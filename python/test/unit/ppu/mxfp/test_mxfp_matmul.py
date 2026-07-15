import math
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton._internal_testing import is_cuda, is_ppu


@triton.jit
def block_scale_mxfp_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_sk, stride_sb, stride_sc, stride_sd: tl.constexpr,  # Need tl.constexpr to pipeline scale load. Why?
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, USE_2D_SCALE_LOAD: tl.constexpr):
    ## This kernel assumes a_scale and b_scale are coming in with shapes
    ## [BLOCK_M(or N) // 128, BLOCK_K // 128, 32, 4, 4] for optimial performance
    ## on nvidia sm100+ HW
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k = tl.arange(0, BLOCK_K_PACKED)

    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128))
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128))

    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // 128) * 32 * 4 * 4)
        a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
    else:
        offs_sk = tl.arange(0, (BLOCK_K // 128))
        offs_sc = tl.arange(0, 32)
        offs_sd = tl.arange(0, 4)
        a_scale_ptr = a_scale + (offs_sm[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])
        b_scale_ptr = b_scale + (offs_sn[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)

        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // 128, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // 128, 32, 4, 4)

        # Scales are coming in for optimial performance, but we reshape here for
        # the canonical inputs to dot_scaled
        # These reshapes and transposes will be optimized away during lowering
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // 32)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // 32)
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)

        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        a_scale_ptr += BLOCK_K // 128 * stride_sb
        b_scale_ptr += BLOCK_K // 128 * stride_sb
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)

def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale

@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (998, 111, 512), (63, 128, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 128, 256), (128, 256, 256)])
@pytest.mark.parametrize("NUM_STAGES", [1, 2, 4])
@pytest.mark.parametrize("USE_2D_SCALE_LOAD", [False, True])
@pytest.mark.skipif(not (is_ppu() and torch.cuda.get_device_capability() == (8, 9)), reason="Requires PPU0015")
def test_blocked_scale_mxfp4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, USE_2D_SCALE_LOAD, device):
    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 2)
    elif BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 3)

    torch.manual_seed(42)
    NUM_STAGES = 1
    VEC_SIZE = 32
    torch.manual_seed(42)
    packing_dim = 1
    scale_type = "float8_e8m0fnu"
    a_mxfp4 = MXFP4Tensor(size=(M, K), device=device).random()
    a = a_mxfp4.to_packed_tensor(dim=packing_dim)
    b_mxfp4 = MXFP4Tensor(size=(N, K), device=device).random()
    b = b_mxfp4.to_packed_tensor(dim=packing_dim).T
    b_ref = b_mxfp4.to(torch.float32).T
    a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
    b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)

    dtype_src_str = "float8e5"
    dtype_dst_str = "float32"
    ceildiv = lambda a, b: math.ceil(a / b)
    a_size = (ceildiv(M, 128), ceildiv(K, 128), 32, 4, 4)
    b_size = (ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4)
    a_scale = torch.rand(a_size, device=device)
    b_scale = torch.rand(b_size, device=device)
    a_scale_ref = MXScaleTensor(a_scale)
    b_scale_ref = MXScaleTensor(b_scale)
    a_scale = a_scale_ref.data
    b_scale = b_scale_ref.data

    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out = block_scale_mxfp_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a_scale.stride(1),
                                        a_scale.stride(2), a_scale.stride(3), a.stride(0), a.stride(1), b.stride(0),
                                        b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K,
                                        NUM_STAGES=NUM_STAGES, USE_2D_SCALE_LOAD=USE_2D_SCALE_LOAD)

    def flatten_scale(scale):
        num_chunk_m, num_chunk_k, _, _, _ = scale.shape
        return scale.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_f32 = flatten_scale(fp8e8m0_to_float32(a_scale))[:M]
    b_scale_f32 = flatten_scale(fp8e8m0_to_float32(b_scale))[:N]
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = a_mxfp4.to(torch.float32) * a_scale_f32
    b = b_mxfp4.to(torch.float32).T * b_scale_f32
    ref_out = torch.matmul(a, b).to(torch.float32)
    output = output.to(torch.float32)
    atol = 0.01
    rtol = 0.02
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
