import pytest
import torch
import triton
import itertools
import numpy as np
import triton.language as tl

from triton._internal_testing import to_numpy, is_ppu


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def matmul_kernel_aiu_fp32_order(A_ptr, B_ptr, C_ptr,
                                M, N, K,
                                stride_am, stride_ak,
                                stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                BLOCK_M: tl.constexpr,
                                BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # A: row-major (1, 0)
        a = tl.aiu_load(A_ptr, (offs_am, offs_k), (BLOCK_M, BLOCK_K), (M, K), tl.float32,(1, 0))
        # B: column-major (0, 1)
        b = tl.aiu_load(B_ptr, (offs_k, offs_bn), (BLOCK_K, BLOCK_N), (K, N), tl.float32, (0, 1))
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

sizes = [32, 64, 128, 256]
@pytest.mark.parametrize("num_stages", [2, 3, 4, 5])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(512, 512, 512)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    list(itertools.product(sizes, sizes, sizes))
)
def test_aiu_matmul_fp32_with_order(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    device = "cuda"
    torch.manual_seed(0)

    if is_ppu() and torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] != 9:
        pytest.skip("fp32 only support ppu1.5")

    if (BLOCK_M * BLOCK_K * 4 + BLOCK_K * BLOCK_N * 4) * max(1, (num_stages - 1)) > 262144:
        pytest.skip("Skip: config will out of resource using pipeline.")

    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)

    C_tri = torch.empty((M, N), device=A.device, dtype=torch.float32)
    C_ref = torch.matmul(A, B.T)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    matmul_kernel_aiu_fp32_order[grid](
        A, B, C_tri,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_tri.stride(0), C_tri.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )

    max_diff = (C_tri - C_ref).abs().max().item()
    torch.set_printoptions(sci_mode=False, precision=6)

    print(f"Max diff: {max_diff:.6e}")
    print(f"allclose: {torch.allclose(C_tri, C_ref, atol=1e-2, rtol=1e-1)}")
    np.testing.assert_allclose(to_numpy(C_tri), to_numpy(C_ref), atol=1e-1, rtol=1e-1)


# test_aiu_matmul_fp32_with_order(1, 16, 16, 16, 16, 16, 16, 1)