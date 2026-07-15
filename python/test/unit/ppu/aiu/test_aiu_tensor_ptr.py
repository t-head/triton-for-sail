import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


@triton.jit
def load_kernel_aiu(a_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_ak = pid_k * BLOCK_SIZE_K

    a_block_ptr = tl.make_block_ptr(a_ptr, (M, K), (K, 1), (offs_am, offs_ak), (BLOCK_SIZE_M, BLOCK_SIZE_K), (1, 0))
    a = tl.aiu_load(a_block_ptr)

    c = a.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + K * offs_cm[:, None] + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptr, c, mask=c_mask)


@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, K", [(1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_K",
    [(32, 32), (32, 64), (64, 64), (64, 128), (64, 256), (128, 64), (128, 128), (128, 256), (256, 256)],
)
def test_aiu_load(num_stages, M, K, BLOCK_M, BLOCK_K, num_warps):
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    C = torch.empty((M, K), dtype=torch.float16, device=device)

    load_kernel_aiu[(triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)](
        A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )

    torch.testing.assert_close(A, C, rtol=1e-3, atol=1e-3)


# test_aiu_load(1, 32, 32)
# test_aiu_load(2, 32, 32)
# test_aiu_load(2, 32, 64, 2)
# test_aiu_load(2, 32, 32, 32, 32, 1)
# test_aiu_load(2, 32, 32, 32, 32, 8)
# test_aiu_load(4, 32, 32, 32, 32, 8)


@triton.jit
def matmul_kernel_aiu(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_tensor_ptr = tl.make_block_ptr(
        a_ptr, (M, K), (stride_am, stride_ak), (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (1, 0)
    )
    b_tensor_ptr = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bk, stride_bn), (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (1, 0)
    )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_tensor_ptr)
        b = tl.aiu_load(b_tensor_ptr)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_tensor_ptr = tl.advance(a_tensor_ptr, (0, BLOCK_SIZE_K))
        b_tensor_ptr = tl.advance(b_tensor_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_tensor_ptr = tl.make_block_ptr(
        c_ptr, (M, N), (stride_cm, stride_cn), (offs_cm, offs_cn), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 0)
    )
    tl.store(c_tensor_ptr, c)


@triton.jit
def matmul_kernel_aiu_mixed_load(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)

    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    b_tensor_ptr = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bk, stride_bn), (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.aiu_load(b_tensor_ptr)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_tensor_ptr = tl.advance(b_tensor_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    [
        (32, 32, 32),
        (64, 64, 64),
        (128, 64, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 64, 128),
        (128, 128, 128),
        (64, 64, 256),
    ],
)
@pytest.mark.parametrize("mixed_load", [False, True])
def test_aiu_matmul(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, mixed_load):
    device = "cuda"

    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)

    if mixed_load is True:
        matmul_kernel_aiu_mixed_load[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
            A,
            B,
            C,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        matmul_kernel_aiu[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
            A,
            B,
            C,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)

    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)


# test_aiu_matmul(2, 32, 32, 32, 32, 32, 32, 4, False)
# test_aiu_matmul(2, 64, 64, 64, 32, 32, 32, 4, False)
