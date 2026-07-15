import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


@triton.jit
def matmul_kernel(
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
        a_ptr, (M, K), (stride_ak, stride_am), (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (0, 1)
    )
    b_tensor_ptr = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bn, stride_bk), (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (0, 1)
    )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_tensor_ptr)
        b = tl.load(b_tensor_ptr)
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
        a_ptr, (M, K), (stride_ak, stride_am), (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (0, 1)
    )
    b_tensor_ptr = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bn, stride_bk), (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (0, 1)
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
def test_aiu_matmul(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    device = "cuda"

    torch.manual_seed(42)
    A = torch.randn((K, M), dtype=torch.float16, device=device)
    B = torch.randn((N, K), dtype=torch.float16, device=device)
    # B = torch.eye(N, dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)

    AT = A.t()
    BT = B.t()

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
    ref_out = torch.matmul(AT.to(torch.float32), BT.to(torch.float32)).to(torch.float16)

    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
