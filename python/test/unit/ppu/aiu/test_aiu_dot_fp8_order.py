import pytest
import torch
import itertools
import triton
import triton.language as tl

from triton._internal_testing import is_ppu

@triton.jit
def matmul_kernel_aiu_fp8_order(a_ptr, b_ptr, c_ptr,
                                stride_am, stride_ak,
                                stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                M, N, K,
                                BLOCK_SIZE_M: tl.constexpr,
                                BLOCK_SIZE_N: tl.constexpr,
                                BLOCK_SIZE_K: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # A: row-major (1, 0)
        a = tl.aiu_load(a_ptr, (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), tl.float8e5,(1, 0))
        # B: column-major (0, 1)
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), tl.float8e5, (0, 1))
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
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
def test_aiu_matmul_fp8_with_order(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    device = "cuda"
    torch.manual_seed(0)

    if is_ppu() and torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] != 9:
        pytest.skip("fp8 only support ppu0015")

    count_ge256 = sum(v >= 256 for v in (BLOCK_M, BLOCK_N, BLOCK_K))
    if count_ge256 >= 2 and num_stages >= 2:
        pytest.skip("Skip: config will out of resource.")

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)

    A = A.to(torch.float8_e5m2)
    B = B.to(torch.float8_e5m2)
    C_tri = torch.empty((M, N), dtype=torch.float16, device=device)
    C_ref = torch.matmul(A.to(torch.float16), B.T.to(torch.float16))

    matmul_kernel_aiu_fp8_order[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        A, B, C_tri,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_tri.stride(0), C_tri.stride(1),
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )

    torch.testing.assert_close(C_ref, C_tri, atol=1e-2, rtol=1e-1)
    if torch.allclose(C_ref, C_tri, atol=1e-2, rtol=1e-1):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


# test_aiu_matmul_fp8_with_order(2, 64, 64, 64, 32, 32, 32, 1)
