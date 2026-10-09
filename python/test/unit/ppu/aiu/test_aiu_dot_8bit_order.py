import pytest
import torch
import itertools
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_aiu_8bit_order(a_ptr, b_ptr, c_ptr,
                                stride_am, stride_ak,
                                stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                M, N, K,
                                BLOCK_SIZE_M: tl.constexpr,
                                BLOCK_SIZE_N: tl.constexpr,
                                BLOCK_SIZE_K: tl.constexpr,
                                DTYPE: tl.constexpr,
                                ACC_DTYPE: tl.constexpr,
                                OUT_DTYPE: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_DTYPE)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # A: row-major (1, 0)
        a = tl.aiu_load(a_ptr, (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), DTYPE.value, (1, 0))
        # B: column-major (0, 1)
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), DTYPE.value, (0, 1))
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=OUT_DTYPE.value)
        offs_k += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# torch dtype -> (tl dtype, accumulator tl dtype, output torch dtype)
dtypes = {
    torch.float8_e5m2: (tl.float8e5, tl.float32, torch.float16),
    torch.int8: (tl.int8, tl.int32, torch.int32),
}

sizes = [32, 64, 128, 256]
@pytest.mark.parametrize("num_stages", [2, 3, 4, 5])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(512, 512, 512)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    list(itertools.product(sizes, sizes, sizes))
)
@pytest.mark.parametrize("dtype", list(dtypes.keys()), ids=["fp8e5", "int8"])
def test_aiu_matmul_8bit_with_order(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, dtype):
    device = "cuda"
    torch.manual_seed(0)

    count_ge256 = sum(v >= 256 for v in (BLOCK_M, BLOCK_N, BLOCK_K))
    if count_ge256 >= 2 and num_stages >= 2:
        pytest.skip("Skip: config will out of resource.")

    if dtype == torch.float8_e5m2:
        A = torch.randn((M, K), device=device, dtype=torch.float16).to(torch.float8_e5m2)
        B = torch.randn((K, N), device=device, dtype=torch.float16).to(torch.float8_e5m2)
    else:
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

    tl_dtype, acc_tl_dtype, out_torch_dtype = dtypes[dtype]
    C_tri = torch.empty((M, N), dtype=out_torch_dtype, device=device)
    # int8 products accumulated in int32 stay exact in float32
    # (max |K * 127 * 128| ~ 8.3M < 2^24), so one reference serves both dtypes.
    C_ref = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(out_torch_dtype)

    matmul_kernel_aiu_8bit_order[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        A, B, C_tri,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_tri.stride(0), C_tri.stride(1),
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
        DTYPE=tl_dtype, ACC_DTYPE=acc_tl_dtype, OUT_DTYPE=acc_tl_dtype
    )

    if dtype == torch.int8:
        # int8 matmul with int32 accumulation is exact.
        assert torch.equal(C_ref, C_tri)
    else:
        torch.testing.assert_close(C_ref, C_tri, atol=1e-2, rtol=1e-1)
