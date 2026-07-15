import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


@triton.jit
def addmm_kernel_aiu(
    bias_ptr,
    alpha,
    beta,
    a_ptr,
    b_ptr,
    c_ptr,
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
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_ptr, (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), tl.float16)
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    offs_biasn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_ptrs = bias_ptr + offs_biasn
    bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0)
    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    # c = accumulator.to(tl.float16)
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
@pytest.mark.parametrize("scalar", [0.001, -0.999, 100.001, -111.999])
def test_aiu_addmm(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, scalar):
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    bias = torch.randn((N,), dtype=torch.float16, device=device)
    alpha = beta = scalar
    addmm_kernel_aiu[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        bias,
        alpha,
        beta,
        A,
        B,
        C,
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
    # ref_out = (bias*beta + torch.matmul(A.to(torch.float32), B.to(torch.float32))*alpha).to(torch.float16)
    ref_out = torch.addmm(bias, A, B, alpha=alpha, beta=beta)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
