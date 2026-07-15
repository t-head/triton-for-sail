import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    src = kernel.src
    for key, value in to_replace.items():
        # kernel.src = kernel.src.replace(key, value)
        src = src.replace(key, value)
        kernel._unsafe_update_src(src)
    return kernel


@triton.jit
def binary_kernel_aiu(a_ptr, b_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_k = pid_k * BLOCK_SIZE_K

    a = tl.aiu_load(a_ptr, [offs_m, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)
    b = tl.aiu_load(b_ptr, [offs_m, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)

    c = GENERATE_TEST_HERE

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + K * offs_cm[:, None] + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptr, c, mask=c_mask)


@triton.jit
def binary_kernel_aiu_mixed_load(a_ptr, b_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ak = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_ak[None, :])
    a = tl.load(a_ptrs, mask=offs_ak[None, :] < K, other=0.0)

    offs_m = pid_m * BLOCK_SIZE_M
    offs_k = pid_k * BLOCK_SIZE_K
    b = tl.aiu_load(b_ptr, [offs_m, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)

    c = GENERATE_TEST_HERE

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + K * offs_cm[:, None] + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptr, c, mask=c_mask)


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, K", [(1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_K",
    [(32, 32), (32, 64), (64, 64), (64, 128), (64, 256), (128, 64), (128, 128), (128, 256), (256, 64)],
)
@pytest.mark.parametrize("mixed_load", [False, True])
def test_aiu_binary(op, num_stages, M, K, BLOCK_M, BLOCK_K, num_warps, mixed_load):
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((M, K), dtype=torch.float16, device=device)
    C = torch.empty((M, K), dtype=torch.float16, device=device)

    A_ref = A.to(torch.float16)
    B_ref = B.to(torch.float16)

    kernel = patch_kernel(binary_kernel_aiu, {"GENERATE_TEST_HERE": f"a {op} b"})
    if mixed_load is True:
        kernel = patch_kernel(binary_kernel_aiu_mixed_load, {"GENERATE_TEST_HERE": f"a {op} b"})

    C_ref = eval(f"A_ref {op} B_ref")

    kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)](
        A, B, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )

    torch.testing.assert_close(C, C_ref, rtol=1e-3, atol=1e-3)


# test_aiu_binary("+", 2, 32, 32, 32, 32, 8, False)
# test_aiu_binary("+", 2, 32, 32, 32, 32, 8, True)
