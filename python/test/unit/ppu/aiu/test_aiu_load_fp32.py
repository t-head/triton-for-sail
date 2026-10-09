import numpy as np
import pytest
import torch
import tempfile
import triton
import triton.language as tl
import itertools

from triton._internal_testing import to_numpy, is_ppu

torch.set_printoptions(profile="full")


@triton.jit
def load_kernel_aiu_fp32(a_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_ak = pid_k * BLOCK_SIZE_K
    a = tl.aiu_load(a_ptr, [offs_am, offs_ak], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float32)
    c = a.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + K * offs_cm[:, None] + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptr, c, mask=c_mask)


@triton.jit
def load_kernel_fp32(a_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + K * offs_am[:, None] + offs_ak[None, :]
    a_mask = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

    c_ptrs = c_ptr + K * offs_am[:, None] + offs_ak[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
    tl.store(c_ptrs, a, mask=c_mask)


sizes = [16, 32, 64, 128, 256]
@pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, K", [(1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_K",
    list(itertools.product(sizes, sizes))
)
def test_load_fp32(num_stages, M, K, BLOCK_M, BLOCK_K, num_warps):
    device = "cuda"
    torch.manual_seed(0)

    if is_ppu() and torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] != 9:
        pytest.skip("fp32 only support ppu1.5")

    A = torch.randn((M, K), device=device, dtype=torch.float32)

    C_tri = torch.empty((M, K), device=A.device, dtype=torch.float32)
    C_ref = torch.empty((M, K), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)
    load_kernel_aiu_fp32[grid](
        A, C_tri,
        M, K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )

    load_kernel_fp32[grid](
        A, C_ref,
        M, K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )

    max_diff = (C_tri - C_ref).abs().max().item()
    torch.set_printoptions(sci_mode=False, precision=6)


    print(f"Max diff: {max_diff:.6e}")
    print(f"allclose: {torch.allclose(C_tri, C_ref, atol=1e-3, rtol=1e-3)}")
    np.testing.assert_allclose(to_numpy(C_tri), to_numpy(C_ref), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(to_numpy(A), to_numpy(C_tri), atol=1e-3, rtol=1e-3)

# test_load_fp32(1, 16, 16, 16, 16, 1)