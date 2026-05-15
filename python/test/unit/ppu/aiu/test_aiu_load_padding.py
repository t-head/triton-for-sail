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
    a = tl.aiu_load(a_ptr, [offs_am, offs_ak], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)
    c = a.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + BLOCK_SIZE_K * offs_cm[:, None] + offs_ck[None, :]
    tl.store(c_ptr, c)

@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, K", [(64, 64)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_K",
                        [(64, 128), (64, 256),
                         (128, 64), (128, 128), (128, 256),
                         (256, 256)
                         ])
def test_aiu_load(monkeypatch, num_stages, M, K, BLOCK_M, BLOCK_K, num_warps):    
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    C = torch.zeros((BLOCK_M, BLOCK_K), dtype=torch.float16, device=device)

    load_kernel_aiu[(1,1,1)](A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps,
                              num_stages=num_stages)
    expected = torch.zeros((BLOCK_M, BLOCK_K), dtype=torch.float16, device=device)
    expected[:M, :K] = A

    torch.testing.assert_close(C, expected, rtol=1e-3, atol=1e-3)
