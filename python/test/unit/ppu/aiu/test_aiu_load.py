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
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_load(monkeypatch, num_stages, M, K, BLOCK_M, BLOCK_K, num_warps, m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("m8 mma only support on ppu1.0")

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    C = torch.empty((M, K), dtype=torch.float16, device=device)
    # Idx = [i for i in range(M*K)]
    # print(Idx)
    # npA = np.array(Idx)
    # npR = npA.reshape(M,K)
    # IdxA = torch.from_numpy(npR).to(torch.float16)
    # A = IdxA.to(device)

    load_kernel_aiu[(triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)](
        A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )
    torch.testing.assert_close(A, C, rtol=1e-3, atol=1e-3)


# test_aiu_load(1, 32, 32)
# test_aiu_load(2, 32, 32)
# test_aiu_load(2, 32, 64, 2)
# test_aiu_load(2, 32, 32, 32, 32, 1)
# test_aiu_load(4, 32, 32, 32, 32, 8)

@pytest.mark.parametrize("num_stages", [1, 2, 4, 7])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, K", [(1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_K", [(16, 16), (16, 32), (16, 64), (32, 16), (32, 32), (32, 64), (64, 16), (64, 32), (64, 64)]
)
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_load_small_block(monkeypatch, num_stages, M, K, BLOCK_M, BLOCK_K, num_warps, m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("m8 mma only support on ppu1.0")

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    C = torch.empty((M, K), dtype=torch.float16, device=device)

    load_kernel_aiu[(triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)](
        A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )
    torch.testing.assert_close(A, C, rtol=1e-3, atol=1e-3)
