import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


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

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_ptr, (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), tl.float16, (0, 1))
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), tl.float16, (0, 1))
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
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
        (16, 16, 16),
        (16, 32, 16),
        (16, 16, 32),
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
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_matmul(monkeypatch, num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("m8 mma only support on ppu1.0")

    device = "cuda"

    torch.manual_seed(42)
    # A = torch.eye(M, dtype=torch.float16, device=device)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    # B = torch.eye(M, dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)

    # Idx = [i for i in range(M*K)]
    # print(Idx)
    # npA = np.array(Idx)
    # npR = npA.reshape(M,K)
    # npT = npA.reshape(64,128)
    # IdxA = torch.from_numpy(npR).to(torch.float16)
    # IdxB = torch.from_numpy(npT).to(torch.float16)
    # print(IdxA)
    # A = IdxA.to(device)
    # B = IdxB.to(device)

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
    AT = A.t()
    BT = B.t()
    ref_out = torch.matmul(AT.to(torch.float32), BT.to(torch.float32)).to(torch.float16)

    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
