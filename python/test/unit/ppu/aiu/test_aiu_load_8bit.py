import pytest
import torch
import triton
import triton.language as tl

torch.set_printoptions(profile="full")


@triton.jit
def load_kernel_aiu_8bit(a_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                         DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_ak = pid_k * BLOCK_SIZE_K
    a = tl.aiu_load(a_ptr, [offs_am, offs_ak], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], DTYPE.value)
    c = a.to(OUT_DTYPE)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptr = c_ptr + K * offs_cm[:, None] + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptr, c, mask=c_mask)


@triton.jit
def pipelined_load_kernel_aiu_8bit(a_ptr, c_ptr, M, K, BLOCK_SIZE_M: tl.constexpr,
                                   BLOCK_SIZE_K: tl.constexpr, DTYPE: tl.constexpr,
                                   OUT_DTYPE: tl.constexpr):
    offs_m = tl.program_id(axis=0) * BLOCK_SIZE_M
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), OUT_DTYPE)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=2):
        a = tl.aiu_load(a_ptr, [offs_m, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], DTYPE.value)
        accumulator += a.to(OUT_DTYPE)
        offs_k += BLOCK_SIZE_K
    offs_cm = offs_m + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = tl.arange(0, BLOCK_SIZE_K)
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < BLOCK_SIZE_K)
    tl.store(c_ptr + BLOCK_SIZE_K * offs_cm[:, None] + offs_ck[None, :], accumulator, mask=c_mask)


# torch dtype -> (tl dtype, output tl dtype, output torch dtype)
dtypes = {
    torch.float8_e5m2: (tl.float8e5, tl.float16, torch.float16),
    torch.int8: (tl.int8, tl.int16, torch.int16),
}


@pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
@pytest.mark.parametrize("M, K", [(1024, 1024)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_K",
    [(16, 32), (32, 32), (32, 64), (64, 32), (64, 64), (64, 128), (64, 256),
     (128, 64), (128, 128), (128, 256), (256, 128), (256, 256), (512, 512)],
)
@pytest.mark.parametrize("m8_mma", [0, 1])
@pytest.mark.parametrize("dtype", list(dtypes.keys()), ids=["fp8e5", "int8"])
def test_aiu_load_8bit(monkeypatch, num_stages, M, K, BLOCK_M, BLOCK_K, num_warps, m8_mma, dtype):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("m8 mma only support on ppu1.0")

    device = "cuda"
    torch.manual_seed(42)
    if dtype == torch.float8_e5m2:
        A = torch.randn((M, K), dtype=torch.float16, device=device)
        A = A.to(torch.float8_e5m2)
    else:
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    tl_dtype, out_tl_dtype, out_torch_dtype = dtypes[dtype]
    C = torch.empty((M, K), dtype=out_torch_dtype, device=device)

    load_kernel_aiu_8bit[(triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), 1, 1)](
        A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages,
        DTYPE=tl_dtype, OUT_DTYPE=out_tl_dtype
    )
    # fp8/int8 -> wider type is lossless, so the load/store roundtrip should match exactly.
    torch.testing.assert_close(A.to(out_torch_dtype), C, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("pipelined", [False, True], ids=["lowering", "pipeliner"])
def test_aiu_load_8bit_rejects_narrow_ppu10_channel(capfd, pipelined):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires PPU 1.0")

    M = BLOCK_M = BLOCK_K = 16
    K = 32 if pipelined else 16
    A = torch.empty((M, K), dtype=torch.int8, device="cuda")
    C = torch.empty((M, BLOCK_K), dtype=torch.int16, device="cuda")
    kernel = pipelined_load_kernel_aiu_8bit if pipelined else load_kernel_aiu_8bit

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        kernel.warmup(
            A, C, M, K, BLOCK_M, BLOCK_K, DTYPE=tl.int8,
            OUT_DTYPE=tl.int16, grid=(1, ), num_warps=2,
            num_stages=2 if pipelined else 1,
        )
    captured = capfd.readouterr()
    assert "unsupported AIU load tile for PPU AIU version 1: channel width is 16 bytes" in (
        captured.out + captured.err
    )
