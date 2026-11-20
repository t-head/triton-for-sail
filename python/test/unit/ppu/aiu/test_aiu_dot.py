import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ppu.compiler import HGGCOptions

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
        a = tl.aiu_load(a_ptr, (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), tl.float16)
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), tl.float16)
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


@triton.jit
def matmul_kernel_aiu_mixed_load(
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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.aiu_load(b_ptr, (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (K, N), tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
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
@pytest.mark.parametrize("mixed_load", [False, True])
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_matmul(monkeypatch, num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, mixed_load, m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    matmul_kernel_aiu.device_caches.clear()
    matmul_kernel_aiu_mixed_load.device_caches.clear()
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("M8 MMA is supported only on PPU0010")
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
    if mixed_load is True:
        matmul_kernel_aiu_mixed_load[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
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
    else:
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
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)


# test_aiu_matmul(2, 32, 32, 64, 32, 32, 32, 1, True)
# test_aiu_matmul(2, 32, 32, 64, 32, 32, 32, 1, True)
# test_aiu_matmul(3, 16, 16, 32, 16, 16, 16, 1, True)
# test_aiu_matmul(3, 64, 64, 64, 64, 64, 64, 1, True)
# test_aiu_matmul(4, 64, 64, 64, 32, 32, 32, 8, False)
# test_aiu_matmul(2, 16, 16, 16, 16, 16, 16, 1, True)


@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(128, 128, 128)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    [
        (16, 16, 16),
        (16, 16, 32),
        (16, 16, 64),
        (16, 16, 128),
        (16, 32, 16),
        (16, 64, 16),
        (16, 128, 16),
        (32, 16, 16),
        (64, 16, 16),
        (128, 16, 16),
    ],
)
@pytest.mark.parametrize("mixed_load", [False, True])
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_matmul_small_block(monkeypatch, num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, mixed_load,
                                m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    matmul_kernel_aiu.device_caches.clear()
    matmul_kernel_aiu_mixed_load.device_caches.clear()
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("M8 MMA is supported only on PPU0010")
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    if mixed_load is True:
        matmul_kernel_aiu_mixed_load[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
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
    else:
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
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(16, 16, 16)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(16, 16, 16)])
@pytest.mark.parametrize("mixed_load", [False, True])
@pytest.mark.parametrize("m8_mma", [0, 1])
def test_aiu_matmul_16x16_block(monkeypatch, num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, mixed_load,
                                m8_mma):
    monkeypatch.setenv("FORCE_USE_M8MMA", str(m8_mma))
    matmul_kernel_aiu.device_caches.clear()
    matmul_kernel_aiu_mixed_load.device_caches.clear()
    capability = torch.cuda.get_device_capability()
    if m8_mma == 1 and not (capability[0] == 8 and capability[1] == 0):
        pytest.skip("M8 MMA is supported only on PPU0010")

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    if mixed_load is True:
        matmul_kernel_aiu_mixed_load[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
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
    else:
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
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)


def _run_aiu_ordinary_route(a, b, num_warps, enable, block_k=None, num_stages=1,
                            enable_tix=False):
    matmul_kernel_aiu.device_caches.clear()
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    compiled = matmul_kernel_aiu[(1, 1, 1)](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        m,
        n,
        k,
        m,
        n,
        block_k or k,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_aiu_ldmatrix_ordinary=enable,
        enable_tix_ldmatrix_swzl=enable_tix,
    )
    torch.cuda.synchronize()
    return c, compiled.asm['llir']


@pytest.mark.parametrize('size,num_warps', [
    (16, 1),
    (32, 1),
    (64, 1),
    (64, 4),
    (64, 8),
    (128, 4),
])
def test_aiu_ldmatrix_ordinary_exact(size, num_warps):
    values = torch.arange(size * size, device='cuda', dtype=torch.int32).remainder(5) - 2
    a = values.to(torch.float16).reshape(size, size)
    b = values.flip(0).to(torch.float16).reshape(size, size)
    reference = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).to(torch.float16)

    default_c, default_llir = _run_aiu_ordinary_route(a, b, num_warps, False)
    ordinary_c, ordinary_llir = _run_aiu_ordinary_route(a, b, num_warps, True)
    final_c, final_llir = _run_aiu_ordinary_route(a, b, num_warps, False)

    assert 'ppu.cp.async.aiu' in default_llir
    assert 'ldmatrix.swzl.sync.bulk.tensor' in default_llir
    assert 'ppu.cp.async.aiu' in ordinary_llir
    assert 'ppu.ldmatrix.sync.aligned' in ordinary_llir
    assert 'ldmatrix.swzl.sync.bulk.tensor' not in ordinary_llir
    default_global = next(line for line in default_llir.splitlines() if '@global_smem =' in line)
    ordinary_global = next(line for line in ordinary_llir.splitlines() if '@global_smem =' in line)
    if size in (32, 64):
        # Canonical standard buffers retain the AIU-required 128-byte alignment.
        assert 'align 128' in ordinary_global
    else:
        assert ordinary_global == default_global
    assert torch.equal(ordinary_c.cpu(), reference)
    assert torch.equal(ordinary_c, default_c)
    assert torch.equal(final_c, default_c)
    assert final_llir == default_llir


@pytest.mark.parametrize('size,num_warps', [(32, 1), (64, 4)])
def test_aiu_standard_tix_exact(size, num_warps):
    values = torch.arange(size * size, device='cuda', dtype=torch.int32).remainder(5) - 2
    a = values.to(torch.float16).reshape(size, size)
    b = values.flip(0).to(torch.float16).reshape(size, size)
    reference = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).to(torch.float16)

    default_c, default_llir = _run_aiu_ordinary_route(a, b, num_warps, False)
    tix_c, tix_llir = _run_aiu_ordinary_route(a, b, num_warps, True, enable_tix=True)

    assert 'ppu.cp.async.aiu' in tix_llir
    assert 'ldmatrix.swzl.sync.bulk.tensor' in tix_llir
    assert 'ppu.ldmatrix.sync.aligned' not in tix_llir
    assert torch.equal(tix_c.cpu(), reference)
    assert torch.equal(tix_c, default_c)


@pytest.mark.parametrize('size,num_warps,block_k,num_stages', [
    (64, 1, 32, 2),
    (64, 4, 16, 2),
    (64, 4, 16, 4),
])
def test_aiu_ldmatrix_ordinary_pipeline(size, num_warps, block_k, num_stages):
    values = torch.arange(size * size, device='cuda', dtype=torch.int32).remainder(5) - 2
    a = values.to(torch.float16).reshape(size, size)
    b = values.flip(0).to(torch.float16).reshape(size, size)
    reference = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).to(torch.float16)

    default_c, _ = _run_aiu_ordinary_route(a, b, num_warps, False, block_k, num_stages)
    ordinary_c, ordinary_llir = _run_aiu_ordinary_route(a, b, num_warps, True, block_k, num_stages)

    assert 'ppu.cp.async.aiu' in ordinary_llir
    assert 'ppu.ldmatrix.sync.aligned' in ordinary_llir
    assert 'ldmatrix.swzl.sync.bulk.tensor' not in ordinary_llir
    assert torch.equal(ordinary_c.cpu(), reference)
    assert torch.equal(ordinary_c, default_c)


def test_aiu_ldmatrix_ordinary_hash():
    assert HGGCOptions(enable_aiu_ldmatrix_ordinary=False).hash() != HGGCOptions(
        enable_aiu_ldmatrix_ordinary=True).hash()

# from _pytest.monkeypatch import MonkeyPatch
# mp = MonkeyPatch()
# test_aiu_matmul_16x16_block(mp, 1, 16, 16, 16, 16, 16, 16, 2, False, 1)
