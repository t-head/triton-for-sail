"""
Test descending AIU and block-ptr load patterns.

Descending-order block-ptr loads (e.g. reverse K-loop in mamba3 dqkv):
The pipeliner's last pre-fetch produces a negative zOffset/start_c,
which violates the PPU AIU hardware constraint (zOffset/start_c must be >= 0).
Fix: peel the pipelined epilogue while retaining AIU loads.

"""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_descend(
    a_ptr, b_ptr, c_ptr, M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % tl.cdiv(M, BLOCK_M)
    pid_n = pid // tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(0, num_k):
        # Descending: iterate K from high to low
        offs_k = (num_k - 1 - k_idx) * BLOCK_K
        a_ptrs = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1),
                                  offsets=(pid_m * BLOCK_M, offs_k),
                                  block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        b_ptrs = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(N, 1),
                                  offsets=(offs_k, pid_n * BLOCK_N),
                                  block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
        # Full tiles allow AIU promotion without boundary checks.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc=acc)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("M, K, N", [(512, 512, 512), (1024, 256, 512)])
def test_descriptor_descend(num_stages, num_warps, M, K, N):
    """Descending block-pointer loads retain AIU through the pipeline."""
    BM, BK, BN = 64, 64, 64
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kernel = _matmul_descend[grid](A, B, C, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                          num_stages=num_stages, num_warps=num_warps)
    assert kernel.asm["ttir"].count("tt.aiu_load") == 2
    assert "cp.async.aiu" in kernel.asm["llir"]
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, atol=1.0, rtol=1e-2)



@triton.jit
def matmul_desc_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_cm, stride_cn,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M) % M
    offs_bn = (pid_n * BLOCK_SIZE_N) % N
    n_iters = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k = (n_iters - 1) * BLOCK_SIZE_K
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, n_iters):
        a = tl.aiu_load(a_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)
        b = tl.aiu_load(b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], [K, N], tl.float16)
        acc = tl.dot(a, b, acc=acc)
        offs_k -= BLOCK_SIZE_K
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@pytest.mark.parametrize("num_stages", [1, 3])
def test_desc_k_negative_prefetch(num_stages):
    """Direct AIU loads with a loop-carried descending K offset."""
    M, N, K = 64, 64, 512
    generator = torch.Generator(device="cpu").manual_seed(7)
    a_cpu = torch.randint(-2, 3, (M, K), dtype=torch.int32, device="cpu", generator=generator)
    b_cpu = torch.randint(-2, 3, (K, N), dtype=torch.int32, device="cpu", generator=generator)
    ref = (a_cpu @ b_cpu).to(torch.float16)
    a = a_cpu.to(device="cuda", dtype=torch.float16)
    b = b_cpu.to(device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")
    kernel = matmul_desc_kernel[(1,)](
        a, b, c, M, N, K, *c.stride(),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_warps=4, num_stages=num_stages,
    )
    assert kernel.asm["ttir"].count("tt.aiu_load") == 2
    assert "cp.async.aiu" in kernel.asm["llir"]
    torch.testing.assert_close(c.cpu().view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)

@triton.jit
def _auto_promoted_block_ptr_load(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_row = tl.cdiv(COLS, BLOCK_COLS)
    block_row = pid // blocks_per_row
    block_col = pid % blocks_per_row
    block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(block_row * BLOCK_ROWS, block_col * BLOCK_COLS),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )
    values = tl.load(block_ptr)
    offsets = (block_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[:, None] * COLS
    offsets += block_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    tl.store(output_ptr + offsets, values)


def test_auto_promoted_block_ptr_load():
    rows, cols = 64, 64
    block_rows, block_cols = 32, 32
    torch.manual_seed(42)
    input = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(input)
    grid = (triton.cdiv(rows, block_rows) * triton.cdiv(cols, block_cols), )
    kernel = _auto_promoted_block_ptr_load[grid](input, output, ROWS=rows, COLS=cols,
                                                 BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols)
    torch.testing.assert_close(output, input, atol=0, rtol=0)
    assert "tt.aiu_load" in kernel.asm["ttir"]
