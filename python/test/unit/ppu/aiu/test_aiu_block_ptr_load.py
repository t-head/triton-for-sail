"""
Test AIU block-ptr load patterns.

Descending-order block-ptr loads (e.g. reverse K-loop in mamba3 dqkv):
The pipeliner's last pre-fetch produces a negative zOffset/start_c,
which violates the PPU AIU hardware constraint (zOffset/start_c must be >= 0).
Fix: detect subtraction in the index chain and fall back to pointer load.

"""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_descend(
    a_ptr, b_ptr, c_ptr, M, N, K,
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
        a = tl.load(a_ptrs, boundary_check=(0, 1))
        b = tl.load(b_ptrs, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc=acc)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("M, K, N", [(512, 512, 512), (1024, 256, 512)])
def test_descriptor_descend(num_stages, num_warps, M, K, N):
    """Descriptor load with descending K-loop: compile-time detection prevents
    AIU conversion, avoiding negative zOffset from pipeliner pre-fetch."""
    BM, BK, BN = 64, 64, 64
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _matmul_descend[grid](A, B, C, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                          num_stages=num_stages, num_warps=num_warps)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, atol=1.0, rtol=1e-2)


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
