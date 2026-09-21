"""
Test AIU descriptor load patterns.
Covers two issues fixed in RewriteTensorDescriptorToPointer.cpp:
1. Descending-order descriptor loads (e.g. reverse K-loop in mamba3 dqkv):
   The pipeliner's last pre-fetch produces a negative zOffset/start_c,
   which violates the PPU AIU hardware constraint (zOffset/start_c must be >= 0).
   Fix: detect subtraction in the index chain and fall back to pointer load.
2. Multi-use descriptor loads (direct + tl.trans, e.g. mamba3 ssm_states):
   A descriptor load result used both directly as dot operand B and via
   tl.trans as another dot operand B.
   Fix: normal AIU load + downstream memdesc_trans for the transposed view.
"""
import pytest
import torch
import triton
import triton.language as tl
def _alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc_fn)
# ============================================================================
# Test 1: Descending-order descriptor load (reverse K-loop matmul)
# ============================================================================
@triton.jit
def _matmul_descend(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % tl.cdiv(M, BLOCK_M)
    pid_n = pid // tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_K)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1],
                                       block_shape=[BLOCK_K, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(0, num_k):
        # Descending: iterate K from high to low
        offs_k = (num_k - 1 - k_idx) * BLOCK_K
        a = a_desc.load([pid_m * BLOCK_M, offs_k])
        b = b_desc.load([offs_k, pid_n * BLOCK_N])
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
# ============================================================================
# Test 2: Descriptor load used both directly and via tl.trans (mamba ssm)
# ============================================================================
@triton.jit
def _ssm_multi_use(
    DO_ptr, Q_ptr, SSM_ptr, Out1_ptr, Out2_ptr,
    seqlen, headdim_v, K_ssm,
    CHUNK: tl.constexpr, HQK: tl.constexpr, HV: tl.constexpr,
):
    num_chunks = tl.cdiv(seqlen, CHUNK)
    do_desc = tl.make_tensor_descriptor(DO_ptr, shape=[seqlen, headdim_v],
        strides=[headdim_v, 1], block_shape=[CHUNK, HV])
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[seqlen, HQK],
        strides=[HQK, 1], block_shape=[CHUNK, HQK])
    ssm_desc = tl.make_tensor_descriptor(SSM_ptr, shape=[headdim_v, K_ssm],
        strides=[K_ssm, 1], block_shape=[HV, HQK])
    acc1 = tl.zeros((CHUNK, HQK), dtype=tl.float32)
    acc2 = tl.zeros((CHUNK, HV), dtype=tl.float32)
    for c in range(num_chunks):
        do_blk = do_desc.load([c * CHUNK, 0])
        q_blk = q_desc.load([c * CHUNK, 0])
        ssm_blk = ssm_desc.load([0, c * HQK])
        # Multi-use: ssm directly as operand B + transposed as operand B
        acc1 += tl.dot(do_blk, ssm_blk)
        acc2 += tl.dot(q_blk, tl.trans(ssm_blk))
    offs_s = tl.arange(0, CHUNK)
    tl.store(Out1_ptr + offs_s[:, None] * HQK + tl.arange(0, HQK)[None, :], acc1.to(tl.bfloat16))
    tl.store(Out2_ptr + offs_s[:, None] * HV + tl.arange(0, HV)[None, :], acc2.to(tl.bfloat16))
@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("seqlen, headdim_qk, headdim_v",
                         [(1024, 64, 16), (512, 64, 16), (256, 64, 16)])
def test_descriptor_multi_use_trans(num_stages, num_warps, seqlen, headdim_qk, headdim_v):
    """Descriptor load used both directly and via tl.trans — the mamba ssm pattern."""
    CHUNK = 64
    num_chunks = seqlen // CHUNK
    K_SSM = num_chunks * headdim_qk
    torch.manual_seed(42)
    DO = torch.randn((seqlen, headdim_v), dtype=torch.bfloat16, device="cuda")
    Q = torch.randn((seqlen, headdim_qk), dtype=torch.bfloat16, device="cuda")
    SSM = torch.randn((headdim_v, K_SSM), dtype=torch.bfloat16, device="cuda")
    O1 = torch.empty((CHUNK, headdim_qk), dtype=torch.bfloat16, device="cuda")
    O2 = torch.empty((CHUNK, headdim_v), dtype=torch.bfloat16, device="cuda")
    _ssm_multi_use[(1,)](DO, Q, SSM, O1, O2, seqlen, headdim_v, K_SSM,
                         CHUNK=CHUNK, HQK=headdim_qk, HV=headdim_v,
                         num_stages=num_stages, num_warps=num_warps)
    ref1 = torch.zeros((CHUNK, headdim_qk), dtype=torch.float32, device="cuda")
    ref2 = torch.zeros((CHUNK, headdim_v), dtype=torch.float32, device="cuda")
    for c in range(num_chunks):
        d = DO[c*CHUNK:(c+1)*CHUNK].float()
        q = Q[c*CHUNK:(c+1)*CHUNK].float()
        s = SSM[:, c*headdim_qk:(c+1)*headdim_qk].float()
        ref1 += d @ s
        ref2 += q @ s.T
    torch.testing.assert_close(O1.float(), ref1, atol=2.0, rtol=1e-2)
    torch.testing.assert_close(O2.float(), ref2, atol=2.0, rtol=1e-2)
# ============================================================================
# Test 3: Single-use descriptor load + tl.trans (fused path)
# ============================================================================
@triton.jit
def _desc_trans_dot(
    A_ptr, B_ptr, C_ptr, K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(A_ptr, shape=[BLOCK_M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(B_ptr, shape=[BLOCK_N, K], strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([0, k * BLOCK_K])
        b = b_desc.load([0, k * BLOCK_K])
        acc += tl.dot(a, tl.trans(b))  # single trans use → fused AIU load
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    tl.store(C_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc.to(tl.bfloat16))
@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("M, K, N", [(64, 256, 64), (64, 512, 64)])
def test_descriptor_trans_dot(num_stages, num_warps, M, K, N):
    """Single-use descriptor load fused with tl.trans into transposed AIU load."""
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    _desc_trans_dot[(1,)](A, B, C, K, BLOCK_M=M, BLOCK_K=64, BLOCK_N=N,
                          num_stages=num_stages, num_warps=num_warps)
    ref = A.float() @ B.float().T
    torch.testing.assert_close(C.float(), ref, atol=1.0, rtol=1e-2)


@triton.jit
def _descriptor_padded_stride(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_row = tl.cdiv(COLS, BLOCK_COLS)
    block_row = pid // blocks_per_row
    block_col = pid % blocks_per_row
    desc = tl.make_tensor_descriptor(input_ptr, shape=[ROWS, COLS], strides=[ROW_STRIDE, 1],
                                     block_shape=[BLOCK_ROWS, BLOCK_COLS])
    values = desc.load([block_row * BLOCK_ROWS, block_col * BLOCK_COLS])
    offsets = (block_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[:, None] * COLS
    offsets += block_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    tl.store(output_ptr + offsets, values)


def test_descriptor_padded_stride():
    rows, cols, row_stride = 32, 32, 48
    block_rows, block_cols = 16, 32
    torch.manual_seed(42)
    input = torch.randn((rows, row_stride), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(rows, block_rows) * triton.cdiv(cols, block_cols), )
    _descriptor_padded_stride[grid](input, output, ROWS=rows, COLS=cols, ROW_STRIDE=row_stride,
                                    BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols)
    torch.testing.assert_close(output, input[:, :cols], atol=0, rtol=0)
