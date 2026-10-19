"""Reduced sglang MoE for zero-trip loop prologues."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def swizzle1d(pid, M: tl.constexpr, N: tl.constexpr, group_size_m: tl.constexpr):
    group_size = N * group_size_m
    group_id = pid // group_size
    base_in_group = group_id * group_size
    off_in_group = pid - base_in_group
    size_m = min(M - base_in_group // N, group_size_m)
    off_m_in_group = off_in_group // N
    off_n_in_group = off_in_group % N
    return base_in_group + off_n_in_group * size_m + off_m_in_group


@triton.jit
def fused_moe_persistence_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    expert_ids_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_tiles,
    num_valid_tokens: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr = 3,
    loop_unroll_factor: tl.constexpr = 4,
    num_sms: tl.constexpr = 20,
    SWIZZLE_SIZE: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    num_workers = tl.num_programs(axis=0)
    swizzle_pid = swizzle1d(pid, num_workers // num_sms, num_sms, SWIZZLE_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    grid_size_n = triton.cdiv(N, BLOCK_SIZE_N)
    all_tiles = num_tiles * grid_size_n

    # same flatten/unroll heuristic as sglang: k_iters < 4 -> flatten and
    # unroll by k_iters; k_iters == 2 or 3 gives a zero-trip remainder loop
    # for every program under the grid chosen below
    loop_k_iters: tl.constexpr = triton.cdiv(K, BLOCK_SIZE_K)
    faltten: tl.constexpr = loop_k_iters < 4
    unroll: tl.constexpr = loop_k_iters if faltten else 1

    for tid in tl.range(
        swizzle_pid,
        all_tiles,
        num_workers,
        num_stages=num_stages,
        loop_unroll_factor=unroll,
        flatten=faltten,
    ):
        tid_m = tid // grid_size_n
        tid_n = tid % grid_size_n

        offs_row = tl.arange(0, BLOCK_SIZE_M)
        offs_token = tid_m * BLOCK_SIZE_M + offs_row
        token_mask = offs_token < num_valid_tokens
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        # loop-carried expert base: loaded inside the loop, so the flattened
        # remainder loop carries it as a poison-initialized value
        off_experts = tl.load(expert_ids_ptr + tid_m).to(tl.int64)

        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + off_experts * stride_be,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, tid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(0, 1),
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, loop_k_iters):
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K))
            b = tl.aiu_load(b_block_ptr)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
        accumulator = accumulator.to(tl.bfloat16)

        offs_cn = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("K, BLOCK_K, unroll", [(128, 64, 2), (192, 64, 3)])
def test_fused_moe_aiu_zero_trip_prologue(K, BLOCK_K, unroll):
    """The remainder loop must run zero times for every program.

    all_tiles / num_workers is a multiple of `unroll`, so the unrolled main
    loop consumes every iteration and the flattened remainder loop has trip
    count 0; its prologue used to issue an AIU load from a poison base.
    """
    N, BLOCK_N, BLOCK_M, top_k = 512, 64, 64, 1
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = 10 * unroll  # 20/30 tiles * 8 col-tiles over 80 workers -> 2/3 each
    num_workers = 80
    rows = num_tiles * BLOCK_M

    a = torch.randn(rows, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(num_tiles, N, K, dtype=torch.bfloat16, device="cuda") * 0.01
    c = torch.empty(rows, top_k, N, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, top_k, device="cuda").to(torch.bfloat16)
    expert_ids = torch.arange(num_tiles, dtype=torch.int32, device="cuda")

    # sanity: even tile split, and per-program trip count a multiple of unroll
    tiles = num_tiles * grid_n
    assert tiles % num_workers == 0, "uneven tile split; pick another grid"
    assert (tiles // num_workers) % unroll == 0, "test requires zero remainder trips"

    fused_moe_persistence_kernel[(num_workers,)](
        a,
        b,
        c,
        weights,
        expert_ids,
        N,
        K,
        num_tiles,
        rows,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        top_k=top_k,
        num_warps=4,
        num_stages=3,
        loop_unroll_factor=4,
        num_sms=20,
        SWIZZLE_SIZE=grid_n,
    )
    torch.cuda.synchronize()

    # each row block m multiplies with weight matrix b[m], scaled by the
    # routed weight (MUL_ROUTED_WEIGHT path of the kernel)
    ref = torch.bmm(
        a.float().view(num_tiles, BLOCK_M, K),
        b.float().transpose(1, 2),
    )
    ref = ref * weights.float().view(num_tiles, BLOCK_M, 1)
    ref = ref.to(torch.bfloat16).view(rows, N)
    torch.testing.assert_close(c.view(rows, N), ref, atol=2e-2, rtol=2e-2)
