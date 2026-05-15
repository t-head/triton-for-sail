import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel_aiu(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    M, N, K,
    sm_scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_SIZE_M

    q = tl.aiu_load(q_ptr, (offs_m, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (M, K), tl.float16)

    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    offs_n = 0
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # dot0: Q @ K^T -> qk  [BLOCK_M, BLOCK_N]
        k = tl.aiu_load(k_ptr, (offs_n, 0), (BLOCK_SIZE_N, BLOCK_SIZE_K), (N, K), tl.float16)
        qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N]

        # online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2((qk - m_ij[:, None]) * 1.44269504)  # [BLOCK_M, BLOCK_N]
        alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        m_i = m_ij

        # dot1: p @ V -> acc
        v = tl.aiu_load(v_ptr, (offs_n, 0), (BLOCK_SIZE_N, BLOCK_SIZE_K), (N, K), tl.float16)
        acc = tl.dot(p.to(tl.float16), v, acc=acc)

        offs_n += BLOCK_SIZE_N

    # epilogue
    acc = acc / l_i[:, None]

    # store O
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ok = tl.arange(0, BLOCK_SIZE_K)
    o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_ok * offs_ok[None, :]
    o_mask = offs_om[:, None] < M
    tl.store(o_ptrs, acc, mask=o_mask)


@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("M, N, K", [(1024, 1024, 64)])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    [(32, 32, 64), (64, 64, 64), (128, 64, 64)],
)
def test_attn(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
              num_warps, num_stages, device="cuda"):
    torch.manual_seed(0)
    sm_scale = 0.5

    Q = torch.randn((M, K), dtype=torch.float16, device=device)
    K_ = torch.randn((N, K), dtype=torch.float16, device=device)
    V = torch.randn((N, K), dtype=torch.float16, device=device)
    O = torch.empty((M, K), dtype=torch.float16, device=device)

    grid = (triton.cdiv(M, BLOCK_M), 1, 1)

    attn_kernel_aiu[grid](
        Q, K_, V, O,
        Q.stride(0),  Q.stride(1),
        K_.stride(0), K_.stride(1),
        V.stride(0),  V.stride(1),
        O.stride(0),  O.stride(1),
        M, N, K,
        sm_scale,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # ref
    qk_ref = torch.matmul(Q.float(), K_.float().T) * sm_scale
    p_ref  = torch.softmax(qk_ref, dim=-1).half()
    ref_out = torch.matmul(p_ref, V)

    torch.testing.assert_close(ref_out, O, rtol=1e-3, atol=1e-3)
    print(
        f"[PASSED] M={M} N={N} K={K} "
        f"BLOCK=({BLOCK_M},{BLOCK_N},{BLOCK_K}) "
        f"warps={num_warps} stages={num_stages} | "
        f"max_err={(ref_out - O).abs().max().item():.5f}"
    )
