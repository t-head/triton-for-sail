from typing import Optional

import torch
import triton
import triton.language as tl


exp = tl.exp


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)

    p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H*V, 1), (i_c * BC, i_v * BV), (BC, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all*H + boc * H + i_h) * K, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_v * all*H + boc * H + i_h) * V, (TC, V), (H*V, 1), (i_c * BC, i_v * BV), (BC, BV), (1, 0))

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)

    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)

        p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [G]
        b_lse = tl.load(p_lse)
        b_delta = tl.load(p_delta)
        # [BC, G]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        # [BC, G] @ [G, BV] -> [BC, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BC, BV] @ [BV, G] -> [BC, G]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BC, G] @ [G, BK] -> [BC, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def parallel_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])


def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)


def test_parallel_nsa_compression_bwd_kernel_dkv():
    B = 1
    T = 4096
    H = 4
    HQ = 64
    G = HQ // H   # 16
    K = 64
    V = 64
    BS = 64
    BC = 64
    BK = 64
    BV = 64
    NV = triton.cdiv(V, BV)
    scale = K ** -0.5
    device = torch.device('cuda:0')
    cu_seqlens = torch.tensor([0, 1024, 2048, 3072, 4096], dtype=torch.int32, device=device)
    TC = triton.cdiv(T, BS)

    q = torch.randn(B, T, HQ, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, TC, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, TC, H, V, dtype=torch.bfloat16, device=device)
    o = torch.randn(B, T, HQ, V, dtype=torch.bfloat16, device=device)
    do = torch.randn(B, T, HQ, V, dtype=torch.bfloat16, device=device)
    lse = torch.randn(B, T, HQ, dtype=torch.float32, device=device)

    delta = parallel_attn_bwd_preprocess(o, do)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS)
    token_indices = prepare_token_indices(cu_seqlens)
    dk = torch.empty(NV, B, TC, H, K, dtype=torch.bfloat16, device=device)
    dv = torch.empty(B, TC, H, V, dtype=torch.bfloat16, device=device)

    NC = len(chunk_indices)
    grid = (NV, NC, B * H)
    parallel_nsa_compression_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    torch.cuda.synchronize()

    print("success")
    return dk, dv


if __name__ == '__main__':
    dk, dv = test_parallel_nsa_compression_bwd_kernel_dkv()
