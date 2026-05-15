# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from typing import Optional


device = 'cuda' if torch.cuda.is_available() else 'cpu'

_is_amd = torch.cuda.is_available() and torch.version.hip is not None
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if _is_amd else [4, 8, 16, 32]


def get_multiprocessor_count(device_index=0):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def assert_close(name, ref, tri, ratio):
    abs_err = (ref.float() - tri.float()).abs().max().item()
    threshold = ratio * ref.float().abs().max().item() + 1e-6
    assert abs_err < threshold, (
        f"[{name}] abs_err={abs_err:.6f} threshold={threshold:.6f}\n"
        f"  ref : max={ref.float().abs().max():.4f} mean={ref.float().abs().mean():.4f}\n"
        f"  tri : max={tri.float().abs().max():.4f} mean={tri.float().abs().mean():.4f}"
    )


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Returns flat tensor of shape [total_chunks * 2]:
      [seq_idx_0, chunk_idx_0, seq_idx_1, chunk_idx_1, ...]
    The kernel reads: chunk_indices[i_t*2] and chunk_indices[i_t*2+1]
    """
    pairs = []
    for i in range(len(cu_seqlens) - 1):
        bos = cu_seqlens[i].item()
        eos = cu_seqlens[i + 1].item()
        seq_len = eos - bos
        num_chunks = triton.cdiv(seq_len, chunk_size)
        for c in range(num_chunks):
            pairs.append(i)
            pairs.append(c)
    return torch.tensor(pairs, dtype=torch.long, device=cu_seqlens.device)


def causal_conv1d_ref_torch(
    x,
    weight,
    bias=None,
    initial_state=None,
    activation=None,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_state is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W', 'NB'],
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_n = i_b
        bos = (i_b * T).to(tl.int64)
        eos = (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(
                x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
            )
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    elif i_t * BT >= W:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(
                x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
            )
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(
                x + bos * D + o_x[:, None] * D + o_d,
                mask=m_x, other=0
            ).to(tl.float32)

            b_yi += tl.load(
                initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None],
                mask=m_c, other=0
            ).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(
            residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
        )
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(
        y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
    )
    tl.store(
        p_y,
        tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'),
        boundary_check=(0, 1)
    )


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, D = x.shape
    W = weight.shape[1]
    BT = min(
        64,
        triton.next_power_of_2(
            triton.cdiv(max(16, B * T), get_multiprocessor_count(x.device.index))
        )
    )
    BW = triton.next_power_of_2(W)
    NB = triton.cdiv(B * T, 1024)

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices) // 2
    else:
        NT = triton.cdiv(T, BT)

    y = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), NT, B)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    return y


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test))
        for test in [
            (4, 500, 128, 3, "swish", True, True, torch.float32),
            (4, 1024, 200, 4, "swish", False, True, torch.float32),
            (4, 500, 128, 3, None, True, False, torch.float16),
            (4, 1024, 1024, 4, None, False, False, torch.float16),
        ]
    ]
)
@torch.no_grad()
def test_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, D).to(device, dtype)
    weight = torch.randn(D, W).to(device, dtype)
    bias = torch.randn(D).to(device, dtype) if has_bias else None
    residual = x.clone() if has_residual else None

    # Reference: run each segment independently
    ref = torch.cat([
        rearrange(
            causal_conv1d_ref_torch(
                x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                weight=weight,
                bias=bias,
                activation=activation,
            ),
            "b d t -> b t d"
        ) + (residual[:, bos:eos] if has_residual else torch.zeros_like(x[:, bos:eos]))
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], dim=1)

    # Triton forward
    tri = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )

    assert_close("y", ref, tri, 1e-3)
