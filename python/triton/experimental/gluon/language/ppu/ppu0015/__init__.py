from __future__ import annotations

from ..._core import builtin, _unwrap_if_constexpr
from .. import _ops
from ..ppu0010 import aiu, async_copy, mbarrier

__all__ = ["aiu", "async_copy", "mbarrier", "mma_v2", "mma_scaled"]


@builtin
def mma_v2(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    return _ops._mma([2, 0], a, b, acc, input_precision, _semantic)


@builtin
def mma_scaled(a, a_scale, a_type, b, b_scale, b_type, acc, _semantic=None):
    """
    Perform a warp-level block-scaled matrix multiply-accumulate (fp4 only) on PPU0015.

    acc = (a * a_scale) @ (b * b_scale) + acc

    Args:
        a (tensor): LHS in packed MXFP4; two e2m1 elements per byte (first
            element in lower bits), shape ``[M, K // 2]`` uint8 for K logical
            elements.
        a_scale (tensor): e8m0 scale for ``a``, shape ``[M, K // 32]``,
            row-major order.
        a_type (str): Element type of ``a``; must be ``"e2m1"`` on PPU0015.
        b (tensor): RHS in packed MXFP4; packed along K, shape ``[K // 2, N]``
            uint8.
        b_scale (tensor): e8m0 scale for ``b``, shape ``[N, K // 32]``,
            row-major order (do NOT transpose).
        b_type (str): Element type of ``b``; must be ``"e2m1"`` on PPU0015.
        acc (tensor): fp32 accumulator of shape ``[M, N]`` with a
            ``PPUMMADistributedLayout(version=[2, 0], instr_shape=[16, 16])``.
    """
    a = _unwrap_if_constexpr(a)
    a_scale = _unwrap_if_constexpr(a_scale)
    b = _unwrap_if_constexpr(b)
    b_scale = _unwrap_if_constexpr(b_scale)
    acc = _unwrap_if_constexpr(acc)
    return _ops._mma_scaled(a, a_scale, a_type, b, b_scale, b_type, acc, _semantic)
