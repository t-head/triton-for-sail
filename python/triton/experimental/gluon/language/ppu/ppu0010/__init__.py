from __future__ import annotations

from triton.language.core import float16

from ..._core import builtin, _unwrap_if_constexpr
from ..._layouts import DotOperandLayout
from .. import _ops
from .._layouts import PPUMMADistributedLayout
from . import aiu, async_copy, mbarrier

__all__ = ["aiu", "async_copy", "mbarrier", "mma_v1"]


@builtin
def mma_v1(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    # PPU0010 MMAv1 uses vec_size=2 only for f16 accumulators; f32 needs 1.
    mma = acc.type.layout
    vec_size = 2 if acc.dtype == float16 else 1
    if isinstance(mma, PPUMMADistributedLayout) and mma.vec_size != vec_size:
        mma = PPUMMADistributedLayout(version=mma.version, warps_per_cta=mma.warps_per_cta,
                                      instr_shape=mma.instr_shape, cga_layout=mma.cga_layout,
                                      vec_size=vec_size)
        a = _semantic.convert_layout(a, DotOperandLayout(operand_index=0, parent=mma, k_width=a.type.layout.k_width))
        b = _semantic.convert_layout(b, DotOperandLayout(operand_index=1, parent=mma, k_width=b.type.layout.k_width))
        acc = _semantic.convert_layout(acc, mma)
    return _ops._mma([1, 0], a, b, acc, input_precision, _semantic)
