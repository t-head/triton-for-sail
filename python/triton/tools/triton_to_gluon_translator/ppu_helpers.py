# type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from triton.experimental.gluon.language.ppu.ppu0010 import mma_v1
from triton.experimental.gluon.language.ppu.ppu0015 import mma_v2, mma_scaled as mma_scaled_v2
from triton.language.target_info import current_target
from triton._utils import is_ppu_device

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout,
    tl_dot_decomposed_block_scales_impl,
)
from triton.tools.triton_to_gluon_translator.hopper_helpers import (
    _get_default_input_precision, )

# ---- architecture detection ----

_IS_PPU_DEVICE: ttgl.constexpr = is_ppu_device()


@gluon.constexpr_function
def _is_ppu0010(target=None):
    return _IS_PPU_DEVICE and target is not None and str(target.arch) == "80"


@gluon.constexpr_function
def _is_ppu0015(target=None):
    return _IS_PPU_DEVICE and target is not None and str(target.arch) == "89"


# ---- PPU MMA layouts ----


@gluon.constexpr_function
def get_mma_layout(shape, num_warps, version):
    rank = len(shape)
    assert rank in [2, 3], "PPU MMA supports 2D or batched 3D shapes"
    if rank == 2:
        return ttgl.ppu.PPUMMADistributedLayout(version=[version, 0], warps_per_cta=[num_warps, 1],
                                                instr_shape=[16, 16])
    return ttgl.ppu.PPUMMADistributedLayout(version=[version, 0], warps_per_cta=[num_warps, 1, 1],
                                            instr_shape=[1, 16, 16])


@gluon.constexpr_function
def get_mma_k_width(a_ty, b_ty):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    return max(32 // min_bitwidth, 1)


# ---- PPU MMA dot (ppu0010/ppu0015) ----


@gluon.jit
def tl_dot_mma(a, b, acc, input_precision, out_dtype, version: ttgl.constexpr, mma_fn: ttgl.constexpr):
    mma_layout: ttgl.constexpr = get_mma_layout(a.type.shape, ttgl.num_warps(), version)
    k_width: ttgl.constexpr = get_mma_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, mma_layout)
    else:
        accumulator = ttgl.zeros(a.shape[:-1] + [b.shape[-1]], out_dtype, layout=mma_layout)

    result = mma_fn(a, b, accumulator, input_precision)

    if acc is not None:
        layout: ttgl.constexpr = acc.type.layout
    else:
        layout: ttgl.constexpr = default_blocked_layout(result.type.shape, ttgl.num_warps())
    return ttgl.convert_layout(result, layout)


# ---- PPU dot dispatch ----


@gluon.jit
def tl_dot(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=ttgl.float32,
):
    target: ttgl.constexpr = current_target()
    input_prec: ttgl.constexpr = _get_default_input_precision(allow_tf32, input_precision)

    if _is_ppu0010(target):
        return tl_dot_mma(a, b, acc, input_prec, out_dtype, 1, mma_v1)
    else:
        return tl_dot_mma(a, b, acc, input_prec, out_dtype, 2, mma_v2)


# ---- PPU dot-scaled ----


@gluon.constexpr_function
def _scaled_native_supported(target, lhs_format, rhs_format):
    if _is_ppu0010(target):
        return False
    if lhs_format != rhs_format:
        return False
    return lhs_format == "e2m1"


@gluon.jit
def tl_dot_scaled_native(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, target: ttgl.constexpr):
    version: ttgl.constexpr = 2
    num_warps: ttgl.constexpr = ttgl.num_warps()
    mma_layout: ttgl.constexpr = ttgl.ppu.PPUMMADistributedLayout(version=[version, 0], warps_per_cta=[num_warps, 1],
                                                                  instr_shape=[16, 16])

    # _ops._mma_scaled converts operands/scales to the required DotOperand /
    # scale layouts internally, so blocked inputs are sufficient here.
    lhs = ttgl.convert_layout(lhs, default_blocked_layout(lhs.type.shape, num_warps))
    rhs = ttgl.convert_layout(rhs, default_blocked_layout(rhs.type.shape, num_warps))
    lhs_scale = ttgl.convert_layout(lhs_scale, default_blocked_layout(lhs_scale.type.shape, num_warps))
    rhs_scale = ttgl.convert_layout(rhs_scale, default_blocked_layout(rhs_scale.type.shape, num_warps))

    c_shape: ttgl.constexpr = lhs.shape[:-1] + [rhs.shape[-1]]
    if acc is None:
        ret_layout: ttgl.constexpr = default_blocked_layout(c_shape, num_warps)
        acc = ttgl.full(c_shape, 0.0, ttgl.float32, layout=mma_layout)
    else:
        ret_layout: ttgl.constexpr = acc.type.layout
        acc = ttgl.convert_layout(acc, mma_layout)

    result = mma_scaled_v2(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc)
    return ttgl.convert_layout(result, ret_layout)


@gluon.jit
def tl_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    target: ttgl.constexpr = current_target()
    native: ttgl.constexpr = _scaled_native_supported(target, lhs_format, rhs_format)
    use_native: ttgl.constexpr = native and (lhs_scale is not None) and (rhs_scale is not None)
    if use_native:
        return tl_dot_scaled_native(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, target)
    else:
        return tl_dot_decomposed_block_scales_impl(
            tl_dot_scaled,
            tl_dot,
            lhs,
            lhs_scale,
            lhs_format,
            rhs,
            rhs_scale,
            rhs_format,
            acc=acc,
            fast_math=fast_math,
            lhs_k_pack=lhs_k_pack,
            rhs_k_pack=rhs_k_pack,
            out_dtype=out_dtype,
        )
