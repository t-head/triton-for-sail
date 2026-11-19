from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check
from triton.language.core import _unwrap_if_constexpr

from .._layouts import DistributedLinearLayout, DotOperandLayout
from ._layouts import PPUMMADistributedLayout


def _verify_mma(version, a, b, acc):
    """ Validate operands for a PPU MMA """
    _check(acc is not None, lambda: "acc is required")

    layout = acc.type.layout
    _check(
        isinstance(layout, PPUMMADistributedLayout) and layout.version == version,
        lambda: f"Expected acc layout to be a PPUMMADistributedLayout with version {version}, got {layout}")

    a_layout = a.type.layout
    _check(
        isinstance(a_layout, DotOperandLayout) and isinstance(a_layout.parent, PPUMMADistributedLayout)
        and a_layout.parent.version == version and a_layout.operand_index == 0, lambda:
        "Expected a's layout to be a DotOperandLayout(operand_index=0) with a matching PPUMMADistributedLayout parent")

    b_layout = b.type.layout
    _check(
        isinstance(b_layout, DotOperandLayout) and isinstance(b_layout.parent, PPUMMADistributedLayout)
        and b_layout.parent.version == version and b_layout.operand_index == 1, lambda:
        "Expected b's layout to be a DotOperandLayout(operand_index=1) with a matching PPUMMADistributedLayout parent")


def _mma(version, a, b, acc, input_precision, semantic):
    """ Shared implementation for PPU MMA operations for Gluon builtins """
    _verify_mma(version, a, b, acc)
    handle = semantic.dot(a, b, acc, input_precision=input_precision, max_num_imprecise_acc=None,
                          out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)


def _dot_scaled_scale_layout_v2(mma, operand_index, operand_shape, scale_shape, semantic):
    """ Scale layout for PPU0015 MMA v2 (inline ``convertScale`` lambda in ScaledBlockedToMMAv2) """
    operand_layout = DotOperandLayout(operand_index=operand_index, parent=mma, k_width=4)
    operand_ll = semantic.to_linear_layout(operand_layout, operand_shape)
    if isinstance(operand_ll, ttgl.constexpr):
        operand_ll = operand_ll.value

    warp_bases = [list(basis) for basis in operand_ll.warp_bases]
    if operand_index == 1:
        warp_bases = [list(reversed(basis)) for basis in warp_bases]

    warps = mma.warps_per_cta
    lane_bases = [
        [16 * warps[operand_index], 0] if 16 * warps[operand_index] < scale_shape[0] else [0, 0],
        [32 * warps[operand_index], 0] if 32 * warps[operand_index] < scale_shape[0] else [0, 0],
        [1, 0],
        [2, 0],
        [4, 0],
    ]
    reg_bases = [[0, 1], [8, 0]]

    def dim_size(dim):
        num_bits = 0
        for bases in (reg_bases, lane_bases, warp_bases):
            num_bits += sum(1 for basis in bases if basis[dim] != 0)
        return 1 << num_bits

    # Fill remaining shape with register bases (dim1 first, then dim0),
    # matching `newLL *= LinearLayout::identity1D(...)` in repOrder {1, 0}.
    for dim in (1, 0):
        size = dim_size(dim)
        while scale_shape[dim] > size:
            reg_bases.append([size if d == dim else 0 for d in range(2)])
            size *= 2

    return DistributedLinearLayout(reg_bases, lane_bases, warp_bases, [], scale_shape)


def _mma_scaled(a, a_scale, a_type, b, b_scale, b_type, acc, semantic):
    """ Shared warp-level block-scaled MMA for PPU0015 (MMA v2) """
    mma = acc.type.layout
    _check(
        isinstance(mma, PPUMMADistributedLayout) and mma.version == [2, 0],
        lambda: f"Expected acc layout to be a PPUMMADistributedLayout with version [2, 0], got {mma}")

    a_type = _unwrap_if_constexpr(a_type)
    b_type = _unwrap_if_constexpr(b_type)
    _check(a_type == "e2m1" and b_type == "e2m1",
           lambda: f"PPU0015 (MMA v2) only supports fp4 (e2m1) scaled dot, got a={a_type}, b={b_type}")
    m, kpack = a.type.shape
    n = b.type.shape[1]
    _check(m >= 16 and n >= 16 and kpack >= 32,
           lambda: f"PPU scaled MMA requires M >= 16, N >= 16, packed K >= 32, got M={m}, N={n}, packedK={kpack}")

    # 32 bits per register group -> kWidth = 4 for packed fp4 (uint8).
    a = semantic.convert_layout(a, DotOperandLayout(operand_index=0, parent=mma, k_width=4))
    b = semantic.convert_layout(b, DotOperandLayout(operand_index=1, parent=mma, k_width=4))

    a_scale = semantic.convert_layout(a_scale,
                                      _dot_scaled_scale_layout_v2(mma, 0, list(a.shape), list(a_scale.shape), semantic))
    b_scale = semantic.convert_layout(b_scale,
                                      _dot_scaled_scale_layout_v2(mma, 1, list(b.shape), list(b_scale.shape), semantic))

    ret = semantic.dot_scaled(a, a_scale, a_type, b, b_scale, b_type, acc, fast_math=False, lhs_k_pack=True,
                              rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(ret.handle, acc.type)
