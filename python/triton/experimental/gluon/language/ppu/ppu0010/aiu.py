from ..._core import _unwrap_if_constexpr, builtin
from triton.language.core import int32

__all__ = ["aiu_load"]


@builtin
def aiu_load(smem, pointer, coord, shape, _semantic=None):
    """
    Asynchronously load from global memory to shared memory via the PPU AIU bulk-copy.

    Args:
        smem (shared_memory_descriptor): Destination shared memory (PPUAIUSharedLayout).
        pointer (tensor): Base global pointer (scalar).
        coord (Sequence): Per-dim i32 tile offsets into the global tensor.
        shape (Sequence): Per-dim i64 global tensor shape.
    """
    coord = [_semantic.make_scalar(_unwrap_if_constexpr(c), int32).handle for c in coord]
    shape = _semantic._convert_to_ir_values([_unwrap_if_constexpr(s) for s in shape])
    _semantic.builder.create_async_aiu_copy_global_to_local(smem.handle, pointer.handle, coord, shape)
