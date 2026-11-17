from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import DistributedLayout, SharedLayout
from triton._C.libtriton.gluon_ir import compute_aiu_load_strategy

__all__ = [
    "PPUMMADistributedLayout",
    "PPUAIUSharedLayout",
]


@dataclass(frozen=True, eq=True)
class PPUMMADistributedLayout(DistributedLayout):
    """
    Represents a distributed layout for PPU MMA (tensor core) operation outputs.

    Args:
        version (List[int]): MMA version.
        warps_per_cta (List[int]): Number of warps per CTA along each dim.
        instr_shape (List[int]): MMA instruction shape.
        cga_layout (List[List[int]]): Bases describing CTA tiling.
        vec_size (int): Accumulator register vector width. PPU0010 MMAv1 only;

    Current supported versions:

    - 1: PPU0010
    - 2: PPU0015
    """
    version: List[int]
    warps_per_cta: List[int]
    instr_shape: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)
    vec_size: int = 2

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        super().__setattr__("vec_size", _unwrap_if_constexpr(self.vec_size))

        assert len(self.version) == 2, \
            f"PPUMMADistributedLayout version must be [major, minor], got {self.version}"
        assert 1 <= self.version[0] <= 2, \
            f"PPUMMADistributedLayout version must be in the [1, 2] range, got {self.version}"

    def _to_ir(self, builder):
        return builder.get_ppu_mma_layout(
            self.version[0],
            self.version[1],
            self.warps_per_cta,
            self.cga_layout,
            self.instr_shape,
            self.vec_size,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return (f"PPUMMA_{self.version}_{self.warps_per_cta}_"
                f"{self.instr_shape}_{cga_layout}_{self.vec_size}_PPUMMA")

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta), tuple(self.instr_shape),
                     tuple(tuple(vec) for vec in self.cga_layout) if self.cga_layout else None, self.vec_size))

    @property
    def rank(self):
        return len(self.warps_per_cta)


@dataclass(frozen=True)
class PPUAIUSharedLayout(SharedLayout):
    """
    Represents a shared memory layout suitable for PPU AIU operations.

    Args:
        version_major (int): AIU generation, 1 = PPU0010, 2 = PPU0015.
        aiu_strategy (List[int]): AIU load strategy vector computed from the
            tile shape / element width (see ``AIUUtility.h`` for more).
        order (List[int]): Memory order, fastest-varying dim first.
        cga_layout (List[List[int]]): Bases describing CTA tiling.
        k_offset (int): Offset along the K dimension.  

    Current supported versions:

    - 1: PPU0010
    - 2: PPU0015
            
    """
    version_major: int
    aiu_strategy: List[int]
    order: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)    
    k_offset: int = 0


    def __post_init__(self):
        super().__setattr__("version_major", _unwrap_if_constexpr(self.version_major))
        super().__setattr__("aiu_strategy", _unwrap_if_constexpr(self.aiu_strategy))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))        
        super().__setattr__("k_offset", _unwrap_if_constexpr(self.k_offset))


        assert self.version_major in (1, 2), \
            f"PPUAIUSharedLayout version_major must be 1 or 2, got {self.version_major}"

    @staticmethod
    def get_default_for(num_warps, shape, element_bitwidth, version_major, order=None, cga_layout=None, k_offset=0):
        """
        Build a PPUAIUSharedLayout with the AIU load strategy computed by the
        backend (host-side ``AIULoadStrategy``).
        """

        shape = [_unwrap_if_constexpr(s) for s in shape]
        rank = len(shape)
        assert rank >= 2, f"AIU load requires a tile of rank >= 2, got {rank}"
        if order is None:
            order = list(range(rank))[::-1]
        order = [_unwrap_if_constexpr(o) for o in order]

        # Mirror AIULowering.cpp
        tile_c = shape[rank - 1]
        tile_w = shape[rank - 2]
        if order[rank - 1] != 0:
            tile_c = shape[rank - 2]
            tile_w = shape[rank - 1]

        element_bitwidth = _unwrap_if_constexpr(element_bitwidth)
        num_warps = _unwrap_if_constexpr(num_warps)
        version_major = _unwrap_if_constexpr(version_major)
        elem_bytes = element_bitwidth // 8
        strategy = compute_aiu_load_strategy(num_warps, tile_w, tile_c, elem_bytes, version_major)
        if strategy is None:
            raise ValueError(
                f"unsupported AIU load tile for version {version_major}: "
                f"contiguous width {tile_w} elems, channel width {tile_c * elem_bytes} bytes, "
                f"num_warps {num_warps}")
        return PPUAIUSharedLayout(version_major=version_major, aiu_strategy=strategy, order=order,
                                  cga_layout=cga_layout or [], k_offset=k_offset)

    def _to_ir(self, builder):
        return builder.get_ppu_aiu_shared_layout(
            self.version_major,
            self.aiu_strategy,
            self.order,
            self.cga_layout,
            self.k_offset,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return (f"PPUAIU_{self.version_major}_{self.aiu_strategy}_{self.order}_"
                f"{cga_layout}_{self.k_offset}_PPUAIU")

    def __hash__(self):
        return hash((self.version_major, tuple(self.aiu_strategy), tuple(self.order), self.k_offset,
                     tuple(tuple(vec) for vec in self.cga_layout) if self.cga_layout else None))
