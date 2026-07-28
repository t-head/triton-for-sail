"""Regression test: tl.math.exp2 and libdevice.exp2 in one kernel.

Both lower to the PPU libdevice symbol __ppu_exp2f but via different paths:
  * tl.math.exp2   -> math.exp2 -> NVVM        -> llvm.func @__nv_exp2f
  * libdevice.exp2 -> tt.extern_elementwise    -> llvm.func @__ppu_exp2f
The ConvertLibdeviceFuncToPPU pass renames __nv_exp2f -> __ppu_exp2f. Before the
symbol-dedup fix that rename collided with the already-present __ppu_exp2f and
make_llir failed with "redefinition of symbol named '__ppu_exp2f'".
"""
import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _exp2_kernel(x_ptr, y_ptr, N: tl.constexpr):
    off = tl.arange(0, N)
    x = tl.load(x_ptr + off)
    a = tl.math.exp2(x)      # -> __nv_exp2f -> renamed __ppu_exp2f
    b = libdevice.exp2(x)    # -> __ppu_exp2f (extern_elementwise)
    tl.store(y_ptr + off, a + b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU device")
def test_exp2_math_and_libdevice_no_symbol_redefinition():
    N = 128
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    _exp2_kernel[(1,)](x, y, N=N)  # must compile without redefinition
    torch.cuda.synchronize()

    expected = 2.0 * torch.exp2(x)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)
