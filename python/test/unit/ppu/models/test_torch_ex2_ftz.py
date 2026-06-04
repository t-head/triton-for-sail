"""
Standalone Triton test to verify FTZ (Flush-to-Zero) behavior.

Derived from the Inductor-generated kernel for aten.pow(2.0, x).
"""

import pytest
import torch
import triton
import triton.language as tl
from decimal import Decimal
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def pow_fused_pow_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = libdevice.exp2(tmp1)
    tl.store(out_ptr0 + tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK), tmp2)


@pytest.mark.xfail(
    reason="PPU does not support native FTZ ex2; ignore FTZ on llvm.nvvm.ex2.approx.ftz.f to avoid performance loss",
    strict=True,
)
def test_ex2_ftz():
    # -128.0 -> 2.0**(-128) ≈ 2.94e-39 (subnormal float32)
    x_in = torch.tensor([-128.0], dtype=torch.float32, device="cuda")
    out = torch.empty(1, dtype=torch.float32, device="cuda")

    grid = lambda meta: (1,)
    pow_fused_pow_kernel[grid](x_in, out, 1, 1)

    result = out.item()
    print(f"Input:    {x_in.item()}")
    print(f"Output:   {result}")
    print(f"Decimal:  {Decimal(result)}")
    print(f"Is zero:  {result == Decimal(0)}")
    assert result == Decimal(0)
