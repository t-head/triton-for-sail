"""
Original ATen: aten._to_copy (prims.convert_element_type int64 -> bfloat16)

Original kernel: triton_poi_fused__to_copy_0
  Load scalar i64, cast via float32 to bfloat16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    """int64 -> bfloat16 type cast pointwise kernel (scalar input)."""
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tl.static_assert(tmp0.dtype == tl.int64)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tl.static_assert(tmp1.dtype == tl.int64)
    tmp2 = tmp1.to(tl.float32)
    tl.static_assert(tmp2.dtype == tl.float32)
    tl.static_assert(tmp2.dtype == tl.float32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp2, None)


def test_scalar_int64_to_bfloat16():
    input_val = -42
    inp = torch.tensor(input_val, dtype=torch.int64, device="cuda:0")
    out = torch.empty((), dtype=torch.bfloat16, device="cuda:0")
    grid = (1,)
    triton_poi_fused__to_copy_0[grid](inp, out, 1, 1)

    expected = torch.tensor(float(input_val), dtype=torch.bfloat16, device="cuda:0")
    assert torch.equal(out, expected), f"got {out.item()}, expected {expected.item()}"


def test_negative_value():
    input_val = 42
    inp = torch.tensor(input_val, dtype=torch.int64, device="cuda:0")
    out = torch.empty((), dtype=torch.bfloat16, device="cuda:0")
    grid = (1,)
    triton_poi_fused__to_copy_0[grid](inp, out, 1, 1)

    expected = torch.tensor(float(input_val), dtype=torch.bfloat16, device="cuda:0")
    assert torch.equal(out, expected)


def test_zero():
    input_val = 0
    inp = torch.tensor(input_val, dtype=torch.int64, device="cuda:0")
    out = torch.empty((), dtype=torch.bfloat16, device="cuda:0")
    grid = (1,)
    triton_poi_fused__to_copy_0[grid](inp, out, 1, 1)
    expected = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda:0")
    assert torch.equal(out, expected)
