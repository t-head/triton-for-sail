import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires PPU device")


@triton.jit
def _trunc_mul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    reduction_ptr,
    rounded_ptr,
    BLOCK: tl.constexpr,
    TRUNC_LHS: tl.constexpr,
    MULTI_USE: tl.constexpr,
    REDUCE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    rounded = a.to(tl.bfloat16)
    if TRUNC_LHS:
        result = rounded * b
    else:
        result = b * rounded
    if MULTI_USE:
        tl.store(rounded_ptr + offsets, rounded)
    tl.store(out_ptr + offsets, result)
    if REDUCE:
        tl.store(reduction_ptr, tl.sum(result.to(tl.float32), axis=0))


def _make_inputs(device, block):
    a = (torch.arange(block, dtype=torch.float32) - block / 2) * 0.03125
    halfway = torch.tensor(
        [
            1.0 + 2**-8,
            1.0 + 3 * 2**-8,
            1.0 - 2**-9,
            -(1.0 + 2**-8),
            -(1.0 + 3 * 2**-8),
            0.0,
            -0.0,
            255.0 + 2**0,
        ],
        dtype=torch.float32,
    )
    count = min(halfway.numel(), block)
    a[:count] = halfway[:count]
    b = ((torch.arange(block, dtype=torch.float32) % 37) - 18) * 0.0625
    return a.to(device), b.to(torch.bfloat16).to(device)


def _run_trunc_mul(a, b, trunc_lhs, multi_use, reduce):
    block = a.numel()
    out = torch.empty_like(b)
    rounded = torch.empty_like(b)
    reduction = torch.empty((), dtype=torch.float32, device=a.device)
    compiled = _trunc_mul_kernel.warmup(
        a,
        b,
        out,
        reduction,
        rounded,
        BLOCK=block,
        TRUNC_LHS=trunc_lhs,
        MULTI_USE=multi_use,
        REDUCE=reduce,
        grid=(1,),
        num_warps=1,
    )
    _trunc_mul_kernel[(1,)](
        a,
        b,
        out,
        reduction,
        rounded,
        BLOCK=block,
        TRUNC_LHS=trunc_lhs,
        MULTI_USE=multi_use,
        REDUCE=reduce,
        num_warps=1,
    )
    return compiled, out, reduction


def _assert_bf16_bitwise_equal(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("trunc_lhs", [True, False])
def test_bf16_trunc_mul_packed_bitwise(device, fresh_triton_cache, trunc_lhs):
    a, b = _make_inputs(device, 256)
    compiled, actual, actual_reduction = _run_trunc_mul(a, b, trunc_lhs, False, True)
    reference_compiled, expected, expected_reduction = _run_trunc_mul(a, b, trunc_lhs, True, True)

    llir = compiled.asm["llir"]
    assert llir.count("ppu.cvt.rn.bf16x2.f32") >= 2
    assert llir.count("ppu.mul.rn.bf16x2") >= 2
    assert "ppu.mul.rn.bf16x2" not in reference_compiled.asm["llir"]
    _assert_bf16_bitwise_equal(actual, expected)
    assert torch.equal(actual_reduction.view(torch.int32), expected_reduction.view(torch.int32))


def test_bf16_trunc_mul_single_element_fallback(device, fresh_triton_cache):
    a, b = _make_inputs(device, 1)
    compiled, actual, _ = _run_trunc_mul(a, b, True, False, False)
    _, expected, _ = _run_trunc_mul(a, b, True, True, False)

    llir = compiled.asm["llir"]
    assert "ppu.cvt.rn.bf16x2.f32" not in llir
    assert "ppu.mul.rn.bf16x2" not in llir
    assert "fmul" in llir
    _assert_bf16_bitwise_equal(actual, expected)


def test_bf16_trunc_mul_multi_use_fallback(device, fresh_triton_cache):
    a, b = _make_inputs(device, 256)
    compiled, _, _ = _run_trunc_mul(a, b, False, True, False)

    llir = compiled.asm["llir"]
    assert "ppu.mul.rn.bf16x2" not in llir
    assert "fmul" in llir
