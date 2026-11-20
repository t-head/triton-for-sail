import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget
from triton.backends.ppu.compiler import HGGCOptions
from triton.compiler import ASTSource


@triton.jit
def dot_kernel(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    a = tl.load(A + rows[:, None] * BK + reduction[None, :])
    b = tl.load(B + reduction[:, None] * BN + cols[None, :])
    acc = tl.dot(a, b)
    tl.store(C + rows[:, None] * BN + cols[None, :], acc)


def _dtype_info(name):
    return {
        'fp16': (torch.float16, 'fp16', torch.float32, 'fp32'),
        'bf16': (torch.bfloat16, 'bf16', torch.float32, 'fp32'),
        'e4m3': (torch.float8_e4m3fn, 'fp8e4nv', torch.float32, 'fp32'),
        'e5m2': (torch.float8_e5m2, 'fp8e5', torch.float32, 'fp32'),
        'int8': (torch.int8, 'i8', torch.int32, 'i32'),
    }[name]


def _tile(name, mode):
    if name in ('fp16', 'bf16'):
        return 64 if mode == 0 else 32
    return 128 if mode == 0 else 64


def _compile_dot(name, mode, warps, enable, tmp_path, artifact, shape=None):
    dtype, type_name, _, out_type = _dtype_info(name)
    tile = _tile(name, mode)
    bm, bn, bk = shape or (tile, tile, tile)
    source = ASTSource(
        dot_kernel,
        {'A': '*' + type_name, 'B': '*' + type_name, 'C': '*' + out_type},
        {'BM': bm, 'BN': bn, 'BK': bk},
    )
    target = GPUTarget('cuda', 89, 32)
    options = {'num_warps': warps, 'num_stages': 1, 'enable_tix_ldmatrix_swzl': enable}
    compiled = triton.compile(source, target=target, options=options)
    per_phase, max_phase = (1, 8) if mode == 0 else (2, 4)
    vec = 128 // dtype.itemsize // 8
    expected = f'vec = {vec}, perPhase = {per_phase}, maxPhase = {max_phase}'
    ttgir = compiled.asm['ttgir'].replace('perPhase = 4, maxPhase = 2', f'perPhase = {per_phase}, maxPhase = {max_phase}')
    assert ttgir.count(expected) == 1
    override = tmp_path / f'{artifact}.ttgir'
    override.write_text(ttgir)
    return triton.compile(str(override), target=target, options=options)


def _inputs(name, mode, shape=None):
    dtype, _, out_dtype, _ = _dtype_info(name)
    tile = _tile(name, mode)
    bm, bn, bk = shape or (tile, tile, tile)
    torch.manual_seed(0)
    if name == 'int8':
        a = torch.randint(-128, 128, (bm, bk), device='cuda', dtype=dtype)
        b = torch.randint(-128, 128, (bk, bn), device='cuda', dtype=dtype)
    else:
        a = torch.randn((bm, bk), device='cuda').to(dtype)
        b = torch.randn((bk, bn), device='cuda').to(dtype)
    return a, b, torch.empty((bm, bn), device='cuda', dtype=out_dtype)


def _launch(compiled, inputs):
    a, b, c = inputs
    compiled._init_handles()
    compiled[(1, 1, 1)](a, b, c)
    torch.cuda.synchronize()
    return c.clone()


def _check_route(compiled, expected):
    used = 'ldmatrix.swzl.sync.bulk.tensor' in compiled.asm['llir']
    assert used == expected
    alignment = 128 if expected else 16
    assert f'@global_smem = external addrspace(3) global [0 x i8], align {alignment}' in compiled.asm['llir']


pytestmark = pytest.mark.skipif(not is_ppu() or torch.cuda.get_device_capability() != (8, 9), reason='Requires PPU0015')


@pytest.mark.parametrize('name', ['fp16', 'bf16', 'e4m3', 'e5m2', 'int8'])
@pytest.mark.parametrize('mode', [0, 1])
def test_ldmatrix_swzl_numerics(name, mode, tmp_path):
    enabled = _compile_dot(name, mode, 1, True, tmp_path, 'enabled')
    ordinary = _compile_dot(name, mode, 1, False, tmp_path, 'ordinary')
    _check_route(enabled, True)
    _check_route(ordinary, False)
    inputs = _inputs(name, mode)
    enabled_c = _launch(enabled, inputs)
    ordinary_c = _launch(ordinary, inputs)
    assert torch.equal(enabled_c, ordinary_c)


@pytest.mark.parametrize('name,mode,warps', [
    ('fp16', 0, 4),
    ('fp16', 0, 8),
    ('fp16', 1, 4),
    ('fp16', 1, 8),
    ('e4m3', 0, 4),
])
def test_ldmatrix_swzl_warps(name, mode, warps, tmp_path):
    enabled = _compile_dot(name, mode, warps, True, tmp_path, 'enabled')
    ordinary = _compile_dot(name, mode, warps, False, tmp_path, 'ordinary')
    _check_route(enabled, True)
    _check_route(ordinary, False)
    inputs = _inputs(name, mode)
    enabled_c = _launch(enabled, inputs)
    ordinary_c = _launch(ordinary, inputs)
    assert torch.equal(enabled_c, ordinary_c)


@pytest.mark.parametrize('name', ['fp16', 'bf16'])
def test_ldmatrix_swzl_wide_b_fallback(name, tmp_path):
    shape = (128, 128, 64)
    enabled = _compile_dot(name, 0, 4, True, tmp_path, 'enabled', shape)
    ordinary = _compile_dot(name, 0, 4, False, tmp_path, 'ordinary', shape)
    _check_route(enabled, True)
    _check_route(ordinary, False)
    assert 'x4.trans.b16' not in enabled.asm['llir']
    inputs = _inputs(name, 0, shape)
    enabled_c = _launch(enabled, inputs)
    ordinary_c = _launch(ordinary, inputs)
    assert torch.equal(enabled_c, ordinary_c)


def test_ldmatrix_swzl_hash_and_final_off(tmp_path):
    off_options = HGGCOptions(enable_tix_ldmatrix_swzl=False)
    on_options = HGGCOptions(enable_tix_ldmatrix_swzl=True)
    assert off_options.hash() == HGGCOptions(enable_tix_ldmatrix_swzl=False).hash()
    assert off_options.hash() != on_options.hash()

    off = _compile_dot('fp16', 0, 1, False, tmp_path, 'off')
    on = _compile_dot('fp16', 0, 1, True, tmp_path, 'on')
    final_off = _compile_dot('fp16', 0, 1, False, tmp_path, 'final-off')
    _check_route(off, False)
    _check_route(on, True)
    _check_route(final_off, False)
    assert final_off.asm['llir'] == off.asm['llir']
