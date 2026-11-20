import json
import statistics
import time

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget
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


@triton.jit
def dot_loop_kernel(A, B, C, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rows[:, None] * K + k + reduction[None, :])
        b = tl.load(B + (k + reduction[:, None]) * BN + cols[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * BN + cols[None, :], acc)


@triton.jit
def dot_tiled_kernel(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rows[:, None] * K + k + reduction[None, :], mask=rows[:, None] < M, other=0.0)
        b = tl.load(B + (k + reduction[:, None]) * N + cols[None, :], mask=cols[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


def _dtype_info(name):
    return {
        'fp16': (torch.float16, 'fp16'),
        'bf16': (torch.bfloat16, 'bf16'),
        'e4m3': (torch.float8_e4m3fn, 'fp8e4nv'),
    }[name]


def _compile(case, enable, tmp_path, route):
    name = case['dtype']
    dtype, type_name = _dtype_info(name)
    bm, bn, bk = case['bm'], case['bn'], case['bk']
    constants = {'BM': bm, 'BN': bn, 'BK': bk}
    signature = {'A': '*' + type_name, 'B': '*' + type_name, 'C': '*fp32'}
    if case.get('tiled'):
        signature.update({'M': 'i32', 'N': 'i32', 'K': 'i32'})
        source = ASTSource(dot_tiled_kernel, signature, constants)
    elif case['repeat'] > 1:
        signature['K'] = 'i32'
        source = ASTSource(dot_loop_kernel, signature, constants)
    else:
        source = ASTSource(dot_kernel, signature, constants)
    target = GPUTarget('cuda', 89, 32)
    options = {
        'num_warps': case['warps'],
        'num_stages': 1,
        'enable_tix_ldmatrix_swzl': enable,
    }
    start = time.perf_counter()
    compiled = triton.compile(source, target=target, options=options)
    mode = case['mode']
    per_phase, max_phase = (1, 8) if mode == 0 else (2, 4)
    vec = 128 // dtype.itemsize // 8
    expected = f'vec = {vec}, perPhase = {per_phase}, maxPhase = {max_phase}'
    ttgir = compiled.asm['ttgir'].replace('perPhase = 4, maxPhase = 2', f'perPhase = {per_phase}, maxPhase = {max_phase}')
    assert ttgir.count(expected) == 1
    override = tmp_path / f"{case['name']}-{route}.ttgir"
    override.write_text(ttgir)
    compiled = triton.compile(str(override), target=target, options=options)
    compile_seconds = time.perf_counter() - start
    used = 'ldmatrix.swzl.sync.bulk.tensor' in compiled.asm['llir']
    assert used == enable
    return compiled, compile_seconds


def _inputs(case):
    dtype, _ = _dtype_info(case['dtype'])
    bm, bn, bk = case['bm'], case['bn'], case['bk']
    if case.get('tiled'):
        bm, bn, k = case['M'], case['N'], case['K']
    else:
        k = bk * case['repeat']
    torch.manual_seed(0)
    a = torch.randn((bm, k), device='cuda').to(dtype)
    b = torch.randn((k, bn), device='cuda').to(dtype)
    c = torch.empty((bm, bn), device='cuda', dtype=torch.float32)
    return a, b, c


def _launch(compiled, case, inputs):
    a, b, c = inputs
    if case.get('tiled'):
        args = (a, b, c, case['M'], case['N'], case['K'])
        grid = (triton.cdiv(case['M'], case['bm']) * triton.cdiv(case['N'], case['bn']), 1, 1)
    elif case['repeat'] > 1:
        args = (a, b, c, case['bk'] * case['repeat'])
        grid = (1, 1, 1)
    else:
        args = (a, b, c)
        grid = (1, 1, 1)
    compiled[grid](*args)


def _timed_launch(compiled, case, inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _launch(compiled, case, inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


CASES = [
    {'name': 'fp16-small-full', 'dtype': 'fp16', 'bm': 64, 'bn': 64, 'bk': 64, 'repeat': 1, 'mode': 0, 'warps': 1},
    {'name': 'fp16-medium-full', 'dtype': 'fp16', 'bm': 128, 'bn': 128, 'bk': 64, 'repeat': 1, 'mode': 0, 'warps': 4},
    {'name': 'fp16-large-192-tiled', 'dtype': 'fp16', 'M': 192, 'N': 192, 'K': 192, 'bm': 64, 'bn': 64, 'bk': 64, 'repeat': 3, 'mode': 0, 'warps': 4, 'tiled': True},
    {'name': 'fp16-medium-half', 'dtype': 'fp16', 'bm': 64, 'bn': 64, 'bk': 32, 'repeat': 1, 'mode': 1, 'warps': 4},
    {'name': 'bf16-medium-full', 'dtype': 'bf16', 'bm': 128, 'bn': 128, 'bk': 64, 'repeat': 1, 'mode': 0, 'warps': 4},
    {'name': 'fp8-medium-full', 'dtype': 'e4m3', 'bm': 128, 'bn': 128, 'bk': 128, 'repeat': 1, 'mode': 0, 'warps': 4},
    {'name': 'fp8-medium-half', 'dtype': 'e4m3', 'bm': 64, 'bn': 64, 'bk': 64, 'repeat': 1, 'mode': 1, 'warps': 4},
    {'name': 'fp16-repeated-full', 'dtype': 'fp16', 'bm': 64, 'bn': 64, 'bk': 64, 'repeat': 4, 'mode': 0, 'warps': 4},
    {'name': 'fp8-repeated-full', 'dtype': 'e4m3', 'bm': 128, 'bn': 128, 'bk': 128, 'repeat': 4, 'mode': 0, 'warps': 4},
]

pytestmark = pytest.mark.skipif(not is_ppu() or torch.cuda.get_device_capability() != (8, 9), reason='Requires PPU0015')


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_ldmatrix_swzl_forced_route_perf(case, tmp_path):
    enabled, enabled_compile = _compile(case, True, tmp_path, 'on')
    ordinary, ordinary_compile = _compile(case, False, tmp_path, 'off')
    enabled._init_handles()
    ordinary._init_handles()
    inputs = _inputs(case)
    _launch(enabled, case, inputs)
    torch.cuda.synchronize()
    enabled_output = inputs[2].clone()
    _launch(ordinary, case, inputs)
    torch.cuda.synchronize()
    assert torch.equal(enabled_output, inputs[2])

    for _ in range(5):
        _launch(enabled, case, inputs)
        _launch(ordinary, case, inputs)
    torch.cuda.synchronize()
    enabled_times = []
    ordinary_times = []
    for sample in range(15):
        routes = ((enabled, enabled_times), (ordinary, ordinary_times))
        if sample % 2:
            routes = routes[::-1]
        for compiled, samples in routes:
            samples.append(_timed_launch(compiled, case, inputs))

    enabled_median = statistics.median(enabled_times)
    ordinary_median = statistics.median(ordinary_times)
    operations = 2 * case.get('M', case['bm']) * case.get('N', case['bn']) * case.get('K', case['bk'] * case['repeat'])
    result = {
        **case,
        'enabled_ms': enabled_median,
        'ordinary_ms': ordinary_median,
        'speedup': ordinary_median / enabled_median,
        'enabled_range_ms': [min(enabled_times), max(enabled_times)],
        'ordinary_range_ms': [min(ordinary_times), max(ordinary_times)],
        'enabled_tflops': operations / (enabled_median * 1e-3) / 1e12,
        'ordinary_tflops': operations / (ordinary_median * 1e-3) / 1e12,
        'enabled_compile_s': enabled_compile,
        'ordinary_compile_s': ordinary_compile,
        'enabled_regs': getattr(enabled, 'n_regs', None),
        'ordinary_regs': getattr(ordinary, 'n_regs', None),
        'enabled_spills': getattr(enabled, 'n_spills', None),
        'ordinary_spills': getattr(ordinary, 'n_spills', None),
        'samples': 15,
    }
    (tmp_path / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
