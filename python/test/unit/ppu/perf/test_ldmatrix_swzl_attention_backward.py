import importlib.util
import json
from pathlib import Path
import statistics

import pytest
import torch
import triton
from triton._internal_testing import is_ppu


CASES = [
    {'name': 'bwd-short', 'z': 1, 'h': 2, 'n': 128, 'd': 64},
    {'name': 'bwd-medium', 'z': 1, 'h': 4, 'n': 256, 'd': 64},
    {'name': 'bwd-long', 'z': 1, 'h': 2, 'n': 1024, 'd': 64},
]


def _module():
    path = Path(__file__).with_name('06-fused-attention.py')
    spec = importlib.util.spec_from_file_location('swzl_backward_attention', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(case):
    mod = _module()
    z, h, n, d = case['z'], case['h'], case['n'], case['d']
    torch.manual_seed(20)
    q = torch.randn((z, h, n, d), device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    m = torch.empty((z, h, n), device='cuda', dtype=torch.float32)
    o = torch.empty_like(q)
    mod._attn_fwd.fn[(triton.cdiv(n, 64), z * h, 1)](
        q, k, v, 0.5, m, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        z, h, n, HEAD_DIM=d, BLOCK_M=64, BLOCK_N=64, STAGE=3,
        num_warps=4, num_stages=1, enable_tix_ldmatrix_swzl=False)
    return mod, q, k, v, dout, m, o


def _prepare(case, enabled, config=None):
    mod, q, k, v, dout, m, o = _setup(case)
    z, h, n, d = case['z'], case['h'], case['n'], case['d']
    config = config or {}
    delta = torch.empty_like(m)
    preprocess = mod._attn_bwd_preprocess[(n // 128, z * h, 1)](
        o, dout, delta, z, h, n, BLOCK_M=128, HEAD_DIM=d,
        num_warps=4, num_stages=1, enable_tix_ldmatrix_swzl=enabled)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    arg_k = k * (0.5 * 1.4426950408889634)
    grid = (n // 128, 1, z * h)
    compiled = mod._attn_bwd.fn[grid](
        q, arg_k, v, 0.5, dout, dq, dk, dv, m, delta,
        *q.stride(), H=h, N_CTX=n,
        BLOCK_M1=config.get('block_m1', 64), BLOCK_N1=128,
        BLOCK_M2=128, BLOCK_N2=config.get('block_n2', 64),
        BLK_SLICE_FACTOR=2, HEAD_DIM=d,
        num_warps=config.get('num_warps', 4), num_stages=config.get('num_stages', 1),
        enable_tix_ldmatrix_swzl=enabled)
    return {
        'mod': mod, 'q': q, 'k': k, 'v': v, 'dout': dout, 'm': m, 'o': o,
        'arg_k': arg_k, 'delta': delta, 'dq': dq, 'dk': dk, 'dv': dv,
        'compiled': compiled, 'preprocess': preprocess, 'grid': grid, 'config': config,
    }


def _launch(case, prepared):
    z, h, n, d = case['z'], case['h'], case['n'], case['d']
    prepared['preprocess'][(n // 128, z * h, 1)](
        prepared['o'], prepared['dout'], prepared['delta'], z, h, n, 128, d)
    prepared['compiled'][prepared['grid']](
        prepared['q'], prepared['arg_k'], prepared['v'], 0.5, prepared['dout'],
        prepared['dq'], prepared['dk'], prepared['dv'], prepared['m'], prepared['delta'],
        *prepared['q'].stride(), h, n,
        prepared['config'].get('block_m1', 64), 128, 128,
        prepared['config'].get('block_n2', 64), 2, d)


def _route(prepared):
    llir = prepared['compiled'].asm['llir']
    return {
        'used_swzl': 'ldmatrix.swzl.sync.bulk.tensor' in llir,
        'bulk_count': llir.count('ldmatrix.swzl.sync.bulk.tensor'),
    }


def _reference(case, prepared):
    q = prepared['q'].clone().requires_grad_()
    k = prepared['k'].clone().requires_grad_()
    v = prepared['v'].clone().requires_grad_()
    n = case['n']
    mask = torch.tril(torch.ones((n, n), device='cuda'))
    p = torch.matmul(q, k.transpose(2, 3)) * 0.5
    p[:, :, mask == 0] = float('-inf')
    p = torch.softmax(p.float(), dim=-1).half()
    out = torch.matmul(p, v)
    out.backward(prepared['dout'])
    torch.testing.assert_close(prepared['dq'], q.grad, atol=0.05, rtol=0)
    torch.testing.assert_close(prepared['dk'], k.grad, atol=0.05, rtol=0)
    torch.testing.assert_close(prepared['dv'], v.grad, atol=0.05, rtol=0)


def _check(case, enabled, ordinary):
    enabled_route = _route(enabled)
    assert enabled_route['used_swzl']
    assert enabled_route['bulk_count'] > 0
    assert not _route(ordinary)['used_swzl']
    assert torch.equal(enabled['dq'], ordinary['dq'])
    assert torch.equal(enabled['dk'], ordinary['dk'])
    assert torch.equal(enabled['dv'], ordinary['dv'])
    _reference(case, ordinary)


def _timed_launch(case, prepared):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _launch(case, prepared)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


pytestmark = pytest.mark.skipif(not is_ppu() or torch.cuda.get_device_capability() != (8, 9), reason='Requires PPU0015')


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_attention_backward_correctness(case):
    enabled = _prepare(case, True)
    ordinary = _prepare(case, False)
    _check(case, enabled, ordinary)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_attention_backward_performance(case, tmp_path):
    enabled = _prepare(case, True)
    ordinary = _prepare(case, False)
    _check(case, enabled, ordinary)
    for _ in range(5):
        _launch(case, enabled)
        _launch(case, ordinary)
    torch.cuda.synchronize()
    enabled_times = []
    ordinary_times = []
    for sample in range(15):
        routes = ((enabled, enabled_times), (ordinary, ordinary_times))
        if sample % 2:
            routes = routes[::-1]
        for prepared, samples in routes:
            samples.append(_timed_launch(case, prepared))
    enabled_ms = statistics.median(enabled_times)
    ordinary_ms = statistics.median(ordinary_times)
    result = {
        **case,
        **_route(enabled),
        'enabled_ms': enabled_ms,
        'ordinary_ms': ordinary_ms,
        'speedup': ordinary_ms / enabled_ms,
        'enabled_range_ms': [min(enabled_times), max(enabled_times)],
        'ordinary_range_ms': [min(ordinary_times), max(ordinary_times)],
        'samples': 15,
    }
    (tmp_path / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
