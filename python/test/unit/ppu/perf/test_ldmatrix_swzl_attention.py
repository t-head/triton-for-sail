import importlib.util
import json
from pathlib import Path
import statistics

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_ppu


@triton.jit
def _attn_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, qk_scale, lo, hi,
                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                MASKED: tl.constexpr, causal_offset, offs_m, offs_n):
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if MASKED:
            mask = (offs_m[:, None] + causal_offset) >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, L, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_os, stride_om, stride_on,
              Z, H_Q, H_KV, N_CTX_Q, N_CTX_KV,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, STAGE: tl.constexpr, NUM_SPLITS: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    split_id = tl.program_id(2)
    off_z = off_hz // H_Q
    off_h = off_hz % H_Q
    off_kh = off_h // (H_Q // H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kh.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_kh.to(tl.int64) * stride_vh
    out_offset = (off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh +
                  split_id.to(tl.int64) * stride_os)

    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX_Q, HEAD_DIM),
                                    strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    fp8_v: tl.constexpr = V.dtype.element_ty == tl.float8e5 or V.dtype.element_ty == tl.float8e4nv
    v_order: tl.constexpr = (0, 1) if fp8_v else (1, 0)
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(N_CTX_KV, HEAD_DIM),
                                    strides=(stride_vk, stride_vn), offsets=(0, 0),
                                    block_shape=(BLOCK_N, HEAD_DIM), order=v_order)
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(HEAD_DIM, N_CTX_KV),
                                    strides=(stride_kk, stride_kn), offsets=(0, 0),
                                    block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + out_offset, shape=(N_CTX_Q, HEAD_DIM),
                                    strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)

    if NUM_SPLITS > 1:
        split_size = N_CTX_KV // NUM_SPLITS
        lo = split_id * split_size
        hi = lo + split_size
        acc, l_i, m_i = _attn_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, qk_scale,
                                    lo, hi, BLOCK_M, HEAD_DIM, BLOCK_N, False, N_CTX_KV - N_CTX_Q, offs_m, offs_n)
    else:
        if STAGE & 1:
            hi = start_m * BLOCK_M if STAGE == 3 else N_CTX_KV
            acc, l_i, m_i = _attn_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, qk_scale,
                                        0, hi, BLOCK_M, HEAD_DIM, BLOCK_N, False, N_CTX_KV - N_CTX_Q, offs_m, offs_n)
        if STAGE & 2:
            lo = start_m * BLOCK_M
            hi = (start_m + 1) * BLOCK_M
            acc, l_i, m_i = _attn_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, qk_scale,
                                        lo, hi, BLOCK_M, HEAD_DIM, BLOCK_N, True, N_CTX_KV - N_CTX_Q, offs_m, offs_n)

    partial_index = (off_hz * NUM_SPLITS + split_id) * N_CTX_Q + offs_m
    if NUM_SPLITS == 1:
        tl.store(M + partial_index, m_i + tl.math.log2(l_i))
        acc = acc / l_i[:, None]
    else:
        tl.store(M + partial_index, m_i)
    tl.store(L + partial_index, l_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_split_reduce(M, L, Partial, Out, N_CTX_Q,
                       HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, NUM_SPLITS: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    for split in range(NUM_SPLITS):
        m = tl.load(M + (off_hz * NUM_SPLITS + split) * N_CTX_Q + offs_m)
        m_global = tl.maximum(m_global, m)
    denominator = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for split in range(NUM_SPLITS):
        m = tl.load(M + (off_hz * NUM_SPLITS + split) * N_CTX_Q + offs_m)
        alpha = tl.math.exp2(m - m_global)
        l = tl.load(L + (off_hz * NUM_SPLITS + split) * N_CTX_Q + offs_m)
        denominator += l * alpha
        partial = tl.load(Partial + ((off_hz * NUM_SPLITS + split) * N_CTX_Q + offs_m)[:, None] * HEAD_DIM + offs_d[None, :])
        acc += partial * alpha[:, None]
    acc = acc / denominator[:, None]
    tl.store(Out + (off_hz * N_CTX_Q + offs_m)[:, None] * HEAD_DIM + offs_d[None, :], acc.to(Out.type.element_ty))


CASES = [
    {'name': 'fp16-mha-short-causal', 'dtype': 'fp16', 'z': 1, 'hq': 2, 'hkv': 2, 'sq': 128, 'skv': 128, 'd': 64, 'splits': 1, 'causal': True},
    {'name': 'fp16-gqa-causal', 'dtype': 'fp16', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 256, 'skv': 256, 'd': 64, 'splits': 1, 'causal': True},
    {'name': 'fp16-gqa-long-causal', 'dtype': 'fp16', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 1024, 'skv': 1024, 'd': 64, 'splits': 1, 'causal': True},
    {'name': 'e5m2-gqa-causal', 'dtype': 'e5m2', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 256, 'skv': 256, 'd': 64, 'splits': 1, 'causal': True},
    {'name': 'e4m3-gqa-causal', 'dtype': 'e4m3', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 256, 'skv': 256, 'd': 64, 'splits': 1, 'causal': True},
    {'name': 'fp16-gqa-split-short', 'dtype': 'fp16', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 128, 'skv': 256, 'd': 64, 'splits': 4, 'causal': False},
    {'name': 'fp16-gqa-split-long', 'dtype': 'fp16', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 256, 'skv': 1024, 'd': 64, 'splits': 4, 'causal': False},
    {'name': 'e5m2-gqa-split-long', 'dtype': 'e5m2', 'z': 1, 'hq': 4, 'hkv': 2, 'sq': 128, 'skv': 1024, 'd': 64, 'splits': 4, 'causal': False},
]


def _dtype(name):
    return {'fp16': torch.float16, 'e5m2': torch.float8_e5m2, 'e4m3': torch.float8_e4m3fn}[name]


def _inputs(case):
    dtype = _dtype(case['dtype'])
    torch.manual_seed(20)
    q = torch.randn((case['z'], case['hq'], case['sq'], case['d']), device='cuda', dtype=torch.float16)
    k = torch.randn((case['z'], case['hkv'], case['skv'], case['d']), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    q = q.to(dtype)
    k = k.to(dtype)
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        v = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    return q, k, v.to(dtype)


def _prepare(case, inputs, enabled):
    q, k, v = inputs
    splits = case['splits']
    partial = torch.empty((case['z'], case['hq'], splits, case['sq'], case['d']), device='cuda',
                          dtype=torch.float32 if splits > 1 else q.dtype)
    m = torch.empty((case['z'], case['hq'], splits, case['sq']), device='cuda', dtype=torch.float32)
    l = torch.empty_like(m)
    out = torch.empty((case['z'], case['hq'], case['sq'], case['d']), device='cuda', dtype=q.dtype)
    grid = (triton.cdiv(case['sq'], 64), case['z'] * case['hq'], splits)
    args = (q, k, v, 0.5, m, l, partial,
            *q.stride(), *k.stride(), *v.stride(),
            partial.stride(0), partial.stride(1), partial.stride(2), partial.stride(3), partial.stride(4),
            case['z'], case['hq'], case['hkv'], case['sq'], case['skv'])
    kwargs = dict(HEAD_DIM=case['d'], BLOCK_M=64, BLOCK_N=64,
                  STAGE=3 if case['causal'] else 1, NUM_SPLITS=splits,
                  num_warps=4, num_stages=1, enable_tix_ldmatrix_swzl=enabled)
    compiled = _attn_fwd[grid](*args, **kwargs)
    reduce_compiled = None
    reduce_args = None
    if splits > 1:
        reduce_grid = (triton.cdiv(case['sq'], 64), case['z'] * case['hq'], 1)
        reduce_args = (m, l, partial, out, case['sq'])
        reduce_compiled = _attn_split_reduce[reduce_grid](
            *reduce_args, HEAD_DIM=case['d'], BLOCK_M=64, NUM_SPLITS=splits,
            num_warps=4, num_stages=1, enable_tix_ldmatrix_swzl=enabled)
    return {
        'compiled': compiled,
        'grid': grid,
        'args': args,
        'kwargs': kwargs,
        'reduce_compiled': reduce_compiled,
        'reduce_grid': reduce_grid if splits > 1 else None,
        'reduce_args': reduce_args,
        'output': out if splits > 1 else partial[:, :, 0],
    }


def _launch(prepared):
    kwargs = prepared['kwargs']
    constexpr = (kwargs['HEAD_DIM'], kwargs['BLOCK_M'], kwargs['BLOCK_N'], kwargs['STAGE'], kwargs['NUM_SPLITS'])
    prepared['compiled'][prepared['grid']](*prepared['args'], *constexpr)
    if prepared['reduce_compiled'] is not None:
        prepared['reduce_compiled'][prepared['reduce_grid']](
            *prepared['reduce_args'], kwargs['HEAD_DIM'], kwargs['BLOCK_M'], kwargs['NUM_SPLITS'])


def _reference(case, inputs):
    q, k, v = (x.float() for x in inputs)
    groups = case['hq'] // case['hkv']
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k.transpose(2, 3)) * 0.5
    if case['causal']:
        query_positions = torch.arange(case['sq'], device='cuda')[:, None]
        key_positions = torch.arange(case['skv'], device='cuda')[None, :]
        mask = key_positions <= query_positions + case['skv'] - case['sq']
        scores = scores.masked_fill(~mask, float('-inf'))
    probabilities = torch.softmax(scores, dim=-1).to(_dtype(case['dtype'])).float()
    return torch.matmul(probabilities, v).to(_dtype(case['dtype'])).float()


def _legacy_reference(case, inputs):
    q, k, v = inputs
    groups = case['hq'] // case['hkv']
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    if case['dtype'] in ('e5m2', 'e4m3'):
        v = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2).to(_dtype(case['dtype']))
    path = Path(__file__).with_name('06-fused-attention.py')
    spec = importlib.util.spec_from_file_location('swzl_legacy_attention', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if case['sq'] != case['skv']:
        q_ref = torch.zeros((case['z'], case['hq'], case['skv'], case['d']), device='cuda', dtype=q.dtype)
        q_ref[:, :, :case['sq']] = q
    else:
        q_ref = q
    out = torch.empty_like(q_ref)
    m = torch.empty((case['z'], case['hq'], case['skv']), device='cuda', dtype=torch.float32)
    grid = (triton.cdiv(case['skv'], 64), case['z'] * case['hq'], 1)
    module._attn_fwd.fn[grid](
        q_ref, k, v, 0.5, m, out,
        *q_ref.stride(), *k.stride(), *v.stride(), *out.stride(),
        case['z'], case['hq'], case['skv'],
        HEAD_DIM=case['d'], BLOCK_M=64, BLOCK_N=64,
        STAGE=3 if case['causal'] else 1,
        num_warps=4, num_stages=1, enable_tix_ldmatrix_swzl=False)
    return out[:, :, :case['sq']].float()


def _route(prepared):
    llir = prepared['compiled'].asm['llir']
    return {
        'used_swzl': 'ldmatrix.swzl.sync.bulk.tensor' in llir,
        'bulk_count': llir.count('ldmatrix.swzl.sync.bulk.tensor'),
    }


def _timed_launch(prepared):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _launch(prepared)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


pytestmark = pytest.mark.skipif(not is_ppu() or torch.cuda.get_device_capability() != (8, 9), reason='Requires PPU0015')


def _check_numerics(case, inputs, enabled, ordinary):
    enabled_route = _route(enabled)
    ordinary_route = _route(ordinary)
    assert enabled_route['used_swzl']
    assert enabled_route['bulk_count'] > 0
    assert not ordinary_route['used_swzl']
    assert torch.equal(enabled['output'], ordinary['output'])
    if case['dtype'] in ('e5m2',):
        reference = _legacy_reference(case, inputs)
    else:
        reference = _reference(case, inputs)
    tolerance = 0.3 if case['dtype'] != 'fp16' else 0.02
    torch.testing.assert_close(ordinary['output'].float(), reference, atol=tolerance, rtol=0)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_attention_swzl_correctness(case):
    inputs = _inputs(case)
    enabled = _prepare(case, inputs, True)
    ordinary = _prepare(case, inputs, False)
    _check_numerics(case, inputs, enabled, ordinary)


def test_attention_wide_b8_fallback():
    case = {'name': 'wide-b8-fallback', 'dtype': 'e4m3', 'z': 1, 'hq': 8, 'hkv': 1,
            'sq': 256, 'skv': 256, 'd': 256, 'splits': 1, 'causal': True}
    inputs = _inputs(case)
    enabled = _prepare(case, inputs, True)
    ordinary = _prepare(case, inputs, False)
    assert not _route(enabled)['used_swzl']
    assert not _route(ordinary)['used_swzl']
    assert torch.equal(enabled['output'], ordinary['output'])
    reference = _reference(case, inputs)
    torch.testing.assert_close(ordinary['output'].float(), reference, atol=0.3, rtol=0)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_attention_swzl_performance(case, tmp_path):
    inputs = _inputs(case)
    enabled = _prepare(case, inputs, True)
    ordinary = _prepare(case, inputs, False)
    _check_numerics(case, inputs, enabled, ordinary)
    for _ in range(5):
        _launch(enabled)
        _launch(ordinary)
    torch.cuda.synchronize()
    enabled_times = []
    ordinary_times = []
    for sample in range(15):
        routes = ((enabled, enabled_times), (ordinary, ordinary_times))
        if sample % 2:
            routes = routes[::-1]
        for prepared, samples in routes:
            samples.append(_timed_launch(prepared))
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
