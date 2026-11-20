import json
import statistics
from unittest.mock import patch

import pytest
import torch
from triton._internal_testing import is_ppu
from triton.runtime.jit import JITFunction
from triton_kernels.matmul import matmul, PrecisionConfig
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels.tensor import make_ragged_tensor_metadata


CASES = [
    {'name': 'ragged-small', 'experts': 4, 'tokens': 64, 'k': 128, 'n': 128},
    {'name': 'ragged-medium', 'experts': 8, 'tokens': 128, 'k': 256, 'n': 256},
    {'name': 'ragged-large', 'experts': 8, 'tokens': 256, 'k': 512, 'n': 512},
]


def _inputs(case):
    torch.manual_seed(0)
    x = torch.randn((case['experts'] * case['tokens'], case['k']), device='cuda', dtype=torch.float16).to(torch.float8_e5m2)
    w_enk = torch.randn((case['experts'], case['n'], case['k']), device='cuda', dtype=torch.float16).to(torch.float8_e5m2)
    metadata = make_ragged_tensor_metadata(
        torch.full((case['experts'],), case['tokens'], device='cuda', dtype=torch.int32), x.shape[0])
    return x, w_enk, w_enk.transpose(1, 2), metadata


def _execute(case, inputs, enabled, capture=False):
    x, _, w, metadata = inputs
    artifacts = []
    original_run = JITFunction.run

    def run_with_option(self, *args, **kwargs):
        kwargs['enable_tix_ldmatrix_swzl'] = enabled
        result = original_run(self, *args, **kwargs)
        if capture and self.__name__ == '_matmul':
            artifacts.append(result)
        return result

    flags = {
        'is_persistent': False,
        'split_k': 1,
        'block_m': 64,
        'block_n': 64,
        'block_k': 64,
        'group_m': 2,
        'num_warps': 4,
        'num_stages': 1,
    }
    with scoped_opt_flags_constraints(flags), patch.object(JITFunction, 'run', run_with_option):
        y = matmul(x, w, None, a_ragged_metadata=metadata,
                   precision_config=PrecisionConfig(out_dtype=torch.float16))
    return y, artifacts[-1] if capture else None


def _reference(case, inputs):
    x, w_enk, _, _ = inputs
    reference = torch.empty((case['experts'] * case['tokens'], case['n']), device='cuda', dtype=torch.float16)
    for expert in range(case['experts']):
        sl = slice(expert * case['tokens'], (expert + 1) * case['tokens'])
        reference[sl] = (x[sl].float() @ w_enk[expert].float().T).to(torch.float16)
    return reference


def _route(artifact):
    llir = artifact.asm['llir']
    return {
        'used_swzl': 'ldmatrix.swzl.sync.bulk.tensor' in llir,
        'bulk_count': llir.count('ldmatrix.swzl.sync.bulk.tensor'),
    }


def _check(case, enabled, ordinary, enabled_artifact, ordinary_artifact):
    assert _route(enabled_artifact)['used_swzl']
    assert not _route(ordinary_artifact)['used_swzl']
    assert torch.equal(enabled, ordinary)
    reference = _reference(case, _inputs(case))
    torch.testing.assert_close(ordinary, reference, atol=0.15, rtol=0)


def _timed_launch(case, inputs, enabled):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _execute(case, inputs, enabled)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


pytestmark = pytest.mark.skipif(not is_ppu() or torch.cuda.get_device_capability() != (8, 9), reason='Requires PPU0015')


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_ragged_matmul_correctness(case):
    inputs = _inputs(case)
    enabled, enabled_artifact = _execute(case, inputs, True, capture=True)
    ordinary, ordinary_artifact = _execute(case, inputs, False, capture=True)
    _check(case, enabled, ordinary, enabled_artifact, ordinary_artifact)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_ragged_matmul_performance(case, tmp_path):
    inputs = _inputs(case)
    enabled, enabled_artifact = _execute(case, inputs, True, capture=True)
    ordinary, ordinary_artifact = _execute(case, inputs, False, capture=True)
    _check(case, enabled, ordinary, enabled_artifact, ordinary_artifact)
    for _ in range(5):
        _execute(case, inputs, True)
        _execute(case, inputs, False)
    torch.cuda.synchronize()
    enabled_times = []
    ordinary_times = []
    for sample in range(15):
        routes = ((True, enabled_times), (False, ordinary_times))
        if sample % 2:
            routes = routes[::-1]
        for enabled_flag, samples in routes:
            samples.append(_timed_launch(case, inputs, enabled_flag))
    enabled_ms = statistics.median(enabled_times)
    ordinary_ms = statistics.median(ordinary_times)
    result = {
        **case,
        **_route(enabled_artifact),
        'enabled_ms': enabled_ms,
        'ordinary_ms': ordinary_ms,
        'speedup': ordinary_ms / enabled_ms,
        'enabled_range_ms': [min(enabled_times), max(enabled_times)],
        'ordinary_range_ms': [min(ordinary_times), max(ordinary_times)],
        'samples': 15,
    }
    (tmp_path / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
