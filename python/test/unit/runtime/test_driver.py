import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_is_ppu_device_cached(monkeypatch):
    from triton import _utils

    calls = []
    monkeypatch.setattr(_utils.shutil, "which", lambda name: calls.append(name) or "/usr/bin/ppu-smi")
    _utils.is_ppu_device.cache_clear()
    try:
        assert _utils.is_ppu_device()
        assert _utils.is_ppu_device()
        assert calls == ["ppu-smi"]
    finally:
        _utils.is_ppu_device.cache_clear()


@pytest.mark.parametrize("ppu_device", [False, True])
def test_nv_ppu_backends_filter_by_device(monkeypatch, ppu_device):
    nvidia = triton.backends.backends["nvidia"]
    ppu = triton.backends.backends["ppu"]
    nvidia_compiler_module = importlib.import_module(nvidia.compiler.__module__)
    ppu_compiler_module = importlib.import_module(ppu.compiler.__module__)
    nvidia_driver_module = importlib.import_module(nvidia.driver.__module__)
    ppu_driver_module = importlib.import_module(ppu.driver.__module__)

    for module in (nvidia_compiler_module, ppu_compiler_module, nvidia_driver_module, ppu_driver_module):
        monkeypatch.setattr(module, "is_ppu_device", lambda: ppu_device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)

    target = GPUTarget("cuda", 80, 32)
    assert nvidia.compiler.supports_target(target) is not ppu_device
    assert ppu.compiler.supports_target(target) is ppu_device
    assert nvidia.driver.is_active() is not ppu_device
    assert ppu.driver.is_active() is ppu_device


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()
