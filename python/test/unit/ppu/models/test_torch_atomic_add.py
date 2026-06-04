"""
Minimal self-contained Triton kernel: atomic_xchg publish -> atomic_add spin.

Two thread blocks:
  Program 0: atomic_xchg(ws[0], 0xDEAD)
  Program 1: while val == 0: val = atomic_add(ws[0], 0)

Used to debug PPU behavior for cross-thread-block
atomic_xchg -> load visibility.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def atomic_xchg_load_spin_kernel(ws_ptr):
    idx = tl.program_id(0)
    ws64 = ws_ptr.to(tl.pointer_type(tl.uint64))

    if idx == 0:
        tl.atomic_xchg(ws64, tl.full([], 0xDEAD, tl.uint64), sem="relaxed")

    if idx == 1:
        val = tl.full([], 0, tl.uint64)
        while val == 0:
            val = tl.atomic_add(ws64, 0, sem="relaxed")


def test_atomic_xchg_load_spin():
    # this ut passes as long as the kernel does not hang and completes all rounds.
    for _ in range(100):
        ws = torch.zeros(2 * 8, dtype=torch.uint8, device="cuda:0")
        atomic_xchg_load_spin_kernel[(2, 1)](ws)
        torch.cuda.synchronize()
