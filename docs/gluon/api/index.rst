API Reference
=============

.. toctree::
   :maxdepth: 1
   :caption: Gluon
   :hidden:

   Runtime API <runtime>
   Language API <language>
   NVIDIA Intrinsics <nvidia>
   AMD Intrinsics <amd>
   PPU Intrinsics <ppu>

.. rubric:: Core APIs

- :doc:`Runtime API <runtime>`: JIT decorators, result handling, and host-side
  tensor descriptors used to launch Gluon kernels.
- :doc:`Language API <language>`: the core Gluon programming model, layouts,
  memory operations, math operations, and compile-time helpers.

.. rubric:: Target Intrinsics

- :doc:`NVIDIA Intrinsics <nvidia>`: target-specific APIs for NVIDIA Ampere,
  Hopper, and Blackwell GPUs.
- :doc:`AMD Intrinsics <amd>`: target-specific APIs for AMD CDNA and RDNA GPU
  generations.
- :doc:`PPU Intrinsics <ppu>`: target-specific APIs for PPU0010 and PPU0015
  generations.
