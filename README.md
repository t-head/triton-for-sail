# Triton for SAIL

> For Triton's language features, compiler design philosophy, general installation, and debugging tips, please refer to the [upstream Triton README](README.triton.md) and the [official Triton documentation](https://triton-lang.org). This document **focuses on the PPU backend's extensions and optimizations relative to upstream Triton**.

## 1. Overview

This repository is the **PPU fork** of Triton, providing support for the PPU independently developed by T-Head Semiconductor.

- **Target hardware**: T-Head PPU, covering two generations of Tensor Core — PPU0010 (MMAv1) and PPU0015 (MMAv2).
- **Runtime**: Requires the T-Head SAIL SDK (PPU SDK) environment.
- **Programming interface**: Write kernels using exactly the same Python programming interface as upstream Triton; the PPU backend transparently handles compilation and execution targeting PPU hardware.

## 2. Core Features and Optimizations

The PPU backend follows Triton's standard multi-stage compilation flow: ttir → ttgir → llir → hgbin, inserting PPU-specific passes at each stage. The core implementation lives in [`third_party/ppu/backend/compiler.py`](third_party/ppu/backend/compiler.py).

Key optimizations on PPU:

- **AIU asynchronous data movement**: Data loads from global memory to shared memory (TSM) are automatically converted into asynchronous copies performed by the AIU (AI accelerator in compute Unit). Combined with software pipelining, this implements multi-buffering inside loops, overlapping tile prefetch with MMA computation to hide global memory access latency and improve compute efficiency.

- **Swizzled shared memory layout**: Triton automatically derives the tiling scheme and swizzle encoding from the number of warps, the tile shape, and the element bit width. On one hand, this eliminates shared memory bank conflicts and guarantees effective bandwidth under concurrent multi-warp access; on the other hand, it aligns the data layout in TSM with the MMA operand layout, reducing extra layout conversion overhead.

- **Tensor Core acceleration and low-precision support**: Triton compiles `tl.dot` into PPU's MMA hardware instructions and automatically derives the tile partitioning granularity to fully leverage Tensor Core acceleration. It provides low-precision / mixed-precision support, including FP8 (E5M2 / E4M3), FP16, and BF16.



## 3. AIU Usage Guide

PPU Triton extends the Triton language with a new `aiu_load` API that moves data from global memory to shared memory via the AIU (AI accelerator in compute Unit) to accelerate data transfer. It currently provides the `tl.aiu_load`, `make_block_ptr`, and `make_tensor_descriptor` interfaces, which users can easily integrate into existing Triton kernels.

### 3.1 tl.aiu_load

```python
def aiu_load(pointer, offsets=(0, 0), block_shape=(0, 0), shape=(0, 0), dtype=void, order=(1, 0), _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, offsets&block_shape&shape should be provided.  In
            this case:
            - `pointer` is the start address of a memory tensor
            - `offsets` is a 2D value indicate the offset of the start pointer of the loading tile to the memory tensor
            - `block_shape` is a 2D value indicate the tile size loaded by aiu_load.
            - `shape` is a 2D value indicate the memory tensor size.
            - `order` is the order of the original data format

        (2) If `pointer` is a block pointer defined by `make_block_ptr`, a tensor is loaded.
    """
```

Use `tl.aiu_load` when you need to load a `BLOCK_M x BLOCK_K` tensor from an `M x K` global data tensor into shared memory.

**Constraints of `aiu_load`:**

- The tensor must be 32-byte aligned.
- `BLOCK_M` must be a multiple of 16.
- The data along the `BLOCK_K` dimension must be a multiple of 32 bytes.
- The block tensor must be contiguous along both the M and K dimensions.
- Supported data types:
  - PPU0010: `b16`
  - PPU0015: `b16`, `b8`


The following is an example matmul kernel using `tl.aiu_load`:

```python
@triton.jit
def matmul_kernel_aiu(a_ptr, b_ptr, c_ptr,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      M, N, K,
                      BLOCK_SIZE_M: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], [M, K], tl.float16)
        b = tl.aiu_load(b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], [K, N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### 3.2 make_block_ptr

Using `make_block_ptr` makes `aiu_load` simpler; it requires `make_block_ptr` / `aiu_load` / `advance` to be used together.

```python
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _semantic=None):
    """
    Returns a pointer to a block in a parent tensor

    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
```

- `tl.make_block_ptr`: constructs a pointer to a block within a parent tensor.
- `tl.aiu_load`: accepts the result of `make_block_ptr` as input.
- `tl.advance`: advances the block pointer.

```python
def advance(base, offsets, _semantic=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
```

Example:

```python
@triton.jit
def matmul_kernel_aiu(a_ptr, b_ptr, c_ptr,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      M, N, K,
                      BLOCK_SIZE_M: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_tensor_ptr = tl.make_block_ptr(a_ptr, (M, K), (stride_am, stride_ak), (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (1, 0))
    b_tensor_ptr = tl.make_block_ptr(b_ptr, (K, N), (stride_bk, stride_bn), (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (1, 0))

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_tensor_ptr)
        b = tl.aiu_load(b_tensor_ptr)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_tensor_ptr = tl.advance(a_tensor_ptr, (0, BLOCK_SIZE_K))
        b_tensor_ptr = tl.advance(b_tensor_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_tensor_ptr = tl.make_block_ptr(c_ptr, (M, N), (stride_cm, stride_cn), (offs_cm, offs_cn), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 0))
    tl.store(c_tensor_ptr, c)
```


### 3.3 make_tensor_descriptor

PPU Triton adds AIU support for `tl.make_tensor_descriptor`; when the conditions are met, a load op can be automatically promoted to an AIU load.

```python
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    :param base: the base pointer of the tensor, must be 16-byte aligned
    :param shape: A list of non-negative integers representing the tensor shape
    :param strides: A list of tensor strides. Leading dimensions must be multiples
        of 16-byte strides and the last dimension must be contiguous.
    :param block_shape: The shape of block to be loaded/stored from global memory
    """
```

When using `tl.make_tensor_descriptor` in Triton kernels, the loads can be automatically promoted to AIU loads when the conditions are met.

### 3.4 More Examples

For more AIU use cases, refer to the tests and performance examples in the repository:

- [`python/test/unit/ppu/aiu/`](python/test/unit/ppu/aiu) — AIU functional tests, covering `aiu_load`, `block pointer`, `tensor descriptor`, `dot`, and `addmm`, along with various data types (FP16, FP8) and order combinations.
- [`python/test/unit/ppu/perf/`](python/test/unit/ppu/perf) — AIU-based performance examples, such as matrix multiplication ([`03-matrix-multiplication-aiu.py`](python/test/unit/ppu/perf/03-matrix-multiplication-aiu.py), [`03-matrix-multiplication-mxfp4-aiu.py`](python/test/unit/ppu/perf/03-matrix-multiplication-mxfp4-aiu.py)) and fused attention ([`06-fused-attention-aiu.py`](python/test/unit/ppu/perf/06-fused-attention-aiu.py)).

## 4. Build and Installation

### 4.1 Prerequisites

- **PPU SDK**: Install the PPU SDK and specify its path via the `PPU_SDK` environment variable (defaults to `/usr/local/PPU_SDK`). The SDK provides headers, libraries, and toolchain executables such as `ppu-llc` and `llvm-irformatter`.
- **Runtime driver**: Install the PPU runtime driver `libhggc.so` and make sure it can be found via `ldconfig` or `LD_LIBRARY_PATH`.
- **Build dependencies**: Python 3.10+, a C++ compiler, CMake, Ninja, etc. (see the [upstream Triton README](README.triton.md)).

```shell
export PPU_SDK=/usr/local/PPU_SDK           # adjust to your actual install path
export LD_LIBRARY_PATH=$PPU_SDK/lib:$LD_LIBRARY_PATH
```

### 4.2 Install from source

```shell
# In the repository root
pip install -r python/requirements.txt      # build-time dependencies
pip install -e .                            # compile and install in editable mode
```

Or with a virtualenv:

```shell
python -m venv .venv --prompt triton
source .venv/bin/activate
pip install -r python/requirements.txt
pip install -e .
```

> Build options (custom LLVM, `ccache`, limiting memory usage with `MAX_JOBS`, etc.) are identical to upstream Triton; see "Building with a custom LLVM" and "Tips for building" in the [upstream Triton README](README.triton.md).

### 4.3 Verify the installation

After building from source, check the import and version with `python -c "import triton; print(triton.__version__)"`. If the version number prints correctly, Triton has been installed successfully and the C++/MLIR extensions load correctly.

Then you can run the examples in the AIU section for end-to-end verification.


### 4.4 Environment variables

For general environment variables, see the [upstream Triton README](README.triton.md). Below are the additional PPU-specific environment variables:

- `PPU_LLC_OPTIONS`: extra options passed to `ppu-llc`.
- `DISABLE_PPU_LLC_OPT`: disable `ppu-llc` optimizations.
- `TRITON_LIBDEVICE_PATH`: specify the PPU libdevice path.
- `TRITON_DUMP_COMPILE_LOG`: export the compilation log.
