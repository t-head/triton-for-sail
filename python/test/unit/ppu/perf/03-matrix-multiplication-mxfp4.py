"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_N):
#        acc = zeros((BLOCK_M, BLOCK_N), dtype=float32)
#        for k in range(0, K, BLOCK_K):
#          a = A[m : m+BLOCK_M, k : k+BLOCK_K]
#          b = B[k : k+BLOCK_K, n : n+BLOCK_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_M, n : n+BLOCK_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_M, k:k+BLOCK_K]` and
# :code:`B[k : k+BLOCK_K, n : n+BLOCK_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_M, k:k+BLOCK_K] =  a_ptr + (m : m+BLOCK_M)[:, None]*A.stride(0) + (k : k+BLOCK_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_K, n:n+BLOCK_N] =  b_ptr + (k : k+BLOCK_K)[:, None]*B.stride(0) + (n : n+BLOCK_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_M` or :code:`N` is not a multiple of :code:`BLOCK_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
#    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
#    offs_k = tl.arange(0, BLOCK_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_K * stride_ak;
#    b_ptrs += BLOCK_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_M, BLOCK_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = tl.program_id(axis=0)
#    grid_n = tl.cdiv(N, BLOCK_N)
#    pid_m = pid // grid_n
#    pid_n = pid % grid_n
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------

import torch
import argparse
import math

import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton._internal_testing import is_ppu

DEVICE = triton.runtime.driver.active.get_active_torch_device()

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="test accuracy before benchmark")
parser.add_argument("--reshape", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="reshape scaleA and schale B to coalesce memory load")
args = parser.parse_args()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    if not args.reshape:
        return [
            triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK}, num_stages = S, num_warps=w)
            for BM in [64, 128, 256] \
            for BN in [64, 128, 256] \
            for BK in [64, 128, 256] \
            for S in [3, 4, 5] \
            for w in [4, 8] \
        ]
    else:
        return [
            triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK, 'USE_2D_SCALE_LOAD': SCALE, 'GROUP_SIZE_M': G}, num_stages = S, num_warps=w)
            for BM in [128, 256] \
            for BN in [128, 256] \
            for BK in [128, 256] \
            for G in [32, 64, 128] \
            for S in [2, 3, 4] \
            for w in [4, 8] \
            for SCALE in [True]
        ]
    return [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK, 'USE_2D_SCALE_LOAD': SCALE, 'GROUP_SIZE_M': G}, num_stages = S, num_warps=w)
        for BM in [128] \
        for BN in [128] \
        for BK in [256] \
        for G in [8, 16, 32, 64, 128] \
        for S in [3] \
        for w in [8] \
        for SCALE in [True]
    ]
    # return [
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5,
    #                   num_warps=2),
    #     triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5,
    #                   num_warps=2),
    #     # Good config for fp8 inputs.
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3,
    #                   num_warps=8),
    #     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4,
    #                   num_warps=4),
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4,
    #                   num_warps=4)
    # ]



def get_autotune_config():
    if is_cuda() or is_ppu():
        return get_cuda_autotune_config()

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def block_scale_fp4_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        num_stages: tl.constexpr, PACK_ALONG_K: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    PACKING_ALONG_M_N: tl.constexpr = 1 if PACK_ALONG_K else 2
    offs_am_packed = (pid_m * (BLOCK_M // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_M // PACKING_ALONG_M_N))
    offs_bn_packed = (pid_n * (BLOCK_N // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_N // PACKING_ALONG_M_N))
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2 if PACK_ALONG_K else BLOCK_K

    # Two e2m1 values per K
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)
    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am_packed[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn_packed[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=num_stages):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator, lhs_k_pack=PACK_ALONG_K,
                                    rhs_k_pack=PACK_ALONG_K)
        a_ptrs += (BLOCK_K_PACKED) * stride_ak
        b_ptrs += (BLOCK_K_PACKED) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // VEC_SIZE
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // VEC_SIZE
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def block_scale_fp4_matmul_reshape(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_sk, stride_sb, stride_sc, stride_sd: tl.constexpr,  # Need tl.constexpr to pipeline scale load. Why?
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        num_stages: tl.constexpr, PACK_ALONG_K: tl.constexpr, USE_2D_SCALE_LOAD: tl.constexpr):  #
    ## This kernel assumes a_scale and b_scale are coming in with shapes
    ## [BLOCK_M(or N) // 128, BLOCK_K // 128, 32, 4, 4] for optimial performance
    ## on nvidia sm100+ HW
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(M, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k = tl.arange(0, BLOCK_K_PACKED)

    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128))
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128))

    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // 128) * 32 * 4 * 4)
        a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
    else:
        offs_sk = tl.arange(0, (BLOCK_K // 128))
        offs_sc = tl.arange(0, 32)
        offs_sd = tl.arange(0, 4)
        a_scale_ptr = a_scale + (offs_sm[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])
        b_scale_ptr = b_scale + (offs_sn[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=num_stages):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)

        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // 128, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // 128, 32, 4, 4)

        # Scales are coming in for optimial performance, but we reshape here for
        # the canonical inputs to dot_scaled
        # These reshapes and transposes will be optimized away during lowering
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // 32)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // 32)
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)

        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        a_scale_ptr += BLOCK_K // 128 * stride_sb
        b_scale_ptr += BLOCK_K // 128 * stride_sb
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, output, M, N, K, a_scale, b_scale, VEC_SIZE, pack_along_k, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    if args.reshape:
        k = block_scale_fp4_matmul_reshape[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a_scale.stride(1),
                                            a_scale.stride(2), a_scale.stride(3), a.stride(0), a.stride(1),
                                        b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE,  PACK_ALONG_K=pack_along_k
                                        )
    else:
        stride_scale = a_scale.stride(0)
        k = block_scale_fp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
                                        b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE,  PACK_ALONG_K=pack_along_k,
                                        )
    return output

def flatten_scale(scale):
    num_chunk_m, num_chunk_k, _, _, _ = scale.shape
    return scale.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale

# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
if args.test:
    M, N, K = 512, 512, 512
    pack_along_k = True
    VEC_SIZE = 32
    torch.manual_seed(42)
    packing_dim = 1 if pack_along_k else 0
    a_mxfp4 = MXFP4Tensor(size=(M, K), device=DEVICE).random()
    a = a_mxfp4.to_packed_tensor(dim=packing_dim)
    # Generate b with k-major layout, pack two e2m1 along k or n, then logical transpose to K, N
    b_mxfp4 = MXFP4Tensor(size=(N, K), device=DEVICE).random()
    b = b_mxfp4.to_packed_tensor(dim=packing_dim).T
    # No need to pack along K since we convert each e2m1 to f32 directly for the reference matmul
    b_ref = b_mxfp4.to(torch.float32).T
    if not args.reshape:

        a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
        b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)
        a_scale = torch.rand(a_size, device=DEVICE)
        b_scale = torch.rand(b_size, device=DEVICE)
        scale_type = "float8_e8m0fnu"
        if scale_type == "float8_e8m0fnu":
            a_scale_ref = MXScaleTensor(a_scale)
            b_scale_ref = MXScaleTensor(b_scale)
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data
        elif scale_type == "float8_e4m3fn":
            a_scale = a_scale.to(torch.float8_e4m3fn)
            b_scale = b_scale.to(torch.float8_e4m3fn)
            a_scale_ref = a_scale
            b_scale_ref = b_scale

        a_scale_ref = a_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]

        kernel_kwargs = {}
        output = a.new_empty((M, N), dtype=torch.float32)
        torch_output = torch.matmul(a_mxfp4.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)
        triton_output = matmul(a, b, output, M, N, K, a_scale, b_scale, VEC_SIZE, pack_along_k)
        print(f"triton_output_with_mxfp4_inputs={triton_output}")
        print(f"torch_output_with_mxfp4_inputs={torch_output}")
        # Bigger tolerance for AMD CDNA2 devices.
        # CDNA2 devices use reduced precision fp16 and bf16 and flush input and
        # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=2e-2):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
    else:
        dtype_dst_str = "float32"
        ceildiv = lambda a, b: math.ceil(a / b)
        a_size = (ceildiv(M, 128), ceildiv(K, 128), 32, 4, 4)
        b_size = (ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4)
        a_scale = torch.rand(a_size, device=DEVICE)
        b_scale = torch.rand(b_size, device=DEVICE)
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

        dtype_dst = getattr(torch, dtype_dst_str)

        kernel_kwargs = {}
        output = a.new_empty((M, N), dtype=torch.float32)
        triton_output = matmul(a, b, output, M, N, K, a_scale, b_scale, VEC_SIZE, pack_along_k)

        a_scale_f32 = flatten_scale(fp8e8m0_to_float32(a_scale))[:M]
        b_scale_f32 = flatten_scale(fp8e8m0_to_float32(b_scale))[:N]
        a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
        b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)

        # b_scales are always col major
        b_scale_f32 = b_scale_f32.T.contiguous()

        a = a_mxfp4.to(torch.float32) * a_scale_f32
        b = b_mxfp4.to(torch.float32).T * b_scale_f32
        torch_output = torch.matmul(a, b).to(torch.float32)
        print(f"triton_output_with_mxfp4_inputs={triton_output}")
        print(f"torch_output_with_mxfp4_inputs={torch_output}")
        # Bigger tolerance for AMD CDNA2 devices.
        # CDNA2 devices use reduced precision fp16 and bf16 and flush input and
        # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=2e-2):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")


# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if (is_cuda() or is_ppu()) else 'rocBLAS'
TORCH_HAS_FP8 = False
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[512 * i for i in range(1, 23)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["triton"],  # Label name for the lines
        line_names=["Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" +
        ("mxfp4"),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    pack_along_k = True
    VEC_SIZE = 32
    packing_dim = 1 if pack_along_k else 0
    scale_type = "float8_e8m0fnu"
    a_mxfp4 = MXFP4Tensor(size=(M, K), device=DEVICE).random()
    a = a_mxfp4.to_packed_tensor(dim=packing_dim)
    # Generate b with k-major layout, pack two e2m1 along k or n, then logical transpose to K, N
    b_mxfp4 = MXFP4Tensor(size=(N, K), device=DEVICE).random()
    b = b_mxfp4.to_packed_tensor(dim=packing_dim).T
    # No need to pack along K since we convert each e2m1 to f32 directly for the reference matmul
    b_ref = b_mxfp4.to(torch.float32).T
    if not args.reshape:
        a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
        b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)
        a_scale = torch.rand(a_size, device=DEVICE)
        b_scale = torch.rand(b_size, device=DEVICE)

        if scale_type == "float8_e8m0fnu":
            a_scale_ref = MXScaleTensor(a_scale)
            b_scale_ref = MXScaleTensor(b_scale)
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data
        elif scale_type == "float8_e4m3fn":
            a_scale = a_scale.to(torch.float8_e4m3fn)
            b_scale = b_scale.to(torch.float8_e4m3fn)
            a_scale_ref = a_scale
            b_scale_ref = b_scale
        a_scale_ref = a_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        quantiles = [0.5, 0.2, 0.8]
        output = a.new_empty((M, N), dtype=torch.float32)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, output, M, N, K, a_scale, b_scale, VEC_SIZE, pack_along_k), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    else:
        a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
        b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)

        ceildiv = lambda a, b: math.ceil(a / b)
        a_size = (ceildiv(M, 128), ceildiv(K, 128), 32, 4, 4)
        b_size = (ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4)
        a_scale = torch.rand(a_size, device=DEVICE)
        b_scale = torch.rand(b_size, device=DEVICE)
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

        output = a.new_empty((M, N), dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        output = a.new_empty((M, N), dtype=torch.float32)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, output, M, N, K, a_scale, b_scale, VEC_SIZE, pack_along_k), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
