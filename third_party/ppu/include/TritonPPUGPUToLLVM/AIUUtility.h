/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_AIU_UTILITY_H
#define TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_AIU_UTILITY_H

#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace LLVM {
namespace PPU {
SmallVector<Value> getStrides(const SharedMemoryObject &smemObj,
                              triton::gpu::MemDescType memDesc, Location loc,
                              RewriterBase &rewriter);

triton::gpu::CGAEncodingAttr
getExpandedCTALayout(MLIRContext *ctx, triton::gpu::CGAEncodingAttr ctaLayout);

Attribute getExpandedEncoding(Attribute encoding);

triton::gpu::MemDescType getExpandedDesc(triton::gpu::MemDescType descTy);

SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape);

Value getSliceKOffset(ConversionPatternRewriter &rewriter, Location loc,
                      Value tensor, int opIdx);

DenseMap<unsigned, Value> getAIUSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals);

inline FailureOr<llvm::SmallVector<unsigned>>
AIULoadStrategy(unsigned numWarps, unsigned xElems, unsigned channelElems,
                unsigned elemBytes, unsigned version = 1) {
  constexpr unsigned minXElems = 16;
  constexpr unsigned maxCubeW = 2048;
  if (numWarps == 0 || xElems < minXElems || channelElems == 0 ||
      elemBytes == 0)
    return failure();
  if ((version == 1 && elemBytes != 1 && elemBytes != 2) ||
      (version == 2 && elemBytes != 1 && elemBytes != 2 && elemBytes != 4))
    return failure();

  if (version == 1) {
    constexpr unsigned sliceByte = 32;
    constexpr unsigned maxSliceBytes = 128;
    unsigned channelBytes = channelElems * elemBytes;
    if (channelBytes < sliceByte)
      return failure();

    unsigned sliceTotal = channelBytes / sliceByte;
    unsigned numSlice;
    if (channelBytes <= maxSliceBytes) {
      numSlice = sliceTotal;
    } else if (sliceTotal % 4 == 0) {
      numSlice = 4;
    } else if (sliceTotal % 2 == 0) {
      numSlice = 2;
    } else {
      numSlice = 1;
    }

    unsigned channelCopy = sliceTotal / numSlice;
    unsigned warpC;
    unsigned warpW;
    unsigned maxWarpW = xElems / minXElems;
    if (numWarps % channelCopy == 0) {
      warpC = channelCopy;
      warpW = std::min<unsigned>(numWarps / channelCopy, maxWarpW);
    } else if (channelCopy % numWarps == 0) {
      warpC = numWarps;
      warpW = 1;
    } else {
      warpC = 1;
      warpW = std::min<unsigned>(numWarps, maxWarpW);
    }

    unsigned cubeW = xElems / warpW;
    if (cubeW > maxCubeW)
      return failure();
    unsigned cubeC = numSlice * sliceByte / elemBytes;
    return llvm::SmallVector<unsigned>{cubeC, cubeW, warpC, warpW,
                                       numSlice};
  }

  if (version == 2) {
    if (channelElems % 64 != 0 && channelElems % 32 != 0 &&
        channelElems % 16 != 0)
      return failure();

    unsigned swizzledBytes = channelElems * elemBytes <= 64 ? 64 : 128;
    unsigned channelBytes = elemBytes * channelElems;
    unsigned cubeC = channelBytes < swizzledBytes
                         ? channelElems
                         : swizzledBytes / elemBytes;
    unsigned warpCMax = (channelElems + cubeC - 1) / cubeC;
    unsigned warpC = std::min<unsigned>(numWarps, warpCMax);
    unsigned warpWMax = xElems / minXElems;
    unsigned warpW = std::min<unsigned>(numWarps / warpC, warpWMax);
    unsigned cubeW = xElems / warpW;
    if (cubeW > maxCubeW)
      return failure();

    return llvm::SmallVector<unsigned>{cubeC, cubeW, warpC, warpW,
                                       swizzledBytes};
  }

  return failure();
}

} // namespace PPU
} // namespace LLVM
} // namespace mlir

#endif