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
#include "triton/Dialect/TritonGPU/IR/SharedPhysicalLayout.h"

namespace mlir {
namespace LLVM {
namespace PPU {
inline constexpr llvm::StringLiteral kAIUWriterStrategyAttr = "ppu.aiu_strategy";
inline constexpr llvm::StringLiteral kAIUSourceAttr = "ppu.aiu_source";
inline constexpr llvm::StringLiteral kDisableChainedDotAttr =
    "ppu.disable_chained_dot";

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

Value getPPUAIUV2SwizzledSharedPtr(
    Location loc, const triton::gpu::SharedPhysicalLayout &layout, Type resElemTy,
    Value dstPtrBase, Value idxRow, Value idxCol, RewriterBase &rewriter);

DenseMap<unsigned, Value> getAIUSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals,
    const triton::gpu::SharedPhysicalLayout *layout = nullptr);

FailureOr<llvm::SmallVector<unsigned>>
AIULoadStrategy(unsigned numWarps, unsigned xElems, unsigned channelElems,
                unsigned elemBytes, unsigned version = 1);

} // namespace PPU
} // namespace LLVM
} // namespace mlir

#endif
