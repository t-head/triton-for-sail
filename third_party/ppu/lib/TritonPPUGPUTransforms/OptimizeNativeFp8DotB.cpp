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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

#define DEBUG_TYPE "tritonppugpu-optimize-native-fp8-dot-b"

namespace mlir {

#define GEN_PASS_DEF_TRITONPPUGPUOPTIMIZENATIVEFP8DOTB
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {

using triton::gpu::ConvertLayoutOp;

bool isHoistedNativeFp8DotB(ConvertLayoutOp cvtOp) {
  if (cvtOp.getResult().use_empty())
    return false;
  for (Operation *user : cvtOp.getResult().getUsers()) {
    auto dot = dyn_cast<triton::DotOp>(user);
    auto loop = dot ? dot->getParentOfType<scf::ForOp>() : scf::ForOp();
    // Repacking is amortized only for operands hoisted out of a loop.
    if (!dot || !loop || loop->isProperAncestor(cvtOp) ||
        dot.getB() != cvtOp.getResult() ||
        !dot.getType().getElementType().isF32())
      return false;
  }
  return true;
}

bool isNativeFp8DotBConversion(ConvertLayoutOp cvtOp) {
  auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
  auto dstType = cast<RankedTensorType>(cvtOp.getType());
  auto srcEncoding = srcType.getEncoding();
  if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
    return false;
  auto dstDotOp =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
  if (!dstDotOp || !cvtNeedsSharedMemory(srcType, dstType))
    return false;
  auto ppuMma =
      dyn_cast<triton::gpu::PPUMmaEncodingAttr>(dstDotOp.getParent());
  if (!ppuMma || ppuMma.getVersionMajor() != 2 || dstDotOp.getOpIdx() != 1 ||
      dstDotOp.getKWidth() != 4)
    return false;
  if (srcType.getRank() != 2 ||
      !isa<Float8E4M3FNType, Float8E5M2Type>(srcType.getElementType()) ||
      !isa<triton::gpu::BlockedEncodingAttr>(srcEncoding) ||
      triton::gpu::getNumCTAs(srcEncoding) != 1)
    return false;
  auto order = triton::gpu::getOrderForMemory(srcType);
  auto shape = srcType.getShape();
  if (order[0] != 1 || shape[0] < 32 || shape[0] % 32 != 0 || shape[1] < 16 ||
      shape[1] % 16 != 0)
    return false;
  return isHoistedNativeFp8DotB(cvtOp);
}

void rewriteToNativeSharedK(ConvertLayoutOp cvtOp) {
  auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
  auto dstType = cast<RankedTensorType>(cvtOp.getType());
  auto srcEncoding = srcType.getEncoding();
  auto dstDotOp =
      cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
  auto ppuMma =
      cast<triton::gpu::PPUMmaEncodingAttr>(dstDotOp.getParent());
  OpBuilder builder(cvtOp);
  auto *ctx = cvtOp.getContext();
  auto blocked = triton::gpu::BlockedEncodingAttr::get(
      ctx, {4, 4}, {4, 8}, ppuMma.getWarpsPerCTA(), {1, 0},
      triton::gpu::getCGALayout(srcEncoding));
  auto sharedSourceType = srcType.cloneWithEncoding(blocked);
  auto sharedSource = ConvertLayoutOp::create(builder, cvtOp.getLoc(),
                                              sharedSourceType, cvtOp.getSrc());
  auto sharedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
      ctx, 16, 2, 4, {0, 1}, triton::gpu::getCGALayout(srcEncoding));
  auto sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
  auto tmpType = triton::gpu::MemDescType::get(
      dstType.getShape(), dstType.getElementType(), sharedEncoding,
      sharedMemorySpace);
  auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(), tmpType,
                                               sharedSource);
  auto newConvert = triton::gpu::LocalLoadOp::create(builder, cvtOp.getLoc(),
                                                     dstType, tmp);
  cvtOp.replaceAllUsesWith(newConvert.getResult());
  cvtOp.erase();
}

} // namespace

class TritonPPUGPUOptimizeNativeFp8DotBPass
    : public impl::TritonPPUGPUOptimizeNativeFp8DotBBase<
          TritonPPUGPUOptimizeNativeFp8DotBPass> {
public:
  void runOnOperation() override {
    SmallVector<ConvertLayoutOp> candidates;
    getOperation().walk(
        [&](ConvertLayoutOp cvtOp) { candidates.push_back(cvtOp); });
    for (ConvertLayoutOp cvtOp : candidates)
      if (isNativeFp8DotBConversion(cvtOp))
        rewriteToNativeSharedK(cvtOp);
  }
};

} // namespace mlir
