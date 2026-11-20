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

#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>

namespace mlir {
namespace triton {
namespace gpu {
namespace {

class AIULoadLowering : public OpRewritePattern<AIULoadOp> {
public:
  AIULoadLowering(MLIRContext *context, bool *encounteredInvalidLoad)
      : OpRewritePattern(context),
        encounteredInvalidLoad(encounteredInvalidLoad) {}

  LogicalResult matchAndRewrite(AIULoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    assert(tensorType.getRank() > 1);
    auto order = llvm::to_vector_of<unsigned>(op.getOrder());
    auto ctaLayout = getCGALayout(tensorType.getEncoding());
    auto elemBytes = tensorType.getElementTypeBitWidth() / 8;
    int numWarps = lookupNumWarps(op);

    auto tileShape = tensorType.getShape();
    size_t rank = tileShape.size();
    auto tileC = tileShape[rank - 1];
    auto tileW = tileShape[rank - 2];
    if (order[rank - 1] != 0) {
      tileC = tileShape[rank - 2];
      tileW = tileShape[rank - 1];
    }

    auto mod = op->getParentOfType<ModuleOp>();
    auto computeCapability = getPPUComputeCapability(mod);
    unsigned version = (computeCapability == 80) ? 1 : 2;
    SmallVector<unsigned> loadStrategy;
    if (version == 2) {
      auto plan = PPUAIUPlan::createLegacyDefault(tensorType, numWarps, order);
      if (!plan) {
        *encounteredInvalidLoad = true;
        return op.emitError()
               << "unsupported AIU load tile for PPU AIU version 2: "
               << llvm::toString(plan.takeError());
      }
      loadStrategy = plan->toLegacyStrategy();
    } else {
      auto legacy =
          LLVM::PPU::AIULoadStrategy(numWarps, tileW, tileC, elemBytes, version);
      if (failed(legacy)) {
        *encounteredInvalidLoad = true;
        return op.emitError()
               << "unsupported AIU load tile for PPU AIU version " << version
               << ": channel width is " << tileC * elemBytes
               << " bytes and contiguous width is " << tileW << " elements";
      }
      loadStrategy = *legacy;
    }
    Attribute encoding = PPUAIUSharedEncodingAttr::get(
        tensorType.getContext(), version, loadStrategy, order, ctaLayout);

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType, Value());

    Operation *copy = triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp::create(
        rewriter, loc, op.getSrcPtr(), op.getIndices(), op.getShape(), alloc);

    Value Zero = arith::ConstantIntOp::create(rewriter, loc, 0, 0);

    Operation *commmit = triton::gpu::AsyncCommitGroupOp::create(
        rewriter, loc, copy->getResult(0));
    Operation *wait = triton::gpu::AsyncWaitOp::create(
        rewriter, loc, commmit->getResult(0), 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }

private:
  bool *encounteredInvalidLoad;
};

std::optional<MemDescType>
canonicalSharedType(MemDescType type, unsigned numWarps) {
  auto plan = PPUAIUPlan::fromLegacy(type, numWarps);
  if (!plan) {
    llvm::consumeError(plan.takeError());
    return std::nullopt;
  }
  auto encoding = plan->toCanonicalSwizzledSharedEncoding(type);
  if (!encoding) {
    llvm::consumeError(encoding.takeError());
    return std::nullopt;
  }
  return MemDescType::get(type.getShape(), type.getElementType(), *encoding,
                          type.getMemorySpace(), type.getMutableMemory(),
                          type.getAllocShape());
}

void collectSharedTypes(Type type, unsigned numWarps,
                        DenseMap<MemDescType, MemDescType> &conversions) {
  auto memDescTy = dyn_cast<MemDescType>(type);
  if (!memDescTy || conversions.contains(memDescTy) ||
      !isa<PPUAIUSharedEncodingAttr>(memDescTy.getEncoding()))
    return;
  if (auto converted = canonicalSharedType(memDescTy, numWarps))
    conversions.insert({memDescTy, *converted});
}

void invalidateMixedConversions(
    ModuleOp module, DenseMap<MemDescType, MemDescType> &conversions) {
  bool changed;
  do {
    changed = false;
    module.walk([&](Operation *op) {
      SmallVector<MemDescType> types;
      auto appendType = [&](Type type) {
        if (auto memDesc = dyn_cast<MemDescType>(type))
          types.push_back(memDesc);
      };
      for (Value operand : op->getOperands())
        appendType(operand.getType());
      for (Value result : op->getResults())
        appendType(result.getType());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            appendType(argument.getType());
      bool hasConverted = llvm::any_of(types, [&](MemDescType type) {
        return conversions.contains(type);
      });
      bool hasLegacy = llvm::any_of(types, [&](MemDescType type) {
        return isa<PPUAIUSharedEncodingAttr>(type.getEncoding()) &&
               !conversions.contains(type);
      });
      if (!hasConverted || !hasLegacy)
        return;
      for (MemDescType type : types)
        changed |= conversions.erase(type);
    });
  } while (changed);
}

ArrayAttr writerStrategyAttr(MLIRContext *context,
                             ArrayRef<unsigned> strategy) {
  SmallVector<Attribute> values;
  auto type = IntegerType::get(context, 32);
  for (unsigned value : strategy)
    values.push_back(IntegerAttr::get(type, value));
  return ArrayAttr::get(context, values);
}

} // namespace
} // namespace gpu
} // namespace triton

#define GEN_PASS_DEF_TRITONPPUAIULOWERINGPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

class TritonPPUAIULoweringPass
    : public impl::TritonPPUAIULoweringPassBase<TritonPPUAIULoweringPass> {
public:
  using impl::TritonPPUAIULoweringPassBase<
      TritonPPUAIULoweringPass>::TritonPPUAIULoweringPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    bool encounteredInvalidLoad = false;
    mlir::RewritePatternSet patterns(context);
    patterns.add<mlir::triton::gpu::AIULoadLowering>(context,
                                                     &encounteredInvalidLoad);
    if (applyPatternsGreedily(m, std::move(patterns)).failed() ||
        encounteredInvalidLoad)
      signalPassFailure();
  }
};

#define GEN_PASS_DEF_TRITONPPUAIUSTANDARDIZESHAREDPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

class TritonPPUAIUStandardizeSharedPass
    : public impl::TritonPPUAIUStandardizeSharedPassBase<
          TritonPPUAIUStandardizeSharedPass> {
public:
  void runOnOperation() override {
    using namespace triton::gpu;
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    unsigned numWarps = lookupNumWarps(module);
    DenseMap<MemDescType, MemDescType> conversions;

    module.walk([&](Operation *op) {
      for (Value result : op->getResults())
        collectSharedTypes(result.getType(), numWarps, conversions);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            collectSharedTypes(argument.getType(), numWarps, conversions);
    });
    module.walk([&](func::FuncOp function) {
      auto type = function.getFunctionType();
      for (Type input : type.getInputs())
        collectSharedTypes(input, numWarps, conversions);
      for (Type result : type.getResults())
        collectSharedTypes(result, numWarps, conversions);
    });
    invalidateMixedConversions(module, conversions);

    bool failed = false;
    module.walk([&](triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp op) {
      auto oldType = dyn_cast<MemDescType>(op.getResult().getType());
      if (!oldType || !conversions.contains(oldType))
        return;
      auto plan = PPUAIUPlan::fromLegacy(oldType, numWarps);
      if (!plan) {
        op.emitError() << "failed to preserve AIU writer strategy: "
                       << llvm::toString(plan.takeError());
        failed = true;
        return;
      }
      op->setAttr(LLVM::PPU::kAIUWriterStrategyAttr,
                  writerStrategyAttr(context, plan->toLegacyStrategy()));
    });
    module.walk([&](LocalLoadOp op) {
      auto sourceType = dyn_cast<MemDescType>(op.getSrc().getType());
      if (sourceType && conversions.contains(sourceType))
        op->setAttr(LLVM::PPU::kAIUSourceAttr, UnitAttr::get(context));
    });
    module.walk([&](LocalAllocOp op) {
      auto oldType = dyn_cast<MemDescType>(op.getType());
      if (!oldType || !conversions.contains(oldType))
        return;
      int32_t alignment = std::max<int32_t>(op.getAlignmentOrDefault(), 128);
      op->setAttr("alignment", IntegerAttr::get(IntegerType::get(context, 32),
                                                alignment));
    });

    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        auto type = dyn_cast<MemDescType>(result.getType());
        if (type && conversions.contains(type))
          result.setType(conversions.lookup(type));
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments()) {
            auto type = dyn_cast<MemDescType>(argument.getType());
            if (type && conversions.contains(type))
              argument.setType(conversions.lookup(type));
          }
    });
    module.walk([&](func::FuncOp function) {
      auto type = function.getFunctionType();
      SmallVector<Type> inputs;
      SmallVector<Type> results;
      for (Type input : type.getInputs()) {
        auto memDesc = dyn_cast<MemDescType>(input);
        inputs.push_back(memDesc && conversions.contains(memDesc)
                             ? conversions.lookup(memDesc)
                             : input);
      }
      for (Type result : type.getResults()) {
        auto memDesc = dyn_cast<MemDescType>(result);
        results.push_back(memDesc && conversions.contains(memDesc)
                              ? conversions.lookup(memDesc)
                              : result);
      }
      function.setFunctionType(FunctionType::get(context, inputs, results));
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace mlir
