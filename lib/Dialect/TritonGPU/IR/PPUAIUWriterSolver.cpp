#include "triton/Dialect/TritonGPU/IR/PPUAIUWriterSolver.h"

namespace mlir::triton::gpu {
namespace {

llvm::Error invalidSolverInput(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<PPUAIUWriterSolution>
PPUAIUWriterSolver::solve(const PPULayoutSpec &layout, unsigned numWarps) {
  if (!numWarps)
    return invalidSolverInput("AIU writer solver requires nonzero numWarps");

  auto inDimNames = layout.getLogicalToPhysical().getInDimNames();
  MLIRContext *context = (*inDimNames.begin()).getContext();
  auto dim0 = StringAttr::get(context, "dim0");
  auto dim1 = StringAttr::get(context, "dim1");
  uint64_t xElems = layout.getRootShape()[0];
  uint64_t channelElems = layout.getRootShape()[1];
  if (layout.getOrder()[1] != 0)
    std::swap(xElems, channelElems);

  auto plans = PPUAIUPlan::enumerateProvenCandidates(
      numWarps, xElems, channelElems, layout.getElementBitWidth(),
      layout.getRootShape(), layout.getOrder());
  if (!plans) {
    llvm::consumeError(plans.takeError());
    return PPUAIUWriterSolution{};
  }

  PPUAIUWriterSolution solution;
  auto legacy = PPUAIUPlan::createLegacyDefault(
      numWarps, xElems, channelElems, layout.getElementBitWidth(),
      layout.getRootShape(), layout.getOrder());
  SmallVector<unsigned> legacyStrategy;
  if (legacy)
    legacyStrategy = legacy->toLegacyStrategy();
  else
    llvm::consumeError(legacy.takeError());

  for (const auto &plan : *plans) {
    auto physical = plan.toSharedPhysicalLayout(*context);
    if (!physical) {
      llvm::consumeError(physical.takeError());
      continue;
    }
    if (physical->physicalStageElems != layout.getPhysicalStageElems() ||
        physical->alignmentBytes != layout.getAlignmentBytes() ||
        physical->order != layout.getOrder() ||
        physical->logicalToPhysical.transposeIns({dim0, dim1}) !=
            layout.getLogicalToPhysical())
      continue;

    if (layout.getCanonicalParameters()) {
      auto cga = CGAEncodingAttr::get1CTALayout(context, 2);
      auto aiuEncoding = PPUAIUSharedEncodingAttr::get(
          context, 2, plan.toLegacyStrategy(), layout.getOrder(), cga);
      auto elementType = IntegerType::get(context, layout.getElementBitWidth());
      auto memDescTy =
          MemDescType::get(layout.getRootShape(), elementType, aiuEncoding,
                           SharedMemorySpaceAttr::get(context),
                           /*mutableMemory=*/true, layout.getRootShape());
      auto encoding = plan.toCanonicalSwizzledSharedEncoding(memDescTy);
      if (!encoding) {
        llvm::consumeError(encoding.takeError());
        continue;
      }
      auto expected = layout.toCanonicalEncoding(*context);
      if (!expected)
        return expected.takeError();
      if (*encoding != *expected)
        continue;
    }

    if (!legacyStrategy.empty() && plan.toLegacyStrategy() == legacyStrategy)
      solution.legacyDefault = plan;
    solution.writers.push_back(plan);
  }
  return solution;
}

} // namespace mlir::triton::gpu
