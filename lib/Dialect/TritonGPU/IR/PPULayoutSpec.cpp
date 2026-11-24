#include "triton/Dialect/TritonGPU/IR/PPULayoutSpec.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::gpu {
namespace {

llvm::Error invalidLayout(llvm::Twine message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 message);
}

llvm::Error validateGeometry(unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
                             ArrayRef<unsigned> order) {
  if (elementBitWidth != 8 && elementBitWidth != 16 && elementBitWidth != 32)
    return invalidLayout("PPU layout supports only 8/16/32-bit elements");
  if (rootShape.size() != 2 || order.size() != 2)
    return invalidLayout("PPU layout requires rank-2 root geometry");
  if (!((order[0] == 1 && order[1] == 0) ||
        (order[0] == 0 && order[1] == 1)))
    return invalidLayout("unsupported PPU layout order");
  for (int64_t dim : rootShape) {
    if (dim <= 0 || !llvm::isPowerOf2_64(dim))
      return invalidLayout("PPU layout requires positive power-of-two roots");
  }
  return llvm::Error::success();
}

} // namespace

PPULayoutSpec::PPULayoutSpec(unsigned elementBitWidth,
                             ArrayRef<int64_t> rootShape,
                             ArrayRef<unsigned> order,
                             uint64_t physicalStageElems,
                             unsigned alignmentBytes,
                             triton::LinearLayout logicalToPhysical)
    : rootShape(rootShape.begin(), rootShape.end()),
      order(order.begin(), order.end()), elementBitWidth(elementBitWidth),
      physicalStageElems(physicalStageElems), alignmentBytes(alignmentBytes),
      logicalToPhysical(std::move(logicalToPhysical)) {}

llvm::Expected<PPULayoutSpec> PPULayoutSpec::createPhysical(
    unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
    ArrayRef<unsigned> order, uint64_t physicalStageElems,
    unsigned alignmentBytes, triton::LinearLayout logicalToPhysical) {
  if (auto error = validateGeometry(elementBitWidth, rootShape, order))
    return std::move(error);
  if (!physicalStageElems || physicalStageElems > std::numeric_limits<int32_t>::max())
    return invalidLayout("PPU layout physical stage exceeds int32 range");
  if (!alignmentBytes || !llvm::isPowerOf2_32(alignmentBytes))
    return invalidLayout("PPU layout alignment must be a nonzero power of two");

  auto inDimNames = logicalToPhysical.getInDimNames();
  if (inDimNames.empty())
    return invalidLayout("PPU layout requires dim0/dim1 inputs and offset output");
  MLIRContext *context = (*inDimNames.begin()).getContext();
  auto dim0 = StringAttr::get(context, "dim0");
  auto dim1 = StringAttr::get(context, "dim1");
  auto offset = StringAttr::get(context, "offset");
  if (!logicalToPhysical.hasInDim(dim0) || !logicalToPhysical.hasInDim(dim1) ||
      !logicalToPhysical.hasOutDim(offset) || logicalToPhysical.getOutDims().size() != 1)
    return invalidLayout("PPU layout requires dim0/dim1 inputs and offset output");
  auto normalized = logicalToPhysical.transposeIns({dim0, dim1});
  if (normalized.getInDimSize(dim0) != rootShape[0] ||
      normalized.getInDimSize(dim1) != rootShape[1] ||
      static_cast<uint64_t>(normalized.getOutDimSize(offset)) != physicalStageElems)
    return invalidLayout("PPU layout dimensions do not match root geometry");
  if (!normalized.isInjective())
    return invalidLayout("PPU layout physical mapping must be collision-free");
  return PPULayoutSpec(elementBitWidth, rootShape, order, physicalStageElems,
                       alignmentBytes, std::move(normalized));
}

llvm::Expected<PPULayoutSpec> PPULayoutSpec::createCanonical(
    MLIRContext &context, unsigned elementBitWidth,
    ArrayRef<int64_t> rootShape, ArrayRef<unsigned> order, unsigned vec,
    unsigned perPhase, unsigned maxPhase) {
  if (auto error = validateGeometry(elementBitWidth, rootShape, order))
    return std::move(error);
  if (!vec || !perPhase || !maxPhase || !llvm::isPowerOf2_32(vec) ||
      !llvm::isPowerOf2_32(perPhase) || !llvm::isPowerOf2_32(maxPhase))
    return invalidLayout("canonical layout parameters must be powers of two");

  auto cga = CGAEncodingAttr::get1CTALayout(&context, 2);
  auto encoding = SwizzledSharedEncodingAttr::get(&context, vec, perPhase,
                                                  maxPhase, order, cga);
  auto standard = encoding.toLinearLayout(rootShape);
  standard = standard.squeezeIns(StringAttr::get(&context, "block"));
  if (!standard.isInvertible())
    return invalidLayout("canonical shared layout must be invertible");
  auto dim0 = StringAttr::get(&context, "dim0");
  auto dim1 = StringAttr::get(&context, "dim1");
  auto physical = standard.invert().transposeIns({dim0, dim1});
  auto result = createPhysical(elementBitWidth, rootShape, order,
                               mlir::product(rootShape), 128, std::move(physical));
  if (!result)
    return result.takeError();
  result->canonicalParameters = CanonicalLayoutParameters{vec, perPhase, maxPhase};
  return result;
}

llvm::Expected<std::vector<PPULayoutSpec>>
PPULayoutSpec::enumerateCanonicalCandidates(MLIRContext &context,
                                            unsigned elementBitWidth,
                                            ArrayRef<int64_t> rootShape,
                                            ArrayRef<unsigned> order) {
  PPULayoutConstraints constraints;
  constraints.elementBitWidth = elementBitWidth;
  constraints.rootShape.assign(rootShape.begin(), rootShape.end());
  constraints.allowedOrders.emplace_back(order.begin(), order.end());
  return enumerateCanonicalCandidates(context, constraints);
}

llvm::Expected<std::vector<PPULayoutSpec>>
PPULayoutSpec::enumerateCanonicalCandidates(
    MLIRContext &context, const PPULayoutConstraints &constraints) {
  if (auto error = validateGeometry(constraints.elementBitWidth,
                                    constraints.rootShape, {1, 0}))
    return std::move(error);
  if (constraints.allowedOrders.empty())
    return invalidLayout("PPU layout requires at least one allowed order");
  for (const auto &order : constraints.allowedOrders) {
    if (auto error = validateGeometry(constraints.elementBitWidth,
                                      constraints.rootShape, order))
      return std::move(error);
  }
  if (!constraints.maxVectorBits || !llvm::isPowerOf2_32(constraints.maxVectorBits) ||
      !constraints.maxPhase || !llvm::isPowerOf2_32(constraints.maxPhase))
    return invalidLayout("PPU layout reader limits must be powers of two");

  std::vector<PPULayoutSpec> result;
  for (const auto &order : constraints.allowedOrders) {
    uint64_t contiguousElems = constraints.rootShape[order[0]];
    uint64_t rows = constraints.rootShape[order[1]];
    unsigned maxVec = std::min<uint64_t>(
        constraints.maxVectorBits / constraints.elementBitWidth, contiguousElems);
    unsigned maxPhase = std::min<uint64_t>(constraints.maxPhase, rows);
    for (unsigned vec = 1; vec <= maxVec; vec <<= 1) {
      for (unsigned perPhase = 1; perPhase <= rows; perPhase <<= 1) {
        for (unsigned phases = 1; phases <= maxPhase; phases <<= 1) {
          if (static_cast<uint64_t>(perPhase) * phases > rows)
            continue;
          auto candidate = createCanonical(
              context, constraints.elementBitWidth, constraints.rootShape,
              order, vec, perPhase, phases);
          if (!candidate) {
            llvm::consumeError(candidate.takeError());
            continue;
          }
          bool duplicate = llvm::any_of(result, [&](const PPULayoutSpec &existing) {
            return existing.getOrder() == candidate->getOrder() &&
                   existing.getLogicalToPhysical() == candidate->getLogicalToPhysical();
          });
          if (!duplicate)
            result.push_back(std::move(*candidate));
        }
      }
    }
  }
  return result;
}

llvm::Expected<SwizzledSharedEncodingAttr>
PPULayoutSpec::toCanonicalEncoding(MLIRContext &context,
                                   ArrayRef<unsigned> fullOrder,
                                   CGAEncodingAttr cgaLayout) const {
  if (!canonicalParameters)
    return invalidLayout("non-canonical PPU layout has no standard encoding");
  if (fullOrder.empty())
    fullOrder = order;
  if (fullOrder.size() == 2) {
    if (!std::equal(fullOrder.begin(), fullOrder.end(), order.begin()))
      return invalidLayout("canonical layout order mismatch");
  } else if (fullOrder.size() == 3) {
    if (fullOrder[2] != 0 || fullOrder[0] == 0 || fullOrder[1] == 0 ||
        fullOrder[0] - 1 != order[0] || fullOrder[1] - 1 != order[1])
      return invalidLayout("unsupported canonical pipelined layout order");
  } else {
    return invalidLayout("canonical layout requires rank-2 or staged order");
  }
  if (!cgaLayout)
    cgaLayout = CGAEncodingAttr::get1CTALayout(&context, fullOrder.size());
  if (cgaLayout != CGAEncodingAttr::get1CTALayout(&context, cgaLayout.getRank()))
    return invalidLayout("canonical PPU layout currently requires one CTA");
  return SwizzledSharedEncodingAttr::get(
      &context, canonicalParameters->vec, canonicalParameters->perPhase,
      canonicalParameters->maxPhase, fullOrder, cgaLayout);
}

} // namespace mlir::triton::gpu
