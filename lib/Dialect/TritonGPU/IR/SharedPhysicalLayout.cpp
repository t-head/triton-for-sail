#include "triton/Dialect/TritonGPU/IR/SharedPhysicalLayout.h"
#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>

namespace mlir::triton::gpu {
namespace {

llvm::Error invalidLayout(llvm::Twine message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 message);
}

uint64_t physicalOffset(const PPUAIUPlan &plan, uint64_t row, uint64_t col,
                        bool applySwizzle) {
  uint64_t rowCube = row / plan.getCubeW();
  uint64_t colCube = col / plan.getCubeC();
  uint64_t warpC = colCube % plan.getWarpCopyC();
  uint64_t warpW = rowCube % plan.getWarpCopyW();
  uint64_t copy = colCube / plan.getWarpCopyC();
  uint64_t base = copy * plan.getWriterCopyElems() +
                  (warpW * plan.getWarpCopyC() + warpC) *
                      plan.getWriterCubeElems();
  uint64_t innerRow = row % plan.getCubeW();
  uint64_t innerCol = col % (plan.getCubeC() * plan.getStorageFactor());
  innerRow += innerCol / plan.getCubeC();
  innerCol %= plan.getCubeC();

  unsigned perPhase = plan.getSwizzledBytes() == 64 ? 2 : 1;
  unsigned maxPhase = plan.getSwizzledBytes() == 64 ? 4 : 8;
  unsigned outVec = 128 / plan.getElementBitWidth();
  uint64_t phase = applySwizzle ? (innerRow / perPhase) % maxPhase : 0;
  uint64_t colChunk = innerCol / outVec;
  uint64_t swizzledChunk = colChunk ^ phase;
  return base + innerRow * plan.getCubeC() * plan.getStorageFactor() +
         swizzledChunk * outVec + innerCol % outVec;
}

llvm::Expected<triton::LinearLayout>
buildLayout(const PPUAIUPlan &plan, MLIRContext &context, bool bulkBase) {
  auto dim0 = StringAttr::get(&context, "dim0");
  auto dim1 = StringAttr::get(&context, "dim1");
  auto offset = StringAttr::get(&context, "offset");
  triton::LinearLayout::BasesT bases;
  for (uint64_t bit = 1; bit < plan.getXElems(); bit <<= 1) {
    auto value = physicalOffset(plan, bit, 0, !bulkBase);
    bases[plan.getOrder()[1] == 0 ? dim0 : dim1].push_back(
        {static_cast<int32_t>(value)});
  }
  for (uint64_t bit = 1; bit < plan.getChannelElems(); bit <<= 1) {
    auto value = physicalOffset(plan, 0, bit, !bulkBase);
    bases[plan.getOrder()[1] == 0 ? dim1 : dim0].push_back(
        {static_cast<int32_t>(value)});
  }
  auto layout = triton::LinearLayout::tryCreate(
      bases, {{offset, static_cast<int32_t>(plan.getPhysicalStageElems())}},
      /*requireSurjective=*/false);
  if (!layout)
    return invalidLayout("failed to construct AIU physical LinearLayout");
  return *layout;
}

llvm::Expected<triton::LinearLayout>
invertPhysicalLayout(const triton::LinearLayout &layout,
                     StringAttr expandedDim = nullptr) {
  auto inDims = llvm::to_vector(layout.getInDimNames());
  llvm::sort(inDims, [](StringAttr a, StringAttr b) {
    return a.str() < b.str();
  });
  auto outDims = llvm::to_vector(layout.getOutDimNames());
  unsigned rows = layout.getTotalOutDimSizeLog2();
  unsigned cols = layout.getTotalInDimSizeLog2();
  if (rows > 64 || cols > 64)
    return invalidLayout("physical layout inversion supports at most 64 bits");

  std::vector<uint64_t> matrix(rows, 0);
  std::vector<uint64_t> transform(rows, 0);
  unsigned row = 0;
  for (StringAttr outDim : outDims) {
    for (int bit = 0; bit < layout.getOutDimSizeLog2(outDim); ++bit, ++row) {
      transform[row] = uint64_t{1} << row;
      unsigned col = 0;
      for (StringAttr inDim : inDims) {
        for (int inBit = 0; inBit < layout.getInDimSizeLog2(inDim);
             ++inBit, ++col) {
          if ((layout.getBasis(inDim, inBit, outDim) >> bit) & 1)
            matrix[row] |= uint64_t{1} << col;
        }
      }
    }
  }

  unsigned rank = 0;
  SmallVector<int> pivotColumns(rows, -1);
  for (unsigned col = 0; col < cols; ++col) {
    unsigned pivot = rank;
    while (pivot < rows && !((matrix[pivot] >> col) & 1))
      ++pivot;
    if (pivot == rows)
      continue;
    std::swap(matrix[rank], matrix[pivot]);
    std::swap(transform[rank], transform[pivot]);
    for (unsigned other = 0; other < rows; ++other) {
      if (other != rank && ((matrix[other] >> col) & 1)) {
        matrix[other] ^= matrix[rank];
        transform[other] ^= transform[rank];
      }
    }
    pivotColumns[rank] = col;
    ++rank;
  }

  if (rank != cols)
    return invalidLayout("physical layout is not injective");

  uint64_t paddingParity = 0;
  unsigned expandedDimIndex = 0;
  if (expandedDim) {
    if (rows != rank + 1 || matrix[rank] != 0)
      return invalidLayout("physical padding has no unique parity row");
    paddingParity = transform[rank];
    auto dim = llvm::find(inDims, expandedDim);
    if (dim == inDims.end())
      return invalidLayout("expanded physical dimension is missing");
    expandedDimIndex = dim - inDims.begin();
  }

  triton::LinearLayout::BasesT bases;
  unsigned outputBit = 0;
  for (StringAttr outDim : outDims) {
    auto &basis = bases[outDim];
    for (int bit = 0; bit < layout.getOutDimSizeLog2(outDim);
         ++bit, ++outputBit) {
      std::vector<int32_t> values(inDims.size(), 0);
      for (unsigned pivotRow = 0; pivotRow < rank; ++pivotRow) {
        if (!((transform[pivotRow] >> outputBit) & 1))
          continue;
        unsigned col = pivotColumns[pivotRow];
        unsigned dim = 0;
        unsigned inBit = col;
        while (inBit >= static_cast<unsigned>(
                            layout.getInDimSizeLog2(inDims[dim]))) {
          inBit -= layout.getInDimSizeLog2(inDims[dim]);
          ++dim;
        }
        values[dim] |= 1 << inBit;
      }
      if (expandedDim && ((paddingParity >> outputBit) & 1))
        values[expandedDimIndex] |= layout.getInDimSize(expandedDim);
      basis.push_back(std::move(values));
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> logicalDims;
  for (StringAttr inDim : inDims) {
    int32_t size = layout.getInDimSize(inDim);
    if (inDim == expandedDim)
      size *= 2;
    logicalDims.push_back({inDim, size});
  }
  auto inverted = triton::LinearLayout::tryCreate(
      bases, logicalDims, /*requireSurjective=*/false);
  if (!inverted)
    return invalidLayout("failed to construct inverted physical layout");
  return *inverted;
}

} // namespace

llvm::Expected<SharedPhysicalLayout>
SharedPhysicalLayout::fromPlan(const PPUAIUPlan &plan, MLIRContext &context) {
  if (plan.getRootShape().size() != 2 ||
      !llvm::isPowerOf2_64(plan.getRootShape()[0]) ||
      !llvm::isPowerOf2_64(plan.getRootShape()[1]))
    return invalidLayout("AIU physical LinearLayout requires power-of-two roots");
  if (plan.getPhysicalStageElems() > std::numeric_limits<int32_t>::max())
    return invalidLayout("AIU physical stage exceeds LinearLayout int32 range");
  auto finalLayout = buildLayout(plan, context, /*bulkBase=*/false);
  if (!finalLayout)
    return finalLayout.takeError();
  auto bulkLayout = buildLayout(plan, context, /*bulkBase=*/true);
  if (!bulkLayout)
    return bulkLayout.takeError();
  bool expectedSurjective = plan.getStorageFactor() == 1;
  if (finalLayout->isSurjective() != expectedSurjective ||
      bulkLayout->isSurjective() != expectedSurjective)
    return invalidLayout("unexpected AIU physical layout surjectivity");

  SharedPhysicalLayout result;
  result.logicalToPhysical = *finalLayout;
  result.logicalToBulkBase = *bulkLayout;
  result.physicalStageElems = plan.getPhysicalStageElems();
  result.order.assign(plan.getOrder().begin(), plan.getOrder().end());
  result.alignmentBytes = 128;
  result.swizzledBytes = plan.getSwizzledBytes();
  return result;
}

llvm::Expected<SharedPhysicalLayout>
PPUAIUPlan::toSharedPhysicalLayout(MLIRContext &context) const {
  return SharedPhysicalLayout::fromPlan(*this, context);
}

llvm::Expected<SwizzledSharedEncodingAttr>
PPUAIUPlan::toCanonicalSwizzledSharedEncoding(MemDescType memDescTy) const {
  MLIRContext &context = *memDescTy.getContext();
  if (getStorageFactor() != 1 ||
      getPhysicalStageElems() != getXElems() * getChannelElems())
    return invalidLayout("AIU canonical shared encoding requires dense factor-1 storage");

  auto shape = getShapePerCTA(memDescTy);
  auto allocShape = memDescTy.getAllocShape();
  if (shape.size() < 2 || allocShape.size() < 2 ||
      !std::equal(getRootShape().begin(), getRootShape().end(),
                  ArrayRef<int64_t>(shape).take_back(2).begin()) ||
      !std::equal(getRootShape().begin(), getRootShape().end(),
                  ArrayRef<int64_t>(allocShape).take_back(2).begin()))
    return invalidLayout("AIU canonical shared encoding requires matching root shapes");
  if (memDescTy.getElementTypeBitWidth() != getElementBitWidth())
    return invalidLayout("AIU canonical shared encoding element type mismatch");

  auto cgaLayout = getCGALayout(memDescTy.getEncoding());
  if (cgaLayout != CGAEncodingAttr::get1CTALayout(&context, cgaLayout.getRank()))
    return invalidLayout("AIU canonical shared encoding currently requires one CTA");

  auto physical = toSharedPhysicalLayout(context);
  if (!physical)
    return physical.takeError();
  unsigned perPhase = getSwizzledBytes() == 64 ? 2 : 1;
  unsigned maxPhase = getSwizzledBytes() == 64 ? 4 : 8;
  auto rootCGA = CGAEncodingAttr::get1CTALayout(&context, 2);
  auto rootEncoding = SwizzledSharedEncodingAttr::get(
      &context, 128 / getElementBitWidth(), perPhase, maxPhase, getOrder(),
      rootCGA);
  auto standardLayout = rootEncoding.toLinearLayout(getRootShape());
  standardLayout = standardLayout.squeezeIns(StringAttr::get(&context, "block"));
  auto dim0 = StringAttr::get(&context, "dim0");
  auto dim1 = StringAttr::get(&context, "dim1");
  auto normalizedPhysical = physical->logicalToPhysical.transposeIns({dim0, dim1});
  auto normalizedStandard = standardLayout.invert().transposeIns({dim0, dim1});
  if (!standardLayout.isInvertible() || normalizedPhysical != normalizedStandard)
    return invalidLayout("AIU physical layout is not canonical swizzled shared");

  auto bulkEncoding = SwizzledSharedEncodingAttr::get(
      &context, 128 / getElementBitWidth(), 1, 1, getOrder(), rootCGA);
  auto bulkLayout = bulkEncoding.toLinearLayout(getRootShape());
  bulkLayout = bulkLayout.squeezeIns(StringAttr::get(&context, "block"));
  auto normalizedBulkPhysical = physical->logicalToBulkBase.transposeIns({dim0, dim1});
  auto normalizedBulkStandard = bulkLayout.invert().transposeIns({dim0, dim1});
  if (!bulkLayout.isInvertible() ||
      normalizedBulkPhysical != normalizedBulkStandard)
    return invalidLayout("AIU bulk base layout is not canonical shared");

  auto fullOrder = mlir::triton::gpu::getOrder(memDescTy);
  auto encoding = SwizzledSharedEncodingAttr::get(
      &context, 128 / getElementBitWidth(), perPhase, maxPhase, fullOrder,
      cgaLayout);
  if (getAllocationElems(encoding, allocShape) != mlir::product(allocShape) ||
      mlir::product(allocShape) % getPhysicalStageElems() != 0)
    return invalidLayout("AIU canonical shared encoding allocation mismatch");
  return encoding;
}

llvm::Expected<Attribute>
PPUAIUPlan::toSharedEncoding(MemDescType memDescTy) const {
  auto canonical = toCanonicalSwizzledSharedEncoding(memDescTy);
  if (canonical)
    return Attribute(*canonical);
  llvm::consumeError(canonical.takeError());

  MLIRContext &context = *memDescTy.getContext();
  auto physical = toSharedPhysicalLayout(context);
  if (!physical)
    return physical.takeError();
  StringAttr expandedDim;
  if (getStorageFactor() == 2)
    expandedDim = StringAttr::get(&context, getOrder()[1] == 0 ? "dim1" : "dim0");
  auto logical = invertPhysicalLayout(physical->logicalToPhysical, expandedDim);
  if (!logical)
    return logical.takeError();
  auto block = StringAttr::get(&context, "block");
  auto bases = logical->getBases();
  bases[block] = {};
  auto shared = triton::LinearLayout::tryCreate(
      bases, logical->getOutDims(), /*requireSurjective=*/false);
  if (!shared)
    return invalidLayout("failed to construct shared linear layout");
  return Attribute(SharedLinearEncodingAttr::get(&context, *shared,
                                                 physical->alignmentBytes));
}

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::fromSharedEncoding(MemDescType memDescTy, unsigned numWarps,
                               ArrayRef<unsigned> strategy) {
  if (!isa<SwizzledSharedEncodingAttr, SharedLinearEncodingAttr>(
          memDescTy.getEncoding()))
    return invalidLayout("AIU writer requires a supported shared encoding");
  auto shape = getShapePerCTA(memDescTy);
  auto order = mlir::triton::gpu::getOrder(memDescTy);
  if (shape.size() < 2 || order.size() < 2)
    return invalidLayout("AIU writer requires rank-2 root geometry");
  auto rootShape = ArrayRef<int64_t>(shape).take_back(2);
  SmallVector<unsigned> rootOrder;
  if (order.size() == 2) {
    rootOrder.assign(order.begin(), order.end());
  } else if (order.size() == 3 && order[2] == 0 && order[0] > 0 &&
             order[1] > 0) {
    rootOrder = {order[0] - 1, order[1] - 1};
  } else {
    return invalidLayout("unsupported shared pipelined memory order");
  }
  uint64_t tileW = rootShape[0];
  uint64_t tileC = rootShape[1];
  if (rootOrder[1] != 0)
    std::swap(tileW, tileC);
  auto plan = create(numWarps, tileW, tileC, memDescTy.getElementTypeBitWidth(),
                     rootShape, rootOrder, strategy);
  if (!plan)
    return plan.takeError();
  auto expected = plan->toSharedEncoding(memDescTy);
  if (!expected)
    return expected.takeError();
  if (*expected != memDescTy.getEncoding())
    return invalidLayout("AIU writer strategy does not match shared encoding");
  return plan;
}

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::fromCanonicalSwizzled(MemDescType memDescTy, unsigned numWarps,
                                  ArrayRef<unsigned> strategy) {
  if (!isa<SwizzledSharedEncodingAttr>(memDescTy.getEncoding()))
    return invalidLayout("canonical AIU plan requires swizzled shared encoding");
  auto shape = getShapePerCTA(memDescTy);
  auto order = mlir::triton::gpu::getOrder(memDescTy);
  if (shape.size() < 2 || order.size() < 2)
    return invalidLayout("canonical AIU plan requires rank-2 root geometry");
  auto rootShape = ArrayRef<int64_t>(shape).take_back(2);
  SmallVector<unsigned> rootOrder;
  if (order.size() == 2) {
    rootOrder.assign(order.begin(), order.end());
  } else if (order.size() == 3 && order[2] == 0 && order[0] > 0 &&
             order[1] > 0) {
    rootOrder = {order[0] - 1, order[1] - 1};
  } else {
    return invalidLayout("unsupported canonical AIU pipelined memory order");
  }
  uint64_t tileW = rootShape[0];
  uint64_t tileC = rootShape[1];
  if (rootOrder[1] != 0)
    std::swap(tileW, tileC);
  auto plan = create(numWarps, tileW, tileC,
                     memDescTy.getElementTypeBitWidth(), rootShape, rootOrder,
                     strategy);
  if (!plan)
    return plan.takeError();
  auto expected = plan->toCanonicalSwizzledSharedEncoding(memDescTy);
  if (!expected)
    return expected.takeError();
  if (*expected != memDescTy.getEncoding())
    return invalidLayout("AIU writer strategy does not match standard shared encoding");
  return plan;
}

} // namespace mlir::triton::gpu
