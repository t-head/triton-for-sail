#ifndef TRITON_DIALECT_TRITONGPU_IR_PPULAYOUTSPEC_H
#define TRITON_DIALECT_TRITONGPU_IR_PPULAYOUTSPEC_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <vector>

namespace mlir::triton::gpu {

struct CanonicalLayoutParameters {
  unsigned vec = 0;
  unsigned perPhase = 0;
  unsigned maxPhase = 0;
};

struct PPULayoutConstraints {
  unsigned elementBitWidth = 0;
  SmallVector<int64_t, 2> rootShape;
  SmallVector<SmallVector<unsigned, 2>, 2> allowedOrders;
  unsigned maxVectorBits = 128;
  unsigned maxPhase = 8;
};

class PPULayoutSpec {
public:
  static llvm::Expected<PPULayoutSpec>
  createCanonical(MLIRContext &context, unsigned elementBitWidth,
                  ArrayRef<int64_t> rootShape, ArrayRef<unsigned> order,
                  unsigned vec, unsigned perPhase, unsigned maxPhase);

  static llvm::Expected<PPULayoutSpec>
  createPhysical(unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
                 ArrayRef<unsigned> order, uint64_t physicalStageElems,
                 unsigned alignmentBytes,
                 triton::LinearLayout logicalToPhysical);

  static llvm::Expected<std::vector<PPULayoutSpec>>
  enumerateCanonicalCandidates(MLIRContext &context,
                               const PPULayoutConstraints &constraints);

  static llvm::Expected<std::vector<PPULayoutSpec>>
  enumerateCanonicalCandidates(MLIRContext &context, unsigned elementBitWidth,
                               ArrayRef<int64_t> rootShape,
                               ArrayRef<unsigned> order);

  llvm::Expected<SwizzledSharedEncodingAttr>
  toCanonicalEncoding(MLIRContext &context, ArrayRef<unsigned> fullOrder = {},
                      CGAEncodingAttr cgaLayout = {}) const;

  ArrayRef<int64_t> getRootShape() const { return rootShape; }
  ArrayRef<unsigned> getOrder() const { return order; }
  unsigned getElementBitWidth() const { return elementBitWidth; }
  uint64_t getPhysicalStageElems() const { return physicalStageElems; }
  unsigned getAlignmentBytes() const { return alignmentBytes; }
  const triton::LinearLayout &getLogicalToPhysical() const {
    return logicalToPhysical;
  }
  const std::optional<CanonicalLayoutParameters> &getCanonicalParameters() const {
    return canonicalParameters;
  }

private:
  PPULayoutSpec(unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
                ArrayRef<unsigned> order, uint64_t physicalStageElems,
                unsigned alignmentBytes,
                triton::LinearLayout logicalToPhysical);

  SmallVector<int64_t, 2> rootShape;
  SmallVector<unsigned, 2> order;
  unsigned elementBitWidth;
  uint64_t physicalStageElems;
  unsigned alignmentBytes;
  triton::LinearLayout logicalToPhysical;
  std::optional<CanonicalLayoutParameters> canonicalParameters;
};

} // namespace mlir::triton::gpu

#endif
