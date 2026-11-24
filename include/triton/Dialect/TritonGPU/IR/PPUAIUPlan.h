#ifndef TRITON_DIALECT_TRITONGPU_IR_PPUAIUPLAN_H
#define TRITON_DIALECT_TRITONGPU_IR_PPUAIUPLAN_H

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/SharedPhysicalLayout.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace mlir::triton::gpu {

class PPUAIUPlan {
public:
  static llvm::Expected<PPUAIUPlan>
  createLegacyDefault(unsigned numWarps, uint64_t xElems,
                      uint64_t channelElems, unsigned elementBitWidth,
                      ArrayRef<int64_t> rootShape, ArrayRef<unsigned> order);

  static llvm::Expected<PPUAIUPlan>
  createLegacyDefault(RankedTensorType tensorTy, unsigned numWarps,
                      ArrayRef<unsigned> order);

  static llvm::Expected<PPUAIUPlan>
  fromLegacy(unsigned numWarps, uint64_t xElems, uint64_t channelElems,
             unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
             ArrayRef<unsigned> order, ArrayRef<unsigned> strategy);

  static llvm::Expected<PPUAIUPlan> fromLegacy(MemDescType memDescTy,
                                               unsigned numWarps);

  // Enumerate every scalar-valid strategy in the current finite AIU model.
  // Canonical standard-layout candidates are a subset; legacy-only plans remain included.
  static llvm::Expected<std::vector<PPUAIUPlan>>
  enumerateProvenCandidates(unsigned numWarps, uint64_t xElems,
                            uint64_t channelElems, unsigned elementBitWidth,
                            ArrayRef<int64_t> rootShape,
                            ArrayRef<unsigned> order);

  llvm::Expected<SharedPhysicalLayout>
  toSharedPhysicalLayout(MLIRContext &context) const;

  llvm::Expected<SwizzledSharedEncodingAttr>
  toCanonicalSwizzledSharedEncoding(MemDescType memDescTy) const;

  static llvm::Expected<PPUAIUPlan>
  fromCanonicalSwizzled(MemDescType memDescTy, unsigned numWarps,
                        ArrayRef<unsigned> strategy);

  SmallVector<unsigned> toLegacyStrategy() const {
    return {cubeC, cubeW, warpCopyC, warpCopyW, swizzledBytes};
  }

  ArrayRef<int64_t> getRootShape() const { return rootShape; }
  ArrayRef<unsigned> getOrder() const { return order; }
  unsigned getElementBitWidth() const { return elementBitWidth; }
  unsigned getNumWarps() const { return numWarps; }
  uint64_t getXElems() const { return xElems; }
  uint64_t getChannelElems() const { return channelElems; }
  unsigned getCubeC() const { return cubeC; }
  unsigned getCubeW() const { return cubeW; }
  unsigned getWarpCopyC() const { return warpCopyC; }
  unsigned getWarpCopyW() const { return warpCopyW; }
  unsigned getSwizzledBytes() const { return swizzledBytes; }
  unsigned getSwizzledElems() const { return swizzledElems; }
  unsigned getStorageFactor() const { return storageFactor; }
  unsigned getCopyWarps() const { return copyWarps; }
  uint64_t getPhysicalStageElems() const { return physicalStageElems; }
  uint64_t getWriterCubeElems() const { return writerCubeElems; }
  uint64_t getWriterCopyElems() const { return writerCopyElems; }
  uint64_t getWriterChannelAdvance() const { return writerChannelAdvance; }
  uint64_t getBulkCubeElems() const { return bulkCubeElems; }
  uint64_t getBulkCopyElems() const { return bulkCopyElems; }

private:
  PPUAIUPlan() = default;
  static llvm::Expected<PPUAIUPlan>
  create(unsigned numWarps, uint64_t xElems, uint64_t channelElems,
         unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
         ArrayRef<unsigned> order, ArrayRef<unsigned> strategy);
  llvm::Error validate();

  SmallVector<int64_t, 2> rootShape;
  SmallVector<unsigned, 2> order;
  unsigned elementBitWidth = 0;
  unsigned numWarps = 0;
  uint64_t xElems = 0;
  uint64_t channelElems = 0;
  unsigned cubeC = 0;
  unsigned cubeW = 0;
  unsigned warpCopyC = 0;
  unsigned warpCopyW = 0;
  unsigned swizzledBytes = 0;
  unsigned swizzledElems = 0;
  unsigned storageFactor = 0;
  unsigned copyWarps = 0;
  uint64_t physicalStageElems = 0;
  uint64_t writerCubeElems = 0;
  uint64_t writerCopyElems = 0;
  uint64_t writerChannelAdvance = 0;
  uint64_t bulkCubeElems = 0;
  uint64_t bulkCopyElems = 0;
};

} // namespace mlir::triton::gpu

#endif
