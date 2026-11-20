#ifndef TRITON_DIALECT_TRITONGPU_IR_SHAREDPHYSICALLAYOUT_H
#define TRITON_DIALECT_TRITONGPU_IR_SHAREDPHYSICALLAYOUT_H

#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/Error.h"

namespace mlir::triton::gpu {

class PPUAIUPlan;

struct SharedPhysicalLayout {
  triton::LinearLayout logicalToPhysical;
  triton::LinearLayout logicalToBulkBase;
  uint64_t physicalStageElems = 0;
  SmallVector<unsigned, 2> order;
  unsigned alignmentBytes = 0;
  unsigned swizzledBytes = 0;

  uint64_t stageOffsetElems(uint64_t stage) const {
    return stage * physicalStageElems;
  }

  static llvm::Expected<SharedPhysicalLayout>
  fromPlan(const PPUAIUPlan &plan, MLIRContext &context);
};

} // namespace mlir::triton::gpu

#endif
