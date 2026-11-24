#ifndef TRITON_DIALECT_TRITONGPU_IR_PPUAIUWRITERSOLVER_H
#define TRITON_DIALECT_TRITONGPU_IR_PPUAIUWRITERSOLVER_H

#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "triton/Dialect/TritonGPU/IR/PPULayoutSpec.h"
#include <optional>
#include <vector>

namespace mlir::triton::gpu {

struct PPUAIUWriterSolution {
  std::vector<PPUAIUPlan> writers;
  std::optional<PPUAIUPlan> legacyDefault;
};

class PPUAIUWriterSolver {
public:
  static llvm::Expected<PPUAIUWriterSolution>
  solve(const PPULayoutSpec &layout, unsigned numWarps);
};

} // namespace mlir::triton::gpu

#endif
