#ifndef PROTONGPU_TO_LLVM_PPU_PATTERN_PROTONGPUOP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_PPU_PATTERN_PROTONGPUOP_TO_LLVM_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
namespace proton::gpu {
namespace PPU {

void populateProtonGPUOpPPUPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

} // namespace PPU
} // namespace proton::gpu
} // namespace mlir::triton

#endif // PROTONGPU_TO_LLVM_PPU_PATTERN_PROTONGPUOP_TO_LLVM_H
