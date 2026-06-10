#ifndef PROTONGPU_TO_LLVM_TARGETINFO_PPU_H
#define PROTONGPU_TO_LLVM_TARGETINFO_PPU_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/ppu/lib/TritonPPUGPUToLLVM/TargetInfo.h"

namespace mlir::triton::proton::gpu::PPU {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::ppu::TargetInfo &helper)
      : mlir::triton::proton::gpu::TargetInfoBase(helper) {}

  const mlir::triton::ppu::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::ppu::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  Value globalTime(ConversionPatternRewriter &rewriter,
                   Location loc) const override;

  Value processorId(ConversionPatternRewriter &rewriter,
                    Location loc) const override;

  int getAddressSpace(Attribute addressSpace) const override;

  int getIndexPtrAddrSpace() const override;

  ~TargetInfo() {}
};
} // namespace mlir::triton::proton::gpu::PPU

#endif // PROTONGPU_TO_LLVM_TARGETINFO_PPU_H
