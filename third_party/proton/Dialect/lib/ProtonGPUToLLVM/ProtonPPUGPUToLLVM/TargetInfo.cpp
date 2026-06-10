#include "Conversion/ProtonGPUToLLVM/ProtonPPUGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/ppu/include/TritonPPUGPUToLLVM/TIXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::proton::gpu::PPU {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {

  auto getClockReg = [&](const std::string &clkName) {
    ppu::TIXBuilder builder;
    auto &movLow = builder.create("ppu.mov")->o("u32");
    auto *destLowOpr = builder.newOperand("=r");
    auto *sRegLowOpr = builder.newConstantOperand(clkName);
    movLow(destLowOpr, sRegLowOpr);
    Value clkLow32 =
        builder.launch(rewriter, loc, rewriter.getIntegerType(32), true);
    return clkLow32;
  };

  Value clkLow32 = getClockReg("%clock");

  if (!isClock64)
    return clkLow32;

  Value clkHigh32 = getClockReg("%clock_hi");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value clkLow64 = b.zext(i64_ty, clkLow32);
  Value clkHigh64 = b.zext(i64_ty, clkHigh32);
  Value clock64 = b.or_(b.shl(clkHigh64, b.i64_val(32)), clkLow64);
  return clock64;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  // globaltimer is a 64-bit global clock counter in nanoseconds.
  ppu::TIXBuilder builder;
  auto &mov = builder.create("ppu.mov")->o("u64");
  auto *destOpr = builder.newOperand("=l");
  auto *sRegOpr = builder.newConstantOperand("%globaltimer");
  mov(destOpr, sRegOpr);
  return builder.launch(rewriter, loc, rewriter.getIntegerType(64), true);
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return NVVM::SmIdOp::create(rewriter, loc, i32_ty);
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "and GlobalMemorySpace for now");
  }
  return spaceId;
}

int TargetInfo::getIndexPtrAddrSpace() const {
  // Internal buffer index is private to each thread, we use generic address
  // space for PPU GPUs (same as NVIDIA).
  return 0;
}

} // namespace mlir::triton::proton::gpu::PPU
