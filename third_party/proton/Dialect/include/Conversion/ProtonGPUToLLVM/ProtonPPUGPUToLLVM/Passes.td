#ifndef PROTONPPUGPU_TO_LLVM_PASSES
#define PROTONPPUGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonPPUGPUToLLVM : Pass<"convert-proton-ppu-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using PPU-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonPPUGPUToLLVMPass(80)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

#endif // PROTONPPUGPU_TO_LLVM_PASSES
