add_triton_library(TritonPPUGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandPPUAIUV1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandPPUAIUV1m8.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandPPUAIUV2.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/PPUMMAv1.cpp
    DotOpToLLVM/PPUMMAv1m8.cpp
    DotOpToLLVM/PPUMMAv2.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    TIXAsmFormat.cpp
    Utility.cpp
    AIUUtility.cpp
    Fp4ToFpOpToLLVM.cpp
    TargetInfo.cpp
    Allocation.cpp
    ConvertLibdeviceFuncToPPU.cpp

    DEPENDS
    TritonPPUGPUConversionPassIncGen
    PPUGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    TritonInstrumentToLLVM
    MLIRReconcileUnrealizedCasts
    PPUGPUIR
    MLIRUBToLLVM
)
