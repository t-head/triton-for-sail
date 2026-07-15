add_triton_library(PPUGPUToLLVM
    PPUGPUToLLVMPass.cpp

    DEPENDS
    PPUGPUConversionPassIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    PPUGPUIR
)
