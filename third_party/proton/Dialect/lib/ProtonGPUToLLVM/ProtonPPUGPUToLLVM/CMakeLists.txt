include_directories(${PROJECT_SOURCE_DIR}/third_party/ppu/include)

add_triton_library(ProtonPPUGPUToLLVM
    TargetInfo.cpp
    PPUPatternProtonGPUOpToLLVM.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonPPUGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonPPUGPUToLLVM
)
