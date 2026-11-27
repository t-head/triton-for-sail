#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>
#include <set>

namespace mlir::triton::gpu {
namespace {

llvm::Error invalidPlan(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

} // namespace

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::create(unsigned numWarps, uint64_t xElems, uint64_t channelElems,
                   unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
                   ArrayRef<unsigned> order, ArrayRef<unsigned> strategy) {
  if (strategy.size() != 5)
    return invalidPlan("AIUStrategy must contain exactly five fields");
  PPUAIUPlan plan;
  plan.rootShape.assign(rootShape.begin(), rootShape.end());
  plan.order.assign(order.begin(), order.end());
  plan.elementBitWidth = elementBitWidth;
  plan.numWarps = numWarps;
  plan.xElems = xElems;
  plan.channelElems = channelElems;
  plan.cubeC = strategy[0];
  plan.cubeW = strategy[1];
  plan.warpCopyC = strategy[2];
  plan.warpCopyW = strategy[3];
  plan.swizzledBytes = strategy[4];
  if (auto error = plan.validate())
    return std::move(error);
  return plan;
}

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::createLegacyDefault(unsigned numWarps, uint64_t xElems,
                                uint64_t channelElems, unsigned elementBitWidth,
                                ArrayRef<int64_t> rootShape,
                                ArrayRef<unsigned> order) {
  if (!numWarps || xElems < 16 || !channelElems ||
      (elementBitWidth != 8 && elementBitWidth != 16 && elementBitWidth != 32))
    return invalidPlan("unsupported AIU tile geometry or element type");
  if (channelElems % 16 != 0)
    return invalidPlan("AIU channel extent must be divisible by 16");
  unsigned elementBytes = elementBitWidth / 8;
  unsigned swizzledBytes = channelElems * elementBytes <= 64 ? 64 : 128;
  unsigned channelBytes = channelElems * elementBytes;
  unsigned cubeC = channelBytes < swizzledBytes ? channelElems
                                                : swizzledBytes / elementBytes;
  unsigned warpCMax = (channelElems + cubeC - 1) / cubeC;
  unsigned warpC = std::min<unsigned>(numWarps, warpCMax);
  unsigned warpWMax = xElems / 16;
  unsigned warpW = std::min<unsigned>(numWarps / warpC, warpWMax);
  unsigned cubeW = xElems / warpW;
  SmallVector<unsigned> strategy{cubeC, cubeW, warpC, warpW, swizzledBytes};
  return create(numWarps, xElems, channelElems, elementBitWidth, rootShape,
                order, strategy);
}

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::createLegacyDefault(RankedTensorType tensorTy, unsigned numWarps,
                                ArrayRef<unsigned> order) {
  auto shape = tensorTy.getShape();
  if (shape.size() != 2 || order.size() != 2)
    return invalidPlan("PPU0015 AIU requires a rank-2 tensor and order");
  uint64_t tileW = shape[0];
  uint64_t tileC = shape[1];
  if (order[1] != 0)
    std::swap(tileW, tileC);
  return createLegacyDefault(numWarps, tileW, tileC,
                             tensorTy.getElementTypeBitWidth(), shape, order);
}

llvm::Expected<PPUAIUPlan>
PPUAIUPlan::fromLegacy(unsigned numWarps, uint64_t xElems,
                       uint64_t channelElems, unsigned elementBitWidth,
                       ArrayRef<int64_t> rootShape, ArrayRef<unsigned> order,
                       ArrayRef<unsigned> strategy) {
  return create(numWarps, xElems, channelElems, elementBitWidth, rootShape,
                order, strategy);
}

llvm::Expected<PPUAIUPlan> PPUAIUPlan::fromLegacy(MemDescType memDescTy,
                                                  unsigned numWarps) {
  auto encoding = dyn_cast<PPUAIUSharedEncodingAttr>(memDescTy.getEncoding());
  if (!encoding || !encoding.isPPU0015())
    return invalidPlan("AIUPlan requires a PPU0015 AIU shared encoding");
  auto shape = getShapePerCTA(memDescTy);
  auto order = encoding.getOrder();
  if (shape.size() < 2 || order.size() < 2)
    return invalidPlan("PPU0015 AIU requires rank-2 root layout geometry");
  auto rootShape = ArrayRef<int64_t>(shape).take_back(2);
  SmallVector<unsigned> rootOrder;
  if (order.size() == 2) {
    rootOrder.assign(order.begin(), order.end());
  } else if (order.size() == 3 && order[2] == 0 && order[0] > 0 &&
             order[1] > 0) {
    rootOrder = {order[0] - 1, order[1] - 1};
  } else {
    return invalidPlan("unsupported AIU pipelined memory order");
  }
  uint64_t tileW = rootShape[0];
  uint64_t tileC = rootShape[1];
  if (rootOrder[1] != 0)
    std::swap(tileW, tileC);
  return create(numWarps, tileW, tileC, memDescTy.getElementTypeBitWidth(),
                rootShape, rootOrder, encoding.getAIUStrategy());
}

llvm::Expected<std::vector<PPUAIUPlan>> PPUAIUPlan::enumerateProvenCandidates(
    unsigned numWarps, uint64_t xElems, uint64_t channelElems,
    unsigned elementBitWidth, ArrayRef<int64_t> rootShape,
    ArrayRef<unsigned> order) {
  auto legacy = createLegacyDefault(numWarps, xElems, channelElems,
                                    elementBitWidth, rootShape, order);
  if (!numWarps || xElems < 16 || !channelElems ||
      (elementBitWidth != 8 && elementBitWidth != 16 && elementBitWidth != 32))
    return legacy.takeError();

  std::vector<PPUAIUPlan> candidates;
  std::set<std::array<unsigned, 5>> seen;
  auto appendCandidate = [&](ArrayRef<unsigned> strategy) {
    std::array<unsigned, 5> key{strategy[0], strategy[1], strategy[2],
                                strategy[3], strategy[4]};
    if (!seen.insert(key).second)
      return;
    auto candidate = fromLegacy(numWarps, xElems, channelElems, elementBitWidth,
                                rootShape, order, strategy);
    if (!candidate) {
      llvm::consumeError(candidate.takeError());
      return;
    }
    candidates.push_back(*candidate);
  };
  if (legacy)
    appendCandidate(legacy->toLegacyStrategy());

  unsigned elementBytes = elementBitWidth / 8;
  for (unsigned swizzledBytes : {64, 128}) {
    unsigned factorOneCubeC = swizzledBytes / elementBytes;
    for (unsigned warpC = 1; warpC <= numWarps; ++warpC) {
      if (channelElems % (static_cast<uint64_t>(factorOneCubeC) * warpC) != 0)
        continue;
      for (unsigned warpW = 1; warpW <= numWarps / warpC; ++warpW) {
        if (xElems % warpW != 0)
          continue;
        uint64_t cubeW = xElems / warpW;
        if (cubeW < 16 || cubeW > 2048)
          continue;
        SmallVector<unsigned> strategy{factorOneCubeC,
                                       static_cast<unsigned>(cubeW), warpC,
                                       warpW, swizzledBytes};
        appendCandidate(strategy);
      }
    }

    if (swizzledBytes == 64) {
      unsigned factorTwoCubeC = 32 / elementBytes;
      if (channelElems == factorTwoCubeC) {
        for (unsigned warpW = 1; warpW <= numWarps; ++warpW) {
          if (xElems % warpW != 0)
            continue;
          uint64_t cubeW = xElems / warpW;
          if (cubeW < 16 || cubeW > 2048)
            continue;
          SmallVector<unsigned> strategy{factorTwoCubeC,
                                         static_cast<unsigned>(cubeW), 1, warpW,
                                         swizzledBytes};
          appendCandidate(strategy);
        }
      }
    }
  }
  if (!legacy) {
    if (candidates.empty())
      return legacy.takeError();
    llvm::consumeError(legacy.takeError());
  }
  return candidates;
}

llvm::Error PPUAIUPlan::validate() {
  if (rootShape.size() != 2 || order.size() != 2)
    return invalidPlan("AIUPlan requires rank-2 root layout geometry");
  if (!((order[0] == 1 && order[1] == 0) || (order[0] == 0 && order[1] == 1)))
    return invalidPlan("unsupported AIU memory order");
  if (rootShape[0] <= 0 || rootShape[1] <= 0 || xElems < 16 || !channelElems)
    return invalidPlan(
        "AIUPlan requires positive root dimensions and at least 16 rows");
  uint64_t expectedX = rootShape[0];
  uint64_t expectedChannel = rootShape[1];
  if (order[1] != 0)
    std::swap(expectedX, expectedChannel);
  if (expectedX != xElems || expectedChannel != channelElems)
    return invalidPlan("AIU root shape does not match canonical tile geometry");
  if (channelElems % 16 != 0)
    return invalidPlan("AIU channel extent must be divisible by 16");
  if (elementBitWidth != 8 && elementBitWidth != 16 && elementBitWidth != 32)
    return invalidPlan("AIUPlan supports only 8/16/32-bit elements");
  if (!numWarps || !cubeC || !cubeW || !warpCopyC || !warpCopyW)
    return invalidPlan("AIUPlan cube and warp fields must be nonzero");
  if (cubeW > 2048)
    return invalidPlan("AIU cubeW exceeds 2048");
  if (static_cast<uint64_t>(cubeW) * warpCopyW != xElems)
    return invalidPlan("AIU cubeW * warpCopyW must exactly cover xElems");
  if (static_cast<uint64_t>(warpCopyC) * warpCopyW > numWarps)
    return invalidPlan("AIU copy warps exceed module numWarps");
  if (swizzledBytes != 64 && swizzledBytes != 128)
    return invalidPlan("AIU swizzle must be 64 or 128 bytes");

  unsigned elementBytes = elementBitWidth / 8;
  swizzledElems = swizzledBytes / elementBytes;
  if (!swizzledElems || swizzledElems % cubeC != 0)
    return invalidPlan("AIU swizzledElems must be divisible by cubeC");
  storageFactor = swizzledElems / cubeC;
  if (storageFactor != 1 && storageFactor != 2)
    return invalidPlan("AIU storageFactor must be 1 or 2");
  if (storageFactor == 2) {
    if (cubeC != 32 / elementBytes)
      return invalidPlan("unsupported factor-2 AIU cube geometry");
    if (channelElems != cubeC || warpCopyC != 1)
      return invalidPlan(
          "factor-2 AIU supports only one channel cube and warpCopyC == 1");
  } else {
    if (warpCopyC > (channelElems + cubeC - 1) / cubeC)
      return invalidPlan("AIU warpCopyC exceeds available channel cubes");
    if (channelElems % (static_cast<uint64_t>(cubeC) * warpCopyC) != 0)
      return invalidPlan("AIU factor-1 requires complete channel copy groups");
  }

  copyWarps = warpCopyC * warpCopyW;
  if (!checkedMul(xElems, channelElems, physicalStageElems) ||
      !checkedMul(physicalStageElems, storageFactor, physicalStageElems))
    return invalidPlan("AIU physical stage size overflows uint64");
  if (!checkedMul(static_cast<uint64_t>(cubeC), cubeW, writerCubeElems) ||
      !checkedMul(writerCubeElems, storageFactor, writerCubeElems) ||
      !checkedMul(writerCubeElems, copyWarps, writerCopyElems) ||
      !checkedMul(static_cast<uint64_t>(swizzledElems), warpCopyC,
                  writerChannelAdvance) ||
      !checkedMul(writerChannelAdvance, storageFactor, writerChannelAdvance) ||
      !checkedMul(static_cast<uint64_t>(cubeC), cubeW, bulkCubeElems) ||
      !checkedMul(bulkCubeElems, copyWarps, bulkCopyElems))
    return invalidPlan("AIU derived copy geometry overflows uint64");
  if (!writerChannelAdvance)
    return invalidPlan("AIU writer channel advance must be nonzero");
  return llvm::Error::success();
}

} // namespace mlir::triton::gpu

namespace mlir::LLVM::PPU {

FailureOr<llvm::SmallVector<unsigned>>
AIULoadStrategy(unsigned numWarps, unsigned xElems, unsigned channelElems,
                unsigned elemBytes, unsigned version) {
  if (version == 2) {
    SmallVector<int64_t> rootShape{static_cast<int64_t>(xElems),
                                   static_cast<int64_t>(channelElems)};
    SmallVector<unsigned> order{1, 0};
    auto plan = triton::gpu::PPUAIUPlan::createLegacyDefault(
        numWarps, xElems, channelElems, elemBytes * 8, rootShape, order);
    if (!plan)
      return failure();
    return plan->toLegacyStrategy();
  }

  constexpr unsigned minXElems = 16;
  constexpr unsigned maxCubeW = 2048;
  if (version != 1 || numWarps == 0 || xElems < minXElems ||
      channelElems == 0 || (elemBytes != 1 && elemBytes != 2 && elemBytes != 4))
    return failure();
  constexpr unsigned sliceByte = 32;
  constexpr unsigned maxSliceBytes = 128;
  unsigned channelBytes = channelElems * elemBytes;
  if (channelBytes < sliceByte)
    return failure();
  unsigned sliceTotal = channelBytes / sliceByte;
  unsigned numSlice;
  if (channelBytes <= maxSliceBytes) {
    numSlice = sliceTotal;
  } else if (sliceTotal % 4 == 0) {
    numSlice = 4;
  } else if (sliceTotal % 2 == 0) {
    numSlice = 2;
  } else {
    numSlice = 1;
  }
  unsigned channelCopy = sliceTotal / numSlice;
  unsigned warpC;
  unsigned warpW;
  unsigned maxWarpW = xElems / minXElems;
  if (numWarps % channelCopy == 0) {
    warpC = channelCopy;
    warpW = std::min<unsigned>(numWarps / channelCopy, maxWarpW);
  } else if (channelCopy % numWarps == 0) {
    warpC = numWarps;
    warpW = 1;
  } else {
    warpC = 1;
    warpW = std::min<unsigned>(numWarps, maxWarpW);
  }
  unsigned cubeW = xElems / warpW;
  if (cubeW > maxCubeW)
    return failure();
  unsigned cubeC = numSlice * sliceByte / elemBytes;
  return llvm::SmallVector<unsigned>{cubeC, cubeW, warpC, warpW, numSlice};
}

} // namespace mlir::LLVM::PPU
