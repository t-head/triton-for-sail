#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <vector>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace llvm;

namespace {

PPUAIUPlan validPlan(unsigned warps, uint64_t x, uint64_t channel,
                     unsigned bits, ArrayRef<unsigned> strategy) {
  SmallVector<int64_t> shape{static_cast<int64_t>(x),
                             static_cast<int64_t>(channel)};
  SmallVector<unsigned> order{1, 0};
  auto plan = PPUAIUPlan::fromLegacy(warps, x, channel, bits, shape, order,
                                     strategy);
  if (!plan)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(plan.takeError())));
  return *plan;
}

PPUAIUPlan defaultPlan(unsigned warps, uint64_t x, uint64_t channel,
                       unsigned bits) {
  SmallVector<int64_t> shape{static_cast<int64_t>(x),
                             static_cast<int64_t>(channel)};
  SmallVector<unsigned> order{1, 0};
  auto plan = PPUAIUPlan::createLegacyDefault(warps, x, channel, bits, shape,
                                              order);
  if (!plan)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(plan.takeError())));
  return *plan;
}

std::vector<PPUAIUPlan> provenCandidates(unsigned warps, uint64_t x,
                                         uint64_t channel, unsigned bits) {
  SmallVector<int64_t> shape{static_cast<int64_t>(x),
                             static_cast<int64_t>(channel)};
  SmallVector<unsigned> order{1, 0};
  auto plans = PPUAIUPlan::enumerateProvenCandidates(warps, x, channel, bits,
                                                     shape, order);
  if (!plans)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(plans.takeError())));
  return *plans;
}

void expectInvalidShared(llvm::Expected<SharedPhysicalLayout> layout) {
  EXPECT_FALSE(layout);
  llvm::consumeError(layout.takeError());
}

void expectInvalidEncoding(
    llvm::Expected<SwizzledSharedEncodingAttr> encoding) {
  EXPECT_FALSE(encoding);
  llvm::consumeError(encoding.takeError());
}

MemDescType aiuMemDesc(const PPUAIUPlan &plan, MLIRContext &context,
                       ArrayRef<int64_t> shape = {},
                       ArrayRef<unsigned> order = {}) {
  if (shape.empty())
    shape = plan.getRootShape();
  if (order.empty())
    order = plan.getOrder();
  auto cga = CGAEncodingAttr::get1CTALayout(&context, order.size());
  auto encoding = PPUAIUSharedEncodingAttr::get(
      &context, 2, plan.toLegacyStrategy(), order, cga);
  auto elementType = mlir::IntegerType::get(&context, plan.getElementBitWidth());
  return MemDescType::get(shape, elementType, encoding,
                          SharedMemorySpaceAttr::get(&context),
                          /*mutableMemory=*/true, shape);
}

MemDescType withEncoding(MemDescType type, mlir::Attribute encoding) {
  return MemDescType::get(type.getShape(), type.getElementType(), encoding,
                          type.getMemorySpace(), type.getMutableMemory(),
                          type.getAllocShape());
}

void expectInvalid(llvm::Expected<PPUAIUPlan> plan) {
  EXPECT_FALSE(plan);
  llvm::consumeError(plan.takeError());
}

unsigned outVec(const PPUAIUPlan &plan) {
  return 128 / plan.getElementBitWidth();
}

uint64_t withinCubeOffset(const PPUAIUPlan &plan, uint64_t row,
                          uint64_t col, bool swizzle) {
  unsigned perPhase = plan.getSwizzledBytes() == 64 ? 2 : 1;
  unsigned maxPhase = plan.getSwizzledBytes() == 64 ? 4 : 8;
  uint64_t phase = (row / perPhase) % maxPhase;
  uint64_t chunk = col / outVec(plan);
  uint64_t swizzledChunk = swizzle ? (chunk ^ phase) : chunk;
  return row * plan.getCubeC() * plan.getStorageFactor() +
         swizzledChunk * outVec(plan) + col % outVec(plan);
}

uint64_t distributedOffset(const PPUAIUPlan &plan, uint64_t row,
                           uint64_t col) {
  uint64_t rowCube = row / plan.getCubeW();
  uint64_t colCube = col / plan.getCubeC();
  uint64_t warpC = colCube % plan.getWarpCopyC();
  uint64_t warpW = rowCube % plan.getWarpCopyW();
  uint64_t copy = colCube / plan.getWarpCopyC();
  uint64_t elemsPerCube = static_cast<uint64_t>(plan.getCubeC()) *
                          plan.getCubeW() * plan.getStorageFactor();
  uint64_t elemsPerCopy = elemsPerCube * plan.getCopyWarps();
  uint64_t base = copy * elemsPerCopy +
                  (warpW * plan.getWarpCopyC() + warpC) * elemsPerCube;
  uint64_t innerRow = row % plan.getCubeW();
  uint64_t innerCol = col % (plan.getCubeC() * plan.getStorageFactor());
  innerRow += innerCol / plan.getCubeC();
  innerCol %= plan.getCubeC();
  return base + withinCubeOffset(plan, innerRow, innerCol, true);
}

uint64_t distributedBulkBase(const PPUAIUPlan &plan, uint64_t row,
                             uint64_t col) {
  uint64_t rowCube = row / plan.getCubeW();
  uint64_t colCube = col / plan.getCubeC();
  uint64_t warpC = colCube % plan.getWarpCopyC();
  uint64_t warpW = rowCube % plan.getWarpCopyW();
  uint64_t copy = colCube / plan.getWarpCopyC();
  uint64_t elemsPerCube = static_cast<uint64_t>(plan.getCubeC()) *
                          plan.getCubeW() * plan.getStorageFactor();
  uint64_t elemsPerCopy = elemsPerCube * plan.getCopyWarps();
  uint64_t base = copy * elemsPerCopy +
                  (warpW * plan.getWarpCopyC() + warpC) * elemsPerCube;
  uint64_t innerRow = row % plan.getCubeW();
  uint64_t innerCol = col % (plan.getCubeC() * plan.getStorageFactor());
  innerRow += innerCol / plan.getCubeC();
  innerCol %= plan.getCubeC();
  return base + withinCubeOffset(plan, innerRow, innerCol, false);
}

uint64_t bulkReaderBase(const PPUAIUPlan &plan, uint64_t row, uint64_t col) {
  uint64_t warpW = row / plan.getCubeW();
  uint64_t copy = col / (plan.getCubeC() * plan.getWarpCopyC());
  uint64_t innerChannel = col % (plan.getCubeC() * plan.getWarpCopyC());
  uint64_t warpC = innerChannel / plan.getCubeC();
  return copy * plan.getBulkCopyElems() +
         warpW * plan.getBulkCubeElems() * plan.getWarpCopyC() *
             plan.getStorageFactor() +
         warpC * plan.getBulkCubeElems() * plan.getStorageFactor() +
         (row % plan.getCubeW()) * plan.getCubeC() *
             plan.getStorageFactor() +
         col % plan.getCubeC();
}

void proveLinearLayout(const PPUAIUPlan &plan) {
  MLIRContext context;
  auto dim0 = StringAttr::get(&context, "dim0");
  auto dim1 = StringAttr::get(&context, "dim1");
  auto physical = plan.toSharedPhysicalLayout(context);
  if (!physical) {
    ADD_FAILURE() << llvm::toString(physical.takeError());
    return;
  }
  EXPECT_EQ(physical->logicalToPhysical.isSurjective(),
            plan.getStorageFactor() == 1);
  EXPECT_EQ(physical->logicalToBulkBase.isSurjective(),
            plan.getStorageFactor() == 1);
  EXPECT_EQ(physical->physicalStageElems, plan.getPhysicalStageElems());
  EXPECT_EQ(physical->alignmentBytes, 128);
  for (uint64_t row = 0; row < plan.getXElems(); ++row) {
    for (uint64_t col = 0; col < plan.getChannelElems(); ++col) {
      auto result = physical->logicalToPhysical.apply({{dim0, row}, {dim1, col}});
      ASSERT_EQ(result.size(), 1);
      EXPECT_EQ(result[0].second, distributedOffset(plan, row, col));
      auto bulk = physical->logicalToBulkBase.apply({{dim0, row}, {dim1, col}});
      ASSERT_EQ(bulk.size(), 1);
      EXPECT_EQ(bulk[0].second, distributedBulkBase(plan, row, col));
      EXPECT_EQ(result[0].second + physical->stageOffsetElems(2),
                2 * plan.getPhysicalStageElems() +
                    distributedOffset(plan, row, col));
    }
  }
}

void provePlan(const PPUAIUPlan &plan) {
  std::vector<unsigned> writers(plan.getXElems() * plan.getChannelElems(), 0);
  std::vector<unsigned> physicalWrites(plan.getPhysicalStageElems(), 0);
  for (uint64_t copy = 0;
       copy * plan.getWriterChannelAdvance() < plan.getChannelElems(); ++copy) {
    for (unsigned warp = 0; warp < plan.getNumWarps(); ++warp) {
      uint64_t warpW = warp / plan.getWarpCopyC();
      uint64_t warpC = warp % plan.getWarpCopyC();
      if (warpW >= plan.getWarpCopyW())
        continue;
      uint64_t startW = warpW * plan.getCubeW();
      uint64_t startC = copy * plan.getWriterChannelAdvance() +
                        warpC * plan.getCubeC();
      if (startC >= plan.getChannelElems())
        continue;
      for (uint64_t r = 0; r < plan.getCubeW(); ++r) {
        for (uint64_t c = 0; c < plan.getCubeC(); ++c) {
          uint64_t row = startW + r;
          uint64_t col = startC + c;
          if (row >= plan.getXElems() || col >= plan.getChannelElems())
            continue;
          uint64_t writer = copy * plan.getWriterCopyElems() +
                            warp * plan.getWriterCubeElems() +
                            withinCubeOffset(plan, r, c, true);
          uint64_t reader = distributedOffset(plan, row, col);
          ASSERT_EQ(writer, reader);
          ASSERT_LT(reader, plan.getPhysicalStageElems());
          ++writers[row * plan.getChannelElems() + col];
          ++physicalWrites[reader];
        }
      }
    }
  }
  for (unsigned count : writers)
    ASSERT_EQ(count, 1);
  for (unsigned count : physicalWrites)
    ASSERT_LE(count, 1);
  for (uint64_t row = 0; row < plan.getXElems(); row += 8) {
    for (uint64_t col = 0; col < plan.getChannelElems(); col += outVec(plan))
      ASSERT_EQ(distributedBulkBase(plan, row, col),
                bulkReaderBase(plan, row, col));
  }
  proveLinearLayout(plan);
}

} // namespace

TEST(PPUAIUPlan, LegacyDefaultsMatchExpected) {
  EXPECT_EQ(defaultPlan(1, 16, 16, 16).toLegacyStrategy(),
            (SmallVector<unsigned>{16, 16, 1, 1, 64}));
  EXPECT_EQ(defaultPlan(4, 64, 64, 16).toLegacyStrategy(),
            (SmallVector<unsigned>{64, 16, 1, 4, 128}));
  EXPECT_EQ(defaultPlan(2, 16, 32, 8).toLegacyStrategy(),
            (SmallVector<unsigned>{32, 16, 1, 1, 64}));
  EXPECT_EQ(defaultPlan(4, 16, 16, 32).toLegacyStrategy(),
            (SmallVector<unsigned>{16, 16, 1, 1, 64}));
  EXPECT_EQ(defaultPlan(2, 64, 128, 16).toLegacyStrategy(),
            (SmallVector<unsigned>{64, 64, 2, 1, 128}));
}

TEST(PPUAIUPlan, PhysicalMappingCoversWriterReadersAndStages) {
  provePlan(defaultPlan(1, 16, 16, 16));
  provePlan(defaultPlan(4, 64, 64, 16));
  provePlan(defaultPlan(2, 64, 128, 16));
  provePlan(validPlan(4, 64, 64, 16, {32, 32, 2, 2, 64}));
  provePlan(defaultPlan(2, 16, 32, 8));
  provePlan(defaultPlan(4, 64, 16, 32));
}

TEST(PPUAIUPlan, ExplicitFactorTwoCoordinates) {
  auto plan = defaultPlan(1, 16, 16, 16);
  EXPECT_EQ(plan.getPhysicalStageElems(), 512);
  EXPECT_EQ(distributedOffset(plan, 0, 0), 0);
  EXPECT_EQ(distributedOffset(plan, 0, 8), 8);
  EXPECT_EQ(distributedOffset(plan, 1, 0), 32);
  EXPECT_EQ(distributedOffset(plan, 2, 0), 72);
  EXPECT_EQ(distributedOffset(plan, 2, 8), 64);
  EXPECT_EQ(distributedOffset(plan, 15, 15), 503);
}

TEST(PPUAIUPlan, EnumeratesOnlyProvenWarpVariants) {
  auto factorOne = provenCandidates(4, 64, 128, 16);
  EXPECT_EQ(factorOne.size(), 5);
  EXPECT_EQ(factorOne.front().toLegacyStrategy(),
            defaultPlan(4, 64, 128, 16).toLegacyStrategy());
  for (const auto &plan : factorOne)
    provePlan(plan);

  auto factorTwo = provenCandidates(4, 64, 16, 16);
  EXPECT_EQ(factorTwo.size(), 3);
  for (const auto &plan : factorTwo) {
    EXPECT_EQ(plan.getChannelElems(), plan.getCubeC());
    EXPECT_EQ(plan.getWarpCopyC(), 1);
    provePlan(plan);
  }

  EXPECT_EQ(provenCandidates(4, 32, 32, 8).size(), 2);
  EXPECT_EQ(provenCandidates(4, 64, 16, 32).size(), 3);
}

TEST(PPUAIUPlan, RejectsUnprovenGeometry) {
  SmallVector<int64_t> shape{16, 16};
  SmallVector<unsigned> order{1, 0};
  expectInvalid(PPUAIUPlan::createLegacyDefault(1, 16, 16, 8, shape, order));
  SmallVector<unsigned> badRows{16, 15, 1, 1, 64};
  expectInvalid(PPUAIUPlan::fromLegacy(1, 16, 16, 16, shape, order, badRows));
  SmallVector<int64_t> nonDivisibleShape{33, 64};
  expectInvalid(PPUAIUPlan::createLegacyDefault(2, 33, 64, 16,
                                                nonDivisibleShape, order));
  SmallVector<int64_t> partialChannelShape{32, 80};
  expectInvalid(PPUAIUPlan::createLegacyDefault(2, 32, 80, 16,
                                                partialChannelShape, order));
}

TEST(PPUAIUPlan, RejectsFactorTwoMultipleChannelCubes) {
  SmallVector<int64_t> shape{16, 32};
  SmallVector<unsigned> order{1, 0};
  SmallVector<unsigned> strategy{16, 16, 2, 1, 64};
  expectInvalid(PPUAIUPlan::fromLegacy(2, 16, 32, 16, shape, order, strategy));
}

TEST(PPUAIUPlan, RejectsNonPowerOfTwoPhysicalLayout) {
  auto plan = validPlan(3, 48, 64, 16, {64, 16, 1, 3, 128});
  MLIRContext context;
  expectInvalidShared(plan.toSharedPhysicalLayout(context));
}

TEST(PPUAIUPlan, CanonicalSharedEncodingMatchesPhysicalLayout) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  auto plan = defaultPlan(4, 64, 64, 16);
  auto type = aiuMemDesc(plan, context);
  auto encoding = plan.toCanonicalSwizzledSharedEncoding(type);
  ASSERT_TRUE(!!encoding);
  EXPECT_EQ(encoding->getVec(), 8);
  EXPECT_EQ(encoding->getPerPhase(), 1);
  EXPECT_EQ(encoding->getMaxPhase(), 8);
  EXPECT_EQ(encoding->getOrder(), ArrayRef<unsigned>({1, 0}));

  auto standardType = withEncoding(type, *encoding);
  auto reconstructed = PPUAIUPlan::fromCanonicalSwizzled(
      standardType, 4, plan.toLegacyStrategy());
  ASSERT_TRUE(!!reconstructed);
  EXPECT_EQ(reconstructed->toLegacyStrategy(), plan.toLegacyStrategy());

  auto warpVariant = validPlan(4, 64, 64, 16, {64, 64, 1, 1, 128});
  auto variantEncoding =
      warpVariant.toCanonicalSwizzledSharedEncoding(aiuMemDesc(warpVariant, context));
  ASSERT_TRUE(!!variantEncoding);
  EXPECT_EQ(*variantEncoding, *encoding);
  auto variantReconstructed = PPUAIUPlan::fromCanonicalSwizzled(
      withEncoding(aiuMemDesc(warpVariant, context), *variantEncoding), 4,
      warpVariant.toLegacyStrategy());
  ASSERT_TRUE(!!variantReconstructed);
  EXPECT_EQ(variantReconstructed->toLegacyStrategy(),
            warpVariant.toLegacyStrategy());
}

TEST(PPUAIUPlan, CanonicalSharedEncodingSupportsTypesOrdersAndStages) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  for (auto bits : {8, 16, 32}) {
    uint64_t channel = bits == 32 ? 16 : 64;
    auto plan = defaultPlan(4, 64, channel, bits);
    auto encoding = plan.toCanonicalSwizzledSharedEncoding(
        aiuMemDesc(plan, context));
    ASSERT_TRUE(!!encoding);
    EXPECT_EQ(encoding->getVec(), 128 / bits);
    EXPECT_EQ(encoding->getPerPhase(), plan.getSwizzledBytes() == 64 ? 2 : 1);
    EXPECT_EQ(encoding->getMaxPhase(), plan.getSwizzledBytes() == 64 ? 4 : 8);
  }

  SmallVector<int64_t> shape{64, 64};
  SmallVector<unsigned> order{0, 1};
  auto transposed = PPUAIUPlan::createLegacyDefault(4, 64, 64, 16, shape, order);
  ASSERT_TRUE(!!transposed);
  auto transposedEncoding = transposed->toCanonicalSwizzledSharedEncoding(
      aiuMemDesc(*transposed, context));
  ASSERT_TRUE(!!transposedEncoding);
  EXPECT_EQ(transposedEncoding->getOrder(), ArrayRef<unsigned>({0, 1}));

  auto plan = defaultPlan(4, 64, 64, 16);
  SmallVector<int64_t> stagedShape{2, 64, 64};
  SmallVector<unsigned> stagedOrder{2, 1, 0};
  auto stagedType = aiuMemDesc(plan, context, stagedShape, stagedOrder);
  auto stagedEncoding = plan.toCanonicalSwizzledSharedEncoding(stagedType);
  ASSERT_TRUE(!!stagedEncoding);
  EXPECT_EQ(stagedEncoding->getOrder(), ArrayRef<unsigned>({2, 1, 0}));
  EXPECT_EQ(getAllocationElems(*stagedEncoding, stagedShape), 8192);
  auto stagedPlan = PPUAIUPlan::fromCanonicalSwizzled(
      withEncoding(stagedType, *stagedEncoding), 4, plan.toLegacyStrategy());
  ASSERT_TRUE(!!stagedPlan);
}

TEST(PPUAIUPlan, RejectsNonCanonicalSharedEncoding) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  auto factorTwo = defaultPlan(1, 16, 16, 16);
  expectInvalidEncoding(
      factorTwo.toCanonicalSwizzledSharedEncoding(aiuMemDesc(factorTwo, context)));

  auto channelCopies = defaultPlan(2, 64, 128, 16);
  expectInvalidEncoding(channelCopies.toCanonicalSwizzledSharedEncoding(
      aiuMemDesc(channelCopies, context)));

  auto blockedChannels = validPlan(4, 64, 64, 16, {32, 32, 2, 2, 64});
  expectInvalidEncoding(blockedChannels.toCanonicalSwizzledSharedEncoding(
      aiuMemDesc(blockedChannels, context)));
}
