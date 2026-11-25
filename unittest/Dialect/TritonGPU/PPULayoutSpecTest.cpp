#include "triton/Dialect/TritonGPU/IR/PPUAIUWriterSolver.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace llvm;

namespace {

PPULayoutSpec canonical(MLIRContext &context, unsigned bits,
                        ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                        unsigned perPhase, unsigned maxPhase) {
  auto spec = PPULayoutSpec::createCanonical(
      context, bits, shape, order, 128 / bits, perPhase, maxPhase);
  if (!spec)
    report_fatal_error(Twine(toString(spec.takeError())));
  return *spec;
}

PPUAIUPlan defaultPlan(unsigned warps, uint64_t x, uint64_t channel,
                       unsigned bits) {
  SmallVector<int64_t> shape{static_cast<int64_t>(x),
                             static_cast<int64_t>(channel)};
  SmallVector<unsigned> order{1, 0};
  auto plan = PPUAIUPlan::createLegacyDefault(warps, x, channel, bits, shape,
                                              order);
  if (!plan)
    report_fatal_error(Twine(toString(plan.takeError())));
  return *plan;
}

PPUAIUPlan explicitPlan(unsigned warps, uint64_t x, uint64_t channel,
                        unsigned bits, ArrayRef<unsigned> strategy) {
  SmallVector<int64_t> shape{static_cast<int64_t>(x),
                             static_cast<int64_t>(channel)};
  SmallVector<unsigned> order{1, 0};
  auto plan = PPUAIUPlan::fromLegacy(warps, x, channel, bits, shape, order,
                                     strategy);
  if (!plan)
    report_fatal_error(Twine(toString(plan.takeError())));
  return *plan;
}

MemDescType aiuType(const PPUAIUPlan &plan, MLIRContext &context) {
  auto cga = CGAEncodingAttr::get1CTALayout(&context, 2);
  auto encoding = PPUAIUSharedEncodingAttr::get(
      &context, 2, plan.toLegacyStrategy(), plan.getOrder(), cga);
  auto elementType = mlir::IntegerType::get(&context, plan.getElementBitWidth());
  return MemDescType::get(plan.getRootShape(), elementType, encoding,
                          SharedMemorySpaceAttr::get(&context), true,
                          plan.getRootShape());
}

MemDescType existingSharedType(MemDescType type, mlir::Attribute encoding) {
  SmallVector<int64_t> allocShape(type.getAllocShape());
  if (auto linear = dyn_cast<SharedLinearEncodingAttr>(encoding)) {
    auto rootShape = linear.getLinearLayout().getOutDimSizes();
    auto rank = linear.getLinearLayout().getNumOutDims();
    for (auto [index, size] : llvm::enumerate(rootShape))
      allocShape[allocShape.size() - rank + index] = size;
  }
  return MemDescType::get(type.getShape(), type.getElementType(), encoding,
                          type.getMemorySpace(), type.getMutableMemory(),
                          allocShape);
}

void expectMatches(const PPULayoutSpec &spec, const PPUAIUPlan &plan) {
  auto inDimNames = spec.getLogicalToPhysical().getInDimNames();
  MLIRContext *context = (*inDimNames.begin()).getContext();
  auto dim0 = StringAttr::get(context, "dim0");
  auto dim1 = StringAttr::get(context, "dim1");
  auto physical = plan.toSharedPhysicalLayout(*context);
  ASSERT_TRUE(!!physical);
  EXPECT_EQ(physical->physicalStageElems, spec.getPhysicalStageElems());
  EXPECT_EQ(physical->alignmentBytes, spec.getAlignmentBytes());
  EXPECT_EQ(physical->order, spec.getOrder());
  EXPECT_EQ(physical->logicalToPhysical.transposeIns({dim0, dim1}),
            spec.getLogicalToPhysical());
}

bool hasParameters(const std::vector<PPULayoutSpec> &specs, unsigned vec,
                   unsigned perPhase, unsigned maxPhase) {
  return llvm::any_of(specs, [&](const PPULayoutSpec &spec) {
    const auto &parameters = *spec.getCanonicalParameters();
    return parameters.vec == vec && parameters.perPhase == perPhase &&
           parameters.maxPhase == maxPhase;
  });
}

} // namespace

TEST(PPULayoutSpec, EnumeratesIndependentCanonicalParameters) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  SmallVector<int64_t> shape{64, 64};
  SmallVector<unsigned> order{1, 0};
  auto specs = PPULayoutSpec::enumerateCanonicalCandidates(context, 16, shape,
                                                           order);
  ASSERT_TRUE(!!specs);
  ASSERT_GT(specs->size(), 3);
  EXPECT_TRUE(hasParameters(*specs, 1, 1, 1));
  EXPECT_TRUE(hasParameters(*specs, 8, 2, 4));
  EXPECT_TRUE(hasParameters(*specs, 8, 1, 8));
  EXPECT_TRUE(hasParameters(*specs, 4, 1, 8));
  auto stagedSpec = canonical(context, 16, shape, order, 1, 8);
  auto staged = stagedSpec.toCanonicalEncoding(context, {2, 1, 0});
  ASSERT_TRUE(!!staged);
  EXPECT_EQ(staged->getOrder(), ArrayRef<unsigned>({2, 1, 0}));
}

TEST(PPULayoutSpec, UsesTileAndReaderConstraintsAndDeduplicatesPhysicalLayouts) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  PPULayoutConstraints constraints;
  constraints.elementBitWidth = 16;
  constraints.rootShape = {16, 32};
  constraints.allowedOrders = {{1, 0}, {0, 1}};
  constraints.maxVectorBits = 64;
  constraints.maxPhase = 4;
  auto specs = PPULayoutSpec::enumerateCanonicalCandidates(context, constraints);
  ASSERT_TRUE(!!specs);
  ASSERT_FALSE(specs->empty());
  size_t rowMajor = 0;
  size_t columnMajor = 0;
  for (size_t i = 0; i < specs->size(); ++i) {
    const auto &spec = (*specs)[i];
    const auto &parameters = *spec.getCanonicalParameters();
    auto order = spec.getOrder();
    rowMajor += order[0] == 1;
    columnMajor += order[0] == 0;
    EXPECT_LE(parameters.vec, 4);
    EXPECT_LE(parameters.vec, constraints.rootShape[order[0]]);
    EXPECT_LE(parameters.maxPhase, 4);
    EXPECT_LE(parameters.perPhase * parameters.maxPhase,
              constraints.rootShape[order[1]]);
    for (size_t j = 0; j < i; ++j) {
      if ((*specs)[j].getOrder() == spec.getOrder())
        EXPECT_NE((*specs)[j].getLogicalToPhysical(), spec.getLogicalToPhysical());
    }
  }
  EXPECT_GT(rowMajor, 0);
  EXPECT_GT(columnMajor, 0);
}

TEST(PPUAIUWriterSolver, SolvesMultipleWritersForOneLayout) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  SmallVector<int64_t> shape{64, 64};
  SmallVector<unsigned> order{1, 0};
  auto spec128 = canonical(context, 16, shape, order, 1, 8);
  auto solution128 = PPUAIUWriterSolver::solve(spec128, 4);
  ASSERT_TRUE(!!solution128);
  ASSERT_GE(solution128->writers.size(), 3);
  ASSERT_TRUE(solution128->legacyDefault.has_value());
  EXPECT_EQ(solution128->legacyDefault->toLegacyStrategy(),
            defaultPlan(4, 64, 64, 16).toLegacyStrategy());
  bool foundSerialWriter = false;
  for (const auto &writer : solution128->writers) {
    expectMatches(spec128, writer);
    foundSerialWriter |= writer.toLegacyStrategy() ==
                         SmallVector<unsigned>{64, 64, 1, 1, 128};
  }
  EXPECT_TRUE(foundSerialWriter);

  SmallVector<int64_t> shape64{64, 32};
  auto spec64 = canonical(context, 16, shape64, order, 2, 4);
  auto solution64 = PPUAIUWriterSolver::solve(spec64, 4);
  ASSERT_TRUE(!!solution64);
  ASSERT_FALSE(solution64->writers.empty());
  EXPECT_TRUE(solution64->legacyDefault.has_value());
  for (const auto &writer : solution64->writers)
    expectMatches(spec64, writer);
}

TEST(PPUAIUWriterSolver, ReturnsEmptyWhenAIUCannotRealizeLayout) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  SmallVector<int64_t> shape{64, 64};
  SmallVector<unsigned> order{1, 0};
  auto unswizzled = canonical(context, 16, shape, order, 1, 1);
  auto solution = PPUAIUWriterSolver::solve(unswizzled, 4);
  ASSERT_TRUE(!!solution);
  EXPECT_TRUE(solution->writers.empty());
  EXPECT_FALSE(solution->legacyDefault.has_value());
}

TEST(PPUAIUWriterSolver, SupportsNonCanonicalPhysicalLayoutAndDefault) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  auto legacy = defaultPlan(1, 16, 16, 16);
  auto physical = legacy.toSharedPhysicalLayout(context);
  ASSERT_TRUE(!!physical);
  auto spec = PPULayoutSpec::createPhysical(
      16, legacy.getRootShape(), legacy.getOrder(), physical->physicalStageElems,
      physical->alignmentBytes, physical->logicalToPhysical);
  ASSERT_TRUE(!!spec);
  EXPECT_FALSE(spec->getCanonicalParameters().has_value());
  auto solution = PPUAIUWriterSolver::solve(*spec, 1);
  ASSERT_TRUE(!!solution);
  ASSERT_EQ(solution->writers.size(), 1);
  ASSERT_TRUE(solution->legacyDefault.has_value());
  EXPECT_EQ(solution->legacyDefault->toLegacyStrategy(),
            legacy.toLegacyStrategy());
  expectMatches(*spec, solution->writers.front());
  auto encoding = spec->toCanonicalEncoding(context);
  EXPECT_FALSE(encoding);
  llvm::consumeError(encoding.takeError());
}

TEST(PPUAIUWriterSolver, SupportsLegacyOnlyPhysicalLayout) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  auto expected = explicitPlan(4, 64, 64, 16, {32, 32, 1, 2, 64});
  auto physical = expected.toSharedPhysicalLayout(context);
  ASSERT_TRUE(!!physical);
  auto spec = PPULayoutSpec::createPhysical(
      16, expected.getRootShape(), expected.getOrder(),
      physical->physicalStageElems, physical->alignmentBytes,
      physical->logicalToPhysical);
  ASSERT_TRUE(!!spec);
  EXPECT_FALSE(spec->getCanonicalParameters().has_value());
  auto solution = PPUAIUWriterSolver::solve(*spec, 4);
  ASSERT_TRUE(!!solution);
  ASSERT_FALSE(solution->writers.empty());
  bool found = false;
  for (const auto &writer : solution->writers) {
    expectMatches(*spec, writer);
    found |= writer.toLegacyStrategy() == expected.toLegacyStrategy();
  }
  EXPECT_TRUE(found);
}

TEST(PPUAIUWriterSolver, UsesExistingSharedEncodingsForAllPhysicalClasses) {
  MLIRContext context;
  context.loadDialect<TritonGPUDialect>();
  auto canonicalPlan = defaultPlan(4, 64, 64, 16);
  auto multiCube = defaultPlan(2, 64, 128, 16);
  auto factorTwo = defaultPlan(1, 16, 16, 16);
  auto factorTwo8Bit = defaultPlan(1, 16, 32, 8);
  auto transposed = PPUAIUPlan::createLegacyDefault(1, 16, 16, 16, {16, 16},
                                                    {0, 1});
  ASSERT_TRUE(!!transposed);
  for (auto plan : {canonicalPlan, multiCube, factorTwo, factorTwo8Bit,
                    *transposed}) {
    auto type = aiuType(plan, context);
    auto encoding = plan.toSharedEncoding(type);
    ASSERT_TRUE(!!encoding);
    EXPECT_TRUE(isa<SwizzledSharedEncodingAttr>(*encoding) ||
                isa<SharedLinearEncodingAttr>(*encoding));
    auto sharedType = existingSharedType(type, *encoding);
    auto physical = plan.toSharedPhysicalLayout(context);
    ASSERT_TRUE(!!physical);
    auto sharedLayout = toLinearLayout(sharedType);
    auto dim0 = StringAttr::get(&context, "dim0");
    auto dim1 = StringAttr::get(&context, "dim1");
    auto offset = StringAttr::get(&context, "offset");
    auto block = StringAttr::get(&context, "block");
    int32_t maxPhysicalOffset = 0;
    for (int32_t row = 0; row < plan.getRootShape()[0]; ++row) {
      for (int32_t col = 0; col < plan.getRootShape()[1]; ++col) {
        auto physicalCoordinates = physical->logicalToPhysical.apply(
            {{dim0, row}, {dim1, col}});
        ASSERT_EQ(physicalCoordinates.size(), 1);
        int32_t physicalOffset = physicalCoordinates[0].second;
        maxPhysicalOffset = std::max(maxPhysicalOffset, physicalOffset);
        auto logicalCoordinates = sharedLayout.apply(
            {{offset, physicalOffset}, {block, 0}});
        ASSERT_EQ(logicalCoordinates.size(), 2);
        EXPECT_EQ(logicalCoordinates[0].second, row);
        EXPECT_EQ(logicalCoordinates[1].second, col);
      }
    }
    EXPECT_GT(getAllocationElems(*encoding, sharedType.getShape(),
                                 sharedType.getAllocShape()),
              maxPhysicalOffset);
    auto writer = PPUAIUPlan::fromSharedEncoding(
        sharedType, plan.getNumWarps(), plan.toLegacyStrategy());
    ASSERT_TRUE(!!writer);
    EXPECT_EQ(writer->toLegacyStrategy(), plan.toLegacyStrategy());
  }
}
