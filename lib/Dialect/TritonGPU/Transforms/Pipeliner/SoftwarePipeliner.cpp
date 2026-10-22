#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/ppu/include/Dialect/TritonPPUGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/Dump.h"
#include "llvm/Support/MathExtras.h"
//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static void pipelineWgmma(ModuleOp moduleOp, unsigned numStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    if (getNumStagesOrDefault(forOp, numStages) >= 1)
      mlir::triton::asyncLaunchDots(forOp);
  }
}

static bool hasMMAv5WaitsInLastStage(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int maxStage = schedule.getNumStages() - 1;
  bool hasMMAv5 = false;
  bool hasWaitInLastStage = false;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<triton::nvidia_gpu::WaitBarrierOp>(op) &&
        schedule[&op].first == maxStage) {
      hasWaitInLastStage = true;
    }
    if (isa<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
      hasMMAv5 = true;
    }
  }
  return hasMMAv5 && hasWaitInLastStage;
}

// Return the per-iteration change of an AIU offset.
static std::optional<int64_t> getAIUOffsetDelta(Value value, scf::ForOp loop,
                                              unsigned depth = 0) {
  if (depth > 16)
    return std::nullopt;
  // Fixed values such as a row offset do not change between iterations.
  if (loop.isDefinedOutsideOfLoop(value))
    return 0;
  // For a loop counter i, delta is the loop step.
  if (value == loop.getInductionVar())
    return getConstantIntValue(getAsOpFoldResult(loop.getStep()));

  // Loop-carried offsets, e.g. offset -= 64 or offset += -64.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != loop.getBody())
      return std::nullopt;
    // scf.yield omits the induction variable at argument 0.
    Value next =
        loop.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
    if (next == value)
      return 0;
    if (auto add = next.getDefiningOp<arith::AddIOp>()) {
      if (add.getLhs() == value)
        return getConstantIntValue(getAsOpFoldResult(add.getRhs()));
      if (add.getRhs() == value)
        return getConstantIntValue(getAsOpFoldResult(add.getLhs()));
    }
    if (auto sub = next.getDefiningOp<arith::SubIOp>()) {
      if (sub.getLhs() == value) {
        auto step = getConstantIntValue(getAsOpFoldResult(sub.getRhs()));
        int64_t delta;
        if (step && !llvm::SubOverflow(int64_t{0}, *step, delta))
          return delta;
      }
    }
    return std::nullopt;
  }

  Operation *op = value.getDefiningOp();
  // Add/sub expressions, e.g. T - 1 - i, combine the operand deltas.
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto lhs = getAIUOffsetDelta(op->getOperand(0), loop, depth + 1);
    auto rhs = getAIUOffsetDelta(op->getOperand(1), loop, depth + 1);
    int64_t delta;
    if (lhs && rhs &&
        !(isa<arith::AddIOp>(op) ? llvm::AddOverflow(*lhs, *rhs, delta)
                               : llvm::SubOverflow(*lhs, *rhs, delta)))
      return delta;
  }
  // Constant scaling, e.g. i * 64 or (T - 1 - i) * 64.
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    Value varying = mul.getLhs();
    auto factor = getConstantIntValue(getAsOpFoldResult(mul.getRhs()));
    if (!factor) {
      factor = getConstantIntValue(getAsOpFoldResult(mul.getLhs()));
      varying = mul.getRhs();
    }
    if (factor) {
      auto step = getAIUOffsetDelta(varying, loop, depth + 1);
      int64_t delta;
      if (step && !llvm::MulOverflow(*step, *factor, delta))
        return delta;
    }
  }
  return std::nullopt;
}

static bool needsAIUEpiloguePeeling(
    scf::ForOp loop, ArrayRef<std::pair<Operation *, unsigned>> schedule) {
  unsigned maxStage = 0;
  for (auto [op, stage] : schedule)
    maxStage = std::max(maxStage, stage);
  for (auto [op, stage] : schedule) {
    auto copy = dyn_cast<triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp>(op);
    // A copy in the last stage does not prefetch future iterations.
    if (!copy || stage == maxStage)
      continue;
    // Transpose can move the contiguous dimension; use the layout order.
    auto encoding =
        cast<PPUAIUSharedEncodingAttr>(copy.getResult().getType().getEncoding());
    auto delta =
        getAIUOffsetDelta(copy.getCoord()[encoding.getOrder().front()], loop);
    // Keep the default pipeline when the coordinate change is unknown.
    if (delta && *delta < 0)
      return true;
  }
  return false;
}

static void expandLoops(ModuleOp moduleOp) {
  DenseSet<MaskOp> peeledMaskOps;
  auto processPeeledEpilogueOp = [&](RewriterBase &rewriter, Operation *op,
                                     bool isEpilogue) -> Operation * {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (auto predOp = dyn_cast<triton::gpu::PredicateStageOp>(op)) {
      if (isEpilogue) {
        // Return false for the predicate of the peeled iteration
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 0);
      }
      if (predOp.getStage() == predOp.getMaxStage() - 1) {
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 1);
      }
      return triton::emitPredicateForStage(
                 rewriter, predOp.getIv(), predOp.getUb(), predOp.getStep(),
                 predOp.getMaxStage(), predOp.getStage())
          .getDefiningOp();
    }
    if (auto maskOp = dyn_cast<triton::gpu::MaskOp>(op)) {
      if (isEpilogue) {
        peeledMaskOps.insert(maskOp);
      }
    }
    return op;
  };

  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      continue;
    }

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    triton::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = needsAIUEpiloguePeeling(forOp, finalSchedule);
    options.predicateFn = wrapInMaskOp;
    options.getScheduleFn =
        [&](scf::ForOp forOp,
            std::vector<std::pair<Operation *, unsigned>> &schedule) {
          schedule = finalSchedule;
        };

    // Testing feature: allow for unresolved predicate stage ops
    // in the loop body.
    bool keepPredicateStage = forOp->hasAttr("__test_keep_predicate_stage");
    // TODO: Enable epilogue peeling for warp specialized loops
    // Heuristic: only peel epilogue for MMAv5 loops with waits in the last
    // stage
    bool customEpiloguePeeling =
        !options.peelEpilogue && hasMMAv5WaitsInLastStage(forOp, schedule) &&
        !forOp->getParentOfType<triton::gpu::WarpSpecializeOp>() &&
        !keepPredicateStage; // do not peel if we are testing the stage
                             // predication

    if (keepPredicateStage || customEpiloguePeeling) {
      options.emitPredicateStageFn =
          [](RewriterBase &rewriter, Value inductionVar, Value upperBound,
             Value step, uint64_t maxStage, uint64_t stage) {
            return triton::gpu::PredicateStageOp::create(
                rewriter, inductionVar.getLoc(), inductionVar, upperBound, step,
                maxStage, stage);
          };
    }
    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        triton::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp)) {
      continue;
    }
    forOp = *newForOp;
    if (customEpiloguePeeling) {
      mlir::triton::peelLoopEpilogue(forOp, processPeeledEpilogueOp);
    }

    // Prune all the statically dead mask ops in the epilogue. This is a
    // hack, ideally we should do it for all the mask ops, but it is incorrect
    // if we have speculatively executed async cp operations that will store to
    // shmem even if the mask is false.
    for (auto maskOp : peeledMaskOps) {
      rewriter.setInsertionPoint(maskOp);
      if (isConstantIntValue(maskOp.getPred(), 0)) {
        SmallVector<Value> results;
        for (auto result : maskOp->getResults()) {
          auto poisonOp = mlir::ub::PoisonOp::create(rewriter, maskOp->getLoc(),
                                                     result.getType());
          results.push_back(poisonOp);
        }
        maskOp->replaceAllUsesWith(results);
        maskOp->erase();
      }
    }
    peeledMaskOps.clear();
  }
  assert(moduleOp.getOps<triton::gpu::PredicateStageOp>().empty() &&
         "PredicateStageOp should be resolved after the pipeline expansion");
  assert(verify(moduleOp).succeeded());
  resolveMaskOp(moduleOp);
}

struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {

  using impl::TritonGPUPipelineBase<PipelinePass>::TritonGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Transform the loop by introducing async operations to prepare it for
    // pipeline expansion.
    if (failed(lowerLoops(moduleOp))) {
      signalPassFailure();
      return;
    }
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: LowerLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Apply the pipeline expansion.
    expandLoops(moduleOp);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: "
             "ExpandLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Cleanup the IR from the pipeline attributes.
    removePipeliningAttributes(moduleOp);

    pipelineWgmma(moduleOp, numStages);

    // schedule the waits
    mlir::triton::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    {
      SmallVector<LoopLikeOpInterface> loops;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp, numStages) > 1)
          loops.push_back(forOp);
      });

      if (numStages > 1) {
        getOperation()->walk(
            [&](scf::WhileOp whileOp) { loops.push_back(whileOp); });
      }

      for (auto loopOp : loops) {
        mlir::triton::pipelineTMAStores(loopOp);
      }
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
