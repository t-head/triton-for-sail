#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"


#define DEBUG_TYPE "tritonppu-reorder-instructions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_TRITONPPUGPUREORDERINSTRUCTIONS
#include "TritonPPUGPUTransforms/Passes.h.inc"

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;
  if (mlir::isa<triton::gpu::DotOperandEncodingAttr>(
          cvt.getType().getEncoding()))
    return true;
  return false;
}

// Return true if it has side effects that are either unknown or writes.
static bool hasWriteSideEffect(Operation *op) {
  auto effects = getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::any_of(*effects, [](MemoryEffects::EffectInstance effect) {
      return !isa<MemoryEffects::Read, MemoryEffects::Allocate,
                  MemoryEffects::Free>(effect.getEffect());
    });
}

// Return true if there is a write side effect on any path between start and end
// ops. This assumes start dominates end.
static bool crossWriteSideEffectingOp(Operation *start, Operation *end) {
  auto ancestor = start->getBlock()->findAncestorOpInBlock(*end);
  // Couldn't find an ancestor in the same block, conservatively assume true.
  if (!ancestor)
    return true;
  Operation *nextOp = start->getNextNode();
  while (nextOp) {
    if ((hasWriteSideEffect(nextOp)))
      return true;
    if (nextOp == ancestor)
      return false;
    nextOp = nextOp->getNextNode();
  }
  assert(false && "op doesn't dominate other");
  return true;
}

class TritonPPUGPUReorderInstructionsPass
    : public impl::TritonPPUGPUReorderInstructionsBase<
          TritonPPUGPUReorderInstructionsPass> {
public:
  TritonPPUGPUReorderInstructionsPass() = default;

  Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> users;
    for (auto user : op->getUsers()) {
    if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
      users.push_back(ancestor);
    }
    auto minOpIt =
      llvm::min_element(users, [](mlir::Operation *a, mlir::Operation *b) {
        return a->isBeforeInBlock(b);
      });
    return minOpIt != users.end() ? *minOpIt : nullptr;
  }

  static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
    return op->getBlock() == forOp.getBody();
  }

  static bool collectOpsToMoveAfter(Operation *op, Operation *anchorOp,
                                      SmallVector<Operation *> &opsToMove) {
    // Rule 1: op and anchorOp must be in the same ForOp.
    auto forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp) {
      LDBG("Unsafe! op is not inside a ForOp");
      return false;
    }
    if (anchorOp->getParentOfType<scf::ForOp>() != forOp) {
      LDBG("Unsafe! op and anchorOp are not in the same ForOp");
      return false;
    }

    // Rule 2: Both op and anchorOp are directly in the ForOp body.
    if (!isDirectlyInForBody(op, forOp)) {
      LDBG("Unsafe! op is not directly in ForOp body");
      return false;
    }
    if (!isDirectlyInForBody(anchorOp, forOp)) {
      LDBG("Unsafe! anchorOp is not directly in ForOp body");
      return false;
    }

    // `op` and `anchorOp` are the same and do not need to be moved.
    if (op == anchorOp) {
      LDBG("op and anchorOp are the same");
      return false;
    }

    // Rule 4: There cannot be a for/while loop between op and anchorOp
    // Determine the traversal direction
    // (regardless of whether op comes before or after anchorOp)
    Operation *first = anchorOp->isBeforeInBlock(op) ? anchorOp : op;
    Operation *second = anchorOp->isBeforeInBlock(op) ? op : anchorOp;
    auto it = std::next(mlir::Block::iterator(first));
    auto end = mlir::Block::iterator(second);
    for (; it != end; ++it) {
      if (isa<scf::ForOp, scf::WhileOp>(&*it)) {
        LDBG("Unsafe! for/while between anchorOp and op: ");
        LLVM_DEBUG(it->dump());
        return false;
      }
    }

    // Collect ops that need to move togethers.
    DenseSet<Operation *> visited;
    SmallVector<Operation *> worklist;
    worklist.push_back(op);

    while (!worklist.empty()) {
      Operation *curr = worklist.pop_back_val();
      if (visited.contains(curr))
        continue;
      visited.insert(curr);

      for (auto operand : curr->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;

        // Safely skip dependencies that are not in the ForOp body or before the anchorOp.
        if (defOp->getBlock() != forOp.getBody() ||
            !anchorOp->isBeforeInBlock(defOp)) {
          LDBG("dep is before anchorOp or outside ForOp body, safe to skip: ");
          LLVM_DEBUG(defOp->dump());
          continue;
        }

        // It depends on anchorOp and needs to be moved together.
        if (!defOp->hasOneUse()) {
          LDBG("Unsafe! dep has multiple uses: ");
          LLVM_DEBUG(defOp->dump());
          return false;
        }

        LDBG("Adding dep to move list: ");
        LLVM_DEBUG(defOp->dump());
        worklist.push_back(defOp);
      }
    }

    // Arrange the ops to be moved in the order they appear in the block.
    SmallVector<Operation *> candidates(visited.begin(), visited.end());
    llvm::sort(candidates, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    opsToMove = candidates;
    return true;
  }

  static bool tryMoveAfterWithDeps(Operation *op, Operation *anchorOp) {
    SmallVector<Operation *> opsToMove;
    if (!collectOpsToMoveAfter(op, anchorOp, opsToMove))
      return false;

    LDBG("Moving " << opsToMove.size() << " ops after anchorOp");
    Operation *insertAfter = anchorOp;
    for (Operation *opToMove : opsToMove) {
      LDBG("Moving op: ");
      LLVM_DEBUG(opToMove->dump());
      opToMove->moveAfter(insertAfter);
      insertAfter = opToMove;
    }
    return true;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    auto dumpIR = [&](const std::string &stage) {
      LLVM_DEBUG({
        DBGS() << "After " << stage << "\n";
        m.dump();
        DBGS() << "End " << stage << "\n";
      });
    };

    // sink conversion after the last dealloc
    // before the first use ancestor in its block
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto curr = mlir::Block::iterator(op);
      auto end = op->getBlock()->end();
      for (; curr != end && &*curr != getFirstUse(op); curr++)
        if (isa<triton::gpu::LocalDeallocOp>(&*curr))
          op->moveAfter(&*curr);
    });
    dumpIR("Stage1: sink ConvertLayout after LocalDealloc");

    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if (std::distance(user_begin, user_end) != 1)
        return;
      if (user_begin->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, *user_begin});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    dumpIR("Stage2: sink high pressure ops into loops");

    // Move alloc(load) immediately after dependent load
    m.walk([&](triton::gpu::LocalAllocOp op) {
      if (!op.getSrc())
        return;
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      // Don't hoist alloc if the src is a scalar as this may increase smem
      // pressure for no benefits.
      if (isa<arith::ConstantOp, triton::SplatOp>(argOp))
        return;
      moveAfter(op, argOp);
    });
    dumpIR("Stage3: move LocalAlloc after source Load");

    // Move transpositions just after their definition
    opToMove.clear();
    m.walk([&](triton::TransposeOpInterface op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    dumpIR("Stage4: move Transpose after definition");

    // Move `dot` operand so that conversions to opIdx=1 happens after
    // conversions to opIdx=0
    m.walk([&](triton::gpu::LocalLoadOp op) {
      auto dstEncoding = mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(
          op.getType().getEncoding());
      if (!dstEncoding)
        return;
      int opIdx = dstEncoding.getOpIdx();
      if (opIdx != 1)
        return;
      if (!op->hasOneUse())
        return;
      auto dotUser = dyn_cast<triton::DotOp>(*op->user_begin());
      if (!dotUser)
        return;
      auto AOp =
          dotUser.getOperand(0).getDefiningOp<triton::gpu::LocalLoadOp>();
      if (!AOp)
        return;
      // Check that the conversion to OpIdx=1 happens before and can be moved
      // after the conversion to OpIdx=0.
      if (!dom.dominates(op.getOperation(), AOp.getOperation()))
        return;
      if (crossWriteSideEffectingOp(op, AOp))
        return;
      moveAfter(op, AOp);
    });
    dumpIR("Stage5: reorder dot operand LocalLoads");

    // Stage6: move local_load right after previous dot
    m.walk([&](scf::ForOp forOp) {
      SmallVector<triton::DotOp> dotOps;
      // only work for local_load in loop.
      for (auto &op : forOp.getBody()->getOperations()) {
        if (auto dotOp = dyn_cast<triton::DotOp>(&op))
          dotOps.push_back(dotOp);
      }

      // multiple dotOp.
      for (int i = 1; i < (int)dotOps.size(); i++) {
        auto dotOp = dotOps[i];
        Operation *anchorOp = dotOps[i - 1].getOperation();

        auto AOp = dotOp.getOperand(0).getDefiningOp<triton::gpu::LocalLoadOp>();
        auto BOp = dotOp.getOperand(1).getDefiningOp<triton::gpu::LocalLoadOp>();

        if (BOp && BOp->hasOneUse()) {
          mlir::triton::gpu::MemDescType memDescType = BOp.getSrc().getType();
          Attribute srcLayout = memDescType.getEncoding();
          if (auto sharedEnc = dyn_cast<mlir::triton::gpu::PPUAIUSharedEncodingAttr>(srcLayout)) {
            if (!tryMoveAfterWithDeps(BOp, anchorOp)) {
              LDBG("Skipping BOp: not safe to move");
            }
          }
        }
        if (AOp && AOp->hasOneUse()) {
          mlir::triton::gpu::MemDescType memDescType = AOp.getSrc().getType();
          Attribute srcLayout = memDescType.getEncoding();
          if (auto sharedEnc = dyn_cast<mlir::triton::gpu::PPUAIUSharedEncodingAttr>(srcLayout)) {
            if (!tryMoveAfterWithDeps(AOp, anchorOp)) {
              LDBG("Skipping AOp: not safe to move");
            }
          }
        }
      }
    });
    dumpIR("Stage6: move LocalLoad right after previous DotOp");

    return;
  }
};

} // namespace mlir
