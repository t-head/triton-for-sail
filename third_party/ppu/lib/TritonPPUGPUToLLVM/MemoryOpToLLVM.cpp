/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Dialect/PPUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

namespace SharedToDotOperandPPUAIUV1 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> aiuLoad);
}

namespace SharedToDotOperandPPUAIUV1m8 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> aiuLoad);
}

namespace SharedToDotOperandPPUAIUV2 {
FailureOr<Value> convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                               Location loc, Value tensor,
                               DotOperandEncodingAttr bEncoding,
                               const SharedMemoryObject &smemObj,
                               const LLVMTypeConverter *typeConverter,
                               Value thread);
}

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::ppu;
using namespace mlir::LLVM::PPU;

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const ppu::TargetInfo &targetInfo, const LLVMTypeConverter *typeConverter) {
  bool isStore = !vals.empty();

  // Remove broadcasting from regLayout
  auto removeBroadcast = actionRemoveBroadcastedRegs(regLayout);
  if (!removeBroadcast.isIdentity()) {
    if (isStore) {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      vals = removeBroadcast.apply(vals);
      return lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                             rewriter, targetInfo, typeConverter);
    } else {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      auto result =
          lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                          rewriter, targetInfo, typeConverter);
      if (succeeded(result)) {
        vals = broadcastAs(vals, regLayout);
      }
      return result;
    }
  }
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding())) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = StringAttr::get(loc.getContext(), "block");
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    return failure();
  }
  cvt = maybeSublayout.value();
  auto smemBase = smemObj.getBase();
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescType);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescType);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (bool transpose : {false, true}) {
    auto result = LLVM::PPU::lowerLdStMatrix(
        loc, cvt, transpose, vals, smemBase, affineOffset, maskSpanAffineOffset,
        llvmElemTy, rewriter, targetInfo);
    if (succeeded(result)) {
      return result;
    }
  }
  return failure();
}

std::optional<Value> lowerPPULdMatrixSwzl(
    Location loc, RankedTensorType tensorTy, MemDescType memDescType,
    bool transpose, Value smemBase, Type llvmElemTy,
    ConversionPatternRewriter &rewriter,
    const LLVMTypeConverter *typeConverter, DotOperandEncodingAttr dotEnc,
    bool isPPU0015, const LinearLayout &cvt, int nVecs) {
  if (!isPPU0015 || nVecs != 4 ||
      SharedMemoryObject::isAffineSharedMemoryAccess(memDescType))
    return std::nullopt;
  auto enc = dyn_cast<SwizzledSharedEncodingAttr>(memDescType.getEncoding());
  if (!enc)
    return std::nullopt;
  int bitwidth = tensorTy.getElementTypeBitWidth();
  if ((bitwidth != 8 && bitwidth != 16) || (transpose && bitwidth != 16))
    return std::nullopt;
  int rowElements = 128 / bitwidth;
  int elementsPerReg = 32 / bitwidth;
  if (enc.getVec() != rowElements || enc.getOrder() != ArrayRef<unsigned>{1, 0})
    return std::nullopt;
  int swizzledBytes;
  if (enc.getMaxPhase() == 8 && enc.getPerPhase() == 1) {
    swizzledBytes = 128;
  } else if (enc.getMaxPhase() == 4 && enc.getPerPhase() == 2) {
    swizzledBytes = 64;
  } else {
    return std::nullopt;
  }
  auto shape = tensorTy.getShape();
  // Restrict the initial path to the full-panel geometry validated on PPU0015.
  int panelCols = swizzledBytes * 8 / bitwidth;
  if (shape.size() != 2 || shape[0] < 32 || shape[0] % 16 != 0 ||
      shape[1] < panelCols || shape[1] % (2 * rowElements) != 0 ||
      (transpose && dotEnc.getOpIdx() == 1 && shape[1] > 64) ||
      // Wide B8 A operands have unproven packet organization.
      (!transpose && bitwidth == 8 && dotEnc.getOpIdx() == 0 && shape[1] > 128))
    return std::nullopt;

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = tensorTy.getContext();
  auto S = [ctx](StringRef value) { return StringAttr::get(ctx, value); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kOffset = S("offset");
  auto kBlock = S("block");
  auto kDim0 = S("dim0");
  auto kDim1 = S("dim1");
  LinearLayout memLayout = triton::gpu::toLinearLayout(memDescType);
  if (!memLayout.isInvertible())
    return std::nullopt;
  LinearLayout logicalToShared = memLayout.invert();
  int step = 4 * elementsPerReg;
  for (int i = 0; i < cvt.getInDimSize(kReg); i += step) {
    for (int warp = 0; warp < cvt.getInDimSize(kWarp); ++warp) {
      auto base = cvt.apply({{kReg, i}, {kLane, 0}, {kWarp, warp}});
      auto origin = memLayout.apply(
          {{kOffset, base[0].second}, {kBlock, 0}});
      int originRow = origin[0].second;
      int originCol = origin[1].second;
      if (originRow % 16 != 0 || originCol % (2 * rowElements) != 0)
        return std::nullopt;
      for (int lane = 0; lane < 32; ++lane) {
        for (int q = 0; q < 4; ++q) {
          // Raw packets are TL, BL, TR, BR; operand B exchanges r1/r2 once.
          int matrixQ = dotEnc.getOpIdx() == 1 && q == 1 ? 2
                        : dotEnc.getOpIdx() == 1 && q == 2 ? 1 : q;
          for (int h = 0; h < elementsPerReg; ++h) {
            int row = originRow + 8 * (matrixQ % 2) +
                      (transpose ? 2 * (lane % 4) + h : lane / 4);
            int col = originCol + rowElements * (matrixQ / 2) +
                      (transpose ? lane / 4
                                 : elementsPerReg * (lane % 4) + h);
            auto expected =
                logicalToShared.apply({{kDim0, row}, {kDim1, col}});
            auto actual = cvt.apply({{kReg, i + q * elementsPerReg + h},
                                     {kLane, lane},
                                     {kWarp, warp}});
            if (actual[0].second != expected[0].second)
              return std::nullopt;
          }
        }
      }
    }
  }

  auto removeBroadcast =
      actionRemoveBroadcastedRegs(triton::gpu::toLinearLayout(tensorTy));
  auto resultLayout =
      removeBroadcast.apply(triton::gpu::toLinearLayout(tensorTy));
  auto structTy = typeConverter->convertType(tensorTy);
  if (cast<LLVM::LLVMStructType>(structTy).getBody().size() !=
      resultLayout.getInDimSize(kReg))
    return std::nullopt;

  auto warpId = getLaneAndWarpId(rewriter, loc).second;
  auto smemPtrTy = ptr_ty(ctx, 3);
  SmallVector<Value> srcVals;
  for (int i = 0; i < cvt.getInDimSize(kReg); i += step) {
    Value offset = applyLinearLayout(loc, rewriter, cvt,
                                     {{kReg, b.i32_val(i)},
                                      {kLane, b.i32_val(0)},
                                      {kWarp, warpId}})[0]
                       .second;
    Value vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                          LLVM::GEPNoWrapFlags::inbounds);
    Value baseDiv16 = b.lshr(b.ptrtoint(i64_ty, vecAddr), b.i64_val(4));
    auto builder = mlir::triton::ppu::TIXBuilder();
    auto resArgs = builder.newListOperand(4, "=r");
    auto sbase = builder.newAddrOperand(baseDiv16, "r");
    auto lbo = builder.newOperand(b.i32_val(1), "r");
    auto sbo = builder.newOperand(b.i32_val(swizzledBytes == 128 ? 64 : 32), "r");
    auto mode = builder.newOperand(b.i32_val(swizzledBytes == 128 ? 0 : 1), "r");
    auto ldmatrix = builder.create("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4")
                        ->o("trans", transpose)
                        .o("b16");
    ldmatrix(resArgs, sbase, lbo, sbo, mode);
    Type matTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(4, i32_ty));
    Value res = builder.launch(rewriter, loc, matTy);

    if (dotEnc.getOpIdx() == 1) {
      SmallVector<Value> elemsI32;
      elemsI32.push_back(b.extract_val(i32_ty, res, 0));
      elemsI32.push_back(b.extract_val(i32_ty, res, 2));
      elemsI32.push_back(b.extract_val(i32_ty, res, 1));
      elemsI32.push_back(b.extract_val(i32_ty, res, 3));
      for (int j = 0; j < 4; ++j) {
        Value output = b.bitcast(elemsI32[j], vec_ty(llvmElemTy, elementsPerReg));
        for (int k = 0; k < elementsPerReg; ++k)
          srcVals.push_back(b.extract_element(llvmElemTy, output, b.i32_val(k)));
      }
    } else {
      for (int j = 0; j < 4; ++j) {
        Value output = b.extract_val(i32_ty, res, j);
        output = b.bitcast(output, vec_ty(llvmElemTy, elementsPerReg));
        for (int k = 0; k < elementsPerReg; ++k)
          srcVals.push_back(b.extract_element(llvmElemTy, output, b.i32_val(k)));
      }
    }
  }

  if (!removeBroadcast.isIdentity())
    srcVals = removeBroadcast.apply(srcVals);
  assert(cast<LLVM::LLVMStructType>(structTy).getBody().size() ==
             srcVals.size() &&
         "Unexpected SWZL result size");
  return packLLElements(loc, typeConverter, srcVals, rewriter, structTy);
}

LogicalResult lowerPPULdMatrix(
    Location loc, RankedTensorType tensorTy, MemDescType memDescType,
    bool transpose, Value &src, // Output for ldmatrix
    Value smemBase, Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const ppu::TargetInfo &targetInfo, const LLVMTypeConverter *typeConverter,
    std::pair<size_t, Type> *const llvmOpCount = nullptr,
    bool enableTixSwzl = false, PPUAIUSharedEncodingAttr aiuEnc = nullptr,
    const SharedPhysicalLayout *aiuLayout = nullptr) {
  // Lower load via ppu.ldmatrix
  if (!targetInfo.supportLdMatrix())
    return failure();

  bool isPPU0010 = false;
  bool isPPU0015 = false;
  bool useM8MMA = false;
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (dotEnc) {
    auto mmaEncoding =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(dotEnc.getParent());
    if (mmaEncoding && (mmaEncoding.isPPU0010() || mmaEncoding.isPPU0015())) {
      isPPU0010 = mmaEncoding.isPPU0010();
      isPPU0015 = mmaEncoding.isPPU0015();
      useM8MMA =
          mmaEncoding.getInstrShape().size() >= 2 &&
          mmaEncoding.getInstrShape()[mmaEncoding.getInstrShape().size() - 2] ==
              8;
    }
  }
  if (!isPPU0010 && !isPPU0015)
    return failure();

  auto rank = tensorTy.getRank();
  auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
  auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
  auto bitwidth = tensorTy.getElementTypeBitWidth();
  auto shape = tensorTy.getShape();

  if (aiuEnc && (!aiuLayout || !supportsAIUOrdinaryLdMatrix(
                                    tensorTy, memDescType, aiuEnc, *aiuLayout,
                                    isPPU0015, /*nVecs=*/0)))
    return failure();

  // Limitation: check kWidth * bitwidth
  if (dotEnc.getKWidth() * bitwidth != 32)
    return failure();

  // Limitation: Only support 2d matrices now but we should
  // be able to support 3D minor changes
  if (rank > 2)
    return failure();

  // Limitation: Minimum tile size
  if (bitwidth == 8) {
    if (shape[kOrder] < 32 || shape[nonKOrder] < 16)
      return failure();
  } else {
    if (shape[kOrder] < (8 * 16 / bitwidth) || shape[nonKOrder] < 8)
      return failure();
  }

  assert(llvmOpCount == nullptr && "NYI");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = tensorTy.getContext();
  auto memL = toLinearLayout(memDescType.getShape(), memDescType.getEncoding());
  auto cvt = minimalCvtLayout(memDescType, tensorTy);

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  auto kDim0 = S("dim0");
  auto kDim1 = S("dim1");
  auto smemPtrTy = ptr_ty(ctx, 3);
  // In the transpose case, consecutive elements are not stored contiguously
  // so we cannot split an fp32
  // We could support bitwidth == 8, but it'd be a rather weird layout
  // so we don't do that for now
  // if ((!transpose && bitwidth > 32) || (transpose && bitwidth != 16))

  // support bitwidth == 8 on PPU
  if ((!transpose && bitwidth > 32) ||
      (transpose && bitwidth != 16 && bitwidth != 8))
    return failure();

  bool Opb8bLdmatrix = isPPU0010 && !useM8MMA && bitwidth == 8 &&
                       dotEnc.getOpIdx() == 1 && transpose;
  if (bitwidth == 8 && transpose && !Opb8bLdmatrix)
    return failure();

  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  auto srcVals = SmallVector<Value>{};

  std::optional<ColumnAction> maybePermutation;
  LinearLayout tile;
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);

    // Find if there is a register permutation that allows us to divideLeft
    // We need to pass the map from regs to offsets, as is cvt
    // maybePermutation = regPermForDivideLeft(cvt, tile);
    maybePermutation = regPermForDivide(cvt, tile, /*left=*/true);
    if (!maybePermutation.has_value()) {
      return failure();
    }
    auto permutation = maybePermutation.value();
    // Check if the action indeed allows us to divideLeft
    cvt = permutation.apply(cvt);
  }

  LinearLayout reps;
  if (!transpose) {
    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }
    reps = zerosLike(tile) * maybeQuot.value();
  } else {
    // Division does not quite work here. To define this properly, we would need
    // to define a different multiplication that does:
    // A *' B = [[0, A], [B, 0]] and define leftDivision for it
    // We do it ad-hoc for now, as I beleive there's not much demand for this op
    // outside of this lowering.

    // We implement leftDivision as above for B = identity1D(8, kLane, kOffset)
    // Divisibility in the sense above is the same as regular divisibility
    // You need to see that the tile A is a sublayout of the matrix, and that
    // it has zeros above it and to its right.

    // In particular, offsets lanes 4, 8, 16 map to offsets 1, 2, 4...
    const auto &laneBases = cvt.getBases().find(kLane)->second;
    for (int i = 0; i < 3; ++i) {
      if (laneBases[i + 2][0] != (1 << i))
        return failure();
    }
    // ... and no other basis should depend on 1, 2, 4
    // Note that this gives us the usual alignment condition, but we have
    // translated it to checking that the matrix to the left of A is all zeros
    for (auto dim : cvt.getInDimNames()) {
      const auto &bases = cvt.getBases().find(dim)->second;
      for (auto [i, basis] : llvm::enumerate(bases)) {
        if (dim == kLane && i >= 2)
          continue;
        if (basis[0] & 0b111)
          return failure();
      }
    }

    // Hack: We are not going to use in the rest of the function reps[kLane][2:]
    // so we don't need to zero them out
    reps = cvt;
  }

  // We must have at least 2 register elements to use stmatrix.trans
  if (transpose && reps.getInDimSizeLog2(kReg) < llvm::Log2_32(32 / bitwidth)) {
    return failure();
  }

  // Choose up to 4 packs of 32-bit elements indexed by the next (at most) two
  // bases as the vectorisation factor. We don't consider the basis of the tile
  // for vectorisation so we substract them
  auto vec = std::min<int32_t>(2, reps.getInDimSizeLog2(kReg) -
                                      llvm::Log2_32(32 / bitwidth));

  // Map from kReg, kLane, kWarp to beginning of each tile
  assert(reps.getOutDimSize(kOffset) == cvt.getOutDimSize(kOffset));

  LinearLayout addrLayout = choosePPULdMatrixLayout(dotEnc, shape, transpose,
                                                    bitwidth, Opb8bLdmatrix);
  LinearLayout regToSharedLayout;
  std::optional<int32_t> maxVecElems = 8 * 16 / bitwidth;
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value regBase;
  if (!aiuEnc) {
    LinearLayout sharedLayout =
        triton::gpu::toLinearLayout(shape, memDescType.getEncoding());
    regToSharedLayout = addrLayout.invertAndCompose(sharedLayout);
    regBase = applyLinearLayout(loc, rewriter, regToSharedLayout,
                                {{kReg, b.i32_val(0)},
                                 {kLane, laneId},
                                 {kWarp, warpId},
                                 {kBlock, b.i32_val(0)}})[0]
                  .second;
  }

  // Elements per op
  auto nVecs = 1 << vec;
  bool useLDmatx2 = false;
  if (useM8MMA) {
    const int vecElems =
        std::min(regToSharedLayout.getNumConsecutiveInOut(),
                 maxVecElems.value_or(std::numeric_limits<int>::max()));
    auto vecTy = vec_ty(llvmElemTy, vecElems);
    auto numElems = vecTy.getNumElements();
    auto numElemsI32 = numElems * bitwidth / 32;
    useLDmatx2 = numElemsI32 == 2; // use ppu.ldmatrix.x2 for PPUv1m8
  }
  if ((useM8MMA && useLDmatx2) || (isPPU0010 && !useM8MMA && Opb8bLdmatrix))
    nVecs = 2;
  if (aiuEnc && nVecs != 4)
    return failure();
  if (enableTixSwzl) {
    if (auto result = lowerPPULdMatrixSwzl(
            loc, tensorTy, memDescType, transpose, smemBase, llvmElemTy,
            rewriter, typeConverter, dotEnc, isPPU0015, cvt, nVecs)) {
      src = result.value();
      return success();
    }
  }
  auto elemsPerVec = 32 / bitwidth;
  auto step = nVecs * elemsPerVec;
  for (int i = 0; i < cvt.getInDimSize(kReg); i += step) {
    Value vecAddr;
    if (aiuEnc) {
      auto logicalCoords = applyLinearLayout(
          loc, rewriter, addrLayout,
          {{kReg, b.i32_val(i)},
           {kLane, laneId},
           {kWarp, warpId},
           {kBlock, b.i32_val(0)}});
      Value idxRow;
      Value idxCol;
      for (auto [dim, value] : logicalCoords) {
        if (dim == kDim0)
          idxRow = value;
        else if (dim == kDim1)
          idxCol = value;
      }
      assert(idxRow && idxCol && "Expected rank-2 AIU logical coordinates");
      vecAddr = LLVM::PPU::getPPUAIUV2SwizzledSharedPtr(
          loc, *aiuLayout, llvmElemTy, smemBase, idxRow, idxCol, rewriter);
    } else {
      auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
      Value offset = b.xor_(regBase, b.i32_val(regIdx));
      vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                      LLVM::GEPNoWrapFlags::inbounds);
    }
    Type packedTy = vec_ty(llvmElemTy, 32 / bitwidth);
    Type matTy = nVecs == 1
                     ? i32_ty
                     : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                           ctx, SmallVector<Type>(nVecs, i32_ty)));
    auto ldmatrixOp = triton::ppugpu::PPULoadMatrixOp::create(
        rewriter, loc, matTy, vecAddr,
        /*needTrans=*/transpose, Opb8bLdmatrix);
    if (isPPU0015)
      ldmatrixOp->setAttr("is_0015", rewriter.getUnitAttr());
    Value res = ldmatrixOp.getResult();

    // Extract result into srcVals
    bool needExchange = false;
    if (isPPU0015 && dotEnc && dotEnc.getOpIdx() == 1) {
      needExchange = true;
    }
    if (needExchange) {
      auto resultType = cast<LLVM::LLVMStructType>(res.getType());
      assert(resultType.getBody().size() == 4 && "Unexpected vector size");
      SmallVector<Value> elemsI32;
      elemsI32.push_back(b.extract_val(i32_ty, res, 0));
      elemsI32.push_back(b.extract_val(i32_ty, res, 2));
      elemsI32.push_back(b.extract_val(i32_ty, res, 1));
      elemsI32.push_back(b.extract_val(i32_ty, res, 3));
      for (int j = 0; j < 4; j++) {
        Value output = b.bitcast(elemsI32[j], vec_ty(llvmElemTy, elemsPerVec));
        for (int k = 0; k < elemsPerVec; k++) {
          srcVals.push_back(
              b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    } else {
      for (int j = 0; j < nVecs; j++) {
        Value output = nVecs == 1 ? res : b.extract_val(i32_ty, res, j);
        output = b.bitcast(output, vec_ty(llvmElemTy, elemsPerVec));
        for (int k = 0; k < elemsPerVec; k++) {
          srcVals.push_back(
              b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    }
  }

  // The tiling above yields one value per register slot of the full layout,
  // i.e. the total element count. The converted tensor struct is unique-sized,
  // so drop the broadcast registers. Identity for an injective layout.
  auto removeBroadcast =
      actionRemoveBroadcastedRegs(triton::gpu::toLinearLayout(tensorTy));
  if (!removeBroadcast.isIdentity())
    srcVals = removeBroadcast.apply(srcVals);

  // The type converter owns the result struct's shape. Packing against a
  // locally derived length would survive as an unrealized_conversion_cast and
  // only fail later during LLVM translation.
  auto structTy = typeConverter->convertType(tensorTy);
  assert(cast<LLVM::LLVMStructType>(structTy).getBody().size() ==
             srcVals.size() &&
         "ldmatrix element count must match the converted struct size");
  src = packLLElements(loc, typeConverter, srcVals, rewriter, structTy);
  return success();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const ppu::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1, bool enableTixSwzl = false,
                        bool enableAIULdMatrixOrdinary = false)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo), enableTixSwzl(enableTixSwzl),
        enableAIULdMatrixOrdinary(enableAIULdMatrixOrdinary) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();
    Attribute srcLayout = memDescType.getEncoding();

    // Try to lower to ppu.ldmatrix.swzl for AIU load
    if (auto sharedEnc = dyn_cast<PPUAIUSharedEncodingAttr>(srcLayout)) {
      Attribute dstLayout = dstTy.getEncoding();
      if (isa<DotOperandEncodingAttr>(dstLayout) &&
          isa<PPUMmaEncodingAttr>(
              cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
        auto tryOrdinary = [&]() -> LogicalResult {
          auto aiuPlan = PPUAIUPlan::fromLegacy(memDescType, lookupNumWarps(op));
          if (!aiuPlan) {
            llvm::consumeError(aiuPlan.takeError());
            return failure();
          }
          auto aiuLayout = aiuPlan->toSharedPhysicalLayout(*op.getContext());
          if (!aiuLayout) {
            llvm::consumeError(aiuLayout.takeError());
            return failure();
          }
          for (bool transpose : {false, true}) {
            Value val;
            if (succeeded(lowerPPULdMatrix(
                    op.getLoc(), dstTy, memDescType, transpose, val, smemBase,
                    llvmElemTy, rewriter, targetInfo, getTypeConverter(),
                    nullptr, /*enableTixSwzl=*/false, sharedEnc, &*aiuLayout))) {
              rewriter.replaceOp(op, val);
              return success();
            }
          }
          return failure();
        };
        if (enableAIULdMatrixOrdinary && succeeded(tryOrdinary()))
          return success();
        if (succeeded(lowerAIUSharedToPPUDotOperand(op, adaptor,
                                                    getTypeConverter(), rewriter)))
          return success();
        return tryOrdinary();
      } else {
        return lowerAIUSharedToDistributed(op, adaptor, getTypeConverter(),
                                           rewriter);
      }
    }

    // Try to lower to ppu.ldmatrix
    bool lowered = false;
    Value val;
    for (bool transpose : {false, true}) {
      lowered = lowerPPULdMatrix(op.getLoc(), dstTy, memDescType, transpose,
                                 val, smemBase, llvmElemTy, rewriter,
                                 targetInfo, getTypeConverter(), nullptr,
                                 enableTixSwzl)
                    .succeeded();
      if (lowered) {
        rewriter.replaceOp(op, val);
        return success();
      }
    }

    // Try to lower to ldmatrix
    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout = toLinearLayout(dstTy).removeZeroBasesAlongDim(
        StringAttr::get(op.getContext(), "register"));
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    auto structTy = LLVM::LLVMStructType::getLiteral(
        op.getLoc().getContext(), SmallVector<Type>(values.size(), llvmElemTy));
    auto value =
        packLLElements(op.getLoc(), typeConverter, values, rewriter, structTy);
    rewriter.replaceOp(op, value);
    return success();
  }

  LogicalResult
  lowerAIUSharedToDistributed(triton::gpu::LocalLoadOp op,
                              triton::gpu::LocalLoadOpAdaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemTy = typeConverter->convertType(dstTy.getElementType());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto dstDistributedLayout = dstTy.getEncoding();
    auto srcSharedLayout =
        cast<triton::gpu::PPUAIUSharedEncodingAttr>(srcTy.getEncoding());
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    LDBG("loadSharedToDistributed elemTy "
         << elemTy << " srcElemTy " << srcElemTy << " dstElemTy " << dstElemTy);

    auto inOrd = llvm::to_vector(srcSharedLayout.getOrder());
    auto outOrd = triton::gpu::getOrder(dstTy);
    unsigned outVec =
        inOrd == outOrd ? triton::gpu::getContigPerThread(dstTy)[outOrd[0]] : 1;

    unsigned inVec = 128 / elemTy.getIntOrFloatBitWidth();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    SmallVector<Value> offsetVals = {
        LLVM::PPU::getStrides(smemObj, srcTy, loc, rewriter).size(),
        b.i32_val(0)};

    auto plan = PPUAIUPlan::fromLegacy(srcTy, lookupNumWarps(op));
    if (!plan)
      return op.emitError() << "invalid AIU plan: "
                            << llvm::toString(plan.takeError());
    auto layout = plan->toSharedPhysicalLayout(*op.getContext());
    if (!layout)
      return op.emitError() << "invalid AIU physical layout: "
                            << llvm::toString(layout.takeError());
    DenseMap<unsigned, Value> sharedPtrs;
    sharedPtrs = LLVM::PPU::getAIUSwizzledSharedPtrs(
        loc, targetInfo, outVec, dstTy, srcTy, srcSharedLayout, elemTy, smemObj,
        rewriter, offsetVals, &*layout);

    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr = b.bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
      auto valVec = b.load(wordTy, smemAddr);
      valVec.setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
      for (unsigned v = 0; v < minVec; ++v) {
        Value currVal = b.extract_element(elemTy, valVec, b.i32_val(v));
        outVals[i * minVec + v] = currVal;
      }
    }

    Value result =
        packTensorElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  lowerAIUSharedToPPUDotOperand(triton::gpu::LocalLoadOp op,
                                triton::gpu::LocalLoadOpAdaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto src = op.getSrc();
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mmaEncoding =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(dotEnc.getParent());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;
    if (auto sharedEnc = cast<PPUAIUSharedEncodingAttr>(srcTy.getEncoding())) {
      auto aiuLoad = sharedEnc.getAIUStrategy();
      auto instrShape = mmaEncoding.getInstrShape();
      bool isM8MMA =
          instrShape.size() >= 2 && instrShape[instrShape.size() - 2] == 8;
      if (mmaEncoding.isPPU0010() && !isM8MMA) {
        res = SharedToDotOperandPPUAIUV1::convertLayout(
            dotEnc.getOpIdx(), rewriter, loc, src, dotEnc, smemObj,
            getTypeConverter(), getThreadId(rewriter, loc), aiuLoad);
      } else if (mmaEncoding.isPPU0010() && isM8MMA) {
        res = SharedToDotOperandPPUAIUV1m8::convertLayout(
            dotEnc.getOpIdx(), rewriter, loc, src, dotEnc, smemObj,
            getTypeConverter(), getThreadId(rewriter, loc), aiuLoad);
      } else if (mmaEncoding.isPPU0015()) {
        auto result = SharedToDotOperandPPUAIUV2::convertLayout(
            dotEnc.getOpIdx(), rewriter, loc, src, dotEnc, smemObj,
            getTypeConverter(), getThreadId(rewriter, loc));
        if (failed(result))
          return failure();
        res = *result;
      } else {
        assert(false && "Unsupported mma layout found");
      }
    } else if (auto sharedEnc =
                   cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding())) {
      // TODO old version for compute the address
      assert(false && "Unsupported mma layout found");
    }

    auto elemsI32 = unpackLLElements(loc, res, rewriter);
    // Unpack i32 values to the original type
    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / bitwidth;
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = b.bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
    }

    // The AIU tiling above yields the total element count; the converted
    // dotOperand struct is unique-sized, so drop the broadcast registers.
    // Identity for an injective dotOperand layout.
    auto removeBroadcast =
        actionRemoveBroadcastedRegs(triton::gpu::toLinearLayout(dstTy));
    if (!removeBroadcast.isIdentity())
      elems = removeBroadcast.apply(elems);

    // The type converter owns the result struct's shape. Packing against a
    // locally derived length would survive as an unrealized_conversion_cast and
    // only fail later during LLVM translation.
    auto structTy = typeConverter->convertType(dstTy);
    assert(cast<LLVM::LLVMStructType>(structTy).getBody().size() ==
               elems.size() &&
           "dotOperand element count must match the converted struct size");
    auto ret = packLLElements(loc, typeConverter, elems, rewriter, structTy);
    rewriter.replaceOp(op, ret);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
  bool enableTixSwzl;
  bool enableAIULdMatrixOrdinary;
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const ppu::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout = toLinearLayout(regTy).removeZeroBasesAlongDim(
        StringAttr::get(op.getContext(), "register"));
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }

    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const ppu::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout = toLinearLayout(srcTy).removeZeroBasesAlongDim(
        StringAttr::get(op.getContext(), "register"));
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};
// Map an RMWOp plus element type onto the ppu.atom/ppu.red opcode and type
// suffix. Returns failure for combinations PPU cannot express, so the op stays
// illegal instead of being miscompiled.
static LogicalResult getSharedAtomicOpcode(RMWOp rmwOp, Type valueElemTy,
                                           std::string &opStr,
                                           std::string &tyStr) {
  unsigned bits = valueElemTy.getIntOrFloatBitWidth();
  // The PPU ISA only supports 16-bit shared atomics for floating-point add.
  if (bits == 16 && rmwOp != RMWOp::FADD)
    return failure();
  std::string sBits = std::to_string(bits);
  switch (rmwOp) {
  case RMWOp::AND:
    opStr = "and";
    tyStr = "b" + sBits;
    return success();
  case RMWOp::OR:
    opStr = "or";
    tyStr = "b" + sBits;
    return success();
  case RMWOp::XOR:
    opStr = "xor";
    tyStr = "b" + sBits;
    return success();
  case RMWOp::ADD:
    opStr = "add";
    tyStr = "u" + sBits;
    return success();
  case RMWOp::MAX:
    opStr = "max";
    tyStr = "s" + sBits;
    return success();
  case RMWOp::MIN:
    opStr = "min";
    tyStr = "s" + sBits;
    return success();
  case RMWOp::UMAX:
    opStr = "max";
    tyStr = "u" + sBits;
    return success();
  case RMWOp::UMIN:
    opStr = "min";
    tyStr = "u" + sBits;
    return success();
  case RMWOp::XCHG:
    opStr = "exch";
    tyStr = "b" + sBits;
    return success();
  case RMWOp::FADD:
    opStr = "add";
    if (bits == 16)
      opStr += ".noftz";
    tyStr = (valueElemTy.isBF16() ? "bf" : "f") + sBits;
    return success();
  default:
    return failure();
  }
}

// Emit a single shared-memory atomic RMW. Uses ppu.red when the old value is
// unused and ppu.atom otherwise.
static FailureOr<Value> emitSharedAtomicRMW(ConversionPatternRewriter &rewriter,
                                            Location loc, Type valueElemTy,
                                            Value ptr, Value value, RMWOp rmwOp,
                                            bool returnOld, Value pred) {
  std::string opStr, tyStr;
  if (failed(getSharedAtomicOpcode(rmwOp, valueElemTy, opStr, tyStr)))
    return failure();

  unsigned bits = valueElemTy.getIntOrFloatBitWidth();
  // Register constraint code for the value operand. Mirrors
  // getRegisterSizeCode() in LoadStoreOpToLLVM.cpp, which is file-local there.
  std::string tyId;
  switch (bits) {
  case 16:
    tyId = "h";
    break;
  case 32:
    tyId = "r";
    break;
  case 64:
    tyId = "l";
    break;
  default:
    return failure();
  }

  TIXBuilder tixBuilder;
  // Suffix order matches PPU's own global atomic RMW
  // (ppu.atom.global.<scope>.<sem>.<op>.<ty>, LoadStoreOpToLLVM.cpp) and
  // NVIDIA's PTX builder. The op carries no sem attribute; upstream hardcodes
  // relaxed for shared scatter atomics, and omitting it would leave the
  // ordering unstated rather than relaxed.
  const char *semStr = "relaxed";
  // PTX has no red.exch, so an unused-result exchange still has to use atom.
  // NVIDIA gets the same effect by trying red and falling back on failure.
  if (!returnOld && rmwOp != RMWOp::XCHG) {
    // Shared-memory addresses use "r" (32-bit), unlike global atomics' "l".
    auto *ptrOpr = tixBuilder.newAddrOperand(ptr, "r");
    auto *valOpr = tixBuilder.newOperand(value, tyId);
    auto &red = *tixBuilder.create("ppu.red");
    red.shared().o("cta").o(semStr).o(opStr).o(tyStr);
    red(ptrOpr, valOpr).maybePredicate(pred);
    return tixBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  // The output operand must be created before the inputs: inline asm requires
  // output constraints to precede input constraints, and TIXBuilder emits them
  // in creation order.
  auto *dstOpr = tixBuilder.newOperand("=" + tyId, /*init=*/true);
  auto *ptrOpr = tixBuilder.newAddrOperand(ptr, "r");
  auto *valOpr = tixBuilder.newOperand(value, tyId);
  auto &atom = *tixBuilder.create("ppu.atom");
  atom.shared().o("cta").o(semStr).o(opStr).o(tyStr);
  atom(dstOpr, ptrOpr, valOpr).maybePredicate(pred);
  return tixBuilder.launch(rewriter, loc, valueElemTy);
}

struct LocalAtomicScatterRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp> {
  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const ppu::TargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp>(converter,
                                                                     benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto lowering = prepareLocalAtomicScatterRMW(
        op, adaptor.getDst(), adaptor.getIndices(), adaptor.getValues(),
        op.getMask() ? adaptor.getMask() : Value(), rewriter, targetInfo,
        getTypeConverter());
    if (failed(lowering))
      return failure();
    LocalAtomicScatterRMWInfo &info = *lowering;

    RMWOp rmwOp = op.getAtomicRmwOp();
    bool returnOld = !op.getResult().use_empty();

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.addrs.size());
    for (auto [i, addrAndValue] :
         llvm::enumerate(llvm::zip(info.addrs, info.values))) {
      auto [addr, value] = addrAndValue;
      // A non-null ctaId means a cluster-scoped shared address. NVIDIA resolves
      // it via its private mapDShared; PPU has no equivalent, and its compiler
      // entry rejects num_ctas > 1, so this should be unreachable. Fail rather
      // than drop the ctaId and address the wrong CTA's shared memory.
      if (addr.ctaId)
        return failure();
      Value pred =
          maybeAnd(rewriter, loc, info.threadPred,
                   info.maskValues.empty() ? Value() : info.maskValues[i]);
      auto old = emitSharedAtomicRMW(rewriter, loc, info.llvmElemTy, addr.ptr,
                                     value, rmwOp, returnOld, pred);
      if (failed(old))
        return failure();
      if (returnOld)
        results.push_back(*old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    finalizeAtomicResults(op, rewriter, results, info.llvmElemTy, b,
                          info.threadPred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};
} // namespace

bool mlir::triton::ppu::supportsAIUOrdinaryLdMatrix(
    RankedTensorType tensorTy, MemDescType memDescType,
    PPUAIUSharedEncodingAttr aiuEnc, const SharedPhysicalLayout &layout,
    bool isPPU0015, int nVecs) {
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (!dotEnc)
    return false;
  auto mmaEncoding = dyn_cast<PPUMmaEncodingAttr>(dotEnc.getParent());
  if (!isPPU0015 || !mmaEncoding || !mmaEncoding.isPPU0015() ||
      !aiuEnc.isPPU0015())
    return false;
  auto strategy = aiuEnc.getAIUStrategy();
  if (tensorTy.getRank() != 2 || tensorTy.getElementTypeBitWidth() != 16 ||
      dotEnc.getKWidth() != 2 || aiuEnc.getOrder() != ArrayRef<unsigned>{1, 0} ||
      layout.order != aiuEnc.getOrder() || aiuEnc.getKOffset() != 0 ||
      SharedMemoryObject::isAffineSharedMemoryAccess(memDescType) ||
      strategy.size() != 5 || (nVecs && nVecs != 4))
    return false;
  unsigned cubeC = strategy[0];
  unsigned cubeW = strategy[1];
  unsigned warpCopyC = strategy[2];
  unsigned warpCopyW = strategy[3];
  unsigned swizzledBytes = strategy[4];
  return cubeC && cubeW && warpCopyC && warpCopyW &&
         (swizzledBytes == 64 || swizzledBytes == 128) &&
         layout.swizzledBytes == swizzledBytes &&
         swizzledBytes % (2 * cubeC) == 0;
}

void mlir::triton::ppu::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit, bool enableTixSwzl,
    bool enableAIULdMatrixOrdinary) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 1, enableTixSwzl,
      enableAIULdMatrixOrdinary);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);
}
