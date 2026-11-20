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

#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "TritonPPUGPUToLLVM/TIXAsmFormat.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"

using namespace mlir;

using ValueTable = std::map<std::array<int, 3>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::PPUAIUSharedEncodingAttr;
using ::mlir::triton::gpu::PPUAIUPlan;
using ::mlir::triton::gpu::SharedPhysicalLayout;

namespace SharedToDotOperandPPUAIUV2 {

std::tuple<Value, Value, Value, Value>
loadX4(ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
       Value lbo, Value sbo, unsigned swizzledBytes, unsigned elemBits,
       Type matTy, bool needTrans) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // The struct should have exactly the same element types.
  auto resTy = cast<LLVM::LLVMStructType>(matTy);
  Type elemTy = cast<LLVM::LLVMStructType>(matTy).getBody()[0];

  // For some reasons, LLVM backend inserts unnecessary (?) integer
  // instructions to pack & unpack b.sub-word integers. A workaround is to
  // store the results of ldmatrix in i32
  if (auto vecElemTy = dyn_cast<VectorType>(elemTy)) {
    Type elemElemTy = vecElemTy.getElementType();
    MLIRContext *ctx = elemElemTy.getContext();
    if (auto intTy = dyn_cast<IntegerType>(elemElemTy)) {
      if (intTy.getWidth() <= 16) {
        elemTy = rewriter.getI32Type();
        resTy =
            LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, elemTy));
      }
    }
  }

  mlir::triton::ppu::TIXBuilder builder;
  // ldmatrix.m8n8.x4 returns 4x2xfp16(that is 4xb32) elements for a thread.
  auto resArgs = builder.newListOperand(4, "=r");

  smemBase = b.ptrtoint(i64_ty, smemBase);
  Value smemBaseDiv16 = b.lshr(smemBase, b.i64_val(4));
  auto sbase = builder.newAddrOperand(smemBaseDiv16, "r");
  // the following args are already divided by 16
  unsigned swzlMode = 0; // swzlMode 0: 128B
  if (swizzledBytes == 64)
    swzlMode = 1; // swzlMode 1: 64B
  auto lboOpnd = builder.newOperand(lbo, "r");
  auto sboOpnd = builder.newOperand(sbo, "r");
  auto swzlModeOpnd = builder.newOperand(b.i32_val(swzlMode), "r");

  auto ldmatrix = builder.create("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4")
                      ->o("trans", needTrans /*predicate*/)
                      .o("b16");

  // will be lowering to tsm.ld.swzl.b32x4.trans_b32
  if (elemBits == 32 && needTrans)
    ldmatrix =
        *builder.create("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.trans.b32");

  ldmatrix(resArgs, sbase, lboOpnd, sboOpnd, swzlModeOpnd);

  // The result type is 4xi32, each i32 is composed of 2xf16
  // elements (adjacent two columns in a row) or a single f32 element.
  Value resV4 = builder.launch(rewriter, loc, resTy);
  return {b.extract_val(elemTy, resV4, 0), b.extract_val(elemTy, resV4, 1),
          b.extract_val(elemTy, resV4, 2), b.extract_val(elemTy, resV4, 3)};
}

Type getSharedMemTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return type::f16Ty(ctx);
  else if (argType.isBF16())
    return type::bf16Ty(ctx);
  else if (argType.isF32())
    return type::f32Ty(ctx);
  else if (argType.getIntOrFloatBitWidth() == 8)
    return type::i8Ty(ctx);
  else
    llvm::report_fatal_error("mma16816 data type not supported");
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int batch, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int b = 0; b < batch; ++b)
    for (int m = 0; m < n0; ++m)
      for (int k = 0; k < n1; ++k) {
        elems.push_back(vals.at({b, 2 * m, 2 * k}));
        elems.push_back(vals.at({b, 2 * m + 1, 2 * k}));
        elems.push_back(vals.at({b, 2 * m, 2 * k + 1}));
        elems.push_back(vals.at({b, 2 * m + 1, 2 * k + 1}));
      }
  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = packLLElements(loc, typeConverter, elems, rewriter, structTy);
  return result;
}

std::function<void(int, int, int)> getLoadMatrixFn(
    MemDescType descTy, const SharedMemoryObject &smemObj,
    PPUMmaEncodingAttr mmaLayout, int warpsPerTile, uint32_t kOrder, int kWidth,
    SmallVector<int> instrShape, SmallVector<int> matShape,
    SmallVector<Value> multiDimWarpId, Value lane, ValueTable &vals, bool isA,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    Location loc) {
  auto shapePerCTA = getShapePerCTA(descTy);
  Type eltTy = descTy.getElementType();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout =
      mlir::cast<PPUAIUSharedEncodingAttr>(descTy.getEncoding());
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto expectedPlan = PPUAIUPlan::fromLegacy(descTy, product(warpsPerCTA));
  if (!expectedPlan)
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(expectedPlan.takeError())));
  auto expectedLayout = expectedPlan->toSharedPhysicalLayout(*eltTy.getContext());
  if (!expectedLayout)
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(expectedLayout.takeError())));
  auto layout = std::make_shared<SharedPhysicalLayout>(*expectedLayout);
  const int elemBytes = descTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  int nPerWarp =
      std::max<int>(shapePerCTA[2] / mmaLayout.getWarpsPerCTA()[2], 16);

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals](int batch, int a, int b) {
    auto builder = TritonLLVMOpBuilder(loc, rewriter);
    unsigned warpM = mmaLayout.getWarpsPerCTA()[1];
    unsigned warpN = mmaLayout.getWarpsPerCTA()[2];
    Type smemTy = getSharedMemTy(eltTy);
    unsigned shapePerWarpM = 16;
    unsigned shapePerWarpN = 16;
    Value warpIdx = builder.udiv(lane, builder.i32_val(32));
    Value warpIdxM = multiDimWarpId[1];
    Value warpIdxN = multiDimWarpId[2];
    Value warpMOff = builder.mul(warpIdxM, builder.i32_val(shapePerWarpM));
    Value warpNOff = builder.mul(warpIdxN, builder.i32_val(shapePerWarpN));
    Value warpOff = warpMOff;
    if (!isA)
      warpOff = warpNOff;

    unsigned swizzledBytes = layout->swizzledBytes;
    unsigned replicaMN = a >> 1;
    unsigned replicaK = b >> 1;
    unsigned replicaMNElements = shapePerWarpM * warpsPerTile;
    unsigned replicaKElements = 32 / elemBytes;
    unsigned replicaMNOff = replicaMNElements * replicaMN;
    unsigned replicaKOff = replicaKElements * replicaK;

    auto smemObjOffsets = smemObj.getOffsets();
    Value smemBase = smemObj.getBase();

    auto needTrans = kOrder != order[0];
    if (elemBytes == 1) {
      assert(!needTrans && "needTrans should be false for fp8 aiu");
    }
    {



      // width*channel elems offset inside of a cube

      // channel elems offset inside of a cube
    }

    Value tileWOff = needTrans
                         ? builder.i32_val(replicaKOff)
                         : builder.add(builder.i32_val(replicaMNOff), warpOff);
    Value tileCOff = needTrans
                         ? builder.add(builder.i32_val(replicaMNOff), warpOff)
                         : builder.i32_val(replicaKOff);
    unsigned nonKOrder = 3 - kOrder;
    Value globalW = builder.add(tileWOff, smemObjOffsets[nonKOrder]);
    Value globalC = builder.add(tileCOff, smemObjOffsets[kOrder]);
    bool rowMajor = layout->order[1] == 0;
    auto dim0 = StringAttr::get(eltTy.getContext(), "dim0");
    auto dim1 = StringAttr::get(eltTy.getContext(), "dim1");
    SmallVector<std::pair<StringAttr, Value>> indices;
    for (auto dim : layout->logicalToBulkBase.getInDimNames()) {
      if (dim == dim0)
        indices.push_back({dim, rowMajor ? globalW : globalC});
      else if (dim == dim1)
        indices.push_back({dim, rowMajor ? globalC : globalW});
    }
    Value offset =
        applyLinearLayout(loc, rewriter, layout->logicalToBulkBase, indices)[0]
            .second;
    smemBase = builder.gep(smemObj.getBase().getType(),
                           smemObj.getBaseElemType(), smemBase, offset);

    auto matTy = LLVM::LLVMStructType::getLiteral(eltTy.getContext(),
                                                  SmallVector<Type>(4, i32_ty));

    Value lbo;
    Value sbo;
    if (!needTrans && !isA) {
      lbo = builder.i32_val(swizzledBytes / 2);
      sbo = builder.i32_val(1);
    } else {
      lbo = builder.i32_val(1);
      sbo = builder.i32_val(swizzledBytes / 2);
    }
    if (elemBytes == 4 && needTrans) {
      if (swizzledBytes == 128) {
        lbo = builder.i32_val(512);
        sbo = builder.i32_val(32);
      }
    }

    // actually load from shared memory
    auto [ha0, ha1, ha2, ha3] =
        loadX4(rewriter, loc, smemBase, lbo, sbo, swizzledBytes, elemBytes * 8,
               matTy, needTrans);

    if (needTrans || !isA) {
      vals[{batch, a, b}] = ha0;
      vals[{batch, a, b + 1}] = ha1;
      vals[{batch, a + 1, b}] = ha2;
      vals[{batch, a + 1, b + 1}] = ha3;
    } else {
      vals[{batch, a, b}] = ha0;
      vals[{batch, a + 1, b}] = ha1;
      vals[{batch, a, b + 1}] = ha2;
      vals[{batch, a + 1, b + 1}] = ha3;
    }

    if (elemBytes == 4 && needTrans) {
      vals[{batch, a, b}] = ha0;
      vals[{batch, a, b + 1}] = ha2;
      vals[{batch, a + 1, b}] = ha1;
      vals[{batch, a + 1, b + 1}] = ha3;
    }
  };

  return load;
}

Value loadArg(ConversionPatternRewriter &rewriter, Location loc,
              MemDescType descTy, DotOperandEncodingAttr encoding,
              const SharedMemoryObject &smemObj,
              const LLVMTypeConverter *typeConverter, Value thread, bool isA) {
  auto shapePerCTA = getShapePerCTA(descTy);
  int bitwidth = descTy.getElementTypeBitWidth();
  auto mmaLayout = mlir::cast<PPUMmaEncodingAttr>(encoding.getParent());
  auto builder = TritonLLVMOpBuilder(loc, rewriter);

  ValueTable vals;
  int mmaInstrM = 16, mmaInstrN = 16, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / bitwidth;

  int kWidth = encoding.getKWidth();
  auto numRep = mmaLayout.getRepForOperand(shapePerCTA, bitwidth, kWidth,
                                           encoding.getOpIdx());

  auto rank = shapePerCTA.size();
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto order = mlir::triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
  Value warp = builder.udiv(thread, builder.i32_val(32));
  warp = LLVM::PPU::toUniformB32(loc, rewriter, warp);
  Value lane = builder.urem(thread, builder.i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA, order);
  Value warpB =
      builder.urem(multiDimWarpId[0], builder.i32_val(shapePerCTA[0]));
  int warpsPerTile;
  Value warpM =
      builder.urem(multiDimWarpId[1], builder.i32_val(shapePerCTA[1] / 16));
  Value warpN =
      builder.urem(multiDimWarpId[2], builder.i32_val(shapePerCTA[2] / 16));
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[1], shapePerCTA[1] / 16);
  else
    warpsPerTile = std::min<int>(warpsPerCTA[2], shapePerCTA[2] / 16);
  std::function<void(int, int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFn(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 2 /*kOrder*/,
        kWidth, {1, mmaInstrM, mmaInstrK} /*instrShape*/,
        {1, matShapeM, matShapeK} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);
  else
    loadFn = getLoadMatrixFn(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 1 /*kOrder*/,
        kWidth, {1, mmaInstrK, mmaInstrN} /*instrShape*/,
        {1, matShapeK, matShapeN} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);

  // Perform loading.
  int numRepBatch = numRep[0];
  int numRepOuter = isA ? numRep[1] : std::max<int>(numRep[2], 1);
  int numRepK = isA ? numRep[2] : numRep[1];
  for (int b = 0; b < numRepBatch; ++b)
    for (int m = 0; m < numRepOuter; ++m)
      for (int k = 0; k < numRepK; ++k)
        loadFn(b, 2 * m, 2 * k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  return composeValuesToDotOperandLayoutStruct(
      vals, numRepBatch, numRepOuter, numRepK, typeConverter, loc, rewriter);
}

bool supportsBulk(MemDescType descTy, DotOperandEncodingAttr encoding) {
  auto sharedEncoding = dyn_cast<PPUAIUSharedEncodingAttr>(descTy.getEncoding());
  auto mmaEncoding = dyn_cast<PPUMmaEncodingAttr>(encoding.getParent());
  if (!sharedEncoding || !sharedEncoding.isPPU0015() || !mmaEncoding ||
      !mmaEncoding.isPPU0015())
    return false;
  auto expandedOrder =
      cast<PPUAIUSharedEncodingAttr>(
          mlir::LLVM::PPU::getExpandedEncoding(sharedEncoding))
          .getOrder();
  unsigned kOrder = encoding.getOpIdx() == 0 ? 2 : 1;
  bool needTrans = kOrder != expandedOrder[0];
  unsigned bitWidth = descTy.getElementTypeBitWidth();
  if ((bitWidth == 8 && needTrans) ||
      (bitWidth != 8 && bitWidth != 16 && bitWidth != 32))
    return false;
  if ((bitWidth == 8 && encoding.getKWidth() != 4) ||
      (bitWidth == 16 && encoding.getKWidth() != 2) ||
      (bitWidth == 32 && encoding.getKWidth() != 1))
    return false;
  auto warpsPerCTA = mmaEncoding.getWarpsPerCTA();
  auto plan = PPUAIUPlan::fromLegacy(
      mlir::LLVM::PPU::getExpandedDesc(descTy), product(warpsPerCTA));
  if (!plan) {
    llvm::consumeError(plan.takeError());
    return false;
  }
  auto layout = plan->toSharedPhysicalLayout(*descTy.getContext());
  if (!layout) {
    llvm::consumeError(layout.takeError());
    return false;
  }
  return true;
}

FailureOr<Value> convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                               Location loc, Value tensor,
                               DotOperandEncodingAttr encoding,
                               const SharedMemoryObject &smemObj,
                               const LLVMTypeConverter *typeConverter,
                               Value thread) {
  auto descTy = cast<MemDescType>(tensor.getType());
  if (!supportsBulk(descTy, encoding))
    return failure();
  auto expandedDescTy = mlir::LLVM::PPU::getExpandedDesc(descTy);
  auto expandedEncoding = cast<DotOperandEncodingAttr>(
      mlir::LLVM::PPU::getExpandedEncoding(encoding));
  auto expandedSmemObj = mlir::LLVM::PPU::getExpandedSharedMemoryObject(
      rewriter, loc, smemObj, descTy.getShape());

  if (opIdx == 0)
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, true);
  else {
    assert(opIdx == 1);
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, false);
  }
}
} // namespace SharedToDotOperandPPUAIUV2
