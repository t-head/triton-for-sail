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

using namespace mlir;

using ValueTable = std::map<std::array<int, 3>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::PPUAIUSharedEncodingAttr;

namespace SharedToDotOperandPPUAIUV1m8 {

// M8MMA: ldmatrix.m8n8.x2 for operand A (M=8)
std::tuple<Value, Value>
loadX2(ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
       Value start_coord_y, Value start_coord_x, Value cube_h, Value cube_w,
       Value cube_n, Value channel_offset, Type matTy, bool needTrans) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto resTy = cast<LLVM::LLVMStructType>(matTy);
  Type elemTy = cast<LLVM::LLVMStructType>(matTy).getBody()[0];

  if (auto vecElemTy = dyn_cast<VectorType>(elemTy)) {
    Type elemElemTy = vecElemTy.getElementType();
    MLIRContext *ctx = elemElemTy.getContext();
    if (auto intTy = dyn_cast<IntegerType>(elemElemTy)) {
      if (intTy.getWidth() <= 16) {
        elemTy = rewriter.getI32Type();
        resTy =
            LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(2, elemTy));
      }
    }
  }

  mlir::triton::ppu::TIXBuilder builder;
  auto resArgs = builder.newListOperand(2, "=r");
  auto sbase = builder.newAddrOperand(smemBase, "r");
  auto Args = builder.newListOperand();
  Args->listAppend(builder.newOperand(start_coord_y, "r"));
  Args->listAppend(builder.newOperand(start_coord_x, "r"));
  Args->listAppend(builder.newOperand(cube_h, "r"));
  Args->listAppend(builder.newOperand(cube_w, "r"));
  Args->listAppend(builder.newOperand(cube_n, "r"));
  Args->listAppend(builder.newOperand(channel_offset, "r"));

  std::string ldstring;
  if (!needTrans) {
    ldstring = "ppu.ldmatrix.sync.aligned.m8n8.x2.swzl";
  } else {
    ldstring = "ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.trans";
  }

  auto ldmatrix = builder.create(ldstring)->o("shared.b16");

  ldmatrix(resArgs, sbase, Args);

  // The result type is 4xi32, each i32 is composed of 2xf16
  // elements (adjacent two columns in a row) or a single f32 element.
  Value resV4 = builder.launch(rewriter, loc, resTy);
  return {b.extract_val(elemTy, resV4, 0), b.extract_val(elemTy, resV4, 1)};
}

// ldmatrix.m8n8.x4 for operand A when needTrans or warpM > 1
std::tuple<Value, Value, Value, Value>
loadX4(ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
       Value start_coord_y, Value start_coord_x, Value cube_h, Value cube_w,
       Value cube_n, Value channel_offset, Type matTy, bool needTrans) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto resTy = cast<LLVM::LLVMStructType>(matTy);
  Type elemTy = cast<LLVM::LLVMStructType>(matTy).getBody()[0];

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
  auto resArgs = builder.newListOperand(4, "=r");
  auto sbase = builder.newAddrOperand(smemBase, "r");
  auto Args = builder.newListOperand();
  Args->listAppend(builder.newOperand(start_coord_y, "r"));
  Args->listAppend(builder.newOperand(start_coord_x, "r"));
  Args->listAppend(builder.newOperand(cube_h, "r"));
  Args->listAppend(builder.newOperand(cube_w, "r"));
  Args->listAppend(builder.newOperand(cube_n, "r"));
  Args->listAppend(builder.newOperand(channel_offset, "r"));

  std::string ldstring;
  if (!needTrans) {
    ldstring = "ppu.ldmatrix.sync.aligned.m8n8.x4.swzl";
  } else {
    ldstring = "ppu.ldmatrix.sync.aligned.m16n16.x1.swzl.trans";
  }

  auto ldmatrix = builder.create(ldstring)->o("shared.b16");
  ldmatrix(resArgs, sbase, Args);

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

// M8MMA: operand A has M tile of 8, so only 2 values per (b,m,k) tile
Value composeValuesToDotOperandLayoutStructM8(
    const ValueTable &vals, int batch, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int b = 0; b < batch; ++b)
    for (int m = 0; m < n0; ++m)
      for (int k = 0; k < n1; ++k) {
        elems.push_back(vals.at({b, m, 2 * k}));
        elems.push_back(vals.at({b, m, 2 * k + 1}));
      }
  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = packLLElements(loc, typeConverter, elems, rewriter, structTy);
  return result;
}

// Operand B uses the standard layout (N tile = 16)
Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int batch, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int b = 0; b < batch; ++b)
    for (int m = 0; m < n0; ++m)
      for (int k = 0; k < n1; ++k) {
        elems.push_back(vals.at({b, 2 * m, 2 * k}));
        elems.push_back(vals.at({b, 2 * m, 2 * k + 1}));
        elems.push_back(vals.at({b, 2 * m + 1, 2 * k}));
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

// M8MMA operand A: uses m8n8.x2 for non-transposed, m8n8.x4 for transposed
std::function<void(int, int, int)> getLoadMatrixFnM8(
    MemDescType descTy, const SharedMemoryObject &smemObj,
    PPUMmaEncodingAttr mmaLayout, int warpsPerTile, uint32_t kOrder, int kWidth,
    SmallVector<int> instrShape, SmallVector<int> matShape,
    SmallVector<Value> multiDimWarpId, Value lane, ValueTable &vals, bool isA,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    Location loc, Value sliceKOffset, ArrayRef<unsigned> aiuLoad, bool shape_m8) {
  auto shapePerCTA = getShapePerCTA(descTy);
  Type eltTy = descTy.getElementType();
  auto sharedLayout =
      mlir::cast<PPUAIUSharedEncodingAttr>(descTy.getEncoding());
  const int elemBytes = descTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  auto load = [=, &rewriter, &vals](int batch, int a, int b) {
    auto builder = TritonLLVMOpBuilder(loc, rewriter);
    unsigned warpM = mmaLayout.getWarpsPerCTA()[1];
    unsigned warpN = mmaLayout.getWarpsPerCTA()[2];
    Type smemTy = getSharedMemTy(eltTy);
    unsigned shapePerWarpM = 8;
    unsigned shapePerWarpN = 16;
    Value warpIdx = builder.udiv(lane, builder.i32_val(32));
    Value warpIdxM = multiDimWarpId[1];
    Value warpIdxN = multiDimWarpId[2];
    Value warpMOff = builder.mul(warpIdxM, builder.i32_val(shapePerWarpM));
    Value warpNOff = builder.mul(warpIdxN, builder.i32_val(shapePerWarpN));
    Value warpOff = warpMOff;
    if (!isA)
      warpOff = warpNOff;

    unsigned cubeC = aiuLoad[0];
    unsigned cubeW = aiuLoad[1];
    unsigned aiuWarpCopyC = aiuLoad[2];
    unsigned aiuWarpCopyW = aiuLoad[3];

    unsigned cubeElems = cubeC * cubeW;
    unsigned cubeElemsPerCopy = cubeElems * aiuWarpCopyC * aiuWarpCopyW;
    unsigned cubeCElemsPerCopy = cubeC * aiuWarpCopyC;
    unsigned replicaMN = a >> 1;
    unsigned replicaK = b >> 1;
    unsigned replicaMNElements = shapePerWarpM * warpsPerTile;
    if (isA)
      replicaMNElements *= 2;
    unsigned replicaKElements = 16;
    unsigned replicaMNOff = replicaMNElements * replicaMN;
    unsigned replicaKOff = replicaKElements * replicaK;

    Value start_coord_y = builder.i32_val(0);
    Value start_coord_x;
    Value cube_h = builder.i32_val(1);
    Value cube_w = builder.i32_val(cubeW);
    Value cube_n = builder.i32_val(1);
    Value channel_offset;
    Value smemBase;
    Value smemStride =
        mlir::LLVM::PPU::getStrides(smemObj, descTy, loc, rewriter)[order[1]];

    auto needTrans = kOrder != order[0];
    if (!needTrans) {
      // w/x offset of the origin aiu load tile
      Value tileWOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
      // operandA has no aiu instruciton copy split in W/X, only warpM split
      // aiuWarpCopyIdxW is the orginal tile split
      Value aiuWarpCopyIdxW = builder.udiv(tileWOff, cube_w);
      // W/X offset inside of cube
      start_coord_x = builder.urem(tileWOff, cube_w);
      // channel first iterated
      Value shMemOffsetAIUWarpW = builder.mul(
          aiuWarpCopyIdxW, builder.i32_val(cubeElems * aiuWarpCopyC));

      // c/z offset of the origin aiu load tile
      // operandA may have slice split in K
      Value tileCOff = builder.add(sliceKOffset, builder.i32_val(replicaKOff));
      // the instruction copy index for channel
      Value copyIdx =
          builder.udiv(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      // channel elements inside of copy
      Value channelElemsInnerCopy =
          builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      // aiu copy WarpN index inside of an instruction copy
      Value aiuWarpCopyIdxC =
          builder.udiv(channelElemsInnerCopy, builder.i32_val(cubeC));
      // channel offset inside of a cube
      channel_offset = builder.mul(
          builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
          builder.i32_val(elemBytes));
      Value shMemOffsetCopy =
          builder.mul(copyIdx, builder.i32_val(cubeElemsPerCopy));
      Value shMemOffsetAIUWarpC =
          builder.mul(aiuWarpCopyIdxC, builder.i32_val(cubeElems));
      Value shMemOffset =
          builder.add(shMemOffsetCopy,
                      builder.add(shMemOffsetAIUWarpW, shMemOffsetAIUWarpC));
      smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
      smemBase = builder.gep(smemObj.getBase().getType(),
                             smemObj.getBaseElemType(), smemBase, shMemOffset);

    } else {
      // operandB may have slice split in W/X
      Value tileWOff = builder.add(sliceKOffset, builder.i32_val(replicaKOff));
      Value aiuWarpCopyIdxW = builder.udiv(tileWOff, cube_w);
      // Value numCubeWIdx = builder.udiv(x_off, cube_w);
      //  W/X offset inside of cube
      start_coord_x = builder.urem(tileWOff, cube_w);
      // Value cubeOuterWOff = builder.mul(numCubeWIdx, cube_w);
      Value shMemOffsetAIUWarpW = builder.mul(
          aiuWarpCopyIdxW, builder.i32_val(cubeElems * aiuWarpCopyC));
      Value tileCOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
      // Value copyIdx = builder.udiv(tileCOff, builder.i32_val(cubeC));
      Value copyIdx =
          builder.udiv(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      Value channelElemsInnerCopy =
          builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      Value aiuWarpCopyIdxC =
          builder.udiv(channelElemsInnerCopy, builder.i32_val(cubeC));
      // channel_offset = builder.mul(builder.urem(tileCOff,
      // builder.i32_val(cubeC)), builder.i32_val(elemBytes));
      channel_offset = builder.mul(
          builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
          builder.i32_val(elemBytes));
      Value sliceElemsOffset = builder.sub(
          builder.i32_val(0), builder.mul(sliceKOffset, smemStride));
      Value shMemOffsetCopy =
          builder.mul(copyIdx, builder.i32_val(cubeElemsPerCopy));
      Value shMemOffsetAIUWarpC =
          builder.mul(aiuWarpCopyIdxC, builder.i32_val(cubeElems));
      Value shMemOffset =
          builder.add(shMemOffsetCopy,
                      builder.add(shMemOffsetAIUWarpW, shMemOffsetAIUWarpC));
      shMemOffset = builder.add(shMemOffset, sliceElemsOffset);
      smemBase =
          builder.gep(smemObj.getBase().getType(), smemObj.getBaseElemType(),
                      smemObj.getBase(), shMemOffset);
    }

    if (!needTrans) {
      auto matTy = LLVM::LLVMStructType::getLiteral(
          eltTy.getContext(), SmallVector<Type>(2, i32_ty));

      // actually load from shared memory
      auto [ha0, ha1] =
          loadX2(rewriter, loc, smemBase, start_coord_y, start_coord_x, cube_h,
                 cube_w, cube_n, channel_offset, matTy, needTrans);

      vals[{batch, a, b}] = ha0;
      vals[{batch, a, b + 1}] = ha1;

      if(!shape_m8) {
        int xoffset = shapePerWarpM * warpM;
        Value start_coord_x_new =
            builder.add(start_coord_x, builder.i32_val(xoffset));
        // if (xoffset >= cubeW) xoffset = shapePerWarpM * warpM * warpM *
        // aiuWarpCopyC;
        if (xoffset >= cubeW) {
          Value tileWOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
          tileWOff = builder.add(tileWOff, builder.i32_val(xoffset));

          Value aiuWarpCopyIdxW = builder.udiv(tileWOff, cube_w);
          // W/X offset inside of cube
          start_coord_x_new = builder.urem(tileWOff, cube_w);
          // channel first iterated
          Value shMemOffsetAIUWarpW = builder.mul(
              aiuWarpCopyIdxW, builder.i32_val(cubeElems * aiuWarpCopyC));

          // c/z offset of the origin aiu load tile
          // operandA may have slice split in K
          Value tileCOff =
              builder.add(sliceKOffset, builder.i32_val(replicaKOff));
          // the instruction copy index for channel
          Value copyIdx =
              builder.udiv(tileCOff, builder.i32_val(cubeCElemsPerCopy));
          // channel elements inside of copy
          Value channelElemsInnerCopy =
              builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
          // aiu copy WarpN index inside of an instruction copy
          Value aiuWarpCopyIdxC =
              builder.udiv(channelElemsInnerCopy, builder.i32_val(cubeC));
          // channel offset inside of a cube
          channel_offset = builder.mul(
              builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
              builder.i32_val(elemBytes));
          Value shMemOffsetCopy =
              builder.mul(copyIdx, builder.i32_val(cubeElemsPerCopy));
          Value shMemOffsetAIUWarpC =
              builder.mul(aiuWarpCopyIdxC, builder.i32_val(cubeElems));
          Value shMemOffset =
              builder.add(shMemOffsetCopy,
                          builder.add(shMemOffsetAIUWarpW, shMemOffsetAIUWarpC));
          smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
          smemBase =
              builder.gep(smemObj.getBase().getType(), smemObj.getBaseElemType(),
                          smemBase, shMemOffset);
        }

        auto [ha2, ha3] =
            loadX2(rewriter, loc, smemBase, start_coord_y, start_coord_x_new,
                  cube_h, cube_w, cube_n, channel_offset, matTy, needTrans);

        vals[{batch, a + 1, b}] = ha2;
        vals[{batch, a + 1, b + 1}] = ha3;
      }
    } else if (warpM == 1) {
      auto matTy = LLVM::LLVMStructType::getLiteral(
          eltTy.getContext(), SmallVector<Type>(4, i32_ty));
      auto [ha0, ha1, ha2, ha3] =
          loadX4(rewriter, loc, smemBase, start_coord_y, start_coord_x, cube_h,
                 cube_w, cube_n, channel_offset, matTy, needTrans);

      vals[{batch, a, b}] = ha0;
      vals[{batch, a, b + 1}] = ha1;
      vals[{batch, a + 1, b}] = ha2;
      vals[{batch, a + 1, b + 1}] = ha3;
    } else {
      auto matTy = LLVM::LLVMStructType::getLiteral(
          eltTy.getContext(), SmallVector<Type>(4, i32_ty));

      // warp1 and warp0 in the same slice and same replica K and same replicaMN
      // (if needTrans) int the same slice but different channel_offset channel
      // offset cannot to decide load x4 data select x2 data by warpID
      auto [ha00, ha01, ha02, ha03] =
          loadX4(rewriter, loc, smemBase, start_coord_y, start_coord_x, cube_h,
                 cube_w, cube_n, channel_offset, matTy, needTrans);

      Value oneConst = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      Value zeroConst = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value warpIdxMLowBit =
          arith::AndIOp::create(rewriter, loc, warpIdxM, oneConst);
      Value isEven = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, warpIdxMLowBit, zeroConst);
      vals[{batch, a, b}] =
          arith::SelectOp::create(rewriter, loc, isEven, ha00, ha02);
      vals[{batch, a, b + 1}] =
          arith::SelectOp::create(rewriter, loc, isEven, ha01, ha03);

      if(!shape_m8) {
        // channel offset update, different slice
        int coffset = shapePerWarpM * warpM;
        Value tileCOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
        tileCOff = builder.add(tileCOff, builder.i32_val(coffset));
        Value channelElemsInnerCopy =
            builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
        channel_offset = builder.mul(
            builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
            builder.i32_val(elemBytes));

        auto [ha10, ha11, ha12, ha13] =
            loadX4(rewriter, loc, smemBase, start_coord_y, start_coord_x, cube_h,
                   cube_w, cube_n, channel_offset, matTy, needTrans);

        vals[{batch, a + 1, b}] =
            arith::SelectOp::create(rewriter, loc, isEven, ha10, ha12);
        vals[{batch, a + 1, b + 1}] =
            arith::SelectOp::create(rewriter, loc, isEven, ha11, ha13);
      }
    }
  };

  return load;
}

std::function<void(int, int, int)> getLoadMatrixFn(
    MemDescType descTy, const SharedMemoryObject &smemObj,
    PPUMmaEncodingAttr mmaLayout, int warpsPerTile, uint32_t kOrder, int kWidth,
    SmallVector<int> instrShape, SmallVector<int> matShape,
    SmallVector<Value> multiDimWarpId, Value lane, ValueTable &vals, bool isA,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    Location loc, Value sliceKOffset, ArrayRef<unsigned> aiuLoad) {
  auto shapePerCTA = getShapePerCTA(descTy);
  Type eltTy = descTy.getElementType();
  auto sharedLayout =
      mlir::cast<PPUAIUSharedEncodingAttr>(descTy.getEncoding());
  const int elemBytes = descTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  auto load = [=, &rewriter, &vals](int batch, int a, int b) {
    auto builder = TritonLLVMOpBuilder(loc, rewriter);
    unsigned warpM = mmaLayout.getWarpsPerCTA()[1];
    unsigned warpN = mmaLayout.getWarpsPerCTA()[2];
    unsigned shapePerWarpM = 16;
    unsigned shapePerWarpN = 16;
    Value warpIdxM = multiDimWarpId[1];
    Value warpIdxN = multiDimWarpId[2];
    Value warpMOff = builder.mul(warpIdxM, builder.i32_val(shapePerWarpM));
    Value warpNOff = builder.mul(warpIdxN, builder.i32_val(shapePerWarpN));
    Value warpOff = warpMOff;
    if(!isA) warpOff = warpNOff;

    unsigned cubeC = aiuLoad[0];
    unsigned cubeW = aiuLoad[1];
    unsigned aiuWarpCopyC = aiuLoad[2];
    unsigned aiuWarpCopyW = aiuLoad[3];

    unsigned cubeElems = cubeC * cubeW;
    unsigned cubeElemsPerCopy = cubeElems * aiuWarpCopyC * aiuWarpCopyW;
    unsigned cubeCElemsPerCopy = cubeC * aiuWarpCopyC;
    unsigned replicaMN = a >> 1;
    unsigned replicaK = b >> 1;
    unsigned replicaMNElements = shapePerWarpM * warpsPerTile;
    unsigned replicaKElements = 16;
    unsigned replicaMNOff = replicaMNElements * replicaMN;
    unsigned replicaKOff = replicaKElements * replicaK;

    Value start_coord_y = builder.i32_val(0);
    Value start_coord_x;
    Value cube_h = builder.i32_val(1);
    Value cube_w = builder.i32_val(cubeW);
    Value cube_n = builder.i32_val(1);
    Value channel_offset;
    Value smemBase;
    Value smemStride =
        mlir::LLVM::PPU::getStrides(smemObj, descTy, loc, rewriter)[order[1]];

    auto needTrans = kOrder != order[0];
    if (!needTrans) {
      // w/x offset of the origin aiu load tile
      Value tileWOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
      // operandA has no aiu instruciton copy split in W/X, only warpM split
      // aiuWarpCopyIdxW is the orginal tile split
      Value aiuWarpCopyIdxW = builder.udiv(tileWOff, cube_w);
      // W/X offset inside of cube
      start_coord_x = builder.urem(tileWOff, cube_w);
      // channel first iterated
      Value shMemOffsetAIUWarpW = builder.mul(
          aiuWarpCopyIdxW, builder.i32_val(cubeElems * aiuWarpCopyC));

      // c/z offset of the origin aiu load tile
      // operandA may have slice split in K
      Value tileCOff = builder.add(sliceKOffset, builder.i32_val(replicaKOff));
      // the instruction copy index for channel
      Value copyIdx =
          builder.udiv(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      // channel elements inside of copy
      Value channelElemsInnerCopy =
          builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      // aiu copy WarpN index inside of an instruction copy
      Value aiuWarpCopyIdxC =
          builder.udiv(channelElemsInnerCopy, builder.i32_val(cubeC));
      // channel offset insice of a cube
      channel_offset = builder.mul(
          builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
          builder.i32_val(elemBytes));
      Value shMemOffsetCopy =
          builder.mul(copyIdx, builder.i32_val(cubeElemsPerCopy));
      Value shMemOffsetAIUWarpC =
          builder.mul(aiuWarpCopyIdxC, builder.i32_val(cubeElems));
      Value shMemOffset =
          builder.add(shMemOffsetCopy,
                      builder.add(shMemOffsetAIUWarpW, shMemOffsetAIUWarpC));
      smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
      smemBase = builder.gep(smemObj.getBase().getType(),
                             smemObj.getBaseElemType(), smemBase, shMemOffset);
    } else {
      // operandB may have slice split in W/X
      Value tileWOff = builder.add(sliceKOffset, builder.i32_val(replicaKOff));
      Value aiuWarpCopyIdxW = builder.udiv(tileWOff, cube_w);
      // Value numCubeWIdx = builder.udiv(x_off, cube_w);
      //  W/X offset inside of cube
      start_coord_x = builder.urem(tileWOff, cube_w);
      // Value cubeOuterWOff = builder.mul(numCubeWIdx, cube_w);
      Value shMemOffsetAIUWarpW = builder.mul(
          aiuWarpCopyIdxW, builder.i32_val(cubeElems * aiuWarpCopyC));
      Value tileCOff = builder.add(builder.i32_val(replicaMNOff), warpOff);
      // Value copyIdx = builder.udiv(tileCOff, builder.i32_val(cubeC));
      Value copyIdx =
          builder.udiv(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      Value channelElemsInnerCopy =
          builder.urem(tileCOff, builder.i32_val(cubeCElemsPerCopy));
      Value aiuWarpCopyIdxC =
          builder.udiv(channelElemsInnerCopy, builder.i32_val(cubeC));
      // channel_offset = builder.mul(builder.urem(tileCOff,
      // builder.i32_val(cubeC)), builder.i32_val(elemBytes));
      channel_offset = builder.mul(
          builder.urem(channelElemsInnerCopy, builder.i32_val(cubeC)),
          builder.i32_val(elemBytes));
      Value sliceElemsOffset = builder.sub(
          builder.i32_val(0), builder.mul(sliceKOffset, smemStride));
      Value shMemOffsetCopy =
          builder.mul(copyIdx, builder.i32_val(cubeElemsPerCopy));
      Value shMemOffsetAIUWarpC =
          builder.mul(aiuWarpCopyIdxC, builder.i32_val(cubeElems));
      Value shMemOffset =
          builder.add(shMemOffsetCopy,
                      builder.add(shMemOffsetAIUWarpW, shMemOffsetAIUWarpC));
      shMemOffset = builder.add(shMemOffset, sliceElemsOffset);
      smemBase =
          builder.gep(smemObj.getBase().getType(), smemObj.getBaseElemType(),
                      smemObj.getBase(), shMemOffset);
    }

    auto matTy = LLVM::LLVMStructType::getLiteral(eltTy.getContext(),
                                                  SmallVector<Type>(4, i32_ty));

    // actually load from shared memory
    auto [ha0, ha1, ha2, ha3] =
        loadX4(rewriter, loc, smemBase, start_coord_y, start_coord_x, cube_h,
               cube_w, cube_n, channel_offset, matTy, needTrans);

    vals[{batch, a, b}] = ha0;
    vals[{batch, a, b + 1}] = ha1;
    vals[{batch, a + 1, b}] = ha2;
    vals[{batch, a + 1, b + 1}] = ha3;
  };

  return load;
}

Value loadArg(ConversionPatternRewriter &rewriter, Location loc,
              MemDescType descTy, DotOperandEncodingAttr encoding,
              const SharedMemoryObject &smemObj,
              const LLVMTypeConverter *typeConverter, Value thread, bool isA,
              Value sliceKOffset, ArrayRef<unsigned> aiuLoad) {
  auto shapePerCTA = getShapePerCTA(descTy);
  int bitwidth = descTy.getElementTypeBitWidth();
  auto mmaLayout = mlir::cast<PPUMmaEncodingAttr>(encoding.getParent());
  auto builder = TritonLLVMOpBuilder(loc, rewriter);

  ValueTable vals;
  int mmaInstrM = 8, mmaInstrN = 16, mmaInstrK = 4 * 64 / bitwidth;
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
  //   auto rank = shapePerCTA.size();
  Value warpM =
      builder.urem(multiDimWarpId[1], builder.i32_val(shapePerCTA[1] / 8));
  Value warpN =
      builder.urem(multiDimWarpId[2], builder.i32_val(shapePerCTA[2] / 16));
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[1], shapePerCTA[1] / 16);
  else
    warpsPerTile = std::min<int>(warpsPerCTA[2], shapePerCTA[2] / 16);

  bool shape_m8 = (numRep[1] == 1);
  std::function<void(int, int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFnM8(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 2 /*kOrder*/,
        kWidth, {1, mmaInstrM, mmaInstrK} /*instrShape*/,
        {1, matShapeM, matShapeK} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/, sliceKOffset, aiuLoad, shape_m8);
  else
    loadFn = getLoadMatrixFn(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 1 /*kOrder*/,
        kWidth, {1, mmaInstrK, mmaInstrN} /*instrShape*/,
        {1, matShapeK, matShapeN} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/, sliceKOffset, aiuLoad);

  // Perform loading.
  int numRepBatch = numRep[0];
  int numRepOuter = isA ? std::max<int>(numRep[1] / 2, 1) : numRep[2];
  int numRepK = isA ? numRep[2] : numRep[1];
  for (int b = 0; b < numRepBatch; ++b)
    for (int m = 0; m < numRepOuter; ++m)
      for (int k = 0; k < numRepK; ++k)
        loadFn(b, 2 * m, 2 * k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  if (isA)
    return composeValuesToDotOperandLayoutStructM8(
        vals, numRepBatch, numRep[1], numRepK, typeConverter, loc, rewriter);
  else
    return composeValuesToDotOperandLayoutStruct(
        vals, numRepBatch, numRepOuter, numRepK, typeConverter, loc, rewriter);
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> aiuLoad) {
  auto descTy = cast<MemDescType>(tensor.getType());
  auto expandedDescTy = mlir::LLVM::PPU::getExpandedDesc(descTy);
  auto expandedEncoding = cast<DotOperandEncodingAttr>(
      mlir::LLVM::PPU::getExpandedEncoding(encoding));
  auto expandedSmemObj = mlir::LLVM::PPU::getExpandedSharedMemoryObject(
      rewriter, loc, smemObj, descTy.getShape());

  Value sliceKOffset =
      mlir::LLVM::PPU::getSliceKOffset(rewriter, loc, tensor, opIdx);
  if (opIdx == 0) {
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, true, sliceKOffset,
                   aiuLoad);
  } else {
    assert(opIdx == 1);
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, false, sliceKOffset,
                   aiuLoad);
  }
}
} // namespace SharedToDotOperandPPUAIUV1m8
