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

#include "TritonPPUGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTLIBDEVICEFUNCTOPPU
#include "TritonPPUGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static constexpr llvm::StringRef kOldPrefix = "__nv_";
static constexpr llvm::StringRef kNewPrefix = "__ppu_";

struct ConvertLibdeviceFuncToPPU
    : public mlir::triton::impl::ConvertLibdeviceFuncToPPUBase<
          ConvertLibdeviceFuncToPPU> {
  using ConvertLibdeviceFuncToPPUBase::ConvertLibdeviceFuncToPPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = mod.getContext();
    llvm::StringMap<StringAttr> cvtMap;

    // The same libdevice function may appear as both __nv_exp2f (via
    // math -> NVVM) and __ppu_exp2f (via tt.extern_elementwise). Renaming the
    // __nv_ one would then redefine an existing symbol, so drop it and repoint
    // its call sites to the __ppu_ symbol already in the module.
    llvm::StringSet<> existingNames;
    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      existingNames.insert(funcOp.getName());
    });

    // update declarations/definitions.
    SmallVector<LLVM::LLVMFuncOp> duplicates;
    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      StringRef name = funcOp.getName();
      if (!name.starts_with(kOldPrefix))
        return;
      std::string newName =
          (kNewPrefix + name.drop_front(kOldPrefix.size())).str();
      cvtMap[name] = StringAttr::get(ctx, newName); // collect functions to convert
      if (existingNames.count(newName)) {
        duplicates.push_back(funcOp); // __ppu_ counterpart already exists
      } else {
        funcOp.setSymName(newName);
        existingNames.insert(newName);
      }
    });

    // update all call sites.
    mod.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCalleeAttr();
      if (!callee)
        return;
      auto it = cvtMap.find(callee.getValue());
      if (it != cvtMap.end()) {
        callOp.setCalleeAttr(FlatSymbolRefAttr::get(ctx, it->second));
      }
    });

    // erase the unused __nv_ duplicates.
    for (LLVM::LLVMFuncOp funcOp : duplicates)
      funcOp.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertLibdeviceFuncToPPUPass() {
  return std::make_unique<ConvertLibdeviceFuncToPPU>();
}
