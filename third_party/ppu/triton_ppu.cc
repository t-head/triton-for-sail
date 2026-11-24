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
#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "PPUGPUToLLVM/Passes.h"
#include "TritonPPUGPUToLLVM/Passes.h"
#include "TritonPPUGPUTransforms/Passes.h"
#include "acblas_instance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "passes.h"
#include "triton/Dialect/TritonGPU/IR/PPUAIUPlan.h"
#include "triton/Dialect/TritonGPU/IR/PPUAIUWriterSolver.h"
#include "llvm/IR/Constants.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;

void init_triton_ppu_passes_ttgpuir(py::module_ &m) {
  using namespace mlir::triton;
  m.def("add_allocate_shared_memory_ppu",
        [](mlir::PassManager &pm, int32_t capability,
           bool enableTixLdMatrixSwzl) {
          pm.addPass(mlir::triton::createAllocateSharedMemoryPPUPass(
              capability, enableTixLdMatrixSwzl));
        },
        py::arg("pm"), py::arg("capability"),
        py::arg("enable_tix_ldmatrix_swzl") = false);
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, int32_t capability,
           bool enableTixLdMatrixSwzl, bool enableAIULdMatrixOrdinary) {
          pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPPUPass(
              capability, enableTixLdMatrixSwzl, enableAIULdMatrixOrdinary));
        },
        py::arg("pm"), py::arg("capability"),
        py::arg("enable_tix_ldmatrix_swzl") = false,
        py::arg("enable_aiu_ldmatrix_ordinary") = false);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", mlir::createTritonPPUGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_convert_libdevice_func_to_ppu",
                     mlir::triton::createConvertLibdeviceFuncToPPUPass);
  ADD_PASS_WRAPPER_0("add_ppu_reorder_instructions",
                     mlir::createTritonPPUGPUReorderInstructions);
}

void init_triton_ppu_passes_ttppugpuir(py::module_ &m) {
  ADD_PASS_WRAPPER_0("add_aiu_lowering", mlir::createTritonPPUAIULoweringPass);
  ADD_PASS_WRAPPER_0("add_standardize_aiu_shared",
                     mlir::createTritonPPUAIUStandardizeSharedPass);
  ADD_PASS_WRAPPER_0("add_ppugpu_to_llvm",
                     mlir::triton::createConvertPPUGPUToLLVM);
}

static void checkMatmulConstraints(const std::string &A_dtype,
                                   const std::string &B_dtype,
                                   const std::string &C_dtype,
                                   const std::vector<int> &A_shape,
                                   const std::vector<int> &B_shape,
                                   const std::vector<int> &C_shape) {
  if (A_dtype != B_dtype || A_dtype != C_dtype) {
    throw std::runtime_error("Data types do not match.");
  }
  if (A_dtype != "torch.float8_e4m3fn" && A_dtype != "torch.float16" &&
      A_dtype != "torch.float32" && A_dtype != "torch.bfloat16") {
    throw std::runtime_error("Unsupported data type.");
  }

  if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
    throw std::runtime_error("Only 2D matrices are supported.");
  }

  int k = A_shape[1];
  if (k != B_shape[1]) {
    throw std::runtime_error(
        "Matrix dimensions do not match. A is [" + std::to_string(A_shape[0]) +
        ", " + std::to_string(A_shape[1]) + "], B is [" +
        std::to_string(B_shape[0]) + ", " + std::to_string(B_shape[1]) +
        "]. Expected A.shape[1] == B.shape[1]. Note "
        "that B needs to be transposed.");
  }

  int m = A_shape[0];
  if (m != C_shape[0]) {
    throw std::runtime_error(
        "Matrix dimensions do not match. A is [" + std::to_string(A_shape[0]) +
        ", " + std::to_string(A_shape[1]) + "], C is [" +
        std::to_string(C_shape[0]) + ", " + std::to_string(C_shape[1]) +
        "]. Expected A.shape[0] == C.shape[0].");
  }

  int n = B_shape[0];
  if (n != C_shape[1]) {
    throw std::runtime_error(
        "Matrix dimensions do not match. B is [" + std::to_string(B_shape[0]) +
        ", " + std::to_string(B_shape[1]) + "], C is [" +
        std::to_string(C_shape[0]) + ", " + std::to_string(C_shape[1]) +
        "]. Expected B.shape[0] == C.shape[1]. Note "
        "that B needs to be transposed.");
  }
}

void init_triton_ppu(py::module_ &m) {
  auto passes = m.def_submodule("passes");
  auto ttgpuir_m = passes.def_submodule("ttgpuir");
  init_triton_ppu_passes_ttgpuir(ttgpuir_m);
  auto ttppugpuir_m = passes.def_submodule("ttppugpuir");
  init_triton_ppu_passes_ttppugpuir(ttppugpuir_m);

  m.def("enumerate_aiu_plans", [](unsigned numWarps, uint64_t xElems,
                                 uint64_t channelElems,
                                 unsigned elementBitWidth,
                                 std::vector<int64_t> rootShape,
                                 std::vector<unsigned> order) {
    auto plans = mlir::triton::gpu::PPUAIUPlan::enumerateProvenCandidates(
        numWarps, xElems, channelElems, elementBitWidth, rootShape, order);
    if (!plans)
      throw std::runtime_error(llvm::toString(plans.takeError()));
    std::vector<std::vector<unsigned>> result;
    for (const auto &plan : *plans) {
      auto strategy = plan.toLegacyStrategy();
      result.emplace_back(strategy.begin(), strategy.end());
    }
    return result;
  });

  m.def("enumerate_layouts", [](unsigned elementBitWidth,
                               std::vector<int64_t> rootShape,
                               std::vector<unsigned> order,
                               unsigned maxVectorBits, unsigned maxPhase) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
    mlir::triton::gpu::PPULayoutConstraints constraints;
    constraints.elementBitWidth = elementBitWidth;
    constraints.rootShape.assign(rootShape.begin(), rootShape.end());
    constraints.allowedOrders.emplace_back(order.begin(), order.end());
    constraints.maxVectorBits = maxVectorBits;
    constraints.maxPhase = maxPhase;
    auto layouts = mlir::triton::gpu::PPULayoutSpec::enumerateCanonicalCandidates(
        context, constraints);
    if (!layouts)
      throw std::runtime_error(llvm::toString(layouts.takeError()));
    std::vector<std::vector<unsigned>> result;
    for (const auto &layout : *layouts) {
      const auto &parameters = *layout.getCanonicalParameters();
      result.push_back({parameters.vec, parameters.perPhase,
                        parameters.maxPhase, order[0], order[1]});
    }
    return result;
  }, py::arg("element_bit_width"), py::arg("root_shape"), py::arg("order"),
     py::arg("max_vector_bits") = 128, py::arg("max_phase") = 8);

  m.def("solve_aiu_writers", [](unsigned numWarps, unsigned elementBitWidth,
                               std::vector<int64_t> rootShape,
                               std::vector<unsigned> order,
                               std::vector<unsigned> layoutParameters) {
    if (order.size() != 2 || layoutParameters.size() != 5 ||
        layoutParameters[3] != order[0] || layoutParameters[4] != order[1])
      throw std::runtime_error("layout parameters must be vec/perPhase/maxPhase/order");
    mlir::MLIRContext context;
    context.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
    auto layout = mlir::triton::gpu::PPULayoutSpec::createCanonical(
        context, elementBitWidth, rootShape, order, layoutParameters[0],
        layoutParameters[1], layoutParameters[2]);
    if (!layout)
      throw std::runtime_error(llvm::toString(layout.takeError()));
    auto solution = mlir::triton::gpu::PPUAIUWriterSolver::solve(*layout, numWarps);
    if (!solution)
      throw std::runtime_error(llvm::toString(solution.takeError()));
    std::vector<std::vector<unsigned>> result;
    for (const auto &writer : solution->writers) {
      auto strategy = writer.toLegacyStrategy();
      result.emplace_back(strategy.begin(), strategy.end());
    }
    return result;
  });

  m.def("legacy_aiu_default", [](unsigned numWarps, uint64_t xElems,
                                uint64_t channelElems,
                                unsigned elementBitWidth,
                                std::vector<int64_t> rootShape,
                                std::vector<unsigned> order) {
    auto plan = mlir::triton::gpu::PPUAIUPlan::createLegacyDefault(
        numWarps, xElems, channelElems, elementBitWidth, rootShape, order);
    if (!plan)
      throw std::runtime_error(llvm::toString(plan.takeError()));
    auto strategy = plan->toLegacyStrategy();
    return std::vector<unsigned>(strategy.begin(), strategy.end());
  });

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::ppu_gpu::TritonPPUGPUDialect,
                    mlir::triton::ppugpu::PPUGPUDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("set_reflect_ftz", [](llvm::Module *mod) {
    // this will enable fast math path in libdevice
    // for example, when enable reflect-ftz, ppu.sqrt.approx.f32 will change to
    // ppu.sqrt.approx.ftz.f32
    using namespace llvm;
    auto &ctx = mod->getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
    auto *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
    auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
    auto *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
    mod->addModuleFlag(reflect);
  });

  // Sets the smemsize property on the given function.
  m.def("set_smemsize", [](llvm::Function *fn, int sharedsize) {
    auto op = llvm::MDNode::get(
        fn->getContext(),
        {
            llvm::ValueAsMetadata::get(fn),
            llvm::MDString::get(fn->getContext(), "smemsize"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(fn->getContext()), sharedsize)),
        });
    fn->getParent()
        ->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(op);
  });

  // Sets the reqntid property on the given function.
  m.def("set_reqntid", [](llvm::Function *fn) {
    if (fn->hasFnAttribute("nvvm.reqntid")) {
      llvm::Attribute attr = fn->getFnAttribute("nvvm.reqntid");
      llvm::StringRef valStr = attr.getValueAsString();

      unsigned reqntidVal = 0;
      valStr.getAsInteger(10, reqntidVal);

      auto op = llvm::MDNode::get(
          fn->getContext(),
          {
              llvm::ValueAsMetadata::get(fn),
              llvm::MDString::get(fn->getContext(), "reqntidx"),
              llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(fn->getContext()), reqntidVal)),
          });
      fn->getParent()
          ->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(op);
    }
  });

  // Sets the attn forward property on the given function.
  m.def("set_attn_fwd", [](llvm::Function *fn) {
    fn->setMetadata("ppu.triton.fwd", llvm::MDNode::get(fn->getContext(), {}));
  });

  // Sets the attn backward property on the given function.
  m.def("set_attn_bwd", [](llvm::Function *fn) {
    fn->setMetadata("ppu.triton.bwd", llvm::MDNode::get(fn->getContext(), {}));
  });

  m.def("attach_datalayout", [](llvm::Module &module) {
    const std::string dataLayout =
        "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:"
        "64-f16:16-f32:32-v16:16-v32:32-n16:32:64";
    module.setDataLayout(dataLayout);
  });

  // acblas
  auto acblas = m.def_submodule("acblas");

  py::class_<AcblasLtInstance>(acblas, "AcblasLt")
      .def(py::new_([](py::object workspace) {
        auto wrk_ptr = py::cast<uint64_t>(workspace.attr("data_ptr")());
        auto wrk_size = py::cast<size_t>(workspace.attr("numel")()) *
                        py::cast<size_t>(workspace.attr("element_size")());
        return new AcblasLtInstance(wrk_ptr, wrk_size);
      }))
      .def("matmul",
           [](AcblasLtInstance &self, py::object &A, py::object &B,
              py::object &C) {
             auto A_ptr = py::cast<uint64_t>(A.attr("data_ptr")());
             auto B_ptr = py::cast<uint64_t>(B.attr("data_ptr")());
             auto C_ptr = py::cast<uint64_t>(C.attr("data_ptr")());

             auto A_shape = py::cast<std::vector<int>>(A.attr("shape"));
             auto B_shape = py::cast<std::vector<int>>(B.attr("shape"));
             auto C_shape = py::cast<std::vector<int>>(C.attr("shape"));

             auto A_dtype =
                 py::cast<std::string>(A.attr("dtype").attr("__str__")());
             auto B_dtype =
                 py::cast<std::string>(B.attr("dtype").attr("__str__")());
             auto C_dtype =
                 py::cast<std::string>(C.attr("dtype").attr("__str__")());

             checkMatmulConstraints(A_dtype, B_dtype, C_dtype, A_shape, B_shape,
                                    C_shape);

             std::string dtype_str =
                 A_dtype.substr(A_dtype.find_last_of('.') + 1);
             hggcDataType_t dtype;
             if (dtype_str == "float8_e4m3fn") {
               dtype = HGGC_R_8F_E4M3;
             } else if (dtype_str == "float16") {
               dtype = HGGC_R_16F;
             } else if (dtype_str == "float32") {
               // Use FP32 inputs with TF32 compute in acblasLt (set in compute
               // type)
               dtype = HGGC_R_32F;
             } else if (dtype_str == "bfloat16") {
               dtype = HGGC_R_16BF;
             } else {
               throw std::runtime_error(
                   "Unsupported dtype for acblasLt.matmul: " + dtype_str);
             }

             self.matmul(A_shape[0], B_shape[0], A_shape[1], A_ptr, B_ptr,
                         C_ptr, dtype);
           })
      .def("gemm", [](AcblasLtInstance &self, py::object &A, py::object &B,
                      py::object &C, py::object &D, float alpha, float beta) {
        auto A_ptr = py::cast<uint64_t>(A.attr("data_ptr")());
        auto B_ptr = py::cast<uint64_t>(B.attr("data_ptr")());
        auto C_ptr = py::cast<uint64_t>(C.attr("data_ptr")());
        auto D_ptr = py::cast<uint64_t>(D.attr("data_ptr")());

        auto A_shape = py::cast<std::vector<int>>(A.attr("shape"));
        auto B_shape = py::cast<std::vector<int>>(B.attr("shape"));
        auto C_shape = py::cast<std::vector<int>>(C.attr("shape"));
        auto D_shape = py::cast<std::vector<int>>(D.attr("shape"));

        auto A_dtype = py::cast<std::string>(A.attr("dtype").attr("__str__")());
        auto B_dtype = py::cast<std::string>(B.attr("dtype").attr("__str__")());
        auto C_dtype = py::cast<std::string>(C.attr("dtype").attr("__str__")());
        auto D_dtype = py::cast<std::string>(D.attr("dtype").attr("__str__")());

        checkMatmulConstraints(A_dtype, B_dtype, D_dtype, A_shape, B_shape,
                               D_shape);
        if (C_dtype != "torch.float16") {
          throw std::runtime_error("C dtype must be float16, got " + C_dtype);
        }
        if (C_shape != D_shape) {
          throw std::runtime_error("C and D shapes must match");
        }

        std::string dtype_str = A_dtype.substr(A_dtype.find_last_of('.') + 1);
        hggcDataType_t dtype;
        if (dtype_str == "float8_e4m3fn") {
          dtype = HGGC_R_8F_E4M3;
        } else if (dtype_str == "float16") {
          dtype = HGGC_R_16F;
        } else if (dtype_str == "float32") {
          dtype = HGGC_R_32F;
        } else if (dtype_str == "bfloat16") {
          dtype = HGGC_R_16BF;
        } else {
          throw std::runtime_error("Unsupported dtype for acblasLt.gemm: " +
                                   dtype_str);
        }

        self.gemm(A_shape[0], B_shape[0], A_shape[1], A_ptr, B_ptr, C_ptr,
                  D_ptr, dtype, alpha, beta);
      });

  m.def("has_extern_deps", [](llvm::Module *dstMod) -> bool {
    // `global_smem` is special cased in Triton, so we ignore it here.
    for (const auto &g : dstMod->globals()) {
      if (g.hasExternalLinkage() && g.getName() != "global_smem") {
        return true;
      }
    }
    for (const auto &f : *dstMod) {
      if (f.hasExternalLinkage() && !f.hasExactDefinition() &&
          !f.isIntrinsic()) {
        return true;
      }
    }
    return false;
  });
}
