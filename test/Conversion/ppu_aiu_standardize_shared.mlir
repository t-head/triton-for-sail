// RUN: triton-opt %s -split-input-file --tritonppu-standardize-aiu-shared | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonppu-standardize-aiu-shared --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false' --convert-triton-gpu-to-llvm-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false enable-aiu-ldmatrix-ordinary=true' | FileCheck %s --check-prefix=ORDINARY
// RUN: triton-opt %s -split-input-file --tritonppu-standardize-aiu-shared --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=true' --convert-triton-gpu-to-llvm-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=true enable-aiu-ldmatrix-ordinary=true' | FileCheck %s --check-prefix=TIX
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false' --convert-triton-gpu-to-llvm-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false enable-aiu-ldmatrix-ordinary=false' | FileCheck %s --check-prefix=OFF

// OFF: ppu.cp.async.aiu
// OFF: ppu.ldmatrix.swzl.sync.bulk.tensor
// OFF-NOT: ppug.ppu_ldmatrix
// OFF-COUNT-3: alignment = 16 : i64
// ORDINARY: ppu.cp.async.aiu
// ORDINARY: ppug.ppu_ldmatrix
// ORDINARY-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor
// TIX: ppu.cp.async.aiu
// TIX: ppu.ldmatrix.swzl.sync.bulk.tensor
// TIX-NOT: ppug.ppu_ldmatrix

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], vecSize = 2}>
#aiu_default = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [64, 16, 1, 4, 128], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @canonical(%P: !tt.ptr<f16>, %A: tensor<64x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i64
    %a = ttg.local_alloc %A : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #aiu_default, #smem>
    %token = ttpg.async_aiu_copy_global_to_local %P[%c0, %c0][%c64, %c64] %a : !tt.ptr<f16> -> !ttg.memdesc<64x64xf16, #aiu_default, #smem>
    %a_dot = ttg.local_load %a : !ttg.memdesc<64x64xf16, #aiu_default, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
// CHECK: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: tt.func public @canonical
// CHECK: alignment = 128 : i32
// CHECK: ppu.aiu_strategy = [64 : i32, 16 : i32, 1 : i32, 4 : i32, 128 : i32]
// CHECK: ppu.aiu_source

// -----

#blocked_tuned = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma_tuned = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], vecSize = 2}>
#aiu_tuned = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [64, 64, 1, 1, 128], order = [1, 0]}>
#smem_tuned = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tuned(%P: !tt.ptr<f16>, %A: tensor<64x64xf16, #blocked_tuned>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i64
    %a = ttg.local_alloc %A : (tensor<64x64xf16, #blocked_tuned>) -> !ttg.memdesc<64x64xf16, #aiu_tuned, #smem_tuned>
    %token = ttpg.async_aiu_copy_global_to_local %P[%c0, %c0][%c64, %c64] %a : !tt.ptr<f16> -> !ttg.memdesc<64x64xf16, #aiu_tuned, #smem_tuned>
    %a_dot = ttg.local_load %a : !ttg.memdesc<64x64xf16, #aiu_tuned, #smem_tuned> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_tuned, kWidth = 2}>>
    tt.return
  }
}
// CHECK: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: tt.func public @tuned
// CHECK: ppu.aiu_strategy = [64 : i32, 64 : i32, 1 : i32, 1 : i32, 128 : i32]
// CHECK: ppu.aiu_source

// -----

#blocked_legacy = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#aiu_legacy = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [16, 16, 1, 1, 64], order = [1, 0]}>
#smem_legacy = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @noncanonical(%A: tensor<16x16xf16, #blocked_legacy>) {
    %a = ttg.local_alloc %A : (tensor<16x16xf16, #blocked_legacy>) -> !ttg.memdesc<16x16xf16, #aiu_legacy, #smem_legacy>
    tt.return
  }
}
// CHECK: #shared = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [16, 16, 1, 1, 64], order = [1, 0]}>
// CHECK-LABEL: tt.func public @noncanonical
// CHECK-NOT: ppu.aiu_strategy
// CHECK-NOT: ppu.aiu_source

// -----

#blocked_stage = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#aiu_stage = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [64, 16, 1, 4, 128], order = [1, 0]}>
#smem_stage = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @staged() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #aiu_stage, #smem_stage, mutable>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @staged
// CHECK: alignment = 128 : i32
// CHECK: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
