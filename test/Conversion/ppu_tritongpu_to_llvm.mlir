// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=true' --convert-triton-gpu-to-llvm-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=true' | FileCheck %s --check-prefix=SWZL
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false' --convert-triton-gpu-to-llvm-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=false' | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu='compute-capability=89 enable-tix-ldmatrix-swzl=true' | FileCheck %s --check-prefix=ALLOC

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], vecSize = 2}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_full(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %cst = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<64x1xi32, #blocked>
    %r3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %a0 = tt.splat %A : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %a1 = tt.addptr %a0, %r2 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %a2 = tt.broadcast %a1 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %r5 = tt.broadcast %r4 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %a3 = tt.addptr %a2, %r5 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %a4 = tt.load %a3 : tensor<64x64x!tt.ptr<f16>, #blocked>
    %a5 = ttg.local_alloc %a4 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %b0 = tt.splat %B : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %b1 = tt.addptr %b0, %r2 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %b2 = tt.broadcast %b1 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %b3 = tt.addptr %b2, %r5 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %b4 = tt.load %b3 : tensor<64x64x!tt.ptr<f16>, #blocked>
    %b5 = ttg.local_alloc %b4 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %a6 = ttg.local_load %a5 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %b6 = ttg.local_load %b5 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %dot = tt.dot %a6, %b6, %zero, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
    %c0 = tt.splat %C : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %c1 = tt.addptr %c0, %r2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %c2 = tt.broadcast %c1 : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %c3 = ttg.convert_layout %dot : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    tt.store %c2, %c3 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// ALLOC: ttg.local_alloc {{.*}}allocation.offset = 0
// ALLOC: ttg.local_alloc {{.*}}allocation.offset = 8192
// SWZL: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 128 : i64}
// SWZL-COUNT-16: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16
// SWZL-COUNT-16: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.trans.b16
// SWZL-NOT: ppu.ldmatrix.sync.aligned
// OFF: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// OFF-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor

// -----

#blocked_half = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma_half = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], vecSize = 2}>
#shared_half = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem_half = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_half(%A: tensor<32x32xf16, #blocked_half>, %B: tensor<32x32xf16, #blocked_half>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma_half>
    %a = ttg.local_alloc %A : (tensor<32x32xf16, #blocked_half>) -> !ttg.memdesc<32x32xf16, #shared_half, #smem_half>
    %b = ttg.local_alloc %B : (tensor<32x32xf16, #blocked_half>) -> !ttg.memdesc<32x32xf16, #shared_half, #smem_half>
    %a_dot = ttg.local_load %a : !ttg.memdesc<32x32xf16, #shared_half, #smem_half> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_half, kWidth = 2}>>
    %b_dot = ttg.local_load %b : !ttg.memdesc<32x32xf16, #shared_half, #smem_half> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_half, kWidth = 2}>>
    %dot = tt.dot %a_dot, %b_dot, %zero, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_half, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_half, kWidth = 2}>> -> tensor<32x32xf32, #mma_half>
    tt.return
  }
}

// ALLOC: ttg.local_alloc {{.*}}allocation.offset = 0
// ALLOC: ttg.local_alloc {{.*}}allocation.offset = 2048
// SWZL: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 128 : i64}
// SWZL-COUNT-4: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16
// SWZL-COUNT-4: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.trans.b16
// OFF: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// OFF-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor

// -----

#blocked_other = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma_other = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], vecSize = 2}>
#shared_other = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#smem_other = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_noncanonical(%A: tensor<16x16xf16, #blocked_other>, %B: tensor<16x16xf16, #blocked_other>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma_other>
    %a = ttg.local_alloc %A : (tensor<16x16xf16, #blocked_other>) -> !ttg.memdesc<16x16xf16, #shared_other, #smem_other>
    %b = ttg.local_alloc %B : (tensor<16x16xf16, #blocked_other>) -> !ttg.memdesc<16x16xf16, #shared_other, #smem_other>
    %a_dot = ttg.local_load %a : !ttg.memdesc<16x16xf16, #shared_other, #smem_other> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_other, kWidth = 2}>>
    %b_dot = ttg.local_load %b : !ttg.memdesc<16x16xf16, #shared_other, #smem_other> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_other, kWidth = 2}>>
    %dot = tt.dot %a_dot, %b_dot, %zero, inputPrecision = tf32 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_other, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_other, kWidth = 2}>> -> tensor<16x16xf32, #mma_other>
    tt.return
  }
}

// SWZL: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// SWZL-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor
// OFF: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// OFF-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor

// -----

#blocked_wide = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma_wide = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], vecSize = 2}>
#shared_wide = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem_wide = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_wide_b(%A: tensor<128x64xf16, #blocked_wide>, %B: tensor<64x128xf16, #blocked_wide>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_wide>
    %a = ttg.local_alloc %A : (tensor<128x64xf16, #blocked_wide>) -> !ttg.memdesc<128x64xf16, #shared_wide, #smem_wide>
    %b = ttg.local_alloc %B : (tensor<64x128xf16, #blocked_wide>) -> !ttg.memdesc<64x128xf16, #shared_wide, #smem_wide>
    %a_dot = ttg.local_load %a : !ttg.memdesc<128x64xf16, #shared_wide, #smem_wide> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_wide, kWidth = 2}>>
    %b_dot = ttg.local_load %b : !ttg.memdesc<64x128xf16, #shared_wide, #smem_wide> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_wide, kWidth = 2}>>
    %dot = tt.dot %a_dot, %b_dot, %zero, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_wide, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_wide, kWidth = 2}>> -> tensor<128x128xf32, #mma_wide>
    tt.return
  }
}

// SWZL: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 128 : i64}
// SWZL: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16
// SWZL-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.trans.b16
// OFF: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// OFF-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor

// -----

#blocked_v1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma_v1 = #ttg.ppu_mma<{versionMajor = 1, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], vecSize = 2}>
#shared_v1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem_v1 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_v1(%A: tensor<64x64xf16, #blocked_v1>, %B: tensor<64x64xf16, #blocked_v1>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_v1>
    %a = ttg.local_alloc %A : (tensor<64x64xf16, #blocked_v1>) -> !ttg.memdesc<64x64xf16, #shared_v1, #smem_v1>
    %b = ttg.local_alloc %B : (tensor<64x64xf16, #blocked_v1>) -> !ttg.memdesc<64x64xf16, #shared_v1, #smem_v1>
    %a_dot = ttg.local_load %a : !ttg.memdesc<64x64xf16, #shared_v1, #smem_v1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_v1, kWidth = 2}>>
    %b_dot = ttg.local_load %b : !ttg.memdesc<64x64xf16, #shared_v1, #smem_v1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_v1, kWidth = 2}>>
    %dot = tt.dot %a_dot, %b_dot, %zero, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_v1, kWidth = 2}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_v1, kWidth = 2}>> -> tensor<64x64xf32, #mma_v1>
    tt.return
  }
}

// SWZL: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 128 : i64}
// SWZL-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor
// OFF: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64}
// OFF-NOT: ppu.ldmatrix.swzl.sync.bulk.tensor
