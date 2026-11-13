// RUN: triton-opt %s -split-input-file -tritonppugpu-optimize-native-fp8-dot-b | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonppugpu-optimize-native-fp8-dot-b -tritongpu-reduce-data-duplication | FileCheck %s --check-prefix=PIPELINE

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @native_fp8_e4m3(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK: #blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1]}>
// CHECK-LABEL: @native_fp8_e4m3
// CHECK: %[[REPACK:.*]] = ttg.convert_layout %arg1 : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #blocked1>
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc %[[REPACK]] : (tensor<32x32xf8E4M3FN, #blocked1>) -> !ttg.memdesc<32x32xf8E4M3FN, #shared, #smem>
// CHECK: %[[LOAD:.*]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<32x32xf8E4M3FN, #shared, #smem> -> tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK-NOT: ttg.convert_layout %arg1 : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #ttg.dot_op
// CHECK: tt.dot %arg0, %[[LOAD]],
// PIPELINE: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1]}>
// PIPELINE-LABEL: @native_fp8_e4m3
// PIPELINE: ttg.local_alloc %{{.*}} : (tensor<32x32xf8E4M3FN, #blocked1>) -> !ttg.memdesc<32x32xf8E4M3FN, #shared, #smem>
// PIPELINE-NOT: ttg.local_alloc
// PIPELINE: tt.return
// PIPELINE-NOT: ttg.convert_layout %arg1 : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #ttg.dot_op

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @native_fp8_e5m2(%a: tensor<32x32xf8E5M2, #dotop_a>, %b: tensor<32x32xf8E5M2, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E5M2, #blocked> -> tensor<32x32xf8E5M2, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E5M2, #dotop_a> * tensor<32x32xf8E5M2, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK: #blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1]}>
// CHECK-LABEL: @native_fp8_e5m2
// CHECK: %[[REPACK2:.*]] = ttg.convert_layout %arg1 : tensor<32x32xf8E5M2, #blocked> -> tensor<32x32xf8E5M2, #blocked1>
// CHECK: %[[ALLOC2:.*]] = ttg.local_alloc %[[REPACK2]] : (tensor<32x32xf8E5M2, #blocked1>) -> !ttg.memdesc<32x32xf8E5M2, #shared, #smem>
// CHECK: %[[LOAD2:.*]] = ttg.local_load %[[ALLOC2]] : !ttg.memdesc<32x32xf8E5M2, #shared, #smem> -> tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: tt.dot %arg0, %[[LOAD2]],
// PIPELINE: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1]}>
// PIPELINE-LABEL: @native_fp8_e5m2
// PIPELINE: ttg.local_alloc %{{.*}} : (tensor<32x32xf8E5M2, #blocked1>) -> !ttg.memdesc<32x32xf8E5M2, #shared, #smem>
// PIPELINE-NOT: ttg.local_alloc
// PIPELINE: tt.return

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 1, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_ppu0010(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_ppu0010
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_dot_a(%a: tensor<32x32xf8E4M3FN, #blocked>, %b: tensor<32x32xf8E4M3FN, #dotop_b>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a_dot = ttg.convert_layout %a : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_a>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a_dot, %b, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_dot_a
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dotop_b2 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_kwidth2(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b2>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b2> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_kwidth2
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_f16(%a: tensor<32x32xf16, #dotop_a>, %b: tensor<32x32xf16, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf16, #dotop_a> * tensor<32x32xf16, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_f16
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked_nk = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [0, 1]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_nk_order(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked_nk>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked_nk> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_nk_order
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_small_k(%a: tensor<32x16xf8E4M3FN, #dotop_a>, %b: tensor<16x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<16x32xf8E4M3FN, #blocked> -> tensor<16x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x16xf8E4M3FN, #dotop_a> * tensor<16x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_small_k
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_small_n(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x8xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x8xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x8xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x8xf8E4M3FN, #blocked> -> tensor<32x8xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x8xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x8xf8E4M3FN, #dotop_b> -> tensor<32x8xf32, #mma>
      scf.yield %d : tensor<32x8xf32, #mma>
    }
    tt.return %out : tensor<32x8xf32, #mma>
  }
}
// CHECK-LABEL: @reject_small_n
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_cvt_inside_loop(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_cvt_inside_loop
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_extra_user(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> (tensor<32x32xf32, #mma>, tensor<32x32xf8E4M3FN, #dotop_b>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out, %b_dot : tensor<32x32xf32, #mma>, tensor<32x32xf8E4M3FN, #dotop_b>
  }
}
// CHECK-LABEL: @reject_extra_user
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_one_shot_dot(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %d = tt.dot %a, %b_dot, %cst : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
    tt.return %d : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_one_shot_dot
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 16], vecSize = 2}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_f16_accumulator(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf16, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf16, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf16, #mma>
      scf.yield %d : tensor<32x32xf16, #mma>
    }
    tt.return %out : tensor<32x32xf16, #mma>
  }
}
// CHECK-LABEL: @reject_f16_accumulator
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.ppu_mma<{versionMajor = 1, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [8, 16], vecSize = 1}>
#dotop_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotop_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_ppu0010_v1m8(%a: tensor<32x32xf8E4M3FN, #dotop_a>, %b: tensor<32x32xf8E4M3FN, #blocked>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x32xf8E4M3FN, #blocked> -> tensor<32x32xf8E4M3FN, #dotop_b>
    %out = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %d = tt.dot %a, %b_dot, %acc : tensor<32x32xf8E4M3FN, #dotop_a> * tensor<32x32xf8E4M3FN, #dotop_b> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}
// CHECK-LABEL: @reject_ppu0010_v1m8
// CHECK: ttg.convert_layout
// CHECK-NOT: ttg.local_alloc
