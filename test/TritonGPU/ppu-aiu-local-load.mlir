// RUN: triton-opt %s -split-input-file -allocate-shared-memory-ppu -convert-triton-gpu-to-llvm-ppu | FileCheck %s

// Which shared->register path an AIU shared layout takes is decided by the
// destination encoding in MemoryOpToLLVM.cpp: a dot operand gets the hardware
// matrix load, anything else gets a loop of plain shared-memory loads.

#mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], vecSize = 2}>
#aiu = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [64, 32, 1, 4, 64], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @aiu_dot_operand_uses_matrix_load
  // CHECK: ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16
  tt.func public @aiu_dot_operand_uses_matrix_load(%arg: !ttg.memdesc<128x128xbf16, #aiu, #smem>) {
    %0 = ttg.local_load %arg : !ttg.memdesc<128x128xbf16, #aiu, #smem>
      -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

// The MXFP4 AIU matmul loads its packed operand into this layout, which the
// pipeliner's AIU shared encoding feeds directly. It is not a dot operand, so
// the dispatch takes the distributed path and no matrix load is emitted.

#aiu1 = #ttg.ppu_aiu_shared<{versionMajor = 2, AIUStrategy = [64, 32, 1, 4, 64], order = [0, 1]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 8], [16, 0], [32, 0], [0, 32], [0, 64]], lane = [[4, 0], [8, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 0]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @aiu_linear_destination_falls_back
  // CHECK-NOT: ppu.ldmatrix
  tt.func public @aiu_linear_destination_falls_back(%arg: !ttg.memdesc<64x128xi8, #aiu1, #smem>) {
    %0 = ttg.local_load %arg : !ttg.memdesc<64x128xi8, #aiu1, #smem>
      -> tensor<64x128xi8, #linear>
    tt.return
  }
}
