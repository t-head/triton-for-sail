// RUN: triton-opt %s -split-input-file -tritonppugpu-accelerate-matmul | FileCheck %s

// An MXFP4 scaled dot must reach the native PPU0015 path. The scale operands
// carry one element per thread, so their layout has no element contiguity, and
// asking `getOrder` for their order answers [0, 1] even though the layout is
// declared `order = [1, 0]`. Reading that as "not row major" declines the
// native path and decomposes the dot into a BF16 upcast plus a BF16 MMA, which
// also costs the AIU operand loads their specialized matrix load.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// One element per thread: no element contiguity in either dimension.
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #mma = #ttg.ppu_mma<{versionMajor = 2, versionMinor = 0
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mxfp4_scaled_dot_stays_native
  // The scaled dot survives, with its operands rewritten onto the PPU MMA layout.
  // CHECK: tt.dot_scaled
  // CHECK-SAME: parent = #mma
  // No BF16 decomposition: that path is what emits the FP4 upcast.
  // CHECK-NOT: ttg.fp4_to_fp
  tt.func public @mxfp4_scaled_dot_stays_native(
      %a: tensor<128x64xi8, #blocked1>, %a_scale: tensor<128x4xi8, #scale>,
      %b: tensor<64x128xi8, #blocked>, %b_scale: tensor<128x4xi8, #scale>,
      %acc: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc
      lhs = e2m1 rhs = e2m1 {fastMath = false}
      : tensor<128x64xi8, #blocked1>, tensor<128x4xi8, #scale>
      * tensor<64x128xi8, #blocked>, tensor<128x4xi8, #scale>
      -> tensor<128x128xf32, #blocked>
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}
