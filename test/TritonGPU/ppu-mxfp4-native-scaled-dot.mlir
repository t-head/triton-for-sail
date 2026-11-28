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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
!a = tensor<128x64xi8, #blocked>
!b = tensor<64x128xi8, #blocked>
!acc = tensor<128x128xf32, #blocked>
!mx_scale = tensor<128x4xi8, #scale>
!nv_scale = tensor<128x8xf8E4M3FN, #scale>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @nvfp4_decomposes
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @nvfp4_decomposes(
      %a: !a, %sa: !nv_scale, %b: !b, %sb: !nv_scale, %acc: !acc) -> !acc {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !a, !nv_scale * !b, !nv_scale -> !acc
    tt.return %0 : !acc
  }

  // Isolate group size from scale type: these E8M0 bytes scale 16 values each.
  // CHECK-LABEL: @group16_byte_scales_decompose
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @group16_byte_scales_decompose(
      %a: !a, %sa: tensor<128x8xi8, #scale>, %b: !b, %sb: tensor<128x8xi8, #scale>,
      %acc: !acc) -> !acc {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !a, tensor<128x8xi8, #scale> * !b, tensor<128x8xi8, #scale> -> !acc
    tt.return %0 : !acc
  }

  // With group size 32, changing only the RHS scale type still requires decomposition.
  // CHECK-LABEL: @fp8_rhs_scale_decomposes
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @fp8_rhs_scale_decomposes(
      %a: !a, %sa: !mx_scale, %b: !b, %sb: tensor<128x4xf8E4M3FN, #scale>,
      %acc: !acc) -> !acc {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !a, !mx_scale * !b, tensor<128x4xf8E4M3FN, #scale> -> !acc
    tt.return %0 : !acc
  }

  // CHECK-LABEL: @f16_accumulator_decomposes
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @f16_accumulator_decomposes(
      %a: !a, %sa: !mx_scale, %b: !b, %sb: !mx_scale,
      %acc: tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #blocked> {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !a, !mx_scale * !b, !mx_scale -> tensor<128x128xf16, #blocked>
    tt.return %0 : tensor<128x128xf16, #blocked>
  }

  // Packed K is 64 bytes, but N=8 is too small for m16n16k64.
  // CHECK-LABEL: @small_n_decomposes
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @small_n_decomposes(
      %a: !a, %sa: !mx_scale, %b: tensor<64x8xi8, #blocked>, %sb: tensor<8x4xi8, #scale>,
      %acc: tensor<128x8xf32, #blocked>) -> tensor<128x8xf32, #blocked> {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !a, !mx_scale * tensor<64x8xi8, #blocked>, tensor<8x4xi8, #scale> -> tensor<128x8xf32, #blocked>
    tt.return %0 : tensor<128x8xf32, #blocked>
  }
}

// -----

#batched = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
!operand = tensor<16x64x64xi8, #batched>
!scale = tensor<16x64x4xi8, #batched>
!acc = tensor<16x64x64xf32, #batched>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:89", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @batched_mxfp4_decomposes
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot %
  tt.func public @batched_mxfp4_decomposes(
      %a: !operand, %sa: !scale, %b: !operand, %sb: !scale, %acc: !acc) -> !acc {
    %0 = tt.dot_scaled %a scale %sa, %b scale %sb, %acc lhs = e2m1 rhs = e2m1 {fastMath = false}
      : !operand, !scale * !operand, !scale -> !acc
    tt.return %0 : !acc
  }
}
