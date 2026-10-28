#loc1 = loc("kernel.cpp":6:6)
#loc2 = loc("kernel.cpp":11:28)
#loc3 = loc("kernel.cpp":12:30)
#loc5 = loc("kernel.cpp":14:32)
#set = affine_set<(d0, d1) : (d0 - d1 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trmm(%arg0: f64 loc("kernel.cpp":6:6), %arg1: memref<60x60xf64> loc("kernel.cpp":6:6), %arg2: memref<60x80xf64> loc("kernel.cpp":6:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 loc("kernel.cpp":11:28) = 0 to 60 {
      affine.for %arg4 loc("kernel.cpp":12:30) = 0 to 80 {
        %0 = affine.load %arg2[%arg3, %arg4] : memref<60x80xf64> loc(#loc4)
        %1 = affine.for %arg5 loc("kernel.cpp":14:32) = 0 to 60 iter_args(%arg6 = %0) -> (f64) {
          %3 = affine.if #set(%arg5, %arg3) -> f64 {
            %4 = affine.load %arg1[%arg5, %arg3] : memref<60x60xf64> loc(#loc7)
            %5 = affine.load %arg2[%arg5, %arg4] : memref<60x80xf64> loc(#loc8)
            %6 = arith.mulf %4, %5 : f64 loc(#loc9)
            %7 = arith.addf %arg6, %6 : f64 loc(#loc10)
            affine.yield %7 : f64 loc(#loc)
          } else {
            affine.yield %arg6 : f64 loc(#loc6)
          } loc(#loc6)
          affine.yield %3 : f64 loc(#loc11)
        } loc(#loc5)
        %2 = arith.mulf %arg0, %1 : f64 loc(#loc12)
        affine.store %2, %arg2[%arg3, %arg4] : memref<60x80xf64> loc(#loc13)
      } loc(#loc3)
    } loc(#loc2)
    return loc(#loc14)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc4 = loc("kernel.cpp":13:20)
#loc6 = loc("kernel.cpp":15:9)
#loc7 = loc("kernel.cpp":16:18)
#loc8 = loc("kernel.cpp":16:28)
#loc9 = loc("kernel.cpp":16:26)
#loc10 = loc("kernel.cpp":16:15)
#loc11 = loc("kernel.cpp":14:14)
#loc12 = loc("kernel.cpp":19:23)
#loc13 = loc("kernel.cpp":19:15)
#loc14 = loc("kernel.cpp":22:1)
