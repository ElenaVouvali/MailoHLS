#loc1 = loc("Example5_Reordered.cpp":3:17)
#loc4 = loc("Example5_Reordered.cpp":4:24)
#loc5 = loc("Example5_Reordered.cpp":7:26)
#loc7 = loc("Example5_Reordered.cpp":11:28)
#loc14 = loc("Example5_Reordered.cpp":19:26)
#set = affine_set<(d0) : (d0 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @MatrixMultiplication(%arg0: memref<?xf64> loc("Example5_Reordered.cpp":3:17), %arg1: memref<?xf64> loc("Example5_Reordered.cpp":3:17), %arg2: memref<?xf64> loc("Example5_Reordered.cpp":3:17)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64 loc(#loc2)
    %alloca = memref.alloca() : memref<1024xf64> loc(#loc3)
    affine.for %arg3 loc("Example5_Reordered.cpp":4:24) = 0 to 1024 {
      affine.for %arg4 loc("Example5_Reordered.cpp":7:26) = 0 to 1024 {
        %0 = affine.load %arg0[%arg4 + %arg3 * 1024] : memref<?xf64> loc(#loc6)
        affine.for %arg5 loc("Example5_Reordered.cpp":11:28) = 0 to 1024 {
          %1 = affine.if #set(%arg4) -> f64 {
            affine.yield %cst : f64 loc(#loc8)
          } else {
            %5 = affine.load %alloca[%arg5] : memref<1024xf64> loc(#loc9)
            affine.yield %5 : f64 loc(#loc8)
          } loc(#loc8)
          %2 = affine.load %arg1[%arg5 + %arg4 * 1024] : memref<?xf64> loc(#loc10)
          %3 = arith.mulf %0, %2 : f64 loc(#loc11)
          %4 = arith.addf %1, %3 : f64 loc(#loc12)
          affine.store %4, %alloca[%arg5] : memref<1024xf64> loc(#loc13)
        } loc(#loc7)
      } loc(#loc5)
      affine.for %arg4 loc("Example5_Reordered.cpp":19:26) = 0 to 1024 {
        %0 = affine.load %alloca[%arg4] : memref<1024xf64> loc(#loc15)
        affine.store %0, %arg2[%arg4 + %arg3 * 1024] : memref<?xf64> loc(#loc16)
      } loc(#loc14)
    } loc(#loc4)
    return loc(#loc17)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("Example5_Reordered.cpp":12:40)
#loc3 = loc("Example5_Reordered.cpp":5:8)
#loc6 = loc("Example5_Reordered.cpp":9:22)
#loc8 = loc("Example5_Reordered.cpp":12:29)
#loc9 = loc("Example5_Reordered.cpp":12:44)
#loc10 = loc("Example5_Reordered.cpp":13:29)
#loc11 = loc("Example5_Reordered.cpp":13:27)
#loc12 = loc("Example5_Reordered.cpp":13:23)
#loc13 = loc("Example5_Reordered.cpp":13:16)
#loc15 = loc("Example5_Reordered.cpp":20:22)
#loc16 = loc("Example5_Reordered.cpp":20:20)
#loc17 = loc("Example5_Reordered.cpp":23:1)
