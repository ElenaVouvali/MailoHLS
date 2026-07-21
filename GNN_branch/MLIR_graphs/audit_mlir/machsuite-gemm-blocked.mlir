#loc1 = loc("gemm.c":10:6)
#loc2 = loc("gemm.c":15:31)
#loc3 = loc("gemm.c":16:35)
#loc4 = loc("gemm.c":17:37)
#loc5 = loc("gemm.c":18:40)
#loc7 = loc("gemm.c":22:44)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bbgemm(%arg0: memref<4096xf64> loc("gemm.c":10:6), %arg1: memref<4096xf64> loc("gemm.c":10:6), %arg2: memref<4096xf64> loc("gemm.c":10:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 loc("gemm.c":15:31) = 0 to 64 step 8 {
      affine.for %arg4 loc("gemm.c":16:35) = 0 to 64 step 8 {
        affine.for %arg5 loc("gemm.c":17:37) = 0 to 64 {
          affine.for %arg6 loc("gemm.c":18:40) = 0 to 8 {
            %0 = affine.load %arg0[%arg4 + %arg6 + %arg5 * 64] : memref<4096xf64> loc(#loc6)
            affine.for %arg7 loc("gemm.c":22:44) = 0 to 8 {
              %1 = affine.load %arg1[%arg3 + %arg7 + %arg6 * 64 + %arg4 * 64] : memref<4096xf64> loc(#loc8)
              %2 = arith.mulf %0, %1 : f64 loc(#loc9)
              %3 = affine.load %arg2[%arg3 + %arg7 + %arg5 * 64] : memref<4096xf64> loc(#loc10)
              %4 = arith.addf %3, %2 : f64 loc(#loc10)
              affine.store %4, %arg2[%arg3 + %arg7 + %arg5 * 64] : memref<4096xf64> loc(#loc10)
            } loc(#loc7)
          } loc(#loc5)
        } loc(#loc4)
      } loc(#loc3)
    } loc(#loc2)
    return loc(#loc11)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc6 = loc("gemm.c":21:30)
#loc8 = loc("gemm.c":23:40)
#loc9 = loc("gemm.c":23:38)
#loc10 = loc("gemm.c":24:46)
#loc11 = loc("gemm.c":30:1)
