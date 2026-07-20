module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32>, %arg1: memref<2097152xf32>, %arg2: memref<1048576xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<512xf32>
    %alloca_0 = memref.alloca() : memref<1024xf32>
    %alloca_1 = memref.alloca() : memref<2xf32>
    affine.for %arg3 = 0 to 2 {
      %0 = affine.load %arg0[%arg3] : memref<2xf32>
      affine.store %0, %alloca_1[%arg3] : memref<2xf32>
    }
    %cast = memref.cast %alloca_0 : memref<1024xf32> to memref<?xf32>
    %cast_2 = memref.cast %arg1 : memref<2097152xf32> to memref<?xf32>
    %cast_3 = memref.cast %alloca_1 : memref<2xf32> to memref<?xf32>
    %cast_4 = memref.cast %alloca : memref<512xf32> to memref<?xf32>
    %cast_5 = memref.cast %arg2 : memref<1048576xf32> to memref<?xf32>
    affine.for %arg3 = 0 to 2048 {
      %0 = arith.index_cast %arg3 : index to i32
      func.call @load(%0, %cast_2, %cast) : (i32, memref<?xf32>, memref<?xf32>) -> ()
      func.call @compute_dist(%cast_3, %cast, %cast_4) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
      func.call @store(%0, %cast_4, %cast_5) : (i32, memref<?xf32>, memref<?xf32>) -> ()
    }
    return
  }
  func.func @load(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to 1024 {
      %1 = affine.load %arg1[%arg3 + symbol(%0) * 1024] : memref<?xf32>
      affine.store %1, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @compute_dist(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 512 {
      %0 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst) -> (f32) {
        %1 = affine.load %arg1[%arg4 + %arg3 * 2] : memref<?xf32>
        %2 = affine.load %arg0[%arg4] : memref<?xf32>
        %3 = arith.subf %1, %2 : f32
        %4 = arith.mulf %3, %3 : f32
        %5 = arith.addf %arg5, %4 : f32
        affine.yield %5 : f32
      }
      affine.store %0, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @store(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to 512 {
      %1 = affine.load %arg1[%arg3] : memref<?xf32>
      affine.store %1, %arg2[%arg3 + symbol(%0) * 512] : memref<?xf32>
    }
    return
  }
}
