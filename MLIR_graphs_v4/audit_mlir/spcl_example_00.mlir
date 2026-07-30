#loc1 = loc("Example0.cpp":3:17)
#loc3 = loc("Example0.cpp":1:19)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @InitiationInterval(%arg0: memref<?xf32> loc("Example0.cpp":3:17), %arg1: memref<?xf32> loc("Example0.cpp":3:17), %arg2: memref<?xf32> loc("Example0.cpp":3:17)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c1024000_i32 = arith.constant 1024000 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %true = arith.constant true loc(#loc5)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc6)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc7)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc8)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = scf.if %true -> (i32) {
              %4 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc)
            } else {
              scf.yield %1 : i32 loc(#loc)
            } loc(#loc)
            %3:4 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %2) : (f32, f32, f32, i32) -> (f32, f32, f32, i32) {
              %4 = arith.cmpi slt, %arg6, %c1024000_i32 : i32 loc(#loc9)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6 : f32, f32, f32, i32 loc(#loc10)
            } do {
            ^bb0(%arg3: f32 loc("Example0.cpp":1:19), %arg4: f32 loc("Example0.cpp":1:19), %arg5: f32 loc("Example0.cpp":1:19), %arg6: i32 loc("Example0.cpp":1:19)):
              %4 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      %11 = arith.index_cast %arg6 : i32 to index loc(#loc11)
                      %12 = "polygeist.subindex"(%arg0, %11) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc12)
                      %13 = affine.load %12[0] : memref<?xf32> loc(#loc12)
                      scf.yield %13 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg5 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg5 : f32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      %11 = arith.index_cast %arg6 : i32 to index loc(#loc13)
                      %12 = "polygeist.subindex"(%arg1, %11) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc14)
                      %13 = affine.load %12[0] : memref<?xf32> loc(#loc14)
                      scf.yield %13 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg4 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg4 : f32 loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      %11 = arith.addf %4, %5 : f32 loc(#loc15)
                      %12 = arith.subf %4, %5 : f32 loc(#loc16)
                      %13 = arith.mulf %11, %12 : f32 loc(#loc17)
                      scf.yield %13 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg3 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg3 : f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = arith.index_cast %arg6 : i32 to index loc(#loc18)
                  %9 = "polygeist.subindex"(%arg2, %8) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc19)
                  affine.store %6, %9[0] : memref<?xf32> loc(#loc20)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg6, %c1_i32 : i32 loc(#loc2)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg6 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6, %5, %4, %7 : f32, f32, f32, i32 loc(#loc10)
            } loc(#loc3)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc21)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("Example0.cpp":4:29)
#loc4 = loc("Example0.cpp":4:19)
#loc5 = loc("Example0.cpp":3:12)
#loc6 = loc("Example0.cpp":11:5)
#loc7 = loc("Example0.cpp":4:11)
#loc8 = loc("Example0.cpp":4:1)
#loc9 = loc("Example0.cpp":4:24)
#loc10 = loc("Example0.cpp":4:6)
#loc11 = loc("Example0.cpp":5:27)
#loc12 = loc("Example0.cpp":5:20)
#loc13 = loc("Example0.cpp":6:27)
#loc14 = loc("Example0.cpp":6:20)
#loc15 = loc("Example0.cpp":11:18)
#loc16 = loc("Example0.cpp":11:28)
#loc17 = loc("Example0.cpp":11:23)
#loc18 = loc("Example0.cpp":13:12)
#loc19 = loc("Example0.cpp":13:5)
#loc20 = loc("Example0.cpp":13:14)
#loc21 = loc("Example0.cpp":15:1)
