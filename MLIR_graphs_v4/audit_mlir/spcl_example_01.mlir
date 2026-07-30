#loc1 = loc("Example1_Pipelined.cpp":24:17)
#loc5 = loc("Example1_Pipelined.cpp":3:6)
#loc7 = loc("Example1_Pipelined.cpp":9:28)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @Example1_Pipelined(%arg0: memref<?xf32> loc("Example1_Pipelined.cpp":24:17), %arg1: memref<?xf32> loc("Example1_Pipelined.cpp":24:17)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true loc(#loc2)
    scf.if %true {
      scf.execute_region {
        func.call @_Z15Simple1DStencilPKfPf(%arg0, %arg1) : (memref<?xf32>, memref<?xf32>) -> () loc(#loc3)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc4)
  } loc(#loc1)
  func.func @_Z15Simple1DStencilPKfPf(%arg0: memref<?xf32> loc("Example1_Pipelined.cpp":3:6), %arg1: memref<?xf32> loc("Example1_Pipelined.cpp":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333000e-01 : f32 loc(#loc6)
    %c1023999_i32 = arith.constant 1023999 : i32 loc(#loc7)
    %c1_i32 = arith.constant 1 : i32 loc(#loc8)
    %true = arith.constant true loc(#loc9)
    %c1 = arith.constant 1 : index loc(#loc)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc10)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc11)
    %2 = scf.if %true -> (f32) {
      %4 = scf.execute_region -> f32 {
        %5 = scf.if %true -> (f32) {
          %6 = scf.execute_region -> f32 {
            %7 = affine.load %arg0[0] : memref<?xf32> loc(#loc12)
            scf.yield %7 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %6 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %5 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %4 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %3 = scf.if %true -> (f32) {
      %4 = scf.execute_region -> f32 {
        %5 = scf.if %true -> (f32) {
          %6 = scf.execute_region -> f32 {
            %7 = "polygeist.subindex"(%arg0, %c1) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc13)
            %8 = affine.load %7[0] : memref<?xf32> loc(#loc13)
            scf.yield %8 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %6 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %5 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %4 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc14)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %4 = scf.if %true -> (i32) {
              %6 = scf.execute_region -> i32 {
                scf.yield %c1_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6 : i32 loc(#loc)
            } else {
              scf.yield %1 : i32 loc(#loc)
            } loc(#loc)
            %5:5 = scf.while (%arg2 = %0, %arg3 = %0, %arg4 = %4, %arg5 = %3, %arg6 = %2) : (f32, f32, i32, f32, f32) -> (f32, f32, i32, f32, f32) {
              %6 = arith.cmpi slt, %arg4, %c1023999_i32 : i32 loc(#loc15)
              scf.condition(%6) %arg2, %arg3, %arg4, %arg5, %arg6 : f32, f32, i32, f32, f32 loc(#loc16)
            } do {
            ^bb0(%arg2: f32 loc("Example1_Pipelined.cpp":9:28), %arg3: f32 loc("Example1_Pipelined.cpp":9:28), %arg4: i32 loc("Example1_Pipelined.cpp":9:28), %arg5: f32 loc("Example1_Pipelined.cpp":9:28), %arg6: f32 loc("Example1_Pipelined.cpp":9:28)):
              %6 = scf.if %true -> (f32) {
                %11 = scf.execute_region -> f32 {
                  %12 = scf.if %true -> (f32) {
                    %13 = scf.execute_region -> f32 {
                      %14 = arith.addi %arg4, %c1_i32 : i32 loc(#loc17)
                      %15 = arith.index_cast %14 : i32 to index loc(#loc18)
                      %16 = "polygeist.subindex"(%arg0, %15) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc19)
                      %17 = affine.load %16[0] : memref<?xf32> loc(#loc19)
                      scf.yield %17 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %13 : f32 loc(#loc)
                  } else {
                    scf.yield %arg3 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %12 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %11 : f32 loc(#loc)
              } else {
                scf.yield %arg3 : f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  scf.if %true {
                    scf.execute_region {
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (f32) {
                %11 = scf.execute_region -> f32 {
                  %12 = scf.if %true -> (f32) {
                    %13 = scf.execute_region -> f32 {
                      %14 = arith.addf %arg6, %arg5 : f32 loc(#loc20)
                      %15 = arith.addf %14, %6 : f32 loc(#loc21)
                      %16 = arith.mulf %15, %cst : f32 loc(#loc22)
                      scf.yield %16 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %13 : f32 loc(#loc)
                  } else {
                    scf.yield %arg2 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %12 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %11 : f32 loc(#loc)
              } else {
                scf.yield %arg2 : f32 loc(#loc)
              } loc(#loc)
              %8 = scf.if %true -> (f32) {
                scf.execute_region {
                  scf.yield loc(#loc)
                } loc(#loc)
                scf.yield %arg5 : f32 loc(#loc)
              } else {
                scf.yield %arg6 : f32 loc(#loc)
              } loc(#loc)
              %9 = scf.if %true -> (f32) {
                scf.execute_region {
                  scf.yield loc(#loc)
                } loc(#loc)
                scf.yield %6 : f32 loc(#loc)
              } else {
                scf.yield %arg5 : f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %11 = arith.index_cast %arg4 : i32 to index loc(#loc23)
                  %12 = "polygeist.subindex"(%arg1, %11) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc24)
                  affine.store %7, %12[0] : memref<?xf32> loc(#loc25)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %10 = scf.if %true -> (i32) {
                %11 = scf.execute_region -> i32 {
                  %12 = arith.addi %arg4, %c1_i32 : i32 loc(#loc26)
                  scf.yield %12 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %11 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %7, %6, %10, %9, %8 : f32, f32, i32, f32, f32 loc(#loc16)
            } loc(#loc7)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc27)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("Example1_Pipelined.cpp":24:12)
#loc3 = loc("Example1_Pipelined.cpp":30:3)
#loc4 = loc("Example1_Pipelined.cpp":31:1)
#loc6 = loc("Example1_Pipelined.cpp":13:30)
#loc8 = loc("Example1_Pipelined.cpp":9:19)
#loc9 = loc("Example1_Pipelined.cpp":3:1)
#loc10 = loc("Example1_Pipelined.cpp":14:5)
#loc11 = loc("Example1_Pipelined.cpp":9:11)
#loc12 = loc("Example1_Pipelined.cpp":6:16)
#loc13 = loc("Example1_Pipelined.cpp":7:18)
#loc14 = loc("Example1_Pipelined.cpp":9:1)
#loc15 = loc("Example1_Pipelined.cpp":9:24)
#loc16 = loc("Example1_Pipelined.cpp":9:6)
#loc17 = loc("Example1_Pipelined.cpp":11:36)
#loc18 = loc("Example1_Pipelined.cpp":11:39)
#loc19 = loc("Example1_Pipelined.cpp":11:24)
#loc20 = loc("Example1_Pipelined.cpp":14:41)
#loc21 = loc("Example1_Pipelined.cpp":14:50)
#loc22 = loc("Example1_Pipelined.cpp":14:33)
#loc23 = loc("Example1_Pipelined.cpp":20:17)
#loc24 = loc("Example1_Pipelined.cpp":20:5)
#loc25 = loc("Example1_Pipelined.cpp":20:19)
#loc26 = loc("Example1_Pipelined.cpp":9:33)
#loc27 = loc("Example1_Pipelined.cpp":22:1)
