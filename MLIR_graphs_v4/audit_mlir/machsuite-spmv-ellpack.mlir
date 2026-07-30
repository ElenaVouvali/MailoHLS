#loc1 = loc("spmv.c":8:6)
#loc3 = loc("./spmv.h":13:11)
#loc4 = loc("./spmv.h":12:11)
#loc7 = loc("./spmv.h":15:14)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @ellpack(%arg0: memref<4940xf64> loc("spmv.c":8:6), %arg1: memref<4940xi32> loc("spmv.c":8:6), %arg2: memref<494xf64> loc("spmv.c":8:6), %arg3: memref<494xf64> loc("spmv.c":8:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc3)
    %c494_i32 = arith.constant 494 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %true = arith.constant true loc(#loc6)
    %0 = "polygeist.undef"() : () -> f64 loc(#loc7)
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
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc8)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc9)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1:3 = scf.while (%arg4 = %0, %arg5 = %0, %arg6 = %c0_i32) : (f64, f64, i32) -> (f64, f64, i32) {
              %2 = arith.cmpi slt, %arg6, %c494_i32 : i32 loc(#loc10)
              scf.condition(%2) %arg4, %arg5, %arg6 : f64, f64, i32 loc(#loc11)
            } do {
            ^bb0(%arg4: f64 loc("./spmv.h":12:11), %arg5: f64 loc("./spmv.h":12:11), %arg6: i32 loc("./spmv.h":12:11)):
              %2 = scf.if %true -> (f64) {
                %5 = scf.execute_region -> f64 {
                  %6 = scf.if %true -> (f64) {
                    %7 = scf.execute_region -> f64 {
                      %8 = arith.index_cast %arg6 : i32 to index loc(#loc12)
                      %9 = "polygeist.subindex"(%arg3, %8) : (memref<494xf64>, index) -> memref<?xf64> loc(#loc13)
                      %10 = affine.load %9[0] : memref<?xf64> loc(#loc13)
                      scf.yield %10 : f64 loc(#loc)
                    } loc(#loc)
                    scf.yield %7 : f64 loc(#loc)
                  } else {
                    scf.yield %arg4 : f64 loc(#loc)
                  } loc(#loc)
                  scf.yield %6 : f64 loc(#loc)
                } loc(#loc)
                scf.yield %5 : f64 loc(#loc)
              } else {
                scf.yield %arg4 : f64 loc(#loc)
              } loc(#loc)
              %3:2 = scf.if %true -> (f64, f64) {
                %5:2 = scf.execute_region -> (f64, f64) {
                  cf.br ^bb1 loc(#loc14)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc15)
                ^bb2:  // pred: ^bb1
                  %6:2 = scf.if %true -> (f64, f64) {
                    %7:2 = scf.execute_region -> (f64, f64) {
                      %8:3 = scf.while (%arg7 = %2, %arg8 = %arg5, %arg9 = %c0_i32) : (f64, f64, i32) -> (f64, f64, i32) {
                        %9 = arith.cmpi slt, %arg9, %c10_i32 : i32 loc(#loc16)
                        scf.condition(%9) %arg7, %arg8, %arg9 : f64, f64, i32 loc(#loc17)
                      } do {
                      ^bb0(%arg7: f64 loc("./spmv.h":15:14), %arg8: f64 loc("./spmv.h":15:14), %arg9: i32 loc("./spmv.h":13:11)):
                        %9 = scf.if %true -> (f64) {
                          %12 = scf.execute_region -> f64 {
                            %13 = arith.muli %arg6, %c10_i32 : i32 loc(#loc18)
                            %14 = arith.addi %arg9, %13 : i32 loc(#loc19)
                            %15 = arith.index_cast %14 : i32 to index loc(#loc20)
                            %16 = "polygeist.subindex"(%arg0, %15) : (memref<4940xf64>, index) -> memref<?xf64> loc(#loc21)
                            %17 = affine.load %16[0] : memref<?xf64> loc(#loc21)
                            %18 = "polygeist.subindex"(%arg1, %15) : (memref<4940xi32>, index) -> memref<?xi32> loc(#loc22)
                            %19 = affine.load %18[0] : memref<?xi32> loc(#loc22)
                            %20 = arith.index_cast %19 : i32 to index loc(#loc23)
                            %21 = "polygeist.subindex"(%arg2, %20) : (memref<494xf64>, index) -> memref<?xf64> loc(#loc24)
                            %22 = affine.load %21[0] : memref<?xf64> loc(#loc24)
                            %23 = arith.mulf %17, %22 : f64 loc(#loc25)
                            scf.yield %23 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : f64 loc(#loc)
                        } else {
                          scf.yield %arg8 : f64 loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (f64) {
                          %12 = scf.execute_region -> f64 {
                            %13 = arith.addf %arg7, %9 : f64 loc(#loc26)
                            scf.yield %13 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : f64 loc(#loc)
                        } else {
                          scf.yield %arg7 : f64 loc(#loc)
                        } loc(#loc)
                        %11 = scf.if %true -> (i32) {
                          %12 = scf.execute_region -> i32 {
                            %13 = arith.addi %arg9, %c1_i32 : i32 loc(#loc2)
                            scf.yield %13 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10, %9, %11 : f64, f64, i32 loc(#loc17)
                      } loc(#loc3)
                      scf.yield %8#0, %8#1 : f64, f64 loc(#loc)
                    } loc(#loc)
                    scf.yield %7#0, %7#1 : f64, f64 loc(#loc)
                  } else {
                    scf.yield %2, %arg5 : f64, f64 loc(#loc)
                  } loc(#loc)
                  scf.yield %6#0, %6#1 : f64, f64 loc(#loc)
                } loc(#loc)
                scf.yield %5#0, %5#1 : f64, f64 loc(#loc)
              } else {
                scf.yield %2, %arg5 : f64, f64 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg6 : i32 to index loc(#loc27)
                  %6 = "polygeist.subindex"(%arg3, %5) : (memref<494xf64>, index) -> memref<?xf64> loc(#loc28)
                  affine.store %3#0, %6[0] : memref<?xf64> loc(#loc29)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg6, %c1_i32 : i32 loc(#loc30)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg6 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3#0, %3#1, %4 : f64, f64, i32 loc(#loc11)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc31)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("spmv.c":15:40)
#loc5 = loc("spmv.c":13:27)
#loc6 = loc("spmv.c":8:1)
#loc8 = loc("spmv.c":13:1)
#loc9 = loc("spmv.c":13:8)
#loc10 = loc("spmv.c":13:31)
#loc11 = loc("spmv.c":13:20)
#loc12 = loc("spmv.c":14:25)
#loc13 = loc("spmv.c":14:20)
#loc14 = loc("spmv.c":15:1)
#loc15 = loc("spmv.c":15:12)
#loc16 = loc("spmv.c":15:35)
#loc17 = loc("spmv.c":15:24)
#loc18 = loc("spmv.c":16:33)
#loc19 = loc("spmv.c":16:30)
#loc20 = loc("spmv.c":16:35)
#loc21 = loc("spmv.c":16:22)
#loc22 = loc("spmv.c":16:43)
#loc23 = loc("spmv.c":16:56)
#loc24 = loc("spmv.c":16:39)
#loc25 = loc("spmv.c":16:37)
#loc26 = loc("spmv.c":17:21)
#loc27 = loc("spmv.c":19:14)
#loc28 = loc("spmv.c":19:9)
#loc29 = loc("spmv.c":19:16)
#loc30 = loc("spmv.c":13:36)
#loc31 = loc("spmv.c":21:1)
