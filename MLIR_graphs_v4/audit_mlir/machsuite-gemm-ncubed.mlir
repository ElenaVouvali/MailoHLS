#loc1 = loc("gemm.c":3:6)
#loc4 = loc("./gemm.h":10:18)
#loc7 = loc("./gemm.h":7:14)
#loc8 = loc("gemm.c":5:5)
#loc15 = loc("gemm.c":9:28)
#loc20 = loc("gemm.c":12:31)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<4096xf64> loc("gemm.c":3:6), %arg1: memref<4096xf64> loc("gemm.c":3:6), %arg2: memref<4096xf64> loc("gemm.c":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f64 loc(#loc3)
    %c64_i32 = arith.constant 64 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %true = arith.constant true loc(#loc6)
    %0 = "polygeist.undef"() : () -> f64 loc(#loc7)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc8)
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
        cf.br ^bb1 loc(#loc9)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc10)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %2:5 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %1, %arg6 = %1, %arg7 = %c0_i32) : (f64, f64, i32, i32, i32) -> (f64, f64, i32, i32, i32) {
              %3 = arith.cmpi slt, %arg7, %c64_i32 : i32 loc(#loc11)
              scf.condition(%3) %arg3, %arg4, %arg5, %arg6, %arg7 : f64, f64, i32, i32, i32 loc(#loc12)
            } do {
            ^bb0(%arg3: f64 loc("./gemm.h":10:18), %arg4: f64 loc("./gemm.h":10:18), %arg5: i32 loc("./gemm.h":10:18), %arg6: i32 loc("./gemm.h":10:18), %arg7: i32 loc("./gemm.h":10:18)):
              %3:4 = scf.if %true -> (f64, f64, i32, i32) {
                %5:4 = scf.execute_region -> (f64, f64, i32, i32) {
                  cf.br ^bb1 loc(#loc13)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc14)
                ^bb2:  // pred: ^bb1
                  %6:4 = scf.if %true -> (f64, f64, i32, i32) {
                    %7:4 = scf.execute_region -> (f64, f64, i32, i32) {
                      %8:5 = scf.while (%arg8 = %arg3, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %c0_i32) : (f64, f64, i32, i32, i32) -> (f64, f64, i32, i32, i32) {
                        %9 = arith.cmpi slt, %arg12, %c64_i32 : i32 loc(#loc15)
                        scf.condition(%9) %arg8, %arg9, %arg10, %arg11, %arg12 : f64, f64, i32, i32, i32 loc(#loc16)
                      } do {
                      ^bb0(%arg8: f64 loc("./gemm.h":7:14), %arg9: f64 loc("./gemm.h":7:14), %arg10: i32 loc("gemm.c":5:5), %arg11: i32 loc("gemm.c":5:5), %arg12: i32 loc("gemm.c":9:28)):
                        %9 = scf.if %true -> (i32) {
                          %13 = scf.execute_region -> i32 {
                            %14 = arith.muli %arg7, %c64_i32 : i32 loc(#loc17)
                            scf.yield %14 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : i32 loc(#loc)
                        } else {
                          scf.yield %arg10 : i32 loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (f64) {
                          %13 = scf.execute_region -> f64 {
                            %14 = scf.if %true -> (f64) {
                              %15 = scf.execute_region -> f64 {
                                scf.yield %cst : f64 loc(#loc)
                              } loc(#loc)
                              scf.yield %15 : f64 loc(#loc)
                            } else {
                              scf.yield %arg8 : f64 loc(#loc)
                            } loc(#loc)
                            scf.yield %14 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : f64 loc(#loc)
                        } else {
                          scf.yield %arg8 : f64 loc(#loc)
                        } loc(#loc)
                        %11:3 = scf.if %true -> (f64, f64, i32) {
                          %13:3 = scf.execute_region -> (f64, f64, i32) {
                            cf.br ^bb1 loc(#loc18)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc19)
                          ^bb2:  // pred: ^bb1
                            %14:3 = scf.if %true -> (f64, f64, i32) {
                              %15:3 = scf.execute_region -> (f64, f64, i32) {
                                %16:4 = scf.while (%arg13 = %10, %arg14 = %arg9, %arg15 = %arg11, %arg16 = %c0_i32) : (f64, f64, i32, i32) -> (f64, f64, i32, i32) {
                                  %17 = arith.cmpi slt, %arg16, %c64_i32 : i32 loc(#loc20)
                                  scf.condition(%17) %arg13, %arg14, %arg15, %arg16 : f64, f64, i32, i32 loc(#loc21)
                                } do {
                                ^bb0(%arg13: f64 loc("./gemm.h":7:14), %arg14: f64 loc("./gemm.h":7:14), %arg15: i32 loc("gemm.c":5:5), %arg16: i32 loc("gemm.c":12:31)):
                                  %17 = scf.if %true -> (i32) {
                                    %21 = scf.execute_region -> i32 {
                                      %22 = arith.muli %arg16, %c64_i32 : i32 loc(#loc22)
                                      scf.yield %22 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %21 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg15 : i32 loc(#loc)
                                  } loc(#loc)
                                  %18 = scf.if %true -> (f64) {
                                    %21 = scf.execute_region -> f64 {
                                      %22 = arith.addi %9, %arg16 : i32 loc(#loc23)
                                      %23 = arith.index_cast %22 : i32 to index loc(#loc24)
                                      %24 = "polygeist.subindex"(%arg0, %23) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc25)
                                      %25 = affine.load %24[0] : memref<?xf64> loc(#loc25)
                                      %26 = arith.addi %17, %arg12 : i32 loc(#loc26)
                                      %27 = arith.index_cast %26 : i32 to index loc(#loc27)
                                      %28 = "polygeist.subindex"(%arg1, %27) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc28)
                                      %29 = affine.load %28[0] : memref<?xf64> loc(#loc28)
                                      %30 = arith.mulf %25, %29 : f64 loc(#loc29)
                                      scf.yield %30 : f64 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %21 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : f64 loc(#loc)
                                  } loc(#loc)
                                  %19 = scf.if %true -> (f64) {
                                    %21 = scf.execute_region -> f64 {
                                      %22 = arith.addf %arg13, %18 : f64 loc(#loc30)
                                      scf.yield %22 : f64 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %21 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : f64 loc(#loc)
                                  } loc(#loc)
                                  %20 = scf.if %true -> (i32) {
                                    %21 = scf.execute_region -> i32 {
                                      %22 = arith.addi %arg16, %c1_i32 : i32 loc(#loc2)
                                      scf.yield %22 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %21 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg16 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %19, %18, %17, %20 : f64, f64, i32, i32 loc(#loc21)
                                } loc(#loc20)
                                scf.yield %16#0, %16#1, %16#2 : f64, f64, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %15#0, %15#1, %15#2 : f64, f64, i32 loc(#loc)
                            } else {
                              scf.yield %10, %arg9, %arg11 : f64, f64, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %14#0, %14#1, %14#2 : f64, f64, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13#0, %13#1, %13#2 : f64, f64, i32 loc(#loc)
                        } else {
                          scf.yield %10, %arg9, %arg11 : f64, f64, i32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %13 = arith.addi %9, %arg12 : i32 loc(#loc31)
                            %14 = arith.index_cast %13 : i32 to index loc(#loc32)
                            %15 = "polygeist.subindex"(%arg2, %14) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc33)
                            affine.store %11#0, %15[0] : memref<?xf64> loc(#loc34)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %12 = scf.if %true -> (i32) {
                          %13 = scf.execute_region -> i32 {
                            %14 = arith.addi %arg12, %c1_i32 : i32 loc(#loc35)
                            scf.yield %14 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : i32 loc(#loc)
                        } else {
                          scf.yield %arg12 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %11#0, %11#1, %9, %11#2, %12 : f64, f64, i32, i32, i32 loc(#loc16)
                      } loc(#loc15)
                      scf.yield %8#0, %8#1, %8#2, %8#3 : f64, f64, i32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %7#0, %7#1, %7#2, %7#3 : f64, f64, i32, i32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg5, %arg6 : f64, f64, i32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %6#0, %6#1, %6#2, %6#3 : f64, f64, i32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %5#0, %5#1, %5#2, %5#3 : f64, f64, i32, i32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg5, %arg6 : f64, f64, i32, i32 loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg7, %c1_i32 : i32 loc(#loc36)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg7 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3#0, %3#1, %3#2, %3#3, %4 : f64, f64, i32, i32, i32 loc(#loc12)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc37)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("gemm.c":12:42)
#loc3 = loc("gemm.c":11:24)
#loc5 = loc("gemm.c":8:20)
#loc6 = loc("gemm.c":3:1)
#loc9 = loc("gemm.c":8:1)
#loc10 = loc("gemm.c":8:8)
#loc11 = loc("gemm.c":8:23)
#loc12 = loc("gemm.c":8:14)
#loc13 = loc("gemm.c":9:1)
#loc14 = loc("gemm.c":9:12)
#loc16 = loc("gemm.c":9:19)
#loc17 = loc("gemm.c":10:23)
#loc18 = loc("gemm.c":12:1)
#loc19 = loc("gemm.c":12:16)
#loc21 = loc("gemm.c":12:22)
#loc22 = loc("gemm.c":13:27)
#loc23 = loc("gemm.c":14:33)
#loc24 = loc("gemm.c":14:36)
#loc25 = loc("gemm.c":14:24)
#loc26 = loc("gemm.c":14:49)
#loc27 = loc("gemm.c":14:52)
#loc28 = loc("gemm.c":14:40)
#loc29 = loc("gemm.c":14:38)
#loc30 = loc("gemm.c":15:21)
#loc31 = loc("gemm.c":17:24)
#loc32 = loc("gemm.c":17:27)
#loc33 = loc("gemm.c":17:13)
#loc34 = loc("gemm.c":17:30)
#loc35 = loc("gemm.c":9:39)
#loc36 = loc("gemm.c":8:34)
#loc37 = loc("gemm.c":20:1)
