#loc1 = loc("Example5_Reordered.cpp":3:17)
#loc4 = loc("./Example5.h":3:19)
#loc7 = loc("Example5_Reordered.cpp":19:13)
#loc8 = loc("Example5_Reordered.cpp":12:9)
#loc17 = loc("Example5_Reordered.cpp":11:15)
#loc18 = loc("Example5_Reordered.cpp":9:7)
#loc19 = loc("Example5_Reordered.cpp":7:13)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @MatrixMultiplication(%arg0: memref<?xf64> loc("Example5_Reordered.cpp":3:17), %arg1: memref<?xf64> loc("Example5_Reordered.cpp":3:17), %arg2: memref<?xf64> loc("Example5_Reordered.cpp":3:17)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f64 loc(#loc3)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %true = arith.constant true loc(#loc6)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc7)
    %1 = "polygeist.undef"() : () -> f64 loc(#loc8)
    %alloca = memref.alloca() : memref<1024xf64> loc(#loc9)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc10)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = scf.if %true -> (i32) {
              %4 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc)
            } else {
              scf.yield %0 : i32 loc(#loc)
            } loc(#loc)
            %3:6 = scf.while (%arg3 = %0, %arg4 = %1, %arg5 = %0, %arg6 = %1, %arg7 = %0, %arg8 = %2) : (i32, f64, i32, f64, i32, i32) -> (i32, f64, i32, f64, i32, i32) {
              %4 = arith.cmpi slt, %arg8, %c1024_i32 : i32 loc(#loc11)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : i32, f64, i32, f64, i32, i32 loc(#loc12)
            } do {
            ^bb0(%arg3: i32 loc("./Example5.h":3:19), %arg4: f64 loc("./Example5.h":3:19), %arg5: i32 loc("./Example5.h":3:19), %arg6: f64 loc("./Example5.h":3:19), %arg7: i32 loc("./Example5.h":3:19), %arg8: i32 loc("./Example5.h":3:19)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc13)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4:4 = scf.if %true -> (f64, i32, f64, i32) {
                %7:4 = scf.execute_region -> (f64, i32, f64, i32) {
                  cf.br ^bb1 loc(#loc14)
                ^bb1:  // pred: ^bb0
                  %8:4 = scf.if %true -> (f64, i32, f64, i32) {
                    %9:4 = scf.execute_region -> (f64, i32, f64, i32) {
                      %10 = scf.if %true -> (i32) {
                        %12 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12 : i32 loc(#loc)
                      } else {
                        scf.yield %arg7 : i32 loc(#loc)
                      } loc(#loc)
                      %11:4 = scf.while (%arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %10) : (f64, i32, f64, i32) -> (f64, i32, f64, i32) {
                        %12 = arith.cmpi slt, %arg12, %c1024_i32 : i32 loc(#loc15)
                        scf.condition(%12) %arg9, %arg10, %arg11, %arg12 : f64, i32, f64, i32 loc(#loc16)
                      } do {
                      ^bb0(%arg9: f64 loc("Example5_Reordered.cpp":12:9), %arg10: i32 loc("Example5_Reordered.cpp":11:15), %arg11: f64 loc("Example5_Reordered.cpp":9:7), %arg12: i32 loc("Example5_Reordered.cpp":7:13)):
                        %12 = scf.if %true -> (f64) {
                          %15 = scf.execute_region -> f64 {
                            %16 = scf.if %true -> (f64) {
                              %17 = scf.execute_region -> f64 {
                                %18 = arith.muli %arg8, %c1024_i32 : i32 loc(#loc20)
                                %19 = arith.addi %18, %arg12 : i32 loc(#loc21)
                                %20 = arith.index_cast %19 : i32 to index loc(#loc22)
                                %21 = "polygeist.subindex"(%arg0, %20) : (memref<?xf64>, index) -> memref<?xf64> loc(#loc23)
                                %22 = affine.load %21[0] : memref<?xf64> loc(#loc23)
                                scf.yield %22 : f64 loc(#loc)
                              } loc(#loc)
                              scf.yield %17 : f64 loc(#loc)
                            } else {
                              scf.yield %arg11 : f64 loc(#loc)
                            } loc(#loc)
                            scf.yield %16 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f64 loc(#loc)
                        } else {
                          scf.yield %arg11 : f64 loc(#loc)
                        } loc(#loc)
                        %13:2 = scf.if %true -> (f64, i32) {
                          %15:2 = scf.execute_region -> (f64, i32) {
                            cf.br ^bb1 loc(#loc24)
                          ^bb1:  // pred: ^bb0
                            %16:2 = scf.if %true -> (f64, i32) {
                              %17:2 = scf.execute_region -> (f64, i32) {
                                %18 = scf.if %true -> (i32) {
                                  %20 = scf.execute_region -> i32 {
                                    scf.yield %c0_i32 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %20 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg10 : i32 loc(#loc)
                                } loc(#loc)
                                %19:2 = scf.while (%arg13 = %arg9, %arg14 = %18) : (f64, i32) -> (f64, i32) {
                                  %20 = arith.cmpi slt, %arg14, %c1024_i32 : i32 loc(#loc25)
                                  scf.condition(%20) %arg13, %arg14 : f64, i32 loc(#loc26)
                                } do {
                                ^bb0(%arg13: f64 loc("Example5_Reordered.cpp":12:9), %arg14: i32 loc("Example5_Reordered.cpp":11:15)):
                                  %20 = scf.if %true -> (f64) {
                                    %22 = scf.execute_region -> f64 {
                                      %23 = scf.if %true -> (f64) {
                                        %24 = scf.execute_region -> f64 {
                                          %25 = arith.cmpi eq, %arg12, %c0_i32 : i32 loc(#loc27)
                                          %26 = scf.if %25 -> (f64) {
                                            scf.yield %cst : f64 loc(#loc28)
                                          } else {
                                            %27 = arith.index_cast %arg14 : i32 to index loc(#loc29)
                                            %28 = "polygeist.subindex"(%alloca, %27) : (memref<1024xf64>, index) -> memref<?xf64> loc(#loc30)
                                            %29 = affine.load %28[0] : memref<?xf64> loc(#loc30)
                                            scf.yield %29 : f64 loc(#loc28)
                                          } loc(#loc28)
                                          scf.yield %26 : f64 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %24 : f64 loc(#loc)
                                      } else {
                                        scf.yield %arg13 : f64 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %23 : f64 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %22 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : f64 loc(#loc)
                                  } loc(#loc)
                                  scf.if %true {
                                    scf.execute_region {
                                      %22 = arith.index_cast %arg14 : i32 to index loc(#loc31)
                                      %23 = "polygeist.subindex"(%alloca, %22) : (memref<1024xf64>, index) -> memref<?xf64> loc(#loc32)
                                      %24 = arith.muli %arg12, %c1024_i32 : i32 loc(#loc33)
                                      %25 = arith.addi %24, %arg14 : i32 loc(#loc34)
                                      %26 = arith.index_cast %25 : i32 to index loc(#loc35)
                                      %27 = "polygeist.subindex"(%arg1, %26) : (memref<?xf64>, index) -> memref<?xf64> loc(#loc36)
                                      %28 = affine.load %27[0] : memref<?xf64> loc(#loc36)
                                      %29 = arith.mulf %12, %28 : f64 loc(#loc37)
                                      %30 = arith.addf %20, %29 : f64 loc(#loc38)
                                      affine.store %30, %23[0] : memref<?xf64> loc(#loc39)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  %21 = scf.if %true -> (i32) {
                                    %22 = scf.execute_region -> i32 {
                                      %23 = arith.addi %arg14, %c1_i32 : i32 loc(#loc2)
                                      scf.yield %23 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %22 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %20, %21 : f64, i32 loc(#loc26)
                                } loc(#loc25)
                                scf.yield %19#0, %19#1 : f64, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %17#0, %17#1 : f64, i32 loc(#loc)
                            } else {
                              scf.yield %arg9, %arg10 : f64, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %16#0, %16#1 : f64, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15#0, %15#1 : f64, i32 loc(#loc)
                        } else {
                          scf.yield %arg9, %arg10 : f64, i32 loc(#loc)
                        } loc(#loc)
                        %14 = scf.if %true -> (i32) {
                          %15 = scf.execute_region -> i32 {
                            %16 = arith.addi %arg12, %c1_i32 : i32 loc(#loc40)
                            scf.yield %16 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : i32 loc(#loc)
                        } else {
                          scf.yield %arg12 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %13#0, %13#1, %12, %14 : f64, i32, f64, i32 loc(#loc16)
                      } loc(#loc15)
                      scf.yield %11#0, %11#1, %11#2, %11#3 : f64, i32, f64, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %9#0, %9#1, %9#2, %9#3 : f64, i32, f64, i32 loc(#loc)
                  } else {
                    scf.yield %arg4, %arg5, %arg6, %arg7 : f64, i32, f64, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %8#0, %8#1, %8#2, %8#3 : f64, i32, f64, i32 loc(#loc)
                } loc(#loc)
                scf.yield %7#0, %7#1, %7#2, %7#3 : f64, i32, f64, i32 loc(#loc)
              } else {
                scf.yield %arg4, %arg5, %arg6, %arg7 : f64, i32, f64, i32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %7 = scf.execute_region -> i32 {
                  cf.br ^bb1 loc(#loc41)
                ^bb1:  // pred: ^bb0
                  %8 = scf.if %true -> (i32) {
                    %9 = scf.execute_region -> i32 {
                      %10 = scf.if %true -> (i32) {
                        %12 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12 : i32 loc(#loc)
                      } else {
                        scf.yield %arg3 : i32 loc(#loc)
                      } loc(#loc)
                      %11 = scf.while (%arg9 = %10) : (i32) -> i32 {
                        %12 = arith.cmpi slt, %arg9, %c1024_i32 : i32 loc(#loc42)
                        scf.condition(%12) %arg9 : i32 loc(#loc43)
                      } do {
                      ^bb0(%arg9: i32 loc("Example5_Reordered.cpp":19:13)):
                        scf.if %true {
                          scf.execute_region {
                            %13 = arith.muli %arg8, %c1024_i32 : i32 loc(#loc44)
                            %14 = arith.addi %13, %arg9 : i32 loc(#loc45)
                            %15 = arith.index_cast %14 : i32 to index loc(#loc46)
                            %16 = "polygeist.subindex"(%arg2, %15) : (memref<?xf64>, index) -> memref<?xf64> loc(#loc47)
                            %17 = arith.index_cast %arg9 : i32 to index loc(#loc48)
                            %18 = "polygeist.subindex"(%alloca, %17) : (memref<1024xf64>, index) -> memref<?xf64> loc(#loc49)
                            %19 = affine.load %18[0] : memref<?xf64> loc(#loc49)
                            affine.store %19, %16[0] : memref<?xf64> loc(#loc50)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %12 = scf.if %true -> (i32) {
                          %13 = scf.execute_region -> i32 {
                            %14 = arith.addi %arg9, %c1_i32 : i32 loc(#loc51)
                            scf.yield %14 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12 : i32 loc(#loc43)
                      } loc(#loc42)
                      scf.yield %11 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %9 : i32 loc(#loc)
                  } else {
                    scf.yield %arg3 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %8 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %7 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (i32) {
                %7 = scf.execute_region -> i32 {
                  %8 = arith.addi %arg8, %c1_i32 : i32 loc(#loc52)
                  scf.yield %8 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %7 : i32 loc(#loc)
              } else {
                scf.yield %arg8 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %5, %4#0, %4#1, %4#2, %4#3, %6 : i32, f64, i32, f64, i32, i32 loc(#loc12)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc53)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("Example5_Reordered.cpp":11:33)
#loc3 = loc("Example5_Reordered.cpp":12:40)
#loc5 = loc("Example5_Reordered.cpp":4:19)
#loc6 = loc("Example5_Reordered.cpp":3:12)
#loc9 = loc("Example5_Reordered.cpp":5:8)
#loc10 = loc("Example5_Reordered.cpp":4:1)
#loc11 = loc("Example5_Reordered.cpp":4:24)
#loc12 = loc("Example5_Reordered.cpp":4:6)
#loc13 = loc("Example5_Reordered.cpp":5:1)
#loc14 = loc("Example5_Reordered.cpp":7:1)
#loc15 = loc("Example5_Reordered.cpp":7:26)
#loc16 = loc("Example5_Reordered.cpp":7:8)
#loc20 = loc("Example5_Reordered.cpp":9:26)
#loc21 = loc("Example5_Reordered.cpp":9:30)
#loc22 = loc("Example5_Reordered.cpp":9:33)
#loc23 = loc("Example5_Reordered.cpp":9:22)
#loc24 = loc("Example5_Reordered.cpp":11:1)
#loc25 = loc("Example5_Reordered.cpp":11:28)
#loc26 = loc("Example5_Reordered.cpp":11:10)
#loc27 = loc("Example5_Reordered.cpp":12:32)
#loc28 = loc("Example5_Reordered.cpp":12:29)
#loc29 = loc("Example5_Reordered.cpp":12:49)
#loc30 = loc("Example5_Reordered.cpp":12:44)
#loc31 = loc("Example5_Reordered.cpp":13:14)
#loc32 = loc("Example5_Reordered.cpp":13:9)
#loc33 = loc("Example5_Reordered.cpp":13:33)
#loc34 = loc("Example5_Reordered.cpp":13:37)
#loc35 = loc("Example5_Reordered.cpp":13:40)
#loc36 = loc("Example5_Reordered.cpp":13:29)
#loc37 = loc("Example5_Reordered.cpp":13:27)
#loc38 = loc("Example5_Reordered.cpp":13:23)
#loc39 = loc("Example5_Reordered.cpp":13:16)
#loc40 = loc("Example5_Reordered.cpp":7:31)
#loc41 = loc("Example5_Reordered.cpp":19:1)
#loc42 = loc("Example5_Reordered.cpp":19:26)
#loc43 = loc("Example5_Reordered.cpp":19:8)
#loc44 = loc("Example5_Reordered.cpp":20:11)
#loc45 = loc("Example5_Reordered.cpp":20:15)
#loc46 = loc("Example5_Reordered.cpp":20:18)
#loc47 = loc("Example5_Reordered.cpp":20:7)
#loc48 = loc("Example5_Reordered.cpp":20:27)
#loc49 = loc("Example5_Reordered.cpp":20:22)
#loc50 = loc("Example5_Reordered.cpp":20:20)
#loc51 = loc("Example5_Reordered.cpp":19:31)
#loc52 = loc("Example5_Reordered.cpp":4:29)
#loc53 = loc("Example5_Reordered.cpp":23:1)
