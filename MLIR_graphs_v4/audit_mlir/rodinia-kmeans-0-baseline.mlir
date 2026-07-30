#loc1 = loc("kmeans.cpp":4:6)
#loc7 = loc("./kmeans.h":14:24)
#loc10 = loc("kmeans.cpp":25:5)
#loc11 = loc("kmeans.cpp":24:18)
#loc20 = loc("kmeans.cpp":22:4)
#loc21 = loc("kmeans.cpp":21:16)
#loc22 = loc("kmeans.cpp":18:3)
#loc23 = loc("kmeans.cpp":17:3)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<?xf32> loc("kmeans.cpp":4:6), %arg1: memref<?xf32> loc("kmeans.cpp":4:6), %arg2: memref<?xi32> loc("kmeans.cpp":4:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c34_i32 = arith.constant 34 : i32 loc(#loc3)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc4)
    %c5_i32 = arith.constant 5 : i32 loc(#loc5)
    %cst_0 = arith.constant 3.40282347E+38 : f32 loc(#loc6)
    %c409600_i32 = arith.constant 409600 : i32 loc(#loc7)
    %c0_i32 = arith.constant 0 : i32 loc(#loc8)
    %true = arith.constant true loc(#loc9)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc10)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc11)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc12)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc13)
      ^bb2:  // pred: ^bb1
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
            %3:7 = scf.while (%arg3 = %0, %arg4 = %1, %arg5 = %0, %arg6 = %1, %arg7 = %1, %arg8 = %0, %arg9 = %2) : (f32, i32, f32, i32, i32, f32, i32) -> (f32, i32, f32, i32, i32, f32, i32) {
              %4 = arith.cmpi slt, %arg9, %c409600_i32 : i32 loc(#loc14)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : f32, i32, f32, i32, i32, f32, i32 loc(#loc15)
            } do {
            ^bb0(%arg3: f32 loc("./kmeans.h":14:24), %arg4: i32 loc("./kmeans.h":14:24), %arg5: f32 loc("./kmeans.h":14:24), %arg6: i32 loc("./kmeans.h":14:24), %arg7: i32 loc("./kmeans.h":14:24), %arg8: f32 loc("./kmeans.h":14:24), %arg9: i32 loc("./kmeans.h":14:24)):
              %4 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      scf.yield %cst_0 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg8 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg8 : f32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      scf.yield %c0_i32 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg7 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg7 : i32 loc(#loc)
              } loc(#loc)
              %6:6 = scf.if %true -> (f32, i32, f32, i32, i32, f32) {
                %8:6 = scf.execute_region -> (f32, i32, f32, i32, i32, f32) {
                  cf.br ^bb1 loc(#loc16)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc17)
                ^bb2:  // pred: ^bb1
                  %9:6 = scf.if %true -> (f32, i32, f32, i32, i32, f32) {
                    %10:6 = scf.execute_region -> (f32, i32, f32, i32, i32, f32) {
                      %11 = scf.if %true -> (i32) {
                        %13 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %13 : i32 loc(#loc)
                      } else {
                        scf.yield %arg6 : i32 loc(#loc)
                      } loc(#loc)
                      %12:6 = scf.while (%arg10 = %arg3, %arg11 = %arg4, %arg12 = %arg5, %arg13 = %11, %arg14 = %5, %arg15 = %4) : (f32, i32, f32, i32, i32, f32) -> (f32, i32, f32, i32, i32, f32) {
                        %13 = arith.cmpi slt, %arg13, %c5_i32 : i32 loc(#loc18)
                        scf.condition(%13) %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : f32, i32, f32, i32, i32, f32 loc(#loc19)
                      } do {
                      ^bb0(%arg10: f32 loc("kmeans.cpp":25:5), %arg11: i32 loc("kmeans.cpp":24:18), %arg12: f32 loc("kmeans.cpp":22:4), %arg13: i32 loc("kmeans.cpp":21:16), %arg14: i32 loc("kmeans.cpp":18:3), %arg15: f32 loc("kmeans.cpp":17:3)):
                        %13 = scf.if %true -> (f32) {
                          %17 = scf.execute_region -> f32 {
                            %18 = scf.if %true -> (f32) {
                              %19 = scf.execute_region -> f32 {
                                scf.yield %cst : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19 : f32 loc(#loc)
                            } else {
                              scf.yield %arg12 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : f32 loc(#loc)
                        } else {
                          scf.yield %arg12 : f32 loc(#loc)
                        } loc(#loc)
                        %14:3 = scf.if %true -> (f32, i32, f32) {
                          %17:3 = scf.execute_region -> (f32, i32, f32) {
                            cf.br ^bb1 loc(#loc24)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc25)
                          ^bb2:  // pred: ^bb1
                            %18:3 = scf.if %true -> (f32, i32, f32) {
                              %19:3 = scf.execute_region -> (f32, i32, f32) {
                                %20 = scf.if %true -> (i32) {
                                  %22 = scf.execute_region -> i32 {
                                    scf.yield %c0_i32 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg11 : i32 loc(#loc)
                                } loc(#loc)
                                %21:3 = scf.while (%arg16 = %arg10, %arg17 = %20, %arg18 = %13) : (f32, i32, f32) -> (f32, i32, f32) {
                                  %22 = arith.cmpi slt, %arg17, %c34_i32 : i32 loc(#loc26)
                                  scf.condition(%22) %arg16, %arg17, %arg18 : f32, i32, f32 loc(#loc27)
                                } do {
                                ^bb0(%arg16: f32 loc("kmeans.cpp":25:5), %arg17: i32 loc("kmeans.cpp":24:18), %arg18: f32 loc("kmeans.cpp":22:4)):
                                  %22 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          %28 = arith.muli %arg9, %c34_i32 : i32 loc(#loc28)
                                          %29 = arith.addi %28, %arg17 : i32 loc(#loc29)
                                          %30 = arith.index_cast %29 : i32 to index loc(#loc30)
                                          %31 = "polygeist.subindex"(%arg0, %30) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc31)
                                          %32 = affine.load %31[0] : memref<?xf32> loc(#loc31)
                                          %33 = arith.muli %arg13, %c34_i32 : i32 loc(#loc32)
                                          %34 = arith.addi %33, %arg17 : i32 loc(#loc33)
                                          %35 = arith.index_cast %34 : i32 to index loc(#loc34)
                                          %36 = "polygeist.subindex"(%arg1, %35) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc35)
                                          %37 = affine.load %36[0] : memref<?xf32> loc(#loc35)
                                          %38 = arith.subf %32, %37 : f32 loc(#loc36)
                                          scf.yield %38 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg16 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg16 : f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = arith.mulf %22, %22 : f32 loc(#loc37)
                                      %27 = arith.addf %arg18, %26 : f32 loc(#loc38)
                                      scf.yield %27 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg18 : f32 loc(#loc)
                                  } loc(#loc)
                                  %24 = scf.if %true -> (i32) {
                                    %25 = scf.execute_region -> i32 {
                                      %26 = arith.addi %arg17, %c1_i32 : i32 loc(#loc2)
                                      scf.yield %26 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg17 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22, %24, %23 : f32, i32, f32 loc(#loc27)
                                } loc(#loc3)
                                scf.yield %21#0, %21#1, %21#2 : f32, i32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19#0, %19#1, %19#2 : f32, i32, f32 loc(#loc)
                            } else {
                              scf.yield %arg10, %arg11, %13 : f32, i32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18#0, %18#1, %18#2 : f32, i32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17#0, %17#1, %17#2 : f32, i32, f32 loc(#loc)
                        } else {
                          scf.yield %arg10, %arg11, %13 : f32, i32, f32 loc(#loc)
                        } loc(#loc)
                        %15:2 = scf.if %true -> (i32, f32) {
                          %17:2 = scf.execute_region -> (i32, f32) {
                            %18:2 = scf.if %true -> (i32, f32) {
                              %19:2 = scf.execute_region -> (i32, f32) {
                                %20 = arith.cmpf olt, %14#2, %arg15 : f32 loc(#loc39)
                                %21:2 = scf.if %20 -> (i32, f32) {
                                  %22 = scf.if %true -> (f32) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %14#2 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg15 : f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (i32) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %arg13 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %23, %22 : i32, f32 loc(#loc40)
                                } else {
                                  scf.yield %arg14, %arg15 : i32, f32 loc(#loc40)
                                } loc(#loc40)
                                scf.yield %21#0, %21#1 : i32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19#0, %19#1 : i32, f32 loc(#loc)
                            } else {
                              scf.yield %arg14, %arg15 : i32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18#0, %18#1 : i32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17#0, %17#1 : i32, f32 loc(#loc)
                        } else {
                          scf.yield %arg14, %arg15 : i32, f32 loc(#loc)
                        } loc(#loc)
                        %16 = scf.if %true -> (i32) {
                          %17 = scf.execute_region -> i32 {
                            %18 = arith.addi %arg13, %c1_i32 : i32 loc(#loc41)
                            scf.yield %18 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : i32 loc(#loc)
                        } else {
                          scf.yield %arg13 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14#0, %14#1, %14#2, %16, %15#0, %15#1 : f32, i32, f32, i32, i32, f32 loc(#loc19)
                      } loc(#loc5)
                      scf.yield %12#0, %12#1, %12#2, %12#3, %12#4, %12#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10#0, %10#1, %10#2, %10#3, %10#4, %10#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg5, %arg6, %5, %4 : f32, i32, f32, i32, i32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9#0, %9#1, %9#2, %9#3, %9#4, %9#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %8#0, %8#1, %8#2, %8#3, %8#4, %8#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg5, %arg6, %5, %4 : f32, i32, f32, i32, i32, f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = arith.index_cast %arg9 : i32 to index loc(#loc42)
                  %9 = "polygeist.subindex"(%arg2, %8) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc43)
                  affine.store %6#4, %9[0] : memref<?xi32> loc(#loc44)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg9, %c1_i32 : i32 loc(#loc45)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg9 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6#0, %6#1, %6#2, %6#3, %6#4, %6#5, %7 : f32, i32, f32, i32, i32, f32, i32 loc(#loc15)
            } loc(#loc7)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc46)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("kmeans.cpp":24:45)
#loc3 = loc("./kmeans.h":15:19)
#loc4 = loc("kmeans.cpp":22:17)
#loc5 = loc("./kmeans.h":16:19)
#loc6 = loc("./kmeans.h":12:17)
#loc8 = loc("kmeans.cpp":16:33)
#loc9 = loc("kmeans.cpp":4:1)
#loc12 = loc("kmeans.cpp":16:1)
#loc13 = loc("kmeans.cpp":16:5)
#loc14 = loc("kmeans.cpp":16:38)
#loc15 = loc("kmeans.cpp":16:20)
#loc16 = loc("kmeans.cpp":21:1)
#loc17 = loc("kmeans.cpp":21:6)
#loc18 = loc("kmeans.cpp":21:29)
#loc19 = loc("kmeans.cpp":21:11)
#loc24 = loc("kmeans.cpp":24:1)
#loc25 = loc("kmeans.cpp":24:7)
#loc26 = loc("kmeans.cpp":24:31)
#loc27 = loc("kmeans.cpp":24:13)
#loc28 = loc("kmeans.cpp":25:36)
#loc29 = loc("kmeans.cpp":25:40)
#loc30 = loc("kmeans.cpp":25:43)
#loc31 = loc("kmeans.cpp":25:18)
#loc32 = loc("kmeans.cpp":25:66)
#loc33 = loc("kmeans.cpp":25:70)
#loc34 = loc("kmeans.cpp":25:73)
#loc35 = loc("kmeans.cpp":25:47)
#loc36 = loc("kmeans.cpp":25:45)
#loc37 = loc("kmeans.cpp":26:18)
#loc38 = loc("kmeans.cpp":26:10)
#loc39 = loc("kmeans.cpp":28:13)
#loc40 = loc("kmeans.cpp":28:4)
#loc41 = loc("kmeans.cpp":21:43)
#loc42 = loc("kmeans.cpp":34:15)
#loc43 = loc("kmeans.cpp":34:3)
#loc44 = loc("kmeans.cpp":34:17)
#loc45 = loc("kmeans.cpp":16:50)
#loc46 = loc("kmeans.cpp":36:1)
