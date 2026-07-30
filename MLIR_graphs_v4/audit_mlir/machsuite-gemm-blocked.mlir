#loc1 = loc("gemm.c":10:6)
#loc3 = loc("./gemm.h":19:20)
#loc4 = loc("./gemm.h":16:18)
#loc7 = loc("./gemm.h":13:14)
#loc8 = loc("gemm.c":12:5)
#loc15 = loc("gemm.c":16:35)
#loc19 = loc("gemm.c":17:37)
#loc34 = loc("gemm.c":22:44)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bbgemm(%arg0: memref<4096xf64> loc("gemm.c":10:6), %arg1: memref<4096xf64> loc("gemm.c":10:6), %arg2: memref<4096xf64> loc("gemm.c":10:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc3)
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
            ^bb0(%arg3: f64 loc("./gemm.h":16:18), %arg4: f64 loc("./gemm.h":16:18), %arg5: i32 loc("./gemm.h":16:18), %arg6: i32 loc("./gemm.h":16:18), %arg7: i32 loc("./gemm.h":16:18)):
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
                      ^bb0(%arg8: f64 loc("./gemm.h":13:14), %arg9: f64 loc("./gemm.h":13:14), %arg10: i32 loc("gemm.c":12:5), %arg11: i32 loc("gemm.c":12:5), %arg12: i32 loc("gemm.c":16:35)):
                        %9:4 = scf.if %true -> (f64, f64, i32, i32) {
                          %11:4 = scf.execute_region -> (f64, f64, i32, i32) {
                            cf.br ^bb1 loc(#loc17)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc18)
                          ^bb2:  // pred: ^bb1
                            %12:4 = scf.if %true -> (f64, f64, i32, i32) {
                              %13:4 = scf.execute_region -> (f64, f64, i32, i32) {
                                %14:5 = scf.while (%arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10, %arg16 = %arg11, %arg17 = %c0_i32) : (f64, f64, i32, i32, i32) -> (f64, f64, i32, i32, i32) {
                                  %15 = arith.cmpi slt, %arg17, %c64_i32 : i32 loc(#loc19)
                                  scf.condition(%15) %arg13, %arg14, %arg15, %arg16, %arg17 : f64, f64, i32, i32, i32 loc(#loc20)
                                } do {
                                ^bb0(%arg13: f64 loc("./gemm.h":13:14), %arg14: f64 loc("./gemm.h":13:14), %arg15: i32 loc("gemm.c":12:5), %arg16: i32 loc("gemm.c":12:5), %arg17: i32 loc("gemm.c":17:37)):
                                  %15:4 = scf.if %true -> (f64, f64, i32, i32) {
                                    %17:4 = scf.execute_region -> (f64, f64, i32, i32) {
                                      cf.br ^bb1 loc(#loc21)
                                    ^bb1:  // pred: ^bb0
                                      cf.br ^bb2 loc(#loc22)
                                    ^bb2:  // pred: ^bb1
                                      %18:4 = scf.if %true -> (f64, f64, i32, i32) {
                                        %19:4 = scf.execute_region -> (f64, f64, i32, i32) {
                                          %20:5 = scf.while (%arg18 = %arg13, %arg19 = %arg14, %arg20 = %arg15, %arg21 = %arg16, %arg22 = %c0_i32) : (f64, f64, i32, i32, i32) -> (f64, f64, i32, i32, i32) {
                                            %21 = arith.cmpi slt, %arg22, %c8_i32 : i32 loc(#loc23)
                                            scf.condition(%21) %arg18, %arg19, %arg20, %arg21, %arg22 : f64, f64, i32, i32, i32 loc(#loc24)
                                          } do {
                                          ^bb0(%arg18: f64 loc("./gemm.h":13:14), %arg19: f64 loc("./gemm.h":13:14), %arg20: i32 loc("gemm.c":12:5), %arg21: i32 loc("gemm.c":12:5), %arg22: i32 loc("./gemm.h":19:20)):
                                            %21 = scf.if %true -> (i32) {
                                              %26 = scf.execute_region -> i32 {
                                                %27 = arith.muli %arg17, %c64_i32 : i32 loc(#loc25)
                                                scf.yield %27 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %26 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg21 : i32 loc(#loc)
                                            } loc(#loc)
                                            %22 = scf.if %true -> (i32) {
                                              %26 = scf.execute_region -> i32 {
                                                %27 = arith.addi %arg22, %arg12 : i32 loc(#loc26)
                                                %28 = arith.muli %27, %c64_i32 : i32 loc(#loc27)
                                                scf.yield %28 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %26 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg20 : i32 loc(#loc)
                                            } loc(#loc)
                                            %23 = scf.if %true -> (f64) {
                                              %26 = scf.execute_region -> f64 {
                                                %27 = arith.addi %21, %arg22 : i32 loc(#loc28)
                                                %28 = arith.addi %27, %arg12 : i32 loc(#loc29)
                                                %29 = arith.index_cast %28 : i32 to index loc(#loc30)
                                                %30 = "polygeist.subindex"(%arg0, %29) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc31)
                                                %31 = affine.load %30[0] : memref<?xf64> loc(#loc31)
                                                scf.yield %31 : f64 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %26 : f64 loc(#loc)
                                            } else {
                                              scf.yield %arg19 : f64 loc(#loc)
                                            } loc(#loc)
                                            %24 = scf.if %true -> (f64) {
                                              %26 = scf.execute_region -> f64 {
                                                cf.br ^bb1 loc(#loc32)
                                              ^bb1:  // pred: ^bb0
                                                cf.br ^bb2 loc(#loc33)
                                              ^bb2:  // pred: ^bb1
                                                %27 = scf.if %true -> (f64) {
                                                  %28 = scf.execute_region -> f64 {
                                                    %29:2 = scf.while (%arg23 = %arg18, %arg24 = %c0_i32) : (f64, i32) -> (f64, i32) {
                                                      %30 = arith.cmpi slt, %arg24, %c8_i32 : i32 loc(#loc34)
                                                      scf.condition(%30) %arg23, %arg24 : f64, i32 loc(#loc35)
                                                    } do {
                                                    ^bb0(%arg23: f64 loc("./gemm.h":13:14), %arg24: i32 loc("gemm.c":22:44)):
                                                      %30 = scf.if %true -> (f64) {
                                                        %32 = scf.execute_region -> f64 {
                                                          %33 = arith.addi %22, %arg24 : i32 loc(#loc36)
                                                          %34 = arith.addi %33, %arg7 : i32 loc(#loc37)
                                                          %35 = arith.index_cast %34 : i32 to index loc(#loc38)
                                                          %36 = "polygeist.subindex"(%arg1, %35) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc39)
                                                          %37 = affine.load %36[0] : memref<?xf64> loc(#loc39)
                                                          %38 = arith.mulf %23, %37 : f64 loc(#loc40)
                                                          scf.yield %38 : f64 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %32 : f64 loc(#loc)
                                                      } else {
                                                        scf.yield %arg23 : f64 loc(#loc)
                                                      } loc(#loc)
                                                      scf.if %true {
                                                        scf.execute_region {
                                                          %32 = arith.addi %21, %arg24 : i32 loc(#loc41)
                                                          %33 = arith.addi %32, %arg7 : i32 loc(#loc42)
                                                          %34 = arith.index_cast %33 : i32 to index loc(#loc43)
                                                          %35 = "polygeist.subindex"(%arg2, %34) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc44)
                                                          %36 = affine.load %35[0] : memref<?xf64> loc(#loc45)
                                                          %37 = arith.addf %36, %30 : f64 loc(#loc45)
                                                          affine.store %37, %35[0] : memref<?xf64> loc(#loc45)
                                                          scf.yield loc(#loc)
                                                        } loc(#loc)
                                                      } loc(#loc)
                                                      %31 = scf.if %true -> (i32) {
                                                        %32 = scf.execute_region -> i32 {
                                                          %33 = arith.addi %arg24, %c1_i32 : i32 loc(#loc2)
                                                          scf.yield %33 : i32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %32 : i32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg24 : i32 loc(#loc)
                                                      } loc(#loc)
                                                      scf.yield %30, %31 : f64, i32 loc(#loc35)
                                                    } loc(#loc34)
                                                    scf.yield %29#0 : f64 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %28 : f64 loc(#loc)
                                                } else {
                                                  scf.yield %arg18 : f64 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %27 : f64 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %26 : f64 loc(#loc)
                                            } else {
                                              scf.yield %arg18 : f64 loc(#loc)
                                            } loc(#loc)
                                            %25 = scf.if %true -> (i32) {
                                              %26 = scf.execute_region -> i32 {
                                                %27 = arith.addi %arg22, %c1_i32 : i32 loc(#loc46)
                                                scf.yield %27 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %26 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg22 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %24, %23, %22, %21, %25 : f64, f64, i32, i32, i32 loc(#loc24)
                                          } loc(#loc3)
                                          scf.yield %20#0, %20#1, %20#2, %20#3 : f64, f64, i32, i32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %19#0, %19#1, %19#2, %19#3 : f64, f64, i32, i32 loc(#loc)
                                      } else {
                                        scf.yield %arg13, %arg14, %arg15, %arg16 : f64, f64, i32, i32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %18#0, %18#1, %18#2, %18#3 : f64, f64, i32, i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %17#0, %17#1, %17#2, %17#3 : f64, f64, i32, i32 loc(#loc)
                                  } else {
                                    scf.yield %arg13, %arg14, %arg15, %arg16 : f64, f64, i32, i32 loc(#loc)
                                  } loc(#loc)
                                  %16 = scf.if %true -> (i32) {
                                    %17 = scf.execute_region -> i32 {
                                      %18 = arith.addi %arg17, %c1_i32 : i32 loc(#loc47)
                                      scf.yield %18 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %17 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg17 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %15#0, %15#1, %15#2, %15#3, %16 : f64, f64, i32, i32, i32 loc(#loc20)
                                } loc(#loc19)
                                scf.yield %14#0, %14#1, %14#2, %14#3 : f64, f64, i32, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %13#0, %13#1, %13#2, %13#3 : f64, f64, i32, i32 loc(#loc)
                            } else {
                              scf.yield %arg8, %arg9, %arg10, %arg11 : f64, f64, i32, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %12#0, %12#1, %12#2, %12#3 : f64, f64, i32, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11#0, %11#1, %11#2, %11#3 : f64, f64, i32, i32 loc(#loc)
                        } else {
                          scf.yield %arg8, %arg9, %arg10, %arg11 : f64, f64, i32, i32 loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (i32) {
                          %11 = scf.execute_region -> i32 {
                            %12 = arith.addi %arg12, %c8_i32 : i32 loc(#loc48)
                            scf.yield %12 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11 : i32 loc(#loc)
                        } else {
                          scf.yield %arg12 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %9#0, %9#1, %9#2, %9#3, %10 : f64, f64, i32, i32, i32 loc(#loc16)
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
                  %6 = arith.addi %arg7, %c8_i32 : i32 loc(#loc49)
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
    return loc(#loc50)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("gemm.c":22:58)
#loc5 = loc("gemm.c":15:25)
#loc6 = loc("gemm.c":10:1)
#loc9 = loc("gemm.c":15:1)
#loc10 = loc("gemm.c":15:8)
#loc11 = loc("gemm.c":15:31)
#loc12 = loc("gemm.c":15:15)
#loc13 = loc("gemm.c":16:1)
#loc14 = loc("gemm.c":16:12)
#loc16 = loc("gemm.c":16:19)
#loc17 = loc("gemm.c":17:1)
#loc18 = loc("gemm.c":17:16)
#loc20 = loc("gemm.c":17:22)
#loc21 = loc("gemm.c":18:1)
#loc22 = loc("gemm.c":18:20)
#loc23 = loc("gemm.c":18:40)
#loc24 = loc("gemm.c":18:26)
#loc25 = loc("gemm.c":19:31)
#loc26 = loc("gemm.c":20:33)
#loc27 = loc("gemm.c":20:39)
#loc28 = loc("gemm.c":21:39)
#loc29 = loc("gemm.c":21:43)
#loc30 = loc("gemm.c":21:47)
#loc31 = loc("gemm.c":21:30)
#loc32 = loc("gemm.c":22:1)
#loc33 = loc("gemm.c":22:24)
#loc35 = loc("gemm.c":22:30)
#loc36 = loc("gemm.c":23:49)
#loc37 = loc("gemm.c":23:53)
#loc38 = loc("gemm.c":23:57)
#loc39 = loc("gemm.c":23:40)
#loc40 = loc("gemm.c":23:38)
#loc41 = loc("gemm.c":24:36)
#loc42 = loc("gemm.c":24:40)
#loc43 = loc("gemm.c":24:44)
#loc44 = loc("gemm.c":24:25)
#loc45 = loc("gemm.c":24:46)
#loc46 = loc("gemm.c":18:54)
#loc47 = loc("gemm.c":17:49)
#loc48 = loc("gemm.c":16:50)
#loc49 = loc("gemm.c":15:46)
#loc50 = loc("gemm.c":30:1)
