#loc1 = loc("stencil.c":3:6)
#loc4 = loc("stencil.c":10:44)
#loc5 = loc("stencil.c":8:47)
#loc6 = loc("stencil.c":7:43)
#loc9 = loc("./stencil.h":11:14)
#loc24 = loc("stencil.c":11:47)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @stencil(%arg0: memref<8192xi32> loc("stencil.c":3:6), %arg1: memref<8192xi32> loc("stencil.c":3:6), %arg2: memref<9xi32> loc("stencil.c":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c64_i32 = arith.constant 64 : i32 loc(#loc3)
    %c3_i32 = arith.constant 3 : i32 loc(#loc4)
    %c62_i32 = arith.constant 62 : i32 loc(#loc5)
    %c126_i32 = arith.constant 126 : i32 loc(#loc6)
    %c0_i32 = arith.constant 0 : i32 loc(#loc7)
    %true = arith.constant true loc(#loc8)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc9)
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
        cf.br ^bb1 loc(#loc10)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc11)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1:3 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
              %2 = arith.cmpi slt, %arg5, %c126_i32 : i32 loc(#loc12)
              scf.condition(%2) %arg3, %arg4, %arg5 : i32, i32, i32 loc(#loc13)
            } do {
            ^bb0(%arg3: i32 loc("stencil.c":7:43), %arg4: i32 loc("stencil.c":7:43), %arg5: i32 loc("stencil.c":7:43)):
              %2:2 = scf.if %true -> (i32, i32) {
                %4:2 = scf.execute_region -> (i32, i32) {
                  cf.br ^bb1 loc(#loc14)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc15)
                ^bb2:  // pred: ^bb1
                  %5:2 = scf.if %true -> (i32, i32) {
                    %6:2 = scf.execute_region -> (i32, i32) {
                      %7:3 = scf.while (%arg6 = %arg3, %arg7 = %arg4, %arg8 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
                        %8 = arith.cmpi slt, %arg8, %c62_i32 : i32 loc(#loc16)
                        scf.condition(%8) %arg6, %arg7, %arg8 : i32, i32, i32 loc(#loc17)
                      } do {
                      ^bb0(%arg6: i32 loc("./stencil.h":11:14), %arg7: i32 loc("./stencil.h":11:14), %arg8: i32 loc("stencil.c":8:47)):
                        %8 = scf.if %true -> (i32) {
                          scf.execute_region {
                            scf.yield loc(#loc)
                          } loc(#loc)
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } else {
                          scf.yield %arg7 : i32 loc(#loc)
                        } loc(#loc)
                        %9:2 = scf.if %true -> (i32, i32) {
                          %11:2 = scf.execute_region -> (i32, i32) {
                            cf.br ^bb1 loc(#loc18)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc19)
                          ^bb2:  // pred: ^bb1
                            %12:2 = scf.if %true -> (i32, i32) {
                              %13:2 = scf.execute_region -> (i32, i32) {
                                %14:3 = scf.while (%arg9 = %arg6, %arg10 = %8, %arg11 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
                                  %15 = arith.cmpi slt, %arg11, %c3_i32 : i32 loc(#loc20)
                                  scf.condition(%15) %arg9, %arg10, %arg11 : i32, i32, i32 loc(#loc21)
                                } do {
                                ^bb0(%arg9: i32 loc("./stencil.h":11:14), %arg10: i32 loc("./stencil.h":11:14), %arg11: i32 loc("stencil.c":10:44)):
                                  %15:2 = scf.if %true -> (i32, i32) {
                                    %17:2 = scf.execute_region -> (i32, i32) {
                                      cf.br ^bb1 loc(#loc22)
                                    ^bb1:  // pred: ^bb0
                                      cf.br ^bb2 loc(#loc23)
                                    ^bb2:  // pred: ^bb1
                                      %18:2 = scf.if %true -> (i32, i32) {
                                        %19:2 = scf.execute_region -> (i32, i32) {
                                          %20:3 = scf.while (%arg12 = %arg9, %arg13 = %arg10, %arg14 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
                                            %21 = arith.cmpi slt, %arg14, %c3_i32 : i32 loc(#loc24)
                                            scf.condition(%21) %arg12, %arg13, %arg14 : i32, i32, i32 loc(#loc25)
                                          } do {
                                          ^bb0(%arg12: i32 loc("./stencil.h":11:14), %arg13: i32 loc("./stencil.h":11:14), %arg14: i32 loc("stencil.c":11:47)):
                                            %21 = scf.if %true -> (i32) {
                                              %24 = scf.execute_region -> i32 {
                                                %25 = arith.muli %arg11, %c3_i32 : i32 loc(#loc26)
                                                %26 = arith.addi %25, %arg14 : i32 loc(#loc27)
                                                %27 = arith.index_cast %26 : i32 to index loc(#loc28)
                                                %28 = "polygeist.subindex"(%arg2, %27) : (memref<9xi32>, index) -> memref<?xi32> loc(#loc29)
                                                %29 = affine.load %28[0] : memref<?xi32> loc(#loc29)
                                                %30 = arith.addi %arg5, %arg11 : i32 loc(#loc30)
                                                %31 = arith.muli %30, %c64_i32 : i32 loc(#loc31)
                                                %32 = arith.addi %31, %arg8 : i32 loc(#loc32)
                                                %33 = arith.addi %32, %arg14 : i32 loc(#loc33)
                                                %34 = arith.index_cast %33 : i32 to index loc(#loc34)
                                                %35 = "polygeist.subindex"(%arg0, %34) : (memref<8192xi32>, index) -> memref<?xi32> loc(#loc35)
                                                %36 = affine.load %35[0] : memref<?xi32> loc(#loc35)
                                                %37 = arith.muli %29, %36 : i32 loc(#loc36)
                                                scf.yield %37 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %24 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg12 : i32 loc(#loc)
                                            } loc(#loc)
                                            %22 = scf.if %true -> (i32) {
                                              %24 = scf.execute_region -> i32 {
                                                %25 = arith.addi %arg13, %21 : i32 loc(#loc37)
                                                scf.yield %25 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %24 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg13 : i32 loc(#loc)
                                            } loc(#loc)
                                            %23 = scf.if %true -> (i32) {
                                              %24 = scf.execute_region -> i32 {
                                                %25 = arith.addi %arg14, %c1_i32 : i32 loc(#loc2)
                                                scf.yield %25 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %24 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg14 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %21, %22, %23 : i32, i32, i32 loc(#loc25)
                                          } loc(#loc24)
                                          scf.yield %20#0, %20#1 : i32, i32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %19#0, %19#1 : i32, i32 loc(#loc)
                                      } else {
                                        scf.yield %arg9, %arg10 : i32, i32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %18#0, %18#1 : i32, i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %17#0, %17#1 : i32, i32 loc(#loc)
                                  } else {
                                    scf.yield %arg9, %arg10 : i32, i32 loc(#loc)
                                  } loc(#loc)
                                  %16 = scf.if %true -> (i32) {
                                    %17 = scf.execute_region -> i32 {
                                      %18 = arith.addi %arg11, %c1_i32 : i32 loc(#loc38)
                                      scf.yield %18 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %17 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg11 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %15#0, %15#1, %16 : i32, i32, i32 loc(#loc21)
                                } loc(#loc4)
                                scf.yield %14#0, %14#1 : i32, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %13#0, %13#1 : i32, i32 loc(#loc)
                            } else {
                              scf.yield %arg6, %8 : i32, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %12#0, %12#1 : i32, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11#0, %11#1 : i32, i32 loc(#loc)
                        } else {
                          scf.yield %arg6, %8 : i32, i32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %11 = arith.muli %arg5, %c64_i32 : i32 loc(#loc39)
                            %12 = arith.addi %11, %arg8 : i32 loc(#loc40)
                            %13 = arith.index_cast %12 : i32 to index loc(#loc41)
                            %14 = "polygeist.subindex"(%arg1, %13) : (memref<8192xi32>, index) -> memref<?xi32> loc(#loc42)
                            affine.store %9#1, %14[0] : memref<?xi32> loc(#loc43)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (i32) {
                          %11 = scf.execute_region -> i32 {
                            %12 = arith.addi %arg8, %c1_i32 : i32 loc(#loc44)
                            scf.yield %12 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11 : i32 loc(#loc)
                        } else {
                          scf.yield %arg8 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %9#0, %9#1, %10 : i32, i32, i32 loc(#loc17)
                      } loc(#loc5)
                      scf.yield %7#0, %7#1 : i32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %6#0, %6#1 : i32, i32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4 : i32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %5#0, %5#1 : i32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %4#0, %4#1 : i32, i32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4 : i32, i32 loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg5, %c1_i32 : i32 loc(#loc45)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg5 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2#0, %2#1, %3 : i32, i32, i32 loc(#loc13)
            } loc(#loc6)
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
#loc2 = loc("stencil.c":11:52)
#loc3 = loc("./stencil.h":6:18)
#loc7 = loc("stencil.c":7:30)
#loc8 = loc("stencil.c":3:1)
#loc10 = loc("stencil.c":7:1)
#loc11 = loc("stencil.c":7:8)
#loc12 = loc("stencil.c":7:34)
#loc13 = loc("stencil.c":7:23)
#loc14 = loc("stencil.c":8:1)
#loc15 = loc("stencil.c":8:12)
#loc16 = loc("stencil.c":8:38)
#loc17 = loc("stencil.c":8:27)
#loc18 = loc("stencil.c":10:1)
#loc19 = loc("stencil.c":10:16)
#loc20 = loc("stencil.c":10:43)
#loc21 = loc("stencil.c":10:31)
#loc22 = loc("stencil.c":11:1)
#loc23 = loc("stencil.c":11:20)
#loc25 = loc("stencil.c":11:35)
#loc26 = loc("stencil.c":12:36)
#loc27 = loc("stencil.c":12:39)
#loc28 = loc("stencil.c":12:43)
#loc29 = loc("stencil.c":12:27)
#loc30 = loc("stencil.c":12:54)
#loc31 = loc("stencil.c":12:58)
#loc32 = loc("stencil.c":12:68)
#loc33 = loc("stencil.c":12:71)
#loc34 = loc("stencil.c":12:74)
#loc35 = loc("stencil.c":12:47)
#loc36 = loc("stencil.c":12:45)
#loc37 = loc("stencil.c":13:26)
#loc38 = loc("stencil.c":10:48)
#loc39 = loc("stencil.c":16:19)
#loc40 = loc("stencil.c":16:30)
#loc41 = loc("stencil.c":16:33)
#loc42 = loc("stencil.c":16:13)
#loc43 = loc("stencil.c":16:35)
#loc44 = loc("stencil.c":8:52)
#loc45 = loc("stencil.c":7:48)
#loc46 = loc("stencil.c":19:1)
