#loc1 = loc("stencil.c":10:6)
#loc6 = loc("./stencil.h":15:18)
#loc7 = loc("./stencil.h":14:18)
#loc10 = loc("./stencil.h":17:14)
#loc33 = loc("stencil.c":21:49)
#loc52 = loc("stencil.c":27:49)
#loc57 = loc("stencil.c":28:38)
#loc71 = loc("stencil.c":36:49)
#loc76 = loc("stencil.c":37:36)
#loc80 = loc("stencil.c":38:51)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @stencil3d(%arg0: memref<2xi32> loc("stencil.c":10:6), %arg1: memref<16384xi32> loc("stencil.c":10:6), %arg2: memref<16384xi32> loc("stencil.c":10:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c992_i32 = arith.constant 992 : i32 loc(#loc3)
    %c15_i32 = arith.constant 15 : i32 loc(#loc4)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc5)
    %c16_i32 = arith.constant 16 : i32 loc(#loc6)
    %c32_i32 = arith.constant 32 : i32 loc(#loc7)
    %c0_i32 = arith.constant 0 : i32 loc(#loc8)
    %true = arith.constant true loc(#loc9)
    %c1 = arith.constant 1 : index loc(#loc)
    %c0 = arith.constant 0 : index loc(#loc10)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc10)
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
        cf.br ^bb1 loc(#loc11)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc12)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
              %2 = arith.cmpi slt, %arg3, %c32_i32 : i32 loc(#loc13)
              scf.condition(%2) %arg3 : i32 loc(#loc14)
            } do {
            ^bb0(%arg3: i32 loc("./stencil.h":14:18)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc15)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc16)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg4, %c16_i32 : i32 loc(#loc17)
                        scf.condition(%4) %arg4 : i32 loc(#loc18)
                      } do {
                      ^bb0(%arg4: i32 loc("./stencil.h":15:18)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg3, %c16_i32 : i32 loc(#loc19)
                            %6 = arith.addi %arg4, %5 : i32 loc(#loc20)
                            %7 = arith.index_cast %6 : i32 to index loc(#loc21)
                            %8 = "polygeist.subindex"(%arg2, %7) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc22)
                            %9 = "polygeist.subindex"(%arg1, %7) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc23)
                            %10 = affine.load %9[0] : memref<?xi32> loc(#loc23)
                            affine.store %10, %8[0] : memref<?xi32> loc(#loc24)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.addi %arg3, %c992_i32 : i32 loc(#loc25)
                            %6 = arith.muli %5, %c16_i32 : i32 loc(#loc19)
                            %7 = arith.addi %arg4, %6 : i32 loc(#loc20)
                            %8 = arith.index_cast %7 : i32 to index loc(#loc26)
                            %9 = "polygeist.subindex"(%arg2, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc27)
                            %10 = "polygeist.subindex"(%arg1, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc28)
                            %11 = affine.load %10[0] : memref<?xi32> loc(#loc28)
                            affine.store %11, %9[0] : memref<?xi32> loc(#loc29)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg4, %c1_i32 : i32 loc(#loc2)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc18)
                      } loc(#loc6)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = arith.addi %arg3, %c1_i32 : i32 loc(#loc30)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2 : i32 loc(#loc14)
            } loc(#loc7)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc31)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc32)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.while (%arg3 = %c1_i32) : (i32) -> i32 {
              %2 = arith.cmpi slt, %arg3, %c31_i32 : i32 loc(#loc34)
              scf.condition(%2) %arg3 : i32 loc(#loc35)
            } do {
            ^bb0(%arg3: i32 loc("stencil.c":21:49)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc36)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc37)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg4, %c16_i32 : i32 loc(#loc38)
                        scf.condition(%4) %arg4 : i32 loc(#loc39)
                      } do {
                      ^bb0(%arg4: i32 loc("./stencil.h":15:18)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg3, %c32_i32 : i32 loc(#loc3)
                            %6 = arith.muli %5, %c16_i32 : i32 loc(#loc19)
                            %7 = arith.addi %arg4, %6 : i32 loc(#loc20)
                            %8 = arith.index_cast %7 : i32 to index loc(#loc40)
                            %9 = "polygeist.subindex"(%arg2, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc41)
                            %10 = "polygeist.subindex"(%arg1, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc42)
                            %11 = affine.load %10[0] : memref<?xi32> loc(#loc42)
                            affine.store %11, %9[0] : memref<?xi32> loc(#loc43)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg3, %c32_i32 : i32 loc(#loc3)
                            %6 = arith.addi %5, %c31_i32 : i32 loc(#loc25)
                            %7 = arith.muli %6, %c16_i32 : i32 loc(#loc19)
                            %8 = arith.addi %arg4, %7 : i32 loc(#loc20)
                            %9 = arith.index_cast %8 : i32 to index loc(#loc44)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc45)
                            %11 = "polygeist.subindex"(%arg1, %9) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc46)
                            %12 = affine.load %11[0] : memref<?xi32> loc(#loc46)
                            affine.store %12, %10[0] : memref<?xi32> loc(#loc47)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg4, %c1_i32 : i32 loc(#loc48)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc39)
                      } loc(#loc6)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = arith.addi %arg3, %c1_i32 : i32 loc(#loc49)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2 : i32 loc(#loc35)
            } loc(#loc33)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc50)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc51)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.while (%arg3 = %c1_i32) : (i32) -> i32 {
              %2 = arith.cmpi slt, %arg3, %c31_i32 : i32 loc(#loc53)
              scf.condition(%2) %arg3 : i32 loc(#loc54)
            } do {
            ^bb0(%arg3: i32 loc("stencil.c":27:49)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc55)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc56)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg4 = %c1_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg4, %c31_i32 : i32 loc(#loc57)
                        scf.condition(%4) %arg4 : i32 loc(#loc58)
                      } do {
                      ^bb0(%arg4: i32 loc("stencil.c":28:38)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg3, %c32_i32 : i32 loc(#loc3)
                            %6 = arith.addi %arg4, %5 : i32 loc(#loc25)
                            %7 = arith.muli %6, %c16_i32 : i32 loc(#loc19)
                            %8 = arith.index_cast %7 : i32 to index loc(#loc59)
                            %9 = "polygeist.subindex"(%arg2, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc60)
                            %10 = "polygeist.subindex"(%arg1, %8) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc61)
                            %11 = affine.load %10[0] : memref<?xi32> loc(#loc61)
                            affine.store %11, %9[0] : memref<?xi32> loc(#loc62)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg3, %c32_i32 : i32 loc(#loc3)
                            %6 = arith.addi %arg4, %5 : i32 loc(#loc25)
                            %7 = arith.muli %6, %c16_i32 : i32 loc(#loc19)
                            %8 = arith.addi %7, %c15_i32 : i32 loc(#loc20)
                            %9 = arith.index_cast %8 : i32 to index loc(#loc63)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc64)
                            %11 = "polygeist.subindex"(%arg1, %9) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc65)
                            %12 = affine.load %11[0] : memref<?xi32> loc(#loc65)
                            affine.store %12, %10[0] : memref<?xi32> loc(#loc66)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg4, %c1_i32 : i32 loc(#loc67)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc58)
                      } loc(#loc57)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = arith.addi %arg3, %c1_i32 : i32 loc(#loc68)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2 : i32 loc(#loc54)
            } loc(#loc52)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc69)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc70)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1:5 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %0, %arg7 = %c1_i32) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32) {
              %2 = arith.cmpi slt, %arg7, %c31_i32 : i32 loc(#loc72)
              scf.condition(%2) %arg3, %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i32, i32 loc(#loc73)
            } do {
            ^bb0(%arg3: i32 loc("stencil.c":36:49), %arg4: i32 loc("stencil.c":36:49), %arg5: i32 loc("stencil.c":36:49), %arg6: i32 loc("stencil.c":36:49), %arg7: i32 loc("stencil.c":36:49)):
              %2:4 = scf.if %true -> (i32, i32, i32, i32) {
                %4:4 = scf.execute_region -> (i32, i32, i32, i32) {
                  cf.br ^bb1 loc(#loc74)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc75)
                ^bb2:  // pred: ^bb1
                  %5:4 = scf.if %true -> (i32, i32, i32, i32) {
                    %6:4 = scf.execute_region -> (i32, i32, i32, i32) {
                      %7:5 = scf.while (%arg8 = %arg3, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %c1_i32) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32) {
                        %8 = arith.cmpi slt, %arg12, %c31_i32 : i32 loc(#loc76)
                        scf.condition(%8) %arg8, %arg9, %arg10, %arg11, %arg12 : i32, i32, i32, i32, i32 loc(#loc77)
                      } do {
                      ^bb0(%arg8: i32 loc("./stencil.h":17:14), %arg9: i32 loc("./stencil.h":17:14), %arg10: i32 loc("./stencil.h":17:14), %arg11: i32 loc("./stencil.h":17:14), %arg12: i32 loc("stencil.c":37:36)):
                        %8:4 = scf.if %true -> (i32, i32, i32, i32) {
                          %10:4 = scf.execute_region -> (i32, i32, i32, i32) {
                            cf.br ^bb1 loc(#loc78)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc79)
                          ^bb2:  // pred: ^bb1
                            %11:4 = scf.if %true -> (i32, i32, i32, i32) {
                              %12:4 = scf.execute_region -> (i32, i32, i32, i32) {
                                %13:5 = scf.while (%arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10, %arg16 = %arg11, %arg17 = %c1_i32) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32) {
                                  %14 = arith.cmpi slt, %arg17, %c15_i32 : i32 loc(#loc81)
                                  scf.condition(%14) %arg13, %arg14, %arg15, %arg16, %arg17 : i32, i32, i32, i32, i32 loc(#loc82)
                                } do {
                                ^bb0(%arg13: i32 loc("./stencil.h":17:14), %arg14: i32 loc("./stencil.h":17:14), %arg15: i32 loc("./stencil.h":17:14), %arg16: i32 loc("./stencil.h":17:14), %arg17: i32 loc("stencil.c":38:51)):
                                  %14 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = arith.muli %arg7, %c32_i32 : i32 loc(#loc3)
                                      %21 = arith.addi %arg12, %20 : i32 loc(#loc25)
                                      %22 = arith.muli %21, %c16_i32 : i32 loc(#loc19)
                                      %23 = arith.addi %arg17, %22 : i32 loc(#loc20)
                                      %24 = arith.index_cast %23 : i32 to index loc(#loc83)
                                      %25 = "polygeist.subindex"(%arg1, %24) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc84)
                                      %26 = affine.load %25[0] : memref<?xi32> loc(#loc84)
                                      scf.yield %26 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg16 : i32 loc(#loc)
                                  } loc(#loc)
                                  %15 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = arith.addi %arg7, %c1_i32 : i32 loc(#loc85)
                                      %21 = arith.muli %20, %c32_i32 : i32 loc(#loc3)
                                      %22 = arith.addi %arg12, %21 : i32 loc(#loc25)
                                      %23 = arith.muli %22, %c16_i32 : i32 loc(#loc19)
                                      %24 = arith.addi %arg17, %23 : i32 loc(#loc20)
                                      %25 = arith.index_cast %24 : i32 to index loc(#loc86)
                                      %26 = "polygeist.subindex"(%arg1, %25) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc87)
                                      %27 = affine.load %26[0] : memref<?xi32> loc(#loc87)
                                      %28 = arith.addi %arg7, %c-1_i32 : i32 loc(#loc88)
                                      %29 = arith.muli %28, %c32_i32 : i32 loc(#loc3)
                                      %30 = arith.addi %arg12, %29 : i32 loc(#loc25)
                                      %31 = arith.muli %30, %c16_i32 : i32 loc(#loc19)
                                      %32 = arith.addi %arg17, %31 : i32 loc(#loc20)
                                      %33 = arith.index_cast %32 : i32 to index loc(#loc89)
                                      %34 = "polygeist.subindex"(%arg1, %33) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc90)
                                      %35 = affine.load %34[0] : memref<?xi32> loc(#loc90)
                                      %36 = arith.addi %27, %35 : i32 loc(#loc91)
                                      %37 = arith.addi %arg12, %c1_i32 : i32 loc(#loc92)
                                      %38 = arith.muli %arg7, %c32_i32 : i32 loc(#loc3)
                                      %39 = arith.addi %37, %38 : i32 loc(#loc25)
                                      %40 = arith.muli %39, %c16_i32 : i32 loc(#loc19)
                                      %41 = arith.addi %arg17, %40 : i32 loc(#loc20)
                                      %42 = arith.index_cast %41 : i32 to index loc(#loc93)
                                      %43 = "polygeist.subindex"(%arg1, %42) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc94)
                                      %44 = affine.load %43[0] : memref<?xi32> loc(#loc94)
                                      %45 = arith.addi %36, %44 : i32 loc(#loc95)
                                      %46 = arith.addi %arg12, %c-1_i32 : i32 loc(#loc96)
                                      %47 = arith.addi %46, %38 : i32 loc(#loc25)
                                      %48 = arith.muli %47, %c16_i32 : i32 loc(#loc19)
                                      %49 = arith.addi %arg17, %48 : i32 loc(#loc20)
                                      %50 = arith.index_cast %49 : i32 to index loc(#loc97)
                                      %51 = "polygeist.subindex"(%arg1, %50) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc98)
                                      %52 = affine.load %51[0] : memref<?xi32> loc(#loc98)
                                      %53 = arith.addi %45, %52 : i32 loc(#loc99)
                                      %54 = arith.addi %arg17, %c1_i32 : i32 loc(#loc100)
                                      %55 = arith.addi %arg12, %38 : i32 loc(#loc25)
                                      %56 = arith.muli %55, %c16_i32 : i32 loc(#loc19)
                                      %57 = arith.addi %54, %56 : i32 loc(#loc20)
                                      %58 = arith.index_cast %57 : i32 to index loc(#loc101)
                                      %59 = "polygeist.subindex"(%arg1, %58) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc102)
                                      %60 = affine.load %59[0] : memref<?xi32> loc(#loc102)
                                      %61 = arith.addi %53, %60 : i32 loc(#loc103)
                                      %62 = arith.addi %arg17, %c-1_i32 : i32 loc(#loc104)
                                      %63 = arith.addi %62, %56 : i32 loc(#loc20)
                                      %64 = arith.index_cast %63 : i32 to index loc(#loc105)
                                      %65 = "polygeist.subindex"(%arg1, %64) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc106)
                                      %66 = affine.load %65[0] : memref<?xi32> loc(#loc106)
                                      %67 = arith.addi %61, %66 : i32 loc(#loc107)
                                      scf.yield %67 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg15 : i32 loc(#loc)
                                  } loc(#loc)
                                  %16 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = "polygeist.subindex"(%arg0, %c0) : (memref<2xi32>, index) -> memref<?xi32> loc(#loc108)
                                      %21 = affine.load %20[0] : memref<?xi32> loc(#loc108)
                                      %22 = arith.muli %14, %21 : i32 loc(#loc109)
                                      scf.yield %22 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : i32 loc(#loc)
                                  } loc(#loc)
                                  %17 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = "polygeist.subindex"(%arg0, %c1) : (memref<2xi32>, index) -> memref<?xi32> loc(#loc110)
                                      %21 = affine.load %20[0] : memref<?xi32> loc(#loc110)
                                      %22 = arith.muli %15, %21 : i32 loc(#loc111)
                                      scf.yield %22 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.if %true {
                                    scf.execute_region {
                                      %19 = arith.muli %arg7, %c32_i32 : i32 loc(#loc3)
                                      %20 = arith.addi %arg12, %19 : i32 loc(#loc25)
                                      %21 = arith.muli %20, %c16_i32 : i32 loc(#loc19)
                                      %22 = arith.addi %arg17, %21 : i32 loc(#loc20)
                                      %23 = arith.index_cast %22 : i32 to index loc(#loc112)
                                      %24 = "polygeist.subindex"(%arg2, %23) : (memref<16384xi32>, index) -> memref<?xi32> loc(#loc113)
                                      %25 = arith.addi %16, %17 : i32 loc(#loc114)
                                      affine.store %25, %24[0] : memref<?xi32> loc(#loc115)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  %18 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = arith.addi %arg17, %c1_i32 : i32 loc(#loc116)
                                      scf.yield %20 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg17 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %17, %16, %15, %14, %18 : i32, i32, i32, i32, i32 loc(#loc82)
                                } loc(#loc80)
                                scf.yield %13#0, %13#1, %13#2, %13#3 : i32, i32, i32, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %12#0, %12#1, %12#2, %12#3 : i32, i32, i32, i32 loc(#loc)
                            } else {
                              scf.yield %arg8, %arg9, %arg10, %arg11 : i32, i32, i32, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %11#0, %11#1, %11#2, %11#3 : i32, i32, i32, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %10#0, %10#1, %10#2, %10#3 : i32, i32, i32, i32 loc(#loc)
                        } else {
                          scf.yield %arg8, %arg9, %arg10, %arg11 : i32, i32, i32, i32 loc(#loc)
                        } loc(#loc)
                        %9 = scf.if %true -> (i32) {
                          %10 = scf.execute_region -> i32 {
                            %11 = arith.addi %arg12, %c1_i32 : i32 loc(#loc117)
                            scf.yield %11 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %10 : i32 loc(#loc)
                        } else {
                          scf.yield %arg12 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %8#0, %8#1, %8#2, %8#3, %9 : i32, i32, i32, i32, i32 loc(#loc77)
                      } loc(#loc76)
                      scf.yield %7#0, %7#1, %7#2, %7#3 : i32, i32, i32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %6#0, %6#1, %6#2, %6#3 : i32, i32, i32, i32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg5, %arg6 : i32, i32, i32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %5#0, %5#1, %5#2, %5#3 : i32, i32, i32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %4#0, %4#1, %4#2, %4#3 : i32, i32, i32, i32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg5, %arg6 : i32, i32, i32, i32 loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg7, %c1_i32 : i32 loc(#loc118)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg7 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2#0, %2#1, %2#2, %2#3, %3 : i32, i32, i32, i32, i32 loc(#loc73)
            } loc(#loc71)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc119)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("stencil.c":16:53)
#loc3 = loc("./stencil.h":22:75)
#loc4 = loc("stencil.c":30:50)
#loc5 = loc("stencil.c":18:59)
#loc8 = loc("stencil.c":15:33)
#loc9 = loc("stencil.c":10:1)
#loc11 = loc("stencil.c":15:1)
#loc12 = loc("stencil.c":15:8)
#loc13 = loc("stencil.c":15:37)
#loc14 = loc("stencil.c":15:27)
#loc15 = loc("stencil.c":16:1)
#loc16 = loc("stencil.c":16:12)
#loc17 = loc("stencil.c":16:41)
#loc18 = loc("stencil.c":16:31)
#loc19 = loc("./stencil.h":22:59)
#loc20 = loc("./stencil.h":22:49)
#loc21 = loc("stencil.c":17:50)
#loc22 = loc("stencil.c":17:13)
#loc23 = loc("stencil.c":17:54)
#loc24 = loc("stencil.c":17:52)
#loc25 = loc("./stencil.h":22:65)
#loc26 = loc("stencil.c":18:62)
#loc27 = loc("stencil.c":18:13)
#loc28 = loc("stencil.c":18:66)
#loc29 = loc("stencil.c":18:64)
#loc30 = loc("stencil.c":15:49)
#loc31 = loc("stencil.c":21:1)
#loc32 = loc("stencil.c":21:8)
#loc34 = loc("stencil.c":21:37)
#loc35 = loc("stencil.c":21:27)
#loc36 = loc("stencil.c":22:1)
#loc37 = loc("stencil.c":22:12)
#loc38 = loc("stencil.c":22:38)
#loc39 = loc("stencil.c":22:28)
#loc40 = loc("stencil.c":23:50)
#loc41 = loc("stencil.c":23:13)
#loc42 = loc("stencil.c":23:54)
#loc43 = loc("stencil.c":23:52)
#loc44 = loc("stencil.c":24:59)
#loc45 = loc("stencil.c":24:13)
#loc46 = loc("stencil.c":24:63)
#loc47 = loc("stencil.c":24:61)
#loc48 = loc("stencil.c":22:50)
#loc49 = loc("stencil.c":21:54)
#loc50 = loc("stencil.c":27:1)
#loc51 = loc("stencil.c":27:8)
#loc53 = loc("stencil.c":27:37)
#loc54 = loc("stencil.c":27:27)
#loc55 = loc("stencil.c":28:1)
#loc56 = loc("stencil.c":28:12)
#loc58 = loc("stencil.c":28:28)
#loc59 = loc("stencil.c":29:50)
#loc60 = loc("stencil.c":29:13)
#loc61 = loc("stencil.c":29:54)
#loc62 = loc("stencil.c":29:52)
#loc63 = loc("stencil.c":30:59)
#loc64 = loc("stencil.c":30:13)
#loc65 = loc("stencil.c":30:63)
#loc66 = loc("stencil.c":30:61)
#loc67 = loc("stencil.c":28:52)
#loc68 = loc("stencil.c":27:54)
#loc69 = loc("stencil.c":36:1)
#loc70 = loc("stencil.c":36:8)
#loc72 = loc("stencil.c":36:35)
#loc73 = loc("stencil.c":36:22)
#loc74 = loc("stencil.c":37:1)
#loc75 = loc("stencil.c":37:12)
#loc77 = loc("stencil.c":37:23)
#loc78 = loc("stencil.c":38:1)
#loc79 = loc("stencil.c":38:16)
#loc81 = loc("stencil.c":38:40)
#loc82 = loc("stencil.c":38:27)
#loc83 = loc("stencil.c":39:62)
#loc84 = loc("stencil.c":39:24)
#loc85 = loc("stencil.c":40:62)
#loc86 = loc("stencil.c":40:66)
#loc87 = loc("stencil.c":40:24)
#loc88 = loc("stencil.c":41:62)
#loc89 = loc("stencil.c":41:66)
#loc90 = loc("stencil.c":41:24)
#loc91 = loc("stencil.c":40:68)
#loc92 = loc("stencil.c":42:59)
#loc93 = loc("stencil.c":42:66)
#loc94 = loc("stencil.c":42:24)
#loc95 = loc("stencil.c":41:68)
#loc96 = loc("stencil.c":43:59)
#loc97 = loc("stencil.c":43:66)
#loc98 = loc("stencil.c":43:24)
#loc99 = loc("stencil.c":42:68)
#loc100 = loc("stencil.c":44:56)
#loc101 = loc("stencil.c":44:66)
#loc102 = loc("stencil.c":44:24)
#loc103 = loc("stencil.c":43:68)
#loc104 = loc("stencil.c":45:56)
#loc105 = loc("stencil.c":45:66)
#loc106 = loc("stencil.c":45:24)
#loc107 = loc("stencil.c":44:68)
#loc108 = loc("stencil.c":46:31)
#loc109 = loc("stencil.c":46:29)
#loc110 = loc("stencil.c":47:31)
#loc111 = loc("stencil.c":47:29)
#loc112 = loc("stencil.c":48:54)
#loc113 = loc("stencil.c":48:17)
#loc114 = loc("stencil.c":48:63)
#loc115 = loc("stencil.c":48:56)
#loc116 = loc("stencil.c":38:57)
#loc117 = loc("stencil.c":37:53)
#loc118 = loc("stencil.c":36:55)
#loc119 = loc("stencil.c":52:1)
