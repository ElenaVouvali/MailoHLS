#loc1 = loc("viterbi.c":3:5)
#loc4 = loc("viterbi.c":50:42)
#loc6 = loc("./viterbi.h":22:19)
#loc11 = loc("viterbi.c":9:3)
#loc12 = loc("viterbi.c":8:3)
#loc16 = loc("viterbi.c":13:28)
#loc33 = loc("viterbi.c":19:39)
#loc36 = loc("viterbi.c":7:3)
#loc86 = loc("viterbi.c":40:27)
#loc112 = loc("viterbi.c":53:31)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @viterbi(%arg0: memref<140xi8> loc("viterbi.c":3:5), %arg1: memref<64xf64> loc("viterbi.c":3:5), %arg2: memref<4096xf64> loc("viterbi.c":3:5), %arg3: memref<4096xf64> loc("viterbi.c":3:5), %arg4: memref<140xi8> loc("viterbi.c":3:5)) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c139 = arith.constant 139 : index loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %c138_i32 = arith.constant 138 : i32 loc(#loc5)
    %c140_i32 = arith.constant 140 : i32 loc(#loc6)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c1_i8 = arith.constant 1 : i8 loc(#loc7)
    %c64_i32 = arith.constant 64 : i32 loc(#loc8)
    %c0_i8 = arith.constant 0 : i8 loc(#loc9)
    %true = arith.constant true loc(#loc10)
    %c0 = arith.constant 0 : index loc(#loc11)
    %0 = "polygeist.undef"() : () -> i8 loc(#loc11)
    %1 = "polygeist.undef"() : () -> f64 loc(#loc12)
    %2 = "polygeist.undef"() : () -> i32 loc(#loc13)
    %alloca = memref.alloca() : memref<140x64xf64> loc(#loc14)
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
        cf.br ^bb1 loc(#loc15)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %8 = scf.while (%arg5 = %c0_i8) : (i8) -> i8 {
              %9 = arith.extsi %arg5 : i8 to i32 loc(#loc16)
              %10 = arith.cmpi slt, %9, %c64_i32 : i32 loc(#loc17)
              scf.condition(%10) %arg5 : i8 loc(#loc18)
            } do {
            ^bb0(%arg5: i8 loc("viterbi.c":13:28)):
              %9 = arith.extsi %arg5 : i8 to i32 loc(#loc16)
              scf.if %true {
                scf.execute_region {
                  %11 = "polygeist.subindex"(%alloca, %c0) : (memref<140x64xf64>, index) -> memref<64xf64> loc(#loc19)
                  %12 = arith.index_cast %arg5 : i8 to index loc(#loc20)
                  %13 = "polygeist.subindex"(%11, %12) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc19)
                  %14 = "polygeist.subindex"(%arg1, %12) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc21)
                  %15 = affine.load %14[0] : memref<?xf64> loc(#loc21)
                  %16 = arith.muli %9, %c64_i32 : i32 loc(#loc22)
                  %17 = "polygeist.subindex"(%arg0, %c0) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc23)
                  %18 = affine.load %17[0] : memref<?xi8> loc(#loc23)
                  %19 = arith.extsi %18 : i8 to i32 loc(#loc23)
                  %20 = arith.addi %16, %19 : i32 loc(#loc24)
                  %21 = arith.index_cast %20 : i32 to index loc(#loc25)
                  %22 = "polygeist.subindex"(%arg3, %21) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc26)
                  %23 = affine.load %22[0] : memref<?xf64> loc(#loc26)
                  %24 = arith.addf %15, %23 : f64 loc(#loc27)
                  affine.store %24, %13[0] : memref<?xf64> loc(#loc28)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %10 = scf.if %true -> (i8) {
                %11 = scf.execute_region -> i8 {
                  %12 = arith.addi %arg5, %c1_i8 : i8 loc(#loc7)
                  scf.yield %12 : i8 loc(#loc)
                } loc(#loc)
                scf.yield %11 : i8 loc(#loc)
              } else {
                scf.yield %arg5 : i8 loc(#loc)
              } loc(#loc)
              scf.yield %10 : i8 loc(#loc18)
            } loc(#loc16)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    %3:2 = scf.if %true -> (f64, f64) {
      %8:2 = scf.execute_region -> (f64, f64) {
        cf.br ^bb1 loc(#loc29)
      ^bb1:  // pred: ^bb0
        %9:2 = scf.if %true -> (f64, f64) {
          %10:2 = scf.execute_region -> (f64, f64) {
            %11:4 = scf.while (%arg5 = %1, %arg6 = %1, %arg7 = %0, %arg8 = %c1_i32) : (f64, f64, i8, i32) -> (f64, f64, i8, i32) {
              %12 = arith.cmpi slt, %arg8, %c140_i32 : i32 loc(#loc30)
              scf.condition(%12) %arg5, %arg6, %arg7, %arg8 : f64, f64, i8, i32 loc(#loc31)
            } do {
            ^bb0(%arg5: f64 loc("viterbi.c":8:3), %arg6: f64 loc("viterbi.c":8:3), %arg7: i8 loc("./viterbi.h":22:19), %arg8: i32 loc("./viterbi.h":22:19)):
              %12:3 = scf.if %true -> (f64, f64, i8) {
                %14:3 = scf.execute_region -> (f64, f64, i8) {
                  cf.br ^bb1 loc(#loc32)
                ^bb1:  // pred: ^bb0
                  %15:3 = scf.if %true -> (f64, f64, i8) {
                    %16:3 = scf.execute_region -> (f64, f64, i8) {
                      %17:4 = scf.while (%arg9 = %arg5, %arg10 = %arg6, %arg11 = %c0_i8, %arg12 = %arg7) : (f64, f64, i8, i8) -> (f64, f64, i8, i8) {
                        %18 = arith.extsi %arg11 : i8 to i32 loc(#loc33)
                        %19 = arith.cmpi slt, %18, %c64_i32 : i32 loc(#loc34)
                        scf.condition(%19) %arg9, %arg10, %arg12, %arg11 : f64, f64, i8, i8 loc(#loc35)
                      } do {
                      ^bb0(%arg9: f64 loc("viterbi.c":8:3), %arg10: f64 loc("viterbi.c":8:3), %arg11: i8 loc("viterbi.c":7:3), %arg12: i8 loc("viterbi.c":19:39)):
                        %18 = arith.extsi %arg12 : i8 to i32 loc(#loc33)
                        %19 = scf.if %true -> (i8) {
                          scf.execute_region {
                            scf.yield loc(#loc)
                          } loc(#loc)
                          scf.yield %c0_i8 : i8 loc(#loc)
                        } else {
                          scf.yield %arg11 : i8 loc(#loc)
                        } loc(#loc)
                        %20 = scf.if %true -> (f64) {
                          %23 = scf.execute_region -> f64 {
                            %24 = arith.addi %arg8, %c-1_i32 : i32 loc(#loc37)
                            %25 = arith.index_cast %24 : i32 to index loc(#loc38)
                            %26 = "polygeist.subindex"(%alloca, %25) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc39)
                            %27 = "polygeist.subindex"(%26, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc39)
                            %28 = arith.index_cast %19 : i8 to index loc(#loc40)
                            %29 = "polygeist.subindex"(%27, %28) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc39)
                            %30 = affine.load %29[0] : memref<?xf64> loc(#loc39)
                            %31 = arith.extsi %19 : i8 to i32 loc(#loc41)
                            %32 = arith.muli %31, %c64_i32 : i32 loc(#loc42)
                            %33 = arith.addi %32, %18 : i32 loc(#loc43)
                            %34 = arith.index_cast %33 : i32 to index loc(#loc44)
                            %35 = "polygeist.subindex"(%arg2, %34) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc45)
                            %36 = affine.load %35[0] : memref<?xf64> loc(#loc45)
                            %37 = arith.addf %30, %36 : f64 loc(#loc46)
                            %38 = arith.muli %18, %c64_i32 : i32 loc(#loc47)
                            %39 = arith.index_cast %arg8 : i32 to index loc(#loc48)
                            %40 = "polygeist.subindex"(%arg0, %39) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc49)
                            %41 = affine.load %40[0] : memref<?xi8> loc(#loc49)
                            %42 = arith.extsi %41 : i8 to i32 loc(#loc49)
                            %43 = arith.addi %38, %42 : i32 loc(#loc50)
                            %44 = arith.index_cast %43 : i32 to index loc(#loc51)
                            %45 = "polygeist.subindex"(%arg3, %44) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc52)
                            %46 = affine.load %45[0] : memref<?xf64> loc(#loc52)
                            %47 = arith.addf %37, %46 : f64 loc(#loc53)
                            scf.yield %47 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %23 : f64 loc(#loc)
                        } else {
                          scf.yield %arg10 : f64 loc(#loc)
                        } loc(#loc)
                        %21:3 = scf.if %true -> (f64, f64, i8) {
                          %23:3 = scf.execute_region -> (f64, f64, i8) {
                            cf.br ^bb1 loc(#loc54)
                          ^bb1:  // pred: ^bb0
                            %24:3 = scf.if %true -> (f64, f64, i8) {
                              %25:3 = scf.execute_region -> (f64, f64, i8) {
                                %26:3 = scf.while (%arg13 = %arg9, %arg14 = %20, %arg15 = %c1_i8) : (f64, f64, i8) -> (f64, f64, i8) {
                                  %27 = arith.extsi %arg15 : i8 to i32 loc(#loc55)
                                  %28 = arith.cmpi slt, %27, %c64_i32 : i32 loc(#loc56)
                                  scf.condition(%28) %arg13, %arg14, %arg15 : f64, f64, i8 loc(#loc57)
                                } do {
                                ^bb0(%arg13: f64 loc("viterbi.c":8:3), %arg14: f64 loc("viterbi.c":8:3), %arg15: i8 loc("viterbi.c":7:3)):
                                  %27 = arith.extsi %arg15 : i8 to i32 loc(#loc55)
                                  %28 = scf.if %true -> (f64) {
                                    %31 = scf.execute_region -> f64 {
                                      %32 = arith.addi %arg8, %c-1_i32 : i32 loc(#loc58)
                                      %33 = arith.index_cast %32 : i32 to index loc(#loc59)
                                      %34 = "polygeist.subindex"(%alloca, %33) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc60)
                                      %35 = "polygeist.subindex"(%34, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc60)
                                      %36 = arith.index_cast %arg15 : i8 to index loc(#loc61)
                                      %37 = "polygeist.subindex"(%35, %36) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc60)
                                      %38 = affine.load %37[0] : memref<?xf64> loc(#loc60)
                                      %39 = arith.muli %27, %c64_i32 : i32 loc(#loc62)
                                      %40 = arith.addi %39, %18 : i32 loc(#loc63)
                                      %41 = arith.index_cast %40 : i32 to index loc(#loc64)
                                      %42 = "polygeist.subindex"(%arg2, %41) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc65)
                                      %43 = affine.load %42[0] : memref<?xf64> loc(#loc65)
                                      %44 = arith.addf %38, %43 : f64 loc(#loc66)
                                      %45 = arith.muli %18, %c64_i32 : i32 loc(#loc67)
                                      %46 = arith.index_cast %arg8 : i32 to index loc(#loc68)
                                      %47 = "polygeist.subindex"(%arg0, %46) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc69)
                                      %48 = affine.load %47[0] : memref<?xi8> loc(#loc69)
                                      %49 = arith.extsi %48 : i8 to i32 loc(#loc69)
                                      %50 = arith.addi %45, %49 : i32 loc(#loc70)
                                      %51 = arith.index_cast %50 : i32 to index loc(#loc71)
                                      %52 = "polygeist.subindex"(%arg3, %51) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc72)
                                      %53 = affine.load %52[0] : memref<?xf64> loc(#loc72)
                                      %54 = arith.addf %44, %53 : f64 loc(#loc73)
                                      scf.yield %54 : f64 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %31 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : f64 loc(#loc)
                                  } loc(#loc)
                                  %29 = scf.if %true -> (f64) {
                                    %31 = scf.execute_region -> f64 {
                                      %32 = scf.if %true -> (f64) {
                                        %33 = scf.execute_region -> f64 {
                                          %34 = arith.cmpf olt, %28, %arg14 : f64 loc(#loc74)
                                          %35 = scf.if %34 -> (f64) {
                                            %36 = scf.if %true -> (f64) {
                                              scf.execute_region {
                                                scf.yield loc(#loc)
                                              } loc(#loc)
                                              scf.yield %28 : f64 loc(#loc)
                                            } else {
                                              scf.yield %arg14 : f64 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %36 : f64 loc(#loc75)
                                          } else {
                                            scf.yield %arg14 : f64 loc(#loc75)
                                          } loc(#loc75)
                                          scf.yield %35 : f64 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %33 : f64 loc(#loc)
                                      } else {
                                        scf.yield %arg14 : f64 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %32 : f64 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %31 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : f64 loc(#loc)
                                  } loc(#loc)
                                  %30 = scf.if %true -> (i8) {
                                    %31 = scf.execute_region -> i8 {
                                      %32 = arith.addi %arg15, %c1_i8 : i8 loc(#loc76)
                                      scf.yield %32 : i8 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %31 : i8 loc(#loc)
                                  } else {
                                    scf.yield %arg15 : i8 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %28, %29, %30 : f64, f64, i8 loc(#loc57)
                                } loc(#loc55)
                                scf.yield %26#0, %26#1, %26#2 : f64, f64, i8 loc(#loc)
                              } loc(#loc)
                              scf.yield %25#0, %25#1, %25#2 : f64, f64, i8 loc(#loc)
                            } else {
                              scf.yield %arg9, %20, %19 : f64, f64, i8 loc(#loc)
                            } loc(#loc)
                            scf.yield %24#0, %24#1, %24#2 : f64, f64, i8 loc(#loc)
                          } loc(#loc)
                          scf.yield %23#0, %23#1, %23#2 : f64, f64, i8 loc(#loc)
                        } else {
                          scf.yield %arg9, %20, %19 : f64, f64, i8 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %23 = arith.index_cast %arg8 : i32 to index loc(#loc77)
                            %24 = "polygeist.subindex"(%alloca, %23) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc78)
                            %25 = "polygeist.subindex"(%24, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc78)
                            %26 = arith.index_cast %arg12 : i8 to index loc(#loc79)
                            %27 = "polygeist.subindex"(%25, %26) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc78)
                            affine.store %21#1, %27[0] : memref<?xf64> loc(#loc80)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %22 = scf.if %true -> (i8) {
                          %23 = scf.execute_region -> i8 {
                            %24 = arith.addi %arg12, %c1_i8 : i8 loc(#loc81)
                            scf.yield %24 : i8 loc(#loc)
                          } loc(#loc)
                          scf.yield %23 : i8 loc(#loc)
                        } else {
                          scf.yield %arg12 : i8 loc(#loc)
                        } loc(#loc)
                        scf.yield %21#0, %21#1, %22, %21#2 : f64, f64, i8, i8 loc(#loc35)
                      } loc(#loc33)
                      scf.yield %17#0, %17#1, %17#2 : f64, f64, i8 loc(#loc)
                    } loc(#loc)
                    scf.yield %16#0, %16#1, %16#2 : f64, f64, i8 loc(#loc)
                  } else {
                    scf.yield %arg5, %arg6, %arg7 : f64, f64, i8 loc(#loc)
                  } loc(#loc)
                  scf.yield %15#0, %15#1, %15#2 : f64, f64, i8 loc(#loc)
                } loc(#loc)
                scf.yield %14#0, %14#1, %14#2 : f64, f64, i8 loc(#loc)
              } else {
                scf.yield %arg5, %arg6, %arg7 : f64, f64, i8 loc(#loc)
              } loc(#loc)
              %13 = scf.if %true -> (i32) {
                %14 = scf.execute_region -> i32 {
                  %15 = arith.addi %arg8, %c1_i32 : i32 loc(#loc82)
                  scf.yield %15 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %14 : i32 loc(#loc)
              } else {
                scf.yield %arg8 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %12#0, %12#1, %12#2, %13 : f64, f64, i8, i32 loc(#loc31)
            } loc(#loc6)
            scf.yield %11#0, %11#1 : f64, f64 loc(#loc)
          } loc(#loc)
          scf.yield %10#0, %10#1 : f64, f64 loc(#loc)
        } else {
          scf.yield %1, %1 : f64, f64 loc(#loc)
        } loc(#loc)
        scf.yield %9#0, %9#1 : f64, f64 loc(#loc)
      } loc(#loc)
      scf.yield %8#0, %8#1 : f64, f64 loc(#loc)
    } else {
      scf.yield %1, %1 : f64, f64 loc(#loc)
    } loc(#loc)
    %4 = scf.if %true -> (i8) {
      %8 = scf.execute_region -> i8 {
        scf.yield %c0_i8 : i8 loc(#loc)
      } loc(#loc)
      scf.yield %8 : i8 loc(#loc)
    } else {
      scf.yield %0 : i8 loc(#loc)
    } loc(#loc)
    %5 = scf.if %true -> (f64) {
      %8 = scf.execute_region -> f64 {
        %9 = "polygeist.subindex"(%alloca, %c139) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc83)
        %10 = "polygeist.subindex"(%9, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc83)
        %11 = arith.index_cast %4 : i8 to index loc(#loc84)
        %12 = "polygeist.subindex"(%10, %11) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc83)
        %13 = affine.load %12[0] : memref<?xf64> loc(#loc83)
        scf.yield %13 : f64 loc(#loc)
      } loc(#loc)
      scf.yield %8 : f64 loc(#loc)
    } else {
      scf.yield %3#1 : f64 loc(#loc)
    } loc(#loc)
    %6:3 = scf.if %true -> (i8, f64, f64) {
      %8:3 = scf.execute_region -> (i8, f64, f64) {
        cf.br ^bb1 loc(#loc85)
      ^bb1:  // pred: ^bb0
        %9:3 = scf.if %true -> (i8, f64, f64) {
          %10:3 = scf.execute_region -> (i8, f64, f64) {
            %11:4 = scf.while (%arg5 = %c1_i8, %arg6 = %4, %arg7 = %3#0, %arg8 = %5) : (i8, i8, f64, f64) -> (i8, f64, f64, i8) {
              %12 = arith.extsi %arg5 : i8 to i32 loc(#loc86)
              %13 = arith.cmpi slt, %12, %c64_i32 : i32 loc(#loc87)
              scf.condition(%13) %arg6, %arg7, %arg8, %arg5 : i8, f64, f64, i8 loc(#loc88)
            } do {
            ^bb0(%arg5: i8 loc("viterbi.c":9:3), %arg6: f64 loc("viterbi.c":8:3), %arg7: f64 loc("viterbi.c":8:3), %arg8: i8 loc("viterbi.c":40:27)):
              %12 = scf.if %true -> (f64) {
                %15 = scf.execute_region -> f64 {
                  %16 = "polygeist.subindex"(%alloca, %c139) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc89)
                  %17 = "polygeist.subindex"(%16, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc89)
                  %18 = arith.index_cast %arg8 : i8 to index loc(#loc90)
                  %19 = "polygeist.subindex"(%17, %18) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc89)
                  %20 = affine.load %19[0] : memref<?xf64> loc(#loc89)
                  scf.yield %20 : f64 loc(#loc)
                } loc(#loc)
                scf.yield %15 : f64 loc(#loc)
              } else {
                scf.yield %arg6 : f64 loc(#loc)
              } loc(#loc)
              %13:2 = scf.if %true -> (i8, f64) {
                %15:2 = scf.execute_region -> (i8, f64) {
                  %16:2 = scf.if %true -> (i8, f64) {
                    %17:2 = scf.execute_region -> (i8, f64) {
                      %18 = arith.cmpf olt, %12, %arg7 : f64 loc(#loc91)
                      %19:2 = scf.if %18 -> (i8, f64) {
                        %20 = scf.if %true -> (f64) {
                          scf.execute_region {
                            scf.yield loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : f64 loc(#loc)
                        } else {
                          scf.yield %arg7 : f64 loc(#loc)
                        } loc(#loc)
                        %21 = scf.if %true -> (i8) {
                          scf.execute_region {
                            scf.yield loc(#loc)
                          } loc(#loc)
                          scf.yield %arg8 : i8 loc(#loc)
                        } else {
                          scf.yield %arg5 : i8 loc(#loc)
                        } loc(#loc)
                        scf.yield %21, %20 : i8, f64 loc(#loc92)
                      } else {
                        scf.yield %arg5, %arg7 : i8, f64 loc(#loc92)
                      } loc(#loc92)
                      scf.yield %19#0, %19#1 : i8, f64 loc(#loc)
                    } loc(#loc)
                    scf.yield %17#0, %17#1 : i8, f64 loc(#loc)
                  } else {
                    scf.yield %arg5, %arg7 : i8, f64 loc(#loc)
                  } loc(#loc)
                  scf.yield %16#0, %16#1 : i8, f64 loc(#loc)
                } loc(#loc)
                scf.yield %15#0, %15#1 : i8, f64 loc(#loc)
              } else {
                scf.yield %arg5, %arg7 : i8, f64 loc(#loc)
              } loc(#loc)
              %14 = scf.if %true -> (i8) {
                %15 = scf.execute_region -> i8 {
                  %16 = arith.addi %arg8, %c1_i8 : i8 loc(#loc93)
                  scf.yield %16 : i8 loc(#loc)
                } loc(#loc)
                scf.yield %15 : i8 loc(#loc)
              } else {
                scf.yield %arg8 : i8 loc(#loc)
              } loc(#loc)
              scf.yield %14, %13#0, %12, %13#1 : i8, i8, f64, f64 loc(#loc88)
            } loc(#loc86)
            scf.yield %11#0, %11#1, %11#2 : i8, f64, f64 loc(#loc)
          } loc(#loc)
          scf.yield %10#0, %10#1, %10#2 : i8, f64, f64 loc(#loc)
        } else {
          scf.yield %4, %3#0, %5 : i8, f64, f64 loc(#loc)
        } loc(#loc)
        scf.yield %9#0, %9#1, %9#2 : i8, f64, f64 loc(#loc)
      } loc(#loc)
      scf.yield %8#0, %8#1, %8#2 : i8, f64, f64 loc(#loc)
    } else {
      scf.yield %4, %3#0, %5 : i8, f64, f64 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %8 = "polygeist.subindex"(%arg4, %c139) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc94)
        affine.store %6#0, %8[0] : memref<?xi8> loc(#loc95)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc96)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %8:4 = scf.while (%arg5 = %6#0, %arg6 = %6#1, %arg7 = %6#2, %arg8 = %c138_i32) : (i8, f64, f64, i32) -> (i8, f64, f64, i32) {
              %9 = arith.cmpi sge, %arg8, %c0_i32 : i32 loc(#loc97)
              scf.condition(%9) %arg5, %arg6, %arg7, %arg8 : i8, f64, f64, i32 loc(#loc98)
            } do {
            ^bb0(%arg5: i8 loc("viterbi.c":50:42), %arg6: f64 loc("viterbi.c":50:42), %arg7: f64 loc("viterbi.c":50:42), %arg8: i32 loc("viterbi.c":50:42)):
              %9 = scf.if %true -> (i8) {
                %13 = scf.execute_region -> i8 {
                  scf.yield %c0_i8 : i8 loc(#loc)
                } loc(#loc)
                scf.yield %13 : i8 loc(#loc)
              } else {
                scf.yield %arg5 : i8 loc(#loc)
              } loc(#loc)
              %10 = scf.if %true -> (f64) {
                %13 = scf.execute_region -> f64 {
                  %14 = arith.index_cast %arg8 : i32 to index loc(#loc99)
                  %15 = "polygeist.subindex"(%alloca, %14) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc100)
                  %16 = "polygeist.subindex"(%15, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc100)
                  %17 = arith.index_cast %9 : i8 to index loc(#loc101)
                  %18 = "polygeist.subindex"(%16, %17) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc100)
                  %19 = affine.load %18[0] : memref<?xf64> loc(#loc100)
                  %20 = arith.extsi %9 : i8 to i32 loc(#loc102)
                  %21 = arith.muli %20, %c64_i32 : i32 loc(#loc103)
                  %22 = arith.addi %arg8, %c1_i32 : i32 loc(#loc104)
                  %23 = arith.index_cast %22 : i32 to index loc(#loc105)
                  %24 = "polygeist.subindex"(%arg4, %23) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc106)
                  %25 = affine.load %24[0] : memref<?xi8> loc(#loc106)
                  %26 = arith.extsi %25 : i8 to i32 loc(#loc106)
                  %27 = arith.addi %21, %26 : i32 loc(#loc107)
                  %28 = arith.index_cast %27 : i32 to index loc(#loc108)
                  %29 = "polygeist.subindex"(%arg2, %28) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc109)
                  %30 = affine.load %29[0] : memref<?xf64> loc(#loc109)
                  %31 = arith.addf %19, %30 : f64 loc(#loc110)
                  scf.yield %31 : f64 loc(#loc)
                } loc(#loc)
                scf.yield %13 : f64 loc(#loc)
              } else {
                scf.yield %arg7 : f64 loc(#loc)
              } loc(#loc)
              %11:3 = scf.if %true -> (i8, f64, f64) {
                %13:3 = scf.execute_region -> (i8, f64, f64) {
                  cf.br ^bb1 loc(#loc111)
                ^bb1:  // pred: ^bb0
                  %14:3 = scf.if %true -> (i8, f64, f64) {
                    %15:3 = scf.execute_region -> (i8, f64, f64) {
                      %16:4 = scf.while (%arg9 = %c1_i8, %arg10 = %9, %arg11 = %arg6, %arg12 = %10) : (i8, i8, f64, f64) -> (i8, f64, f64, i8) {
                        %17 = arith.extsi %arg9 : i8 to i32 loc(#loc112)
                        %18 = arith.cmpi slt, %17, %c64_i32 : i32 loc(#loc113)
                        scf.condition(%18) %arg10, %arg11, %arg12, %arg9 : i8, f64, f64, i8 loc(#loc114)
                      } do {
                      ^bb0(%arg9: i8 loc("viterbi.c":9:3), %arg10: f64 loc("viterbi.c":8:3), %arg11: f64 loc("viterbi.c":8:3), %arg12: i8 loc("viterbi.c":53:31)):
                        %17 = arith.extsi %arg12 : i8 to i32 loc(#loc112)
                        %18 = scf.if %true -> (f64) {
                          %21 = scf.execute_region -> f64 {
                            %22 = arith.index_cast %arg8 : i32 to index loc(#loc115)
                            %23 = "polygeist.subindex"(%alloca, %22) : (memref<140x64xf64>, index) -> memref<?x64xf64> loc(#loc116)
                            %24 = "polygeist.subindex"(%23, %c0) : (memref<?x64xf64>, index) -> memref<64xf64> loc(#loc116)
                            %25 = arith.index_cast %arg12 : i8 to index loc(#loc117)
                            %26 = "polygeist.subindex"(%24, %25) : (memref<64xf64>, index) -> memref<?xf64> loc(#loc116)
                            %27 = affine.load %26[0] : memref<?xf64> loc(#loc116)
                            %28 = arith.muli %17, %c64_i32 : i32 loc(#loc118)
                            %29 = arith.addi %arg8, %c1_i32 : i32 loc(#loc119)
                            %30 = arith.index_cast %29 : i32 to index loc(#loc120)
                            %31 = "polygeist.subindex"(%arg4, %30) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc121)
                            %32 = affine.load %31[0] : memref<?xi8> loc(#loc121)
                            %33 = arith.extsi %32 : i8 to i32 loc(#loc121)
                            %34 = arith.addi %28, %33 : i32 loc(#loc122)
                            %35 = arith.index_cast %34 : i32 to index loc(#loc123)
                            %36 = "polygeist.subindex"(%arg2, %35) : (memref<4096xf64>, index) -> memref<?xf64> loc(#loc124)
                            %37 = affine.load %36[0] : memref<?xf64> loc(#loc124)
                            %38 = arith.addf %27, %37 : f64 loc(#loc125)
                            scf.yield %38 : f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %21 : f64 loc(#loc)
                        } else {
                          scf.yield %arg10 : f64 loc(#loc)
                        } loc(#loc)
                        %19:2 = scf.if %true -> (i8, f64) {
                          %21:2 = scf.execute_region -> (i8, f64) {
                            %22:2 = scf.if %true -> (i8, f64) {
                              %23:2 = scf.execute_region -> (i8, f64) {
                                %24 = arith.cmpf olt, %18, %arg11 : f64 loc(#loc126)
                                %25:2 = scf.if %24 -> (i8, f64) {
                                  %26 = scf.if %true -> (f64) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %18 : f64 loc(#loc)
                                  } else {
                                    scf.yield %arg11 : f64 loc(#loc)
                                  } loc(#loc)
                                  %27 = scf.if %true -> (i8) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %arg12 : i8 loc(#loc)
                                  } else {
                                    scf.yield %arg9 : i8 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %27, %26 : i8, f64 loc(#loc127)
                                } else {
                                  scf.yield %arg9, %arg11 : i8, f64 loc(#loc127)
                                } loc(#loc127)
                                scf.yield %25#0, %25#1 : i8, f64 loc(#loc)
                              } loc(#loc)
                              scf.yield %23#0, %23#1 : i8, f64 loc(#loc)
                            } else {
                              scf.yield %arg9, %arg11 : i8, f64 loc(#loc)
                            } loc(#loc)
                            scf.yield %22#0, %22#1 : i8, f64 loc(#loc)
                          } loc(#loc)
                          scf.yield %21#0, %21#1 : i8, f64 loc(#loc)
                        } else {
                          scf.yield %arg9, %arg11 : i8, f64 loc(#loc)
                        } loc(#loc)
                        %20 = scf.if %true -> (i8) {
                          %21 = scf.execute_region -> i8 {
                            %22 = arith.addi %arg12, %c1_i8 : i8 loc(#loc128)
                            scf.yield %22 : i8 loc(#loc)
                          } loc(#loc)
                          scf.yield %21 : i8 loc(#loc)
                        } else {
                          scf.yield %arg12 : i8 loc(#loc)
                        } loc(#loc)
                        scf.yield %20, %19#0, %18, %19#1 : i8, i8, f64, f64 loc(#loc114)
                      } loc(#loc112)
                      scf.yield %16#0, %16#1, %16#2 : i8, f64, f64 loc(#loc)
                    } loc(#loc)
                    scf.yield %15#0, %15#1, %15#2 : i8, f64, f64 loc(#loc)
                  } else {
                    scf.yield %9, %arg6, %10 : i8, f64, f64 loc(#loc)
                  } loc(#loc)
                  scf.yield %14#0, %14#1, %14#2 : i8, f64, f64 loc(#loc)
                } loc(#loc)
                scf.yield %13#0, %13#1, %13#2 : i8, f64, f64 loc(#loc)
              } else {
                scf.yield %9, %arg6, %10 : i8, f64, f64 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %13 = arith.index_cast %arg8 : i32 to index loc(#loc129)
                  %14 = "polygeist.subindex"(%arg4, %13) : (memref<140xi8>, index) -> memref<?xi8> loc(#loc130)
                  affine.store %11#0, %14[0] : memref<?xi8> loc(#loc131)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %12 = scf.if %true -> (i32) {
                %13 = scf.execute_region -> i32 {
                  %14 = arith.addi %arg8, %c-1_i32 : i32 loc(#loc132)
                  scf.yield %14 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %13 : i32 loc(#loc)
              } else {
                scf.yield %arg8 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %11#0, %11#1, %11#2, %12 : i8, f64, f64, i32 loc(#loc98)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    %7 = scf.if %true -> (i32) {
      %8 = scf.execute_region -> i32 {
        %9 = scf.if %true -> (i32) {
          %10 = scf.execute_region -> i32 {
            scf.yield %c0_i32 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %10 : i32 loc(#loc)
        } else {
          scf.yield %2 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %9 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %8 : i32 loc(#loc)
    } else {
      scf.yield %2 : i32 loc(#loc)
    } loc(#loc)
    return %7 : i32 loc(#loc133)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("viterbi.c":18:29)
#loc3 = loc("viterbi.c":47:15)
#loc5 = loc("viterbi.c":50:35)
#loc7 = loc("viterbi.c":13:41)
#loc8 = loc("./viterbi.h":21:19)
#loc9 = loc("viterbi.c":13:25)
#loc10 = loc("viterbi.c":3:1)
#loc13 = loc("viterbi.c":6:3)
#loc14 = loc("viterbi.c":5:10)
#loc15 = loc("viterbi.c":13:10)
#loc17 = loc("viterbi.c":13:29)
#loc18 = loc("viterbi.c":13:18)
#loc19 = loc("viterbi.c":14:5)
#loc20 = loc("viterbi.c":14:15)
#loc21 = loc("viterbi.c":14:19)
#loc22 = loc("viterbi.c":14:39)
#loc23 = loc("viterbi.c":14:49)
#loc24 = loc("viterbi.c":14:48)
#loc25 = loc("viterbi.c":14:55)
#loc26 = loc("viterbi.c":14:29)
#loc27 = loc("viterbi.c":14:27)
#loc28 = loc("viterbi.c":14:17)
#loc29 = loc("viterbi.c":18:10)
#loc30 = loc("viterbi.c":18:33)
#loc31 = loc("viterbi.c":18:22)
#loc32 = loc("viterbi.c":19:12)
#loc34 = loc("viterbi.c":19:43)
#loc35 = loc("viterbi.c":19:26)
#loc37 = loc("viterbi.c":22:22)
#loc38 = loc("viterbi.c":22:24)
#loc39 = loc("viterbi.c":22:15)
#loc40 = loc("viterbi.c":22:30)
#loc41 = loc("viterbi.c":23:26)
#loc42 = loc("viterbi.c":23:30)
#loc43 = loc("viterbi.c":23:39)
#loc44 = loc("viterbi.c":23:44)
#loc45 = loc("viterbi.c":23:15)
#loc46 = loc("viterbi.c":22:32)
#loc47 = loc("viterbi.c":24:28)
#loc48 = loc("viterbi.c":24:43)
#loc49 = loc("viterbi.c":24:38)
#loc50 = loc("viterbi.c":24:37)
#loc51 = loc("viterbi.c":24:44)
#loc52 = loc("viterbi.c":24:15)
#loc53 = loc("viterbi.c":23:46)
#loc54 = loc("viterbi.c":25:14)
#loc55 = loc("viterbi.c":25:41)
#loc56 = loc("viterbi.c":25:45)
#loc57 = loc("viterbi.c":25:28)
#loc58 = loc("viterbi.c":26:20)
#loc59 = loc("viterbi.c":26:22)
#loc60 = loc("viterbi.c":26:13)
#loc61 = loc("viterbi.c":26:28)
#loc62 = loc("viterbi.c":27:28)
#loc63 = loc("viterbi.c":27:37)
#loc64 = loc("viterbi.c":27:42)
#loc65 = loc("viterbi.c":27:13)
#loc66 = loc("viterbi.c":26:30)
#loc67 = loc("viterbi.c":28:26)
#loc68 = loc("viterbi.c":28:41)
#loc69 = loc("viterbi.c":28:36)
#loc70 = loc("viterbi.c":28:35)
#loc71 = loc("viterbi.c":28:42)
#loc72 = loc("viterbi.c":28:13)
#loc73 = loc("viterbi.c":27:44)
#loc74 = loc("viterbi.c":29:14)
#loc75 = loc("viterbi.c":29:9)
#loc76 = loc("viterbi.c":25:60)
#loc77 = loc("viterbi.c":33:14)
#loc78 = loc("viterbi.c":33:7)
#loc79 = loc("viterbi.c":33:20)
#loc80 = loc("viterbi.c":33:22)
#loc81 = loc("viterbi.c":19:58)
#loc82 = loc("viterbi.c":18:42)
#loc83 = loc("viterbi.c":39:11)
#loc84 = loc("viterbi.c":39:31)
#loc85 = loc("viterbi.c":40:10)
#loc87 = loc("viterbi.c":40:28)
#loc88 = loc("viterbi.c":40:17)
#loc89 = loc("viterbi.c":41:9)
#loc90 = loc("viterbi.c":41:25)
#loc91 = loc("viterbi.c":42:10)
#loc92 = loc("viterbi.c":42:5)
#loc93 = loc("viterbi.c":40:40)
#loc94 = loc("viterbi.c":47:3)
#loc95 = loc("viterbi.c":47:17)
#loc96 = loc("viterbi.c":50:10)
#loc97 = loc("viterbi.c":50:40)
#loc98 = loc("viterbi.c":50:23)
#loc99 = loc("viterbi.c":52:20)
#loc100 = loc("viterbi.c":52:13)
#loc101 = loc("viterbi.c":52:27)
#loc102 = loc("viterbi.c":52:42)
#loc103 = loc("viterbi.c":52:47)
#loc104 = loc("viterbi.c":52:63)
#loc105 = loc("viterbi.c":52:65)
#loc106 = loc("viterbi.c":52:57)
#loc107 = loc("viterbi.c":52:56)
#loc108 = loc("viterbi.c":52:66)
#loc109 = loc("viterbi.c":52:31)
#loc110 = loc("viterbi.c":52:29)
#loc111 = loc("viterbi.c":53:12)
#loc113 = loc("viterbi.c":53:32)
#loc114 = loc("viterbi.c":53:21)
#loc115 = loc("viterbi.c":54:18)
#loc116 = loc("viterbi.c":54:11)
#loc117 = loc("viterbi.c":54:21)
#loc118 = loc("viterbi.c":54:37)
#loc119 = loc("viterbi.c":54:53)
#loc120 = loc("viterbi.c":54:55)
#loc121 = loc("viterbi.c":54:47)
#loc122 = loc("viterbi.c":54:46)
#loc123 = loc("viterbi.c":54:56)
#loc124 = loc("viterbi.c":54:25)
#loc125 = loc("viterbi.c":54:23)
#loc126 = loc("viterbi.c":55:12)
#loc127 = loc("viterbi.c":55:7)
#loc128 = loc("viterbi.c":53:44)
#loc129 = loc("viterbi.c":60:11)
#loc130 = loc("viterbi.c":60:5)
#loc131 = loc("viterbi.c":60:13)
#loc132 = loc("viterbi.c":50:46)
#loc133 = loc("viterbi.c":64:1)
