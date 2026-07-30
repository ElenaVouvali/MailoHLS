#loc1 = loc("knn.cpp":38:6)
#loc4 = loc("./knn.h":15:25)
#loc24 = loc("./knn.h":16:39)
#loc37 = loc("knn.cpp":3:6)
#loc39 = loc("knn.cpp":7:51)
#loc41 = loc("./knn.h":17:30)
#loc57 = loc("knn.cpp":12:6)
#loc63 = loc("knn.cpp":20:21)
#loc64 = loc("knn.cpp":16:2)
#loc72 = loc("knn.cpp":18:17)
#loc73 = loc("knn.cpp":15:5)
#loc94 = loc("knn.cpp":29:6)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32> loc("knn.cpp":38:6), %arg1: memref<2097152xf32> loc("knn.cpp":38:6), %arg2: memref<1048576xf32> loc("knn.cpp":38:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2048_i32 = arith.constant 2048 : i32 loc(#loc2)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %c2_i32 = arith.constant 2 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %true = arith.constant true loc(#loc6)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc7)
    %alloca = memref.alloca() : memref<512xf32> loc(#loc8)
    %alloca_0 = memref.alloca() : memref<1024xf32> loc(#loc9)
    %alloca_1 = memref.alloca() : memref<2xf32> loc(#loc10)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc11)
      ^bb1:  // pred: ^bb0
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
        cf.br ^bb1 loc(#loc12)
      ^bb1:  // pred: ^bb0
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
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc14)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc15)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.if %true -> (i32) {
              %3 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc)
            } else {
              scf.yield %0 : i32 loc(#loc)
            } loc(#loc)
            %2 = scf.while (%arg3 = %1) : (i32) -> i32 {
              %3 = arith.cmpi slt, %arg3, %c2_i32 : i32 loc(#loc16)
              scf.condition(%3) %arg3 : i32 loc(#loc17)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":15:25)):
              scf.if %true {
                scf.execute_region {
                  %4 = arith.index_cast %arg3 : i32 to index loc(#loc18)
                  %5 = "polygeist.subindex"(%alloca_1, %4) : (memref<2xf32>, index) -> memref<?xf32> loc(#loc19)
                  %6 = "polygeist.subindex"(%arg0, %4) : (memref<2xf32>, index) -> memref<?xf32> loc(#loc20)
                  %7 = affine.load %6[0] : memref<?xf32> loc(#loc20)
                  affine.store %7, %5[0] : memref<?xf32> loc(#loc21)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg3, %c1_i32 : i32 loc(#loc3)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc17)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc22)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc23)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.if %true -> (i32) {
              %3 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc)
            } else {
              scf.yield %0 : i32 loc(#loc)
            } loc(#loc)
            %2 = scf.while (%arg3 = %1) : (i32) -> i32 {
              %3 = arith.cmpi slt, %arg3, %c2048_i32 : i32 loc(#loc25)
              scf.condition(%3) %arg3 : i32 loc(#loc26)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":16:39)):
              scf.if %true {
                scf.execute_region {
                  %cast = memref.cast %alloca_0 : memref<1024xf32> to memref<?xf32> loc(#loc27)
                  %cast_2 = memref.cast %arg1 : memref<2097152xf32> to memref<?xf32> loc(#loc28)
                  func.call @load(%arg3, %cast_2, %cast) : (i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc28)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %cast = memref.cast %alloca_1 : memref<2xf32> to memref<?xf32> loc(#loc29)
                  %cast_2 = memref.cast %alloca_0 : memref<1024xf32> to memref<?xf32> loc(#loc30)
                  %cast_3 = memref.cast %alloca : memref<512xf32> to memref<?xf32> loc(#loc31)
                  func.call @compute_dist(%cast, %cast_2, %cast_3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> () loc(#loc32)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %cast = memref.cast %alloca : memref<512xf32> to memref<?xf32> loc(#loc33)
                  %cast_2 = memref.cast %arg2 : memref<1048576xf32> to memref<?xf32> loc(#loc34)
                  func.call @store(%arg3, %cast, %cast_2) : (i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc34)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg3, %c1_i32 : i32 loc(#loc35)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc26)
            } loc(#loc24)
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
    return loc(#loc36)
  } loc(#loc1)
  func.func @load(%arg0: i32 loc("knn.cpp":3:6), %arg1: memref<?xf32> loc("knn.cpp":3:6), %arg2: memref<?xf32> loc("knn.cpp":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc38)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc39)
    %c0_i32 = arith.constant 0 : i32 loc(#loc40)
    %c2_i32 = arith.constant 2 : i32 loc(#loc4)
    %c512_i32 = arith.constant 512 : i32 loc(#loc41)
    %true = arith.constant true loc(#loc42)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc43)
    %1 = scf.if %true -> (i32) {
      %2 = scf.execute_region -> i32 {
        %3 = scf.if %true -> (i32) {
          %4 = scf.execute_region -> i32 {
            %5 = arith.muli %arg0, %c512_i32 : i32 loc(#loc44)
            %6 = arith.muli %5, %c2_i32 : i32 loc(#loc45)
            scf.yield %6 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %4 : i32 loc(#loc)
        } else {
          scf.yield %0 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %3 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %2 : i32 loc(#loc)
    } else {
      scf.yield %0 : i32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc46)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc47)
      ^bb2:  // pred: ^bb1
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
            %3 = scf.while (%arg3 = %2) : (i32) -> i32 {
              %4 = arith.cmpi slt, %arg3, %c1024_i32 : i32 loc(#loc48)
              scf.condition(%4) %arg3 : i32 loc(#loc49)
            } do {
            ^bb0(%arg3: i32 loc("knn.cpp":7:51)):
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg3 : i32 to index loc(#loc50)
                  %6 = "polygeist.subindex"(%arg2, %5) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc51)
                  %7 = arith.addi %1, %arg3 : i32 loc(#loc52)
                  %8 = arith.index_cast %7 : i32 to index loc(#loc53)
                  %9 = "polygeist.subindex"(%arg1, %8) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc54)
                  %10 = affine.load %9[0] : memref<?xf32> loc(#loc54)
                  affine.store %10, %6[0] : memref<?xf32> loc(#loc55)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg3, %c1_i32 : i32 loc(#loc38)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc49)
            } loc(#loc39)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc56)
  } loc(#loc37)
  func.func @compute_dist(%arg0: memref<?xf32> loc("knn.cpp":12:6), %arg1: memref<?xf32> loc("knn.cpp":12:6), %arg2: memref<?xf32> loc("knn.cpp":12:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc58)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc59)
    %c2_i32 = arith.constant 2 : i32 loc(#loc60)
    %c512_i32 = arith.constant 512 : i32 loc(#loc41)
    %c0_i32 = arith.constant 0 : i32 loc(#loc61)
    %true = arith.constant true loc(#loc62)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc63)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc64)
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
        cf.br ^bb1 loc(#loc65)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc66)
      ^bb2:  // pred: ^bb1
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
            %3:5 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %2, %arg6 = %1, %arg7 = %1) : (i32, i32, i32, f32, f32) -> (i32, i32, i32, f32, f32) {
              %4 = arith.cmpi slt, %arg5, %c512_i32 : i32 loc(#loc67)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, f32, f32 loc(#loc68)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":17:30), %arg4: i32 loc("./knn.h":17:30), %arg5: i32 loc("./knn.h":17:30), %arg6: f32 loc("./knn.h":17:30), %arg7: f32 loc("./knn.h":17:30)):
              %4:4 = scf.if %true -> (i32, i32, f32, f32) {
                %6:4 = scf.execute_region -> (i32, i32, f32, f32) {
                  cf.br ^bb1 loc(#loc69)
                ^bb1:  // pred: ^bb0
                  %7:4 = scf.if %true -> (i32, i32, f32, f32) {
                    %8:4 = scf.execute_region -> (i32, i32, f32, f32) {
                      %9 = scf.if %true -> (i32) {
                        %11 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %11 : i32 loc(#loc)
                      } else {
                        scf.yield %arg4 : i32 loc(#loc)
                      } loc(#loc)
                      %10:4 = scf.while (%arg8 = %arg3, %arg9 = %9, %arg10 = %arg6, %arg11 = %arg7) : (i32, i32, f32, f32) -> (i32, i32, f32, f32) {
                        %11 = arith.cmpi slt, %arg9, %c2_i32 : i32 loc(#loc70)
                        scf.condition(%11) %arg8, %arg9, %arg10, %arg11 : i32, i32, f32, f32 loc(#loc71)
                      } do {
                      ^bb0(%arg8: i32 loc("knn.cpp":20:21), %arg9: i32 loc("knn.cpp":18:17), %arg10: f32 loc("knn.cpp":16:2), %arg11: f32 loc("knn.cpp":15:5)):
                        %11 = scf.if %true -> (f32) {
                          %14 = scf.execute_region -> f32 {
                            scf.yield %cst : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %14 : f32 loc(#loc)
                        } else {
                          scf.yield %arg11 : f32 loc(#loc)
                        } loc(#loc)
                        %12:3 = scf.if %true -> (i32, f32, f32) {
                          %14:3 = scf.execute_region -> (i32, f32, f32) {
                            cf.br ^bb1 loc(#loc74)
                          ^bb1:  // pred: ^bb0
                            %15:3 = scf.if %true -> (i32, f32, f32) {
                              %16:3 = scf.execute_region -> (i32, f32, f32) {
                                %17 = scf.if %true -> (i32) {
                                  %19 = scf.execute_region -> i32 {
                                    scf.yield %c0_i32 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %19 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg8 : i32 loc(#loc)
                                } loc(#loc)
                                %18:3 = scf.while (%arg12 = %17, %arg13 = %arg10, %arg14 = %11) : (i32, f32, f32) -> (i32, f32, f32) {
                                  %19 = arith.cmpi slt, %arg12, %c2_i32 : i32 loc(#loc75)
                                  scf.condition(%19) %arg12, %arg13, %arg14 : i32, f32, f32 loc(#loc76)
                                } do {
                                ^bb0(%arg12: i32 loc("knn.cpp":20:21), %arg13: f32 loc("knn.cpp":16:2), %arg14: f32 loc("knn.cpp":15:5)):
                                  %19 = scf.if %true -> (f32) {
                                    %22 = scf.execute_region -> f32 {
                                      %23 = arith.addi %arg5, %arg9 : i32 loc(#loc77)
                                      %24 = arith.muli %23, %c2_i32 : i32 loc(#loc78)
                                      %25 = arith.addi %24, %arg12 : i32 loc(#loc79)
                                      %26 = arith.index_cast %25 : i32 to index loc(#loc80)
                                      %27 = "polygeist.subindex"(%arg1, %26) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc81)
                                      %28 = affine.load %27[0] : memref<?xf32> loc(#loc81)
                                      %29 = arith.index_cast %arg12 : i32 to index loc(#loc82)
                                      %30 = "polygeist.subindex"(%arg0, %29) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc83)
                                      %31 = affine.load %30[0] : memref<?xf32> loc(#loc83)
                                      %32 = arith.subf %28, %31 : f32 loc(#loc84)
                                      scf.yield %32 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %22 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : f32 loc(#loc)
                                  } loc(#loc)
                                  %20 = scf.if %true -> (f32) {
                                    %22 = scf.execute_region -> f32 {
                                      %23 = arith.mulf %19, %19 : f32 loc(#loc85)
                                      %24 = arith.addf %arg14, %23 : f32 loc(#loc86)
                                      scf.yield %24 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %22 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : f32 loc(#loc)
                                  } loc(#loc)
                                  %21 = scf.if %true -> (i32) {
                                    %22 = scf.execute_region -> i32 {
                                      %23 = arith.addi %arg12, %c1_i32 : i32 loc(#loc58)
                                      scf.yield %23 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %22 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg12 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %21, %19, %20 : i32, f32, f32 loc(#loc76)
                                } loc(#loc75)
                                scf.yield %18#0, %18#1, %18#2 : i32, f32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %16#0, %16#1, %16#2 : i32, f32, f32 loc(#loc)
                            } else {
                              scf.yield %arg8, %arg10, %11 : i32, f32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %15#0, %15#1, %15#2 : i32, f32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %14#0, %14#1, %14#2 : i32, f32, f32 loc(#loc)
                        } else {
                          scf.yield %arg8, %arg10, %11 : i32, f32, f32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %14 = arith.addi %arg5, %arg9 : i32 loc(#loc87)
                            %15 = arith.index_cast %14 : i32 to index loc(#loc88)
                            %16 = "polygeist.subindex"(%arg2, %15) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc89)
                            affine.store %12#2, %16[0] : memref<?xf32> loc(#loc90)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (i32) {
                          %14 = scf.execute_region -> i32 {
                            %15 = arith.addi %arg9, %c1_i32 : i32 loc(#loc91)
                            scf.yield %15 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %14 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12#0, %13, %12#1, %12#2 : i32, i32, f32, f32 loc(#loc71)
                      } loc(#loc60)
                      scf.yield %10#0, %10#1, %10#2, %10#3 : i32, i32, f32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %8#0, %8#1, %8#2, %8#3 : i32, i32, f32, f32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg6, %arg7 : i32, i32, f32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %7#0, %7#1, %7#2, %7#3 : i32, i32, f32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %6#0, %6#1, %6#2, %6#3 : i32, i32, f32, f32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg6, %arg7 : i32, i32, f32, f32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %6 = scf.execute_region -> i32 {
                  %7 = arith.addi %arg5, %c2_i32 : i32 loc(#loc92)
                  scf.yield %7 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %6 : i32 loc(#loc)
              } else {
                scf.yield %arg5 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4#0, %4#1, %5, %4#2, %4#3 : i32, i32, i32, f32, f32 loc(#loc68)
            } loc(#loc41)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc93)
  } loc(#loc57)
  func.func @store(%arg0: i32 loc("knn.cpp":29:6), %arg1: memref<?xf32> loc("knn.cpp":29:6), %arg2: memref<?xf32> loc("knn.cpp":29:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc95)
    %c0_i32 = arith.constant 0 : i32 loc(#loc96)
    %c512_i32 = arith.constant 512 : i32 loc(#loc41)
    %true = arith.constant true loc(#loc97)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc98)
    %1 = scf.if %true -> (i32) {
      %2 = scf.execute_region -> i32 {
        %3 = scf.if %true -> (i32) {
          %4 = scf.execute_region -> i32 {
            %5 = arith.muli %arg0, %c512_i32 : i32 loc(#loc99)
            scf.yield %5 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %4 : i32 loc(#loc)
        } else {
          scf.yield %0 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %3 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %2 : i32 loc(#loc)
    } else {
      scf.yield %0 : i32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc100)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc101)
      ^bb2:  // pred: ^bb1
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
            %3 = scf.while (%arg3 = %2) : (i32) -> i32 {
              %4 = arith.cmpi slt, %arg3, %c512_i32 : i32 loc(#loc102)
              scf.condition(%4) %arg3 : i32 loc(#loc103)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":17:30)):
              scf.if %true {
                scf.execute_region {
                  %5 = arith.addi %1, %arg3 : i32 loc(#loc104)
                  %6 = arith.index_cast %5 : i32 to index loc(#loc105)
                  %7 = "polygeist.subindex"(%arg2, %6) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc106)
                  %8 = arith.index_cast %arg3 : i32 to index loc(#loc107)
                  %9 = "polygeist.subindex"(%arg1, %8) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc108)
                  %10 = affine.load %9[0] : memref<?xf32> loc(#loc108)
                  affine.store %10, %7[0] : memref<?xf32> loc(#loc109)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg3, %c1_i32 : i32 loc(#loc95)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc103)
            } loc(#loc41)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc110)
  } loc(#loc94)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("./knn.h":18:45)
#loc3 = loc("knn.cpp":55:52)
#loc5 = loc("knn.cpp":55:33)
#loc6 = loc("knn.cpp":38:1)
#loc7 = loc("knn.cpp":59:21)
#loc8 = loc("knn.cpp":53:5)
#loc9 = loc("knn.cpp":52:5)
#loc10 = loc("knn.cpp":51:5)
#loc11 = loc("knn.cpp":51:1)
#loc12 = loc("knn.cpp":52:1)
#loc13 = loc("knn.cpp":53:1)
#loc14 = loc("knn.cpp":55:1)
#loc15 = loc("knn.cpp":55:5)
#loc16 = loc("knn.cpp":55:38)
#loc17 = loc("knn.cpp":55:22)
#loc18 = loc("knn.cpp":56:21)
#loc19 = loc("knn.cpp":56:3)
#loc20 = loc("knn.cpp":56:25)
#loc21 = loc("knn.cpp":56:23)
#loc22 = loc("knn.cpp":59:1)
#loc23 = loc("knn.cpp":59:6)
#loc25 = loc("knn.cpp":59:46)
#loc26 = loc("knn.cpp":59:16)
#loc27 = loc("knn.cpp":60:31)
#loc28 = loc("knn.cpp":60:3)
#loc29 = loc("knn.cpp":61:16)
#loc30 = loc("knn.cpp":61:34)
#loc31 = loc("knn.cpp":61:53)
#loc32 = loc("knn.cpp":61:3)
#loc33 = loc("knn.cpp":62:25)
#loc34 = loc("knn.cpp":62:9)
#loc35 = loc("knn.cpp":59:58)
#loc36 = loc("knn.cpp":66:1)
#loc38 = loc("knn.cpp":7:65)
#loc40 = loc("knn.cpp":7:27)
#loc42 = loc("knn.cpp":3:1)
#loc43 = loc("knn.cpp":7:21)
#loc44 = loc("knn.cpp":6:27)
#loc45 = loc("knn.cpp":6:46)
#loc46 = loc("knn.cpp":7:1)
#loc47 = loc("knn.cpp":7:5)
#loc48 = loc("knn.cpp":7:33)
#loc49 = loc("knn.cpp":7:16)
#loc50 = loc("knn.cpp":8:22)
#loc51 = loc("knn.cpp":8:3)
#loc52 = loc("knn.cpp":8:47)
#loc53 = loc("knn.cpp":8:49)
#loc54 = loc("knn.cpp":8:26)
#loc55 = loc("knn.cpp":8:24)
#loc56 = loc("knn.cpp":10:1)
#loc58 = loc("knn.cpp":20:49)
#loc59 = loc("knn.cpp":19:19)
#loc60 = loc("./knn.h":19:27)
#loc61 = loc("knn.cpp":17:32)
#loc62 = loc("knn.cpp":12:1)
#loc65 = loc("knn.cpp":17:1)
#loc66 = loc("knn.cpp":17:5)
#loc67 = loc("knn.cpp":17:37)
#loc68 = loc("knn.cpp":17:19)
#loc69 = loc("knn.cpp":18:1)
#loc70 = loc("knn.cpp":18:30)
#loc71 = loc("knn.cpp":18:12)
#loc74 = loc("knn.cpp":20:1)
#loc75 = loc("knn.cpp":20:34)
#loc76 = loc("knn.cpp":20:16)
#loc77 = loc("knn.cpp":21:53)
#loc78 = loc("knn.cpp":21:56)
#loc79 = loc("knn.cpp":21:68)
#loc80 = loc("knn.cpp":21:70)
#loc81 = loc("knn.cpp":21:33)
#loc82 = loc("knn.cpp":21:92)
#loc83 = loc("knn.cpp":21:74)
#loc84 = loc("knn.cpp":21:72)
#loc85 = loc("knn.cpp":22:37)
#loc86 = loc("knn.cpp":22:21)
#loc87 = loc("knn.cpp":24:29)
#loc88 = loc("knn.cpp":24:31)
#loc89 = loc("knn.cpp":24:13)
#loc90 = loc("knn.cpp":24:33)
#loc91 = loc("knn.cpp":18:47)
#loc92 = loc("knn.cpp":17:58)
#loc93 = loc("knn.cpp":27:1)
#loc95 = loc("knn.cpp":33:54)
#loc96 = loc("knn.cpp":33:28)
#loc97 = loc("knn.cpp":29:1)
#loc98 = loc("knn.cpp":33:22)
#loc99 = loc("knn.cpp":32:28)
#loc100 = loc("knn.cpp":33:1)
#loc101 = loc("knn.cpp":33:5)
#loc102 = loc("knn.cpp":33:34)
#loc103 = loc("knn.cpp":33:17)
#loc104 = loc("knn.cpp":34:27)
#loc105 = loc("knn.cpp":34:29)
#loc106 = loc("knn.cpp":34:9)
#loc107 = loc("knn.cpp":34:49)
#loc108 = loc("knn.cpp":34:33)
#loc109 = loc("knn.cpp":34:31)
#loc110 = loc("knn.cpp":36:1)
