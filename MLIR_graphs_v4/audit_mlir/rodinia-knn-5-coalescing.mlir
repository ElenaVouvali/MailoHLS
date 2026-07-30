#loc1 = loc("knn.cpp":52:6)
#loc2 = loc("./knn.h":15:25)
#loc30 = loc("./knn.h":16:39)
#loc67 = loc("knn.cpp":3:6)
#loc71 = loc("./knn.h":17:30)
#loc81 = loc("knn.cpp":8:58)
#loc91 = loc("knn.cpp":14:6)
#loc99 = loc("knn.cpp":28:25)
#loc105 = loc("knn.cpp":19:48)
#loc111 = loc("knn.cpp":24:29)
#loc112 = loc("knn.cpp":22:21)
#loc113 = loc("knn.cpp":21:25)
#loc114 = loc("knn.cpp":20:21)
#loc144 = loc("knn.cpp":41:6)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32> loc("knn.cpp":52:6), %arg1: memref<131072xi32> loc("knn.cpp":52:6), %arg2: memref<65536xi32> loc("knn.cpp":52:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-2_i32 = arith.constant -2 : i32 loc(#loc2)
    %c2049_i32 = arith.constant 2049 : i32 loc(#loc3)
    %c2050_i32 = arith.constant 2050 : i32 loc(#loc4)
    %c2048_i32 = arith.constant 2048 : i32 loc(#loc5)
    %c1_i32 = arith.constant 1 : i32 loc(#loc6)
    %false = arith.constant false loc(#loc)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc7)
    %true = arith.constant true loc(#loc8)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc9)
    %alloca = memref.alloca() : memref<32xi32> loc(#loc10)
    %alloca_0 = memref.alloca() : memref<32xi32> loc(#loc11)
    %alloca_1 = memref.alloca() : memref<64xi32> loc(#loc12)
    %alloca_2 = memref.alloca() : memref<64xi32> loc(#loc13)
    %alloca_3 = memref.alloca() : memref<2xf32> loc(#loc14)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc15)
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
        cf.br ^bb1 loc(#loc16)
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
        cf.br ^bb1 loc(#loc17)
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
        cf.br ^bb1 loc(#loc18)
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
        cf.br ^bb1 loc(#loc19)
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
        cf.br ^bb1 loc(#loc20)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc21)
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
              %3 = arith.cmpi slt, %arg3, %c2_i32 : i32 loc(#loc22)
              scf.condition(%3) %arg3 : i32 loc(#loc23)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":15:25)):
              scf.if %true {
                scf.execute_region {
                  %4 = arith.index_cast %arg3 : i32 to index loc(#loc24)
                  %5 = "polygeist.subindex"(%alloca_3, %4) : (memref<2xf32>, index) -> memref<?xf32> loc(#loc25)
                  %6 = "polygeist.subindex"(%arg0, %4) : (memref<2xf32>, index) -> memref<?xf32> loc(#loc26)
                  %7 = affine.load %6[0] : memref<?xf32> loc(#loc26)
                  affine.store %7, %5[0] : memref<?xf32> loc(#loc27)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg3, %c1_i32 : i32 loc(#loc6)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc23)
            } loc(#loc2)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc28)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc29)
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
            %2:4 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %1) : (i32, i32, i32, i32) -> (i32, i32, i32, i32) {
              %3 = arith.cmpi slt, %arg6, %c2050_i32 : i32 loc(#loc31)
              scf.condition(%3) %arg3, %arg4, %arg5, %arg6 : i32, i32, i32, i32 loc(#loc32)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":16:39), %arg4: i32 loc("./knn.h":16:39), %arg5: i32 loc("./knn.h":16:39), %arg6: i32 loc("./knn.h":16:39)):
              %3 = arith.cmpi slt, %arg6, %c2050_i32 : i32 loc(#loc31)
              %4 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      %11 = arith.cmpi sge, %arg6, %c0_i32 : i32 loc(#loc33)
                      %12 = scf.if %11 -> (i1) {
                        %14 = arith.cmpi slt, %arg6, %c2048_i32 : i32 loc(#loc35)
                        scf.yield %14 : i1 loc(#loc34)
                      } else {
                        scf.yield %false : i1 loc(#loc34)
                      } loc(#loc34)
                      %13 = arith.extui %12 : i1 to i32 loc(#loc36)
                      scf.yield %13 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg5 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg5 : i32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      %11 = arith.cmpi sge, %arg6, %c1_i32 : i32 loc(#loc37)
                      %12 = scf.if %11 -> (i1) {
                        %14 = arith.cmpi slt, %arg6, %c2049_i32 : i32 loc(#loc39)
                        scf.yield %14 : i1 loc(#loc38)
                      } else {
                        scf.yield %false : i1 loc(#loc38)
                      } loc(#loc38)
                      %13 = arith.extui %12 : i1 to i32 loc(#loc40)
                      scf.yield %13 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg4 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      %11 = arith.cmpi sge, %arg6, %c2_i32 : i32 loc(#loc41)
                      %12 = scf.if %11 -> (i1) {
                        scf.yield %3 : i1 loc(#loc42)
                      } else {
                        scf.yield %false : i1 loc(#loc42)
                      } loc(#loc42)
                      %13 = arith.extui %12 : i1 to i32 loc(#loc43)
                      scf.yield %13 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg3 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  scf.if %true {
                    scf.execute_region {
                      %8 = arith.remsi %arg6, %c2_i32 : i32 loc(#loc44)
                      %9 = arith.cmpi eq, %8, %c0_i32 : i32 loc(#loc45)
                      scf.if %9 {
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_2 : memref<64xi32> to memref<?xi32> loc(#loc47)
                            %cast_4 = memref.cast %arg1 : memref<131072xi32> to memref<?xi32> loc(#loc48)
                            func.call @load(%4, %arg6, %cast_4, %cast) : (i32, i32, memref<?xi32>, memref<?xi32>) -> () loc(#loc48)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_3 : memref<2xf32> to memref<?xf32> loc(#loc49)
                            %cast_4 = memref.cast %alloca_1 : memref<64xi32> to memref<?xi32> loc(#loc50)
                            %cast_5 = memref.cast %alloca : memref<32xi32> to memref<?xi32> loc(#loc51)
                            func.call @compute(%5, %cast, %cast_4, %cast_5) : (i32, memref<?xf32>, memref<?xi32>, memref<?xi32>) -> () loc(#loc52)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.addi %arg6, %c-2_i32 : i32 loc(#loc53)
                            %cast = memref.cast %alloca_0 : memref<32xi32> to memref<?xi32> loc(#loc54)
                            %cast_4 = memref.cast %arg2 : memref<65536xi32> to memref<?xi32> loc(#loc55)
                            func.call @store(%6, %10, %cast, %cast_4) : (i32, i32, memref<?xi32>, memref<?xi32>) -> () loc(#loc55)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } else {
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_1 : memref<64xi32> to memref<?xi32> loc(#loc56)
                            %cast_4 = memref.cast %arg1 : memref<131072xi32> to memref<?xi32> loc(#loc57)
                            func.call @load(%4, %arg6, %cast_4, %cast) : (i32, i32, memref<?xi32>, memref<?xi32>) -> () loc(#loc57)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_3 : memref<2xf32> to memref<?xf32> loc(#loc58)
                            %cast_4 = memref.cast %alloca_2 : memref<64xi32> to memref<?xi32> loc(#loc59)
                            %cast_5 = memref.cast %alloca_0 : memref<32xi32> to memref<?xi32> loc(#loc60)
                            func.call @compute(%5, %cast, %cast_4, %cast_5) : (i32, memref<?xf32>, memref<?xi32>, memref<?xi32>) -> () loc(#loc61)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.addi %arg6, %c-2_i32 : i32 loc(#loc62)
                            %cast = memref.cast %alloca : memref<32xi32> to memref<?xi32> loc(#loc63)
                            %cast_4 = memref.cast %arg2 : memref<65536xi32> to memref<?xi32> loc(#loc64)
                            func.call @store(%6, %10, %cast, %cast_4) : (i32, i32, memref<?xi32>, memref<?xi32>) -> () loc(#loc64)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } loc(#loc46)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg6, %c1_i32 : i32 loc(#loc65)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg6 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6, %5, %4, %7 : i32, i32, i32, i32 loc(#loc32)
            } loc(#loc30)
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
    return loc(#loc66)
  } loc(#loc1)
  func.func @load(%arg0: i32 loc("knn.cpp":3:6), %arg1: i32 loc("knn.cpp":3:6), %arg2: memref<?xi32> loc("knn.cpp":3:6), %arg3: memref<?xi32> loc("knn.cpp":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc68)
    %c1_i32 = arith.constant 1 : i32 loc(#loc69)
    %c16_i32 = arith.constant 16 : i32 loc(#loc70)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %c512_i32 = arith.constant 512 : i32 loc(#loc71)
    %c0_i32 = arith.constant 0 : i32 loc(#loc72)
    %true = arith.constant true loc(#loc73)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc74)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %1 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc72)
            scf.if %1 {
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = scf.if %true -> (i32) {
                    %5 = scf.execute_region -> i32 {
                      %6 = arith.muli %arg1, %c512_i32 : i32 loc(#loc76)
                      %7 = arith.muli %6, %c2_i32 : i32 loc(#loc77)
                      %8 = arith.divsi %7, %c16_i32 : i32 loc(#loc78)
                      scf.yield %8 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %5 : i32 loc(#loc)
                  } else {
                    scf.yield %0 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %0 : i32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc79)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc80)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.if %true -> (i32) {
                        scf.execute_region {
                          scf.yield loc(#loc)
                        } loc(#loc)
                        scf.yield %c0_i32 : i32 loc(#loc)
                      } else {
                        scf.yield %0 : i32 loc(#loc)
                      } loc(#loc)
                      %4 = scf.while (%arg4 = %3) : (i32) -> i32 {
                        %5 = arith.cmpi slt, %arg4, %c64_i32 : i32 loc(#loc82)
                        scf.condition(%5) %arg4 : i32 loc(#loc83)
                      } do {
                      ^bb0(%arg4: i32 loc("knn.cpp":8:58)):
                        scf.if %true {
                          scf.execute_region {
                            %6 = arith.index_cast %arg4 : i32 to index loc(#loc84)
                            %7 = "polygeist.subindex"(%arg3, %6) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc85)
                            %8 = arith.addi %2, %arg4 : i32 loc(#loc86)
                            %9 = arith.index_cast %8 : i32 to index loc(#loc87)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc88)
                            %11 = affine.load %10[0] : memref<?xi32> loc(#loc88)
                            affine.store %11, %7[0] : memref<?xi32> loc(#loc89)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %5 = scf.if %true -> (i32) {
                          %6 = scf.execute_region -> i32 {
                            %7 = arith.addi %arg4, %c1_i32 : i32 loc(#loc69)
                            scf.yield %7 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %6 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5 : i32 loc(#loc83)
                      } loc(#loc81)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc75)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc90)
  } loc(#loc67)
  func.func @compute(%arg0: i32 loc("knn.cpp":14:6), %arg1: memref<?xf32> loc("knn.cpp":14:6), %arg2: memref<?xi32> loc("knn.cpp":14:6), %arg3: memref<?xi32> loc("knn.cpp":14:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc92)
    %c1_i32 = arith.constant 1 : i32 loc(#loc93)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc94)
    %c2_i32 = arith.constant 2 : i32 loc(#loc95)
    %false = arith.constant false loc(#loc)
    %c16_i32 = arith.constant 16 : i32 loc(#loc70)
    %c0_i32 = arith.constant 0 : i32 loc(#loc96)
    %true = arith.constant true loc(#loc97)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc98)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc99)
    %alloca = memref.alloca() : memref<1xi32> loc(#loc100)
    affine.store %0, %alloca[0] : memref<1xi32> loc(#loc100)
    %alloca_0 = memref.alloca() : memref<1xf32> loc(#loc101)
    affine.store %1, %alloca_0[0] : memref<1xf32> loc(#loc101)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %2 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc96)
            scf.if %2 {
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc103)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc104)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.if %true -> (i32) {
                        scf.execute_region {
                          scf.yield loc(#loc)
                        } loc(#loc)
                        scf.yield %c0_i32 : i32 loc(#loc)
                      } else {
                        scf.yield %0 : i32 loc(#loc)
                      } loc(#loc)
                      %4:6 = scf.while (%arg4 = %1, %arg5 = %0, %arg6 = %1, %arg7 = %0, %arg8 = %0, %arg9 = %3) : (f32, i32, f32, i32, i32, i32) -> (f32, i32, f32, i32, i32, i32) {
                        %5 = arith.cmpi slt, %arg9, %c64_i32 : i32 loc(#loc106)
                        scf.condition(%5) %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : f32, i32, f32, i32, i32, i32 loc(#loc107)
                      } do {
                      ^bb0(%arg4: f32 loc("knn.cpp":19:48), %arg5: i32 loc("knn.cpp":19:48), %arg6: f32 loc("knn.cpp":19:48), %arg7: i32 loc("knn.cpp":19:48), %arg8: i32 loc("knn.cpp":19:48), %arg9: i32 loc("knn.cpp":19:48)):
                        %5:5 = scf.if %true -> (f32, i32, f32, i32, i32) {
                          %7:5 = scf.execute_region -> (f32, i32, f32, i32, i32) {
                            cf.br ^bb1 loc(#loc108)
                          ^bb1:  // pred: ^bb0
                            %8:5 = scf.if %true -> (f32, i32, f32, i32, i32) {
                              %9:5 = scf.execute_region -> (f32, i32, f32, i32, i32) {
                                %10 = scf.if %true -> (i32) {
                                  scf.execute_region {
                                    scf.yield loc(#loc)
                                  } loc(#loc)
                                  scf.yield %c0_i32 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg8 : i32 loc(#loc)
                                } loc(#loc)
                                %11:5 = scf.while (%arg10 = %arg4, %arg11 = %arg5, %arg12 = %arg6, %arg13 = %arg7, %arg14 = %10) : (f32, i32, f32, i32, i32) -> (f32, i32, f32, i32, i32) {
                                  %12 = arith.cmpi slt, %arg14, %c2_i32 : i32 loc(#loc109)
                                  scf.condition(%12) %arg10, %arg11, %arg12, %arg13, %arg14 : f32, i32, f32, i32, i32 loc(#loc110)
                                } do {
                                ^bb0(%arg10: f32 loc("knn.cpp":28:25), %arg11: i32 loc("knn.cpp":24:29), %arg12: f32 loc("knn.cpp":22:21), %arg13: i32 loc("knn.cpp":21:25), %arg14: i32 loc("knn.cpp":20:21)):
                                  %12:4 = scf.if %true -> (f32, i32, f32, i32) {
                                    %14:4 = scf.execute_region -> (f32, i32, f32, i32) {
                                      cf.br ^bb1 loc(#loc115)
                                    ^bb1:  // pred: ^bb0
                                      %15:4 = scf.if %true -> (f32, i32, f32, i32) {
                                        %16:4 = scf.execute_region -> (f32, i32, f32, i32) {
                                          %17 = scf.if %true -> (i32) {
                                            scf.execute_region {
                                              scf.yield loc(#loc)
                                            } loc(#loc)
                                            scf.yield %c0_i32 : i32 loc(#loc)
                                          } else {
                                            scf.yield %arg13 : i32 loc(#loc)
                                          } loc(#loc)
                                          %18:4 = scf.while (%arg15 = %arg10, %arg16 = %arg11, %arg17 = %arg12, %arg18 = %17) : (f32, i32, f32, i32) -> (f32, i32, f32, i32) {
                                            %19 = arith.cmpi slt, %arg18, %c16_i32 : i32 loc(#loc116)
                                            scf.condition(%19) %arg15, %arg16, %arg17, %arg18 : f32, i32, f32, i32 loc(#loc117)
                                          } do {
                                          ^bb0(%arg15: f32 loc("knn.cpp":28:25), %arg16: i32 loc("knn.cpp":24:29), %arg17: f32 loc("knn.cpp":22:21), %arg18: i32 loc("knn.cpp":21:25)):
                                            %19 = scf.if %true -> (f32) {
                                              %22 = scf.execute_region -> f32 {
                                                %23 = scf.if %true -> (f32) {
                                                  %24 = scf.execute_region -> f32 {
                                                    scf.yield %cst : f32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %24 : f32 loc(#loc)
                                                } else {
                                                  scf.yield %arg17 : f32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %23 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %22 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg17 : f32 loc(#loc)
                                            } loc(#loc)
                                            scf.if %true {
                                              scf.execute_region {
                                                scf.if %true {
                                                  scf.execute_region {
                                                    affine.store %cst, %alloca_0[0] : memref<1xf32> loc(#loc101)
                                                    scf.yield loc(#loc)
                                                  } loc(#loc)
                                                } loc(#loc)
                                                scf.yield loc(#loc)
                                              } loc(#loc)
                                            } loc(#loc)
                                            %20:3 = scf.if %true -> (f32, i32, f32) {
                                              %22:3 = scf.execute_region -> (f32, i32, f32) {
                                                cf.br ^bb1 loc(#loc118)
                                              ^bb1:  // pred: ^bb0
                                                %23:3 = scf.if %true -> (f32, i32, f32) {
                                                  %24:3 = scf.execute_region -> (f32, i32, f32) {
                                                    %25 = scf.if %true -> (i32) {
                                                      scf.execute_region {
                                                        scf.yield loc(#loc)
                                                      } loc(#loc)
                                                      scf.yield %c0_i32 : i32 loc(#loc)
                                                    } else {
                                                      scf.yield %arg16 : i32 loc(#loc)
                                                    } loc(#loc)
                                                    %26:3 = scf.while (%arg19 = %arg15, %arg20 = %25, %arg21 = %19) : (f32, i32, f32) -> (f32, i32, f32) {
                                                      %27 = arith.cmpi slt, %arg20, %c2_i32 : i32 loc(#loc119)
                                                      scf.condition(%27) %arg19, %arg20, %arg21 : f32, i32, f32 loc(#loc120)
                                                    } do {
                                                    ^bb0(%arg19: f32 loc("knn.cpp":28:25), %arg20: i32 loc("knn.cpp":24:29), %arg21: f32 loc("knn.cpp":22:21)):
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
                                                              %30 = arith.addi %arg9, %arg14 : i32 loc(#loc121)
                                                              %31 = arith.index_cast %30 : i32 to index loc(#loc122)
                                                              %32 = "polygeist.subindex"(%arg2, %31) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc123)
                                                              %33 = affine.load %32[0] : memref<?xi32> loc(#loc124)
                                                              affine.store %33, %alloca[0] : memref<1xi32> loc(#loc100)
                                                              scf.yield loc(#loc)
                                                            } loc(#loc)
                                                          } loc(#loc)
                                                          scf.yield loc(#loc)
                                                        } loc(#loc)
                                                      } loc(#loc)
                                                      %27 = scf.if %true -> (f32) {
                                                        %30 = scf.execute_region -> f32 {
                                                          %31 = scf.if %true -> (f32) {
                                                            %32 = scf.execute_region -> f32 {
                                                              %33 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr loc(#loc125)
                                                              %34 = "polygeist.pointer2memref"(%33) : (!llvm.ptr) -> memref<?xf32> loc(#loc125)
                                                              %35 = affine.load %34[0] : memref<?xf32> loc(#loc126)
                                                              scf.yield %35 : f32 loc(#loc)
                                                            } loc(#loc)
                                                            scf.yield %32 : f32 loc(#loc)
                                                          } else {
                                                            scf.yield %arg19 : f32 loc(#loc)
                                                          } loc(#loc)
                                                          scf.yield %31 : f32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %30 : f32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg19 : f32 loc(#loc)
                                                      } loc(#loc)
                                                      %28 = scf.if %true -> (f32) {
                                                        %30 = scf.execute_region -> f32 {
                                                          %31 = arith.index_cast %arg20 : i32 to index loc(#loc127)
                                                          %32 = "polygeist.subindex"(%arg1, %31) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc128)
                                                          %33 = affine.load %32[0] : memref<?xf32> loc(#loc128)
                                                          %34 = arith.subf %27, %33 : f32 loc(#loc129)
                                                          scf.yield %34 : f32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %30 : f32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg21 : f32 loc(#loc)
                                                      } loc(#loc)
                                                      scf.if %true {
                                                        scf.execute_region {
                                                          %30 = arith.mulf %28, %28 : f32 loc(#loc130)
                                                          %31 = affine.load %alloca_0[0] : memref<1xf32> loc(#loc131)
                                                          %32 = arith.addf %31, %30 : f32 loc(#loc131)
                                                          affine.store %32, %alloca_0[0] : memref<1xf32> loc(#loc131)
                                                          scf.yield loc(#loc)
                                                        } loc(#loc)
                                                      } loc(#loc)
                                                      %29 = scf.if %true -> (i32) {
                                                        %30 = scf.execute_region -> i32 {
                                                          %31 = arith.addi %arg20, %c1_i32 : i32 loc(#loc93)
                                                          scf.yield %31 : i32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %30 : i32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg20 : i32 loc(#loc)
                                                      } loc(#loc)
                                                      scf.yield %27, %29, %28 : f32, i32, f32 loc(#loc120)
                                                    } loc(#loc119)
                                                    scf.yield %26#0, %26#1, %26#2 : f32, i32, f32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %24#0, %24#1, %24#2 : f32, i32, f32 loc(#loc)
                                                } else {
                                                  scf.yield %arg15, %arg16, %19 : f32, i32, f32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %23#0, %23#1, %23#2 : f32, i32, f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %22#0, %22#1, %22#2 : f32, i32, f32 loc(#loc)
                                            } else {
                                              scf.yield %arg15, %arg16, %19 : f32, i32, f32 loc(#loc)
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
                                                    scf.while : () -> () {
                                                      %22 = scf.execute_region -> i1 {
                                                        scf.if %true {
                                                          scf.execute_region {
                                                            %23 = arith.addi %arg9, %arg14 : i32 loc(#loc132)
                                                            %24 = arith.divsi %23, %c2_i32 : i32 loc(#loc133)
                                                            %25 = arith.index_cast %24 : i32 to index loc(#loc134)
                                                            %26 = "polygeist.subindex"(%arg3, %25) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc135)
                                                            %27 = "polygeist.memref2pointer"(%alloca_0) : (memref<1xf32>) -> !llvm.ptr loc(#loc136)
                                                            %28 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?xi32> loc(#loc136)
                                                            %29 = affine.load %28[0] : memref<?xi32> loc(#loc137)
                                                            affine.store %29, %26[0] : memref<?xi32> loc(#loc138)
                                                            scf.yield loc(#loc)
                                                          } loc(#loc)
                                                        } loc(#loc)
                                                        cf.br ^bb1 loc(#loc139)
                                                      ^bb1:  // pred: ^bb0
                                                        scf.yield %false : i1 loc(#loc)
                                                      } loc(#loc)
                                                      scf.condition(%22) loc(#loc139)
                                                    } do {
                                                      scf.yield loc(#loc139)
                                                    } loc(#loc)
                                                    scf.yield loc(#loc)
                                                  } loc(#loc)
                                                } loc(#loc)
                                                scf.yield loc(#loc)
                                              } loc(#loc)
                                            } loc(#loc)
                                            %21 = scf.if %true -> (i32) {
                                              %22 = scf.execute_region -> i32 {
                                                %23 = arith.addi %arg18, %c2_i32 : i32 loc(#loc140)
                                                scf.yield %23 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %22 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg18 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %20#0, %20#1, %20#2, %21 : f32, i32, f32, i32 loc(#loc117)
                                          } loc(#loc116)
                                          scf.yield %18#0, %18#1, %18#2, %18#3 : f32, i32, f32, i32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %16#0, %16#1, %16#2, %16#3 : f32, i32, f32, i32 loc(#loc)
                                      } else {
                                        scf.yield %arg10, %arg11, %arg12, %arg13 : f32, i32, f32, i32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %15#0, %15#1, %15#2, %15#3 : f32, i32, f32, i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %14#0, %14#1, %14#2, %14#3 : f32, i32, f32, i32 loc(#loc)
                                  } else {
                                    scf.yield %arg10, %arg11, %arg12, %arg13 : f32, i32, f32, i32 loc(#loc)
                                  } loc(#loc)
                                  %13 = scf.if %true -> (i32) {
                                    %14 = scf.execute_region -> i32 {
                                      %15 = arith.addi %arg14, %c1_i32 : i32 loc(#loc141)
                                      scf.yield %15 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %14 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %12#0, %12#1, %12#2, %12#3, %13 : f32, i32, f32, i32, i32 loc(#loc110)
                                } loc(#loc95)
                                scf.yield %11#0, %11#1, %11#2, %11#3, %11#4 : f32, i32, f32, i32, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %9#0, %9#1, %9#2, %9#3, %9#4 : f32, i32, f32, i32, i32 loc(#loc)
                            } else {
                              scf.yield %arg4, %arg5, %arg6, %arg7, %arg8 : f32, i32, f32, i32, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %8#0, %8#1, %8#2, %8#3, %8#4 : f32, i32, f32, i32, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %7#0, %7#1, %7#2, %7#3, %7#4 : f32, i32, f32, i32, i32 loc(#loc)
                        } else {
                          scf.yield %arg4, %arg5, %arg6, %arg7, %arg8 : f32, i32, f32, i32, i32 loc(#loc)
                        } loc(#loc)
                        %6 = scf.if %true -> (i32) {
                          %7 = scf.execute_region -> i32 {
                            %8 = arith.addi %arg9, %c2_i32 : i32 loc(#loc142)
                            scf.yield %8 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %7 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5#0, %5#1, %5#2, %5#3, %5#4, %6 : f32, i32, f32, i32, i32, i32 loc(#loc107)
                      } loc(#loc105)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc102)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc143)
  } loc(#loc91)
  func.func @store(%arg0: i32 loc("knn.cpp":41:6), %arg1: i32 loc("knn.cpp":41:6), %arg2: memref<?xi32> loc("knn.cpp":41:6), %arg3: memref<?xi32> loc("knn.cpp":41:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32_i32 = arith.constant 32 : i32 loc(#loc145)
    %c1_i32 = arith.constant 1 : i32 loc(#loc146)
    %c16_i32 = arith.constant 16 : i32 loc(#loc70)
    %c512_i32 = arith.constant 512 : i32 loc(#loc71)
    %c0_i32 = arith.constant 0 : i32 loc(#loc147)
    %true = arith.constant true loc(#loc148)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc149)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %1 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc147)
            scf.if %1 {
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = scf.if %true -> (i32) {
                    %5 = scf.execute_region -> i32 {
                      %6 = arith.muli %arg1, %c512_i32 : i32 loc(#loc151)
                      %7 = arith.divsi %6, %c16_i32 : i32 loc(#loc152)
                      scf.yield %7 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %5 : i32 loc(#loc)
                  } else {
                    scf.yield %0 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %0 : i32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc153)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc154)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.if %true -> (i32) {
                        scf.execute_region {
                          scf.yield loc(#loc)
                        } loc(#loc)
                        scf.yield %c0_i32 : i32 loc(#loc)
                      } else {
                        scf.yield %0 : i32 loc(#loc)
                      } loc(#loc)
                      %4 = scf.while (%arg4 = %3) : (i32) -> i32 {
                        %5 = arith.cmpi slt, %arg4, %c32_i32 : i32 loc(#loc155)
                        scf.condition(%5) %arg4 : i32 loc(#loc156)
                      } do {
                      ^bb0(%arg4: i32 loc("./knn.h":17:30)):
                        scf.if %true {
                          scf.execute_region {
                            %6 = arith.addi %2, %arg4 : i32 loc(#loc157)
                            %7 = arith.index_cast %6 : i32 to index loc(#loc158)
                            %8 = "polygeist.subindex"(%arg3, %7) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc159)
                            %9 = arith.index_cast %arg4 : i32 to index loc(#loc160)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc161)
                            %11 = affine.load %10[0] : memref<?xi32> loc(#loc161)
                            affine.store %11, %8[0] : memref<?xi32> loc(#loc162)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %5 = scf.if %true -> (i32) {
                          %6 = scf.execute_region -> i32 {
                            %7 = arith.addi %arg4, %c1_i32 : i32 loc(#loc146)
                            scf.yield %7 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %6 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5 : i32 loc(#loc156)
                      } loc(#loc71)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc150)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc163)
  } loc(#loc144)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("knn.cpp":79:60)
#loc4 = loc("knn.cpp":77:56)
#loc5 = loc("./knn.h":18:45)
#loc6 = loc("knn.cpp":73:53)
#loc7 = loc("knn.cpp":73:34)
#loc8 = loc("knn.cpp":52:1)
#loc9 = loc("knn.cpp":80:3)
#loc10 = loc("knn.cpp":71:9)
#loc11 = loc("knn.cpp":70:9)
#loc12 = loc("knn.cpp":68:8)
#loc13 = loc("knn.cpp":67:8)
#loc14 = loc("knn.cpp":65:5)
#loc15 = loc("knn.cpp":65:1)
#loc16 = loc("knn.cpp":67:1)
#loc17 = loc("knn.cpp":68:1)
#loc18 = loc("knn.cpp":70:1)
#loc19 = loc("knn.cpp":71:1)
#loc20 = loc("knn.cpp":73:1)
#loc21 = loc("knn.cpp":73:6)
#loc22 = loc("knn.cpp":73:39)
#loc23 = loc("knn.cpp":73:23)
#loc24 = loc("knn.cpp":74:21)
#loc25 = loc("knn.cpp":74:3)
#loc26 = loc("knn.cpp":74:25)
#loc27 = loc("knn.cpp":74:23)
#loc28 = loc("knn.cpp":77:1)
#loc29 = loc("knn.cpp":77:6)
#loc31 = loc("knn.cpp":77:46)
#loc32 = loc("knn.cpp":77:16)
#loc33 = loc("knn.cpp":78:28)
#loc34 = loc("knn.cpp":78:33)
#loc35 = loc("knn.cpp":78:45)
#loc36 = loc("knn.cpp":78:19)
#loc37 = loc("knn.cpp":79:31)
#loc38 = loc("knn.cpp":79:36)
#loc39 = loc("knn.cpp":79:48)
#loc40 = loc("knn.cpp":79:22)
#loc41 = loc("knn.cpp":80:29)
#loc42 = loc("knn.cpp":80:34)
#loc43 = loc("knn.cpp":80:20)
#loc44 = loc("knn.cpp":82:19)
#loc45 = loc("knn.cpp":82:23)
#loc46 = loc("knn.cpp":82:6)
#loc47 = loc("knn.cpp":83:46)
#loc48 = loc("knn.cpp":83:7)
#loc49 = loc("knn.cpp":84:26)
#loc50 = loc("knn.cpp":84:44)
#loc51 = loc("knn.cpp":84:65)
#loc52 = loc("knn.cpp":84:4)
#loc53 = loc("knn.cpp":85:30)
#loc54 = loc("knn.cpp":85:34)
#loc55 = loc("knn.cpp":85:4)
#loc56 = loc("knn.cpp":88:46)
#loc57 = loc("knn.cpp":88:7)
#loc58 = loc("knn.cpp":89:26)
#loc59 = loc("knn.cpp":89:44)
#loc60 = loc("knn.cpp":89:65)
#loc61 = loc("knn.cpp":89:4)
#loc62 = loc("knn.cpp":90:30)
#loc63 = loc("knn.cpp":90:34)
#loc64 = loc("knn.cpp":90:4)
#loc65 = loc("knn.cpp":77:60)
#loc66 = loc("knn.cpp":95:1)
#loc68 = loc("knn.cpp":8:70)
#loc69 = loc("knn.cpp":8:85)
#loc70 = loc("./knn.h":44:35)
#loc72 = loc("knn.cpp":6:9)
#loc73 = loc("knn.cpp":3:1)
#loc74 = loc("knn.cpp":8:28)
#loc75 = loc("knn.cpp":6:5)
#loc76 = loc("knn.cpp":7:34)
#loc77 = loc("knn.cpp":7:53)
#loc78 = loc("knn.cpp":7:67)
#loc79 = loc("knn.cpp":8:1)
#loc80 = loc("knn.cpp":8:12)
#loc82 = loc("knn.cpp":8:40)
#loc83 = loc("knn.cpp":8:23)
#loc84 = loc("knn.cpp":9:32)
#loc85 = loc("knn.cpp":9:13)
#loc86 = loc("knn.cpp":9:57)
#loc87 = loc("knn.cpp":9:59)
#loc88 = loc("knn.cpp":9:36)
#loc89 = loc("knn.cpp":9:34)
#loc90 = loc("knn.cpp":12:1)
#loc92 = loc("knn.cpp":19:60)
#loc93 = loc("knn.cpp":24:57)
#loc94 = loc("knn.cpp":22:43)
#loc95 = loc("./knn.h":19:27)
#loc96 = loc("knn.cpp":17:9)
#loc97 = loc("knn.cpp":14:1)
#loc98 = loc("knn.cpp":32:21)
#loc100 = loc("knn.cpp":27:25)
#loc101 = loc("knn.cpp":23:21)
#loc102 = loc("knn.cpp":17:5)
#loc103 = loc("knn.cpp":18:9)
#loc104 = loc("knn.cpp":19:1)
#loc106 = loc("knn.cpp":19:30)
#loc107 = loc("knn.cpp":19:12)
#loc108 = loc("knn.cpp":20:1)
#loc109 = loc("knn.cpp":20:34)
#loc110 = loc("knn.cpp":20:16)
#loc115 = loc("knn.cpp":21:1)
#loc116 = loc("knn.cpp":21:38)
#loc117 = loc("knn.cpp":21:20)
#loc118 = loc("knn.cpp":24:1)
#loc119 = loc("knn.cpp":24:42)
#loc120 = loc("knn.cpp":24:24)
#loc121 = loc("knn.cpp":27:78)
#loc122 = loc("knn.cpp":27:80)
#loc123 = loc("knn.cpp":27:59)
#loc124 = loc("./knn.h":45:42)
#loc125 = loc("knn.cpp":28:54)
#loc126 = loc("knn.cpp":28:52)
#loc127 = loc("knn.cpp":29:80)
#loc128 = loc("knn.cpp":29:62)
#loc129 = loc("knn.cpp":29:60)
#loc130 = loc("knn.cpp":30:46)
#loc131 = loc("knn.cpp":30:29)
#loc132 = loc("knn.cpp":34:48)
#loc133 = loc("knn.cpp":34:51)
#loc134 = loc("knn.cpp":34:53)
#loc135 = loc("knn.cpp":34:31)
#loc136 = loc("knn.cpp":34:83)
#loc137 = loc("./knn.h":46:53)
#loc138 = loc("./knn.h":46:51)
#loc139 = loc("./knn.h":46:42)
#loc140 = loc("knn.cpp":21:55)
#loc141 = loc("knn.cpp":20:51)
#loc142 = loc("knn.cpp":19:76)
#loc143 = loc("knn.cpp":39:1)
#loc145 = loc("knn.cpp":46:59)
#loc146 = loc("knn.cpp":46:74)
#loc147 = loc("knn.cpp":44:9)
#loc148 = loc("knn.cpp":41:1)
#loc149 = loc("knn.cpp":46:29)
#loc150 = loc("knn.cpp":44:5)
#loc151 = loc("knn.cpp":45:35)
#loc152 = loc("knn.cpp":45:54)
#loc153 = loc("knn.cpp":46:1)
#loc154 = loc("knn.cpp":46:12)
#loc155 = loc("knn.cpp":46:41)
#loc156 = loc("knn.cpp":46:24)
#loc157 = loc("knn.cpp":47:31)
#loc158 = loc("knn.cpp":47:33)
#loc159 = loc("knn.cpp":47:13)
#loc160 = loc("knn.cpp":47:53)
#loc161 = loc("knn.cpp":47:37)
#loc162 = loc("knn.cpp":47:35)
#loc163 = loc("knn.cpp":50:1)
