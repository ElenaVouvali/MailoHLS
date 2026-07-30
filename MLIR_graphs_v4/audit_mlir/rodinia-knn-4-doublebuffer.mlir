#loc1 = loc("knn.cpp":44:6)
#loc2 = loc("./knn.h":15:25)
#loc30 = loc("./knn.h":16:39)
#loc67 = loc("knn.cpp":3:6)
#loc69 = loc("knn.cpp":8:58)
#loc70 = loc("./knn.h":17:30)
#loc88 = loc("knn.cpp":14:6)
#loc94 = loc("knn.cpp":23:25)
#loc95 = loc("knn.cpp":19:9)
#loc104 = loc("knn.cpp":21:21)
#loc105 = loc("knn.cpp":18:9)
#loc126 = loc("knn.cpp":33:6)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32> loc("knn.cpp":44:6), %arg1: memref<2097152xf32> loc("knn.cpp":44:6), %arg2: memref<1048576xf32> loc("knn.cpp":44:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %alloca = memref.alloca() : memref<512xf32> loc(#loc10)
    %alloca_0 = memref.alloca() : memref<512xf32> loc(#loc11)
    %alloca_1 = memref.alloca() : memref<1024xf32> loc(#loc12)
    %alloca_2 = memref.alloca() : memref<1024xf32> loc(#loc13)
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
                            %cast = memref.cast %alloca_2 : memref<1024xf32> to memref<?xf32> loc(#loc47)
                            %cast_4 = memref.cast %arg1 : memref<2097152xf32> to memref<?xf32> loc(#loc48)
                            func.call @load(%4, %arg6, %cast_4, %cast) : (i32, i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc48)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_3 : memref<2xf32> to memref<?xf32> loc(#loc49)
                            %cast_4 = memref.cast %alloca_1 : memref<1024xf32> to memref<?xf32> loc(#loc50)
                            %cast_5 = memref.cast %alloca : memref<512xf32> to memref<?xf32> loc(#loc51)
                            func.call @compute(%5, %cast, %cast_4, %cast_5) : (i32, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> () loc(#loc52)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.addi %arg6, %c-2_i32 : i32 loc(#loc53)
                            %cast = memref.cast %alloca_0 : memref<512xf32> to memref<?xf32> loc(#loc54)
                            %cast_4 = memref.cast %arg2 : memref<1048576xf32> to memref<?xf32> loc(#loc55)
                            func.call @store(%6, %10, %cast, %cast_4) : (i32, i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc55)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } else {
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_1 : memref<1024xf32> to memref<?xf32> loc(#loc56)
                            %cast_4 = memref.cast %arg1 : memref<2097152xf32> to memref<?xf32> loc(#loc57)
                            func.call @load(%4, %arg6, %cast_4, %cast) : (i32, i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc57)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_3 : memref<2xf32> to memref<?xf32> loc(#loc58)
                            %cast_4 = memref.cast %alloca_2 : memref<1024xf32> to memref<?xf32> loc(#loc59)
                            %cast_5 = memref.cast %alloca_0 : memref<512xf32> to memref<?xf32> loc(#loc60)
                            func.call @compute(%5, %cast, %cast_4, %cast_5) : (i32, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> () loc(#loc61)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.addi %arg6, %c-2_i32 : i32 loc(#loc62)
                            %cast = memref.cast %alloca : memref<512xf32> to memref<?xf32> loc(#loc63)
                            %cast_4 = memref.cast %arg2 : memref<1048576xf32> to memref<?xf32> loc(#loc64)
                            func.call @store(%6, %10, %cast, %cast_4) : (i32, i32, memref<?xf32>, memref<?xf32>) -> () loc(#loc64)
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
  func.func @load(%arg0: i32 loc("knn.cpp":3:6), %arg1: i32 loc("knn.cpp":3:6), %arg2: memref<?xf32> loc("knn.cpp":3:6), %arg3: memref<?xf32> loc("knn.cpp":3:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc68)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc69)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %c512_i32 = arith.constant 512 : i32 loc(#loc70)
    %c0_i32 = arith.constant 0 : i32 loc(#loc71)
    %true = arith.constant true loc(#loc72)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc73)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %1 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc71)
            scf.if %1 {
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = scf.if %true -> (i32) {
                    %5 = scf.execute_region -> i32 {
                      %6 = arith.muli %arg1, %c512_i32 : i32 loc(#loc75)
                      %7 = arith.muli %6, %c2_i32 : i32 loc(#loc76)
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
                  cf.br ^bb1 loc(#loc77)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc78)
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
                        %5 = arith.cmpi slt, %arg4, %c1024_i32 : i32 loc(#loc79)
                        scf.condition(%5) %arg4 : i32 loc(#loc80)
                      } do {
                      ^bb0(%arg4: i32 loc("knn.cpp":8:58)):
                        scf.if %true {
                          scf.execute_region {
                            %6 = arith.index_cast %arg4 : i32 to index loc(#loc81)
                            %7 = "polygeist.subindex"(%arg3, %6) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc82)
                            %8 = arith.addi %2, %arg4 : i32 loc(#loc83)
                            %9 = arith.index_cast %8 : i32 to index loc(#loc84)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc85)
                            %11 = affine.load %10[0] : memref<?xf32> loc(#loc85)
                            affine.store %11, %7[0] : memref<?xf32> loc(#loc86)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %5 = scf.if %true -> (i32) {
                          %6 = scf.execute_region -> i32 {
                            %7 = arith.addi %arg4, %c1_i32 : i32 loc(#loc68)
                            scf.yield %7 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %6 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5 : i32 loc(#loc80)
                      } loc(#loc69)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc74)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc87)
  } loc(#loc67)
  func.func @compute(%arg0: i32 loc("knn.cpp":14:6), %arg1: memref<?xf32> loc("knn.cpp":14:6), %arg2: memref<?xf32> loc("knn.cpp":14:6), %arg3: memref<?xf32> loc("knn.cpp":14:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc89)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc90)
    %c2_i32 = arith.constant 2 : i32 loc(#loc91)
    %c512_i32 = arith.constant 512 : i32 loc(#loc70)
    %c0_i32 = arith.constant 0 : i32 loc(#loc92)
    %true = arith.constant true loc(#loc93)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc94)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc95)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %2 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc92)
            scf.if %2 {
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
                  cf.br ^bb1 loc(#loc97)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc98)
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
                      %4:5 = scf.while (%arg4 = %0, %arg5 = %0, %arg6 = %3, %arg7 = %1, %arg8 = %1) : (i32, i32, i32, f32, f32) -> (i32, i32, i32, f32, f32) {
                        %5 = arith.cmpi slt, %arg6, %c512_i32 : i32 loc(#loc99)
                        scf.condition(%5) %arg4, %arg5, %arg6, %arg7, %arg8 : i32, i32, i32, f32, f32 loc(#loc100)
                      } do {
                      ^bb0(%arg4: i32 loc("./knn.h":17:30), %arg5: i32 loc("./knn.h":17:30), %arg6: i32 loc("./knn.h":17:30), %arg7: f32 loc("./knn.h":17:30), %arg8: f32 loc("./knn.h":17:30)):
                        %5:4 = scf.if %true -> (i32, i32, f32, f32) {
                          %7:4 = scf.execute_region -> (i32, i32, f32, f32) {
                            cf.br ^bb1 loc(#loc101)
                          ^bb1:  // pred: ^bb0
                            %8:4 = scf.if %true -> (i32, i32, f32, f32) {
                              %9:4 = scf.execute_region -> (i32, i32, f32, f32) {
                                %10 = scf.if %true -> (i32) {
                                  scf.execute_region {
                                    scf.yield loc(#loc)
                                  } loc(#loc)
                                  scf.yield %c0_i32 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg5 : i32 loc(#loc)
                                } loc(#loc)
                                %11:4 = scf.while (%arg9 = %arg4, %arg10 = %10, %arg11 = %arg7, %arg12 = %arg8) : (i32, i32, f32, f32) -> (i32, i32, f32, f32) {
                                  %12 = arith.cmpi slt, %arg10, %c2_i32 : i32 loc(#loc102)
                                  scf.condition(%12) %arg9, %arg10, %arg11, %arg12 : i32, i32, f32, f32 loc(#loc103)
                                } do {
                                ^bb0(%arg9: i32 loc("knn.cpp":23:25), %arg10: i32 loc("knn.cpp":21:21), %arg11: f32 loc("knn.cpp":19:9), %arg12: f32 loc("knn.cpp":18:9)):
                                  %12 = scf.if %true -> (f32) {
                                    %15 = scf.execute_region -> f32 {
                                      scf.yield %cst : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %15 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg12 : f32 loc(#loc)
                                  } loc(#loc)
                                  %13:3 = scf.if %true -> (i32, f32, f32) {
                                    %15:3 = scf.execute_region -> (i32, f32, f32) {
                                      cf.br ^bb1 loc(#loc106)
                                    ^bb1:  // pred: ^bb0
                                      %16:3 = scf.if %true -> (i32, f32, f32) {
                                        %17:3 = scf.execute_region -> (i32, f32, f32) {
                                          %18 = scf.if %true -> (i32) {
                                            scf.execute_region {
                                              scf.yield loc(#loc)
                                            } loc(#loc)
                                            scf.yield %c0_i32 : i32 loc(#loc)
                                          } else {
                                            scf.yield %arg9 : i32 loc(#loc)
                                          } loc(#loc)
                                          %19:3 = scf.while (%arg13 = %18, %arg14 = %arg11, %arg15 = %12) : (i32, f32, f32) -> (i32, f32, f32) {
                                            %20 = arith.cmpi slt, %arg13, %c2_i32 : i32 loc(#loc107)
                                            scf.condition(%20) %arg13, %arg14, %arg15 : i32, f32, f32 loc(#loc108)
                                          } do {
                                          ^bb0(%arg13: i32 loc("knn.cpp":23:25), %arg14: f32 loc("knn.cpp":19:9), %arg15: f32 loc("knn.cpp":18:9)):
                                            %20 = scf.if %true -> (f32) {
                                              %23 = scf.execute_region -> f32 {
                                                %24 = arith.addi %arg6, %arg10 : i32 loc(#loc109)
                                                %25 = arith.muli %24, %c2_i32 : i32 loc(#loc110)
                                                %26 = arith.addi %25, %arg13 : i32 loc(#loc111)
                                                %27 = arith.index_cast %26 : i32 to index loc(#loc112)
                                                %28 = "polygeist.subindex"(%arg2, %27) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc113)
                                                %29 = affine.load %28[0] : memref<?xf32> loc(#loc113)
                                                %30 = arith.index_cast %arg13 : i32 to index loc(#loc114)
                                                %31 = "polygeist.subindex"(%arg1, %30) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc115)
                                                %32 = affine.load %31[0] : memref<?xf32> loc(#loc115)
                                                %33 = arith.subf %29, %32 : f32 loc(#loc116)
                                                scf.yield %33 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %23 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg14 : f32 loc(#loc)
                                            } loc(#loc)
                                            %21 = scf.if %true -> (f32) {
                                              %23 = scf.execute_region -> f32 {
                                                %24 = arith.mulf %20, %20 : f32 loc(#loc117)
                                                %25 = arith.addf %arg15, %24 : f32 loc(#loc118)
                                                scf.yield %25 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %23 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg15 : f32 loc(#loc)
                                            } loc(#loc)
                                            %22 = scf.if %true -> (i32) {
                                              %23 = scf.execute_region -> i32 {
                                                %24 = arith.addi %arg13, %c1_i32 : i32 loc(#loc89)
                                                scf.yield %24 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %23 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg13 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %22, %20, %21 : i32, f32, f32 loc(#loc108)
                                          } loc(#loc107)
                                          scf.yield %19#0, %19#1, %19#2 : i32, f32, f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %17#0, %17#1, %17#2 : i32, f32, f32 loc(#loc)
                                      } else {
                                        scf.yield %arg9, %arg11, %12 : i32, f32, f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %16#0, %16#1, %16#2 : i32, f32, f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %15#0, %15#1, %15#2 : i32, f32, f32 loc(#loc)
                                  } else {
                                    scf.yield %arg9, %arg11, %12 : i32, f32, f32 loc(#loc)
                                  } loc(#loc)
                                  scf.if %true {
                                    scf.execute_region {
                                      %15 = arith.addi %arg6, %arg10 : i32 loc(#loc119)
                                      %16 = arith.index_cast %15 : i32 to index loc(#loc120)
                                      %17 = "polygeist.subindex"(%arg3, %16) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc121)
                                      affine.store %13#2, %17[0] : memref<?xf32> loc(#loc122)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  %14 = scf.if %true -> (i32) {
                                    %15 = scf.execute_region -> i32 {
                                      %16 = arith.addi %arg10, %c1_i32 : i32 loc(#loc123)
                                      scf.yield %16 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %15 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg10 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %13#0, %14, %13#1, %13#2 : i32, i32, f32, f32 loc(#loc103)
                                } loc(#loc91)
                                scf.yield %11#0, %11#1, %11#2, %11#3 : i32, i32, f32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %9#0, %9#1, %9#2, %9#3 : i32, i32, f32, f32 loc(#loc)
                            } else {
                              scf.yield %arg4, %arg5, %arg7, %arg8 : i32, i32, f32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %8#0, %8#1, %8#2, %8#3 : i32, i32, f32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %7#0, %7#1, %7#2, %7#3 : i32, i32, f32, f32 loc(#loc)
                        } else {
                          scf.yield %arg4, %arg5, %arg7, %arg8 : i32, i32, f32, f32 loc(#loc)
                        } loc(#loc)
                        %6 = scf.if %true -> (i32) {
                          %7 = scf.execute_region -> i32 {
                            %8 = arith.addi %arg6, %c2_i32 : i32 loc(#loc124)
                            scf.yield %8 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %7 : i32 loc(#loc)
                        } else {
                          scf.yield %arg6 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5#0, %5#1, %6, %5#2, %5#3 : i32, i32, i32, f32, f32 loc(#loc100)
                      } loc(#loc70)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc96)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc125)
  } loc(#loc88)
  func.func @store(%arg0: i32 loc("knn.cpp":33:6), %arg1: i32 loc("knn.cpp":33:6), %arg2: memref<?xf32> loc("knn.cpp":33:6), %arg3: memref<?xf32> loc("knn.cpp":33:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc127)
    %c512_i32 = arith.constant 512 : i32 loc(#loc70)
    %c0_i32 = arith.constant 0 : i32 loc(#loc128)
    %true = arith.constant true loc(#loc129)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc130)
    scf.if %true {
      scf.execute_region {
        scf.if %true {
          scf.execute_region {
            %1 = arith.cmpi ne, %arg0, %c0_i32 : i32 loc(#loc128)
            scf.if %1 {
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = scf.if %true -> (i32) {
                    %5 = scf.execute_region -> i32 {
                      %6 = arith.muli %arg1, %c512_i32 : i32 loc(#loc132)
                      scf.yield %6 : i32 loc(#loc)
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
                  cf.br ^bb1 loc(#loc133)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc134)
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
                        %5 = arith.cmpi slt, %arg4, %c512_i32 : i32 loc(#loc135)
                        scf.condition(%5) %arg4 : i32 loc(#loc136)
                      } do {
                      ^bb0(%arg4: i32 loc("./knn.h":17:30)):
                        scf.if %true {
                          scf.execute_region {
                            %6 = arith.addi %2, %arg4 : i32 loc(#loc137)
                            %7 = arith.index_cast %6 : i32 to index loc(#loc138)
                            %8 = "polygeist.subindex"(%arg3, %7) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc139)
                            %9 = arith.index_cast %arg4 : i32 to index loc(#loc140)
                            %10 = "polygeist.subindex"(%arg2, %9) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc141)
                            %11 = affine.load %10[0] : memref<?xf32> loc(#loc141)
                            affine.store %11, %8[0] : memref<?xf32> loc(#loc142)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %5 = scf.if %true -> (i32) {
                          %6 = scf.execute_region -> i32 {
                            %7 = arith.addi %arg4, %c1_i32 : i32 loc(#loc127)
                            scf.yield %7 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %6 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %5 : i32 loc(#loc136)
                      } loc(#loc70)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
            } loc(#loc131)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc143)
  } loc(#loc126)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("knn.cpp":71:60)
#loc4 = loc("knn.cpp":69:56)
#loc5 = loc("./knn.h":18:45)
#loc6 = loc("knn.cpp":65:53)
#loc7 = loc("knn.cpp":65:34)
#loc8 = loc("knn.cpp":44:1)
#loc9 = loc("knn.cpp":72:3)
#loc10 = loc("knn.cpp":63:9)
#loc11 = loc("knn.cpp":62:8)
#loc12 = loc("knn.cpp":60:8)
#loc13 = loc("knn.cpp":59:8)
#loc14 = loc("knn.cpp":57:5)
#loc15 = loc("knn.cpp":57:1)
#loc16 = loc("knn.cpp":59:1)
#loc17 = loc("knn.cpp":60:1)
#loc18 = loc("knn.cpp":62:1)
#loc19 = loc("knn.cpp":63:1)
#loc20 = loc("knn.cpp":65:1)
#loc21 = loc("knn.cpp":65:6)
#loc22 = loc("knn.cpp":65:39)
#loc23 = loc("knn.cpp":65:23)
#loc24 = loc("knn.cpp":66:21)
#loc25 = loc("knn.cpp":66:3)
#loc26 = loc("knn.cpp":66:25)
#loc27 = loc("knn.cpp":66:23)
#loc28 = loc("knn.cpp":69:1)
#loc29 = loc("knn.cpp":69:6)
#loc31 = loc("knn.cpp":69:46)
#loc32 = loc("knn.cpp":69:16)
#loc33 = loc("knn.cpp":70:28)
#loc34 = loc("knn.cpp":70:33)
#loc35 = loc("knn.cpp":70:45)
#loc36 = loc("knn.cpp":70:19)
#loc37 = loc("knn.cpp":71:31)
#loc38 = loc("knn.cpp":71:36)
#loc39 = loc("knn.cpp":71:48)
#loc40 = loc("knn.cpp":71:22)
#loc41 = loc("knn.cpp":72:29)
#loc42 = loc("knn.cpp":72:34)
#loc43 = loc("knn.cpp":72:20)
#loc44 = loc("knn.cpp":74:19)
#loc45 = loc("knn.cpp":74:23)
#loc46 = loc("knn.cpp":74:6)
#loc47 = loc("knn.cpp":75:46)
#loc48 = loc("knn.cpp":75:7)
#loc49 = loc("knn.cpp":76:26)
#loc50 = loc("knn.cpp":76:44)
#loc51 = loc("knn.cpp":76:65)
#loc52 = loc("knn.cpp":76:4)
#loc53 = loc("knn.cpp":77:30)
#loc54 = loc("knn.cpp":77:34)
#loc55 = loc("knn.cpp":77:4)
#loc56 = loc("knn.cpp":80:46)
#loc57 = loc("knn.cpp":80:7)
#loc58 = loc("knn.cpp":81:26)
#loc59 = loc("knn.cpp":81:44)
#loc60 = loc("knn.cpp":81:65)
#loc61 = loc("knn.cpp":81:4)
#loc62 = loc("knn.cpp":82:30)
#loc63 = loc("knn.cpp":82:34)
#loc64 = loc("knn.cpp":82:4)
#loc65 = loc("knn.cpp":69:60)
#loc66 = loc("knn.cpp":87:1)
#loc68 = loc("knn.cpp":8:72)
#loc71 = loc("knn.cpp":6:9)
#loc72 = loc("knn.cpp":3:1)
#loc73 = loc("knn.cpp":8:28)
#loc74 = loc("knn.cpp":6:5)
#loc75 = loc("knn.cpp":7:34)
#loc76 = loc("knn.cpp":7:53)
#loc77 = loc("knn.cpp":8:1)
#loc78 = loc("knn.cpp":8:12)
#loc79 = loc("knn.cpp":8:40)
#loc80 = loc("knn.cpp":8:23)
#loc81 = loc("knn.cpp":9:32)
#loc82 = loc("knn.cpp":9:13)
#loc83 = loc("knn.cpp":9:57)
#loc84 = loc("knn.cpp":9:59)
#loc85 = loc("knn.cpp":9:36)
#loc86 = loc("knn.cpp":9:34)
#loc87 = loc("knn.cpp":12:1)
#loc89 = loc("knn.cpp":23:53)
#loc90 = loc("knn.cpp":22:23)
#loc91 = loc("./knn.h":19:27)
#loc92 = loc("knn.cpp":17:9)
#loc93 = loc("knn.cpp":14:1)
#loc96 = loc("knn.cpp":17:5)
#loc97 = loc("knn.cpp":20:1)
#loc98 = loc("knn.cpp":20:12)
#loc99 = loc("knn.cpp":20:44)
#loc100 = loc("knn.cpp":20:26)
#loc101 = loc("knn.cpp":21:1)
#loc102 = loc("knn.cpp":21:34)
#loc103 = loc("knn.cpp":21:16)
#loc106 = loc("knn.cpp":23:1)
#loc107 = loc("knn.cpp":23:38)
#loc108 = loc("knn.cpp":23:20)
#loc109 = loc("knn.cpp":24:57)
#loc110 = loc("knn.cpp":24:60)
#loc111 = loc("knn.cpp":24:72)
#loc112 = loc("knn.cpp":24:74)
#loc113 = loc("knn.cpp":24:37)
#loc114 = loc("knn.cpp":24:96)
#loc115 = loc("knn.cpp":24:78)
#loc116 = loc("knn.cpp":24:76)
#loc117 = loc("knn.cpp":25:41)
#loc118 = loc("knn.cpp":25:25)
#loc119 = loc("knn.cpp":27:33)
#loc120 = loc("knn.cpp":27:35)
#loc121 = loc("knn.cpp":27:17)
#loc122 = loc("knn.cpp":27:37)
#loc123 = loc("knn.cpp":21:51)
#loc124 = loc("knn.cpp":20:65)
#loc125 = loc("knn.cpp":31:1)
#loc127 = loc("knn.cpp":38:61)
#loc128 = loc("knn.cpp":36:9)
#loc129 = loc("knn.cpp":33:1)
#loc130 = loc("knn.cpp":38:29)
#loc131 = loc("knn.cpp":36:5)
#loc132 = loc("knn.cpp":37:35)
#loc133 = loc("knn.cpp":38:1)
#loc134 = loc("knn.cpp":38:12)
#loc135 = loc("knn.cpp":38:41)
#loc136 = loc("knn.cpp":38:24)
#loc137 = loc("knn.cpp":39:31)
#loc138 = loc("knn.cpp":39:33)
#loc139 = loc("knn.cpp":39:13)
#loc140 = loc("knn.cpp":39:53)
#loc141 = loc("knn.cpp":39:37)
#loc142 = loc("knn.cpp":39:35)
#loc143 = loc("knn.cpp":42:1)
