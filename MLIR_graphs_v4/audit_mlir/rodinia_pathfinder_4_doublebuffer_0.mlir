#loc1 = loc("pathfinder.cpp":40:6)
#loc6 = loc("pathfinder.cpp":58:38)
#loc7 = loc("./pathfinder.h":14:19)
#loc8 = loc("./pathfinder.h":12:14)
#loc64 = loc("pathfinder.cpp":34:6)
#loc78 = loc("pathfinder.cpp":5:6)
#loc81 = loc("pathfinder.cpp":27:37)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<1048576xi32> loc("pathfinder.cpp":40:6), %arg1: memref<1024xi32> loc("pathfinder.cpp":40:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c63_i32 = arith.constant 63 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %c2_i32 = arith.constant 2 : i32 loc(#loc5)
    %c64_i32 = arith.constant 64 : i32 loc(#loc6)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c16_i64 = arith.constant 16 : i64 loc(#loc7)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc8)
    %true = arith.constant true loc(#loc9)
    %c1024 = arith.constant 1024 : index loc(#loc)
    %alloca = memref.alloca() : memref<16384xi32> loc(#loc10)
    %alloca_0 = memref.alloca() : memref<16384xi32> loc(#loc11)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc12)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc13)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc14)
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
        %0 = "polygeist.memref2pointer"(%alloca_2) : (memref<1024xi32>) -> !llvm.ptr loc(#loc18)
        %1 = "polygeist.memref2pointer"(%arg0) : (memref<1048576xi32>) -> !llvm.ptr loc(#loc19)
        %2 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc20)
        %3 = arith.index_cast %2 : index to i64 loc(#loc20)
        %4 = arith.muli %3, %c1024_i64 : i64 loc(#loc21)
        "llvm.intr.memcpy"(%0, %1, %4) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc22)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %0 = "polygeist.memref2pointer"(%alloca_0) : (memref<16384xi32>) -> !llvm.ptr loc(#loc23)
        %1 = "polygeist.subindex"(%arg0, %c1024) : (memref<1048576xi32>, index) -> memref<?xi32> loc(#loc24)
        %2 = "polygeist.memref2pointer"(%1) : (memref<?xi32>) -> !llvm.ptr loc(#loc25)
        %3 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc26)
        %4 = arith.index_cast %3 : index to i64 loc(#loc26)
        %5 = arith.muli %4, %c1024_i64 : i64 loc(#loc27)
        %6 = arith.muli %5, %c16_i64 : i64 loc(#loc28)
        "llvm.intr.memcpy"(%0, %2, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc29)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc30)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc31)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %0 = scf.while (%arg2 = %c1_i32) : (i32) -> i32 {
              %1 = arith.cmpi slt, %arg2, %c64_i32 : i32 loc(#loc32)
              scf.condition(%1) %arg2 : i32 loc(#loc33)
            } do {
            ^bb0(%arg2: i32 loc("pathfinder.cpp":58:38)):
              scf.if %true {
                scf.execute_region {
                  scf.if %true {
                    scf.execute_region {
                      %2 = arith.remsi %arg2, %c2_i32 : i32 loc(#loc34)
                      %3 = arith.cmpi eq, %2, %c0_i32 : i32 loc(#loc35)
                      scf.if %3 {
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_0 : memref<16384xi32> to memref<?xi32> loc(#loc37)
                            %cast_3 = memref.cast %arg0 : memref<1048576xi32> to memref<?xi32> loc(#loc38)
                            func.call @load(%cast, %cast_3, %arg2) : (memref<?xi32>, memref<?xi32>, i32) -> () loc(#loc38)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %cast = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32> loc(#loc39)
                            %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32> loc(#loc40)
                            %cast_4 = memref.cast %alloca : memref<16384xi32> to memref<?xi32> loc(#loc41)
                            %4 = arith.addi %arg2, %c-1_i32 : i32 loc(#loc42)
                            func.call @pathfinder_kernel(%cast, %cast_3, %cast_4, %4) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> () loc(#loc43)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } else {
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.cmpi eq, %2, %c1_i32 : i32 loc(#loc44)
                            scf.if %4 {
                              scf.if %true {
                                scf.execute_region {
                                  %cast = memref.cast %alloca : memref<16384xi32> to memref<?xi32> loc(#loc46)
                                  %cast_3 = memref.cast %arg0 : memref<1048576xi32> to memref<?xi32> loc(#loc47)
                                  func.call @load(%cast, %cast_3, %arg2) : (memref<?xi32>, memref<?xi32>, i32) -> () loc(#loc47)
                                  scf.yield loc(#loc)
                                } loc(#loc)
                              } loc(#loc)
                              scf.if %true {
                                scf.execute_region {
                                  %cast = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32> loc(#loc48)
                                  %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32> loc(#loc49)
                                  %cast_4 = memref.cast %alloca_0 : memref<16384xi32> to memref<?xi32> loc(#loc50)
                                  %5 = arith.addi %arg2, %c-1_i32 : i32 loc(#loc51)
                                  func.call @pathfinder_kernel(%cast, %cast_3, %cast_4, %5) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> () loc(#loc52)
                                  scf.yield loc(#loc)
                                } loc(#loc)
                              } loc(#loc)
                            } loc(#loc45)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } loc(#loc36)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %1 = scf.if %true -> (i32) {
                %2 = scf.execute_region -> i32 {
                  %3 = arith.addi %arg2, %c1_i32 : i32 loc(#loc53)
                  scf.yield %3 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %2 : i32 loc(#loc)
              } else {
                scf.yield %arg2 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %1 : i32 loc(#loc33)
            } loc(#loc6)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %cast = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32> loc(#loc54)
        %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32> loc(#loc55)
        %cast_4 = memref.cast %alloca : memref<16384xi32> to memref<?xi32> loc(#loc56)
        func.call @pathfinder_kernel(%cast, %cast_3, %cast_4, %c63_i32) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> () loc(#loc57)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %0 = "polygeist.memref2pointer"(%arg1) : (memref<1024xi32>) -> !llvm.ptr loc(#loc58)
        %1 = "polygeist.memref2pointer"(%alloca_2) : (memref<1024xi32>) -> !llvm.ptr loc(#loc59)
        %2 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc60)
        %3 = arith.index_cast %2 : index to i64 loc(#loc60)
        %4 = arith.muli %3, %c1024_i64 : i64 loc(#loc61)
        "llvm.intr.memcpy"(%0, %1, %4) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc62)
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
    return loc(#loc63)
  } loc(#loc1)
  func.func @load(%arg0: memref<?xi32> loc("pathfinder.cpp":34:6), %arg1: memref<?xi32> loc("pathfinder.cpp":34:6), %arg2: i32 loc("pathfinder.cpp":34:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16_i64 = arith.constant 16 : i64 loc(#loc7)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc8)
    %c1_i32 = arith.constant 1 : i32 loc(#loc65)
    %c16_i32 = arith.constant 16 : i32 loc(#loc7)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc8)
    %true = arith.constant true loc(#loc66)
    scf.if %true {
      scf.execute_region {
        %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr loc(#loc67)
        %1 = arith.muli %arg2, %c16_i32 : i32 loc(#loc68)
        %2 = arith.addi %1, %c1_i32 : i32 loc(#loc69)
        %3 = arith.muli %2, %c1024_i32 : i32 loc(#loc70)
        %4 = arith.index_cast %3 : i32 to index loc(#loc71)
        %5 = "polygeist.subindex"(%arg1, %4) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc71)
        %6 = "polygeist.memref2pointer"(%5) : (memref<?xi32>) -> !llvm.ptr loc(#loc72)
        %7 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc73)
        %8 = arith.index_cast %7 : index to i64 loc(#loc73)
        %9 = arith.muli %8, %c1024_i64 : i64 loc(#loc74)
        %10 = arith.muli %9, %c16_i64 : i64 loc(#loc75)
        "llvm.intr.memcpy"(%0, %6, %10) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc76)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc77)
  } loc(#loc64)
  func.func @pathfinder_kernel(%arg0: memref<?xi32> loc("pathfinder.cpp":5:6), %arg1: memref<?xi32> loc("pathfinder.cpp":5:6), %arg2: memref<?xi32> loc("pathfinder.cpp":5:6), %arg3: i32 loc("pathfinder.cpp":5:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc79)
    %c2_i32 = arith.constant 2 : i32 loc(#loc80)
    %c512_i32 = arith.constant 512 : i32 loc(#loc81)
    %c1023_i32 = arith.constant 1023 : i32 loc(#loc82)
    %c1_i32 = arith.constant 1 : i32 loc(#loc79)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc8)
    %c16_i32 = arith.constant 16 : i32 loc(#loc7)
    %c0_i32 = arith.constant 0 : i32 loc(#loc83)
    %true = arith.constant true loc(#loc84)
    %alloca = memref.alloca() : memref<1xi32> loc(#loc85)
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32> loc(#loc85)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc85)
    affine.store %0, %alloca[0] : memref<1xi32> loc(#loc85)
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
        cf.br ^bb1 loc(#loc86)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc87)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
              %2 = arith.cmpi slt, %arg4, %c16_i32 : i32 loc(#loc88)
              scf.condition(%2) %arg4 : i32 loc(#loc89)
            } do {
            ^bb0(%arg4: i32 loc("./pathfinder.h":14:19)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc90)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc91)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg5, %c1024_i32 : i32 loc(#loc92)
                        scf.condition(%4) %arg5 : i32 loc(#loc93)
                      } do {
                      ^bb0(%arg5: i32 loc("./pathfinder.h":12:14)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.index_cast %arg5 : i32 to index loc(#loc94)
                            %6 = "polygeist.subindex"(%arg0, %5) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc95)
                            %7 = affine.load %6[0] : memref<?xi32> loc(#loc95)
                            affine.store %7, %alloca[0] : memref<1xi32> loc(#loc96)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            scf.if %true {
                              scf.execute_region {
                                %5 = arith.cmpi sgt, %arg5, %c0_i32 : i32 loc(#loc97)
                                scf.if %5 {
                                  scf.if %true {
                                    scf.execute_region {
                                      %6 = affine.load %alloca[0] : memref<1xi32> loc(#loc99)
                                      %7 = arith.addi %arg5, %c-1_i32 : i32 loc(#loc100)
                                      %8 = arith.index_cast %7 : i32 to index loc(#loc101)
                                      %9 = "polygeist.subindex"(%arg0, %8) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc102)
                                      %10 = affine.load %9[0] : memref<?xi32> loc(#loc103)
                                      %11 = arith.cmpi sle, %6, %10 : i32 loc(#loc104)
                                      %12 = scf.if %11 -> (memref<?xi32>) {
                                        scf.yield %cast : memref<?xi32> loc(#loc99)
                                      } else {
                                        scf.yield %9 : memref<?xi32> loc(#loc99)
                                      } loc(#loc99)
                                      %13 = affine.load %12[0] : memref<?xi32> loc(#loc105)
                                      affine.store %13, %alloca[0] : memref<1xi32> loc(#loc106)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                } loc(#loc98)
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
                                %5 = arith.cmpi slt, %arg5, %c1023_i32 : i32 loc(#loc107)
                                scf.if %5 {
                                  scf.if %true {
                                    scf.execute_region {
                                      %6 = affine.load %alloca[0] : memref<1xi32> loc(#loc99)
                                      %7 = arith.addi %arg5, %c1_i32 : i32 loc(#loc109)
                                      %8 = arith.index_cast %7 : i32 to index loc(#loc110)
                                      %9 = "polygeist.subindex"(%arg0, %8) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc111)
                                      %10 = affine.load %9[0] : memref<?xi32> loc(#loc103)
                                      %11 = arith.cmpi sle, %6, %10 : i32 loc(#loc104)
                                      %12 = scf.if %11 -> (memref<?xi32>) {
                                        scf.yield %cast : memref<?xi32> loc(#loc99)
                                      } else {
                                        scf.yield %9 : memref<?xi32> loc(#loc99)
                                      } loc(#loc99)
                                      %13 = affine.load %12[0] : memref<?xi32> loc(#loc105)
                                      affine.store %13, %alloca[0] : memref<1xi32> loc(#loc112)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                } loc(#loc108)
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
                                %5 = arith.muli %arg3, %c16_i32 : i32 loc(#loc113)
                                %6 = arith.addi %5, %arg4 : i32 loc(#loc114)
                                %7 = arith.cmpi slt, %6, %c1023_i32 : i32 loc(#loc115)
                                scf.if %7 {
                                  scf.if %true {
                                    scf.execute_region {
                                      %8 = arith.index_cast %arg5 : i32 to index loc(#loc117)
                                      %9 = "polygeist.subindex"(%arg1, %8) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc118)
                                      %10 = arith.muli %arg4, %c1024_i32 : i32 loc(#loc119)
                                      %11 = arith.addi %10, %arg5 : i32 loc(#loc120)
                                      %12 = arith.index_cast %11 : i32 to index loc(#loc121)
                                      %13 = "polygeist.subindex"(%arg2, %12) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc122)
                                      %14 = affine.load %13[0] : memref<?xi32> loc(#loc122)
                                      %15 = affine.load %alloca[0] : memref<1xi32> loc(#loc123)
                                      %16 = arith.addi %14, %15 : i32 loc(#loc124)
                                      affine.store %16, %9[0] : memref<?xi32> loc(#loc125)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                } loc(#loc116)
                                scf.yield loc(#loc)
                              } loc(#loc)
                            } loc(#loc)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg5, %c1_i32 : i32 loc(#loc126)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg5 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc93)
                      } loc(#loc8)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc127)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc128)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg5, %c512_i32 : i32 loc(#loc129)
                        scf.condition(%4) %arg5 : i32 loc(#loc130)
                      } do {
                      ^bb0(%arg5: i32 loc("pathfinder.cpp":27:37)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg5, %c2_i32 : i32 loc(#loc131)
                            %6 = arith.index_cast %5 : i32 to index loc(#loc132)
                            %7 = "polygeist.subindex"(%arg0, %6) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc133)
                            %8 = "polygeist.subindex"(%arg1, %6) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc134)
                            %9 = affine.load %8[0] : memref<?xi32> loc(#loc134)
                            affine.store %9, %7[0] : memref<?xi32> loc(#loc135)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.muli %arg5, %c2_i32 : i32 loc(#loc136)
                            %6 = arith.addi %5, %c1_i32 : i32 loc(#loc137)
                            %7 = arith.index_cast %6 : i32 to index loc(#loc138)
                            %8 = "polygeist.subindex"(%arg0, %7) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc139)
                            %9 = "polygeist.subindex"(%arg1, %7) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc140)
                            %10 = affine.load %9[0] : memref<?xi32> loc(#loc140)
                            affine.store %10, %8[0] : memref<?xi32> loc(#loc141)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg5, %c1_i32 : i32 loc(#loc142)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg5 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc130)
                      } loc(#loc81)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = arith.addi %arg4, %c1_i32 : i32 loc(#loc143)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2 : i32 loc(#loc89)
            } loc(#loc7)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc144)
  } loc(#loc78)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("pathfinder.cpp":58:27)
#loc3 = loc("pathfinder.cpp":67:48)
#loc4 = loc("pathfinder.cpp":59:15)
#loc5 = loc("pathfinder.cpp":59:10)
#loc9 = loc("pathfinder.cpp":40:1)
#loc10 = loc("pathfinder.cpp":53:5)
#loc11 = loc("pathfinder.cpp":52:5)
#loc12 = loc("pathfinder.cpp":49:5)
#loc13 = loc("pathfinder.cpp":48:5)
#loc14 = loc("pathfinder.cpp":48:1)
#loc15 = loc("pathfinder.cpp":49:1)
#loc16 = loc("pathfinder.cpp":52:1)
#loc17 = loc("pathfinder.cpp":53:1)
#loc18 = loc("pathfinder.cpp":55:9)
#loc19 = loc("pathfinder.cpp":55:13)
#loc20 = loc("pathfinder.cpp":55:15)
#loc21 = loc("pathfinder.cpp":55:31)
#loc22 = loc("pathfinder.cpp":55:2)
#loc23 = loc("pathfinder.cpp":56:9)
#loc24 = loc("pathfinder.cpp":56:16)
#loc25 = loc("pathfinder.cpp":56:14)
#loc26 = loc("pathfinder.cpp":56:24)
#loc27 = loc("pathfinder.cpp":56:40)
#loc28 = loc("pathfinder.cpp":56:47)
#loc29 = loc("pathfinder.cpp":56:2)
#loc30 = loc("pathfinder.cpp":58:1)
#loc31 = loc("pathfinder.cpp":58:5)
#loc32 = loc("pathfinder.cpp":58:32)
#loc33 = loc("pathfinder.cpp":58:19)
#loc34 = loc("pathfinder.cpp":59:8)
#loc35 = loc("pathfinder.cpp":59:12)
#loc36 = loc("pathfinder.cpp":59:3)
#loc37 = loc("pathfinder.cpp":60:9)
#loc38 = loc("pathfinder.cpp":60:4)
#loc39 = loc("pathfinder.cpp":61:22)
#loc40 = loc("pathfinder.cpp":61:26)
#loc41 = loc("pathfinder.cpp":61:30)
#loc42 = loc("pathfinder.cpp":61:37)
#loc43 = loc("pathfinder.cpp":61:4)
#loc44 = loc("pathfinder.cpp":62:18)
#loc45 = loc("pathfinder.cpp":62:9)
#loc46 = loc("pathfinder.cpp":63:9)
#loc47 = loc("pathfinder.cpp":63:4)
#loc48 = loc("pathfinder.cpp":64:22)
#loc49 = loc("pathfinder.cpp":64:26)
#loc50 = loc("pathfinder.cpp":64:30)
#loc51 = loc("pathfinder.cpp":64:36)
#loc52 = loc("pathfinder.cpp":64:4)
#loc53 = loc("pathfinder.cpp":58:51)
#loc54 = loc("pathfinder.cpp":67:20)
#loc55 = loc("pathfinder.cpp":67:24)
#loc56 = loc("pathfinder.cpp":67:28)
#loc57 = loc("pathfinder.cpp":67:2)
#loc58 = loc("pathfinder.cpp":69:9)
#loc59 = loc("pathfinder.cpp":69:14)
#loc60 = loc("pathfinder.cpp":69:18)
#loc61 = loc("pathfinder.cpp":69:34)
#loc62 = loc("pathfinder.cpp":69:2)
#loc63 = loc("pathfinder.cpp":72:1)
#loc65 = loc("pathfinder.cpp":37:36)
#loc66 = loc("pathfinder.cpp":34:1)
#loc67 = loc("pathfinder.cpp":37:9)
#loc68 = loc("pathfinder.cpp":37:25)
#loc69 = loc("pathfinder.cpp":37:35)
#loc70 = loc("pathfinder.cpp":37:22)
#loc71 = loc("pathfinder.cpp":37:16)
#loc72 = loc("pathfinder.cpp":37:14)
#loc73 = loc("pathfinder.cpp":37:39)
#loc74 = loc("pathfinder.cpp":37:55)
#loc75 = loc("pathfinder.cpp":37:62)
#loc76 = loc("pathfinder.cpp":37:2)
#loc77 = loc("pathfinder.cpp":38:1)
#loc79 = loc("pathfinder.cpp":15:25)
#loc80 = loc("pathfinder.cpp":28:8)
#loc82 = loc("pathfinder.cpp":18:15)
#loc83 = loc("pathfinder.cpp":10:26)
#loc84 = loc("pathfinder.cpp":5:1)
#loc85 = loc("pathfinder.cpp":8:2)
#loc86 = loc("pathfinder.cpp":10:1)
#loc87 = loc("pathfinder.cpp":10:5)
#loc88 = loc("pathfinder.cpp":10:32)
#loc89 = loc("pathfinder.cpp":10:17)
#loc90 = loc("pathfinder.cpp":11:1)
#loc91 = loc("pathfinder.cpp":11:6)
#loc92 = loc("pathfinder.cpp":11:33)
#loc93 = loc("pathfinder.cpp":11:20)
#loc94 = loc("pathfinder.cpp":12:15)
#loc95 = loc("pathfinder.cpp":12:10)
#loc96 = loc("pathfinder.cpp":12:8)
#loc97 = loc("pathfinder.cpp":14:9)
#loc98 = loc("pathfinder.cpp":14:4)
#loc99 = loc("./pathfinder.h":16:19)
#loc100 = loc("pathfinder.cpp":15:24)
#loc101 = loc("pathfinder.cpp":15:26)
#loc102 = loc("pathfinder.cpp":15:19)
#loc103 = loc("./pathfinder.h":16:24)
#loc104 = loc("./pathfinder.h":16:22)
#loc105 = loc("./pathfinder.h":16:18)
#loc106 = loc("pathfinder.cpp":15:9)
#loc107 = loc("pathfinder.cpp":18:9)
#loc108 = loc("pathfinder.cpp":18:4)
#loc109 = loc("pathfinder.cpp":19:24)
#loc110 = loc("pathfinder.cpp":19:26)
#loc111 = loc("pathfinder.cpp":19:19)
#loc112 = loc("pathfinder.cpp":19:9)
#loc113 = loc("pathfinder.cpp":22:9)
#loc114 = loc("pathfinder.cpp":22:21)
#loc115 = loc("pathfinder.cpp":22:26)
#loc116 = loc("pathfinder.cpp":22:4)
#loc117 = loc("pathfinder.cpp":23:10)
#loc118 = loc("pathfinder.cpp":23:5)
#loc119 = loc("pathfinder.cpp":23:22)
#loc120 = loc("pathfinder.cpp":23:29)
#loc121 = loc("pathfinder.cpp":23:32)
#loc122 = loc("pathfinder.cpp":23:14)
#loc123 = loc("pathfinder.cpp":23:34)
#loc124 = loc("pathfinder.cpp":23:33)
#loc125 = loc("pathfinder.cpp":23:12)
#loc126 = loc("pathfinder.cpp":11:42)
#loc127 = loc("pathfinder.cpp":27:1)
#loc128 = loc("pathfinder.cpp":27:6)
#loc129 = loc("pathfinder.cpp":27:31)
#loc130 = loc("pathfinder.cpp":27:18)
#loc131 = loc("pathfinder.cpp":28:9)
#loc132 = loc("pathfinder.cpp":28:11)
#loc133 = loc("pathfinder.cpp":28:4)
#loc134 = loc("pathfinder.cpp":28:15)
#loc135 = loc("pathfinder.cpp":28:13)
#loc136 = loc("pathfinder.cpp":29:9)
#loc137 = loc("pathfinder.cpp":29:11)
#loc138 = loc("pathfinder.cpp":29:13)
#loc139 = loc("pathfinder.cpp":29:4)
#loc140 = loc("pathfinder.cpp":29:17)
#loc141 = loc("pathfinder.cpp":29:15)
#loc142 = loc("pathfinder.cpp":27:42)
#loc143 = loc("pathfinder.cpp":10:47)
#loc144 = loc("pathfinder.cpp":32:1)
