#loc1 = loc("pathfinder.cpp":28:6)
#loc3 = loc("./pathfinder.h":13:14)
#loc4 = loc("pathfinder.cpp":43:38)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<1048576xi32> loc("pathfinder.cpp":28:6), %arg1: memref<1024xi32> loc("pathfinder.cpp":28:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc3)
    %c1023_i32 = arith.constant 1023 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc3)
    %true = arith.constant true loc(#loc6)
    %alloca = memref.alloca() : memref<1xi32> loc(#loc7)
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32> loc(#loc7)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc7)
    affine.store %0, %alloca[0] : memref<1xi32> loc(#loc7)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc8)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc9)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc10)
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
        %1 = "polygeist.memref2pointer"(%alloca_1) : (memref<1024xi32>) -> !llvm.ptr loc(#loc12)
        %2 = "polygeist.memref2pointer"(%arg0) : (memref<1048576xi32>) -> !llvm.ptr loc(#loc13)
        %3 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc14)
        %4 = arith.index_cast %3 : index to i64 loc(#loc14)
        %5 = arith.muli %4, %c1024_i64 : i64 loc(#loc15)
        "llvm.intr.memcpy"(%1, %2, %5) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc16)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc17)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc18)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %1 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
              %2 = arith.cmpi slt, %arg2, %c1023_i32 : i32 loc(#loc19)
              scf.condition(%2) %arg2 : i32 loc(#loc20)
            } do {
            ^bb0(%arg2: i32 loc("pathfinder.cpp":43:38)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc21)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc22)
                ^bb2:  // pred: ^bb1
                  scf.if %true {
                    scf.execute_region {
                      %3 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
                        %4 = arith.cmpi slt, %arg3, %c1024_i32 : i32 loc(#loc23)
                        scf.condition(%4) %arg3 : i32 loc(#loc24)
                      } do {
                      ^bb0(%arg3: i32 loc("./pathfinder.h":13:14)):
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.index_cast %arg3 : i32 to index loc(#loc25)
                            %6 = "polygeist.subindex"(%alloca_1, %5) : (memref<1024xi32>, index) -> memref<?xi32> loc(#loc26)
                            %7 = affine.load %6[0] : memref<?xi32> loc(#loc26)
                            affine.store %7, %alloca[0] : memref<1xi32> loc(#loc27)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            scf.if %true {
                              scf.execute_region {
                                %5 = arith.cmpi sgt, %arg3, %c0_i32 : i32 loc(#loc28)
                                scf.if %5 {
                                  scf.if %true {
                                    scf.execute_region {
                                      %6 = affine.load %alloca[0] : memref<1xi32> loc(#loc30)
                                      %7 = arith.addi %arg3, %c-1_i32 : i32 loc(#loc31)
                                      %8 = arith.index_cast %7 : i32 to index loc(#loc32)
                                      %9 = "polygeist.subindex"(%alloca_1, %8) : (memref<1024xi32>, index) -> memref<?xi32> loc(#loc33)
                                      %10 = affine.load %9[0] : memref<?xi32> loc(#loc34)
                                      %11 = arith.cmpi sle, %6, %10 : i32 loc(#loc35)
                                      %12 = scf.if %11 -> (memref<?xi32>) {
                                        scf.yield %cast : memref<?xi32> loc(#loc30)
                                      } else {
                                        scf.yield %9 : memref<?xi32> loc(#loc30)
                                      } loc(#loc30)
                                      %13 = affine.load %12[0] : memref<?xi32> loc(#loc36)
                                      affine.store %13, %alloca[0] : memref<1xi32> loc(#loc37)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                } loc(#loc29)
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
                                %5 = arith.cmpi slt, %arg3, %c1023_i32 : i32 loc(#loc38)
                                scf.if %5 {
                                  scf.if %true {
                                    scf.execute_region {
                                      %6 = affine.load %alloca[0] : memref<1xi32> loc(#loc30)
                                      %7 = arith.addi %arg3, %c1_i32 : i32 loc(#loc40)
                                      %8 = arith.index_cast %7 : i32 to index loc(#loc41)
                                      %9 = "polygeist.subindex"(%alloca_1, %8) : (memref<1024xi32>, index) -> memref<?xi32> loc(#loc42)
                                      %10 = affine.load %9[0] : memref<?xi32> loc(#loc34)
                                      %11 = arith.cmpi sle, %6, %10 : i32 loc(#loc35)
                                      %12 = scf.if %11 -> (memref<?xi32>) {
                                        scf.yield %cast : memref<?xi32> loc(#loc30)
                                      } else {
                                        scf.yield %9 : memref<?xi32> loc(#loc30)
                                      } loc(#loc30)
                                      %13 = affine.load %12[0] : memref<?xi32> loc(#loc36)
                                      affine.store %13, %alloca[0] : memref<1xi32> loc(#loc43)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                } loc(#loc39)
                                scf.yield loc(#loc)
                              } loc(#loc)
                            } loc(#loc)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %5 = arith.index_cast %arg3 : i32 to index loc(#loc44)
                            %6 = "polygeist.subindex"(%alloca_0, %5) : (memref<1024xi32>, index) -> memref<?xi32> loc(#loc45)
                            %7 = arith.addi %arg2, %c1_i32 : i32 loc(#loc46)
                            %8 = arith.muli %7, %c1024_i32 : i32 loc(#loc47)
                            %9 = arith.addi %8, %arg3 : i32 loc(#loc48)
                            %10 = arith.index_cast %9 : i32 to index loc(#loc49)
                            %11 = "polygeist.subindex"(%arg0, %10) : (memref<1048576xi32>, index) -> memref<?xi32> loc(#loc50)
                            %12 = affine.load %11[0] : memref<?xi32> loc(#loc50)
                            %13 = affine.load %alloca[0] : memref<1xi32> loc(#loc51)
                            %14 = arith.addi %12, %13 : i32 loc(#loc52)
                            affine.store %14, %6[0] : memref<?xi32> loc(#loc53)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %4 = scf.if %true -> (i32) {
                          %5 = scf.execute_region -> i32 {
                            %6 = arith.addi %arg3, %c1_i32 : i32 loc(#loc54)
                            scf.yield %6 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %5 : i32 loc(#loc)
                        } else {
                          scf.yield %arg3 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %4 : i32 loc(#loc24)
                      } loc(#loc3)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %3 = "polygeist.memref2pointer"(%alloca_1) : (memref<1024xi32>) -> !llvm.ptr loc(#loc55)
                  %4 = "polygeist.memref2pointer"(%alloca_0) : (memref<1024xi32>) -> !llvm.ptr loc(#loc56)
                  %5 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc57)
                  %6 = arith.index_cast %5 : index to i64 loc(#loc57)
                  %7 = arith.muli %6, %c1024_i64 : i64 loc(#loc58)
                  "llvm.intr.memcpy"(%3, %4, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc59)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %2 = scf.if %true -> (i32) {
                %3 = scf.execute_region -> i32 {
                  %4 = arith.addi %arg2, %c1_i32 : i32 loc(#loc60)
                  scf.yield %4 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %3 : i32 loc(#loc)
              } else {
                scf.yield %arg2 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %2 : i32 loc(#loc20)
            } loc(#loc4)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %1 = "polygeist.memref2pointer"(%arg1) : (memref<1024xi32>) -> !llvm.ptr loc(#loc61)
        %2 = "polygeist.memref2pointer"(%alloca_1) : (memref<1024xi32>) -> !llvm.ptr loc(#loc62)
        %3 = "polygeist.typeSize"() <{source = i32}> : () -> index loc(#loc63)
        %4 = arith.index_cast %3 : index to i64 loc(#loc63)
        %5 = arith.muli %4, %c1024_i64 : i64 loc(#loc64)
        "llvm.intr.memcpy"(%1, %2, %5) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc65)
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
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("pathfinder.cpp":48:25)
#loc5 = loc("pathfinder.cpp":43:27)
#loc6 = loc("pathfinder.cpp":28:1)
#loc7 = loc("pathfinder.cpp":39:2)
#loc8 = loc("pathfinder.cpp":37:5)
#loc9 = loc("pathfinder.cpp":36:5)
#loc10 = loc("pathfinder.cpp":36:1)
#loc11 = loc("pathfinder.cpp":37:1)
#loc12 = loc("pathfinder.cpp":41:9)
#loc13 = loc("pathfinder.cpp":41:13)
#loc14 = loc("pathfinder.cpp":41:15)
#loc15 = loc("pathfinder.cpp":41:31)
#loc16 = loc("pathfinder.cpp":41:2)
#loc17 = loc("pathfinder.cpp":43:1)
#loc18 = loc("pathfinder.cpp":43:5)
#loc19 = loc("pathfinder.cpp":43:32)
#loc20 = loc("pathfinder.cpp":43:19)
#loc21 = loc("pathfinder.cpp":44:1)
#loc22 = loc("pathfinder.cpp":44:6)
#loc23 = loc("pathfinder.cpp":44:33)
#loc24 = loc("pathfinder.cpp":44:20)
#loc25 = loc("pathfinder.cpp":45:15)
#loc26 = loc("pathfinder.cpp":45:10)
#loc27 = loc("pathfinder.cpp":45:8)
#loc28 = loc("pathfinder.cpp":47:9)
#loc29 = loc("pathfinder.cpp":47:4)
#loc30 = loc("./pathfinder.h":16:19)
#loc31 = loc("pathfinder.cpp":48:24)
#loc32 = loc("pathfinder.cpp":48:26)
#loc33 = loc("pathfinder.cpp":48:19)
#loc34 = loc("./pathfinder.h":16:24)
#loc35 = loc("./pathfinder.h":16:22)
#loc36 = loc("./pathfinder.h":16:18)
#loc37 = loc("pathfinder.cpp":48:9)
#loc38 = loc("pathfinder.cpp":51:9)
#loc39 = loc("pathfinder.cpp":51:4)
#loc40 = loc("pathfinder.cpp":52:24)
#loc41 = loc("pathfinder.cpp":52:26)
#loc42 = loc("pathfinder.cpp":52:19)
#loc43 = loc("pathfinder.cpp":52:9)
#loc44 = loc("pathfinder.cpp":55:9)
#loc45 = loc("pathfinder.cpp":55:4)
#loc46 = loc("pathfinder.cpp":55:17)
#loc47 = loc("pathfinder.cpp":55:21)
#loc48 = loc("pathfinder.cpp":55:28)
#loc49 = loc("pathfinder.cpp":55:31)
#loc50 = loc("pathfinder.cpp":55:13)
#loc51 = loc("pathfinder.cpp":55:33)
#loc52 = loc("pathfinder.cpp":55:32)
#loc53 = loc("pathfinder.cpp":55:11)
#loc54 = loc("pathfinder.cpp":44:42)
#loc55 = loc("pathfinder.cpp":57:10)
#loc56 = loc("pathfinder.cpp":57:14)
#loc57 = loc("pathfinder.cpp":57:18)
#loc58 = loc("pathfinder.cpp":57:34)
#loc59 = loc("pathfinder.cpp":57:3)
#loc60 = loc("pathfinder.cpp":43:43)
#loc61 = loc("pathfinder.cpp":59:9)
#loc62 = loc("pathfinder.cpp":59:14)
#loc63 = loc("pathfinder.cpp":59:18)
#loc64 = loc("pathfinder.cpp":59:34)
#loc65 = loc("pathfinder.cpp":59:2)
#loc66 = loc("pathfinder.cpp":62:1)
