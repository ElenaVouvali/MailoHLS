#loc1 = loc("knn.cpp":33:6)
#loc4 = loc("./knn.h":15:25)
#loc24 = loc("./knn.h":16:39)
#loc37 = loc("knn.cpp":3:6)
#loc39 = loc("knn.cpp":6:51)
#loc41 = loc("./knn.h":17:30)
#loc57 = loc("knn.cpp":11:6)
#loc62 = loc("knn.cpp":17:11)
#loc63 = loc("knn.cpp":14:2)
#loc71 = loc("knn.cpp":13:5)
#loc86 = loc("knn.cpp":25:6)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32> loc("knn.cpp":33:6), %arg1: memref<2097152xf32> loc("knn.cpp":33:6), %arg2: memref<1048576xf32> loc("knn.cpp":33:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
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
            ^bb0(%arg3: i32 loc("knn.cpp":6:51)):
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
  func.func @compute_dist(%arg0: memref<?xf32> loc("knn.cpp":11:6), %arg1: memref<?xf32> loc("knn.cpp":11:6), %arg2: memref<?xf32> loc("knn.cpp":11:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc58)
    %c2_i32 = arith.constant 2 : i32 loc(#loc4)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc59)
    %c512_i32 = arith.constant 512 : i32 loc(#loc41)
    %c0_i32 = arith.constant 0 : i32 loc(#loc60)
    %true = arith.constant true loc(#loc61)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc62)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc63)
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
        cf.br ^bb1 loc(#loc64)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc65)
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
            %3:4 = scf.while (%arg3 = %0, %arg4 = %2, %arg5 = %1, %arg6 = %1) : (i32, i32, f32, f32) -> (i32, i32, f32, f32) {
              %4 = arith.cmpi slt, %arg4, %c512_i32 : i32 loc(#loc66)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6 : i32, i32, f32, f32 loc(#loc67)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":17:30), %arg4: i32 loc("./knn.h":17:30), %arg5: f32 loc("./knn.h":17:30), %arg6: f32 loc("./knn.h":17:30)):
              %4 = scf.if %true -> (f32) {
                %7 = scf.execute_region -> f32 {
                  scf.yield %cst : f32 loc(#loc)
                } loc(#loc)
                scf.yield %7 : f32 loc(#loc)
              } else {
                scf.yield %arg6 : f32 loc(#loc)
              } loc(#loc)
              %5:3 = scf.if %true -> (i32, f32, f32) {
                %7:3 = scf.execute_region -> (i32, f32, f32) {
                  cf.br ^bb1 loc(#loc68)
                ^bb1:  // pred: ^bb0
                  %8:3 = scf.if %true -> (i32, f32, f32) {
                    %9:3 = scf.execute_region -> (i32, f32, f32) {
                      %10 = scf.if %true -> (i32) {
                        %12 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12 : i32 loc(#loc)
                      } else {
                        scf.yield %arg3 : i32 loc(#loc)
                      } loc(#loc)
                      %11:3 = scf.while (%arg7 = %10, %arg8 = %arg5, %arg9 = %4) : (i32, f32, f32) -> (i32, f32, f32) {
                        %12 = arith.cmpi slt, %arg7, %c2_i32 : i32 loc(#loc69)
                        scf.condition(%12) %arg7, %arg8, %arg9 : i32, f32, f32 loc(#loc70)
                      } do {
                      ^bb0(%arg7: i32 loc("knn.cpp":17:11), %arg8: f32 loc("knn.cpp":14:2), %arg9: f32 loc("knn.cpp":13:5)):
                        %12 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = arith.muli %arg4, %c2_i32 : i32 loc(#loc72)
                            %17 = arith.addi %16, %arg7 : i32 loc(#loc73)
                            %18 = arith.index_cast %17 : i32 to index loc(#loc74)
                            %19 = "polygeist.subindex"(%arg1, %18) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc75)
                            %20 = affine.load %19[0] : memref<?xf32> loc(#loc75)
                            %21 = arith.index_cast %arg7 : i32 to index loc(#loc76)
                            %22 = "polygeist.subindex"(%arg0, %21) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc77)
                            %23 = affine.load %22[0] : memref<?xf32> loc(#loc77)
                            %24 = arith.subf %20, %23 : f32 loc(#loc78)
                            scf.yield %24 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg8 : f32 loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = arith.mulf %12, %12 : f32 loc(#loc79)
                            %17 = arith.addf %arg9, %16 : f32 loc(#loc80)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg9 : f32 loc(#loc)
                        } loc(#loc)
                        %14 = scf.if %true -> (i32) {
                          %15 = scf.execute_region -> i32 {
                            %16 = arith.addi %arg7, %c1_i32 : i32 loc(#loc58)
                            scf.yield %16 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : i32 loc(#loc)
                        } else {
                          scf.yield %arg7 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14, %12, %13 : i32, f32, f32 loc(#loc70)
                      } loc(#loc4)
                      scf.yield %11#0, %11#1, %11#2 : i32, f32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %9#0, %9#1, %9#2 : i32, f32, f32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg5, %4 : i32, f32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %8#0, %8#1, %8#2 : i32, f32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %7#0, %7#1, %7#2 : i32, f32, f32 loc(#loc)
              } else {
                scf.yield %arg3, %arg5, %4 : i32, f32, f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %7 = arith.index_cast %arg4 : i32 to index loc(#loc81)
                  %8 = "polygeist.subindex"(%arg2, %7) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc82)
                  affine.store %5#2, %8[0] : memref<?xf32> loc(#loc83)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (i32) {
                %7 = scf.execute_region -> i32 {
                  %8 = arith.addi %arg4, %c1_i32 : i32 loc(#loc84)
                  scf.yield %8 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %7 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %5#0, %6, %5#1, %5#2 : i32, i32, f32, f32 loc(#loc67)
            } loc(#loc41)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc85)
  } loc(#loc57)
  func.func @store(%arg0: i32 loc("knn.cpp":25:6), %arg1: memref<?xf32> loc("knn.cpp":25:6), %arg2: memref<?xf32> loc("knn.cpp":25:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc87)
    %c0_i32 = arith.constant 0 : i32 loc(#loc88)
    %c512_i32 = arith.constant 512 : i32 loc(#loc41)
    %true = arith.constant true loc(#loc89)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc90)
    %1 = scf.if %true -> (i32) {
      %2 = scf.execute_region -> i32 {
        %3 = scf.if %true -> (i32) {
          %4 = scf.execute_region -> i32 {
            %5 = arith.muli %arg0, %c512_i32 : i32 loc(#loc91)
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
        cf.br ^bb1 loc(#loc92)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc93)
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
              %4 = arith.cmpi slt, %arg3, %c512_i32 : i32 loc(#loc94)
              scf.condition(%4) %arg3 : i32 loc(#loc95)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":17:30)):
              scf.if %true {
                scf.execute_region {
                  %5 = arith.addi %1, %arg3 : i32 loc(#loc96)
                  %6 = arith.index_cast %5 : i32 to index loc(#loc97)
                  %7 = "polygeist.subindex"(%arg2, %6) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc98)
                  %8 = arith.index_cast %arg3 : i32 to index loc(#loc99)
                  %9 = "polygeist.subindex"(%arg1, %8) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc100)
                  %10 = affine.load %9[0] : memref<?xf32> loc(#loc100)
                  affine.store %10, %7[0] : memref<?xf32> loc(#loc101)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg3, %c1_i32 : i32 loc(#loc87)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc95)
            } loc(#loc41)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc102)
  } loc(#loc86)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("./knn.h":18:45)
#loc3 = loc("knn.cpp":50:52)
#loc5 = loc("knn.cpp":50:33)
#loc6 = loc("knn.cpp":33:1)
#loc7 = loc("knn.cpp":54:20)
#loc8 = loc("knn.cpp":48:5)
#loc9 = loc("knn.cpp":47:5)
#loc10 = loc("knn.cpp":46:5)
#loc11 = loc("knn.cpp":46:1)
#loc12 = loc("knn.cpp":47:1)
#loc13 = loc("knn.cpp":48:1)
#loc14 = loc("knn.cpp":50:1)
#loc15 = loc("knn.cpp":50:5)
#loc16 = loc("knn.cpp":50:38)
#loc17 = loc("knn.cpp":50:22)
#loc18 = loc("knn.cpp":51:21)
#loc19 = loc("knn.cpp":51:3)
#loc20 = loc("knn.cpp":51:25)
#loc21 = loc("knn.cpp":51:23)
#loc22 = loc("knn.cpp":54:1)
#loc23 = loc("knn.cpp":54:5)
#loc25 = loc("knn.cpp":54:45)
#loc26 = loc("knn.cpp":54:15)
#loc27 = loc("knn.cpp":55:31)
#loc28 = loc("knn.cpp":55:3)
#loc29 = loc("knn.cpp":56:16)
#loc30 = loc("knn.cpp":56:34)
#loc31 = loc("knn.cpp":56:53)
#loc32 = loc("knn.cpp":56:3)
#loc33 = loc("knn.cpp":57:25)
#loc34 = loc("knn.cpp":57:9)
#loc35 = loc("knn.cpp":54:57)
#loc36 = loc("knn.cpp":61:1)
#loc38 = loc("knn.cpp":6:65)
#loc40 = loc("knn.cpp":6:27)
#loc42 = loc("knn.cpp":3:1)
#loc43 = loc("knn.cpp":6:21)
#loc44 = loc("knn.cpp":5:27)
#loc45 = loc("knn.cpp":5:46)
#loc46 = loc("knn.cpp":6:1)
#loc47 = loc("knn.cpp":6:5)
#loc48 = loc("knn.cpp":6:33)
#loc49 = loc("knn.cpp":6:16)
#loc50 = loc("knn.cpp":7:22)
#loc51 = loc("knn.cpp":7:3)
#loc52 = loc("knn.cpp":7:47)
#loc53 = loc("knn.cpp":7:49)
#loc54 = loc("knn.cpp":7:26)
#loc55 = loc("knn.cpp":7:24)
#loc56 = loc("knn.cpp":9:1)
#loc58 = loc("knn.cpp":17:39)
#loc59 = loc("knn.cpp":16:15)
#loc60 = loc("knn.cpp":15:32)
#loc61 = loc("knn.cpp":11:1)
#loc64 = loc("knn.cpp":15:1)
#loc65 = loc("knn.cpp":15:5)
#loc66 = loc("knn.cpp":15:37)
#loc67 = loc("knn.cpp":15:19)
#loc68 = loc("knn.cpp":17:1)
#loc69 = loc("knn.cpp":17:24)
#loc70 = loc("knn.cpp":17:6)
#loc72 = loc("knn.cpp":18:39)
#loc73 = loc("knn.cpp":18:51)
#loc74 = loc("knn.cpp":18:53)
#loc75 = loc("knn.cpp":18:20)
#loc76 = loc("knn.cpp":18:75)
#loc77 = loc("knn.cpp":18:57)
#loc78 = loc("knn.cpp":18:55)
#loc79 = loc("knn.cpp":19:24)
#loc80 = loc("knn.cpp":19:8)
#loc81 = loc("knn.cpp":21:25)
#loc82 = loc("knn.cpp":21:9)
#loc83 = loc("knn.cpp":21:27)
#loc84 = loc("knn.cpp":15:57)
#loc85 = loc("knn.cpp":23:1)
#loc87 = loc("knn.cpp":28:54)
#loc88 = loc("knn.cpp":28:28)
#loc89 = loc("knn.cpp":25:1)
#loc90 = loc("knn.cpp":28:22)
#loc91 = loc("knn.cpp":27:28)
#loc92 = loc("knn.cpp":28:1)
#loc93 = loc("knn.cpp":28:5)
#loc94 = loc("knn.cpp":28:34)
#loc95 = loc("knn.cpp":28:17)
#loc96 = loc("knn.cpp":29:27)
#loc97 = loc("knn.cpp":29:29)
#loc98 = loc("knn.cpp":29:9)
#loc99 = loc("knn.cpp":29:49)
#loc100 = loc("knn.cpp":29:33)
#loc101 = loc("knn.cpp":29:31)
#loc102 = loc("knn.cpp":31:1)
