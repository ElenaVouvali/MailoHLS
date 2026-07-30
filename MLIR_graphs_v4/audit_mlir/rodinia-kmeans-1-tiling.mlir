#loc1 = loc("kmeans.cpp":57:6)
#loc15 = loc("./kmeans.h":14:24)
#loc22 = loc("kmeans.cpp":4:6)
#loc25 = loc("./kmeans.h":16:19)
#loc28 = loc("kmeans.cpp":7:11)
#loc43 = loc("kmeans.cpp":13:6)
#loc45 = loc("./kmeans.h":17:19)
#loc48 = loc("kmeans.cpp":16:11)
#loc68 = loc("kmeans.cpp":22:6)
#loc74 = loc("kmeans.cpp":35:17)
#loc75 = loc("kmeans.cpp":34:27)
#loc83 = loc("kmeans.cpp":32:13)
#loc84 = loc("kmeans.cpp":31:22)
#loc85 = loc("kmeans.cpp":28:9)
#loc86 = loc("kmeans.cpp":27:9)
#loc110 = loc("kmeans.cpp":50:6)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<?xf32> loc("kmeans.cpp":57:6), %arg1: memref<?xf32> loc("kmeans.cpp":57:6), %arg2: memref<?xi32> loc("kmeans.cpp":57:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %true = arith.constant true loc(#loc5)
    %alloca = memref.alloca() : memref<170xf32> loc(#loc6)
    %alloca_0 = memref.alloca() : memref<139264xf32> loc(#loc7)
    %alloca_1 = memref.alloca() : memref<4096xi32> loc(#loc8)
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
        %cast = memref.cast %arg1 : memref<?xf32> to memref<170xf32> loc(#loc12)
        func.call @load_local_cluster(%alloca, %cast) : (memref<170xf32>, memref<170xf32>) -> () loc(#loc12)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc13)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc14)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %0 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
              %1 = arith.cmpi slt, %arg3, %c100_i32 : i32 loc(#loc16)
              scf.condition(%1) %arg3 : i32 loc(#loc17)
            } do {
            ^bb0(%arg3: i32 loc("./kmeans.h":14:24)):
              scf.if %true {
                scf.execute_region {
                  %cast = memref.cast %arg0 : memref<?xf32> to memref<13926400xf32> loc(#loc18)
                  func.call @load_local_feature(%alloca_0, %cast, %arg3) : (memref<139264xf32>, memref<13926400xf32>, i32) -> () loc(#loc18)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  func.call @compute_local_membership(%alloca_0, %alloca, %alloca_1) : (memref<139264xf32>, memref<170xf32>, memref<4096xi32>) -> () loc(#loc19)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %cast = memref.cast %arg2 : memref<?xi32> to memref<409600xi32> loc(#loc20)
                  func.call @store_local_membership(%alloca_1, %cast, %arg3) : (memref<4096xi32>, memref<409600xi32>, i32) -> () loc(#loc20)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %1 = scf.if %true -> (i32) {
                %2 = scf.execute_region -> i32 {
                  %3 = arith.addi %arg3, %c1_i32 : i32 loc(#loc3)
                  scf.yield %3 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %2 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %1 : i32 loc(#loc17)
            } loc(#loc15)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc21)
  } loc(#loc1)
  func.func @load_local_cluster(%arg0: memref<170xf32> loc("kmeans.cpp":4:6), %arg1: memref<170xf32> loc("kmeans.cpp":4:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc23)
    %c34_i32 = arith.constant 34 : i32 loc(#loc24)
    %c5_i32 = arith.constant 5 : i32 loc(#loc25)
    %c0_i32 = arith.constant 0 : i32 loc(#loc26)
    %true = arith.constant true loc(#loc27)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc28)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc29)
      ^bb1:  // pred: ^bb0
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
            %2:2 = scf.while (%arg2 = %0, %arg3 = %1) : (i32, i32) -> (i32, i32) {
              %3 = arith.cmpi slt, %arg3, %c5_i32 : i32 loc(#loc30)
              scf.condition(%3) %arg2, %arg3 : i32, i32 loc(#loc31)
            } do {
            ^bb0(%arg2: i32 loc("./kmeans.h":16:19), %arg3: i32 loc("./kmeans.h":16:19)):
              %3 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  cf.br ^bb1 loc(#loc32)
                ^bb1:  // pred: ^bb0
                  %6 = scf.if %true -> (i32) {
                    %7 = scf.execute_region -> i32 {
                      %8 = scf.if %true -> (i32) {
                        %10 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10 : i32 loc(#loc)
                      } else {
                        scf.yield %arg2 : i32 loc(#loc)
                      } loc(#loc)
                      %9 = scf.while (%arg4 = %8) : (i32) -> i32 {
                        %10 = arith.cmpi slt, %arg4, %c34_i32 : i32 loc(#loc33)
                        scf.condition(%10) %arg4 : i32 loc(#loc34)
                      } do {
                      ^bb0(%arg4: i32 loc("kmeans.cpp":7:11)):
                        scf.if %true {
                          scf.execute_region {
                            %11 = arith.muli %arg3, %c34_i32 : i32 loc(#loc35)
                            %12 = arith.addi %11, %arg4 : i32 loc(#loc36)
                            %13 = arith.index_cast %12 : i32 to index loc(#loc37)
                            %14 = "polygeist.subindex"(%arg0, %13) : (memref<170xf32>, index) -> memref<?xf32> loc(#loc38)
                            %15 = "polygeist.subindex"(%arg1, %13) : (memref<170xf32>, index) -> memref<?xf32> loc(#loc39)
                            %16 = affine.load %15[0] : memref<?xf32> loc(#loc39)
                            affine.store %16, %14[0] : memref<?xf32> loc(#loc40)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (i32) {
                          %11 = scf.execute_region -> i32 {
                            %12 = arith.addi %arg4, %c1_i32 : i32 loc(#loc23)
                            scf.yield %12 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10 : i32 loc(#loc34)
                      } loc(#loc24)
                      scf.yield %9 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %7 : i32 loc(#loc)
                  } else {
                    scf.yield %arg2 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg2 : i32 loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg3, %c1_i32 : i32 loc(#loc41)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3, %4 : i32, i32 loc(#loc31)
            } loc(#loc25)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc42)
  } loc(#loc22)
  func.func @load_local_feature(%arg0: memref<139264xf32> loc("kmeans.cpp":13:6), %arg1: memref<13926400xf32> loc("kmeans.cpp":13:6), %arg2: i32 loc("kmeans.cpp":13:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc44)
    %c34_i32 = arith.constant 34 : i32 loc(#loc24)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc45)
    %c0_i32 = arith.constant 0 : i32 loc(#loc46)
    %true = arith.constant true loc(#loc47)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc48)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc49)
      ^bb1:  // pred: ^bb0
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
            %2:2 = scf.while (%arg3 = %0, %arg4 = %1) : (i32, i32) -> (i32, i32) {
              %3 = arith.cmpi slt, %arg4, %c4096_i32 : i32 loc(#loc50)
              scf.condition(%3) %arg3, %arg4 : i32, i32 loc(#loc51)
            } do {
            ^bb0(%arg3: i32 loc("./kmeans.h":17:19), %arg4: i32 loc("./kmeans.h":17:19)):
              %3 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  cf.br ^bb1 loc(#loc52)
                ^bb1:  // pred: ^bb0
                  %6 = scf.if %true -> (i32) {
                    %7 = scf.execute_region -> i32 {
                      %8 = scf.if %true -> (i32) {
                        %10 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10 : i32 loc(#loc)
                      } else {
                        scf.yield %arg3 : i32 loc(#loc)
                      } loc(#loc)
                      %9 = scf.while (%arg5 = %8) : (i32) -> i32 {
                        %10 = arith.cmpi slt, %arg5, %c34_i32 : i32 loc(#loc53)
                        scf.condition(%10) %arg5 : i32 loc(#loc54)
                      } do {
                      ^bb0(%arg5: i32 loc("kmeans.cpp":16:11)):
                        scf.if %true {
                          scf.execute_region {
                            %11 = arith.muli %arg4, %c34_i32 : i32 loc(#loc55)
                            %12 = arith.addi %11, %arg5 : i32 loc(#loc56)
                            %13 = arith.index_cast %12 : i32 to index loc(#loc57)
                            %14 = "polygeist.subindex"(%arg0, %13) : (memref<139264xf32>, index) -> memref<?xf32> loc(#loc58)
                            %15 = arith.muli %arg2, %c4096_i32 : i32 loc(#loc59)
                            %16 = arith.addi %15, %arg4 : i32 loc(#loc60)
                            %17 = arith.muli %16, %c34_i32 : i32 loc(#loc61)
                            %18 = arith.addi %17, %arg5 : i32 loc(#loc62)
                            %19 = arith.index_cast %18 : i32 to index loc(#loc63)
                            %20 = "polygeist.subindex"(%arg1, %19) : (memref<13926400xf32>, index) -> memref<?xf32> loc(#loc64)
                            %21 = affine.load %20[0] : memref<?xf32> loc(#loc64)
                            affine.store %21, %14[0] : memref<?xf32> loc(#loc65)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %10 = scf.if %true -> (i32) {
                          %11 = scf.execute_region -> i32 {
                            %12 = arith.addi %arg5, %c1_i32 : i32 loc(#loc44)
                            scf.yield %12 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %11 : i32 loc(#loc)
                        } else {
                          scf.yield %arg5 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10 : i32 loc(#loc54)
                      } loc(#loc24)
                      scf.yield %9 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %7 : i32 loc(#loc)
                  } else {
                    scf.yield %arg3 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg4, %c1_i32 : i32 loc(#loc66)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3, %4 : i32, i32 loc(#loc51)
            } loc(#loc45)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc67)
  } loc(#loc43)
  func.func @compute_local_membership(%arg0: memref<139264xf32> loc("kmeans.cpp":22:6), %arg1: memref<170xf32> loc("kmeans.cpp":22:6), %arg2: memref<4096xi32> loc("kmeans.cpp":22:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc69)
    %c34_i32 = arith.constant 34 : i32 loc(#loc24)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc70)
    %c5_i32 = arith.constant 5 : i32 loc(#loc25)
    %cst_0 = arith.constant 3.40282347E+38 : f32 loc(#loc71)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc45)
    %c0_i32 = arith.constant 0 : i32 loc(#loc72)
    %true = arith.constant true loc(#loc73)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc74)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc75)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc76)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = scf.if %true -> (i32) {
              %4 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc)
            } else {
              scf.yield %1 : i32 loc(#loc)
            } loc(#loc)
            %3:7 = scf.while (%arg3 = %0, %arg4 = %1, %arg5 = %0, %arg6 = %1, %arg7 = %1, %arg8 = %0, %arg9 = %2) : (f32, i32, f32, i32, i32, f32, i32) -> (f32, i32, f32, i32, i32, f32, i32) {
              %4 = arith.cmpi slt, %arg9, %c4096_i32 : i32 loc(#loc77)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : f32, i32, f32, i32, i32, f32, i32 loc(#loc78)
            } do {
            ^bb0(%arg3: f32 loc("./kmeans.h":17:19), %arg4: i32 loc("./kmeans.h":17:19), %arg5: f32 loc("./kmeans.h":17:19), %arg6: i32 loc("./kmeans.h":17:19), %arg7: i32 loc("./kmeans.h":17:19), %arg8: f32 loc("./kmeans.h":17:19), %arg9: i32 loc("./kmeans.h":17:19)):
              %4 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      scf.yield %cst_0 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg8 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg8 : f32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      scf.yield %c0_i32 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg7 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg7 : i32 loc(#loc)
              } loc(#loc)
              %6:6 = scf.if %true -> (f32, i32, f32, i32, i32, f32) {
                %8:6 = scf.execute_region -> (f32, i32, f32, i32, i32, f32) {
                  cf.br ^bb1 loc(#loc79)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc80)
                ^bb2:  // pred: ^bb1
                  %9:6 = scf.if %true -> (f32, i32, f32, i32, i32, f32) {
                    %10:6 = scf.execute_region -> (f32, i32, f32, i32, i32, f32) {
                      %11 = scf.if %true -> (i32) {
                        %13 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %13 : i32 loc(#loc)
                      } else {
                        scf.yield %arg6 : i32 loc(#loc)
                      } loc(#loc)
                      %12:6 = scf.while (%arg10 = %arg3, %arg11 = %arg4, %arg12 = %arg5, %arg13 = %11, %arg14 = %5, %arg15 = %4) : (f32, i32, f32, i32, i32, f32) -> (f32, i32, f32, i32, i32, f32) {
                        %13 = arith.cmpi slt, %arg13, %c5_i32 : i32 loc(#loc81)
                        scf.condition(%13) %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : f32, i32, f32, i32, i32, f32 loc(#loc82)
                      } do {
                      ^bb0(%arg10: f32 loc("kmeans.cpp":35:17), %arg11: i32 loc("kmeans.cpp":34:27), %arg12: f32 loc("kmeans.cpp":32:13), %arg13: i32 loc("kmeans.cpp":31:22), %arg14: i32 loc("kmeans.cpp":28:9), %arg15: f32 loc("kmeans.cpp":27:9)):
                        %13 = scf.if %true -> (f32) {
                          %17 = scf.execute_region -> f32 {
                            %18 = scf.if %true -> (f32) {
                              %19 = scf.execute_region -> f32 {
                                scf.yield %cst : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19 : f32 loc(#loc)
                            } else {
                              scf.yield %arg12 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : f32 loc(#loc)
                        } else {
                          scf.yield %arg12 : f32 loc(#loc)
                        } loc(#loc)
                        %14:3 = scf.if %true -> (f32, i32, f32) {
                          %17:3 = scf.execute_region -> (f32, i32, f32) {
                            cf.br ^bb1 loc(#loc87)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc88)
                          ^bb2:  // pred: ^bb1
                            %18:3 = scf.if %true -> (f32, i32, f32) {
                              %19:3 = scf.execute_region -> (f32, i32, f32) {
                                %20 = scf.if %true -> (i32) {
                                  %22 = scf.execute_region -> i32 {
                                    scf.yield %c0_i32 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg11 : i32 loc(#loc)
                                } loc(#loc)
                                %21:3 = scf.while (%arg16 = %arg10, %arg17 = %20, %arg18 = %13) : (f32, i32, f32) -> (f32, i32, f32) {
                                  %22 = arith.cmpi slt, %arg17, %c34_i32 : i32 loc(#loc89)
                                  scf.condition(%22) %arg16, %arg17, %arg18 : f32, i32, f32 loc(#loc90)
                                } do {
                                ^bb0(%arg16: f32 loc("kmeans.cpp":35:17), %arg17: i32 loc("kmeans.cpp":34:27), %arg18: f32 loc("kmeans.cpp":32:13)):
                                  %22 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          %28 = arith.muli %arg9, %c34_i32 : i32 loc(#loc91)
                                          %29 = arith.addi %28, %arg17 : i32 loc(#loc92)
                                          %30 = arith.index_cast %29 : i32 to index loc(#loc93)
                                          %31 = "polygeist.subindex"(%arg0, %30) : (memref<139264xf32>, index) -> memref<?xf32> loc(#loc94)
                                          %32 = affine.load %31[0] : memref<?xf32> loc(#loc94)
                                          %33 = arith.muli %arg13, %c34_i32 : i32 loc(#loc95)
                                          %34 = arith.addi %33, %arg17 : i32 loc(#loc96)
                                          %35 = arith.index_cast %34 : i32 to index loc(#loc97)
                                          %36 = "polygeist.subindex"(%arg1, %35) : (memref<170xf32>, index) -> memref<?xf32> loc(#loc98)
                                          %37 = affine.load %36[0] : memref<?xf32> loc(#loc98)
                                          %38 = arith.subf %32, %37 : f32 loc(#loc99)
                                          scf.yield %38 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg16 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg16 : f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = arith.mulf %22, %22 : f32 loc(#loc100)
                                      %27 = arith.addf %arg18, %26 : f32 loc(#loc101)
                                      scf.yield %27 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg18 : f32 loc(#loc)
                                  } loc(#loc)
                                  %24 = scf.if %true -> (i32) {
                                    %25 = scf.execute_region -> i32 {
                                      %26 = arith.addi %arg17, %c1_i32 : i32 loc(#loc69)
                                      scf.yield %26 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg17 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22, %24, %23 : f32, i32, f32 loc(#loc90)
                                } loc(#loc24)
                                scf.yield %21#0, %21#1, %21#2 : f32, i32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19#0, %19#1, %19#2 : f32, i32, f32 loc(#loc)
                            } else {
                              scf.yield %arg10, %arg11, %13 : f32, i32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18#0, %18#1, %18#2 : f32, i32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17#0, %17#1, %17#2 : f32, i32, f32 loc(#loc)
                        } else {
                          scf.yield %arg10, %arg11, %13 : f32, i32, f32 loc(#loc)
                        } loc(#loc)
                        %15:2 = scf.if %true -> (i32, f32) {
                          %17:2 = scf.execute_region -> (i32, f32) {
                            %18:2 = scf.if %true -> (i32, f32) {
                              %19:2 = scf.execute_region -> (i32, f32) {
                                %20 = arith.cmpf olt, %14#2, %arg15 : f32 loc(#loc102)
                                %21:2 = scf.if %20 -> (i32, f32) {
                                  %22 = scf.if %true -> (f32) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %14#2 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg15 : f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (i32) {
                                    scf.execute_region {
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                    scf.yield %arg13 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg14 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %23, %22 : i32, f32 loc(#loc103)
                                } else {
                                  scf.yield %arg14, %arg15 : i32, f32 loc(#loc103)
                                } loc(#loc103)
                                scf.yield %21#0, %21#1 : i32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19#0, %19#1 : i32, f32 loc(#loc)
                            } else {
                              scf.yield %arg14, %arg15 : i32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18#0, %18#1 : i32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17#0, %17#1 : i32, f32 loc(#loc)
                        } else {
                          scf.yield %arg14, %arg15 : i32, f32 loc(#loc)
                        } loc(#loc)
                        %16 = scf.if %true -> (i32) {
                          %17 = scf.execute_region -> i32 {
                            %18 = arith.addi %arg13, %c1_i32 : i32 loc(#loc104)
                            scf.yield %18 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : i32 loc(#loc)
                        } else {
                          scf.yield %arg13 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14#0, %14#1, %14#2, %16, %15#0, %15#1 : f32, i32, f32, i32, i32, f32 loc(#loc82)
                      } loc(#loc25)
                      scf.yield %12#0, %12#1, %12#2, %12#3, %12#4, %12#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10#0, %10#1, %10#2, %10#3, %10#4, %10#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg5, %arg6, %5, %4 : f32, i32, f32, i32, i32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9#0, %9#1, %9#2, %9#3, %9#4, %9#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %8#0, %8#1, %8#2, %8#3, %8#4, %8#5 : f32, i32, f32, i32, i32, f32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg5, %arg6, %5, %4 : f32, i32, f32, i32, i32, f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = arith.index_cast %arg9 : i32 to index loc(#loc105)
                  %9 = "polygeist.subindex"(%arg2, %8) : (memref<4096xi32>, index) -> memref<?xi32> loc(#loc106)
                  affine.store %6#4, %9[0] : memref<?xi32> loc(#loc107)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg9, %c1_i32 : i32 loc(#loc108)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg9 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6#0, %6#1, %6#2, %6#3, %6#4, %6#5, %7 : f32, i32, f32, i32, i32, f32, i32 loc(#loc78)
            } loc(#loc45)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc109)
  } loc(#loc68)
  func.func @store_local_membership(%arg0: memref<4096xi32> loc("kmeans.cpp":50:6), %arg1: memref<409600xi32> loc("kmeans.cpp":50:6), %arg2: i32 loc("kmeans.cpp":50:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc111)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc45)
    %c0_i32 = arith.constant 0 : i32 loc(#loc112)
    %true = arith.constant true loc(#loc113)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc114)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc115)
      ^bb1:  // pred: ^bb0
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
              %3 = arith.cmpi slt, %arg3, %c4096_i32 : i32 loc(#loc116)
              scf.condition(%3) %arg3 : i32 loc(#loc117)
            } do {
            ^bb0(%arg3: i32 loc("./kmeans.h":17:19)):
              scf.if %true {
                scf.execute_region {
                  %4 = arith.muli %arg2, %c4096_i32 : i32 loc(#loc118)
                  %5 = arith.addi %4, %arg3 : i32 loc(#loc119)
                  %6 = arith.index_cast %5 : i32 to index loc(#loc120)
                  %7 = "polygeist.subindex"(%arg1, %6) : (memref<409600xi32>, index) -> memref<?xi32> loc(#loc121)
                  %8 = arith.index_cast %arg3 : i32 to index loc(#loc122)
                  %9 = "polygeist.subindex"(%arg0, %8) : (memref<4096xi32>, index) -> memref<?xi32> loc(#loc123)
                  %10 = affine.load %9[0] : memref<?xi32> loc(#loc123)
                  affine.store %10, %7[0] : memref<?xi32> loc(#loc124)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg3, %c1_i32 : i32 loc(#loc111)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3 : i32 loc(#loc117)
            } loc(#loc45)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc125)
  } loc(#loc110)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("./kmeans.h":21:30)
#loc3 = loc("kmeans.cpp":77:50)
#loc4 = loc("kmeans.cpp":77:33)
#loc5 = loc("kmeans.cpp":57:1)
#loc6 = loc("kmeans.cpp":73:9)
#loc7 = loc("kmeans.cpp":72:9)
#loc8 = loc("kmeans.cpp":71:8)
#loc9 = loc("kmeans.cpp":71:1)
#loc10 = loc("kmeans.cpp":72:1)
#loc11 = loc("kmeans.cpp":73:1)
#loc12 = loc("kmeans.cpp":75:5)
#loc13 = loc("kmeans.cpp":77:1)
#loc14 = loc("kmeans.cpp":77:9)
#loc16 = loc("kmeans.cpp":77:38)
#loc17 = loc("kmeans.cpp":77:25)
#loc18 = loc("kmeans.cpp":79:6)
#loc19 = loc("kmeans.cpp":80:6)
#loc20 = loc("kmeans.cpp":81:9)
#loc21 = loc("kmeans.cpp":83:1)
#loc23 = loc("kmeans.cpp":7:34)
#loc24 = loc("./kmeans.h":15:19)
#loc26 = loc("kmeans.cpp":6:16)
#loc27 = loc("kmeans.cpp":4:1)
#loc29 = loc("kmeans.cpp":6:1)
#loc30 = loc("kmeans.cpp":6:21)
#loc31 = loc("kmeans.cpp":6:5)
#loc32 = loc("kmeans.cpp":7:1)
#loc33 = loc("kmeans.cpp":7:22)
#loc34 = loc("kmeans.cpp":7:6)
#loc35 = loc("kmeans.cpp":8:20)
#loc36 = loc("kmeans.cpp":8:30)
#loc37 = loc("kmeans.cpp":8:32)
#loc38 = loc("kmeans.cpp":8:4)
#loc39 = loc("kmeans.cpp":8:36)
#loc40 = loc("kmeans.cpp":8:34)
#loc41 = loc("kmeans.cpp":6:33)
#loc42 = loc("kmeans.cpp":11:1)
#loc44 = loc("kmeans.cpp":16:34)
#loc46 = loc("kmeans.cpp":15:16)
#loc47 = loc("kmeans.cpp":13:1)
#loc49 = loc("kmeans.cpp":15:1)
#loc50 = loc("kmeans.cpp":15:21)
#loc51 = loc("kmeans.cpp":15:5)
#loc52 = loc("kmeans.cpp":16:1)
#loc53 = loc("kmeans.cpp":16:22)
#loc54 = loc("kmeans.cpp":16:6)
#loc55 = loc("kmeans.cpp":17:19)
#loc56 = loc("kmeans.cpp":17:29)
#loc57 = loc("kmeans.cpp":17:31)
#loc58 = loc("kmeans.cpp":17:4)
#loc59 = loc("kmeans.cpp":17:52)
#loc60 = loc("kmeans.cpp":17:62)
#loc61 = loc("kmeans.cpp":17:65)
#loc62 = loc("kmeans.cpp":17:75)
#loc63 = loc("kmeans.cpp":17:77)
#loc64 = loc("kmeans.cpp":17:35)
#loc65 = loc("kmeans.cpp":17:33)
#loc66 = loc("kmeans.cpp":15:33)
#loc67 = loc("kmeans.cpp":20:1)
#loc69 = loc("kmeans.cpp":34:54)
#loc70 = loc("kmeans.cpp":32:26)
#loc71 = loc("./kmeans.h":12:17)
#loc72 = loc("kmeans.cpp":26:21)
#loc73 = loc("kmeans.cpp":22:1)
#loc76 = loc("kmeans.cpp":26:1)
#loc77 = loc("kmeans.cpp":26:26)
#loc78 = loc("kmeans.cpp":26:8)
#loc79 = loc("kmeans.cpp":31:1)
#loc80 = loc("kmeans.cpp":31:12)
#loc81 = loc("kmeans.cpp":31:35)
#loc82 = loc("kmeans.cpp":31:17)
#loc87 = loc("kmeans.cpp":34:1)
#loc88 = loc("kmeans.cpp":34:16)
#loc89 = loc("kmeans.cpp":34:40)
#loc90 = loc("kmeans.cpp":34:22)
#loc91 = loc("kmeans.cpp":35:54)
#loc92 = loc("kmeans.cpp":35:58)
#loc93 = loc("kmeans.cpp":35:61)
#loc94 = loc("kmeans.cpp":35:30)
#loc95 = loc("kmeans.cpp":35:90)
#loc96 = loc("kmeans.cpp":35:94)
#loc97 = loc("kmeans.cpp":35:97)
#loc98 = loc("kmeans.cpp":35:65)
#loc99 = loc("kmeans.cpp":35:63)
#loc100 = loc("kmeans.cpp":36:30)
#loc101 = loc("kmeans.cpp":36:22)
#loc102 = loc("kmeans.cpp":39:22)
#loc103 = loc("kmeans.cpp":39:13)
#loc104 = loc("kmeans.cpp":31:49)
#loc105 = loc("kmeans.cpp":46:27)
#loc106 = loc("kmeans.cpp":46:9)
#loc107 = loc("kmeans.cpp":46:29)
#loc108 = loc("kmeans.cpp":26:40)
#loc109 = loc("kmeans.cpp":48:1)
#loc111 = loc("kmeans.cpp":52:33)
#loc112 = loc("kmeans.cpp":52:16)
#loc113 = loc("kmeans.cpp":50:1)
#loc114 = loc("kmeans.cpp":52:10)
#loc115 = loc("kmeans.cpp":52:1)
#loc116 = loc("kmeans.cpp":52:21)
#loc117 = loc("kmeans.cpp":52:5)
#loc118 = loc("kmeans.cpp":53:22)
#loc119 = loc("kmeans.cpp":53:32)
#loc120 = loc("kmeans.cpp":53:34)
#loc121 = loc("kmeans.cpp":53:3)
#loc122 = loc("kmeans.cpp":53:56)
#loc123 = loc("kmeans.cpp":53:38)
#loc124 = loc("kmeans.cpp":53:36)
#loc125 = loc("kmeans.cpp":55:1)
