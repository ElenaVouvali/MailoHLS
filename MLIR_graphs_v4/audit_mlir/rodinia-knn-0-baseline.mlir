#loc1 = loc("knn.cpp":4:6)
#loc5 = loc("./knn.h":16:39)
#loc8 = loc("knn.cpp":21:16)
#loc9 = loc("knn.cpp":18:5)
#loc16 = loc("knn.cpp":17:5)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<2xf32> loc("knn.cpp":4:6), %arg1: memref<2097152xf32> loc("knn.cpp":4:6), %arg2: memref<1048576xf32> loc("knn.cpp":4:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c2_i32 = arith.constant 2 : i32 loc(#loc3)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc4)
    %c1048576_i32 = arith.constant 1048576 : i32 loc(#loc5)
    %c0_i32 = arith.constant 0 : i32 loc(#loc6)
    %true = arith.constant true loc(#loc7)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc8)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc9)
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
              %4 = arith.cmpi slt, %arg4, %c1048576_i32 : i32 loc(#loc11)
              scf.condition(%4) %arg3, %arg4, %arg5, %arg6 : i32, i32, f32, f32 loc(#loc12)
            } do {
            ^bb0(%arg3: i32 loc("./knn.h":16:39), %arg4: i32 loc("./knn.h":16:39), %arg5: f32 loc("./knn.h":16:39), %arg6: f32 loc("./knn.h":16:39)):
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
                  cf.br ^bb1 loc(#loc13)
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
                        %12 = arith.cmpi slt, %arg7, %c2_i32 : i32 loc(#loc14)
                        scf.condition(%12) %arg7, %arg8, %arg9 : i32, f32, f32 loc(#loc15)
                      } do {
                      ^bb0(%arg7: i32 loc("knn.cpp":21:16), %arg8: f32 loc("knn.cpp":18:5), %arg9: f32 loc("knn.cpp":17:5)):
                        %12 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = arith.muli %arg4, %c2_i32 : i32 loc(#loc17)
                            %17 = arith.addi %16, %arg7 : i32 loc(#loc18)
                            %18 = arith.index_cast %17 : i32 to index loc(#loc19)
                            %19 = "polygeist.subindex"(%arg1, %18) : (memref<2097152xf32>, index) -> memref<?xf32> loc(#loc20)
                            %20 = affine.load %19[0] : memref<?xf32> loc(#loc20)
                            %21 = arith.index_cast %arg7 : i32 to index loc(#loc21)
                            %22 = "polygeist.subindex"(%arg0, %21) : (memref<2xf32>, index) -> memref<?xf32> loc(#loc22)
                            %23 = affine.load %22[0] : memref<?xf32> loc(#loc22)
                            %24 = arith.subf %20, %23 : f32 loc(#loc23)
                            scf.yield %24 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg8 : f32 loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = arith.mulf %12, %12 : f32 loc(#loc24)
                            %17 = arith.addf %arg9, %16 : f32 loc(#loc25)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg9 : f32 loc(#loc)
                        } loc(#loc)
                        %14 = scf.if %true -> (i32) {
                          %15 = scf.execute_region -> i32 {
                            %16 = arith.addi %arg7, %c1_i32 : i32 loc(#loc2)
                            scf.yield %16 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : i32 loc(#loc)
                        } else {
                          scf.yield %arg7 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14, %12, %13 : i32, f32, f32 loc(#loc15)
                      } loc(#loc3)
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
                  %7 = arith.index_cast %arg4 : i32 to index loc(#loc26)
                  %8 = "polygeist.subindex"(%arg2, %7) : (memref<1048576xf32>, index) -> memref<?xf32> loc(#loc27)
                  affine.store %5#2, %8[0] : memref<?xf32> loc(#loc28)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (i32) {
                %7 = scf.execute_region -> i32 {
                  %8 = arith.addi %arg4, %c1_i32 : i32 loc(#loc29)
                  scf.yield %8 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %7 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %5#0, %6, %5#1, %5#2 : i32, i32, f32, f32 loc(#loc12)
            } loc(#loc5)
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
    return loc(#loc30)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("knn.cpp":21:44)
#loc3 = loc("./knn.h":15:25)
#loc4 = loc("knn.cpp":20:15)
#loc6 = loc("knn.cpp":19:20)
#loc7 = loc("knn.cpp":4:1)
#loc10 = loc("knn.cpp":19:1)
#loc11 = loc("knn.cpp":19:25)
#loc12 = loc("knn.cpp":19:8)
#loc13 = loc("knn.cpp":21:1)
#loc14 = loc("knn.cpp":21:29)
#loc15 = loc("knn.cpp":21:12)
#loc17 = loc("knn.cpp":22:42)
#loc18 = loc("knn.cpp":22:54)
#loc19 = loc("knn.cpp":22:56)
#loc20 = loc("knn.cpp":22:29)
#loc21 = loc("knn.cpp":22:72)
#loc22 = loc("knn.cpp":22:60)
#loc23 = loc("knn.cpp":22:58)
#loc24 = loc("knn.cpp":23:33)
#loc25 = loc("knn.cpp":23:17)
#loc26 = loc("knn.cpp":25:19)
#loc27 = loc("knn.cpp":25:9)
#loc28 = loc("knn.cpp":25:21)
#loc29 = loc("knn.cpp":19:50)
#loc30 = loc("knn.cpp":29:1)
