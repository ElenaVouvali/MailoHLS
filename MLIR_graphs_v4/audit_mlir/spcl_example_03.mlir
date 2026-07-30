#loc1 = loc("Example3.cpp":3:17)
#loc3 = loc("Example3.cpp":16:28)
#loc5 = loc("./Example3.h":4:19)
#loc8 = loc("Example3.cpp":24:7)
#loc9 = loc("Example3.cpp":17:13)
#loc37 = loc("Example3.cpp":21:7)
#loc38 = loc("Example3.cpp":20:7)
#loc39 = loc("Example3.cpp":19:7)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @Stencil2D(%arg0: memref<1000000xf32> loc("Example3.cpp":3:17), %arg1: memref<1000000xf32> loc("Example3.cpp":3:17)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333000e-01 : f32 loc(#loc2)
    %c999_i32 = arith.constant 999 : i32 loc(#loc3)
    %c1_i32 = arith.constant 1 : i32 loc(#loc4)
    %c1000_i32 = arith.constant 1000 : i32 loc(#loc5)
    %c0_i32 = arith.constant 0 : i32 loc(#loc6)
    %true = arith.constant true loc(#loc7)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc8)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc9)
    %alloca = memref.alloca() : memref<1000xf32> loc(#loc10)
    %alloca_0 = memref.alloca() : memref<1000xf32> loc(#loc11)
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
            %3 = scf.while (%arg2 = %2) : (i32) -> i32 {
              %4 = arith.cmpi slt, %arg2, %c1000_i32 : i32 loc(#loc15)
              scf.condition(%4) %arg2 : i32 loc(#loc16)
            } do {
            ^bb0(%arg2: i32 loc("./Example3.h":4:19)):
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg2 : i32 to index loc(#loc17)
                  %6 = "polygeist.subindex"(%alloca_0, %5) : (memref<1000xf32>, index) -> memref<?xf32> loc(#loc18)
                  %7 = "polygeist.subindex"(%arg0, %5) : (memref<1000000xf32>, index) -> memref<?xf32> loc(#loc19)
                  %8 = affine.load %7[0] : memref<?xf32> loc(#loc19)
                  affine.store %8, %6[0] : memref<?xf32> loc(#loc20)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg2, %c1_i32 : i32 loc(#loc4)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg2 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc16)
            } loc(#loc5)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc21)
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
            %3 = scf.while (%arg2 = %2) : (i32) -> i32 {
              %4 = arith.cmpi slt, %arg2, %c1000_i32 : i32 loc(#loc22)
              scf.condition(%4) %arg2 : i32 loc(#loc23)
            } do {
            ^bb0(%arg2: i32 loc("./Example3.h":4:19)):
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg2 : i32 to index loc(#loc24)
                  %6 = "polygeist.subindex"(%alloca, %5) : (memref<1000xf32>, index) -> memref<?xf32> loc(#loc25)
                  %7 = arith.addi %arg2, %c1000_i32 : i32 loc(#loc26)
                  %8 = arith.index_cast %7 : i32 to index loc(#loc27)
                  %9 = "polygeist.subindex"(%arg0, %8) : (memref<1000000xf32>, index) -> memref<?xf32> loc(#loc28)
                  %10 = affine.load %9[0] : memref<?xf32> loc(#loc28)
                  affine.store %10, %6[0] : memref<?xf32> loc(#loc29)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg2, %c1_i32 : i32 loc(#loc30)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg2 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc23)
            } loc(#loc5)
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
        scf.if %true {
          scf.execute_region {
            %2 = scf.if %true -> (i32) {
              %4 = scf.execute_region -> i32 {
                scf.yield %c1_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4 : i32 loc(#loc)
            } else {
              scf.yield %1 : i32 loc(#loc)
            } loc(#loc)
            %3:6 = scf.while (%arg2 = %0, %arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %1, %arg7 = %2) : (f32, f32, f32, f32, i32, i32) -> (f32, f32, f32, f32, i32, i32) {
              %4 = arith.cmpi slt, %arg7, %c999_i32 : i32 loc(#loc32)
              scf.condition(%4) %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : f32, f32, f32, f32, i32, i32 loc(#loc33)
            } do {
            ^bb0(%arg2: f32 loc("Example3.cpp":16:28), %arg3: f32 loc("Example3.cpp":16:28), %arg4: f32 loc("Example3.cpp":16:28), %arg5: f32 loc("Example3.cpp":16:28), %arg6: i32 loc("Example3.cpp":16:28), %arg7: i32 loc("Example3.cpp":16:28)):
              %4:5 = scf.if %true -> (f32, f32, f32, f32, i32) {
                %6:5 = scf.execute_region -> (f32, f32, f32, f32, i32) {
                  cf.br ^bb1 loc(#loc34)
                ^bb1:  // pred: ^bb0
                  %7:5 = scf.if %true -> (f32, f32, f32, f32, i32) {
                    %8:5 = scf.execute_region -> (f32, f32, f32, f32, i32) {
                      %9 = scf.if %true -> (i32) {
                        %11 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %11 : i32 loc(#loc)
                      } else {
                        scf.yield %arg6 : i32 loc(#loc)
                      } loc(#loc)
                      %10:5 = scf.while (%arg8 = %arg2, %arg9 = %arg3, %arg10 = %arg4, %arg11 = %arg5, %arg12 = %9) : (f32, f32, f32, f32, i32) -> (f32, f32, f32, f32, i32) {
                        %11 = arith.cmpi slt, %arg12, %c1000_i32 : i32 loc(#loc35)
                        scf.condition(%11) %arg8, %arg9, %arg10, %arg11, %arg12 : f32, f32, f32, f32, i32 loc(#loc36)
                      } do {
                      ^bb0(%arg8: f32 loc("Example3.cpp":24:7), %arg9: f32 loc("Example3.cpp":21:7), %arg10: f32 loc("Example3.cpp":20:7), %arg11: f32 loc("Example3.cpp":19:7), %arg12: i32 loc("Example3.cpp":17:13)):
                        %11 = scf.if %true -> (f32) {
                          %16 = scf.execute_region -> f32 {
                            %17 = scf.if %true -> (f32) {
                              %18 = scf.execute_region -> f32 {
                                %19 = arith.index_cast %arg12 : i32 to index loc(#loc40)
                                %20 = "polygeist.subindex"(%alloca_0, %19) : (memref<1000xf32>, index) -> memref<?xf32> loc(#loc41)
                                %21 = affine.load %20[0] : memref<?xf32> loc(#loc41)
                                scf.yield %21 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %18 : f32 loc(#loc)
                            } else {
                              scf.yield %arg11 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %16 : f32 loc(#loc)
                        } else {
                          scf.yield %arg11 : f32 loc(#loc)
                        } loc(#loc)
                        %12 = scf.if %true -> (f32) {
                          %16 = scf.execute_region -> f32 {
                            %17 = scf.if %true -> (f32) {
                              %18 = scf.execute_region -> f32 {
                                %19 = arith.index_cast %arg12 : i32 to index loc(#loc42)
                                %20 = "polygeist.subindex"(%alloca, %19) : (memref<1000xf32>, index) -> memref<?xf32> loc(#loc43)
                                %21 = affine.load %20[0] : memref<?xf32> loc(#loc43)
                                scf.yield %21 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %18 : f32 loc(#loc)
                            } else {
                              scf.yield %arg10 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %16 : f32 loc(#loc)
                        } else {
                          scf.yield %arg10 : f32 loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (f32) {
                          %16 = scf.execute_region -> f32 {
                            %17 = scf.if %true -> (f32) {
                              %18 = scf.execute_region -> f32 {
                                %19 = arith.addi %arg7, %c1_i32 : i32 loc(#loc44)
                                %20 = arith.muli %19, %c1000_i32 : i32 loc(#loc45)
                                %21 = arith.addi %20, %arg12 : i32 loc(#loc46)
                                %22 = arith.index_cast %21 : i32 to index loc(#loc47)
                                %23 = "polygeist.subindex"(%arg0, %22) : (memref<1000000xf32>, index) -> memref<?xf32> loc(#loc48)
                                %24 = affine.load %23[0] : memref<?xf32> loc(#loc48)
                                scf.yield %24 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %18 : f32 loc(#loc)
                            } else {
                              scf.yield %arg9 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %16 : f32 loc(#loc)
                        } else {
                          scf.yield %arg9 : f32 loc(#loc)
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
                        %14 = scf.if %true -> (f32) {
                          %16 = scf.execute_region -> f32 {
                            %17 = scf.if %true -> (f32) {
                              %18 = scf.execute_region -> f32 {
                                %19 = arith.addf %11, %12 : f32 loc(#loc49)
                                %20 = arith.addf %19, %13 : f32 loc(#loc50)
                                %21 = arith.mulf %20, %cst : f32 loc(#loc51)
                                scf.yield %21 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %18 : f32 loc(#loc)
                            } else {
                              scf.yield %arg8 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %16 : f32 loc(#loc)
                        } else {
                          scf.yield %arg8 : f32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %16 = arith.muli %arg7, %c1000_i32 : i32 loc(#loc52)
                            %17 = arith.addi %16, %arg12 : i32 loc(#loc53)
                            %18 = arith.index_cast %17 : i32 to index loc(#loc54)
                            %19 = "polygeist.subindex"(%arg1, %18) : (memref<1000000xf32>, index) -> memref<?xf32> loc(#loc55)
                            affine.store %14, %19[0] : memref<?xf32> loc(#loc56)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %15 = scf.if %true -> (i32) {
                          %16 = scf.execute_region -> i32 {
                            %17 = arith.addi %arg12, %c1_i32 : i32 loc(#loc57)
                            scf.yield %17 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %16 : i32 loc(#loc)
                        } else {
                          scf.yield %arg12 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14, %13, %12, %11, %15 : f32, f32, f32, f32, i32 loc(#loc36)
                      } loc(#loc5)
                      scf.yield %10#0, %10#1, %10#2, %10#3, %10#4 : f32, f32, f32, f32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %8#0, %8#1, %8#2, %8#3, %8#4 : f32, f32, f32, f32, i32 loc(#loc)
                  } else {
                    scf.yield %arg2, %arg3, %arg4, %arg5, %arg6 : f32, f32, f32, f32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %7#0, %7#1, %7#2, %7#3, %7#4 : f32, f32, f32, f32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %6#0, %6#1, %6#2, %6#3, %6#4 : f32, f32, f32, f32, i32 loc(#loc)
              } else {
                scf.yield %arg2, %arg3, %arg4, %arg5, %arg6 : f32, f32, f32, f32, i32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %6 = scf.execute_region -> i32 {
                  %7 = arith.addi %arg7, %c1_i32 : i32 loc(#loc58)
                  scf.yield %7 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %6 : i32 loc(#loc)
              } else {
                scf.yield %arg7 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4#0, %4#1, %4#2, %4#3, %4#4, %5 : f32, f32, f32, f32, i32, i32 loc(#loc33)
            } loc(#loc3)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc59)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("Example3.cpp":23:32)
#loc4 = loc("Example3.cpp":8:29)
#loc6 = loc("Example3.cpp":8:19)
#loc7 = loc("Example3.cpp":3:12)
#loc10 = loc("Example3.cpp":6:6)
#loc11 = loc("Example3.cpp":5:6)
#loc12 = loc("Example3.cpp":5:1)
#loc13 = loc("Example3.cpp":6:1)
#loc14 = loc("Example3.cpp":8:1)
#loc15 = loc("Example3.cpp":8:24)
#loc16 = loc("Example3.cpp":8:6)
#loc17 = loc("Example3.cpp":9:19)
#loc18 = loc("Example3.cpp":9:5)
#loc19 = loc("Example3.cpp":9:23)
#loc20 = loc("Example3.cpp":9:21)
#loc21 = loc("Example3.cpp":12:1)
#loc22 = loc("Example3.cpp":12:24)
#loc23 = loc("Example3.cpp":12:6)
#loc24 = loc("Example3.cpp":13:20)
#loc25 = loc("Example3.cpp":13:5)
#loc26 = loc("Example3.cpp":13:36)
#loc27 = loc("Example3.cpp":13:39)
#loc28 = loc("Example3.cpp":13:24)
#loc29 = loc("Example3.cpp":13:22)
#loc30 = loc("Example3.cpp":12:29)
#loc31 = loc("Example3.cpp":16:1)
#loc32 = loc("Example3.cpp":16:24)
#loc33 = loc("Example3.cpp":16:6)
#loc34 = loc("Example3.cpp":17:1)
#loc35 = loc("Example3.cpp":17:26)
#loc36 = loc("Example3.cpp":17:8)
#loc40 = loc("Example3.cpp":19:40)
#loc41 = loc("Example3.cpp":19:26)
#loc42 = loc("Example3.cpp":20:42)
#loc43 = loc("Example3.cpp":20:27)
#loc44 = loc("Example3.cpp":21:39)
#loc45 = loc("Example3.cpp":21:44)
#loc46 = loc("Example3.cpp":21:48)
#loc47 = loc("Example3.cpp":21:51)
#loc48 = loc("Example3.cpp":21:26)
#loc49 = loc("Example3.cpp":24:44)
#loc50 = loc("Example3.cpp":24:53)
#loc51 = loc("Example3.cpp":24:35)
#loc52 = loc("Example3.cpp":26:20)
#loc53 = loc("Example3.cpp":26:24)
#loc54 = loc("Example3.cpp":26:27)
#loc55 = loc("Example3.cpp":26:7)
#loc56 = loc("Example3.cpp":26:29)
#loc57 = loc("Example3.cpp":17:31)
#loc58 = loc("Example3.cpp":16:33)
#loc59 = loc("Example3.cpp":29:1)
