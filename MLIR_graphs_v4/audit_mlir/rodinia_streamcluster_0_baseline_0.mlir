#loc1 = loc("streamcluster.cpp":4:6)
#loc4 = loc("./streamcluster.h":9:13)
#loc6 = loc("./streamcluster.h":10:20)
#loc17 = loc("streamcluster.cpp":45:13)
#loc18 = loc("streamcluster.cpp":43:9)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<?xf32> loc("streamcluster.cpp":4:6), %arg1: memref<?xf32> loc("streamcluster.cpp":4:6), %arg2: memref<?xf32> loc("streamcluster.cpp":4:6), %arg3: memref<?xf32> loc("streamcluster.cpp":4:6), %arg4: memref<?xi32> loc("streamcluster.cpp":4:6), %arg5: memref<?xi32> loc("streamcluster.cpp":4:6), %arg6: memref<?xi8> loc("streamcluster.cpp":4:6), %arg7: memref<?xf32> loc("streamcluster.cpp":4:6), %arg8: i32 loc("streamcluster.cpp":4:6), %arg9: memref<?xf32> loc("streamcluster.cpp":4:6), %arg10: i32 loc("streamcluster.cpp":4:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i8 = arith.constant 1 : i8 loc(#loc2)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %c200_i32 = arith.constant 200 : i32 loc(#loc4)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc5)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc6)
    %c0_i32 = arith.constant 0 : i32 loc(#loc7)
    %true = arith.constant true loc(#loc8)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc9)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc10)
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
        cf.br ^bb1 loc(#loc11)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2:5 = scf.while (%arg11 = %0, %arg12 = %1, %arg13 = %1, %arg14 = %1, %arg15 = %c0_i32) : (i32, f32, f32, f32, i32) -> (i32, f32, f32, f32, i32) {
              %3 = arith.cmpi slt, %arg15, %c1024_i32 : i32 loc(#loc12)
              scf.condition(%3) %arg11, %arg12, %arg13, %arg14, %arg15 : i32, f32, f32, f32, i32 loc(#loc13)
            } do {
            ^bb0(%arg11: i32 loc("./streamcluster.h":10:20), %arg12: f32 loc("./streamcluster.h":10:20), %arg13: f32 loc("./streamcluster.h":10:20), %arg14: f32 loc("./streamcluster.h":10:20), %arg15: i32 loc("./streamcluster.h":10:20)):
              %3 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      scf.yield %cst : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg14 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg14 : f32 loc(#loc)
              } loc(#loc)
              %4:2 = scf.if %true -> (f32, f32) {
                %8:2 = scf.execute_region -> (f32, f32) {
                  cf.br ^bb1 loc(#loc14)
                ^bb1:  // pred: ^bb0
                  %9:2 = scf.if %true -> (f32, f32) {
                    %10:2 = scf.execute_region -> (f32, f32) {
                      %11:3 = scf.while (%arg16 = %arg13, %arg17 = %3, %arg18 = %c0_i32) : (f32, f32, i32) -> (f32, f32, i32) {
                        %12 = arith.cmpi slt, %arg18, %c200_i32 : i32 loc(#loc15)
                        scf.condition(%12) %arg16, %arg17, %arg18 : f32, f32, i32 loc(#loc16)
                      } do {
                      ^bb0(%arg16: f32 loc("streamcluster.cpp":45:13), %arg17: f32 loc("streamcluster.cpp":43:9), %arg18: i32 loc("./streamcluster.h":9:13)):
                        %12 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = scf.if %true -> (f32) {
                              %17 = scf.execute_region -> f32 {
                                %18 = arith.muli %arg15, %c200_i32 : i32 loc(#loc19)
                                %19 = arith.addi %18, %arg18 : i32 loc(#loc20)
                                %20 = arith.index_cast %19 : i32 to index loc(#loc21)
                                %21 = "polygeist.subindex"(%arg0, %20) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc22)
                                %22 = affine.load %21[0] : memref<?xf32> loc(#loc22)
                                %23 = arith.index_cast %arg18 : i32 to index loc(#loc23)
                                %24 = "polygeist.subindex"(%arg3, %23) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc24)
                                %25 = affine.load %24[0] : memref<?xf32> loc(#loc24)
                                %26 = arith.subf %22, %25 : f32 loc(#loc25)
                                scf.yield %26 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %17 : f32 loc(#loc)
                            } else {
                              scf.yield %arg16 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %16 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg16 : f32 loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (f32) {
                          %15 = scf.execute_region -> f32 {
                            %16 = arith.mulf %12, %12 : f32 loc(#loc26)
                            %17 = arith.addf %arg17, %16 : f32 loc(#loc27)
                            scf.yield %17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : f32 loc(#loc)
                        } else {
                          scf.yield %arg17 : f32 loc(#loc)
                        } loc(#loc)
                        %14 = scf.if %true -> (i32) {
                          %15 = scf.execute_region -> i32 {
                            %16 = arith.addi %arg18, %c1_i32 : i32 loc(#loc3)
                            scf.yield %16 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %15 : i32 loc(#loc)
                        } else {
                          scf.yield %arg18 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %12, %13, %14 : f32, f32, i32 loc(#loc16)
                      } loc(#loc4)
                      scf.yield %11#0, %11#1 : f32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10#0, %10#1 : f32, f32 loc(#loc)
                  } else {
                    scf.yield %arg13, %3 : f32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9#0, %9#1 : f32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %8#0, %8#1 : f32, f32 loc(#loc)
              } else {
                scf.yield %arg13, %3 : f32, f32 loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      %11 = arith.index_cast %arg15 : i32 to index loc(#loc28)
                      %12 = "polygeist.subindex"(%arg1, %11) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc29)
                      %13 = affine.load %12[0] : memref<?xf32> loc(#loc29)
                      %14 = arith.mulf %4#1, %13 : f32 loc(#loc30)
                      %15 = "polygeist.subindex"(%arg2, %11) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc31)
                      %16 = affine.load %15[0] : memref<?xf32> loc(#loc31)
                      %17 = arith.subf %14, %16 : f32 loc(#loc32)
                      scf.yield %17 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg12 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg12 : f32 loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = scf.if %true -> (i32) {
                    %10 = scf.execute_region -> i32 {
                      %11 = arith.index_cast %arg15 : i32 to index loc(#loc33)
                      %12 = "polygeist.subindex"(%arg4, %11) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc34)
                      %13 = affine.load %12[0] : memref<?xi32> loc(#loc34)
                      %14 = arith.index_cast %13 : i32 to index loc(#loc35)
                      %15 = "polygeist.subindex"(%arg5, %14) : (memref<?xi32>, index) -> memref<?xi32> loc(#loc36)
                      %16 = affine.load %15[0] : memref<?xi32> loc(#loc36)
                      scf.yield %16 : i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : i32 loc(#loc)
                  } else {
                    scf.yield %arg11 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg11 : i32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  scf.if %true {
                    scf.execute_region {
                      %8 = arith.cmpf olt, %5, %cst : f32 loc(#loc37)
                      scf.if %8 {
                        scf.if %true {
                          scf.execute_region {
                            %9 = arith.index_cast %arg15 : i32 to index loc(#loc39)
                            %10 = "polygeist.subindex"(%arg6, %9) : (memref<?xi8>, index) -> memref<?xi8> loc(#loc40)
                            affine.store %c1_i8, %10[0] : memref<?xi8> loc(#loc41)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %9 = affine.load %arg9[0] : memref<?xf32> loc(#loc42)
                            %10 = arith.addf %9, %5 : f32 loc(#loc42)
                            affine.store %10, %arg9[0] : memref<?xf32> loc(#loc42)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } else {
                        scf.if %true {
                          scf.execute_region {
                            %9 = arith.index_cast %6 : i32 to index loc(#loc43)
                            %10 = "polygeist.subindex"(%arg7, %9) : (memref<?xf32>, index) -> memref<?xf32> loc(#loc44)
                            %11 = affine.load %10[0] : memref<?xf32> loc(#loc45)
                            %12 = arith.subf %11, %5 : f32 loc(#loc45)
                            affine.store %12, %10[0] : memref<?xf32> loc(#loc45)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                      } loc(#loc38)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg15, %c1_i32 : i32 loc(#loc46)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg15 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %6, %5, %4#0, %4#1, %7 : i32, f32, f32, f32, i32 loc(#loc13)
            } loc(#loc6)
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
    return loc(#loc47)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("streamcluster.cpp":56:36)
#loc3 = loc("streamcluster.cpp":44:34)
#loc5 = loc("streamcluster.cpp":43:21)
#loc7 = loc("streamcluster.cpp":42:17)
#loc8 = loc("streamcluster.cpp":4:1)
#loc9 = loc("streamcluster.cpp":49:9)
#loc10 = loc("streamcluster.cpp":48:9)
#loc11 = loc("streamcluster.cpp":42:1)
#loc12 = loc("streamcluster.cpp":42:22)
#loc13 = loc("streamcluster.cpp":42:8)
#loc14 = loc("streamcluster.cpp":44:1)
#loc15 = loc("streamcluster.cpp":44:26)
#loc16 = loc("streamcluster.cpp":44:12)
#loc19 = loc("streamcluster.cpp":45:31)
#loc20 = loc("streamcluster.cpp":45:37)
#loc21 = loc("streamcluster.cpp":45:40)
#loc22 = loc("streamcluster.cpp":45:23)
#loc23 = loc("streamcluster.cpp":45:52)
#loc24 = loc("streamcluster.cpp":45:44)
#loc25 = loc("streamcluster.cpp":45:42)
#loc26 = loc("streamcluster.cpp":46:22)
#loc27 = loc("streamcluster.cpp":46:17)
#loc28 = loc("streamcluster.cpp":48:44)
#loc29 = loc("streamcluster.cpp":48:36)
#loc30 = loc("streamcluster.cpp":48:34)
#loc31 = loc("streamcluster.cpp":48:48)
#loc32 = loc("streamcluster.cpp":48:46)
#loc33 = loc("streamcluster.cpp":49:55)
#loc34 = loc("streamcluster.cpp":49:47)
#loc35 = loc("streamcluster.cpp":49:56)
#loc36 = loc("streamcluster.cpp":49:34)
#loc37 = loc("streamcluster.cpp":50:26)
#loc38 = loc("streamcluster.cpp":50:9)
#loc39 = loc("streamcluster.cpp":56:32)
#loc40 = loc("streamcluster.cpp":56:13)
#loc41 = loc("streamcluster.cpp":56:34)
#loc42 = loc("streamcluster.cpp":57:34)
#loc43 = loc("streamcluster.cpp":69:40)
#loc44 = loc("streamcluster.cpp":69:13)
#loc45 = loc("streamcluster.cpp":69:42)
#loc46 = loc("streamcluster.cpp":42:37)
#loc47 = loc("streamcluster.cpp":74:1)
