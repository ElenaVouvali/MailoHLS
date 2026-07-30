#loc1 = loc("dilate.cpp":5:6)
#loc8 = loc("./dilate.h":12:19)
#loc13 = loc("dilate.cpp":27:6)
#loc14 = loc("dilate.cpp":23:4)
#loc23 = loc("dilate.cpp":26:6)
#loc24 = loc("dilate.cpp":25:13)
#loc25 = loc("dilate.cpp":24:12)
#loc26 = loc("dilate.cpp":22:11)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<278528xf32> loc("dilate.cpp":5:6), %arg1: memref<280576xf32> loc("dilate.cpp":5:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i8 = arith.constant 0 : i8 loc(#loc2)
    %c1_i8 = arith.constant 1 : i8 loc(#loc3)
    %c1_i32 = arith.constant 1 : i32 loc(#loc4)
    %c5_i32 = arith.constant 5 : i32 loc(#loc5)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc6)
    %c512_i32 = arith.constant 512 : i32 loc(#loc7)
    %c544_i32 = arith.constant 544 : i32 loc(#loc8)
    %c0_i32 = arith.constant 0 : i32 loc(#loc9)
    %c2_i32 = arith.constant 2 : i32 loc(#loc10)
    %false = arith.constant false loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc11)
    %true = arith.constant true loc(#loc12)
    %c1 = arith.constant 1 : index loc(#loc)
    %c2 = arith.constant 2 : index loc(#loc)
    %c3 = arith.constant 3 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c5 = arith.constant 5 : index loc(#loc)
    %c6 = arith.constant 6 : index loc(#loc)
    %c7 = arith.constant 7 : index loc(#loc)
    %c8 = arith.constant 8 : index loc(#loc)
    %c9 = arith.constant 9 : index loc(#loc)
    %c10 = arith.constant 10 : index loc(#loc)
    %c11 = arith.constant 11 : index loc(#loc)
    %c12 = arith.constant 12 : index loc(#loc)
    %c13 = arith.constant 13 : index loc(#loc)
    %c14 = arith.constant 14 : index loc(#loc)
    %c15 = arith.constant 15 : index loc(#loc)
    %c16 = arith.constant 16 : index loc(#loc)
    %c17 = arith.constant 17 : index loc(#loc)
    %c18 = arith.constant 18 : index loc(#loc)
    %c19 = arith.constant 19 : index loc(#loc)
    %c20 = arith.constant 20 : index loc(#loc)
    %c21 = arith.constant 21 : index loc(#loc)
    %c22 = arith.constant 22 : index loc(#loc)
    %c23 = arith.constant 23 : index loc(#loc)
    %c24 = arith.constant 24 : index loc(#loc)
    %c0 = arith.constant 0 : index loc(#loc13)
    %0 = "polygeist.undef"() : () -> i32 loc(#loc13)
    %1 = "polygeist.undef"() : () -> f32 loc(#loc14)
    %alloca = memref.alloca() : memref<25xi8> loc(#loc15)
    %2 = scf.if %true -> (i32) {
      %5 = scf.execute_region -> i32 {
        %6 = scf.if %true -> (i32) {
          %7 = scf.execute_region -> i32 {
            scf.yield %c1024_i32 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %7 : i32 loc(#loc)
        } else {
          scf.yield %0 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %6 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %5 : i32 loc(#loc)
    } else {
      scf.yield %0 : i32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc16)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %5 = "polygeist.subindex"(%alloca, %c0) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %5[0] : memref<?xi8> loc(#loc15)
            %6 = "polygeist.subindex"(%alloca, %c1) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %6[0] : memref<?xi8> loc(#loc15)
            %7 = "polygeist.subindex"(%alloca, %c2) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %7[0] : memref<?xi8> loc(#loc15)
            %8 = "polygeist.subindex"(%alloca, %c3) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %8[0] : memref<?xi8> loc(#loc15)
            %9 = "polygeist.subindex"(%alloca, %c4) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %9[0] : memref<?xi8> loc(#loc15)
            %10 = "polygeist.subindex"(%alloca, %c5) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %10[0] : memref<?xi8> loc(#loc15)
            %11 = "polygeist.subindex"(%alloca, %c6) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %11[0] : memref<?xi8> loc(#loc15)
            %12 = "polygeist.subindex"(%alloca, %c7) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %12[0] : memref<?xi8> loc(#loc15)
            %13 = "polygeist.subindex"(%alloca, %c8) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %13[0] : memref<?xi8> loc(#loc15)
            %14 = "polygeist.subindex"(%alloca, %c9) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %14[0] : memref<?xi8> loc(#loc15)
            %15 = "polygeist.subindex"(%alloca, %c10) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %15[0] : memref<?xi8> loc(#loc15)
            %16 = "polygeist.subindex"(%alloca, %c11) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %16[0] : memref<?xi8> loc(#loc15)
            %17 = "polygeist.subindex"(%alloca, %c12) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %17[0] : memref<?xi8> loc(#loc15)
            %18 = "polygeist.subindex"(%alloca, %c13) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %18[0] : memref<?xi8> loc(#loc15)
            %19 = "polygeist.subindex"(%alloca, %c14) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %19[0] : memref<?xi8> loc(#loc15)
            %20 = "polygeist.subindex"(%alloca, %c15) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %20[0] : memref<?xi8> loc(#loc15)
            %21 = "polygeist.subindex"(%alloca, %c16) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %21[0] : memref<?xi8> loc(#loc15)
            %22 = "polygeist.subindex"(%alloca, %c17) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %22[0] : memref<?xi8> loc(#loc15)
            %23 = "polygeist.subindex"(%alloca, %c18) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %23[0] : memref<?xi8> loc(#loc15)
            %24 = "polygeist.subindex"(%alloca, %c19) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %24[0] : memref<?xi8> loc(#loc15)
            %25 = "polygeist.subindex"(%alloca, %c20) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %25[0] : memref<?xi8> loc(#loc15)
            %26 = "polygeist.subindex"(%alloca, %c21) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %26[0] : memref<?xi8> loc(#loc15)
            %27 = "polygeist.subindex"(%alloca, %c22) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c1_i8, %27[0] : memref<?xi8> loc(#loc15)
            %28 = "polygeist.subindex"(%alloca, %c23) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %28[0] : memref<?xi8> loc(#loc15)
            %29 = "polygeist.subindex"(%alloca, %c24) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc15)
            affine.store %c0_i8, %29[0] : memref<?xi8> loc(#loc15)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    %3 = scf.if %true -> (i32) {
      %5 = scf.execute_region -> i32 {
        %6 = scf.if %true -> (i32) {
          %7 = scf.execute_region -> i32 {
            scf.yield %c2_i32 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %7 : i32 loc(#loc)
        } else {
          scf.yield %0 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %6 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %5 : i32 loc(#loc)
    } else {
      scf.yield %0 : i32 loc(#loc)
    } loc(#loc)
    %4 = scf.if %true -> (i32) {
      %5 = scf.execute_region -> i32 {
        %6 = scf.if %true -> (i32) {
          %7 = scf.execute_region -> i32 {
            scf.yield %c2_i32 : i32 loc(#loc)
          } loc(#loc)
          scf.yield %7 : i32 loc(#loc)
        } else {
          scf.yield %0 : i32 loc(#loc)
        } loc(#loc)
        scf.yield %6 : i32 loc(#loc)
      } loc(#loc)
      scf.yield %5 : i32 loc(#loc)
    } else {
      scf.yield %0 : i32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc17)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %5 = scf.if %true -> (i32) {
              %7 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %7 : i32 loc(#loc)
            } else {
              scf.yield %0 : i32 loc(#loc)
            } loc(#loc)
            %6:8 = scf.while (%arg2 = %0, %arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %1, %arg7 = %1, %arg8 = %0, %arg9 = %5) : (i32, i32, i32, i32, f32, f32, i32, i32) -> (i32, i32, i32, i32, f32, f32, i32, i32) {
              %7 = arith.cmpi slt, %arg9, %c544_i32 : i32 loc(#loc18)
              scf.condition(%7) %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : i32, i32, i32, i32, f32, f32, i32, i32 loc(#loc19)
            } do {
            ^bb0(%arg2: i32 loc("./dilate.h":12:19), %arg3: i32 loc("./dilate.h":12:19), %arg4: i32 loc("./dilate.h":12:19), %arg5: i32 loc("./dilate.h":12:19), %arg6: f32 loc("./dilate.h":12:19), %arg7: f32 loc("./dilate.h":12:19), %arg8: i32 loc("./dilate.h":12:19), %arg9: i32 loc("./dilate.h":12:19)):
              %7:7 = scf.if %true -> (i32, i32, i32, i32, f32, f32, i32) {
                %9:7 = scf.execute_region -> (i32, i32, i32, i32, f32, f32, i32) {
                  cf.br ^bb1 loc(#loc20)
                ^bb1:  // pred: ^bb0
                  %10:7 = scf.if %true -> (i32, i32, i32, i32, f32, f32, i32) {
                    %11:7 = scf.execute_region -> (i32, i32, i32, i32, f32, f32, i32) {
                      %12 = scf.if %true -> (i32) {
                        %14 = scf.execute_region -> i32 {
                          scf.yield %c0_i32 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %14 : i32 loc(#loc)
                      } else {
                        scf.yield %arg8 : i32 loc(#loc)
                      } loc(#loc)
                      %13:7 = scf.while (%arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7, %arg16 = %12) : (i32, i32, i32, i32, f32, f32, i32) -> (i32, i32, i32, i32, f32, f32, i32) {
                        %14 = arith.cmpi slt, %arg16, %c512_i32 : i32 loc(#loc21)
                        scf.condition(%14) %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16 : i32, i32, i32, i32, f32, f32, i32 loc(#loc22)
                      } do {
                      ^bb0(%arg10: i32 loc("dilate.cpp":27:6), %arg11: i32 loc("dilate.cpp":26:6), %arg12: i32 loc("dilate.cpp":25:13), %arg13: i32 loc("dilate.cpp":24:12), %arg14: f32 loc("dilate.cpp":23:4), %arg15: f32 loc("dilate.cpp":23:4), %arg16: i32 loc("dilate.cpp":22:11)):
                        %14 = scf.if %true -> (f32) {
                          %17 = scf.execute_region -> f32 {
                            %18 = scf.if %true -> (f32) {
                              %19 = scf.execute_region -> f32 {
                                scf.yield %cst : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19 : f32 loc(#loc)
                            } else {
                              scf.yield %arg15 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : f32 loc(#loc)
                        } else {
                          scf.yield %arg15 : f32 loc(#loc)
                        } loc(#loc)
                        %15:6 = scf.if %true -> (i32, i32, i32, i32, f32, f32) {
                          %17:6 = scf.execute_region -> (i32, i32, i32, i32, f32, f32) {
                            cf.br ^bb1 loc(#loc27)
                          ^bb1:  // pred: ^bb0
                            %18:6 = scf.if %true -> (i32, i32, i32, i32, f32, f32) {
                              %19:6 = scf.execute_region -> (i32, i32, i32, i32, f32, f32) {
                                %20 = scf.if %true -> (i32) {
                                  %22 = scf.execute_region -> i32 {
                                    scf.yield %c0_i32 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22 : i32 loc(#loc)
                                } else {
                                  scf.yield %arg13 : i32 loc(#loc)
                                } loc(#loc)
                                %21:6 = scf.while (%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %20, %arg21 = %arg14, %arg22 = %14) : (i32, i32, i32, i32, f32, f32) -> (i32, i32, i32, i32, f32, f32) {
                                  %22 = arith.cmpi slt, %arg20, %c5_i32 : i32 loc(#loc28)
                                  scf.condition(%22) %arg17, %arg18, %arg19, %arg20, %arg21, %arg22 : i32, i32, i32, i32, f32, f32 loc(#loc29)
                                } do {
                                ^bb0(%arg17: i32 loc("dilate.cpp":27:6), %arg18: i32 loc("dilate.cpp":26:6), %arg19: i32 loc("dilate.cpp":25:13), %arg20: i32 loc("dilate.cpp":24:12), %arg21: f32 loc("dilate.cpp":23:4), %arg22: f32 loc("dilate.cpp":23:4)):
                                  %22:5 = scf.if %true -> (i32, i32, i32, f32, f32) {
                                    %24:5 = scf.execute_region -> (i32, i32, i32, f32, f32) {
                                      cf.br ^bb1 loc(#loc30)
                                    ^bb1:  // pred: ^bb0
                                      %25:5 = scf.if %true -> (i32, i32, i32, f32, f32) {
                                        %26:5 = scf.execute_region -> (i32, i32, i32, f32, f32) {
                                          %27 = scf.if %true -> (i32) {
                                            %29 = scf.execute_region -> i32 {
                                              scf.yield %c0_i32 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %29 : i32 loc(#loc)
                                          } else {
                                            scf.yield %arg19 : i32 loc(#loc)
                                          } loc(#loc)
                                          %28:5 = scf.while (%arg23 = %arg17, %arg24 = %arg18, %arg25 = %27, %arg26 = %arg21, %arg27 = %arg22) : (i32, i32, i32, f32, f32) -> (i32, i32, i32, f32, f32) {
                                            %29 = arith.cmpi slt, %arg25, %c5_i32 : i32 loc(#loc31)
                                            scf.condition(%29) %arg23, %arg24, %arg25, %arg26, %arg27 : i32, i32, i32, f32, f32 loc(#loc32)
                                          } do {
                                          ^bb0(%arg23: i32 loc("dilate.cpp":27:6), %arg24: i32 loc("dilate.cpp":26:6), %arg25: i32 loc("dilate.cpp":25:13), %arg26: f32 loc("dilate.cpp":23:4), %arg27: f32 loc("dilate.cpp":23:4)):
                                            %29 = scf.if %true -> (i32) {
                                              %33 = scf.execute_region -> i32 {
                                                %34 = scf.if %true -> (i32) {
                                                  %35 = scf.execute_region -> i32 {
                                                    %36 = arith.subi %arg9, %3 : i32 loc(#loc33)
                                                    %37 = arith.addi %36, %arg20 : i32 loc(#loc34)
                                                    scf.yield %37 : i32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %35 : i32 loc(#loc)
                                                } else {
                                                  scf.yield %arg24 : i32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %34 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %33 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg24 : i32 loc(#loc)
                                            } loc(#loc)
                                            %30 = scf.if %true -> (i32) {
                                              %33 = scf.execute_region -> i32 {
                                                %34 = scf.if %true -> (i32) {
                                                  %35 = scf.execute_region -> i32 {
                                                    %36 = arith.subi %arg16, %4 : i32 loc(#loc35)
                                                    %37 = arith.addi %36, %arg25 : i32 loc(#loc36)
                                                    scf.yield %37 : i32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %35 : i32 loc(#loc)
                                                } else {
                                                  scf.yield %arg23 : i32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %34 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %33 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg23 : i32 loc(#loc)
                                            } loc(#loc)
                                            %31:2 = scf.if %true -> (f32, f32) {
                                              %33:2 = scf.execute_region -> (f32, f32) {
                                                %34:2 = scf.if %true -> (f32, f32) {
                                                  %35:2 = scf.execute_region -> (f32, f32) {
                                                    %36 = arith.cmpi sge, %29, %c0_i32 : i32 loc(#loc37)
                                                    %37 = scf.if %36 -> (i1) {
                                                      %42 = arith.cmpi sge, %30, %c0_i32 : i32 loc(#loc39)
                                                      scf.yield %42 : i1 loc(#loc38)
                                                    } else {
                                                      scf.yield %false : i1 loc(#loc38)
                                                    } loc(#loc38)
                                                    %38 = scf.if %37 -> (i1) {
                                                      %42 = arith.cmpi slt, %29, %c544_i32 : i32 loc(#loc41)
                                                      scf.yield %42 : i1 loc(#loc40)
                                                    } else {
                                                      scf.yield %false : i1 loc(#loc40)
                                                    } loc(#loc40)
                                                    %39 = scf.if %38 -> (i1) {
                                                      %42 = arith.cmpi slt, %30, %c512_i32 : i32 loc(#loc43)
                                                      scf.yield %42 : i1 loc(#loc42)
                                                    } else {
                                                      scf.yield %false : i1 loc(#loc42)
                                                    } loc(#loc42)
                                                    %40 = scf.if %39 -> (i1) {
                                                      %42 = arith.muli %arg20, %c5_i32 : i32 loc(#loc45)
                                                      %43 = arith.addi %42, %arg25 : i32 loc(#loc46)
                                                      %44 = arith.index_cast %43 : i32 to index loc(#loc47)
                                                      %45 = "polygeist.subindex"(%alloca, %44) : (memref<25xi8>, index) -> memref<?xi8> loc(#loc48)
                                                      %46 = affine.load %45[0] : memref<?xi8> loc(#loc48)
                                                      %47 = arith.extui %46 : i8 to i32 loc(#loc48)
                                                      %48 = arith.cmpi ne, %47, %c0_i32 : i32 loc(#loc49)
                                                      scf.yield %48 : i1 loc(#loc44)
                                                    } else {
                                                      scf.yield %false : i1 loc(#loc44)
                                                    } loc(#loc44)
                                                    %41:2 = scf.if %40 -> (f32, f32) {
                                                      %42 = scf.if %true -> (f32) {
                                                        %44 = scf.execute_region -> f32 {
                                                          %45 = arith.muli %29, %c512_i32 : i32 loc(#loc51)
                                                          %46 = arith.addi %2, %45 : i32 loc(#loc52)
                                                          %47 = arith.addi %46, %30 : i32 loc(#loc53)
                                                          %48 = arith.index_cast %47 : i32 to index loc(#loc54)
                                                          %49 = "polygeist.subindex"(%arg1, %48) : (memref<280576xf32>, index) -> memref<?xf32> loc(#loc55)
                                                          %50 = affine.load %49[0] : memref<?xf32> loc(#loc55)
                                                          scf.yield %50 : f32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %44 : f32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg26 : f32 loc(#loc)
                                                      } loc(#loc)
                                                      %43 = scf.if %true -> (f32) {
                                                        %44 = scf.execute_region -> f32 {
                                                          %45 = scf.if %true -> (f32) {
                                                            %46 = scf.execute_region -> f32 {
                                                              %47 = arith.cmpf ogt, %42, %arg27 : f32 loc(#loc56)
                                                              %48 = scf.if %47 -> (f32) {
                                                                scf.yield %42 : f32 loc(#loc57)
                                                              } else {
                                                                scf.yield %arg27 : f32 loc(#loc57)
                                                              } loc(#loc57)
                                                              scf.yield %48 : f32 loc(#loc)
                                                            } loc(#loc)
                                                            scf.yield %46 : f32 loc(#loc)
                                                          } else {
                                                            scf.yield %arg27 : f32 loc(#loc)
                                                          } loc(#loc)
                                                          scf.yield %45 : f32 loc(#loc)
                                                        } loc(#loc)
                                                        scf.yield %44 : f32 loc(#loc)
                                                      } else {
                                                        scf.yield %arg27 : f32 loc(#loc)
                                                      } loc(#loc)
                                                      scf.yield %42, %43 : f32, f32 loc(#loc50)
                                                    } else {
                                                      scf.yield %arg26, %arg27 : f32, f32 loc(#loc50)
                                                    } loc(#loc50)
                                                    scf.yield %41#0, %41#1 : f32, f32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %35#0, %35#1 : f32, f32 loc(#loc)
                                                } else {
                                                  scf.yield %arg26, %arg27 : f32, f32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %34#0, %34#1 : f32, f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %33#0, %33#1 : f32, f32 loc(#loc)
                                            } else {
                                              scf.yield %arg26, %arg27 : f32, f32 loc(#loc)
                                            } loc(#loc)
                                            %32 = scf.if %true -> (i32) {
                                              %33 = scf.execute_region -> i32 {
                                                %34 = arith.addi %arg25, %c1_i32 : i32 loc(#loc4)
                                                scf.yield %34 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %33 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg25 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %30, %29, %32, %31#0, %31#1 : i32, i32, i32, f32, f32 loc(#loc32)
                                          } loc(#loc31)
                                          scf.yield %28#0, %28#1, %28#2, %28#3, %28#4 : i32, i32, i32, f32, f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %26#0, %26#1, %26#2, %26#3, %26#4 : i32, i32, i32, f32, f32 loc(#loc)
                                      } else {
                                        scf.yield %arg17, %arg18, %arg19, %arg21, %arg22 : i32, i32, i32, f32, f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %25#0, %25#1, %25#2, %25#3, %25#4 : i32, i32, i32, f32, f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %24#0, %24#1, %24#2, %24#3, %24#4 : i32, i32, i32, f32, f32 loc(#loc)
                                  } else {
                                    scf.yield %arg17, %arg18, %arg19, %arg21, %arg22 : i32, i32, i32, f32, f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (i32) {
                                    %24 = scf.execute_region -> i32 {
                                      %25 = arith.addi %arg20, %c1_i32 : i32 loc(#loc58)
                                      scf.yield %25 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %24 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg20 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22#0, %22#1, %22#2, %23, %22#3, %22#4 : i32, i32, i32, i32, f32, f32 loc(#loc29)
                                } loc(#loc5)
                                scf.yield %21#0, %21#1, %21#2, %21#3, %21#4, %21#5 : i32, i32, i32, i32, f32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %19#0, %19#1, %19#2, %19#3, %19#4, %19#5 : i32, i32, i32, i32, f32, f32 loc(#loc)
                            } else {
                              scf.yield %arg10, %arg11, %arg12, %arg13, %arg14, %14 : i32, i32, i32, i32, f32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %18#0, %18#1, %18#2, %18#3, %18#4, %18#5 : i32, i32, i32, i32, f32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17#0, %17#1, %17#2, %17#3, %17#4, %17#5 : i32, i32, i32, i32, f32, f32 loc(#loc)
                        } else {
                          scf.yield %arg10, %arg11, %arg12, %arg13, %arg14, %14 : i32, i32, i32, i32, f32, f32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %17 = arith.muli %arg9, %c512_i32 : i32 loc(#loc59)
                            %18 = arith.addi %17, %arg16 : i32 loc(#loc60)
                            %19 = arith.index_cast %18 : i32 to index loc(#loc61)
                            %20 = "polygeist.subindex"(%arg0, %19) : (memref<278528xf32>, index) -> memref<?xf32> loc(#loc62)
                            affine.store %15#5, %20[0] : memref<?xf32> loc(#loc63)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %16 = scf.if %true -> (i32) {
                          %17 = scf.execute_region -> i32 {
                            %18 = arith.addi %arg16, %c1_i32 : i32 loc(#loc64)
                            scf.yield %18 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : i32 loc(#loc)
                        } else {
                          scf.yield %arg16 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %15#0, %15#1, %15#2, %15#3, %15#4, %15#5, %16 : i32, i32, i32, i32, f32, f32, i32 loc(#loc22)
                      } loc(#loc7)
                      scf.yield %13#0, %13#1, %13#2, %13#3, %13#4, %13#5, %13#6 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
                  } else {
                    scf.yield %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %10#0, %10#1, %10#2, %10#3, %10#4, %10#5, %10#6 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %9#0, %9#1, %9#2, %9#3, %9#4, %9#5, %9#6 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
              } else {
                scf.yield %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : i32, i32, i32, i32, f32, f32, i32 loc(#loc)
              } loc(#loc)
              %8 = scf.if %true -> (i32) {
                %9 = scf.execute_region -> i32 {
                  %10 = arith.addi %arg9, %c1_i32 : i32 loc(#loc65)
                  scf.yield %10 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %9 : i32 loc(#loc)
              } else {
                scf.yield %arg9 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %8 : i32, i32, i32, i32, f32, f32, i32, i32 loc(#loc19)
            } loc(#loc8)
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
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("dilate.cpp":16:24)
#loc3 = loc("dilate.cpp":16:30)
#loc4 = loc("dilate.cpp":25:41)
#loc5 = loc("./dilate.h":20:20)
#loc6 = loc("dilate.cpp":23:16)
#loc7 = loc("./dilate.h":13:19)
#loc9 = loc("dilate.cpp":21:18)
#loc10 = loc("dilate.cpp":18:28)
#loc11 = loc("dilate.cpp":14:31)
#loc12 = loc("dilate.cpp":5:1)
#loc15 = loc("dilate.cpp":16:5)
#loc16 = loc("dilate.cpp":16:1)
#loc17 = loc("dilate.cpp":21:1)
#loc18 = loc("dilate.cpp":21:23)
#loc19 = loc("dilate.cpp":21:5)
#loc20 = loc("dilate.cpp":22:1)
#loc21 = loc("dilate.cpp":22:24)
#loc22 = loc("dilate.cpp":22:6)
#loc27 = loc("dilate.cpp":24:1)
#loc28 = loc("dilate.cpp":24:25)
#loc29 = loc("dilate.cpp":24:7)
#loc30 = loc("dilate.cpp":25:1)
#loc31 = loc("dilate.cpp":25:26)
#loc32 = loc("dilate.cpp":25:8)
#loc33 = loc("dilate.cpp":26:16)
#loc34 = loc("dilate.cpp":26:27)
#loc35 = loc("dilate.cpp":27:16)
#loc36 = loc("dilate.cpp":27:27)
#loc37 = loc("dilate.cpp":28:12)
#loc38 = loc("dilate.cpp":28:17)
#loc39 = loc("dilate.cpp":28:22)
#loc40 = loc("dilate.cpp":28:27)
#loc41 = loc("dilate.cpp":28:32)
#loc42 = loc("dilate.cpp":28:44)
#loc43 = loc("dilate.cpp":28:49)
#loc44 = loc("dilate.cpp":28:61)
#loc45 = loc("dilate.cpp":28:72)
#loc46 = loc("dilate.cpp":28:85)
#loc47 = loc("dilate.cpp":28:88)
#loc48 = loc("dilate.cpp":28:64)
#loc49 = loc("dilate.cpp":28:90)
#loc50 = loc("dilate.cpp":28:6)
#loc51 = loc("dilate.cpp":29:35)
#loc52 = loc("dilate.cpp":29:31)
#loc53 = loc("dilate.cpp":29:47)
#loc54 = loc("dilate.cpp":29:50)
#loc55 = loc("dilate.cpp":29:14)
#loc56 = loc("dilate.cpp":30:16)
#loc57 = loc("dilate.cpp":30:7)
#loc58 = loc("dilate.cpp":24:40)
#loc59 = loc("dilate.cpp":34:13)
#loc60 = loc("dilate.cpp":34:25)
#loc61 = loc("dilate.cpp":34:28)
#loc62 = loc("dilate.cpp":34:4)
#loc63 = loc("dilate.cpp":34:30)
#loc64 = loc("dilate.cpp":22:38)
#loc65 = loc("dilate.cpp":21:37)
#loc66 = loc("dilate.cpp":39:1)
