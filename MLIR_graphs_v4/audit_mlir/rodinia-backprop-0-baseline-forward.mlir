#loc1 = loc("backprop_kernel.cpp":7:6)
#loc5 = loc("backprop_kernel.cpp":4:19)
#loc7 = loc("backprop_kernel.cpp":34:36)
#loc10 = loc("backprop_kernel.cpp":30:24)
#loc15 = loc("backprop_kernel.cpp":39:21)
#loc16 = loc("backprop_kernel.cpp":35:13)
#loc35 = loc("backprop_kernel.cpp":18:5)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<65537xf32> loc("backprop_kernel.cpp":7:6), %arg1: memref<17xf32> loc("backprop_kernel.cpp":7:6), %arg2: memref<1114129xf32> loc("backprop_kernel.cpp":7:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64 loc(#loc2)
    %c65537_i32 = arith.constant 65537 : i32 loc(#loc3)
    %c17_i64 = arith.constant 17 : i64 loc(#loc4)
    %c512_i32 = arith.constant 512 : i32 loc(#loc5)
    %c65536_i32 = arith.constant 65536 : i32 loc(#loc6)
    %c66048_i32 = arith.constant 66048 : i32 loc(#loc7)
    %c0_i32 = arith.constant 0 : i32 loc(#loc8)
    %cst_0 = arith.constant 0.000000e+00 : f32 loc(#loc9)
    %c17_i32 = arith.constant 17 : i32 loc(#loc10)
    %c1_i32 = arith.constant 1 : i32 loc(#loc11)
    %cst_1 = arith.constant 1.000000e+00 : f32 loc(#loc12)
    %c65537_i64 = arith.constant 65537 : i64 loc(#loc13)
    %true = arith.constant true loc(#loc14)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc15)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc16)
    %alloca = memref.alloca() : memref<17xf32> loc(#loc17)
    %alloca_2 = memref.alloca() : memref<8704xf32> loc(#loc18)
    %alloca_3 = memref.alloca() : memref<65537xf32> loc(#loc19)
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
        cf.br ^bb1 loc(#loc20)
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
        cf.br ^bb1 loc(#loc21)
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
        cf.br ^bb1 loc(#loc22)
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
        %2 = "polygeist.memref2pointer"(%alloca_3) : (memref<65537xf32>) -> !llvm.ptr loc(#loc23)
        %3 = "polygeist.memref2pointer"(%arg0) : (memref<65537xf32>) -> !llvm.ptr loc(#loc24)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc25)
        %5 = arith.index_cast %4 : index to i64 loc(#loc25)
        %6 = arith.muli %5, %c65537_i64 : i64 loc(#loc26)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc27)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        affine.store %cst_1, %alloca_3[0] : memref<65537xf32> loc(#loc28)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc29)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2:4 = scf.while (%arg3 = %0, %arg4 = %1, %arg5 = %c1_i32, %arg6 = %0) : (f32, i32, i32, f32) -> (f32, i32, i32, f32) {
              %3 = arith.cmpi slt, %arg5, %c17_i32 : i32 loc(#loc30)
              scf.condition(%3) %arg3, %arg4, %arg5, %arg6 : f32, i32, i32, f32 loc(#loc31)
            } do {
            ^bb0(%arg3: f32 loc("backprop_kernel.cpp":30:24), %arg4: i32 loc("backprop_kernel.cpp":30:24), %arg5: i32 loc("backprop_kernel.cpp":30:24), %arg6: f32 loc("backprop_kernel.cpp":30:24)):
              %3 = scf.if %true -> (f32) {
                %6 = scf.execute_region -> f32 {
                  scf.yield %cst_0 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %6 : f32 loc(#loc)
              } else {
                scf.yield %arg6 : f32 loc(#loc)
              } loc(#loc)
              %4:3 = scf.if %true -> (f32, i32, f32) {
                %6:3 = scf.execute_region -> (f32, i32, f32) {
                  cf.br ^bb1 loc(#loc32)
                ^bb1:  // pred: ^bb0
                  %7:3 = scf.if %true -> (f32, i32, f32) {
                    %8:3 = scf.execute_region -> (f32, i32, f32) {
                      %9:4 = scf.while (%arg7 = %arg3, %arg8 = %arg4, %arg9 = %c0_i32, %arg10 = %3) : (f32, i32, i32, f32) -> (f32, i32, f32, i32) {
                        %10 = arith.cmpi slt, %arg9, %c66048_i32 : i32 loc(#loc33)
                        scf.condition(%10) %arg7, %arg8, %arg10, %arg9 : f32, i32, f32, i32 loc(#loc34)
                      } do {
                      ^bb0(%arg7: f32 loc("backprop_kernel.cpp":39:21), %arg8: i32 loc("backprop_kernel.cpp":35:13), %arg9: f32 loc("backprop_kernel.cpp":18:5), %arg10: i32 loc("backprop_kernel.cpp":34:36)):
                        %10 = scf.if %true -> (i32) {
                          %13 = scf.execute_region -> i32 {
                            %14 = scf.if %true -> (i32) {
                              %15 = scf.execute_region -> i32 {
                                %16 = arith.cmpi eq, %arg10, %c65536_i32 : i32 loc(#loc36)
                                %17 = scf.if %16 -> (i32) {
                                  scf.yield %c1_i32 : i32 loc(#loc37)
                                } else {
                                  scf.yield %c512_i32 : i32 loc(#loc37)
                                } loc(#loc37)
                                scf.yield %17 : i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %15 : i32 loc(#loc)
                            } else {
                              scf.yield %arg8 : i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %14 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : i32 loc(#loc)
                        } else {
                          scf.yield %arg8 : i32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %13 = "polygeist.memref2pointer"(%alloca_2) : (memref<8704xf32>) -> !llvm.ptr loc(#loc38)
                            %14 = arith.muli %arg10, %c17_i32 : i32 loc(#loc39)
                            %15 = arith.index_cast %14 : i32 to index loc(#loc40)
                            %16 = "polygeist.subindex"(%arg2, %15) : (memref<1114129xf32>, index) -> memref<?xf32> loc(#loc40)
                            %17 = "polygeist.memref2pointer"(%16) : (memref<?xf32>) -> !llvm.ptr loc(#loc41)
                            %18 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc42)
                            %19 = arith.index_cast %18 : index to i64 loc(#loc42)
                            %20 = arith.extsi %10 : i32 to i64 loc(#loc43)
                            %21 = arith.muli %19, %20 : i64 loc(#loc44)
                            %22 = arith.muli %21, %c17_i64 : i64 loc(#loc45)
                            "llvm.intr.memcpy"(%13, %17, %22) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc46)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %11:2 = scf.if %true -> (f32, f32) {
                          %13:2 = scf.execute_region -> (f32, f32) {
                            cf.br ^bb1 loc(#loc47)
                          ^bb1:  // pred: ^bb0
                            %14:2 = scf.if %true -> (f32, f32) {
                              %15:2 = scf.execute_region -> (f32, f32) {
                                %16:3 = scf.while (%arg11 = %arg7, %arg12 = %c0_i32, %arg13 = %arg9) : (f32, i32, f32) -> (f32, f32, i32) {
                                  %17 = arith.cmpi slt, %arg12, %c512_i32 : i32 loc(#loc48)
                                  scf.condition(%17) %arg11, %arg13, %arg12 : f32, f32, i32 loc(#loc49)
                                } do {
                                ^bb0(%arg11: f32 loc("backprop_kernel.cpp":39:21), %arg12: f32 loc("backprop_kernel.cpp":18:5), %arg13: i32 loc("backprop_kernel.cpp":4:19)):
                                  %17:2 = scf.if %true -> (f32, f32) {
                                    %19:2 = scf.execute_region -> (f32, f32) {
                                      %20:2 = scf.if %true -> (f32, f32) {
                                        %21:2 = scf.execute_region -> (f32, f32) {
                                          %22 = arith.addi %arg13, %arg10 : i32 loc(#loc50)
                                          %23 = arith.cmpi slt, %22, %c65537_i32 : i32 loc(#loc51)
                                          %24:2 = scf.if %23 -> (f32, f32) {
                                            %25 = scf.if %true -> (f32) {
                                              %27 = scf.execute_region -> f32 {
                                                %28 = scf.if %true -> (f32) {
                                                  %29 = scf.execute_region -> f32 {
                                                    %30 = arith.muli %arg13, %c17_i32 : i32 loc(#loc53)
                                                    %31 = arith.addi %30, %arg5 : i32 loc(#loc54)
                                                    %32 = arith.index_cast %31 : i32 to index loc(#loc55)
                                                    %33 = "polygeist.subindex"(%alloca_2, %32) : (memref<8704xf32>, index) -> memref<?xf32> loc(#loc56)
                                                    %34 = affine.load %33[0] : memref<?xf32> loc(#loc56)
                                                    %35 = arith.index_cast %22 : i32 to index loc(#loc57)
                                                    %36 = "polygeist.subindex"(%alloca_3, %35) : (memref<65537xf32>, index) -> memref<?xf32> loc(#loc58)
                                                    %37 = affine.load %36[0] : memref<?xf32> loc(#loc58)
                                                    %38 = arith.mulf %34, %37 : f32 loc(#loc59)
                                                    scf.yield %38 : f32 loc(#loc)
                                                  } loc(#loc)
                                                  scf.yield %29 : f32 loc(#loc)
                                                } else {
                                                  scf.yield %arg11 : f32 loc(#loc)
                                                } loc(#loc)
                                                scf.yield %28 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %27 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg11 : f32 loc(#loc)
                                            } loc(#loc)
                                            %26 = scf.if %true -> (f32) {
                                              %27 = scf.execute_region -> f32 {
                                                %28 = arith.addf %arg12, %25 : f32 loc(#loc60)
                                                scf.yield %28 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %27 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg12 : f32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %25, %26 : f32, f32 loc(#loc52)
                                          } else {
                                            scf.yield %arg11, %arg12 : f32, f32 loc(#loc52)
                                          } loc(#loc52)
                                          scf.yield %24#0, %24#1 : f32, f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %21#0, %21#1 : f32, f32 loc(#loc)
                                      } else {
                                        scf.yield %arg11, %arg12 : f32, f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %20#0, %20#1 : f32, f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19#0, %19#1 : f32, f32 loc(#loc)
                                  } else {
                                    scf.yield %arg11, %arg12 : f32, f32 loc(#loc)
                                  } loc(#loc)
                                  %18 = scf.if %true -> (i32) {
                                    %19 = scf.execute_region -> i32 {
                                      %20 = arith.addi %arg13, %c1_i32 : i32 loc(#loc61)
                                      scf.yield %20 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %19 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg13 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %17#0, %18, %17#1 : f32, i32, f32 loc(#loc49)
                                } loc(#loc5)
                                scf.yield %16#0, %16#1 : f32, f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %15#0, %15#1 : f32, f32 loc(#loc)
                            } else {
                              scf.yield %arg7, %arg9 : f32, f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %14#0, %14#1 : f32, f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13#0, %13#1 : f32, f32 loc(#loc)
                        } else {
                          scf.yield %arg7, %arg9 : f32, f32 loc(#loc)
                        } loc(#loc)
                        %12 = scf.if %true -> (i32) {
                          %13 = scf.execute_region -> i32 {
                            %14 = arith.addi %arg10, %c512_i32 : i32 loc(#loc62)
                            scf.yield %14 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %13 : i32 loc(#loc)
                        } else {
                          scf.yield %arg10 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %11#0, %10, %12, %11#1 : f32, i32, i32, f32 loc(#loc34)
                      } loc(#loc7)
                      scf.yield %9#0, %9#1, %9#2 : f32, i32, f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %8#0, %8#1, %8#2 : f32, i32, f32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %3 : f32, i32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %7#0, %7#1, %7#2 : f32, i32, f32 loc(#loc)
                } loc(#loc)
                scf.yield %6#0, %6#1, %6#2 : f32, i32, f32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %3 : f32, i32, f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %6 = arith.index_cast %arg5 : i32 to index loc(#loc63)
                  %7 = "polygeist.subindex"(%alloca, %6) : (memref<17xf32>, index) -> memref<?xf32> loc(#loc64)
                  %8 = arith.negf %4#2 : f32 loc(#loc65)
                  %9 = math.exp %8 : f32 loc(#loc66)
                  %10 = arith.extf %9 : f32 to f64 loc(#loc66)
                  %11 = arith.addf %10, %cst : f64 loc(#loc67)
                  %12 = arith.divf %cst, %11 : f64 loc(#loc68)
                  %13 = arith.truncf %12 : f64 to f32 loc(#loc69)
                  affine.store %13, %7[0] : memref<?xf32> loc(#loc70)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %5 = scf.if %true -> (i32) {
                %6 = scf.execute_region -> i32 {
                  %7 = arith.addi %arg5, %c1_i32 : i32 loc(#loc71)
                  scf.yield %7 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %6 : i32 loc(#loc)
              } else {
                scf.yield %arg5 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4#0, %4#1, %5, %4#2 : f32, i32, i32, f32 loc(#loc31)
            } loc(#loc10)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%arg1) : (memref<17xf32>) -> !llvm.ptr loc(#loc72)
        %3 = "polygeist.memref2pointer"(%alloca) : (memref<17xf32>) -> !llvm.ptr loc(#loc73)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc74)
        %5 = arith.index_cast %4 : index to i64 loc(#loc74)
        %6 = arith.muli %5, %c17_i64 : i64 loc(#loc75)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc76)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc77)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("backprop_kernel.cpp":45:22)
#loc3 = loc("backprop_kernel.cpp":38:30)
#loc4 = loc("backprop_kernel.cpp":36:65)
#loc6 = loc("backprop_kernel.cpp":35:31)
#loc8 = loc("backprop_kernel.cpp":34:22)
#loc9 = loc("backprop_kernel.cpp":32:15)
#loc11 = loc("backprop_kernel.cpp":30:17)
#loc12 = loc("backprop_kernel.cpp":27:17)
#loc13 = loc("backprop_kernel.cpp":24:40)
#loc14 = loc("backprop_kernel.cpp":7:1)
#loc17 = loc("backprop_kernel.cpp":22:8)
#loc18 = loc("backprop_kernel.cpp":21:8)
#loc19 = loc("backprop_kernel.cpp":20:8)
#loc20 = loc("backprop_kernel.cpp":20:1)
#loc21 = loc("backprop_kernel.cpp":21:1)
#loc22 = loc("backprop_kernel.cpp":22:1)
#loc23 = loc("backprop_kernel.cpp":24:12)
#loc24 = loc("backprop_kernel.cpp":24:20)
#loc25 = loc("backprop_kernel.cpp":24:24)
#loc26 = loc("backprop_kernel.cpp":24:38)
#loc27 = loc("backprop_kernel.cpp":24:5)
#loc28 = loc("backprop_kernel.cpp":27:15)
#loc29 = loc("backprop_kernel.cpp":30:1)
#loc30 = loc("backprop_kernel.cpp":30:22)
#loc31 = loc("backprop_kernel.cpp":30:8)
#loc32 = loc("backprop_kernel.cpp":34:1)
#loc33 = loc("backprop_kernel.cpp":34:28)
#loc34 = loc("backprop_kernel.cpp":34:12)
#loc36 = loc("backprop_kernel.cpp":35:28)
#loc37 = loc("backprop_kernel.cpp":35:24)
#loc38 = loc("backprop_kernel.cpp":36:20)
#loc39 = loc("backprop_kernel.cpp":36:37)
#loc40 = loc("backprop_kernel.cpp":36:34)
#loc41 = loc("backprop_kernel.cpp":36:30)
#loc42 = loc("backprop_kernel.cpp":36:42)
#loc43 = loc("backprop_kernel.cpp":36:58)
#loc44 = loc("backprop_kernel.cpp":36:56)
#loc45 = loc("backprop_kernel.cpp":36:63)
#loc46 = loc("backprop_kernel.cpp":36:13)
#loc47 = loc("backprop_kernel.cpp":37:1)
#loc48 = loc("backprop_kernel.cpp":37:30)
#loc49 = loc("backprop_kernel.cpp":37:16)
#loc50 = loc("backprop_kernel.cpp":38:23)
#loc51 = loc("backprop_kernel.cpp":38:28)
#loc52 = loc("backprop_kernel.cpp":38:17)
#loc53 = loc("backprop_kernel.cpp":39:48)
#loc54 = loc("backprop_kernel.cpp":39:53)
#loc55 = loc("backprop_kernel.cpp":39:56)
#loc56 = loc("backprop_kernel.cpp":39:37)
#loc57 = loc("backprop_kernel.cpp":39:73)
#loc58 = loc("backprop_kernel.cpp":39:60)
#loc59 = loc("backprop_kernel.cpp":39:58)
#loc60 = loc("backprop_kernel.cpp":40:25)
#loc61 = loc("backprop_kernel.cpp":37:44)
#loc62 = loc("backprop_kernel.cpp":34:52)
#loc63 = loc("backprop_kernel.cpp":45:17)
#loc64 = loc("backprop_kernel.cpp":45:9)
#loc65 = loc("backprop_kernel.cpp":45:39)
#loc66 = loc("backprop_kernel.cpp":45:35)
#loc67 = loc("backprop_kernel.cpp":45:33)
#loc68 = loc("backprop_kernel.cpp":45:26)
#loc69 = loc("backprop_kernel.cpp":45:21)
#loc70 = loc("backprop_kernel.cpp":45:19)
#loc71 = loc("backprop_kernel.cpp":30:29)
#loc72 = loc("backprop_kernel.cpp":47:12)
#loc73 = loc("backprop_kernel.cpp":47:16)
#loc74 = loc("backprop_kernel.cpp":47:24)
#loc75 = loc("backprop_kernel.cpp":47:38)
#loc76 = loc("backprop_kernel.cpp":47:5)
#loc77 = loc("backprop_kernel.cpp":48:1)
