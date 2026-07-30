#loc1 = loc("backprop_kernel.cpp":20:6)
#loc4 = loc("backprop_kernel.cpp":17:19)
#loc7 = loc("backprop_kernel.cpp":51:33)
#loc18 = loc("backprop_kernel.cpp":32:5)
#loc73 = loc("backprop_kernel.cpp":55:39)
#loc117 = loc("backprop_kernel.cpp":65:32)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<17xf32> loc("backprop_kernel.cpp":20:6), %arg1: memref<65537xf32> loc("backprop_kernel.cpp":20:6), %arg2: memref<1048592xf32> loc("backprop_kernel.cpp":20:6), %arg3: memref<1048592xf32> loc("backprop_kernel.cpp":20:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %cst = arith.constant 3.000000e-01 : f64 loc(#loc3)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc4)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %c4096_i64 = arith.constant 4096 : i64 loc(#loc4)
    %c16_i32 = arith.constant 16 : i32 loc(#loc6)
    %c65537_i32 = arith.constant 65537 : i32 loc(#loc7)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c65536_i64 = arith.constant 65536 : i64 loc(#loc8)
    %c16_i64 = arith.constant 16 : i64 loc(#loc9)
    %true = arith.constant true loc(#loc10)
    %c0 = arith.constant 0 : index loc(#loc11)
    %c1 = arith.constant 1 : index loc(#loc)
    %alloca = memref.alloca() : memref<4096x16xf32> loc(#loc12)
    %alloca_0 = memref.alloca() : memref<4096x16xf32> loc(#loc13)
    %alloca_1 = memref.alloca() : memref<65536xf32> loc(#loc14)
    %alloca_2 = memref.alloca() : memref<16xf32> loc(#loc15)
    %alloca_3 = memref.alloca() : memref<16xf32> loc(#loc16)
    %alloca_4 = memref.alloca() : memref<16xf32> loc(#loc17)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc18)
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
        cf.br ^bb1 loc(#loc23)
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
        cf.br ^bb1 loc(#loc24)
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
        %2 = "polygeist.memref2pointer"(%alloca_2) : (memref<16xf32>) -> !llvm.ptr loc(#loc25)
        %3 = "polygeist.subindex"(%arg0, %c1) : (memref<17xf32>, index) -> memref<?xf32> loc(#loc26)
        %4 = "polygeist.memref2pointer"(%3) : (memref<?xf32>) -> !llvm.ptr loc(#loc27)
        %5 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc28)
        %6 = arith.index_cast %5 : index to i64 loc(#loc28)
        %7 = arith.muli %6, %c16_i64 : i64 loc(#loc29)
        "llvm.intr.memcpy"(%2, %4, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc30)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%alloca_1) : (memref<65536xf32>) -> !llvm.ptr loc(#loc31)
        %3 = "polygeist.subindex"(%arg1, %c1) : (memref<65537xf32>, index) -> memref<?xf32> loc(#loc32)
        %4 = "polygeist.memref2pointer"(%3) : (memref<?xf32>) -> !llvm.ptr loc(#loc33)
        %5 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc34)
        %6 = arith.index_cast %5 : index to i64 loc(#loc34)
        %7 = arith.muli %6, %c65536_i64 : i64 loc(#loc35)
        "llvm.intr.memcpy"(%2, %4, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc36)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%alloca_4) : (memref<16xf32>) -> !llvm.ptr loc(#loc37)
        %3 = "polygeist.memref2pointer"(%arg2) : (memref<1048592xf32>) -> !llvm.ptr loc(#loc38)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc39)
        %5 = arith.index_cast %4 : index to i64 loc(#loc39)
        %6 = arith.muli %5, %c16_i64 : i64 loc(#loc40)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc41)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%alloca_3) : (memref<16xf32>) -> !llvm.ptr loc(#loc42)
        %3 = "polygeist.memref2pointer"(%arg3) : (memref<1048592xf32>) -> !llvm.ptr loc(#loc43)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc44)
        %5 = arith.index_cast %4 : index to i64 loc(#loc44)
        %6 = arith.muli %5, %c16_i64 : i64 loc(#loc45)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc46)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    %1 = scf.if %true -> (f32) {
      %2 = scf.execute_region -> f32 {
        cf.br ^bb1 loc(#loc47)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc48)
      ^bb2:  // pred: ^bb1
        %3 = scf.if %true -> (f32) {
          %4 = scf.execute_region -> f32 {
            %5:2 = scf.while (%arg4 = %c1_i32, %arg5 = %0) : (i32, f32) -> (f32, i32) {
              %6 = arith.cmpi slt, %arg4, %c65537_i32 : i32 loc(#loc49)
              scf.condition(%6) %arg5, %arg4 : f32, i32 loc(#loc50)
            } do {
            ^bb0(%arg4: f32 loc("backprop_kernel.cpp":32:5), %arg5: i32 loc("backprop_kernel.cpp":51:33)):
              scf.if %true {
                scf.execute_region {
                  %8 = "polygeist.subindex"(%alloca_0, %c0) : (memref<4096x16xf32>, index) -> memref<16xf32> loc(#loc51)
                  %9 = "polygeist.memref2pointer"(%8) : (memref<16xf32>) -> !llvm.ptr loc(#loc51)
                  %10 = arith.muli %arg5, %c16_i32 : i32 loc(#loc52)
                  %11 = arith.index_cast %10 : i32 to index loc(#loc53)
                  %12 = "polygeist.subindex"(%arg2, %11) : (memref<1048592xf32>, index) -> memref<?xf32> loc(#loc53)
                  %13 = "polygeist.memref2pointer"(%12) : (memref<?xf32>) -> !llvm.ptr loc(#loc54)
                  %14 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc55)
                  %15 = arith.index_cast %14 : index to i64 loc(#loc55)
                  %16 = arith.muli %15, %c4096_i64 : i64 loc(#loc56)
                  %17 = arith.muli %16, %c16_i64 : i64 loc(#loc57)
                  "llvm.intr.memcpy"(%9, %13, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc58)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = "polygeist.subindex"(%alloca, %c0) : (memref<4096x16xf32>, index) -> memref<16xf32> loc(#loc59)
                  %9 = "polygeist.memref2pointer"(%8) : (memref<16xf32>) -> !llvm.ptr loc(#loc59)
                  %10 = arith.muli %arg5, %c16_i32 : i32 loc(#loc60)
                  %11 = arith.index_cast %10 : i32 to index loc(#loc61)
                  %12 = "polygeist.subindex"(%arg3, %11) : (memref<1048592xf32>, index) -> memref<?xf32> loc(#loc61)
                  %13 = "polygeist.memref2pointer"(%12) : (memref<?xf32>) -> !llvm.ptr loc(#loc62)
                  %14 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc63)
                  %15 = arith.index_cast %14 : index to i64 loc(#loc63)
                  %16 = arith.muli %15, %c4096_i64 : i64 loc(#loc64)
                  %17 = arith.muli %16, %c16_i64 : i64 loc(#loc65)
                  "llvm.intr.memcpy"(%9, %13, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc66)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %6 = scf.if %true -> (f32) {
                %8 = scf.execute_region -> f32 {
                  cf.br ^bb1 loc(#loc67)
                ^bb1:  // pred: ^bb0
                  cf.br ^bb2 loc(#loc68)
                ^bb2:  // pred: ^bb1
                  %9 = scf.if %true -> (f32) {
                    %10 = scf.execute_region -> f32 {
                      %11:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg4) : (i32, f32) -> (f32, i32) {
                        %12 = arith.cmpi slt, %arg6, %c4096_i32 : i32 loc(#loc69)
                        scf.condition(%12) %arg7, %arg6 : f32, i32 loc(#loc70)
                      } do {
                      ^bb0(%arg6: f32 loc("backprop_kernel.cpp":32:5), %arg7: i32 loc("backprop_kernel.cpp":17:19)):
                        %12 = scf.if %true -> (f32) {
                          %14 = scf.execute_region -> f32 {
                            cf.br ^bb1 loc(#loc71)
                          ^bb1:  // pred: ^bb0
                            cf.br ^bb2 loc(#loc72)
                          ^bb2:  // pred: ^bb1
                            %15 = scf.if %true -> (f32) {
                              %16 = scf.execute_region -> f32 {
                                %17:2 = scf.while (%arg8 = %c0_i32, %arg9 = %arg6) : (i32, f32) -> (f32, i32) {
                                  %18 = arith.cmpi slt, %arg8, %c16_i32 : i32 loc(#loc74)
                                  scf.condition(%18) %arg9, %arg8 : f32, i32 loc(#loc75)
                                } do {
                                ^bb0(%arg8: f32 loc("backprop_kernel.cpp":32:5), %arg9: i32 loc("backprop_kernel.cpp":55:39)):
                                  %18 = scf.if %true -> (f32) {
                                    %20 = scf.execute_region -> f32 {
                                      %21 = arith.index_cast %arg9 : i32 to index loc(#loc76)
                                      %22 = "polygeist.subindex"(%alloca_2, %21) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc77)
                                      %23 = affine.load %22[0] : memref<?xf32> loc(#loc77)
                                      %24 = arith.extf %23 : f32 to f64 loc(#loc77)
                                      %25 = arith.mulf %24, %cst : f64 loc(#loc78)
                                      %26 = arith.addi %arg7, %arg5 : i32 loc(#loc79)
                                      %27 = arith.addi %26, %c-1_i32 : i32 loc(#loc80)
                                      %28 = arith.index_cast %27 : i32 to index loc(#loc81)
                                      %29 = "polygeist.subindex"(%alloca_1, %28) : (memref<65536xf32>, index) -> memref<?xf32> loc(#loc82)
                                      %30 = affine.load %29[0] : memref<?xf32> loc(#loc82)
                                      %31 = arith.extf %30 : f32 to f64 loc(#loc82)
                                      %32 = arith.mulf %25, %31 : f64 loc(#loc83)
                                      %33 = arith.index_cast %arg7 : i32 to index loc(#loc84)
                                      %34 = "polygeist.subindex"(%alloca, %33) : (memref<4096x16xf32>, index) -> memref<?x16xf32> loc(#loc85)
                                      %35 = "polygeist.subindex"(%34, %c0) : (memref<?x16xf32>, index) -> memref<16xf32> loc(#loc85)
                                      %36 = "polygeist.subindex"(%35, %21) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc85)
                                      %37 = affine.load %36[0] : memref<?xf32> loc(#loc85)
                                      %38 = arith.extf %37 : f32 to f64 loc(#loc85)
                                      %39 = arith.mulf %38, %cst : f64 loc(#loc86)
                                      %40 = arith.addf %32, %39 : f64 loc(#loc87)
                                      %41 = arith.truncf %40 : f64 to f32 loc(#loc3)
                                      scf.yield %41 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %20 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg8 : f32 loc(#loc)
                                  } loc(#loc)
                                  scf.if %true {
                                    scf.execute_region {
                                      %20 = arith.index_cast %arg7 : i32 to index loc(#loc88)
                                      %21 = "polygeist.subindex"(%alloca_0, %20) : (memref<4096x16xf32>, index) -> memref<?x16xf32> loc(#loc89)
                                      %22 = "polygeist.subindex"(%21, %c0) : (memref<?x16xf32>, index) -> memref<16xf32> loc(#loc89)
                                      %23 = arith.index_cast %arg9 : i32 to index loc(#loc90)
                                      %24 = "polygeist.subindex"(%22, %23) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc89)
                                      %25 = affine.load %24[0] : memref<?xf32> loc(#loc91)
                                      %26 = arith.addf %25, %18 : f32 loc(#loc91)
                                      affine.store %26, %24[0] : memref<?xf32> loc(#loc91)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  scf.if %true {
                                    scf.execute_region {
                                      %20 = arith.index_cast %arg7 : i32 to index loc(#loc92)
                                      %21 = "polygeist.subindex"(%alloca, %20) : (memref<4096x16xf32>, index) -> memref<?x16xf32> loc(#loc93)
                                      %22 = "polygeist.subindex"(%21, %c0) : (memref<?x16xf32>, index) -> memref<16xf32> loc(#loc93)
                                      %23 = arith.index_cast %arg9 : i32 to index loc(#loc94)
                                      %24 = "polygeist.subindex"(%22, %23) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc93)
                                      affine.store %18, %24[0] : memref<?xf32> loc(#loc95)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  %19 = scf.if %true -> (i32) {
                                    %20 = scf.execute_region -> i32 {
                                      %21 = arith.addi %arg9, %c1_i32 : i32 loc(#loc96)
                                      scf.yield %21 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %20 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg9 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %19, %18 : i32, f32 loc(#loc75)
                                } loc(#loc73)
                                scf.yield %17#0 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %16 : f32 loc(#loc)
                            } else {
                              scf.yield %arg6 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %15 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %14 : f32 loc(#loc)
                        } else {
                          scf.yield %arg6 : f32 loc(#loc)
                        } loc(#loc)
                        %13 = scf.if %true -> (i32) {
                          %14 = scf.execute_region -> i32 {
                            %15 = arith.addi %arg7, %c1_i32 : i32 loc(#loc97)
                            scf.yield %15 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %14 : i32 loc(#loc)
                        } else {
                          scf.yield %arg7 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %13, %12 : i32, f32 loc(#loc70)
                      } loc(#loc4)
                      scf.yield %11#0 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %10 : f32 loc(#loc)
                  } else {
                    scf.yield %arg4 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %9 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : f32 loc(#loc)
              } else {
                scf.yield %arg4 : f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = arith.muli %arg5, %c16_i32 : i32 loc(#loc98)
                  %9 = arith.index_cast %8 : i32 to index loc(#loc99)
                  %10 = "polygeist.subindex"(%arg2, %9) : (memref<1048592xf32>, index) -> memref<?xf32> loc(#loc99)
                  %11 = "polygeist.memref2pointer"(%10) : (memref<?xf32>) -> !llvm.ptr loc(#loc100)
                  %12 = "polygeist.subindex"(%alloca_0, %c0) : (memref<4096x16xf32>, index) -> memref<16xf32> loc(#loc101)
                  %13 = "polygeist.memref2pointer"(%12) : (memref<16xf32>) -> !llvm.ptr loc(#loc101)
                  %14 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc102)
                  %15 = arith.index_cast %14 : index to i64 loc(#loc102)
                  %16 = arith.muli %15, %c4096_i64 : i64 loc(#loc103)
                  %17 = arith.muli %16, %c16_i64 : i64 loc(#loc104)
                  "llvm.intr.memcpy"(%11, %13, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc105)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %8 = arith.muli %arg5, %c16_i32 : i32 loc(#loc106)
                  %9 = arith.index_cast %8 : i32 to index loc(#loc107)
                  %10 = "polygeist.subindex"(%arg3, %9) : (memref<1048592xf32>, index) -> memref<?xf32> loc(#loc107)
                  %11 = "polygeist.memref2pointer"(%10) : (memref<?xf32>) -> !llvm.ptr loc(#loc108)
                  %12 = "polygeist.subindex"(%alloca, %c0) : (memref<4096x16xf32>, index) -> memref<16xf32> loc(#loc109)
                  %13 = "polygeist.memref2pointer"(%12) : (memref<16xf32>) -> !llvm.ptr loc(#loc109)
                  %14 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc110)
                  %15 = arith.index_cast %14 : index to i64 loc(#loc110)
                  %16 = arith.muli %15, %c4096_i64 : i64 loc(#loc111)
                  %17 = arith.muli %16, %c16_i64 : i64 loc(#loc112)
                  "llvm.intr.memcpy"(%11, %13, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc113)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %7 = scf.if %true -> (i32) {
                %8 = scf.execute_region -> i32 {
                  %9 = arith.addi %arg5, %c4096_i32 : i32 loc(#loc114)
                  scf.yield %9 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %8 : i32 loc(#loc)
              } else {
                scf.yield %arg5 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %7, %6 : i32, f32 loc(#loc50)
            } loc(#loc7)
            scf.yield %5#0 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %4 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %3 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %2 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc115)
      ^bb1:  // pred: ^bb0
        cf.br ^bb2 loc(#loc116)
      ^bb2:  // pred: ^bb1
        scf.if %true {
          scf.execute_region {
            %2:2 = scf.while (%arg4 = %c0_i32, %arg5 = %1) : (i32, f32) -> (i32, f32) {
              %3 = arith.cmpi slt, %arg4, %c16_i32 : i32 loc(#loc118)
              scf.condition(%3) %arg4, %arg5 : i32, f32 loc(#loc119)
            } do {
            ^bb0(%arg4: i32 loc("backprop_kernel.cpp":65:32), %arg5: f32 loc("backprop_kernel.cpp":65:32)):
              %3 = scf.if %true -> (f32) {
                %5 = scf.execute_region -> f32 {
                  %6 = arith.index_cast %arg4 : i32 to index loc(#loc120)
                  %7 = "polygeist.subindex"(%alloca_2, %6) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc121)
                  %8 = affine.load %7[0] : memref<?xf32> loc(#loc121)
                  %9 = arith.extf %8 : f32 to f64 loc(#loc121)
                  %10 = arith.mulf %9, %cst : f64 loc(#loc122)
                  %11 = "polygeist.subindex"(%alloca_3, %6) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc123)
                  %12 = affine.load %11[0] : memref<?xf32> loc(#loc123)
                  %13 = arith.extf %12 : f32 to f64 loc(#loc123)
                  %14 = arith.mulf %13, %cst : f64 loc(#loc124)
                  %15 = arith.addf %10, %14 : f64 loc(#loc125)
                  %16 = arith.truncf %15 : f64 to f32 loc(#loc3)
                  scf.yield %16 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : f32 loc(#loc)
              } else {
                scf.yield %arg5 : f32 loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg4 : i32 to index loc(#loc126)
                  %6 = "polygeist.subindex"(%alloca_4, %5) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc127)
                  %7 = affine.load %6[0] : memref<?xf32> loc(#loc128)
                  %8 = arith.addf %7, %3 : f32 loc(#loc128)
                  affine.store %8, %6[0] : memref<?xf32> loc(#loc128)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  %5 = arith.index_cast %arg4 : i32 to index loc(#loc129)
                  %6 = "polygeist.subindex"(%alloca_3, %5) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc130)
                  affine.store %3, %6[0] : memref<?xf32> loc(#loc131)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg4, %c1_i32 : i32 loc(#loc132)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %4, %3 : i32, f32 loc(#loc119)
            } loc(#loc117)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%arg2) : (memref<1048592xf32>) -> !llvm.ptr loc(#loc133)
        %3 = "polygeist.memref2pointer"(%alloca_4) : (memref<16xf32>) -> !llvm.ptr loc(#loc134)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc135)
        %5 = arith.index_cast %4 : index to i64 loc(#loc135)
        %6 = arith.muli %5, %c16_i64 : i64 loc(#loc136)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc137)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        %2 = "polygeist.memref2pointer"(%arg3) : (memref<1048592xf32>) -> !llvm.ptr loc(#loc138)
        %3 = "polygeist.memref2pointer"(%alloca_3) : (memref<16xf32>) -> !llvm.ptr loc(#loc139)
        %4 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc140)
        %5 = arith.index_cast %4 : index to i64 loc(#loc140)
        %6 = arith.muli %5, %c16_i64 : i64 loc(#loc141)
        "llvm.intr.memcpy"(%2, %3, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc142)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc143)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("backprop_kernel.cpp":51:25)
#loc3 = loc("backprop_kernel.cpp":15:13)
#loc5 = loc("backprop_kernel.cpp":54:27)
#loc6 = loc("backprop_kernel.cpp":52:31)
#loc8 = loc("backprop_kernel.cpp":47:40)
#loc9 = loc("backprop_kernel.cpp":46:46)
#loc10 = loc("backprop_kernel.cpp":20:1)
#loc11 = loc("backprop_kernel.cpp":33:5)
#loc12 = loc("backprop_kernel.cpp":43:8)
#loc13 = loc("backprop_kernel.cpp":42:8)
#loc14 = loc("backprop_kernel.cpp":40:8)
#loc15 = loc("backprop_kernel.cpp":38:8)
#loc16 = loc("backprop_kernel.cpp":36:8)
#loc17 = loc("backprop_kernel.cpp":35:8)
#loc19 = loc("backprop_kernel.cpp":35:1)
#loc20 = loc("backprop_kernel.cpp":36:1)
#loc21 = loc("backprop_kernel.cpp":38:1)
#loc22 = loc("backprop_kernel.cpp":40:1)
#loc23 = loc("backprop_kernel.cpp":42:1)
#loc24 = loc("backprop_kernel.cpp":43:1)
#loc25 = loc("backprop_kernel.cpp":46:12)
#loc26 = loc("backprop_kernel.cpp":46:28)
#loc27 = loc("backprop_kernel.cpp":46:23)
#loc28 = loc("backprop_kernel.cpp":46:32)
#loc29 = loc("backprop_kernel.cpp":46:45)
#loc30 = loc("backprop_kernel.cpp":46:5)
#loc31 = loc("backprop_kernel.cpp":47:12)
#loc32 = loc("backprop_kernel.cpp":47:22)
#loc33 = loc("backprop_kernel.cpp":47:20)
#loc34 = loc("backprop_kernel.cpp":47:26)
#loc35 = loc("backprop_kernel.cpp":47:39)
#loc36 = loc("backprop_kernel.cpp":47:5)
#loc37 = loc("backprop_kernel.cpp":48:12)
#loc38 = loc("backprop_kernel.cpp":48:20)
#loc39 = loc("backprop_kernel.cpp":48:23)
#loc40 = loc("backprop_kernel.cpp":48:36)
#loc41 = loc("backprop_kernel.cpp":48:5)
#loc42 = loc("backprop_kernel.cpp":49:12)
#loc43 = loc("backprop_kernel.cpp":49:23)
#loc44 = loc("backprop_kernel.cpp":49:29)
#loc45 = loc("backprop_kernel.cpp":49:42)
#loc46 = loc("backprop_kernel.cpp":49:5)
#loc47 = loc("backprop_kernel.cpp":51:1)
#loc48 = loc("backprop_kernel.cpp":51:8)
#loc49 = loc("backprop_kernel.cpp":51:31)
#loc50 = loc("backprop_kernel.cpp":51:15)
#loc51 = loc("backprop_kernel.cpp":52:16)
#loc52 = loc("backprop_kernel.cpp":52:30)
#loc53 = loc("backprop_kernel.cpp":52:27)
#loc54 = loc("backprop_kernel.cpp":52:26)
#loc55 = loc("backprop_kernel.cpp":52:35)
#loc56 = loc("backprop_kernel.cpp":52:49)
#loc57 = loc("backprop_kernel.cpp":52:61)
#loc58 = loc("backprop_kernel.cpp":52:9)
#loc59 = loc("backprop_kernel.cpp":53:16)
#loc60 = loc("backprop_kernel.cpp":53:36)
#loc61 = loc("backprop_kernel.cpp":53:33)
#loc62 = loc("backprop_kernel.cpp":53:29)
#loc63 = loc("backprop_kernel.cpp":53:41)
#loc64 = loc("backprop_kernel.cpp":53:55)
#loc65 = loc("backprop_kernel.cpp":53:67)
#loc66 = loc("backprop_kernel.cpp":53:9)
#loc67 = loc("backprop_kernel.cpp":54:1)
#loc68 = loc("backprop_kernel.cpp":54:12)
#loc69 = loc("backprop_kernel.cpp":54:32)
#loc70 = loc("backprop_kernel.cpp":54:18)
#loc71 = loc("backprop_kernel.cpp":55:1)
#loc72 = loc("backprop_kernel.cpp":55:16)
#loc74 = loc("backprop_kernel.cpp":55:37)
#loc75 = loc("backprop_kernel.cpp":55:23)
#loc76 = loc("backprop_kernel.cpp":56:43)
#loc77 = loc("backprop_kernel.cpp":56:32)
#loc78 = loc("backprop_kernel.cpp":56:30)
#loc79 = loc("backprop_kernel.cpp":56:55)
#loc80 = loc("backprop_kernel.cpp":56:58)
#loc81 = loc("backprop_kernel.cpp":56:60)
#loc82 = loc("backprop_kernel.cpp":56:47)
#loc83 = loc("backprop_kernel.cpp":56:45)
#loc84 = loc("backprop_kernel.cpp":56:85)
#loc85 = loc("backprop_kernel.cpp":56:75)
#loc86 = loc("backprop_kernel.cpp":56:73)
#loc87 = loc("backprop_kernel.cpp":56:62)
#loc88 = loc("backprop_kernel.cpp":57:24)
#loc89 = loc("backprop_kernel.cpp":57:17)
#loc90 = loc("backprop_kernel.cpp":57:27)
#loc91 = loc("backprop_kernel.cpp":57:29)
#loc92 = loc("backprop_kernel.cpp":58:27)
#loc93 = loc("backprop_kernel.cpp":58:17)
#loc94 = loc("backprop_kernel.cpp":58:30)
#loc95 = loc("backprop_kernel.cpp":58:32)
#loc96 = loc("backprop_kernel.cpp":55:44)
#loc97 = loc("backprop_kernel.cpp":54:46)
#loc98 = loc("backprop_kernel.cpp":61:20)
#loc99 = loc("backprop_kernel.cpp":61:17)
#loc100 = loc("backprop_kernel.cpp":61:16)
#loc101 = loc("backprop_kernel.cpp":61:25)
#loc102 = loc("backprop_kernel.cpp":61:35)
#loc103 = loc("backprop_kernel.cpp":61:49)
#loc104 = loc("backprop_kernel.cpp":61:61)
#loc105 = loc("backprop_kernel.cpp":61:9)
#loc106 = loc("backprop_kernel.cpp":62:23)
#loc107 = loc("backprop_kernel.cpp":62:20)
#loc108 = loc("backprop_kernel.cpp":62:16)
#loc109 = loc("backprop_kernel.cpp":62:28)
#loc110 = loc("backprop_kernel.cpp":62:41)
#loc111 = loc("backprop_kernel.cpp":62:55)
#loc112 = loc("backprop_kernel.cpp":62:67)
#loc113 = loc("backprop_kernel.cpp":62:9)
#loc114 = loc("backprop_kernel.cpp":51:42)
#loc115 = loc("backprop_kernel.cpp":65:1)
#loc116 = loc("backprop_kernel.cpp":65:9)
#loc118 = loc("backprop_kernel.cpp":65:30)
#loc119 = loc("backprop_kernel.cpp":65:16)
#loc120 = loc("backprop_kernel.cpp":66:35)
#loc121 = loc("backprop_kernel.cpp":66:24)
#loc122 = loc("backprop_kernel.cpp":66:22)
#loc123 = loc("backprop_kernel.cpp":66:50)
#loc124 = loc("backprop_kernel.cpp":66:48)
#loc125 = loc("backprop_kernel.cpp":66:37)
#loc126 = loc("backprop_kernel.cpp":67:17)
#loc127 = loc("backprop_kernel.cpp":67:9)
#loc128 = loc("backprop_kernel.cpp":67:19)
#loc129 = loc("backprop_kernel.cpp":68:20)
#loc130 = loc("backprop_kernel.cpp":68:9)
#loc131 = loc("backprop_kernel.cpp":68:22)
#loc132 = loc("backprop_kernel.cpp":65:37)
#loc133 = loc("backprop_kernel.cpp":71:12)
#loc134 = loc("backprop_kernel.cpp":71:15)
#loc135 = loc("backprop_kernel.cpp":71:23)
#loc136 = loc("backprop_kernel.cpp":71:36)
#loc137 = loc("backprop_kernel.cpp":71:5)
#loc138 = loc("backprop_kernel.cpp":72:12)
#loc139 = loc("backprop_kernel.cpp":72:18)
#loc140 = loc("backprop_kernel.cpp":72:29)
#loc141 = loc("backprop_kernel.cpp":72:42)
#loc142 = loc("backprop_kernel.cpp":72:5)
#loc143 = loc("backprop_kernel.cpp":73:1)
