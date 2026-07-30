#loc1 = loc("hotspot.cpp":67:6)
#loc10 = loc("hotspot.cpp":103:39)
#loc11 = loc("hotspot.cpp":102:33)
#loc88 = loc("hotspot.cpp":112:39)
#loc122 = loc("hotspot.cpp":20:6)
#loc132 = loc("hotspot.cpp":38:35)
#loc133 = loc("./hotspot.h":18:21)
#loc134 = loc("hotspot.cpp":32:35)
#loc156 = loc("hotspot.cpp":33:28)
#loc257 = loc("hotspot.cpp":56:43)
#loc280 = loc("hotspot.cpp":7:7)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<262144xf32> loc("hotspot.cpp":67:6), %arg1: memref<262144xf32> loc("hotspot.cpp":67:6), %arg2: memref<262144xf32> loc("hotspot.cpp":67:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.125000e-05 : f64 loc(#loc2)
    %cst_0 = arith.constant 4.375000e+02 : f64 loc(#loc3)
    %cst_1 = arith.constant 1.000000e-01 : f64 loc(#loc4)
    %cst_2 = arith.constant 68.571428571428569 : f64 loc(#loc5)
    %c1_i32 = arith.constant 1 : i32 loc(#loc6)
    %c64_i64 = arith.constant 64 : i64 loc(#loc7)
    %c512_i64 = arith.constant 512 : i64 loc(#loc8)
    %c66_i64 = arith.constant 66 : i64 loc(#loc9)
    %c512_i32 = arith.constant 512 : i32 loc(#loc8)
    %c64_i32 = arith.constant 64 : i32 loc(#loc7)
    %c8_i32 = arith.constant 8 : i32 loc(#loc10)
    %c32_i32 = arith.constant 32 : i32 loc(#loc11)
    %c0_i32 = arith.constant 0 : i32 loc(#loc12)
    %cst_3 = arith.constant 1.000000e+00 : f32 loc(#loc13)
    %cst_4 = arith.constant 1.000000e+03 : f64 loc(#loc14)
    %cst_5 = arith.constant 1.000000e-03 : f64 loc(#loc15)
    %cst_6 = arith.constant 1.000000e+02 : f32 loc(#loc16)
    %cst_7 = arith.constant 5.000000e-04 : f64 loc(#loc17)
    %true = arith.constant true loc(#loc18)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc19)
    %alloca = memref.alloca() : memref<32768xf32> loc(#loc20)
    %alloca_8 = memref.alloca() : memref<33792xf32> loc(#loc21)
    %alloca_9 = memref.alloca() : memref<32768xf32> loc(#loc22)
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
        cf.br ^bb1 loc(#loc25)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    %1 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.truncf %cst : f64 to f32 loc(#loc26)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %2 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.truncf %cst : f64 to f32 loc(#loc27)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %3 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.extf %2 : f32 to f64 loc(#loc28)
            %17 = arith.mulf %16, %cst_0 : f64 loc(#loc29)
            %18 = arith.extf %1 : f32 to f64 loc(#loc30)
            %19 = arith.mulf %17, %18 : f64 loc(#loc31)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc32)
            scf.yield %20 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %4 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.extf %2 : f32 to f64 loc(#loc33)
            %17 = arith.extf %1 : f32 to f64 loc(#loc34)
            %18 = arith.mulf %17, %cst_1 : f64 loc(#loc35)
            %19 = arith.divf %16, %18 : f64 loc(#loc36)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc33)
            scf.yield %20 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %5 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.extf %1 : f32 to f64 loc(#loc37)
            %17 = arith.extf %2 : f32 to f64 loc(#loc38)
            %18 = arith.mulf %17, %cst_1 : f64 loc(#loc39)
            %19 = arith.divf %16, %18 : f64 loc(#loc40)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc37)
            scf.yield %20 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %6 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.mulf %1, %cst_6 : f32 loc(#loc41)
            %17 = arith.mulf %16, %2 : f32 loc(#loc42)
            %18 = arith.extf %17 : f32 to f64 loc(#loc43)
            %19 = arith.divf %cst_7, %18 : f64 loc(#loc44)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc17)
            scf.yield %20 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %7 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.truncf %cst_2 : f64 to f32 loc(#loc45)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %8 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.extf %7 : f32 to f64 loc(#loc46)
            %17 = arith.divf %cst_5, %16 : f64 loc(#loc47)
            %18 = arith.divf %17, %cst_4 : f64 loc(#loc48)
            %19 = arith.truncf %18 : f64 to f32 loc(#loc15)
            scf.yield %19 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %9 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.divf %cst_3, %4 : f32 loc(#loc49)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %10 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.divf %cst_3, %5 : f32 loc(#loc50)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %11 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.divf %cst_3, %6 : f32 loc(#loc51)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %12 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.divf %8, %3 : f32 loc(#loc52)
            scf.yield %16 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %15 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %14 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %13 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
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
        cf.br ^bb1 loc(#loc53)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %13 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
              %14 = arith.cmpi slt, %arg3, %c32_i32 : i32 loc(#loc54)
              scf.condition(%14) %arg3 : i32 loc(#loc55)
            } do {
            ^bb0(%arg3: i32 loc("hotspot.cpp":102:33)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc56)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %15 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
                        %16 = arith.cmpi slt, %arg4, %c8_i32 : i32 loc(#loc57)
                        scf.condition(%16) %arg4 : i32 loc(#loc58)
                      } do {
                      ^bb0(%arg4: i32 loc("hotspot.cpp":103:39)):
                        scf.if %true {
                          scf.execute_region {
                            %17 = "polygeist.memref2pointer"(%alloca_8) : (memref<33792xf32>) -> !llvm.ptr loc(#loc59)
                            %18 = arith.muli %arg4, %c64_i32 : i32 loc(#loc60)
                            %19 = arith.muli %18, %c512_i32 : i32 loc(#loc61)
                            %20 = arith.index_cast %19 : i32 to index loc(#loc62)
                            %21 = "polygeist.subindex"(%arg1, %20) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc62)
                            %22 = "polygeist.memref2pointer"(%21) : (memref<?xf32>) -> !llvm.ptr loc(#loc63)
                            %23 = llvm.getelementptr %22[-512] : (!llvm.ptr) -> !llvm.ptr, f32 loc(#loc63)
                            %24 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc64)
                            %25 = arith.index_cast %24 : index to i64 loc(#loc64)
                            %26 = arith.muli %25, %c66_i64 : i64 loc(#loc65)
                            %27 = arith.muli %26, %c512_i64 : i64 loc(#loc66)
                            "llvm.intr.memcpy"(%17, %23, %27) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc67)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %17 = "polygeist.memref2pointer"(%alloca) : (memref<32768xf32>) -> !llvm.ptr loc(#loc68)
                            %18 = arith.muli %arg4, %c64_i32 : i32 loc(#loc69)
                            %19 = arith.muli %18, %c512_i32 : i32 loc(#loc70)
                            %20 = arith.index_cast %19 : i32 to index loc(#loc71)
                            %21 = "polygeist.subindex"(%arg2, %20) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc71)
                            %22 = "polygeist.memref2pointer"(%21) : (memref<?xf32>) -> !llvm.ptr loc(#loc72)
                            %23 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc73)
                            %24 = arith.index_cast %23 : index to i64 loc(#loc73)
                            %25 = arith.muli %24, %c64_i64 : i64 loc(#loc74)
                            %26 = arith.muli %25, %c512_i64 : i64 loc(#loc75)
                            "llvm.intr.memcpy"(%17, %22, %26) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc76)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            func.call @hotspot(%alloca_9, %alloca_8, %alloca, %12, %9, %10, %11, %arg4) : (memref<32768xf32>, memref<33792xf32>, memref<32768xf32>, f32, f32, f32, f32, i32) -> () loc(#loc77)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %17 = arith.muli %arg4, %c64_i32 : i32 loc(#loc78)
                            %18 = arith.muli %17, %c512_i32 : i32 loc(#loc79)
                            %19 = arith.index_cast %18 : i32 to index loc(#loc80)
                            %20 = "polygeist.subindex"(%arg0, %19) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc80)
                            %21 = "polygeist.memref2pointer"(%20) : (memref<?xf32>) -> !llvm.ptr loc(#loc81)
                            %22 = "polygeist.memref2pointer"(%alloca_9) : (memref<32768xf32>) -> !llvm.ptr loc(#loc82)
                            %23 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc83)
                            %24 = arith.index_cast %23 : index to i64 loc(#loc83)
                            %25 = arith.muli %24, %c64_i64 : i64 loc(#loc84)
                            %26 = arith.muli %25, %c512_i64 : i64 loc(#loc85)
                            "llvm.intr.memcpy"(%21, %22, %26) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc86)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %16 = scf.if %true -> (i32) {
                          %17 = scf.execute_region -> i32 {
                            %18 = arith.addi %arg4, %c1_i32 : i32 loc(#loc6)
                            scf.yield %18 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %16 : i32 loc(#loc58)
                      } loc(#loc10)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc87)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %15 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
                        %16 = arith.cmpi slt, %arg4, %c8_i32 : i32 loc(#loc89)
                        scf.condition(%16) %arg4 : i32 loc(#loc90)
                      } do {
                      ^bb0(%arg4: i32 loc("hotspot.cpp":112:39)):
                        scf.if %true {
                          scf.execute_region {
                            %17 = "polygeist.memref2pointer"(%alloca_8) : (memref<33792xf32>) -> !llvm.ptr loc(#loc91)
                            %18 = arith.muli %arg4, %c64_i32 : i32 loc(#loc92)
                            %19 = arith.muli %18, %c512_i32 : i32 loc(#loc93)
                            %20 = arith.index_cast %19 : i32 to index loc(#loc94)
                            %21 = "polygeist.subindex"(%arg0, %20) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc94)
                            %22 = "polygeist.memref2pointer"(%21) : (memref<?xf32>) -> !llvm.ptr loc(#loc95)
                            %23 = llvm.getelementptr %22[-512] : (!llvm.ptr) -> !llvm.ptr, f32 loc(#loc95)
                            %24 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc96)
                            %25 = arith.index_cast %24 : index to i64 loc(#loc96)
                            %26 = arith.muli %25, %c66_i64 : i64 loc(#loc97)
                            %27 = arith.muli %26, %c512_i64 : i64 loc(#loc98)
                            "llvm.intr.memcpy"(%17, %23, %27) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc99)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %17 = "polygeist.memref2pointer"(%alloca) : (memref<32768xf32>) -> !llvm.ptr loc(#loc100)
                            %18 = arith.muli %arg4, %c64_i32 : i32 loc(#loc101)
                            %19 = arith.muli %18, %c512_i32 : i32 loc(#loc102)
                            %20 = arith.index_cast %19 : i32 to index loc(#loc103)
                            %21 = "polygeist.subindex"(%arg2, %20) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc103)
                            %22 = "polygeist.memref2pointer"(%21) : (memref<?xf32>) -> !llvm.ptr loc(#loc104)
                            %23 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc105)
                            %24 = arith.index_cast %23 : index to i64 loc(#loc105)
                            %25 = arith.muli %24, %c64_i64 : i64 loc(#loc106)
                            %26 = arith.muli %25, %c512_i64 : i64 loc(#loc107)
                            "llvm.intr.memcpy"(%17, %22, %26) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc108)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            func.call @hotspot(%alloca_9, %alloca_8, %alloca, %12, %9, %10, %11, %arg4) : (memref<32768xf32>, memref<33792xf32>, memref<32768xf32>, f32, f32, f32, f32, i32) -> () loc(#loc109)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %17 = arith.muli %arg4, %c64_i32 : i32 loc(#loc110)
                            %18 = arith.muli %17, %c512_i32 : i32 loc(#loc111)
                            %19 = arith.index_cast %18 : i32 to index loc(#loc112)
                            %20 = "polygeist.subindex"(%arg1, %19) : (memref<262144xf32>, index) -> memref<?xf32> loc(#loc112)
                            %21 = "polygeist.memref2pointer"(%20) : (memref<?xf32>) -> !llvm.ptr loc(#loc113)
                            %22 = "polygeist.memref2pointer"(%alloca_9) : (memref<32768xf32>) -> !llvm.ptr loc(#loc114)
                            %23 = "polygeist.typeSize"() <{source = f32}> : () -> index loc(#loc115)
                            %24 = arith.index_cast %23 : index to i64 loc(#loc115)
                            %25 = arith.muli %24, %c64_i64 : i64 loc(#loc116)
                            %26 = arith.muli %25, %c512_i64 : i64 loc(#loc117)
                            "llvm.intr.memcpy"(%21, %22, %26) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> () loc(#loc118)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %16 = scf.if %true -> (i32) {
                          %17 = scf.execute_region -> i32 {
                            %18 = arith.addi %arg4, %c1_i32 : i32 loc(#loc119)
                            scf.yield %18 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %17 : i32 loc(#loc)
                        } else {
                          scf.yield %arg4 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %16 : i32 loc(#loc90)
                      } loc(#loc88)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %14 = scf.if %true -> (i32) {
                %15 = scf.execute_region -> i32 {
                  %16 = arith.addi %arg3, %c1_i32 : i32 loc(#loc120)
                  scf.yield %16 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %15 : i32 loc(#loc)
              } else {
                scf.yield %arg3 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %14 : i32 loc(#loc55)
            } loc(#loc11)
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
    return loc(#loc121)
  } loc(#loc1)
  func.func @hotspot(%arg0: memref<32768xf32> loc("hotspot.cpp":20:6), %arg1: memref<33792xf32> loc("hotspot.cpp":20:6), %arg2: memref<32768xf32> loc("hotspot.cpp":20:6), %arg3: f32 loc("hotspot.cpp":20:6), %arg4: f32 loc("hotspot.cpp":20:6), %arg5: f32 loc("hotspot.cpp":20:6), %arg6: f32 loc("hotspot.cpp":20:6), %arg7: i32 loc("hotspot.cpp":20:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc123)
    %c65_i32 = arith.constant 65 : i32 loc(#loc124)
    %c2048_i32 = arith.constant 2048 : i32 loc(#loc125)
    %c32 = arith.constant 32 : index loc(#loc126)
    %c31_i32 = arith.constant 31 : i32 loc(#loc127)
    %c2016_i32 = arith.constant 2016 : i32 loc(#loc128)
    %c7_i32 = arith.constant 7 : i32 loc(#loc129)
    %c64 = arith.constant 64 : index loc(#loc130)
    %c15_i32 = arith.constant 15 : i32 loc(#loc131)
    %c64_i32 = arith.constant 64 : i32 loc(#loc7)
    %c32_i32 = arith.constant 32 : i32 loc(#loc132)
    %false = arith.constant false loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc123)
    %c16_i32 = arith.constant 16 : i32 loc(#loc133)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc134)
    %c0_i32 = arith.constant 0 : i32 loc(#loc135)
    %true = arith.constant true loc(#loc136)
    %c0 = arith.constant 0 : index loc(#loc137)
    %alloca = memref.alloca() : memref<16x65xf32> loc(#loc138)
    %alloca_0 = memref.alloca() : memref<16xf32> loc(#loc139)
    %alloca_1 = memref.alloca() : memref<16xf32> loc(#loc140)
    %alloca_2 = memref.alloca() : memref<16xf32> loc(#loc141)
    %alloca_3 = memref.alloca() : memref<16xf32> loc(#loc142)
    %alloca_4 = memref.alloca() : memref<16xf32> loc(#loc143)
    %alloca_5 = memref.alloca() : memref<16xf32> loc(#loc144)
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
        cf.br ^bb1 loc(#loc145)
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
        cf.br ^bb1 loc(#loc146)
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
        cf.br ^bb1 loc(#loc147)
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
        cf.br ^bb1 loc(#loc148)
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
        cf.br ^bb1 loc(#loc149)
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
        cf.br ^bb1 loc(#loc150)
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
        cf.br ^bb1 loc(#loc151)
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
        cf.br ^bb1 loc(#loc152)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %0 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
              %1 = arith.cmpi slt, %arg8, %c65_i32 : i32 loc(#loc153)
              scf.condition(%1) %arg8 : i32 loc(#loc154)
            } do {
            ^bb0(%arg8: i32 loc("hotspot.cpp":32:35)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc155)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %2 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
                        %3 = arith.cmpi slt, %arg9, %c16_i32 : i32 loc(#loc156)
                        scf.condition(%3) %arg9 : i32 loc(#loc157)
                      } do {
                      ^bb0(%arg9: i32 loc("hotspot.cpp":33:28)):
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc158)
                            %5 = "polygeist.subindex"(%alloca, %4) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc159)
                            %6 = "polygeist.subindex"(%5, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc159)
                            %7 = arith.index_cast %arg8 : i32 to index loc(#loc160)
                            %8 = "polygeist.subindex"(%6, %7) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc159)
                            %9 = arith.muli %arg8, %c16_i32 : i32 loc(#loc161)
                            %10 = arith.addi %9, %arg9 : i32 loc(#loc162)
                            %11 = arith.index_cast %10 : i32 to index loc(#loc163)
                            %12 = "polygeist.subindex"(%arg1, %11) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc164)
                            %13 = affine.load %12[0] : memref<?xf32> loc(#loc164)
                            affine.store %13, %8[0] : memref<?xf32> loc(#loc165)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %3 = scf.if %true -> (i32) {
                          %4 = scf.execute_region -> i32 {
                            %5 = arith.addi %arg9, %c1_i32 : i32 loc(#loc166)
                            scf.yield %5 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %4 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %3 : i32 loc(#loc157)
                      } loc(#loc156)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %1 = scf.if %true -> (i32) {
                %2 = scf.execute_region -> i32 {
                  %3 = arith.addi %arg8, %c1_i32 : i32 loc(#loc167)
                  scf.yield %3 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %2 : i32 loc(#loc)
              } else {
                scf.yield %arg8 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %1 : i32 loc(#loc154)
            } loc(#loc134)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc168)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %0 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
              %1 = arith.cmpi slt, %arg8, %c2048_i32 : i32 loc(#loc169)
              scf.condition(%1) %arg8 : i32 loc(#loc170)
            } do {
            ^bb0(%arg8: i32 loc("hotspot.cpp":38:35)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc171)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %2 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
                        %3 = arith.cmpi slt, %arg9, %c16_i32 : i32 loc(#loc172)
                        scf.condition(%3) %arg9 : i32 loc(#loc173)
                      } do {
                      ^bb0(%arg9: i32 loc("./hotspot.h":18:21)):
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc174)
                            %5 = "polygeist.subindex"(%alloca_1, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc175)
                            %6 = "polygeist.subindex"(%alloca, %4) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc176)
                            %7 = "polygeist.subindex"(%6, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc176)
                            %8 = "polygeist.subindex"(%7, %c32) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc176)
                            %9 = affine.load %8[0] : memref<?xf32> loc(#loc176)
                            affine.store %9, %5[0] : memref<?xf32> loc(#loc177)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc178)
                            %5 = "polygeist.subindex"(%alloca_5, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc179)
                            %6 = arith.cmpi slt, %arg8, %c32_i32 : i32 loc(#loc180)
                            %7 = scf.if %6 -> (i1) {
                              %10 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc182)
                              scf.yield %10 : i1 loc(#loc181)
                            } else {
                              scf.yield %false : i1 loc(#loc181)
                            } loc(#loc181)
                            %8 = scf.if %7 -> (memref<?xf32>) {
                              %10 = "polygeist.subindex"(%alloca_1, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc184)
                              scf.yield %10 : memref<?xf32> loc(#loc183)
                            } else {
                              %10 = "polygeist.subindex"(%alloca, %4) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc185)
                              %11 = "polygeist.subindex"(%10, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc185)
                              %cast = memref.cast %11 : memref<65xf32> to memref<?xf32> loc(#loc185)
                              scf.yield %cast : memref<?xf32> loc(#loc183)
                            } loc(#loc183)
                            %9 = affine.load %8[0] : memref<?xf32> loc(#loc183)
                            affine.store %9, %5[0] : memref<?xf32> loc(#loc186)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc187)
                            %5 = "polygeist.subindex"(%alloca_4, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc188)
                            %6 = arith.remsi %arg8, %c32_i32 : i32 loc(#loc189)
                            %7 = arith.cmpi eq, %6, %c0_i32 : i32 loc(#loc190)
                            %8 = scf.if %7 -> (i1) {
                              %11 = arith.cmpi eq, %arg9, %c0_i32 : i32 loc(#loc192)
                              scf.yield %11 : i1 loc(#loc191)
                            } else {
                              scf.yield %false : i1 loc(#loc191)
                            } loc(#loc191)
                            %9 = scf.if %8 -> (memref<?xf32>) {
                              %11 = "polygeist.subindex"(%alloca_1, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc194)
                              scf.yield %11 : memref<?xf32> loc(#loc193)
                            } else {
                              %11 = arith.addi %arg9, %c-1_i32 : i32 loc(#loc195)
                              %12 = arith.addi %11, %c16_i32 : i32 loc(#loc196)
                              %13 = arith.remsi %12, %c16_i32 : i32 loc(#loc197)
                              %14 = arith.index_cast %13 : i32 to index loc(#loc198)
                              %15 = "polygeist.subindex"(%alloca, %14) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc199)
                              %16 = "polygeist.subindex"(%15, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc199)
                              %17 = arith.cmpi eq, %arg9, %c0_i32 : i32 loc(#loc200)
                              %18 = arith.extui %17 : i1 to i32 loc(#loc201)
                              %19 = arith.subi %c32_i32, %18 : i32 loc(#loc202)
                              %20 = arith.index_cast %19 : i32 to index loc(#loc203)
                              %21 = "polygeist.subindex"(%16, %20) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc199)
                              scf.yield %21 : memref<?xf32> loc(#loc193)
                            } loc(#loc193)
                            %10 = affine.load %9[0] : memref<?xf32> loc(#loc193)
                            affine.store %10, %5[0] : memref<?xf32> loc(#loc204)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc205)
                            %5 = "polygeist.subindex"(%alloca_3, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc206)
                            %6 = arith.remsi %arg8, %c32_i32 : i32 loc(#loc207)
                            %7 = arith.cmpi eq, %6, %c31_i32 : i32 loc(#loc208)
                            %8 = scf.if %7 -> (i1) {
                              %11 = arith.cmpi eq, %arg9, %c15_i32 : i32 loc(#loc210)
                              scf.yield %11 : i1 loc(#loc209)
                            } else {
                              scf.yield %false : i1 loc(#loc209)
                            } loc(#loc209)
                            %9 = scf.if %8 -> (memref<?xf32>) {
                              %11 = "polygeist.subindex"(%alloca_1, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc212)
                              scf.yield %11 : memref<?xf32> loc(#loc211)
                            } else {
                              %11 = arith.addi %arg9, %c1_i32 : i32 loc(#loc213)
                              %12 = arith.addi %11, %c16_i32 : i32 loc(#loc214)
                              %13 = arith.remsi %12, %c16_i32 : i32 loc(#loc215)
                              %14 = arith.index_cast %13 : i32 to index loc(#loc216)
                              %15 = "polygeist.subindex"(%alloca, %14) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc217)
                              %16 = "polygeist.subindex"(%15, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc217)
                              %17 = arith.cmpi eq, %arg9, %c15_i32 : i32 loc(#loc218)
                              %18 = arith.extui %17 : i1 to i32 loc(#loc219)
                              %19 = arith.addi %18, %c32_i32 : i32 loc(#loc220)
                              %20 = arith.index_cast %19 : i32 to index loc(#loc221)
                              %21 = "polygeist.subindex"(%16, %20) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc217)
                              scf.yield %21 : memref<?xf32> loc(#loc211)
                            } loc(#loc211)
                            %10 = affine.load %9[0] : memref<?xf32> loc(#loc211)
                            affine.store %10, %5[0] : memref<?xf32> loc(#loc222)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc223)
                            %5 = "polygeist.subindex"(%alloca_2, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc224)
                            %6 = arith.cmpi sge, %arg8, %c2016_i32 : i32 loc(#loc225)
                            %7 = scf.if %6 -> (i1) {
                              %10 = arith.cmpi eq, %arg7, %c7_i32 : i32 loc(#loc227)
                              scf.yield %10 : i1 loc(#loc226)
                            } else {
                              scf.yield %false : i1 loc(#loc226)
                            } loc(#loc226)
                            %8 = scf.if %7 -> (memref<?xf32>) {
                              %10 = "polygeist.subindex"(%alloca_1, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc229)
                              scf.yield %10 : memref<?xf32> loc(#loc228)
                            } else {
                              %10 = "polygeist.subindex"(%alloca, %4) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc230)
                              %11 = "polygeist.subindex"(%10, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc230)
                              %12 = "polygeist.subindex"(%11, %c64) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc230)
                              scf.yield %12 : memref<?xf32> loc(#loc228)
                            } loc(#loc228)
                            %9 = affine.load %8[0] : memref<?xf32> loc(#loc228)
                            affine.store %9, %5[0] : memref<?xf32> loc(#loc231)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc232)
                            %5 = "polygeist.subindex"(%alloca_0, %4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc233)
                            %6 = arith.muli %arg8, %c16_i32 : i32 loc(#loc234)
                            %7 = arith.addi %6, %arg9 : i32 loc(#loc235)
                            %8 = arith.index_cast %7 : i32 to index loc(#loc236)
                            %9 = "polygeist.subindex"(%arg2, %8) : (memref<32768xf32>, index) -> memref<?xf32> loc(#loc237)
                            %10 = affine.load %9[0] : memref<?xf32> loc(#loc237)
                            affine.store %10, %5[0] : memref<?xf32> loc(#loc238)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.muli %arg8, %c16_i32 : i32 loc(#loc239)
                            %5 = arith.addi %4, %arg9 : i32 loc(#loc240)
                            %6 = arith.index_cast %5 : i32 to index loc(#loc241)
                            %7 = "polygeist.subindex"(%arg0, %6) : (memref<32768xf32>, index) -> memref<?xf32> loc(#loc242)
                            %8 = arith.index_cast %arg9 : i32 to index loc(#loc243)
                            %9 = "polygeist.subindex"(%alloca_5, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc244)
                            %10 = affine.load %9[0] : memref<?xf32> loc(#loc244)
                            %11 = "polygeist.subindex"(%alloca_4, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc245)
                            %12 = affine.load %11[0] : memref<?xf32> loc(#loc245)
                            %13 = "polygeist.subindex"(%alloca_3, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc246)
                            %14 = affine.load %13[0] : memref<?xf32> loc(#loc246)
                            %15 = "polygeist.subindex"(%alloca_2, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc247)
                            %16 = affine.load %15[0] : memref<?xf32> loc(#loc247)
                            %17 = "polygeist.subindex"(%alloca_1, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc248)
                            %18 = affine.load %17[0] : memref<?xf32> loc(#loc248)
                            %19 = "polygeist.subindex"(%alloca_0, %8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc249)
                            %20 = affine.load %19[0] : memref<?xf32> loc(#loc249)
                            %21 = func.call @hotspot_stencil_core(%10, %12, %14, %16, %18, %20, %arg3, %arg4, %arg5, %arg6) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> f32 loc(#loc250)
                            affine.store %21, %7[0] : memref<?xf32> loc(#loc251)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %3 = scf.if %true -> (i32) {
                          %4 = scf.execute_region -> i32 {
                            %5 = arith.addi %arg9, %c1_i32 : i32 loc(#loc252)
                            scf.yield %5 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %4 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %3 : i32 loc(#loc173)
                      } loc(#loc133)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc253)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %2 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
                        %3 = arith.cmpi slt, %arg9, %c16_i32 : i32 loc(#loc254)
                        scf.condition(%3) %arg9 : i32 loc(#loc255)
                      } do {
                      ^bb0(%arg9: i32 loc("./hotspot.h":18:21)):
                        scf.if %true {
                          scf.execute_region {
                            cf.br ^bb1 loc(#loc256)
                          ^bb1:  // pred: ^bb0
                            scf.if %true {
                              scf.execute_region {
                                %4 = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
                                  %5 = arith.cmpi slt, %arg10, %c64_i32 : i32 loc(#loc258)
                                  scf.condition(%5) %arg10 : i32 loc(#loc259)
                                } do {
                                ^bb0(%arg10: i32 loc("hotspot.cpp":56:43)):
                                  scf.if %true {
                                    scf.execute_region {
                                      %6 = arith.index_cast %arg9 : i32 to index loc(#loc260)
                                      %7 = "polygeist.subindex"(%alloca, %6) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc261)
                                      %8 = "polygeist.subindex"(%7, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc261)
                                      %9 = arith.index_cast %arg10 : i32 to index loc(#loc262)
                                      %10 = "polygeist.subindex"(%8, %9) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc261)
                                      %11 = arith.addi %arg10, %c1_i32 : i32 loc(#loc263)
                                      %12 = arith.index_cast %11 : i32 to index loc(#loc264)
                                      %13 = "polygeist.subindex"(%8, %12) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc265)
                                      %14 = affine.load %13[0] : memref<?xf32> loc(#loc265)
                                      affine.store %14, %10[0] : memref<?xf32> loc(#loc266)
                                      scf.yield loc(#loc)
                                    } loc(#loc)
                                  } loc(#loc)
                                  %5 = scf.if %true -> (i32) {
                                    %6 = scf.execute_region -> i32 {
                                      %7 = arith.addi %arg10, %c1_i32 : i32 loc(#loc267)
                                      scf.yield %7 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %6 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg10 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %5 : i32 loc(#loc259)
                                } loc(#loc257)
                                scf.yield loc(#loc)
                              } loc(#loc)
                            } loc(#loc)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %4 = arith.index_cast %arg9 : i32 to index loc(#loc268)
                            %5 = "polygeist.subindex"(%alloca, %4) : (memref<16x65xf32>, index) -> memref<?x65xf32> loc(#loc269)
                            %6 = "polygeist.subindex"(%5, %c0) : (memref<?x65xf32>, index) -> memref<65xf32> loc(#loc269)
                            %7 = "polygeist.subindex"(%6, %c64) : (memref<65xf32>, index) -> memref<?xf32> loc(#loc269)
                            %8 = arith.addi %arg8, %c1_i32 : i32 loc(#loc270)
                            %9 = arith.muli %8, %c16_i32 : i32 loc(#loc271)
                            %10 = arith.addi %9, %c1024_i32 : i32 loc(#loc272)
                            %11 = arith.addi %10, %arg9 : i32 loc(#loc273)
                            %12 = arith.index_cast %11 : i32 to index loc(#loc274)
                            %13 = "polygeist.subindex"(%arg1, %12) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc275)
                            %14 = affine.load %13[0] : memref<?xf32> loc(#loc275)
                            affine.store %14, %7[0] : memref<?xf32> loc(#loc276)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %3 = scf.if %true -> (i32) {
                          %4 = scf.execute_region -> i32 {
                            %5 = arith.addi %arg9, %c1_i32 : i32 loc(#loc277)
                            scf.yield %5 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %4 : i32 loc(#loc)
                        } else {
                          scf.yield %arg9 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %3 : i32 loc(#loc255)
                      } loc(#loc133)
                      scf.yield loc(#loc)
                    } loc(#loc)
                  } loc(#loc)
                  scf.yield loc(#loc)
                } loc(#loc)
              } loc(#loc)
              %1 = scf.if %true -> (i32) {
                %2 = scf.execute_region -> i32 {
                  %3 = arith.addi %arg8, %c1_i32 : i32 loc(#loc278)
                  scf.yield %3 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %2 : i32 loc(#loc)
              } else {
                scf.yield %arg8 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %1 : i32 loc(#loc170)
            } loc(#loc132)
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
    return loc(#loc279)
  } loc(#loc122)
  func.func @hotspot_stencil_core(%arg0: f32 loc("hotspot.cpp":7:7), %arg1: f32 loc("hotspot.cpp":7:7), %arg2: f32 loc("hotspot.cpp":7:7), %arg3: f32 loc("hotspot.cpp":7:7), %arg4: f32 loc("hotspot.cpp":7:7), %arg5: f32 loc("hotspot.cpp":7:7), %arg6: f32 loc("hotspot.cpp":7:7), %arg7: f32 loc("hotspot.cpp":7:7), %arg8: f32 loc("hotspot.cpp":7:7), %arg9: f32 loc("hotspot.cpp":7:7)) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 8.000000e+01 : f32 loc(#loc281)
    %true = arith.constant true loc(#loc282)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc283)
    %1 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = scf.if %true -> (f32) {
          %12 = scf.execute_region -> f32 {
            %13 = arith.addf %arg4, %arg4 : f32 loc(#loc284)
            scf.yield %13 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %12 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %11 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %2 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = scf.if %true -> (f32) {
          %12 = scf.execute_region -> f32 {
            %13 = arith.addf %arg0, %arg3 : f32 loc(#loc285)
            %14 = arith.subf %13, %1 : f32 loc(#loc286)
            scf.yield %14 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %12 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %11 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %3 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = scf.if %true -> (f32) {
          %12 = scf.execute_region -> f32 {
            %13 = arith.addf %arg1, %arg2 : f32 loc(#loc287)
            %14 = arith.subf %13, %1 : f32 loc(#loc288)
            scf.yield %14 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %12 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %11 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %4 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = scf.if %true -> (f32) {
          %12 = scf.execute_region -> f32 {
            %13 = arith.subf %cst, %arg4 : f32 loc(#loc289)
            scf.yield %13 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %12 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %11 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    %5 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = arith.mulf %2, %arg8 : f32 loc(#loc290)
        %12 = arith.addf %arg5, %11 : f32 loc(#loc291)
        scf.yield %12 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %arg5 : f32 loc(#loc)
    } loc(#loc)
    %6 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = arith.mulf %3, %arg7 : f32 loc(#loc292)
        %12 = arith.addf %5, %11 : f32 loc(#loc293)
        scf.yield %12 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %5 : f32 loc(#loc)
    } loc(#loc)
    %7 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = arith.mulf %4, %arg9 : f32 loc(#loc294)
        %12 = arith.addf %6, %11 : f32 loc(#loc295)
        scf.yield %12 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %6 : f32 loc(#loc)
    } loc(#loc)
    %8 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = arith.mulf %arg6, %7 : f32 loc(#loc296)
        %12 = arith.addf %arg4, %11 : f32 loc(#loc297)
        scf.yield %12 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %arg4 : f32 loc(#loc)
    } loc(#loc)
    %9 = scf.if %true -> (f32) {
      %10 = scf.execute_region -> f32 {
        %11 = scf.if %true -> (f32) {
          scf.execute_region {
            scf.yield loc(#loc)
          } loc(#loc)
          scf.yield %8 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %11 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %10 : f32 loc(#loc)
    } else {
      scf.yield %0 : f32 loc(#loc)
    } loc(#loc)
    return %9 : f32 loc(#loc298)
  } loc(#loc280)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("hotspot.cpp":84:35)
#loc3 = loc("hotspot.cpp":86:44)
#loc4 = loc("hotspot.cpp":88:42)
#loc5 = loc("hotspot.cpp":91:30)
#loc6 = loc("hotspot.cpp":103:53)
#loc7 = loc("./hotspot.h":16:19)
#loc8 = loc("./hotspot.h":14:19)
#loc9 = loc("hotspot.cpp":104:94)
#loc12 = loc("hotspot.cpp":102:18)
#loc13 = loc("hotspot.cpp":94:16)
#loc14 = loc("hotspot.cpp":92:42)
#loc15 = loc("./hotspot.h":27:19)
#loc16 = loc("./hotspot.h":29:14)
#loc17 = loc("./hotspot.h":36:16)
#loc18 = loc("hotspot.cpp":67:1)
#loc19 = loc("hotspot.cpp":97:5)
#loc20 = loc("hotspot.cpp":81:9)
#loc21 = loc("hotspot.cpp":80:9)
#loc22 = loc("hotspot.cpp":79:9)
#loc23 = loc("hotspot.cpp":79:1)
#loc24 = loc("hotspot.cpp":80:1)
#loc25 = loc("hotspot.cpp":81:1)
#loc26 = loc("./hotspot.h":37:21)
#loc27 = loc("./hotspot.h":38:20)
#loc28 = loc("hotspot.cpp":86:55)
#loc29 = loc("hotspot.cpp":86:53)
#loc30 = loc("hotspot.cpp":86:68)
#loc31 = loc("hotspot.cpp":86:66)
#loc32 = loc("./hotspot.h":32:21)
#loc33 = loc("hotspot.cpp":87:16)
#loc34 = loc("hotspot.cpp":87:52)
#loc35 = loc("hotspot.cpp":87:50)
#loc36 = loc("hotspot.cpp":87:27)
#loc37 = loc("hotspot.cpp":88:16)
#loc38 = loc("hotspot.cpp":88:53)
#loc39 = loc("hotspot.cpp":88:51)
#loc40 = loc("hotspot.cpp":88:28)
#loc41 = loc("hotspot.cpp":89:31)
#loc42 = loc("hotspot.cpp":89:45)
#loc43 = loc("hotspot.cpp":89:25)
#loc44 = loc("hotspot.cpp":89:23)
#loc45 = loc("./hotspot.h":24:17)
#loc46 = loc("hotspot.cpp":92:30)
#loc47 = loc("hotspot.cpp":92:28)
#loc48 = loc("hotspot.cpp":92:40)
#loc49 = loc("hotspot.cpp":94:20)
#loc50 = loc("hotspot.cpp":95:20)
#loc51 = loc("hotspot.cpp":96:20)
#loc52 = loc("hotspot.cpp":97:24)
#loc53 = loc("hotspot.cpp":102:1)
#loc54 = loc("hotspot.cpp":102:23)
#loc55 = loc("hotspot.cpp":102:9)
#loc56 = loc("hotspot.cpp":103:1)
#loc57 = loc("hotspot.cpp":103:27)
#loc58 = loc("hotspot.cpp":103:13)
#loc59 = loc("hotspot.cpp":104:20)
#loc60 = loc("hotspot.cpp":104:41)
#loc61 = loc("hotspot.cpp":104:53)
#loc62 = loc("hotspot.cpp":104:37)
#loc63 = loc("hotspot.cpp":104:65)
#loc64 = loc("hotspot.cpp":104:78)
#loc65 = loc("hotspot.cpp":104:92)
#loc66 = loc("hotspot.cpp":104:110)
#loc67 = loc("hotspot.cpp":104:13)
#loc68 = loc("hotspot.cpp":105:20)
#loc69 = loc("hotspot.cpp":105:43)
#loc70 = loc("hotspot.cpp":105:55)
#loc71 = loc("hotspot.cpp":105:39)
#loc72 = loc("hotspot.cpp":105:33)
#loc73 = loc("hotspot.cpp":105:68)
#loc74 = loc("hotspot.cpp":105:82)
#loc75 = loc("hotspot.cpp":105:94)
#loc76 = loc("hotspot.cpp":105:13)
#loc77 = loc("hotspot.cpp":107:13)
#loc78 = loc("hotspot.cpp":109:31)
#loc79 = loc("hotspot.cpp":109:43)
#loc80 = loc("hotspot.cpp":109:27)
#loc81 = loc("hotspot.cpp":109:20)
#loc82 = loc("hotspot.cpp":109:56)
#loc83 = loc("hotspot.cpp":109:70)
#loc84 = loc("hotspot.cpp":109:84)
#loc85 = loc("hotspot.cpp":109:96)
#loc86 = loc("hotspot.cpp":109:13)
#loc87 = loc("hotspot.cpp":112:1)
#loc89 = loc("hotspot.cpp":112:27)
#loc90 = loc("hotspot.cpp":112:13)
#loc91 = loc("hotspot.cpp":113:20)
#loc92 = loc("hotspot.cpp":113:43)
#loc93 = loc("hotspot.cpp":113:55)
#loc94 = loc("hotspot.cpp":113:39)
#loc95 = loc("hotspot.cpp":113:67)
#loc96 = loc("hotspot.cpp":113:80)
#loc97 = loc("hotspot.cpp":113:94)
#loc98 = loc("hotspot.cpp":113:112)
#loc99 = loc("hotspot.cpp":113:13)
#loc100 = loc("hotspot.cpp":114:20)
#loc101 = loc("hotspot.cpp":114:43)
#loc102 = loc("hotspot.cpp":114:55)
#loc103 = loc("hotspot.cpp":114:39)
#loc104 = loc("hotspot.cpp":114:33)
#loc105 = loc("hotspot.cpp":114:68)
#loc106 = loc("hotspot.cpp":114:82)
#loc107 = loc("hotspot.cpp":114:94)
#loc108 = loc("hotspot.cpp":114:13)
#loc109 = loc("hotspot.cpp":116:13)
#loc110 = loc("hotspot.cpp":118:29)
#loc111 = loc("hotspot.cpp":118:41)
#loc112 = loc("hotspot.cpp":118:25)
#loc113 = loc("hotspot.cpp":118:20)
#loc114 = loc("hotspot.cpp":118:54)
#loc115 = loc("hotspot.cpp":118:69)
#loc116 = loc("hotspot.cpp":118:83)
#loc117 = loc("hotspot.cpp":118:95)
#loc118 = loc("hotspot.cpp":118:13)
#loc119 = loc("hotspot.cpp":112:53)
#loc120 = loc("hotspot.cpp":102:38)
#loc121 = loc("hotspot.cpp":124:1)
#loc123 = loc("hotspot.cpp":32:55)
#loc124 = loc("hotspot.cpp":32:53)
#loc125 = loc("hotspot.cpp":38:49)
#loc126 = loc("hotspot.cpp":40:65)
#loc127 = loc("hotspot.cpp":46:92)
#loc128 = loc("hotspot.cpp":48:61)
#loc129 = loc("./hotspot.h":43:37)
#loc130 = loc("hotspot.cpp":60:51)
#loc131 = loc("hotspot.cpp":46:117)
#loc135 = loc("hotspot.cpp":32:17)
#loc136 = loc("hotspot.cpp":20:1)
#loc137 = loc("hotspot.cpp":22:5)
#loc138 = loc("hotspot.cpp":30:8)
#loc139 = loc("hotspot.cpp":28:5)
#loc140 = loc("hotspot.cpp":27:5)
#loc141 = loc("hotspot.cpp":26:5)
#loc142 = loc("hotspot.cpp":25:5)
#loc143 = loc("hotspot.cpp":24:5)
#loc144 = loc("hotspot.cpp":23:5)
#loc145 = loc("hotspot.cpp":23:1)
#loc146 = loc("hotspot.cpp":24:1)
#loc147 = loc("hotspot.cpp":25:1)
#loc148 = loc("hotspot.cpp":26:1)
#loc149 = loc("hotspot.cpp":27:1)
#loc150 = loc("hotspot.cpp":28:1)
#loc151 = loc("hotspot.cpp":30:1)
#loc152 = loc("hotspot.cpp":32:1)
#loc153 = loc("hotspot.cpp":32:23)
#loc154 = loc("hotspot.cpp":32:8)
#loc155 = loc("hotspot.cpp":33:1)
#loc157 = loc("hotspot.cpp":33:12)
#loc158 = loc("hotspot.cpp":34:23)
#loc159 = loc("hotspot.cpp":34:13)
#loc160 = loc("hotspot.cpp":34:26)
#loc161 = loc("hotspot.cpp":34:36)
#loc162 = loc("hotspot.cpp":34:49)
#loc163 = loc("hotspot.cpp":34:53)
#loc164 = loc("hotspot.cpp":34:30)
#loc165 = loc("hotspot.cpp":34:28)
#loc166 = loc("hotspot.cpp":33:45)
#loc167 = loc("hotspot.cpp":32:59)
#loc168 = loc("hotspot.cpp":38:1)
#loc169 = loc("hotspot.cpp":38:23)
#loc170 = loc("hotspot.cpp":38:9)
#loc171 = loc("hotspot.cpp":39:1)
#loc172 = loc("hotspot.cpp":39:27)
#loc173 = loc("hotspot.cpp":39:13)
#loc174 = loc("hotspot.cpp":40:26)
#loc175 = loc("hotspot.cpp":40:13)
#loc176 = loc("hotspot.cpp":40:31)
#loc177 = loc("hotspot.cpp":40:29)
#loc178 = loc("hotspot.cpp":42:23)
#loc179 = loc("hotspot.cpp":42:13)
#loc180 = loc("hotspot.cpp":42:34)
#loc181 = loc("hotspot.cpp":42:60)
#loc182 = loc("hotspot.cpp":42:78)
#loc183 = loc("hotspot.cpp":42:31)
#loc184 = loc("hotspot.cpp":42:88)
#loc185 = loc("hotspot.cpp":42:105)
#loc186 = loc("hotspot.cpp":42:29)
#loc187 = loc("hotspot.cpp":44:24)
#loc188 = loc("hotspot.cpp":44:13)
#loc189 = loc("hotspot.cpp":44:35)
#loc190 = loc("hotspot.cpp":44:64)
#loc191 = loc("hotspot.cpp":44:69)
#loc192 = loc("hotspot.cpp":44:74)
#loc193 = loc("hotspot.cpp":44:31)
#loc194 = loc("hotspot.cpp":44:82)
#loc195 = loc("hotspot.cpp":44:110)
#loc196 = loc("hotspot.cpp":44:114)
#loc197 = loc("hotspot.cpp":44:129)
#loc198 = loc("hotspot.cpp":44:142)
#loc199 = loc("hotspot.cpp":44:99)
#loc200 = loc("hotspot.cpp":44:173)
#loc201 = loc("hotspot.cpp":44:170)
#loc202 = loc("hotspot.cpp":44:168)
#loc203 = loc("hotspot.cpp":44:179)
#loc204 = loc("hotspot.cpp":44:29)
#loc205 = loc("hotspot.cpp":46:25)
#loc206 = loc("hotspot.cpp":46:13)
#loc207 = loc("hotspot.cpp":46:35)
#loc208 = loc("hotspot.cpp":46:64)
#loc209 = loc("hotspot.cpp":46:97)
#loc210 = loc("hotspot.cpp":46:102)
#loc211 = loc("hotspot.cpp":46:31)
#loc212 = loc("hotspot.cpp":46:124)
#loc213 = loc("hotspot.cpp":46:152)
#loc214 = loc("hotspot.cpp":46:156)
#loc215 = loc("hotspot.cpp":46:171)
#loc216 = loc("hotspot.cpp":46:184)
#loc217 = loc("hotspot.cpp":46:141)
#loc218 = loc("hotspot.cpp":46:215)
#loc219 = loc("hotspot.cpp":46:212)
#loc220 = loc("hotspot.cpp":46:210)
#loc221 = loc("hotspot.cpp":46:237)
#loc222 = loc("hotspot.cpp":46:29)
#loc223 = loc("hotspot.cpp":48:26)
#loc224 = loc("hotspot.cpp":48:13)
#loc225 = loc("hotspot.cpp":48:34)
#loc226 = loc("hotspot.cpp":48:79)
#loc227 = loc("hotspot.cpp":48:97)
#loc228 = loc("hotspot.cpp":48:31)
#loc229 = loc("hotspot.cpp":48:110)
#loc230 = loc("hotspot.cpp":48:127)
#loc231 = loc("hotspot.cpp":48:29)
#loc232 = loc("hotspot.cpp":50:27)
#loc233 = loc("hotspot.cpp":50:13)
#loc234 = loc("hotspot.cpp":50:39)
#loc235 = loc("hotspot.cpp":50:53)
#loc236 = loc("hotspot.cpp":50:56)
#loc237 = loc("hotspot.cpp":50:31)
#loc238 = loc("hotspot.cpp":50:29)
#loc239 = loc("hotspot.cpp":52:22)
#loc240 = loc("hotspot.cpp":52:36)
#loc241 = loc("hotspot.cpp":52:39)
#loc242 = loc("hotspot.cpp":52:13)
#loc243 = loc("hotspot.cpp":52:74)
#loc244 = loc("hotspot.cpp":52:64)
#loc245 = loc("hotspot.cpp":52:77)
#loc246 = loc("hotspot.cpp":52:91)
#loc247 = loc("hotspot.cpp":52:106)
#loc248 = loc("hotspot.cpp":52:122)
#loc249 = loc("hotspot.cpp":52:138)
#loc250 = loc("hotspot.cpp":52:43)
#loc251 = loc("hotspot.cpp":52:41)
#loc252 = loc("hotspot.cpp":39:43)
#loc253 = loc("hotspot.cpp":55:1)
#loc254 = loc("hotspot.cpp":55:27)
#loc255 = loc("hotspot.cpp":55:13)
#loc256 = loc("hotspot.cpp":56:1)
#loc258 = loc("hotspot.cpp":56:31)
#loc259 = loc("hotspot.cpp":56:17)
#loc260 = loc("hotspot.cpp":57:26)
#loc261 = loc("hotspot.cpp":57:17)
#loc262 = loc("hotspot.cpp":57:29)
#loc263 = loc("hotspot.cpp":57:46)
#loc264 = loc("hotspot.cpp":57:49)
#loc265 = loc("hotspot.cpp":57:33)
#loc266 = loc("hotspot.cpp":57:31)
#loc267 = loc("hotspot.cpp":56:63)
#loc268 = loc("hotspot.cpp":60:22)
#loc269 = loc("hotspot.cpp":60:13)
#loc270 = loc("hotspot.cpp":60:79)
#loc271 = loc("hotspot.cpp":60:84)
#loc272 = loc("hotspot.cpp":60:74)
#loc273 = loc("hotspot.cpp":60:98)
#loc274 = loc("hotspot.cpp":60:101)
#loc275 = loc("hotspot.cpp":60:55)
#loc276 = loc("hotspot.cpp":60:53)
#loc277 = loc("hotspot.cpp":55:43)
#loc278 = loc("hotspot.cpp":38:65)
#loc279 = loc("hotspot.cpp":65:1)
#loc281 = loc("./hotspot.h":40:18)
#loc282 = loc("hotspot.cpp":7:1)
#loc283 = loc("hotspot.cpp":12:2)
#loc284 = loc("hotspot.cpp":9:36)
#loc285 = loc("hotspot.cpp":10:31)
#loc286 = loc("hotspot.cpp":10:52)
#loc287 = loc("hotspot.cpp":11:32)
#loc288 = loc("hotspot.cpp":11:52)
#loc289 = loc("hotspot.cpp":12:31)
#loc290 = loc("hotspot.cpp":13:31)
#loc291 = loc("hotspot.cpp":13:15)
#loc292 = loc("hotspot.cpp":14:31)
#loc293 = loc("hotspot.cpp":14:15)
#loc294 = loc("hotspot.cpp":15:31)
#loc295 = loc("hotspot.cpp":15:15)
#loc296 = loc("hotspot.cpp":16:30)
#loc297 = loc("hotspot.cpp":16:14)
#loc298 = loc("hotspot.cpp":18:1)
