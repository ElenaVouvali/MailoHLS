#loc = loc(unknown)
#loc1 = loc("hotspot.cpp":30:6)
#loc10 = loc("hotspot.cpp":66:38)
#loc11 = loc("hotspot.cpp":64:32)
#loc88 = loc("hotspot.cpp":75:38)
#loc122 = loc("hotspot.cpp":5:6)
#loc131 = loc("hotspot.cpp":16:13)
#loc133 = loc("hotspot.cpp":10:17)
#loc140 = loc("hotspot.cpp":15:13)
#loc141 = loc("hotspot.cpp":14:13)
#loc142 = loc("hotspot.cpp":13:13)
#loc143 = loc("hotspot.cpp":7:5)
#loc215 = loc("hotspot.cpp":9:13)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<262144xf32> loc("hotspot.cpp":30:6), %arg1: memref<262144xf32> loc("hotspot.cpp":30:6), %arg2: memref<262144xf32> loc("hotspot.cpp":30:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %alloca = memref.alloca() : memref<32768xf32> loc(#loc19)
    %alloca_8 = memref.alloca() : memref<33792xf32> loc(#loc20)
    %alloca_9 = memref.alloca() : memref<32768xf32> loc(#loc21)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc22)
    %1 = scf.if %true -> (f32) {
      %13 = scf.execute_region -> f32 {
        %14 = scf.if %true -> (f32) {
          %15 = scf.execute_region -> f32 {
            %16 = arith.truncf %cst : f64 to f32 loc(#loc23)
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
            %16 = arith.truncf %cst : f64 to f32 loc(#loc24)
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
            %16 = arith.extf %2 : f32 to f64 loc(#loc25)
            %17 = arith.mulf %16, %cst_0 : f64 loc(#loc26)
            %18 = arith.extf %1 : f32 to f64 loc(#loc27)
            %19 = arith.mulf %17, %18 : f64 loc(#loc28)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc29)
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
            %16 = arith.extf %2 : f32 to f64 loc(#loc30)
            %17 = arith.extf %1 : f32 to f64 loc(#loc31)
            %18 = arith.mulf %17, %cst_1 : f64 loc(#loc32)
            %19 = arith.divf %16, %18 : f64 loc(#loc33)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc30)
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
            %16 = arith.extf %1 : f32 to f64 loc(#loc34)
            %17 = arith.extf %2 : f32 to f64 loc(#loc35)
            %18 = arith.mulf %17, %cst_1 : f64 loc(#loc36)
            %19 = arith.divf %16, %18 : f64 loc(#loc37)
            %20 = arith.truncf %19 : f64 to f32 loc(#loc34)
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
            %16 = arith.mulf %1, %cst_6 : f32 loc(#loc38)
            %17 = arith.mulf %16, %2 : f32 loc(#loc39)
            %18 = arith.extf %17 : f32 to f64 loc(#loc40)
            %19 = arith.divf %cst_7, %18 : f64 loc(#loc41)
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
            %16 = arith.truncf %cst_2 : f64 to f32 loc(#loc42)
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
            %16 = arith.extf %7 : f32 to f64 loc(#loc43)
            %17 = arith.divf %cst_5, %16 : f64 loc(#loc44)
            %18 = arith.divf %17, %cst_4 : f64 loc(#loc45)
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
            %16 = arith.divf %cst_3, %4 : f32 loc(#loc46)
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
            %16 = arith.divf %cst_3, %5 : f32 loc(#loc47)
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
            %16 = arith.divf %cst_3, %6 : f32 loc(#loc48)
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
            %16 = arith.divf %8, %3 : f32 loc(#loc49)
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
        cf.br ^bb1 loc(#loc50)
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
        cf.br ^bb1 loc(#loc51)
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
        cf.br ^bb1 loc(#loc52)
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
        cf.br ^bb1 loc(#loc53)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %13 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
              %14 = arith.cmpi slt, %arg3, %c32_i32 : i32 loc(#loc54)
              scf.condition(%14) %arg3 : i32 loc(#loc55)
            } do {
            ^bb0(%arg3: i32 loc("hotspot.cpp":64:32)):
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
                  cf.br ^bb1 loc(#loc56)
                ^bb1:  // pred: ^bb0
                  scf.if %true {
                    scf.execute_region {
                      %15 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
                        %16 = arith.cmpi slt, %arg4, %c8_i32 : i32 loc(#loc57)
                        scf.condition(%16) %arg4 : i32 loc(#loc58)
                      } do {
                      ^bb0(%arg4: i32 loc("hotspot.cpp":66:38)):
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
                      ^bb0(%arg4: i32 loc("hotspot.cpp":75:38)):
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
  func.func @hotspot(%arg0: memref<32768xf32> loc("hotspot.cpp":5:6), %arg1: memref<33792xf32> loc("hotspot.cpp":5:6), %arg2: memref<32768xf32> loc("hotspot.cpp":5:6), %arg3: f32 loc("hotspot.cpp":5:6), %arg4: f32 loc("hotspot.cpp":5:6), %arg5: f32 loc("hotspot.cpp":5:6), %arg6: f32 loc("hotspot.cpp":5:6), %arg7: i32 loc("hotspot.cpp":5:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc123)
    %c7_i32 = arith.constant 7 : i32 loc(#loc124)
    %cst = arith.constant 2.000000e+00 : f32 loc(#loc125)
    %c63_i32 = arith.constant 63 : i32 loc(#loc126)
    %c511_i32 = arith.constant 511 : i32 loc(#loc127)
    %c1_i32 = arith.constant 1 : i32 loc(#loc123)
    %c512_i32 = arith.constant 512 : i32 loc(#loc8)
    %false = arith.constant false loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc7)
    %c0_i32 = arith.constant 0 : i32 loc(#loc128)
    %cst_0 = arith.constant 8.000000e+01 : f32 loc(#loc129)
    %true = arith.constant true loc(#loc130)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc131)
    %alloca = memref.alloca() : memref<1xf32> loc(#loc132)
    %cast = memref.cast %alloca : memref<1xf32> to memref<?xf32> loc(#loc132)
    affine.store %0, %alloca[0] : memref<1xf32> loc(#loc132)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc133)
    %2 = scf.if %true -> (f32) {
      %3 = scf.execute_region -> f32 {
        %4 = scf.if %true -> (f32) {
          %5 = scf.execute_region -> f32 {
            scf.yield %cst_0 : f32 loc(#loc)
          } loc(#loc)
          scf.yield %5 : f32 loc(#loc)
        } else {
          scf.yield %0 : f32 loc(#loc)
        } loc(#loc)
        scf.yield %4 : f32 loc(#loc)
      } loc(#loc)
      scf.yield %3 : f32 loc(#loc)
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
        cf.br ^bb1 loc(#loc134)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %3 = scf.if %true -> (i32) {
              %5 = scf.execute_region -> i32 {
                scf.yield %c0_i32 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %5 : i32 loc(#loc)
            } else {
              scf.yield %1 : i32 loc(#loc)
            } loc(#loc)
            %4:7 = scf.while (%arg8 = %0, %arg9 = %0, %arg10 = %0, %arg11 = %0, %arg12 = %1, %arg13 = %3, %arg14 = %0) : (f32, f32, f32, f32, i32, i32, f32) -> (f32, f32, f32, f32, i32, i32, f32) {
              %5:8 = scf.execute_region -> (i1, f32, f32, f32, f32, i32, i32, f32) {
                %6 = arith.cmpi slt, %arg13, %c64_i32 : i32 loc(#loc135)
                cf.cond_br %6, ^bb1, ^bb3(%false, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14 : i1, f32, f32, f32, f32, i32, i32, f32) loc(#loc136)
              ^bb1:  // pred: ^bb0
                cf.br ^bb2 loc(#loc137)
              ^bb2:  // pred: ^bb1
                %7:6 = scf.if %true -> (f32, f32, f32, f32, i32, f32) {
                  %17:6 = scf.execute_region -> (f32, f32, f32, f32, i32, f32) {
                    %18 = scf.if %true -> (i32) {
                      %20 = scf.execute_region -> i32 {
                        scf.yield %c0_i32 : i32 loc(#loc)
                      } loc(#loc)
                      scf.yield %20 : i32 loc(#loc)
                    } else {
                      scf.yield %arg12 : i32 loc(#loc)
                    } loc(#loc)
                    %19:6 = scf.while (%arg15 = %arg8, %arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11, %arg19 = %18, %arg20 = %arg14) : (f32, f32, f32, f32, i32, f32) -> (f32, f32, f32, f32, i32, f32) {
                      %20 = arith.cmpi slt, %arg19, %c512_i32 : i32 loc(#loc138)
                      scf.condition(%20) %arg15, %arg16, %arg17, %arg18, %arg19, %arg20 : f32, f32, f32, f32, i32, f32 loc(#loc139)
                    } do {
                    ^bb0(%arg15: f32 loc("hotspot.cpp":16:13), %arg16: f32 loc("hotspot.cpp":15:13), %arg17: f32 loc("hotspot.cpp":14:13), %arg18: f32 loc("hotspot.cpp":13:13), %arg19: i32 loc("hotspot.cpp":10:17), %arg20: f32 loc("hotspot.cpp":7:5)):
                      scf.if %true {
                        scf.execute_region {
                          scf.if %true {
                            scf.execute_region {
                              %26 = arith.muli %arg13, %c512_i32 : i32 loc(#loc144)
                              %27 = arith.addi %26, %c512_i32 : i32 loc(#loc145)
                              %28 = arith.addi %27, %arg19 : i32 loc(#loc146)
                              %29 = arith.index_cast %28 : i32 to index loc(#loc147)
                              %30 = "polygeist.subindex"(%arg1, %29) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc148)
                              %31 = affine.load %30[0] : memref<?xf32> loc(#loc148)
                              affine.store %31, %alloca[0] : memref<1xf32> loc(#loc132)
                              scf.yield loc(#loc)
                            } loc(#loc)
                          } loc(#loc)
                          scf.yield loc(#loc)
                        } loc(#loc)
                      } loc(#loc)
                      %20 = scf.if %true -> (f32) {
                        %26 = scf.execute_region -> f32 {
                          %27 = scf.if %true -> (f32) {
                            %28 = scf.execute_region -> f32 {
                              %29 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc149)
                              %30 = scf.if %29 -> (i1) {
                                %33 = arith.cmpi eq, %arg13, %c0_i32 : i32 loc(#loc151)
                                scf.yield %33 : i1 loc(#loc150)
                              } else {
                                scf.yield %false : i1 loc(#loc150)
                              } loc(#loc150)
                              %31 = scf.if %30 -> (memref<?xf32>) {
                                scf.yield %cast : memref<?xf32> loc(#loc152)
                              } else {
                                %33 = arith.addi %arg13, %c-1_i32 : i32 loc(#loc153)
                                %34 = arith.muli %33, %c512_i32 : i32 loc(#loc154)
                                %35 = arith.addi %34, %c512_i32 : i32 loc(#loc155)
                                %36 = arith.addi %35, %arg19 : i32 loc(#loc156)
                                %37 = arith.index_cast %36 : i32 to index loc(#loc157)
                                %38 = "polygeist.subindex"(%arg1, %37) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc158)
                                scf.yield %38 : memref<?xf32> loc(#loc152)
                              } loc(#loc152)
                              %32 = affine.load %31[0] : memref<?xf32> loc(#loc152)
                              scf.yield %32 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %28 : f32 loc(#loc)
                          } else {
                            scf.yield %arg18 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %27 : f32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : f32 loc(#loc)
                      } else {
                        scf.yield %arg18 : f32 loc(#loc)
                      } loc(#loc)
                      %21 = scf.if %true -> (f32) {
                        %26 = scf.execute_region -> f32 {
                          %27 = scf.if %true -> (f32) {
                            %28 = scf.execute_region -> f32 {
                              %29 = arith.cmpi eq, %arg19, %c511_i32 : i32 loc(#loc159)
                              %30 = scf.if %29 -> (memref<?xf32>) {
                                scf.yield %cast : memref<?xf32> loc(#loc160)
                              } else {
                                %32 = arith.muli %arg13, %c512_i32 : i32 loc(#loc161)
                                %33 = arith.addi %32, %c512_i32 : i32 loc(#loc162)
                                %34 = arith.addi %33, %arg19 : i32 loc(#loc163)
                                %35 = arith.addi %34, %c1_i32 : i32 loc(#loc164)
                                %36 = arith.index_cast %35 : i32 to index loc(#loc165)
                                %37 = "polygeist.subindex"(%arg1, %36) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc166)
                                scf.yield %37 : memref<?xf32> loc(#loc160)
                              } loc(#loc160)
                              %31 = affine.load %30[0] : memref<?xf32> loc(#loc160)
                              scf.yield %31 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %28 : f32 loc(#loc)
                          } else {
                            scf.yield %arg17 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %27 : f32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : f32 loc(#loc)
                      } else {
                        scf.yield %arg17 : f32 loc(#loc)
                      } loc(#loc)
                      %22 = scf.if %true -> (f32) {
                        %26 = scf.execute_region -> f32 {
                          %27 = scf.if %true -> (f32) {
                            %28 = scf.execute_region -> f32 {
                              %29 = arith.cmpi eq, %arg7, %c7_i32 : i32 loc(#loc167)
                              %30 = scf.if %29 -> (i1) {
                                %33 = arith.cmpi eq, %arg13, %c63_i32 : i32 loc(#loc169)
                                scf.yield %33 : i1 loc(#loc168)
                              } else {
                                scf.yield %false : i1 loc(#loc168)
                              } loc(#loc168)
                              %31 = scf.if %30 -> (memref<?xf32>) {
                                scf.yield %cast : memref<?xf32> loc(#loc170)
                              } else {
                                %33 = arith.addi %arg13, %c1_i32 : i32 loc(#loc171)
                                %34 = arith.muli %33, %c512_i32 : i32 loc(#loc172)
                                %35 = arith.addi %34, %c512_i32 : i32 loc(#loc173)
                                %36 = arith.addi %35, %arg19 : i32 loc(#loc174)
                                %37 = arith.index_cast %36 : i32 to index loc(#loc175)
                                %38 = "polygeist.subindex"(%arg1, %37) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc176)
                                scf.yield %38 : memref<?xf32> loc(#loc170)
                              } loc(#loc170)
                              %32 = affine.load %31[0] : memref<?xf32> loc(#loc170)
                              scf.yield %32 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %28 : f32 loc(#loc)
                          } else {
                            scf.yield %arg16 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %27 : f32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : f32 loc(#loc)
                      } else {
                        scf.yield %arg16 : f32 loc(#loc)
                      } loc(#loc)
                      %23 = scf.if %true -> (f32) {
                        %26 = scf.execute_region -> f32 {
                          %27 = scf.if %true -> (f32) {
                            %28 = scf.execute_region -> f32 {
                              %29 = arith.cmpi eq, %arg19, %c0_i32 : i32 loc(#loc177)
                              %30 = scf.if %29 -> (memref<?xf32>) {
                                scf.yield %cast : memref<?xf32> loc(#loc178)
                              } else {
                                %32 = arith.muli %arg13, %c512_i32 : i32 loc(#loc179)
                                %33 = arith.addi %32, %c512_i32 : i32 loc(#loc180)
                                %34 = arith.addi %33, %arg19 : i32 loc(#loc181)
                                %35 = arith.addi %34, %c-1_i32 : i32 loc(#loc182)
                                %36 = arith.index_cast %35 : i32 to index loc(#loc183)
                                %37 = "polygeist.subindex"(%arg1, %36) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc184)
                                scf.yield %37 : memref<?xf32> loc(#loc178)
                              } loc(#loc178)
                              %31 = affine.load %30[0] : memref<?xf32> loc(#loc178)
                              scf.yield %31 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %28 : f32 loc(#loc)
                          } else {
                            scf.yield %arg15 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %27 : f32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : f32 loc(#loc)
                      } else {
                        scf.yield %arg15 : f32 loc(#loc)
                      } loc(#loc)
                      %24 = scf.if %true -> (f32) {
                        %26 = scf.execute_region -> f32 {
                          %27 = arith.muli %arg13, %c512_i32 : i32 loc(#loc185)
                          %28 = arith.addi %27, %arg19 : i32 loc(#loc186)
                          %29 = arith.index_cast %28 : i32 to index loc(#loc187)
                          %30 = "polygeist.subindex"(%arg2, %29) : (memref<32768xf32>, index) -> memref<?xf32> loc(#loc188)
                          %31 = affine.load %30[0] : memref<?xf32> loc(#loc188)
                          %32 = arith.addf %20, %22 : f32 loc(#loc189)
                          %33 = affine.load %alloca[0] : memref<1xf32> loc(#loc190)
                          %34 = arith.mulf %33, %cst : f32 loc(#loc191)
                          %35 = arith.subf %32, %34 : f32 loc(#loc192)
                          %36 = arith.mulf %35, %arg5 : f32 loc(#loc193)
                          %37 = arith.addf %31, %36 : f32 loc(#loc194)
                          %38 = arith.addf %21, %23 : f32 loc(#loc195)
                          %39 = arith.subf %38, %34 : f32 loc(#loc196)
                          %40 = arith.mulf %39, %arg4 : f32 loc(#loc197)
                          %41 = arith.addf %37, %40 : f32 loc(#loc198)
                          %42 = arith.subf %2, %33 : f32 loc(#loc199)
                          %43 = arith.mulf %42, %arg6 : f32 loc(#loc200)
                          %44 = arith.addf %41, %43 : f32 loc(#loc201)
                          %45 = arith.mulf %arg3, %44 : f32 loc(#loc202)
                          scf.yield %45 : f32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : f32 loc(#loc)
                      } else {
                        scf.yield %arg20 : f32 loc(#loc)
                      } loc(#loc)
                      scf.if %true {
                        scf.execute_region {
                          %26 = arith.muli %arg13, %c512_i32 : i32 loc(#loc203)
                          %27 = arith.addi %26, %arg19 : i32 loc(#loc204)
                          %28 = arith.index_cast %27 : i32 to index loc(#loc205)
                          %29 = "polygeist.subindex"(%arg0, %28) : (memref<32768xf32>, index) -> memref<?xf32> loc(#loc206)
                          %30 = arith.addi %26, %c512_i32 : i32 loc(#loc207)
                          %31 = arith.addi %30, %arg19 : i32 loc(#loc208)
                          %32 = arith.index_cast %31 : i32 to index loc(#loc209)
                          %33 = "polygeist.subindex"(%arg1, %32) : (memref<33792xf32>, index) -> memref<?xf32> loc(#loc210)
                          %34 = affine.load %33[0] : memref<?xf32> loc(#loc210)
                          %35 = arith.addf %34, %24 : f32 loc(#loc211)
                          affine.store %35, %29[0] : memref<?xf32> loc(#loc212)
                          scf.yield loc(#loc)
                        } loc(#loc)
                      } loc(#loc)
                      %25 = scf.if %true -> (i32) {
                        %26 = scf.execute_region -> i32 {
                          %27 = arith.addi %arg19, %c1_i32 : i32 loc(#loc213)
                          scf.yield %27 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %26 : i32 loc(#loc)
                      } else {
                        scf.yield %arg19 : i32 loc(#loc)
                      } loc(#loc)
                      scf.yield %23, %22, %21, %20, %25, %24 : f32, f32, f32, f32, i32, f32 loc(#loc139)
                    } loc(#loc8)
                    scf.yield %19#0, %19#1, %19#2, %19#3, %19#4, %19#5 : f32, f32, f32, f32, i32, f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %17#0, %17#1, %17#2, %17#3, %17#4, %17#5 : f32, f32, f32, f32, i32, f32 loc(#loc)
                } else {
                  scf.yield %arg8, %arg9, %arg10, %arg11, %arg12, %arg14 : f32, f32, f32, f32, i32, f32 loc(#loc)
                } loc(#loc)
                %8 = scf.if %true -> (i32) {
                  %17 = scf.execute_region -> i32 {
                    %18 = arith.addi %arg13, %c1_i32 : i32 loc(#loc214)
                    scf.yield %18 : i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %17 : i32 loc(#loc)
                } else {
                  scf.yield %arg13 : i32 loc(#loc)
                } loc(#loc)
                cf.br ^bb3(%true, %7#0, %7#1, %7#2, %7#3, %7#4, %8, %7#5 : i1, f32, f32, f32, f32, i32, i32, f32) loc(#loc136)
              ^bb3(%9: i1 loc(unknown), %10: f32 loc("hotspot.cpp":16:13), %11: f32 loc("hotspot.cpp":15:13), %12: f32 loc("hotspot.cpp":14:13), %13: f32 loc("hotspot.cpp":13:13), %14: i32 loc("hotspot.cpp":10:17), %15: i32 loc("hotspot.cpp":9:13), %16: f32 loc("hotspot.cpp":7:5)):  // 2 preds: ^bb0, ^bb2
                scf.yield %9, %10, %11, %12, %13, %14, %15, %16 : i1, f32, f32, f32, f32, i32, i32, f32 loc(#loc)
              } loc(#loc)
              scf.condition(%5#0) %5#1, %5#2, %5#3, %5#4, %5#5, %5#6, %5#7 : f32, f32, f32, f32, i32, i32, f32 loc(#loc136)
            } do {
            ^bb0(%arg8: f32 loc("hotspot.cpp":16:13), %arg9: f32 loc("hotspot.cpp":15:13), %arg10: f32 loc("hotspot.cpp":14:13), %arg11: f32 loc("hotspot.cpp":13:13), %arg12: i32 loc("hotspot.cpp":10:17), %arg13: i32 loc("hotspot.cpp":9:13), %arg14: f32 loc("hotspot.cpp":7:5)):
              scf.yield %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14 : f32, f32, f32, f32, i32, i32, f32 loc(#loc136)
            } loc(#loc7)
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
    return loc(#loc216)
  } loc(#loc122)
} loc(#loc)
#loc2 = loc("hotspot.cpp":44:35)
#loc3 = loc("hotspot.cpp":46:44)
#loc4 = loc("hotspot.cpp":48:42)
#loc5 = loc("hotspot.cpp":51:30)
#loc6 = loc("hotspot.cpp":66:52)
#loc7 = loc("./hotspot.h":16:19)
#loc8 = loc("./hotspot.h":14:19)
#loc9 = loc("hotspot.cpp":67:94)
#loc12 = loc("hotspot.cpp":64:17)
#loc13 = loc("hotspot.cpp":54:16)
#loc14 = loc("hotspot.cpp":52:42)
#loc15 = loc("./hotspot.h":27:19)
#loc16 = loc("./hotspot.h":29:14)
#loc17 = loc("./hotspot.h":36:16)
#loc18 = loc("hotspot.cpp":30:1)
#loc19 = loc("hotspot.cpp":61:8)
#loc20 = loc("hotspot.cpp":60:8)
#loc21 = loc("hotspot.cpp":59:8)
#loc22 = loc("hotspot.cpp":57:5)
#loc23 = loc("./hotspot.h":37:21)
#loc24 = loc("./hotspot.h":38:20)
#loc25 = loc("hotspot.cpp":46:55)
#loc26 = loc("hotspot.cpp":46:53)
#loc27 = loc("hotspot.cpp":46:68)
#loc28 = loc("hotspot.cpp":46:66)
#loc29 = loc("./hotspot.h":32:21)
#loc30 = loc("hotspot.cpp":47:16)
#loc31 = loc("hotspot.cpp":47:52)
#loc32 = loc("hotspot.cpp":47:50)
#loc33 = loc("hotspot.cpp":47:27)
#loc34 = loc("hotspot.cpp":48:16)
#loc35 = loc("hotspot.cpp":48:53)
#loc36 = loc("hotspot.cpp":48:51)
#loc37 = loc("hotspot.cpp":48:28)
#loc38 = loc("hotspot.cpp":49:31)
#loc39 = loc("hotspot.cpp":49:45)
#loc40 = loc("hotspot.cpp":49:25)
#loc41 = loc("hotspot.cpp":49:23)
#loc42 = loc("./hotspot.h":24:17)
#loc43 = loc("hotspot.cpp":52:30)
#loc44 = loc("hotspot.cpp":52:28)
#loc45 = loc("hotspot.cpp":52:40)
#loc46 = loc("hotspot.cpp":54:20)
#loc47 = loc("hotspot.cpp":55:20)
#loc48 = loc("hotspot.cpp":56:20)
#loc49 = loc("hotspot.cpp":57:24)
#loc50 = loc("hotspot.cpp":59:1)
#loc51 = loc("hotspot.cpp":60:1)
#loc52 = loc("hotspot.cpp":61:1)
#loc53 = loc("hotspot.cpp":64:1)
#loc54 = loc("hotspot.cpp":64:22)
#loc55 = loc("hotspot.cpp":64:8)
#loc56 = loc("hotspot.cpp":66:1)
#loc57 = loc("hotspot.cpp":66:26)
#loc58 = loc("hotspot.cpp":66:12)
#loc59 = loc("hotspot.cpp":67:20)
#loc60 = loc("hotspot.cpp":67:41)
#loc61 = loc("hotspot.cpp":67:53)
#loc62 = loc("hotspot.cpp":67:37)
#loc63 = loc("hotspot.cpp":67:65)
#loc64 = loc("hotspot.cpp":67:78)
#loc65 = loc("hotspot.cpp":67:92)
#loc66 = loc("hotspot.cpp":67:110)
#loc67 = loc("hotspot.cpp":67:13)
#loc68 = loc("hotspot.cpp":68:20)
#loc69 = loc("hotspot.cpp":68:43)
#loc70 = loc("hotspot.cpp":68:55)
#loc71 = loc("hotspot.cpp":68:39)
#loc72 = loc("hotspot.cpp":68:33)
#loc73 = loc("hotspot.cpp":68:68)
#loc74 = loc("hotspot.cpp":68:82)
#loc75 = loc("hotspot.cpp":68:94)
#loc76 = loc("hotspot.cpp":68:13)
#loc77 = loc("hotspot.cpp":70:13)
#loc78 = loc("hotspot.cpp":72:31)
#loc79 = loc("hotspot.cpp":72:43)
#loc80 = loc("hotspot.cpp":72:27)
#loc81 = loc("hotspot.cpp":72:20)
#loc82 = loc("hotspot.cpp":72:56)
#loc83 = loc("hotspot.cpp":72:70)
#loc84 = loc("hotspot.cpp":72:84)
#loc85 = loc("hotspot.cpp":72:96)
#loc86 = loc("hotspot.cpp":72:13)
#loc87 = loc("hotspot.cpp":75:1)
#loc89 = loc("hotspot.cpp":75:26)
#loc90 = loc("hotspot.cpp":75:12)
#loc91 = loc("hotspot.cpp":76:20)
#loc92 = loc("hotspot.cpp":76:43)
#loc93 = loc("hotspot.cpp":76:55)
#loc94 = loc("hotspot.cpp":76:39)
#loc95 = loc("hotspot.cpp":76:67)
#loc96 = loc("hotspot.cpp":76:80)
#loc97 = loc("hotspot.cpp":76:94)
#loc98 = loc("hotspot.cpp":76:112)
#loc99 = loc("hotspot.cpp":76:13)
#loc100 = loc("hotspot.cpp":77:20)
#loc101 = loc("hotspot.cpp":77:43)
#loc102 = loc("hotspot.cpp":77:55)
#loc103 = loc("hotspot.cpp":77:39)
#loc104 = loc("hotspot.cpp":77:33)
#loc105 = loc("hotspot.cpp":77:68)
#loc106 = loc("hotspot.cpp":77:82)
#loc107 = loc("hotspot.cpp":77:94)
#loc108 = loc("hotspot.cpp":77:13)
#loc109 = loc("hotspot.cpp":79:13)
#loc110 = loc("hotspot.cpp":81:29)
#loc111 = loc("hotspot.cpp":81:41)
#loc112 = loc("hotspot.cpp":81:25)
#loc113 = loc("hotspot.cpp":81:20)
#loc114 = loc("hotspot.cpp":81:54)
#loc115 = loc("hotspot.cpp":81:69)
#loc116 = loc("hotspot.cpp":81:83)
#loc117 = loc("hotspot.cpp":81:95)
#loc118 = loc("hotspot.cpp":81:13)
#loc119 = loc("hotspot.cpp":75:52)
#loc120 = loc("hotspot.cpp":64:37)
#loc121 = loc("hotspot.cpp":86:1)
#loc123 = loc("hotspot.cpp":13:94)
#loc124 = loc("./hotspot.h":43:37)
#loc125 = loc("hotspot.cpp":19:37)
#loc126 = loc("hotspot.cpp":15:74)
#loc127 = loc("hotspot.cpp":14:43)
#loc128 = loc("hotspot.cpp":9:21)
#loc129 = loc("hotspot.cpp":6:22)
#loc130 = loc("hotspot.cpp":5:1)
#loc132 = loc("hotspot.cpp":12:13)
#loc134 = loc("hotspot.cpp":9:1)
#loc135 = loc("hotspot.cpp":9:26)
#loc136 = loc("hotspot.cpp":9:8)
#loc137 = loc("hotspot.cpp":10:1)
#loc138 = loc("hotspot.cpp":10:30)
#loc139 = loc("hotspot.cpp":10:12)
#loc144 = loc("hotspot.cpp":12:46)
#loc145 = loc("hotspot.cpp":12:43)
#loc146 = loc("hotspot.cpp":12:57)
#loc147 = loc("hotspot.cpp":12:60)
#loc148 = loc("hotspot.cpp":12:28)
#loc149 = loc("hotspot.cpp":13:42)
#loc150 = loc("hotspot.cpp":13:50)
#loc151 = loc("hotspot.cpp":13:55)
#loc152 = loc("hotspot.cpp":13:25)
#loc153 = loc("hotspot.cpp":13:92)
#loc154 = loc("hotspot.cpp":13:96)
#loc155 = loc("hotspot.cpp":13:87)
#loc156 = loc("hotspot.cpp":13:107)
#loc157 = loc("hotspot.cpp":13:110)
#loc158 = loc("hotspot.cpp":13:72)
#loc159 = loc("hotspot.cpp":14:30)
#loc160 = loc("hotspot.cpp":14:27)
#loc161 = loc("hotspot.cpp":14:77)
#loc162 = loc("hotspot.cpp":14:74)
#loc163 = loc("hotspot.cpp":14:88)
#loc164 = loc("hotspot.cpp":14:92)
#loc165 = loc("hotspot.cpp":14:95)
#loc166 = loc("hotspot.cpp":14:59)
#loc167 = loc("hotspot.cpp":15:45)
#loc168 = loc("hotspot.cpp":15:56)
#loc169 = loc("hotspot.cpp":15:61)
#loc170 = loc("hotspot.cpp":15:28)
#loc171 = loc("hotspot.cpp":15:110)
#loc172 = loc("hotspot.cpp":15:114)
#loc173 = loc("hotspot.cpp":15:105)
#loc174 = loc("hotspot.cpp":15:125)
#loc175 = loc("hotspot.cpp":15:128)
#loc176 = loc("hotspot.cpp":15:90)
#loc177 = loc("hotspot.cpp":16:29)
#loc178 = loc("hotspot.cpp":16:26)
#loc179 = loc("hotspot.cpp":16:64)
#loc180 = loc("hotspot.cpp":16:61)
#loc181 = loc("hotspot.cpp":16:75)
#loc182 = loc("hotspot.cpp":16:79)
#loc183 = loc("hotspot.cpp":16:82)
#loc184 = loc("hotspot.cpp":16:46)
#loc185 = loc("hotspot.cpp":18:38)
#loc186 = loc("hotspot.cpp":18:49)
#loc187 = loc("hotspot.cpp":18:52)
#loc188 = loc("hotspot.cpp":18:31)
#loc189 = loc("hotspot.cpp":19:26)
#loc190 = loc("hotspot.cpp":19:41)
#loc191 = loc("hotspot.cpp":19:40)
#loc192 = loc("hotspot.cpp":19:35)
#loc193 = loc("hotspot.cpp":19:49)
#loc194 = loc("hotspot.cpp":18:54)
#loc195 = loc("hotspot.cpp":20:28)
#loc196 = loc("hotspot.cpp":20:35)
#loc197 = loc("hotspot.cpp":20:49)
#loc198 = loc("hotspot.cpp":19:56)
#loc199 = loc("hotspot.cpp":21:31)
#loc200 = loc("hotspot.cpp":21:41)
#loc201 = loc("hotspot.cpp":20:56)
#loc202 = loc("hotspot.cpp":18:28)
#loc203 = loc("hotspot.cpp":23:21)
#loc204 = loc("hotspot.cpp":23:32)
#loc205 = loc("hotspot.cpp":23:35)
#loc206 = loc("hotspot.cpp":23:13)
#loc207 = loc("hotspot.cpp":23:54)
#loc208 = loc("hotspot.cpp":23:68)
#loc209 = loc("hotspot.cpp":23:71)
#loc210 = loc("hotspot.cpp":23:39)
#loc211 = loc("hotspot.cpp":23:73)
#loc212 = loc("hotspot.cpp":23:37)
#loc213 = loc("hotspot.cpp":10:44)
#loc214 = loc("hotspot.cpp":9:40)
#loc216 = loc("hotspot.cpp":27:1)
