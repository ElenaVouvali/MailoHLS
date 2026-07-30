#loc1 = loc("backprop_kernel.cpp":7:6)
#loc3 = loc("backprop_kernel.cpp":25:28)
#loc5 = loc("backprop_kernel.cpp":24:24)
#loc10 = loc("backprop_kernel.cpp":19:5)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<17xf32> loc("backprop_kernel.cpp":7:6), %arg1: memref<65537xf32> loc("backprop_kernel.cpp":7:6), %arg2: memref<1114129xf32> loc("backprop_kernel.cpp":7:6), %arg3: memref<1114129xf32> loc("backprop_kernel.cpp":7:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.000000e-01 : f64 loc(#loc2)
    %c65537_i32 = arith.constant 65537 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %c17_i32 = arith.constant 17 : i32 loc(#loc5)
    %c1_i32 = arith.constant 1 : i32 loc(#loc6)
    %cst_0 = arith.constant 1.000000e+00 : f32 loc(#loc7)
    %true = arith.constant true loc(#loc8)
    %c0 = arith.constant 0 : index loc(#loc9)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc10)
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
        %1 = "polygeist.subindex"(%arg1, %c0) : (memref<65537xf32>, index) -> memref<?xf32> loc(#loc11)
        affine.store %cst_0, %1[0] : memref<?xf32> loc(#loc12)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc13)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %1:2 = scf.while (%arg4 = %c1_i32, %arg5 = %0) : (i32, f32) -> (i32, f32) {
              %2 = arith.cmpi slt, %arg4, %c17_i32 : i32 loc(#loc14)
              scf.condition(%2) %arg4, %arg5 : i32, f32 loc(#loc15)
            } do {
            ^bb0(%arg4: i32 loc("backprop_kernel.cpp":24:24), %arg5: f32 loc("backprop_kernel.cpp":24:24)):
              %2 = scf.if %true -> (f32) {
                %4 = scf.execute_region -> f32 {
                  cf.br ^bb1 loc(#loc16)
                ^bb1:  // pred: ^bb0
                  %5 = scf.if %true -> (f32) {
                    %6 = scf.execute_region -> f32 {
                      %7:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg5) : (i32, f32) -> (f32, i32) {
                        %8 = arith.cmpi slt, %arg6, %c65537_i32 : i32 loc(#loc17)
                        scf.condition(%8) %arg7, %arg6 : f32, i32 loc(#loc18)
                      } do {
                      ^bb0(%arg6: f32 loc("backprop_kernel.cpp":19:5), %arg7: i32 loc("backprop_kernel.cpp":25:28)):
                        %8 = scf.if %true -> (f32) {
                          %10 = scf.execute_region -> f32 {
                            %11 = arith.index_cast %arg4 : i32 to index loc(#loc19)
                            %12 = "polygeist.subindex"(%arg0, %11) : (memref<17xf32>, index) -> memref<?xf32> loc(#loc20)
                            %13 = affine.load %12[0] : memref<?xf32> loc(#loc20)
                            %14 = arith.extf %13 : f32 to f64 loc(#loc20)
                            %15 = arith.mulf %14, %cst : f64 loc(#loc21)
                            %16 = arith.index_cast %arg7 : i32 to index loc(#loc22)
                            %17 = "polygeist.subindex"(%arg1, %16) : (memref<65537xf32>, index) -> memref<?xf32> loc(#loc23)
                            %18 = affine.load %17[0] : memref<?xf32> loc(#loc23)
                            %19 = arith.extf %18 : f32 to f64 loc(#loc23)
                            %20 = arith.mulf %15, %19 : f64 loc(#loc24)
                            %21 = arith.muli %arg7, %c17_i32 : i32 loc(#loc25)
                            %22 = arith.addi %21, %arg4 : i32 loc(#loc26)
                            %23 = arith.index_cast %22 : i32 to index loc(#loc27)
                            %24 = "polygeist.subindex"(%arg3, %23) : (memref<1114129xf32>, index) -> memref<?xf32> loc(#loc28)
                            %25 = affine.load %24[0] : memref<?xf32> loc(#loc28)
                            %26 = arith.extf %25 : f32 to f64 loc(#loc28)
                            %27 = arith.mulf %26, %cst : f64 loc(#loc29)
                            %28 = arith.addf %20, %27 : f64 loc(#loc30)
                            %29 = arith.truncf %28 : f64 to f32 loc(#loc31)
                            scf.yield %29 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %10 : f32 loc(#loc)
                        } else {
                          scf.yield %arg6 : f32 loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.muli %arg7, %c17_i32 : i32 loc(#loc32)
                            %11 = arith.addi %10, %arg4 : i32 loc(#loc33)
                            %12 = arith.index_cast %11 : i32 to index loc(#loc34)
                            %13 = "polygeist.subindex"(%arg2, %12) : (memref<1114129xf32>, index) -> memref<?xf32> loc(#loc35)
                            %14 = affine.load %13[0] : memref<?xf32> loc(#loc36)
                            %15 = arith.addf %14, %8 : f32 loc(#loc36)
                            affine.store %15, %13[0] : memref<?xf32> loc(#loc36)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        scf.if %true {
                          scf.execute_region {
                            %10 = arith.muli %arg7, %c17_i32 : i32 loc(#loc37)
                            %11 = arith.addi %10, %arg4 : i32 loc(#loc38)
                            %12 = arith.index_cast %11 : i32 to index loc(#loc39)
                            %13 = "polygeist.subindex"(%arg3, %12) : (memref<1114129xf32>, index) -> memref<?xf32> loc(#loc40)
                            affine.store %8, %13[0] : memref<?xf32> loc(#loc41)
                            scf.yield loc(#loc)
                          } loc(#loc)
                        } loc(#loc)
                        %9 = scf.if %true -> (i32) {
                          %10 = scf.execute_region -> i32 {
                            %11 = arith.addi %arg7, %c1_i32 : i32 loc(#loc42)
                            scf.yield %11 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %10 : i32 loc(#loc)
                        } else {
                          scf.yield %arg7 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %9, %8 : i32, f32 loc(#loc18)
                      } loc(#loc3)
                      scf.yield %7#0 : f32 loc(#loc)
                    } loc(#loc)
                    scf.yield %6 : f32 loc(#loc)
                  } else {
                    scf.yield %arg5 : f32 loc(#loc)
                  } loc(#loc)
                  scf.yield %5 : f32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : f32 loc(#loc)
              } else {
                scf.yield %arg5 : f32 loc(#loc)
              } loc(#loc)
              %3 = scf.if %true -> (i32) {
                %4 = scf.execute_region -> i32 {
                  %5 = arith.addi %arg4, %c1_i32 : i32 loc(#loc43)
                  scf.yield %5 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %4 : i32 loc(#loc)
              } else {
                scf.yield %arg4 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3, %2 : i32, f32 loc(#loc15)
            } loc(#loc5)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    return loc(#loc44)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("backprop_kernel.cpp":3:13)
#loc4 = loc("backprop_kernel.cpp":25:21)
#loc6 = loc("backprop_kernel.cpp":24:17)
#loc7 = loc("backprop_kernel.cpp":22:13)
#loc8 = loc("backprop_kernel.cpp":7:1)
#loc9 = loc("backprop_kernel.cpp":20:5)
#loc11 = loc("backprop_kernel.cpp":22:5)
#loc12 = loc("backprop_kernel.cpp":22:11)
#loc13 = loc("backprop_kernel.cpp":24:1)
#loc14 = loc("backprop_kernel.cpp":24:22)
#loc15 = loc("backprop_kernel.cpp":24:8)
#loc16 = loc("backprop_kernel.cpp":25:1)
#loc17 = loc("backprop_kernel.cpp":25:26)
#loc18 = loc("backprop_kernel.cpp":25:12)
#loc19 = loc("backprop_kernel.cpp":26:37)
#loc20 = loc("backprop_kernel.cpp":26:30)
#loc21 = loc("backprop_kernel.cpp":26:28)
#loc22 = loc("backprop_kernel.cpp":26:45)
#loc23 = loc("backprop_kernel.cpp":26:41)
#loc24 = loc("backprop_kernel.cpp":26:39)
#loc25 = loc("backprop_kernel.cpp":26:69)
#loc26 = loc("backprop_kernel.cpp":26:74)
#loc27 = loc("backprop_kernel.cpp":26:77)
#loc28 = loc("backprop_kernel.cpp":26:62)
#loc29 = loc("backprop_kernel.cpp":26:60)
#loc30 = loc("backprop_kernel.cpp":26:48)
#loc31 = loc("backprop_kernel.cpp":26:22)
#loc32 = loc("backprop_kernel.cpp":27:17)
#loc33 = loc("backprop_kernel.cpp":27:22)
#loc34 = loc("backprop_kernel.cpp":27:25)
#loc35 = loc("backprop_kernel.cpp":27:13)
#loc36 = loc("backprop_kernel.cpp":27:27)
#loc37 = loc("backprop_kernel.cpp":28:20)
#loc38 = loc("backprop_kernel.cpp":28:25)
#loc39 = loc("backprop_kernel.cpp":28:28)
#loc40 = loc("backprop_kernel.cpp":28:13)
#loc41 = loc("backprop_kernel.cpp":28:30)
#loc42 = loc("backprop_kernel.cpp":25:36)
#loc43 = loc("backprop_kernel.cpp":24:29)
#loc44 = loc("backprop_kernel.cpp":31:1)
