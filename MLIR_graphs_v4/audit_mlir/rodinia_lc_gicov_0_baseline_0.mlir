#loc1 = loc("lc_gicov.cpp":5:6)
#loc5 = loc("./lc_gicov.h":21:17)
#loc6 = loc("./lc_gicov.h":13:19)
#loc25 = loc("lc_gicov.cpp":60:5)
#loc26 = loc("lc_gicov.cpp":36:3)
#loc41 = loc("lc_gicov.cpp":38:29)
#loc43 = loc("lc_gicov.cpp":57:5)
#loc44 = loc("lc_gicov.cpp":55:5)
#loc45 = loc("lc_gicov.cpp":40:4)
#loc47 = loc("lc_gicov.cpp":43:21)
#loc107 = loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":463:3)
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @workload(%arg0: memref<65536xf32> loc("lc_gicov.cpp":5:6), %arg1: memref<65536xf32> loc("lc_gicov.cpp":5:6), %arg2: memref<65536xf32> loc("lc_gicov.cpp":5:6)) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c253_i32 = arith.constant 253 : i32 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc3)
    %cst = arith.constant 1.500000e+01 : f32 loc(#loc4)
    %cst_0 = arith.constant 1.600000e+01 : f32 loc(#loc5)
    %c16_i32 = arith.constant 16 : i32 loc(#loc5)
    %c256_i32 = arith.constant 256 : i32 loc(#loc6)
    %c-2_i32 = arith.constant -2 : i32 loc(#loc7)
    %c2_i32 = arith.constant 2 : i32 loc(#loc8)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc9)
    %c0_i32 = arith.constant 0 : i32 loc(#loc10)
    %c1_i32 = arith.constant 1 : i32 loc(#loc11)
    %cst_1 = arith.constant -1.83697015E-16 : f32 loc(#loc12)
    %cst_2 = arith.constant 6.12323426E-17 : f32 loc(#loc13)
    %cst_3 = arith.constant -1.000000e+00 : f32 loc(#loc14)
    %cst_4 = arith.constant -0.923879504 : f32 loc(#loc15)
    %cst_5 = arith.constant -0.707106769 : f32 loc(#loc16)
    %cst_6 = arith.constant -0.382683426 : f32 loc(#loc17)
    %cst_7 = arith.constant 1.22464685E-16 : f32 loc(#loc18)
    %cst_8 = arith.constant 1.000000e+00 : f32 loc(#loc19)
    %cst_9 = arith.constant 0.923879504 : f32 loc(#loc20)
    %cst_10 = arith.constant 0.707106769 : f32 loc(#loc21)
    %cst_11 = arith.constant 0.382683426 : f32 loc(#loc22)
    %cst_12 = arith.constant 0.000000e+00 : f32 loc(#loc23)
    %true = arith.constant true loc(#loc24)
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
    %c0 = arith.constant 0 : index loc(#loc25)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc25)
    %1 = "polygeist.undef"() : () -> i32 loc(#loc26)
    %alloca = memref.alloca() : memref<16xf32> loc(#loc27)
    %alloca_13 = memref.alloca() : memref<2x16xi32> loc(#loc28)
    %alloca_14 = memref.alloca() : memref<2x16xi32> loc(#loc29)
    %alloca_15 = memref.alloca() : memref<16xf32> loc(#loc30)
    %alloca_16 = memref.alloca() : memref<16xf32> loc(#loc31)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc32)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = "polygeist.subindex"(%alloca_16, %c0) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_12, %2[0] : memref<?xf32> loc(#loc31)
            %3 = "polygeist.subindex"(%alloca_16, %c1) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_11, %3[0] : memref<?xf32> loc(#loc31)
            %4 = "polygeist.subindex"(%alloca_16, %c2) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_10, %4[0] : memref<?xf32> loc(#loc31)
            %5 = "polygeist.subindex"(%alloca_16, %c3) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_9, %5[0] : memref<?xf32> loc(#loc31)
            %6 = "polygeist.subindex"(%alloca_16, %c4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_8, %6[0] : memref<?xf32> loc(#loc31)
            %7 = "polygeist.subindex"(%alloca_16, %c5) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_9, %7[0] : memref<?xf32> loc(#loc31)
            %8 = "polygeist.subindex"(%alloca_16, %c6) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_10, %8[0] : memref<?xf32> loc(#loc31)
            %9 = "polygeist.subindex"(%alloca_16, %c7) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_11, %9[0] : memref<?xf32> loc(#loc31)
            %10 = "polygeist.subindex"(%alloca_16, %c8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_7, %10[0] : memref<?xf32> loc(#loc31)
            %11 = "polygeist.subindex"(%alloca_16, %c9) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_6, %11[0] : memref<?xf32> loc(#loc31)
            %12 = "polygeist.subindex"(%alloca_16, %c10) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_5, %12[0] : memref<?xf32> loc(#loc31)
            %13 = "polygeist.subindex"(%alloca_16, %c11) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_4, %13[0] : memref<?xf32> loc(#loc31)
            %14 = "polygeist.subindex"(%alloca_16, %c12) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_3, %14[0] : memref<?xf32> loc(#loc31)
            %15 = "polygeist.subindex"(%alloca_16, %c13) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_4, %15[0] : memref<?xf32> loc(#loc31)
            %16 = "polygeist.subindex"(%alloca_16, %c14) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_5, %16[0] : memref<?xf32> loc(#loc31)
            %17 = "polygeist.subindex"(%alloca_16, %c15) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc31)
            affine.store %cst_6, %17[0] : memref<?xf32> loc(#loc31)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc33)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = "polygeist.subindex"(%alloca_15, %c0) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_8, %2[0] : memref<?xf32> loc(#loc30)
            %3 = "polygeist.subindex"(%alloca_15, %c1) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_9, %3[0] : memref<?xf32> loc(#loc30)
            %4 = "polygeist.subindex"(%alloca_15, %c2) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_10, %4[0] : memref<?xf32> loc(#loc30)
            %5 = "polygeist.subindex"(%alloca_15, %c3) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_11, %5[0] : memref<?xf32> loc(#loc30)
            %6 = "polygeist.subindex"(%alloca_15, %c4) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_2, %6[0] : memref<?xf32> loc(#loc30)
            %7 = "polygeist.subindex"(%alloca_15, %c5) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_6, %7[0] : memref<?xf32> loc(#loc30)
            %8 = "polygeist.subindex"(%alloca_15, %c6) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_5, %8[0] : memref<?xf32> loc(#loc30)
            %9 = "polygeist.subindex"(%alloca_15, %c7) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_4, %9[0] : memref<?xf32> loc(#loc30)
            %10 = "polygeist.subindex"(%alloca_15, %c8) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_3, %10[0] : memref<?xf32> loc(#loc30)
            %11 = "polygeist.subindex"(%alloca_15, %c9) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_4, %11[0] : memref<?xf32> loc(#loc30)
            %12 = "polygeist.subindex"(%alloca_15, %c10) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_5, %12[0] : memref<?xf32> loc(#loc30)
            %13 = "polygeist.subindex"(%alloca_15, %c11) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_6, %13[0] : memref<?xf32> loc(#loc30)
            %14 = "polygeist.subindex"(%alloca_15, %c12) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_1, %14[0] : memref<?xf32> loc(#loc30)
            %15 = "polygeist.subindex"(%alloca_15, %c13) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_11, %15[0] : memref<?xf32> loc(#loc30)
            %16 = "polygeist.subindex"(%alloca_15, %c14) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_10, %16[0] : memref<?xf32> loc(#loc30)
            %17 = "polygeist.subindex"(%alloca_15, %c15) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc30)
            affine.store %cst_9, %17[0] : memref<?xf32> loc(#loc30)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc34)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = "polygeist.subindex"(%alloca_14, %c0) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc29)
            %3 = "polygeist.subindex"(%2, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc29)
            %4 = "polygeist.subindex"(%3, %c0) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c1_i32, %4[0] : memref<?xi32> loc(#loc29)
            %5 = "polygeist.subindex"(%3, %c1) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %5[0] : memref<?xi32> loc(#loc29)
            %6 = "polygeist.subindex"(%3, %c2) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %6[0] : memref<?xi32> loc(#loc29)
            %7 = "polygeist.subindex"(%3, %c3) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %7[0] : memref<?xi32> loc(#loc29)
            %8 = "polygeist.subindex"(%3, %c4) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %8[0] : memref<?xi32> loc(#loc29)
            %9 = "polygeist.subindex"(%3, %c5) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %9[0] : memref<?xi32> loc(#loc29)
            %10 = "polygeist.subindex"(%3, %c6) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %10[0] : memref<?xi32> loc(#loc29)
            %11 = "polygeist.subindex"(%3, %c7) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %11[0] : memref<?xi32> loc(#loc29)
            %12 = "polygeist.subindex"(%3, %c8) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %12[0] : memref<?xi32> loc(#loc29)
            %13 = "polygeist.subindex"(%3, %c9) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %13[0] : memref<?xi32> loc(#loc29)
            %14 = "polygeist.subindex"(%3, %c10) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %14[0] : memref<?xi32> loc(#loc29)
            %15 = "polygeist.subindex"(%3, %c11) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %15[0] : memref<?xi32> loc(#loc29)
            %16 = "polygeist.subindex"(%3, %c12) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %16[0] : memref<?xi32> loc(#loc29)
            %17 = "polygeist.subindex"(%3, %c13) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %17[0] : memref<?xi32> loc(#loc29)
            %18 = "polygeist.subindex"(%3, %c14) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %18[0] : memref<?xi32> loc(#loc29)
            %19 = "polygeist.subindex"(%3, %c15) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %19[0] : memref<?xi32> loc(#loc29)
            %20 = "polygeist.subindex"(%alloca_14, %c1) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc29)
            %21 = "polygeist.subindex"(%20, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc29)
            %22 = "polygeist.subindex"(%21, %c0) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c2_i32, %22[0] : memref<?xi32> loc(#loc29)
            %23 = "polygeist.subindex"(%21, %c1) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c1_i32, %23[0] : memref<?xi32> loc(#loc29)
            %24 = "polygeist.subindex"(%21, %c2) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c1_i32, %24[0] : memref<?xi32> loc(#loc29)
            %25 = "polygeist.subindex"(%21, %c3) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %25[0] : memref<?xi32> loc(#loc29)
            %26 = "polygeist.subindex"(%21, %c4) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %26[0] : memref<?xi32> loc(#loc29)
            %27 = "polygeist.subindex"(%21, %c5) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %27[0] : memref<?xi32> loc(#loc29)
            %28 = "polygeist.subindex"(%21, %c6) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-2_i32, %28[0] : memref<?xi32> loc(#loc29)
            %29 = "polygeist.subindex"(%21, %c7) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-2_i32, %29[0] : memref<?xi32> loc(#loc29)
            %30 = "polygeist.subindex"(%21, %c8) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-2_i32, %30[0] : memref<?xi32> loc(#loc29)
            %31 = "polygeist.subindex"(%21, %c9) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-2_i32, %31[0] : memref<?xi32> loc(#loc29)
            %32 = "polygeist.subindex"(%21, %c10) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-2_i32, %32[0] : memref<?xi32> loc(#loc29)
            %33 = "polygeist.subindex"(%21, %c11) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c-1_i32, %33[0] : memref<?xi32> loc(#loc29)
            %34 = "polygeist.subindex"(%21, %c12) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %34[0] : memref<?xi32> loc(#loc29)
            %35 = "polygeist.subindex"(%21, %c13) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c0_i32, %35[0] : memref<?xi32> loc(#loc29)
            %36 = "polygeist.subindex"(%21, %c14) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c1_i32, %36[0] : memref<?xi32> loc(#loc29)
            %37 = "polygeist.subindex"(%21, %c15) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc29)
            affine.store %c1_i32, %37[0] : memref<?xi32> loc(#loc29)
            scf.yield loc(#loc)
          } loc(#loc)
        } loc(#loc)
        scf.yield loc(#loc)
      } loc(#loc)
    } loc(#loc)
    scf.if %true {
      scf.execute_region {
        cf.br ^bb1 loc(#loc35)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2 = "polygeist.subindex"(%alloca_13, %c0) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc28)
            %3 = "polygeist.subindex"(%2, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc28)
            %4 = "polygeist.subindex"(%3, %c0) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %4[0] : memref<?xi32> loc(#loc28)
            %5 = "polygeist.subindex"(%3, %c1) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %5[0] : memref<?xi32> loc(#loc28)
            %6 = "polygeist.subindex"(%3, %c2) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %6[0] : memref<?xi32> loc(#loc28)
            %7 = "polygeist.subindex"(%3, %c3) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %7[0] : memref<?xi32> loc(#loc28)
            %8 = "polygeist.subindex"(%3, %c4) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %8[0] : memref<?xi32> loc(#loc28)
            %9 = "polygeist.subindex"(%3, %c5) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %9[0] : memref<?xi32> loc(#loc28)
            %10 = "polygeist.subindex"(%3, %c6) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %10[0] : memref<?xi32> loc(#loc28)
            %11 = "polygeist.subindex"(%3, %c7) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %11[0] : memref<?xi32> loc(#loc28)
            %12 = "polygeist.subindex"(%3, %c8) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %12[0] : memref<?xi32> loc(#loc28)
            %13 = "polygeist.subindex"(%3, %c9) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %13[0] : memref<?xi32> loc(#loc28)
            %14 = "polygeist.subindex"(%3, %c10) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %14[0] : memref<?xi32> loc(#loc28)
            %15 = "polygeist.subindex"(%3, %c11) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %15[0] : memref<?xi32> loc(#loc28)
            %16 = "polygeist.subindex"(%3, %c12) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %16[0] : memref<?xi32> loc(#loc28)
            %17 = "polygeist.subindex"(%3, %c13) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %17[0] : memref<?xi32> loc(#loc28)
            %18 = "polygeist.subindex"(%3, %c14) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %18[0] : memref<?xi32> loc(#loc28)
            %19 = "polygeist.subindex"(%3, %c15) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %19[0] : memref<?xi32> loc(#loc28)
            %20 = "polygeist.subindex"(%alloca_13, %c1) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc28)
            %21 = "polygeist.subindex"(%20, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc28)
            %22 = "polygeist.subindex"(%21, %c0) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %22[0] : memref<?xi32> loc(#loc28)
            %23 = "polygeist.subindex"(%21, %c1) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %23[0] : memref<?xi32> loc(#loc28)
            %24 = "polygeist.subindex"(%21, %c2) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c1_i32, %24[0] : memref<?xi32> loc(#loc28)
            %25 = "polygeist.subindex"(%21, %c3) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c1_i32, %25[0] : memref<?xi32> loc(#loc28)
            %26 = "polygeist.subindex"(%21, %c4) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c1_i32, %26[0] : memref<?xi32> loc(#loc28)
            %27 = "polygeist.subindex"(%21, %c5) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c1_i32, %27[0] : memref<?xi32> loc(#loc28)
            %28 = "polygeist.subindex"(%21, %c6) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c1_i32, %28[0] : memref<?xi32> loc(#loc28)
            %29 = "polygeist.subindex"(%21, %c7) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %29[0] : memref<?xi32> loc(#loc28)
            %30 = "polygeist.subindex"(%21, %c8) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c0_i32, %30[0] : memref<?xi32> loc(#loc28)
            %31 = "polygeist.subindex"(%21, %c9) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %31[0] : memref<?xi32> loc(#loc28)
            %32 = "polygeist.subindex"(%21, %c10) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-2_i32, %32[0] : memref<?xi32> loc(#loc28)
            %33 = "polygeist.subindex"(%21, %c11) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-2_i32, %33[0] : memref<?xi32> loc(#loc28)
            %34 = "polygeist.subindex"(%21, %c12) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-2_i32, %34[0] : memref<?xi32> loc(#loc28)
            %35 = "polygeist.subindex"(%21, %c13) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-2_i32, %35[0] : memref<?xi32> loc(#loc28)
            %36 = "polygeist.subindex"(%21, %c14) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-2_i32, %36[0] : memref<?xi32> loc(#loc28)
            %37 = "polygeist.subindex"(%21, %c15) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc28)
            affine.store %c-1_i32, %37[0] : memref<?xi32> loc(#loc28)
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
        cf.br ^bb1 loc(#loc36)
      ^bb1:  // pred: ^bb0
        scf.if %true {
          scf.execute_region {
            %2:7 = scf.while (%arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %0, %arg7 = %1, %arg8 = %1, %arg9 = %c3_i32) : (f32, f32, f32, f32, i32, i32, i32) -> (f32, f32, f32, f32, i32, i32, i32) {
              %3 = arith.cmpi slt, %arg9, %c253_i32 : i32 loc(#loc37)
              scf.condition(%3) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : f32, f32, f32, f32, i32, i32, i32 loc(#loc38)
            } do {
            ^bb0(%arg3: f32 loc("./lc_gicov.h":13:19), %arg4: f32 loc("./lc_gicov.h":13:19), %arg5: f32 loc("./lc_gicov.h":13:19), %arg6: f32 loc("./lc_gicov.h":13:19), %arg7: i32 loc("./lc_gicov.h":13:19), %arg8: i32 loc("./lc_gicov.h":13:19), %arg9: i32 loc("./lc_gicov.h":13:19)):
              scf.if %true {
                scf.execute_region {
                  cf.br ^bb1 loc(#loc39)
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
              %3:6 = scf.if %true -> (f32, f32, f32, f32, i32, i32) {
                %5:6 = scf.execute_region -> (f32, f32, f32, f32, i32, i32) {
                  cf.br ^bb1 loc(#loc40)
                ^bb1:  // pred: ^bb0
                  %6:6 = scf.if %true -> (f32, f32, f32, f32, i32, i32) {
                    %7:6 = scf.execute_region -> (f32, f32, f32, f32, i32, i32) {
                      %8:7 = scf.while (%arg10 = %arg3, %arg11 = %arg4, %arg12 = %arg5, %arg13 = %arg6, %arg14 = %arg7, %arg15 = %arg8, %arg16 = %c3_i32) : (f32, f32, f32, f32, i32, i32, i32) -> (f32, f32, f32, f32, i32, i32, i32) {
                        %9 = arith.cmpi slt, %arg16, %c253_i32 : i32 loc(#loc41)
                        scf.condition(%9) %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16 : f32, f32, f32, f32, i32, i32, i32 loc(#loc42)
                      } do {
                      ^bb0(%arg10: f32 loc("lc_gicov.cpp":60:5), %arg11: f32 loc("lc_gicov.cpp":57:5), %arg12: f32 loc("lc_gicov.cpp":55:5), %arg13: f32 loc("lc_gicov.cpp":40:4), %arg14: i32 loc("lc_gicov.cpp":36:3), %arg15: i32 loc("lc_gicov.cpp":36:3), %arg16: i32 loc("lc_gicov.cpp":38:29)):
                        %9 = scf.if %true -> (f32) {
                          %12 = scf.execute_region -> f32 {
                            %13 = scf.if %true -> (f32) {
                              %14 = scf.execute_region -> f32 {
                                scf.yield %cst_12 : f32 loc(#loc)
                              } loc(#loc)
                              scf.yield %14 : f32 loc(#loc)
                            } else {
                              scf.yield %arg13 : f32 loc(#loc)
                            } loc(#loc)
                            scf.yield %13 : f32 loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : f32 loc(#loc)
                        } else {
                          scf.yield %arg13 : f32 loc(#loc)
                        } loc(#loc)
                        %10:6 = scf.if %true -> (f32, f32, f32, f32, i32, i32) {
                          %12:6 = scf.execute_region -> (f32, f32, f32, f32, i32, i32) {
                            cf.br ^bb1 loc(#loc46)
                          ^bb1:  // pred: ^bb0
                            %13:6 = scf.if %true -> (f32, f32, f32, f32, i32, i32) {
                              %14:6 = scf.execute_region -> (f32, f32, f32, f32, i32, i32) {
                                %15:7 = scf.while (%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %9, %arg21 = %arg14, %arg22 = %arg15, %arg23 = %c0_i32) : (f32, f32, f32, f32, i32, i32, i32) -> (f32, f32, f32, f32, i32, i32, i32) {
                                  %16 = arith.cmpi slt, %arg23, %c2_i32 : i32 loc(#loc47)
                                  scf.condition(%16) %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23 : f32, f32, f32, f32, i32, i32, i32 loc(#loc48)
                                } do {
                                ^bb0(%arg17: f32 loc("lc_gicov.cpp":60:5), %arg18: f32 loc("lc_gicov.cpp":57:5), %arg19: f32 loc("lc_gicov.cpp":55:5), %arg20: f32 loc("lc_gicov.cpp":40:4), %arg21: i32 loc("lc_gicov.cpp":36:3), %arg22: i32 loc("lc_gicov.cpp":36:3), %arg23: i32 loc("lc_gicov.cpp":43:21)):
                                  %16:2 = scf.if %true -> (i32, i32) {
                                    %25:2 = scf.execute_region -> (i32, i32) {
                                      cf.br ^bb1 loc(#loc49)
                                    ^bb1:  // pred: ^bb0
                                      %26:2 = scf.if %true -> (i32, i32) {
                                        %27:2 = scf.execute_region -> (i32, i32) {
                                          %28:3 = scf.while (%arg24 = %arg21, %arg25 = %arg22, %arg26 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
                                            %29 = arith.cmpi slt, %arg26, %c16_i32 : i32 loc(#loc50)
                                            scf.condition(%29) %arg24, %arg25, %arg26 : i32, i32, i32 loc(#loc51)
                                          } do {
                                          ^bb0(%arg24: i32 loc("lc_gicov.cpp":36:3), %arg25: i32 loc("lc_gicov.cpp":36:3), %arg26: i32 loc("./lc_gicov.h":21:17)):
                                            %29 = scf.if %true -> (i32) {
                                              %32 = scf.execute_region -> i32 {
                                                %33 = arith.index_cast %arg23 : i32 to index loc(#loc52)
                                                %34 = "polygeist.subindex"(%alloca_13, %33) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc53)
                                                %35 = "polygeist.subindex"(%34, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc53)
                                                %36 = arith.index_cast %arg26 : i32 to index loc(#loc54)
                                                %37 = "polygeist.subindex"(%35, %36) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc53)
                                                %38 = affine.load %37[0] : memref<?xi32> loc(#loc53)
                                                %39 = arith.addi %arg16, %38 : i32 loc(#loc55)
                                                scf.yield %39 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg24 : i32 loc(#loc)
                                            } loc(#loc)
                                            %30 = scf.if %true -> (i32) {
                                              %32 = scf.execute_region -> i32 {
                                                %33 = arith.index_cast %arg23 : i32 to index loc(#loc56)
                                                %34 = "polygeist.subindex"(%alloca_14, %33) : (memref<2x16xi32>, index) -> memref<?x16xi32> loc(#loc57)
                                                %35 = "polygeist.subindex"(%34, %c0) : (memref<?x16xi32>, index) -> memref<16xi32> loc(#loc57)
                                                %36 = arith.index_cast %arg26 : i32 to index loc(#loc58)
                                                %37 = "polygeist.subindex"(%35, %36) : (memref<16xi32>, index) -> memref<?xi32> loc(#loc57)
                                                %38 = affine.load %37[0] : memref<?xi32> loc(#loc57)
                                                %39 = arith.addi %arg9, %38 : i32 loc(#loc59)
                                                scf.yield %39 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg25 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.if %true {
                                              scf.execute_region {
                                                %32 = arith.index_cast %arg26 : i32 to index loc(#loc60)
                                                %33 = "polygeist.subindex"(%alloca, %32) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc61)
                                                %34 = arith.muli %29, %c256_i32 : i32 loc(#loc62)
                                                %35 = arith.addi %34, %30 : i32 loc(#loc63)
                                                %36 = arith.index_cast %35 : i32 to index loc(#loc64)
                                                %37 = "polygeist.subindex"(%arg1, %36) : (memref<65536xf32>, index) -> memref<?xf32> loc(#loc65)
                                                %38 = affine.load %37[0] : memref<?xf32> loc(#loc65)
                                                %39 = "polygeist.subindex"(%alloca_15, %32) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc66)
                                                %40 = affine.load %39[0] : memref<?xf32> loc(#loc66)
                                                %41 = arith.mulf %38, %40 : f32 loc(#loc67)
                                                %42 = "polygeist.subindex"(%arg2, %36) : (memref<65536xf32>, index) -> memref<?xf32> loc(#loc68)
                                                %43 = affine.load %42[0] : memref<?xf32> loc(#loc68)
                                                %44 = "polygeist.subindex"(%alloca_16, %32) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc69)
                                                %45 = affine.load %44[0] : memref<?xf32> loc(#loc69)
                                                %46 = arith.mulf %43, %45 : f32 loc(#loc70)
                                                %47 = arith.addf %41, %46 : f32 loc(#loc71)
                                                affine.store %47, %33[0] : memref<?xf32> loc(#loc72)
                                                scf.yield loc(#loc)
                                              } loc(#loc)
                                            } loc(#loc)
                                            %31 = scf.if %true -> (i32) {
                                              %32 = scf.execute_region -> i32 {
                                                %33 = arith.addi %arg26, %c1_i32 : i32 loc(#loc73)
                                                scf.yield %33 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg26 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %29, %30, %31 : i32, i32, i32 loc(#loc51)
                                          } loc(#loc5)
                                          scf.yield %28#0, %28#1 : i32, i32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27#0, %27#1 : i32, i32 loc(#loc)
                                      } else {
                                        scf.yield %arg21, %arg22 : i32, i32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26#0, %26#1 : i32, i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25#0, %25#1 : i32, i32 loc(#loc)
                                  } else {
                                    scf.yield %arg21, %arg22 : i32, i32 loc(#loc)
                                  } loc(#loc)
                                  %17 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          scf.yield %cst_12 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg19 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg19 : f32 loc(#loc)
                                  } loc(#loc)
                                  %18 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      cf.br ^bb1 loc(#loc74)
                                    ^bb1:  // pred: ^bb0
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          %28:2 = scf.while (%arg24 = %17, %arg25 = %c0_i32) : (f32, i32) -> (f32, i32) {
                                            %29 = arith.cmpi slt, %arg25, %c16_i32 : i32 loc(#loc75)
                                            scf.condition(%29) %arg24, %arg25 : f32, i32 loc(#loc76)
                                          } do {
                                          ^bb0(%arg24: f32 loc("lc_gicov.cpp":55:5), %arg25: i32 loc("./lc_gicov.h":21:17)):
                                            %29 = arith.index_cast %arg25 : i32 to index loc(#loc77)
                                            %30 = "polygeist.subindex"(%alloca, %29) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc78)
                                            %31 = affine.load %30[0] : memref<?xf32> loc(#loc78)
                                            %32 = arith.addf %arg24, %31 : f32 loc(#loc79)
                                            %33 = scf.if %true -> (i32) {
                                              %34 = scf.execute_region -> i32 {
                                                %35 = arith.addi %arg25, %c1_i32 : i32 loc(#loc80)
                                                scf.yield %35 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %34 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg25 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %32, %33 : f32, i32 loc(#loc76)
                                          } loc(#loc5)
                                          scf.yield %28#0 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %17 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %17 : f32 loc(#loc)
                                  } loc(#loc)
                                  %19 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          %28 = arith.divf %18, %cst_0 : f32 loc(#loc81)
                                          scf.yield %28 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg18 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg18 : f32 loc(#loc)
                                  } loc(#loc)
                                  %20 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          scf.yield %cst_12 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg17 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg17 : f32 loc(#loc)
                                  } loc(#loc)
                                  %21:2 = scf.if %true -> (f32, f32) {
                                    %25:2 = scf.execute_region -> (f32, f32) {
                                      cf.br ^bb1 loc(#loc82)
                                    ^bb1:  // pred: ^bb0
                                      %26:2 = scf.if %true -> (f32, f32) {
                                        %27:2 = scf.execute_region -> (f32, f32) {
                                          %28:3 = scf.while (%arg24 = %20, %arg25 = %18, %arg26 = %c0_i32) : (f32, f32, i32) -> (f32, f32, i32) {
                                            %29 = arith.cmpi slt, %arg26, %c16_i32 : i32 loc(#loc83)
                                            scf.condition(%29) %arg24, %arg25, %arg26 : f32, f32, i32 loc(#loc84)
                                          } do {
                                          ^bb0(%arg24: f32 loc("lc_gicov.cpp":60:5), %arg25: f32 loc("lc_gicov.cpp":55:5), %arg26: i32 loc("./lc_gicov.h":21:17)):
                                            %29 = scf.if %true -> (f32) {
                                              %32 = scf.execute_region -> f32 {
                                                %33 = arith.index_cast %arg26 : i32 to index loc(#loc85)
                                                %34 = "polygeist.subindex"(%alloca, %33) : (memref<16xf32>, index) -> memref<?xf32> loc(#loc86)
                                                %35 = affine.load %34[0] : memref<?xf32> loc(#loc86)
                                                %36 = arith.subf %35, %19 : f32 loc(#loc87)
                                                scf.yield %36 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg25 : f32 loc(#loc)
                                            } loc(#loc)
                                            %30 = scf.if %true -> (f32) {
                                              %32 = scf.execute_region -> f32 {
                                                %33 = arith.mulf %29, %29 : f32 loc(#loc88)
                                                %34 = arith.addf %arg24, %33 : f32 loc(#loc89)
                                                scf.yield %34 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg24 : f32 loc(#loc)
                                            } loc(#loc)
                                            %31 = scf.if %true -> (i32) {
                                              %32 = scf.execute_region -> i32 {
                                                %33 = arith.addi %arg26, %c1_i32 : i32 loc(#loc90)
                                                scf.yield %33 : i32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %32 : i32 loc(#loc)
                                            } else {
                                              scf.yield %arg26 : i32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %30, %29, %31 : f32, f32, i32 loc(#loc84)
                                          } loc(#loc5)
                                          scf.yield %28#0, %28#1 : f32, f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27#0, %27#1 : f32, f32 loc(#loc)
                                      } else {
                                        scf.yield %20, %18 : f32, f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26#0, %26#1 : f32, f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25#0, %25#1 : f32, f32 loc(#loc)
                                  } else {
                                    scf.yield %20, %18 : f32, f32 loc(#loc)
                                  } loc(#loc)
                                  %22 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = arith.divf %21#0, %cst : f32 loc(#loc91)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %21#0 : f32 loc(#loc)
                                  } loc(#loc)
                                  %23 = scf.if %true -> (f32) {
                                    %25 = scf.execute_region -> f32 {
                                      %26 = scf.if %true -> (f32) {
                                        %27 = scf.execute_region -> f32 {
                                          %28 = arith.mulf %19, %19 : f32 loc(#loc92)
                                          %29 = arith.divf %28, %22 : f32 loc(#loc93)
                                          %30 = arith.cmpf ogt, %29, %arg20 : f32 loc(#loc94)
                                          %31 = scf.if %30 -> (f32) {
                                            scf.if %true {
                                              scf.execute_region {
                                                %33 = arith.muli %arg16, %c256_i32 : i32 loc(#loc96)
                                                %34 = arith.addi %33, %arg9 : i32 loc(#loc97)
                                                %35 = arith.index_cast %34 : i32 to index loc(#loc98)
                                                %36 = "polygeist.subindex"(%arg0, %35) : (memref<65536xf32>, index) -> memref<?xf32> loc(#loc99)
                                                %37 = func.call @_ZSt4sqrtf(%22) : (f32) -> f32 loc(#loc100)
                                                %38 = arith.divf %19, %37 : f32 loc(#loc101)
                                                affine.store %38, %36[0] : memref<?xf32> loc(#loc102)
                                                scf.yield loc(#loc)
                                              } loc(#loc)
                                            } loc(#loc)
                                            %32 = scf.if %true -> (f32) {
                                              %33 = scf.execute_region -> f32 {
                                                scf.yield %29 : f32 loc(#loc)
                                              } loc(#loc)
                                              scf.yield %33 : f32 loc(#loc)
                                            } else {
                                              scf.yield %arg20 : f32 loc(#loc)
                                            } loc(#loc)
                                            scf.yield %32 : f32 loc(#loc95)
                                          } else {
                                            scf.yield %arg20 : f32 loc(#loc95)
                                          } loc(#loc95)
                                          scf.yield %31 : f32 loc(#loc)
                                        } loc(#loc)
                                        scf.yield %27 : f32 loc(#loc)
                                      } else {
                                        scf.yield %arg20 : f32 loc(#loc)
                                      } loc(#loc)
                                      scf.yield %26 : f32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : f32 loc(#loc)
                                  } else {
                                    scf.yield %arg20 : f32 loc(#loc)
                                  } loc(#loc)
                                  %24 = scf.if %true -> (i32) {
                                    %25 = scf.execute_region -> i32 {
                                      %26 = arith.addi %arg23, %c1_i32 : i32 loc(#loc103)
                                      scf.yield %26 : i32 loc(#loc)
                                    } loc(#loc)
                                    scf.yield %25 : i32 loc(#loc)
                                  } else {
                                    scf.yield %arg23 : i32 loc(#loc)
                                  } loc(#loc)
                                  scf.yield %22, %19, %21#1, %23, %16#0, %16#1, %24 : f32, f32, f32, f32, i32, i32, i32 loc(#loc48)
                                } loc(#loc47)
                                scf.yield %15#0, %15#1, %15#2, %15#3, %15#4, %15#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                              } loc(#loc)
                              scf.yield %14#0, %14#1, %14#2, %14#3, %14#4, %14#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                            } else {
                              scf.yield %arg10, %arg11, %arg12, %9, %arg14, %arg15 : f32, f32, f32, f32, i32, i32 loc(#loc)
                            } loc(#loc)
                            scf.yield %13#0, %13#1, %13#2, %13#3, %13#4, %13#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %12#0, %12#1, %12#2, %12#3, %12#4, %12#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                        } else {
                          scf.yield %arg10, %arg11, %arg12, %9, %arg14, %arg15 : f32, f32, f32, f32, i32, i32 loc(#loc)
                        } loc(#loc)
                        %11 = scf.if %true -> (i32) {
                          %12 = scf.execute_region -> i32 {
                            %13 = arith.addi %arg16, %c1_i32 : i32 loc(#loc104)
                            scf.yield %13 : i32 loc(#loc)
                          } loc(#loc)
                          scf.yield %12 : i32 loc(#loc)
                        } else {
                          scf.yield %arg16 : i32 loc(#loc)
                        } loc(#loc)
                        scf.yield %10#0, %10#1, %10#2, %10#3, %10#4, %10#5, %11 : f32, f32, f32, f32, i32, i32, i32 loc(#loc42)
                      } loc(#loc41)
                      scf.yield %8#0, %8#1, %8#2, %8#3, %8#4, %8#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                    } loc(#loc)
                    scf.yield %7#0, %7#1, %7#2, %7#3, %7#4, %7#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                  } else {
                    scf.yield %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : f32, f32, f32, f32, i32, i32 loc(#loc)
                  } loc(#loc)
                  scf.yield %6#0, %6#1, %6#2, %6#3, %6#4, %6#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
                } loc(#loc)
                scf.yield %5#0, %5#1, %5#2, %5#3, %5#4, %5#5 : f32, f32, f32, f32, i32, i32 loc(#loc)
              } else {
                scf.yield %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : f32, f32, f32, f32, i32, i32 loc(#loc)
              } loc(#loc)
              %4 = scf.if %true -> (i32) {
                %5 = scf.execute_region -> i32 {
                  %6 = arith.addi %arg9, %c1_i32 : i32 loc(#loc105)
                  scf.yield %6 : i32 loc(#loc)
                } loc(#loc)
                scf.yield %5 : i32 loc(#loc)
              } else {
                scf.yield %arg9 : i32 loc(#loc)
              } loc(#loc)
              scf.yield %3#0, %3#1, %3#2, %3#3, %3#4, %3#5, %4 : f32, f32, f32, f32, i32, i32, i32 loc(#loc38)
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
    return loc(#loc106)
  } loc(#loc1)
  func.func @_ZSt4sqrtf(%arg0: f32 loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":463:3)) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %true = arith.constant true loc(#loc108)
    %0 = "polygeist.undef"() : () -> f32 loc(#loc109)
    %1 = scf.if %true -> (f32) {
      %2 = scf.execute_region -> f32 {
        %3 = scf.if %true -> (f32) {
          %4 = scf.execute_region -> f32 {
            %5 = math.sqrt %arg0 : f32 loc(#loc110)
            scf.yield %5 : f32 loc(#loc)
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
    return %1 : f32 loc(#loc111)
  } loc(#loc107)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("lc_gicov.cpp":34:40)
#loc3 = loc("./lc_gicov.h":29:32)
#loc4 = loc("lc_gicov.cpp":65:24)
#loc7 = loc("lc_gicov.cpp":22:102)
#loc8 = loc("lc_gicov.cpp":22:83)
#loc9 = loc("lc_gicov.cpp":22:41)
#loc10 = loc("lc_gicov.cpp":22:29)
#loc11 = loc("lc_gicov.cpp":22:26)
#loc12 = loc("lc_gicov.cpp":20:237)
#loc13 = loc("lc_gicov.cpp":20:91)
#loc14 = loc("lc_gicov.cpp":18:233)
#loc15 = loc("lc_gicov.cpp":18:213)
#loc16 = loc("lc_gicov.cpp":18:193)
#loc17 = loc("lc_gicov.cpp":18:173)
#loc18 = loc("lc_gicov.cpp":18:151)
#loc19 = loc("lc_gicov.cpp":18:91)
#loc20 = loc("lc_gicov.cpp":18:72)
#loc21 = loc("lc_gicov.cpp":18:53)
#loc22 = loc("lc_gicov.cpp":18:34)
#loc23 = loc("lc_gicov.cpp":18:31)
#loc24 = loc("lc_gicov.cpp":5:1)
#loc27 = loc("lc_gicov.cpp":35:6)
#loc28 = loc("lc_gicov.cpp":24:8)
#loc29 = loc("lc_gicov.cpp":22:8)
#loc30 = loc("lc_gicov.cpp":20:8)
#loc31 = loc("lc_gicov.cpp":18:8)
#loc32 = loc("lc_gicov.cpp":18:1)
#loc33 = loc("lc_gicov.cpp":20:1)
#loc34 = loc("lc_gicov.cpp":22:1)
#loc35 = loc("lc_gicov.cpp":24:1)
#loc36 = loc("lc_gicov.cpp":34:1)
#loc37 = loc("lc_gicov.cpp":34:28)
#loc38 = loc("lc_gicov.cpp":34:5)
#loc39 = loc("lc_gicov.cpp":35:1)
#loc40 = loc("lc_gicov.cpp":38:1)
#loc42 = loc("lc_gicov.cpp":38:6)
#loc46 = loc("lc_gicov.cpp":43:1)
#loc48 = loc("lc_gicov.cpp":43:7)
#loc49 = loc("lc_gicov.cpp":45:1)
#loc50 = loc("lc_gicov.cpp":45:22)
#loc51 = loc("lc_gicov.cpp":45:8)
#loc52 = loc("lc_gicov.cpp":47:18)
#loc53 = loc("lc_gicov.cpp":47:14)
#loc54 = loc("lc_gicov.cpp":47:21)
#loc55 = loc("lc_gicov.cpp":47:12)
#loc56 = loc("lc_gicov.cpp":48:18)
#loc57 = loc("lc_gicov.cpp":48:14)
#loc58 = loc("lc_gicov.cpp":48:21)
#loc59 = loc("lc_gicov.cpp":48:12)
#loc60 = loc("lc_gicov.cpp":51:12)
#loc61 = loc("lc_gicov.cpp":51:6)
#loc62 = loc("lc_gicov.cpp":51:25)
#loc63 = loc("lc_gicov.cpp":51:37)
#loc64 = loc("lc_gicov.cpp":51:40)
#loc65 = loc("lc_gicov.cpp":51:16)
#loc66 = loc("lc_gicov.cpp":51:44)
#loc67 = loc("lc_gicov.cpp":51:42)
#loc68 = loc("lc_gicov.cpp":51:59)
#loc69 = loc("lc_gicov.cpp":51:87)
#loc70 = loc("lc_gicov.cpp":51:85)
#loc71 = loc("lc_gicov.cpp":51:57)
#loc72 = loc("lc_gicov.cpp":51:14)
#loc73 = loc("lc_gicov.cpp":45:34)
#loc74 = loc("lc_gicov.cpp":56:1)
#loc75 = loc("lc_gicov.cpp":56:23)
#loc76 = loc("lc_gicov.cpp":56:9)
#loc77 = loc("lc_gicov.cpp":56:52)
#loc78 = loc("lc_gicov.cpp":56:46)
#loc79 = loc("lc_gicov.cpp":56:43)
#loc80 = loc("lc_gicov.cpp":56:35)
#loc81 = loc("lc_gicov.cpp":57:22)
#loc82 = loc("lc_gicov.cpp":61:1)
#loc83 = loc("lc_gicov.cpp":61:23)
#loc84 = loc("lc_gicov.cpp":61:9)
#loc85 = loc("lc_gicov.cpp":62:18)
#loc86 = loc("lc_gicov.cpp":62:12)
#loc87 = loc("lc_gicov.cpp":62:20)
#loc88 = loc("lc_gicov.cpp":63:17)
#loc89 = loc("lc_gicov.cpp":63:10)
#loc90 = loc("lc_gicov.cpp":61:35)
#loc91 = loc("lc_gicov.cpp":65:15)
#loc92 = loc("lc_gicov.cpp":68:14)
#loc93 = loc("lc_gicov.cpp":68:21)
#loc94 = loc("lc_gicov.cpp":68:27)
#loc95 = loc("lc_gicov.cpp":68:5)
#loc96 = loc("lc_gicov.cpp":69:15)
#loc97 = loc("lc_gicov.cpp":69:27)
#loc98 = loc("lc_gicov.cpp":69:30)
#loc99 = loc("lc_gicov.cpp":69:6)
#loc100 = loc("lc_gicov.cpp":69:41)
#loc101 = loc("lc_gicov.cpp":69:39)
#loc102 = loc("lc_gicov.cpp":69:32)
#loc103 = loc("lc_gicov.cpp":43:34)
#loc104 = loc("lc_gicov.cpp":38:56)
#loc105 = loc("lc_gicov.cpp":34:55)
#loc106 = loc("lc_gicov.cpp":77:1)
#loc108 = loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":462:3)
#loc109 = loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":463:8)
#loc110 = loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":464:12)
#loc111 = loc("/usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/cmath":464:34)
