; ModuleID = 'serrano-kalman-filter.cpp'
source_filename = "serrano-kalman-filter.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ap_uint = type { i64 }
%union.raw_float = type { i32 }
%"struct.ap_uint<256>::range_proxy" = type { i64* }

$_ZN7ap_uintILi256EEC2Em = comdat any

$_ZN7ap_uintILi256EE5rangeEii = comdat any

$_ZNK7ap_uintILi256EE11range_proxycvmEv = comdat any

$_ZN7ap_uintILi256EEaSEm = comdat any

$_ZN7ap_uintILi256EE11range_proxyaSEm = comdat any

$_ZN7ap_uintILi256EE11range_proxyC2ERm = comdat any

; Function Attrs: noinline optnone uwtable
define dso_local void @krnl_KALMAN(%struct.ap_uint* %in, %struct.ap_uint* %out) #0 {
entry:
  %in.addr = alloca %struct.ap_uint*, align 8
  %out.addr = alloca %struct.ap_uint*, align 8
  %in_local = alloca [262144 x float], align 16
  %out_local = alloca [262144 x float], align 16
  %temp = alloca %struct.ap_uint, align 8
  %counter = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %t = alloca %union.raw_float, align 4
  %ref.tmp = alloca %"struct.ap_uint<256>::range_proxy", align 8
  %u_hat_arr = alloca [64 x float], align 16
  %p_arr_1 = alloca [64 x float], align 16
  %k_arr = alloca [64 x float], align 16
  %calc_temp_arr = alloca [64 x float], align 16
  %i18 = alloca i32, align 4
  %t37 = alloca i32, align 4
  %i41 = alloca i32, align 4
  %i92 = alloca i32, align 4
  %j98 = alloca i32, align 4
  %t103 = alloca %union.raw_float, align 4
  %ref.tmp109 = alloca %"struct.ap_uint<256>::range_proxy", align 8
  store %struct.ap_uint* %in, %struct.ap_uint** %in.addr, align 8
  store %struct.ap_uint* %out, %struct.ap_uint** %out.addr, align 8
  call void @_ZN7ap_uintILi256EEC2Em(%struct.ap_uint* %temp, i64 0)
  store i32 0, i32* %counter, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc15, %entry
  %0 = load i32, i32* %i, align 4
  %conv = sext i32 %0 to i64
  %cmp = icmp ult i64 %conv, 32768
  br i1 %cmp, label %for.body, label %for.end17

for.body:                                         ; preds = %for.cond
  %1 = load %struct.ap_uint*, %struct.ap_uint** %in.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds %struct.ap_uint, %struct.ap_uint* %1, i64 %idxprom
  %3 = bitcast %struct.ap_uint* %temp to i8*
  %4 = bitcast %struct.ap_uint* %arrayidx to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %3, i8* align 8 %4, i64 8, i1 false)
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %5 = load i32, i32* %j, align 4
  %conv2 = sext i32 %5 to i64
  %cmp3 = icmp ult i64 %conv2, 8
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond1
  %6 = load i32, i32* %j, align 4
  %add = add nsw i32 %6, 1
  %conv5 = sext i32 %add to i64
  %mul = mul i64 %conv5, 32
  %sub = sub i64 %mul, 1
  %conv6 = trunc i64 %sub to i32
  %7 = load i32, i32* %j, align 4
  %conv7 = sext i32 %7 to i64
  %mul8 = mul i64 %conv7, 32
  %conv9 = trunc i64 %mul8 to i32
  %call = call i64* @_ZN7ap_uintILi256EE5rangeEii(%struct.ap_uint* %temp, i32 %conv6, i32 %conv9)
  %coerce.dive = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %ref.tmp, i32 0, i32 0
  store i64* %call, i64** %coerce.dive, align 8
  %call10 = call i64 @_ZNK7ap_uintILi256EE11range_proxycvmEv(%"struct.ap_uint<256>::range_proxy"* %ref.tmp)
  %conv11 = trunc i64 %call10 to i32
  %raw_val = bitcast %union.raw_float* %t to i32*
  store i32 %conv11, i32* %raw_val, align 4
  %float_val = bitcast %union.raw_float* %t to float*
  %8 = load float, float* %float_val, align 4
  %9 = load i32, i32* %counter, align 4
  %idxprom12 = sext i32 %9 to i64
  %arrayidx13 = getelementptr inbounds [262144 x float], [262144 x float]* %in_local, i64 0, i64 %idxprom12
  store float %8, float* %arrayidx13, align 4
  %10 = load i32, i32* %counter, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %counter, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %11 = load i32, i32* %j, align 4
  %inc14 = add nsw i32 %11, 1
  store i32 %inc14, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc15

for.inc15:                                        ; preds = %for.end
  %12 = load i32, i32* %i, align 4
  %inc16 = add nsw i32 %12, 1
  store i32 %inc16, i32* %i, align 4
  br label %for.cond

for.end17:                                        ; preds = %for.cond
  store i32 0, i32* %i18, align 4
  br label %for.cond19

for.cond19:                                       ; preds = %for.inc34, %for.end17
  %13 = load i32, i32* %i18, align 4
  %cmp20 = icmp slt i32 %13, 64
  br i1 %cmp20, label %for.body21, label %for.end36

for.body21:                                       ; preds = %for.cond19
  %14 = load i32, i32* %i18, align 4
  %mul22 = mul nsw i32 %14, 4096
  %idxprom23 = sext i32 %mul22 to i64
  %arrayidx24 = getelementptr inbounds [262144 x float], [262144 x float]* %in_local, i64 0, i64 %idxprom23
  %15 = load float, float* %arrayidx24, align 4
  %16 = load i32, i32* %i18, align 4
  %idxprom25 = sext i32 %16 to i64
  %arrayidx26 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom25
  store float %15, float* %arrayidx26, align 4
  %17 = load i32, i32* %i18, align 4
  %idxprom27 = sext i32 %17 to i64
  %arrayidx28 = getelementptr inbounds [64 x float], [64 x float]* %p_arr_1, i64 0, i64 %idxprom27
  store float 5.000000e-01, float* %arrayidx28, align 4
  %18 = load i32, i32* %i18, align 4
  %idxprom29 = sext i32 %18 to i64
  %arrayidx30 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom29
  %19 = load float, float* %arrayidx30, align 4
  %20 = load i32, i32* %i18, align 4
  %mul31 = mul nsw i32 %20, 4096
  %idxprom32 = sext i32 %mul31 to i64
  %arrayidx33 = getelementptr inbounds [262144 x float], [262144 x float]* %out_local, i64 0, i64 %idxprom32
  store float %19, float* %arrayidx33, align 4
  br label %for.inc34

for.inc34:                                        ; preds = %for.body21
  %21 = load i32, i32* %i18, align 4
  %inc35 = add nsw i32 %21, 1
  store i32 %inc35, i32* %i18, align 4
  br label %for.cond19

for.end36:                                        ; preds = %for.cond19
  store i32 1, i32* %t37, align 4
  br label %for.cond38

for.cond38:                                       ; preds = %for.inc89, %for.end36
  %22 = load i32, i32* %t37, align 4
  %cmp39 = icmp slt i32 %22, 4096
  br i1 %cmp39, label %for.body40, label %for.end91

for.body40:                                       ; preds = %for.cond38
  store i32 0, i32* %i41, align 4
  br label %for.cond42

for.cond42:                                       ; preds = %for.inc86, %for.body40
  %23 = load i32, i32* %i41, align 4
  %cmp43 = icmp slt i32 %23, 64
  br i1 %cmp43, label %for.body44, label %for.end88

for.body44:                                       ; preds = %for.cond42
  %24 = load i32, i32* %i41, align 4
  %idxprom45 = sext i32 %24 to i64
  %arrayidx46 = getelementptr inbounds [64 x float], [64 x float]* %p_arr_1, i64 0, i64 %idxprom45
  %25 = load float, float* %arrayidx46, align 4
  %add47 = fadd float %25, 0x3F847AE140000000
  %26 = load i32, i32* %i41, align 4
  %idxprom48 = sext i32 %26 to i64
  %arrayidx49 = getelementptr inbounds [64 x float], [64 x float]* %calc_temp_arr, i64 0, i64 %idxprom48
  store float %add47, float* %arrayidx49, align 4
  %27 = load i32, i32* %i41, align 4
  %idxprom50 = sext i32 %27 to i64
  %arrayidx51 = getelementptr inbounds [64 x float], [64 x float]* %calc_temp_arr, i64 0, i64 %idxprom50
  %28 = load float, float* %arrayidx51, align 4
  %29 = load i32, i32* %i41, align 4
  %idxprom52 = sext i32 %29 to i64
  %arrayidx53 = getelementptr inbounds [64 x float], [64 x float]* %p_arr_1, i64 0, i64 %idxprom52
  %30 = load float, float* %arrayidx53, align 4
  %add54 = fadd float %30, 0x3FF028F5C0000000
  %div = fdiv float %28, %add54
  %31 = load i32, i32* %i41, align 4
  %idxprom55 = sext i32 %31 to i64
  %arrayidx56 = getelementptr inbounds [64 x float], [64 x float]* %k_arr, i64 0, i64 %idxprom55
  store float %div, float* %arrayidx56, align 4
  %32 = load i32, i32* %i41, align 4
  %idxprom57 = sext i32 %32 to i64
  %arrayidx58 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom57
  %33 = load float, float* %arrayidx58, align 4
  %34 = load i32, i32* %i41, align 4
  %idxprom59 = sext i32 %34 to i64
  %arrayidx60 = getelementptr inbounds [64 x float], [64 x float]* %k_arr, i64 0, i64 %idxprom59
  %35 = load float, float* %arrayidx60, align 4
  %36 = load i32, i32* %i41, align 4
  %mul61 = mul nsw i32 %36, 4096
  %37 = load i32, i32* %t37, align 4
  %add62 = add nsw i32 %mul61, %37
  %idxprom63 = sext i32 %add62 to i64
  %arrayidx64 = getelementptr inbounds [262144 x float], [262144 x float]* %in_local, i64 0, i64 %idxprom63
  %38 = load float, float* %arrayidx64, align 4
  %39 = load i32, i32* %i41, align 4
  %idxprom65 = sext i32 %39 to i64
  %arrayidx66 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom65
  %40 = load float, float* %arrayidx66, align 4
  %sub67 = fsub float %38, %40
  %mul68 = fmul float %35, %sub67
  %add69 = fadd float %33, %mul68
  %41 = load i32, i32* %i41, align 4
  %idxprom70 = sext i32 %41 to i64
  %arrayidx71 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom70
  store float %add69, float* %arrayidx71, align 4
  %42 = load i32, i32* %i41, align 4
  %idxprom72 = sext i32 %42 to i64
  %arrayidx73 = getelementptr inbounds [64 x float], [64 x float]* %k_arr, i64 0, i64 %idxprom72
  %43 = load float, float* %arrayidx73, align 4
  %sub74 = fsub float 1.000000e+00, %43
  %44 = load i32, i32* %i41, align 4
  %idxprom75 = sext i32 %44 to i64
  %arrayidx76 = getelementptr inbounds [64 x float], [64 x float]* %calc_temp_arr, i64 0, i64 %idxprom75
  %45 = load float, float* %arrayidx76, align 4
  %mul77 = fmul float %sub74, %45
  %46 = load i32, i32* %i41, align 4
  %idxprom78 = sext i32 %46 to i64
  %arrayidx79 = getelementptr inbounds [64 x float], [64 x float]* %p_arr_1, i64 0, i64 %idxprom78
  store float %mul77, float* %arrayidx79, align 4
  %47 = load i32, i32* %i41, align 4
  %idxprom80 = sext i32 %47 to i64
  %arrayidx81 = getelementptr inbounds [64 x float], [64 x float]* %u_hat_arr, i64 0, i64 %idxprom80
  %48 = load float, float* %arrayidx81, align 4
  %49 = load i32, i32* %i41, align 4
  %mul82 = mul nsw i32 %49, 4096
  %50 = load i32, i32* %t37, align 4
  %add83 = add nsw i32 %mul82, %50
  %idxprom84 = sext i32 %add83 to i64
  %arrayidx85 = getelementptr inbounds [262144 x float], [262144 x float]* %out_local, i64 0, i64 %idxprom84
  store float %48, float* %arrayidx85, align 4
  br label %for.inc86

for.inc86:                                        ; preds = %for.body44
  %51 = load i32, i32* %i41, align 4
  %inc87 = add nsw i32 %51, 1
  store i32 %inc87, i32* %i41, align 4
  br label %for.cond42

for.end88:                                        ; preds = %for.cond42
  br label %for.inc89

for.inc89:                                        ; preds = %for.end88
  %52 = load i32, i32* %t37, align 4
  %inc90 = add nsw i32 %52, 1
  store i32 %inc90, i32* %t37, align 4
  br label %for.cond38

for.end91:                                        ; preds = %for.cond38
  store i32 0, i32* %counter, align 4
  store i32 0, i32* %i92, align 4
  br label %for.cond93

for.cond93:                                       ; preds = %for.inc127, %for.end91
  %53 = load i32, i32* %i92, align 4
  %conv94 = sext i32 %53 to i64
  %cmp95 = icmp ult i64 %conv94, 32768
  br i1 %cmp95, label %for.body96, label %for.end129

for.body96:                                       ; preds = %for.cond93
  %call97 = call dereferenceable(8) %struct.ap_uint* @_ZN7ap_uintILi256EEaSEm(%struct.ap_uint* %temp, i64 0)
  store i32 0, i32* %j98, align 4
  br label %for.cond99

for.cond99:                                       ; preds = %for.inc122, %for.body96
  %54 = load i32, i32* %j98, align 4
  %conv100 = sext i32 %54 to i64
  %cmp101 = icmp ult i64 %conv100, 8
  br i1 %cmp101, label %for.body102, label %for.end124

for.body102:                                      ; preds = %for.cond99
  %55 = load i32, i32* %counter, align 4
  %idxprom104 = sext i32 %55 to i64
  %arrayidx105 = getelementptr inbounds [262144 x float], [262144 x float]* %out_local, i64 0, i64 %idxprom104
  %56 = load float, float* %arrayidx105, align 4
  %float_val106 = bitcast %union.raw_float* %t103 to float*
  store float %56, float* %float_val106, align 4
  %raw_val107 = bitcast %union.raw_float* %t103 to i32*
  %57 = load i32, i32* %raw_val107, align 4
  %conv108 = sext i32 %57 to i64
  %58 = load i32, i32* %j98, align 4
  %add110 = add nsw i32 %58, 1
  %conv111 = sext i32 %add110 to i64
  %mul112 = mul i64 %conv111, 32
  %sub113 = sub i64 %mul112, 1
  %conv114 = trunc i64 %sub113 to i32
  %59 = load i32, i32* %j98, align 4
  %conv115 = sext i32 %59 to i64
  %mul116 = mul i64 %conv115, 32
  %conv117 = trunc i64 %mul116 to i32
  %call118 = call i64* @_ZN7ap_uintILi256EE5rangeEii(%struct.ap_uint* %temp, i32 %conv114, i32 %conv117)
  %coerce.dive119 = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %ref.tmp109, i32 0, i32 0
  store i64* %call118, i64** %coerce.dive119, align 8
  %call120 = call dereferenceable(8) %"struct.ap_uint<256>::range_proxy"* @_ZN7ap_uintILi256EE11range_proxyaSEm(%"struct.ap_uint<256>::range_proxy"* %ref.tmp109, i64 %conv108)
  %60 = load i32, i32* %counter, align 4
  %inc121 = add nsw i32 %60, 1
  store i32 %inc121, i32* %counter, align 4
  br label %for.inc122

for.inc122:                                       ; preds = %for.body102
  %61 = load i32, i32* %j98, align 4
  %inc123 = add nsw i32 %61, 1
  store i32 %inc123, i32* %j98, align 4
  br label %for.cond99

for.end124:                                       ; preds = %for.cond99
  %62 = load %struct.ap_uint*, %struct.ap_uint** %out.addr, align 8
  %63 = load i32, i32* %i92, align 4
  %idxprom125 = sext i32 %63 to i64
  %arrayidx126 = getelementptr inbounds %struct.ap_uint, %struct.ap_uint* %62, i64 %idxprom125
  %64 = bitcast %struct.ap_uint* %arrayidx126 to i8*
  %65 = bitcast %struct.ap_uint* %temp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %64, i8* align 8 %65, i64 8, i1 false)
  br label %for.inc127

for.inc127:                                       ; preds = %for.end124
  %66 = load i32, i32* %i92, align 4
  %inc128 = add nsw i32 %66, 1
  store i32 %inc128, i32* %i92, align 4
  br label %for.cond93

for.end129:                                       ; preds = %for.cond93
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN7ap_uintILi256EEC2Em(%struct.ap_uint* %this, i64 %x) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.ap_uint*, align 8
  %x.addr = alloca i64, align 8
  store %struct.ap_uint* %this, %struct.ap_uint** %this.addr, align 8
  store i64 %x, i64* %x.addr, align 8
  %this1 = load %struct.ap_uint*, %struct.ap_uint** %this.addr, align 8
  %v = getelementptr inbounds %struct.ap_uint, %struct.ap_uint* %this1, i32 0, i32 0
  %0 = load i64, i64* %x.addr, align 8
  store i64 %0, i64* %v, align 8
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64* @_ZN7ap_uintILi256EE5rangeEii(%struct.ap_uint* %this, i32 %0, i32 %1) #0 comdat align 2 {
entry:
  %retval = alloca %"struct.ap_uint<256>::range_proxy", align 8
  %this.addr = alloca %struct.ap_uint*, align 8
  %.addr = alloca i32, align 4
  %.addr1 = alloca i32, align 4
  store %struct.ap_uint* %this, %struct.ap_uint** %this.addr, align 8
  store i32 %0, i32* %.addr, align 4
  store i32 %1, i32* %.addr1, align 4
  %this2 = load %struct.ap_uint*, %struct.ap_uint** %this.addr, align 8
  %v = getelementptr inbounds %struct.ap_uint, %struct.ap_uint* %this2, i32 0, i32 0
  call void @_ZN7ap_uintILi256EE11range_proxyC2ERm(%"struct.ap_uint<256>::range_proxy"* %retval, i64* dereferenceable(8) %v)
  %coerce.dive = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %retval, i32 0, i32 0
  %2 = load i64*, i64** %coerce.dive, align 8
  ret i64* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNK7ap_uintILi256EE11range_proxycvmEv(%"struct.ap_uint<256>::range_proxy"* %this) #1 comdat align 2 {
entry:
  %this.addr = alloca %"struct.ap_uint<256>::range_proxy"*, align 8
  store %"struct.ap_uint<256>::range_proxy"* %this, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  %this1 = load %"struct.ap_uint<256>::range_proxy"*, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  %ref = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %this1, i32 0, i32 0
  %0 = load i64*, i64** %ref, align 8
  %1 = load i64, i64* %0, align 8
  ret i64 %1
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ap_uint* @_ZN7ap_uintILi256EEaSEm(%struct.ap_uint* %this, i64 %x) #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.ap_uint*, align 8
  %x.addr = alloca i64, align 8
  store %struct.ap_uint* %this, %struct.ap_uint** %this.addr, align 8
  store i64 %x, i64* %x.addr, align 8
  %this1 = load %struct.ap_uint*, %struct.ap_uint** %this.addr, align 8
  %0 = load i64, i64* %x.addr, align 8
  %v = getelementptr inbounds %struct.ap_uint, %struct.ap_uint* %this1, i32 0, i32 0
  store i64 %0, i64* %v, align 8
  ret %struct.ap_uint* %this1
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local dereferenceable(8) %"struct.ap_uint<256>::range_proxy"* @_ZN7ap_uintILi256EE11range_proxyaSEm(%"struct.ap_uint<256>::range_proxy"* %this, i64 %x) #1 comdat align 2 {
entry:
  %this.addr = alloca %"struct.ap_uint<256>::range_proxy"*, align 8
  %x.addr = alloca i64, align 8
  store %"struct.ap_uint<256>::range_proxy"* %this, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  store i64 %x, i64* %x.addr, align 8
  %this1 = load %"struct.ap_uint<256>::range_proxy"*, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  %0 = load i64, i64* %x.addr, align 8
  %ref = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %this1, i32 0, i32 0
  %1 = load i64*, i64** %ref, align 8
  store i64 %0, i64* %1, align 8
  ret %"struct.ap_uint<256>::range_proxy"* %this1
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN7ap_uintILi256EE11range_proxyC2ERm(%"struct.ap_uint<256>::range_proxy"* %this, i64* dereferenceable(8) %r) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %"struct.ap_uint<256>::range_proxy"*, align 8
  %r.addr = alloca i64*, align 8
  store %"struct.ap_uint<256>::range_proxy"* %this, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  store i64* %r, i64** %r.addr, align 8
  %this1 = load %"struct.ap_uint<256>::range_proxy"*, %"struct.ap_uint<256>::range_proxy"** %this.addr, align 8
  %ref = getelementptr inbounds %"struct.ap_uint<256>::range_proxy", %"struct.ap_uint<256>::range_proxy"* %this1, i32 0, i32 0
  %0 = load i64*, i64** %r.addr, align 8
  store i64* %0, i64** %ref, align 8
  ret void
}

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind willreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0-4ubuntu1~18.04.2 "}
