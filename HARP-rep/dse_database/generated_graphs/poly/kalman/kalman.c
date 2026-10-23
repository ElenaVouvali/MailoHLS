#include <ap_int.h>
#include "merlin_type_define.h"
#pragma ACCEL kernel

void krnl_KALMAN(INTERFACE_WIDTH in[32768],INTERFACE_WIDTH out[32768])
{
  
  
/*L1:*/
  DTYPE in_local[262144];
/*L2:*/
  DTYPE out_local[262144];
  INTERFACE_WIDTH temp;
  int counter = 0;
/*L3:*/
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; ((unsigned long )i) < ((unsigned long )262144) / (((unsigned long )256) / (4 * ((unsigned long )8))); i++) {
    temp = in[i];
/*L4:*/
    for (int j = 0; ((unsigned long )j) < ((unsigned long )256) / (4 * ((unsigned long )8)); j++) {
      raw_float t;
      t . raw_val = ((int )(temp . range(((int )(((unsigned long )(j + 1)) * (4 * ((unsigned long )8)) - ((unsigned long )1))),((int )(((unsigned long )j) * (4 * ((unsigned long )8)))))));
      in_local[counter] = ap_fixed< 19,13,AP_RND,AP_WRAP,0> (t . float_val);
      counter++;
    }
  }
/*L5:*/
  DTYPE u_hat_arr[64];
/*L6:*/
  DTYPE p_arr_1[64];
/*L7:*/
  DTYPE k_arr[64];
/*L8:*/
  DTYPE calc_temp_arr[64];
/*L9:*/
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 64; i++) {
    u_hat_arr[i] = in_local[i * 4096];
    p_arr_1[i] = 0.5;
    out_local[i * 4096] = u_hat_arr[i];
  }
/*L10:*/
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (int t = 1; t < 4096; t++) {
/*L11:*/
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (int i = 0; i < 64; i++) {
      calc_temp_arr[i] = p_arr_1[i]+ap_fixed< 19,13,AP_RND,AP_WRAP,0> (0.01);
      k_arr[i] = calc_temp_arr[i]/(p_arr_1[i]+ap_fixed< 19,13,AP_RND,AP_WRAP,0> (1.01));
      u_hat_arr[i] = u_hat_arr[i]+k_arr[i]*(in_local[i * 4096 + t]-u_hat_arr[i]);
      p_arr_1[i] = (ap_fixed< 19,13,AP_RND,AP_WRAP,0> (1)-k_arr[i])*calc_temp_arr[i];
      out_local[i * 4096 + t] = u_hat_arr[i];
    }
  }
  counter = 0;
/*L12:*/
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (int i = 0; ((unsigned long )i) < ((unsigned long )262144) / (((unsigned long )256) / (4 * ((unsigned long )8))); i++) {
    temp = 0;
/*L13:*/
    for (int j = 0; ((unsigned long )j) < ((unsigned long )256) / (4 * ((unsigned long )8)); j++) {
      raw_float t;
      t . float_val = ((float )(out_local[counter]));
      temp . range(((int )(((unsigned long )(j + 1)) * (4 * ((unsigned long )8)) - ((unsigned long )1))),((int )(((unsigned long )j) * (4 * ((unsigned long )8))))) = ((unsigned long )t . raw_val);
      counter++;
    }
    out[i] = temp;
  }
}
