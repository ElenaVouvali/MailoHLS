/*
 * Apply batching
 */

#define BATCH_SIZE_1 262144
#define INTERNAL_BATCH_SIZE 4096
#define INTERNAL_BATCHES 64

/*
 * Use custom data type
 */

#include "ap_fixed.h"

#define DEC_BITS 6
#define INT_BITS 13

typedef ap_fixed<INT_BITS + DEC_BITS, INT_BITS, AP_RND> DTYPE;

/*
 * Use the maximum AXI width
 */

#include "ap_int.h"

#define WIDTH 256
#define FLOAT_BITS (sizeof(float) * 8)
#define FLOATS_PER_ELEMENT (WIDTH / FLOAT_BITS)

typedef ap_uint<WIDTH> INTERFACE_WIDTH;

typedef union
{
	int raw_val;
	float float_val;
} raw_float;

extern "C"{

	void krnl_KALMAN(INTERFACE_WIDTH in[BATCH_SIZE_1 / FLOATS_PER_ELEMENT], INTERFACE_WIDTH out[BATCH_SIZE_1 / FLOATS_PER_ELEMENT]) {
		#pragma HLS INTERFACE m_axi port=in bundle=gmem0
		#pragma HLS INTERFACE m_axi port=out bundle=gmem1

/*L1:*/	DTYPE in_local[BATCH_SIZE_1];
#pragma HLS array_partition variable=in_local type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
/*L2:*/	DTYPE out_local[BATCH_SIZE_1];
#pragma HLS array_partition variable=out_local type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}

		INTERFACE_WIDTH temp;
		int counter = 0;
/*L3:*/	for(int i = 0; i < BATCH_SIZE_1 / FLOATS_PER_ELEMENT; i++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
			temp = in[i];

/*L4:*/	for(int j = 0; j < FLOATS_PER_ELEMENT; j++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
				raw_float t;
			        t.raw_val = temp.range((j + 1) * FLOAT_BITS - 1, j * FLOAT_BITS);
			        in_local[counter] = (DTYPE)t.float_val;

				counter++;
			}
		}

/*L5:*/	DTYPE u_hat_arr[INTERNAL_BATCHES];
#pragma HLS array_partition variable=u_hat_arr type=auto{_ARRAY_T_L5} factor=auto{_ARRAY_F_L5} dim=auto{_ARRAY_D_L5}
/*L6:*/	DTYPE p_arr_1[INTERNAL_BATCHES];
#pragma HLS array_partition variable=p_arr_1 type=auto{_ARRAY_T_L6} factor=auto{_ARRAY_F_L6} dim=auto{_ARRAY_D_L6}
/*L7:*/	DTYPE k_arr[INTERNAL_BATCHES];
#pragma HLS array_partition variable=k_arr type=auto{_ARRAY_T_L7} factor=auto{_ARRAY_F_L7} dim=auto{_ARRAY_D_L7}
/*L8:*/	DTYPE calc_temp_arr[INTERNAL_BATCHES];
#pragma HLS array_partition variable=calc_temp_arr type=auto{_ARRAY_T_L8} factor=auto{_ARRAY_F_L8} dim=auto{_ARRAY_D_L8}

/*L9:*/	for (int i = 0; i < INTERNAL_BATCHES; i++) {
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
			u_hat_arr[i] = in_local[i * INTERNAL_BATCH_SIZE];
			p_arr_1[i] = 0.5;
			out_local[i * INTERNAL_BATCH_SIZE] = u_hat_arr[i];
		}

/*L10:*/	for (int t = 1; t < INTERNAL_BATCH_SIZE; t++) {
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
/*L11:*/	for (int i = 0; i < INTERNAL_BATCHES; i++) {
#pragma HLS pipeline II=auto{_PIPE_L11}
#pragma HLS unroll factor=auto{_UNROLL_L11}
				calc_temp_arr[i] = p_arr_1[i] + (DTYPE)0.01;

				k_arr[i] = calc_temp_arr[i] / (p_arr_1[i] + (DTYPE)1.01);
				
				u_hat_arr[i] = u_hat_arr[i] + k_arr[i] * (in_local[i * INTERNAL_BATCH_SIZE + t] - u_hat_arr[i]);
				
				p_arr_1[i] = ((DTYPE)1 - k_arr[i]) * calc_temp_arr[i];
				
				out_local[i * INTERNAL_BATCH_SIZE + t] = u_hat_arr[i];
			}
		}

		counter = 0;
/*L12:*/	for(int i = 0; i < BATCH_SIZE_1 / FLOATS_PER_ELEMENT; i++) {
#pragma HLS pipeline II=auto{_PIPE_L12}
#pragma HLS unroll factor=auto{_UNROLL_L12}
			temp = 0;

/*L13:*/	for(int j = 0; j < FLOATS_PER_ELEMENT; j++) {
#pragma HLS pipeline II=auto{_PIPE_L13}
#pragma HLS unroll factor=auto{_UNROLL_L13}
				raw_float t;
				t.float_val = (float)out_local[counter];

 				temp.range((j + 1) * FLOAT_BITS - 1, j * FLOAT_BITS) = t.raw_val;

				counter++;
			}

			out[i] = temp;
		}
	}
}
