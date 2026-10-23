#include "pathfinder.h"
#include "mc.h"

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]){
	int32_t temp;
	int32_t i, t, tt, n;
	int32_t min, before;

L1:	KERNEL2: for(t = 0; t < ROWS-1 ;t++){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:		for(n = 0; n < COLS; n++){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
			min = Jout[n];
			
			if(n > 0){
				min = MIN(min,before);
			}
	
			if(n < COLS-1){
				min = MIN(min,Jout[n+1]);
			}
			before = Jout[n];
			
			Jout[n] = J[(t+1) * COLS + n]+min;
		}
	}
}

extern "C" {
void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]) {
  
 	#pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem
 	#pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem
  	#pragma HLS INTERFACE s_axilite port=J bundle=control
  	#pragma HLS INTERFACE s_axilite port=Jout bundle=control
  	#pragma HLS INTERFACE s_axilite port=return bundle=control

L3:	int32_t dst[COLS];
#pragma HLS array_partition variable=dst type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}
L4:	int32_t src[COLS];
#pragma HLS array_partition variable=src type=auto{_ARRAY_T_L4} factor=auto{_ARRAY_F_L4} dim=auto{_ARRAY_D_L4}
	int32_t i, t, tt, n;
	int32_t min;

	memcpy(dst,J,sizeof(int32_t) * COLS);

L5:	KERNEL_OUTER: for(t = 0; t < ROWS-1 ;t++){
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
L6:		KERNEL_INNER: for(n = 0; n < COLS; n++){
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
			min = dst[n];
			
			if(n > 0){
				min = MIN(min,dst[n-1]);
			}
	
			if(n < COLS-1){
				min = MIN(min,dst[n+1]);
			}
			
			src[n] = J[(t+1) * COLS + n]+min;
		}
		memcpy(dst,src,sizeof(int32_t) * COLS);
	}  	
	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
							  
}
}
