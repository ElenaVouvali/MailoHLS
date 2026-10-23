#include "dilate.h"

extern "C" {

void workload(float result[GRID_ROWS * GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS])
{

	#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=img offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=result bundle=control
	#pragma HLS INTERFACE s_axilite port=img bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	int starting_idx = MAX_RADIUS*GRID_COLS;

L1:	bool strel[25] = { 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0 };
#pragma HLS array_partition variable=strel type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}

	int radius_p = STREL_ROWS / 2;
	int radius_q = STREL_COLS / 2;

L2:	for (int i = 0; i < GRID_ROWS; i++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
L3:		for (int j = 0; j < GRID_COLS; j++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
			float max = 0.0, temp;
L4:			for (int m = 0; m < STREL_ROWS; m++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
L5:				for (int n = 0; n < STREL_COLS; n++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
					int p = i - radius_p + m;
					int q = j - radius_q + n;
					if (p >= 0 && q >= 0 && p < GRID_ROWS && q < GRID_COLS && strel[m * STREL_COLS + n] != 0) {
						temp = img[starting_idx + p * GRID_COLS + q];
						if (temp > max) max = temp;
					}
				}
			}
			result[i * GRID_COLS + j] = max;
		}
	}

	return;
}

}