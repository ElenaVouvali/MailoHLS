#include "dilate.h"

extern "C" {

	void lc_dilate(float result[TILE_ROWS * GRID_COLS], float img[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary)
	{
L1:	    bool strel[25] = { 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0 };
#pragma HLS array_partition variable=strel type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}

	    int radius_p = STREL_ROWS / 2;
	    int radius_q = STREL_COLS / 2;

L2:	    for (int i = 0; i < TILE_ROWS; i++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
L3:	        for (int j = 0; j < GRID_COLS; j++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}

	            float max = 0.0, temp;

L4:	            for (int m = 0; m < STREL_ROWS; m++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
L5:	                for (int n = 0; n < STREL_COLS; n++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
	                    int p = i - radius_p + m;
	                    int q = j - radius_q + n;
	                    if ((p >= 0 || which_boundary != TOP) &&
	                    		q >= 0 &&
								(p < TILE_ROWS || which_boundary != BOTTOM) &&
								q < GRID_COLS &&
								strel[m * STREL_COLS + n] != 0)
	                    {
	                        temp = img[p * GRID_COLS + q + GRID_COLS * MAX_RADIUS];
	                        if (temp > max) max = temp;
	                    }
	                }
	            }
	            result[i * GRID_COLS + j] = max;
	        }
	    }
	    return;
	}

	void load_data_tile(float img_bram[(TILE_ROWS+2*MAX_RADIUS)*GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
L6:		for (int row=0; row<TILE_ROWS+2*MAX_RADIUS; ++row){
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
L7:			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
				img_bram[row*GRID_COLS+col] = img[starting_index+row*GRID_COLS+col];
			}
		}
	}

	void store_result_tile(float result_bram[TILE_ROWS * GRID_COLS], float result[GRID_ROWS*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
L8:		for (int row=0; row<TILE_ROWS; ++row){
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
L9:			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
				result[starting_index+row*GRID_COLS+col] = result_bram[row*GRID_COLS+col];
			}
		}
	}

	void workload(float result[GRID_COLS * GRID_ROWS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS])
	{

	    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
	    #pragma HLS INTERFACE m_axi port=img offset=slave bundle=img

	    #pragma HLS INTERFACE s_axilite port=result bundle=control
	    #pragma HLS INTERFACE s_axilite port=img bundle=control

	    #pragma HLS INTERFACE s_axilite port=return bundle=control

		int i, n, k;

L10:	    float result_inner_0   [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_0 type=auto{_ARRAY_T_L10} factor=auto{_ARRAY_F_L10} dim=auto{_ARRAY_D_L10}
L11:	    float img_inner_0      [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=img_inner_0 type=auto{_ARRAY_T_L11} factor=auto{_ARRAY_F_L11} dim=auto{_ARRAY_D_L11}

L12:		for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
#pragma HLS pipeline II=auto{_PIPE_L12}
#pragma HLS unroll factor=auto{_UNROLL_L12}
			load_data_tile(img_inner_0, img, k);
			lc_dilate(result_inner_0, img_inner_0, k);
			store_result_tile(result_inner_0, result, k);
		}

		return;
	}

}