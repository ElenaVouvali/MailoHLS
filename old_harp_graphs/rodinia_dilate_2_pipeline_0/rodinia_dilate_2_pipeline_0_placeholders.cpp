#include "dilate.h"

extern "C" {

	#define PARA_FACTOR 32

	float lc_dilate_stencil_core(float img_sample[STREL_ROWS * STREL_COLS])
	{
	    float max = 0.0;
L1:	    for (int i = 0; i < STREL_ROWS; i++)
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:	        for (int j = 0; j < STREL_COLS; j++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
	            float temp = img_sample[i * STREL_COLS + j];
	            if (temp > max) max = temp;
	        }
	    return max;
	}

	void lc_dilate(float result[TILE_ROWS * GRID_COLS], float img[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary)
	{

L3:	    bool strel[25] = { 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0 };
#pragma HLS array_partition variable=strel type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}

	    int radius_p = STREL_ROWS / 2;

L4:	    float img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
#pragma HLS array_partition variable=img_rf type=auto{_ARRAY_T_L4} factor=auto{_ARRAY_F_L4} dim=auto{_ARRAY_D_L4}


L5:	    for (int i = 0; i < MAX_RADIUS; i++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
	        img_rf[i] = 0.0;
	    }

L6:	    for (int i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
	        img_rf[i + MAX_RADIUS] = img[i];
	    }

L7:	    for (int i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}

L8:	        for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
L9:	            float img_sample[STREL_ROWS * STREL_COLS];
#pragma HLS array_partition variable=img_sample type=auto{_ARRAY_T_L9} factor=auto{_ARRAY_F_L9} dim=auto{_ARRAY_D_L9}

L10:	            for (int m = 0; m < STREL_ROWS; m++) {
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
L11:	                for (int n = 0; n < STREL_COLS; n++) {
#pragma HLS pipeline II=auto{_PIPE_L11}
#pragma HLS unroll factor=auto{_UNROLL_L11}
	                    if (((i % (GRID_COLS / PARA_FACTOR)) * PARA_FACTOR - MAX_RADIUS + n + k < 0) ||
	                    	((i % (GRID_COLS / PARA_FACTOR)) * PARA_FACTOR - MAX_RADIUS + n + k >= GRID_COLS) ||
							(which_boundary == TOP && (i < GRID_COLS / PARA_FACTOR) && m < MAX_RADIUS) ||
							(which_boundary == BOTTOM && (i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - 1)) && m > MAX_RADIUS) ||
							strel[m * STREL_COLS + n] != 1  )
	                    {
	                    	img_sample[m * STREL_COLS + n] = 0;
	                    }
	                    else {
	                    	img_sample[m * STREL_COLS + n] = img_rf[GRID_COLS * m + n + k];
	                    }
	                }
	            }
	            result[i * PARA_FACTOR + k] = lc_dilate_stencil_core(img_sample);
	        }

L12:	        for (int k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
#pragma HLS pipeline II=auto{_PIPE_L12}
#pragma HLS unroll factor=auto{_UNROLL_L12}
	            img_rf[k] = img_rf[k + PARA_FACTOR];
	        }

L13:	        for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS pipeline II=auto{_PIPE_L13}
#pragma HLS unroll factor=auto{_UNROLL_L13}
	        	if ((GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k) < ((TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS))
	        		img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] =
	        		   img[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
	        	else
	        		img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = 0.0;
	        }

	    }

	    return;
	}

	void load_data_tile(float img_bram[(TILE_ROWS+2*MAX_RADIUS)*GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
L14:		for (int row=0; row<TILE_ROWS+2*MAX_RADIUS; ++row){
#pragma HLS pipeline II=auto{_PIPE_L14}
#pragma HLS unroll factor=auto{_UNROLL_L14}
L15:			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS pipeline II=auto{_PIPE_L15}
#pragma HLS unroll factor=auto{_UNROLL_L15}
				img_bram[row*GRID_COLS+col] = img[starting_index+row*GRID_COLS+col];
			}
		}
	}

	void store_result_tile(float result_bram[TILE_ROWS * GRID_COLS], float result[GRID_ROWS*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
L16:		for (int row=0; row<TILE_ROWS; ++row){
#pragma HLS pipeline II=auto{_PIPE_L16}
#pragma HLS unroll factor=auto{_UNROLL_L16}
L17:			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS pipeline II=auto{_PIPE_L17}
#pragma HLS unroll factor=auto{_UNROLL_L17}
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

L18:	    float result_inner_0   [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_0 type=auto{_ARRAY_T_L18} factor=auto{_ARRAY_F_L18} dim=auto{_ARRAY_D_L18}
L19:	    float img_inner_0      [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=img_inner_0 type=auto{_ARRAY_T_L19} factor=auto{_ARRAY_F_L19} dim=auto{_ARRAY_D_L19}

L20:		for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
#pragma HLS pipeline II=auto{_PIPE_L20}
#pragma HLS unroll factor=auto{_UNROLL_L20}
			load_data_tile(img_inner_0, img, k);
			lc_dilate(result_inner_0, img_inner_0, k);
			store_result_tile(result_inner_0, result, k);
		}

		return;
	}

}