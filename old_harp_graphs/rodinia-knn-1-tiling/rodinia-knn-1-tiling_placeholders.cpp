#include "knn.h"
extern "C"{
void load (int load_idx, float* searchSpace, float* local_searchSpace)
{
	int start_idx = load_idx * NUM_PT_IN_BUFFER * NUM_FEATURE;
L1:	LOAD_TILE: for (int i(0); i < NUM_PT_IN_BUFFER*NUM_FEATURE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
		local_searchSpace[i] = searchSpace[start_idx+i];
	}
}

void compute_dist (float* local_inputQuery, float* local_searchSpace, float* local_distance)
{
    float sum;
	float feature_delta;
L2:	COMPUTE_TILE: for (int i = 0; i < NUM_PT_IN_BUFFER; ++i) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
        sum = 0.0;
L3:		for (int j = 0; j < NUM_FEATURE; ++j){
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
			feature_delta = local_searchSpace[i*NUM_FEATURE+j] - local_inputQuery[j];
			sum += feature_delta*feature_delta;
		}
        local_distance[i] = sum;
	}	
}

void store (int store_idx, float* local_distance, float* distance)
{
	int start_idx = store_idx * NUM_PT_IN_BUFFER;
L4:	STORE_TILE: for (int i(0); i < NUM_PT_IN_BUFFER; ++i){
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
        distance[start_idx+i] = local_distance[i];
	}        
}

void workload(
	float inputQuery[NUM_FEATURE],
	float searchSpace[NUM_PT_IN_SEARCHSPACE*NUM_FEATURE],
    float distance[NUM_PT_IN_SEARCHSPACE]
){
    #pragma HLS INTERFACE m_axi port=inputQuery offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=inputQuery bundle=control
    #pragma HLS INTERFACE m_axi port=searchSpace offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=searchSpace bundle=control
    #pragma HLS INTERFACE m_axi port=distance offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=distance bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

L5:	float local_inputQuery[NUM_FEATURE];
#pragma HLS array_partition variable=local_inputQuery type=auto{_ARRAY_T_L5} factor=auto{_ARRAY_F_L5} dim=auto{_ARRAY_D_L5}
L6:	float local_searchSpace[NUM_PT_IN_BUFFER*NUM_FEATURE];
#pragma HLS array_partition variable=local_searchSpace type=auto{_ARRAY_T_L6} factor=auto{_ARRAY_F_L6} dim=auto{_ARRAY_D_L6}
L7:	float local_distance[NUM_PT_IN_BUFFER];
#pragma HLS array_partition variable=local_distance type=auto{_ARRAY_T_L7} factor=auto{_ARRAY_F_L7} dim=auto{_ARRAY_D_L7}

L8:	LOAD_INPUTQUERY: for (int i(0); i<NUM_FEATURE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
		local_inputQuery[i] = inputQuery[i];
    }
	
L9:	TILED_PE: for (int tile_idx(0); tile_idx<NUM_TILES; ++tile_idx){
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
		load(tile_idx, searchSpace, local_searchSpace);
		compute_dist(local_inputQuery, local_searchSpace, local_distance);
        store(tile_idx, local_distance, distance);
	}

	return;
}
}
