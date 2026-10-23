#include "knn.h"
extern "C"{
void load (int load_idx, float* searchSpace, float* local_searchSpace)
{
#pragma HLS INLINE OFF
	int start_idx = load_idx * NUM_PT_IN_BUFFER * NUM_FEATURE;
L1:	LOAD_TILE: for (int i(0); i < NUM_PT_IN_BUFFER*NUM_FEATURE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
		local_searchSpace[i] = searchSpace[start_idx+i];
	}
}

void compute_dist (float* local_inputQuery, float* local_searchSpace, float* local_distance)
{
#pragma HLS INLINE OFF
    float sum;
	float feature_delta;
L2:	COMPUTE_TILE: for (int i = 0; i < NUM_PT_IN_BUFFER; i+=UNROLL_FACTOR) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
L3:        for (int j = 0; j < UNROLL_FACTOR; ++j){
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
            sum = 0.0;
L4:            for (int k = 0; k < NUM_FEATURE; ++k){
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
                feature_delta = local_searchSpace[(i+j)*NUM_FEATURE+k] - local_inputQuery[k];
                sum += feature_delta*feature_delta;
            }
            local_distance[i+j] = sum;
        }
	}	
}

void store (int store_idx, float* local_distance, float* distance)
{
#pragma HLS INLINE OFF
	int start_idx = store_idx * NUM_PT_IN_BUFFER;
L5:	STORE_TILE: for (int i(0); i < NUM_PT_IN_BUFFER; ++i){
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
        distance[start_idx+i] = local_distance[i];
	}        
}

void workload(
	float inputQuery[NUM_FEATURE],
	float searchSpace[NUM_PT_IN_SEARCHSPACE*NUM_FEATURE],
    float distance[NUM_PT_IN_SEARCHSPACE]
){
    // #pragma HLS INTERFACE m_axi port=inputQuery offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=inputQuery bundle=control
    // #pragma HLS INTERFACE m_axi port=searchSpace offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=searchSpace bundle=control
    // #pragma HLS INTERFACE m_axi port=distance offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=distance bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

L6:	float local_inputQuery[NUM_FEATURE];
#pragma HLS array_partition variable=local_inputQuery type=auto{_ARRAY_T_L6} factor=auto{_ARRAY_F_L6} dim=auto{_ARRAY_D_L6}
L7:	float local_searchSpace[NUM_PT_IN_BUFFER*NUM_FEATURE];
#pragma HLS array_partition variable=local_searchSpace type=auto{_ARRAY_T_L7} factor=auto{_ARRAY_F_L7} dim=auto{_ARRAY_D_L7}
L8:	float local_distance[NUM_PT_IN_BUFFER];
#pragma HLS array_partition variable=local_distance type=auto{_ARRAY_T_L8} factor=auto{_ARRAY_F_L8} dim=auto{_ARRAY_D_L8}
    
L9:	LOAD_INPUTQUERY: for (int i(0); i<NUM_FEATURE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
		local_inputQuery[i] = inputQuery[i];
    }
	
L10:	TILED_PE: for (int tile_idx(0); tile_idx<NUM_TILES; ++tile_idx){
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
		load(tile_idx, searchSpace, local_searchSpace);
		compute_dist(local_inputQuery, local_searchSpace, local_distance);
        store(tile_idx, local_distance, distance);
	}

	return;
}
}
