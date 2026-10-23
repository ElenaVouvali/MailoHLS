#include "knn.h"

extern "C"{
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
    
    float sum;
    float feature_delta;
L1:    for(int i = 0; i < NUM_PT_IN_SEARCHSPACE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
        sum = 0.0;
L2:        for(int j = 0; j < NUM_FEATURE; ++j){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
            feature_delta = searchSpace[i*NUM_FEATURE+j] - inputQuery[j];
            sum += feature_delta*feature_delta;
        }
        distance[i] = sum;
    }

    return;
}
}