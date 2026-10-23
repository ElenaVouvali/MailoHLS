#include "streamcluster.h"
#define BUF_SIZE 256

extern "C" {

void load(int i, float* coord, float* weight, float* cost, int* assign,
			float* buffer_coord, float* buffer_weight, float* buffer_cost, int* buffer_assign){
	memcpy(buffer_coord, coord + i * DIM, BUF_SIZE * DIM * sizeof(float));
	memcpy(buffer_weight, weight + i, BUF_SIZE * sizeof(float));
	memcpy(buffer_cost, cost + i, BUF_SIZE * sizeof(float));
	memcpy(buffer_assign, assign + i, BUF_SIZE * sizeof(int));
}

//Input of compute should be the buffer 
void compute(int num, int k, float* coord, float* weight, float* target, float* cost, int * assign, 
		int* center_table, char* switch_membership, float* cost_of_opening_x, float* work_mem, float* x_cost){
L1:	pre:for(int i = 0; i < BUF_SIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
		#pragma HLS DEPENDENCE variable=x_cost inter false 
		float sum = 0;
L2:		pre_inner:for(int j = 0; j < DIM; j++){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
			float a = coord[i * DIM + j] - target[j];
			sum += a * a;
		}
		x_cost[i] = sum * weight[i];
	}

L3:	after:for(int i = 0; i < BUF_SIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
		float current_cost = x_cost[i] - cost[i];
		//int local_center_index = center_table[assign[i]];

		if(current_cost < 0){
			switch_membership[k + i] = 1;
			cost_of_opening_x[0] += current_cost;
		} else{
			int local_center_index = center_table[assign[i]];
			work_mem[local_center_index] -= current_cost;
		}
	}
}

void store(int num, int numcenter, float* work_mem, char* switch_membership, float* cost_of_opening_x,
			float* buffer_work_mem, char* buffer_switch_membership, float* buffer_cost_of_opening_x){
	memcpy(work_mem, buffer_work_mem, BATCH_SIZE * sizeof(float));
	memcpy(switch_membership, buffer_switch_membership, BATCH_SIZE * sizeof(char));
	cost_of_opening_x[0] = buffer_cost_of_opening_x[0];
}

void workload(    
    float* coord,                      
    float* weight,                      
    float* cost, 
    float* target,
    int* assign,
    int* center_table,
    char* switch_membership,
    float* work_mem,
    int num,
    float* cost_of_opening_x,
    int numcenter            
)
{
    #pragma HLS INTERFACE m_axi port=coord offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=cost offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=target offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=assign offset=slave bundle=gmemi
    #pragma HLS INTERFACE m_axi port=center_table offset=slave bundle=gmemi
    #pragma HLS INTERFACE m_axi port=switch_membership offset=slave bundle=gmemc
    #pragma HLS INTERFACE m_axi port=work_mem offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=cost_of_opening_x offset=slave bundle=gmemf

    #pragma HLS INTERFACE s_axilite port=coord bundle=control
    #pragma HLS INTERFACE s_axilite port=weight bundle=control
    #pragma HLS INTERFACE s_axilite port=cost bundle=control
    #pragma HLS INTERFACE s_axilite port=target bundle=control
    #pragma HLS INTERFACE s_axilite port=assign bundle=control
    #pragma HLS INTERFACE s_axilite port=center_table bundle=control
    #pragma HLS INTERFACE s_axilite port=switch_membership bundle=control
    #pragma HLS INTERFACE s_axilite port=work_mem bundle=control
    #pragma HLS INTERFACE s_axilite port=num bundle=control
    #pragma HLS INTERFACE s_axilite port=cost_of_opening_x bundle=control
    #pragma HLS INTERFACE s_axilite port=numcenter bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

L4:	float buffer_target[DIM];
#pragma HLS array_partition variable=buffer_target type=auto{_ARRAY_T_L4} factor=auto{_ARRAY_F_L4} dim=auto{_ARRAY_D_L4}

L5:	int buffer_center_table[BATCH_SIZE];
#pragma HLS array_partition variable=buffer_center_table type=auto{_ARRAY_T_L5} factor=auto{_ARRAY_F_L5} dim=auto{_ARRAY_D_L5}
	
L6:	float buffer_work_mem[BATCH_SIZE];
#pragma HLS array_partition variable=buffer_work_mem type=auto{_ARRAY_T_L6} factor=auto{_ARRAY_F_L6} dim=auto{_ARRAY_D_L6}
L7:	char buffer_switch_membership[BATCH_SIZE];
#pragma HLS array_partition variable=buffer_switch_membership type=auto{_ARRAY_T_L7} factor=auto{_ARRAY_F_L7} dim=auto{_ARRAY_D_L7}
L8:	float buffer_cost_of_opening_x[1];
#pragma HLS array_partition variable=buffer_cost_of_opening_x type=auto{_ARRAY_T_L8} factor=auto{_ARRAY_F_L8} dim=auto{_ARRAY_D_L8}
	buffer_cost_of_opening_x[0] = cost_of_opening_x[0];

L9:	int buffer_assign[BUF_SIZE];
#pragma HLS array_partition variable=buffer_assign type=auto{_ARRAY_T_L9} factor=auto{_ARRAY_F_L9} dim=auto{_ARRAY_D_L9}
L10:	float buffer_cost[BUF_SIZE];
#pragma HLS array_partition variable=buffer_cost type=auto{_ARRAY_T_L10} factor=auto{_ARRAY_F_L10} dim=auto{_ARRAY_D_L10}
L11:	float buffer_weight[BUF_SIZE];
#pragma HLS array_partition variable=buffer_weight type=auto{_ARRAY_T_L11} factor=auto{_ARRAY_F_L11} dim=auto{_ARRAY_D_L11}
L12:	float buffer_coord[BUF_SIZE * DIM];
#pragma HLS array_partition variable=buffer_coord type=auto{_ARRAY_T_L12} factor=auto{_ARRAY_F_L12} dim=auto{_ARRAY_D_L12}

L13:	float x_cost[BUF_SIZE];
#pragma HLS array_partition variable=x_cost type=auto{_ARRAY_T_L13} factor=auto{_ARRAY_F_L13} dim=auto{_ARRAY_D_L13}

	memcpy(buffer_work_mem, work_mem, BATCH_SIZE * sizeof(float));
	memcpy(buffer_switch_membership, switch_membership, BATCH_SIZE * sizeof(char));
	
	memcpy(buffer_target, target, DIM * sizeof(float));
	memcpy(buffer_center_table, center_table, BATCH_SIZE * sizeof(int));

L14:	process:for(int i = 0; i < BATCH_SIZE; i += BUF_SIZE){
#pragma HLS pipeline II=auto{_PIPE_L14}
#pragma HLS unroll factor=auto{_UNROLL_L14}
		load(i, coord, weight, cost, assign, buffer_coord, buffer_weight, buffer_cost, buffer_assign);
		compute(num, i, buffer_coord, buffer_weight, buffer_target, buffer_cost, buffer_assign, 
				buffer_center_table, buffer_switch_membership, buffer_cost_of_opening_x, buffer_work_mem, x_cost);
	}

	store(num, numcenter, work_mem, switch_membership, cost_of_opening_x,
		buffer_work_mem, buffer_switch_membership, buffer_cost_of_opening_x);

	return;
}
}
