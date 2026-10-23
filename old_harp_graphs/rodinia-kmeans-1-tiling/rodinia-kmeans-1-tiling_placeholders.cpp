#include "kmeans.h"

extern "C"{
void load_local_cluster(float local_clusters[NCLUSTERS * NFEATURES], float clusters[NCLUSTERS * NFEATURES])
{
L1:	for (int i(0); i<NCLUSTERS; ++i){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:		for (int j(0); j<NFEATURES; ++j){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
			local_clusters[i*NFEATURES+j] = clusters[i*NFEATURES+j];
		}
	}
}

void load_local_feature(float local_feature[TILE_SIZE * NFEATURES], float feature[NPOINTS * NFEATURES], int tile_idx)
{
L3:	for (int i(0); i<TILE_SIZE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
L4:		for (int j(0); j<NFEATURES; ++j){
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
			local_feature[i*NFEATURES+j] = feature[(tile_idx*TILE_SIZE+i)*NFEATURES+j];
		}
	}
}

void compute_local_membership(float local_feature[TILE_SIZE * NFEATURES],
		float local_clusters[NCLUSTERS * NFEATURES],int local_membership[TILE_SIZE])
{

L5:    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
        float min_dist = FLT_MAX;
        int index = 0;

        /* find the cluster center id with min distance to pt */
L6:        MIN: for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
            float dist = 0.0;

L7:            DIST: for (int k = 0; k < NFEATURES; k++) {
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
                float diff = local_feature[NFEATURES * i + k] - local_clusters[NFEATURES * j + k];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        /* assign the membership to object i */
        local_membership[i] = index;
    }
}

void store_local_membership(int local_membership[TILE_SIZE], int membership[NPOINTS], int tile_idx)
{
L8:	for (int i(0); i<TILE_SIZE; ++i){
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
		membership[tile_idx*TILE_SIZE+i] = local_membership[i];
	}
}

void workload(float  *feature, /* [npoints][nfeatures] */
              float  *clusters, /* [n_clusters][n_features] */
			  int *membership)
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int i, ii, j, k, index;

L9:    int local_membership[TILE_SIZE];
#pragma HLS array_partition variable=local_membership type=auto{_ARRAY_T_L9} factor=auto{_ARRAY_F_L9} dim=auto{_ARRAY_D_L9}
L10:    float local_feature[TILE_SIZE * NFEATURES];
#pragma HLS array_partition variable=local_feature type=auto{_ARRAY_T_L10} factor=auto{_ARRAY_F_L10} dim=auto{_ARRAY_D_L10}
L11:    float local_clusters[NCLUSTERS * NFEATURES];
#pragma HLS array_partition variable=local_clusters type=auto{_ARRAY_T_L11} factor=auto{_ARRAY_F_L11} dim=auto{_ARRAY_D_L11}

    load_local_cluster(local_clusters, clusters);

L12:    UPDATE_MEMBER1: for (ii=0; ii<NUM_TILES; ++ii)
#pragma HLS pipeline II=auto{_PIPE_L12}
#pragma HLS unroll factor=auto{_UNROLL_L12}
    {
    	load_local_feature(local_feature, feature, ii);
    	compute_local_membership(local_feature, local_clusters, local_membership);
        store_local_membership(local_membership, membership, ii);
    }
}
}