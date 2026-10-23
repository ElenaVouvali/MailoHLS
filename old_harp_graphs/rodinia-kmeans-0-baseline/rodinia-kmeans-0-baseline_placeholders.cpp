#include "kmeans.h"

extern "C"{
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

L1:	UPDATE_MEMBER: for (int i = 0; i < NPOINTS; i++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
		float min_dist = FLT_MAX;
		int index = 0;

		/* find the cluster center id with min distance to pt */
L2:		MIN: for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
			float dist = 0.0;

L3:			DIST: for (int k = 0; k < NFEATURES; k++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
				float diff = feature[NFEATURES * i + k] - clusters[NFEATURES * j + k];
				dist += diff * diff;
			}
			if (dist < min_dist) {
				min_dist = dist;
				index = j;
			}
		}
		/* assign the membership to object i */
		membership[i] = index;
	}
}
}