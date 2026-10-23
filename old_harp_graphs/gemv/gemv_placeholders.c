#define M 32
#define N 32

void gemv() {
    /*L1:*/	float A[N][M];
#pragma HLS array_partition variable=A type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
    /*L2:*/	float x[M];
#pragma HLS array_partition variable=x type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}
    /*L3:*/	float y[N];
#pragma HLS array_partition variable=y type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}

    /*L4:*/	for (int i = 0; i < N; i++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
        float acc = 0;
        /*L5:*/	for (int j = 0; j < M; j++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
            acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
}
