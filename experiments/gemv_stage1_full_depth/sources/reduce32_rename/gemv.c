#define N 32
#define M 32
void gemv(const float inputA[N][M], const float inputx[M], float output[N*M+N]) {
    /*L1:*/ float matrix[N][M];
#pragma HLS array_partition variable=matrix type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
    /*L2:*/ float vector[M];
#pragma HLS array_partition variable=vector type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}
    /*L3:*/ float result[N];
#pragma HLS array_partition variable=result type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}
    for (int r=0; r<N; ++r) for (int c=0; c<M; ++c) matrix[r][c]=inputA[r][c];
    for (int c=0; c<M; ++c) vector[c]=inputx[c];
    /*L4:*/ for (int row=0; row<N; ++row) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
        float total=0;
        /*L5:*/ for (int col=0; col<M; ++col) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
            total += matrix[row][col] * vector[col];
            output[row*M+col]=total;
        }
        result[row]=total;
        output[N*M+row]=result[row];
    }
}
