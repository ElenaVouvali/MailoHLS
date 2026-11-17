// MailoHLS external/unseen application input.
// Adapted from the HLSyn/HierarchicalMoE trmm-opt kernel.
// Optimization pragmas from the upstream DSE template were removed; MailoHLS Lk action anchors were added.

extern "C" {
void kernel_trmm(double alpha, double A[60][60], double B[60][80])
{
  // BLAS parameters: SIDE='L', UPLO='L', TRANSA='T', DIAG='U'.
  // B := alpha * A^T * B.

/*L1:*/  for (int i = 0; i < 60; i++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
/*L2:*/    for (int j = 0; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
      double sum = B[i][j];
/*L3:*/      for (int k = 0; k < 60; k++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
        if (k > i) {
          sum += A[k][i] * B[k][j];
        }
      }
      B[i][j] = alpha * sum;
    }
  }
}
}
