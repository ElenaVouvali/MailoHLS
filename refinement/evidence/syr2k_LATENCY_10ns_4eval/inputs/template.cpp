// MailoHLS external/unseen application input.
// Adapted from the HLSyn/HierarchicalMoE syr2k kernel.
// Optimization pragmas from the upstream DSE template were removed; MailoHLS Lk action anchors were added.

extern "C" {
void kernel_syr2k(double alpha, double beta,
                  double C[80][80], double A[80][60], double B[80][60])
{
  int i;
  int j;
  int k;
  // BLAS parameters: UPLO='L', TRANS='N'; A/B are NxM and C is NxN.

/*L1:*/  for (i = 0; i < 80; i++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
/*L2:*/    for (j = 0; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
      if (j <= i) {
        C[i][j] *= beta;
      }
    }

/*L3:*/    for (k = 0; k < 60; k++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
/*L4:*/      for (j = 0; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
        if (j <= i) {
          C[i][j] += A[j][k] * alpha * B[i][k]
                   + B[j][k] * alpha * A[i][k];
        }
      }
    }
  }
}
}
