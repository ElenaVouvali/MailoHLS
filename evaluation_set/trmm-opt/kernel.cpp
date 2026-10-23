// MailoHLS external/unseen application input.
// Adapted from the HLSyn/HierarchicalMoE trmm-opt kernel.
// Optimization pragmas from the upstream DSE template were removed; MailoHLS Lk action anchors were added.

extern "C" {
void kernel_trmm(double alpha, double A[60][60], double B[60][80])
{
  // BLAS parameters: SIDE='L', UPLO='L', TRANSA='T', DIAG='U'.
  // B := alpha * A^T * B.

/*L1:*/  for (int i = 0; i < 60; i++) {
/*L2:*/    for (int j = 0; j < 80; j++) {
      double sum = B[i][j];
/*L3:*/      for (int k = 0; k < 60; k++) {
        if (k > i) {
          sum += A[k][i] * B[k][j];
        }
      }
      B[i][j] = alpha * sum;
    }
  }
}
}
