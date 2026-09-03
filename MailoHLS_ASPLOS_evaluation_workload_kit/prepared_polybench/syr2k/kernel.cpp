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
/*L2:*/    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }

/*L3:*/    for (k = 0; k < 60; k++) {
/*L4:*/      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += A[j][k] * alpha * B[i][k]
                   + B[j][k] * alpha * A[i][k];
        }
      }
    }
  }
}
}
