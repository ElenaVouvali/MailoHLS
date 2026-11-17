// MailoHLS external/unseen application input.
// Adapted from the HLSyn/HierarchicalMoE jacobi-2d kernel.
// Optimization pragmas from the upstream DSE template were removed; MailoHLS Lk action anchors were added.

extern "C" {
void kernel_jacobi_2d(int tsteps, int n, double A[90][90], double B[90][90])
{
  int t;
  int i;
  int j;

/*L1:*/  for (t = 0; t < 40; t++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
/*L2:*/    for (i = 1; i < 89; i++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
/*L3:*/      for (j = 1; j < 89; j++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j]
                         + A[1 + i][j] + A[i - 1][j]);
      }
    }

/*L4:*/    for (i = 1; i < 89; i++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
/*L5:*/      for (j = 1; j < 89; j++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
        A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][1 + j]
                         + B[1 + i][j] + B[i - 1][j]);
      }
    }
  }
}
}
