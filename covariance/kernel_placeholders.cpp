// MailoHLS external/unseen application input.
// Adapted from the HLSyn/HierarchicalMoE covariance kernel.
// Optimization pragmas from the upstream DSE template were removed; MailoHLS Lk action anchors were added.

extern "C" {
void kernel_covariance(int m, int n, double float_n,
                       double data[100][80],
                       double cov[80][80],
                       double mean[80])
{
  int i;
  int j;
  int k;

/*L1:*/  for (j = 0; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
    mean[j] = 0.0;
/*L2:*/    for (i = 0; i < 100; i++) {
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }

/*L3:*/  for (i = 0; i < 100; i++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
/*L4:*/    for (j = 0; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
      data[i][j] -= mean[j];
    }
  }

/*L5:*/  for (i = 0; i < 80; i++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
/*L6:*/    for (j = i; j < 80; j++) {
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
      cov[i][j] = 0.0;
/*L7:*/      for (k = 0; k < 100; k++) {
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
        cov[i][j] += data[k][i] * data[k][j];
      }
      cov[i][j] /= float_n - 1.0;
      cov[j][i] = cov[i][j];
    }
  }
}
}
