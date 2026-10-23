#include "Example5.h"

void MatrixMultiplication(const double A[], const double B[], double C[]) {
L1:  for (int n = 0; n < N; ++n) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:    double acc[M];  // Buffer an output row of C
#pragma HLS array_partition variable=acc type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}

L3:    for (int k = 0; k < K; ++k) {  // Collapsed dimension moved out
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}

      const auto a = A[n * K + k];  // We only need to read A once per row of B

L4:      for (int m = 0; m < M; ++m) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
        const double prev = (k == 0) ? 0 : acc[m];  // Automatic "reset" during
        acc[m] = prev + a * B[k * M + m];           // first iteration of M-loop
        #pragma HLS DEPENDENCE variable=acc false
      }
    }

    // Write out resulting row of C
L5:    for (int m = 0; m < M; ++m) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
      C[n * M + m] = acc[m];
    }
  }
}
