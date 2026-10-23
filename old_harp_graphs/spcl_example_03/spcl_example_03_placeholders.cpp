#include "Example3.h" // Defines N, M, and T

void Stencil2D(float const memory_in[N * M], float memory_out[N * M]) {

L1:  float above_buffer[M];
#pragma HLS array_partition variable=above_buffer type=auto{_ARRAY_T_L1} factor=auto{_ARRAY_F_L1} dim=auto{_ARRAY_D_L1}
L2:  float center_buffer[M];
#pragma HLS array_partition variable=center_buffer type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}

L3:  for (int i = 0; i < M; ++i) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
    above_buffer[i] = memory_in[i];
  }

L4:  for (int i = 0; i < M; ++i) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
    center_buffer[i] = memory_in[M + i];
  }

L5:  for (int i = 1; i < N - 1; ++i) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
L6:    for (int j = 0; j < M; ++j) {
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}

      const auto above = above_buffer[j];
      const auto center = center_buffer[j];
      const auto below = memory_in[(i + 1) * M + j];

      constexpr float factor = 0.3333;
      const auto average = factor * (above + center + below);

      memory_out[i * M + j] = average;
    }
  }
}
