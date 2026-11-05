Fresh V4 Stage-1 observations
============================

| Action | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| L1 — A partition | block, 2, 1 | **block, 2, 2** | block, 2, 1 | block, 2, 1 |
| L2 — x partition | cyclic, 2, 1 | **cyclic, 4, 1** | cyclic, 2, 1 | cyclic, 2, 1 |
| L3 — y partition | cyclic, 4, 1 | **cyclic, 2, 1** | cyclic, 4, 1 | cyclic, 4, 1 |
| L4 — outer (PIPE, UNROLL) | 0, 2 | 0, 2 | 0, 2 | 0, 2 |
| L5 — inner (PIPE, UNROLL) | 0, 2 | **1, 0** | 0, 2 | 0, 2 |

All values below come from saved vectors and candidate scores; inference checkpoint identity is in checkpoint.json.

| Mode | Action | Field | Boundary | Largest relative L2 | Layer | Contrast | Cosine |
|---|---|---|---|---:|---:|---|---:|
| controlled | L1 | auto{_ARRAY_D_L1} | post_decoder_residual | 0.1150948 | 32 | R1-vs-R2 | 0.9953924 |
| controlled | L1 | auto{_ARRAY_D_L1} | post_self_attention_residual | 0.0732621 | 16 | R1-vs-R4 | 0.9973128 |
| controlled | L1 | auto{_ARRAY_F_L1} | post_decoder_residual | 0.0959510 | 32 | R1-vs-R4 | 0.9964633 |
| controlled | L1 | auto{_ARRAY_F_L1} | post_self_attention_residual | 0.1097062 | 32 | R1-vs-R4 | 0.9940513 |
| controlled | L1 | auto{_ARRAY_T_L1} | post_decoder_residual | 0.0829780 | 8 | R1-vs-R4 | 0.9965621 |
| controlled | L1 | auto{_ARRAY_T_L1} | post_self_attention_residual | 0.0816430 | 16 | R1-vs-R3 | 0.9966639 |
| controlled | L2 | auto{_ARRAY_D_L2} | post_decoder_residual | 0.0574533 | 24 | R1-vs-R4 | 0.9983482 |
| controlled | L2 | auto{_ARRAY_D_L2} | post_self_attention_residual | 0.0592869 | 32 | R1-vs-R4 | 0.9982755 |
| controlled | L2 | auto{_ARRAY_F_L2} | post_decoder_residual | 0.0720400 | 24 | R1-vs-R4 | 0.9974255 |
| controlled | L2 | auto{_ARRAY_F_L2} | post_self_attention_residual | 0.0817286 | 32 | R1-vs-R4 | 0.9967245 |
| controlled | L2 | auto{_ARRAY_T_L2} | post_decoder_residual | 0.0601206 | 8 | R1-vs-R3 | 0.9982231 |
| controlled | L2 | auto{_ARRAY_T_L2} | post_self_attention_residual | 0.0627951 | 32 | R1-vs-R4 | 0.9980339 |
| controlled | L3 | auto{_ARRAY_D_L3} | post_decoder_residual | 0.0789577 | 16 | R1-vs-R4 | 0.9968905 |
| controlled | L3 | auto{_ARRAY_D_L3} | post_self_attention_residual | 0.0790309 | 16 | R1-vs-R4 | 0.9968796 |
| controlled | L3 | auto{_ARRAY_F_L3} | post_decoder_residual | 0.0806253 | 24 | R1-vs-R4 | 0.9967462 |
| controlled | L3 | auto{_ARRAY_F_L3} | post_self_attention_residual | 0.0781513 | 24 | R1-vs-R4 | 0.9969441 |
| controlled | L3 | auto{_ARRAY_T_L3} | post_decoder_residual | 0.0810659 | 8 | R1-vs-R4 | 0.9967150 |
| controlled | L3 | auto{_ARRAY_T_L3} | post_self_attention_residual | 0.0823802 | 8 | R1-vs-R4 | 0.9966076 |
| controlled | L4 | auto{_PIPE_L4} | post_decoder_residual | 0.0955446 | 32 | R1-vs-R4 | 0.9962525 |
| controlled | L4 | auto{_PIPE_L4} | post_self_attention_residual | 0.0913510 | 32 | R1-vs-R4 | 0.9958699 |
| controlled | L4 | auto{_UNROLL_L4} | post_decoder_residual | 0.0669629 | 24 | R1-vs-R4 | 0.9977678 |
| controlled | L4 | auto{_UNROLL_L4} | post_self_attention_residual | 0.0725051 | 32 | R1-vs-R4 | 0.9973831 |
| controlled | L5 | auto{_PIPE_L5} | post_decoder_residual | 0.0643354 | 32 | R1-vs-R4 | 0.9983088 |
| controlled | L5 | auto{_PIPE_L5} | post_self_attention_residual | 0.0674906 | 32 | R1-vs-R4 | 0.9977494 |
| controlled | L5 | auto{_UNROLL_L5} | post_decoder_residual | 0.0863856 | 24 | R1-vs-R4 | 0.9962618 |
| controlled | L5 | auto{_UNROLL_L5} | post_self_attention_residual | 0.0835768 | 24 | R1-vs-R4 | 0.9965020 |
| natural | L1 | auto{_ARRAY_D_L1} | post_decoder_residual | 0.1150948 | 32 | R1-vs-R2 | 0.9953924 |
| natural | L1 | auto{_ARRAY_D_L1} | post_self_attention_residual | 0.0732621 | 16 | R1-vs-R4 | 0.9973128 |
| natural | L1 | auto{_ARRAY_F_L1} | post_decoder_residual | 0.0959510 | 32 | R1-vs-R4 | 0.9964633 |
| natural | L1 | auto{_ARRAY_F_L1} | post_self_attention_residual | 0.1097062 | 32 | R1-vs-R4 | 0.9940513 |
| natural | L1 | auto{_ARRAY_T_L1} | post_decoder_residual | 0.0829780 | 8 | R1-vs-R4 | 0.9965621 |
| natural | L1 | auto{_ARRAY_T_L1} | post_self_attention_residual | 0.0816430 | 16 | R1-vs-R3 | 0.9966639 |
| natural | L1 | source | post_decoder_residual | 0.4136067 | 24 | R1-vs-R3 | 0.9183648 |
| natural | L1 | source | post_self_attention_residual | 0.4003990 | 24 | R1-vs-R3 | 0.9249339 |
| natural | L2 | auto{_ARRAY_D_L2} | post_decoder_residual | 0.3947716 | 16 | R1-vs-R2 | 0.9197690 |
| natural | L2 | auto{_ARRAY_D_L2} | post_self_attention_residual | 0.3882879 | 16 | R1-vs-R2 | 0.9223590 |
| natural | L2 | auto{_ARRAY_F_L2} | post_decoder_residual | 0.0720400 | 24 | R1-vs-R4 | 0.9974255 |
| natural | L2 | auto{_ARRAY_F_L2} | post_self_attention_residual | 0.0817286 | 32 | R1-vs-R4 | 0.9967245 |
| natural | L2 | auto{_ARRAY_T_L2} | post_decoder_residual | 0.0601206 | 8 | R1-vs-R3 | 0.9982231 |
| natural | L2 | auto{_ARRAY_T_L2} | post_self_attention_residual | 0.0627951 | 32 | R1-vs-R4 | 0.9980339 |
| natural | L2 | source | post_decoder_residual | 0.5049478 | 24 | R1-vs-R4 | 0.8796688 |
| natural | L2 | source | post_self_attention_residual | 0.4899855 | 24 | R1-vs-R4 | 0.8868005 |
| natural | L3 | auto{_ARRAY_D_L3} | post_decoder_residual | 0.4684390 | 16 | R1-vs-R2 | 0.8916692 |
| natural | L3 | auto{_ARRAY_D_L3} | post_self_attention_residual | 0.4625278 | 16 | R1-vs-R2 | 0.8978025 |
| natural | L3 | auto{_ARRAY_F_L3} | post_decoder_residual | 0.3727281 | 24 | R1-vs-R2 | 0.9328246 |
| natural | L3 | auto{_ARRAY_F_L3} | post_self_attention_residual | 0.3659815 | 24 | R1-vs-R2 | 0.9351625 |
| natural | L3 | auto{_ARRAY_T_L3} | post_decoder_residual | 0.1093178 | 8 | R1-vs-R2 | 0.9940117 |
| natural | L3 | auto{_ARRAY_T_L3} | post_self_attention_residual | 0.1101893 | 8 | R1-vs-R2 | 0.9939479 |
| natural | L3 | source | post_decoder_residual | 0.4219663 | 16 | R1-vs-R4 | 0.9083990 |
| natural | L3 | source | post_self_attention_residual | 0.4147487 | 16 | R1-vs-R4 | 0.9107931 |
| natural | L4 | auto{_PIPE_L4} | post_decoder_residual | 0.0955446 | 32 | R1-vs-R4 | 0.9962525 |
| natural | L4 | auto{_PIPE_L4} | post_self_attention_residual | 0.0913510 | 32 | R1-vs-R4 | 0.9958699 |
| natural | L4 | auto{_UNROLL_L4} | post_decoder_residual | 0.0716347 | 24 | R1-vs-R2 | 0.9974484 |
| natural | L4 | auto{_UNROLL_L4} | post_self_attention_residual | 0.0744415 | 32 | R1-vs-R2 | 0.9972802 |
| natural | L4 | source | post_decoder_residual | 0.3110321 | 8 | R1-vs-R4 | 0.9542432 |
| natural | L4 | source | post_self_attention_residual | 0.3095437 | 8 | R1-vs-R4 | 0.9540408 |
| natural | L5 | auto{_PIPE_L5} | post_decoder_residual | 0.0701675 | 32 | R1-vs-R2 | 0.9983912 |
| natural | L5 | auto{_PIPE_L5} | post_self_attention_residual | 0.0674906 | 32 | R1-vs-R4 | 0.9977494 |
| natural | L5 | auto{_UNROLL_L5} | post_decoder_residual | 0.7501929 | 24 | R1-vs-R2 | 0.7083776 |
| natural | L5 | auto{_UNROLL_L5} | post_self_attention_residual | 0.7397311 | 32 | R1-vs-R2 | 0.7428948 |
| natural | L5 | source | post_decoder_residual | 0.2136113 | 16 | R1-vs-R4 | 0.9769216 |
| natural | L5 | source | post_self_attention_residual | 0.2178907 | 16 | R1-vs-R4 | 0.9759781 |

Candidate preferences (13 fields × 3 contrasts per mode):

- natural: 5/39 choices change; 13 rankings change; 0 domains differ.
- controlled: 1/39 choices change; 9 rankings change; 0 domains differ.

Consult pairwise.csv for all cosine, relative L2, baseline/compared norms and signed relative norm changes; candidate_comparisons.csv for field-level margins and domain identity. These describe sensitivity, not optimization quality.
