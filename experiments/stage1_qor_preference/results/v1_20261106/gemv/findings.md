Fresh V4-based QoR-DPO Stage-1 observations
============================

| Action | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| L1 — A partition | block, 2, 1 | **block, 2, 2** | block, 2, 1 | block, 2, 1 |
| L2 — x partition | cyclic, 2, 1 | **cyclic, 4, 1** | cyclic, 2, 1 | cyclic, 2, 1 |
| L3 — y partition | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 | complete, 0, 1 |
| L4 — outer (PIPE, UNROLL) | 0, 2 | 0, 2 | 0, 2 | 0, 2 |
| L5 — inner (PIPE, UNROLL) | 0, 2 | **1, 0** | 0, 2 | 0, 2 |

All values below come from saved vectors and candidate scores; inference checkpoint identity is in checkpoint.json.

| Mode | Action | Field | Boundary | Largest relative L2 | Layer | Contrast | Cosine |
|---|---|---|---|---:|---:|---|---:|
| controlled | L1 | auto{_ARRAY_D_L1} | post_decoder_residual | 0.1269798 | 32 | R1-vs-R2 | 0.9939908 |
| controlled | L1 | auto{_ARRAY_D_L1} | post_self_attention_residual | 0.0718208 | 32 | R1-vs-R2 | 0.9975503 |
| controlled | L1 | auto{_ARRAY_F_L1} | post_decoder_residual | 0.0859120 | 32 | R1-vs-R4 | 0.9969387 |
| controlled | L1 | auto{_ARRAY_F_L1} | post_self_attention_residual | 0.1021630 | 32 | R1-vs-R4 | 0.9948115 |
| controlled | L1 | auto{_ARRAY_T_L1} | post_decoder_residual | 0.0792717 | 8 | R1-vs-R3 | 0.9968544 |
| controlled | L1 | auto{_ARRAY_T_L1} | post_self_attention_residual | 0.0803171 | 16 | R1-vs-R3 | 0.9967695 |
| controlled | L2 | auto{_ARRAY_D_L2} | post_decoder_residual | 0.0543538 | 24 | R1-vs-R4 | 0.9985219 |
| controlled | L2 | auto{_ARRAY_D_L2} | post_self_attention_residual | 0.0570480 | 32 | R1-vs-R4 | 0.9983920 |
| controlled | L2 | auto{_ARRAY_F_L2} | post_decoder_residual | 0.0698726 | 24 | R1-vs-R4 | 0.9975797 |
| controlled | L2 | auto{_ARRAY_F_L2} | post_self_attention_residual | 0.0790025 | 32 | R1-vs-R4 | 0.9969239 |
| controlled | L2 | auto{_ARRAY_T_L2} | post_decoder_residual | 0.0593575 | 8 | R1-vs-R3 | 0.9982668 |
| controlled | L2 | auto{_ARRAY_T_L2} | post_self_attention_residual | 0.0603417 | 32 | R1-vs-R4 | 0.9981926 |
| controlled | L3 | auto{_ARRAY_D_L3} | post_decoder_residual | 0.0754115 | 16 | R1-vs-R4 | 0.9971705 |
| controlled | L3 | auto{_ARRAY_D_L3} | post_self_attention_residual | 0.0750207 | 16 | R1-vs-R4 | 0.9971887 |
| controlled | L3 | auto{_ARRAY_F_L3} | post_decoder_residual | 0.0751590 | 24 | R1-vs-R4 | 0.9971720 |
| controlled | L3 | auto{_ARRAY_F_L3} | post_self_attention_residual | 0.0729866 | 24 | R1-vs-R4 | 0.9973337 |
| controlled | L3 | auto{_ARRAY_T_L3} | post_decoder_residual | 0.0765071 | 8 | R1-vs-R4 | 0.9970722 |
| controlled | L3 | auto{_ARRAY_T_L3} | post_self_attention_residual | 0.0776017 | 8 | R1-vs-R4 | 0.9969898 |
| controlled | L4 | auto{_PIPE_L4} | post_decoder_residual | 0.0787188 | 32 | R1-vs-R4 | 0.9973508 |
| controlled | L4 | auto{_PIPE_L4} | post_self_attention_residual | 0.0762518 | 32 | R1-vs-R4 | 0.9971184 |
| controlled | L4 | auto{_UNROLL_L4} | post_decoder_residual | 0.0622490 | 24 | R1-vs-R4 | 0.9980713 |
| controlled | L4 | auto{_UNROLL_L4} | post_self_attention_residual | 0.0671353 | 32 | R1-vs-R4 | 0.9977527 |
| controlled | L5 | auto{_PIPE_L5} | post_decoder_residual | 0.0542748 | 8 | R1-vs-R4 | 0.9985334 |
| controlled | L5 | auto{_PIPE_L5} | post_self_attention_residual | 0.0557509 | 8 | R1-vs-R4 | 0.9984453 |
| controlled | L5 | auto{_UNROLL_L5} | post_decoder_residual | 0.0848788 | 24 | R1-vs-R4 | 0.9963917 |
| controlled | L5 | auto{_UNROLL_L5} | post_self_attention_residual | 0.0821765 | 24 | R1-vs-R4 | 0.9966185 |
| natural | L1 | auto{_ARRAY_D_L1} | post_decoder_residual | 0.1269798 | 32 | R1-vs-R2 | 0.9939908 |
| natural | L1 | auto{_ARRAY_D_L1} | post_self_attention_residual | 0.0718208 | 32 | R1-vs-R2 | 0.9975503 |
| natural | L1 | auto{_ARRAY_F_L1} | post_decoder_residual | 0.0859120 | 32 | R1-vs-R4 | 0.9969387 |
| natural | L1 | auto{_ARRAY_F_L1} | post_self_attention_residual | 0.1021630 | 32 | R1-vs-R4 | 0.9948115 |
| natural | L1 | auto{_ARRAY_T_L1} | post_decoder_residual | 0.0792717 | 8 | R1-vs-R3 | 0.9968544 |
| natural | L1 | auto{_ARRAY_T_L1} | post_self_attention_residual | 0.0803171 | 16 | R1-vs-R3 | 0.9967695 |
| natural | L1 | source | post_decoder_residual | 0.4094317 | 24 | R1-vs-R3 | 0.9204489 |
| natural | L1 | source | post_self_attention_residual | 0.4045982 | 16 | R1-vs-R3 | 0.9256303 |
| natural | L2 | auto{_ARRAY_D_L2} | post_decoder_residual | 0.3941798 | 16 | R1-vs-R2 | 0.9199764 |
| natural | L2 | auto{_ARRAY_D_L2} | post_self_attention_residual | 0.3873444 | 16 | R1-vs-R2 | 0.9227079 |
| natural | L2 | auto{_ARRAY_F_L2} | post_decoder_residual | 0.0698726 | 24 | R1-vs-R4 | 0.9975797 |
| natural | L2 | auto{_ARRAY_F_L2} | post_self_attention_residual | 0.0790025 | 32 | R1-vs-R4 | 0.9969239 |
| natural | L2 | auto{_ARRAY_T_L2} | post_decoder_residual | 0.0593575 | 8 | R1-vs-R3 | 0.9982668 |
| natural | L2 | auto{_ARRAY_T_L2} | post_self_attention_residual | 0.0603417 | 32 | R1-vs-R4 | 0.9981926 |
| natural | L2 | source | post_decoder_residual | 0.3697609 | 24 | R1-vs-R4 | 0.9341128 |
| natural | L2 | source | post_self_attention_residual | 0.3541000 | 24 | R1-vs-R4 | 0.9397290 |
| natural | L3 | auto{_ARRAY_D_L3} | post_decoder_residual | 0.2059616 | 16 | R1-vs-R2 | 0.9786420 |
| natural | L3 | auto{_ARRAY_D_L3} | post_self_attention_residual | 0.2104297 | 16 | R1-vs-R2 | 0.9780329 |
| natural | L3 | auto{_ARRAY_F_L3} | post_decoder_residual | 0.1853842 | 16 | R1-vs-R2 | 0.9827609 |
| natural | L3 | auto{_ARRAY_F_L3} | post_self_attention_residual | 0.1854251 | 16 | R1-vs-R2 | 0.9826815 |
| natural | L3 | auto{_ARRAY_T_L3} | post_decoder_residual | 0.1037660 | 8 | R1-vs-R2 | 0.9946083 |
| natural | L3 | auto{_ARRAY_T_L3} | post_self_attention_residual | 0.1042648 | 8 | R1-vs-R2 | 0.9945830 |
| natural | L3 | source | post_decoder_residual | 0.3768706 | 24 | R1-vs-R4 | 0.9274915 |
| natural | L3 | source | post_self_attention_residual | 0.3682305 | 16 | R1-vs-R4 | 0.9300935 |
| natural | L4 | auto{_PIPE_L4} | post_decoder_residual | 0.0639287 | 32 | R1-vs-R4 | 0.9981029 |
| natural | L4 | auto{_PIPE_L4} | post_self_attention_residual | 0.0696037 | 32 | R1-vs-R4 | 0.9975885 |
| natural | L4 | auto{_UNROLL_L4} | post_decoder_residual | 0.1208088 | 32 | R1-vs-R2 | 0.9958739 |
| natural | L4 | auto{_UNROLL_L4} | post_self_attention_residual | 0.0982778 | 32 | R1-vs-R2 | 0.9954891 |
| natural | L4 | source | post_decoder_residual | 0.2880319 | 16 | R1-vs-R4 | 0.9593284 |
| natural | L4 | source | post_self_attention_residual | 0.2863659 | 16 | R1-vs-R4 | 0.9594434 |
| natural | L5 | auto{_PIPE_L5} | post_decoder_residual | 0.0800915 | 32 | R1-vs-R2 | 0.9978908 |
| natural | L5 | auto{_PIPE_L5} | post_self_attention_residual | 0.0624979 | 32 | R1-vs-R2 | 0.9981623 |
| natural | L5 | auto{_UNROLL_L5} | post_decoder_residual | 0.7544874 | 24 | R1-vs-R2 | 0.7117888 |
| natural | L5 | auto{_UNROLL_L5} | post_self_attention_residual | 0.7476346 | 32 | R1-vs-R2 | 0.7376482 |
| natural | L5 | source | post_decoder_residual | 0.2108908 | 16 | R1-vs-R4 | 0.9775099 |
| natural | L5 | source | post_self_attention_residual | 0.2137893 | 16 | R1-vs-R4 | 0.9768938 |

Candidate preferences (13 fields × 3 contrasts per mode):

- natural: 4/39 choices change; 4 rankings change; 0 domains differ.
- controlled: 2/39 choices change; 4 rankings change; 0 domains differ.

Consult pairwise.csv for all cosine, relative L2, baseline/compared norms and signed relative norm changes; candidate_comparisons.csv for field-level margins and domain identity. These describe sensitivity, not optimization quality.
