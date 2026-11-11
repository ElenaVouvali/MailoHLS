> Revision3 completed the corrected action-normalized multi-reference baseline (150updates, selected50). Four heads×64,8,683,524trainable scalars; no nine-request assignment change versus Stage1. Same-route evidence shuffle is a mathematical placebo; cross-route interventions slightly change scores but no choices on two reference cases. No fair layer/gate-layout superiority claim follows.
>
> Current evidence: [Revision-3 results](RESULTS_V3.md); [technical architecture](FINAL_SALT_HLS_ARCHITECTURE.md).

# Cross-attention qualification

The existing gated block and `post_self_attention_residual` native layernorm pre-hook preserve the decoder MLP. RC3 already corrected native residual dtype handling. Selected RHS positions use forward-filled absolute action IDs, including the causal position scoring the first RHS token. A single prefix vector does not determine a multi-token candidate score.

The focused suite passed56 tests including historical V4b, structural attention and multikernel contracts, plus two new matched-pool regressions. This is CPU correctness evidence. The inherited real-decoder zero-gate parity receipt is historical; no new real LLM parity run occurred here.

The previous C pilot20 updates selected step0 with unchanged validation accuracy. It does not establish convergence or fusion inferiority. No gate, placement, routing or memory format is newly selected. A/B action memory and variable-length C evidence are separate schemas. Metadata type/status is not automatically consumed as learned features.

First Davinci experiment: exact V4 zero-gate decoder parity, one-batch gradient/update smoke, then one resumable static-memory run to a validation plateau or declared500-update/8-GPU-hour bound. Record gate and Q/K/V gradients after the first update, skipped updates, structural/residual norms, natural complete decode, and controlled-prefix authentic/gate-off/shuffled/witness-disabled interventions. Compare small nonzero initialization only with identical projection weights and schedule. Then compare matched placements; do not reject untrained alternatives. No convergence-qualified baseline completed in this run.
