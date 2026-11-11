# Selected specification and evidence

This is a research specification. `implemented` identifies executable code;
`tested` identifies contract checks; `measured` identifies actual study receipts;
`unqualified` means the component has not earned default deployment; `planned`
identifies work without a qualified implementation. Flags may coexist. Tests
establish interface behavior, not hardware improvement. Current execution checks
are recorded in [consolidation status](CONSOLIDATION_STATUS.md).

| Component / selected interface | Status | Implementation and primary evidence | Confidence and reversal condition |
|---|---|---|---|
| Source validity and absolute `Lk → k−1` action registration | implemented, tested | [Domains](../LLM_branch/common/directive_domains.py), [memory](../LLM_branch/common/structural_memory.py), [structural audit](evidence/latency_gnn_structure_audit.md) | Binding is a required interface. Missing compiler binding remains source-only; gaps remain masked. Any source/hash mismatch invalidates the export. |
| Native RC3 compiler graph, relation certainty and provenance | implemented, tested | [Graph builder](../GNN_branch/mlir_graph_v2.py), [RC3 manifest](graph-release-v22-rc3.json), [graph analysis](evidence/graph_encoder_analysis.md) | Facts belong to their pinned frontend/schema. Unknown dependence is never independence. A matched source/coverage failure requires repair and requalification. |
| Historical V1 four-layer encoder and registered models | implemented, measured | [Registry](../FINAL_ARTIFACTS.json), [encoder analysis](evidence/graph_encoder_analysis.md) | Preserved historical lineage. Historical ADP qualification does not qualify the new latency architecture. |
| RC3 two-layer encoder and 128D memory used by the corrected fused-policy trial | implemented, tested, measured, unqualified | [Fusion receipt](evidence/receipts/corrected_graph_fusion.json), [fusion analysis](evidence/graph_fusion_analysis.md) | Executable control; no demonstrated latency gain. Never substitute another 128D encoder without fresh memory and fusion training. |
| New four-layer latency-GNN candidate | implemented, tested, measured, unqualified | [Pushed initial report](evidence/latency_gnn_initial_results.md), [local completed study](evidence/latency_gnn_completed_local.md) | Completion says `FREEZE_NOT_YET_JUSTIFIED`; selected preliminary checkpoint is not a production promotion. Reconsider only after independent-family matched end-task qualification. |
| Target-conditioned Stage-1 decoder | implemented, measured; release replay not performed | [SFT loss](../LLM_branch/train/stage1_decision_loss.py), [protocol](evidence/training_and_reproduction.md) | Preserve objective, part, clock, absolute budget, tokenizer and adapter identity. New complete real-checkpoint decoding is still required. |
| Gated graph cross-attention | implemented, tested, measured, unqualified | [Fusion](../LLM_branch/common/structural_xattn.py), [completed trial](evidence/latency_experiment_report.md) | Corrected step 50 changed 0/9 generated assignments versus Stage 1. Require measured feasible-latency benefit with wrong-route/zero-gate controls. |
| Full-policy offline DPO against exact frozen reference | implemented, tested, measured, unqualified | [Preference runner](../experiments/final_architecture_v3/train_preferences.py), [receipt](evidence/receipts/whole_configuration_preference_training.json) | Mandatory to qualify; current DPO and SFT each achieved 25/54 pair accuracy. DPO changed 0/9 assignments. Do not promote this pilot as default. |
| Complete legal RHS decoding | implemented, tested | [Decoder](../LLM_branch/inference/eval_stage1_stage2_stage3.py), [generated comparison](evidence/receipts/generated_assignment_comparison.json) | Preserve full causal suffix, newline and mean score reduction. Source legality alone does not establish synthesis or functional correctness. |
| Optional measured refinement | implemented, tested, measured, unqualified | [CLI](../LLM_branch/inference/optimize.py), [physical receipt](evidence/receipts/refinement_physical_validation.json) | Disabled means zero oracle/SSH/HLS calls. Enabled uses a finite charged ledger and Coroni identity. A successful matched 1/4/8-query comparison is still needed. |
| Deep decoder/query FiLM and richer learned typed witnesses | planned | [Design note](FILM_GMT_AND_HISTORICAL_ENCODER.md), [neutral final-head FiLM pilot](evidence/hardware_conditioning_ablation.md) | Architectural hypotheses. Require deployed-precision and independent measured switch/stay controls before promotion. |

Every copied report and JSON has an authenticated source entry in
[SOURCE_MAP.tsv](evidence/SOURCE_MAP.tsv). The later local completion is separately
identified by [LOCAL_SOURCE_MAP.json](evidence/LOCAL_SOURCE_MAP.json). Original
reports retain historical terms and relative links; open their original source
commit when following those links. See [limitations](LIMITATIONS.md) for the
remaining scientific and release gates.
