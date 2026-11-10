# Execution status

**FREEZE_NOT_YET_JUSTIFIED — original A and B trained/exported; ranking-first continuation is running.**

The original Liono runs completed normally under their original feasibility-first early-stopping rule: A completed 11 epochs / 3,256 optimizer steps; B completed 12 epochs / 3,552 steps. Both exported memory and passed full-source XAttn checks. Their original selected epochs were 3 and 4, respectively. These complete runs remain untouched under `runs/liono_v3_20261110/qualified_v3/{A,B}`.

The user subsequently requested checkpoint selection by **Kendall tau-b first, log2 latency MAE second, feasible Regret@1 third**, with feasible-selection rate reported as a diagnostic. This is an explicitly post hoc policy revision; `predeclared_policy.json` retains the original policy. B has restored its own last model, optimizer, scheduler, sampler and Torch/CUDA RNG states and resumed at epoch 13, preserving 3,552 optimizer steps. Its epoch-12 state is its highest-tau historical checkpoint, so revised stale count is zero. A is queued to continue from its own epoch-11 state under the same policy. The cap remains 40 **total** epochs per arm, with patience 8; loss weights, architecture, data and optimizer schedule are unchanged.

The continuation root is `runs/liono_v3_20261110/qualified_v3/ranking_selection_v2/`. Persistent tmux session `salt_gnn_ranking_v2` runs B then A, checking idle GPU ownership before allocation and never terminating other workloads. Session `salt_gnn_ranking_v2_finish` waits for both continuations and runs the matched evaluation and selected-memory/XAttn checks. Live logs are `B.log`, then `A.log`; `resource_status.json` and `evaluation_status.json` distinguish active training from pending evaluation. No continued-run completion or improved-metric claim is made before the corresponding receipts exist.

The earlier GPU-block snapshot is retained in `receipts/liono_gpu_status.json`. The later successful idle-at-launch receipt and actual optimizer progress are in `receipts/liono_execution_progress.json`. Availability is checked again immediately before CUDA model allocation. Other workloads can subsequently appear; no other process is stopped.

An earlier Davinci A attempt genuinely trained for 11 epochs / 3,256 optimizer steps, but failed the stronger full-source-action XAttn routing qualification. It remains preserved as an unqualified result. Additional Davinci v3 progress is also preserved and excluded from the final Liono comparison following the user's explicit host instruction.

The shared v3 repair represents the five missing registered array actions in each of two Hotspot kernels as isolated source-only nodes. No compiler binding or dependence is invented. The original RC3 archive and fixed four-layer GNN architecture remain unchanged. A real maintained XAttn forward/backward probe passed for all registered source actions; that probe uses preserved weights and is not a new training run.

The final validation cohort is frozen before both Liono arms: sort-radix (L6 explicitly ignored), Viterbi, Pathfinder baseline and Pathfinder doublebuffer. It contains 13,119 observed configurations across four kernels / three families. Sort-radix comparisons concern ten authenticated action assignments; its eleventh directive is not reconstructed.

Corrected training contains 90,248 eligible unique configurations from 37 kernels / 17 families. Raw-table recovery retains 3,215 over-capacity training rows and 118 development rows before aggregation and mapping checks. Original raw percentages, zeros, measured latency and provenance are preserved. Locked-test labels remain unused.

The revised-policy four-metric comparison, checkpoint recommendation and selected memory bank remain pending completion. The original post-training finalizer failed an exact static-memory assertion after both arms completed: it omitted training's deterministic CUDA setup. The finalizer now invokes the same deterministic seed setup before inference; the exact independence check remains intact. This export failure did not stop or invalidate B's completed training. Existing models and production registrations are unchanged. Downstream LLM improvement remains unverified.

Arm A’s original-policy four-metric report, per-kernel/per-context tables and curves are in [results_A/REPORT.md](results_A/REPORT.md). That report concerns the original epoch-3 checkpoint and does not predict the revised-policy outcome.
