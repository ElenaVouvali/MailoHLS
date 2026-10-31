# Experiment index

Historical results remain at their original paths. A publication date or branch
name does not establish compatibility with selected checkpoints. Commands below
are discoverability entry points; use each guide's exact configuration and fresh
output paths before executing training or synthesis.

| Question / experiment | Implementation and reproduction entry | Data/configuration | Status and evidence |
|---|---|---|---|
| Selected SALT-HLS models | [Production launchers](shell-scripts.md); `scripts/reproduction/prepare_final_experiment.sh` | GNΩSIS, selected seed-123 split and `FINAL_ARTIFACTS.json` | Existing production V1/LLM contracts; objective-specific pending stages remain pending. |
| HARP versus MLIR structure | `scripts/experiments/representation/train_paired_gnn.sh`, `train_harp_gnosis.sh` | Script-specific paired profiles | Historical configs retained; no new comparison executed here. |
| Stage-1 collapse and recovery | [Recovery guide](../experiments/stage1_recovery/README.md); `scripts/experiments/stage1_recovery_controls.sh` | Frozen recovery inputs and measured validation; excludes silently repaired Kalman data | Experimental controls, not promoted checkpoint replacements. |
| Full-data Stage-1 quality | [Measured-quality branch](https://github.com/ElenaVouvali/MailoHLS/tree/research/stage1-measured-quality) | Original branch contracts | Assessment documents incomplete run status; not integrated into production. |
| Hardware graph learning readiness | [Readiness guide](../experiments/target_graph_readiness/README.md) | Three-family-kernel pilot and original configs/reports | CONDITIONAL GO; tensor validity and tiny learning result do not prove target generalization. |
| Target-aware V2.2 supervised learning | [Resumable branch protocol](https://github.com/ElenaVouvali/MailoHLS/blob/research/v22-gnn-resumable-20261030/experiments/v22_gnn_training/RUN_PROTOCOL.md) | 104,147 retained measured records; 88,466/8,976/6,705 split | Last committed receipt: preflight passes, full training not launched; no selected V2.2 model. Test is Kalman only. |
| Device transfer / clock adaptation | [U50 guide](../experiments/u50_adaptation/README.md); `experiments/u50_adaptation/run_device_adapt.sh` | Existing U50 assets and parent adapter contract | Separate adaptation experiment; does not establish V2.2 zero-shot performance. |
| Feedback-guided synthesis refinement | [Refinement](../experiments/refinement/README.md); `python -m experiments.refinement.experiment --mode prepare` | Versioned configurations and measured report cache | Optional synthesis study; prepare alone launches no synthesis. |
| Budgeted local search | [Controller](../experiments/refinement_search/README.md); `python -m experiments.refinement_search.controller --mode prepare` | Existing policy budgets/cache | Separate proposal policies; retains objective definitions and report identities. |
| Structural routing audit | [Audit](../experiments/structural_audit/README.md); `python -m experiments.structural_audit.audit --mode fixtures` | Compiler fixtures and existing memories | Read-only diagnostics, not prediction improvements. |
| V2.1 refinement comparison | [Guide](../experiments/v21_refinement/README.md); `python -m experiments.v21_refinement.experiment --mode prepare` | Historical frozen V2.1 inputs | Retained for reproduction, not latest compiler schema. |
| Structural memory ablation | [Guide](../experiments/asplos_memory_ablation/README.md) | ADP/Kalman configurations and fixed adapters | Historical experiment; preserve measurements, no rerun here. |

`experiments/latency_target_search/` and root `latency_target_search/` have
experiment-specific code/results; they are not assumed equivalent merely because
the names overlap. Paper scripts, notebooks, `asplos_results/` and baseline
measurements remain at their documented original locations pending individual
provenance review. See [development status](framework-development-status.md) for
additional graph/action-demand branches and [third-party provenance](third-party-provenance.json).
