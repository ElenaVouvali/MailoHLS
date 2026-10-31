# Framework development status

The maintained framework is `main`. Selected models and objective-specific stage
availability are defined in `FINAL_ARTIFACTS.json`; production GNN tensors remain
V1. The frozen MLIR V2.2 RC3 representation still awaits independent human
semantic adjudication. Its automated validation does not establish improved QoR.

The three ongoing research branches are:

- `research/stage1-refinement`: full-data recovery, decision-aware supervision,
  measured-quality diagnostics and resumable experimental training. No recovery
  adapter is promoted to production; Kalman corrections remain a separate variant.
- `research/gnn-refinement`: static/candidate demand, semantic tensorization,
  coverage audits and resumable candidate-conditioned GNN experiments. The latest
  coverage variant has its own tensor schema; historical V3 preflight receipts
  remain historical and are not relabeled as V4 results. The last committed full
  paired-training receipt reported no selected checkpoint. This is not a live
  process monitor or a new QoR/generalization evaluation.
- `research/refinement-mechanism`: independent development over the stable
  refinement/search utilities already available on main. No new algorithm is
  introduced merely to differentiate the branch.

`mlir-target` and `llvm-ir` remain immutable historical implementations. Source
commits, configurations, experiment reports and verified offline preservation
replace the need for a separate branch or Release for every experiment. See
[experiments](experiments.md), [reproduction](reproducibility.md) and
[development](development.md) for the working interfaces.
