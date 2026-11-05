# Framework development status

The maintained framework is `main`. Selected models and objective-specific stage
availability are defined in `FINAL_ARTIFACTS.json`; production GNN tensors remain
V1. V2.2 RC3 is a frozen research graph representation awaiting independent
semantic adjudication. Automated validation does not establish improved QoR.

## Research branches — 5 November 2026

- `research/stage1-refinement`: both Stage-1 quality-SFT V4 latency arms completed
  one training pass; six legal Kalman inference outputs are recorded. Measured
  QoR evaluation and production promotion remain outstanding.
- `research/gnn-refinement`: full-cohort corrected-v3 V4b controls, evaluation,
  and alignment interventions completed. Evidence is mixed; meaningful held-out
  device/clock preference changes are absent in two validation families. No V4b
  model is promoted. The selected next research baseline is the V2.2 RC3 canonical
  compiler graph, V4 semantic tensorization, and source-verified absolute action
  alignment, with the FPGA hardware overlay kept separate. See the [GNN guide](gnn-refinement/README.md).
- `research/refinement-mechanism`: independent development over the stable
  refinement/search utilities already available on main.

`mlir-target` and `llvm-ir` remain historical implementations. See
[experiments](experiments.md), [reproducibility](reproducibility.md), and
[development](development.md) for the maintained working interfaces.
