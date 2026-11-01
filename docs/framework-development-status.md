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
- `research/refinement-mechanism`: authenticated synthesis-feedback evaluation
  over the stable refinement/search utilities. The completed R11 assessment
  finds measured refinement gains but does not establish structural or hybrid
  superiority. Its reporting code, evidence exports and conclusions remain on
  that research branch; production models and search policies are unchanged.

## Refinement evidence assessed 2026-11-01

Research assessment commit
[`871c3d4`](https://github.com/ElenaVouvali/MailoHLS/commit/871c3d4b0c83ce0ae29fbd3fed5d2d21bad937fb)
authenticates all 144 logical observations across GAN, covariance and Kalman:
six policies/controls, seed 123, eight evaluations each, Vitis HLS 2021.1,
ZCU104 and a 10 ns requested clock. The historical experiment used 86 new
syntheses and 58 authenticated reuses. The assessment required no additional
synthesis or C simulation and loaded no LLM or GNN.

The best observed ADP improvements are 42.694% for GAN (matched control),
28.754% for covariance (compose), and 0.142% for Kalman (matched control).
Structural beats matched only on covariance; hybrid loses substantial ground
to compose there. Covariance's one explicit composition adds 0.454 percentage
points beyond its best single edit. The scientific decision is
`REFINEMENT_BENEFICIAL_BUT_STRUCTURAL_GAIN_UNPROVEN`. Winners were selected
retrospectively across six eight-query trajectories, rather than representing
a preselected policy's guaranteed performance with eight total observations.

All three selected finalists have authenticated finite C-simulation passes with
zero observed error/mismatches. Historical compiler-label quality and remaining
matching differences limit causal attribution. The three-kernel/one-seed scope,
small Kalman gain, finite functional inputs and absence of post-route evidence
remain explicit limitations. The research suite passed 143 tests with one
optional saved-tensor prerequisite skip. This main-branch update publishes only
the status and links; it promotes no experimental algorithm or checkpoint.

The pinned [results and evidence](https://github.com/ElenaVouvali/MailoHLS/blob/871c3d4b0c83ce0ae29fbd3fed5d2d21bad937fb/experiments/refinement_search/RESULTS.md)
include complete budget curves, costs, configurations, hashes and functional
receipts. The [methodology and design rationale](https://github.com/ElenaVouvali/MailoHLS/blob/871c3d4b0c83ce0ae29fbd3fed5d2d21bad937fb/experiments/refinement_search/README.md#authenticated-r11-assessment)
explain replay, authentication, controls and reproduction.

`mlir-target` and `llvm-ir` remain immutable historical implementations. Source
commits, configurations, experiment reports and verified offline preservation
replace the need for a separate branch or Release for every experiment. See
[experiments](experiments.md), [reproduction](reproducibility.md) and
[development](development.md) for the working interfaces.
