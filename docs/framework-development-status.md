# Framework development status

The maintained framework is `main`. Selected models and objective-specific stage
availability are defined in `FINAL_ARTIFACTS.json`; production graph tensors and
Stage-2 checkpoints remain on the V1 contract. Research-branch outcomes do not
change production artifacts by themselves.

## Research branches — 5 November 2026

- `research/stage1-refinement`: the quality-SFT V4 single-reference and
  multi-reference latency runs completed one pass. Their final root adapters
  produced six complete source-legal Kalman outputs. Those probes do not show
  measured QoR or adapter superiority; no adapter is promoted.
- `research/gnn-refinement`: full-cohort corrected-v3 V4b controls and alignment
  interventions completed. Results are mixed, with insufficient target-preference
  coverage in the two held-out families. Its selected next research baseline is
  the V2.2 RC3 canonical compiler graph, V4 semantic tensorization, and
  source-verified absolute action alignment, with hardware overlay separate. See
  the [pinned GNN guide](https://github.com/ElenaVouvali/MailoHLS/blob/aa6ad1a20239250fc9236a3e87b35b1df99e6ca9/docs/gnn-refinement/README.md).
- `research/refinement-mechanism`: independent work over the stable
  refinement/search utilities already available on main.

`mlir-target` and `llvm-ir` remain historical implementations. See
[experiments](experiments.md), [reproduction](reproducibility.md), and
[development](development.md) for maintained working interfaces.
