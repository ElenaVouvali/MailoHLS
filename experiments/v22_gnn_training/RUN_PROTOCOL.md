> Current development branch: `research/gnn-refinement`. This document retains
> the original V3 preflight host paths and measurements as provenance. Start
> new work in a clean checkout of the consolidated branch and use your installed
> compatible Python environment. The current V4 schema requires a fresh protocol
> identity; do not resume V3 states as V4 or reuse old output directories.
> `python -m experiments.v22_gnn_training.train --help` exposes the actual CLI.

# V2.2 target-aware GNN run protocol

## Frozen inputs

- Source branch: `research/v22-gnn-resumable-20261030`.
- Base implementation: `research/v22-target-aware-gnn-training` at
  `43431de1e77efaf42692db3cfece7773541bb37c`.
- Resumability/loss-scale fix: `53f293c8b1755252db061f8ba7077ae551048971`.
- Graph schema: `salt-hls-mlir-structural-v2.2`, semantic revision RC3.
- Overlay schema: `salt-hls-target-enriched-v22-v1`.
- Tensor schema: `salt-hls-target-tensors-v3-semantic-experimental`.
- Candidate head schema: `salt-hls-v22-candidate-qor-head-v1-experimental`.
- Archive SHA256:
  `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
- Dataset SHA256:
  `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
- Seed-123 split SHA256:
  `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
- Eligible rows: 88,466 train / 8,976 validation / 6,705 test.

The launch refuses a dirty source worktree. `protocol.json` fingerprints the
source hashes, dataset/split/release identities, feature schemas, normalization
statistics, hardware/software context, optimizer, loss, sampling and command
configuration. Resume is refused if this identity changes.

## Model arms

Both arms use the existing two-layer, edge-aware V3 `TransformerConv` encoder,
64-wide hidden state, typed forward/reverse relations, semantic edge features,
static action-demand projection, semantic action summary and 128D absolute
action slots. Both use the same candidate head, complete legal assignment,
candidate-demand descriptor, target numerical context, labels and training
examples.

1. `numerical`: remove FPGA hardware-resource message edges; keep late numeric
   device, clock and deployment-budget context.
2. `hardware`: retain hardware-resource nodes and their messages, while keeping
   the identical late numeric context.

Each arm is initialized with the same seed and follows the same RNG sampling
sequence. The numerical baseline is not target-free; the comparison isolates
the added hardware-relational message pathway as closely as this architecture
allows. Candidate directives and candidate-demand descriptors do not enter
candidate-independent structural memory.

## Labels and losses

The five measured targets are latency in ms and physical BRAM_18K, DSP, FF and
LUT counts. CSV resource utilization percentage is converted using the
authenticated physical capacity from the selected device profile. The
transformed target is `log1p([latency_ms, physical_counts])`, standardized by
means and standard deviations fit on eligible training rows only. Prediction
inversion is `expm1` followed by a zero lower bound; resource predictions are
not capped at physical capacity.

The base QoR loss is mean SmoothL1 over the five standardized outputs. Ranking
is pairwise logistic within the same kernel/device/clock group and only among
measured-feasible candidates. Relative measured objective gaps of at most 5%
are ignored. `PARETO_LATENCY` uses measured latency; `PARETO_AREA` uses the
existing mean of four physical utilization fractions; `PARETO_ADP` uses
latency-ms times that area proxy. Predicted objective scores use monotonic
`log1p` forms and are divided by the median within-training-context standard
deviation of the corresponding `log1p` measured objective. This ranking scale
is fitted only from training groups. The coefficient remains 0.1:

`L = SmoothL1(QoR) + 0.1 * pairwise_logistic(rank)`.

Budgets are 100% or 50% of physical capacities. They change measured
feasibility and the requested graph context, never the underlying physical QoR
labels. Objective and feasibility comparisons never cross device/clock
contexts.

## Optimizer and sampling

- Seed: 123.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Maximum: 8,000 updates per arm.
- Candidate configurations per update: 8.
- One kernel/device/clock graph context per update; one shared encoder forward
  is reused across those candidates without detaching embeddings.
- Kernel/context is sampled uniformly; candidates are sampled without
  replacement within that context.
- One of the three existing optimization objectives and one of the two budget
  fractions is sampled per update.
- Gradient norm clipping: 1.0.
- Validation every 500 updates; deterministic configuration-hash prefix of at
  most 256 candidates per context.
- Early stopping patience: 8 validation events.
- Periodic atomic checkpoint: every 250 updates, as well as validation,
  interruption and arm completion.

Validation selection uses the deterministic bounded score already implemented
by the trainer: 0.4 feasibility error, 0.2 normalized ranking error, 0.2
bounded evaluated-pool regret and 0.2 bounded QoR error. Validation subset
regret is explicitly marked as subset regret. Test evaluation occurs only
after both arms have validation-selected checkpoints. Test metrics never
select checkpoints.

## Runtime

Use `/tmp/mailohls-v22-gnn-env-20261030/bin/python`, an isolated Python 3.10.20
venv inheriting the existing CUDA-enabled PyTorch 2.6.0+cu124 and adding PyG
2.6.1 and SciPy 1.15.3. This did not modify the environment used by the
concurrent Stage-1 run. The tested GPU is the NVIDIA A2, selected by UUID
`GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35` (15,356 MiB). The launcher checks
the device name, utilization, free memory and compute processes immediately
before detaching. It refuses busy devices by default. It does not fall back to
the V100S.

## Detached launch and operations

The output directory must be new and empty:

```bash
cd /tmp/MailoHLS-v22-gnn-resumable-20261030
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m \
  experiments.v22_gnn_training.launch \
  --gpu-name A2 \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-target-aware-gnn-s123-a2-20261030 \
  --steps 8000 --candidates 8 --checkpoint-every 250 --progress-every 25
```

The launcher writes `launcher_pid.json`, `trainer_pid.json`, `launcher_spec.json`,
`run.log`, `status.json`, `progress.jsonl` and `exit_status.json`. Inspect the
latest status and progress without attaching to the process:

```bash
cat /tmp/v22-target-aware-gnn-s123-a2-20261030/status.json
tail -n 20 /tmp/v22-target-aware-gnn-s123-a2-20261030/progress.jsonl
tail -f /tmp/v22-target-aware-gnn-s123-a2-20261030/run.log
```

Resume only after the launcher has exited and the checkpoint is present:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m \
  experiments.v22_gnn_training.launch \
  --gpu-name A2 \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-target-aware-gnn-s123-a2-20261030 \
  --resume --steps 8000 --candidates 8 --checkpoint-every 250 --progress-every 25
```

The exact resume arguments must match the run. A fingerprint mismatch or live
launcher/trainer is rejected. The launcher creates a new session and redirects
stdout/stderr to the run log, so it survives the interactive terminal closing.

## Preflight outputs

The authenticated audit and A2 profiling reports are kept outside the source
tree at `/tmp/v22-gnn-resumable-preflight-20261030/`. Their hashes and outcomes
are summarized in `AUDIT.md`; the experiment's own output directory receives a
fresh audit and protocol record at launch. No checkpoint/cache files are
committed.
