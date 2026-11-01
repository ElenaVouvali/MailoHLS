# V4 target-aware GNN run protocol

## Current frozen contract

- Research branch: `research/gnn-refinement`.
- Graph schema/release: `salt-hls-mlir-structural-v2.2`, RC3.
- RC3 archive SHA256: `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
- Overlay schema: `salt-hls-target-enriched-v22-v1`.
- Tensor schema: `salt-hls-target-tensors-v4-semantic-wide-affine-experimental`.
- Candidate QoR head: `salt-hls-v22-candidate-qor-head-v1-experimental`.
- Selected data SHA256: `afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820`.
- Seed-123 family split SHA256: `d0886fa8be67329b25dc6923bd6d1fb5e1372df60f2f0feffe6ad07f83ec360c`.
- Eligible measured records: 88,466 train / 8,976 validation / 6,705 test.
- CUDA runtime: isolated Python 3.10.20, PyTorch 2.6.0+cu124, PyG 2.6.1, SciPy 1.15.3 at `/tmp/mailohls-v22-gnn-env-20261030/bin/python`.
- Intended GPU: NVIDIA A2 UUID `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`.

V4 uses a fresh protocol identity. Never label V3 weights or training state as
V4, or resume V3 state as V4. The experiment fingerprints code, environment,
data, split, graph archive, feature schemas and training arguments. The
detached launcher refuses dirty source worktrees, mismatched resume identities,
active runs and busy GPUs. Do not use `--allow-shared-gpu`; it must not fall
back to the reserved V100S.

## Frozen matched experiment

The numerical arm removes all edges incident to hardware nodes but retains the
same numerical device/clock/budget vector. The hardware arm retains those
typed messages. Both arms use the same two-layer 64-wide TransformerConv,
128D absolute action slots, complete directive candidate, demand descriptor,
QoR labels and candidate head. Candidate directives and measurements enter
only the candidate-conditioned prediction pathway.

Frozen settings: seed 123; AdamW at `3e-4`, weight decay `1e-4`; 8 candidate
designs per update; maximum 8,000 updates per arm; ranking coefficient 0.1;
gradient clipping 1.0; validation every 500 updates, at most 256 candidates
per context; patience 8; checkpoint interval 250; budget fractions 1.0 and
0.5; latency, area and ADP objectives. Do not tune these settings after
observing validation or test outcomes.

The selected validation score is unchanged: 0.4 feasibility error, 0.2
ranking error, 0.2 bounded evaluated-pool regret and 0.2 bounded QoR error.
After both arm checkpoints are selected, run the complete validation report
and only then the held-out test report. The test split is Kalman alone and is
not an untouched benchmark; no broad family, U50 or unseen-clock claim follows.

## Restore and verify the authenticated graph release

From a clean `research/gnn-refinement` worktree:

```bash
python scripts/artifacts.py verify --artifact salt-hls-graphs-v22-20261027-rc3 --extract-to /tmp/salt-v22-training-release
sha256sum GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz
```

The extracted release must contain `release.json`; do not regenerate RC3 graphs.
Before the audit, ensure the authenticated preprocessed CSV tables are present
under `LLM_branch/Data/preprocessed_CSVS/`. These tables are ignored generated
inputs; restore only copies whose hashes match the committed preprocessing
manifest.

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.audit \
  --release-dir /tmp/salt-v22-training-release \
  --output /tmp/v22-gnn-v4-audit/data-audit.json
```

Expected audit identity and explicit exclusions are documented in `AUDIT.md`.

## Required preflight before GPU launch

Run CPU tests and the data audit first. Then verify available memory, disk,
threads and active host processes. Inspect GPU state immediately before each
GPU operation:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

The selected A2 must match the UUID above and be idle. Never touch or share the
V100S. Run V4 tensorization, memory profiling and the paired optimizer smoke
only on an idle A2. The real graceful interruption/resume smoke must complete
on a disposable V4 output before full training. If A2 is occupied, stop at CPU
validation; do not launch, wait indefinitely, or use another GPU.

The current CPU test command is:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m pytest -q \
  tests/test_v22_gnn_resume.py tests/test_v22_gnn_training_contracts.py \
  tests/test_target_graph_semantic_tensor_v22.py tests/test_action_demand_tensor_v22.py \
  tests/test_target_graph_v22.py tests/test_mlir_graph_v2.py \
  tests/test_structural_audit.py tests/test_directive_domains.py
```

The actual V4 profiling CLI is:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.profile_memory \
  --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-memory-profile.json \
  --expected-gpu A2
```

## Fresh smoke and full training commands

Run from the clean, committed research worktree. Use a new, durable output
directory and verify sufficient free space first.

```bash
cd /tmp/MailoHLS-gnn-refinement
mkdir -p /home/elvouvali/MailoHLS-GNN-runs
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.launch \
  --gpu-name A2 --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-smoke-s123 --steps 10 --candidates 8 \
  --val-every 5 --checkpoint-every 5 --progress-every 1 --skip-test
```

Only after that two-arm smoke and a separate graceful interruption/resume
smoke pass, launch the predeclared full comparison in a different empty
directory:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.launch \
  --gpu-name A2 --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-s123 --steps 8000 --candidates 8 \
  --checkpoint-every 250 --progress-every 25
```

The launcher applies its default validation cadence and patience when omitted.
No Stage-1, HLS synthesis, or second GNN run is part of this protocol.

For the disposable resume smoke, use a separate empty output and the same
arguments in both invocations:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.launch \
  --gpu-name A2 --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-resume-smoke-s123 \
  --steps 100 --candidates 8 --val-every 5 --checkpoint-every 5 \
  --progress-every 1 --skip-test
```

Wait until `progress.jsonl` records an atomic checkpoint at update 5, then
send SIGTERM only to this smoke's detached launcher PID from its
`launcher_pid.json`:

```bash
python -c "import json,os,signal; p=json.load(open('/home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-resume-smoke-s123/launcher_pid.json'))['pid']; os.kill(p,signal.SIGTERM)"
```

Confirm an `interrupted` status and a completed
`training_state.pt`. Resume with:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.launch \
  --gpu-name A2 --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-resume-smoke-s123 \
  --steps 100 --candidates 8 --val-every 5 --checkpoint-every 5 \
  --progress-every 1 --skip-test --resume
```

Verify
the completed update count and sample-identity trace continue without
duplicating update 5. Do not signal another PID or interrupt a production run.

## Monitor, resume, and verify completion

```bash
cat /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-s123/status.json
tail -n 20 /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-s123/progress.jsonl
cat /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-s123/exit_status.json
```

Resume only after both launcher and trainer have exited, the last atomic
`training_state.pt` is valid, the original source commit and protocol
fingerprint still match, and the A2 is idle:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python -m experiments.v22_gnn_training.launch \
  --gpu-name A2 --release-dir /tmp/salt-v22-training-release \
  --output /home/elvouvali/MailoHLS-GNN-runs/v22-gnn-v4-s123 --resume --steps 8000 \
  --candidates 8 --checkpoint-every 250 --progress-every 25
```

Completion requires both `numerical_selected.pt` and `hardware_selected.pt`,
matching protocol fingerprints, recorded validation-selected steps, completed
test evaluation, and `exit_status.json` with return code zero. Keep model
weights and large run logs outside Git.

## Historical V3 provenance

Earlier V3 runs used the same RC3 graph archive but a different tensor schema.
Their A2 profiling and the previous V100S pre-resumability attempt are
historical only. Neither is a V4 training result or V4 checkpoint. V4 begins
from a fresh initialization and a new empty output directory.

## V4b full-cohort action-preference protocol (2026-11-02)

This is a new protocol; it does not relabel or overwrite any V4 run. Its
frozen 16-candidate, QoR-blind, full-eligible cohort manifest is
`v4b/receipts/preference_cohort_manifest.json`; the paired preference and
target coverage receipts are beside it. Data identities are the authenticated
seed-123 split, RC3 graph release and dataset hashes recorded in
`v4b/receipts/eligibility.json`. Kalman is not loaded by the V4b runner.

Before launching, verify the selected GPU UUID, driver and zero active compute
processes; verify durable disk capacity and use a new empty output path. A2 is
the default and no automatic fallback occurs. The V100S is accepted only by
an explicit `--gpu-name V100S --gpu-uuid GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`
request; use it only when a user has authorized it and live inspection shows
it is idle. The pinned environment is Python 3.10.20, PyTorch
2.6.0+cu124 and PyG 2.6.1. The CPU-only audit command is:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.v4b.audit \
  --release-dir /path/to/authenticated-rc3-release \
  --authenticated-audit experiments/v22_gnn_training/data-audit.json \
  --output-dir /durable/path/v4b-audit-<unique-id> \
  --candidates-per-context 16
```

The bounded CUDA training command is:

```bash
CUDA_VISIBLE_DEVICES=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
SALT_HLS_GPU_UUID=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35 \
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python \
  -m experiments.v22_gnn_training.v4b.train_probe \
  --release-dir /path/to/authenticated-rc3-release \
  --authenticated-audit experiments/v22_gnn_training/data-audit.json \
  --manifest experiments/v22_gnn_training/v4b/receipts/preference_cohort_manifest.json \
  --output /durable/path/v4b-run-<unique-id> \
  --gpu-name V100S --gpu-uuid GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb \
  --updates 1500 --validate-every 250 --seed 123
```

First run a separate one-update smoke (`--updates 1 --validate-every 1`) in a
fresh output directory. The V4b entry point verifies the explicitly selected
GPU inventory/UUID, active compute processes, utilization, free memory, and
the UUID environment; it aborts rather than sharing or falling back.

The runner refuses an existing output directory and refuses to run without
CUDA. All arms share a family-balanced sampled pair trace, fixed candidate
pools, train-only QoR normalization, AdamW settings, pair loss and validation
steps. It selects checkpoints only on validation feasible-selection failure
plus feasible top-1 ADP regret, with preference accuracy as a tie-break. This
run has not been launched because the host driver was unavailable. Do not
report preflight or audit outputs as model results. The pairwise difference
arm's pool selection uses deterministic net wins; document possible
non-transitive cycles in any result. Stage-2 and production artifacts remain
out of scope.
