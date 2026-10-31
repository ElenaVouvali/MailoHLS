# V2.2 target-aware GNN training results

## Preflight (completed)

- Authenticated data audit: **PASS WITH EXPLICIT EXCLUSIONS**. 104,147 eligible
  configurations: 88,466 train, 8,976 validation and 6,705 test. Exclusions and
  identities are in [`AUDIT.md`](AUDIT.md).
- Focused regression tests: **112 passed** in the compatible Python 3.10
  environment (the final resumability/loss-specific subset separately reports
  15 passed after the latest training-contract change; full rerun follows).
- A2 CUDA operation: succeeded on NVIDIA A2 using GPU UUID
  `GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35`.
- A2 small/medium/AES forward/backward memory profiles: passed without OOM.
  AES peak allocated memory was 656,113,664 bytes; CPU tensor payload was
  60,297,588 bytes.
- Real measured-candidate updates: both arms and all three objectives had
  finite losses and gradients. Candidate/candidate-demand gradients were
  nonzero; hardware relation gradients were nonzero only in the hardware arm.
- Ranking diagnostic: initial raw-unit scaling suppressed the ranking
  gradient, so scoring now uses a training-only median within-context standard
  deviation of monotonic `log1p` objectives. The coefficient remains 0.1.
  The measured ranking-to-QoR gradient ratios are in the raw A2 profile cited
  in [`AUDIT.md`](AUDIT.md); they remain small at initialization and are not
  evidence of model quality.

## Paired interruption/resume smoke

**Pending.** This must complete at least one validation event, be interrupted
with SIGTERM, resume from its atomic checkpoint under the same fingerprint, and
finish both validation selections without reading the held-out test partition.

## Full paired training

**Pending.** No completed Arm A or Arm B checkpoint, validation-selected model,
or held-out test metric is reported here until files are present and verified
in the detached run output. The test contains Kalman only and will be described
as a narrow held-out-family evaluation. U50 and 6.67 ns are not part of this
experiment's training data.

The run directory, PID, GPU, progress, and exit status will be filled from the
launcher and process records after launch. Validation and test metrics will be
reported separately. No QoR advantage or target-generalization claim is made
from preflight evidence.
