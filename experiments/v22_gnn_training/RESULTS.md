# V2.2 target-aware GNN training results

## Preflight (completed)

- Authenticated data audit: **PASS WITH EXPLICIT EXCLUSIONS**. 104,147 eligible
  configurations: 88,466 train, 8,976 validation and 6,705 test. Exclusions and
  identities are in [`AUDIT.md`](AUDIT.md).
- Focused regression tests: **115 passed in 181.12 s** in the compatible Python
  3.10 environment (including archive authentication, graph semantics,
  tensorization, demand features, trainer contracts and resume/launcher tests).
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

**Blocked by GPU availability.** The detached launcher refused to use the A2
because an unrelated `VLLM::EngineCore` process (PID 582660) occupied it with
10,540 MiB. It then refused the V100S because a separate
`VLLM::EngineCore` process (PID 543208) occupied it with 30,598 MiB. No process
was signaled or modified, and neither smoke created a training state or read
the test split. The deterministic interruption/resume unit test passed, but it
does not replace the required real paired GPU smoke.

## Full paired training

**Not launched.** No Arm A or Arm B checkpoint, validation-selected model, or
held-out test metric exists for this protocol. The test contains Kalman only
and would be described as a narrow held-out-family evaluation. U50 and 6.67 ns
are not part of this experiment's training data.

## Current decision

`BLOCKED_GPU_AVAILABILITY`. The A2 was idle for the successful memory and
measured-gradient preflight, then both it and the V100S became occupied by
unrelated VLLM processes before the interruption/resume smoke. The launcher
correctly declined both. Re-run the smoke then full launch commands in
[`RUN_PROTOCOL.md`](RUN_PROTOCOL.md) once a compatible GPU is idle; the launcher
will select the named GPU by UUID, verify availability, and refuse to share.

No run PID or active output directory exists. Failed safe-launch attempts
produced no training data or checkpoint. The standalone audit, per-kernel CSV,
memory profile and gradient diagnostics are under
`/tmp/v22-gnn-resumable-preflight-20261030/`. Validation and test metrics remain
unreported. No QoR advantage or target-generalization claim is made from
preflight evidence.
