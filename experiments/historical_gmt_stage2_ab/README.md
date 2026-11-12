# Historical structural-memory Stage-2 A/B on Liono

This ports the supplied A30 package to Liono's UUID-selected **A2 (15 GiB)**.
The independent original LLVM reconstruction continues on Liono's V100S.
No Davinci training is launched by these scripts.

For the measured validation workload, reasons for the silent `0/2` smoke phase,
and settings to review before a full run, see
[VALIDATION_COST.md](VALIDATION_COST.md). It also explains the existing early
stopping policy and why 300 updates are a cap rather than a guaranteed matched
realized budget. The running launch is unchanged.

A uses the authenticated historical epoch-9 four-layer GNN's 64x128 action
anchors. B preserves those exact anchors and adds up to three typed, learned
compiler-evidence summaries per action. GMT receives compiler-only auxiliary
reconstruction training on Stage-2 training families, then stays frozen during
Stage-2. This bounded comparison does **not** jointly train GMT through the LLM
objective. Both arms use the same frozen V4 parent, split, fusion initialization,
training budget and legal directive domains. It does not establish physical QoR.

The historical encoder SHA-256 is
`34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623`.
The V4 parent SHA-256 is
`28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`.
Both were authenticated locally. CPU checks passed strict historical loading,
source/action and graph matching, and validation-family separation from GNN
pretraining. The matched cohort has **14 training families and 2 validation
families**, 948/36 raw rows. One CFD kernel is excluded because its evidence
count exceeds the existing 8,192-record limit; evidence is never truncated.
All 984 source/answer rows pass the conservative 4,096-token check (maximum
3,811 tokens). The trainer selects 237 train and 36 validation examples using
its existing supervision policy.

The smoke preparation completed 40 auxiliary GMT updates and exported both
memory banks. The preparation receipt is
[liono_smoke_preparation.json](liono_smoke_preparation.json). It is not evidence
of a completed Stage-2 comparison. The first smoke attempt stopped before a training update
because the supplied 4,096 context limit violated the frozen parent's 7,168 contract. The retry uses 7,168; actual row lengths are unchanged.
Learned LLM alignment and A/B conclusions
remain unavailable until the actual training and evaluation finish.

## Launch and monitoring

Run from `/home/elvouvali/MailoHLS/worktrees/historical-mlir-gmt-gemv`.
The executing package is
`/home/elvouvali/MailoHLS/experiment_packages/historical_gmt_ab_liono`;
these tracked scripts preserve its source. `run_smoke_then_full.sh` first runs
2 Stage-2 updates per arm (GMT 40 updates), then starts 300 updates per arm for
seeds 123 and 456 (GMT 600 updates) **only after smoke success**. Existing run
IDs are protected against overwrite. Do not start a duplicate while the current
launcher is alive.

```bash
bash experiments/historical_gmt_stage2_ab/run_smoke_then_full.sh
tail -f /home/elvouvali/MailoHLS/experiment_packages/historical_gmt_ab_liono/liono_a2_smoke_retry2_then_full.log
```

The launcher refuses a busy A2 and requires at least 14,000 MiB free. It uses
`/home/elvouvali/.hls-llm/bin/python` for historical GNN/GMT preparation and
`/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python` for Stage-2 training.
Neither environment is modified. The A2 uses the trainer's existing BF16/NF4
support. An OOM or context failure stops the sequence rather than changing the
matched protocol silently.

Outputs are under `artifacts/historical_gmt_stage2_ab/` in the historical
worktree, with run IDs `historical_gmt_ab_liono_a2_smoke_retry2_20261112` and
`historical_gmt_ab_liono_a2_full_20261112`. Weights remain outside Git.
`summarize.py` creates the comparison from actual validation reports only.

The original package had a checkpoint-directory/file path bug, now fixed.
Its historical audit filter is retained as a patch and as the worktree's
`experiments/rc3_stage2_multikernel/audit.py` change. The audit uses RC3 coverage
only to screen the supervised cohort; historical graph/source authentication
is performed separately and historical GNN memories are never replaced by RC3.

**Physical QoR improvement: NOT EVALUATED.** GMT summaries remain provisional;
comparison results, summary-removal interventions and exact HLS synthesis are
needed before any usefulness or hardware-improvement claim.
