# Reproduction and Davinci handoff

Pull research/final-architecture-decision-20261109 into an isolated persistent worktree. Git contains code and compact receipts, not large checkpoints/tables/caches/memory packs. Verify SHA identities in artifact_manifest.json before use. Install pinned dependencies in an isolated environment; do not alter shared server packages.

```bash
git fetch origin research/final-architecture-decision-20261109
git worktree add ~/MailoHLS-final-architecture-20261109 origin/research/final-architecture-decision-20261109
cd ~/MailoHLS-final-architecture-20261109
export SALT_V4B_RUN_ROOT=/local/path/to/v4b-v2-20261102-8f3593b
export SALT_PREPROCESSED_DIR=/local/path/to/preprocessed_CSVS
python -m experiments.final_architecture_20261109.run audit
python -m experiments.final_architecture_20261109.run audit --execute
python -m experiments.final_architecture_20261109.run prepare --execute
python -m experiments.final_architecture_20261109.evaluate
python -m experiments.final_architecture_20261109.run screen --execute
python -m experiments.final_architecture_20261109.summarize
```

CPU environment used Python3.10, Torch2.6.0+cu124, PyG2.6.1, SciPy1.15.3. PyG/SciPy were restored into an isolated venv with shared LLM packages read-only. `audit.py` verifies cache/graph/model/source hashes, rebuilds fresh candidate tensors and replays historical ADP selection. Artifacts live under artifacts/final_architecture_20261109. Existing complete receipts are immutable; a rerun must use a new output when results differ. The report generator is deterministic and loads no LLM. Full candidate-hash set ledgers remain in the ignored raw JSON; compact Git receipts contain exact set hashes and counts.

GPU preflight:
```bash
hostname
nvidia-smi --query-gpu=uuid,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
python -m experiments.final_architecture_20261109.run baseline   --stage1 /local/v4/multi_reference/run --data-dir /local/frozen_requests   --run-dir /local/new_baseline --gpu-uuid GPU-VERIFIED-ON-THIS-HOST
```
Add --execute only after reviewing the dry-run. `requests.jsonl` must use the canonical frozen request schema. The wrapper verifies V4 hash, rejects occupied/insufficient GPUs, checks CUDA, then uses the production decoder. It neither guesses UUIDs nor terminates Ray. Liono V100S had6105MiB free and an active26382MiB allocation; Davinci SSH failed host-key verification. No GPU forward/optimizer update occurred in this run.

For static training use the portable wrapper around `experiments.rc3_stage2_multikernel.train.command`:
```bash
python -m experiments.final_architecture_20261109.run static_smoke  --stage1 /local/v4/multi_reference/run --data-dir /local/audit  --memory-dir /local/memory_c/memory --run-dir /local/new_run/smoke  --gpu-uuid GPU-VERIFIED-ON-THIS-HOST
python -m experiments.final_architecture_20261109.run static_train  --stage1 /local/v4/multi_reference/run --data-dir /local/audit  --memory-dir /local/memory_c/memory --run-dir /local/new_run/static  --gpu-uuid GPU-VERIFIED-ON-THIS-HOST --steps 500
```
Both commands default to dry-run. Add --execute after review and successful exact-decoder parity/gradient smoke. Resume static training with --resume /local/new_run/static/checkpoint-N; the production Trainer restores optimizer/scheduler/RNG state. The smoke uses one layer32 update; static training uses the historical four-layer schedule as an unqualified anchor. No automatic early plateau claim is made. Memory manifest/source alignment must pass the production loader; no old V1 memory is substituted. Changing data/layer/precision contracts requires a new run directory.

The wrapper uses explicit local V4/data/memory/output paths. First run the existing smoke command dry-run. Its historical launcher is liono-specific; do not execute its hard-coded UUID on Davinci. The new portable preflight can validate a Davinci UUID. A bounded convergence protocol must expose steps, validation curves, optimizer/RNG resume and exact decoder parity; the historical20-step pilot is insufficient. This package deliberately refuses train_finalists while those gates are unmet. It does not provide an untested automatic convergence launcher.

Synthesis remains Coroni-only. Existing sealed bundle instructions are in ../architecture-finalization/README.md. Preserve tool/version/source identities and all failed jobs. Do not run the pending bank until finite anchor latency is resolved. Revision2 spent0 new HLS jobs,0 GPU hours and0 post-route checks;2 disposable CPU FiLM optimizer steps tested gradient mechanics. Limits remain8GPUh/30initial new HLS/48maximum/6configured post-route.

No locked-test metrics or external workload outcomes were read for selection. No new checkpoint was trained or production model overwritten.
