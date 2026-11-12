# MailoHLS / GNOSIS Stage-1 target conditioning A/B — Davinci V100

**Status:** the 6-step Davinci V100 smoke completed for both arms. It is a *matched continuation from an authenticated common Stage-1 SFT parent*, not a clean retrain of DeepSeek-Coder from pretrained initialization. The comparison is **inconclusive**: it has one independent validation family and two complete groups. It does not qualify a final deployable Stage-1 checkpoint or physical FPGA QoR.

## Contents

- `rebuild_non_test.py`: rebuild *only* canonical, complete-source/action train/val GNOSIS rows from the restored raw CSVs. Verifies raw SHA256s against `Data/manifest.json`, stages symlinks for **non-test** kernels, runs unchanged production `Preprocessing.data_preprocess --mode llm` and `Preprocessing.create_jsonl`, and creates a new row-index split. Never opens locked-test measurements.
- `prepare_gnosis_ab.py`: selects safe, legal measured assignments, applies full/standard/boundary resource budgets, keeps up to three near-optimal feasible SFT references per context, constructs matching two-context budget and clock reversals. Requires authentic parent directive-domain registry; fails/skips invalid source/target mappings. Reports exact counts. Resource percent feasibility is **HLS-estimate proxy** only (±1 percentage-point conservative boundary guard), not exact BRAM counts, actual timing, or post-route feasibility.
- `train_gnosis_ab.py`: identical residual LoRA parent, training rows, RNG schedule, optimizer and evaluation for arms A and B. A has no FiLM; B applies 15 context features and directive-kind-aware, intermediate-layer, FP32-computed residual FiLM before later decoder computation. Gradient-checkpoint recomputation cannot see the wrong context because each candidate is backwarded before rebinding. Uses `selected_logps` from existing `experiments/final_architecture_v3/preferences_objective.py` and production deterministic RHS/value-only action-balanced supervision, each candidate on its own valid causal prefix. AdamW, fp16 AMP with GradScaler, checkpoint/optimizer/RNG state saved for resumption.
- `compare_ab.py`: paired held-out comparison, mandatory B-on vs B-off FiLM ablation, macro two-context group correctness, SFT NLL. Does not promote if there are fewer than 2 held-out contrast families or 8 complete two-context groups.
- `run_v100s.sh`: compatibility entry point that forwards Liono launches to Davinci. On Davinci it runs A and B sequentially on the pinned **Tesla V100-PCIE-32GB** and refuses another host, an occupied GPU, model mismatch, or pre-existing result directories.
- `run_davinci.sh`: verifies the remote V100, Stage-1 Python environment, authenticated parent, and cached base-model config before transferring the experiment code and already-built non-test data. It runs in an isolated Davinci work directory and copies the result artifacts back to Liono. It never selects Liono's V100S or falls back to another GPU.
- `test_film_mechanism.py`: CPU forward/backward, gradient checkpoint hook, kind-mask, and context-effect tests.

## Completed smoke run

The compact [smoke result summary](results/stage1_ab_smoke_20261112.json) records six optimizer updates per arm, gradient norms, FiLM gradient norms and gates, held-out metrics, authenticated Stage-1 parent hash, and Davinci V100 identity. Raw checkpoints and runtime caches remain outside Git.

Arm A and B both completed step 6. Arm B received nonzero FiLM gradients, but B-on did not improve the measured two-context group accuracy over A or B-off. The comparison therefore remains inconclusive; this smoke is an integration check, not evidence to promote FiLM or a final Stage-1 model.

## Exact commands

On **local laptop** after downloading `gnosis_stage1_ab.zip`:

```bash
scp ~/Downloads/gnosis_stage1_ab.zip elvouvali@liono.microlab.ntua.gr:~/
ssh elvouvali@liono.microlab.ntua.gr
```

On **liono**, start from your existing repository, create a separate worktree of the canonical data plus v3 Stage-2 code:

```bash
cd ~/MailoHLS
git fetch origin research/final-architecture-v3-20261110
git worktree add -b research/gnosis-stage1-film-ab-20261112 \
  worktrees/gnosis-stage1-film-ab origin/research/final-architecture-v3-20261110
cd worktrees/gnosis-stage1-film-ab
mkdir -p experiments/gnosis_stage1_ab
unzip -j ~/gnosis_stage1_ab.zip 'gnosis_stage1_ab/*' -d experiments/gnosis_stage1_ab/
chmod +x experiments/gnosis_stage1_ab/run_v100s.sh
```

The remote launcher uses `$HOME/MailoHLS/.venv/bin/python` on Davinci and requires the base DeepSeek-Coder model in that server's local cache. It sets `PARENT` to Davinci's `$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run`. The script requires an authenticated parent with `training_contract.json`, `adapter_model.safetensors`, tokenizer and domain registry. The runner rejects a different adapter hash and never falls back to the older final-latency path. The local checkpoint can be inspected before launch with:

```bash
export PARENT="$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run"
# If your checkpoint lives elsewhere, locate it first, then adjust PARENT.
find "$HOME" -path '*/stage1_quality_sft_latency_v4/multi_reference/run/training_contract.json' -print 2>/dev/null
ls -lh "$PARENT"/{adapter_model.safetensors,training_contract.json,directive_domain_registry.json}
sha256sum "$PARENT/adapter_model.safetensors"
# Verified final-architecture adapter SHA256:
# 28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
```

Launch the existing smoke on Liono using the same command. The compatibility entry point dispatches execution to Davinci's pinned V100 after a read-only availability and runtime preflight; it exits before transferring data if Davinci is unavailable or occupied. The launcher requires the local `gnosis_rebuilt` dataset to exist first. For a fresh `RUN_ID`, prepare it with the existing Liono research Python:

```bash
RUN_ID=stage1_ab_smoke_20261112
/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  experiments/gnosis_stage1_ab/rebuild_non_test.py \
  --out "artifacts/gnosis_stage1_ab/$RUN_ID/gnosis_rebuilt" \
  --canonical-manifest Data/manifest.json
```

Set `RUN_ID` to the same value for preparation and launch. To reuse the already-built dataset, use the existing run ID and `REUSE_REBUILD=1`.

**Fast integration smoke** (two contexts, only 6 optimizer steps; **not a meaningful model comparison**):

```bash
RUN_ID=stage1_ab_smoke_20261112 STEPS=6 CONTEXTS=2 EVAL_EVERY=2 \
  VAL_SFT=2 VAL_PAIRS=2 \
  bash experiments/gnosis_stage1_ab/run_v100s.sh
```

**Full matched experiment** in a new immutable output folder:

```bash
RUN_ID=stage1_ab_full_20261112 STEPS=720 CONTEXTS=240 EVAL_EVERY=60 \
  VAL_SFT=32 VAL_PAIRS=24 MAX_LENGTH=6144 \
  bash experiments/gnosis_stage1_ab/run_v100s.sh
```

The runner is pinned to Davinci V100 UUID `GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f` and verifies the device name, UUID, free memory, and active compute processes twice before training. `DAVINCI_SSH_TARGET` may change the SSH spelling for the same host; `DAVINCI_V100_UUID` cannot redirect work to another device. If Davinci's V100 is occupied, it exits before transferring files or creating a new result bank. If CUDA OOMs at 6144, **do not silently truncate**; choose a smaller verified source-complete cohort and make a new `RUN_ID`, or redesign batching. A different sequence-length regime is a distinct experiment. Do not relax the token alignment assertions.

**Results:**

```bash
cat artifacts/gnosis_stage1_ab/stage1_ab_full_20261112/architecture_comparison.json
cat artifacts/gnosis_stage1_ab/stage1_ab_full_20261112/bank/manifest.json
cat artifacts/gnosis_stage1_ab/stage1_ab_full_20261112/A/best.json
cat artifacts/gnosis_stage1_ab/stage1_ab_full_20261112/B/best.json
```

For a resume of *an interrupted individual arm*, use the last complete `checkpoint-NNNN` of that arm and its existing output directory. Do not rerun the top-level script, which intentionally refuses to overwrite the immutable dataset and A/B outcomes. Example:

```bash
# Run on Davinci, inside the isolated remote attempt directory.
export CUDA_VISIBLE_DEVICES=GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="$HOME/MailoHLS/.venv/bin/python"
R="$PWD/artifacts/gnosis_stage1_ab/stage1_ab_full_20261112"
"$PYTHON" experiments/gnosis_stage1_ab/train_gnosis_ab.py \
  --parent "$PARENT" --bank "$R/bank" --out "$R/B" --arm B \
  --steps 720 --train-contexts 240 --max-length 6144 --eval-every 60 \
  --val-sft 32 --val-pairs 24 --seed 123 \
  --lr 8e-5 --film-lr 3e-4 --rank-budget .2 --rank-clock .05 \
  --resume-from "$R/B/checkpoint-0060"
```

## Interpretation and limitations

**A and B both train Stage-1 from identical initial weights** (the frozen SFT parent + new residual LoRA), using better multi-reference / conditional GNOSIS supervision. This isolates whether FiLM adds value over an identically supervised LoRA. It does *not* compare a brand-new from-base Stage-1 trained with the entire GNOSIS corpus, nor is it a controlled comparison of LoRA ranks.

The selected adapter is experimental until its inference-time parser handles *exactly* the same directive-kind FiLM masks, and a matched test of **complete generated assignments** establishes feasible measured latency regret. Current script evaluates full-prefix action-weighted candidate scores, not free-running decode or synthesis. Cached decoding must explicitly carry the current directive kind for every token; otherwise FiLM application is inconsistent.

For a paper claim, report (i) independent family coverage, (ii) same-candidate two-context switch correctness and stay correctness, (iii) B ON vs OFF using the same B LoRA weights, (iv) complete-design prediction/feasibility on independent held-out programs, and (v) the nonbinding budget controls. With just 3 source-complete canonical validation kernels, the qualification may remain **INCONCLUSIVE**, and the conservative choice is A. No code guarantees FiLM superiority.

The previous final-hidden-state FiLM failed with a 1.86e-6 gate, FP16 attenuation and 54-pair no-change ablation. This script initializes modest *nonzero* gates, modulates intermediate layers before later attention, computes residuals in FP32, explicitly records FiLM gradients and ON/OFF behavior, and uses paired clock/budget training signals. These are engineering corrections, not experimental proof.

## User-facing training objective

Let `s_theta(d | P,c)` be action-normalized, causal-completion log probability, with source/action legal tokens. The SFT objective samples all near-best measured feasible references per (program, device, clock, budget), but gives each *context* equal training opportunity. For each measured two-context pair, the ranking component is `softplus(-(s_preferred-s_rejected))`. The bounded ranking weights are `0.2` for conservative budget switches and `0.05` for clock pair reversals (often not near-optimal). The total loss is SFT plus that rank loss. No unmeasured configuration is assigned a QoR label.

**V100 compatibility**: Davinci's Tesla V100 has 32GB, Volta compute capability 7.0; use NF4 frozen base, FP16 AMP and gradient checkpointing. BF16 kernels are not used. The checkpointed FiLM context is bound for the whole forward-and-backward of a candidate, avoiding a subtle source/target-context bug in naive hook implementations.

## v1.1 correction: `Empty family-disjoint SFT split` during bank preparation

The first zip had a **serialization-case bug** in `prepare_gnosis_ab.py`: it
uppercased LHS keys obtained from the source plan, but compared them against
lowercase `auto{...}` keys returned by `build_rhs_map_from_target`. This rejected
every otherwise valid source/target pair. The same mistaken casing affected the
registry-value lookup. The error occurred after the successfully completed
90,908-example, 37-kernel non-test data rebuild, **before GPU training**.

v1.1 normalizes both sides to the same uppercase LHS identity, while preserving
RHS values and the official source/registry sequential-domain semantics. It
adds five focused case/domain regression tests and per-split rejection summaries.
A real source-to-domain mismatch remains a hard rejection; we do not silently
invent directives or modify the V4 registry.

**Reuse the already rebuilt data after copying this corrected ZIP onto liono:**

```bash
cd ~/MailoHLS/worktrees/gnosis-stage1-film-ab
unzip -o -j ~/gnosis_stage1_ab_fix_20261112.zip 'gnosis_stage1_ab/*' \
  -d experiments/gnosis_stage1_ab/
chmod +x experiments/gnosis_stage1_ab/run_v100s.sh
export PARENT="$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run"
REUSE_REBUILD=1 RUN_ID=stage1_ab_smoke_20261112 STEPS=6 CONTEXTS=2 \
  EVAL_EVERY=2 VAL_SFT=2 VAL_PAIRS=2 \
  bash experiments/gnosis_stage1_ab/run_v100s.sh
```

`REUSE_REBUILD=1` verifies the saved dataset/split SHA-256 receipts, refuses
existing `bank`, `A`, or `B` directories, and skips preprocessing. Do not delete
or overwrite the previous `gnosis_rebuilt` directory. A fresh full experiment
still requires a new `RUN_ID` (without `REUSE_REBUILD`).

**Validation:** Python AST/syntax checks, Bash syntax and synthetic CPU checks
have passed for this correction. Full source/registry qualification and the
6.7B model's forward/backward must still pass on liono. If the new
`[SOURCE-DOMAIN-AUDIT]` indicates genuine parent-registry incompatibilities,
resolve them explicitly; never bypass the registry check to force training.
