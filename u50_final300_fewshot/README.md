# SALT-HLS: U50 few-shot target calibration of FiLM+GMT Stage-2 checkpoint 300

## Objective and scientific protocol

**Target:** Alveo U50, `xcu50-fsvh2104-2-e`, requested 10 ns, 100% of the device resources. This is **device-transfer** at an already observed clock period (10 ns); it is **not** evidence of adaptation to an unseen frequency. Frozen source-only historical GNN/GMT representations are device-independent. The evaluation objective is latency under resource and HLS-estimated timing constraints.

**Starting model:** Final FiLM Arm B checkpoint-0600 + historical original-anchor GNN/GMT memory + compiler-guided structural XAttn Stage-2 **checkpoint-0300**. The V4 parent LoRA/token deltas, residual LoRA, original FiLM, GNN, GMT, and the four gated XAttn modules are **all frozen**. No Stage-1-only device adapter or legacy RC3/DPO checkpoint is loaded. The model remains the identical checkpoint-300 architecture.

**Unseen-target baseline:** The U50 target prompt uses the existing `<DEV=UNKNOWN>` token, accurate U50 physical resource capacities, and *two zero device-category bits* in the FiLM context `[0,0, log(clock/10), log1p(capacities)/16, budget fractions, log1p(available)/16]`. Existing known-device one-hot slots are never renamed. This explicit 15D representation is tested identically before and after adaptation. No fabricated U50 token or extra tokenizer embedding is introduced.

**Small learned adapter:** A **5×64 = 320 parameter** target-specific residual `E_U50`, initialized at zero, is injected into the existing FiLM bottleneck before the **frozen** FiLM output projection: `z = FiLM_enc(context, kind) + 0.5*tanh(E_U50[kind])`, for five directive-field kinds. This is differentiable parameter-efficient **fine-tuning** (soft target conditioning) of the full-composite model's output policy, without changing the pretrained or structural weights. A zero code is mathematically the same U50 baseline policy. It is not classic LoRA on the LLM and cannot correct an absent compiler graph.

**Offline supervision:** The already prepared repository `mailohls_u50_adaptation/artifacts/u50_10ns_sft.jsonl` contains **72** preprocessed complete-design examples across 20 kernels. They were collected from an earlier synthesis experiment and are **not** checkpoint-300 predictions. They were not all independent original HLS experiments: the repository also includes 79 success records (candidate/provenance duplicates exist). Do **not** infer 72 new syntheses or 72 independent policies. The older `experiments/u50_adaptation/run_device_adapt.sh` is Stage-1-only and is deliberately not used.

**Shots:** `4, 8, 16, 22` *previously synthesized complete designs* for training, specifically **2 measured configurations per source family**. Within eligible families, the two candidates are selected by deterministic hash before consulting their measured QoR; the feasible lower-latency design is the chosen completion. A support count of 8 means four target families × two measured configurations, **not** eight families with an unseen free oracle. Support selection is retrospective from preprocessed data. Therefore this measures **K labeled samples used**, not that a prospective engineer would need only K total HLS calls including historical candidate preselection.

**Training loss:** action-normalized, nonforced RHS chosen-design SFT plus a weighted measured pairwise logistic preference objective, using exactly the Stage-2 `bind_sample` and `gn.Encoded` interfaces. Only 320 floats receive gradients. Stage-2's target-aware ranking and structural masks remain active. Learning rate .02, 20 successful updates, evaluation every 5 steps, fixed seed123. These are **pilot hyperparameters**, not proven optimal. The baseline (update 0) is eligible for selection. The chosen checkpoint minimizes heldout U50 validation chosen-NLL; pair accuracy is the secondary key. A non-improving run honestly selects zero.

**Split:** Original U50 adaptation split is 16 training kernels, four validation kernels (three validation families). However the selected Stage-2 *historical original-anchor* memory authenticates only its original 31+2 GNOSIS source subset. The provided `audit` prints exactly which U50 training/validation kernels qualify and which do not. Expected on the published provenance is about 11 eligible calibration kernels and two validation kernels/families. This is a serious limitation; **no missing frozen memory is invented, no new GNN is trained, and no unrelated RC3 pack is substituted**. To extend all 20 U50 kernels, first qualify authentic epoch-9 source/graph/action caches and export them through the existing frozen GMT, plus source-derived directive domains; that is a separate engineering/evaluation step.

**Final test:** GAN, Kalman, chstone-jpeg, covariance from the *existing externally prepared* nine-kernel source-only historical GNN/GMT memory (`eval_gmt_memory_B_compat9`). Their measured U50 outcomes must not enter support sampling or model/checkpoint selection. Generate independent **zero-shot** and **adapted** outputs using the same checkpoint, prompt, full original source, domain policy, decoder and GMT memory. Then run **new paired U50 Vitis HLS synthesis** on Coroni; record failures, undefined latency, capacities and estimated timing. Interpret validation teacher-forced pair accuracy as a development diagnostic, never the external HLS metric.

**Hard caveat:** The original U50 measurements document synthesized latency/resource counts but not necessarily achieved timing or functional equivalence. Exact final reports should come from the new HLS synthesis. Requested period is not actual achieved period. This is a small retrospective pilot, not evidence of generalization without a positive heldout synthesis comparison and more families/seeds.

## Files

- `u50_fewshot.py`: immutable checkpoint audit, deterministic measured-pair support, PEFT FiLM code training, original FiLM-GMT decoder inference, physical-comparison report.
- `coroni_batch.py`: source-preserving authenticated final300 campaign Coroni runner, with U50 capacities/alias added; copied from the uploaded final300 campaign ZIP. Other target logic unchanged.
- `run_liono.sh`: ready-to-run `audit`, `train`, `predict-gan`, `predict`, `compare` modes.
- `run_coroni.sh`: two HLS batch runs on Coroni (zero/adapt) with manifest/source/TCL SHA verification. Explicit `CONFIRM_SYNTHESIS=YES` required.
- `test_offline.py`: no-GPU unit tests, including actual U50 synthesis-bundle preparation.

## 0. Upload and install (run on Liono)

Copy `SALT_U50_FiLM_GMT_fewshot.zip` to `/home/elvouvali/` on Liono. From Liono:

```bash
mkdir -p ~/MailoHLS/u50_final300_fewshot
unzip -oq ~/SALT_U50_FiLM_GMT_fewshot.zip -d ~/MailoHLS/u50_final300_fewshot
export PKG=$HOME/MailoHLS/u50_final300_fewshot
export ROOT=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
export PY=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
$PY -m unittest discover -s "$PKG" -p test_offline.py -v
```

**Prerequisites:** The deployment mirrored from Davinci must exist under:

```text
$ROOT/deployments/film-gmt-stage2-6ee1390-20261113/
  runs/campaign_s123_v1/final/checkpoint-0300/{receipt.json,structural_xattn.pt}
  eval_sources_compat9/{GAN,Kalman,chstone-jpeg,covariance}/
  eval_gmt_memory_B_compat9/memory_manifest.json
```

Other pinned Liono files are `PARENT=/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run`, `FILM_CP=/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B/checkpoint-0600` and `$ROOT/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B/`. All must exist; no new training is performed during preflight. The dataset must also be present under `$ROOT/mailohls_u50_adaptation`. If the worktree lacks it, copy the exact repository-tracked `mailohls_u50_adaptation` tree from a matching checkout; never fabricate SFT rows.

## 1. CPU-only audited preflight (zero GPU, zero synthesis)

```bash
SHOTS=8 SEED=123 MODE=audit bash "$PKG/run_liono.sh" | tee "$HOME/u50_audit.log"
```

Check the `blocked_kernels` report carefully. Failures mean missing or mismatched authenticated sources/memory, not a reason to bypass checks. The 8-shot bank requires at least four eligible *independent* training families and two validation families. The CPU audit authenticates the checkpoint, input source hashes, historical GMT file checksums, family separation and measured-example count. It does **not** prove the entire GPU decoder works. GPU run provides remaining functional preflight.

## 2. Train 320 U50-only parameters on Liono's V100S

```bash
SHOTS=8 SEED=123 MODE=train bash "$PKG/run_liono.sh" \
  2>&1 | tee "$HOME/u50_fewshot_8_seed123_train.log"
```

This takes the existing `.salt_final_v100s_global.lock` and fails if another cooperating Liono job holds it. Do not run it concurrently with your routing ablation. Writes:

```text
$ROOT/artifacts/u50_fewshot_20261114/shots8_seed123/
  audit.json
  protocol.json
  training_trace.json
  result.json
  u50_code.pt
```

Only `u50_code.pt` (~320 floats plus receipt) is learned. `result.json` reports whether validation actually selected an adapted checkpoint; if not, the zero-code baseline remains selected. No test outcomes read.

## 3. Real external inference smoke on GAN (zero HLS calls)

```bash
SHOTS=8 SEED=123 MODE=predict-gan bash "$PKG/run_liono.sh" \
  2>&1 | tee "$HOME/u50_fewshot_gan_predict.log"
```

Produces independent one-row `zero_shot.jsonl` and `few_shot.jsonl` for GAN, with both directives and provenance; same source and checkpoint SHA.

## 4. One GAN HLS smoke (two HLS jobs)

Inspect Coroni availability first:

```bash
ssh bonus24@coroni.microlab.ntua.gr "pgrep -u bonus24 -af '[v]itis_hls' || true"
```

Only if no conflicting job is running, execute from Liono:

```bash
export OUT="$ROOT/artifacts/u50_fewshot_20261114"
SHOTS=8 SEED=123 \
PRED="$OUT/smoke_gan_shots8_seed123" \
HLS_ROOT="$OUT/gan_smoke_hls" \
REMOTE_TAG=smoke CONFIRM_SYNTHESIS=YES \
  bash "$PKG/run_coroni.sh" \
  2>&1 | tee "$HOME/u50_fewshot_gan_hls.log"
```

This runs two HLS jobs (zero vs adapted), subject to specified timeouts; failure and undefined-latency reports are retained. They are **not** assumed successful.

## 5. Full independent generation and synthesis (four kernels × two policies)

```bash
SHOTS=8 SEED=123 MODE=predict bash "$PKG/run_liono.sh" \
  2>&1 | tee "$HOME/u50_fewshot_full_predict.log"

SHOTS=8 SEED=123 REMOTE_TAG=full CONFIRM_SYNTHESIS=YES \
  bash "$PKG/run_coroni.sh" \
  2>&1 | tee "$HOME/u50_fewshot_full_hls.log"

SHOTS=8 SEED=123 MODE=compare bash "$PKG/run_liono.sh"
```

Result:

```text
$ROOT/artifacts/u50_fewshot_20261114/
  predictions_shots8_seed123/{zero_shot,few_shot}.jsonl
  hls_shots8_seed123/{zero_shot,few_shot}/summary/results.jsonl
  comparison_shots8_seed123.json
```

The comparator only computes a numerical speedup when both configurations have finite positive HLS latency, report feasibility, and estimated target clock ≤10 ns. Cases with timeouts or timing failures are reported but not assigned fictitious speedups. It reports the geometric mean over the *comparable subset only*, whose denominator must always be disclosed. A robust conclusion additionally needs the overall feasibility rate.

## 6. Shot curve and seeds

Repeat steps 1–5 for shot counts `4`, `8`, `16`, `22` and seeds `123`, `456`, `789`, on **fresh paths**. Use paired source/evaluation contexts and the same hyperparameters across shots. If K=22 fails because fewer than eleven independent memory-qualified families are available, report it as not covered; never substitute a different representation. Checkpoint selection must only inspect the U50 validation families. Locked external test outcomes must not be used to tune K, seed, LR, epochs or hyperparameters. Instead precommit the full protocol and report all outcomes.

## Novelty and limitation

This is a device-specific *context bottleneck* adaptation, not a new LLM training stage nor an online HLS refinement controller. The hypothesis is that the original FiLM context map has learned transferable hardware structure, and only a small correction is needed to accommodate an unseen device. Because the historical GNN/GMT memory stays source-derived and frozen, all improvement can be attributed to the U50 code on top of the identical baseline. This clean attribution is the strongest design choice in the pilot.

An instructive second experiment, **only after the primary few-shot curve is fixed**, is a matched trainable FiLM-layer or residual-attention LoRA arm. That could test whether 320 parameters are too restrictive, but needs stronger overfitting controls and matched syntheses. Do not silently compare against the older U50 Stage-1 adaptation or publish training-NLL as an actual hardware-speedup claim.


## Domain-qualified measured U50 records (2026-11-14 correction)

The original retrospective U50 preprocessing contains measured directive
values that are *not representable* by the frozen checkpoint-300 source RHS
menu. In particular, `rodinia_pathfinder_4_doublebuffer_0` has one measured
`auto{_UNROLL_L1} = 1` design, while the frozen domain is `0,2,4,8,16`.
This is a data-contract mismatch, not an exception to suppress.

`audit` and `train` now apply the **same CPU domain qualification** using
the checkpoint parent's original domain registry and the repository's
source-derived extension for otherwise missing kernels. Unsupported measured
rows are excluded with their original target hashes and exact rejection
reasons in `domain_exclusions`; no measurement or directive value is rewritten.
Training support pairs are reselected **after** that qualification. Both
validation families are still required.

Furthermore, the original eight-shot pilot included an exact QoR tie for
`rodinia-hotspot-2-pipeline`. This version will not call equal- or nearly
equal-latency configurations a preference. It excludes pairs with <0.5%
relative latency separation and continues deterministically to the next
eligible family; this retrospective quality filter is disclosed and the
number of measured designs remains exactly the declared `SHOTS`. A support
budget that cannot be populated from legal, distinguishable measurements
fails closed, rather than silently changing the number of shots.

A previously created run should not be reused across package versions. The
training protocol records `adapter_source_sha256`, support IDs and the exact
selected validation families. Preserve any older log, but use a new run name
for a corrected campaign if an earlier run completed.
