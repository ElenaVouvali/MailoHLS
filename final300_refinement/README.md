# SALT-HLS — final FiLM+GMT checkpoint-300 budgeted refinement (Kalman pilot)

**Scope:** One external kernel, **Kalman**, ZCU104 (`xczu7ev-ffvc1156-2-e`), **10 ns, 100%** resource budget, LATENCY. Frozen Stage-2 checkpoint **300**. **Eight logical HLS evaluations maximum including the unchanged SALT proposal**. The same historical GNN+GMT inference memory and selected FiLM checkpoint are required; no surrogate or fabricated memory is accepted.

**Status:** Packaged code passed local Python compilation and five offline tests (on 2026-11-14). Actual Liono model loading, Coroni tool access and physical HLS evaluations **have not been executed** here; server-side preflight and one-query smoke are required. Do not treat passing local tests as physical validation.

## Algorithm

1. Read the genuine final300 Kalman output; verify the saved structural-XAttn tensor SHA256, training protocol SHA256, full source, and independent frozen GMT memory.
2. Extract the **historical action-to-action relation eligibility** from the authenticated `Kalman.memory.pt`. These compiler-derived routes identify candidate action pairs; they do **not** prove memory dependence or performance benefit. Keep memory fixed throughout refinement.
3. Generate only semantically valid, source-derived, **legacy/native-domain** candidate edits with the repository's `experiments.refinement_search.policies.single_proposals` and `experiments.refinement.configurations` APIs. Consider related two-action combinations and combinations of previously measured beneficial single edits when their source reference agrees.
4. Use HLS outcome history (feasible improvements, infeasibility, explicit labeled loop-II evidence when available) to prioritize the next candidate. At evaluation 4, test a related two-action combination if one is legal and available. This is an exploratory rule, not a proven optimum.
5. In `model` mode, score at most three proposals with the **unchanged checkpoint 300 and compiler-guided GMT-XAttn** using each candidate's actual causal prefix and FiLM target conditioning. The scored quantity is **change in mean RHS log-likelihood at edited nonforced sites**, not a sequence probability or latency estimator. It receives only a bounded search-priority weight. Unscorable values remain legal HLS candidates but have no learned score.
6. Durable-reserve one complete candidate, materialize its original full C/C++ source, run Vitis HLS 2021.1 **on Coroni** and authenticate the result. If the job times out or reports undefined latency, charge the query, preserve it, and never promote it as a winner.
7. Update the best **resource-feasible, HLS-estimated-timing-qualified** measured incumbent, repeat until the eight logical-query budget is exhausted or the legal neighborhood is exhausted. No model weights or frozen memory are updated.

### Important boundaries

- This uses HLS *C synthesis* estimates, not post-route clock closure or functional equivalence validation.
- HLS **interactions are sequential** and adaptive; the synthesis jobs themselves use isolated source/Tcl/target contexts.
- `structural` (no GPU after baseline) and `coordinate` are ablation controls. Use a separate output directory for each policy. The historical GMT eligibility is an architectural prior, not a verified loop-array memory-port bottleneck.
- The `loop_ii` priority only activates if an actual CSynth XML loop entry names the exact Lk action with a numeric II; no guessed loop attribution is allowed. Otherwise the optimizer uses measured latency and feasibility feedback.
- A scored candidate outside the selected model's native RHS domain is marked `unscorable` and cannot be assigned a fictitious score. This first pilot deliberately restricts modifications to the **legacy native directive domain**; a TCAD-expanded-domain study should be separate.
- Each job is materialized and checked with the supplied `coroni_batch.py` copied from the authenticated final300 archive, with **one minimal provenance modification**: additional candidate outputs are labeled `refinement_proposal_not_synthesized`, not falsely described as original model predictions.
- On a broken remote HLS attempt, the same reserved query is resumed; incomplete runs are not silently discarded. Each policy gets its own remote directory keyed by experiment identity.

## 0. Install on Liono

From your local Ubuntu computer, assuming the downloaded archive is in `~/Downloads`:

```bash
scp ~/Downloads/SALT_final300_refinement.zip elvouvali@liono.microlab.ntua.gr:~/
ssh elvouvali@liono.microlab.ntua.gr
```

Then, on Liono:

```bash
mkdir -p "$HOME/MailoHLS"
unzip -oq "$HOME/SALT_final300_refinement.zip" -d "$HOME/MailoHLS"
export PKG="$HOME/MailoHLS/final300_refinement"
export ROOT=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final
export D="$ROOT/deployments/film-gmt-stage2-6ee1390-20261113"
export PY=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python

"$PY" -m unittest discover -s "$PKG" -p 'test_offline.py' -v
"$PY" "$PKG/setup_scorer.py"
```

These commands create no HLS jobs and do not mutate repository training/inference or checkpoint files.

## 1. Ensure checkpoint + Kalman assets + baseline inference exist

```bash
# This entire mirror may already exist from the routing ablation.
DAV=evouvali@davinci.microlab.ntua.gr
DD=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113

if [[ ! -f "$D/runs/campaign_s123_v1/final/checkpoint-0300/receipt.json" ]]; then
  mkdir -p "$D"
  rsync -a --exclude='/venv/' --exclude='/final300_eval_20261113/' "$DAV:$DD/" "$D/"
fi

P="$ROOT/artifacts/stage2_eval_9_20261113/final300_eval_20261113/predictions"
mkdir -p "$P"
if [[ ! -f "$P/zcu104_10ns_all9.jsonl" ]]; then
  rsync -a "$DAV:$DD/final300_eval_20261113/predictions/zcu104_10ns_all9.jsonl" "$P/"
fi

# The original multi-reference Stage-1 parent must still be available on Liono.
test -f /home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run/training_contract.json

test -f "$D/eval_sources_compat9/Kalman/kernel_info.txt"
test -f "$D/eval_gmt_memory_B_compat9/memory_manifest.json"
test -f "$D/runs/campaign_s123_v1/final/checkpoint-0300/structural_xattn.pt"
test -f "$P/zcu104_10ns_all9.jsonl"

ssh -o BatchMode=yes bonus24@coroni.microlab.ntua.gr 'test -x /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls'
```

If any `test` fails, **stop** and restore the corresponding original artifact. No synthetic substitutes or path guesses.

## 2. CPU preflight (zero HLS calls)

```bash
bash "$PKG/run_kalman.sh" audit
bash "$PKG/run_kalman.sh" plan
```

`audit` checks checkpoint/data/memory identities and invokes a *real local* one-job `coroni_batch.py prepare` in a temporary directory. This validates source/action/Tcl construction without opening a Vitis session.

`plan` prints initial legal proposals, with no GPU or HLS work.

## 3. GPU-only model-score smoke (zero HLS calls)

```bash
unset CUDA_VISIBLE_DEVICES
bash "$PKG/run_kalman.sh" score-smoke
```

This loads the exact frozen FiLM+GMT checkpoint300 on the Liono **V100S**, under your existing `$ROOT/artifacts/.salt_final_v100s_global.lock`, and scores up to three legal candidate/reference comparisons. It exits after saving the score slate. It **does not** perform synthesis or training. A busy GPU will cause it to wait on the lock rather than collide with the routing ablation. Check both `scoring_slate.json` and `scored.json` in the `score_smoke/` output.

**Do not start step 4 until this passes.** If it fails, inspect the full traceback; do not alter the checkpoint/receipt to satisfy it.

## 4. First *physical* Kalman HLS smoke: exactly one logical call

Confirm no old Coroni queue is processing jobs before the test:

```bash
ssh bonus24@coroni.microlab.ntua.gr "pgrep -u bonus24 -af '[v]itis_hls' || true"
```

Then, from Liono:

```bash
bash "$PKG/run_kalman.sh" baseline 2>&1 | tee "$HOME/final300_refinement_kalman_baseline.log"
```

This evaluates **the unchanged SALT prediction**. It does not replace the baseline with an invented, unmeasured configuration. The HLS job has a 3600-second per-job timeout. An HLS timeout still counts as the one logical evaluation; it does not create a fake successful measurement.

Inspect:

```bash
OUT="$ROOT/artifacts/refinement_final300_kalman_20261114/model"
cat "$OUT/summary.json"
cat "$OUT/report.md"
```

At this point the physical smoke is complete. Validate the Coroni return, provenance, and any failure before expanding the budget.

## 5. Continue the same closed-loop experiment to four and eight calls

```bash
# Three *additional* evaluations: cumulative total <= 4
bash "$PKG/run_kalman.sh" next3 2>&1 | tee "$HOME/final300_refinement_kalman_next3.log"

# Four *additional* evaluations: cumulative total <= 8
bash "$PKG/run_kalman.sh" rest4 2>&1 | tee "$HOME/final300_refinement_kalman_rest4.log"

bash "$PKG/run_kalman.sh" report
cat "$ROOT/artifacts/refinement_final300_kalman_20261114/model/report.md"
```

The three invocations use the *same* run identity and existing history. If `next3` or `rest4` is repeated, already measured steps are skipped, and new steps are bounded by `--limit` (unless earlier attempts were incomplete, which are resumed).

For a disconnected SSH session, wrap a stage under `nohup bash -c 'bash ...'` after the earlier interactive preflight; do not launch competing controllers. All runs use an OS file lock in the run directory.

## 6. Comparators for research (separate outputs and query budgets)

Use the Python driver directly with the same input and target paths but change `--policy coordinate` or `--policy structural` and `--out` to a **fresh** directory. The `run_kalman.sh` wrapper intentionally stays pinned to `--policy model` for the first real-smoke experiment. Each comparator starts from the same checkpoint-300 baseline; never share the model policy's observed history with a comparator.

## Output contents

```
artifacts/refinement_final300_kalman_20261114/model/
  identity.json               exact sources/checkpoint/memory/policy/target/budget identity
  score_smoke/{scoring_slate.json,scored.json}   GPU-only scorer smoke (no HLS)
  step_01/
    query.json                durable reservation, original SALT candidate
    proposal.jsonl            exact complete directive configuration
    bundle/                   canonical source, Tcl, checksums, pinned HLS runner
    results/                  authenticated Coroni run receipt, raw XML and log
    summary/results.jsonl     parsed physical result
    measurement.json          extracted latency/resources/estimated timing/loop-II
  step_02/ ... step_08/
  summary.json                best verified feasible measured configuration
  report.md                   per-query trajectory and missing/failed results
```

**Do not interpret** a `success` status alone as meeting resource or clock constraints. The reported winner requires all three: finite latency, component-wise resource feasibility and HLS estimated timing <= requested period.

## GitHub provenance inspected

- Branch `research/film-gmt-stage2-final-20261113` commit `6ee1390611c9047a4b8e2b07200d223b93ee8d16`: final checkpoint training `experiments/film_gmt_stage2_final/train.py`, frozen historical B-GMT memory exporter, action-domain and causal-XAttn interfaces.
- Branch `research/stage2-evaluated-20261113` commit `3a64895afed87600d1b326ab45b260cea2dfabe7`: `experiments/stage2_eval_compat9/stage2_infer_compact_views.py`, `experiments/refinement_search/policies.py`, `experiments/refinement/configurations.py` and existing Coroni refiners.
- Earlier RC3/Stage3 refine CLI is **not** directly interchangeable with this selected FiLM+GMT checkpoint. No Stage3 DPO tensor or RC3 memory is loaded.
- Your uploaded `SALT_final300_campaign_20261113(1).zip` provided the exact `stage2_infer_reference.py` and original `coroni_batch.py`. The scorer generator fail-closes on the supplied inference source SHA256. The derivative HLS runner changes only the admission of accurately labeled unsynthesized refinement candidates.

## What counts as proof of the contribution?

The Kalman pilot validates functionality and end-to-end control, **not a generalization claim**. For a TCAD comparison, run coordinate-only, structure-only and model-guided refiners with equal paid logical budgets (1,4,8) on additional held-out kernel families, confirm identical source/tool/target contexts, and report physical latency, resource feasibility, timing estimates and synthesis cost. If Kalman is part of a locked test set, avoid using its results to retune the controller and subsequently reporting it as untouched test evidence.
