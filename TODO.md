# MailoHLS GNN Embedding Roadmap

Status: Stage C completed; causal Stage-2 memory validation is the next gate.

Repository basis: `ElenaVouvali/MailoHLS` commit `9fa010eb44e3a450177b5b3925200d3b9c854fae` (13 August 2026).

## Objective

Train the GNN to provide structural and pragma-response information that improves the LLM's design choices on unseen kernels. An embedding is useful only if aligned, real GNN memory improves held-out-kernel decisions over both no information and deliberately misaligned information.

## Current evidence and limits

- [x] Stage C tiny-overfit proves the model and targets can fit a small same-kernel sample: latency MAPE 4.81%, latency Kendall tau 0.920; area MAPE 3.98%, area tau 0.962.
- [x] Full Stage C learns useful average relative effects: best validation loss ratios versus the zero-delta predictor were approximately 0.558 for performance and 0.444 for area.
- [x] Full Stage C does **not** establish robust unseen-kernel transfer. The held-out double-buffered Pathfinder case had the wrong mean performance-effect sign, latency tau about 0.094, and a loss ratio about 1.289 versus zero delta.
- [ ] Do not claim that the exported node memory is useful to the LLM until the four-arm causal experiment below passes.
- [ ] Keep Serrano locked as final test data. Do not use it for architecture, checkpoint, threshold, or hyperparameter decisions.

## Target-mode decision — keep `reference_delta`

- [x] Keep HARP's useful representation split: encode the pragma-free program first, apply NPT as the transformation path, and reason from both program structure and the transformation-induced change.
- [x] Keep Stage C `reference_delta` as the supervised QoR target while an authenticated neutral measurement is available.
- [ ] Do not revert to HARP's default absolute/global-reference target merely to be closer to HARP. That would reintroduce large between-kernel scale variation without strengthening the structural representation.
- [ ] For an unknown kernel, require one neutral result from the same source, Vitis version, FPGA part, clock, and metric definition to recover calibrated absolute QoR.
- [ ] If MailoHLS must support a strict zero-HLS-call mode, add a separate learned static-baseline head with uncertainty and retain the measured-baseline mode as the calibrated path. Do not replace Stage C before that mode is evaluated.
- [ ] Treat `reference_delta` and HARP's P/T separation as complementary, not competing: the former defines the target; the latter defines how structure and pragma transformation are represented.

## Priority 0 — Freeze the experimental contract

- [ ] Record the exact Stage C checkpoint path and SHA-256 hash. The checkpoint itself is not in the GitHub results commit.
- [ ] Record the Git commit, dataset manifest/hash, Vitis version, device `xczu7ev-ffvc1156-2-e`, clock period `10.0 ns`, split JSON, and all seeds.
- [ ] Use `LLM_branch/train/train_SFT_xattn_new.py`; do not mix it with the older training script.
- [ ] Extract a shared prompt/model contract used by both current training and inference. The current inference script uses an older prompt and defaults to cross-attention every 16 layers, while current training uses target/device-aware prompts and defaults to every 8 layers.
- [x] Save the common `training_contract.json` beside every checkpoint, with a Stage-2 `structural` section for memory and cross-attention configuration.
- [x] Make inference load the common contract and fail on incompatible structural overrides or missing/unexpected cross-attention keys.
- [ ] Use one frozen Stage-1 adapter and one saved split for every memory arm.
- [ ] Verify identical initial Stage-2 trainable weights across arms by hashing the newly initialized cross-attention/gate state before the first optimizer step.

## Priority 1 — Causally test the memory path

### 1. Export exact static pre-NPT memory

- [ ] Add `Net.forward_static_node_embed(data)` in `GNN_branch/model.py`.
- [ ] Return the JKN-combined structural node tensor immediately before NPT and the post-NPT convolution.
- [ ] Preserve the existing `forward_node_embed()` behavior for compatibility.
- [ ] Add `--embedding_mode {current_zero_mask,static_pre_npt}` to `GNN_branch/build_harp_memory.py`.
- [ ] `current_zero_mask`: reproduce the existing zero-value/zero-scope export exactly.
- [ ] `static_pre_npt`: call `forward_static_node_embed()` explicitly; do not rely on zero masks to bypass NPT.
- [ ] Keep the same slot ordering, clipping rule, padding, dtype, and maximum slots in both banks.
- [ ] Save explicit metadata: `embedding_mode`, checkpoint path/hash, critical GNN config fingerprint, Git commit, source PT manifest hash, dimension, maximum slots, and slot labels/categories.

Minimal model shape:

```python
def forward_static_node_embed(self, data):
    return self._node_embed(data, embedding_mode="static_pre_npt")

def forward_node_embed(self, data):
    return self._node_embed(data, embedding_mode="conditioned")

def _node_embed(self, data, embedding_mode="conditioned"):
    # existing encoder, convolutions and JKN
    static_node_embeddings = out
    if embedding_mode == "static_pre_npt":
        return static_node_embeddings
    if embedding_mode != "conditioned":
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}")
    # existing NPT and post-NPT convolution
    return out
```

Place the early return at the exact point where the normal forward path currently assigns `static_node_embeddings`, before NPT.

### 2. Derive controlled ablation banks

- [ ] Add `GNN_branch/make_memory_ablation.py`.
- [ ] Derive both controls from the exact static bank:
  - `zero`: zero every vector and set every memory mask to false while preserving tensor shapes and metadata.
  - `shuffle_slots`: permute only active vectors within each kernel; keep masks, tensor shapes, slot count, vector multiset, norms, labels, and slot categories fixed.
- [ ] Use a stable SHA-256-derived seed from `(global_seed, kernel_name)`, not Python's randomized `hash()`.
- [ ] Shuffle once when creating the bank; never reshuffle by batch or epoch.
- [ ] Record kernels with fewer than two active slots, for which a within-kernel permutation is impossible.
- [ ] Refuse accidental output-directory overwrite.
- [ ] Optionally add a secondary cross-kernel shuffle later, matched by slot category and slot count. Do not use it as the primary control because it introduces schema and distribution confounds.

### 3. Validate all banks before GPU training

- [ ] Add `GNN_branch/validate_memory_bank.py`.
- [ ] Assert identical kernel file sets, dimensions, maximum slots, labels, categories, and finite tensors across arms.
- [ ] Assert current/static masks and slot ordering match.
- [ ] Assert the zero bank has no active memory.
- [ ] Assert each shuffled kernel preserves the active-vector multiset and mask but differs whenever at least two non-identical vectors exist.
- [ ] Emit a bank-level `memory_manifest.json` and SHA-256 file hashes.

### 4. Run the four Stage-2 arms

- [ ] Arm A: `zero` bank — cross-attention remains present and trainable, but receives no information. This is the clean no-information control.
- [ ] Arm B: existing `current_zero_mask` / post-NPT bank.
- [ ] Arm C: exact `static_pre_npt` bank.
- [ ] Arm D: deterministic `shuffle_slots` bank.
- [ ] Do **not** compare Arm A to a Stage-1-only architecture as the primary test; that changes both information and model capacity/training path.
- [ ] Also report the frozen Stage-1-only model as a secondary operational baseline, but do not substitute it for Arm A in the causal four-arm comparison.
- [ ] Hold fixed Stage-1 adapter, split, rows, sampler order, candidate seeds, prompt, tokenization, optimizer, schedule, epochs, hardware, and Stage-2 initialization.
- [ ] Start with `PARETO_ADP`, seed 123, as the inexpensive causal smoke test. `PARETO_ADP` is the balanced objective name in `train_SFT_xattn_new.py`; `PARETO_KNEE` belongs to the older trainer/scripts.
- [ ] If a directional signal appears, repeat all production objectives and at least seeds 123, 456, and 789.
- [ ] Treat held-out kernels as the statistical unit; do not inflate confidence by treating many rows from one kernel as independent.

Suggested single-stage Stage-2 settings for each bank:

```bash
python -u LLM_branch/train/train_SFT_xattn_new.py \
  --run_mode single \
  --objective PARETO_ADP \
  --init_adapter_dir "$STAGE1_DIR/best_custom_stage1" \
  --split_json "$SPLIT_JSON" \
  --memory_dir "$MEMORY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --initial_state_reference runs/memory_ablation/PARETO_ADP_s123/initial_harp_state_post_sa_pre_mlp_s123.json \
  --best_dir_name best_custom_stage2 \
  --lr_lora 0 --lr_embed 0 \
  --lr_xattn 1e-4 --lr_gate 2e-4 \
  --lr_ff 0 --lr_gate_ff 0 \
  --epochs 4 --seed 123 \
  --mem_dim -1 --max_slots 64 \
  --every_n_layers 8 \
  --require_pragma_free_memory
```

Confirm the exact CLI spelling against the checked-out commit before launching. In particular, do not silently let inference use `mem_dim=32` or `every_n_layers=16` when the trained checkpoint used inferred dimension and interval 8.

Do not launch the four arms through the checked-in `designsplit_2_stages_3_goals.sh` or `serrano_holdout_2_stages_3_goals.sh`: they still call `train_SFT_xattn.py`, use `PARETO_KNEE`, and hard-code older memory settings.

### 5. Measure the right outcomes

- [ ] Schema validity: parse success, exactly one assignment per expected slot, no missing/extra/duplicate slots, and legal values.
- [ ] Per-slot directive accuracy, reported by `<Lk>` slot, pragma kind, kernel, and macro-average across kernels.
- [ ] Resource feasibility using measured utilization/budgets; report it separately from syntactic validity.
- [ ] Generate the same top-k candidates under paired sampling seeds for every arm.
- [ ] For candidates present in the measured dataset, join by a canonical ordered pragma signature and use measured QoR.
- [ ] Report top-k design quality, normalized regret, and Pareto hypervolume with one fixed normalization/reference convention per kernel.
- [ ] Report candidate uniqueness and measured-dataset coverage. An unseen candidate is not syntactically invalid; it simply lacks ground-truth QoR until synthesized.
- [ ] Do not use the GNN's own predicted QoR as ground truth for validating whether its embeddings help the LLM.
- [ ] Use paired per-kernel differences and bootstrap confidence intervals across kernels/cases.
- [ ] Keep checkpoint selection independent of final Serrano results.

### 6. Decision rule

- [ ] `static_pre_npt > zero` and `static_pre_npt > shuffle_slots`: aligned structural memory is causally useful.
- [ ] `static_pre_npt ≈ shuffle_slots`: the LLM ignores alignment, or the memory lacks slot-relevant information.
- [ ] `zero > real memory`: the cross-attention path is actively harmful or overfits.
- [ ] `static_pre_npt > current_zero_mask`: the exporter mismatch is confirmed.
- [ ] `current_zero_mask ≈ static_pre_npt > controls`: exporter mismatch is not practically important for this task.
- [ ] Require consistency on held-out-kernel design-quality metrics, not merely training loss or seen-row directive accuracy.

## Priority 2 — Test the structural-analogy hypothesis directly

- [ ] Build one pragma-free structural prototype per training kernel from exact static embeddings.
- [ ] Measure nearest-neighbor retrieval for held-out kernels without using QoR labels.
- [ ] Test whether structural distance predicts similarity of complete pragma-response vectors, not just family labels.
- [ ] Define response similarity from matched pragma actions: sign agreement, delta correlation, and ranking agreement for latency and area.
- [ ] Compare against simple baselines: graph size/count features, kernel-family label, and random retrieval.
- [ ] Visualize UMAP only as a diagnostic; do not treat a visually attractive cluster as evidence.
- [ ] Success means closer structural embeddings statistically imply more similar pragma effects on held-out kernels.

## Priority 3 — Make checkpoint qualification transfer-aware

- [ ] Report macro metrics per kernel and per pragma family, not only row-weighted aggregate loss.
- [ ] Track sign accuracy, Kendall tau, calibration/bias, and loss ratio versus the zero-delta predictor for both targets.
- [ ] Reject a checkpoint that improves the mean but catastrophically fails a validation family.
- [ ] Select with a robust criterion such as worst-target macro tau plus a guardrail on every validation kernel's zero-baseline loss ratio.
- [ ] Add a held-out transformation-style split, especially double buffering, in addition to held-out kernel families.

## Priority 4 — Train the information the LLM actually needs

- [ ] Add pairwise action supervision within the same kernel: given actions `a_i` and `a_j`, predict which improves latency/area and by how much.
- [ ] Add sign and ranking losses alongside Smooth L1 on reference deltas.
- [ ] Weight comparisons by decision relevance: Pareto-near and feasibility-boundary pairs deserve more weight than obviously dominated pairs.
- [ ] Export slot-level action tokens or deltas aligned to `<Lk>` sites, rather than assuming a graph-level QoR head automatically makes every node vector useful.
- [ ] Test an auxiliary contrastive objective: same structural site under similar pragma effects close; dissimilar effects separated.

## Priority 5 — Separate transferable tendency from kernel-specific residual

- [ ] Model `delta(G,a) = mean_effect(a, context) + residual(G,a)`.
- [ ] Use a transparent global/contextual action-effect baseline as the mean component.
- [ ] Train the GNN on the residual and on uncertainty.
- [ ] On an unknown kernel, fall back toward the mean effect when the structural embedding is out of distribution or uncertain.
- [ ] Feed the LLM the retrieved analogues, predicted effect/sign, confidence, and applicability mask—not a large opaque vector alone.

## Priority 6 — Robustness and ablations

- [ ] Repeat decisive experiments over at least three seeds.
- [ ] Ablate graph-only memory, slot-only memory, graph-plus-slot memory, and predicted-effect summaries.
- [ ] Compare cross-attention with a small textual/soft-token summary to determine whether the bottleneck is the GNN representation or memory injection.
- [ ] Measure memory-gate activations and attention mass by layer/slot; these are diagnostics, not causal proof.
- [ ] Test missing-memory and out-of-distribution behavior explicitly.
- [ ] Calibrate an OOD score from training-kernel structural distances and expose it to the LLM or use it to gate memory strength.

## Priority 7 — Final locked evaluation

- [ ] Freeze architecture, Stage C checkpoint, exporter, Stage-2 recipe, objective handling, and all thresholds before Serrano.
- [ ] Evaluate Serrano once as the final unknown-kernel test.
- [ ] Report all four causal arms, all objectives, seeds, failures, feasibility, regret, hypervolume, and confidence intervals.
- [ ] Preserve raw prompts, generated candidates, canonical pragma signatures, measured QoR joins, and manifests for reproduction.

## Definition of done for Priority 1

- [ ] Four validated immutable banks exist and are provenance-tracked.
- [ ] Four Stage-2 runs use identical non-memory conditions and initial trainable state.
- [ ] Current training and inference share one prompt/model contract.
- [ ] Held-out results include schema, slot accuracy, feasibility, top-k measured QoR, regret, and hypervolume.
- [ ] Real aligned memory beats both zero and shuffled memory with a consistent paired held-out-kernel effect, or the experiment clearly falsifies that hypothesis.

## Result table template

| Arm | Schema valid | Slot acc. | Feasible | Top-k regret | Hypervolume | Held-out macro | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Zero memory | | | | | | | |
| Current post-NPT | | | | | | | |
| Exact static pre-NPT | | | | | | | |
| Shuffled static slots | | | | | | | |

## Related-work anchors

- HARP: decouple program structure from transformations and use a node-level pragma transformer: https://ieeexplore.ieee.org/document/10323853
- HARP implementation: https://github.com/UCLA-VAST/HARP
- ProgSG: align source/program semantics with graph representations for HLS performance modeling: https://arxiv.org/abs/2305.10838
- compareXplore: representation-learning and transfer considerations for HLS design-space exploration: https://arxiv.org/abs/2409.13138
- DiffHLS: recent independent evidence for expressing design outcomes relative to a kernel baseline: https://arxiv.org/abs/2604.09240





I expect v16 to improve Kendall τ and top-k design ranking. I do not promise better MAPE. The change directly optimizes the ordering used in DSE, while the current Stage C only uses ranking for checkpoint selection.

Success criteria:

median three-seed worst-target kernel-macro τ greater than 0.326;
every target still beats the neutral validation baseline;
no major absolute regression—preferably within 5–10% of Stage C;
Pathfinder-4 latency τ meaningfully above approximately 0.09;
real exported memory later beats shuffled memory in Stage 2.
Next GNN change after v16—not in the same run

mlir_data.py already parses BRAM, DSP, FF and LUT utilization, but discards them before constructing point tensors.

Persist these four fractions and add small absolute auxiliary resource heads:

log1p(BRAM utilization)
log1p(DSP utilization)
log1p(FF utilization)
log1p(LUT utilization)

This is more aligned with MailoHLS than a single aggregate area target because the LLM receives individual resource budgets. Test it separately from the rank loss so you can attribute improvements.

Finally, make the memory exporter load a resolved checkpoint configuration. It currently hashes config.py, not the exact resolved flags. Save and hash:

resolved GNN flags;
feature-schema JSON;
checkpoint;
source PT manifest;
embedding mode.








