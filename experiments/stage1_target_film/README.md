# Stage-1 residual target LoRA versus action-aware FiLM

Research branch `research/stage1-target-film-20261110`, parent `58c098936c14cfb021154884d5ad75cdd5fcf068`.
No GNN, structural fusion, DPO, synthesis or production promotion is part of this experiment.

Frozen V4 latency adapter SHA256 `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164`.
The original base, objective adapter and learned-token rows remain frozen. The new residual uses the parent's rank8, alpha16 and q/k/v/o projection scope, reusing `device_adaptation.activate` and its parent-token serialization fix. Both adapters remain active. The new bundle schema is distinct from the historical single-device bundle; it does not relabel that contract.

## Evidence and limits

[Exact selected-pair audit](pair_bank_audit.json): 128 training pairs, 16 families, 249 unique physical designs; 54 validation pairs, 2 families, 105 physical designs. No identical candidate-pair switch or stay occurs across target contexts in this selected bank. Unmatched candidate pools cannot establish reversals. These are exploratory same-request latency preferences, with proxy resource feasibility, not exact resource counts or timing qualification.

The larger historical common-pool audit has training clock reversal3/stay130, device reversal2 correlated views/stay188, budget proxy boundaries74/no-change525. Validation has clock stay10, device stay16, budget proxy boundaries11/no-change43, and no clock/device reversals. These counts do not describe the sampled 128/54 preferences. Suspect spcl data were excluded from the pair bank. GAN and locked test are excluded from all selection.

Each preferred complete design is within1% of its request's measured feasible proxy pool best; its rejected complete design is at least10% slower. Reference sampling preserves real correlated assignments; this bounded pilot samples one authenticated near-optimal preferred completion per retained request, not all near-optimal references. Neither candidate uses the other's causal prefix.

## Mechanism

A trains only residual LoRA. B trains the same LoRA and one shared final-hidden-state FiLM generator:15 physical/context inputs,8-dimensional directive-kind embedding,32-unit SiLU bottleneck, bounded gamma/beta and tanh gate. Five kinds are PIPE/UNROLL/ARRAY_T/ARRAY_F/ARRAY_D. Lk is never a semantic feature. Device capacities come directly from `mailohls_contract.DEVICE_RESOURCES`, not from preferred QoR. The gate starts at0 with a nonzero random branch.

FiLM is an LM-head pre-hook. Explicit kinds are assigned at position i-1 to predict RHS token i. Its state includes only request context and current field kind. Prompt/source logits pass through unchanged. Native output head, vocabulary, domain filtering, complete serialization and mean-RHS candidate decoder are preserved. `infer.py` uses the same FiLM class and explicit candidate-position masks; it does not infer actions from free text.

Loss: action-normalized nonforced value-token SFT +0.1 softplus(-(score_preferred-score_rejected)); score is the negative same action-normalized NLL. Ranking temperature1. Fixed coefficient avoids a validation sweep. Both arms independently restore identical initial residual tensors; seed123, AdamW2e-5, effective batch2,64 successful updates, identical pair schedule, validation at0/16/32/48/64. Checkpoint selection: family-macro preference accuracy, then preferred NLL. This is exploratory pair validation, not measured generated QoR.

## Actual execution

Davinci V100 UUID `GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f`; A30 untouched. Environment `/home/evouvali/MailoHLS/.venv/bin/python`, persistent root `/home/evouvali/mailohls_runs/stage1_target_film_20261110`.

The focused CPU test passed. Real NF4 V100 smoke completed2 updates per arm. A/B step0 validation and first training losses matched. B gate gradient on update1 was0.0003781; generator output-weight gradient on update2 was1.868e-6. A failed initial gradient-checkpointing smoke was preserved; the fix installs the input-gradient hook after PEFT token-wrapper installation. This changes gradient connectivity, not frozen parameter ownership.

The matched64-update run is launched; completion is established only by each arm's `completed.json`. Reports will distinguish selected step0 from an updated checkpoint if validation retains the parent. Checkpoints save residual-only PEFT weights and FiLM, never the backbone/parent. Progress includes optimizer/scaler/RNG for explicit resume. `--resume /path/A/checkpoint-N` restores that arm before proceeding to B.

```bash
python -m experiments.stage1_target_film.audit --pairs PAIRS --out NEW_AUDIT.json
python -m pytest -p no:cacheprovider experiments/stage1_target_film/test_model.py -q
python -m experiments.stage1_target_film.train --parent V4 --pairs PAIRS --out NEW_RUN \
  --gpu-uuid VERIFIED_IDLE_UUID --steps 64
python -m experiments.stage1_target_film.infer --parent V4 --adaptation SELECTED_CHECKPOINT \
  --prepared SEALED_GAN --out NEW_GAN_OUTPUT --gpu-uuid VERIFIED_IDLE_UUID
python -m experiments.stage1_target_film.report_gan --v4 V4_PREDICTIONS --a A_PREDICTIONS \
  --b B_PREDICTIONS --out NEW_REPORT
```

`continue_gan` waits for B's completed training receipt, reads the already selected development checkpoints, verifies the historical18-case V4 reference receipt, then executes A/B sequentially through the native decoder and builds the comparison. No HLS command exists in this path. GPU preflight refuses contention. Fresh outputs are required; partial output is preserved for inspection.

Do not claim target adaptation from pair ranking or GAN directive diversity. Exact-count feasible regret, timing-qualified benefit and independent device/clock switches remain unavailable in this cohort.
