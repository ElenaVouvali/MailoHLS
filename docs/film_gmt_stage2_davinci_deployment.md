# Davinci Stage-2 deployment — READY

Qualified on 2026-11-13 for **original-film-31+2 only**. Full 450-update training
has **not** been launched. The isolated smoke performed one disposable structural
optimizer update; it produced no selected Stage-2 training checkpoint. The V4,
selected Arm B step-600 residual/FiLM, historical GNN and GMT weights were frozen.

Server: `evouvali@davinci.microlab.ntua.gr`.
Deployment: `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113`.
Exact code HEAD: `6ee1390611c9047a4b8e2b07200d223b93ee8d16`, branch
`research/film-gmt-stage2-final-20261113`; tree `3840e763bda143315364efd944c20dba025d1977`.
Both Liono's original checkout and Davinci's independent shallow checkout are clean.
No push or original implementation edit was performed.

| Variable | Absolute Davinci path / value |
|---|---|
| ROOT | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/code` |
| PARENT | `/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run` |
| FILM_CP | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/assets/film/B/checkpoint-0600` |
| GNOSIS_BANK | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/code/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank_original_anchors` |
| GMT_MEMORY | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/code/artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B` |
| GPU_UUID | `GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f` |
| ENTRY | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/adapter/entry.py` |
| RUN_ROOT | `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/runs/campaign_s123_v1` |

ENTRY is a deployment-only filename adapter outside the committed code. It calls
the unchanged `ROOT/experiments/film_gmt_stage2_final/train.py` and unchanged
`commands.sh`. An explicit longest-prefix map in `adapter/paths.json` relocates
only filesystem lookups in the protected-receipt verifier. All original JSON/PT
payloads, expected digests and verification control flow remain unchanged. An
unmapped Liono path fails; a deliberately altered expected digest in a temporary
receipt was rejected. No global pathlib or model behavior was patched.

The code, adapter, launcher, model/data payloads and source checksum inventories
are read-only. Run outputs and operational receipts remain writable. Every mode
checks clean exact Git HEAD; training also authenticates the model/data/memory
dependencies. Fresh-output guards refuse to overwrite a previous campaign.

**Exact identities (SHA-256)**

| Artifact | SHA-256 |
|---|---|
| Core trainer | `ee43bc54dbaaa9a89b88c44bf6ceff0fc8350ac4f081f9907e537b26ddd9f233` |
| Original receipt verifier | `5618841cd62f1e7f67c9062b8229c762801baf93db6795342e19e73dc3e170a9` |
| V4 adapter | `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164` |
| V4 training contract | `1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671` |
| V4 directive registry | `41d0f6f90db4c2370bdc02fbcdda4e114ec65cf57b89a567bfab56606d8d0c1c` |
| Arm B step-600 residual LoRA | `6628fe1f268f93ce760428ca0c50b46dbcbfe55e00d45fba97a20142e89b2f80` |
| Arm B step-600 FiLM | `d2697977c97c467c7f433b134dff3031c4e842f3b8f2cf9f0c57613c1ecf46ae` |
| Historical epoch-9 GNN | `34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623` |
| GMT, selected update 330 of original 600 | `1d4cdf3071375349b94809062d05d30ad3e5c981bb6e2885b6f99c9e94398c12` |
| Selected original bank manifest | `2b5745ab029a32f3f0088367b80a7478212249912a7ee8e7b890fdaffc2cbf0c` |
| Four-file dataset identity | `958da104bb3ad1a20f4d4cae3c1bc9464e6ac1c3b1a9e2ad3144a4b7c339a9cf` |
| Stage-2 manifest | `ffa6bf2049e51662a71261d0063051e2cee0c6555089cb31a1f3324764cf0fd6` |
| Final memory manifest | `0e15b178022d6a5decc17444e67ffe9e386eaa0423591d45801245f70a5bff85` |
| Complete memory identity | `07ab6c531bb5c34b32761f970a0bd90b3c915e89cd917ac0e9a73fcdab3e4a02` |
| Filename-map configuration | `e82f662051a9e494b651049193aefa75de15633ad54d29d27a08499cde7b5ed2` |
| Source artifact inventory | `00305c81e337705759fd8dc1ade17ba82722075e50748cef61b10ab1fbe93708` |
| Full deployment payload inventory | `d1e702d3d7327150f2eda4d56c4156e64232da7f55cb018b1ae45e3a72afe4e5` |

`receipts/source_artifacts.json` records all 172 artifact source/destination/hash
associations, including source symlinks when present; authenticated payloads were
materialized as regular files. The committed tree's 115 symlinks were preserved
by `rsync -a`. `receipts/payload_sha256.json` authenticates 3,409 payload paths;
Git authenticates the tracked tree and symlink identities. The final post-smoke
transfer check passed, including 190 artifact/reuse checks: 172 transferred
artifacts, all 11 top-level V4 files, and all seven cached base-model files.
Historical GNN weights, normalization, training provenance, compiler graphs,
original anchor caches, GMT checkpoint, both anchor and augmented memories, and
the committed protected receipts are present. No anchor, graph, checkpoint or
measurement was regenerated.

The reused base model is `deepseek-ai/deepseek-coder-6.7b-base`, pinned revision
`ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912`, at
`/home/evouvali/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912`.
Both safetensor shards and every snapshot file match Liono by SHA-256.
Model loading is offline. The existing V4 directory was reused without edits.

**Environment and qualification**

Python: `/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113/venv/bin/python`,
3.10.12; Linux 5.15.0-126, glibc 2.35; NVIDIA driver 550.54.14, Torch CUDA 12.4.
All eight principal package versions match Liono: torch 2.6.0+cu124,
transformers 4.51.0, peft 0.18.1, accelerate 1.13.0, bitsandbytes 0.49.2,
tokenizers 0.21.4, numpy 2.2.6, pandas 2.3.3. Python patch/compiler differ from
Liono; the actual Davinci audit and smoke validate this runtime. The small venv
installs pinned data dependencies and reuses
`/home/evouvali/MailoHLS/.venv/lib/python3.10/site-packages` read-only for the
matching large CUDA stack. The original shared environment was not modified.
`receipts/environment_freeze.txt` and `environment_gpu.json` record the environment.

Exactly one CUDA device was visible: **Tesla V100-PCIE-32GB, capability 7.0**, the
requested UUID. The other GPU, A30 UUID
`GPU-ac0112df-7098-6c59-5c4f-a57fa666f808`, was excluded. Process inspection during
smoke showed only the requested V100 in use. The adapter rejects another UUID,
multiple visible GPUs, a non-Volta device or a non-FP16 autocast policy. Both base
NF4 compute and model autocast use native FP16; existing quantization is preserved.

| Check | Davinci result |
|---|---|
| Full `MODE=audit` | PASS; 31 train / 2 validation kernels, 33 authenticated memories, no excluded bank kernels |
| Source/action, graph, checkpoint, split and receipt checks | PASS; original anchors preserved; no test outcomes consumed |
| Full bank token audit | NO_TRUNCATION; maximum 6,660, limit 7,168 |
| Full `MODE=smoke` | SMOKE_PASS, native `torch.float16` |
| Longest training SFT | CFD coalescing, index 2165, 6,659 tokens; `code_truncated=0` |
| Zero-gate FiLM ON/OFF parity | Absolute score difference **0.0** |
| Four zero-gate gradient norms | 0.5793344, 1.8031571, 0.2967444, 0.1339233; finite, nonzero |
| Open-gate projection gradients | All finite; range 1.7929e-5 to 0.0101618 |
| Causal routing | Exact supervised RHS Lk binding and compiler-relational keys checked by every smoke forward/backward |
| Structural ON/OFF at declared diagnostic tanh(g)=0.05 | Mean absolute score difference **0.0001549721**, all four post-rounding effects nonzero |
| Post-rounding RMS ratios, layers 8/16/24/32 | 0.01029053 / 0.00449681 / 0.00191901 / 0.00080482 |
| Frozen composite before/after | Identical: `a14a805f989afdbd03a015f45c3586e05cc2f3c8cbc90211af6abb4e7f926b59` |
| Checkpoint save/resume, exact committed functions | PASS on CPU synthetic weights; weights/optimizer/scheduler/RNG restored; wrong protocol rejected |
| Post-smoke payload and reused-model checks | PASS; HEAD clean and exact |

GPU diagnostics are a pathway qualification, not evidence of learned QoR gains.
The full model smoke used a disposable structural update and never fit validation
data. No GNN, GMT, FiLM or V4 training was performed.

**Coverage and objective**

The bank retains 3,276 training SFT references, 498 pair rows (249 complete groups),
31 training kernels / 15 families, 1,237 budget contexts and 181 physical
kernel/device/clock targets. Validation retains 216 SFT references, 28 pair rows,
two kernels / two families, 72 budget contexts and 12 physical targets. Fixed
evaluation uses 24 SFT references spanning both validation kernels, all 12 targets,
20 budgets, and eight complete preference groups (16 pair rows). Existing absent
Cartesian targets remain absent; no missing measurement or target was added.

Each optimizer update processes **four SFT contexts** plus **one two-context
preference group**. SFT uses four accumulated microbatches; each ranking context
scores preferred/rejected alternatives and backpropagates their two contributions.
Over 450 updates the deterministic SFT schedule has **1,800 context visits**:
674 contexts once and 563 twice. All 1,237 are reached during update 310. All 249
preference groups (128 budget, 121 clock) are scheduled: 201 twice, 48 once, always
with both target contexts. Context-order SHA-256:
`2a29ed69be00b98c971746646ed9f9fb6805bfb83f31beaae59c2cbc2e2abfce`.

These are **scheduled exposures**. Realized full-campaign exposure is zero because
training has not started. The isolated smoke is not part of that schedule.
Successful `training.jsonl` updates and the deterministic ordering permit auditing
realized coverage after launch; a stopped run does not inherit the planned total.

The objective is **action-normalized nonforced-RHS SFT plus measured target-switch
logistic pair ranking**, **not standard reference-corrected DPO**. For each pair,
`s = -action_normalized_NLL`; the pair loss is
`weight * softplus(-(s_preferred - s_rejected))`, averaged over the group's two
contexts. Budget weight is 0.20; clock weight is 0.05. There is no reference-model
log-ratio subtraction or DPO beta. The imported helper module also defines a DPO
function, but this trainer does not call it. This deployment does not change the
objective, 450-update budget, context schedule, four XAttn layers, zero gates or
60-update gate curriculum.

**Launch manually on Davinci**

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
nohup env MODE=train bash "$D/launch.sh" > "$D/receipts/train_driver.log" 2>&1 < /dev/null &
echo $!
```

This creates the fresh `runs/campaign_s123_v1/final` directory; it was verified
absent after qualification. Original commit guards reject an existing final run.
Training first repeats the authentication, parity and gradient checks, then
evaluates at step 0 before the first campaign update. The fixed validation is
retained. Checkpoints are saved at steps **0, 150, 300 and 450** with structural
weights, optimizer, scaler, scheduler, Python/Torch/CUDA RNG and receipts.
`best.json` names the selected checkpoint (which may differ from step 450);
`COMPLETE.json` marks successful full completion. Full frozen parent/residual/FiLM
weights are referenced by identity, not duplicated in each structural checkpoint.

Monitor on Davinci:

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
tail -f "$D/receipts/train_driver.log"
tail -n 5 "$D/runs/campaign_s123_v1/final/training.jsonl"
cat "$D/runs/campaign_s123_v1/final/best.json"
nvidia-smi -i GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f
```

`training.jsonl` contains loss, ranking margin, gate/projection gradients, gate
values, learning rates and timing. `val_0000.json`, `val_0150.json`, etc. contain
validation and structural ON/OFF diagnostics. Driver progress prints every ten
updates. For a genuine interruption, use the latest complete saved checkpoint,
for example (do not run concurrently with training):

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
nohup env MODE=resume RESUME_FROM="$D/runs/campaign_s123_v1/final/checkpoint-0150" bash "$D/launch.sh" > "$D/receipts/resume_0150_driver.log" 2>&1 < /dev/null &
```

The same paths and settings are required by the saved protocol. Resume reloads
and authenticates frozen inputs; it does not refit or regenerate data/GNN/GMT.

Retrieve **after completion** to Liono (all structural checkpoints and receipts):

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
DEST=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/artifacts/davinci_stage2_deployment_20261113/retrieved_campaign
mkdir -p "$DEST"
rsync -a --checksum --info=progress2 "evouvali@davinci.microlab.ntua.gr:$D/runs/campaign_s123_v1/final/" "$DEST/final/"
rsync -a --checksum "evouvali@davinci.microlab.ntua.gr:$D/receipts/" "$DEST/receipts/"
rsync -a --checksum "evouvali@davinci.microlab.ntua.gr:$D/adapter/" "$DEST/adapter/"
rsync -a --checksum "evouvali@davinci.microlab.ntua.gr:$D/launch.sh" "$DEST/launch.sh"
cat "$DEST/final/best.json"
```

The selected local checkpoint is `final/checkpoint-NNNN` for the step in
`best.json`; that JSON preserves the original Davinci absolute path. Retain the
authenticated frozen assets on Liono alongside these structural weights.

The completed qualification commands used `MODE=audit` and `MODE=smoke` with
`RUN_ROOT=$D/runs/qualification_v1`. To repeat qualification use a fresh root:

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
MODE=audit RUN_ROOT="$D/runs/qualification_v2" bash "$D/launch.sh"
MODE=smoke RUN_ROOT="$D/runs/qualification_v2" bash "$D/launch.sh"
```

CPU audit, GPU smoke, environment, tamper-rejection, checkpoint-roundtrip and
post-transfer receipts are in `receipts/` and `runs/qualification_v1/` on Davinci,
and `verified/` beside this report on Liono. The original receipts were copied
unchanged under `code/docs/provenance/film_gmt_stage2_final_20261113_original_anchors`.
Expanded 49+4 qualification remains outside this deployment. **READY for the
explicit original 31+2 campaign; no reason from this deployment to delay it.**
