# New LLVM latency checkpoint reconstruction

This experiment trains a **new** latency-oriented framework checkpoint from the
original `llvm-ir` implementation. It does not recover the original paper's LLM
weights and does not use an MLIR encoder or GMT tokenizer. Training is currently
in progress on Davinci's A30; no completed framework checkpoint or predictions
are claimed yet. See [manifest.json](manifest.json) for the run identity.

## Architecture and assets

The source parent is `llvm-ir` commit
`0b5581b2b22becf1d20f476e22e92ea3fd533a64`, preserving paper snapshot
`c9b2f43e97508115c4d8d58bc24900adf437b41a`. The original SFT, HARP cross-attention,
DPO and constrained complete-RHS inference implementations are retained.

- Base: existing cached DeepSeek-Coder 6.7B revision
  `ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912`. The archived launchers call it
  `deepseek-coder-7b-base`; both local cache aliases have the same revision/config.
- GNN: original LLVM `Net`, recovered `run1/val_model_state_dict.pth`, SHA-256
  `c6bbba75fa8dffb9e29d0cbcf2da689bd875d54d5e67f038dff76cbf4d8409f3`.
  Strict state loading passed. Actual feature dimensions are **356/1808**;
  the archived `model_info.txt` dimensions **361/1807** are stale.
  [gnn_validation.json](gnn_validation.json) records the check. The architecture
  has four message-passing layers before JK-max and an additional post-pragma-MLP
  convolution, as defined by the archived `Net` code.
- Original 55-kernel pragma-free memory bank: `[64,64]` per kernel, absolute
  `Lk -> k-1` slots. A numerical replay of the archived GEMM graph did **not**
  reproduce the archived vectors (11.15% relative L2 difference), despite matching
  checkpoint-path metadata. The reconstruction rebuilds this bank from the
  actual recovered weights and archived graph tensors before Stage 2. This
  rebuild has completed for all 55 kernels, preserving every archived action
  label and compiler-node binding. See
  [memory_rebuild_manifest.json](memory_rebuild_manifest.json). The unreproduced
  archive bank is preserved separately and is excluded from structural training.
  [memory_reproduction.json](memory_reproduction.json) records the failed replay;
  the cause is not established. The launchers' `mem_dim=32` is incompatible with
  the actual assets; the reconstruction uses width **64**.
- Frozen one-hot encoders SHA-256:
  `22aaf32d3fd0df4a902488785ad75d674de0b67d5177aa3b848cabac3f8a72e7`.
  Encoders are not refitted for the four external kernels.

The original assets were recovered from the user's Drive `Thesis` archive.
Large weights, source graph tensors and the materialized dataset remain outside
Git. Exact asset hashes are in the manifests. The original LLVM tensorizer's
lexicographic edge-relabeling convention is preserved; this reconstruction does
not establish that the historical representation is free of alignment defects.

## Measured training data and protocol

The archived 1.1 GB training JSONL exceeds the Drive connector's 256 MB download
limit. This run therefore reconstructs the original trainer's input schema from
the existing local measured compact dataset and source registry. It filters to
**ZCU104 (`xczu7ev-ffvc1156-2-e`), 10 ns** and retains measured latency and area.
It must not be described as an exact reproduction of the unavailable JSONL.

There are **25,139 usable designs / 54 kernels** after excluding 18 nonpositive
measurements. A seeded, per-kernel random design split yields 20,111/2,514/2,514
raw train/validation/test rows. The original latency/Pareto selection chooses
322/290/290 examples independently within those partitions. No duplicate
kernel/assignment pairs were found in the reconstructed cohort. All four
requested external kernels remain outside training.

The source/target directive schemas and required original memory slots pass
coverage checks. Selected examples have at most 6,492 tokens, so the 7,168-token
limit retains complete source and target context. See
[data_manifest.json](data_manifest.json) and [preflight.json](preflight.json).

The original three stages are:

1. Four epochs of RHS-only SFT: LoRA rank 8 / alpha 16 / dropout 0.05; LoRA and
   special-token embedding LR `5e-5`. Structural fusion is disabled.
2. Four epochs training HARP cross-attention and attention gates at LR
   `1e-4` / `2e-4`, freezing Stage 1. Fusion occurs every eight decoder layers.
3. The archived DPO refinement budget of 60 updates, beta 0.1, training only
   structural cross-attention/gates at `5e-5` / `2e-5`. Stage 2 is the frozen
   reference. No external-kernel labels enter preference training.

The preference preflight produces **85 training pairs across 23 kernels**, 113
validation pairs and 94 test pairs under the archived filters. These partitions
remain separate. See [preference_preflight.json](preference_preflight.json).

Microbatch 1 and accumulation 8 preserve effective batch 8. Seed is 123.
The A30 uses BF16 compute and NF4 double quantization. Validation and saving
occur every 40 SFT updates; the installed Trainer schedules 160 updates for
four epochs. Six validation kernels use the archived candidate-scoring selection.
The initial startup run was stopped solely to correct an archived validation
interval of 164 that would miss this run's final step; its log is preserved.

Compatibility changes are explicit: configurable runtime paths, GPU-supported
precision, two retained optimizer checkpoints, true freezing of zero-LR groups,
final Trainer-state saving, final validation if required, and failure on missing
or incompatible fusion checkpoints. The original objective, prompt, RHS-only
loss, graph representation and action-routing architecture are retained.

## Four-kernel inference

[cases.jsonl](cases.jsonl) preserves the exact sources requested previously:
TRMM uses the available **`trmm-opt`** variant (60x60 A, 60x80 B), followed by
Jacobi-2D, covariance and SYR2K. CPU LLVM10/ProGraML inference has exported
memories for all 19 action sites, covering 38 directive fields. Their compiler
graphs contain respectively 203/411/455/293 nodes and 706/1462/1600/1024 edges.
Bindings and hashes are in [external_memory_manifest.json](external_memory_manifest.json).

After training completes, inference uses the selected Stage-1 LoRA parent plus
the selected Stage-3 fusion weights. DPO leaves the LoRA frozen. Candidate
domains come only from selected training rows. The original decoder scores each
complete RHS and newline continuation causally, with mean token log-probability
and full preceding directive context. Output is
`runtime/predictions/llvm_latency_four_kernels.jsonl`.

**Physical QoR improvement: NOT EVALUATED.** New synthesis is required for these
exact assignments, sources, target and requested clock. Likelihoods and
directive changes cannot establish hardware improvement.

## Reproduction

Run from this isolated LLVM worktree, using an existing environment with the
original dependencies. The executing LLM environment is Davinci's
`/home/evouvali/MailoHLS/.venv/bin/python` (PyTorch 2.6.0+cu124, Transformers
4.51.0, PEFT 0.18.1, bitsandbytes 0.49.2). The CPU graph export uses Liono's
`/home/elvouvali/.hls-llm/bin/python` with LLVM10 and ProGraML.

```bash
python experiments/llvm_latency_reconstruction/prepare.py \
  --compact /path/to/mailohls_sft.jsonl \
  --sources /path/to/mailohls_sft.sources.json \
  --memory "$LLVM_RUN_ROOT/memory" --output "$LLVM_RUN_ROOT/dataset.jsonl" \
  --manifest experiments/llvm_latency_reconstruction/data_manifest.json

python experiments/llvm_latency_reconstruction/preflight.py \
  --runtime "$LLVM_RUN_ROOT" --model "$LLVM_BASE_MODEL" \
  --output experiments/llvm_latency_reconstruction/preflight.json

# Use the CPU graph environment for this command. The 55 archived graph tensors
# must be restored under runtime/original_graphs before rebuilding the bank.
CUDA_VISIBLE_DEVICES='' python experiments/llvm_latency_reconstruction/rebuild_memory.py \
  --runtime "$LLVM_RUN_ROOT" --output "$LLVM_RUN_ROOT/memory_rebuilt"

PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' python \
  experiments/llvm_latency_reconstruction/export_external_memory.py \
  --runtime "$LLVM_RUN_ROOT" \
  --cases experiments/llvm_latency_reconstruction/cases.jsonl \
  --gnn "$LLVM_RUN_ROOT/assets/gnn_val.pth" \
  --encoders "$LLVM_RUN_ROOT/assets/encoders.klepto"

export LLVM_RUN_ROOT=/home/evouvali/mailohls_runs/llvm_latency_reconstruction_20261112/runtime
export LLVM_BASE_MODEL=/home/evouvali/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912
export LLVM_PYTHON=/home/evouvali/MailoHLS/.venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-ac0112df-7098-6c59-5c4f-a57fa666f808
# Recheck occupancy before allocating the selected GPU.
bash experiments/llvm_latency_reconstruction/run_training.sh
python experiments/llvm_latency_reconstruction/progress.py --runtime "$LLVM_RUN_ROOT"
bash experiments/llvm_latency_reconstruction/run_inference.sh

# Alternatively verify/hash the completed stages and generate both the native
# detailed results and a compact per-kernel directives JSONL:
python experiments/llvm_latency_reconstruction/finish.py \
  --runtime "$LLVM_RUN_ROOT" --inference

python -m unittest discover -s experiments/llvm_latency_reconstruction \
  -p 'test_reconstruction.py' -v
```

The six focused CPU tests pass: measured-data filtering/schema, absolute action
routing, zero-gate parity/invalid memory, complete multi-token causal RHS scoring
including newline, rejection of untrained completion markers, and missing fusion
checkpoint rejection. Static Python and shell checks pass. Training completion,
validation outcomes and final checkpoint identities must be read from executed
stage completion reports, not inferred from this planned protocol.
