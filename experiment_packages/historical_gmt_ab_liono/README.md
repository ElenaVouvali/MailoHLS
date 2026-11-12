# Historical MailoHLS GNN vs GMT — A30 matched Stage-2 experiment

## Scientific question
Does three-role compiler-evidence GMT memory **add decision-relevant information** to the authenticated historical epoch-9 four-layer TransformerConv+JK action vectors, when all other Stage-2 parameters and training examples are controlled?

- **A:** immutable 64 historical action embeddings, 128D; historical compiler action routing.
- **B:** exact same first 64 action embeddings **plus** up to 3 learned 128D memory/compute-dependence/control summary keys for each active action, with compiler-grounded provenance and per-action routing.
- Same frozen DeepSeek-Coder-6.7B Stage-1 V4 reference, seeded fresh Stage-2 cross-attention at decoder layers 8/16/24/32, heads 4 and dimension 128, identical training rows/seeds/evaluation.
- A30 24GiB; 4-bit NF4 decoder base from production trainer, no change to V100 workload. Trained GMT summary generator is a **compiler-only auxiliary warm-start**, selected without using Stage-2 validation kernels. GMT is frozen during Stage-2 in this bounded study.
- These are two arms of **one** structural-memory experiment, not two different experiments. Production checkpoint not automatically replaced.

> IMPORTANT: The word GMT here refers primarily to *Graph-as-Memory Tuning* (2025), which motivates deep cross-attention to compressed graph memory. The three-role action summaries also use query-based pooling related to Graph Multiset Transformer (ICLR 2021). Neither paper's LLaMA-specific wrapper is copied.

## Repo basis (inspect before running)

Branch `research/historical-mlir-gmt-gemv`:
- `experiments/historical_mlir_gmt_gemv/historical.py` authenticates the frozen historical 4-layer epoch-9 GNN and original 64x128 action memory.
- `experiments/historical_mlir_gmt_gemv/historical_gmt_slots.py` implements typed compiler evidence, 3-role attention pooling, and train-only auxiliary reconstruction.
- `GNN_branch/MLIR_graphs/*.gexf` contains historical compiler graphs; not interchangeable with RC3.
- `experiments/rc3_stage2_multikernel/audit.py` produces **only the source/target-split supervised dataset**. Despite its name, its RC3 *memory builder* is NOT used here.
- `LLM_branch/common/structural_memory.py`, `.../structural_xattn.py` and `LLM_branch/train/train_SFT_xattn_new.py` support variable-length evidence memory, absolute slots and 4-bit training. B uses the generic variable-key interface currently named `rc3_action_evidence_c_v1`; its custom memory `schema` and source/GNN hashes distinguish the *historical* content. No RC3 checkpoint is loaded.

## Requirements / setup

**Run on Davinci** in an isolated checkout/worktree of the historical branch (not in the Liono V100 worktree). The original history commit `bf82c648273a5cd5eb30363059a3809e7b3979db` must be locally accessible to `git archive`.

You need the authenticated **untracked original epoch-9** checkpoint and its original provenance assets at the absolute `HISTORICAL_ASSETS_ROOT`, and the original frozen V4 Stage-1 adapter and domain registry at `PARENT`. The checkpoint .pth is **not in GitHub**. Transfer files via an approved SSH/rsync route; do not copy the current trained FiLM adapter into this test.

`PYTHON` must be an existing Davinci environment containing `torch`, `torch_geometric`, `networkx`, `numpy`, `transformers`, `peft`, `bitsandbytes` and the maintained MailoHLS training dependencies. The launcher does not install packages or change the machine's other GPU processes.

Copy the 4 files of this package into **any folder on Davinci**; run from the MailoHLS historical-branch worktree:

Rows whose V4 input omits or adds source-action placeholders relative to the
authenticated source action map are excluded as whole kernels by the audit.
The applied strict filter is recorded as `audit_v4_action_filter.patch` in each
run output; it does not infer or add actions. The downstream historical graph
and source identity checks remain mandatory.

```bash
cd ~/MailoHLS/worktrees/historical-mlir-gmt-gemv
export PYTHON="$HOME/MailoHLS/.venv/bin/python"    # change if your Davinci environment differs
export PARENT="$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run"
export HISTORICAL_ASSETS_ROOT="$HOME/MailoHLS"
export GEXF_DIR="$PWD/GNN_branch/MLIR_graphs"
export RUN_ID="historical_gmt_ab_a30_20261112"
export STEPS=300 GMT_STEPS=600 MAX_LENGTH=4096 SEEDS=123,456
bash /path/to/this_package/run_a30.sh
```

For a **no-Stage-2** preparation check: `PREPARE_ONLY=1 bash /path/to/this_package/run_a30.sh` with fresh `RUN_ID`. Do not attempt to reuse the same directory for a subsequent train run; inspect the receipt and use a new run ID.

## Main guarantees

- Identical measured supervised examples, original 64 action embeddings, Stage-1 reference, train/validation family split, cross-attention architecture, and initialization reference for both arms.
- Historical source-file SHA must match graph provenance and declared action ID set. Missing/mismatched kernels are reported, not replaced with RC3/zero vectors.
- Minimum 8 independently named training families, 2 validation families. Historical pretraining overlap with Stage-2 val kernels causes refusal.
- GMT auxiliary training uses **Stage-2 training families only**, with one internal training-family dev partition. No GEMV supervised training or held-out measurement leakage.
- No locked-test reads in the supervised data audit. Resource budgets and HLS estimates are NOT certified post-route physical measurements.
- The 4-bit DeepSeek-Coder base uses the existing production trainer. A30 selection is UUID-scoped with a free-memory/process check; it never starts a V100 job.
- `MAX_LENGTH=4096` is only a starting memory budget. A conservative tokenizer check rejects likely truncated source/answer pairs. If more context is needed, a full no-truncation run on A30 may be infeasible.

## Expected outputs

`$OUT/prepared/PREPARATION.json` records coverage, source identities and memory shape; two banks `memory_A` and `memory_B`; `$OUT/prepared/gmt_selected.pt`; matched SFT/split; `seed_*/A` and `seed_*/B` Stage-2 checkpoints and `val_selection_step_*.json`; `$OUT/architecture_comparison.json` and training stdout.

**Decision boundary:** A repeatable B superiority in complete directive decisions across held-out families, with genuine B-ON vs B-summary-masked sensitivity, supports promotion to a Stage-2 qualification run. Parity, regression or no meaningful decisions favors the simpler A for now. **This script does not establish a final structural-architecture freeze:** only two val families, no matched HLS synthesis of selected designs, GMT pooling frozen during Stage-2, and a potentially changing Stage-1 parent. The report therefore deliberately returns `NEEDS_REVIEW`, not an invented yes/no claim. After your new Stage-1 is finalized, replay the selected structural comparison against that parent.

## If preflight refuses

Do **not** bypass mismatched source hashes, missing historical checkpoint, overlap with historical GNN training, memory OOM, truncation or missing val families by inventing data. Such a refusal is informative: the intended *historical* matched experiment is not executable from available verified data. At that point compare structural memories over the available authenticated historical cohort or regenerate verified historical graphs using the original compiler, preserving the original source/action identities.

## References

- GMT Graph-as-Memory Tuning: https://github.com/tongruiliu/GMT
- Original Graph Multiset Transformer: https://github.com/JinheonBaek/GMT
- MailoHLS historical GMT branch: https://github.com/ElenaVouvali/MailoHLS/tree/research/historical-mlir-gmt-gemv
