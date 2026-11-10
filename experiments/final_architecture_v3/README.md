# Revision-3 architecture qualification

Parent: `8ad95155ec38d9a389e3b8c70dc4a1127dacc28e`. Worktree/branch:
`/home/elvouvali/MailoHLS-final-architecture-v3-20261110`,
`research/final-architecture-v3-20261110`. Original worktrees, registries and
historical receipts remain unchanged. [Receipt reconciliation](RECEIPT_RECONCILIATION.md)
separates historical reports from current evidence. The completed V4b audit is reused.

The reference is **PARETO_LATENCY / xczu7ev-ffvc1156-2-e /10 ns /100% BRAM,DSP,FF,LUT**.
DPO is a mandatory qualification gate. Runtime refinement is explicitly opt-in.
No release benefit or architecture freeze is implied by implemented interfaces.

## Commands actually implemented

Use `/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python` for GPU runs.
The existing isolated `.../MailoHLS-final-architecture-20261109/artifacts/final_architecture_20261109/cpu-env/bin/python`
has PyG for graph/CPU tests. Do not install packages into shared environments.

Dry-run by default; add `--execute` after inspecting the printed command:

```bash
python -m experiments.final_architecture_v3.launch --stage baseline \
  --run-dir /home/elvouvali/mailohls_runs/final_architecture_v3/static_corrected_s123 --steps 150
```

`launch` supports explicit `--stage1`, `--data-dir`, `--memory-dir`, `--gpu-uuid`
and `--resume`; paths are not assumed shared across hosts. Its preflight refuses
occupied GPUs. The corrected wrapper retains all sealed multi-reference exposures,
disables duplicate compaction, and uses the pinned V4 action-normalized value loss.
The parser's inherited `top_k=1` no longer selects/deduplicates rows in this scoped
wrapper; the new contract records that distinction explicitly.

Analyze the already completed immutable traces:

```bash
python -m experiments.final_architecture_v3.analyze_traces \
  --trace /home/elvouvali/MailoHLS-final-architecture-20261109/artifacts/final_architecture_20261109/fixed_prefix_liono.jsonl \
  --out /path/to/new/fixed_prefix_analysis.json
python -m experiments.final_architecture_v3.plot_traces \
  --input experiments/final_architecture_v3/results_v3/fixed_prefix_analysis.json \
  --out /path/to/figures
```

Stage3 and matched chosen-only SFT, after selecting and authenticating a completed
corrected Stage2 parent (this command **executes**, so run GPU preflight first):

```bash
python -m experiments.final_architecture_v3.train_preferences \
  --parent /path/to/corrected-stage2-checkpoint \
  --pairs artifacts/final_architecture_v3/preferences_v3 \
  --memory-dir /home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/memory_c/memory \
  --out /home/elvouvali/mailohls_runs/final_architecture_v3/preference_s123 --steps 34
```

The reference is the exact starting Stage2 model, cached before updates. Cache
identity includes parent fusion/contract, frozen Stage1 runtime hash, memory,
pairs, encoding and masks. It is never the bare pretrained model. One NF4/FP16
backbone is reused. Each completion has its own causal prefix. DPO sums selected
nonforced value-token log probabilities in float32; the production decoder still
ranks each RHS by mean continuation log probability including its newline.
Chosen-only SFT uses the action-normalized loss. These scores are deliberately
not conflated. Both arms start from the same fusion state and use the same chosen
schedule. DPO requires extra rejected forwards; report that compute cost.
`--resume /path/to/checkpoint-N` restores fusion/optimizer/scaler/RNG/cache identity.

## User-controlled refinement

`python -m LLM_branch.inference.optimize` passes ordinary inference flags to the
existing decoder, defaults to `--stage stage3`, and adds:

- `--refinement-enabled` (absent by default: no oracle, SSH or HLS calls).
- `--refinement-budget 4` (logical queries **including baseline**, maximum16).
- `--refinement-policy coordinate` (`structural`/`hybrid` require supported verified relations).
- `--refinement-request FILE --refinement-run-dir DIR`.
- `--refinement-archive FILE` for exact authenticated receipt replay, or
  `--refinement-coroni-runner FILE --refinement-coroni-directory RUN_NAME`
  with `--refinement-max-new-calls 8` for bounded Coroni execution.

Only the selected Stage3 assignment initializes refinement. Without refinement,
`final_assignment` is exactly that complete assignment. With refinement the existing
search controller maintains its paid causal history and returns its best verified,
resource-feasible and HLS-estimated-timing-qualified incumbent. Unknown outcomes
are never numerical scores. The objective is cycles within one fixed requested
period; it does not claim post-route timing. All physical identity fields, source
headers, tool/vendor header versions and flow settings must match.

The backend is **AVAILABLE_EXPERIMENTAL** pending real Stage3-output experiments.
Coordinate search works with source-derived TCAD tuples. RC3 structural-relation
translation remains explicitly unsupported at the public CLI; it fails closed
rather than inventing old graph relations. The original ADP controller entry point
retains its original sealed context and default objective.

Tests:

```bash
python -m pytest -p no:cacheprovider experiments/final_architecture_v3/tests \
  LLM_branch/tests/test_stage3_dpo_contract.py tests/test_refinement_search.py \
  tests/test_refinement.py -q
```

Large pair manifests, model weights, caches and logs are ignored. Compact results,
figures, scripts and source hashes are tracked. Locked test labels remain unopened.
