# MailoHLS final patch bundle

Audited against `ElenaVouvali/MailoHLS`, branch `stage2-analysis-refactor`, commit
`a6e38321f92fb0401737e74b98b8d1ae091999b6` (`GNN final`).

## Apply and inspect

```bash
python apply_mailohls_final_simplification_patch.py
python -m pytest -q LLM_branch/tests/test_stage1_final_contract.py
python -m pytest -q GNN_branch/tests
git diff --check -- LLM_branch GNN_branch
git diff -- LLM_branch GNN_branch
```

The patch intentionally refuses a different HEAD or modified audited source
blob. Review the diff before committing.

## Optional: add GNΩSIS LUD0 to the Stage-1 corpus

```bash
python import_gnosis_lud0.py
python -m Preprocessing.data_preprocess --mode llm --force
python -m Preprocessing.create_jsonl --force
python -m Preprocessing.build_family_split \
  --dataset_jsonl artifacts/llm/mailohls_sft.jsonl \
  --output_json mailohls_runs/mailohls_final_family_split_s123.json \
  --seed 123 \
  --val_kernels "machsuite-sort-radix,machsuite-viterbi,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0" \
  --test_kernels "serrano-kalman-filter"
```

LUD0 is imported only for the LLM corpus. The existing GNN MLIR corpus does not
need to be regenerated merely to add this Stage-1 example.

## Commit before the final Stage-1 run

`run_final_mailohls_stage1.sh` uses `--require_clean_git`, so commit the final
tracked source/data changes after tests pass.

## Final GNN

```bash
bash run_final_mailohls_gnn.sh
```

Final GNN methodology:

- primary QoR target: latency reference delta;
- physical resource heads: BRAM, DSP, FF, LUT;
- no separate aggregate-area target;
- no pairwise ranking loss;
- checkpoint selection, plateau scheduling, and early stopping all use the
  complete validation training objective.

## Final Stage-1

```bash
bash run_final_mailohls_stage1.sh
```

Stage-1 optimization uses only full RHS cross-entropy. Checkpoint selection is
lexicographic:

1. kernel-macro static-field accuracy;
2. kernel-macro joint-action accuracy;
3. negative validation CE.

The two accuracies are validation/model-selection metrics, not training losses.
