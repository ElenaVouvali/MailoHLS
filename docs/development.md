# Developer conventions

Extend maintained code in `GNN_branch/`, `LLM_branch/` or `Preprocessing/`.
Keep research-specific methods/configurations/reports under `experiments/` and
launchers under `scripts/`; use existing modules before adding another entry
point. Preserve source-legal directive domains, tokenizer IDs and checkpoint schemas.

Four legacy `src/` entries (`data`, `cpp_to_gnn_emb`, `filter_jsonl`,
`inference_qLoRA`) execute the canonical source in their original module namespace.
This keeps old imports, pickle class names, `__main__` behavior and caller-selected
LLVM-era sibling imports. The rest of `src/` contains distinct historical code.
Do not change its import resolution merely to modernize syntax. The original
exact source copies remain at maintenance base `7c6844f` in Git.

Runtime weights, tensors, logs, HLS build trees and large inventories belong in
ignored output paths. Do not overwrite selected or active runs. Register immutable
artifacts in `configs/artifacts.json`, providing SHA256, size, schema, commit and
real URLs. Remove tracked scientific content only after downloading and verifying
an independently accessible replacement. Keep raw datasets and split membership
unchanged; dataset corrections need their own version and compatibility review.

For FPGA profiles, use the independently validated registry and its tests; retain
BRAM18K units, sourced capacities and separate deployment budgets. For experiments,
record source/data/profile/schema identities, train-only normalization, deterministic
seeds and actual outcomes. Do not equate mechanical validity with QoR improvement.

```bash
python -m pytest -q tests LLM_branch/tests
python scripts/maintenance/inventory.py --ref HEAD --output /tmp/mailohls-inventory-new
python scripts/artifacts.py verify --artifact salt-hls-graphs-v22-20261027-rc3
```

The inventory uses committed Git blobs, records exact duplicate hashes and AST
imports, and marks unresolved ownership/dependency decisions for human review.
It does not prove that unreferenced code is dead. Large reports stay outside Git;
commit concise findings and reproduction commands. Use clean worktrees, logical
commits and reviewable PRs; never clean another researcher's checkout.
