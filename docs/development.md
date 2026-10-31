# Developer conventions

Extend maintained code in `GNN_branch/`, `LLM_branch/` or `Preprocessing/`.
Keep research-specific methods/configurations/reports under `experiments/` and
launchers under `scripts/`; use existing modules before adding another entry
point. Preserve source-legal directive domains, tokenizer IDs and checkpoint schemas.

Twelve legacy `src/` entries (the mapping is in [legacy-entrypoints.json](legacy-entrypoints.json)) execute the canonical source in their original module namespace.
This keeps old imports, pickle class names, `__main__` behavior and caller-selected
LLVM-era sibling imports. The rest of `src/` contains distinct historical code.
Do not change its import resolution merely to modernize syntax. The original
exact source copies remain at maintenance base `7c6844f` in Git.

Runtime weights, tensors, logs, HLS build trees and large inventories belong in
ignored output paths. Do not overwrite selected or active runs. Register immutable
artifacts in `configs/artifacts.json`, providing SHA256, size, schema, immutable source commit and sealed member
identity. Prefer exact recovery from reachable Git objects. Do not publish a
Release for each experiment. Remove scientific content only after validated
restoration and preservation. Keep raw datasets and split membership
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

## Three independent research branches

Work on `research/stage1-refinement` for LLM/Stage-1 changes,
`research/gnn-refinement` for graph/encoder/training changes, and
`research/refinement-mechanism` for feedback/search. These share finalized `main`
and retain independent experiment modules, configurations and results.

```bash
git fetch origin
# With a clean research worktree on its own branch:
git merge origin/main
# Resolve conflicts explicitly, run relevant tests, then push that branch.
```

Do not reset a research branch to main, force-push, or merge all unfinished work
into production. Integrate a mature improvement selectively after checking its
data, schema and checkpoint contracts. Keep the two historical branches immutable.
Existing local worktrees on retired branch names remain valid; when idle, preserve
their changes, create a fresh worktree for the corresponding canonical research
branch, and deliberately transfer any additional changes. Never switch/reset a
running or dirty checkout merely to update its upstream.
