# Structural representation audit

This read-only CPU diagnostic checks the path from labeled C/C++ source through
the compiler graph, action projection, saved GNN tensors, structural memory and
Stage-2 routing. It does not launch HLS, load the language model, run the GNN,
change model behavior or regenerate production artifacts.

`audit.py` is the only implementation module. It reuses the actual production
projector and pure encoder function bodies through scoped AST extraction,
avoiding their training CLI side effects. Existing refinement, inference,
training and graph generator files remain unchanged.

The optional [V1/V2 comparison](../../docs/structural-graph-v2.md) is exposed by
`python -m experiments.structural_audit.compare_v2`. It reuses the same raw MLIR
and source references, validates call-scoped V2 queries and compact-view
preservation, and keeps this original audit and all production consumers unchanged.

## Run from the repository root

Use the project's Python environment with torch, networkx and pytest (Python
3.10+). No GPU is needed.

```bash
# Independently specified unit fixtures; no compiler invocation.
python -B -m experiments.structural_audit.audit --mode fixtures \
  --output-dir /tmp/mailohls-structural-unit-check

# Also generate eight tiny graphs using the existing pinned native pipeline.
python -B -m experiments.structural_audit.audit --mode fixtures --compile-fixtures \
  --polygeist-build /path/to/Polygeist/build-mailohls-assertions \
  --mlir-python /path/to/python3.11

# Audit real kernels; include the preceding compiler receipts in the report.
python -B -m experiments.structural_audit.audit --mode audit \
  --kernels GAN covariance Kalman

python -B -m pytest -q -p no:cacheprovider tests/test_structural_audit.py \
  tests/test_refinement.py tests/test_refinement_search.py \
  tests/test_refinement_benchmark.py tests/test_directive_domains.py \
  LLM_branch/tests/test_sparse_memory_slots.py \
  LLM_branch/tests/test_structural_xattn.py \
  GNN_branch/tests/test_multiscale_memory.py GNN_branch/tests/test_mlir_top_selection.py
```

The default audit also includes five original kernels: simple arithmetic,
a helper call, nested GEMM, indirect-index SpMV and a multi-function LU kernel
with memory views. Use `--no-include-originals` for only the requested kernels.
Selection is fixed independently of projector successes.

`--shared-manifest PATH` selects an existing refinement source/graph archive;
the default is `experiments/refinement/results/search/pilot/shared/manifest.json`.
Source, metadata and graph hashes are authenticated before comparison. The
original kernels use tracked `GNN_branch/MLIR_graphs` and hash-matched sources
under `Data/ApplicationDataset`. Matching raw IR is recovered from tracked
`audit_mlir` or the local R0 archive by its recorded hash. Missing prerequisites
are explicit blocked/partial diagnostics. No LLVM/HARP graph is substituted.
Saved schema/contracts, tensors and memory packs are local research artifacts;
without them the semantic audit still runs, but learned-interface checks are
reported unavailable. `FINAL_ARTIFACTS.json` selects the actual production ADP
Stage-2 and GNN artifacts; pending area/latency experiments are not treated as
production checkpoints.

## Independent references and uncertainty

`fixtures/annotations.json` contains source-written expected labels, including
positives, source-supported negatives and an unresolved alias case. These labels
are **machine-assisted drafts**, not independently reviewed ground truth. Each
output record attaches source quotes/line numbers, hashes, compiler operations,
graph edges and projection observations **after** loading the expectation.
Expected labels do not come from the projector or verifier being evaluated.

Use `--annotations PATH` for reviewed records. An independently reviewed record
must explicitly include `review_status: independently_reviewed`, `reviewer`,
and should pin `source_sha256`. The tool verifies pinned hashes and source
quotes; it cannot authenticate a human reviewer's identity or judgment.
Unreviewed records contribute only `draft_TP/FP/FN/TN`, never reviewed precision
or recall. Unresolved facts and inactive actions are excluded from negatives.
No confidence intervals or statistical generalization are claimed.

`fixtures/semantics.json` specifies eight small programs and expected semantic
checks before running either implementation. Unit graphs are explicitly
hand-authored inputs, not fabricated compiler outputs. With `--compile-fixtures`,
the actual pinned generator produces separate GEXF, raw MLIR, native analysis
and logs. Source expectations are checked against those outputs; disagreements
remain visible rather than changing labels to fit the implementation. A positive
unit case with precise call IDs demonstrates the required future evidence; the
legacy compiler schema does not supply those IDs and must remain conservative.

## Isolated verification corrections

The versioned `strict_callsite_and_access_v1` audit policy operates on a graph
copy and calls the unchanged legacy verifier:

1. Actual-to-formal bindings need an explicit originating `callsite`, observed
   actual operand position, consistent callee and binding position. Matching
   argument position and callee alone is insufficient.
2. Native writes are operation→root (`write`), with root→operation
   `write_reverse` edges. The old verifier overlooks this orientation. The audit
   recognizes a write only with both matching native edges, recording them.
3. A `func.call` memory-effect summary or a root propagated across function
   contexts is not a concrete load/store in a loop's own region. It cannot become
   direct proof just because it has a root-access edge.

This policy is **not installed into production or existing search policies**.
Past queues remain reproducible. Future refinement evidence can explicitly use
this stricter audit interface; migrating the graph schema or cross-attention
requires a separate controlled change.

The GAN regression illustrates why this distinction matters. L2–L11 is a legacy
access overapproximation through shared helpers. L3–L11 is source-positive, but
the legacy verifier's “direct” witness is a call summary. Under strict compiler
proof requirements it is uncertain until a specific call binding is available.
That is an evidence-confidence correction, not a claim that the source access
does not exist.

## Outputs and reproducibility

Generated files remain ignored under `results/`:

- `manifest.json`: actual input/code/checkpoint hashes, Git/worktree state,
  environment and protected historical journal hashes.
- `relationship_annotations.jsonl`: reviewable source expectations and observed
  source→anchor→root/operation→edge→relation evidence.
- `relationship_comparison.csv`: reviewed and draft counts, uncertainty,
  reviewed precision/recall only where applicable.
- `relation_errors.jsonl`: explicit missing prerequisites and disagreements.
- `representation_trace.json`: checkpoint features/flags, exact saved-tensor
  comparisons, action alignment, memory checks and a future ranker feature contract.
- `structural_audit_report.md`: findings, limitations and priorities.
- `fixtures/results.json`: unit and optional real compiler outcomes/provenance.

Use `--output-dir PATH` for another audit. A prepared manifest cannot be reused
with changed input hashes; select a new directory. Audit input files and protected
historical journals are checked again after execution. All original branches,
graphs, checkpoints and measurements are preserved.
The real-kernel audit also works on its own. For a combined report, run compiled
fixtures first as above. Adding fixture receipts to an already frozen real-kernel
audit requires a new directory. Completed fixture runs authenticate and reuse
their existing artifacts rather than recompiling them.

The saved GNN uses typed categorical/numeric edge features. The inspected
Stage-2 path uses Boolean action visibility, rather than explicit typed relation
biases. Exact tensor/slot checks establish interface preservation, **not** that
every compiler fact can be recovered from a learned embedding. No embedding
probe, QoR ranker or typed-attention ablation is trained here.

Semantic basis: [MLIR Func](https://mlir.llvm.org/docs/Dialects/Func/),
[MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/),
[Affine](https://mlir.llvm.org/docs/Dialects/Affine/) and the
[language reference](https://mlir.llvm.org/docs/LangRef/).
