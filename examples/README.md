# GEMV: source actions to compiler evidence

This small inspection path uses checked-in files and makes no model, GPU or HLS
calls. It demonstrates the historical production graph lineage; it does not
substitute RC3 or a new latency encoder into its checkpoint contract.

Run the standard-library inspection from any working directory:

```bash
python examples/inspect_gemv.py
```

It authenticates the retained graph against the figure's SHA-256, checks the
metadata action set/kinds, counts graph elements, and prints the selected
absolute memory slots. It does not construct learned memory or a model output.

1. Compare the [original source](../evaluation_set/gemv/gemv.c) with the
   [historical annotated source](../evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c)
   and [graph-side metadata](../evaluation_set/gemv/code_to_memory_outputs/meta/kernel_info.txt).
2. Inspect the [structured MLIR](../paper_examples/gemv/gemv.mlir) and
   [compiler graph](../paper_examples/gemv/gemv_mlir.gexf).
3. Inspect the [reduced graph provenance](../paper_figures/gemv_mlir_reduced.json)
   and [figure](../paper_figures/gemv_mlir_paper.svg).

The verified walkthrough uses L1 for array A and L4/L5 for the loop actions.
An exported action memory must retain slots 0, 3 and 4 respectively; absent IDs
do not shift later slots. Source-domain legality is checked before a complete
assignment can be accepted. Graph edges and decoder routing are different
relations: routing cannot manufacture compiler dependence evidence.
These source variants are retained as distinct historical inputs; the inspection
does not claim their source bytes, frontend inputs or functional behavior are identical.

For a real one-pass model run, follow the authenticated checkpoint gate in
[reproduction](../docs/REPRODUCIBILITY.md). It remains unperformed for this
consolidation. The opt-in refinement path is documented separately in the
[maintained protocol](../experiments/final_architecture_v3/README.md).
