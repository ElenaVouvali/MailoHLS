# GEMV 32×32 — standalone structural V2.2 RC3

Open `gemv.rc3.overview.svg` (or PNG/PDF) for the operation overview and `gemv.rc3.full.svg` for the complete compact view. Canonical JSON contains 66 nodes / 106 edges; compact JSON contains 49 / 89. GEXF is provided for Gephi (nested attributes encoded as JSON strings).

All five source actions match: L1=A, L2=x, L3=y, L4=outer loop, L5=inner loop. Both compiler-proven loop trip counts are 32. Source y[32] is promoted to scalar compiler storage; its actual compiler memref type is preserved.

Purple dashed edges are action targets, green edges are memory accesses, orange edges are compiler-proven depends_on in native graph direction. Blue overview edges are explicitly labelled SSA projections through defines/operand paths. Grey edges show sequencing and loop nesting. Overview scope/SSA nodes are omitted only from display. Full canonical JSON is authoritative; parallel identical visual edges are grouped in the drawings.

GEMV is outside the frozen 63-snapshot RC3 release. This separate generation uses the archived, labelled 32×32 prepared recovery source and matching metadata, not the older V1 paper graph. Source/metadata hashes and graph-construction code match the RC3 archive. Inputs and frontend MLIR are included. The frozen archive is unchanged. Native repeat generation agrees exactly on nodes/edges; DLTI attribute ordering changes raw-module hash and graph identity (see manifest).

Re-render from the research branch root:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python paper_figures/gemv_rc3_20261105/render.py
```

Native-generation CLI: `GNN_branch.mlir_graph_v2 --revision 2.2 --semantic-revision rc3`, source `evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c`, metadata `evaluation_set/gemv/code_to_memory_outputs/meta/kernel_info.txt`. Use the pinned Polygeist build recorded in `manifest.json` and its MLIR Python bindings. No GPU, model training or synthesis is required.


## Trained V4b embedding exports

`embeddings/` contains actual compiler, hardware-message and target-FiLM memory
matrices, active-slot CSVs, heatmaps and the checkpoint-hash report for this
standalone GEMV graph at 10 ns on ZCU104/U200. These are out-of-cohort inference
outputs, not new training examples or evidence of hardware selection benefit.
No candidate/QoR labels are fabricated. The frozen training cache has no GEMV.

The original inspector was invoked with `PYTHONPATH` set to the repository
root. `inspect_embeddings.py` includes that repository import bootstrap and
resolves the committed checkpoint manifest relative to the repository, so it
can be invoked from other working directories. It additionally records input
provenance; the computation is unchanged. The stored export report was produced
by the original inspector. Cross-arm `film_effect` plots compare independently
trained checkpoints and do not isolate a causal FiLM-gate effect.

Reproduce from the repository root using the pinned V4b Python environment:

```bash
PYTHONPATH=. /tmp/mailohls-v22-gnn-env-20261030/bin/python paper_figures/gemv_rc3_20261105/prepare_embedding_cache.py --output /tmp/gemv_rc3_embedding_input.pt
/tmp/mailohls-v22-gnn-env-20261030/bin/python paper_figures/gemv_rc3_20261105/inspect_embeddings.py --cache /tmp/gemv_rc3_embedding_input.pt --run-dir /home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37 --kernel gemv --clock 10 --profiles zcu104 u200 --out-dir /tmp/gemv_rc3_embeddings_fresh
```

Use new output paths. The trusted local model checkpoints and input tensor cache
remain outside Git. Exact checkpoint/cache hashes accompany the exports.
