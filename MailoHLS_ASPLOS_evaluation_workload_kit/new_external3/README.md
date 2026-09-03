# MailoHLS / SALT-HLS external evaluation: CHStone AES, CHStone JPEG, Rosetta 3D Rendering

This package adds three structurally different external workloads to the current
`ElenaVouvali/MailoHLS` `stage2-analysis-refactor` inference contract.

## Why the package materializes pinned upstream code instead of shipping a hand-edited fork

CHStone AES/JPEG are multi-file benchmarks and the Rosetta rendering accelerator uses
Xilinx `ap_int.h`.  To keep the ASPLOS artifact auditable, `materialize_upstream.py`
fetches **pinned upstream Git blobs**, verifies their Git blob hashes, mechanically
flattens benchmark-local includes into a single translation unit, removes only
pre-existing HLS *optimization* pragmas (Rosetta interface pragmas are retained), and
derives the MailoHLS Lk action contract without reading any QoR/result file.

This avoids silently rewriting a benchmark while still producing the same source/action
shape used by the existing MailoHLS external inputs.

Pinned sources:

* CHStone: `ferrandi/CHStone@2b7e20ffd3365016faf1e4e2b86496a5c95445fb`
* Rosetta: `cornell-zhang/rosetta@2feed1ce02d871603bf1fc344a65051837ac780f`

## 1. Materialize the three inputs

On a machine with HTTPS access:

```bash
python materialize_upstream.py
python validate_package.py
```

The generated per-workload directories contain:

* `kernel.c` / `kernel.cpp`: pragma-clean computational source with stable `/*Lk:*/` anchors;
* `kernel_info.txt`: top function and loop/local-array actions;
* `kernel_placeholders.c` / `.cpp`: prompt-facing PIPE/UNROLL/ARRAY_PARTITION template;
* `ACTION_AUDIT.json`: exact actions, static bounds/dimensions, skipped dynamic loops, and any L64 truncation;
* `code_to_memory_outputs/{meta,gexf,pt,memory}`.

If your cluster has no internet, run the materializer elsewhere and copy this entire
package directory to the cluster.  Alternatively provide local pinned checkouts directly:

```bash
python materialize_upstream.py --offline \
  --chstone-root /path/to/CHStone \
  --rosetta-root /path/to/rosetta
```

The script still verifies every Git blob against the pinned hash.

## 2. Generate compiler graphs

From the MailoHLS repository root with the normal MLIR/Polygeist environment active:

```bash
MAILOHLS_ROOT=$PWD /path/to/package/prepare_mlir_graphs.sh
```

Rosetta 3D Rendering includes `ap_int.h`; expose the Vitis HLS include path to the same
cgeist environment used for MailoHLS if it is not already visible.

## 3. Generate checkpoint-compatible structural memory

Do **not** fabricate or reuse another kernel's `.memory.pt`.  Convert each generated
GEXF with the current MailoHLS GEXF-to-PT path and export memory with the exact GNN
checkpoint, feature schema, normalization artifact, `embedding_mode`, `max_slots`, and
relation schema referenced by the final Stage-2 training contract.  Put the resulting
three `.memory.pt` files in a copy of the frozen Stage-2 memory bank; do not modify the
frozen parent `memory_manifest.json`.

## 4. Build inference requests

Example ZCU104 specified-clock requests:

```bash
python make_cases.py \
  --device xczu7ev-ffvc1156-2-e \
  --clock 10.0 \
  --bram 624 --dsp 1728 --ff 460800 --lut 230400 \
  --objective PARETO_ADP \
  --output external3_zcu104_10ns.jsonl
```

Point current `eval_stage1_stage2_stage3.py` at that JSONL, the relevant frozen
checkpoint, and (for Stage 2/3) the memory-bank copy.  Use this package root as the
external `application_dataset_dir` so source-derived directive domains find each
`kernel_info.txt` without adding these workloads to the training corpus.

## Evaluation lock

`EVALUATION_WORKLOAD_LOCK.json` records the proposed nine-workload ASPLOS set:
Kalman, GAN, Jacobi-2D, TRMM-opt, CHStone AES, CHStone JPEG, Rosetta 3D Rendering,
SYR2K, and Covariance.  Keep failed cases and report both kernel-macro and
suite/domain-macro aggregates.
