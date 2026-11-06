# GEMV Stage-1 full-depth diagnostic

This package studies the frozen V4 multi-reference latency Stage-1 model. It does not train a model, inject GNN memory, run synthesis, or modify production inference. The experiment ran 43 complete requests on the required V100S and saved all selected vectors/scores.

## Reports

- `STAGE1_32_LAYER_ANALYSIS.md` — full-depth target and prefix behavior.
- `HARDWARE_ENCODING_AUDIT.md` — hardware prompt tokens and learned embedding rows.
- `GEMV_STRUCTURE_TARGET_ANALYSIS.md` — RC3-verified reduction and syntax controls.
- `XATTN_PLACEMENT_RECOMMENDATION.md` — evidence-ranked Stage-2 design tests.
- `PRESENTATION_NOTES.md` — eight-slide narrative.
- `DIRECTIVE_GRID.md` — all 18 canonical target predictions.

## Reproduction

Run from `/tmp/MailoHLS-gemv-stage1-full-depth` with `mailohls-llm-v2`, local base-model cache, and read-only access to `/tmp/MailoHLS-gnn-refinement` for the authenticated RC3 compiler pipeline:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.prepare --out /tmp/gemv-full-depth-run

CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.run --out /tmp/gemv-full-depth-run

CUDA_VISIBLE_DEVICES=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.patching --out /tmp/gemv-full-depth-run

MPLCONFIGDIR=/tmp/gemv-full-depth-mpl PYTHONPATH=. MPLBACKEND=Agg \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.analyze --out /tmp/gemv-full-depth-run

PYTHONPATH=. /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.report --out /tmp/gemv-full-depth-run

PYTHONPATH=. /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_stage1_full_depth.checks --out /tmp/gemv-full-depth-run
```

`analyze` regenerates all PNG/SVG plots and numerical tables from saved CPU-side evidence without loading the LLM. Do not reuse an existing output directory for inference; the launcher refuses overwrite. The report’s checkpoint path, source bytes, request/prompt hashes, RC3 toolchain, and model outputs are retained in `artifacts/gemv_stage1_full_depth/v1_20261106/`.

Focused synthetic read-only hook test:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m pytest -q experiments/gemv_stage1_full_depth/test_full_depth.py
```

Original source and action metadata are byte-identical copies from the GNN RC3 authority. The task’s original GEMV is intentionally not numerically valid as a complete GEMV because it has uninitialized local arrays and only `y[0]` is observable; new source variants are separate initialized diagnostics.
