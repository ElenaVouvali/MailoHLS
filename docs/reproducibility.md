# Reproducing MailoHLS

Use the selected paths and status in [`FINAL_ARTIFACTS.json`](../FINAL_ARTIFACTS.json).
Weights are supplied separately. Do not substitute a recovery adapter or a V2.2
GNN into that production contract. Local ignored runtime assets are not guaranteed
to exist in a fresh checkout. Stage-2 initial states and automatic-clock assets
remain required as explained in [repository layout](repository-layout.md).

## Artifacts

The registry is [`configs/artifacts.json`](../configs/artifacts.json). Raw
measurements and frozen compiler archives remain tracked. Historical/generated
payloads are recovered from exact original Git objects; no new Release is part
of normal development. Use a full clone and fetch preservation tags if necessary:

```bash
git fetch origin --tags
python scripts/artifacts.py restore-git --artifact harp-generated-graphs-v1
python scripts/artifacts.py restore-git --artifact derived-training-tables-v1
```

Other exact-path groups are `checkpoint-evidence-v1` and `synthesis-evidence-v1`.
The source commits remain reachable through main/research history and existing
preservation tags. Recovery checks source object IDs and the original sealed
member manifest (per-file SHA256, size, Git blob and executable mode). It rejects
conflicts and unsafe links and leaves identical files alone. The archive's gzip
bytes can depend on the runtime; `restore-git` authenticates the original payload,
not a silently substituted archive checksum. Original compressed archives and
checksums are also retained in the verified durable offline preservation set.

For the complete graph catalog:

```bash
python scripts/artifacts.py restore-git --artifact v22-complete-target-catalog-v1 --root /fresh/review
python scripts/graphs/publish_catalog.py verify --catalog /fresh/review/graphs/latest
```

This recovers the original catalog from commit `42f52df`, verifies the sealed
checksum file and all 2,218 payload entries, and does not rerun the compiler.
`publish_catalog.py build` can regenerate graphs from retained RC3 inputs and
profiles, but runtime receipts/provenance text need not reproduce the published
archive byte-for-byte. Exact catalog recovery uses Git history instead.

```bash
python -m scripts.maintenance.validate_restored_research --git-source --root . --report /tmp/restored-consumers.json
```

The [migration manifest](../configs/artifact-migration.json) maps externalized
paths to their original snapshots. All archival packaging reads committed blobs,
never ignored local weights or active outputs. Raw GNΩSIS data, selected contracts
and source/action mappings are not modified. Existing benchmark/baseline licenses
remain applicable; HARP's BSD notice is retained. `verify` can still authenticate
an original offline archive. No new tags or Releases are needed for routine work.

## Dataset and compiler

Original measurements are in `Data/CSVS/`; source/metadata/APL identity and the
seed-123 split are scientific inputs. The preparation entry point is:

```bash
bash scripts/reproduction/prepare_final_experiment.sh
```

Inspect settings before execution; preparation writes derived data. Use
[`toolchain/README.md`](../toolchain/README.md) for the pinned Polygeist/native
analysis build. Read frozen V2.2 graphs without a compiler using the artifact tool;
new graphs use the explicit revision command in [the compiler guide](structural-graph-v22.md).
Do not repair source labels or substitute modified Kalman data silently.

## Training, inference and synthesis

[Shell launchers](shell-scripts.md) list exact maintained commands and historical
path replacements. Production launchers are `scripts/training/train_gnn.sh`,
`train_stage1.sh`, `train_stage2.sh` and `export_gnn_memory.sh`. Contract-driven
Stage 3 uses `LLM_branch/inference/run_stage3_contract.sh`. These execute real
jobs and require configured weights, datasets, dependencies and fresh outputs.

Inference entry point `LLM_branch/inference/eval_stage1_stage2_stage3.py` accepts
the model and memory contracts; consult its `--help` and [shell guide](shell-scripts.md).
The [refinement experiment](../experiments/refinement/README.md) documents explicit
Vitis synthesis execution and report reuse. No new measurements are required to
inspect graphs or verify artifact integrity.

## Experiments and paper evidence

[Experiment index](experiments.md) routes to original configurations, outcomes and
limitations. `paper_plots/`, `plots/`, final figures and historical checkpoint
source snapshots remain intact. [Third-party provenance](third-party-provenance.json)
records the HARP source; the modified baseline remains in `HARP-rep/`. LIFT paths
and historical launchers retain their original environments and machine paths.
A fresh clone can run artifact verification and CPU fixtures without local model
caches; full training, inference and synthesis require separately supplied tools
and weights. Missing dependencies are reported as limitations, never passing results.
