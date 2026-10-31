# Reproducing MailoHLS

Use the selected paths and status in [`FINAL_ARTIFACTS.json`](../FINAL_ARTIFACTS.json).
Weights are supplied separately. Do not substitute a recovery adapter or a V2.2
GNN into that production contract. Local ignored runtime assets are not guaranteed
to exist in a fresh checkout. Stage-2 initial states and automatic-clock assets
remain required as explained in [repository layout](repository-layout.md).

## Artifacts

The standard-library registry is [`configs/artifacts.json`](../configs/artifacts.json),
schema `mailohls-artifact-registry-v1`. Each record has an ID, path, SHA256, byte
size, source commit, graph/dataset identity, dependencies, reproduction guide and
public URL where available. Current entries are tracked compiler graph releases.
Their URLs reference immutable published Git commits. RC3 was independently
downloaded and checksum-verified during maintenance; the other entries were
verified locally. No GitHub Release download is asserted. No scientific artifact was removed in this refactor.

```bash
python scripts/artifacts.py list
python scripts/artifacts.py verify --artifact salt-hls-graphs-v22-20261027-rc3
python scripts/artifacts.py fetch --artifact salt-hls-graphs-v22-20261027-rc3 --root /tmp/mailohls-restored
python scripts/artifacts.py verify --artifact salt-hls-graphs-v22-20261027-rc3 --root /tmp/mailohls-restored --extract-to /tmp/mailohls-graphs
```

Choose fresh restore/extraction paths. Fetch requires HTTPS; verification is
offline. Existing files are never overwritten. Downloaded bytes are checked before
publication. Extraction rejects absolute/traversing paths, links, devices and
special files, validates all members before writing, and stages content in the
destination filesystem. Restoring a registry entry does not establish that its
scientific interpretation has been independently adjudicated.

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
