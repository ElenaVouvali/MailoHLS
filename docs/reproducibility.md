# Reproducing MailoHLS

Use the selected paths and status in [`FINAL_ARTIFACTS.json`](../FINAL_ARTIFACTS.json).
Weights are supplied separately. Do not substitute a recovery adapter or a V2.2
GNN into that production contract. Local ignored runtime assets are not guaranteed
to exist in a fresh checkout. Stage-2 initial states and automatic-clock assets
remain required as explained in [repository layout](repository-layout.md).

## Artifacts

The registry is [`configs/artifacts.json`](../configs/artifacts.json). Frozen
compiler archives stay tracked; raw Git URLs for those copies are not considered
independent archival preservation. The following genuine Releases have been
published, downloaded independently over public HTTPS, restored and validated:

- [Complete V2.2 catalog](https://github.com/ElenaVouvali/MailoHLS/releases/tag/archive/v22-graph-catalog-20261029): all 1,512 overlays, source/MLIR/profile/action provenance and 2,218 verified checksums.
- [Research evidence v1](https://github.com/ElenaVouvali/MailoHLS/releases/tag/artifacts/research-evidence-v1-20261031): historical checkpoint evidence, generated HARP graphs, full synthesis snapshots and derived training CSVs. Includes public checksum and artifact manifests. No local model weights are included.

Restore only the groups needed for an experiment:

```bash
python scripts/artifacts.py list
python scripts/artifacts.py fetch --artifact harp-generated-graphs-v1
python scripts/artifacts.py restore --artifact harp-generated-graphs-v1
python scripts/artifacts.py fetch --artifact derived-training-tables-v1
python scripts/artifacts.py restore --artifact derived-training-tables-v1
```

The other exact-path groups are `checkpoint-evidence-v1` and
`synthesis-evidence-v1`. Both use the same fetch/restore commands. Archives are
cached at ignored `artifacts/releases/`. `--root /fresh/reproduction` explicitly
chooses a different restoration root, including the cache. Existing identical
files are verified and left alone; differing files, modes or symlink paths are
rejected rather than overwritten. A conflict must be reviewed by the user, not
resolved with a force flag. A failed concurrent publication may leave a subset
of verified files; rerun safely after resolving the conflict. Individual file
publication is atomic, while a multi-file restore is not a filesystem transaction.

Each sealed tree records its immutable source commit, every original file path,
Git blob, SHA256, size and Git executable mode in `MAILOHLS_ARTIFACT.json`.
Archive checksums bind that manifest. Offline `verify` checks an existing cache;
`restore` verifies every member and original path. Links, devices, traversal and
unexpected members are rejected. The [migration manifest](../configs/artifact-migration.json)
records exactly which indexed copies were externalized and which contracts remain.
Only files from already-public committed Git trees were packaged; existing rights
are unchanged and HARP's BSD license is included. Raw GNΩSIS redistribution terms
were not re-licensed, and its original CSVs remain tracked.

The legacy catalog uses its own frozen checksum layout and extracts to a fresh
review directory rather than installing into a source checkout:

```bash
python scripts/artifacts.py fetch --artifact v22-complete-target-catalog-v1
python scripts/artifacts.py verify --artifact v22-complete-target-catalog-v1 --extract-to /fresh/catalog
# From /fresh/catalog/graphs/latest: sha256sum -c SHA256SUMS
```

Maintainers can reproduce any sealed tree using only committed blobs:

```bash
python scripts/artifacts.py package --artifact checkpoint-evidence-v1 --output /fresh/checkpoint-evidence-v1.tar.gz
python -m scripts.maintenance.validate_restored_research --root /fresh/restored --report /fresh/consumer-report.json
```

Packaging refuses existing outputs, symlinks and model-weight extensions; ignored
local files are not read. Publishing never uses overwrite/clobber flags. Versioned
assets and preservation tags must not be rewritten; create a new version instead.
After a download, verify the registered checksum even if GitHub reports a digest.
No externalized scientific content is regenerated or relabeled during restoration.

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
