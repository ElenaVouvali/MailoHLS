# Proposed compact MailoHLS layout

Status: proposal only, based on the checkout inspected on 2026-10-26. No source,
dataset, model, or result directories have been moved by this proposal.

## Recommended top-level layout

Keep the familiar `GNN_branch`, `LLM_branch`, and `Data` names. Add `Refinement`,
`Experiments`, `Baselines`, and `Models`, and retain `docs` and `tests`. These
nine directories provide distinct owners for implementation, inputs, trained
artifacts, experiments, comparison systems, documentation, and verification.
Use the exact capitalization consistently in imports, commands, and links.

```text
MailoHLS/
├── README.md
├── FINAL_ARTIFACTS.json
├── requirements.txt
├── requirements-llm-colab.txt
├── .gitignore
├── GNN_branch/
│   ├── README.md
│   ├── graphs/                 # graph builders, schemas, action queries
│   ├── memory/                 # structural-memory exporters and validators
│   ├── tools/                  # diagnostics and analysis utilities
│   ├── scripts/                # GNN training and memory-export launchers
│   ├── toolchain/              # pinned compiler source locks and patches
│   ├── legacy/                 # distinct historical GNN implementations
│   └── ...                     # model, loader, config, training entry point
├── LLM_branch/
│   ├── README.md
│   ├── common/                 # contracts, directive domains, attention
│   ├── train/                  # SFT, structural conditioning, DPO
│   ├── inference/              # generation and evaluation
│   ├── clock_adapt/            # automatic-clock implementation
│   ├── tools/                  # checkpoint and routing diagnostics
│   ├── scripts/                # LLM training/evaluation launchers
│   └── legacy/                 # distinct historical LLM implementations
├── Data/
│   ├── README.md
│   ├── kernels/{training,evaluation,examples}/
│   ├── metadata/               # source/action maps and target information
│   ├── raw/                    # original synthesis measurements
│   ├── processed/{gnn,llm}/    # validated model-specific tables and corpora
│   ├── graphs/{mlir,harp}/     # graph artifacts, separate from builder code
│   ├── splits/                 # frozen family and adaptation split manifests
│   ├── preprocessing/          # shared measurement/data preparation code
│   └── cache/                  # ignored regenerable tensors
├── Refinement/
│   ├── README.md
│   ├── core/                   # assignments, relations, synthesis, CLI
│   └── search/                 # policies, controller, matched-budget benchmark
├── Experiments/
│   ├── README.md
│   ├── reproduction/           # end-to-end preparation and orchestration
│   ├── presets/                # representation, loss, objective, seed variants
│   ├── ablations/              # fixed-checkpoint memory/target comparisons
│   ├── adaptation/             # U50 experimental preparation and execution
│   ├── structural_audit/       # cross-component diagnostics and fixtures
│   ├── latency_target_search/ # experiment-specific driver
│   ├── analysis/               # measured-result selection and plotting code
│   ├── results/                # curated measurements and compact summaries
│   ├── notebooks/              # exploratory notebooks
│   └── workspaces/             # ignored HLS projects, raw run output, caches
├── Baselines/
│   ├── README.md
│   ├── HARP/                   # vendor tree, license and reproduction patches
│   └── LIFT/                   # comparator implementation and configuration
├── Models/
│   ├── README.md
│   ├── gnn/                    # checkpoints and their original run metadata
│   ├── llm/                    # selected adapters, runs and initial states
│   ├── memory/                 # learned memory packs and control variants
│   ├── clock/                  # selected automatic-clock assets
│   └── baselines/              # local comparator checkpoints
├── docs/
│   ├── README.md
│   ├── ...                     # pipeline, setup, schemas, migration, provenance
│   └── paper/{figures,examples,manuscripts}/
└── tests/
    ├── README.md
    ├── gnn/
    ├── llm/
    ├── data/
    ├── refinement/
    └── integration/
```

Subdirectories describe responsibilities. Keep small, cohesive module groups
flat; do not add a directory per Python file, seed, paper section, or revision.
The tree describes the intended end state rather than an instruction to create
empty directories in advance. Additional packaging or test configuration may
remain at the root when introduced for an actual implementation need.

## Why these boundaries fit the existing implementation

- `Refinement` is a meaningful component: its existing four-module core owns
  assignment validation, compiler relations and HLS execution. The separate
  search package imports that implementation. Combining their ownership keeps
  one synthesis and measurement-authentication implementation.
- `Experiments` owns study setup, comparisons, and reporting. The reusable
  refinement algorithm belongs in `Refinement`; fixed seeds, budgets, and
  study outcomes belong in `Experiments`.
- `Baselines` keeps upstream HARP code and LIFT comparison logic discoverable
  without mixing them with MailoHLS production model code. Preserve HARP's
  internal directory layout and license during the initial relocation.
- `Models` consolidates the current checkpoint, adapter, memory, and clock
  roots. It contains trained artifacts and their contracts, not architecture
  implementation. Keep `model.py` with the component that implements it.
- `Data` becomes the owner of model-independent sources and measurements as
  well as explicitly identified derived datasets. GNN and LLM processed tables
  remain separate because their preprocessing policies differ.
- `docs/paper` groups publication assets. Measured experiment records belong
  under `Experiments/results`, and plotting programs under
  `Experiments/analysis`; figure exports belong under `docs/paper/figures`.
- Central `tests` removes the present split across root, GNN, LLM and
  preprocessing test directories. Component READMEs link to their tests.

The compiler toolchain moves beside its graph frontend under
`GNN_branch/toolchain`. Refinement and audits reuse that compiler; relocating
its ownership must not create copies of locks, patches, or native analysis.
An independent top-level compiler package can be considered if its standalone
public interface grows, but is unnecessary for this compact layout.

## Mapping the current directories

| Current path | Proposed owner and destination |
| --- | --- |
| `GNN_branch/` | Keep the owner; group graph, memory, diagnostics and launchers internally. |
| `LLM_branch/` | Keep `common`, `train`, `inference`, `clock_adapt` and `tools`; add its README and owned launchers. |
| `Data/ApplicationDataset/` | `Data/kernels/training/`. |
| `evaluation_set/` | `Data/kernels/evaluation/`; preserve complete per-kernel header and metadata layouts. |
| `Data/GAN/`, `Histogram/` | `Data/kernels/` under their documented training/evaluation/example role; compare source versions before consolidating with other GAN/histogram copies. |
| `Data/ApplicationAPLMapping/`, application information CSVs | `Data/metadata/`, retaining action identities and source bindings. |
| `Data/CSVS/` | `Data/raw/`; original measurements retain their contents and provenance. |
| `Preprocessing/` | Code to `Data/preprocessing/`; tests to `tests/data/`. |
| `GNN_branch/Data/`, `LLM_branch/Data/` | `Data/processed/gnn/` and `Data/processed/llm/`; do not merge tables by filename. |
| `GNN_branch/MLIR_graphs/`, `HARP_graphs/` | `Data/graphs/mlir/` and `Data/graphs/harp/`. Keep production and research schemas explicit. |
| `GNN_branch/MLIR_dataset/`, `artifacts/gnn_datasets/` | Dataset manifests to `Data/processed/gnn/`; regenerable binary tensors to ignored `Data/cache/gnn/`. |
| `old_harp_graphs/` | Historical graph artifacts to `Data/graphs/harp/legacy/`; source/header material to its provenance-preserving kernel owner, with relative symlinks repaired. |
| `experiments/refinement/`, `experiments/refinement_search/` | Implementation to `Refinement/core/` and `Refinement/search/`. New working output uses `Experiments/workspaces/refinement/`. |
| Other `experiments/` packages | `Experiments/` by purpose; retain distinct experiments and exact parameter sets. |
| `scripts/training/`, `scripts/evaluation/` | GNN launchers to `GNN_branch/scripts/`, LLM/clock launchers to `LLM_branch/scripts/`; cross-component evaluation to `Experiments/reproduction/`. |
| `scripts/reproduction/` | `Experiments/reproduction/`. |
| `scripts/experiments/` | `Experiments/presets/`; preserve actual loss, representation and objective differences. |
| `scripts/asplos_ablation/` | `Experiments/ablations/` with the corresponding experiment. |
| Root-level `scripts/*.py` | Synthesis drivers to the owning experiment; analysis/selection programs to `Experiments/analysis/`. |
| `scripts/legacy/`, historical `src/` code | Classify by dependency and function; see the source audit below. |
| `HARP-rep/` | `Baselines/HARP/`, preserving vendor internals in the first pass. |
| `LIFT_perf/`, `LIFT_pareto/` | Comparator settings/code to `Baselines/LIFT/`; predictions to `Experiments/results/baselines/lift/{performance,pareto}/`; weights to `Models/baselines/lift/`. |
| `Checkpoints/`, `old_GNN_checkpoint/` | `Models/gnn/`, with explicit selected/historical status and original run identifiers. |
| `mailohls_runs/` | Adapters/contracts to `Models/llm/`; family split to `Data/splits/`; automatic-clock assets to `Models/clock/`. Preserve Stage-2 initial states under `Models/llm/`. |
| `artifacts/gnn/`, `artifacts/stage2_inputs/` | Learned memory and controls to `Models/memory/`; these are trained-model-dependent assets. |
| `artifacts/llm/` | SFT corpus and its manifests/registry to `Data/processed/llm/`, with generated bulk output ignored. |
| Remaining `artifacts/` | Classify individually: curated measurements to `Experiments/results/`, raw projects/logs to `Experiments/workspaces/`, source snapshots retained with provenance. |
| `mailohls_u50_adaptation/` | Sources/raw tables/processed tables/splits to a U50 namespace in `Data/`; model states to `Models/`; result records to `Experiments/results/adaptation/u50/`. |
| `asplos_results/`, `paper_plots/`, `latency_target_search/` | `Experiments/results/`, preserving study identity; the root latency-search directory contains evidence, while the experiment driver is under `experiments/`. |
| `figures/`, `paper_figures/`, `paper_examples/`, root manuscript PDF | Exports to `docs/paper/figures/`, figure-generation programs to `Experiments/analysis/`, example inputs to `docs/paper/examples/`, manuscript to `docs/paper/manuscripts/`. |
| `Notebooks/` | `Experiments/notebooks/`, linked from the relevant component README. |
| Root measurement CSVs/JSONL | Classify by schema/provenance: immutable source measurements to `Data/raw/`; consolidated evaluation/target-response records to `Experiments/results/`. |
| `toolchain/` | `GNN_branch/toolchain/`, preserving the patch chain, lock and verification commands. |
| Root and component tests | Central `tests/{gnn,llm,data,refinement,integration}/`. |
| Empty `runs/` | Remove if still empty when implementation begins; new outputs use the declared workspace owner. |
| `.local-archive/`, `.git/`, managed dot-directories | Outside the public layout reorganization; preserve existing recovery and environment behavior. |

## The current `src` directory needs classification

It contains 48 tracked entries rather than one coherent package. A byte-content
comparison found 12 Python modules with identical copies in other maintained
owners. Examples include `data.py`, `parameter.py`, `cpp_to_gnn_emb.py`, and
`pt_to_gnn_emb.py` in GNN; `inference_qLoRA.py` in LLM; and `filter_jsonl.py` and
`filter_pt.py` in preprocessing. Update callers and retain a single implementation
for each exact duplicate. File equality alone does not prove a path is unused.

Distinct older GNN builders, loaders, models and training code belong in
`GNN_branch/legacy`; distinct older SFT/DPO/inference code belongs in
`LLM_branch/legacy`. Historical preprocessing code belongs in
`Data/preprocessing/legacy`. Memory builders, embedding exporters and reranking
tools require an individual caller and checkpoint audit before choosing active
versus legacy ownership. Diagnostic and plotting scripts follow their component
or `Experiments/analysis` owner. The root one-off
`fix_mailohls_static_domain_teacher_eval.py` belongs in
`LLM_branch/legacy/tools/` if retained for provenance; relocation must not run
it against the current trainer. The ignored ZIP archive is a separate data
preservation question: if retained, move it intact to
`Data/processed/legacy_imports/` after recording its contents and provenance.
It is not deleted by this layout proposal.

In particular, GNN historical graph builders still invoke `src/clang_script.sh`,
and legacy launchers refer to `src/train_SFT_xattn.py` and
`src/train_DPO_harp_xattn.py`. Removing the directory before migrating those
callers would break their interfaces.

## README responsibilities

Every one of the nine main directories gets `README.md`. Keep existing detailed
subcomponent READMEs where they document an independent workflow.

| README | Required content |
| --- | --- |
| Root | Purpose, pipeline diagram, nine-folder map, short setup, complete reproduction route, selected-artifact manifest, branch roles. |
| `GNN_branch/README.md` | Source-to-graph-to-model-to-memory flow; current entry points; V1 production versus V2 research status; compiler build link; required data; train/export commands; graph/feature/checkpoint contracts; tests. |
| `LLM_branch/README.md` | Stage 1/2/3 flow; prompts/domains; structural memory inputs; train/infer/evaluate commands; clock/device adaptation; selected versus pending objectives; checkpoint requirements; tests. |
| `Data/README.md` | Dataset inventory, provenance/licenses, kernel roles, action mappings, hardware/clock labels, raw/derived boundaries, split construction and leakage rules, preparation commands, regeneration cost. |
| `Refinement/README.md` | Stored-prediction inputs, core versus sequential search, domain policy, compiler prerequisites, prepare/run/resume/summarize commands, logical/physical budgets, output records, tests. |
| `Experiments/README.md` | Study index, reproduction commands/configuration, expected inputs, seeds/budgets, metrics, curated result locations, workspace policy, interpretation limits. |
| `Baselines/README.md` | HARP/LIFT provenance and licenses, local changes, setup and reproduction commands, required weights, target/objective alignment, links to comparison results. |
| `Models/README.md` | Selected-artifact inventory, how to obtain/generate weights, file hashes/contracts, memory dependencies, pending objectives, resume rules, tracked metadata versus ignored binaries. |
| `docs/README.md` | Navigation for setup, pipeline, schema/design, migration/provenance, manuscript assets and figure reproduction. |
| `tests/README.md` | Component test commands, CPU-only checks, GPU/compiler/HLS requirements, integration checks, fixture policy, operations that launch expensive jobs. |

Commands and file contracts have one authoritative owner. Other READMEs link to
that owner rather than copying full parameter lists that can diverge.

## Migration and acceptance conditions

1. Resolve the existing checkout/GitHub discrepancy before planning moves.
   The inspected checkout has 386 missing tracked files: 313 under `plots`,
   71 in `MailoHLS_ASPLOS_evaluation_workload_kit`, one paper table, and
   `import_gnosis_lud0.py`. They remain in the committed tree inspected here.
   Classify these as intentional removals or recover them before mapping them.
   Any retained `plots` code belongs in `Experiments/analysis`, report evidence
   in `Experiments/results`, and exports in `docs/paper/figures`. A retained
   workload kit belongs under `Data/kernels/evaluation` with its provenance and
   preparation tools. Preserve unrelated ongoing graph-development changes.
2. Record an old-to-new path manifest and hashes for tracked inputs, selected
   checkpoints, memory packs, frozen splits and measurements. Do not rename
   run identifiers, schema IDs, action IDs or checkpoint tensor keys for visual
   consistency. Add the nine ownership READMEs with the intended target paths.
3. Relocate low-dependency documentation/publication assets, experiment results,
   and comparator roots; fix links, vendor launchers and symlinks. Upstream
   vendor internals remain intact in the first pass.
4. Move one active component at a time, updating Python imports, shell entry
   points, default paths, tests and root discovery together. GNN currently uses
   bare imports such as `from config import FLAGS`; refinement assumes
   `Path(__file__).resolve().parents[2]`; audits import refinement directly.
   Capitalizing `experiments` and changing nesting require real code updates.
   Avoid combining file relocation with model/algorithm changes or broad
   filename renaming.
5. Migrate data and local models with manifest/path validation. Keep
   `FINAL_ARTIFACTS.json` at the root and update it with consumers. Preserve
   selected GNN/LLM weights, automatic-clock cases/calibration, Stage-2 initial
   states, split membership and memory slot/normalization contracts.
6. Historical experiment manifests contain source hashes, code identities and
   repository-relative paths. Do not edit sealed historical manifests merely
   to make paths pretty. Use a documented path resolver or preserved read-only
   replay bundle; generate new manifests for future experiments. Temporary
   old CLI aliases may support migration but should not become duplicate code.
7. Set selective ignore rules: binary weights/tensors, caches, logs and raw HLS
   workspaces remain local; licenses, model contracts, source manifests, frozen
   splits, original measurements and selected summaries remain tracked. The
   public directory view and the local 29 GiB workspace are different concerns.
8. Validate each move with affected CPU tests, shell syntax and path/import
   checks, and existing representative graph/inference artifact checks when
   dependencies are available. Confirm frozen input hashes and model outputs
   remain equivalent. Do not launch training or synthesis simply to verify
   directory relocation. Commit reviewable component moves separately.

This plan reduces the inspected 29 visible local top-level directories to nine
public ownership directories. It improves navigation; it does not by itself
delete research history, reclaim model storage, or shrink Git history.
