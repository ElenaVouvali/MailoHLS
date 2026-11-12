# SALT-HLS paper GNN workflow

The subsequent [canonical data restoration](../../Data/README.md) recovered
authentic deleted directive columns for six kernels and repaired a cross-kernel
lavaMD CSV copy. `Data/manifest.json` is now the single live data registry;
historical tables below remain frozen replay inputs. Direct source inspection
finds 30 unproven action states across 17 kernels, including four labels omitted
from the current CFD step-factor metadata. No sort-radix mapping recovery was
found. Six repaired metadata inventories require new RC3 graphs; the registry
also flags the preexisting Kalman metadata mismatch. The existing paper loader
must not mix these repaired inputs with its old graph/checkpoint identities.

The [action-mapping audit](audits/action_mapping_20261110.md) checks all 55
kernel registries and CSV schemas. It identifies 26 unmapped source loops
across 14 non-test kernels, plus mapped-column gaps in lavaMD and the frozen
Kalman test table. All 26 loops are absent from the legacy loop-analysis
inventory, suggesting a structural-versus-measured-action interface mismatch.
No missing directive state has been assumed or repaired by this audit.

**Current status: three legacy-filtered-cohort runs are completed; the requested
resource-filter correction and corrected-cohort reruns are unfinished.** The
inherited preprocessing excludes utilization >=100% even for records with
measured latency and synthesis time. The raw-data audit found 3,215 training
and 118 development records with finite positive latency and >=100% resource
utilization (within the old latency ceiling). These are not automatically HLS
failures. Consequently, the published runs do not yet satisfy the requirement
to train resource heads on all successfully measured over-capacity designs.

The shared `Data/preprocessed_CSVS` cache uses `retain_measured_overflow=True`
and preserves recorded constant no-directive columns. It does not invent missing
states and is not yet wired to the frozen paper cohort. In particular,
`PaperData` still consumes the legacy preprocessed tables. Do not describe that
flag alone as a completed data fix. Preserve these checkpoints as preliminary
results; integrate the corrected raw-data policy and rerun the same frozen
methodology before calling the original paper GNN task complete.

The sequential controller trains latency, validates its exported memory with the
maintained XAttn module, then trains area and ADP from fresh initialization.
Completion is established by `completion.json` plus a passed XAttn receipt,
not by a configuration or a running process. No downstream LLM or measured
synthesis superiority is claimed.

## Fixed architecture and compatibility

`GNN_branch/model.py:PaperRC3Net` constructs the historical `Net` and reuses
its first TransformerConv, the next three pre-directive convolutions, JK MAX,
and its single post-directive TransformerConv. The four static layers have
64 channels and one attention head, using historical ELU activation. There
are four pre-directive layers. No six-layer variant, FiLM or hardware messages
are introduced. Prediction heads reuse `MLP` and `make_resource_heads`.

`data.py` authenticates the RC3 archive/member hashes against the release
manifest, and checks source and absolute action-metadata identities. Compiler
features use node categories, a training-only operation vocabulary, and
allowlisted numerical compiler facts with known-value indicators. Parallel
edges, relation types and evidence status are preserved, with explicitly typed
reverse edges. Full compiler metadata remains in authenticated graph JSON and
derived action relation records. No target overlays enter this adapter.

Candidate injection happens after the static encoder, followed by the existing
post-directive layer. A separate complete-assignment MLP ensures actions with
unresolved compiler bindings still condition predictions. Neither candidate
path participates in `encode()` or `memory()`. Device capacities and clock
enter only the prediction heads. Budgets enter ranking/feasibility only.

Historical V1 tensors/checkpoints have a different vocabulary. Current RC3
A/B/C banks load a two-layer V4b encoder; the historical multiscale exporter
writes `[JK | centered conv1]`. None is compatible with this representation
based only on a shared width. Historical weights and registrations are preserved.

## Supervision, losses and split

The source dataset uses `Preprocessing.create_jsonl.build_examples`, existing
source-aware TCAD domains and authenticated FPGA capacities. Only train and
validation kernels are parsed. The selected preprocessing manifest and split
must match. Locked-test tables may be hashed for identity but their labels
are not parsed for training/evaluation. Missing measured APL columns, illegal
complete assignments and duplicate physical designs are counted; conflicting
duplicate measurements fail the gate. Absolute `Lk -> k-1` slots never compact.

The authenticated accepted cohort has 97,442 unique designs from 143,321
train/development table rows; 45,879 rows lack a fully measured source-action
mapping. Eligible training uses 37 kernels. Development uses Viterbi and two
Pathfinder variants: three kernels in **two independent families**. Sort-radix
remains in the locked development split but is excluded for missing measured
action columns; it is not replaced or reassigned. `measurement_coverage.json`
reports raw validity filtering and replica aggregation separately.

The existing paper preprocessing convention is latency in milliseconds and
area as the arithmetic mean of BRAM/DSP/FF/LUT utilization percentages. The
split's training-only area floor handles reported zero area. ADP is their
product; within each fixed-clock pool its ordering and relative regret equal
cycles times area. No cross-clock ranking occurs. Log-QoR MAE uses log2 units.
Resource targets are physical demands reconstructed from reported measured
percentages and authenticated capacities. Counts can be fractional because
percentages are rounded; this provenance is explicit.

```
L_QoR = 0.5 * (SmoothL1(log_latency, log_area)
              + SmoothL1(selected log_objective))
L = L_QoR + 0.10 * L_resource + 0.05 * L_objective_ranking
```

Regression uses training-only standardization and Smooth-L1 beta 0.5. Resources
use log1p transforms. All accepted successful designs train resource heads,
including over-budget designs. The authenticated source tables contain complete
finite resource labels; this workflow does not restore partial-resource rows
removed by upstream preprocessing. Such exclusions must be reported, never
relabelled resource-infeasible. The helper feasibility rule has three states:
known excess proves infeasibility, unknown otherwise remains unknown. Timing
qualification is reported separately and remains unknown without timing evidence.

Pairwise softplus ranking compares unique complete configurations from one
kernel/device/clock context, measured feasible under full/half-capacity budgets,
with a 5% relative near-tie threshold. Budget-view losses are averaged inside
the physical context; regression is not expanded into budget replicas.

Each epoch gives every eligible kernel eight updates of up to 16 unique designs
(128 samples; smaller pools use all designs). Contexts rotate deterministically.
Adam uses LR 3e-5, weight decay 1e-4, gradient clipping 1, dropout 0.1, maximum
40 epochs, plateau LR reduction and early stopping patience 8. No loss-weight
sweep is run. Independent objective seeds are latency123, area124 and ADP125;
no weights are warm-started across objectives.

Selection uses only held-out development families, in predeclared order:
feasible-selection rate descending, conditional feasible Regret@1 ascending,
Kendall tau-b descending, log-QoR MAE ascending. Budget views average within
contexts, then contexts within kernels, kernels within families, then families
equally. Regret is undefined on infeasible/unknown/no selections; those failures
count in feasible-selection rate. Conditional regret must not be read alone.
Regret@1 is `selected measured objective / best measured-feasible objective - 1`;
reported values are ratios (1.0 means 100% regret).
Constant/fewer-than-two rankings yield undefined tau. The locked test is unused.

`last.pt` saves optimizer, scheduler, Torch/CUDA RNG, independent sampler RNG,
epoch, stopping state and curves. Resume requires matching protocol, source,
split, graph/schema and normalization identities. `selected.pt` is selected
using the declared development ordering.

## Versioned 128-D structural memory

Representation: `salt-paper-rc3-transformer4-jkmax-conv1-jk-v1`.
Tensor schema: `salt-paper-rc3-compiler-tensors-v1`.
Each of 64 absolute action slots has exactly128 columns:

```
[first-layer64 / train_RMS(first-layer) | JK-max64 / train_RMS(JK)] / sqrt(2)
```

RMS statistics weight training kernels equally and use active slots only.
They are frozen for export and use no candidate, target, budget or measured
QoR input. Learned encoder weights are objective-specific. Padding is exactly
zero. The memory is static and candidate-independent.

Each bank records checkpoint, graph, schema, protocol and normalization hashes,
action masks, labels and compiler relation records/masks. Export currently
covers train/development kernels. A separately qualified test stage can export
the locked-test graph from frozen weights without using measurement labels.
The new bank requires an explicit representation identity in the maintained
memory loader; historical fusion weights cannot be silently reused.

Fresh Stage-2 training uses the existing `train_SFT_xattn_new.py` with
`--memory_representation_schema salt-paper-rc3-transformer4-jkmax-conv1-jk-v1`,
`--mem_dim 128`, the matching objective and the same explicit family split.
Arbitrary `--init_structural_xattn_from` is rejected for these banks. A fusion
resume must match both representation and bank manifest; inference and DPO
inherit that identity from their saved parent contract. Stage-1 LoRA weights
can still initialize a fresh fusion run through the existing Stage-1 path.

`encoder_checks.json` verifies exact layout and static invariance to extraneous
candidate/target/QoR/budget fields. `compatibility.py` authenticates exported
packs and loads them through the maintained loader, then exercises the real
`GatedCrossAttentionBlock` under global, action-segment and compiler-relation
visibility. It checks exact zero-gate identity and finite nonzero backward
gradients with fresh fusion weights for every active kernel. These are real
XAttn module checks, not full pretrained LLM generations or adapter training.
The V100 is preferred when idle; an explicit A30 receipt identifies a fallback.

`replay_check.py` exercises saved model/Adam/RNG/sampler state with two
disposable synthetic-loss updates. Its receipt checks exact same-device replay;
it is not a cross-GPU bitwise guarantee or an additional QoR training result.

## Execution and artifacts

SSH: `evouvali@davinci.microlab.ntua.gr`.
A30: `GPU-ac0112df-7098-6c59-5c4f-a57fa666f808`.
V100: `GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f`.
Preflight rejects a training launch on a GPU with another compute process.
Existing workloads are not terminated. Source and missing packages are isolated
under `~/mailohls_runs/paper_gnn_20261110/` on Davinci; datasets are referenced
read-only from its existing repository.

```bash
export PYTHONPATH="$HOME/mailohls_runs/paper_gnn_20261110/deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=GPU-ac0112df-7098-6c59-5c4f-a57fa666f808
export PYTHONHASHSEED=123
~/MailoHLS/.venv/bin/python -u GNN_branch/train_GNN.py --paper-workflow \
  --objective PARETO_LATENCY --seed 123 \
  --output "$HOME/mailohls_runs/paper_gnn_20261110/latency"
# Add --resume only with identical source/protocol/inputs.

# Sequential execution, including area124 and ADP125 and compatibility gates:
~/MailoHLS/.venv/bin/python -u -m experiments.paper_gnn.execute \
  --root "$HOME/mailohls_runs/paper_gnn_20261110"

# Re-evaluate saved complete development pools:
python -m experiments.paper_gnn.workflow evaluate \
  --rows /path/to/latency/selected_predictions.jsonl \
  --objective PARETO_LATENCY --output /tmp/latency-metrics.json
```

Each objective directory contains `selected.pt`, `last.pt`, `curves.json`,
`selected_metrics.json`, `selected_predictions.jsonl`, `protocol.json`,
`data_identity.json`, `gpu_receipt.json`, `encoder_checks.json`, versioned
`memory/`, `completion.json`, and an XAttn receipt. Exact commands, environment
freeze and the execution journal live at the run root.

The superseded initial head attempt is preserved under
`preserved_initial_head_attempt`; it exposed the need to condition predictions
on unbound source actions. It is not a production registration or a competing
architecture qualification result.

Downstream superiority still requires matched Stage-1 versus fresh
objective-specific GNN+XAttn complete directive generations and measured feasible
HLS QoR. Standalone GNN metrics do not establish that result.
