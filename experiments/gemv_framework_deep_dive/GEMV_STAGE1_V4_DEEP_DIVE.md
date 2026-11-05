# GEMV Stage-1 V4 deep dive — completed V100S experiment

**All four requests completed on November 6, 2026, on the specified V100S.**
The V4 multi-reference model changes five official fields under the 20% budget,
and none under the clock/device contrasts. With prior assignments fixed to R1,
only A's partition dimension changes. Internal states and some candidate rankings
respond to all three contrasts, including those with invariant final directives.

The previous GPU blocker was **sandbox device isolation**, not a broken host
driver. My earlier diagnosis was incorrect. Approved execution outside the sandbox
exposed the required GPU; no driver, service, package or device assignment changed.
The blocked reports and original manifest remain under the `v4_20261105` artifact
directory. The successful experiment uses `v4_20261106`.

## 1. SALT-HLS objective and the GEMV kernel

SALT-HLS selects complete HLS directive configurations conditioned on C/C++ source,
objective, FPGA, clock and available resources. Legal syntax, compiler
admissibility, resource feasibility, timing and latency optimality are separate
questions: parallelism, memory banking and arithmetic recurrences interact.

The exact [source](inputs/gemv.c) declares A[32][32], x[32], y[32], an outer i loop
and an inner j loop accumulating `acc += A[i][j] * x[j]`. The authenticated
[metadata](inputs/kernel_info.txt) defines five actions: three array actions with
type/factor/dimension fields and two loop actions with PIPE/UNROLL fields, totaling
**13 decisions**. Figure 1 is an annotated presentation excerpt; inference uses
the original source bytes and placeholder directives.

The source contains uninitialized local input arrays and returns only y[0]. It is
preserved for compiler/graph consistency, **not treated as a functionally valid,
fully observable numerical GEMV benchmark**. No synthesis or physical QoR is
reported.

## 2. Authenticated V4 checkpoint, repository and GPU

This package is in `/tmp/MailoHLS-stage1-refinement`, on
`research/stage1-refinement` at
`3b7fd7cd39eb052b5661e09ad3562b98a5412036`, matching the requested inspected
commit. The supplied `/home/elvouvali/MailoHLS` checkout is detached at
`9a4533a8a49307285a4748e71967d5e98b23bb7c` with substantial unrelated staged
work. Neither checkout's preexisting changes were altered. The GEMV package and
completed evidence are published separately from unrelated changes.

Only `/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run`
was used. Neither historical production Stage-1 nor V4 single-reference was run.
The checkpoint was user-specified, not selected using GEMV.

| Component | SHA-256 / identity |
|---|---|
| Adapter | `28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164` |
| Training contract | `1c7a2034dba45030e51525fd415367e6fcf276cceddd89eb1132edfd66840671` |
| Base model | `deepseek-ai/deepseek-coder-6.7b-base` |
| Pinned revision | `ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912` |
| Tokenizer JSON | `eef84d1f080adcb66a40c467931055a8b95e3d37fd6534c7340aa6c63b966891` |
| Saved directive registry | `41d0f6f90db4c2370bdc02fbcdda4e114ec65cf57b89a567bfab56606d8d0c1c` |
| Source | `c6e8b0aab3919a433803bacf29c7af6b56a6a5cb4d7a511a6875a21593a7e8f6` |
| Metadata | `72dcbe726d2066c34854ff1efe04fe6f2dba9d5b4f050104c0edbc9d607c487a` |

The existing authenticated loader checked the rank-8 LoRA, Stage-1 contract,
objective PARETO_LATENCY, schema-2 response policies and saved tokenizer mapping.
The full token-to-ID mapping is in [checkpoint.json](results/checkpoint.json).
GEMV is absent from the saved registry: after hash verification, the existing
`add_missing_source_directive_domains` path derives an in-memory GEMV extension
from the authenticated source metadata. Saved registries and weights were unchanged.

Host `liono`; Python
`/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python`; PyTorch 2.6.0+cu124,
Transformers 4.51.0, PEFT 0.18.1. GPU:
**Tesla V100S-PCIE-32GB**, UUID
`GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb`. Before model loading it had 1 MiB
used of 32768 MiB, 0% utilization and no compute processes. UUID mapping exposed
exactly one CUDA device; its name and compute capability (7.0) were checked.
The repository loader used **4-bit NF4, double quantization, FP16 computation**.
The A2 was not used.

`environment.json` records the original *sandboxed preparation* and therefore
reports CUDA unavailable. It is not the execution environment receipt.
[gpu_access_resolution.json](results/gpu_access_resolution.json) and
[launch.json](results/launch.json) document successful host device access.

## 3. Four precisely controlled requests

| ID | FPGA | Clock | Budget | Available BRAM / DSP / FF / LUT | Prompt tokens |
|---|---|---:|---:|---|---:|
| R1 | ZCU104 | 10 ns | 100% | 624 / 1728 / 460800 / 230400 | 604 |
| R2 | ZCU104 | 10 ns | 20% | 125 / 346 / 92160 / 46080 | 597 |
| R3 | ZCU104 | 3.33 ns | 100% | 624 / 1728 / 460800 / 230400 | 608 |
| R4 | XCU200 | 10 ns | 100% | 4320 / 6840 / 2364480 / 1182240 | 608 |

Device identifiers are exactly `xczu7ev-ffvc1156-2-e` and
`xcu200-fsgd2104-2-e`. Capacities are taken from
`mailohls_contract.DEVICE_RESOURCES`. R2 uses `int(round(capacity * 0.2))` per
resource; the integer BRAM/DSP budgets are approximately 20.0321%/20.0231%, both
displayed as 20.0% by the canonical prompt.

All four requests share source bytes, metadata, objective PARETO_LATENCY,
specified-clock mode, one primary output, greedy field selection, `mean` candidate
scoring, no candidate-prefix truncation and checkpoint-native `legacy`
admissibility. R3 consistently changes the clock token `<CLK=3P33NS>`, supported
clock text and fixed selected-clock response. R4 changes the FPGA and its absolute
full capacities; it is not a device-token-only ablation.

The [manifest](results/manifest.json) freezes request/prompt/token hashes, source
and metadata identities, original capacities and requested resources. The
[four-row JSONL](results/requests.jsonl) SHA-256 is
`222e2b347f8929c60841bbf5a710dea347abe2ee11d15b12b1d2ed1eb5fe5a4f`.
The successful manifest adds the loader-supported `obj_mode=PARETO_LATENCY`
field to the earlier prepared requests. The earlier `objective`-only spelling was
rejected before model loading. **All four rendered prompts and token-ID sequences
remain identical to the original frozen preparation.** The earlier manifest and
failed launch are preserved rather than overwritten.

## 4. Actual generated directives

Changed values relative to R1 are bold. Array entries are type/factor/dimension;
loop entries are the literal PIPE/UNROLL values, not achieved hardware II.

| Action | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| L1 — A partition | block, 2, 1 | block, 2, **2** | block, 2, 1 | block, 2, 1 |
| L2 — x partition | cyclic, 2, 1 | cyclic, **4**, 1 | cyclic, 2, 1 | cyclic, 2, 1 |
| L3 — y partition | cyclic, 4, 1 | cyclic, **2**, 1 | cyclic, 4, 1 | cyclic, 4, 1 |
| L4 — outer PIPE/UNROLL | 0, 2 | 0, 2 | 0, 2 | 0, 2 |
| L5 — inner PIPE/UNROLL | 0, 2 | **1**, **0** | 0, 2 | 0, 2 |

All four official outputs contain exactly 13 fields and pass both source validity
and the separate current V2 tool-policy audit. No V2 candidate filtering was used
for decoding. All three target contrasts have identical active candidate sets at
corresponding sites in this experiment. The x/y dimension decisions are semantic
singletons; their official zero scores are synthetic, not measured preferences.

R1 and R4 reproduce the authenticated November 5 V4 multi-reference assignments
**and recorded aggregate scores exactly**: -2.796139289624989 and
-2.7520894668996334. New R2/R3 aggregate scores are -2.594824457541108 and
-2.823949109762907. These scores are sums of per-site mean token log-scores,
**not hardware-quality estimates**; they cannot rank requests by achieved latency.

## 5. Inference and faithful representation capture

The process-local launcher calls the unchanged
[`eval_stage1_stage2_stage3.main`](../../LLM_branch/inference/eval_stage1_stage2_stage3.py)
with `--stage stage1`. Its model, contract, tokenizer, registry and objective
validation paths remain active. It adds no alternative decoder or trainable module.

`build_prompt_sections` renders the target header, source with `<SRC_Lk>` markers
and suffix; each section is tokenized independently. The fixed selected-clock
response is appended using `selected_clock_response_token_ids`, which separately
tokenizes the fixed assignment text and numeric value. Retokenizing the full
rendered string would not preserve this contract.

The real decoder appends each `<Lk>\n` anchor and each `auto{_FIELD_Lk} = ` prefix
as separately tokenized chunks. `score_rhs_candidate_suffix` evaluates each legal
RHS **plus newline**; candidate logits start at `base_len-1`. Log-softmax and
gathering run in float32. Greedy ranking uses mean log-score, then sum log-score,
then stable candidate ordering. Token lengths differ: for example, `block\n`
uses two tokens whereas `cyclic\n` uses four. Those native scoring semantics are
preserved. Singleton domains bypass scoring with a synthetic zero.

The actual backbone is LlamaForCausalLM: 32 layers, 4096 hidden dimensions and
32 attention heads. The installed native decoder computes:

```text
u = h + SelfAttention(input_layernorm(h))
y = u + MLP(post_attention_layernorm(u))
```

At one-based layers **8,16,24,32**, read-only pre-hooks capture **u** at the input
to `post_attention_layernorm`; layer-output hooks capture **y**. The observed
PEFT module path is `base_model.model.model.layers`, using indices 7,15,23,31.
The last-layer y is before the backbone's final RMS norm and language-model head;
those native operations produce the final logits. A prefix vector is upstream of
the first RHS logit, and does not alone explain the full multi-token candidate score.

The archived Stage-2 receipt uses 64-dimensional memory and
`post_decoder_residual`; the actual local production ADP contract uses
128-dimensional memory and `post_self_attention_residual`. Both schedule layers
8/16/24/32. [Their full contracts and hashes](results/stage2_contracts.json) are
retained. Neither contract establishes the best V4 latency insertion point.
The memory interface preserves absolute Lk slots and masks; relational routing
requires an explicit action-relation mask. No structural memory or cross-attention
was inserted in these runs.

For every request, the complete no-hook decode was compared with a hook-attached
decode: assignments and rankings match, and all scores agree at
`atol=rtol=1e-5`. Exact same-input selected-candidate replay also passed extraction
off/on score checks. All runs use eval/inference mode, no dropout, no parameter
updates and no attention-matrix retention. Checkpoint file hashes are unchanged.

## 6. Hidden states and the causal prefix control

**992 actual vectors**, each shape [4096], are saved as float32
`hidden_states.npz` with tensor shape **[992,4096]**. This covers four requests ×
(five source markers + 13 natural decision sites + 13 controlled decision sites)
× four layers × two boundaries. `hidden_index.json` records request/action/field,
exact token index/ID/text, boundary, layer, native module, candidate offsets/IDs,
prefix hashes and preceding output token IDs. Vectors are copied to CPU immediately.

Source markers `<SRC_L1>`–`<SRC_L5>` have IDs 32034–32038. Their R1 positions
are 251,311,369,428,489. Semantic identities, not absolute positions, align requests.
The actual R1 decision token is ID **1412**, decoded ` = `: its positions are
631/645 for A factor/dimension, 751/764 for outer PIPE/UNROLL and 778/791 for
inner PIPE/UNROLL. All 13 fields are captured.

Source markers precede their source entity. In particular `<SRC_L5>` cannot
attend to the subsequent inner-loop header or accumulator expression. Its state
is a left-edge context representation; the later directive decision can attend to
the full source. This is a causal scope distinction, not evidence that the LLM
lacks recurrence understanding.

Natural replay uses each request's authentic earlier assignments. Controlled
replay uses **fresh R1's earlier assignments for all requests**, even when the
current controlled argmax differs. Prior output token IDs are asserted identical
at every controlled site. Controlled results are per-site counterfactuals, not a
new jointly generated configuration.

The three numerical analyses use float32 cosine, relative L2 movement
`||h_i-h_1||/(||h_1||+1e-12)`, and both norms with signed relative norm change.
There are **744 pairwise rows** and **78 field-preference comparisons**.

### 6.1 Which contrast and layer move most?

Maxima below scan all **decision fields**, all four layers and both boundaries;
they do not mix source-marker and output-decision positions.

| Contrast | Natural maximum relative L2 | Site / layer / boundary | Controlled maximum relative L2 | Site / layer / boundary |
|---|---:|---|---:|---|
| Budget R1–R2 | 0.750193 | L5 UNROLL / 24 / post-layer | 0.115095 | L1 dimension / 32 / post-layer |
| Clock R1–R3 | 0.081643 | L1 type / 16 / post-attention | 0.081643 | same |
| Device R1–R4 | 0.109706 | L1 factor / 32 / post-attention | 0.109706 | same |

The unweighted mean relative L2 over 13 decision fields × four layers × two
boundaries is 0.145640/0.044188/0.065207 for natural budget/clock/device contrasts,
versus 0.038094/0.044188/0.065207 under controlled prefixes. Thus **budget causes
the largest natural trajectory movement**, while **device has the largest mean
controlled decision movement**. The single largest controlled decision movement
is still budget at A dimension. Neither aggregation is a hardware-quality metric.

### 6.2 L5 shows why natural and controlled prefixes cannot be conflated

At L5 UNROLL, layer 24, comparing R2 with R1:

| Prefix | Boundary | Cosine | Relative L2 | R1 norm | R2 norm | Relative norm change |
|---|---|---:|---:|---:|---:|---:|
| Natural | post-attention | 0.722957 | 0.733389 | 231.6063 | 224.4214 | -3.1022% |
| Controlled | post-attention | 0.998809 | 0.048959 | 231.6063 | 232.2303 | +0.2694% |
| Natural | post-layer | 0.708378 | 0.750193 | 245.6977 | 236.4888 | -3.7481% |
| Controlled | post-layer | 0.998788 | 0.049462 | 245.6977 | 246.5976 | +0.3663% |

Freezing the prior assignments changes the observed movement from **75.02% to
4.95%** at the same post-layer site. The natural R2 prefix includes changed array
decisions and PIPE=1; controlled replay restores every preceding R1 value,
including PIPE=0. This establishes a strong dependence on the generated prefix;
it does not isolate the contribution of each earlier field or an additive causal
decomposition.

L1's controlled maximum is 0.115095 at dimension/layer32/post-layer (cosine
0.995392, norms 608.2011→566.6812). L4's controlled maximum is device-sensitive
PIPE/layer32/post-layer: relative L2 0.095545, cosine 0.996253, norms
680.4406→650.2712, despite invariant PIPE=0. L5's controlled maximum is device
UNROLL/layer24/post-layer: 0.086386, cosine 0.996262, despite invariant UNROLL=2.
The inspected actions therefore respond differently.

### 6.3 Source states also respond, with different maxima

Across the five source markers, the maxima are 0.271554 for budget (L2/layer24/
post-layer, cosine 0.963604), 0.413607 for clock (L1/layer24/post-layer, cosine
0.918365), and **0.504948 for device** (L2/layer24/post-layer, cosine 0.879669;
norms 227.3127→238.5289). For the focused L1/L4/L5 source markers, device maxima
are respectively 0.403571 at layer24/post-layer, 0.311032 at layer8/post-layer,
and 0.217891 at layer16/post-attention. A single global “most sensitive layer”
would hide this site/boundary dependence.

The clock/device outputs and prefixes are identical to R1, so natural and
controlled decision metrics are identical for those contrasts. Their nonzero
movements establish observed target-context sensitivity without changed argmax.
Different prompt lengths also shift positions; the prefix control does not isolate
pure hardware meaning from all tokenization/position effects.

### 6.4 A general final-state view, without an intermediate-layer schedule

For a general account of what the model represents immediately before choosing
directive values, the natural readout is the **final normalized hidden state at
each field's last prefix token**. The model has 32 decoder layers, so this readout
follows layer 32 by architecture, not because of the fusion-layer schedule.

There is no single hidden vector for a complete configuration. Each request has
13 contextual field positions, giving a selected tensor of **[4,13,4096]**.
The same last token (` = `, ID 1412) has different contextual states because the
source, target, field identity and prior selected values contribute to it. Two
singleton fields per request are recorded through replay, since their production
choices bypass model scoring.

The exact language-model-head input is `FinalRMSNorm(final_decoder_residual)`.
Our saved layer-32 post-decoder vectors are immediately **before** that final
normalization; normalized pre-logit vectors were not separately captured. The
following observed relative L2 movements therefore describe the saved residual,
not an estimated normalized state:

| Final decision state | R1→R2 budget | R1→R3 clock | R1→R4 device |
|---|---:|---:|---:|
| A partition factor | 3.44% | 3.92% | 9.60% |
| A partition dimension | 11.51% | 4.22% | 6.26% |
| Outer PIPE | 5.47% | 5.84% | 9.55% |
| Outer UNROLL | 7.12% | 5.88% | 6.36% |
| Inner PIPE | 7.02% | 2.95% | 6.43% |
| Inner UNROLL | **54.95%** | 3.21% | 6.98% |

All entries use authentic natural prefixes. At this final residual, R2's inner
UNROLL state has cosine **0.878737** to R1. With prior assignments fixed to R1,
its movement falls to **3.93%** and cosine rises to **0.999449**. Thus the same
prefix-dependence conclusion holds at the final decoder output, not only at
the intermediate layers discussed above.

[Figure 8](../../artifacts/gemv_framework_deep_dive/v4_20261106/figures/fig8_final_decision_states.svg)
shows all 13 fields; [the CSV](../../artifacts/gemv_framework_deep_dive/v4_20261106/final_decision_states.csv)
includes cosine, relative L2 and both norms for all 39 contrasts. These are dense
contextual vectors, not readable statements about recurrence understanding, and
one prefix state explains only the first RHS token's prediction context. The
full candidate score also includes subsequent RHS tokens and the newline.

## 7. Candidate preference changes and compiler interpretation

The active candidate domains are identical in **all 78 comparisons**; no observed
choice flip is forced by a changed domain.

| Contrast | Natural choices / full rankings changed | Controlled choices / full rankings changed |
|---|---|---|
| Budget | 5/13 / 7/13 | 1/13 / 3/13 |
| Clock | 0/13 / 3/13 | 0/13 / 3/13 |
| Device | 0/13 / 3/13 | 0/13 / 3/13 |

A dimension is the one controlled argmax change. Its candidates are {1,2};
mean scores for (1,2) are (-0.318177,-0.405307) in R1 and
(-0.370338,-0.347625) in R2, reversing the preference. R1's margin is 0.087129;
R2's margin for dimension 2 is 0.022713. A factor stays 2 in every request;
its lead over factor 4 is 0.330883/0.331017/0.307371/0.300647 for R1–R4.

Outer PIPE remains 0 and UNROLL remains 2. Controlled margins are respectively
2.500866/2.516826/2.595290/2.444379 and
1.814638/1.853469/1.869313/1.791190 for R1–R4. This is robust model preference
within these domains, not proof that the chosen pragmas match optimal hardware.

The inner PIPE preference is less separated: R1 chooses 0 over 1 by 0.070868.
Natural R2 chooses 1 by 0.116677; controlled R2 retains 0 by only **0.000515**
(scores -0.346811 versus -0.347325). That narrow margin should not be called
calibrated confidence or assumed stable under a different numerical runtime.

For L5 UNROLL, R1 prefers 2 over 4 by 0.238679. Natural R2 prefers 0 over 16
by 3.939606; its score for 2 is -4.438067. Controlled R2 instead prefers 2
(-0.365011) over 4 (-0.581930), margin 0.216918. Since all six candidates
{0,2,4,8,16,32} remain legal, this is a model preference shift with the prefix,
not a decoder-domain restriction. No individual earlier directive is isolated by
the all-prior-fields control.

### Compiler-derived information: explicit facts versus untested benefits

The read-only GNN branch at `172e05a69650a666c323955ea78cb501fd9364c2` contains
the authenticated GEMV V2.2 RC3 canonical graph: **66 nodes, 106 edges, five
matched actions**; graph identity
`42fe84cb8b8c2f5665dc016fcfb6a969f93b848a3407c7343100b6b4bcabae21`.
Its source and metadata hashes match this experiment.

| Action | C information available to the LLM | Explicit compiler facts | Observed Stage-1 behavior | Hypothesis requiring future intervention |
|---|---|---|---|---|
| L1 A | Two dimensions, indexing A[i][j], nested loops | Bound memory root, matching [32,32] source/compiler shape, affine access and loop relations | Budget changes dimension under both prefixes; factor remains 2 despite score/state changes | Explicit access/action relations may alter partition preference; no evidence yet of better banking/QoR |
| L4 outer | 32 iterations, child loop and workload | Affine-proven trip count 32, depth 0, child L5, scoped arithmetic/memory operations | PIPE0/UNROLL2 invariant, with nonzero controlled movements and changed scores | Test whether explicit nesting changes useful decisions; invariance alone is not missing structural understanding |
| L5 reduction | 32 iterations, float multiply/add, acc+= recurrence | Depth 1; f32 mul/add; verified scalar SSA forwarding through iter_args and yield | Natural budget response is strongly prefix-dependent; controlled PIPE0 has a narrow margin and UNROLL2 remains preferred | Recurrence/alignment information is a concrete candidate to test, not a demonstrated missing concept |

MLIR exposes `iter_args(%arg2 = %cst)`, `arith.addf %arg2, %4`, and
`affine.yield %5`. Three graph `loop_carried` relations are marked proven with
`verified_MLIR_region_SSA` provenance: initialization, backedge and loop result.
These are distinct from separate memory-dependence analysis. Absence of a proof
elsewhere is not proof of independence. Source y[32] is lowered here to scalar
`memref<f32>` with a guarded store, reflecting the limited observable output.

The standalone V4b representation is not production V1 memory. Absolute action
alignment and dimension compatibility alone do not make its weights interchangeable
with a production Stage-2 checkpoint. No GNN or fusion training was run.

## 8. What is established, and what remains unknown

The V4 LLM **does distinguish these requests internally** at the measured
positions/boundaries, and its candidate scores respond to all contrasts. Budget
changes five official fields; clock/device changes leave all choices invariant
while three full rankings change. Thus invariant output is not evidence that the
target was ignored.

The supported limitation of an output-only or uncontrolled hidden-state account
is concrete: later budget-response behavior mixes target context with generated
prefix changes. The controlled run removes four of five argmax changes and greatly
reduces the L5 movement. It does not prove that any choice is physically wrong,
that the LLM lacks recurrence knowledge, or that explicit graph memory helps.

One kernel, three contrasts, one checkpoint/runtime and four inspected layers
cannot establish generalization or optimal placement. Prefix length/tokenization,
single-run numerical effects, singleton decisions and complete-candidate token
scoring limit interpretation. There is no functional numerical validation,
synthesis, achieved latency, feasibility, timing closure, target-optimality,
production-model comparison or structural-fusion benefit measurement.

## 9. Artifacts, figures and validation

All generated results are under
`artifacts/gemv_framework_deep_dive/v4_20261106/` in this worktree:

- `official.jsonl`, R1–R4 official JSON, exact prompts and token IDs.
- `candidate_scores.json`: 104 site records, full candidate scores/domains,
  selections/margins for 52 natural and 52 controlled decisions.
- `hidden_states.npz` **[992,4096] float32**, with `hidden_index.json`.
- `pairwise.csv` (744 rows), `candidate_comparisons.csv` (78),
  `final_decision_states.csv` (39), `sensitivity_summary.json`,
  `numerical_findings.json`, and `findings.md`.
- Validation, checkpoint, code, GPU access, launch, graph and Stage-2 receipts.

The [small tracked receipts](results/) retain inputs, checkpoint identity,
validation and numerical summaries. [Presentation notes](PRESENTATION_NOTES.md)
provide seven slides grounded in these actual observations.

All required PNG/SVG figures are generated from recorded data: source/actions,
four requests, a 13-field directive matrix, L1/L4/L5 source/natural/controlled
cosine heatmaps, relative movement, candidate scores, and the architecture.
Cosine panels explicitly label their ranges. Movement panels share a scale across
natural/controlled rows for each action. The architecture marks structural fusion
as future work. No missing-data cards are used for the completed run.

All GPU equivalence checks passed, all controlled preceding tokens matched, all
four official configurations (52 fields) are complete/source-valid/V2-tool-admissible, and R1/R4
historical assignments and aggregate scores reproduced exactly. Focused CPU tests
also validate native hook algebra, loader request parsing, prefix alignment and
metric calculations; see `results/cpu_tests.json` for the final test receipt.

Figure-only reproduction does **not** load the checkpoint:

```bash
cd /tmp/MailoHLS-stage1-refinement
MPLCONFIGDIR=/tmp/gemv-matplotlib PYTHONPATH=. \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  -m experiments.gemv_framework_deep_dive.analyze
```

Use a new output directory for another inference run; existing evidence is
protected against overwrite. [README.md](README.md) gives exact commands. GPU
commands require approved access outside the restricted sandbox; a sandboxed
`nvidia-smi` failure must not be diagnosed as host driver failure without that check.

## 10. Single recommended next experiment

**Run a single-field prefix-intervention sweep for R2's L5 decisions.** Swap each
earlier changed array field and the preceding L5 PIPE field individually between
the natural and R1-reference prefixes; retain the same target, weights, candidate
domains and scorer. Measure L5 PIPE/UNROLL rankings and the selected layer24
residual states. This directly tests which feedback terms account for the five-to-one
argmax reduction and 0.750193-to-0.049462 movement reduction.

The all-prior-fields control cannot currently attribute these effects to A's
dimension, vector factors, PIPE, or their interactions. Resolving that attribution
is more informative now than assuming a structural-memory deficit. It would give
the subsequent GNN action-memory trace and controlled fusion experiment a concrete
behavior to explain; the verified recurrence/access relations remain candidates,
not demonstrated missing knowledge.
