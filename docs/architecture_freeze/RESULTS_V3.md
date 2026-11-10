# Revision-3 completed evidence and release decision

This is the current evidence update for `research/final-architecture-v3-20261110`.
It supersedes earlier RUNNING descriptions of the identified training/inference
runs, without changing their receipts. **FREEZE=NO.** The intended DPO-default
release and useful target adaptation are not qualified. The independently
registered latency/area/ADP defaults and `FINAL_ARTIFACTS.json` are unchanged.

## Corrected Stage-2 baseline

The historical uniform-field/top1 experiment is retained as an incompatible
supervision ablation. The new run preserves V4's whole measured multi-reference
exposures and action-normalized nonforced RHS loss. It completed **150 successful
updates**, seed123, batch1×accum8, fresh Memory-C fusion at layers8/16/24/32,
post-self-attention residual, four heads of64, GNN width128 and LLM width4096.
Stage1 base/LoRA/learned tokens and V4b remain frozen and authenticated.

There are1164 training exposures,291 requests,705 distinct selected physical
complete designs,102 source/device/clock contexts,17 training families. Of291
requests,238 sample multiple distinct complete references, all within1% recorded
latency. Validation has54 rows,18 physical contexts,26 distinct selected designs,
three kernels but only **two independent families**. These are reference counts,
not a newly timing-qualified exact-resource training bank.

| Update | Action-normalized validation loss | Mean decision accuracy | Mean candidate margin |
|---:|---:|---:|---:|
| 0 | 1.221354 | .289420 | −.604804 |
| 50 | 1.220023 | .313725 | −.528060 |
| 100 | 1.215775 | .303922 | −.564617 |
| 150 | 1.215412 | .303922 | −.563948 |

The predeclared constrained-validation checkpoint key selected **50**, not150.
Only one of18 validation assignments differs at50 from0. Exact-reference-design
accuracy remains0. This is a bounded pilot showing a limited decision-score
plateau, not proof of general convergence or physical benefit. A lower
teacher-forced loss does not establish better feasible latency. The new weighted
loss scale cannot be compared directly to the historical uniform-field loss.

[Completed receipt](../../experiments/final_architecture_v3/results_v3/corrected_static_completed.json),
[training/gate/gradient plots](figures/v3/completed/training_curves.svg).
Run: `/home/elvouvali/mailohls_runs/final_architecture_v3/static_corrected_s123`.
Portable parent: `.../qualification/stage2_selected_export` under the same run root.
Registry export supplies an omitted dependency; it changes no weights/contracts.

## Mandatory Stage-3 actually trained

Starting from the exact selected Step50 policy, **SFT and DPO each completed34
successful optimizer updates**, accumulation4, identical seed and chosen schedule.
Only fusion/gates train. Reference scores are computed before updates with the
same authentic memory, independent causal prefixes, masks and precision. DPO uses
summed nonforced value log probabilities, beta.1, attention LR1e−5/gate LR2e−5.
SFT uses the corrected chosen-only action-normalized objective. Both use128
training pairs from16 families and54 validation pairs fromtwo families. Resource
labels remain proxies; suspicious spcl units are excluded. This is an ordinal
pilot, not exact-feasibility preference proof.

| Policy | Training pair accuracy | Validation pair accuracy | Validation chosen NLL |
|---|---:|---:|---:|
| Exact frozen Stage2 reference |104/128 |25/54 | cached reference |
| Matched chosen-only SFT |103/128 |25/54 |1.184501 |
| Stage3 reference DPO |103/128 |25/54 |1.185257 |

DPO mean validation reference-corrected summed margin moves+.007731, with+.0412% chosen-NLL drift, but no ranking winner changes. The matched SFT margin moves−.002168. [Health diagnostics](../../experiments/final_architecture_v3/results_v3/stage3_health.json) record all34 successful/no-skipped updates, gradients and gates. These tiny score shifts are not physical gains.

The predeclared anchored repair did not trigger: no validation ranking decline
and no >5% chosen-NLL increase. This does **not** establish convergence or DPO
superiority. Runtime recorded within arms is511.46s SFT and753.51s DPO, including
evaluation/capture; the whole loading/reference/training interval is accounted
separately. Peak allocated VRAM12,753,344,000bytes. Frozen Stage1 verification
passed for both. Optimizer/scaler/RNG and intermediate checkpoints are retained.

A concrete checkpointing defect was fixed and regression-tested: recomputation
must restore each completion's structural conditioning when chosen/rejected use
separate forwards. Variable-length C batching now uses the shared padding
contract. Selected log-softmax operates in FP32 at supervised positions only.
These fixes do not prove DPO will improve hardware decisions.

[Actual completion](../../experiments/final_architecture_v3/results_v3/stage3_completed.json).
Checkpoints: `/home/elvouvali/mailohls_runs/final_architecture_v3/qualification/preferences/{sft,dpo}/checkpoint-34`.
The full starting reference, pair/schedule/encoding hashes and frozen model
identity are cached alongside. Executed source was reconstructed exactly against
preserved Python3.10 bytecode: a post-load edit added only deployment-role contract
metadata. Historical cache identities are not rewritten to claim a clean newer
commit. A compact reconstruction receipt and ignored source snapshot preserve it.

**DPO_REQUIRED_BUT_NOT_QUALIFIED.** The normal CLI refuses these unqualified
checkpoints unless the explicit research-only override is supplied. No knowingly
regressing policy is promoted to satisfy the intended DPO-on default.

## Genuine four-model generation

Liono training finished before Ray occupied both GPUs. Ray was not interrupted.
A checksum-matched persistent worktree and three small checkpoint bundles were
transferred to idle **Davinci V100**, UUID
`GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f`. The existing pinned backbone was reused.
No private keys/backbone shards were bundled. Nine identical requests cover all
three validation sources, ZCU10410ns/5ns andU20010ns at full budgets. Pathfinder4
was added before new generation outcomes to include the third validation source;
this is still onlytwo families. Source domains, mean RHS constrained scoring,
registry, tokens and memory match across models. No final-test labels were read.

| Policy | Complete/schema-valid | Assignments different from Stage1 | Family-macro reference field accuracy |
|---|---:|---:|---:|
| Stage1 |9/9 |0 |.429902 |
| Selected corrected Stage2 |9/9 |0 |.429902 |
| Matched continued SFT |9/9 |1 |.429902 |
| Stage3 DPO |9/9 |0 |.429902 |

A [request-response analysis](../../experiments/final_architecture_v3/results_v3/request_response_diagnostics.json) confirms one device-dependent configuration change per model: Viterbi PIPE_L6 is1 on ZCU104 and0 onU200, with all previous values identical at that decision. Stage1 shared-candidate scores change by up to.11405 there. No complete clock-induced assignment changes occur in this nine-request cohort. The device comparison uses full-device fractions, not common absolute resource limits, so it does not isolate device effects from capacities.

There is **no demonstrated generated-policy improvement**. Model logits can move
without changing winners. Identical assignments do not replace exact-context
synthesis or prove quality across other requests. No nearest-measured-design QoR
is imputed. A subsequent [exact cache audit](../../experiments/final_architecture_v3/results_v3/exact_cached_physical_joins.json) matches16/36 model/request rows: per model, three Pathfinder0 requests have exact resource-feasible and HLS-estimated-timing-qualified historical receipts with **undefined worst-case latency**, and Viterbi/ZCU10410ns has an exact historical timeout. The other five requests lack an exact historical receipt. These are useful partial observations, not16 independent programs or finite-latency outcomes. Viterbi family field agreement .509804, Pathfinder .350000; these are
reference-agreement diagnostics, not feasible-latency scores or confidence bounds.
With two families, report these exact outcomes rather than pseudo-replicated CIs.

[All36 request/model rows and shared top-k score changes](../../experiments/final_architecture_v3/results_v3/decoder_comparison.json).
Raw outputs/host receipts are in ignored `artifacts/final_architecture_v3/davinci_generated/`.

## Embedding/evidence semantics

The old fixed-prefix trace is real: six contexts,33 unforced decisions, layers
8/16/24/32, identical previous directives. Its results belong to the historical
mismatched-supervision checkpoint150, not the corrected model.

A crucial control correction: permuting K/V records **within identical eligibility
routes** preserves every query's accessible multiset. With no memory positional
embedding it is a mathematical symmetry/placebo. Its invariance cannot reject
semantic evidence use. New tests verify the symmetry and a cross-route change.

For corrected Step50 on the supplementary A2 FP16 path, type/status/knownness-
matched content is now moved **between different route profiles**, preserving the
original masks/action vectors. Viterbi moves228 evidence records,213 across routes,
mean relative vector change .56854; Pathfinder0 moves131,77 across routes,
mean change .45856. Shared top-k candidate-score changes average .002220/.001848,
max .010417/.012275. Gate-off changes scores too, but **neither intervention
changes either complete assignment**. This is weak score sensitivity, not useful
compiler-fact reasoning or causal physical benefit. Natural prefixes happen to
remain identical in this comparison because all selected values remain identical.

A separate family-disjoint probe captured actual frozen V4 decision states for20
kernels on the idle A2 before Ray started. Hook on/off scoring parity passed20/20.
74 training and13 validation actions have positive compiler-proven trip counts;
17 train families/two validation families. Family-macro MAE(log2 trip count):
mean1.7378, lexical-program1.1803, GNN1.2066, LLM SA layer32 2.7582,
LLM+GNN layer32 2.5881. High-dimensional small-sample linear recoverability does
not establish absent information or a winning fusion layer. Trip count is an
explicit compiler input; the cheap lexical baseline is competitive. No learned
policy/architecture is selected from these descriptive probes.

[Joint probe](../../experiments/final_architecture_v3/results_v3/joint_trip_probe.json),
[controls](../../experiments/final_architecture_v3/results_v3/decoder_comparison.json).
C0 consumes vectors/masks, not explicit type/status embeddings; edge composites
are diagnostic, not already semantically trained facts. A legacy ledger merge can
replace a normalized relation name with a broad raw kind. Routing/proven witnesses
are preserved; the v1 pack is not silently relabeled or repacked under trained weights.

## New exact-count physical contrast bank

Twelve new Coroni-only Vitis2021.1 jobs completed: three complete contenders per
source, two clocks, two development families. Sources, headers, top, normalized
assignments, Tcl, tool/settings and raw XML/log hashes are authenticated. The first
candidate rule used legality/structural distance/hash order, not QoR ranking;
a prospectively logged four-job extension raised compiler-linked unroll/partition.

| Source/contender |10ns cycles |3.33ns cycles |10ns BRAM18K/DSP | Estimated timing at3.33ns |
|---|---:|---:|---|---:|
| Viterbi A |730163 |1495738 |36/6 |5.405ns |
| Viterbi B |730172 |1495705 |36/6 |5.405ns |
| Viterbi co-edit8 |372816 |506753 |56/12 |3.485ns |
| Pathfinder4 A |3242863 |3242895 |76/0 |2.431ns |
| Pathfinder4 B |3242859 |3242891 |76/0 |2.431ns |
| Pathfinder4 co-edit8 |3243895 |3243943 |108/0 |2.431ns |

All three identical configuration hashes are measured at both clocks for each
source. There is **no meaningful clock winner reversal** in this bank. Pathfinder
is an acceptable-set stay; Viterbi's co-edit is faster in both raw-cycle pools,
but **none of its candidates qualifies estimated3.33ns timing**. Nominal requested-
period products are not achieved execution times. No new C/RTL or post-route
verification is implied by successful HLS synthesis.

At10ns the Viterbi co-edit reduces cycles48.94%, spending more resources. The
predeclared component-midpoint rule yields BRAM18K≤46 (7.37%capacity), excluding
it while allowing the slower contender. DSP≤9 does the same. These are correlated
bank-derived **out-of-training-range** stress views, not two independent target
adaptation successes. Timing qualifies both10ns candidates. This is a real exact-
count feasibility boundary, but **not a model improvement**, a broad development
switch cohort, or a reason to train a larger target-conditioned architecture.

[Measurements, exact identities and analysis](../../experiments/final_architecture_v3/results_v3/measured_bank_analysis.json).
No new device reversal established. Latency-specific useful target adaptation
remains **NOT_IDENTIFIABLE** across independent selection families.

## Refinement and next release gate

The maintained `LLM_branch.inference.optimize` path supports DPO Stage3 outputs and
explicit user enable/disable. Disabled means exact assignment and zero extra HLS
calls. Enabled uses the existing causal controller, absolute slots, authenticated
RC3 proven pairs, complete legal local moves, fixed physical request, charged
logical queries, exact opaque cache and best verified feasible incumbent.
Unknown latency/resources/timing and failures never score. Default candidate policy
is coordinate/feedback; compiler-pair proposals remain an experimental comparison.

The actual Stage3 Pathfinder4 assignment, not an old ADP prediction, starts a
sealed coordinate-versus-structural trial with logical budgets1/4/8,300s/job,
maximum15 new physical calls and prior23 attempts. Its first queries timed out;
see the final machine receipt for completion/cancellation and exact spend. Timeout
means unknown hardware outcome, not infeasibility or zero latency. Until a finite,
resource- and timing-qualified baseline/incumbent exists, refinement gain is
**unmeasured** and the mode remains **AVAILABLE_EXPERIMENTAL**.

Mandatory remaining release work: exact-resource/timing preference labels and
independent latency switch/stay families; a non-regressing useful fusion endpoint;
DPO/default end-task qualification; finite Stage3-start refinement outcomes and
cost. Hardware messages, FiLM, deeper/JK encoders, type-aware C and alternate
layouts are **NOT_IDENTIFIABLE/NOT_RUN**, not rejected by unfair short training.
Locked test remains unopened until a defensible finalist and protocol are frozen.
Historical area/ADP claims retain their own registered lineages; they are not
latency proof. The TCAD paper must separate V1 results, RC3 mechanism pilots and
new measured bank results, and cannot claim a new target-adaptation or DPO gain.
