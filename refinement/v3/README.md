# CG-BDR v3.0 frozen implementation checkpoint

The frozen executable modules below are now published on `llvm-ir`.
The [November 18 first campaign](first_campaign/README.md) adds the subsequent
native Jacobi measurement and the current six-kernel accounting. It contains
six verified baseline reuses and one new HLS call; no V3 latency improvement
has been demonstrated. The offline replay in `replay/` predates that call.

V3 proposes small, compiler-action-grounded **best-case latency** experiments.
It makes no improvement claim before synthesis. All implementation validation and
archived replays are offline: **zero new Vitis calls**. Original controllers,
predictions, native evidence, plots, model architecture and weights are unchanged.
The reviewed base is `e0cf0f722e2e45ce78ca7fc2770f37e9ee08f489`.

## Policy and evidence

`semantic_v3.py` preserves every scheduling row with its log line, grouped
min/max/median, target/final II and count. Differing variants have no single exact
II. Warnings are grouped by type, canonical memory, explicit scheduled loop,
source-owning loop, source location and affected-loop family. Counts do not affect
rank. Complete original logs remain in each imported/synthesized project.

Source spans, exact diagnostic-line accesses, source loop labels and verified
parent chains support attribution. In graph mode, existing v2 ProGraML action
anchors and array-scope/IR access links also gate edits. They never prove alias
freedom, data dependence, physical banking or hardware causality. Source parsing
is conservative: unbraced/nonliteral loops can remain unknown. V3 imports the
unchanged v2 standard-library helpers, whose hashes are also sealed; do not copy
v3 alone to a machine without those two helpers.

`rank_v3.py` offers existing-partition banking repairs with documented dimension
and unit stride; nearest enclosing unroll reduction when the scheduled loop has
no local replication; and local unroll reduction only with a plausible throughput
benefit. A locally unrolled loop is diagnosed at that replication layer first,
rather than speculating about a second concurrency layer. One action changes per
proposal. No new pragma sites, pragma types, transformations or dependence overrides
are generated. Pipeline relocation is disabled, including in scout mode: this
implementation supplies no compiler dependence-safety proof for it.

Rank is lexicographic: supported versus exploratory evidence, available enclosing
report context, preserved versus lost lanes, resource risk, modification count,
then full directive signature. The preview exposes each component. All exact
loop criticality remains **unknown**; enclosing XML latencies give context only,
and are never summed. There are no calibrated latency predictions or fitted
kernel-specific rules. A noncalibrated `ceil(source_trip/U) * II` work proxy uses
only matching literal, unit-stride source loops, never metadata bounds as trips.
Local proportional II relief with marginal/no work benefit causes abstention;
ancestor proposals explicitly expose optimistic and proportional scenarios.

Only a measured infeasible diagnosis permits resource/timing relief, requiring
source replication plus an explicit compiler unrolling record. Feasible designs
receive no area-only priority under LATENCY. Optional `--mode scout` allows **one**
labelled replication-reduction scout on a named, measured high-impact top-level
source loop with resource headroom. It uses the same time, disk, budget and manual
approval guards. There is no unroll-doubling search. Unsuccessful local families
are suppressed in an unchanged evidence context; identical metrics and scheduling
records mark an edit `hardware-insensitive / neutral`. Diagnosis always returns
to the best measured feasible incumbent. If none exists, only a measured
infeasible baseline is available for feasibility recovery, never as a winner.

## Measurement and immutable bundles

`cg_bdr.py` implements `prepare`, `import-baseline`, `preview`, `run`, `summary`.
Preparation seals initial prediction bytes, source/header/template/info/graph
hashes, graph/source evidence, extraction provenance, top, tool version, part,
clock, capacities, exact flags, controller, semantic, ranker, runner and helper
hashes. A manifest checksum detects accidental edits. This is a local integrity
seal, not a digital signature against an adversarial writer. V0.1/v2 bundles
cannot resume as v3. Any code change requires a new bundle/checkpoint.

Only numeric `0 < Best-caseLatency <= Worst-caseLatency`, known positive estimated
clock <=10 ns, and each resource <= its full ZCU104 capacity qualify a native
Vitis candidate. Selection is exclusively by **best-case cycles** (v2 already
selected this way). Worst-case latency, II variants, estimated timing, resources,
errors and native hashes are retained. Estimated clock >7.3 ns produces a separate
schedule-margin warning; this does not change eligibility or establish routed
closure. Synthetic fixtures are labelled and cannot become native incumbents.

`import-baseline` accepts only native packaged archives with verified inventory,
XML/log/Tcl/receipt/candidate/input hashes and matching initial directives,
source, headers, graph, top, part, clock, capacities, flags and Vitis 2021.1.
It copies evidence into a **new** bundle without changing archives. Reuse charges
slot **1/4**, records `new_hls_call=false`, and is not a fresh invocation. A rejected
import leaves the baseline unsynthesized; a new baseline requires its own typed
approval. No nonbaseline measurement reuse is currently implemented: an already
charged candidate signature is rejected, and cross-experiment reuse is forbidden.
Apply exactly this import rule and charged budget to controls.

At most four total evaluations, including baseline. Each actual process has a
fresh typed `RUN <kernel> <slot> <full-signature>` approval. Bare `yes` cannot
approve a call or campaign. One nonblocking `flock` writer per bundle;
`pending.json` reserves the slot **before** execution. Failures and interruptions
consume budget. Never delete a pending reservation for a free retry; unresolved
pending work blocks new calls. Successful receipt/state commit moves the charged
reservation into history. Counters separately report total charged, actual new
Vitis processes, verified reuse, synthesis failures and incomplete reservations.
Interrupted process counts can be uncertain if the controller dies in the tiny
interval between process launch and saving its PID; the slot remains charged.
There is deliberately no automatic interrupted-run recovery.

`preview --explain`, `summary` and `report` are read-only. Previews show baseline
and incumbent best/worst, grouped constraints, all scheduling variants, top three
eligible signatures, directive diffs, evidence/proxy/risk/rank, rejection reasons
and changes from v2. The eligible count may exceed three. Summary/report carry
incumbents forward after early abstention at budgets 1,2,3,4 and retain full
trajectories. Historical v0.1 (original worst-case selection) and v2 (best-case
selection) are labelled separately, including best-case replay curves; they are
never merged into new v3 measurements.

## liono workflow

Run from the repository root with Python 3.8+ (also tested with liono Python 3.7).
Only graph-mode preparation additionally requires NetworkX. Bundled compressed
graphs are extracted into a **new** `runs-root/extracted/` directory. Existing runs
are never overwritten. No remote HLS execution occurs during prepare.

```bash
python3 refinement/v3/runner_v3.py audit
python3 refinement/v3/runner_v3.py prepare --kernels GAN jacobi-2d Kalman trmm-opt covariance syr2k --runs-root ~/cg_bdr_v3_runs --reuse-verified-baselines
python3 refinement/v3/runner_v3.py preview --mode strict --kernels GAN jacobi-2d Kalman trmm-opt covariance syr2k --runs-root ~/cg_bdr_v3_runs --explain
python3 refinement/v3/runner_v3.py pilot --mode strict --kernels jacobi-2d GAN --runs-root ~/cg_bdr_v3_runs --no-hls
# Only after inspecting previews: one typed approval per actual call.
python3 refinement/v3/runner_v3.py pilot --mode strict --kernels jacobi-2d GAN --runs-root ~/cg_bdr_v3_runs
python3 refinement/v3/runner_v3.py collect --runs-root ~/cg_bdr_v3_runs
python3 refinement/v3/runner_v3.py report --runs-root ~/cg_bdr_v3_runs
```

`pilot` executes **at most one call per requested kernel per invocation**, with a
new approval for each, never an automatic four-slot exhaustion loop. It checks
remote paths, Python 3.8 and genuine vendor headers, transfers/compares sealed
code and manifests, then previews authoritative remote state before approval.
Host: `bonus24@coroni.microlab.ntua.gr`; Python: `/usr/bin/python3.8`;
Vitis: `/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls`. Remote paths are printed and
chosen under `/home/bonus24/cg_bdr_v3_runs/<local-root-name>-<path-hash>/`.
`--no-hls` is an entirely local dry run, prints these paths, does not contact SSH,
and never reserves an evaluation. `collect` creates dated snapshots; `preview`,
dry-run `pilot` and `report` use the latest collected bundle per kernel, or prepared
local bundles if absent. Collect after remote calls to refresh these local views;
a real `pilot` always previews authoritative remote state before requesting approval.
Actual SSH transfer and Vitis execution were not exercised during implementation.
The external liono `~/cg_bdr_v2_tool/runner_v2.py` was inspected, not modified,
and is not required or assumed to exist when using this wrapper.

Exact HLS settings are `config_compile -pipeline_loops 0`,
`config_array_partition -complete_threshold 0 -throughput_driven off`,
ZCU104 `xczu7ev-ffvc1156-2-e`, 10 ns and 100% budgets; vendor include flags match
the archives. Each call has a <=7200 s timeout and >=20 GiB free-disk guard.
The XML and log identify the HLS version; no extra `vitis_hls -version` process is
launched. Source the vendor environment on Coroni as required by its installation.

Equivalent explicit lower-level commands on Coroni are:

```bash
/usr/bin/python3.8 refinement/v3/cg_bdr.py preview --bundle /path/to/new-v3-bundle --explain
/usr/bin/python3.8 refinement/v3/cg_bdr.py run --bundle /path/to/new-v3-bundle
/usr/bin/python3.8 refinement/v3/cg_bdr.py summary --bundle /path/to/new-v3-bundle
```

`run` prints the proposal and requests the full token interactively; it never
calls Vitis when approval is absent. `--approval` is solely for forwarding that
exact token from the human-gated wrapper. No approval is granted by these examples.
For custom inputs use `cg_bdr.py prepare --help`, followed optionally by
`import-baseline --bundle NEW --archive NATIVE_ARCHIVE`.

For a matched feedback-only control, prepare a separate fresh runs root with
`--policy feedback-only --reuse-verified-baselines`. It reads exact logs and
source labels/hierarchy but never graph topology or graph-derived action filters;
the compressed graph hash is retained solely for frozen-input matching. Other
contracts, ranker and budgets are shared. No feedback-only synthesis was run.

## Offline verification and decision gate

```bash
python3 -m unittest discover -s refinement/v3/tests -v
python3 -m unittest discover -s refinement/v2/tests -v
python3 -m unittest discover -s refinement/tests -v
python3 refinement/v3/audit_evidence.py
/usr/bin/python3.8 -m py_compile refinement/v3/*.py refinement/v3/tests/*.py
```

[replay/](replay/) contains reproducible dry-run commands, full previews, source
matching/counters, summaries, historical comparisons and test/audit logs. See
[replay/PROPOSED.md](replay/PROPOSED.md) for six-kernel first repairs and uncertainty.
These are imported baseline measurements and proposed interventions, not new
candidate measurements. The first experimental pass starts with Jacobi and GAN;
inspect the other four previews before considering their calls.

Freeze this v3.0 implementation before any experiments. A second code pass is
allowed only for a specific reproducible attribution/ranking defect found in the
first pass, preserving this checkpoint. No learned model or action-space expansion.
Six kernels cannot establish causality or a general optimizer. If multiple non-GAN
improvements and added value over a matched feedback-only control remain unsupported,
freeze refinement as an optional exploratory feature for DAC.
