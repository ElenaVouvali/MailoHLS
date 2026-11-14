# Coroni synthesis guide for ChatGPT

Verified against `bonus24@coroni.microlab.ntua.gr` on **2026-11-13**. This is a
specification for generating a reproducible batch runner for nine kernel families.
The inspection did **not** launch synthesis, simulation or change existing server
projects. Only a version-query scratch directory was created on Coroni.

The accompanying private bundle is
`artifacts/coroni_synthesis_guide_20261113/`. It contains a hash-verified snapshot
of all nine application folders, an exact source inventory, historical report
fixtures, an example request, a Python 3.6 report parser, and a short ChatGPT prompt.
The source snapshot is inspection evidence; its existing pragmas are **not** a
prediction configuration to inherit. Its GAN software stub must not be staged for
native-AP HLS compilation. Do not publish project data or redistribute the
installed Xilinx headers with a generated runner.

## Access: what a user without a server account can actually do

ChatGPT can generate and validate scripts offline from this bundle. Running them
on Coroni still requires an **authorized execution path**: an SSH-enabled agent
using an authorized account, or an authorized Coroni operator who receives the
bundle, launches it and returns logs/results. A chat session without an SSH tool
cannot execute or live-monitor Coroni merely because it knows these paths.

For an offline user, have ChatGPT create a self-contained `coroni_batch.zip` with
the request JSONL, source/version manifest, materialized cases, runner and README.
The operator uploads and executes it. Return `summary.csv`, `results.jsonl`, case
receipts, synthesis logs, top XML/RPT reports and any explicitly requested
co-simulation reports. ChatGPT then parses these artifacts and presents the table.
Do not ask for private keys or embed credentials in scripts. This guide supplies
no account rights; use the existing authorized SSH connection or an operator.

## Verified toolchain and correct kernel registry

| Setting | Verified value |
|---|---|
| Authorized account used for inspection | `bonus24@coroni.microlab.ntua.gr` |
| Home / application root | `/home/bonus24` / `/home/bonus24/Application` |
| HLS executable | `/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls` |
| Environment | `/opt/Xilinx/Vitis_HLS/2021.1/settings64.sh` |
| Version | Vitis HLS 2021.1, SW Build 3247384 |
| Native AP include directory | `/opt/Xilinx/Vitis_HLS/2021.1/include` |
| Server default Python | **3.6.9**; use the standard library or an explicitly provided newer environment |
| Execution | CPU synthesis; no GPU or physical FPGA board is required for C synthesis |

Every source below is relative to `/home/bonus24/Application`. Compile **only the
selected design translation unit**, with its required headers. Do not add both
`kernel.cpp` and `kernel_placeholders.cpp`, or both a design and its testbench as
design files. Source language and function linkage must be preserved.

| User name / accepted alias | Folder | Selected source | `set_top` | Source action map |
|---|---|---|---|---|
| GAN | `GAN` | `network.cpp` | `forward_propagation` | 11: arrays L1–L3, loops L4–L11 |
| Kalman / serrano-kalman-filter | `serrano-kalman-filter` | `kernel_placeholders.cpp` | `krnl_KALMAN` | 13: arrays L1,L2,L5–L8; loops L3,L4,L9–L13 |
| trmm-opt | `trmm-opt` | `kernel.cpp` | `kernel_trmm` | 3 loops L1–L3 |
| aes / chstone-aes | `chstone-aes` | `kernel_placeholders.c` | `chstone_aes` | 6 loops L1–L6 |
| jacobi-2d | `jacobi-2d` | `kernel_placeholders.cpp` | `kernel_jacobi_2d` | 5 loops L1–L5 |
| syr2k | `syr2k` | `kernel_placeholders.cpp` | `kernel_syr2k` | 4 loops L1–L4 |
| covariance | `covariance` | `kernel_placeholders.cpp` | `kernel_covariance` | 7 loops L1–L7 |
| jpeg / chstone-jpeg | `chstone-jpeg` | `kernel_placeholders.c` | `chstone_jpeg` | 21: arrays L5–L7,L15,L16; remaining IDs are loops |
| 3d rendering / rosetta-3d-rendering | `rosetta-3d-rendering` | `kernel_placeholders.cpp` | `rendering` | **15**: arrays L1,L10–L14; loops L2–L9,L15 |

The inventory JSON records each source/header SHA-256. Verify it before preparing
cases; if a file changes, stop or explicitly qualify a new source version. Do not
match by filename alone. Do not alter array sizes, numeric types, loop bounds,
arithmetic, interfaces, top signatures or data fixtures to make synthesis easier.

**Actual server traps to handle:**

* `/home/bonus24/script_syr2k.tcl` currently selects `krnl_KALMAN` and a Kalman source.
  `project_syr2k` even contains Kalman reports. Never infer a kernel from a project name.
* `/home/bonus24/script_trmm.tcl` currently selects `gemv` and a GEMV source.
  Generate per-case Tcl from the verified registry above.
* Many `kernel_placeholders.*` files currently contain **concrete pragmas**, not
  placeholders. trmm has commented pragmas; AES still has `auto{...}` placeholders.
  The materializer must handle these forms and verify the result, rather than
  perform string substitution and assume predictions were applied.
* Rendering's `kernel_info.txt` and `ACTION_AUDIT.json` describe an older **12-action**
  layout. The selected source and `SALT_experiments.jsonl` use **15 actions**.
  Its alternate `kernel.cpp` also has duplicate/wrong comment IDs (e.g. L2 and L15).
  Use the selected 15-action source, verified declarations and `action_map.json`;
  reject a 12-action prediction or resolve it as a different version. Do not relabel
  predictions or trust the stale audit's stored hashes.
* Rendering has an untargeted `RAST2` pipeline pragma. Declare and preserve fixed
  pragmas separately from the 15 prediction sites. Record cleanup of non-operational
  zero-factor syntax; do not silently optimize additional sites.
* GAN needs `network.h`, `weight_definitions.h` and `tanh.h`. Its application folder
  also contains a **software `ap_fixed.h` stub**. Do **not** stage that stub in an
  HLS case. Preserve the project headers, use Vitis's real `ap_fixed.h` / `ap_int.h`,
  and record their hashes and include path. A quoted include searches its local
  directory first: adding `-I` alone does not fix a staged shadowing stub.
  The inspected November V4 GAN runner already excludes AP stubs and pins vendor
  headers; some older September GAN runs copied the stub. Native-AP results must
  not be presented as equivalent to those old measurements without proof.
* Preserve Kalman's fixed-point, 256-bit interface types and AXI pragmas. Its source
  uses Vitis AP headers; do not replace them with software stand-ins.

The new native-AP compilation profile should be explicitly named, e.g.
`coroni-vitis2021.1-native-ap-v1`. It is a declared effective compilation input,
not a silent repair to a historical dataset. Existing server sources are read-only
inputs; stage and modify copies under each new case directory.

## Require explicit device and frequency for every request

The inspected repository file `paper_plots/SALT_experiments.jsonl` has **40 rows**
across these nine families. Its keys are `kernel`, `stage`, `objective`, `latency`,
`resources`, `directives`. **It has no device or clock/frequency fields.** Therefore
it is not a complete synthesis queue until the user supplies targets, either in
each row or in an explicit sidecar/CLI default recorded in every case receipt.
Its stored latency/resources are historical annotations, never new measurements.

The separate `Target_general_experiments.jsonl` illustrates `device` and
`clock_period_ns`, but is a different 12-row experiment. Do not replace the requested
queue with it or silently borrow its contexts.

These exact FPGA parts are present in Coroni's installed-part inventory:

| Alias | Exact part |
|---|---|
| ZCU104 | `xczu7ev-ffvc1156-2-e` |
| U200 | `xcu200-fsgd2104-2-e` |
| U50 | `xcu50-fsvh2104-2-e` |

Accept another exact part only after verifying it in the selected tool installation.
Never substitute speed grade/package. Recheck the recorded inventory at execution
time if the tool installation changes. Part support is independent of a physical
board being plugged into the server.

Use nanoseconds for Tcl: **clock_period_ns = 1000 / frequency_mhz**.
10 ns = 100 MHz; 5 ns = 200 MHz; 3.33 ns is approximately **300.3003 MHz**, not
exactly 300 MHz. For exactly 300 MHz use the corresponding 3.333333… ns period.
If period and frequency are both supplied, validate their agreement; do not silently
round/change one. Preserve full requested precision in Tcl and receipts. XML prints
periods to two decimals, so report identity checking may allow only that formatting
rounding tolerance (0.005 ns), separately from request validation.

Preferred input shape (one JSON object per line; this is a complete syr2k example):

```json
{"case_id":"syr2k_s1_zcu104_100","kernel":"syr2k","stage":"stage-1","objective":"LATENCY","device":"ZCU104","clock_period_ns":10.0,"frequency_mhz":100.0,"source_profile":"coroni-vitis2021.1-native-ap-v1","directives":[{"action_id":"L1","kind":"loop","pipeline_ii":0,"unroll_factor":4},{"action_id":"L2","kind":"loop","pipeline_ii":0,"unroll_factor":8},{"action_id":"L3","kind":"loop","pipeline_ii":0,"unroll_factor":4},{"action_id":"L4","kind":"loop","pipeline_ii":0,"unroll_factor":8}]}
```

For arrays require `action_id`, `kind:"array"`, verified `variable`,
`partition_type`, `partition_factor`, `partition_dim`. A raw MailoHLS `prediction`
or `target` completion can be supported as an additional input format by parsing
every `auto{_PIPE_Lk}`, `_UNROLL_Lk`, `_ARRAY_T_Lk`, `_ARRAY_F_Lk`, `_ARRAY_D_Lk`
assignment strictly. Reject malformed, duplicate, missing, extra or conflicting
assignments. If structured and text predictions coexist they must agree.

Preserve every requested row and target. Do not select only LATENCY/stage-3 rows
because an old search script did so. Give each input line an ID; a fingerprint must
include source/header identities, native-AP profile, directives, exact part/period,
tool/settings and flags. Kernel/stage alone is not a configuration identifier.

## Materialize the requested configuration exactly

Generate a per-source action map recording ID, loop/array kind, owning function,
source anchor/declaration, variable/dimensions where applicable, and source hash.
`action_map.json` in this bundle provides verified lexical anchors, types and array
names for the selected nine sources; owning scopes must be checked when using Tcl
directives. Rendering's version distinction is explicit. Metadata trip-count
annotations alone are not proof of exact bounds or valid compiler scope.

Require a complete assignment for every selected action. Missing does **not** mean
disabled. A row with `directives:null` is valid only as an explicitly requested
baseline with a separately declared baseline policy and source/target context.

For this JSON convention:

* `pipeline_ii:0`: **omit** that site's explicit pipeline pragma. It does not mean
  `II=0`, and it does not mean `PIPELINE off`. Emit `#pragma HLS pipeline II=N`
  only for a positive requested integer N.
* `unroll_factor:0`: omit the unroll pragma. Positive integers emit
  `#pragma HLS unroll factor=N`. Preserve factor 1 if explicitly requested.
* `partition_type:"none"`: omit the whole partition pragma. For this profile require
  the disabled tuple `none,0,0`, rather than guessing what inconsistent tuples mean.
* `block`/`cyclic`: emit, for example,
  `#pragma HLS array_partition variable=HuffBuff type=cyclic factor=2 dim=1`.
  Require a positive factor and a valid dimension. Dimension 0 denotes all dimensions.
* `complete`: emit, for example,
  `#pragma HLS array_partition variable=max_min type=complete dim=1`.
  Do not emit `factor=0`; a factor is irrelevant for complete partitioning. Preserve
  the original requested tuple in the receipt, including any conventional factor 0.

These partition/dimension rules follow the pinned
[Vitis HLS 2021.1 directive reference](https://docs.amd.com/r/2021.1-English/ug1399-vitis-hls/set_directive_array_partition).
Do not force factors to divide loop bounds, clamp a prediction to a smaller factor,
replace an illegal request with a fallback, or change a pipeline II to obtain a
report. Reject invalid requests or record tool rejection without changing them.

Remove/replace existing **controlled** pragmas at each bound action before emitting
the new ones. Insert loop pragmas in the intended loop body and array pragmas after
the correct declaration. Match comments `/*Lk:*/` and real labels `Lk:`; nested loops
and braces require a scope-aware scanner or explicit audited templates. A regex
covering "Lk through the next label" can remove a parent's pragma or bind the wrong
array. Comment IDs are not real HLS loop labels: `set_directive_pipeline top/Lk`
does not target a `/*Lk:*/` comment. Source pragma materialization avoids this trap.

Before HLS, verify an emitted-directive ledger against the request, no `auto{...}`
tokens, no active `II=0`, `factor=0` or `type=none` pragmas, unique action binding,
unchanged computation/interfaces/constants, and expected source/header hashes.
Save the original source, materialized source, diff and ledger. Preserve unrelated
fixed/interface pragmas and record their policy. Disabled user directives do not
guarantee the tool will avoid all internally induced transformations; collect
ignored/unmet directive warnings and report requested versus realized behavior.

## Per-case HLS recipe and logging contract

Use a fresh root such as `/home/bonus24/salt_jsonl_synth/RUN_ID`, never a shared
`project_*` or `/home/bonus24/Application` directory. Acquire a batch lock. Default
to one case at a time; parallelism is an explicit operator choice. Put project,
temporary files and logs under the individual case so concurrent cases cannot mix.
Use safe case IDs and quote/validate paths; never evaluate JSON text as shell/Tcl.

Generic **synthesis-only** Tcl (replace the illustrative top/source/part/period):

```tcl
# Run with the process working directory set to this case directory.
open_project -reset hls_project
set_top kernel_syr2k
add_files -cflags {-I/opt/Xilinx/Vitis_HLS/2021.1/include} {source/kernel_placeholders.cpp}
open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10.0 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
csynth_design
exit
```

The Vivado IP flow and these automatic-optimization settings match the inspected
research launchers; changing them changes the experiment profile. Record any
explicit clock uncertainty or compiler defines; do not invent additional flags.
Start from `source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh`, then use the absolute
2021.1 executable, and record `-version`, executable hash, settings and AP hashes.
Run the version query in the new batch directory because the tool writes logs.

The runner must print, with immediate flushing, before **every** launch:

```text
[3/40] START case=... kernel=... config_sha256=...
top=... source_sha256=... part=... clock_ns=... frequency_mhz=...
requested_directives=<complete normalized configuration>
emitted_directives=<ledger or source diff path>
log=<absolute case/synthesis.log>
```

Stream merged stdout/stderr both to `synthesis.log` and to the user's console,
retaining full raw logs even when the final summary is compact. Save `process.json`
with PID/process-group ID and timestamps. Use `Popen(..., start_new_session=True)`
and a real wall-clock timeout covering silent periods; on timeout stop **that case's**
process group and reap it. Never use a broad `pkill vitis_hls` or kill other users'
jobs. A default six-hour limit may be configured and recorded; it is not a runtime
estimate. Catch failures per case, save them, then continue the queue.

Python must work on 3.6.9: no `dataclasses`, `text=True`, `capture_output=True`,
`Path.is_relative_to`, `copytree(dirs_exist_ok=True)` or built-in generic annotations
unless the operator explicitly selects a newer interpreter. Use standard-library
JSON/XML/CSV, `universal_newlines=True` and a manual streaming/copy loop.

Store `request.json`, `receipt.json`, source/header/AP hashes, normalized directives,
settings, Tcl, start/end times, return code, timeout state, tool warnings, report
hashes and `result.json` for every case. Flush append-only results/events as each
case finishes. Resume may skip only an identical, validated completed case with
matching fingerprint and report hash. Preserve incomplete/failed attempts and use a
new attempt directory for retry; never delete a case or reuse a stale XML blindly.

The HLS process returning zero is not sufficient alone; inspect errors and the
expected fresh, parseable top-level report. Conversely, a report appearing before
the process exits is not permission to kill the process and announce success.
Nonzero/timeout with a report must remain a distinct `failed_with_report` /
`timeout_with_report` status, with metrics explicitly provisional. Save complete
logs; a request that the tool ignores is not proof that the requested optimization
was realized.

## Extract exactly latency and normalized utilization

Read this case's exact report, not the first XML returned by a recursive search:

```text
CASE/hls_project/solution1/syn/report/TOP_csynth.xml
CASE/hls_project/solution1/syn/report/TOP_csynth.rpt
```

Validate `ReportVersion/Version=2021.1`, `UserAssignments/TopModelName`, `Part`,
`FlowTarget=vivado`, `unit=ns`, and requested `TargetClockPeriod`. Numeric-report
clock rounding tolerance is separate from exact request/Tcl identity.

| Requested metric | XML path under `profile` |
|---|---|
| Primary latency cycles | `PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency` |
| Best/average latency, retained for provenance | same parent, `Best-caseLatency` / `Average-caseLatency` |
| Resource count X | `AreaEstimates/Resources/X` |
| Device capacity X | `AreaEstimates/AvailableResources/X` |

For X = `BRAM_18K`, `DSP`, `FF`, `LUT`, compute **100 × used / available** from that
same report. DSP aliases such as `DSP48E` can be supported explicitly. Zero usage
is a real 0%; absent fields or a zero/missing denominator are unknown, not 0%.
Do not use board marketing capacities, resource-budget limits, training-set
statistics or rounded integer percentages from text tables as denominators.
Do not clamp percentages above 100%; that design exceeds the part's capacity.
BRAM is in the report's 18K-block units, not an assumed BRAM36 count.

`parse_csynth.py` in the bundle is a tested Python 3.6-compatible reference parser.
It accepts the expected top/part/period, returns raw latency fields and counts plus
percentages, and never substitutes a loop latency for top-level latency. It does
not certify process success; the runner must combine it with execution evidence.

**Undefined latency must be handled explicitly.** The inspected AES, JPEG and
rendering top-level csynth XMLs all report `undef` for best/average/worst latency.
This reflects data-dependent control in those source versions; it is not a missing
integer to fill from another report. For synthesis-only mode, print `N/A (HLS
latency undefined)` while still reporting available utilization. Do not copy the
old `SALT_experiments.jsonl` latency or sum subfunction/loop latencies.

If the user requires a numeric cycle count for such cases, offer a **separate
explicit C/RTL co-simulation mode**, with a validated, hashed testbench/data set:

* JPEG: existing `tb_jpeg.c`; it calls `chstone_jpeg` and checks its return value.
* Rendering: existing `tb_rendering.cpp`, `input_data.h`, `outputs_golden.txt` for
  the 3,192-triangle bear input. The inspected historical Tcl uses
  `-D__gmp_const=const` on both design and testbench for an old GMP/MPFR compatibility
  issue. Retain and record that flag in this profile if co-simulating it.
* AES: there is no separate testbench in the inspected application folder. A new
  `tb_aes.c` can call `int chstone_aes(void)` from `main` and return its status;
  label and hash this as a new wrapper around the source's embedded test vectors.
  Validate C simulation before attempting RTL co-simulation.

Add testbenches with `add_files -tb`, and rendering's golden data also as `-tb`.
Run `csim_design`, `csynth_design`, then
`cosim_design -rtl verilog -tool xsim -trace_level none`. Keep all logs/statuses
separate. Parse `TOP_cosim.rpt` only for a **Pass** RTL row. Its latency min/avg/max
describe tested transactions; use the max with basis `cosim_max_for_testbench` and
the testbench/data hash. They are not a static worst-case guarantee over all inputs.
Do not use "Total Execution Time" as function latency or mix interval with latency.

Existing historical JPEG/rendering co-simulation reports were inspected to verify
their format; their cycle values are **not** predictions for the next queue.
If co-simulation fails or times out, retain valid HLS resources but leave latency
unknown. A numeric count for every arbitrary configuration cannot be promised.

Distinguish requested frequency from `EstimatedClockPeriod`. These are HLS estimates,
not post-route timing closure or board measurements. Preserve estimated timing and
warnings in receipts without dumping extra report sections into the final table.

## User-facing output and operating commands

During execution show the start record, complete directive configuration, requested
part/frequency and live synthesis log, then an END record with status and log/report
paths. At the end show **only these report metrics**, with identifying columns:

```text
Case | Kernel | Config | FPGA part | MHz | Latency cycles | Latency basis | BRAM % | DSP % | FF % | LUT % | Status
```

Round percentages for display (e.g. four decimals); retain full precision/counts
in JSON. Show failed/timeout/undefined rows rather than silently dropping them.
Do not print unrelated timing/interval/operator/interface report sections, infer
model quality from a successful compile, or claim a run occurred without tool logs.
Do not start architecture search or silently alter directives to improve QoR.

The following are the **interface the generated runner should implement**, not
claims that a new batch runner has already been installed on Coroni:

```bash
# On Coroni, after the authorized operator uploads/unpacks the generated bundle:
source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
python3 run_batch.py --input SALT_experiments.jsonl --output /home/bonus24/salt_jsonl_synth/RUN_ID --dry-run
# Inspect plan, missing targets, materialized-source ledger and hashes first.
nohup python3 -u run_batch.py --input SALT_experiments.jsonl --output /home/bonus24/salt_jsonl_synth/RUN_ID --workers 1 --timeout-sec 21600 > driver.log 2>&1 < /dev/null &
tail -f driver.log
tail -f /home/bonus24/salt_jsonl_synth/RUN_ID/cases/CASE_ID/synthesis.log
tail -n 5 /home/bonus24/salt_jsonl_synth/RUN_ID/results.jsonl
```

`--dry-run` must not invoke HLS or consume the final execution namespace. It should
return all validation failures and a plan. The runner may require `--plan` for an
immutable plan instead of re-reading changed inputs; document the chosen interface.

From an authorized workstation, upload with `rsync -a`/`scp` and retrieve:

```bash
rsync -a --checksum bonus24@coroni.microlab.ntua.gr:/home/bonus24/salt_jsonl_synth/RUN_ID/ ./coroni_results/RUN_ID/
```

For a user without SSH, an operator performs these steps and returns a results
archive. A ChatGPT session can display periodic log excerpts only if it has a
connected execution tool; otherwise provide `tail -f` commands and downloadable
logs, not fabricated real-time progress.

## Acceptance checklist for the scripts ChatGPT generates

1. All requested rows/targets appear in the plan; missing targets/assignments fail
   before HLS. Source/action versions match, especially rendering 15 versus 12.
2. Native vendor AP types are selected; the GAN stub is absent from active includes.
3. Each source diff/ledger implements exactly the supplied predictions, including
   zero/none omission. No silent legalization, unrelated source rewrites or stale
   pragma inheritance. Original sources stay intact.
4. Every run has an isolated project, identity receipt, readable start/config log,
   raw synthesis log, timeout/process-group handling, and explicit failure status.
5. Reports match top/part/clock/flow and are fresh. Raw resources and their same-report
   denominators determine the four percentages. Missing latency stays unknown.
6. Optional co-simulation is explicitly declared and input-specific, with a passing
   testbench before numeric measured latency is reported.
7. Tests cover real 2021.1 XMLs, undefined latency, zero DSP, wrong top/part/clock,
   missing capacities, nested-loop materialization and rendering version rejection.
   Dry-run makes no HLS calls. No new HLS run was used to write this guide.
