# Copy this prompt into the ChatGPT session that will generate the runner

I want a reproducible batch HLS synthesis workflow for my own authorized Coroni
account/operator. Read the attached `coroni_chatgpt_synthesis_guide.md`, source
snapshot, `source_inventory.json`, `action_map.json`, toolchain receipt and
`parse_csynth.py` first. Then inspect my actual input JSONL; do not assume its
schema from its filename. These are instructions for producing scripts, not
permission to invent executed jobs or measurements.

Generate a self-contained, documented Python 3.6-compatible runner and case builder
for GAN, Kalman, trmm-opt, AES, jacobi-2d, syr2k, covariance, JPEG and 3D rendering.
Use the exact source/top mappings and versioned native-AP compilation profile in
the guide. Use `/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls` after its settings script,
Vivado IP flow, `pipeline_loops=0`, and the recorded automatic array-partition
settings. Preserve exact device/package/speed grade and requested clock per row.

Require explicit part/device and frequency/period, either per row or from an
explicitly declared common target. The original `SALT_experiments.jsonl` has 40
rows but no targets; do not guess them. Preserve all requested rows/stages/objectives.
Ignore input latency/resource annotations when reporting new synthesis results.
Resolve conflicts/missing information before running HLS while doing independent
offline preparation where possible.

Materialize every supplied action exactly, with complete action coverage, no
silent factor/II repair, no inherited concrete pragmas, no unresolved placeholders,
and correct zero/none omission. Retain raw predictions and an emitted-directive
ledger/source diff. Use real Vitis AP types, exclude GAN's local AP software stub
from active include paths, preserve its three project headers, and verify the
15-action rendering source instead of its stale 12-action metadata. Do not compile
the alternate malformed rendering source or trust stale top-level Tcl filenames.

Implement `--dry-run`, an immutable plan/manifest, safe per-case directories,
batch locking, sequential default execution, per-process-group timeouts, preserved
retry attempts, checksummed resume, and incremental `results.jsonl`/`summary.csv`.
Validate each result's top-level report identity, requested part/clock and run
completion. Keep failed/timeout-with-report states distinct from success.

Before every case print index/total, kernel, case/config ID, full directive
configuration, exact FPGA part, requested frequency and period, source hash and
log path. Stream merged Vitis stdout/stderr to both terminal and that case's full
`synthesis.log`. At completion print status and result/report paths.

At the end show a compact table containing identifying columns and only these
report metrics: top-level latency cycles, BRAM %, DSP %, FF %, LUT %. Compute
percentages as 100 * used / available from the SAME top-level report. Preserve raw
counts/denominators and full precision in JSON. Missing fields remain N/A; zero
usage is 0%; utilization above 100% must not be clamped.

Use HLS top-level worst-case cycles when numeric. If HLS reports `undef`, show
N/A rather than copying historical values or summing subfunctions. If I explicitly
request numeric measured latency, add validated C/RTL co-simulation with hashed
testbench/data and label its cycles as input-specific co-simulation latency, not a
static worst-case bound. Refer to the guide for JPEG/rendering/AES testbenches and
the rendering GMP compatibility flag. Never promise numeric latency for all cases.

Provide focused tests for materialization/action binding and report parsing using
the supplied historical fixtures, without launching synthesis as a test. Provide
exact upload, dry-run, launch, live-monitor, resume and result-retrieval commands.

If you have an authorized SSH execution tool, first perform preflight/dry-run and
then follow my explicit batch execution authorization. Otherwise generate a ZIP
for the authorized operator and tell me which commands the operator must run;
after I return logs/results, parse them. Never claim a server run or live progress
without actual execution evidence. Do not request or embed private SSH keys.

Deliver `run_batch.py`, case-building/materialization code, report parser, request
schema/examples, source/action/tool manifests, focused tests, and an operator
README. State actual blockers precisely. Do not modify shared Application sources,
old projects, model checkpoints, historical results or my requested directives.
