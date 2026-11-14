# MailoHLS — CHStone prompt views and Coroni JSONL synthesis

## Scope and verified basis

This package was built from the publicly readable MailoHLS research revision
`6ee1390611c9047a4b8e2b07200d223b93ee8d16` and the reported Davinci
`stage2_infer.py --help`. The separately named `coroni_chatgpt_guide_bundle.zip`,
`docs/coroni_chatgpt_synthesis_guide.md`, Coroni source snapshots, and **actual
Stage-2 JSONL** were not supplied and were not accessible at that GitHub revision.
The scripts therefore follow the existing repository's 2021.1 HLS and report
contracts rather than claiming tested compatibility with that unavailable guide.

**Tested here:** Python syntax, four offline unit tests: placeholder rendering,
strict action mapping, provenance rejection, and `undef` XML semantics. **Not
executed here:** Davinci Stage-2 model inference; real nine-source preparation;
Vitis HLS runs on Coroni; validation of Coroni's nine historical reports.

## Part A — Minimal LLM-only AES/JPEG source views

The original AES source `evaluation_set/chstone-aes/kernel.c` is 44,657 bytes,
with action sites L1–L6. The JPEG source `evaluation_set/chstone-jpeg/kernel.c`
is 160,673 bytes, with L1–L21 and embedded binary test data. Neither should be
fed in full to a 7,168-token-trained model. `make_prompt_view.py` removes
non-action helper bodies, comments except Lk labels, and large literal global
array initializers. It retains complete bodies of action-bearing functions and
the original top-level entrypoint/call context. **The views are not executable C
and must never replace the original source in HLS, compiler graph extraction,
validation, or GMT provenance.** These are explicitly non-equivalent compact
prompt representations; do not report them as full-source inference.

Transfer these three files (`make_prompt_view.py`, `patch_inference_views.py`,
`README.md`) to Davinci. Run there:

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
I=/home/evouvali/MailoHLS/inference
mkdir -p "$D/eval_prompt_views"
python3 make_prompt_view.py --kernel chstone-aes \
  --source "$D/eval_sources_compat9/chstone-aes/source.c" \
  --output "$D/eval_prompt_views/chstone-aes.c"
python3 make_prompt_view.py --kernel chstone-jpeg \
  --source "$D/eval_sources_compat9/chstone-jpeg/source.c" \
  --output "$D/eval_prompt_views/chstone-jpeg.c"
python3 patch_inference_views.py --runner "$I/stage2_infer.py" \
  --output "$I/stage2_infer_compact_views.py"
```

The patch writes **a separate inference-runner copy**, not a change to the
original. It replaces `prepared` only after canonical source/action validation,
only for AES and JPEG, and verifies exact directive placeholder key equality.
The original full source is still used to calculate source hashes and for HLS.

Preflight:

```bash
D=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
I=/home/evouvali/MailoHLS/inference
CUDA_VISIBLE_DEVICES="" "$D/venv/bin/python" "$I/stage2_infer_compact_views.py" \
  --checkpoint-step 300 --kernel all --device zcu104 --clock-ns 10 \
  --budget 100 --objective latency \
  --sources-root "$D/eval_sources_compat9" \
  --memory-dir "$D/eval_gmt_memory_B_compat9" --dry-run
```

**Important:** the published runner's `--dry-run` checks input contracts but may
not tokenize the complete prompt. Test AES and JPEG individually on the GPU
before `--kernel all`. No exact DeepSeek token count was available here: if a
view still exceeds 7,168 tokens, it must be further shortened and its new hash
recorded; never increase the model budget or silently truncate. Ensure V100 is
idle, then:

```bash
GPU=GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f
CUDA_VISIBLE_DEVICES="$GPU" "$D/venv/bin/python" "$I/stage2_infer_compact_views.py" \
  --checkpoint-step 300 --kernel chstone-aes --device zcu104 --clock-ns 10 \
  --budget 100 --objective latency --sources-root "$D/eval_sources_compat9" \
  --memory-dir "$D/eval_gmt_memory_B_compat9" --output "$I/aes_compact_step300.jsonl"
CUDA_VISIBLE_DEVICES="$GPU" "$D/venv/bin/python" "$I/stage2_infer_compact_views.py" \
  --checkpoint-step 300 --kernel chstone-jpeg --device zcu104 --clock-ns 10 \
  --budget 100 --objective latency --sources-root "$D/eval_sources_compat9" \
  --memory-dir "$D/eval_gmt_memory_B_compat9" --output "$I/jpeg_compact_step300.jsonl"
# If both pass, a fresh output path:
CUDA_VISIBLE_DEVICES="$GPU" "$D/venv/bin/python" "$I/stage2_infer_compact_views.py" \
  --checkpoint-step 300 --kernel all --device zcu104 --clock-ns 10 \
  --budget 100 --objective latency --sources-root "$D/eval_sources_compat9" \
  --memory-dir "$D/eval_gmt_memory_B_compat9" \
  --output "$I/stage2_eval9_step300_compact_v1.jsonl"
```

Keep checkpoint-450 partial results intact. The compact-source views are a
*separate* evaluation representation; Stage-1 must use the same view for matched
comparisons.

## Part B — Offline preparation of a full-source HLS synthesis batch

On Liono, use `P=/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final/artifacts/stage2_eval_9_20261113`, where `P/sources` already contains nine
canonical sources and metadata. Copy the JSONL from Davinci if needed. Install
this `coroni_batch.py` file in any directory; Python 3.6+ stdlib is enough.

```bash
python3 coroni_batch.py prepare \
  --predictions /path/to/stage2_eval9_step300_compact_v1.jsonl \
  --sources-root "$P/sources" \
  --device zcu104 --clock-ns 10 --budget-percent 100 \
  --require-nine --out "$P/coroni_stage2_step300_zcu104_10ns"
```

This creates a sealed bundle with `manifest.json`, `SHA256SUMS`, `jobs/*/kernel.c(pp)`,
`jobs/*/run_hls.tcl` and a self-contained `coroni_batch.py`. It does NOT synthesize.
All nine actions must match the original `kernel_info.txt`; every predicted
field must be present. The original full source is used, never the compact view.
No implicit target from JSONL is trusted: pass it explicitly as above.

For a **target sweep**, put this in a `targets.json` file:

```json
[
  {"device":"zcu104","clock_ns":10,"budget_percent":100},
  {"device":"zcu104","frequency_mhz":150,"budget_percent":50},
  {"device":"u200","clock_ns":10,"budget_percent":100}
]
```

Then add `--targets-json targets.json` instead of relying on CLI single-target
settings. For another FPGA, supply its *exact* `part` and authoritative
`capacities` (BRAM_18K/DSP/FF/LUT) in the target JSON. The XML parser verifies
the HLS report against these values. A 50% budget is a **posthoc feasibility
limit**, not a different physical device synthesis constraint.

## Part C — Execution on Coroni

`CORONI` below is your **actual** authorized SSH target; the guide bundle did
not provide a verified hostname. From Liono:

```bash
CORONI=your-verified-coroni-ssh-host
B="$P/coroni_stage2_step300_zcu104_10ns"
REMOTE=/home/YOUR_CORONI_USERNAME/salt_eval9_stage2_300
ssh "$CORONI" "test ! -e '$REMOTE' && mkdir -p '$REMOTE/bundle' '$REMOTE/results'"
rsync -av "$B/" "$CORONI:$REMOTE/bundle/"
ssh "$CORONI" "cd '$REMOTE/bundle' && sha256sum -c SHA256SUMS"
ssh -t "$CORONI" "cd '$REMOTE/bundle' && python3 coroni_batch.py run \
  --bundle . --results '$REMOTE/results' \
  --tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls \
  --timeout-seconds 3600"
```

Each job prints prefixed live logs and stores its full Vitis transcript, Tcl,
source, `run_receipt.json`, and synthesis reports under a fresh result directory.
Already-existing results are skipped, not overwritten. No guessed `script_*.tcl`
is reused: each job uses the `kernel_info.txt` top function and its own input.

## Part D — Collect the authenticated results on Liono

```bash
mkdir -p "$P/coroni_stage2_step300_zcu104_10ns_results"
rsync -av "$CORONI:$REMOTE/results/" "$P/coroni_stage2_step300_zcu104_10ns_results/"
python3 coroni_batch.py collect \
  --bundle "$P/coroni_stage2_step300_zcu104_10ns" \
  --results "$P/coroni_stage2_step300_zcu104_10ns_results" \
  --summary "$P/coroni_stage2_step300_zcu104_10ns_summary"
```

`summary.csv` and `results.jsonl` contain **only synthesized measurements**:
worst-case latency cycles (null if undefined) plus the four resource percentages.
Raw `resources`, percentages and budget feasibility are kept separately in
receipts/parsed state as appropriate; the summary table shows no invented cycles.
No simulation or post-route timing is performed; source compilation plus `csynth`
is not proof of functional correctness, bounded latency, or routed timing closure.

## Safety/contract notes

* Never use the synthetic `script_syr2k.tcl` or `script_trmm.tcl` from the old
  Coroni working directory: previous inspection found incorrect `set_top` values.
* Real Vitis `ap_fixed.h`/`ap_int.h` must be found through the installed tool;
  the software-only GAN AP stub is intentionally not copied into HLS jobs.
* Rosetta's 12-action metadata from `eval_sources_compat9` is the source
  contract; never substitute a separate 15-action Coroni rendering source.
* Never infer device/frequency from a bare `SALT_experiments.jsonl`; always pass
  explicit target contexts.
* A successfully generated output may have `Worst-caseLatency=undef`. This is
  reported as `latency_undefined`, with resource percentages retained. It is not
  automatically reinterpreted as synthesis failure or zero latency.
* A physically measured QoR result requires actual Coroni execution. In this
  environment only preparation tests were executed.
