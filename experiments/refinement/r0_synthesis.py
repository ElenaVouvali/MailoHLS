"""Conservative Vitis runner: isolated attempts, exact context, no stale reports."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from .r0_common import ROOT, sha256, object_hash, repo_path, write_json, write_jsonl, read_jsonl, display_path
from .r0_metrics import evaluate_metrics, PART, CLOCK_NS
from .r0_candidates import parse_assignment

HLS_VERSION = "2021.1"
SETTINGS = ["config_array_partition -complete_threshold 0 -throughput_driven off",
            "config_compile -pipeline_loops 0"]
VENDOR_HEADERS = {"ap_fixed.h", "ap_int.h"}
PH = re.compile(r"auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*)\}", re.I)


def materialize(template, target):
    assignments = parse_assignment(target)
    for line in template.splitlines():
        if re.match(r"\s*#\s*pragma\s+HLS\s+(pipeline|unroll|array_partition)\b", line, re.I) and not PH.search(line):
            raise ValueError("Ambiguous concrete optimization pragma outside the complete placeholder assignment")
    observed = ["AUTO{" + match[1].upper() + "}" for match in PH.finditer(template)]
    if set(observed) != set(assignments) or len(observed) != len(set(observed)):
        raise ValueError("Source/assignment placeholders are missing, extra, or duplicated")
    text = PH.sub(lambda match: assignments["AUTO{" + match[1].upper() + "}"], template)
    if "auto{" in text.lower():
        raise ValueError("Unresolved source placeholder")
    output = []
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*#\s*pragma\s+HLS\s+(?:pipeline\s+II=0|unroll\s+factor=0)\s*$", line, re.I):
            continue
        if re.match(r"\s*#\s*pragma\s+HLS\s+array_partition\b", line, re.I):
            if re.search(r"\btype=none\b", line, re.I):
                continue
            if re.search(r"\btype=complete\b", line, re.I):
                # SALT encodes complete partition with factor=0; Vitis should
                # see the complete directive, not an invalid factor=0 token.
                line = re.sub(r"\s+factor=0\b", "", line, flags=re.I)
        output.append(line)
    return "".join(output)


def tcl_quote(value):
    value = str(value)
    if any(c in value for c in "{}\n\r"):
        raise ValueError("Unsafe Tcl path/value")
    return "{" + value + "}"


def make_tcl(source_name, top, part=PART, clock=CLOCK_NS):
    if not re.fullmatch(r"[A-Za-z_]\w*", top) or part != PART or float(clock) != CLOCK_NS:
        raise ValueError("Invalid top or changed R0 deployment context")
    return "\n".join([
        "open_project -reset r0_project", "set_top " + top,
        "add_files " + tcl_quote(source_name),
        "open_solution -reset solution1 -flow_target vivado", "set_part " + tcl_quote(part),
        f"create_clock -period {float(clock):g} -name default", *SETTINGS,
        "csynth_design", "exit", ""])


class ReportError(ValueError):
    pass


def parse_report(path, top, part=PART, clock=CLOCK_NS, version=HLS_VERSION):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ReportError(f"Unreadable/partial XML: {exc}") from exc
    def text(query):
        return root.findtext(query)
    checks = {".//UserAssignments/Part": part, ".//UserAssignments/TopModelName": top,
              ".//ReportVersion/Version": version, ".//UserAssignments/FlowTarget": "vivado"}
    for query, expected in checks.items():
        if text(query) != expected:
            raise ReportError(f"Report provenance mismatch: {query}={text(query)!r}, expected {expected!r}")
    if text(".//UserAssignments/unit") != "ns" or text(".//SummaryOfOverallLatency/unit") != "clock cycles":
        raise ReportError("Report timing units do not match nanoseconds / clock cycles")
    try:
        actual_clock = float(text(".//UserAssignments/TargetClockPeriod"))
        if not math.isfinite(actual_clock) or abs(actual_clock - clock) > 1e-6:
            raise ValueError("changed target clock")
    except (ValueError, TypeError) as exc:
        raise ReportError("Missing/wrong requested clock") from exc
    def count(query):
        raw = text(query)
        if raw is None or not re.fullmatch(r"[0-9]+", raw.strip()):
            raise ReportError(f"Missing/invalid count: {query}={raw!r}")
        return int(raw)
    latency = count(".//SummaryOfOverallLatency/Worst-caseLatency")
    if latency <= 0:
        raise ReportError("Nonpositive worst-case latency")
    resources = {name: count(".//AreaEstimates/Resources/" + name) for name in ("BRAM_18K", "DSP", "FF", "LUT")}
    estimated = text(".//SummaryOfTimingAnalysis/EstimatedClockPeriod")
    try:
        estimated = float(estimated)
        if not math.isfinite(estimated) or estimated <= 0:
            estimated = None
    except (ValueError, TypeError):
        estimated = None
    return {**evaluate_metrics(latency, resources), "estimated_clock_ns": estimated,
            "timing_status": "unknown" if estimated is None else "HLS estimate only; not post-route timing",
            "estimated_clock_le_requested": None if estimated is None else estimated <= clock,
            "requested_clock_ns": actual_clock, "part": part, "top": top, "hls_version": version}


def tool_info(executable):
    path = shutil.which(str(executable))
    if path is None:
        raise FileNotFoundError(f"Vitis HLS not found: {executable}. Preparation/summarization need no HLS; run on a Vitis HLS 2021.1 host with --vitis-hls /path/to/vitis_hls.")
    outputs = []
    for flag in ("-version", "--version"):
        result = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        outputs.append(output)
        if result.returncode == 0 and re.search(r"(?:Vitis|Vivado|Version).*?\b(?:v)?2021\.1\b", output, re.I):
            headers = {}
            for parent in list(Path(path).resolve().parents)[:4]:
                for name in VENDOR_HEADERS:
                    header = parent / "include" / name
                    if header.is_file():
                        payload = header.read_text(errors="replace")
                        if ("Xilinx" not in payload and "Advanced Micro Devices" not in payload) or "AP_FIXED_STUB_H" in payload or "AP_INT_STUB_H" in payload:
                            continue
                        headers.setdefault(name, {"path": str(header), "sha256": sha256(header)})
            return {"executable": str(Path(path).resolve()), "version": HLS_VERSION,
                    "executable_sha256": sha256(path), "version_output": output.strip(),
                    "vendor_headers": headers}
    raise ValueError("The executable did not identify as Vitis HLS 2021.1: " + "\n".join(outputs))


def verify_source(kernel):
    relation = kernel.get("relations", {})
    if relation.get("status") == "available":
        if sha256(repo_path(relation["artifact"])) != relation["artifact_hash"]:
            raise ValueError("Compiler relation artifact changed since preparation")
        path = relation.get("compiler_active_metadata_path")
        if path and sha256(repo_path(path)) != relation["compiler_active_metadata_sha256"]:
            raise ValueError("Compiler-active metadata changed since preparation")
    if sha256(repo_path(kernel["template_path"])) != kernel["template_sha256"]:
        raise ValueError("Prepared source template changed")
    for entry in kernel["source_provenance"]["files"]:
        path = repo_path(entry["snapshot"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Source snapshot changed since preparation: {path}")


def run_candidate(candidate, kernel, output, tool, timeout=21600):
    verify_source(kernel)
    root = Path(output) / "projects" / candidate["candidate_id"]
    number = 1
    while (root / f"attempt-{number:03}").exists():
        number += 1
    work = root / f"attempt-{number:03}"
    work.mkdir(parents=True, exist_ok=False)
    source_name = Path(kernel["source_provenance"]["source"]).name
    template = repo_path(kernel["template_path"]).read_text()
    source_text = materialize(template, candidate["target"])
    (work / source_name).write_text(source_text)
    # Vitis must use its genuine fixed-point/AXI integer implementation, never
    # the repository's explicitly simplified compiler-analysis header stubs.
    analysis_headers = []
    for entry in kernel["source_provenance"]["files"]:
        path = repo_path(entry["snapshot"])
        if path.suffix.lower() not in {".h", ".hpp", ".hh", ".inc"}:
            continue
        if path.name in VENDOR_HEADERS:
            header = tool.get("vendor_headers", {}).get(path.name)
            if not header or sha256(header["path"]) != header["sha256"]:
                raise ValueError(f"Cannot verify genuine Vitis header {path.name}; analysis stubs are forbidden for synthesis")
            analysis_headers.append(path.name)
            continue
        destination = work / path.relative_to(repo_path(kernel["source_provenance"]["snapshot_source"]).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    tcl = make_tcl(source_name, kernel["top"])
    if analysis_headers:
        include_dirs = sorted({str(Path(tool["vendor_headers"][name]["path"]).parent) for name in analysis_headers})
        flags = " ".join('-I"' + directory + '"' for directory in include_dirs)
        tcl = tcl.replace("add_files " + tcl_quote(source_name), "add_files -cflags " + tcl_quote(flags) + " " + tcl_quote(source_name))
    (work / "run_hls.tcl").write_text(tcl)
    receipt = {"candidate_id": candidate["candidate_id"], "canonical_signature": candidate["canonical_signature"],
               "evaluation_context_hash": candidate["evaluation_context_hash"],
               "materialized_source_sha256": sha256(work / source_name),
               "tcl_sha256": sha256(work / "run_hls.tcl"), "tool": tool,
               "analysis_stubs_not_copied": analysis_headers,
               "header_policy": "Vitis supplies ap_fixed.h/ap_int.h; remaining source headers copied exactly"}
    write_json(work / "receipt.json", receipt)
    log = work / "synthesis.log"
    report = work / "r0_project/solution1/syn/report" / (kernel["top"] + "_csynth.xml")
    assert not report.exists()
    start = time.monotonic()
    status, error, returncode = "failed", None, None
    metrics = {**evaluate_metrics(None, None), "estimated_clock_ns": None, "timing_status": "unknown"}
    with log.open("w") as stream:
        process = subprocess.Popen([tool["executable"], "-f", "run_hls.tcl"], cwd=work,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            status, error = "timeout", f"Exceeded {timeout} seconds"
    runtime = time.monotonic() - start
    if status != "timeout":
        if returncode != 0:
            error = f"Vitis exited with code {returncode}; report not ranked even if present"
        elif not report.is_file():
            status, error = "missing_report", "Vitis returned without the top-level csynth report"
        else:
            try:
                metrics = parse_report(report, kernel["top"])
                status = "success"
            except ReportError as exc:
                status, error = "invalid_report", str(exc)
    result = {"candidate_id": candidate["candidate_id"], "kernel": candidate["kernel"],
              "candidate_group": candidate["candidate_group"], "synthesis_status": status,
              "evaluation_context_hash": candidate["evaluation_context_hash"],
              "tool_hash": object_hash(tool), "attempt_count": number, "hls_runtime_seconds": runtime,
              "report_path": display_path(report), "log_path": display_path(log),
              "receipt_path": display_path(work / "receipt.json"), "returncode": returncode,
              "error": error, "requested_clock_ns": CLOCK_NS, "part": PART, "top": kernel["top"],
              "report_sha256": sha256(report) if report.is_file() else None, **metrics}
    if status != "success":
        result["rankable"] = False
    write_json(work / "result.json", result)
    return result


def run_queue(manifest, candidates, output, executable, timeout=21600, limit=None, smoke=None, retry_failed=False):
    tool = tool_info(executable)  # Fail before creating jobs or false QoR rows.
    output = Path(output)
    results_path = output / "synthesis_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    if len({r["candidate_id"] for r in results}) != len(results):
        raise ValueError("Duplicate synthesis result IDs")
    done = {r["candidate_id"]: r for r in results}
    selected = candidates
    if smoke:
        selected = []
        for group in ("baseline", "independent", "structural"):
            rows = [c for c in candidates if c["kernel"] == smoke and c["candidate_group"] == group]
            if not rows:
                raise ValueError(f"Smoke test requires a baseline and both neighborhoods for {smoke}")
            selected.append(rows[0])
    launched = 0
    for candidate in selected:
        old = done.get(candidate["candidate_id"])
        if old:
            if old["evaluation_context_hash"] != candidate["evaluation_context_hash"] or old["tool_hash"] != object_hash(tool):
                raise ValueError("Resume context/tool mismatch; choose a new output directory")
            if old["synthesis_status"] == "success":
                path = repo_path(old["report_path"])
                if not path.is_file() or sha256(path) != old["report_sha256"]:
                    raise ValueError("Completed report missing/modified; refusing blind cache reuse")
                continue
            if not retry_failed:
                continue
        if limit is not None and launched >= limit:
            break
        if candidate["candidate_group"] != "baseline":
            bases = [r for r in done.values() if r["kernel"] == candidate["kernel"] and r["candidate_group"] == "baseline"]
            if not bases or bases[0]["synthesis_status"] != "success":
                continue
        kernel = manifest["kernels"][candidate["kernel"]]
        result = run_candidate(candidate, kernel, output, tool, timeout)
        done[result["candidate_id"]] = result
        write_jsonl(results_path, [done[c["candidate_id"]] for c in candidates if c["candidate_id"] in done])
        launched += 1
    return launched
