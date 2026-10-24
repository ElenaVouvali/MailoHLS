"""Read-only R0 recovery, strict measurement authentication and context identity."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from LLM_branch.common import directive_domains as dd
from experiments.refinement import r0_candidates as c, r0_synthesis as hls
from experiments.refinement.r0_common import ROOT, object_hash, sha256, read_jsonl, repo_path, display_path
from experiments.refinement.r0_local_headroom import CONTEXT, code_hashes


def tool_identity(tool):
    if tool.get("version") != "2021.1" or not tool.get("executable_sha256"):
        raise ValueError("Unverified HLS version/executable")
    headers = tool.get("vendor_headers", {})
    if any(not headers.get(name, {}).get("sha256") for name in hls.VENDOR_HEADERS):
        raise ValueError("Missing vendor-header provenance")
    return {"version": tool["version"], "executable_sha256": tool["executable_sha256"],
        "vendor_headers": {name: headers[name]["sha256"] for name in sorted(headers)}}


def source_context(kernel):
    prov = kernel["source_provenance"]
    source_parent = Path(prov["snapshot_source"]).parent
    headers = {str(Path(f["snapshot"]).relative_to(source_parent)): f["sha256"] for f in prov["files"]
               if Path(f["snapshot"]).suffix.lower() in {".h", ".hpp", ".hh", ".inc"}}
    return {"deployment": CONTEXT, "top": kernel["top"], "source_sha256": prov["source_sha256"],
        "metadata_sha256": prov["metadata_sha256"], "template_sha256": kernel["template_sha256"],
        "snapshot_headers": headers, "header_policy": "genuine verified vendor AP headers; other headers copied exactly"}


def comparison_id(kernel, tool):
    return object_hash({"source_context": source_context(kernel), "tool": tool_identity(tool)})


def expected_tcl(source_name, top, files, tool):
    text = hls.make_tcl(source_name, top)
    required = {Path(f["snapshot"]).name for f in files if Path(f["snapshot"]).name in hls.VENDOR_HEADERS}
    if required:
        paths = sorted({str(Path(tool["vendor_headers"][name]["path"]).parent) for name in required})
        flags = " ".join('-I"' + path + '"' for path in paths)
        text = text.replace("add_files " + hls.tcl_quote(source_name),
            "add_files -cflags " + hls.tcl_quote(flags) + " " + hls.tcl_quote(source_name))
    return text


def authenticate_work(candidate, result, kernel, resolve, *, expected_evaluation_hash):
    """Reparse actual XML and verify source, headers, receipts, Tcl and completion."""
    if result.get("synthesis_status") != "success" or result.get("returncode") != 0:
        raise ValueError("Synthesis did not successfully complete")
    signature = c.signature(candidate["assignment"])
    if signature != candidate["canonical_signature"] or c.parse_assignment(candidate["target"]) != candidate["assignment"]:
        raise ValueError("Candidate signature/serialization mismatch")
    if result.get("evaluation_context_hash") != expected_evaluation_hash:
        raise ValueError("Evaluation context hash mismatch")
    for f in kernel["source_provenance"]["files"]:
        if sha256(resolve(f["snapshot"])) != f["sha256"]:
            raise ValueError("Source/metadata/header snapshot hash mismatch")
    prov = kernel["source_provenance"]
    if sha256(resolve(prov["snapshot_source"])) != prov["source_sha256"] or sha256(resolve(prov["snapshot_metadata"])) != prov["metadata_sha256"]:
        raise ValueError("Source or metadata identity mismatch")
    template_path = resolve(kernel["template_path"])
    if sha256(template_path) != kernel["template_sha256"]:
        raise ValueError("Source template hash mismatch")
    receipt_path, report_path, log_path = (resolve(result[field]) for field in ("receipt_path", "report_path", "log_path"))
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("candidate_id") != candidate["candidate_id"] or receipt.get("canonical_signature") != signature
            or receipt.get("evaluation_context_hash") != expected_evaluation_hash):
        raise ValueError("Receipt identity mismatch")
    tool = receipt["tool"]
    identity = tool_identity(tool)
    if result.get("tool_hash") != object_hash(tool):
        raise ValueError("Tool receipt hash mismatch")
    work = receipt_path.parent
    source_name = Path(prov["source"]).name
    source = work / source_name
    if sha256(source) != receipt["materialized_source_sha256"] or source.read_text() != hls.materialize(template_path.read_text(), candidate["target"]):
        raise ValueError("Materialized source differs from exact complete assignment")
    if (sha256(work / "run_hls.tcl") != receipt["tcl_sha256"]
            or (work / "run_hls.tcl").read_text() != expected_tcl(source_name, kernel["top"], prov["files"], tool)):
        raise ValueError("Tcl synthesis settings differ")
    required = set()
    for entry in prov["files"]:
        snapshot = Path(entry["snapshot"])
        if snapshot.suffix.lower() not in {".h", ".hpp", ".hh", ".inc"}:
            continue
        relative = snapshot.relative_to(Path(prov["snapshot_source"]).parent)
        if snapshot.name in hls.VENDOR_HEADERS:
            required.add(snapshot.name)
            if (work / relative).exists():
                raise ValueError("Analysis AP stub was copied into synthesis work")
        elif sha256(work / relative) != entry["sha256"]:
            raise ValueError("Synthesis dependency header mismatch")
    if required != set(receipt.get("analysis_stubs_not_copied", [])):
        raise ValueError("Unverified vendor header substitution")
    log = log_path.read_text(errors="replace")
    if "Finished Command csynth_design" not in log or "Exiting vitis_hls" not in log:
        raise ValueError("Missing completed synthesis log evidence")
    if sha256(report_path) != result.get("report_sha256"):
        raise ValueError("Report integrity mismatch")
    parsed = hls.parse_report(report_path, kernel["top"])
    xml = ET.parse(report_path).getroot()
    available = {key: int(xml.findtext(".//AreaEstimates/AvailableResources/" + key)) for key in CONTEXT["device_capacities"]}
    if available != CONTEXT["device_capacities"]:
        raise ValueError("Report device capacities differ")
    for key in ("latency_cycles", "resources", "ADP", "area_proxy", "normalized_utilization", "feasible", "rankable", "part", "requested_clock_ns", "top", "hls_version"):
        if result.get(key) != parsed[key]:
            raise ValueError(f"Stored QoR/provenance differs from actual XML: {key}")
    return {**parsed, "tool_identity": identity, "comparison_context_id": comparison_id(kernel, tool),
        "evidence": {str(p): sha256(p) for p in (receipt_path, report_path, log_path, source, work / "run_hls.tcl")}}


class R0Archive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.kernel = self.manifest["kernels"]["GAN"]
        self.prefix = Path(self.kernel["template_path"]).parents[2]
        self.candidates = read_jsonl(self.path / "candidates.jsonl")
        self.pool = read_jsonl(self.path / "candidate_pool.jsonl")
        self.results = read_jsonl(self.path / "synthesis_results.jsonl")
        if self.manifest["context"] != CONTEXT:
            raise ValueError("R0 deployment context is incompatible")
        self.actions = dd.load_source_actions("GAN", self.resolve(self.kernel["source_provenance"]["snapshot_metadata"]).parent)
        template = self.resolve(self.kernel["template_path"]).read_text()
        self.native_domains = dd.source_site_domains("GAN", template,
            self.resolve(self.kernel["source_provenance"]["snapshot_metadata"]).parent)
        self.base = self.kernel["initial_assignment"]
        if c.from_paper(self.kernel["paper_baseline"], self.actions) != self.base:
            raise ValueError("Stored R0 baseline differs from its complete published directives")
        self.domains = c.published_factor_domains(self.base, self.actions, self.native_domains)
        c.validate(self.base, self.actions, self.domains)
        if self.domains != self.kernel["domains"]:
            raise ValueError("R0 recorded domains differ from source/baseline reconstruction")

    def resolve(self, value):
        path = Path(value)
        try:
            relative = path.relative_to(self.prefix)
        except ValueError:
            return repo_path(value)
        return self.path / relative

    def authenticate(self):
        by_id = {candidate["candidate_id"]: candidate for candidate in self.candidates if candidate["kernel"] == "GAN"}
        audits, authenticated = [], {}
        for result in self.results:
            if result.get("kernel") != "GAN":
                continue
            record = {"candidate_id": result.get("candidate_id"), "status": "historical_unverified"}
            try:
                if self.manifest["code_sha256"] != code_hashes() or object_hash(self.candidates) != self.manifest["candidates_sha256"]:
                    raise ValueError("R0 code or frozen candidate queue differs")
                candidate = by_id[result["candidate_id"]]
                c.validate(candidate["assignment"], self.actions, self.domains)
                expected = object_hash({"context": CONTEXT, "source": self.kernel["source_provenance"],
                    "template_sha256": self.kernel["template_sha256"], "top": self.kernel["top"],
                    "assignment": candidate["canonical_signature"], "code": self.manifest["code_sha256"]})
                if candidate["evaluation_context_hash"] != expected:
                    raise ValueError("Candidate context is not the R0 frozen context")
                verified = authenticate_work(candidate, result, self.kernel, self.resolve, expected_evaluation_hash=expected)
                record.update(status="authenticated_reuse", **verified)
                authenticated[candidate["canonical_signature"]] = {**result, **verified,
                    "authentication": "authenticated_reuse", "original_candidate": candidate}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record["reason"] = str(exc)
            audits.append(record)
        return audits, authenticated


def snapshot_kernel(kernel, resolve, output):
    """Relocate frozen sources/graph into R0.1, preserving content identities."""
    result = copy.deepcopy(kernel)
    mapping = {}
    old_parent = Path(kernel["source_provenance"]["snapshot_source"]).parent
    for entry in result["source_provenance"]["files"]:
        old = entry["snapshot"]
        target = Path(output) / "sources/GAN" / Path(old).relative_to(old_parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if sha256(resolve(old)) != entry["sha256"]:
            raise ValueError("Frozen source snapshot differs")
        if target.exists() and sha256(target) != entry["sha256"]:
            raise ValueError("Existing R0.1 source snapshot differs")
        shutil.copy2(resolve(old), target)
        mapping[old] = display_path(target)
        entry["snapshot"] = display_path(target)
    prov = result["source_provenance"]
    for key in ("snapshot_source", "snapshot_metadata"):
        prov[key] = mapping[prov[key]]
    target = Path(output) / "sources/GAN/source_template.txt"
    shutil.copy2(resolve(kernel["template_path"]), target)
    result["template_path"] = display_path(target)
    relation = result.get("relations", {})
    if relation.get("status") == "available":
        target = Path(output) / "graphs/GAN.gexf"
        target.parent.mkdir(parents=True, exist_ok=True)
        if sha256(resolve(relation["artifact"])) != relation["artifact_hash"]:
            raise ValueError("R0 graph artifact hash mismatch")
        shutil.copy2(resolve(relation["artifact"]), target)
        relation["artifact"] = display_path(target)
    return result


def baseline_comparison(archive, authenticated):
    paper = archive.kernel["paper_baseline"]
    result = {"paper_reference": paper, "paper_file_unchanged": True,
        "status": "unresolved_baseline_discrepancy", "conclusions": []}
    baseline = authenticated.get(c.signature(archive.base))
    if not baseline:
        result["conclusions"].append("No authenticated new baseline yet")
        return result
    result["new_baseline"] = {key: baseline[key] for key in ("latency_cycles", "resources", "ADP", "tool_identity", "comparison_context_id")}
    evidence_path = archive.path / "historical_evidence/GAN"
    try:
        audit = json.loads((archive.path / "gan_historical_header_audit.json").read_text())
        for name, expected in audit["archived_files_sha256"].items():
            if sha256(evidence_path / name) != expected:
                raise ValueError("Historical evidence integrity mismatch")
        xml = ET.parse(evidence_path / "forward_propagation_csynth.xml").getroot()
        resources = {key: int(xml.findtext(".//AreaEstimates/Resources/" + key)) for key in CONTEXT["device_capacities"]}
        expected_resources = {key: paper["resources"][old] for key, old in
            {"BRAM_18K": "bram", "DSP": "dsp", "FF": "ff", "LUT": "lut"}.items()}
        old_source = evidence_path / "kernel_placeholders.cpp"
        new_source = archive.resolve(baseline["receipt_path"]).parent / Path(archive.kernel["source_provenance"]["source"]).name
        def normalize(path):
            return re.sub(r"\s+", "", re.sub(r"/\*L[0-9]+:\*/|\bL[0-9]+:", "", path.read_text()))
        checks = {
            "report_matches_paper": resources == expected_resources and int(xml.findtext(".//SummaryOfOverallLatency/Worst-caseLatency")) == paper["latency"],
            "source_equal_except_labels_whitespace": normalize(old_source) == normalize(new_source),
            "other_headers_identical": all(sha256(evidence_path / name) == sha256(new_source.parent / name) for name in ("network.h", "weight_definitions.h", "tanh.h")),
            "historical_header_is_float_stub": "AP_FIXED_STUB_H" in (evidence_path / "ap_fixed.h").read_text() and "float v;" in (evidence_path / "ap_fixed.h").read_text(),
            "matching_tool_target_clock": xml.findtext(".//ReportVersion/Version") == "2021.1" and xml.findtext(".//UserAssignments/Part") == CONTEXT["part"] and float(xml.findtext(".//UserAssignments/TargetClockPeriod")) == 10.0,
            "historical_tcl_matches": (evidence_path / "run_hls.tcl").read_text().replace("hls_project", "r0_project").replace("kernel_placeholders.cpp", "network.cpp").split() == hls.make_tcl("network.cpp", archive.kernel["top"]).replace("{network.cpp}", "network.cpp").split(),
        }
        result["evidence_checks"] = checks
        result["historical_evidence_sha256"] = audit["archived_files_sha256"]
        if all(checks.values()):
            result["status"] = "explained_header_semantics_difference"
            result["conclusions"].append("Matching archived GAN run used float-based ap_fixed analysis stubs; new authenticated R0 uses genuine vendor headers. Numerical implementations differ; paper data is retained, not replaced.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["unresolved_reason"] = str(exc)
    return result
