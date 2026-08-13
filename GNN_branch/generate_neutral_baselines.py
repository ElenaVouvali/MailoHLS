#!/usr/bin/env python3
"""Synthesize one directive-neutral Vitis HLS reference per MailoHLS kernel.

Run this with the same Vitis release, FPGA part, and clock as the QoR CSVs.
The configured test kernel is excluded by default and cannot be synthesized
without an explicit unlock, preserving validation-only model development.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDS = (
    "kernel",
    "status",
    "source_sha256",
    "toolchain_id",
    "device",
    "clock_period_ns",
    "baseline_latency_ms",
    "baseline_area_score",
    "bram_util_percent",
    "dsp_util_percent",
    "ff_util_percent",
    "lut_util_percent",
    "report_path",
    "error",
)
SEARCH_DIRECTIVE = re.compile(
    r"^\s*#\s*pragma\s+HLS\s+(pipeline|unroll|array_partition)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--application-information",
        type=Path,
        default=repo / "Data" / "ApplicationInformation.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "GNN_branch" / "baselines" / "neutral_vitis_2021_1.csv",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo / "neutral_baseline_build",
    )
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--expected-tool-version", default="2021.1")
    parser.add_argument("--device", default="xczu7ev-ffvc1156-2-e")
    parser.add_argument("--clock-period-ns", type=float, default=10.0)
    parser.add_argument(
        "--exclude-kernels",
        default="serrano-kalman-filter",
        help="Comma-separated locked kernels; the default is the final test kernel.",
    )
    parser.add_argument(
        "--kernels",
        default=None,
        help="Optional comma-separated subset; otherwise read ALL_KERNEL from config.py.",
    )
    parser.add_argument(
        "--allow-test-kernels",
        action="store_true",
        help="One-shot final phase only: permit explicitly requested locked kernels.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun-successes", action="store_true")
    return parser.parse_args()


def comma_set(value):
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def configured_kernels(config_path):
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ALL_KERNEL"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                break
            return value
    raise RuntimeError(f"Could not read ALL_KERNEL from {config_path}")


def source_tree_sha256(app_dir):
    """Hash the translation unit and local headers in deterministic order."""
    digest = hashlib.sha256()
    source_files = sorted(
        path for path in app_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".h", ".hpp"}
    )
    if not source_files:
        raise RuntimeError(f"No source files found in {app_dir}")
    for path in source_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def tcl_braced(value):
    text = str(value)
    if "}" in text or "{" in text:
        raise ValueError(f"Unsupported brace in Tcl path: {text}")
    return "{" + text + "}"


def xml_values(root):
    values = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if element.text and element.text.strip():
            values.setdefault(tag, element.text.strip())
    return values


def as_number(values, names):
    for name in names:
        if name in values:
            text = values[name].replace(",", "").strip()
            if text not in ("", "N/A", "undef"):
                return float(text)
    raise KeyError(f"None of the XML fields is present: {names}")


def parse_csynth_report(report, clock_period_ns):
    root = ET.parse(report).getroot()
    values = xml_values(root)
    cycles = as_number(
        values,
        ("Worst-caseLatency", "WorstCaseLatency", "Average-caseLatency"),
    )

    area_node = next(
        (
            element
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "AreaEstimates"
        ),
        None,
    )
    if area_node is None:
        raise RuntimeError(f"No AreaEstimates in {report}")
    children = {
        child.tag.rsplit("}", 1)[-1]: xml_values(child)
        for child in area_node
    }
    resources = children.get("Resources", {})
    available = children.get("AvailableResources", {})

    def utilization(resource_names):
        for resource_name in resource_names:
            if resource_name in resources and resource_name in available:
                used = float(resources[resource_name].replace(",", ""))
                capacity = float(available[resource_name].replace(",", ""))
                if capacity > 0:
                    return 100.0 * used / capacity
        raise RuntimeError(
            f"Missing resource/capacity fields {resource_names} in {report}"
        )

    bram = utilization(("BRAM_18K", "BRAM"))
    dsp = utilization(("DSP48E", "DSP"))
    ff = utilization(("FF",))
    lut = utilization(("LUT",))
    floored = [max(value, 1.0) for value in (bram, dsp, ff, lut)]
    return {
        "baseline_latency_ms": cycles * float(clock_period_ns) / 1_000_000.0,
        "baseline_area_score": sum(floored) / 4.0,
        "bram_util_percent": bram,
        "dsp_util_percent": dsp,
        "ff_util_percent": ff,
        "lut_util_percent": lut,
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    args = parse_args()
    repo = args.repo_root.resolve()
    excluded = comma_set(args.exclude_kernels)
    if args.kernels:
        kernels = sorted(comma_set(args.kernels))
    else:
        kernels = configured_kernels(repo / "GNN_branch" / "config.py")
    requested_locked = set(kernels) & excluded
    if requested_locked and not args.allow_test_kernels:
        kernels = [kernel for kernel in kernels if kernel not in excluded]
        print("Locked kernels excluded: " + ", ".join(sorted(requested_locked)))
    elif requested_locked:
        print("WARNING: final test-kernel synthesis explicitly unlocked")
    if args.limit is not None:
        kernels = kernels[: args.limit]

    with args.application_information.open(newline="", encoding="utf-8") as handle:
        metadata = {row["app_name"]: row for row in csv.DictReader(handle)}
    missing = set(kernels) - set(metadata)
    if missing:
        raise RuntimeError("Missing application metadata: " + ", ".join(sorted(missing)))

    try:
        version_result = subprocess.run(
            [args.vitis_hls, "-version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot run {args.vitis_hls!r}: {error}") from error
    toolchain_id = " ".join(version_result.stdout.split())[:500]
    if args.expected_tool_version not in toolchain_id:
        raise RuntimeError(
            f"Expected Vitis HLS {args.expected_tool_version}, got: {toolchain_id}"
        )

    existing = {}
    if args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as handle:
            existing = {row["kernel"]: row for row in csv.DictReader(handle)}
    requested = set(kernels)
    # A final one-kernel invocation appends to the development manifest instead
    # of deleting its already authenticated rows.
    rows = [
        row for kernel, row in sorted(existing.items())
        if kernel not in requested
    ]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    for position, kernel in enumerate(kernels, start=1):
        if (
            not args.rerun_successes
            and existing.get(kernel, {}).get("status") == "success"
        ):
            rows.append(existing[kernel])
            print(f"[{position}/{len(kernels)}] {kernel}: resume success")
            continue

        info = metadata[kernel]
        app_dir = repo / "Data" / "ApplicationDataset" / kernel
        source = app_dir / info["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        source_text = source.read_text(encoding="utf-8", errors="replace")
        if SEARCH_DIRECTIVE.search(source_text):
            raise RuntimeError(
                f"{source} already contains a searched HLS directive; it is not neutral"
            )

        kernel_work = args.work_dir.resolve() / kernel
        project = kernel_work / "project"
        solution = "neutral"
        script = kernel_work / "run.tcl"
        kernel_work.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "\n".join(
                (
                    f"open_project -reset {tcl_braced(project)}",
                    f"set_top {tcl_braced(info['top_level_function'])}",
                    f"add_files {tcl_braced(source.resolve())} -cflags {tcl_braced('-I' + str(app_dir.resolve()))}",
                    f"open_solution -reset {tcl_braced(solution)} -flow_target vivado",
                    f"set_part {tcl_braced(args.device)}",
                    f"create_clock -period {args.clock_period_ns} -name default",
                    "csynth_design",
                    "exit",
                    "",
                )
            ),
            encoding="utf-8",
        )
        log = kernel_work / "vitis_hls.log"
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "kernel": kernel,
                "status": "failed",
                "source_sha256": source_tree_sha256(app_dir),
                "toolchain_id": toolchain_id,
                "device": args.device,
                "clock_period_ns": args.clock_period_ns,
            }
        )
        print(f"[{position}/{len(kernels)}] {kernel}: synthesizing")
        try:
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [args.vitis_hls, "-f", str(script)],
                    cwd=kernel_work,
                    check=True,
                    text=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            reports = sorted(project.glob(f"{solution}/syn/report/*_csynth.xml"))
            if len(reports) != 1:
                raise RuntimeError(
                    f"Expected one csynth XML report, found {len(reports)}"
                )
            row.update(parse_csynth_report(reports[0], args.clock_period_ns))
            try:
                row["report_path"] = str(reports[0].resolve().relative_to(repo))
            except ValueError:
                row["report_path"] = str(reports[0].resolve())
            row["status"] = "success"
        except Exception as error:  # Persist failure provenance and continue.
            row["error"] = f"{type(error).__name__}: {error}"[:1000]
            print(f"  FAILED: {row['error']}")
        rows.append(row)
        write_manifest(args.output, rows)

    write_manifest(args.output, rows)
    failures = [row["kernel"] for row in rows if row["status"] != "success"]
    print(f"Manifest: {args.output.resolve()}")
    print(f"Successful baselines: {len(rows) - len(failures)}/{len(rows)}")
    if failures:
        raise SystemExit("Failed kernels: " + ", ".join(failures))


if __name__ == "__main__":
    main()
