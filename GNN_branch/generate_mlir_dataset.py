#!/usr/bin/env python3
"""
Generate the complete 55-kernel MailoHLS MLIR GEXF dataset.

The authoritative kernel set is read from GNN_branch/config.py::ALL_KERNEL.
Source filenames and top-level functions are read from
Data/ApplicationInformation.csv.

Each kernel is processed by GNN_branch/mlir_graph_gen.py, which performs:
    C/C++ -> cgeist/Polygeist MLIR -> validated deterministic GEXF

Examples
--------
From the MailoHLS repository root:

    PYTHONHASHSEED=0 \
    python GNN_branch/generate_mlir_graphs.py

Resume an interrupted run (default: existing valid outputs are skipped):

    PYTHONHASHSEED=0 \
    python GNN_branch/generate_mlir_graphs.py

Regenerate everything:

    PYTHONHASHSEED=0 \
    python GNN_branch/generate_mlir_graphs.py --force

Test selected kernels first:

    PYTHONHASHSEED=0 \
    python GNN_branch/generate_mlir_graphs.py \
      --only rodinia-knn-1-tiling machsuite-viterbi

The cgeist executable is resolved by mlir_graph_gen.py from --cgeist,
$CGEIST, or PATH.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class KernelRow:
    app_name: str
    top_level_function: str
    file_name: str
    file_name_extension: str


def repository_root() -> Path:
    # This script is expected at <repo>/GNN_branch/generate_mlir_graphs.py.
    return Path(__file__).resolve().parents[1]


def load_all_kernel_names(config_path: Path) -> list[str]:
    """Parse ALL_KERNEL without importing config.py (which parses CLI flags)."""
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALL_KERNEL":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                        raise RuntimeError("GNN_branch/config.py::ALL_KERNEL is not a list[str].")
                    if len(value) != len(set(value)):
                        raise RuntimeError("GNN_branch/config.py::ALL_KERNEL contains duplicates.")
                    return value
    raise RuntimeError(f"Could not find ALL_KERNEL in {config_path}")


def load_application_rows(csv_path: Path) -> dict[str, KernelRow]:
    rows: dict[str, KernelRow] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "app_name",
            "top_level_function",
            "file_name",
            "file_name_extension",
        }
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_path} is missing columns: {sorted(missing)}")

        for raw in reader:
            row = KernelRow(
                app_name=raw["app_name"].strip(),
                top_level_function=raw["top_level_function"].strip(),
                file_name=raw["file_name"].strip(),
                file_name_extension=raw["file_name_extension"].strip(),
            )
            if not row.app_name:
                continue
            if row.app_name in rows:
                raise RuntimeError(f"Duplicate app_name in {csv_path}: {row.app_name}")
            rows[row.app_name] = row
    return rows


def validate_existing_graph(path: Path, expected_kernel: str) -> tuple[bool, str]:
    """Return whether an existing GEXF is safe to reuse."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        graph = nx.read_gexf(path)
    except Exception as exc:
        return False, f"unreadable: {exc}"

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return False, "empty graph"

    action_ids = {
        str(data.get("action_id", "")).strip()
        for _, data in graph.nodes(data=True)
        if str(data.get("action_id", "")).strip()
    }
    if not action_ids:
        return False, "no action_id nodes"

    kernel = str(graph.graph.get("kernel", "")).strip()
    if kernel and kernel != expected_kernel:
        return False, f"graph kernel={kernel!r}, expected {expected_kernel!r}"

    return True, f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(action_ids)} actions"


def build_command(
    *,
    python_executable: str,
    generator: Path,
    source: Path,
    output: Path,
    cgeist: str | None,
    mlir_audit_dir: Path | None,
) -> list[str]:
    command = [
        python_executable,
        str(generator),
        str(source),
        "--output",
        str(output),
    ]
    if cgeist:
        command += ["--cgeist", cgeist]
    if mlir_audit_dir is not None:
        command += [
            "--mlir-output",
            str(mlir_audit_dir / f"{output.stem}.mlir"),
        ]
    return command


def write_manifest(
    path: Path,
    records: Iterable[dict[str, object]],
) -> None:
    records = list(records)
    fields = [
        "app_name",
        "status",
        "source",
        "output",
        "seconds",
        "detail",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repository_root(),
        help="MailoHLS repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/GNN_branch/MLIR_graphs",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python with matching MLIR bindings (default: current interpreter).",
    )
    parser.add_argument(
        "--cgeist",
        default=os.environ.get("CGEIST", ""),
        help="cgeist executable; default is $CGEIST, then mlir_graph_gen.py resolves PATH.",
    )
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="KERNEL",
        help="Generate only these kernel directory names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even when a valid GEXF already exists.",
    )
    parser.add_argument(
        "--keep-mlir",
        action="store_true",
        help="Preserve emitted MLIR under <output-dir>/audit_mlir.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Attempt remaining kernels after a failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else repo / "GNN_branch" / "MLIR_graphs"
    )
    generator = repo / "GNN_branch" / "mlir_graph_gen.py"
    config_path = repo / "GNN_branch" / "config.py"
    app_csv = repo / "Data" / "ApplicationInformation.csv"
    dataset_root = repo / "Data" / "ApplicationDataset"

    for required in (generator, config_path, app_csv, dataset_root):
        if not required.exists():
            raise FileNotFoundError(f"Required repository path does not exist: {required}")

    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "PYTHONHASHSEED must be set before Python starts.\n"
            "Run: PYTHONHASHSEED=0 python GNN_branch/generate_mlir_graphs.py"
        )

    all_kernels = load_all_kernel_names(config_path)
    rows = load_application_rows(app_csv)

    missing_metadata = [name for name in all_kernels if name not in rows]
    if missing_metadata:
        raise RuntimeError(
            "The 55-kernel config set is missing from ApplicationInformation.csv: "
            + ", ".join(missing_metadata)
        )

    selected = all_kernels
    if args.only:
        requested = list(dict.fromkeys(args.only))
        unknown = [name for name in requested if name not in all_kernels]
        if unknown:
            raise RuntimeError(
                "--only contains kernels not present in config.py::ALL_KERNEL: "
                + ", ".join(unknown)
            )
        selected = requested

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit_mlir" if args.keep_mlir else None
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"Repository: {repo}")
    print(f"Kernel set: {len(selected)} of {len(all_kernels)} configured kernels")
    print(f"Output:     {output_dir}")
    print(f"Python:     {args.python}")
    print(f"cgeist:    {args.cgeist or '$CGEIST / PATH'}")
    print()

    for index, app_name in enumerate(selected, start=1):
        row = rows[app_name]
        source = dataset_root / app_name / row.file_name
        output = output_dir / f"{app_name}.gexf"
        log_path = log_dir / f"{app_name}.log"

        if not source.is_file():
            detail = f"source missing: {source}"
            print(f"[{index:02d}/{len(selected):02d}] FAIL {app_name}: {detail}")
            failures.append(app_name)
            records.append(
                {
                    "app_name": app_name,
                    "status": "failed",
                    "source": str(source),
                    "output": str(output),
                    "seconds": "0.000",
                    "detail": detail,
                    "command": "",
                }
            )
            if not args.continue_on_error:
                break
            continue

        reusable, reuse_detail = validate_existing_graph(output, app_name)
        if reusable and not args.force:
            print(f"[{index:02d}/{len(selected):02d}] SKIP {app_name}: {reuse_detail}")
            records.append(
                {
                    "app_name": app_name,
                    "status": "skipped",
                    "source": str(source),
                    "output": str(output),
                    "seconds": "0.000",
                    "detail": reuse_detail,
                    "command": "",
                }
            )
            continue

        command = build_command(
            python_executable=args.python,
            generator=generator,
            source=source,
            output=output,
            cgeist=args.cgeist or None,
            mlir_audit_dir=audit_dir,
        )
        for flag in args.cflag:
            command += ["--cflag", flag]
        print(f"[{index:02d}/{len(selected):02d}] RUN  {app_name}")
        print("  " + subprocess.list2cmdline(command))

        start = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        seconds = time.monotonic() - start
        log_path.write_text(completed.stdout, encoding="utf-8")

        if completed.returncode != 0:
            detail = f"exit={completed.returncode}; log={log_path}"
            print(f"  FAIL in {seconds:.2f}s: {detail}")
            failures.append(app_name)
            status = "failed"
            if not args.continue_on_error:
                records.append(
                    {
                        "app_name": app_name,
                        "status": status,
                        "source": str(source),
                        "output": str(output),
                        "seconds": f"{seconds:.3f}",
                        "detail": detail,
                        "command": subprocess.list2cmdline(command),
                    }
                )
                break
        else:
            valid, detail = validate_existing_graph(output, app_name)
            if not valid:
                failures.append(app_name)
                status = "failed"
                detail = f"generator returned success but output validation failed: {detail}"
                print(f"  FAIL in {seconds:.2f}s: {detail}")
                if not args.continue_on_error:
                    records.append(
                        {
                            "app_name": app_name,
                            "status": status,
                            "source": str(source),
                            "output": str(output),
                            "seconds": f"{seconds:.3f}",
                            "detail": detail,
                            "command": subprocess.list2cmdline(command),
                        }
                    )
                    break
            else:
                status = "generated"
                print(f"  OK   in {seconds:.2f}s: {detail}")

        records.append(
            {
                "app_name": app_name,
                "status": status,
                "source": str(source),
                "output": str(output),
                "seconds": f"{seconds:.3f}",
                "detail": detail,
                "command": subprocess.list2cmdline(command),
            }
        )

    manifest = output_dir / "generation_manifest.csv"
    write_manifest(manifest, records)

    valid_outputs = []
    for app_name in selected:
        valid, _ = validate_existing_graph(output_dir / f"{app_name}.gexf", app_name)
        if valid:
            valid_outputs.append(app_name)

    print()
    print(f"Valid outputs: {len(valid_outputs)}/{len(selected)}")
    print(f"Manifest:      {manifest}")
    if failures:
        print("Failures:      " + ", ".join(dict.fromkeys(failures)))
        return 2
    if len(valid_outputs) != len(selected):
        print("Dataset is incomplete even though no subprocess failure was recorded.")
        return 3

    print("MLIR graph dataset is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())