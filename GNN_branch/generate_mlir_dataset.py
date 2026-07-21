#!/usr/bin/env python3
"""
Generate and strictly validate the complete MailoHLS MLIR GEXF dataset.

The authoritative kernel set is read from GNN_branch/config.py::ALL_KERNEL.
Source filenames and top-level functions are read from
Data/ApplicationInformation.csv.

Each kernel is processed by GNN_branch/mlir_graph_gen.py:

    C/C++ -> cgeist/Polygeist MLIR -> validated deterministic GEXF

A graph is reusable only when it matches the current:

  * MailoHLS MLIR graph schema;
  * top-level function;
  * complete kernel_info.txt action set;
  * source-file SHA-256;
  * kernel_info.txt SHA-256;
  * cgeist binary SHA-256;
  * strict action-mapping policy (no fallback resolutions).

Generation is atomic: a new graph is first written to a temporary path,
strictly validated, and only then replaces the previous graph. Therefore a
failed forced regeneration never destroys a previously valid graph.

Examples
--------
From the repository root:

    PYTHONHASHSEED=0 \
    PYTHONPATH="$MLIR_PYTHON_ROOT" \
    "$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py

Regenerate everything:

    PYTHONHASHSEED=0 \
    PYTHONPATH="$MLIR_PYTHON_ROOT" \
    "$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
      --force --keep-mlir --continue-on-error

Test selected kernels:

    PYTHONHASHSEED=0 \
    PYTHONPATH="$MLIR_PYTHON_ROOT" \
    "$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
      --force --keep-mlir --continue-on-error \
      --only machsuite-viterbi rodinia-knn-5-coalescing
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


EXPECTED_SCHEMA_VERSION = "mailohls-mlir-graph"
GRAPH_METADATA_PREFIX = "mailohls-meta-v1:"
ACTION_ID_RE = re.compile(r"^L[1-9][0-9]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class KernelRow:
    app_name: str
    top_level_function: str
    file_name: str
    file_name_extension: str


@dataclass(frozen=True)
class KernelContract:
    top_level_function: str
    action_ids: frozenset[str]
    source_sha256: str
    action_sha256: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_executable(requested: str) -> Path:
    """Resolve cgeist once so generation and validation use the same binary."""
    name = requested.strip() or os.environ.get("CGEIST", "").strip() or "cgeist"
    expanded = Path(name).expanduser()

    if expanded.is_absolute() or expanded.parent != Path("."):
        if not expanded.is_file():
            raise FileNotFoundError(f"cgeist was not found at {expanded}")
        if not os.access(expanded, os.X_OK):
            raise PermissionError(f"cgeist is not executable: {expanded}")
        return expanded.resolve()

    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {name!r} on PATH. "
            "Pass --cgeist /absolute/path/to/cgeist or export CGEIST."
        )
    return Path(found).resolve()


def load_all_kernel_names(config_path: Path) -> list[str]:
    """Parse ALL_KERNEL without importing config.py."""
    tree = ast.parse(
        config_path.read_text(encoding="utf-8"),
        filename=str(config_path),
    )

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != "ALL_KERNEL":
                continue

            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise RuntimeError(
                    "GNN_branch/config.py::ALL_KERNEL is not a list[str]."
                )
            if len(value) != len(set(value)):
                raise RuntimeError(
                    "GNN_branch/config.py::ALL_KERNEL contains duplicates."
                )
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
            raise RuntimeError(
                f"{csv_path} is missing columns: {sorted(missing)}"
            )

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
                raise RuntimeError(
                    f"Duplicate app_name in {csv_path}: {row.app_name}"
                )
            if not row.top_level_function:
                raise RuntimeError(
                    f"{csv_path}: {row.app_name} has no top_level_function"
                )
            if not row.file_name:
                raise RuntimeError(
                    f"{csv_path}: {row.app_name} has no file_name"
                )

            rows[row.app_name] = row

    return rows


def load_kernel_contract(
    source: Path,
    kernel_info: Path,
    expected_top: str,
) -> KernelContract:
    """Read and validate the exact source/action contract."""
    if not source.is_file():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not kernel_info.is_file():
        raise FileNotFoundError(f"kernel_info.txt does not exist: {kernel_info}")

    lines = [
        line.strip()
        for line in kernel_info.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"Empty kernel metadata: {kernel_info}")

    metadata_top = lines[0]
    if metadata_top != expected_top:
        raise RuntimeError(
            f"Top-function mismatch for {source.parent.name}: "
            f"ApplicationInformation.csv={expected_top!r}, "
            f"kernel_info.txt={metadata_top!r}"
        )

    action_ids: list[str] = []
    for line_number, line in enumerate(lines[1:], start=2):
        action_id = line.split(",", 1)[0].strip()
        if not ACTION_ID_RE.fullmatch(action_id):
            raise RuntimeError(
                f"{kernel_info}:{line_number}: invalid action id "
                f"{action_id!r}; expected L1, L2, ..."
            )
        action_ids.append(action_id)

    if not action_ids:
        raise RuntimeError(f"{kernel_info} defines no MailoHLS actions")

    duplicates = sorted(
        action_id
        for action_id in set(action_ids)
        if action_ids.count(action_id) > 1
    )
    if duplicates:
        raise RuntimeError(
            f"{kernel_info} contains duplicate action ids: {duplicates}"
        )

    return KernelContract(
        top_level_function=metadata_top,
        action_ids=frozenset(action_ids),
        source_sha256=sha256_file(source),
        action_sha256=sha256_file(kernel_info),
    )


def graph_action_ids(graph: nx.MultiDiGraph) -> set[str]:
    return {
        str(data.get("action_id", "")).strip()
        for _, data in graph.nodes(data=True)
        if str(data.get("action_id", "")).strip()
    }


def graph_fallback_resolutions(graph: nx.MultiDiGraph) -> set[str]:
    return {
        str(data.get("action_resolution", "")).strip()
        for _, data in graph.nodes(data=True)
        if "fallback"
        in str(data.get("action_resolution", "")).strip().lower()
    }


def read_graph_metadata(graph: nx.MultiDiGraph) -> tuple[dict[str, object] | None, str]:
    """Decode the deterministic metadata envelope stored in the GEXF name."""
    raw_name = str(graph.graph.get("name", "")).strip()

    if not raw_name:
        return None, "GEXF has no MailoHLS metadata envelope"

    if not raw_name.startswith(GRAPH_METADATA_PREFIX):
        return (
            None,
            "GEXF graph name does not start with "
            f"{GRAPH_METADATA_PREFIX!r}",
        )

    payload = raw_name[len(GRAPH_METADATA_PREFIX):]
    try:
        metadata = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, f"invalid MailoHLS metadata JSON: {exc}"

    if not isinstance(metadata, dict):
        return None, "MailoHLS metadata envelope is not a JSON object"

    return metadata, ""


def _metadata_string(
    metadata: dict[str, object],
    key: str,
) -> str:
    return str(metadata.get(key, "")).strip()


def validate_existing_graph(
    path: Path,
    contract: KernelContract,
    expected_cgeist_sha256: str,
    expected_generator_sha256: str,
) -> tuple[bool, str]:
    """Return whether a GEXF is safe to reuse as current training data."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty"

    try:
        graph = nx.read_gexf(path)
    except Exception as exc:
        return False, f"unreadable: {exc}"

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return False, "empty graph"

    metadata, metadata_error = read_graph_metadata(graph)
    if metadata is None:
        return False, metadata_error

    schema = _metadata_string(metadata, "schema_version")
    if schema != EXPECTED_SCHEMA_VERSION:
        return (
            False,
            f"schema_version={schema!r}, "
            f"expected {EXPECTED_SCHEMA_VERSION!r}",
        )

    kernel = _metadata_string(metadata, "kernel")
    if kernel != contract.top_level_function:
        return (
            False,
            f"graph kernel={kernel!r}, "
            f"expected {contract.top_level_function!r}",
        )

    actual_actions = graph_action_ids(graph)
    expected_actions = set(contract.action_ids)
    if actual_actions != expected_actions:
        missing = sorted(expected_actions - actual_actions)
        unexpected = sorted(actual_actions - expected_actions)
        return (
            False,
            "action-set mismatch: "
            f"missing={missing}, unexpected={unexpected}",
        )

    fallback_resolutions = graph_fallback_resolutions(graph)
    if fallback_resolutions:
        return (
            False,
            "contains non-exact action mappings: "
            f"{sorted(fallback_resolutions)}",
        )

    metadata_resolutions = metadata.get("action_resolutions", {})
    if not isinstance(metadata_resolutions, dict):
        return False, "metadata action_resolutions is not an object"

    normalized_resolutions: dict[str, int] = {}
    try:
        for name, count in metadata_resolutions.items():
            normalized_resolutions[str(name)] = int(count)
    except (TypeError, ValueError) as exc:
        return False, f"invalid action_resolutions counts: {exc}"

    metadata_fallbacks = {
        name
        for name, count in normalized_resolutions.items()
        if "fallback" in name.lower() and count > 0
    }
    if metadata_fallbacks:
        return (
            False,
            "metadata contains non-exact action mappings: "
            f"{sorted(metadata_fallbacks)}",
        )

    mapped_action_count = sum(normalized_resolutions.values())
    if mapped_action_count != len(expected_actions):
        return (
            False,
            "action-resolution count mismatch: "
            f"metadata={mapped_action_count}, "
            f"expected={len(expected_actions)}",
        )

    source_sha256 = _metadata_string(metadata, "source_sha256")
    if source_sha256 != contract.source_sha256:
        return (
            False,
            "source_sha256 mismatch: graph was generated from a different "
            "source revision",
        )

    action_sha256 = _metadata_string(metadata, "action_sha256")
    if action_sha256 != contract.action_sha256:
        return (
            False,
            "action_sha256 mismatch: graph was generated from a different "
            "kernel_info.txt revision",
        )

    cgeist_sha256 = _metadata_string(metadata, "cgeist_sha256")
    if cgeist_sha256 != expected_cgeist_sha256:
        return (
            False,
            "cgeist_sha256 mismatch: graph was generated with a different "
            "Polygeist frontend binary",
        )

    generator_sha256 = _metadata_string(metadata, "generator_sha256")
    if generator_sha256 != expected_generator_sha256:
        return (
            False,
            "generator_sha256 mismatch: graph was generated with a different "
            "mlir_graph_gen.py revision",
        )

    mlir_sha256 = _metadata_string(metadata, "mlir_sha256")
    if not SHA256_RE.fullmatch(mlir_sha256):
        return False, "metadata contains an invalid or missing mlir_sha256"

    frontend_policy = _metadata_string(metadata, "frontend_policy")
    if not frontend_policy:
        return False, "metadata contains no frontend_policy"

    return (
        True,
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, "
        f"{len(actual_actions)} actions",
    )


def build_command(
    *,
    python_executable: str,
    generator: Path,
    source: Path,
    output: Path,
    cgeist: Path,
    mlir_output: Path | None,
    cflags: list[str],
) -> list[str]:
    command = [
        python_executable,
        str(generator),
        str(source),
        "--output",
        str(output),
        "--cgeist",
        str(cgeist),
    ]

    if mlir_output is not None:
        command += ["--mlir-output", str(mlir_output)]

    for flag in cflags:
        command += ["--cflag", flag]

    return command


def remove_if_exists(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


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

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
        help="Python with matching MLIR bindings.",
    )
    parser.add_argument(
        "--cgeist",
        default=os.environ.get("CGEIST", ""),
        help="cgeist executable; default is $CGEIST, then cgeist on PATH.",
    )
    parser.add_argument(
        "--cflag",
        action="append",
        default=[],
        help="Forward one additional flag to mlir_graph_gen.py.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="KERNEL",
        help="Generate only these configured kernel directory names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a strictly valid graph already exists.",
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


def make_record(
    *,
    app_name: str,
    status: str,
    source: Path,
    output: Path,
    seconds: float,
    detail: str,
    command: list[str] | None,
) -> dict[str, object]:
    return {
        "app_name": app_name,
        "status": status,
        "source": str(source),
        "output": str(output),
        "seconds": f"{seconds:.3f}",
        "detail": detail,
        "command": subprocess.list2cmdline(command or []),
    }


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
            raise FileNotFoundError(
                f"Required repository path does not exist: {required}"
            )

    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "PYTHONHASHSEED must be set before Python starts.\n"
            "Run: PYTHONHASHSEED=0 "
            "python GNN_branch/generate_mlir_dataset.py"
        )

    cgeist = resolve_executable(args.cgeist)
    cgeist_sha256 = sha256_file(cgeist)
    generator_sha256 = sha256_file(generator)

    all_kernels = load_all_kernel_names(config_path)
    rows = load_application_rows(app_csv)

    missing_metadata = [
        name for name in all_kernels if name not in rows
    ]
    if missing_metadata:
        raise RuntimeError(
            "The configured kernel set is missing from "
            "ApplicationInformation.csv: "
            + ", ".join(missing_metadata)
        )

    selected = all_kernels
    if args.only:
        requested = list(dict.fromkeys(args.only))
        unknown = [
            name for name in requested if name not in all_kernels
        ]
        if unknown:
            raise RuntimeError(
                "--only contains kernels not present in "
                "config.py::ALL_KERNEL: "
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
    contracts: dict[str, KernelContract] = {}

    print(f"Repository: {repo}")
    print(
        f"Kernel set: {len(selected)} of "
        f"{len(all_kernels)} configured kernels"
    )
    print(f"Output:     {output_dir}")
    print(f"Python:     {args.python}")
    print(f"cgeist:    {cgeist}")
    print(f"cgeist SHA-256:  {cgeist_sha256}")
    print(f"generator SHA-256: {generator_sha256}")
    print()

    for index, app_name in enumerate(selected, start=1):
        row = rows[app_name]
        source = dataset_root / app_name / row.file_name
        kernel_info = source.parent / "kernel_info.txt"

        output = output_dir / f"{app_name}.gexf"
        temporary_output = output_dir / f".{app_name}.gexf.tmp"
        log_path = log_dir / f"{app_name}.log"

        audit_output = (
            audit_dir / f"{app_name}.mlir"
            if audit_dir is not None
            else None
        )
        temporary_audit = (
            audit_dir / f".{app_name}.mlir.tmp"
            if audit_dir is not None
            else None
        )

        remove_if_exists(temporary_output)
        remove_if_exists(temporary_audit)

        try:
            contract = load_kernel_contract(
                source,
                kernel_info,
                row.top_level_function,
            )
            contracts[app_name] = contract
        except (
            FileNotFoundError,
            PermissionError,
            RuntimeError,
            ValueError,
        ) as exc:
            detail = str(exc)
            print(
                f"[{index:02d}/{len(selected):02d}] "
                f"FAIL {app_name}: {detail}"
            )
            failures.append(app_name)
            records.append(
                make_record(
                    app_name=app_name,
                    status="failed",
                    source=source,
                    output=output,
                    seconds=0.0,
                    detail=detail,
                    command=None,
                )
            )
            if not args.continue_on_error:
                break
            continue

        reusable, reuse_detail = validate_existing_graph(
            output,
            contract,
            cgeist_sha256,
            generator_sha256,
        )
        if reusable and not args.force:
            print(
                f"[{index:02d}/{len(selected):02d}] "
                f"SKIP {app_name}: {reuse_detail}"
            )
            records.append(
                make_record(
                    app_name=app_name,
                    status="skipped",
                    source=source,
                    output=output,
                    seconds=0.0,
                    detail=reuse_detail,
                    command=None,
                )
            )
            continue

        command = build_command(
            python_executable=args.python,
            generator=generator,
            source=source,
            output=temporary_output,
            cgeist=cgeist,
            mlir_output=temporary_audit,
            cflags=list(args.cflag),
        )

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
            remove_if_exists(temporary_output)
            remove_if_exists(temporary_audit)

            detail = (
                f"exit={completed.returncode}; log={log_path}"
            )
            print(f"  FAIL in {seconds:.2f}s: {detail}")
            failures.append(app_name)
            status = "failed"
        else:
            valid, detail = validate_existing_graph(
                temporary_output,
                contract,
                cgeist_sha256,
                generator_sha256,
            )

            if not valid:
                remove_if_exists(temporary_output)
                remove_if_exists(temporary_audit)

                failures.append(app_name)
                status = "failed"
                detail = (
                    "generator returned success but strict temporary-output "
                    f"validation failed: {detail}"
                )
                print(f"  FAIL in {seconds:.2f}s: {detail}")
            else:
                os.replace(temporary_output, output)

                if temporary_audit is not None:
                    if temporary_audit.is_file():
                        assert audit_output is not None
                        os.replace(temporary_audit, audit_output)
                    else:
                        remove_if_exists(audit_output)

                status = "generated"
                print(f"  OK   in {seconds:.2f}s: {detail}")

        records.append(
            make_record(
                app_name=app_name,
                status=status,
                source=source,
                output=output,
                seconds=seconds,
                detail=detail,
                command=command,
            )
        )

        if status == "failed" and not args.continue_on_error:
            break

    manifest = output_dir / "generation_manifest.csv"
    write_manifest(manifest, records)

    valid_outputs: list[str] = []
    invalid_outputs: list[tuple[str, str]] = []

    for app_name in selected:
        row = rows[app_name]
        source = dataset_root / app_name / row.file_name
        kernel_info = source.parent / "kernel_info.txt"

        try:
            contract = contracts.get(app_name)
            if contract is None:
                contract = load_kernel_contract(
                    source,
                    kernel_info,
                    row.top_level_function,
                )

            valid, detail = validate_existing_graph(
                output_dir / f"{app_name}.gexf",
                contract,
                cgeist_sha256,
                generator_sha256,
            )
        except (
            FileNotFoundError,
            PermissionError,
            RuntimeError,
            ValueError,
        ) as exc:
            valid = False
            detail = str(exc)

        if valid:
            valid_outputs.append(app_name)
        else:
            invalid_outputs.append((app_name, detail))

    print()
    print(f"Valid outputs: {len(valid_outputs)}/{len(selected)}")
    print(f"Manifest:      {manifest}")

    if invalid_outputs:
        print("Invalid or missing outputs:")
        for app_name, detail in invalid_outputs:
            print(f"  - {app_name}: {detail}")

    if failures:
        print(
            "Failures:      "
            + ", ".join(dict.fromkeys(failures))
        )
        return 2

    if len(valid_outputs) != len(selected):
        print(
            "Dataset is incomplete even though no subprocess "
            "failure was recorded."
        )
        return 3

    print("MLIR graph dataset is complete and strictly validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())