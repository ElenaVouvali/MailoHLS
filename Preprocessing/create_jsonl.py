#!/usr/bin/env python3
"""Build deterministic target-aware SFT examples from preprocessed QoR tables.

The input tables are produced by "data_preprocess.py --mode llm".  Each JSONL
record contains:

* a key for the labeled source template stored once in a companion file;
* a complete directive assignment (including explicit no-directive values);
* the measured device, clock period, QoR, and resource utilization; and
* provenance fields used by the SFT selector and audit manifest.

This script selects no optimization objective and no winning design point.
Those choices are made after the train/validation/test split by the SFT trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_DIR = (
    REPOSITORY_ROOT / "LLM_branch" / "Data" / "preprocessed_CSVS"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "llm" / "mailohls_sft.jsonl"
APPLICATION_DIR = REPOSITORY_ROOT / "Data" / "ApplicationDataset"
APL_DIR = REPOSITORY_ROOT / "Data" / "ApplicationAPLMapping"
APPLICATION_TABLE = REPOSITORY_ROOT / "Data" / "ApplicationInformation.csv"

ACTION_ID_RE = re.compile(r"^L[1-9][0-9]*$")
LOOP_RE = re.compile(r"^(pipeline|unroll)_([1-9][0-9]*)$")
PARTIAL_ARRAY_RE = re.compile(r"^(block|cyclic)_([1-9][0-9]*)_([1-9][0-9]*)$")
COMPLETE_ARRAY_RE = re.compile(r"^complete_([1-9][0-9]*)$")

UTILIZATION_FIELDS = {
    "BRAM_Utilization_percentage": "bram_util_%",
    "DSP_Utilization_percentage": "dsp_util_%",
    "FF_Utilization_percentage": "ff_util_%",
    "LUT_Utilization_percentage": "lut_util_%",
}
REQUIRED_TABLE_COLUMNS = {
    "Device",
    "Clock_Period_nsec",
    "Latency_msec",
    "Area",
    "Weight",
    "is_pareto",
    *UTILIZATION_FIELDS,
}


@dataclass(frozen=True)
class Action:
    """One source action from ``kernel_info.txt`` and its CSV column."""

    action_id: str
    kind: str
    csv_column: str | None
    trip_count: int | None = None
    dimensions: frozenset[int] = frozenset()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preprocessing_manifest(preprocessed_dir: Path) -> Path:
    """Reject incomplete, non-LLM, or stale preprocessed table collections."""
    manifest_path = preprocessed_dir / "preprocessing_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mailohls-qor-preprocessing-v1":
        raise ValueError(f"Unsupported preprocessing schema: {manifest_path}")
    if payload.get("mode") != "llm":
        raise ValueError(f"SFT construction requires an LLM-mode manifest: {manifest_path}")

    records = payload.get("kernels")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No kernels recorded in {manifest_path}")
    expected = {str(record.get("kernel", "")) for record in records}
    observed = {
        kernel_name_from_preprocessed_path(path)
        for path in preprocessed_dir.glob("preprocessed-*.csv")
    }
    if "" in expected or expected != observed:
        raise ValueError(
            f"Preprocessing manifest/table mismatch: expected={len(expected)}, "
            f"observed={len(observed)}"
        )
    for record in records:
        table = preprocessed_dir / f"preprocessed-{record['kernel']}.csv"
        if record.get("output_sha256") != file_sha256(table):
            raise ValueError(f"Preprocessed table hash mismatch: {table}")
    return manifest_path


def repository_path(path: Path) -> str:
    """Use portable repository-relative provenance whenever possible."""
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def application_sources(required_kernels: set[str]) -> dict[str, Path]:
    table = pd.read_csv(APPLICATION_TABLE, dtype=str).fillna("")
    required = {"app_name", "file_name", "file_name_extension"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{APPLICATION_TABLE} is missing columns: {missing}")

    result: dict[str, Path] = {}
    for record in table.to_dict("records"):
        kernel = record["app_name"].strip()
        filename = record["file_name"].strip()
        extension = record["file_name_extension"].strip().lstrip(".")
        if not Path(filename).suffix and extension:
            filename = f"{filename}.{extension}"
        if kernel not in required_kernels:
            continue
        path = APPLICATION_DIR / kernel / filename
        if kernel in result:
            raise ValueError(f"Duplicate kernel in {APPLICATION_TABLE}: {kernel}")
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found for {kernel}: {path}")
        result[kernel] = path
    return result


def _records(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append([field.strip() for field in line.split(",")])
    return rows


def load_actions(kernel: str, source_text: str) -> list[Action]:
    """Join kernel actions with APL columns without guessing missing mappings."""
    info_path = APPLICATION_DIR / kernel / "kernel_info.txt"
    mapping_path = APL_DIR / f"{kernel}.txt"
    info_rows = _records(info_path)
    if not info_rows:
        raise ValueError(f"Empty action manifest: {info_path}")

    definitions: dict[str, tuple[str, int | None, frozenset[int]]] = {}
    order: list[str] = []
    for fields in info_rows[1:]:
        if len(fields) < 3 or not ACTION_ID_RE.fullmatch(fields[0]):
            raise ValueError(f"Malformed action in {info_path}: {fields}")
        action_id, kind = fields[0], fields[1].lower()
        if action_id in definitions:
            raise ValueError(f"Duplicate action {kernel}:{action_id}")
        if not re.search(rf"\b{re.escape(action_id)}\s*:", source_text):
            raise ValueError(f"Source marker {kernel}:{action_id} is missing")

        if kind == "loop":
            trip_count = int(fields[2])
            if trip_count <= 0:
                raise ValueError(f"Invalid trip count for {kernel}:{action_id}")
            definition = (kind, trip_count, frozenset())
        elif kind == "array":
            if len(fields) < 5 or (len(fields) - 3) % 2:
                raise ValueError(f"Malformed array action in {info_path}: {fields}")
            dimensions = frozenset(int(fields[index]) for index in range(3, len(fields), 2))
            if not dimensions or any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"Invalid dimensions for {kernel}:{action_id}")
            definition = (kind, None, dimensions)
        else:
            raise ValueError(f"Unsupported action kind {kind!r} in {info_path}")
        definitions[action_id] = definition
        order.append(action_id)

    action_to_column: dict[str, str] = {}
    for fields in _records(mapping_path):
        if len(fields) != 2 or not ACTION_ID_RE.fullmatch(fields[1]):
            raise ValueError(f"Malformed mapping in {mapping_path}: {fields}")
        column, action_id = fields
        if action_id not in definitions:
            # Some historical APL files retain inactive columns for actions no
            # longer declared in kernel_info.txt.  data_preprocess.py already
            # rejects such a column when it is active; it is irrelevant here.
            continue
        if action_id in action_to_column:
            raise ValueError(f"Multiple CSV columns map to {kernel}:{action_id}")
        expected_kind = "array" if column.lower().startswith("array") else "loop"
        if definitions[action_id][0] != expected_kind:
            raise ValueError(f"Action kind mismatch for {kernel}:{action_id}")
        action_to_column[action_id] = column

    actions: list[Action] = []
    for action_id in order:
        kind, trip_count, dimensions = definitions[action_id]
        actions.append(Action(
            action_id=action_id,
            kind=kind,
            csv_column=action_to_column.get(action_id),
            trip_count=trip_count,
            dimensions=dimensions,
        ))
    return actions


def source_with_placeholders(source_path: Path) -> str:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from GNN_branch.insert_placeholders import insert_placeholders

    return "".join(insert_placeholders(str(source_path))).strip()


def cell_text(row: pd.Series, column: str | None) -> str:
    if not column or column not in row.index or pd.isna(row[column]):
        return ""
    value = str(row[column]).strip().lower()
    return "" if value in {"", "ndir", "none", "auto"} else value


def action_assignments(action: Action, value: str) -> list[str]:
    """Return a complete deterministic assignment for one action."""
    if action.kind == "loop":
        pipeline_ii = unroll_factor = 0
        if value:
            match = LOOP_RE.fullmatch(value)
            if not match:
                raise ValueError(f"Invalid loop directive {value!r} for {action.action_id}")
            amount = int(match.group(2))
            if match.group(1) == "pipeline":
                pipeline_ii = amount
            else:
                if amount > int(action.trip_count or 0):
                    raise ValueError(f"Unroll exceeds trip count for {action.action_id}")
                unroll_factor = amount
        return [
            f"auto{{_PIPE_{action.action_id}}} = {pipeline_ii}",
            f"auto{{_UNROLL_{action.action_id}}} = {unroll_factor}",
        ]

    partition_type, factor, dimension = "none", 0, 0
    if value:
        partial = PARTIAL_ARRAY_RE.fullmatch(value)
        complete = COMPLETE_ARRAY_RE.fullmatch(value)
        if partial:
            partition_type = partial.group(1)
            factor, dimension = int(partial.group(2)), int(partial.group(3))
        elif complete:
            partition_type, dimension = "complete", int(complete.group(1))
        else:
            raise ValueError(f"Invalid array directive {value!r} for {action.action_id}")
        if dimension not in action.dimensions:
            raise ValueError(f"Invalid partition dimension for {action.action_id}")
    return [
        f"auto{{_ARRAY_T_{action.action_id}}} = {partition_type}",
        f"auto{{_ARRAY_F_{action.action_id}}} = {factor}",
        f"auto{{_ARRAY_D_{action.action_id}}} = {dimension}",
    ]


def finite_positive(row: pd.Series, field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def build_examples(
    preprocessed_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    tables = sorted(preprocessed_dir.glob("preprocessed-*.csv"))
    if not tables:
        raise FileNotFoundError(f"No preprocessed CSV files in {preprocessed_dir}")
    table_kernels = {
        kernel_name_from_preprocessed_path(path)
        for path in tables
    }
    sources = application_sources(table_kernels)

    examples: list[dict[str, Any]] = []
    kernel_stats: list[dict[str, Any]] = []
    source_templates: dict[str, str] = {}
    for table_path in tables:
        kernel = kernel_name_from_preprocessed_path(table_path)
        source_path = sources.get(kernel)
        if source_path is None:
            raise ValueError(f"{kernel} is absent from {APPLICATION_TABLE}")
        source_text = source_path.read_text(encoding="utf-8", errors="strict")
        actions = load_actions(kernel, source_text)
        template = source_with_placeholders(source_path)
        source_templates[kernel] = template

        for action in actions:
            required = (
                [f"auto{{_PIPE_{action.action_id}}}", f"auto{{_UNROLL_{action.action_id}}}"]
                if action.kind == "loop"
                else [
                    f"auto{{_ARRAY_T_{action.action_id}}}",
                    f"auto{{_ARRAY_F_{action.action_id}}}",
                    f"auto{{_ARRAY_D_{action.action_id}}}",
                ]
            )
            missing = [placeholder for placeholder in required if placeholder not in template]
            if missing:
                raise ValueError(f"{kernel} source placeholder(s) missing: {missing}")

        frame = pd.read_csv(table_path, dtype={"Device": str})
        missing_columns = sorted(REQUIRED_TABLE_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{table_path} is missing columns: {missing_columns}")
        target_counts: Counter[tuple[str, float]] = Counter()
        for row_index, row in frame.iterrows():
            target = "\n".join(
                line
                for action in actions
                for line in action_assignments(action, cell_text(row, action.csv_column))
            )
            device = str(row["Device"]).strip()
            clock = finite_positive(row, "Clock_Period_nsec")
            record: dict[str, Any] = {
                "schema": "mailohls-sft-example-v2-compact-source",
                "kernel_name": kernel,
                "source_key": kernel,
                "source_file": repository_path(source_path),
                "target": target,
                "device": device,
                "clock_period": clock,
                "latency": finite_positive(row, "Latency_msec"),
                "area": finite_positive(row, "Area"),
                "weight": finite_positive(row, "Weight"),
                "is_pareto": boolean(row["is_pareto"]),
                "preprocessed_row": int(row_index),
            }
            for source_field, output_field in UTILIZATION_FIELDS.items():
                utilization = float(row[source_field])
                if not math.isfinite(utilization) or utilization < 0.0:
                    raise ValueError(f"Invalid {source_field} in {table_path}")
                record[output_field] = utilization
            examples.append(record)
            target_counts[(device, clock)] += 1

        kernel_stats.append({
            "kernel": kernel,
            "rows": int(len(frame)),
            "actions": len(actions),
            "source": repository_path(source_path),
            "source_sha256": file_sha256(source_path),
            "table": repository_path(table_path),
            "table_sha256": file_sha256(table_path),
            "target_rows": [
                {"device": device, "clock_period": clock, "rows": count}
                for (device, clock), count in sorted(target_counts.items())
            ],
        })
    return examples, kernel_stats, source_templates


def kernel_name_from_preprocessed_path(path: Path) -> str:
    """Extract the kernel name from preprocessed-<kernel>.csv."""
    prefix = "preprocessed-"
    suffix = ".csv"
    filename = path.name

    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Unexpected preprocessed table name: {filename}")

    return filename[len(prefix):-len(suffix)]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output = arguments.output
    manifest_path = output.with_suffix(".manifest.json")
    sources_path = output.with_suffix(".sources.json")
    for path in (output, manifest_path, sources_path):
        if path.exists() and not arguments.force:
            raise SystemExit(f"Refusing to overwrite {path}; pass --force")

    input_manifest = validate_preprocessing_manifest(arguments.preprocessed_dir)
    examples, kernel_stats, source_templates = build_examples(arguments.preprocessed_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps({
        "schema": "mailohls-sft-sources-v1",
        "templates": source_templates,
    }, indent=2, sort_keys=True) + "\n")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(output)

    manifest = {
        "schema": "mailohls-sft-jsonl-manifest-v2-compact-source",
        "examples": len(examples),
        "kernels": len(kernel_stats),
        "output": repository_path(output),
        "output_sha256": file_sha256(output),
        "sources": repository_path(sources_path),
        "sources_sha256": file_sha256(sources_path),
        "preprocessing_manifest": repository_path(input_manifest),
        "preprocessing_manifest_sha256": file_sha256(input_manifest),
        "kernel_statistics": kernel_stats,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(examples)} examples for {len(kernel_stats)} kernels")
    print(f"JSONL:    {output}")
    print(f"Sources:  {sources_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())