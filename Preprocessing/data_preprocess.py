#!/usr/bin/env python3
"""Prepare MailoHLS QoR tables for either GNN or target-aware LLM training.

GNN mode can keep one FPGA target or every measured FPGA/clock target. LLM mode
keeps every measured target. Pareto weights are always computed independently
for each (device, clock-period) design space.

Both modes canonicalize equivalent directive spellings (e.g. pipeline == pipeline_1) 
and aggregate repeated measurements after canonicalization.  Action columns are accepted
when CSV -> APL -> kernel_info -> labeled source is a complete chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "Data" / "CSVS"
DEFAULT_LLM_OUTPUT_DIR = REPOSITORY_ROOT / "LLM_branch" / "Data" / "preprocessed_CSVS"
DEFAULT_GNN_OUTPUT_DIR = REPOSITORY_ROOT / "GNN_branch" / "Data" / "preprocessed_CSVS"
APPLICATION_DIR = REPOSITORY_ROOT / "Data" / "ApplicationDataset"
APL_DIR = REPOSITORY_ROOT / "Data" / "ApplicationAPLMapping"

DEFAULT_GNN_DEVICE = "xczu7ev-ffvc1156-2-e"
DEFAULT_GNN_CLOCK_NS = 10.0

UTILIZATION_COLUMNS = (
    "BRAM_Utilization_percentage",
    "DSP_Utilization_percentage",
    "FF_Utilization_percentage",
    "LUT_Utilization_percentage",
)
REQUIRED_COLUMNS = (
    "Version",
    "Device",
    "Clock_Period_nsec",
    "Latency_msec",
    *UTILIZATION_COLUMNS,
)
MEASUREMENT_COLUMNS = {
    *REQUIRED_COLUMNS,
    "Synthesis_Time",
    "Synthesis_Time_sec",
}
ACTION_ID_RE = re.compile(r"^L[1-9][0-9]*$")
PIPELINE_RE = re.compile(r"^pipeline(?:_([1-9][0-9]*))?$")
UNROLL_RE = re.compile(r"^unroll(?:_([1-9][0-9]*))?$")
PARTIAL_PARTITION_RE = re.compile(r"^(block|cyclic)_([1-9][0-9]*)_([1-9][0-9]*)$")
COMPLETE_PARTITION_RE = re.compile(r"^complete_([1-9][0-9]*)$")


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    kind: str
    trip_count: int | None = None
    array_dimensions: frozenset[int] = frozenset()


def _split_record(line: str) -> list[str]:
    return [field.strip() for field in line.split(",")]


def load_action_definitions(kernel: str) -> dict[str, ActionDefinition]:
    """Return active CSV column -> fully validated action definition."""
    kernel_dir = APPLICATION_DIR / kernel
    kernel_info = kernel_dir / "kernel_info.txt"
    apl_file = APL_DIR / f"{kernel}.txt"
    if not kernel_info.is_file() or not apl_file.is_file():
        raise FileNotFoundError(f"Missing action metadata for {kernel}")

    label_to_definition: dict[str, ActionDefinition] = {}
    lines = [line.strip() for line in kernel_info.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty action manifest: {kernel_info}")

    for line in lines[1:]:
        fields = _split_record(line)
        if len(fields) < 3 or not ACTION_ID_RE.fullmatch(fields[0]):
            raise ValueError(f"Malformed action record in {kernel_info}: {line!r}")
        action_id, kind = fields[0], fields[1].lower()
        if action_id in label_to_definition:
            raise ValueError(f"Duplicate action {action_id} in {kernel_info}")
        if kind == "loop":
            trip_count = int(fields[2])
            if trip_count <= 0:
                raise ValueError(f"Non-positive trip count for {kernel}:{action_id}")
            definition = ActionDefinition(action_id, kind, trip_count=trip_count)
        elif kind == "array":
            if len(fields) < 5 or (len(fields) - 3) % 2:
                raise ValueError(f"Malformed array record in {kernel_info}: {line!r}")
            dimensions: set[int] = set()
            for index in range(3, len(fields), 2):
                dimension, extent = int(fields[index]), int(fields[index + 1])
                if dimension <= 0 or extent <= 0 or dimension in dimensions:
                    raise ValueError(f"Invalid array dimension in {kernel_info}: {line!r}")
                dimensions.add(dimension)
            definition = ActionDefinition(
                action_id, kind, array_dimensions=frozenset(dimensions)
            )
        else:
            raise ValueError(f"Unsupported action kind in {kernel_info}: {kind!r}")
        label_to_definition[action_id] = definition

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(kernel_dir.iterdir())
        if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    )
    column_to_definition: dict[str, ActionDefinition] = {}
    for raw_line in apl_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_record(line)
        if len(fields) != 2 or not ACTION_ID_RE.fullmatch(fields[1]):
            raise ValueError(f"Malformed APL record in {apl_file}: {line!r}")
        column, action_id = fields
        if column in column_to_definition:
            raise ValueError(f"Duplicate CSV column {column!r} in {apl_file}")
        definition = label_to_definition.get(action_id)
        if definition is None:
            # Keep the incomplete entry visible so an active CSV column fails below.
            continue
        expected_kind = "array" if column.lower().startswith("array") else "loop"
        if definition.kind != expected_kind:
            raise ValueError(
                f"{kernel}:{column} maps to {action_id}, which is a "
                f"{definition.kind}, not a {expected_kind}"
            )
        if not re.search(rf"\b{re.escape(action_id)}\s*:", source_text):
            raise ValueError(f"{kernel}:{action_id} is absent from labeled source")
        column_to_definition[column] = definition
    return column_to_definition


def active_directive_columns(frame: pd.DataFrame) -> list[str]:
    """Return ordered directive columns that contain at least one directive."""
    candidates = [column for column in frame.columns if column not in MEASUREMENT_COLUMNS]
    active: list[str] = []
    for column in candidates:
        values = frame[column].fillna("").astype(str).str.strip().str.upper()
        if (values != "NDIR").any() and (values != "").any():
            active.append(column)
    return active


def canonicalize_directive(token: object, action: ActionDefinition) -> str:
    """Map equivalent CSV spellings to the representation consumed by mlir_data.py."""
    value = "" if pd.isna(token) else str(token).strip().lower()
    if value in {"", "ndir", "auto", "none"}:
        return ""

    if action.kind == "loop":
        pipeline = PIPELINE_RE.fullmatch(value)
        if pipeline:
            # Dataset convention: bare pipeline requests the ideal II=1 case.
            return f"pipeline_{int(pipeline.group(1) or 1)}"
        unroll = UNROLL_RE.fullmatch(value)
        if unroll:
            factor = int(unroll.group(1) or action.trip_count or 0)
            if factor <= 0 or factor > int(action.trip_count or 0):
                raise ValueError(
                    f"Invalid unroll factor {factor} for {action.action_id} "
                    f"with trip count {action.trip_count}"
                )
            return f"unroll_{factor}"
        raise ValueError(f"Invalid loop directive {value!r} for {action.action_id}")

    partial = PARTIAL_PARTITION_RE.fullmatch(value)
    if partial:
        kind, factor, dimension = partial.group(1), int(partial.group(2)), int(partial.group(3))
        if dimension not in action.array_dimensions:
            raise ValueError(f"Invalid partition dimension in {value!r} for {action.action_id}")
        return f"{kind}_{factor}_{dimension}"
    complete = COMPLETE_PARTITION_RE.fullmatch(value)
    if complete:
        dimension = int(complete.group(1))
        if dimension not in action.array_dimensions:
            raise ValueError(f"Invalid partition dimension in {value!r} for {action.action_id}")
        return f"complete_{dimension}"
    raise ValueError(f"Invalid array directive {value!r} for {action.action_id}")


def pareto_mask(latency: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Exact non-dominated mask for minimization of latency and area."""
    result = np.ones(len(latency), dtype=bool)
    for index in range(len(latency)):
        dominates = (
            (latency <= latency[index])
            & (area <= area[index])
            & ((latency < latency[index]) | (area < area[index]))
        )
        if np.any(dominates):
            result[index] = False
    return result


def pareto_weights(
    latency: np.ndarray,
    area: np.ndarray,
    minimum_weight: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pareto flags and bounded proximity weights for one target."""
    epsilon = 1.0e-12
    latency_n = (latency - latency.min()) / (np.ptp(latency) + epsilon)
    area_n = (area - area.min()) / (np.ptp(area) + epsilon)
    frontier = pareto_mask(latency_n, area_n)
    frontier_points = np.column_stack((latency_n[frontier], area_n[frontier]))
    points = np.column_stack((latency_n, area_n))

    distance_to_front = np.min(
        np.linalg.norm(points[:, None, :] - frontier_points[None, :, :], axis=2),
        axis=1,
    )
    best_frontier_point = frontier_points[np.argmin(frontier_points.sum(axis=1))]
    distance_to_best = np.linalg.norm(points - best_frontier_point, axis=1)
    total_distance = distance_to_front + distance_to_best
    normalized = (total_distance - total_distance.min()) / (
        np.ptp(total_distance) + epsilon
    )
    weights = minimum_weight + (1.0 - minimum_weight) * (1.0 - normalized) ** gamma
    weights[frontier] = 1.0
    return frontier, weights


def aggregate_repeated_measurements(
    frame: pd.DataFrame,
    directive_columns: list[str],
) -> pd.DataFrame:
    """Collapse rows with the same target and effective pragma assignment."""
    keys = ["Device", "Clock_Period_nsec", *directive_columns]
    numeric_measurements = [
        column
        for column in (
            "Latency_msec",
            "Synthesis_Time_sec",
            "Synthesis_Time",
            *UTILIZATION_COLUMNS,
        )
        if column in frame.columns
    ]
    records: list[dict[str, object]] = []
    ordered = frame.sort_values([*keys, "Version"], kind="stable")
    for key, group in ordered.groupby(keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key_values))
        record["Version"] = sorted(group["Version"].astype(str))[0]
        record["Replicate_Count"] = int(len(group))
        for column in numeric_measurements:
            record[column] = float(group[column].median())
        records.append(record)
    return pd.DataFrame.from_records(records)


def assign_target_local_weights(
    frame: pd.DataFrame,
    minimum_weight: float,
    gamma: float,
) -> pd.DataFrame:
    """Compute Area and Pareto labels independently for every hardware target."""
    result = frame.copy()
    result["Area"] = result[list(UTILIZATION_COLUMNS)].sum(axis=1) / 4.0
    result["Weight"] = 0.0
    result["is_pareto"] = False
    for _, indices in result.groupby(
        ["Device", "Clock_Period_nsec"], sort=True
    ).groups.items():
        positions = list(indices)
        frontier, weights = pareto_weights(
            result.loc[positions, "Latency_msec"].to_numpy(dtype=float),
            result.loc[positions, "Area"].to_numpy(dtype=float),
            minimum_weight,
            gamma,
        )
        result.loc[positions, "Weight"] = weights
        result.loc[positions, "is_pareto"] = frontier
    return result


def preprocess_kernel(
    csv_path: Path,
    mode: str,
    device: str,
    clock_period_ns: float,
    minimum_weight: float,
    gamma: float,
    all_targets: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    kernel = csv_path.stem
    frame = pd.read_csv(csv_path, dtype={"Device": str})
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    for column in ("Clock_Period_nsec", "Latency_msec", *UTILIZATION_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("Synthesis_Time", "Synthesis_Time_sec"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if mode == "gnn" and not all_targets:
        frame = frame[
            (frame["Device"] == device)
            & np.isclose(frame["Clock_Period_nsec"], clock_period_ns, rtol=0.0, atol=1e-9)
        ].copy()
        if frame.empty:
            raise ValueError(f"{kernel} has no measurements for ({device}, {clock_period_ns} ns)")

    directive_columns = active_directive_columns(frame)
    action_definitions = load_action_definitions(kernel)
    missing_actions = [column for column in directive_columns if column not in action_definitions]
    if missing_actions:
        raise ValueError(
            f"{kernel} has active CSV actions absent from its complete APL/kernel_info/source "
            f"chain: {missing_actions}"
        )
    action_ids = [action_definitions[column].action_id for column in directive_columns]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError(f"{kernel} maps multiple active CSV columns to one action ID")

    for column in directive_columns:
        action = action_definitions[column]
        frame[column] = [canonicalize_directive(value, action) for value in frame[column]]

    finite_qor = np.isfinite(frame["Latency_msec"])
    finite_qor &= frame["Latency_msec"].gt(0) & frame["Latency_msec"].le(100000)
    for column in UTILIZATION_COLUMNS:
        finite_qor &= np.isfinite(frame[column]) & frame[column].ge(0) & frame[column].lt(100)
    valid = frame.loc[finite_qor].copy()
    if valid.empty:
        raise ValueError(f"{kernel} contains no valid QoR rows")

    # Zero is a reported utilization, not permission to invent one percent.
    # Numerical safeguards belong in logarithmic objectives, not measurements.
    aggregated = aggregate_repeated_measurements(valid, directive_columns)
    aggregated = assign_target_local_weights(aggregated, minimum_weight, gamma)

    ordered_columns = [
        "Version",
        "Device",
        "Clock_Period_nsec",
        "Latency_msec",
        *(["Synthesis_Time_sec"] if "Synthesis_Time_sec" in aggregated else []),
        *(["Synthesis_Time"] if "Synthesis_Time" in aggregated else []),
        *UTILIZATION_COLUMNS,
        *directive_columns,
        "Replicate_Count",
        "Area",
        "Weight",
        "is_pareto",
    ]
    aggregated = aggregated[ordered_columns].sort_values(
        ["Device", "Clock_Period_nsec", *directive_columns], kind="stable"
    ).reset_index(drop=True)

    effective_keys = ["Device", "Clock_Period_nsec", *directive_columns]
    if aggregated.duplicated(effective_keys).any():
        raise AssertionError(f"Duplicate effective pragma points remain for {kernel}")
    targets = aggregated[["Device", "Clock_Period_nsec"]].drop_duplicates()
    if mode == "gnn" and not all_targets and len(targets) != 1:
        raise AssertionError(f"GNN output for {kernel} is not single-target")

    stats = {
        "kernel": kernel,
        "raw_rows": int(len(frame)),
        "valid_rows": int(len(valid)),
        "output_rows": int(len(aggregated)),
        "targets": int(len(targets)),
        "active_actions": len(directive_columns),
    }
    return aggregated, stats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gnn", "llm"), required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=DEFAULT_GNN_DEVICE)
    parser.add_argument("--clock-period-ns", type=float, default=DEFAULT_GNN_CLOCK_NS)
    parser.add_argument(
        "--all-targets", action="store_true",
        help="For GNN mode, retain every measured device and clock period.",
    )
    parser.add_argument(
        "--exclude-kernels",
        default="",
        help="Comma-separated locked kernels whose measured GNN tables must not be read.",
    )
    parser.add_argument("--minimum-weight", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if not 0.0 < arguments.minimum_weight <= 1.0 or arguments.gamma <= 0.0:
        raise SystemExit("minimum-weight must be in (0, 1] and gamma must be positive")
    if arguments.all_targets and arguments.mode != "gnn":
        raise SystemExit("--all-targets is only meaningful for --mode gnn")
    excluded_kernels = {
        kernel.strip() for kernel in arguments.exclude_kernels.split(",")
        if kernel.strip()
    }
    if excluded_kernels and arguments.mode != "gnn":
        raise SystemExit("--exclude-kernels is only meaningful for --mode gnn")
    output_dir = arguments.output_dir or (
        DEFAULT_GNN_OUTPUT_DIR if arguments.mode == "gnn" else DEFAULT_LLM_OUTPUT_DIR
    )
    csv_files = sorted(arguments.input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {arguments.input_dir}")
    unknown_exclusions = excluded_kernels - {path.stem for path in csv_files}
    if unknown_exclusions:
        raise SystemExit(f"Unknown excluded GNN kernels: {sorted(unknown_exclusions)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in sorted(excluded_kernels):
        stale_output = output_dir / f"preprocessed-{kernel}.csv"
        if stale_output.exists():
            if not arguments.force:
                raise SystemExit(f"Locked kernel still has an old output: {stale_output}; pass --force")
            stale_output.unlink()
    csv_files = [path for path in csv_files if path.stem not in excluded_kernels]
    if not csv_files:
        raise SystemExit("No GNN training/validation CSVs remain after exclusions")

    manifest: dict[str, object] = {
        "schema": "mailohls-qor-preprocessing-v1",
        "mode": arguments.mode,
        "device": arguments.device if arguments.mode == "gnn" and not arguments.all_targets else None,
        "clock_period_ns": arguments.clock_period_ns if arguments.mode == "gnn" and not arguments.all_targets else None,
        "target_policy": (
            "all_measured_targets" if arguments.mode == "llm" or arguments.all_targets
            else "single_measured_target"
        ),
        "input_dir": str(arguments.input_dir.resolve()),
        "utilization_policy": "reported_percentages_unchanged",
        "area_metric": "arithmetic_mean_of_bram_dsp_ff_lut_percentages",
        "excluded_kernels": sorted(excluded_kernels),
        "kernels": [],
    }
    for index, csv_path in enumerate(csv_files, start=1):
        output_path = output_dir / f"preprocessed-{csv_path.name}"
        if output_path.exists() and not arguments.force:
            raise SystemExit(f"Refusing to overwrite {output_path}; pass --force")
        processed, stats = preprocess_kernel(
            csv_path,
            arguments.mode,
            arguments.device,
            arguments.clock_period_ns,
            arguments.minimum_weight,
            arguments.gamma,
            all_targets=arguments.all_targets,
        )
        processed.to_csv(output_path, index=False, float_format="%.12g")
        stats["input_sha256"] = file_sha256(csv_path)
        stats["output_sha256"] = file_sha256(output_path)
        manifest["kernels"].append(stats)
        print(
            f"[{index:02d}/{len(csv_files):02d}] {csv_path.stem}: "
            f"{stats['valid_rows']} valid -> {stats['output_rows']} unique, "
            f"{stats['targets']} target(s), {stats['active_actions']} action(s)"
        )

    manifest_path = output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(csv_files)} preprocessed tables to {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
