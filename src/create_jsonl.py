#!/usr/bin/env python3
"""
create_jsonl_multi_target.py

Build one JSONL dataset from all preprocessed per-kernel CSVs.

Compared to the previous single-target create_jsonl.py, this version keeps the
same core JSONL fields:
  - input
  - target
  - latency
  - area
  - kernel_name

and adds target-platform/QoR metadata:
  - device
  - clock_period
  - bram_util_%
  - dsp_util_%
  - ff_util_%
  - lut_util_%

It assumes each input CSV was produced by the corrected multi-target
preprocessing script and therefore already contains only valid designs from all
available devices/frequencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from insert_placeholders import insert_placeholders


NO_DIRECTIVE_VALUES = {"", "auto", "ndir", "none", "nan", "null"}
SOURCE_EXTS = (".cpp", ".cc", ".cxx", ".c")

REQUIRED_QOR_COLUMNS = [
    "Device",
    "Clock_Period_nsec",
    "Latency_msec",
    "BRAM_Utilization_percentage",
    "DSP_Utilization_percentage",
    "FF_Utilization_percentage",
    "LUT_Utilization_percentage",
    "Area",
]


def _safe_float(value, field_name: str, *, default: Optional[float] = None) -> Optional[float]:
    """Convert CSV value to float with a clear error message."""
    if pd.isna(value):
        if default is not None:
            return default
        raise ValueError(f"Missing numeric field: {field_name}")
    try:
        return float(value)
    except Exception as exc:
        if default is not None:
            return default
        raise ValueError(f"Could not parse {field_name}={value!r} as float") from exc


def _safe_int(value, field_name: str, *, default: Optional[int] = None) -> Optional[int]:
    """Convert CSV/directive value to int with a clear error message."""
    if value is None or pd.isna(value):
        if default is not None:
            return default
        raise ValueError(f"Missing integer field: {field_name}")
    try:
        return int(float(str(value).strip()))
    except Exception as exc:
        if default is not None:
            return default
        raise ValueError(f"Could not parse {field_name}={value!r} as int") from exc


def _parse_bool(value) -> bool:
    """Robust bool parser for pandas/string values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _normalize_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _candidate_files(root: Path, kernel_name: str, suffix: str) -> Iterable[Path]:
    """Try exact and hyphen/underscore variants."""
    variants = {
        kernel_name,
        kernel_name.replace("-", "_"),
        kernel_name.replace("_", "-"),
    }
    for v in sorted(variants):
        yield root / f"{v}{suffix}"


def _find_existing_file(root: Path, kernel_name: str, suffix: str) -> Optional[Path]:
    for p in _candidate_files(root, kernel_name, suffix):
        if p.is_file():
            return p
    return None


def load_kernel_metadata(
    kernel_name: str,
    application_dataset_root: str | Path,
    apl_mapping_root: str | Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build:
      LOOP_MAP:            {csv_col -> loop_label}
      ARRAY_MAP:           {csv_col -> array_label}
      LOOP_ITERATIONS_MAP: {loop_label -> iteration_count_as_string}

    Uses:
      {application_dataset_root}/{kernel_name}/kernel_info.txt
      {apl_mapping_root}/{kernel_name}.txt
    """
    application_dataset_root = Path(application_dataset_root)
    apl_mapping_root = Path(apl_mapping_root)

    kernel_dir = application_dataset_root / kernel_name
    kernel_info_file = kernel_dir / "kernel_info.txt"
    apl_mapping_file = _find_existing_file(apl_mapping_root, kernel_name, ".txt")

    if not kernel_info_file.is_file():
        raise FileNotFoundError(f"Missing kernel_info.txt: {kernel_info_file}")
    if apl_mapping_file is None:
        raise FileNotFoundError(
            f"Missing APL mapping for kernel={kernel_name} under {apl_mapping_root}"
        )

    LOOP_MAP: Dict[str, str] = {}
    ARRAY_MAP: Dict[str, str] = {}
    LOOP_ITERATIONS_MAP: Dict[str, str] = {}

    with open(kernel_info_file, "r", encoding="utf-8") as f:
        info_lines = [ln.strip() for ln in f if ln.strip()]

    # First line is top function / kernel name; skip it.
    for line in info_lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        label = parts[0]
        kind = parts[1].lower()

        if kind == "loop" and len(parts) >= 3:
            LOOP_ITERATIONS_MAP[label] = parts[2]

    with open(apl_mapping_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            csv_col, label = parts[0], parts[1]
            if csv_col.lower().startswith("array"):
                ARRAY_MAP[csv_col] = label
            else:
                LOOP_MAP[csv_col] = label

    return LOOP_MAP, ARRAY_MAP, LOOP_ITERATIONS_MAP


def find_source_file(kernel_dir: str | Path) -> Optional[Path]:
    """Find the source file used for placeholder insertion."""
    kernel_dir = Path(kernel_dir)

    candidates = []
    for p in sorted(kernel_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SOURCE_EXTS:
            continue
        if "placeholder" in p.stem.lower():
            continue
        candidates.append(p)

    if not candidates:
        return None

    # Prefer .cpp if multiple are present, otherwise deterministic first file.
    cpp_candidates = [p for p in candidates if p.suffix.lower() == ".cpp"]
    return cpp_candidates[0] if cpp_candidates else candidates[0]


def decode_loop_cell(
    value,
    loop_label: str,
    loop_iterations_map: Dict[str, str],
) -> Tuple[int, int]:
    """
    Decode a loop directive cell into (pipe_ii, unroll_factor).

    Important compatibility choice:
      If the CSV cell is NDIR/auto/empty, we still emit PIPE=0 and UNROLL=0.
      This preserves explicit supervision for loop no-op decisions.
    """
    val = _normalize_cell(value)

    pipe = 0
    unroll = 0

    if val in NO_DIRECTIVE_VALUES:
        return pipe, unroll

    if val == "unroll":
        iter_count = loop_iterations_map.get(loop_label)
        if iter_count not in (None, ""):
            unroll = _safe_int(iter_count, f"iteration_count[{loop_label}]", default=0) or 0
    elif val.startswith("unroll_"):
        unroll = _safe_int(val.split("_", 1)[1], f"{loop_label}:{val}", default=0) or 0
    elif val == "pipeline":
        pipe = 1
    elif val.startswith("pipeline_"):
        pipe = _safe_int(val.split("_", 1)[1], f"{loop_label}:{val}", default=1) or 1
    else:
        # Unknown loop directive token: keep a conservative no-op but make it visible.
        print(f"[WARN] Unknown loop directive token for {loop_label}: {value!r}; using PIPE=0, UNROLL=0")

    return pipe, unroll


def append_array_directive_lines(pragma_lines, value, array_label: str) -> None:
    """
    Decode array directive cell and append target assignment lines.

    Compatibility choice:
      If the CSV cell is NDIR/auto/empty, we emit no array lines, matching the
      previous dataset behavior where absent array pragmas were not supervised.
    """
    val = _normalize_cell(value)
    if val in NO_DIRECTIVE_VALUES:
        return

    parts = val.split("_")

    if parts[0] == "complete":
        # Usually: complete_1, where the second field is dim.
        dim = _safe_int(parts[1], f"array_dim[{array_label}]", default=1) if len(parts) >= 2 else 1
        pragma_lines.append(f"auto{{_ARRAY_T_{array_label}}} = complete")
        pragma_lines.append(f"auto{{_ARRAY_F_{array_label}}} = 0")
        pragma_lines.append(f"auto{{_ARRAY_D_{array_label}}} = {dim}")
        return

    if parts[0] in {"block", "cyclic"} and len(parts) >= 3:
        factor = _safe_int(parts[1], f"array_factor[{array_label}]", default=0) or 0
        dim = _safe_int(parts[2], f"array_dim[{array_label}]", default=1) or 1
        pragma_lines.append(f"auto{{_ARRAY_T_{array_label}}} = {parts[0]}")
        pragma_lines.append(f"auto{{_ARRAY_F_{array_label}}} = {factor}")
        pragma_lines.append(f"auto{{_ARRAY_D_{array_label}}} = {dim}")
        return

    print(f"[WARN] Unknown array directive token for {array_label}: {value!r}; skipping array directive")


def build_json_entry(
    row,
    df_columns,
    input_template: str,
    kernel_name: str,
    loop_map: Dict[str, str],
    array_map: Dict[str, str],
    loop_iterations_map: Dict[str, str],
) -> Optional[dict]:
    pragma_lines = []

    # Loop pragmas.
    for csv_col, loop_label in loop_map.items():
        if csv_col not in df_columns:
            continue
        pipe, unroll = decode_loop_cell(row[csv_col], loop_label, loop_iterations_map)
        pragma_lines.append(f"auto{{_PIPE_{loop_label}}} = {pipe}")
        pragma_lines.append(f"auto{{_UNROLL_{loop_label}}} = {unroll}")

    # Array pragmas.
    for csv_col, array_label in array_map.items():
        if csv_col not in df_columns:
            continue
        append_array_directive_lines(pragma_lines, row[csv_col], array_label)

    if not pragma_lines:
        return None

    # Existing fields.
    json_entry = {
        "input": input_template,
        "target": "\n".join(pragma_lines),
        "latency": _safe_float(row["Latency_msec"], "Latency_msec"),
        "area": _safe_float(row["Area"], "Area"),
        "kernel_name": kernel_name,
    }

    # New multi-target fields requested by the user.
    json_entry.update({
        "device": str(row["Device"]).strip(),
        "clock_period": _safe_float(row["Clock_Period_nsec"], "Clock_Period_nsec"),
        "bram_util_%": _safe_float(row["BRAM_Utilization_percentage"], "BRAM_Utilization_percentage"),
        "dsp_util_%": _safe_float(row["DSP_Utilization_percentage"], "DSP_Utilization_percentage"),
        "ff_util_%": _safe_float(row["FF_Utilization_percentage"], "FF_Utilization_percentage"),
        "lut_util_%": _safe_float(row["LUT_Utilization_percentage"], "LUT_Utilization_percentage"),
    })

    return json_entry


def create_llm_data_json(
    csv_file: str | Path,
    cpp_file: str | Path,
    kernel_name: str,
    application_dataset_root: str | Path,
    apl_mapping_root: str | Path,
    output_jsonl: str | Path,
    append: bool = False,
) -> int:
    """
    Append JSONL rows for one kernel.

    Each JSONL line contains the old core fields plus target-platform metadata.
    """
    csv_file = Path(csv_file)
    cpp_file = Path(cpp_file)
    output_jsonl = Path(output_jsonl)

    if not csv_file.is_file():
        raise FileNotFoundError(f"Missing preprocessed CSV: {csv_file}")
    if not cpp_file.is_file():
        raise FileNotFoundError(f"Missing source file: {cpp_file}")

    loop_map, array_map, loop_iterations_map = load_kernel_metadata(
        kernel_name,
        application_dataset_root=application_dataset_root,
        apl_mapping_root=apl_mapping_root,
    )

    placeholder_lines = insert_placeholders(str(cpp_file))
    input_template = "".join(placeholder_lines).strip()

    df = pd.read_csv(csv_file)
    missing = [c for c in REQUIRED_QOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_file} is missing required columns: {missing}")

    mode = "a" if append else "w"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_jsonl, mode, encoding="utf-8") as out_f:
        for _, row in df.iterrows():
            entry = build_json_entry(
                row=row,
                df_columns=df.columns,
                input_template=input_template,
                kernel_name=kernel_name,
                loop_map=loop_map,
                array_map=array_map,
                loop_iterations_map=loop_iterations_map,
            )
            if entry is None:
                continue
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[OK] {kernel_name}: wrote {written}/{len(df)} rows from {csv_file.name}"
    )
    return written


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--application_dataset_root",
        default="/home/elvouvali/Data4LLMPrompting/ApplicationDataset",
        help="Directory containing one subdirectory per kernel with source + kernel_info.txt.",
    )
    ap.add_argument(
        "--apl_mapping_root",
        default="/home/elvouvali/Data4LLMPrompting/ApplicationAPLMapping",
        help="Directory containing <kernel>.txt APL mapping files.",
    )
    ap.add_argument(
        "--preprocessed_csv_dir",
        default="/home/elvouvali/Data4LLMPrompting/preprocessed_CSVS",
        help="Directory containing preprocessed-<kernel>.csv files.",
    )
    ap.add_argument(
        "--output_jsonl",
        default="/home/elvouvali/LLM_data/all_kernels_llm_data_multi_target.jsonl",
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to output_jsonl instead of overwriting it.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first failed kernel instead of skipping it.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    application_dataset_root = Path(args.application_dataset_root)
    preprocessed_csv_dir = Path(args.preprocessed_csv_dir)
    output_jsonl = Path(args.output_jsonl)

    if not application_dataset_root.is_dir():
        raise FileNotFoundError(f"Application dataset root not found: {application_dataset_root}")
    if not preprocessed_csv_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed CSV directory not found: {preprocessed_csv_dir}")

    kernel_names = sorted(
        d.name for d in application_dataset_root.iterdir()
        if d.is_dir()
    )

    print(f"Found {len(kernel_names)} kernels under {application_dataset_root}")
    print(f"Output JSONL: {output_jsonl}")

    if output_jsonl.exists() and not args.append:
        output_jsonl.unlink()
        print(f"Removed existing output JSONL because --append was not set: {output_jsonl}")

    total_written = 0
    ok = 0
    failed = 0
    skipped = 0

    for idx, kernel_name in enumerate(kernel_names, start=1):
        print(f"\n[{idx}/{len(kernel_names)}] Processing kernel: {kernel_name}")

        csv_file = preprocessed_csv_dir / f"preprocessed-{kernel_name}.csv"
        if not csv_file.is_file():
            # Fallback for hyphen/underscore naming differences.
            candidates = list(_candidate_files(preprocessed_csv_dir, f"preprocessed-{kernel_name}", ".csv"))
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                print(f"[SKIP] Missing preprocessed CSV for {kernel_name}: {csv_file}")
                skipped += 1
                continue
            csv_file = found

        cpp_file = find_source_file(application_dataset_root / kernel_name)
        if cpp_file is None:
            print(f"[SKIP] No source file found for {kernel_name}")
            skipped += 1
            continue

        try:
            written = create_llm_data_json(
                csv_file=csv_file,
                cpp_file=cpp_file,
                kernel_name=kernel_name,
                application_dataset_root=application_dataset_root,
                apl_mapping_root=args.apl_mapping_root,
                output_jsonl=output_jsonl,
                append=True,
            )
            total_written += written
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {kernel_name}: {exc}")
            if args.strict:
                raise

    print("\nDone.")
    print(f"Kernels processed successfully: {ok}")
    print(f"Kernels skipped: {skipped}")
    print(f"Kernels failed: {failed}")
    print(f"Total JSONL rows written: {total_written}")
    print(f"Output JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
