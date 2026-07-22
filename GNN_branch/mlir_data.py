#!/usr/bin/env python3
"""
mlir_data.py
============

MailoHLS dataset builder for the GEXF graphs emitted by "mlir_graph_gen.py".

The script keeps the compact dataset contract used by the existing MailoHLS
GNN pipeline:

    one static graph tensor pack per kernel
    + one design-point tensor pack per kernel
    + one lightweight global index

Every design point therefore reuses the same structural tensors while changing
only:
  * the pragma values injected at the corresponding action scopes;
  * the flat padded pragma vector;
  * the QoR targets.

This is intentionally a *drop-in data module* for the current model.py and
train_GNN.py interfaces.  It produces PyG Data objects with the same fields:

    x, edge_index, edge_attr,
    X_contextnids, X_pragmanids, X_pragmascopenids,
    X_pseudonids, X_arrayscopenids,
    X_pipeline_scopeids, X_unroll_scopeids,
    X_array_partition_scopeids, X_scopenids, X_icmpnids,
    X_pragma_per_node, pragmas,
    perf, actual_perf, kernel_speedup, area, actual_area.

Why MLIR-specific encoding?
---------------------------
The previous HARP encoder one-hot encoded graph-local LLVM block/function IDs.
Those IDs are useful for reconstructing a single lowered graph but do not carry
stable semantics across kernels.  The MLIR graph already contains explicit
region, loop, SSA, MemRef, access, and dependence relations.  This encoder
therefore uses:

Categorical node features
  * node kind;
  * exact MLIR operation/value token;
  * canonical operation family;
  * canonical value/result type;
  * pragma kind;
  * SSA kind;
  * feature kind.

Numeric node features
  * loop depth, bounds, step, and trip count;
  * operand/result/use counts;
  * memory rank and approximate static volume;
  * action, loop, memory, block-argument, and source-location indicators.

Categorical edge features
  * semantic flow type;
  * semantic role;
  * dependence certainty.

Numeric edge features
  * bounded/log-scaled position and operand index;
  * affine-access indicator;
  * signed/log-scaled dependence distance.

The exact graph topology is retained.  Parallel edges are preserved and node
and edge tensors are built from one deterministic traversal, preventing
"edge_index" / "edge_attr" misalignment.

Expected repository layout
--------------------------
The script searches the following locations automatically:

GEXF graphs:
  GNN_branch/MLIR_graphs/*.gexf

Application metadata:
  GNN_branch/Data/ApplicationDataset/<kernel>/kernel_info.txt
  Data/ApplicationDataset/<kernel>/kernel_info.txt
  Data4LLMPrompting/ApplicationDataset/<kernel>/kernel_info.txt

CSV design points:
  GNN_branch/Data/preprocessed_CSVS/
  Data/preprocessed_CSVS/
  Data4LLMPrompting/preprocessed_CSVS/

APL mappings:
  GNN_branch/Data/ApplicationAPLMapping/
  Data/ApplicationAPLMapping/
  Data4LLMPrompting/ApplicationAPLMapping/

Usage
-----
1. In config.py set:
       dataset = "mlir"
       force_regen = True

2. Change the imports in main_GNN.py and train_GNN.py from "data" to
   "mlir_data" (or rename this file to data.py for an isolated MLIR run).

3. Run:
       python main_GNN.py --dataset mlir --force_regen True

The first run fits new MLIR encoders.  Do not reuse the HARP encoders because
the node and edge vocabularies are different.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset

from config import FLAGS, ALL_KERNEL
from utils import get_root_path


# ---------------------------------------------------------------------------
# Repository paths and persisted dataset layout.
# ---------------------------------------------------------------------------

ROOT = Path(get_root_path()).resolve()
DATASET_NAME = "MLIR_dataset"

GEXF_DIR_CANDIDATES = (
    ROOT / "GNN_branch" / "MLIR_graphs",
    ROOT / "MLIR_graphs",
)

APPLICATION_DIR_CANDIDATES = (
    ROOT / "GNN_branch" / "Data" / "ApplicationDataset",
    ROOT / "Data" / "ApplicationDataset",
    ROOT / "Data4LLMPrompting" / "ApplicationDataset",
)

CSV_DIR_CANDIDATES = (
    ROOT / "GNN_branch" / "Data" / "preprocessed_CSVS",
    ROOT / "Data" / "preprocessed_CSVS",
    ROOT / "Data4LLMPrompting" / "preprocessed_CSVS",
)

APL_DIR_CANDIDATES = (
    ROOT / "GNN_branch" / "Data" / "ApplicationAPLMapping",
    ROOT / "Data" / "ApplicationAPLMapping",
    ROOT / "Data4LLMPrompting" / "ApplicationAPLMapping",
)

SAVE_DIR = ROOT / "GNN_branch" / DATASET_NAME / "all_kernels"
GRAPH_DIR = SAVE_DIR / "graphs"
POINT_DIR = SAVE_DIR / "points"
INDEX_PATH = SAVE_DIR / "index.pt"
ENCODER_PATH = SAVE_DIR / "encoders.pkl"
PRAGMA_DIM_PATH = SAVE_DIR / "pragma_dim.pt"
SCHEMA_PATH = SAVE_DIR / "feature_schema.json"


# ---------------------------------------------------------------------------
# MailoHLS graph constants.
# ---------------------------------------------------------------------------

NODE_TYPE_OP = 0
NODE_TYPE_VALUE = 1
NODE_TYPE_IMMEDIATE = 2
NODE_TYPE_PSEUDO_BLOCK = 4
NODE_TYPE_PRAGMA = 100
NODE_TYPE_ARRAY_SCOPE = 104

FLOW_PRAGMA = 200

PRAGMA_VECTOR_WIDTH = 5
PIPELINE_COL = 0
UNROLL_COL = 1
PARTITION_TYPE_COL = 2
PARTITION_FACTOR_COL = 3
PARTITION_DIM_COL = 4

ARRAY_TYPE_ENCODING = {
    "": 0,
    "none": 0,
    "cyclic": 100,
    "block": 200,
    "complete": 300,
}

AUTO_KEY_RE = re.compile(r"auto\{([^}]+)\}")
ACTION_ID_RE = re.compile(r"^L[1-9][0-9]*$")
INTEGER_RE = re.compile(r"[-+]?[0-9]+")


# ---------------------------------------------------------------------------
# Small records for design-point metadata.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CSVResult:
    point: dict[str, Any]
    perf: float
    area: float
    res_util: dict[str, float]
    row_idx: int
    src_csv: str


# ---------------------------------------------------------------------------
# General helpers.
# ---------------------------------------------------------------------------

def _first_existing_dir(candidates: Sequence[Path], description: str) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n  {tried}")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool01(value: Any) -> float:
    if isinstance(value, str):
        return float(value.strip().lower() in {"1", "true", "yes", "y"})
    return float(bool(_as_int(value, 0)))


def _signed_log1p(value: Any) -> float:
    """Compress an integer-like magnitude while preserving its sign."""
    x = _as_float(value, 0.0)
    return math.copysign(math.log1p(abs(x)), x)


def _log1p_nonnegative(value: Any) -> float:
    return math.log1p(max(0.0, _as_float(value, 0.0)))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() == "":
        return []
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, list) else []
    except json.JSONDecodeError:
        return []


def _make_onehot_encoder() -> OneHotEncoder:
    """
    Support both current scikit-learn (sparse_output) and older releases
    (sparse).  Unknown tokens map to all zeros for inference compatibility.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )


def _fit_onehot(tokens: Iterable[str]) -> OneHotEncoder:
    values = sorted(set(str(t) for t in tokens))
    if not values:
        values = ["<none>"]
    encoder = _make_onehot_encoder()
    encoder.fit(np.asarray(values, dtype=object).reshape(-1, 1))
    return encoder


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _node_id_order(graph: nx.Graph) -> tuple[list[str], dict[str, int]]:
    """
    Use the canonical numeric IDs emitted by mlir_graph_gen.py.

    We do not call ``convert_node_labels_to_integers`` because edge tensors and
    node tensors should be built from one explicit mapping.
    """
    nodes = list(graph.nodes())
    try:
        ordered = sorted(nodes, key=lambda node: int(node))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "MLIR GEXF node IDs must be numeric strings emitted by "
            "mlir_graph_gen.py."
        ) from exc

    numeric = [int(node) for node in ordered]
    expected = list(range(len(ordered)))
    if numeric != expected:
        raise RuntimeError(
            f"Expected contiguous node IDs 0..{len(ordered)-1}, got "
            f"min={min(numeric, default=-1)}, max={max(numeric, default=-1)}."
        )

    return ordered, {node: index for index, node in enumerate(ordered)}


def _iter_edges(graph: nx.Graph):
    """
    Yield every edge exactly once, preserving MultiDiGraph parallel edges.
    """
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for source, target, attrs in graph.edges(data=True):
            yield source, target, 0, attrs


# ---------------------------------------------------------------------------
# Kernel, APL, and CSV parsing.
# ---------------------------------------------------------------------------

def _name_variants(name: str) -> list[str]:
    return list(dict.fromkeys([
        name,
        name.replace("-", "_"),
        name.replace("_", "-"),
    ]))


def _find_application_dir(kernel: str) -> Path:
    for base in APPLICATION_DIR_CANDIDATES:
        for variant in _name_variants(kernel):
            candidate = base / variant
            if (candidate / "kernel_info.txt").is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find ApplicationDataset directory for kernel '{kernel}'."
    )


def _find_apl_mapping(kernel: str) -> Path:
    for base in APL_DIR_CANDIDATES:
        for variant in _name_variants(kernel):
            candidate = base / f"{variant}.txt"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find ApplicationAPLMapping file for kernel '{kernel}'."
    )


def _find_csv(kernel: str) -> Path:
    csv_base = _first_existing_dir(CSV_DIR_CANDIDATES, "preprocessed CSV directory")
    candidates: list[Path] = []
    for variant in _name_variants(kernel):
        candidates.extend([
            csv_base / f"preprocessed-{variant}.csv",
            csv_base / f"preprocessed_{variant}.csv",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Conservative fallback: require an unambiguous fuzzy match.
    matches: list[Path] = []
    for variant in _name_variants(kernel):
        matches.extend(csv_base.glob(f"preprocessed*{variant}*.csv"))
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No preprocessed CSV found for '{kernel}'.")
    raise RuntimeError(
        f"Ambiguous CSV match for '{kernel}': {[str(p) for p in matches]}"
    )


def _load_label_to_columns(kernel: str) -> dict[str, list[str]]:
    mapping_file = _find_apl_mapping(kernel)
    label_to_columns: dict[str, list[str]] = defaultdict(list)

    with mapping_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) < 2:
                continue
            column, action_id = fields[0], fields[1]
            if ACTION_ID_RE.fullmatch(action_id):
                label_to_columns[action_id].append(column)

    if not label_to_columns:
        raise RuntimeError(f"APL mapping is empty or invalid: {mapping_file}")
    return dict(label_to_columns)


def parse_kernel_info(kernel: str) -> dict[str, tuple[Any, ...]]:
    """
    Return CSV-column -> action metadata, preserving current MailoHLS logic.

    Loop entry:
        column -> (Lk, loop_bound)

    Array entry:
        column -> (Lk, {dimension: bound}, array_name)
    """
    app_dir = _find_application_dir(kernel)
    kernel_info = app_dir / "kernel_info.txt"
    label_to_columns = _load_label_to_columns(kernel)

    mapping: dict[str, tuple[Any, ...]] = {}
    nonempty_lines = [
        line.strip()
        for line in kernel_info.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not nonempty_lines:
        raise RuntimeError(f"Empty kernel_info.txt: {kernel_info}")

    # First non-empty line is the top-level function.
    for line in nonempty_lines[1:]:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 3:
            continue

        action_id, kind = fields[0], fields[1].lower()
        columns = label_to_columns.get(action_id, [])
        if not columns:
            continue

        if kind == "loop":
            loop_bound = _as_int(fields[2], 0)
            for column in columns:
                mapping[column] = (action_id, loop_bound)
        elif kind == "array":
            array_name = fields[2]
            dimensions: dict[int, int] = {}
            cursor = 3
            while cursor + 1 < len(fields):
                dim = _as_int(fields[cursor], -1)
                bound = _as_int(fields[cursor + 1], -1)
                if dim < 0 or bound <= 0:
                    break
                dimensions[dim] = bound
                cursor += 2
            if dimensions:
                for column in columns:
                    mapping[column] = (action_id, dimensions, array_name)

    if not mapping:
        raise RuntimeError(
            f"No CSV columns could be mapped for '{kernel}'. Check "
            f"{kernel_info} and the APL mapping."
        )
    return mapping


def parse_pragma_token(
    token: str,
    action_id: str,
    loop_bound: int | Mapping[int, int],
) -> dict[str, Any]:
    token = (token or "").strip().lower()

    pipe_key = f"_PIPE_{action_id}"
    unroll_key = f"_UNROLL_{action_id}"
    array_type_key = f"_ARRAY_T_{action_id}"
    array_factor_key = f"_ARRAY_F_{action_id}"
    array_dim_key = f"_ARRAY_D_{action_id}"

    if token == "":
        return {}

    if token == "pipeline":
        return {pipe_key: 1, unroll_key: 0}
    if token.startswith("pipeline_"):
        return {
            pipe_key: _as_int(token.split("_", 1)[1], 1),
            unroll_key: 0,
        }

    if token == "unroll":
        return {
            pipe_key: 0,
            unroll_key: _as_int(loop_bound, 0),
        }
    if token.startswith("unroll_"):
        return {
            pipe_key: 0,
            unroll_key: _as_int(token.split("_", 1)[1], 0),
        }

    if token.startswith(("cyclic_", "block_")):
        parts = token.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Malformed array pragma '{token}' for action {action_id}."
            )
        return {
            array_type_key: parts[0],
            array_factor_key: _as_int(parts[1], 0),
            array_dim_key: _as_int(parts[2], 0),
        }

    if token.startswith("complete_"):
        parts = token.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Malformed array pragma '{token}' for action {action_id}."
            )
        return {
            array_type_key: "complete",
            array_factor_key: 0,
            array_dim_key: _as_int(parts[1], 0),
        }

    raise ValueError(
        f"Unsupported pragma token '{token}' for action {action_id}."
    )


def load_csv_results(kernel: str) -> list[CSVResult]:
    csv_path = _find_csv(kernel)
    kernel_info_map = parse_kernel_info(kernel)
    results: list[CSVResult] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            point: dict[str, Any] = {}
            for column, metadata in kernel_info_map.items():
                if column not in row:
                    continue
                action_id = metadata[0]
                auxiliary = metadata[1]
                point.update(
                    parse_pragma_token(row[column], action_id, auxiliary)
                )

            perf = _as_float(
                row.get("Latency_msec", row.get("Latency", 0.0)),
                0.0,
            )
            area = _as_float(row.get("Area", 0.0), 0.0)
            res_util = {
                "util-BRAM": _as_float(
                    row.get("BRAM_Utilization_percentage", 0.0)
                ) / 100.0,
                "util-DSP": _as_float(
                    row.get("DSP_Utilization_percentage", 0.0)
                ) / 100.0,
                "util-FF": _as_float(
                    row.get("FF_Utilization_percentage", 0.0)
                ) / 100.0,
                "util-LUT": _as_float(
                    row.get("LUT_Utilization_percentage", 0.0)
                ) / 100.0,
            }
            results.append(
                CSVResult(
                    point=point,
                    perf=perf,
                    area=area,
                    res_util=res_util,
                    row_idx=row_idx,
                    src_csv=csv_path.name,
                )
            )

    if results and all(not result.point for result in results):
        raise RuntimeError(
            f"All design points for '{kernel}' are empty. The CSV headers do "
            "not match the APL mapping."
        )
    return results


# ---------------------------------------------------------------------------
# MLIR semantic feature extraction.
# ---------------------------------------------------------------------------

def node_kind(attrs: Mapping[str, Any]) -> str:
    node_type = _as_int(attrs.get("type"), -1)
    return {
        NODE_TYPE_OP: "operation",
        NODE_TYPE_VALUE: "ssa_value",
        NODE_TYPE_IMMEDIATE: "immediate",
        NODE_TYPE_PSEUDO_BLOCK: "pseudo_block",
        NODE_TYPE_PRAGMA: "pragma",
        NODE_TYPE_ARRAY_SCOPE: "array_scope",
    }.get(node_type, f"node_type_{node_type}")


def pragma_kind(attrs: Mapping[str, Any]) -> str:
    if _as_int(attrs.get("type"), -1) != NODE_TYPE_PRAGMA:
        return "NONE"
    token = str(attrs.get("text", "")).strip().upper()
    return token if token in {"PIPELINE", "UNROLL", "ARRAY_PARTITION"} else "NONE"


def canonical_op_family(attrs: Mapping[str, Any]) -> str:
    """
    Coarse semantic family used in addition to the exact MLIR operation token.
    """
    text = str(attrs.get("text", "")).strip().lower()
    kind = node_kind(attrs)

    if kind != "operation":
        return kind
    if text in {"affine.for", "affine.parallel", "scf.for", "scf.parallel",
                "scf.forall", "scf.while"}:
        return "loop"
    if text in {"affine.load", "memref.load", "vector.load",
                "vector.transfer_read", "affine.vector_load"}:
        return "memory_read"
    if text in {"affine.store", "memref.store", "vector.store",
                "vector.transfer_write", "affine.vector_store"}:
        return "memory_write"
    if "atomic" in text:
        return "memory_atomic"
    if text.startswith("memref.") and any(
        part in text for part in
        ("cast", "view", "subview", "reshape", "transpose",
         "reinterpret", "collapse", "expand")
    ):
        return "memory_view"
    if text in {"memref.alloc", "memref.alloca", "llvm.alloca"}:
        return "memory_alloc"
    if text.startswith(("arith.add", "arith.sub", "arith.mul", "arith.div",
                        "math.", "complex.")):
        return "arithmetic"
    if text.startswith(("arith.cmp", "affine.if", "scf.if")):
        return "compare_or_condition"
    if text in {"func.call", "llvm.call"}:
        return "call"
    if text in {"func.func", "llvm.func"}:
        return "function"
    if text in {"func.return", "llvm.return"}:
        return "return"
    if text.endswith("yield") or text.endswith("condition"):
        return "region_terminator"
    if text.startswith(("cf.br", "cf.cond_br", "llvm.br")):
        return "branch"
    if text == "arith.constant":
        return "constant"
    if text.startswith(("arith.ext", "arith.trunc", "arith.index_cast",
                        "arith.bitcast", "memref.cast",
                        "unrealized_conversion_cast")):
        return "cast"
    return "other_operation"


def canonical_type(attrs: Mapping[str, Any]) -> str:
    raw = str(
        attrs.get("value_type")
        or attrs.get("result_types")
        or attrs.get("operand_types")
        or attrs.get("text")
        or "unknown"
    ).replace(" ", "")

    if raw.startswith("memref<") or "memref<" in raw:
        rank = _as_int(attrs.get("memory_rank"), -1)
        element_match = re.findall(
            r"(?:memref<[^>]*x)?(f16|bf16|f32|f64|i[0-9]+|ui[0-9]+|si[0-9]+)",
            raw,
        )
        element = element_match[-1] if element_match else "unknown"
        return f"memref_rank{rank}_{element}"
    if raw.startswith("tensor<") or "tensor<" in raw:
        return "tensor"
    if raw.startswith("vector<") or "vector<" in raw:
        return "vector"
    if "!llvm.ptr" in raw or raw.startswith("ptr"):
        return "pointer"
    if re.search(r"\bf(?:16|32|64)\b", raw):
        return "floating"
    if re.search(r"\b(?:s|u)?i[0-9]+\b", raw):
        return "integer"
    if "index" in raw:
        return "index"
    return raw[:96] if raw else "unknown"


def _memory_volume_log(attrs: Mapping[str, Any]) -> float:
    dims = _json_list(attrs.get("memory_shape"))
    if not dims:
        return 0.0

    volume = 1
    has_static_dim = False
    for dim in dims:
        value = _as_int(dim, -1)
        if value <= 0:
            continue
        volume *= value
        has_static_dim = True
        if volume > 2**50:
            volume = 2**50
            break
    return math.log2(volume + 1.0) if has_static_dim else 0.0


def node_numeric_features(attrs: Mapping[str, Any]) -> list[float]:
    """
    Dense, bounded, transferable MLIR properties.

    Raw source line, graph-local block ID, graph-local function ID, and action
    ordinal are intentionally excluded: they are identifiers, not semantics.
    """
    return [
        _as_bool01(attrs.get("is_loop")),
        _log1p_nonnegative(attrs.get("loop_depth")),
        _signed_log1p(attrs.get("loop_lower")),
        _signed_log1p(attrs.get("loop_upper")),
        _signed_log1p(attrs.get("loop_step")),
        math.log2(max(0, _as_int(attrs.get("trip_count"), 0)) + 1.0),
        _as_bool01(attrs.get("trip_count_static")),
        _log1p_nonnegative(attrs.get("operand_count")),
        _log1p_nonnegative(attrs.get("result_count")),
        _log1p_nonnegative(attrs.get("ssa_use_count")),
        _as_bool01(attrs.get("is_block_argument")),
        _as_bool01(attrs.get("is_memory")),
        _log1p_nonnegative(attrs.get("memory_rank")),
        _as_bool01(attrs.get("memory_static_shape")),
        _memory_volume_log(attrs),
        _log1p_nonnegative(attrs.get("memory_root_count")),
        _as_bool01(attrs.get("is_function")),
        _as_bool01(bool(str(attrs.get("action_id", "")).strip())),
        _as_bool01(bool(str(attrs.get("source_location", "")).strip())),
        _as_bool01(attrs.get("is_memory_root")),
    ]


NODE_NUMERIC_NAMES = [
    "is_loop",
    "log1p_loop_depth",
    "signed_log1p_loop_lower",
    "signed_log1p_loop_upper",
    "signed_log1p_loop_step",
    "log2_trip_count_plus_1",
    "trip_count_static",
    "log1p_operand_count",
    "log1p_result_count",
    "log1p_ssa_use_count",
    "is_block_argument",
    "is_memory",
    "log1p_memory_rank",
    "memory_static_shape",
    "log2_memory_volume_plus_1",
    "log1p_memory_root_count",
    "is_function",
    "is_action_anchor",
    "has_source_location",
    "is_memory_root",
]


def edge_role(attrs: Mapping[str, Any]) -> str:
    role = str(attrs.get("role", "")).strip()
    return role if role else "<none>"


def edge_certainty(attrs: Mapping[str, Any]) -> str:
    certainty = str(attrs.get("certainty", "")).strip().lower()
    return certainty if certainty else "<none>"


def _distance_scalar(value: Any) -> float:
    """
    Dependence distances may be plain integers or strings/lists.  Use the first
    integer as a compact directional signal and leave unknown distance at zero.
    """
    if value is None:
        return 0.0
    match = INTEGER_RE.search(str(value))
    return _signed_log1p(match.group(0)) if match else 0.0


def edge_numeric_features(attrs: Mapping[str, Any]) -> list[float]:
    position = _as_int(attrs.get("position"), 0)
    operand_index = _as_int(attrs.get("operand_index"), -1)
    access_function = attrs.get("access_function", "")
    return [
        _signed_log1p(max(-1, min(position, 1024))),
        _signed_log1p(max(-1, min(operand_index, 1024))),
        _as_bool01(str(access_function).strip() not in {"", "0", "None"}),
        _distance_scalar(attrs.get("distance")),
    ]


EDGE_NUMERIC_NAMES = [
    "signed_log1p_position",
    "signed_log1p_operand_index",
    "has_affine_access_function",
    "signed_log1p_dependence_distance",
]


# ---------------------------------------------------------------------------
# Pragma-action scope mapping.
# ---------------------------------------------------------------------------

def _attached_pragma_nodes(
    graph: nx.Graph,
    node: str,
) -> dict[str, str]:
    """
    Return pragma-kind -> pragma-node for every pragma attached to ``node``.

    mlir_graph_gen.py writes explicit pragma edges.  We accept either edge
    direction because the graph stores semantic reverse relations for some
    edge families and older generated files may differ in orientation.
    """
    attached: dict[str, str] = {}

    def inspect(neighbor: str, attrs: Mapping[str, Any]) -> None:
        neighbor_attrs = graph.nodes[neighbor]
        kind = pragma_kind(neighbor_attrs)
        if kind == "NONE":
            return
        if _as_int(attrs.get("flow"), -1) == FLOW_PRAGMA or "pragma" in edge_role(attrs).lower():
            attached[kind] = neighbor

    if graph.is_multigraph():
        for _, neighbor, _, attrs in graph.out_edges(node, keys=True, data=True):
            inspect(neighbor, attrs)
        for neighbor, _, _, attrs in graph.in_edges(node, keys=True, data=True):
            inspect(neighbor, attrs)
    else:
        for _, neighbor, attrs in graph.out_edges(node, data=True):
            inspect(neighbor, attrs)
        for neighbor, _, attrs in graph.in_edges(node, data=True):
            inspect(neighbor, attrs)

    # Fallback for graphs where the explicit pragma edge retained a legacy flow
    # category: accept direct pragma neighbors, but only direct neighbors.
    if not attached:
        for neighbor in set(graph.predecessors(node)) | set(graph.successors(node)):
            kind = pragma_kind(graph.nodes[neighbor])
            if kind != "NONE":
                attached[kind] = neighbor

    return attached


def _point_value(point: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    value = point.get(key, default)
    if value is None:
        return default
    return value


def pragma_vector_from_node(
    pragma_attrs: Mapping[str, Any],
    point: Mapping[str, Any],
) -> list[int]:
    vector = [0, 0, 0, 0, 0]
    kind = pragma_kind(pragma_attrs)
    full_text = str(pragma_attrs.get("full_text", ""))
    action_id = str(pragma_attrs.get("action_id", "")).strip()

    keys = AUTO_KEY_RE.findall(full_text)
    if not keys and ACTION_ID_RE.fullmatch(action_id):
        if kind == "PIPELINE":
            keys = [f"_PIPE_{action_id}"]
        elif kind == "UNROLL":
            keys = [f"_UNROLL_{action_id}"]
        elif kind == "ARRAY_PARTITION":
            keys = [
                f"_ARRAY_T_{action_id}",
                f"_ARRAY_F_{action_id}",
                f"_ARRAY_D_{action_id}",
            ]

    for key in keys:
        value = _point_value(point, key, 0)
        if key.startswith("_PIPE_"):
            vector[PIPELINE_COL] = _as_int(value, 0)
        elif key.startswith("_UNROLL_"):
            vector[UNROLL_COL] = _as_int(value, 0)
        elif key.startswith("_ARRAY_T_"):
            if isinstance(value, str):
                vector[PARTITION_TYPE_COL] = ARRAY_TYPE_ENCODING.get(
                    value.strip().lower(), 0
                )
            else:
                vector[PARTITION_TYPE_COL] = _as_int(value, 0)
        elif key.startswith("_ARRAY_F_"):
            vector[PARTITION_FACTOR_COL] = _as_int(value, 0)
        elif key.startswith("_ARRAY_D_"):
            vector[PARTITION_DIM_COL] = _as_int(value, 0)

    return vector


def build_scope_masks_and_dynamic_pragmas(
    graph: nx.Graph,
    ordered_nodes: Sequence[str],
    point: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    point = point or {}

    context: list[bool] = []
    pragma_nodes: list[bool] = []
    pragma_scope: list[bool] = []
    pseudo_nodes: list[bool] = []
    array_scope_nodes: list[bool] = []
    pipeline_scope: list[bool] = []
    unroll_scope: list[bool] = []
    partition_scope: list[bool] = []
    all_scopes: list[bool] = []
    icmp_nodes: list[bool] = []
    pragma_per_node: list[list[int]] = []

    for node in ordered_nodes:
        attrs = graph.nodes[node]
        kind = node_kind(attrs)
        is_pragma = kind == "pragma"
        is_pseudo = kind == "pseudo_block"
        is_array_scope = kind == "array_scope"

        attached = {} if is_pragma else _attached_pragma_nodes(graph, node)
        has_pipeline = "PIPELINE" in attached
        has_unroll = "UNROLL" in attached
        has_partition = "ARRAY_PARTITION" in attached
        is_action_scope = has_pipeline or has_unroll or has_partition

        vector = [0, 0, 0, 0, 0]
        for pragma_node in attached.values():
            candidate = pragma_vector_from_node(
                graph.nodes[pragma_node], point
            )
            # Each kind writes disjoint columns, so max safely merges values.
            vector = [max(a, b) for a, b in zip(vector, candidate)]

        pragma_nodes.append(is_pragma)
        pseudo_nodes.append(is_pseudo)
        array_scope_nodes.append(is_array_scope)
        pipeline_scope.append(has_pipeline)
        unroll_scope.append(has_unroll)
        partition_scope.append(has_partition)
        pragma_scope.append(is_action_scope)
        all_scopes.append(is_pseudo or is_array_scope or is_action_scope)

        # Structural pseudo/action nodes are pooled separately from program
        # context nodes in the existing model.
        context.append(not (is_pragma or is_pseudo or is_array_scope))

        text = str(attrs.get("text", "")).lower()
        full_text = str(attrs.get("full_text", "")).lower()
        icmp_nodes.append("cmp" in text or "icmp" in full_text)
        pragma_per_node.append(vector)

    return {
        "X_contextnids": torch.tensor(context, dtype=torch.bool),
        "X_pragmanids": torch.tensor(pragma_nodes, dtype=torch.bool),
        "X_pragmascopenids": torch.tensor(pragma_scope, dtype=torch.bool),
        "X_pseudonids": torch.tensor(pseudo_nodes, dtype=torch.bool),
        "X_arrayscopenids": torch.tensor(array_scope_nodes, dtype=torch.bool),
        "X_pipeline_scopeids": torch.tensor(pipeline_scope, dtype=torch.bool),
        "X_unroll_scopeids": torch.tensor(unroll_scope, dtype=torch.bool),
        "X_array_partition_scopeids": torch.tensor(
            partition_scope, dtype=torch.bool
        ),
        "X_scopenids": torch.tensor(all_scopes, dtype=torch.bool),
        "X_icmpnids": torch.tensor(icmp_nodes, dtype=torch.bool),
        "X_pragma_per_node": torch.tensor(
            pragma_per_node, dtype=torch.int16
        ),
    }


# ---------------------------------------------------------------------------
# Encoder fitting and graph tensor construction.
# ---------------------------------------------------------------------------

NODE_CATEGORICAL_FIELDS = (
    "node_kind",
    "exact_text",
    "op_family",
    "canonical_type",
    "pragma_kind",
    "ssa_kind",
    "feature_kind",
)

EDGE_CATEGORICAL_FIELDS = (
    "flow",
    "role",
    "certainty",
)


def node_categorical_row(attrs: Mapping[str, Any]) -> list[str]:
    return [
        node_kind(attrs),
        str(attrs.get("text", "<none>")),
        canonical_op_family(attrs),
        canonical_type(attrs),
        pragma_kind(attrs),
        str(attrs.get("ssa_kind", "<none>") or "<none>"),
        str(attrs.get("feature_kind", "<none>") or "<none>"),
    ]


def edge_categorical_row(attrs: Mapping[str, Any]) -> list[str]:
    return [
        str(_as_int(attrs.get("flow"), -1)),
        edge_role(attrs),
        edge_certainty(attrs),
    ]


def fit_encoders(graph_files: Sequence[Path]) -> dict[str, Any]:
    node_tokens: dict[str, set[str]] = {
        field: set() for field in NODE_CATEGORICAL_FIELDS
    }
    edge_tokens: dict[str, set[str]] = {
        field: set() for field in EDGE_CATEGORICAL_FIELDS
    }

    for graph_file in graph_files:
        graph = nx.read_gexf(graph_file)
        ordered_nodes, _ = _node_id_order(graph)

        for node in ordered_nodes:
            for field, token in zip(
                NODE_CATEGORICAL_FIELDS,
                node_categorical_row(graph.nodes[node]),
            ):
                node_tokens[field].add(token)

        for _, _, _, attrs in _iter_edges(graph):
            for field, token in zip(
                EDGE_CATEGORICAL_FIELDS,
                edge_categorical_row(attrs),
            ):
                edge_tokens[field].add(token)

        del graph
        gc.collect()

    return {
        "node": {
            field: _fit_onehot(node_tokens[field])
            for field in NODE_CATEGORICAL_FIELDS
        },
        "edge": {
            field: _fit_onehot(edge_tokens[field])
            for field in EDGE_CATEGORICAL_FIELDS
        },
    }


def _encode_categorical_columns(
    rows: Sequence[Sequence[str]],
    fields: Sequence[str],
    encoders: Mapping[str, OneHotEncoder],
):
    matrices = []
    for column, field in enumerate(fields):
        values = np.asarray(
            [row[column] for row in rows], dtype=object
        ).reshape(-1, 1)
        matrices.append(encoders[field].transform(values))
    return matrices


def encode_static_graph(
    graph: nx.Graph,
    graph_name: str,
    kernel_name: str,
    encoders: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_nodes, node_to_index = _node_id_order(graph)

    node_categories = [
        node_categorical_row(graph.nodes[node])
        for node in ordered_nodes
    ]
    node_numeric = np.asarray(
        [node_numeric_features(graph.nodes[node]) for node in ordered_nodes],
        dtype=np.float32,
    )
    node_matrices = _encode_categorical_columns(
        node_categories,
        NODE_CATEGORICAL_FIELDS,
        encoders["node"],
    )
    node_matrix = hstack(node_matrices).toarray().astype(np.float32)
    x = torch.from_numpy(
        np.concatenate([node_matrix, node_numeric], axis=1)
    ).contiguous()

    edge_pairs: list[list[int]] = []
    edge_categories: list[list[str]] = []
    edge_numeric: list[list[float]] = []

    for source, target, _, attrs in _iter_edges(graph):
        edge_pairs.append([
            node_to_index[source],
            node_to_index[target],
        ])
        edge_categories.append(edge_categorical_row(attrs))
        edge_numeric.append(edge_numeric_features(attrs))

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs, dtype=torch.long
        ).t().contiguous()
        edge_matrices = _encode_categorical_columns(
            edge_categories,
            EDGE_CATEGORICAL_FIELDS,
            encoders["edge"],
        )
        edge_matrix = hstack(edge_matrices).toarray().astype(np.float32)
        edge_numeric_array = np.asarray(edge_numeric, dtype=np.float32)
        edge_attr = torch.from_numpy(
            np.concatenate(
                [edge_matrix, edge_numeric_array],
                axis=1,
            )
        ).to(torch.float16).contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        categorical_width = sum(
            len(encoders["edge"][field].categories_[0])
            for field in EDGE_CATEGORICAL_FIELDS
        )
        edge_attr = torch.empty(
            (0, categorical_width + len(EDGE_NUMERIC_NAMES)),
            dtype=torch.float16,
        )

    masks = build_scope_masks_and_dynamic_pragmas(
        graph, ordered_nodes, point=None
    )

    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite node features in {graph_name}.")
    if not torch.isfinite(edge_attr.float()).all():
        raise RuntimeError(f"Non-finite edge features in {graph_name}.")

    payload = {
        "graph_name": graph_name,
        "kernel_name": kernel_name,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    for key, value in masks.items():
        if key != "X_pragma_per_node":
            payload[key] = value
    return payload


# ---------------------------------------------------------------------------
# Flat and per-node pragma tensors.
# ---------------------------------------------------------------------------

def point_to_ordered_values(point: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for key, value in sorted(point.items()):
        if not key.startswith((
            "_PIPE_", "_UNROLL_", "_ARRAY_T_",
            "_ARRAY_F_", "_ARRAY_D_",
        )):
            continue
        if key.startswith("_ARRAY_T_") and isinstance(value, str):
            values.append(ARRAY_TYPE_ENCODING.get(value.strip().lower(), 0))
        else:
            values.append(_as_int(value, 0))
    return values


def normalize_targets(
    perf: float,
    area: float,
    reference_perf: float,
) -> tuple[float, float, float]:
    epsilon = float(getattr(FLAGS, "epsilon", 1e-6))
    norm_method = str(getattr(FLAGS, "norm_method", "log2"))

    if norm_method == "log2":
        perf_y = math.log2(perf + epsilon)
    elif norm_method == "const":
        perf_y = perf * float(getattr(FLAGS, "normalizer", 1.0))
    elif norm_method == "off":
        perf_y = perf
    elif "speedup" in norm_method:
        if perf <= 0:
            perf_y = 0.0
        else:
            speedup = float(getattr(FLAGS, "normalizer", 1.0)) / perf
            perf_y = (
                math.log2(speedup + epsilon)
                if norm_method == "speedup-log2"
                else speedup
            )
    else:
        raise NotImplementedError(
            f"Unsupported norm_method: {norm_method}"
        )

    area_safe = area if area > 0 else epsilon
    if norm_method == "const":
        area_y = area_safe * float(
            getattr(FLAGS, "util_normalizer", 1.0)
        )
    elif norm_method == "off":
        area_y = area_safe
    else:
        area_y = math.log2(area_safe + epsilon)

    kernel_speedup = (
        math.log2(reference_perf / perf)
        if reference_perf > 0 and perf > 0
        else 0.0
    )
    return perf_y, area_y, kernel_speedup


# ---------------------------------------------------------------------------
# GEXF discovery and kernel matching.
# ---------------------------------------------------------------------------

def discover_graph_files() -> list[Path]:
    gexf_dir = _first_existing_dir(
        GEXF_DIR_CANDIDATES, "MLIR graph directory"
    )
    files = sorted(gexf_dir.glob("*.gexf"))
    if not files:
        raise RuntimeError(f"No .gexf files found in {gexf_dir}")
    return files


def resolve_kernel_from_graph(graph_file: Path) -> str:
    stem = graph_file.stem

    # Longest-first prevents a shorter kernel name from matching a longer one.
    candidates = sorted(ALL_KERNEL, key=len, reverse=True)
    for kernel in candidates:
        variants = _name_variants(kernel)
        if any(stem == variant or stem.startswith(variant + "_")
               for variant in variants):
            return kernel

    # The generated graph often has exactly the application-directory name.
    for kernel in candidates:
        if stem in _name_variants(kernel):
            return kernel

    raise RuntimeError(
        f"Could not map MLIR graph '{graph_file.name}' to ALL_KERNEL."
    )


# ---------------------------------------------------------------------------
# Compact PyG dataset.
# ---------------------------------------------------------------------------

class MyOwnDataset(Dataset):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        data_files=None,
    ):
        self.records = (
            data_files
            if data_files is not None
            else torch.load(INDEX_PATH, weights_only=False)
        )
        self._graph_cache: dict[str, dict[str, Any]] = {}
        self._point_cache: dict[str, dict[str, Any]] = {}
        super().__init__(
            root=str(SAVE_DIR),
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return str(SAVE_DIR)

    @property
    def processed_file_names(self):
        return ["index.pt"]

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.records)

    def __len__(self):
        return self.len()

    def _load_graph(self, graph_name: str) -> dict[str, Any]:
        if graph_name not in self._graph_cache:
            payload = torch.load(
                GRAPH_DIR / f"{graph_name}.pt",
                weights_only=False,
            )
            self._graph_cache[graph_name] = payload
        return self._graph_cache[graph_name]

    def _load_points(self, graph_name: str) -> dict[str, Any]:
        if graph_name not in self._point_cache:
            payload = torch.load(
                POINT_DIR / f"{graph_name}.pt",
                weights_only=False,
            )
            self._point_cache[graph_name] = payload
        return self._point_cache[graph_name]

    def get(self, index: int) -> Data:
        record = self.records[index]
        graph_name = record["graph_name"]
        local_idx = record["local_idx"]

        graph = self._load_graph(graph_name)
        points = self._load_points(graph_name)

        kwargs = {
            "gname": graph["kernel_name"],
            "graph_name": graph_name,
            "kernel": graph["kernel_name"],
            "key": points["keys"][local_idx],
            "x": graph["x"].float(),
            "edge_index": graph["edge_index"],
            "edge_attr": graph["edge_attr"].float(),
            "X_contextnids": graph["X_contextnids"].float(),
            "X_pragmanids": graph["X_pragmanids"].float(),
            "X_pragmascopenids": graph["X_pragmascopenids"].float(),
            "X_pseudonids": graph["X_pseudonids"].float(),
            "X_arrayscopenids": graph["X_arrayscopenids"].float(),
            "X_pipeline_scopeids": graph[
                "X_pipeline_scopeids"
            ].float(),
            "X_unroll_scopeids": graph[
                "X_unroll_scopeids"
            ].float(),
            "X_array_partition_scopeids": graph[
                "X_array_partition_scopeids"
            ].float(),
            "X_scopenids": graph["X_scopenids"].float(),
            "X_icmpnids": graph["X_icmpnids"].float(),
            "X_pragma_per_node": points[
                "X_pragma_per_node"
            ][local_idx].float(),
            "pragmas": points["pragmas"][local_idx].float().unsqueeze(0),
        }

        if str(getattr(FLAGS, "task", "regression")) == "regression":
            kwargs.update({
                "perf": points["perf"][local_idx].view(1).float(),
                "actual_perf": points[
                    "actual_perf"
                ][local_idx].view(1).float(),
                "kernel_speedup": points[
                    "kernel_speedup"
                ][local_idx].view(1).float(),
                "area": points["area"][local_idx].view(1).float(),
                "actual_area": points[
                    "actual_area"
                ][local_idx].view(1).float(),
            })
        else:
            kwargs["perf"] = points["perf"][local_idx].view(1).long()

        data = Data(**kwargs)

        for name in ("x", "edge_attr", "X_pragma_per_node", "pragmas"):
            tensor = getattr(data, name)
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"Non-finite tensor {name} in {graph_name}, "
                    f"point {local_idx}."
                )
        return data


# ---------------------------------------------------------------------------
# Dataset split helpers kept compatible with train_GNN.py.
# ---------------------------------------------------------------------------

def split_dataset(dataset, train, val, dataset_test=None):
    records = dataset.records
    splits = random_split(
        records,
        [train, val, len(records) - train - val],
        generator=torch.Generator().manual_seed(
            int(getattr(FLAGS, "random_seed", 0))
        ),
    )
    train_records = [splits[0][i] for i in range(len(splits[0]))]
    val_records = [splits[1][i] for i in range(len(splits[1]))]
    test_records = (
        [splits[2][i] for i in range(len(splits[2]))]
        if dataset_test is None
        else dataset_test
    )
    return [
        MyOwnDataset(data_files=train_records),
        MyOwnDataset(data_files=val_records),
        MyOwnDataset(data_files=test_records),
    ]


def split_dataset_resample(dataset, train, val, test, test_id=0):
    records = dataset.records
    generator = torch.Generator().manual_seed(100)
    num_batches = int(round(1.0 / test))
    sizes = [int(len(records) * test)] * num_batches
    sizes[-1] = len(records) - sum(sizes[:-1])
    folds = random_split(records, sizes, generator=generator)

    test_records = [
        folds[test_id][i] for i in range(len(folds[test_id]))
    ]
    remaining = [
        folds[fold][i]
        for fold in range(num_batches)
        if fold != test_id
        for i in range(len(folds[fold]))
    ]
    train_size = int(len(remaining) * train / (train + val))
    train_fold, val_fold = random_split(
        remaining,
        [train_size, len(remaining) - train_size],
        generator=generator,
    )
    return [
        MyOwnDataset(
            data_files=[train_fold[i] for i in range(len(train_fold))]
        ),
        MyOwnDataset(
            data_files=[val_fold[i] for i in range(len(val_fold))]
        ),
        MyOwnDataset(data_files=test_records),
    ]


def get_kernel_samples(dataset):
    target = getattr(FLAGS, "target_kernel", None)
    if target is None:
        return dataset
    records = [
        record for record in dataset.records
        if record.get("kernel_name", "").startswith(target)
        or record["graph_name"].startswith(target)
    ]
    return MyOwnDataset(data_files=records)


def split_train_test_kernel(dataset):
    test_kernels = getattr(FLAGS, "test_kernels", None)
    if test_kernels is None:
        return {"train": dataset, "test": None}
    if isinstance(test_kernels, str):
        test_kernels = [
            item.strip()
            for item in test_kernels.split(",")
            if item.strip()
        ]
    test_set = set(test_kernels)
    train_records = [
        record for record in dataset.records
        if record.get("kernel_name") not in test_set
    ]
    test_records = [
        record for record in dataset.records
        if record.get("kernel_name") in test_set
    ]
    return {
        "train": MyOwnDataset(data_files=train_records),
        "test": MyOwnDataset(data_files=test_records),
    }


# ---------------------------------------------------------------------------
# Full two-pass dataset construction.
# ---------------------------------------------------------------------------

def _keep_result(result: CSVResult) -> bool:
    if str(getattr(FLAGS, "task", "regression")) != "regression":
        return True
    if bool(getattr(FLAGS, "invalid", False)):
        return True
    return result.perf >= float(
        getattr(FLAGS, "min_allowed_latency", 0.1)
    )


def _write_schema(
    encoders: Mapping[str, Any],
    node_dim: int,
    edge_dim: int,
) -> None:
    schema = {
        "dataset": DATASET_NAME,
        "node_categorical_fields": list(NODE_CATEGORICAL_FIELDS),
        "node_numeric_fields": NODE_NUMERIC_NAMES,
        "edge_categorical_fields": list(EDGE_CATEGORICAL_FIELDS),
        "edge_numeric_fields": EDGE_NUMERIC_NAMES,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "node_categories": {
            field: [
                str(value)
                for value in encoders["node"][field].categories_[0]
            ]
            for field in NODE_CATEGORICAL_FIELDS
        },
        "edge_categories": {
            field: [
                str(value)
                for value in encoders["edge"][field].categories_[0]
            ]
            for field in EDGE_CATEGORICAL_FIELDS
        },
    }
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def get_data_list():
    """
    Build or load the compact MLIR MailoHLS dataset.

    Returns:
        dataset: MyOwnDataset
        init_feat_dict:
            graph_name -> [kernel-local pragma length, global max length]
    """
    graph_files = discover_graph_files()

    # Exclude unsupported graphs by presence, rather than assuming 55/55.
    graph_records: list[tuple[Path, str]] = []
    for graph_file in graph_files:
        kernel = resolve_kernel_from_graph(graph_file)
        graph_records.append((graph_file, kernel))

    if not bool(getattr(FLAGS, "force_regen", False)):
        if not INDEX_PATH.is_file():
            raise FileNotFoundError(
                f"{INDEX_PATH} does not exist. Set force_regen=True once."
            )
        return (
            MyOwnDataset(),
            torch.load(PRAGMA_DIM_PATH, weights_only=False)
            if PRAGMA_DIM_PATH.is_file()
            else None,
        )

    # Fit MLIR-specific encoders on training graph structure only when an
    # explicit held-out kernel set is configured.  This avoids vocabulary
    # leakage in family-level zero-shot evaluation.  Unknown exact operation
    # tokens are safely ignored by OneHotEncoder, while coarse operation
    # families and numeric semantic features remain available.
    configured_test_kernels = getattr(FLAGS, "test_kernels", None)
    if isinstance(configured_test_kernels, str):
        configured_test_kernels = {
            item.strip()
            for item in configured_test_kernels.split(",")
            if item.strip()
        }
    elif configured_test_kernels is None:
        configured_test_kernels = set()
    else:
        configured_test_kernels = set(configured_test_kernels)

    encoder_fit_files = [
        path
        for path, kernel in graph_records
        if kernel not in configured_test_kernels
    ]
    if not encoder_fit_files:
        raise RuntimeError(
            "No training graphs remain for fitting MLIR encoders after "
            "applying FLAGS.test_kernels."
        )
    encoders = fit_encoders(encoder_fit_files)

    # Metadata pass: determine global pragma padding width.
    point_results: dict[str, list[CSVResult]] = {}
    local_pragma_dims: dict[str, int] = {}
    max_pragma_length = 0

    for graph_file, kernel in graph_records:
        graph_name = graph_file.stem
        results = [
            result for result in load_csv_results(kernel)
            if _keep_result(result)
        ]
        if not results:
            print(f"[WARN] No valid design points for {kernel}; skipping.")
            continue

        lengths = {
            len(point_to_ordered_values(result.point))
            for result in results
        }
        if len(lengths) != 1:
            raise RuntimeError(
                f"Inconsistent pragma vector lengths for {kernel}: "
                f"{sorted(lengths)}"
            )

        local_dim = next(iter(lengths))
        point_results[graph_name] = results
        local_pragma_dims[graph_name] = local_dim
        max_pragma_length = max(max_pragma_length, local_dim)

    if max_pragma_length <= 0:
        raise RuntimeError("No valid MLIR design points were found.")

    tmp_dir = Path(str(SAVE_DIR) + "_tmp")
    if tmp_dir.exists():
        raise RuntimeError(
            f"Temporary directory already exists: {tmp_dir}. "
            "Inspect and remove it before rerunning."
        )
    graph_tmp = tmp_dir / "graphs"
    point_tmp = tmp_dir / "points"
    graph_tmp.mkdir(parents=True)
    point_tmp.mkdir(parents=True)

    global_index: list[dict[str, Any]] = []
    init_feat_dict: dict[str, list[int]] = {}
    first_node_dim: int | None = None
    first_edge_dim: int | None = None

    try:
        for graph_file, kernel in graph_records:
            graph_name = graph_file.stem
            results = point_results.get(graph_name)
            if not results:
                continue

            graph = nx.read_gexf(graph_file)
            ordered_nodes, _ = _node_id_order(graph)
            static_payload = encode_static_graph(
                graph,
                graph_name=graph_name,
                kernel_name=kernel,
                encoders=encoders,
            )
            torch.save(
                static_payload,
                graph_tmp / f"{graph_name}.pt",
            )

            first_node_dim = first_node_dim or static_payload["x"].shape[1]
            first_edge_dim = first_edge_dim or static_payload[
                "edge_attr"
            ].shape[1]

            reference_perf = max(
                (result.perf for result in results if result.perf > 0),
                default=0.0,
            )

            keys: list[str] = []
            pragmas: list[torch.Tensor] = []
            per_node_pragmas: list[torch.Tensor] = []
            perf_values: list[float] = []
            actual_perf_values: list[float] = []
            speedup_values: list[float] = []
            area_values: list[float] = []
            actual_area_values: list[float] = []

            for local_idx, result in enumerate(results):
                flat = point_to_ordered_values(result.point)
                if len(flat) != local_pragma_dims[graph_name]:
                    raise RuntimeError(
                        f"Pragma length changed for {graph_name}."
                    )
                padded = flat + [0] * (
                    max_pragma_length - len(flat)
                )

                masks = build_scope_masks_and_dynamic_pragmas(
                    graph,
                    ordered_nodes,
                    point=result.point,
                )

                # Every non-zero action must reach at least one scope node.
                nonzero_point = any(value != 0 for value in flat)
                if nonzero_point and not torch.any(
                    masks["X_pragma_per_node"] != 0
                ):
                    raise RuntimeError(
                        f"Design point {result.row_idx} for {kernel} has "
                        "non-zero directives but no MLIR action scope received "
                        "them. Check pragma-edge/action mapping."
                    )

                perf_y, area_y, speedup = normalize_targets(
                    result.perf,
                    result.area,
                    reference_perf,
                )

                keys.append(f"csvrow_{result.row_idx}")
                pragmas.append(
                    torch.tensor(padded, dtype=torch.int16)
                )
                per_node_pragmas.append(
                    masks["X_pragma_per_node"]
                )
                perf_values.append(perf_y)
                actual_perf_values.append(result.perf)
                speedup_values.append(speedup)
                area_values.append(area_y)
                actual_area_values.append(result.area)

                global_index.append({
                    "graph_name": graph_name,
                    "kernel_name": kernel,
                    "local_idx": local_idx,
                })

            point_payload = {
                "graph_name": graph_name,
                "kernel_name": kernel,
                "keys": keys,
                "pragmas": torch.stack(pragmas, dim=0),
                "X_pragma_per_node": torch.stack(
                    per_node_pragmas, dim=0
                ),
                "perf": torch.tensor(
                    perf_values, dtype=torch.float32
                ),
                "actual_perf": torch.tensor(
                    actual_perf_values, dtype=torch.float32
                ),
                "kernel_speedup": torch.tensor(
                    speedup_values, dtype=torch.float32
                ),
                "area": torch.tensor(
                    area_values, dtype=torch.float32
                ),
                "actual_area": torch.tensor(
                    actual_area_values, dtype=torch.float32
                ),
            }
            torch.save(
                point_payload,
                point_tmp / f"{graph_name}.pt",
            )

            init_feat_dict[graph_name] = [
                local_pragma_dims[graph_name],
                max_pragma_length,
            ]
            print(
                f"[OK] {kernel}: nodes={graph.number_of_nodes()}, "
                f"edges={graph.number_of_edges()}, "
                f"points={len(results)}, pragma_dim="
                f"{local_pragma_dims[graph_name]}"
            )

            del graph, static_payload, point_payload
            gc.collect()

        torch.save(global_index, tmp_dir / "index.pt")
        torch.save(init_feat_dict, tmp_dir / "pragma_dim.pt")
        _save_pickle(encoders, tmp_dir / "encoders.pkl")

        # Schema is written inside tmp_dir before atomic replacement.
        global SCHEMA_PATH
        previous_schema_path = SCHEMA_PATH
        SCHEMA_PATH = tmp_dir / "feature_schema.json"
        _write_schema(
            encoders,
            node_dim=int(first_node_dim or 0),
            edge_dim=int(first_edge_dim or 0),
        )
        SCHEMA_PATH = previous_schema_path

        if SAVE_DIR.exists():
            rmtree(SAVE_DIR)
        tmp_dir.rename(SAVE_DIR)

    except Exception:
        # Keep the temporary directory for forensic inspection.
        print(f"[ERROR] Dataset build stopped; partial data kept at {tmp_dir}")
        raise

    dataset = MyOwnDataset()
    print(
        f"[DONE] Built {len(dataset)} design points from "
        f"{len(init_feat_dict)} MLIR graphs."
    )
    print(
        f"[DIMS] node_features={first_node_dim}, "
        f"edge_features={first_edge_dim}, "
        f"max_pragma_length={max_pragma_length}"
    )
    return dataset, init_feat_dict


def load_encoders():
    return _load_pickle(ENCODER_PATH)


if __name__ == "__main__":
    dataset, pragma_dim = get_data_list()
    print(f"Dataset samples: {len(dataset)}")
    print(f"Encoded graphs: {len(pragma_dim or {})}")