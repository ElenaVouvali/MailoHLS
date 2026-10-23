#!/usr/bin/env python3
"""
Rerank LLM-generated HLS directive candidates with the frozen GNN QoR surrogate.

Expected input JSONL (one row per unique candidate), compatible with the user's
flatten_unique_candidates_for_gnn() helper:
    {
      "kernel_name": "...",
      "obj_mode": "PARETO_LATENCY_EXTREME",
      "w_lat": 1.0,
      "w_area": 0.0,
      "sample_id": 0,
      "prediction": "<L1>\nauto{_PIPE_L1} = 1\n...",
      "sequence_score": -0.123,
      "sequence_sum_logprob": -12.34
    }

Output JSONL (one row per kernel/objective), compatible with direct concatenation
into the final synthesis-ready file:
    {
      "kernel_name": "...",
      "obj_mode": "PARETO_LATENCY_EXTREME",
      "prediction": "<L1>\nauto{_PIPE_L1} = 1\n...",
      "sample_id": 0,
      "gnn_pred_perf": 12.34,
      "gnn_pred_area": 8.76
    }
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ASSIGN_RE = re.compile(
    r"^(auto\{_([A-Z0-9]+(?:_[A-Z0-9]+)*)_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)

LABEL_KEY_RE = re.compile(r"_L(\d+)\b", re.IGNORECASE)
MAX_COMPLETE_ELEMENTS = 1024
COMPLETE_FALLBACK_FACTOR = 16
MAX_FALLBACK_PARTITION_FACTOR = 64


def key_to_label(key: str) -> str:
    m = LABEL_KEY_RE.search(key)
    if m is None:
        raise ValueError(f"Could not extract label from key: {key}")
    return f"L{int(m.group(1))}"


def parse_kernel_constraints(kernel_info_path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    if not kernel_info_path.exists():
        raise FileNotFoundError(f"kernel_info.txt not found: {kernel_info_path}")

    with open(kernel_info_path, "r", encoding="utf-8") as f:
        # first non-empty line = top func
        while True:
            line = f.readline()
            if not line:
                return out
            if line.strip():
                break

        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            fields = [x.strip() for x in raw.split(",")]
            if len(fields) < 3:
                continue

            label = fields[0]
            kind = fields[1].lower()

            if kind == "loop":
                try:
                    bound = int(fields[2])
                except Exception:
                    continue
                out[label] = {"kind": "loop", "bound": bound}

            elif kind == "array":
                array_name = fields[2]
                dim_bounds = {}
                i = 3
                while i + 1 < len(fields):
                    try:
                        dim_idx = int(fields[i])
                        dim_bound = int(fields[i + 1])
                    except Exception:
                        break
                    dim_bounds[dim_idx] = dim_bound
                    i += 2

                if dim_bounds:
                    out[label] = {
                        "kind": "array",
                        "array_name": array_name,
                        "dim_bounds": dim_bounds,
                    }

    return out


def legalize_unroll_value(factor: int, loop_bound: Optional[int]) -> Optional[int]:
    factor = int(factor)
    if loop_bound is not None and loop_bound > 0:
        factor = min(factor, int(loop_bound))
    return factor


def choose_fallback_partition_factor(
    predicted_factor: int,
    bound: int,
    max_factor: int = MAX_FALLBACK_PARTITION_FACTOR,
) -> Optional[int]:
    """
    Keep exactly aligned with batch_hls_from_predictions.py.

    Used when type=complete is unsafe because the selected dimension is too large.
    If the LLM predicted factor=0/1, use max_factor as the fallback partition factor,
    then clamp to the real dimension bound.
    """
    factor = int(predicted_factor)
    if factor <= 1:
        factor = int(max_factor)
    factor = min(factor, int(bound), int(max_factor))
    return factor


def legalize_pipeline_value(ii: int) -> int:
    ii = int(ii)
    if ii < 0:
        ii = 0
    return ii


def legalize_unroll_value(factor: int, loop_bound: Optional[int]) -> int:
    factor = int(factor)
    if factor < 0:
        factor = 0
    if loop_bound is not None and loop_bound > 0:
        factor = min(factor, int(loop_bound))
    return factor


def legalize_array_partition(
    part_type: str,
    factor: int,
    dim: int,
    dim_bounds: Dict[int, int],
    max_complete_elements: int = MAX_COMPLETE_ELEMENTS,
    complete_fallback_type: str = "cyclic",
    max_fallback_partition_factor: int = MAX_FALLBACK_PARTITION_FACTOR,
) -> Tuple[Optional[dict], List[str]]:
    """
    Must stay behaviorally identical to batch_hls_from_predictions.py.

    Rules:
      - invalid type -> drop
      - complete ignores factor and emits no factor in C
      - block/cyclic require factor > 1
      - invalid dim is clamped to first valid dim
      - factor > bound is clamped
      - complete on very large dim is downgraded to cyclic/block fallback
    """
    notes: List[str] = []
    part_type = str(part_type).strip().lower()
    factor = int(factor)
    dim = int(dim)

    if part_type not in {"block", "cyclic", "complete"}:
        notes.append(f"invalid array type '{part_type}' -> drop pragma")
        return None, notes

    if not dim_bounds:
        notes.append("missing dim bounds -> sanitize array pragma")

        if dim <= 0:
            notes.append(f"dim={dim} invalid without bounds -> clamped to 1")
            dim = 1

        if part_type == "complete":
            return {"type": "complete", "dim": dim}, notes

        if factor <= 1:
            notes.append(f"factor={factor} invalid/trivial -> drop partition pragma")
            return None, notes

        return {"type": part_type, "factor": factor, "dim": dim}, notes

    valid_dims = sorted(dim_bounds.keys())

    if dim not in dim_bounds:
        new_dim = valid_dims[0]
        notes.append(f"dim {dim} invalid -> clamped to {new_dim}")
        dim = new_dim

    bound = int(dim_bounds[dim])

    if bound <= 1:
        notes.append(f"dim {dim} bound={bound} -> drop partition pragma")
        return None, notes

    if part_type == "complete":
        if bound <= max_complete_elements:
            return {"type": "complete", "dim": dim}, notes

        fallback_factor = choose_fallback_partition_factor(
            predicted_factor=factor,
            bound=bound,
            max_factor=max_fallback_partition_factor,
        )

        if fallback_factor is None:
            notes.append(
                f"complete partition on bound={bound} exceeds max_complete_elements={max_complete_elements}, "
                f"but no valid fallback factor -> drop pragma"
            )
            return None, notes

        notes.append(
            f"complete partition on bound={bound} exceeds max_complete_elements={max_complete_elements} "
            f"-> downgrade to {complete_fallback_type} factor={fallback_factor}"
        )
        return {
            "type": complete_fallback_type,
            "factor": fallback_factor,
            "dim": dim,
        }, notes

    # block / cyclic
    if factor <= 1:
        notes.append(f"factor={factor} invalid/trivial -> drop partition pragma")
        return None, notes

    if factor > bound:
        notes.append(f"factor={factor} > bound={bound} -> clamped to {bound}")
        factor = bound

    if factor <= 1:
        notes.append(f"factor became {factor} after clamp -> drop partition pragma")
        return None, notes

    return {"type": part_type, "factor": factor, "dim": dim}, notes


def stable_json_sig(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def resolve_kernel_info_path(kernel_name: str, gexf_path: str, kernel_info_root: str = "") -> Path:
    gp = Path(gexf_path).resolve()
    candidates: List[Path] = []

    bases = {
        kernel_name,
        kernel_name.replace("-", "_"),
        kernel_name.replace("_", "-"),
        normalize_kname(kernel_name),
    }

    if kernel_info_root:
        root = Path(kernel_info_root).resolve()
        candidates.append(root / "kernel_info.txt")
        candidates.append(root / "meta" / "kernel_info.txt")
        for b in sorted(bases):
            candidates.append(root / b / "kernel_info.txt")
            candidates.append(root / b / "code_to_memory_outputs" / "meta" / "kernel_info.txt")

    # code_to_memory_outputs case:
    #   .../<kernel>/code_to_memory_outputs/gexf/<kernel>_processed_result.gexf
    if gp.parent.name == "gexf":
        candidates.append(gp.parent.parent / "meta" / "kernel_info.txt")

    # default harp processed case:
    #   .../harp/processed/.../<kernel>_processed_result.gexf
    for p in gp.parents:
        if p.name == "processed" and p.parent.name == "harp":
            harp_root = p.parent
            for b in sorted(bases):
                candidates.append(harp_root / b / "kernel_info.txt")
            break

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"Could not resolve kernel_info.txt for kernel='{kernel_name}' from gexf='{gexf_path}'. "
        f"Pass --kernel_info_root explicitly."
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_jsonl", type=str, required=True)
    ap.add_argument("--gnn_ckpt", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)

    ap.add_argument(
        "--objective_tag",
        type=str,
        default="",
        help=(
            "Optional fallback objective for legacy single-objective files. "
            "Not required when each JSONL row already contains obj_mode/objective/w_lat+w_area."
        ),
    )
    ap.add_argument(
        "--gexf_dir",
        type=str,
        default="/home/ubuntu/harp/processed/extended-pseudo-block-connected-hierarchy",
        help="Directory containing the processed HARP .gexf graphs.",
    )    
    ap.add_argument(
        "--kernel_info_root",
        type=str,
        default="",
        help=(
            "Optional root used to resolve kernel_info.txt. "
            "If omitted, the script first tries a sibling meta/kernel_info.txt next to the GEXF "
            "and then falls back to <harp_root>/<kernel>/kernel_info.txt."
        ),
    )
    ap.add_argument(
        "--max_pragma_length",
        type=int,
        default=93,
        help="Must match the dataset-compatible gexf_to_pt.py setting.",
    )
    ap.add_argument(
        "--temp_dir",
        type=str,
        default="",
        help="Optional temp directory for point JSON / PT artifacts. Defaults next to output_jsonl.",
    )
    ap.add_argument(
        "--keep_temps",
        action="store_true",
        help="Keep intermediate point JSON / PT files for debugging.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--disable_pragma_injection",
        action="store_true",
        help="Ablation only. By default, the reranker uses the candidate pragma values.",
    )
    ap.add_argument(
        "--domination_penalty",
        type=float,
        default=0.25,
        help="Same spirit as SFT goal ranking; mild penalty for dominated candidates.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
    )
    return ap.parse_args()


def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())


def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


def canonical_objective_from_tag(tag: str) -> str:
    t = normalize_name(tag)
    if t in {"pareto_latency_extreme", "latency_extreme", "latency", "min_latency", "min_lat"}:
        return "PARETO_LATENCY_EXTREME"
    if t in {"pareto_area_extreme", "area_extreme", "area", "min_area"}:
        return "PARETO_AREA_EXTREME"
    if t in {"pareto_knee", "knee", "balanced", "balance"}:
        return "PARETO_KNEE"
    raise ValueError(f"Unknown objective_tag: {tag}")


def normalize_weight_pair(w_lat: float, w_area: float) -> Tuple[float, float]:
    s = float(w_lat) + float(w_area)
    if s <= 0:
        raise ValueError("w_lat + w_area must be > 0")
    return float(w_lat) / s, float(w_area) / s


def mode_from_weights(w_lat: float, w_area: float) -> str:
    eps = 1e-9
    if abs(w_lat - 1.0) < eps and abs(w_area - 0.0) < eps:
        return "PARETO_LATENCY_EXTREME"
    if abs(w_lat - 0.0) < eps and abs(w_area - 1.0) < eps:
        return "PARETO_AREA_EXTREME"
    return "PARETO_KNEE"


def infer_obj_mode_from_row(row: dict, fallback_obj_mode: Optional[str] = None) -> str:
    if row.get("obj_mode"):
        return canonical_objective_from_tag(str(row["obj_mode"]))

    if row.get("objective"):
        return canonical_objective_from_tag(str(row["objective"]))

    if "w_lat" in row and "w_area" in row:
        w_lat, w_area = normalize_weight_pair(float(row["w_lat"]), float(row["w_area"]))
        return mode_from_weights(w_lat, w_area)

    if fallback_obj_mode is not None:
        return fallback_obj_mode

    raise ValueError(
        "Could not infer objective for row. Expected one of: "
        "'obj_mode', 'objective', or ('w_lat' and 'w_area')."
    )


def parse_prediction_to_point(prediction: str) -> Dict[str, Any]:
    """
    Convert canonical directive text into the point-json format expected by gexf_to_pt.py:
      auto{_PIPE_L1} = 1      -> {"_PIPE_L1": 1}
      auto{_ARRAY_T_L2} = cyclic -> {"_ARRAY_T_L2": "cyclic"}
    """
    point: Dict[str, Any] = {}

    for raw_line in prediction.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<L"):
            continue

        m = ASSIGN_RE.match(line)
        if m is None:
            continue

        lhs_full = m.group(1).strip()    # auto{_PIPE_L1}
        kind = m.group(2).strip().upper()  # PIPE / UNROLL / ARRAY_T / ...
        rhs = m.group(3).strip()

        key = lhs_full[lhs_full.index("{") + 1 : lhs_full.index("}")]

        if kind == "ARRAY_T":
            point[key] = rhs.lower()
        else:
            try:
                point[key] = int(rhs)
            except ValueError:
                # Be permissive for unexpected string-like values.
                point[key] = rhs

    if not point:
        raise ValueError("Prediction did not contain any directive assignments.")

    return point


def build_synthesis_aligned_point(
    prediction: str,
    kernel_constraints: Dict[str, dict],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply the same repair/legalization logic as batch_hls_from_predictions.py,
    but produce a point-json dict for gexf_to_pt() instead of rendered C source.
    """
    raw_point = parse_prediction_to_point(prediction)
    repaired: Dict[str, Any] = {}
    repair_notes: List[str] = []

    array_groups: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for raw_key, raw_val in raw_point.items():
        key = raw_key.upper()
        label = key_to_label(key)

        if "_ARRAY_T_" in key:
            array_groups[label]["T"] = str(raw_val).lower()
            continue
        if "_ARRAY_F_" in key:
            array_groups[label]["F"] = int(raw_val)
            continue
        if "_ARRAY_D_" in key:
            array_groups[label]["D"] = int(raw_val)
            continue

        if "_UNROLL_" in key:
            info = kernel_constraints.get(label, {})
            loop_bound = info.get("bound") if info.get("kind") == "loop" else None
            repaired[key] = legalize_unroll_value(int(raw_val), loop_bound)
            continue

        if "_PIPE_" in key:
            repaired[key] = legalize_pipeline_value(int(raw_val))
            continue

        repaired[key] = raw_val

    for label in sorted(array_groups.keys(), key=lambda s: int(s[1:])):
        grp = array_groups[label]

        if not all(k in grp for k in ("T", "F", "D")):
            repair_notes.append(f"{label}: incomplete array assignment -> kept raw group")
            if "T" in grp:
                repaired[f"_ARRAY_T_{label}"] = grp["T"]
            if "F" in grp:
                repaired[f"_ARRAY_F_{label}"] = grp["F"]
            if "D" in grp:
                repaired[f"_ARRAY_D_{label}"] = grp["D"]
            continue

        info = kernel_constraints.get(label, {})
        dim_bounds = info.get("dim_bounds", {}) if info.get("kind") == "array" else {}

        legalized, notes = legalize_array_partition(
            part_type=grp["T"],
            factor=int(grp["F"]),
            dim=int(grp["D"]),
            dim_bounds=dim_bounds,
        )
        repair_notes.extend([f"{label}: {x}" for x in notes])

        if legalized is None:
            # Exact alignment with synthesis: dropped pragma => omit ARRAY_T/F/D keys.
            continue

        repaired[f"_ARRAY_T_{label}"] = legalized["type"]
        repaired[f"_ARRAY_D_{label}"] = int(legalized["dim"])

        # Important:
        # For complete, synthesis emits no factor. The graph-side closest equivalent
        # is to omit ARRAY_F. If your gexf_to_pt requires ARRAY_F to exist, replace
        # this branch by keeping grp["F"] as a don't-care fallback.
        if legalized["type"] == "complete":
            repaired[f"_ARRAY_F_{label}"] = 0
        else:
            repaired[f"_ARRAY_F_{label}"] = int(legalized["factor"])

    if not repaired:
        raise ValueError("Repaired candidate became empty after synthesis-alignment.")

    return repaired, repair_notes


def load_candidate_rows(path: str, fallback_obj_mode: Optional[str] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)

            row["obj_mode"] = infer_obj_mode_from_row(row, fallback_obj_mode=fallback_obj_mode)
            row.setdefault("sequence_score", 0.0)
            row.setdefault("sequence_sum_logprob", 0.0)

            rows.append(row)

    if not rows:
        raise ValueError(f"No candidate rows found in {path}")
    return rows


def _sanitize_argv_for_project_imports() -> Tuple[List[str], str]:
    saved_argv = list(sys.argv)
    saved_cwd = os.getcwd()
    sys.argv = [sys.argv[0]]
    return saved_argv, saved_cwd


def import_project_modules(script_dir: str):
    """
    The project config.py does parser.parse_args() at import time.
    Therefore we must temporarily strip our custom CLI args before importing.
    """
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    saved_argv, saved_cwd = _sanitize_argv_for_project_imports()
    os.chdir(script_dir)
    try:
        import config as config_mod  # type: ignore
        from config import FLAGS  # type: ignore
        import data as data_py  # type: ignore
        from model import Net  # type: ignore
        from gexf_to_pt import gexf_to_pt  # type: ignore
        from pt_to_gnn_emb import load_and_clean_graph  # type: ignore
        from torch_geometric.data import Batch  # type: ignore
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)

    return config_mod, FLAGS, data_py, Net, gexf_to_pt, load_and_clean_graph, Batch


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


def load_gnn_model(Net, FLAGS, ckpt_path: str, sample_graph, device: str):
    num_features = sample_graph.x.size(-1)
    edge_dim = sample_graph.edge_attr.size(-1) if getattr(sample_graph, "edge_attr", None) is not None else 0

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    state = _strip_known_prefixes(state)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load GNN checkpoint cleanly. Missing={missing[:20]} Unexpected={unexpected[:20]}"
            )

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def disable_pragma_conditioning(data):
    for name in (
        "X_pragmascopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
    ):
        if hasattr(data, name):
            setattr(data, name, torch.zeros_like(getattr(data, name)))

    if hasattr(data, "X_pragma_per_node"):
        data.X_pragma_per_node = torch.zeros_like(data.X_pragma_per_node)
    if hasattr(data, "pragmas"):
        data.pragmas = torch.zeros_like(data.pragmas)
    return data


def inverse_perf(pred: float, FLAGS) -> float:
    nm = str(getattr(FLAGS, "norm_method", "log2"))
    eps = float(getattr(FLAGS, "epsilon", 1e-6))
    normalizer = float(getattr(FLAGS, "normalizer", 1.0))

    if nm == "log2":
        return max(0.0, (2.0 ** pred) - eps)
    if nm == "const":
        return pred / max(normalizer, 1e-12)
    if nm == "off":
        return pred
    if nm == "speedup-log2":
        speedup = max(0.0, (2.0 ** pred) - eps)
        return float("inf") if speedup <= 0 else normalizer / speedup
    if nm == "speedup":
        speedup = pred
        return float("inf") if speedup <= 0 else normalizer / speedup

    # Fallback: preserve ordering.
    return pred


def inverse_area(pred: float, FLAGS) -> float:
    nm = str(getattr(FLAGS, "norm_method", "log2"))
    eps = float(getattr(FLAGS, "epsilon", 1e-6))
    util_normalizer = float(getattr(FLAGS, "util_normalizer", 1.0))

    if nm == "log2":
        return max(0.0, (2.0 ** pred) - eps)
    if nm == "const":
        return pred / max(util_normalizer, 1e-12)
    if nm == "off":
        return pred

    # Fallback: preserve ordering.
    return pred


def pareto_nondominated_mask(lat_vals: np.ndarray, area_vals: np.ndarray) -> np.ndarray:
    n = len(lat_vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = (
            (lat_vals <= lat_vals[i])
            & (area_vals <= area_vals[i])
            & ((lat_vals < lat_vals[i]) | (area_vals < area_vals[i]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def safe_log2_latency(lat_vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Match stage-1 SFT goal geometry: normalize log2(latency), not raw latency.
    The transform is monotonic, so Pareto dominance on raw latency is preserved,
    but the knee / distance geometry becomes aligned with training.
    """
    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    return np.log2(np.maximum(lat_vals, eps))


def goal_distance_to_ideal(lat_n: float, area_n: float, goal_mode: str) -> float:
    if goal_mode == "PARETO_LATENCY_EXTREME":
        return lat_n
    if goal_mode == "PARETO_AREA_EXTREME":
        return area_n
    if goal_mode == "PARETO_KNEE":
        return math.sqrt(lat_n ** 2 + area_n ** 2)
    raise ValueError(f"Unknown goal_mode: {goal_mode}")


def goal_sort_key(rec: dict, goal_mode: str, domination_penalty: float = 0.0) -> Tuple[float, ...]:
    lat_n = float(rec["lat_n"])
    area_n = float(rec["area_n"])
    dom_gap = float(rec.get("dom_gap", 0.0))

    primary = goal_distance_to_ideal(lat_n, area_n, goal_mode)
    if domination_penalty > 0.0:
        primary += domination_penalty * dom_gap

    if goal_mode == "PARETO_LATENCY_EXTREME":
        return (primary, area_n)
    if goal_mode == "PARETO_AREA_EXTREME":
        return (primary, lat_n)
    return (primary, abs(lat_n - area_n), lat_n + area_n)


def rerank_group(rows: List[dict], goal_mode: str, domination_penalty: float) -> List[dict]:
    # Raw predicted QoR from the frozen surrogate
    lat_raw = np.array([float(r["gnn_pred_perf"]) for r in rows], dtype=np.float64)
    area_raw = np.array([float(r["gnn_pred_area"]) for r in rows], dtype=np.float64)

    # Stage-1 alignment:
    # use log2(latency) for goal geometry, but keep raw latency/area for Pareto dominance.
    lat_geom = safe_log2_latency(lat_raw)
    area_geom = area_raw.copy()

    lat_n = minmax(lat_geom)
    area_n = minmax(area_geom)

    # Dominance uses the original QoR axes.
    nd_mask = pareto_nondominated_mask(lat_raw, area_raw)

    frontier = [(lat_n[i], area_n[i]) for i in range(len(rows)) if nd_mask[i]]

    decorated = []
    for i, r in enumerate(rows):
        dom_gap = 0.0
        if not nd_mask[i] and frontier:
            dom_gap = min(
                max(0.0, lat_n[i] - fl) + max(0.0, area_n[i] - fa)
                for fl, fa in frontier
            )

        rec = dict(r)
        rec["lat_raw"] = float(lat_raw[i])
        rec["area_raw"] = float(area_raw[i])
        rec["lat_geom"] = float(lat_geom[i])
        rec["area_geom"] = float(area_geom[i])
        rec["lat_n"] = float(lat_n[i])
        rec["area_n"] = float(area_n[i])
        rec["is_pareto"] = bool(nd_mask[i])
        rec["dom_gap"] = float(dom_gap)
        decorated.append(rec)

    decorated.sort(
        key=lambda r: (
            goal_sort_key(r, goal_mode, domination_penalty=domination_penalty),
            -float(r.get("sequence_score", 0.0)),
            -float(r.get("sequence_sum_logprob", 0.0)),
            int(r.get("sample_id", 10**9)),
        )
    )
    return decorated


def find_gexf_for_kernel(gexf_dir: str, kernel_name: str) -> str:
    bases = {
        kernel_name,
        kernel_name.replace("-", "_"),
        kernel_name.replace("_", "-"),
        normalize_kname(kernel_name),
    }

    exact = []
    fuzzy = []
    for base in sorted(bases):
        exact.extend(
            [
                os.path.join(gexf_dir, f"{base}.gexf"),
                os.path.join(gexf_dir, f"{base}_connected_hierarchy.gexf"),
            ]
        )
        fuzzy.extend(glob.glob(os.path.join(gexf_dir, "**", f"{base}*.gexf"), recursive=True))

    exact_hits = [p for p in exact if os.path.isfile(p)]
    if exact_hits:
        return sorted(exact_hits, key=lambda p: (len(os.path.basename(p)), p))[0]

    fuzzy_hits = [p for p in fuzzy if os.path.isfile(p)]
    if not fuzzy_hits:
        raise FileNotFoundError(f"Could not find a .gexf for kernel '{kernel_name}' under {gexf_dir}")

    # Prefer the shortest basename match for determinism.
    fuzzy_hits = sorted(set(fuzzy_hits), key=lambda p: (len(os.path.basename(p)), p))
    return fuzzy_hits[0]


def tensor_signature(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def get_graph_pragma_stats(graph) -> Dict[str, Any]:
    stats = {
        "pragmas_nnz": int(torch.count_nonzero(graph.pragmas).item()) if hasattr(graph, "pragmas") else -1,
        "x_pragma_per_node_nnz": int(torch.count_nonzero(graph.X_pragma_per_node).item()) if hasattr(graph, "X_pragma_per_node") else -1,
        "x_pragmanids_nnz": int(torch.count_nonzero(graph.X_pragmanids).item()) if hasattr(graph, "X_pragmanids") else -1,
        "x_pragmascopenids_nnz": int(torch.count_nonzero(graph.X_pragmascopenids).item()) if hasattr(graph, "X_pragmascopenids") else -1,
        "pragmas_sig": tensor_signature(graph.pragmas) if hasattr(graph, "pragmas") else "",
        "x_pragma_per_node_sig": tensor_signature(graph.X_pragma_per_node) if hasattr(graph, "X_pragma_per_node") else "",
    }
    return stats


def aligned_point_has_active_values(point: Dict[str, Any]) -> bool:
    """
    True only if the repaired point contains at least one nonzero/effective
    directive value that should affect the GNN pragma tensors.

    PIPE=0, UNROLL=0, ARRAY_F=0, ARRAY_D=0 are inactive.
    ARRAY_T=block/cyclic/complete is active because data.py maps it to 200/100/300.
    """
    for k, v in point.items():
        ku = k.upper()

        if ku.startswith("_ARRAY_T_"):
            if str(v).strip().lower() in {"block", "cyclic", "complete"}:
                return True
            try:
                if int(v) != 0:
                    return True
            except Exception:
                pass
            continue

        try:
            if int(v) != 0:
                return True
        except Exception:
            return True

    return False


def validate_candidate_graph(
    graph,
    row: dict,
    disable_pragma_injection_flag: bool,
    aligned_point: Dict[str, Any],
) -> Dict[str, Any]:
    stats = get_graph_pragma_stats(graph)

    required = ["pragmas", "X_pragma_per_node"]
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise RuntimeError(
            f"{row['kernel_name']} sample_id={row.get('sample_id')} missing PT fields: {missing}. "
            "This usually means the wrong gexf_to_pt.py or incompatible preprocessing was used."
        )

    if not disable_pragma_injection_flag:
        expected_active = aligned_point_has_active_values(aligned_point)
        stats["expected_active_pragma_values"] = bool(expected_active)

        if expected_active and stats["pragmas_nnz"] <= 0:
            raise RuntimeError(
                f"{row['kernel_name']} sample_id={row.get('sample_id')} has active aligned_point values "
                "but graph.pragmas is all zero. Candidate values are not reaching the PT graph."
            )

        if expected_active and stats["x_pragma_per_node_nnz"] <= 0:
            raise RuntimeError(
                f"{row['kernel_name']} sample_id={row.get('sample_id')} has active aligned_point values "
                "but X_pragma_per_node is all zero. Candidate values are not being encoded per node."
            )

    return stats


def evaluate_candidate(
    *,
    row: dict,
    gexf_path: str,
    temp_root: str,
    gexf_to_pt,
    load_and_clean_graph,
    Batch,
    model,
    FLAGS,
    device: str,
    max_pragma_length: int,
    disable_pragma_injection_flag: bool,
    aligned_point: Dict[str, Any],
    repair_notes: List[str],
) -> dict:
    uid = hashlib.sha1(
        (
            row["kernel_name"]
            + "\n"
            + row["obj_mode"]
            + "\n"
            + stable_json_sig(aligned_point)
        ).encode("utf-8")
    ).hexdigest()[:16]

    point_json_path = os.path.join(temp_root, f"{uid}.point.json")
    pt_path = os.path.join(temp_root, f"{uid}.processed_result.pt")

    with open(point_json_path, "w", encoding="utf-8") as f:
        json.dump(aligned_point, f, indent=2, ensure_ascii=False, sort_keys=True)

    gexf_to_pt(
        gexf_path=gexf_path,
        point_json=point_json_path,
        out_pt=pt_path,
        key_name=f"cand_{row.get('sample_id', 0)}",
        perf=0.0,
        area=0.0,
        max_pragma_length=max_pragma_length,
    )

    graph = load_and_clean_graph(pt_path)
    pragma_stats = validate_candidate_graph(
        graph,
        row=row,
        disable_pragma_injection_flag=disable_pragma_injection_flag,
        aligned_point=aligned_point,
    )
    batch = Batch.from_data_list([graph]).to(device)
    if disable_pragma_injection_flag:
        batch = disable_pragma_conditioning(batch)

    with torch.no_grad():
        out_dict, _, _, _ = model(batch)

    pred_perf_norm = float(out_dict["perf"][0].item())
    pred_area_norm = float(out_dict["area"][0].item())

    result = dict(row)
    result["gnn_pred_perf_norm"] = pred_perf_norm
    result["gnn_pred_area_norm"] = pred_area_norm
    result["gnn_pred_perf"] = float(inverse_perf(pred_perf_norm, FLAGS))
    result["gnn_pred_area"] = float(inverse_area(pred_area_norm, FLAGS))
    result["pragma_stats"] = pragma_stats
    result["repair_notes"] = repair_notes
    result["aligned_design_sig"] = stable_json_sig(aligned_point)
    result["aligned_point"] = aligned_point
    return result


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main() -> None:
    args = parse_args()

    fallback_obj_mode = None
    if args.objective_tag:
        fallback_obj_mode = canonical_objective_from_tag(args.objective_tag)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_mod, FLAGS, data_py, Net, gexf_to_pt, load_and_clean_graph, Batch = import_project_modules(script_dir)

    # Make the imported project respect the device chosen for this script.
    FLAGS.device = args.device
    FLAGS.task = "regression"
    FLAGS.target = ["perf", "area"]

    candidate_rows = load_candidate_rows(
        args.candidates_jsonl,
        fallback_obj_mode=fallback_obj_mode,
    )

    if fallback_obj_mode is not None:
        bad_modes = sorted({r["obj_mode"] for r in candidate_rows if r["obj_mode"] != fallback_obj_mode})
        if bad_modes and args.verbose:
            print(f"[WARN] fallback objective_tag={fallback_obj_mode}, but found row obj_modes={bad_modes}")

    # Resolve one graph first so the model can be instantiated with the right input dims.
    first_gexf = find_gexf_for_kernel(args.gexf_dir, candidate_rows[0]["kernel_name"])

    temp_root = args.temp_dir
    created_temp_root = False
    if not temp_root:
        out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        temp_prefix = normalize_name(args.objective_tag) if args.objective_tag else "mixed_objectives"
        temp_root = tempfile.mkdtemp(prefix=f"rerank_{temp_prefix}_", dir=out_dir)
        created_temp_root = True
    os.makedirs(temp_root, exist_ok=True)

    try:
        bootstrap_gexf = first_gexf
        bootstrap_kernel = candidate_rows[0]["kernel_name"]
        bootstrap_kinfo_path = resolve_kernel_info_path(
            bootstrap_kernel,
            bootstrap_gexf,
            kernel_info_root=args.kernel_info_root,
        )
        bootstrap_constraints = parse_kernel_constraints(bootstrap_kinfo_path)
        bootstrap_point, bootstrap_notes = build_synthesis_aligned_point(
            candidate_rows[0]["prediction"],
            bootstrap_constraints,
        )

        bootstrap_pt = os.path.join(temp_root, "_bootstrap.pt")
        bootstrap_point_json = os.path.join(temp_root, "_bootstrap.point.json")
        with open(bootstrap_point_json, "w", encoding="utf-8") as f:
            json.dump(bootstrap_point, f, indent=2, ensure_ascii=False, sort_keys=True)

        gexf_to_pt(
            gexf_path=bootstrap_gexf,
            point_json=bootstrap_point_json,
            out_pt=bootstrap_pt,
            key_name="bootstrap",
            perf=0.0,
            area=0.0,
            max_pragma_length=args.max_pragma_length,
        )

        sample_graph = load_and_clean_graph(bootstrap_pt)
        model = load_gnn_model(Net, FLAGS, args.gnn_ckpt, sample_graph, args.device)

        resolved_gexf_cache: Dict[str, str] = {}
        kernel_constraints_cache: Dict[str, Dict[str, dict]] = {}
        aligned_eval_cache: Dict[Tuple[str, str], dict] = {}
        scored_rows: List[dict] = []

        for idx, row in enumerate(candidate_rows, start=1):
            kname = row["kernel_name"]

            if kname not in resolved_gexf_cache:
                resolved_gexf_cache[kname] = find_gexf_for_kernel(args.gexf_dir, kname)

            gexf_path = resolved_gexf_cache[kname]

            if kname not in kernel_constraints_cache:
                kinfo_path = resolve_kernel_info_path(
                    kname,
                    gexf_path,
                    kernel_info_root=args.kernel_info_root,
                )
                kernel_constraints_cache[kname] = parse_kernel_constraints(kinfo_path)

            aligned_point, repair_notes = build_synthesis_aligned_point(
                row["prediction"],
                kernel_constraints_cache[kname],
            )
            aligned_sig = stable_json_sig(aligned_point)
            cache_key = (kname, aligned_sig)

            if args.verbose:
                note_suffix = f" repairs={len(repair_notes)}" if repair_notes else ""
                print(
                    f"[GNN-RERANK] {idx}/{len(candidate_rows)} "
                    f"kernel={kname} sample_id={row.get('sample_id')} "
                    f"aligned_sig={aligned_sig}{note_suffix}"
                )
                if repair_notes:
                    for note in repair_notes[:10]:
                        print(f"  [REPAIR] {note}")

            if cache_key in aligned_eval_cache:
                cached = dict(aligned_eval_cache[cache_key])
                scored = dict(row)
                scored.update(cached)
            else:
                scored = evaluate_candidate(
                    row=row,
                    gexf_path=gexf_path,
                    temp_root=temp_root,
                    gexf_to_pt=gexf_to_pt,
                    load_and_clean_graph=load_and_clean_graph,
                    Batch=Batch,
                    model=model,
                    FLAGS=FLAGS,
                    device=args.device,
                    max_pragma_length=args.max_pragma_length,
                    disable_pragma_injection_flag=args.disable_pragma_injection,
                    aligned_point=aligned_point,
                    repair_notes=repair_notes,
                )
                aligned_eval_cache[cache_key] = {
                    "gnn_pred_perf_norm": scored["gnn_pred_perf_norm"],
                    "gnn_pred_area_norm": scored["gnn_pred_area_norm"],
                    "gnn_pred_perf": scored["gnn_pred_perf"],
                    "gnn_pred_area": scored["gnn_pred_area"],
                    "pragma_stats": scored["pragma_stats"],
                    "repair_notes": scored["repair_notes"],
                    "aligned_design_sig": scored["aligned_design_sig"],
                    "aligned_point": scored["aligned_point"],
                }

            scored_rows.append(scored)

        by_case: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for r in scored_rows:
            by_case[(r["kernel_name"], r["obj_mode"])].append(r)

        final_rows: List[dict] = []
        for (kernel_name, obj_mode), rows in sorted(by_case.items()):
            ranked = rerank_group(rows, goal_mode=obj_mode, domination_penalty=args.domination_penalty)
            best = ranked[0]

            out_row = {
                "kernel_name": kernel_name,
                "obj_mode": obj_mode,
                "prediction": best["prediction"],          # keep raw canonical text for HLS script
                "sample_id": int(best.get("sample_id", -1)),
                "gnn_pred_perf": float(best["gnn_pred_perf"]),
                "gnn_pred_area": float(best["gnn_pred_area"]),
                "gnn_pred_perf_norm": float(best["gnn_pred_perf_norm"]),
                "gnn_pred_area_norm": float(best["gnn_pred_area_norm"]),
                "aligned_design_sig": best["aligned_design_sig"],
                "aligned_point": best.get("aligned_point", {}),
                "pragma_stats": best.get("pragma_stats", {}),
                "repair_notes": best.get("repair_notes", []),
            }
            final_rows.append(out_row)

            if args.verbose:
                print(
                    "[SELECT] "
                    f"kernel={kernel_name} obj={obj_mode} sample_id={out_row['sample_id']} "
                    f"perf={out_row['gnn_pred_perf']:.6f} area={out_row['gnn_pred_area']:.6f}"
                )

        ensure_parent(args.output_jsonl)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in final_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[DONE] wrote reranked JSONL -> {args.output_jsonl}")
        print(f"[DONE] selected {len(final_rows)} final candidates from {len(candidate_rows)} unique candidates")

    finally:
        if created_temp_root and not args.keep_temps:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
