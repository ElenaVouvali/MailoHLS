#!/usr/bin/env python3
"""
gexf_to_pt.py (dataset-compatible, pragma-aware)

Goal:
  Produce a .pt that matches the encoding conventions of the current data.py
  pipeline, while preserving *actual pragma values* when a real point-json is
  provided.

Behavior:
  - If point_json is a real JSON file: use the actual pragma configuration.
  - If point_json is None / "" / "NONE": build a zero-point from auto{...}
    placeholders found in the GEXF (useful for pragma-free PT generation).
  - Emit the updated scope-mask fields and the X_llm_* slot-anchor tensors that
    the memory pipeline expects.
"""

import os
import json
import math
import re
from typing import Any, Dict, List

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

from config import FLAGS
import data as data_py


_ALLOWED_PREFIXES = ("_PIPE_", "_UNROLL_", "_ARRAY_T_", "_ARRAY_F_", "_ARRAY_D_")
_ARRAY_TMAP = {"cyclic": 100, "block": 200, "complete": 300}
_AUTO_RE = re.compile(r"auto\{([^}]+)\}")
_LABEL_RE = re.compile(r"_L(\d+)\b")


def _zero_point_from_gexf(g: nx.Graph) -> Dict[str, int]:
    """
    Extract every auto{_PIPE_...}, auto{_UNROLL_...}, auto{_ARRAY_*_...}
    from node full_text and assign 0.

    This is the pragma-free path used by code_to_memory_pt.py / pragma_free_pts.py.
    """
    point: Dict[str, int] = {}
    for _, ndata in g.nodes(data=True):
        ft = ndata.get("full_text", "")
        if not isinstance(ft, str):
            continue
        for key in _AUTO_RE.findall(ft):
            if key.startswith(_ALLOWED_PREFIXES):
                point[key] = 0
    return point


def _require_numeric_string_nodes(g: nx.Graph) -> None:
    """
    data.py::_encode_X_dict() sorts nodes by int(node_id) and asserts that node
    ids are exactly 0..N-1 as numeric strings.
    """
    nodes = list(g.nodes())
    for n in nodes:
        if not isinstance(n, str) or not n.isdigit():
            raise RuntimeError(
                "Dataset-compatible encoding requires numeric-string node ids. "
                f"Found node id: {n!r} (type={type(n)})"
            )

    ints = sorted(int(n) for n in nodes)
    if not ints:
        raise RuntimeError("Graph has no nodes.")

    if ints[0] != 0 or ints[-1] != len(ints) - 1:
        raise RuntimeError(
            f"Node ids must be contiguous 0..N-1. "
            f"Got min={ints[0]}, max={ints[-1]}, N={len(ints)}."
        )


def _normalize_point_for_pragmas_vector(point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the point mostly as provided for data.py::_encode_X_dict(),
    but normalize simple JSON oddities.
    """
    out: Dict[str, Any] = {}
    for k, v in point.items():
        if v is None:
            out[k] = 0
        elif isinstance(v, (int, np.integer, bool, np.bool_)):
            out[k] = int(v)
        elif isinstance(v, float) and float(v).is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _build_pragmas_vector(point: Dict[str, Any], max_pragma_length: int) -> torch.Tensor:
    """
    Build the flat `pragmas` tensor in the same key-order/value-conversion style
    expected by the project.

    Layout:
      - iterate in sorted(point.items()) order
      - keep only _PIPE_ / _UNROLL_ / _ARRAY_*
      - map ARRAY_T strings to {100, 200, 300}
      - pad to max_pragma_length
    """
    pragmas: List[int] = []

    for name, value in sorted(point.items()):
        if not name.startswith(_ALLOWED_PREFIXES):
            continue

        if isinstance(value, str):
            v = value.strip().lower()
            if name.startswith("_ARRAY_T_"):
                value = _ARRAY_TMAP.get(v, 0)
            else:
                try:
                    value = int(v)
                except ValueError as e:
                    raise ValueError(f"Non-numeric pragma value '{value}' for key {name}") from e
        elif isinstance(value, (int, np.integer, bool, np.bool_)):
            value = int(value)
        else:
            raise ValueError(f"Unexpected pragma value type: {type(value)} for key {name}")

        pragmas.append(int(value))

    if len(pragmas) > max_pragma_length:
        raise RuntimeError(
            f"Pragmas length {len(pragmas)} exceeds max_pragma_length {max_pragma_length}. "
            "Use the correct dataset-compatible max_pragma_length."
        )

    pragmas.extend([0] * (max_pragma_length - len(pragmas)))
    return torch.tensor([pragmas], dtype=torch.float32)


def _fill_targets_like_data_py(xy_dict: Dict[str, Any], perf_val: float, area_val: float) -> None:
    """
    Create regression targets using the current config normalization style.
    For inference-side PT generation, dummy perf/area are acceptable.
    """
    if getattr(FLAGS, "task", "regression") != "regression":
        xy_dict["perf"] = torch.LongTensor([0])
        return

    eps = float(getattr(FLAGS, "epsilon", 1e-6))
    norm_method = str(getattr(FLAGS, "norm_method", "log2"))

    def norm_perf(v: float) -> float:
        if norm_method == "log2":
            return math.log2(v + eps)
        if norm_method == "const":
            return v * float(getattr(FLAGS, "normalizer", 1.0))
        if norm_method == "off":
            return v
        if "speedup" in norm_method:
            if v <= 0:
                return 0.0
            speedup = float(getattr(FLAGS, "normalizer", 1.0)) / v
            if norm_method == "speedup-log2":
                return math.log2(speedup + eps)
            return speedup
        return math.log2(v + eps)

    xy_dict["perf"] = torch.tensor([norm_perf(float(perf_val))], dtype=torch.float32)
    xy_dict["actual_perf"] = torch.tensor([float(perf_val)], dtype=torch.float32)
    xy_dict["kernel_speedup"] = torch.tensor([0.0], dtype=torch.float32)

    area_safe = float(area_val) if float(area_val) > 0.0 else eps
    if norm_method == "const":
        area_norm = area_safe * float(getattr(FLAGS, "util_normalizer", 1.0))
    elif norm_method == "off":
        area_norm = area_safe
    else:
        area_norm = math.log2(area_safe + eps)

    xy_dict["area"] = torch.tensor([area_norm], dtype=torch.float32)
    xy_dict["actual_area"] = torch.tensor([area_safe], dtype=torch.float32)


def _extract_label_ids_from_full_text(ft: Any) -> List[int]:
    if not isinstance(ft, str):
        return []

    lids = set()

    for key in _AUTO_RE.findall(ft):
        if key.startswith(_ALLOWED_PREFIXES):
            m = _LABEL_RE.search(key)
            if m:
                lids.add(int(m.group(1)))

    if not lids:
        for m in _LABEL_RE.finditer(ft):
            lids.add(int(m.group(1)))

    return sorted(lids)


def _build_llm_scope_tensors(g: nx.Graph) -> Dict[str, torch.Tensor]:
    """
    Build slot-anchor tensors aligned with current data.py semantics:

      - loop pragmas anchor on pseudo-block scope nodes
      - array_partition pragmas anchor on array-scope nodes

    Current memory format requires exactly one source label Lk per anchor node.
    """
    n_nodes = g.number_of_nodes()
    scopeids = torch.zeros((n_nodes,), dtype=torch.long)
    scopecat = torch.zeros((n_nodes,), dtype=torch.long)   # 0 none, 1 pseudo-loop, 2 array-scope
    label_ids = torch.full((n_nodes,), -1, dtype=torch.long)

    label_to_anchor = {}
    anchor_to_label = {}

    for node, ndata in sorted(g.nodes(data=True), key=lambda x: int(x[0])):
        if data_py.is_pseudo_block_node(ndata):
            allowed_kinds = {"pipeline", "unroll"}
            anchor_cat = 1
        elif data_py.is_array_scope_node(ndata):
            allowed_kinds = {"array_partition"}
            anchor_cat = 2
        else:
            continue

        neighbor_pragmas = data_py.find_attached_pragmas(
            g,
            node,
            allowed_kinds=allowed_kinds,
        )
        if not neighbor_pragmas:
            continue

        lids = set()
        for _, pid in sorted(neighbor_pragmas.items()):
            ft = str(g.nodes[pid].get("full_text", ""))
            lids.update(_extract_label_ids_from_full_text(ft))

        if not lids:
            continue

        if len(lids) != 1:
            raise RuntimeError(
                f"Scope anchor node {node} maps to multiple labels: {sorted(lids)}. "
                "Current X_llm_labelid format requires one Lk per anchor node."
            )

        lid = next(iter(lids))
        anchor_nid = int(node)

        prev = label_to_anchor.get(lid)
        if prev is not None and prev != (anchor_nid, anchor_cat):
            raise RuntimeError(
                f"Placeholder L{lid} maps to multiple anchors: "
                f"{prev} and {(anchor_nid, anchor_cat)}"
            )
        label_to_anchor[lid] = (anchor_nid, anchor_cat)

        prev_lid = anchor_to_label.get(anchor_nid)
        if prev_lid is not None and prev_lid != lid:
            raise RuntimeError(
                f"Anchor node {anchor_nid} corresponds to multiple labels: "
                f"L{prev_lid} and L{lid}"
            )
        anchor_to_label[anchor_nid] = lid

    for lid, (anchor_nid, anchor_cat) in label_to_anchor.items():
        scopeids[anchor_nid] = 1
        scopecat[anchor_nid] = anchor_cat
        label_ids[anchor_nid] = lid

    return {
        "X_llm_scopeids": scopeids,
        "X_llm_scopecat": scopecat,
        "X_llm_labelid": label_ids,
    }


def gexf_to_pt(
    gexf_path: str,
    point_json: str,
    out_pt: str,
    key_name: str,
    perf: float = 0.0,
    area: float = 0.0,
    max_pragma_length: int = 93,
) -> None:
    # Load encoders exactly as data.py does
    encoders = data_py.load_encoders()
    enc_ntype = encoders["enc_ntype"]
    enc_ptype = encoders["enc_ptype"]
    enc_itype = encoders["enc_itype"]
    enc_ftype = encoders["enc_ftype"]
    enc_btype = encoders["enc_btype"]
    enc_ftype_edge = encoders["enc_ftype_edge"]
    enc_ptype_edge = encoders["enc_ptype_edge"]

    gexf_path = os.path.expanduser(gexf_path)
    g = nx.read_gexf(gexf_path)

    _require_numeric_string_nodes(g)

    graph_name = os.path.basename(gexf_path).split(".")[0]
    kernel_name = graph_name.split("_")[0]

    # Load real design-point values, or build a pragma-free zero-point.
    if point_json is None or str(point_json).strip() == "" or str(point_json).upper() == "NONE":
        point = _zero_point_from_gexf(g)
    else:
        with open(os.path.expanduser(point_json), "r", encoding="utf-8") as f:
            point_raw = json.load(f)
        point = _normalize_point_for_pragmas_vector(point_raw)

    # Current dataset-compatible graph encoding path
    edge_index = data_py.create_edge_index(g).contiguous()
    edge_dict = data_py._encode_edge_dict(g, ftypes=None, ptypes=None)

    xy_dict = data_py._encode_X_dict(
        g,
        ntypes=None,
        ptypes=None,
        numerics=None,
        itypes=None,
        ftypes=None,
        btypes=None,
        point=point,
    )

    # Updated placeholder-aligned scope anchors used by the memory pipeline
    llm_scope = _build_llm_scope_tensors(g)

    # Keep actual pragma values in pragma-aware mode; zero-point path naturally stays zero
    xy_dict["pragmas"] = _build_pragmas_vector(point, max_pragma_length=max_pragma_length)

    # Dummy targets for compatibility
    _fill_targets_like_data_py(xy_dict, perf_val=float(perf), area_val=float(area))

    X = data_py._encode_X_torch(
        xy_dict,
        enc_ntype,
        enc_ptype,
        enc_itype,
        enc_ftype,
        enc_btype,
    ).to(torch.float32).contiguous()

    edge_attr = data_py._encode_edge_torch(
        edge_dict,
        enc_ftype_edge,
        enc_ptype_edge,
    ).to(torch.float32).contiguous()

    if not torch.isfinite(X).all():
        bad = (~torch.isfinite(X)).sum().item()
        raise RuntimeError(f"Non-finite node features before save for {graph_name}: bad_count={bad}")

    if not torch.isfinite(edge_attr.float()).all():
        bad = (~torch.isfinite(edge_attr.float())).sum().item()
        raise RuntimeError(f"Non-finite edge_attr before save for {graph_name}: bad_count={bad}")

    data_obj = Data(
        gname=kernel_name,
        x=X,
        key=key_name,
        edge_index=edge_index,
        edge_attr=edge_attr,
        kernel=graph_name,

        X_contextnids=xy_dict["X_contextnids"],
        X_pragmanids=xy_dict["X_pragmanids"],
        X_pragmascopenids=xy_dict["X_pragmascopenids"],
        X_pseudonids=xy_dict["X_pseudonids"],
        X_arrayscopenids=xy_dict["X_arrayscopenids"],
        X_pipeline_scopeids=xy_dict["X_pipeline_scopeids"],
        X_unroll_scopeids=xy_dict["X_unroll_scopeids"],
        X_array_partition_scopeids=xy_dict["X_array_partition_scopeids"],
        X_scopenids=xy_dict["X_scopenids"],
        X_icmpnids=xy_dict["X_icmpnids"],

        X_pragma_per_node=xy_dict["X_pragma_per_node"],
        pragmas=xy_dict["pragmas"],
        perf=xy_dict["perf"],

        X_llm_scopeids=llm_scope["X_llm_scopeids"],
        X_llm_scopecat=llm_scope["X_llm_scopecat"],
        X_llm_labelid=llm_scope["X_llm_labelid"],
    )

    if "actual_perf" in xy_dict:
        data_obj.actual_perf = xy_dict["actual_perf"]
    if "kernel_speedup" in xy_dict:
        data_obj.kernel_speedup = xy_dict["kernel_speedup"]
    if "area" in xy_dict:
        data_obj.area = xy_dict["area"]
    if "actual_area" in xy_dict:
        data_obj.actual_area = xy_dict["actual_area"]

    out_pt = os.path.expanduser(out_pt)
    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    torch.save(data_obj, out_pt)

    print(f"[OK] Saved: {out_pt}")
