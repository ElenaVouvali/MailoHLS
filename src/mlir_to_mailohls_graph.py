#!/usr/bin/env python3
"""
mlir_to_mailohls_graph.py

Build the *final* MailoHLS-MLIR graph that is intended to feed the current
MailoHLS structural GNN backbone.

Inputs
------
  1) affine/scf MLIR emitted by emit_mlir.py / cgeist
  2) ap_schema.json emitted by emit_mlir.py
  3) optional design-point JSON with pragma values, e.g.
       {"_PIPE_L1": 1, "_UNROLL_L1": 4, "_PIPE_L2": 0, ...}

Outputs
-------
  - mailohls_mlir_graph.gexf       NetworkX graph with numeric node IDs
  - mailohls_mlir_graph.dot        Graphviz DOT visualization
  - mailohls_mlir_graph.svg/png    rendered if `dot` is available
  - mailohls_mlir_graph.pt         PyTorch Geometric Data object, if torch/PyG exists
  - mailohls_mlir_graph_summary.json
  - mailohls_mlir_graph_vocab.json

Design philosophy
-----------------
This script does NOT mutate the textual MLIR.  It treats MLIR as the compiler
structural substrate and adds MailoHLS-specific graph objects in the graph/PyG
layer:

  action_point_Lk nodes  = stable MailoHLS decision/memory anchors
  array_scope nodes      = stable memory-object anchors
  target_context node    = FPGA/clock conditioning context
  HLS shortcut edges     = explicit loop <-> array access relationships

The GNN should extract structural memory from action_point_Lk nodes, because
these nodes correspond one-to-one with the placeholders that the LLM must fill.

Notes
-----
This is a practical first implementation.  The MLIR parser is regex/stack based
and targets cgeist-style affine/scf text.  For a production parser, replace the
text parser with MLIR Python bindings or a custom C++ MLIR pass, while keeping
the graph schema emitted here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Torch/PyG are optional for visualization-only use.
try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except Exception:
    torch = None
    Data = None
    HAS_PYG = False


# ---------------------------------------------------------------------------
# Regexes for cgeist-style MLIR
# ---------------------------------------------------------------------------

FUNC_RE = re.compile(r"^\s*func\.func\s+@(?P<name>[A-Za-z_]\w*)\((?P<args>.*?)\)")
ARG_RE = re.compile(r"(?P<ssa>%[A-Za-z0-9_]+)\s*:\s*(?P<type>memref<[^>]+>|[A-Za-z0-9_!?.<>]+)")

# Examples:
#   affine.for %arg3 = 0 to 64 {
#   %0 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %cst) -> (f64) {
FOR_RE = re.compile(
    r"^\s*(?:(?P<result>%[A-Za-z0-9_]+)\s*=\s*)?(?P<dialect>affine|scf)\.for\s+"
    r"(?P<iv>%[A-Za-z0-9_]+)\s*=\s*(?P<lb>[^\s]+)\s+to\s+(?P<ub>[^\s{]+)(?P<rest>.*)\{\s*$"
)

LOAD_RE = re.compile(
    r"^\s*(?P<result>%[A-Za-z0-9_]+)\s*=\s*(?P<dialect>affine|memref)\.load\s+"
    r"(?P<mem>%[A-Za-z0-9_]+)\[(?P<idx>[^\]]*)\]\s*:\s*(?P<type>memref<[^>]+>)"
)
STORE_RE = re.compile(
    r"^\s*(?P<dialect>affine|memref)\.store\s+(?P<value>%[A-Za-z0-9_]+),\s*"
    r"(?P<mem>%[A-Za-z0-9_]+)\[(?P<idx>[^\]]*)\]\s*:\s*(?P<type>memref<[^>]+>)"
)

ALLOCA_RE = re.compile(
    r"^\s*(?P<result>%[A-Za-z0-9_]+)\s*=\s*memref\.alloca\([^)]*\)\s*:\s*(?P<type>memref<[^>]+>)"
)

ARITH_RE = re.compile(
    r"^\s*(?P<result>%[A-Za-z0-9_]+)\s*=\s*arith\.(?P<op>[A-Za-z0-9_]+)\s+(?P<operands>.*?)\s*:\s*(?P<type>.+)$"
)
CONST_RE = re.compile(r"^\s*(?P<result>%[A-Za-z0-9_]+)\s*=\s*arith\.constant\s+(?P<value>.*?)\s*:\s*(?P<type>.+)$")
YIELD_RE = re.compile(r"^\s*affine\.yield\s+(?P<value>%[A-Za-z0-9_]+)")
RETURN_RE = re.compile(r"^\s*return\b")

# Source function signature parser.  Good enough for HLS kernels.
FUNC_SRC_RE_TMPL = r"\b(?P<ret>[A-Za-z_][\w:\s\*\&<>]*)\s+{fname}\s*\((?P<params>[^)]*)\)\s*\{{"
PARAM_RE = re.compile(r"(?P<type>[A-Za-z_][\w:\s\*\&<>]*?)\s+(?P<name>[A-Za-z_]\w*)\s*(?P<dims>(?:\[[^\]]*\])*)\s*$")
DIM_RE = re.compile(r"\[([^\]]*)\]")

ARRAY_TMAP = {"cyclic": 100, "block": 200, "complete": 300}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_") or "node"


def log2p(v: Any) -> float:
    try:
        f = float(v)
        if f <= 0:
            return 0.0
        return math.log2(f + 1.0)
    except Exception:
        return 0.0


def extract_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    m = re.fullmatch(r"\s*[-+]?\d+\s*", str(s))
    return int(str(s)) if m else None


def parse_memref_shape(memref_type: str) -> Tuple[List[Optional[int]], str]:
    # memref<64x64xf32>, memref<?xf64>, memref<512xf32>
    m = re.match(r"memref<(?P<body>[^>]+)>", memref_type.strip())
    if not m:
        return [], ""
    parts = m.group("body").split("x")
    if not parts:
        return [], ""
    elem_type = parts[-1]
    dims: List[Optional[int]] = []
    for p in parts[:-1]:
        p = p.strip()
        dims.append(int(p) if p.isdigit() else None)
    return dims, elem_type


def load_point(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    obj = json.loads(read_text(path))
    if isinstance(obj, dict):
        return obj
    raise ValueError("--point-json must contain a JSON object")


def pragma_value(point: Dict[str, Any], key: str) -> int:
    v = point.get(key, 0)
    if v is None:
        return 0
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ARRAY_TMAP:
            return ARRAY_TMAP[vl]
        try:
            return int(vl)
        except Exception:
            return 0
    try:
        return int(v)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Source metadata: source parameter names/shapes from ap_schema source_file
# ---------------------------------------------------------------------------

def parse_source_function_params(source_file: Optional[str], func: str) -> List[Dict[str, Any]]:
    if not source_file or not Path(source_file).exists():
        return []
    text = read_text(Path(source_file))
    text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text_no_comments = re.sub(r"//.*", "", text_no_comments)
    pat = re.compile(FUNC_SRC_RE_TMPL.format(fname=re.escape(func)), flags=re.S)
    m = pat.search(text_no_comments)
    if not m:
        return []
    params = []
    for raw in m.group("params").split(","):
        raw = raw.strip()
        if not raw or raw == "void":
            continue
        pm = PARAM_RE.match(raw)
        if not pm:
            # Fallback for pointer syntax: float* x / float *x
            pm2 = re.match(r"(?P<type>.+?\*)\s*(?P<name>[A-Za-z_]\w*)$", raw)
            if not pm2:
                params.append({"name": raw, "type": "", "dims_expr": []})
                continue
            params.append({"name": pm2.group("name"), "type": pm2.group("type").strip(), "dims_expr": []})
            continue
        dims = [d.strip() for d in DIM_RE.findall(pm.group("dims") or "")]
        params.append({"name": pm.group("name"), "type": " ".join(pm.group("type").split()), "dims_expr": dims})
    return params


def eval_dim_expr(expr: str, constants: Dict[str, Any]) -> Optional[int]:
    # Small, safe-ish evaluator for arithmetic macro expressions already resolved in ap_schema.
    if not expr:
        return None
    e = expr
    for k, v in sorted(constants.items(), key=lambda kv: -len(kv[0])):
        if isinstance(v, int):
            e = re.sub(rf"\b{re.escape(k)}\b", str(v), e)
    if not re.fullmatch(r"[0-9+\-*/%() \t]+", e):
        return None
    try:
        return int(eval(e, {"__builtins__": {}}, {}))
    except Exception:
        return None


def infer_logical_shape(name: str, dims: List[Optional[int]], constants: Dict[str, Any]) -> List[Optional[int]]:
    # Helpful for flattened GEMM/KNN arrays: N=row_size*col_size, etc.
    rs = constants.get("row_size")
    cs = constants.get("col_size")
    if len(dims) == 1 and isinstance(rs, int) and isinstance(cs, int) and dims[0] == rs * cs:
        return [rs, cs]
    return dims


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class HLSGraphBuilder:
    def __init__(self, schema: Dict[str, Any], point: Dict[str, Any], include_reverse_edges: bool = True):
        self.schema = schema
        self.point = point
        self.include_reverse_edges = include_reverse_edges
        self.g = nx.MultiDiGraph()
        self._next_id = 0
        self.ssa_producer: Dict[str, str] = {}
        self.memref_to_array_node: Dict[Tuple[str, str], str] = {}  # (func, %arg0) -> node id
        self.func_param_meta: Dict[str, List[Dict[str, Any]]] = {}
        self.loop_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.ap_node_by_label: Dict[str, str] = {}
        self.mlir_loop_by_label: Dict[str, str] = {}
        self.current_function: Optional[str] = None
        self.loop_stack: List[Dict[str, Any]] = []  # items with mlir_node, ap_node, iv, label
        self.node_order_labels: Dict[str, int] = {}

        for ap in schema.get("action_points", []):
            if ap.get("kind") == "loop":
                self.loop_queues[ap.get("function", "GLOBAL")].append(ap)

    def add_node(self, kind: str, text: str, **attrs: Any) -> str:
        nid = str(self._next_id)
        self._next_id += 1
        base = {
            "kind": kind,
            "text": text,
            "full_text": attrs.pop("full_text", text),
            "type": kind,  # keep a string type for MLIR graph variant
        }
        base.update({k: v for k, v in attrs.items() if v is not None})
        self.g.add_node(nid, **base)
        return nid

    def add_edge(self, src: Optional[str], dst: Optional[str], kind: str, **attrs: Any) -> None:
        if src is None or dst is None or src == dst:
            return
        edge_attrs = {"kind": kind, "flow": kind}
        edge_attrs.update({k: v for k, v in attrs.items() if v is not None})
        self.g.add_edge(src, dst, **edge_attrs)
        if self.include_reverse_edges and not kind.endswith("_rev"):
            rev_attrs = dict(edge_attrs)
            rev_attrs["kind"] = kind + "_rev"
            rev_attrs["flow"] = kind + "_rev"
            self.g.add_edge(dst, src, **rev_attrs)

    def make_action_point_node(self, ap: Dict[str, Any], loop_depth: int, mlir_iv: Optional[str], mlir_loop_node: str) -> str:
        label = ap["label"]
        ph = ap.get("placeholders") or []
        has_pipe = any(p.startswith("_PIPE_") for p in ph)
        has_unroll = any(p.startswith("_UNROLL_") for p in ph)
        has_array = any(p.startswith("_ARRAY_") for p in ph)
        nid = self.add_node(
            "action_point",
            f"{label}: {ap.get('named_label') or ap.get('kind')}",
            full_text=json.dumps(ap, sort_keys=True),
            label=label,
            label_id=int(label[1:]),
            function=ap.get("function"),
            ap_kind=ap.get("kind"),
            source_line=ap.get("line"),
            named_label=ap.get("named_label") or "",
            induction_var=ap.get("induction_var") or "",
            mlir_iv=mlir_iv or "",
            loop_depth=loop_depth,
            trip_count=ap.get("loop_bound_value") or 0,
            has_pipeline_slot=int(has_pipe),
            has_unroll_slot=int(has_unroll),
            has_array_partition_slot=int(has_array),
            is_llm_anchor=1,
            scope_category=1 if ap.get("kind") == "loop" else 2,
        )
        self.ap_node_by_label[label] = nid
        self.add_edge(nid, mlir_loop_node, "ap_anchors_mlir_loop")
        return nid

    def add_target_context(self) -> str:
        target = self.schema.get("target", {}) or {}
        text = f"target_context\\n{target.get('target_device')} @ {target.get('clock_period_ns')} ns"
        nid = self.add_node(
            "target_context",
            text,
            full_text=json.dumps(target, sort_keys=True),
            target_device=target.get("target_device") or "",
            clock_period_ns=target.get("clock_period_ns") or 0.0,
        )
        return nid

    def add_array_scope(self, function: str, name: str, memref_ssa: str, elem_type: str = "", dims: Optional[List[Optional[int]]] = None,
                        source_role: str = "argument", source_dims_expr: Optional[List[str]] = None) -> str:
        key = (function, memref_ssa)
        if key in self.memref_to_array_node:
            return self.memref_to_array_node[key]
        constants = self.schema.get("constants", {}) or {}
        dims = dims or []
        logical = infer_logical_shape(name, dims, constants)
        text = f"array_scope {name}"
        if logical:
            text += f"\\nshape={logical}"
        nid = self.add_node(
            "array_scope",
            text,
            full_text=text,
            function=function,
            array_name=name,
            memref_ssa=memref_ssa,
            elem_type=elem_type,
            source_role=source_role,
            dims=json.dumps(dims),
            logical_shape=json.dumps(logical),
            source_dims_expr=json.dumps(source_dims_expr or []),
            array_size=(math.prod([d for d in dims if isinstance(d, int)]) if dims and all(isinstance(d, int) for d in dims) else 0),
            is_array_scope=1,
        )
        self.memref_to_array_node[key] = nid
        return nid

    def create_source_array_scopes(self) -> None:
        # Add explicit array-scope nodes for labeled local arrays from ap_schema.
        for ap in self.schema.get("action_points", []):
            if ap.get("kind") != "array":
                continue
            function = ap.get("function", "GLOBAL")
            name = ap.get("array_name") or ap.get("label")
            dims = ap.get("array_dims_value") or []
            elem = ap.get("array_type") or ""
            nid = self.add_array_scope(function, name, f"source:{ap['label']}:{name}", elem, dims, "local_array", ap.get("array_dims_expr") or [])
            self.ap_node_by_label[ap["label"]] = nid
            # Mark this array_scope as an LLM/action anchor too.
            self.g.nodes[nid].update({
                "kind": "action_array_scope",
                "label": ap["label"],
                "label_id": int(ap["label"][1:]),
                "ap_kind": "array",
                "is_llm_anchor": 1,
                "scope_category": 2,
                "has_array_partition_slot": 1,
                "has_pipeline_slot": 0,
                "has_unroll_slot": 0,
            })

    def init_function_args(self, func: str, args_text: str) -> None:
        params = parse_source_function_params(self.schema.get("source_file"), func)
        constants = self.schema.get("constants", {}) or {}
        self.func_param_meta[func] = params
        mlir_args = list(ARG_RE.finditer(args_text))
        for idx, am in enumerate(mlir_args):
            ssa = am.group("ssa")
            typ = am.group("type")
            if not typ.startswith("memref"):
                continue
            dims, elem = parse_memref_shape(typ)
            pname = f"arg{idx}"
            source_dims_expr: List[str] = []
            if idx < len(params):
                pname = params[idx].get("name", pname)
                source_dims_expr = params[idx].get("dims_expr", []) or []
                eval_dims = [eval_dim_expr(d, constants) for d in source_dims_expr]
                if any(d is not None for d in eval_dims):
                    dims = eval_dims
            self.add_array_scope(func, pname, ssa, elem, dims, "function_arg", source_dims_expr)

    def current_parent_node(self) -> Optional[str]:
        if self.loop_stack:
            return self.loop_stack[-1].get("mlir_node")
        return None

    def current_ap_node(self) -> Optional[str]:
        if self.loop_stack:
            return self.loop_stack[-1].get("ap_node")
        return None

    def replace_iv_names(self, expr: str) -> str:
        out = expr
        # Replace innermost first so human labels appear in access strings.
        for item in self.loop_stack:
            if item.get("iv") and item.get("source_iv"):
                out = re.sub(rf"\b{re.escape(item['iv'])}\b", item["source_iv"], out)
        return out

    def parse_mlir(self, mlir_text: str) -> None:
        # Root + target context
        module = self.add_node("module", "module")
        target_node = self.add_target_context()
        self.add_edge(target_node, module, "target_conditions_module")
        self.create_source_array_scopes()

        stack: List[Dict[str, Any]] = [{"kind": "module", "node": module}]
        active_func_node: Optional[str] = None

        for lineno, line in enumerate(mlir_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            # Close regions.  Handle lines like '}' or '} attributes ...'.
            if stripped.startswith("}"):
                if self.loop_stack:
                    self.loop_stack.pop()
                if len(stack) > 1:
                    popped = stack.pop()
                    if popped.get("kind") == "function":
                        self.current_function = None
                        active_func_node = None
                continue

            fm = FUNC_RE.match(line)
            if fm:
                fname = fm.group("name")
                self.current_function = fname
                fn = self.add_node("function", f"func @{fname}", function=fname, full_text=stripped)
                self.add_edge(stack[-1]["node"], fn, "region_contains")
                stack.append({"kind": "function", "node": fn})
                active_func_node = fn
                self.init_function_args(fname, fm.group("args"))
                # Connect target to function.
                self.add_edge(target_node, fn, "target_conditions_function")
                continue

            # Constants.
            cm = CONST_RE.match(line)
            if cm and self.current_function:
                cn = self.add_node("constant", f"const {cm.group('value')}", function=self.current_function, full_text=stripped,
                                   result=cm.group("result"), value=cm.group("value"), elem_type=cm.group("type"))
                self.add_edge(stack[-1]["node"], cn, "region_contains")
                self.ssa_producer[cm.group("result")] = cn
                continue

            # Local memref allocation.
            am = ALLOCA_RE.match(line)
            if am and self.current_function:
                dims, elem = parse_memref_shape(am.group("type"))
                arr = self.add_array_scope(self.current_function, am.group("result"), am.group("result"), elem, dims, "alloca")
                self.add_edge(stack[-1]["node"], arr, "region_contains")
                self.ssa_producer[am.group("result")] = arr
                continue

            # Loops.
            lm = FOR_RE.match(line)
            if lm and self.current_function:
                iv = lm.group("iv")
                ub = lm.group("ub")
                dialect = lm.group("dialect")
                is_reduction = "iter_args" in (lm.group("rest") or "")
                depth = len(self.loop_stack) + 1
                mlir_loop = self.add_node(
                    "mlir_loop",
                    f"{dialect}.for {iv}=0..{ub}",
                    function=self.current_function,
                    full_text=stripped,
                    mlir_op=f"{dialect}.for",
                    mlir_iv=iv,
                    lb=lm.group("lb"),
                    ub=ub,
                    loop_depth=depth,
                    trip_count=extract_int(ub) or 0,
                    is_reduction_loop=int(is_reduction),
                )
                self.add_edge(stack[-1]["node"], mlir_loop, "region_contains")

                # Match this MLIR loop to the next source AP in the same function.
                ap_node = None
                source_iv = None
                if self.loop_queues[self.current_function]:
                    ap = self.loop_queues[self.current_function].pop(0)
                    source_iv = ap.get("induction_var")
                    ap_node = self.make_action_point_node(ap, depth, iv, mlir_loop)
                    self.mlir_loop_by_label[ap["label"]] = mlir_loop
                    # Target context and loop hierarchy at AP level.
                    self.add_edge(target_node, ap_node, "target_conditions_ap")
                    if self.loop_stack and self.loop_stack[-1].get("ap_node"):
                        self.add_edge(self.loop_stack[-1]["ap_node"], ap_node, "ap_contains")
                    if is_reduction:
                        self.g.nodes[ap_node]["is_reduction_loop"] = 1
                        self.add_edge(ap_node, mlir_loop, "ap_has_reduction_loop")
                else:
                    # Not all MLIR loops necessarily have Lk labels.
                    ap_node = None

                self.loop_stack.append({"mlir_node": mlir_loop, "ap_node": ap_node, "iv": iv, "source_iv": source_iv})
                stack.append({"kind": "loop", "node": mlir_loop})
                continue

            # Loads.
            ldm = LOAD_RE.match(line)
            if ldm and self.current_function:
                mem = ldm.group("mem")
                idx_raw = ldm.group("idx")
                idx_human = self.replace_iv_names(idx_raw)
                arr_node = self.memref_to_array_node.get((self.current_function, mem))
                if arr_node is None:
                    dims, elem = parse_memref_shape(ldm.group("type"))
                    arr_node = self.add_array_scope(self.current_function, mem, mem, elem, dims, "unknown_memref")

                load_node = self.add_node(
                    "access_load",
                    f"load {self.g.nodes[arr_node].get('array_name', mem)}[{idx_human}]",
                    function=self.current_function,
                    full_text=stripped,
                    result=ldm.group("result"),
                    memref_ssa=mem,
                    access_expr=idx_human,
                    raw_access_expr=idx_raw,
                    mlir_op=f"{ldm.group('dialect')}.load",
                )
                self.add_edge(stack[-1]["node"], load_node, "region_contains")
                self.add_edge(load_node, arr_node, "mlir_reads_array")
                self.ssa_producer[ldm.group("result")] = load_node
                # HLS shortcut: current action point reads this array.
                ap = self.current_ap_node()
                if ap:
                    self.add_edge(ap, arr_node, "hls_reads", access_expr=idx_human)
                    self.add_edge(ap, load_node, "ap_observes_load")
                continue

            # Stores.
            sm = STORE_RE.match(line)
            if sm and self.current_function:
                mem = sm.group("mem")
                val = sm.group("value")
                idx_raw = sm.group("idx")
                idx_human = self.replace_iv_names(idx_raw)
                arr_node = self.memref_to_array_node.get((self.current_function, mem))
                if arr_node is None:
                    dims, elem = parse_memref_shape(sm.group("type"))
                    arr_node = self.add_array_scope(self.current_function, mem, mem, elem, dims, "unknown_memref")

                store_node = self.add_node(
                    "access_store",
                    f"store {self.g.nodes[arr_node].get('array_name', mem)}[{idx_human}]",
                    function=self.current_function,
                    full_text=stripped,
                    value=val,
                    memref_ssa=mem,
                    access_expr=idx_human,
                    raw_access_expr=idx_raw,
                    mlir_op=f"{sm.group('dialect')}.store",
                )
                self.add_edge(stack[-1]["node"], store_node, "region_contains")
                prod = self.ssa_producer.get(val)
                if prod:
                    self.add_edge(prod, store_node, "mlir_def_use")
                self.add_edge(store_node, arr_node, "mlir_writes_array")
                ap = self.current_ap_node()
                if ap:
                    self.add_edge(ap, arr_node, "hls_writes", access_expr=idx_human)
                    self.add_edge(ap, store_node, "ap_observes_store")
                continue

            # Arithmetic ops.
            arm = ARITH_RE.match(line)
            if arm and self.current_function:
                op = arm.group("op")
                ops = re.findall(r"%[A-Za-z0-9_]+", arm.group("operands"))
                an = self.add_node(
                    "arith",
                    f"arith.{op}",
                    function=self.current_function,
                    full_text=stripped,
                    result=arm.group("result"),
                    mlir_op=f"arith.{op}",
                    elem_type=arm.group("type"),
                )
                self.add_edge(stack[-1]["node"], an, "region_contains")
                for o in ops:
                    prod = self.ssa_producer.get(o)
                    if prod:
                        self.add_edge(prod, an, "mlir_def_use")
                self.ssa_producer[arm.group("result")] = an
                ap = self.current_ap_node()
                if ap:
                    self.add_edge(ap, an, "ap_observes_compute")
                continue

            # affine.yield -- important for reductions.
            ym = YIELD_RE.match(line)
            if ym and self.current_function:
                yn = self.add_node("yield", "affine.yield", function=self.current_function, full_text=stripped)
                self.add_edge(stack[-1]["node"], yn, "region_contains")
                prod = self.ssa_producer.get(ym.group("value"))
                if prod:
                    self.add_edge(prod, yn, "mlir_def_use")
                ap = self.current_ap_node()
                if ap:
                    self.add_edge(ap, yn, "hls_reduction_yield")
                continue

            if RETURN_RE.match(line) and self.current_function:
                rn = self.add_node("return", "return", function=self.current_function, full_text=stripped)
                self.add_edge(stack[-1]["node"], rn, "region_contains")
                continue

    def connect_array_aps_to_arrays(self) -> None:
        # For labeled array action points, connect their anchor to same-name array scopes if present.
        name_to_arr = defaultdict(list)
        for n, d in self.g.nodes(data=True):
            if "array_scope" in d.get("kind", "") or d.get("is_array_scope"):
                name_to_arr[d.get("array_name")].append(n)
        for ap in self.schema.get("action_points", []):
            if ap.get("kind") != "array":
                continue
            ap_node = self.ap_node_by_label.get(ap["label"])
            for n in name_to_arr.get(ap.get("array_name"), []):
                if n != ap_node:
                    self.add_edge(ap_node, n, "ap_aliases_array_scope")

    def finalize(self) -> nx.MultiDiGraph:
        self.connect_array_aps_to_arrays()
        # Add deterministic index attribute.
        for i, n in enumerate(sorted(self.g.nodes(), key=lambda x: int(x))):
            self.g.nodes[n]["node_id"] = int(n)
        return self.g


# ---------------------------------------------------------------------------
# PyG conversion
# ---------------------------------------------------------------------------

def build_or_load_vocab(g: nx.MultiDiGraph, vocab_in: Optional[Path], vocab_out: Optional[Path]) -> Dict[str, Dict[str, int]]:
    if vocab_in and vocab_in.exists():
        return json.loads(read_text(vocab_in))

    node_kinds = sorted({str(d.get("kind", "unknown")) for _, d in g.nodes(data=True)})
    edge_kinds = sorted({str(d.get("kind", "unknown")) for _, _, d in g.edges(data=True)})
    mlir_ops = sorted({str(d.get("mlir_op", "")) for _, d in g.nodes(data=True) if d.get("mlir_op")})
    vocab = {
        "node_kind": {k: i for i, k in enumerate(node_kinds)},
        "edge_kind": {k: i for i, k in enumerate(edge_kinds)},
        "mlir_op": {k: i for i, k in enumerate(mlir_ops)},
    }
    if vocab_out:
        dump_json(vocab_out, vocab)
    return vocab


def one_hot(idx: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def node_features(d: Dict[str, Any], vocab: Dict[str, Dict[str, int]]) -> List[float]:
    nk = vocab["node_kind"]
    mo = vocab["mlir_op"]
    kind_vec = one_hot(nk.get(str(d.get("kind", "unknown")), -1), len(nk))
    op_vec = one_hot(mo.get(str(d.get("mlir_op", "")), -1), len(mo))
    numeric = [
        float(d.get("is_llm_anchor", 0) or 0),
        float(d.get("is_array_scope", 0) or 0),
        float(d.get("has_pipeline_slot", 0) or 0),
        float(d.get("has_unroll_slot", 0) or 0),
        float(d.get("has_array_partition_slot", 0) or 0),
        log2p(d.get("label_id", 0)),
        log2p(d.get("trip_count", 0)),
        log2p(d.get("loop_depth", 0)),
        float(d.get("is_reduction_loop", 0) or 0),
        log2p(d.get("array_size", 0)),
        float(d.get("clock_period_ns", 0.0) or 0.0) / 100.0,
    ]
    return kind_vec + op_vec + numeric


def edge_features(d: Dict[str, Any], vocab: Dict[str, Dict[str, int]]) -> List[float]:
    ek = vocab["edge_kind"]
    return one_hot(ek.get(str(d.get("kind", "unknown")), -1), len(ek))


def build_pyg_data(g: nx.MultiDiGraph, schema: Dict[str, Any], point: Dict[str, Any], vocab: Dict[str, Dict[str, int]],
                   perf: float, area: float, max_pragma_length: int) -> Optional[Any]:
    if not HAS_PYG:
        return None

    node_ids = sorted(g.nodes(), key=lambda x: int(x))
    N = len(node_ids)
    id_to_i = {n: i for i, n in enumerate(node_ids)}
    x = torch.tensor([node_features(g.nodes[n], vocab) for n in node_ids], dtype=torch.float32)

    edges = []
    eattrs = []
    for u, v, d in g.edges(data=True):
        edges.append([id_to_i[u], id_to_i[v]])
        eattrs.append(edge_features(d, vocab))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(eattrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(vocab["edge_kind"])), dtype=torch.float32)

    X_pipeline_scopeids = torch.zeros((N,), dtype=torch.bool)
    X_unroll_scopeids = torch.zeros((N,), dtype=torch.bool)
    X_array_partition_scopeids = torch.zeros((N,), dtype=torch.bool)
    X_llm_scopeids = torch.zeros((N,), dtype=torch.bool)
    X_llm_scopecat = torch.zeros((N,), dtype=torch.long)
    X_llm_labelid = torch.full((N,), -1, dtype=torch.long)
    X_pragma_per_node = torch.zeros((N, 5), dtype=torch.float32)
    X_pseudonids = torch.zeros((N,), dtype=torch.bool)
    X_arrayscopenids = torch.zeros((N,), dtype=torch.bool)
    X_contextnids = torch.zeros((N,), dtype=torch.bool)
    X_pragmanids = torch.zeros((N,), dtype=torch.bool)
    X_pragmascopenids = torch.zeros((N,), dtype=torch.bool)
    X_scopenids = torch.zeros((N,), dtype=torch.bool)
    X_icmpnids = torch.zeros((N,), dtype=torch.bool)

    for n in node_ids:
        i = id_to_i[n]
        d = g.nodes[n]
        kind = d.get("kind")
        if kind == "target_context":
            X_contextnids[i] = True
        if "array_scope" in str(kind) or d.get("is_array_scope"):
            X_arrayscopenids[i] = True
        if kind == "action_point":
            X_pseudonids[i] = True   # reuse existing pooling branch as AP/scope branch
            X_scopenids[i] = True
        if d.get("is_llm_anchor"):
            X_llm_scopeids[i] = True
            X_llm_scopecat[i] = int(d.get("scope_category", 0) or 0)
            X_llm_labelid[i] = int(d.get("label_id", -1) or -1)

        label = d.get("label")
        if label:
            if int(d.get("has_pipeline_slot", 0) or 0):
                X_pipeline_scopeids[i] = True
                X_pragma_per_node[i, 0] = pragma_value(point, f"_PIPE_{label}")
            if int(d.get("has_unroll_slot", 0) or 0):
                X_unroll_scopeids[i] = True
                X_pragma_per_node[i, 1] = pragma_value(point, f"_UNROLL_{label}")
            if int(d.get("has_array_partition_slot", 0) or 0):
                X_array_partition_scopeids[i] = True
                X_pragma_per_node[i, 2] = pragma_value(point, f"_ARRAY_T_{label}")
                X_pragma_per_node[i, 3] = pragma_value(point, f"_ARRAY_F_{label}")
                X_pragma_per_node[i, 4] = pragma_value(point, f"_ARRAY_D_{label}")

    # Flat pragma vector, mostly for compatibility/debugging.
    pragma_vals: List[int] = []
    for k, v in sorted(point.items()):
        if k.startswith(("_PIPE_", "_UNROLL_", "_ARRAY_T_", "_ARRAY_F_", "_ARRAY_D_")):
            pragma_vals.append(pragma_value(point, k))
    if len(pragma_vals) > max_pragma_length:
        raise RuntimeError(f"Pragma vector length {len(pragma_vals)} exceeds --max-pragma-length {max_pragma_length}")
    pragma_vals += [0] * (max_pragma_length - len(pragma_vals))
    pragmas = torch.tensor([pragma_vals], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.X_pipeline_scopeids = X_pipeline_scopeids
    data.X_unroll_scopeids = X_unroll_scopeids
    data.X_array_partition_scopeids = X_array_partition_scopeids
    data.X_llm_scopeids = X_llm_scopeids
    data.X_llm_scopecat = X_llm_scopecat
    data.X_llm_labelid = X_llm_labelid
    data.X_pragma_per_node = X_pragma_per_node
    data.X_pseudonids = X_pseudonids
    data.X_arrayscopenids = X_arrayscopenids
    data.X_contextnids = X_contextnids
    data.X_pragmanids = X_pragmanids
    data.X_pragmascopenids = X_pragmascopenids
    data.X_scopenids = X_scopenids
    data.X_icmpnids = X_icmpnids
    data.pragmas = pragmas
    data.perf = torch.tensor([float(perf)], dtype=torch.float32)
    data.area = torch.tensor([float(area)], dtype=torch.float32)
    data.actual_perf = torch.tensor([float(perf)], dtype=torch.float32)
    data.actual_area = torch.tensor([float(area)], dtype=torch.float32)
    return data


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

NODE_STYLE = {
    "module": ("gray90", "box"),
    "function": ("gray85", "box"),
    "target_context": ("lightgray", "box"),
    "action_point": ("orange", "oval"),
    "action_array_scope": ("deepskyblue", "oval"),
    "array_scope": ("lightskyblue", "cylinder"),
    "mlir_loop": ("tomato", "box"),
    "access_load": ("khaki", "box"),
    "access_store": ("gold", "box"),
    "arith": ("palegreen", "box"),
    "yield": ("mediumseagreen", "box"),
    "constant": ("white", "box"),
    "return": ("white", "box"),
}


def dot_escape(s: Any) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_dot(g: nx.MultiDiGraph, path: Path, title: str) -> None:
    lines = ["digraph G {", "  graph [rankdir=LR, splines=true, overlap=false, fontsize=18];", "  node [fontname=Helvetica, fontsize=10];", "  edge [fontname=Helvetica, fontsize=8];"]
    lines.append(f'  label="{dot_escape(title)}"; labelloc=t;')
    for n, d in sorted(g.nodes(data=True), key=lambda x: int(x[0])):
        kind = d.get("kind", "unknown")
        color, shape = NODE_STYLE.get(kind, ("white", "box"))
        label = d.get("text", kind)
        if d.get("label") and not str(label).startswith(str(d.get("label"))):
            label = f"{d.get('label')}\\n{label}"
        lines.append(f'  n{n} [label="{dot_escape(label)}", style=filled, fillcolor="{color}", shape={shape}];')
    for u, v, key, d in g.edges(keys=True, data=True):
        kind = d.get("kind", "edge")
        style = "dashed" if str(kind).startswith("hls_") or str(kind).startswith("ap_") or str(kind).startswith("target_") else "solid"
        color = "firebrick" if str(kind).startswith("hls_") else "black"
        label = kind
        if d.get("access_expr"):
            label += f"\\n{d.get('access_expr')}"
        lines.append(f'  n{u} -> n{v} [label="{dot_escape(label)}", style={style}, color="{color}"];')
    lines.append("}")
    write_text(path, "\n".join(lines) + "\n")


def render_dot(dot_path: Path, fmt: str) -> Optional[Path]:
    dot = shutil.which("dot")
    if not dot:
        return None
    out = dot_path.with_suffix(f".{fmt}")
    subprocess.run([dot, f"-T{fmt}", str(dot_path), "-o", str(out)], check=False)
    return out if out.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final MailoHLS-MLIR graph for GNN training/visualization.")
    p.add_argument("--mlir", required=True, help="Input affine/scf MLIR file, e.g. affine_opt.mlir or affine.mlir.")
    p.add_argument("--schema", required=True, help="ap_schema.json from emit_mlir.py.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--prefix", default="mailohls_mlir_graph", help="Output file prefix.")
    p.add_argument("--point-json", default=None, help="Optional concrete pragma design-point JSON.")
    p.add_argument("--perf", type=float, default=1.0, help="Optional/dummy performance target for PT output.")
    p.add_argument("--area", type=float, default=1.0, help="Optional/dummy area target for PT output.")
    p.add_argument("--max-pragma-length", type=int, default=128, help="Pad flat pragmas vector to this length.")
    p.add_argument("--no-reverse-edges", action="store_true", help="Disable reverse edges. Default: add reverse edges for message passing.")
    p.add_argument("--vocab-in", default=None, help="Optional existing vocab JSON for consistent dataset dimensions.")
    p.add_argument("--vocab-out", default=None, help="Optional vocab JSON output. Defaults to <prefix>_vocab.json.")
    p.add_argument("--no-pt", action="store_true", help="Do not emit PyG .pt file.")
    p.add_argument("--render", action="store_true", help="Try rendering DOT to SVG/PNG if Graphviz `dot` exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mlir_path = Path(args.mlir).resolve()
    schema_path = Path(args.schema).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(read_text(schema_path))
    point = load_point(Path(args.point_json).resolve() if args.point_json else None)

    builder = HLSGraphBuilder(schema, point, include_reverse_edges=not args.no_reverse_edges)
    builder.parse_mlir(read_text(mlir_path))
    g = builder.finalize()

    prefix = args.prefix
    gexf_path = out_dir / f"{prefix}.gexf"
    dot_path = out_dir / f"{prefix}.dot"
    vocab_path = Path(args.vocab_out).resolve() if args.vocab_out else out_dir / f"{prefix}_vocab.json"
    vocab_in = Path(args.vocab_in).resolve() if args.vocab_in else None

    # GEXF wants simple attribute types. Convert lists/dicts already serialized above.
    nx.write_gexf(g, gexf_path)
    write_dot(g, dot_path, title=f"MailoHLS-MLIR graph: {Path(schema.get('source_file', mlir_path.name)).name}")

    rendered = []
    if args.render:
        for fmt in ("svg", "png"):
            r = render_dot(dot_path, fmt)
            if r:
                rendered.append(str(r))

    vocab = build_or_load_vocab(g, vocab_in, vocab_path)
    pt_path = None
    if not args.no_pt:
        data = build_pyg_data(g, schema, point, vocab, perf=args.perf, area=args.area, max_pragma_length=args.max_pragma_length)
        if data is None:
            print("[WARN] torch/torch_geometric not available; skipped PT output.", file=sys.stderr)
        else:
            pt_path = out_dir / f"{prefix}.pt"
            torch.save(data, pt_path)

    summary = {
        "mlir": str(mlir_path),
        "schema": str(schema_path),
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "action_points": sorted(builder.ap_node_by_label.keys(), key=lambda x: int(x[1:])),
        "ap_node_by_label": builder.ap_node_by_label,
        "mlir_loop_by_label": builder.mlir_loop_by_label,
        "outputs": {
            "gexf": str(gexf_path),
            "dot": str(dot_path),
            "rendered": rendered,
            "pt": str(pt_path) if pt_path else None,
            "vocab": str(vocab_path),
        },
        "notes": [
            "Action-point nodes are the intended extraction points for MailoHLS structural memory.",
            "Raw MLIR nodes remain in the graph as compiler-structural evidence.",
            "HLS shortcut edges summarize loop-array access relations extracted from affine.load/store.",
        ],
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    dump_json(summary_path, summary)

    print("[OK] Built MailoHLS-MLIR graph")
    print("  GEXF:", gexf_path)
    print("  DOT :", dot_path)
    if rendered:
        for r in rendered:
            print("  Rendered:", r)
    elif args.render:
        print("  [WARN] Graphviz `dot` not found or render failed; DOT was still written.")
    if pt_path:
        print("  PT  :", pt_path)
    print("  Vocab:", vocab_path)
    print("  Summary:", summary_path)
    print("[NEXT] Inspect the DOT/SVG and verify each Lk maps to exactly one action_point node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
