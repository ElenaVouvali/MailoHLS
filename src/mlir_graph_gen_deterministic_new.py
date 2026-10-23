#!/usr/bin/env python3
"""
Deterministic MLIR graph construction for the MailoHLS GNN pipeline.

This file intentionally keeps the public structure and the serialized GEXF
contract of the original graph_gen_deterministic.py and of the first
mlir_graph_gen_deterministic.py prototype:

  * the same operation/value/immediate/pragma/array-scope node types;
  * the same four output folders used by data.py;
  * deterministic canonical relabeling and GEXF serialization;
  * pseudo-block, connected-block, and loop-hierarchy graph variants.

The important change is semantic, not cosmetic.  The original MLIR prototype
parsed one textual line at a time.  This version walks the actual MLIR
Operation -> Region -> Block -> Value object model through the official MLIR
Python bindings.  Consequently it preserves:

  * repeated operands and their positions;
  * operation results and block arguments;
  * real blocks, successors, nested regions, and structured-control entry/exit;
  * SCF/Affine loop nesting and loop-carried values;
  * function calls, actual/formal arguments, and returned values;
  * memref roots, views, read/write accesses, and conservative memory hazards;
  * a stable action contract between MailoHLS Lk placeholders and MLIR scopes.

The GNN architecture does not need to change.  data.py already learns one-hot
node/edge vocabularies, so the extra flow types below are consumed by the same
edge-aware TransformerConv backbone after regenerating encoders and retraining
the GNN.  Existing LLVM/ProGraML encoders and checkpoints must not be reused.

Required runtime
----------------
Use the MLIR Python package built from the same LLVM/MLIR revision as the
Polygeist/cgeist binary that produced the input.  This is normally exposed as:

    export PYTHONPATH=/path/to/llvm-build/tools/mlir/python_packages/mlir_core

An unrelated package named "mlir" from PyPI is not a substitute.

Recommended action manifest
---------------------------
MLIR transformations may rename SSA values, so action locations must not be
recovered by substring matching.  Give each optimization position a stable Lk
either as a mailohls.action_id operation attribute or with this manifest:

{
  "schema_version": 1,
  "actions": [
    {
      "id": "L4",
      "kind": "loop",
      "function": "gemv",
      "loop_ordinal": 0,
      "directives": ["pipeline", "unroll"]
    },
    {
      "id": "L1",
      "kind": "array",
      "function": "gemv",
      "argument_index": 0,
      "variable": "A",
      "directives": ["array_partition"]
    }
  ]
}

loop_ordinal is the deterministic preorder among loops in one function and is
provided only as a migration bridge.  The strongest contract is to preserve
mailohls.action_id = "Lk" directly on the MLIR operation.  For a publication
pipeline, run with --require-actions so every manifest action must resolve
exactly once and every pragma scope remains compatible with
gexf_to_pt_zero.py's Lk-aligned structural memory.

Scope of the dependence graph
-----------------------------
The default memory-dependence pass is deliberately conservative.  It emits
may-RAW, may-WAR, and may-WAW edges between accesses to the same traced memref
root.  It never claims that a dependence is absent merely because Python MLIR
bindings do not expose a complete affine/alias analysis.  Exact dependence
edges from a C++ MLIR analysis pass may be overlaid with
--dependence-manifest.  This distinction is essential for scientific claims:
the graph captures compiler-visible HLS-relevant dependencies, not post-route
timing, placement, routing, or resource-binding effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import networkx as nx


# ---------------------------------------------------------------------------
# MailoHLS-compatible node and edge schema.
# ---------------------------------------------------------------------------

PRAGMA_POSITION = {
    "PIPELINE": 0,
    "UNROLL": 1,
    "ARRAY_PARTITION": 2,
}

NODE_TYPE_OP = 0
NODE_TYPE_VALUE = 1
NODE_TYPE_IMMEDIATE = 2
NODE_TYPE_PSEUDO_BLOCK = 4
NODE_TYPE_PRAGMA = 100
NODE_TYPE_ARRAY_SCOPE = 104

FLOW_CONTROL = 0
FLOW_DATA = 1
FLOW_CALL = 2
FLOW_PSEUDO_BLOCK = 4
FLOW_PSEUDO_CONNECTED = 5
FLOW_LOOP_HIERARCHY = 6
FLOW_ARRAY_SCOPE = 7

# New MLIR relations.  data.py treats flow as a learned categorical feature,
# so these relations keep the existing GNN architecture unchanged.
FLOW_REGION = 8
FLOW_MEMORY_VIEW = 9
FLOW_MEMORY_ACCESS = 10
FLOW_LOOP_CARRIED = 11
FLOW_MEMORY_DEPENDENCE = 12

FLOW_PRAGMA = 200

ALL_FLOWS = {
    FLOW_CONTROL,
    FLOW_DATA,
    FLOW_CALL,
    FLOW_PSEUDO_BLOCK,
    FLOW_PSEUDO_CONNECTED,
    FLOW_LOOP_HIERARCHY,
    FLOW_ARRAY_SCOPE,
    FLOW_REGION,
    FLOW_MEMORY_VIEW,
    FLOW_MEMORY_ACCESS,
    FLOW_LOOP_CARRIED,
    FLOW_MEMORY_DEPENDENCE,
    FLOW_PRAGMA,
}

ARRAY_SCOPE_TEXT = "array_scope"
SCHEMA_VERSION = "mailohls-mlir-graph-v2"
ACTION_ID_RE = re.compile(r"^L([1-9][0-9]*)$")
ACTION_ID_SEARCH_RE = re.compile(r"\bL([1-9][0-9]*)\b")

LOOP_OPS = {
    "scf.for",
    "scf.forall",
    "scf.parallel",
    "scf.while",
    "affine.for",
    "affine.parallel",
}

RETURN_OPS = {"func.return", "llvm.return"}

VIEW_OPS = {
    "memref.cast",
    "memref.subview",
    "memref.reinterpret_cast",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.memory_space_cast",
    "bufferization.to_memref",
    "unrealized_conversion_cast",
}

READ_OPS = {
    "memref.load",
    "affine.load",
    "vector.load",
    "vector.transfer_read",
    "memref.prefetch",
}

WRITE_OPS = {
    "memref.store",
    "affine.store",
    "vector.store",
    "vector.transfer_write",
}

READ_WRITE_OPS = {
    "memref.atomic_rmw",
    "memref.generic_atomic_rmw",
}

MEMORY_DEPENDENCE_POSITION = {"RAW": 0, "WAR": 1, "WAW": 2}
MEMORY_ACCESS_POSITION = {
    ("read", "forward"): 0,
    ("read", "reverse"): 1,
    ("write", "forward"): 2,
    ("write", "reverse"): 3,
    ("readwrite", "forward"): 4,
    ("readwrite", "reverse"): 5,
}


# ---------------------------------------------------------------------------
# Small immutable records.  These make the graph construction auditable and
# keep the code close to the dataclass-based structure of the first prototype.
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    action_id: str
    kind: str
    function: str
    directives: tuple[str, ...]
    loop_ordinal: int | None = None
    location: str | None = None
    op_name: str | None = None
    argument_index: int | None = None
    value_name: str | None = None
    variable: str | None = None
    matched: bool = False


@dataclass
class BlockRecord:
    key: Any
    function_id: int
    block_id: int
    parent_op_node: int
    parent_op_block: int
    region_index: int
    arguments: list[Any] = field(default_factory=list)
    operations: list["OperationRecord"] = field(default_factory=list)


@dataclass
class OperationRecord:
    key: Any
    operation: Any
    node: int
    function_id: int
    function_name: str
    block_id: int
    block_key: Any
    block_order: int
    function_ordinal: int
    op_name: str
    parent_op_node: int | None
    parent_region_index: int | None
    loop_stack: tuple[int, ...]
    attributes: dict[str, str]
    location: str
    operands: list[Any]
    results: list[Any]


@dataclass
class LoopInfo:
    function_id: int
    function_name: str
    loop_ordinal: int
    op_record: OperationRecord
    op_node: int
    op_block: int
    body_blocks: list[int]
    scope_block: int
    parent_index: int | None
    children: list[int] = field(default_factory=list)
    action_id: str | None = None


@dataclass
class MemoryAccess:
    function_id: int
    op_node: int
    op_ordinal: int
    block_id: int
    root_node: int
    mode: str
    loop_stack: tuple[int, ...]


@dataclass
class ParseResult:
    graph: nx.MultiDiGraph
    functions: dict[int, str]
    function_name_to_id: dict[str, int]
    function_nodes: dict[int, int]
    function_arguments: dict[int, list[int]]
    loops: list[LoopInfo]
    blocks: dict[Any, BlockRecord]
    operation_records: list[OperationRecord]
    memory_accesses: list[MemoryAccess]
    actions: list[ActionSpec]


# ---------------------------------------------------------------------------
# Generic deterministic helpers, retained from the original implementation.
# ---------------------------------------------------------------------------

def require_pythonhashseed() -> None:
    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "Determinism requires PYTHONHASHSEED to be set before Python starts.\n"
            "Run, for example:\n"
            "  PYTHONHASHSEED=0 python mlir_graph_gen_deterministic.py ..."
        )


def det_sha_label(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def det_get_full_text(data: dict[str, Any]) -> str:
    if data.get("full_text") is not None:
        return str(data["full_text"])
    features = data.get("features")
    if isinstance(features, dict):
        full_text = features.get("full_text")
        if isinstance(full_text, list) and full_text:
            return str(full_text[0])
    return ""


def det_node_sort_key(node: Any, data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        det_get_full_text(data),
        str(data.get("action_id", "")),
        str(data.get("op_uid", "")),
        str(node),
    )


def det_edge_sort_key(
    source: Any,
    target: Any,
    data: dict[str, Any],
    node_rank: dict[Any, int],
) -> tuple[Any, ...]:
    return (
        node_rank.get(source, 10**18),
        node_rank.get(target, 10**18),
        int(data.get("flow", -1)),
        int(data.get("position", -1)),
        str(data.get("certainty", "")),
        str(source),
        str(target),
    )


def canonicalize_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    canonical = nx.MultiDiGraph()
    canonical.graph.update(deepcopy(graph.graph))
    nodes = sorted(graph.nodes(data=True), key=lambda item: det_node_sort_key(item[0], item[1]))
    for node, data in nodes:
        canonical.add_node(node, **deepcopy(data))

    rank = {node: index for index, (node, _) in enumerate(nodes)}
    edges = [
        (source, target, deepcopy(data))
        for source, target, _, data in graph.edges(keys=True, data=True)
    ]
    for edge_id, (source, target, data) in enumerate(
        sorted(edges, key=lambda edge: det_edge_sort_key(edge[0], edge[1], edge[2], rank))
    ):
        data["id"] = edge_id
        canonical.add_edge(source, target, key=edge_id, **data)
    return canonical


def relabel_nodes_canonically(graph: nx.MultiDiGraph, rounds: int = 3) -> nx.MultiDiGraph:
    labels = {
        node: det_sha_label(det_node_sort_key(node, data))
        for node, data in graph.nodes(data=True)
    }

    for _ in range(max(0, rounds)):
        new_labels: dict[Any, str] = {}
        for node in graph.nodes():
            outgoing = sorted(
                (
                    "o",
                    labels.get(target, ""),
                    int(data.get("flow", -1)),
                    int(data.get("position", -1)),
                )
                for _, target, _, data in graph.out_edges(node, keys=True, data=True)
            )
            incoming = sorted(
                (
                    "i",
                    labels.get(source, ""),
                    int(data.get("flow", -1)),
                    int(data.get("position", -1)),
                )
                for source, _, _, data in graph.in_edges(node, keys=True, data=True)
            )
            new_labels[node] = det_sha_label(
                {"self": labels.get(node, ""), "out": outgoing, "in": incoming}
            )
        labels = new_labels

    ordered = sorted(
        graph.nodes(),
        key=lambda node: (
            labels.get(node, ""),
            int(graph.in_degree(node)),
            int(graph.out_degree(node)),
            det_node_sort_key(node, graph.nodes[node]),
            str(node),
        ),
    )
    mapping = {old: new for new, old in enumerate(ordered)}
    return nx.relabel_nodes(graph, mapping, copy=True)


def stringify_attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prepare_graph_for_write(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    output = nx.MultiDiGraph()
    output.graph.update({key: stringify_attr(value) for key, value in graph.graph.items()})
    for node, data in graph.nodes(data=True):
        output.add_node(node, **{key: stringify_attr(value) for key, value in data.items()})
    for source, target, key, data in graph.edges(keys=True, data=True):
        output.add_edge(
            source,
            target,
            key=key,
            **{name: stringify_attr(value) for name, value in data.items()},
        )
    return output


def write_gexf_deterministic(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(prepare_graph_for_write(graph), path, prettyprint=False)


def prune_redundant_nodes(graph: nx.MultiDiGraph) -> None:
    while True:
        isolated = [
            node for node in sorted(graph.nodes(), key=str)
            if node is None or graph.degree(node) == 0
        ]
        if not isolated:
            return
        graph.remove_nodes_from(isolated)


def finalize_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    prune_redundant_nodes(graph)
    graph = canonicalize_graph(graph)
    graph = relabel_nodes_canonically(graph, rounds=3)
    return canonicalize_graph(graph)


# ---------------------------------------------------------------------------
# Action and exact-dependence manifests.
# ---------------------------------------------------------------------------

def _normalise_action_id(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.startswith("_L"):
        text = text[1:]
    match = ACTION_ID_SEARCH_RE.search(text)
    if not match:
        raise ValueError(f"Invalid MailoHLS action id: {value!r}; expected L1, L2, ...")
    action_id = f"L{int(match.group(1))}"
    if not ACTION_ID_RE.fullmatch(action_id):
        raise ValueError(f"Invalid MailoHLS action id: {value!r}")
    return action_id


def load_action_manifest(path: Path | None) -> list[ActionSpec]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_actions = payload
    elif isinstance(payload, dict):
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError(f"Unsupported action-manifest schema: {payload.get('schema_version')}")
        raw_actions = payload.get("actions", [])
    else:
        raise TypeError("Action manifest must be a JSON object or list.")

    actions: list[ActionSpec] = []
    seen: set[str] = set()
    for raw in raw_actions:
        action_id = _normalise_action_id(raw["id"])
        if action_id in seen:
            raise ValueError(f"Duplicate action id in manifest: {action_id}")
        seen.add(action_id)

        kind = str(raw["kind"]).strip().lower()
        if kind not in {"loop", "array"}:
            raise ValueError(f"Action {action_id}: kind must be loop or array, got {kind!r}")

        default_directives = ("pipeline", "unroll") if kind == "loop" else ("array_partition",)
        directives = tuple(
            str(item).strip().lower()
            for item in raw.get("directives", default_directives)
        )
        allowed = {"pipeline", "unroll"} if kind == "loop" else {"array_partition"}
        if not directives or not set(directives) <= allowed:
            raise ValueError(
                f"Action {action_id}: invalid directives {directives}; allowed={sorted(allowed)}"
            )

        function = str(raw.get("function", "")).strip()
        if not function:
            raise ValueError(f"Action {action_id}: function is required.")

        action = ActionSpec(
            action_id=action_id,
            kind=kind,
            function=function,
            directives=directives,
            loop_ordinal=(
                int(raw["loop_ordinal"]) if raw.get("loop_ordinal") is not None else None
            ),
            location=(str(raw["location"]) if raw.get("location") is not None else None),
            op_name=(str(raw["op_name"]) if raw.get("op_name") is not None else None),
            argument_index=(
                int(raw["argument_index"])
                if raw.get("argument_index") is not None
                else None
            ),
            value_name=(
                str(raw["value_name"]) if raw.get("value_name") is not None else None
            ),
            variable=(str(raw["variable"]) if raw.get("variable") is not None else None),
        )

        if kind == "array" and action.argument_index is None and action.value_name is None:
            raise ValueError(
                f"Array action {action_id}: provide argument_index or value_name."
            )
        actions.append(action)
    return actions


def load_dependence_manifest(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    dependencies = payload if isinstance(payload, list) else payload.get("dependencies", [])
    for dep in dependencies:
        if str(dep.get("kind", "")).upper() not in MEMORY_DEPENDENCE_POSITION:
            raise ValueError(f"Invalid exact memory dependence: {dep}")
        if "function" not in dep or "source_op" not in dep or "target_op" not in dep:
            raise ValueError(
                "Each exact dependence requires function, source_op, target_op, and kind."
            )
    return dependencies


# ---------------------------------------------------------------------------
# Thin compatibility layer over MLIR Python bindings.
# ---------------------------------------------------------------------------

def import_mlir_ir() -> Any:
    try:
        from mlir import ir  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official MLIR Python bindings. Build MLIR with "
            "MLIR_ENABLE_BINDINGS_PYTHON=ON and add the matching mlir_core package "
            "to PYTHONPATH. Do not install the unrelated PyPI package named mlir."
        ) from exc
    return ir


def raw_operation(operation: Any) -> Any:
    return getattr(operation, "operation", operation)


def object_key(obj: Any) -> tuple[str, int]:
    try:
        return (type(obj).__name__, hash(obj))
    except Exception:
        return (type(obj).__name__, id(obj))


def operation_name(operation: Any) -> str:
    return str(raw_operation(operation).name)


def operation_first_line(operation: Any) -> str:
    text = str(raw_operation(operation)).strip()
    return text.splitlines()[0].strip() if text else operation_name(operation)


def operation_location(operation: Any) -> str:
    try:
        return str(raw_operation(operation).location)
    except Exception:
        return "loc(unknown)"


def attribute_items(operation: Any) -> dict[str, str]:
    attrs = raw_operation(operation).attributes
    out: dict[str, str] = {}

    # Newer bindings expose items(); older versions expose indexed
    # NamedAttribute objects.  Supporting both keeps the script tied to the
    # MLIR object model without tying it to one minor Python API revision.
    try:
        for name, value in attrs.items():
            out[str(name)] = str(value)
        return dict(sorted(out.items()))
    except Exception:
        pass

    try:
        length = len(attrs)
    except Exception:
        length = 0
    for index in range(length):
        try:
            named = attrs[index]
            name = str(getattr(named, "name"))
            value = getattr(named, "attr", getattr(named, "attribute", ""))
            out[name] = str(value)
        except Exception:
            continue
    return dict(sorted(out.items()))


def get_attribute(operation: Any, *names: str) -> str | None:
    attrs = raw_operation(operation).attributes
    for name in names:
        try:
            return str(attrs[name])
        except Exception:
            continue
    return None


def strip_mlir_string(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip().strip('"').strip("'")
    if text.startswith("@"):
        text = text[1:]
    return text


def operation_regions(operation: Any) -> list[Any]:
    try:
        return list(raw_operation(operation).regions)
    except Exception:
        return []


def region_blocks(region: Any) -> list[Any]:
    try:
        return list(region.blocks)
    except Exception:
        try:
            return list(region)
        except Exception:
            return []


def block_operations(block: Any) -> list[Any]:
    try:
        return list(block.operations)
    except Exception:
        return list(block)


def block_arguments(block: Any) -> list[Any]:
    try:
        return list(block.arguments)
    except Exception:
        return []


def operation_operands(operation: Any) -> list[Any]:
    try:
        return list(raw_operation(operation).operands)
    except Exception:
        return []


def operation_results(operation: Any) -> list[Any]:
    try:
        return list(raw_operation(operation).results)
    except Exception:
        return []


def operation_successors(operation: Any) -> list[Any]:
    try:
        return list(raw_operation(operation).successors)
    except Exception:
        return []


def value_type(value: Any) -> str:
    try:
        return str(value.type)
    except Exception:
        return "unknown"


def value_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<value>"


def canonical_type_token(type_text: str) -> str:
    compact = re.sub(r"\s+", "", type_text)
    if compact.startswith("memref<"):
        rank = compact.split("<", 1)[1].split("x")
        dimensions = max(0, len(rank) - 1)
        element = rank[-1].split(",", 1)[0].rstrip(">")
        return f"memref_rank{dimensions}_{element}"
    if compact.startswith("tensor<"):
        rank = compact.split("<", 1)[1].split("x")
        dimensions = max(0, len(rank) - 1)
        element = rank[-1].split(",", 1)[0].rstrip(">")
        return f"tensor_rank{dimensions}_{element}"
    if compact.startswith("vector<"):
        return "vector"
    if compact.startswith("!llvm.ptr") or compact.startswith("ptr"):
        return "pointer"
    if re.fullmatch(r"[sui]?[0-9]+", compact):
        return compact
    if compact in {"index", "f16", "bf16", "f32", "f64"}:
        return compact
    return compact[:80] if compact else "unknown"


def is_memory_type(type_text: str) -> bool:
    compact = type_text.replace(" ", "")
    return (
        compact.startswith("memref<")
        or compact.startswith("!llvm.ptr")
        or compact.startswith("ptr")
    )


def parse_integer_attr(text: str | None) -> int | None:
    if text is None:
        return None
    match = re.search(r"(?<![A-Za-z0-9_])-?[0-9]+", text)
    return int(match.group(0)) if match else None


def action_id_from_attributes(attributes: dict[str, str]) -> str | None:
    candidates = (
        "mailohls.action_id",
        "hls.action_id",
        "action_id",
        "mailohls.label",
        "hls.label",
    )
    for name in candidates:
        if name not in attributes:
            continue
        try:
            return _normalise_action_id(attributes[name])
        except ValueError:
            continue
    return None


def parse_mlir_module(path: Path, allow_unregistered_dialects: bool) -> tuple[Any, Any, str]:
    ir = import_mlir_ir()
    text = path.read_text(encoding="utf-8")
    context = ir.Context()
    context.allow_unregistered_dialects = allow_unregistered_dialects
    context.__enter__()
    try:
        module = ir.Module.parse(text)
    except Exception:
        context.__exit__(*sys.exc_info())
        raise
    # The caller keeps context alive until graph construction finishes.
    return context, module, text


# ---------------------------------------------------------------------------
# MLIR object-model graph builder.
# ---------------------------------------------------------------------------

class MlirGraphBuilder:
    def __init__(
        self,
        module: Any,
        mlir_text: str,
        actions: list[ActionSpec],
        exact_dependencies: list[dict[str, Any]],
        conservative_memory_dependencies: bool = True,
        require_actions: bool = False,
    ) -> None:
        self.module = module
        self.mlir_text = mlir_text
        self.actions = actions
        self.exact_dependencies = exact_dependencies
        self.conservative_memory_dependencies = conservative_memory_dependencies
        self.require_actions = require_actions

        self.graph = nx.MultiDiGraph()
        self.graph.graph["schema_version"] = SCHEMA_VERSION
        self.graph.graph["input_sha256"] = hashlib.sha256(
            mlir_text.encode("utf-8")
        ).hexdigest()

        self.next_node_id = 0
        self.next_block_id = 0
        self.next_function_op_ordinal: dict[int, int] = defaultdict(int)

        self.functions: dict[int, str] = {}
        self.function_name_to_id: dict[str, int] = {}
        self.function_nodes: dict[int, int] = {}
        self.function_arguments: dict[int, list[int]] = {}
        self.function_operations: dict[int, Any] = {}
        self.function_returns: dict[int, list[OperationRecord]] = defaultdict(list)

        self.blocks: dict[Any, BlockRecord] = {}
        self.operation_records: list[OperationRecord] = []
        self.operation_by_key: dict[Any, OperationRecord] = {}
        self.operation_by_uid: dict[tuple[str, int], OperationRecord] = {}
        self.value_nodes: dict[Any, int] = {}
        self.value_def_op: dict[Any, int] = {}
        self.value_is_block_argument: set[Any] = set()
        self.constant_values: dict[Any, int] = {}

        self.loops: list[LoopInfo] = []
        self.loop_count_by_function: dict[int, int] = defaultdict(int)
        self.memory_root_by_value: dict[Any, int] = {}
        self.memory_accesses: list[MemoryAccess] = []
        self.block_edges: set[tuple[int, int, int, int]] = set()
        self.attached_action_ids: set[str] = set()

    def build(self) -> ParseResult:
        functions = self._discover_functions()
        if not functions:
            raise RuntimeError("No func.func or llvm.func operation found in the MLIR module.")

        self.function_name_to_id = {
            name: index for index, (name, _) in enumerate(functions)
        }
        for name, operation in functions:
            function_id = self.function_name_to_id[name]
            self.functions[function_id] = name
            self.function_operations[function_id] = operation
            self._index_function(function_id, name, operation)

        self._add_ssa_edges()
        for loop in self.loops:
            self._annotate_loop_features(loop)
        self._add_control_and_region_edges()
        self._add_loop_carried_edges()
        self._add_call_edges()
        self._build_memory_relations()
        self._attach_actions()

        self.graph.graph["block_edges"] = [
            list(edge) for edge in sorted(self.block_edges)
        ]
        self.graph.graph["function_names"] = self.functions
        self.graph.graph["action_ids"] = sorted(
            spec.action_id for spec in self.actions if spec.matched
        )
        self.graph = finalize_graph(self.graph)

        # Canonical relabeling changes graph node ids.  Update records through
        # stable op_uid/action_id attributes so hierarchy construction remains
        # correct without relying on pre-canonical integer ids.
        self._refresh_record_node_ids()

        return ParseResult(
            graph=self.graph,
            functions=self.functions,
            function_name_to_id=self.function_name_to_id,
            function_nodes=self.function_nodes,
            function_arguments=self.function_arguments,
            loops=self.loops,
            blocks=self.blocks,
            operation_records=self.operation_records,
            memory_accesses=self.memory_accesses,
            actions=self.actions,
        )

    def _new_node(self, attrs: dict[str, Any]) -> int:
        node = self.next_node_id
        self.next_node_id += 1
        self.graph.add_node(node, **attrs)
        return node

    def _discover_functions(self) -> list[tuple[str, Any]]:
        found: list[tuple[str, Any]] = []

        def visit(operation: Any) -> None:
            name = operation_name(operation)
            if name in {"func.func", "llvm.func"}:
                symbol = strip_mlir_string(
                    get_attribute(operation, "sym_name", "function_name")
                )
                if not symbol:
                    raise RuntimeError(
                        f"Function operation has no sym_name: {operation_first_line(operation)}"
                    )
                found.append((symbol, operation))
                return
            for region in operation_regions(operation):
                for block in region_blocks(region):
                    for child in block_operations(block):
                        visit(child)

        visit(self.module.operation)
        names = [name for name, _ in found]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate function symbols in MLIR module: {names}")
        return sorted(found, key=lambda item: item[0])

    def _assign_blocks(self, operation: Any, function_id: int) -> None:
        for region_index, region in enumerate(operation_regions(operation)):
            for block in region_blocks(region):
                key = object_key(block)
                if key not in self.blocks:
                    self.blocks[key] = BlockRecord(
                        key=key,
                        function_id=function_id,
                        block_id=self.next_block_id,
                        parent_op_node=-1,
                        parent_op_block=-1,
                        region_index=region_index,
                        arguments=block_arguments(block),
                    )
                    self.next_block_id += 1
                for child in block_operations(block):
                    self._assign_blocks(child, function_id)

    def _entry_block_id(self, function_operation: Any) -> int:
        regions = operation_regions(function_operation)
        if not regions or not region_blocks(regions[0]):
            # Declaration-only functions get one synthetic deterministic block.
            block_id = self.next_block_id
            self.next_block_id += 1
            return block_id
        return self.blocks[object_key(region_blocks(regions[0])[0])].block_id

    def _index_function(self, function_id: int, function_name: str, operation: Any) -> None:
        self._assign_blocks(operation, function_id)
        entry_block = self._entry_block_id(operation)
        attributes = attribute_items(operation)
        function_node = self._new_node(
            {
                "block": entry_block,
                "function": function_id,
                "text": operation_name(operation),
                "type": NODE_TYPE_OP,
                "full_text": operation_first_line(operation),
                "mlir_attrs": attributes,
                "source_location": operation_location(operation),
                "op_uid": f"{function_name}:function",
                "is_function": 1,
            }
        )
        self.function_nodes[function_id] = function_node

        self._index_regions(
            owner_operation=operation,
            owner_node=function_node,
            owner_block=entry_block,
            function_id=function_id,
            function_name=function_name,
            loop_stack=(),
        )

        regions = operation_regions(operation)
        if regions and region_blocks(regions[0]):
            args = block_arguments(region_blocks(regions[0])[0])
            self.function_arguments[function_id] = [
                self.value_nodes[object_key(argument)] for argument in args
            ]
            for argument_index, node in enumerate(self.function_arguments[function_id]):
                self.graph.nodes[node]["function_argument_index"] = argument_index
        else:
            self.function_arguments[function_id] = []

    def _index_regions(
        self,
        owner_operation: Any,
        owner_node: int,
        owner_block: int,
        function_id: int,
        function_name: str,
        loop_stack: tuple[int, ...],
    ) -> None:
        for region_index, region in enumerate(operation_regions(owner_operation)):
            for block in region_blocks(region):
                block_key = object_key(block)
                record = self.blocks[block_key]
                record.parent_op_node = owner_node
                record.parent_op_block = owner_block
                record.region_index = region_index

                for arg_index, argument in enumerate(block_arguments(block)):
                    self._get_or_create_value(
                        argument,
                        function_id,
                        record.block_id,
                        is_block_argument=True,
                        argument_index=arg_index,
                    )

                operations = block_operations(block)
                for block_order, operation in enumerate(operations):
                    op_name = operation_name(operation)
                    attributes = attribute_items(operation)
                    ordinal = self.next_function_op_ordinal[function_id]
                    self.next_function_op_ordinal[function_id] += 1
                    op_uid = f"{function_name}:op{ordinal}"

                    node = self._new_node(
                        {
                            "block": record.block_id,
                            "function": function_id,
                            "text": op_name,
                            "type": NODE_TYPE_OP,
                            "full_text": operation_first_line(operation),
                            "mlir_attrs": attributes,
                            "source_location": operation_location(operation),
                            "op_uid": op_uid,
                            "op_ordinal": ordinal,
                            "is_loop": 1 if op_name in LOOP_OPS else 0,
                            "loop_depth": len(loop_stack),
                            "operand_count": len(operation_operands(operation)),
                            "result_count": len(operation_results(operation)),
                            "operand_types": [value_type(v) for v in operation_operands(operation)],
                            "result_types": [value_type(v) for v in operation_results(operation)],
                        }
                    )
                    op_record = OperationRecord(
                        key=object_key(raw_operation(operation)),
                        operation=operation,
                        node=node,
                        function_id=function_id,
                        function_name=function_name,
                        block_id=record.block_id,
                        block_key=block_key,
                        block_order=block_order,
                        function_ordinal=ordinal,
                        op_name=op_name,
                        parent_op_node=owner_node,
                        parent_region_index=region_index,
                        loop_stack=loop_stack,
                        attributes=attributes,
                        location=operation_location(operation),
                        operands=operation_operands(operation),
                        results=operation_results(operation),
                    )
                    record.operations.append(op_record)
                    self.operation_records.append(op_record)
                    self.operation_by_key[op_record.key] = op_record
                    self.operation_by_uid[(function_name, ordinal)] = op_record

                    for result_index, result in enumerate(op_record.results):
                        self._get_or_create_value(
                            result,
                            function_id,
                            record.block_id,
                            is_block_argument=False,
                            argument_index=result_index,
                        )
                        self.value_def_op[object_key(result)] = node

                    if op_name in RETURN_OPS:
                        self.function_returns[function_id].append(op_record)

                    child_loop_stack = loop_stack
                    if op_name in LOOP_OPS:
                        loop_index = len(self.loops)
                        parent_index = loop_stack[-1] if loop_stack else None
                        body_blocks = [
                            self.blocks[object_key(child_block)].block_id
                            for child_region in operation_regions(operation)
                            for child_block in region_blocks(child_region)
                        ]
                        scope_block = body_blocks[0] if body_blocks else record.block_id
                        loop_ordinal = self.loop_count_by_function[function_id]
                        self.loop_count_by_function[function_id] += 1
                        loop = LoopInfo(
                            function_id=function_id,
                            function_name=function_name,
                            loop_ordinal=loop_ordinal,
                            op_record=op_record,
                            op_node=node,
                            op_block=record.block_id,
                            body_blocks=body_blocks,
                            scope_block=scope_block,
                            parent_index=parent_index,
                        )
                        self.loops.append(loop)
                        if parent_index is not None:
                            self.loops[parent_index].children.append(loop_index)
                        child_loop_stack = loop_stack + (loop_index,)
                    self._index_regions(
                        owner_operation=operation,
                        owner_node=node,
                        owner_block=record.block_id,
                        function_id=function_id,
                        function_name=function_name,
                        loop_stack=child_loop_stack,
                    )

    def _get_or_create_value(
        self,
        value: Any,
        function_id: int,
        block_id: int,
        is_block_argument: bool,
        argument_index: int,
    ) -> int:
        key = object_key(value)
        if key in self.value_nodes:
            return self.value_nodes[key]
        type_text = value_type(value)
        node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                # Normalized type token avoids learning unstable SSA names.
                "text": canonical_type_token(type_text),
                "type": NODE_TYPE_VALUE,
                "full_text": f"{value_text(value)} : {type_text}",
                "value_name": value_text(value),
                "value_type": type_text,
                "is_block_argument": 1 if is_block_argument else 0,
                "argument_index": argument_index,
                "is_memory": 1 if is_memory_type(type_text) else 0,
            }
        )
        self.value_nodes[key] = node
        if is_block_argument:
            self.value_is_block_argument.add(key)
        return node

    def _add_ssa_edges(self) -> None:
        for record in sorted(
            self.operation_records,
            key=lambda item: (item.function_id, item.function_ordinal),
        ):
            for position, operand in enumerate(record.operands):
                value_node = self._get_or_create_value(
                    operand,
                    record.function_id,
                    record.block_id,
                    is_block_argument=False,
                    argument_index=-1,
                )
                self.graph.add_edge(
                    value_node,
                    record.node,
                    flow=FLOW_DATA,
                    position=position,
                    role="operand",
                )

            for position, result in enumerate(record.results):
                value_node = self.value_nodes[object_key(result)]
                self.graph.add_edge(
                    record.node,
                    value_node,
                    flow=FLOW_DATA,
                    position=position,
                    role="result",
                )

            if record.op_name in {"arith.constant", "index.constant", "llvm.mlir.constant"}:
                raw_value = record.attributes.get("value")
                if raw_value is None:
                    raw_value = operation_first_line(record.operation)
                integer = parse_integer_attr(raw_value)
                text = str(integer) if integer is not None else str(raw_value)[:80]
                immediate = self._new_node(
                    {
                        "block": record.block_id,
                        "function": record.function_id,
                        "text": text,
                        "type": NODE_TYPE_IMMEDIATE,
                        "full_text": str(raw_value),
                        "feature_kind": "constant",
                    }
                )
                self.graph.add_edge(
                    immediate,
                    record.node,
                    flow=FLOW_DATA,
                    position=1000,
                    role="immediate",
                )
                if integer is not None:
                    for result in record.results:
                        self.constant_values[object_key(result)] = integer

    def _add_control_and_region_edges(self) -> None:
        for block in sorted(self.blocks.values(), key=lambda item: item.block_id):
            operations = sorted(block.operations, key=lambda item: item.block_order)
            for left, right in zip(operations[:-1], operations[1:]):
                self.graph.add_edge(
                    left.node,
                    right.node,
                    flow=FLOW_CONTROL,
                    position=0,
                    role="next_in_block",
                )

            if operations:
                first = operations[0]
                last = operations[-1]
                self.graph.add_edge(
                    block.parent_op_node,
                    first.node,
                    flow=FLOW_REGION,
                    position=2 * block.region_index,
                    role="region_entry",
                )
                self.graph.add_edge(
                    last.node,
                    block.parent_op_node,
                    flow=FLOW_REGION,
                    position=2 * block.region_index + 1,
                    role="region_exit",
                )

            if block.parent_op_block >= 0 and block.parent_op_block != block.block_id:
                self.block_edges.add(
                    (
                        block.function_id,
                        block.parent_op_block,
                        block.block_id,
                        10 + 2 * block.region_index,
                    )
                )
                self.block_edges.add(
                    (
                        block.function_id,
                        block.block_id,
                        block.parent_op_block,
                        11 + 2 * block.region_index,
                    )
                )

            if not operations:
                continue
            terminator = operations[-1]
            for successor_index, successor in enumerate(
                operation_successors(terminator.operation)
            ):
                successor_record = self.blocks.get(object_key(successor))
                if successor_record is None:
                    continue
                successor_ops = sorted(
                    successor_record.operations,
                    key=lambda item: item.block_order,
                )
                if successor_ops:
                    self.graph.add_edge(
                        terminator.node,
                        successor_ops[0].node,
                        flow=FLOW_CONTROL,
                        position=1 + successor_index,
                        role="block_successor",
                    )
                self.block_edges.add(
                    (
                        block.function_id,
                        block.block_id,
                        successor_record.block_id,
                        1 + successor_index,
                    )
                )

    def _constant_from_value(self, value: Any) -> int | None:
        return self.constant_values.get(object_key(value))

    def _annotate_loop_features(self, loop: LoopInfo) -> None:
        record = loop.op_record
        lower: int | None = None
        upper: int | None = None
        step: int | None = None

        if record.op_name == "scf.for" and len(record.operands) >= 3:
            lower = self._constant_from_value(record.operands[0])
            upper = self._constant_from_value(record.operands[1])
            step = self._constant_from_value(record.operands[2])
        elif record.op_name == "affine.for":
            line = operation_first_line(record.operation)
            match = re.search(
                r"affine\.for\s+%[^=]+\s*=\s*(-?[0-9]+)\s+to\s+(-?[0-9]+)"
                r"(?:\s+step\s+(-?[0-9]+))?",
                line,
            )
            if match:
                lower = int(match.group(1))
                upper = int(match.group(2))
                step = int(match.group(3) or 1)

        trip_count: int | None = None
        if lower is not None and upper is not None and step not in (None, 0):
            if step > 0:
                trip_count = max(0, math.ceil((upper - lower) / step))
            else:
                trip_count = max(0, math.ceil((lower - upper) / (-step)))

        data = self.graph.nodes[loop.op_node]
        data["loop_ordinal"] = loop.loop_ordinal
        data["loop_lower"] = lower if lower is not None else -1
        data["loop_upper"] = upper if upper is not None else -1
        data["loop_step"] = step if step is not None else -1
        data["trip_count"] = trip_count if trip_count is not None else -1
        data["trip_count_static"] = 1 if trip_count is not None else 0

        if trip_count is not None:
            bucket = 0 if trip_count <= 1 else int(math.ceil(math.log2(trip_count)))
            feature = self._new_node(
                {
                    "block": loop.op_block,
                    "function": loop.function_id,
                    "text": f"tripcount_log2_{bucket}",
                    "type": NODE_TYPE_IMMEDIATE,
                    "full_text": str(trip_count),
                    "feature_kind": "loop_trip_count",
                }
            )
            self.graph.add_edge(
                feature,
                loop.op_node,
                flow=FLOW_DATA,
                position=900,
                role="loop_trip_count",
            )

    def _add_loop_carried_edges(self) -> None:
        for loop in self.loops:
            record = loop.op_record
            regions = operation_regions(record.operation)
            if not regions or not region_blocks(regions[0]):
                continue
            body = region_blocks(regions[0])[0]
            arguments = block_arguments(body)

            if record.op_name == "scf.for":
                # operands: lower, upper, step, init_args...
                init_values = record.operands[3:]
                iter_args = arguments[1:] if arguments else []
            elif record.op_name == "affine.for":
                # Affine bounds may also be SSA operands.  Align iter_args from
                # the end so we never invent a dependency for a bound operand.
                iter_args = arguments[1:] if arguments else []
                init_values = record.operands[-len(iter_args):] if iter_args else []
            else:
                continue

            for position, (initial, argument) in enumerate(zip(init_values, iter_args)):
                initial_node = self.value_nodes[object_key(initial)]
                argument_node = self.value_nodes[object_key(argument)]
                self.graph.add_edge(
                    initial_node,
                    argument_node,
                    flow=FLOW_LOOP_CARRIED,
                    position=position,
                    role="iter_init",
                )

            body_key = object_key(body)
            body_record = self.blocks.get(body_key)
            if body_record is None or not body_record.operations:
                continue
            terminator = body_record.operations[-1]
            if terminator.op_name not in {"scf.yield", "affine.yield"}:
                continue
            loop_results = record.results
            for position, yielded in enumerate(terminator.operands):
                yielded_node = self.value_nodes[object_key(yielded)]
                if position < len(iter_args):
                    iter_node = self.value_nodes[object_key(iter_args[position])]
                    self.graph.add_edge(
                        yielded_node,
                        iter_node,
                        flow=FLOW_LOOP_CARRIED,
                        position=100 + position,
                        role="loop_backedge",
                    )
                if position < len(loop_results):
                    result_node = self.value_nodes[object_key(loop_results[position])]
                    self.graph.add_edge(
                        yielded_node,
                        result_node,
                        flow=FLOW_LOOP_CARRIED,
                        position=200 + position,
                        role="loop_result",
                    )

    def _callee_name(self, record: OperationRecord) -> str:
        if record.op_name not in {"func.call", "llvm.call"}:
            return ""
        for name in ("callee", "callee_name"):
            if name in record.attributes:
                return strip_mlir_string(record.attributes[name])
        line = operation_first_line(record.operation)
        match = re.search(r"@([A-Za-z_.$0-9-]+)", line)
        return match.group(1) if match else ""

    def _add_call_edges(self) -> None:
        for record in self.operation_records:
            callee_name = self._callee_name(record)
            if not callee_name or callee_name not in self.function_name_to_id:
                continue
            callee_id = self.function_name_to_id[callee_name]
            callee_node = self.function_nodes[callee_id]
            self.graph.add_edge(
                record.node,
                callee_node,
                flow=FLOW_CALL,
                position=0,
                role="calls",
            )

            formals = self.function_arguments.get(callee_id, [])
            for position, (actual, formal_node) in enumerate(zip(record.operands, formals)):
                actual_node = self.value_nodes[object_key(actual)]
                self.graph.add_edge(
                    actual_node,
                    formal_node,
                    flow=FLOW_CALL,
                    position=1 + position,
                    role="actual_to_formal",
                )

            returns = self.function_returns.get(callee_id, [])
            if not returns:
                continue
            # Multiple returns are conservatively connected to the same call
            # results; control edges decide which return is executable.
            for return_record in returns:
                for position, (returned, call_result) in enumerate(
                    zip(return_record.operands, record.results)
                ):
                    returned_node = self.value_nodes[object_key(returned)]
                    call_result_node = self.value_nodes[object_key(call_result)]
                    self.graph.add_edge(
                        returned_node,
                        call_result_node,
                        flow=FLOW_CALL,
                        position=100 + position,
                        role="return_to_call",
                    )

    def _memory_operands(self, record: OperationRecord) -> list[tuple[int, str]]:
        memory_positions = [
            index
            for index, value in enumerate(record.operands)
            if is_memory_type(value_type(value))
        ]
        if not memory_positions:
            return []

        name = record.op_name
        if name == "memref.copy":
            out: list[tuple[int, str]] = []
            if len(memory_positions) >= 1:
                out.append((memory_positions[0], "read"))
            if len(memory_positions) >= 2:
                out.append((memory_positions[1], "write"))
            return out
        if name in READ_OPS:
            return [(memory_positions[0], "read")]
        if name in WRITE_OPS:
            return [(memory_positions[0], "write")]
        if name in READ_WRITE_OPS:
            return [(memory_positions[0], "readwrite")]
        if name.startswith("linalg."):
            return [(position, "readwrite") for position in memory_positions]
        if name in {"func.call", "llvm.call"}:
            # Without an interprocedural effect summary, treating memref actuals
            # as read-write is conservative and scientifically safer than
            # silently assuming purity.
            return [(position, "readwrite") for position in memory_positions]
        if "load" in name or name.endswith(".read"):
            return [(memory_positions[0], "read")]
        if "store" in name or name.endswith(".write"):
            return [(memory_positions[0], "write")]
        if "atomic" in name:
            return [(memory_positions[0], "readwrite")]
        return []

    def _build_memory_relations(self) -> None:
        # First establish canonical roots.  View-like results inherit the root
        # of their first memory operand; allocations, globals, arguments, and
        # unknown producers remain distinct roots.
        for key, node in self.value_nodes.items():
            if int(self.graph.nodes[node].get("is_memory", 0)) == 1:
                self.memory_root_by_value[key] = node

        for record in sorted(
            self.operation_records,
            key=lambda item: (item.function_id, item.function_ordinal),
        ):
            memory_operands = [
                operand
                for operand in record.operands
                if is_memory_type(value_type(operand))
            ]
            if record.op_name in VIEW_OPS and memory_operands:
                source_key = object_key(memory_operands[0])
                root = self.memory_root_by_value.get(
                    source_key,
                    self.value_nodes[source_key],
                )
                for result in record.results:
                    if not is_memory_type(value_type(result)):
                        continue
                    result_key = object_key(result)
                    result_node = self.value_nodes[result_key]
                    self.memory_root_by_value[result_key] = root
                    self.graph.add_edge(
                        root,
                        result_node,
                        flow=FLOW_MEMORY_VIEW,
                        position=0,
                        role="view_of",
                    )
                    self.graph.add_edge(
                        result_node,
                        root,
                        flow=FLOW_MEMORY_VIEW,
                        position=1,
                        role="view_to_root",
                    )

            for result in record.results:
                key = object_key(result)
                if key in self.memory_root_by_value:
                    root = self.memory_root_by_value[key]
                    self.graph.nodes[root]["is_memory_root"] = 1

        for record in sorted(
            self.operation_records,
            key=lambda item: (item.function_id, item.function_ordinal),
        ):
            for operand_index, mode in self._memory_operands(record):
                value = record.operands[operand_index]
                key = object_key(value)
                value_node = self.value_nodes[key]
                root = self.memory_root_by_value.get(key, value_node)
                self.graph.nodes[root]["is_memory_root"] = 1

                if mode == "read":
                    forward, reverse = root, record.node
                elif mode == "write":
                    forward, reverse = record.node, root
                else:
                    forward, reverse = root, record.node

                self.graph.add_edge(
                    forward,
                    reverse,
                    flow=FLOW_MEMORY_ACCESS,
                    position=MEMORY_ACCESS_POSITION[(mode, "forward")],
                    role=mode,
                    operand_index=operand_index,
                )
                self.graph.add_edge(
                    reverse,
                    forward,
                    flow=FLOW_MEMORY_ACCESS,
                    position=MEMORY_ACCESS_POSITION[(mode, "reverse")],
                    role=f"{mode}_reverse",
                    operand_index=operand_index,
                )

                self.memory_accesses.append(
                    MemoryAccess(
                        function_id=record.function_id,
                        op_node=record.node,
                        op_ordinal=record.function_ordinal,
                        block_id=record.block_id,
                        root_node=root,
                        mode=mode,
                        loop_stack=record.loop_stack,
                    )
                )

        if self.conservative_memory_dependencies:
            self._add_conservative_memory_dependencies()
        self._add_exact_memory_dependencies()

    def _add_memory_dependence(
        self,
        source: int,
        target: int,
        kind: str,
        certainty: str,
        distance: Any = None,
    ) -> None:
        if source == target:
            return
        self.graph.add_edge(
            source,
            target,
            flow=FLOW_MEMORY_DEPENDENCE,
            position=MEMORY_DEPENDENCE_POSITION[kind],
            role=kind,
            certainty=certainty,
            distance=[] if distance is None else distance,
        )

    def _add_conservative_memory_dependencies(self) -> None:
        groups: dict[tuple[int, int], list[MemoryAccess]] = defaultdict(list)
        for access in self.memory_accesses:
            groups[(access.function_id, access.root_node)].append(access)

        for accesses in groups.values():
            accesses.sort(key=lambda item: item.op_ordinal)
            last_write: MemoryAccess | None = None
            reads_since_write: list[MemoryAccess] = []

            for access in accesses:
                reads = access.mode in {"read", "readwrite"}
                writes = access.mode in {"write", "readwrite"}

                if reads and last_write is not None:
                    self._add_memory_dependence(
                        last_write.op_node,
                        access.op_node,
                        "RAW",
                        "may",
                    )

                if writes:
                    if last_write is not None:
                        self._add_memory_dependence(
                            last_write.op_node,
                            access.op_node,
                            "WAW",
                            "may",
                        )
                    for prior_read in reads_since_write:
                        self._add_memory_dependence(
                            prior_read.op_node,
                            access.op_node,
                            "WAR",
                            "may",
                        )
                    reads_since_write = []
                    last_write = access

                if reads and not writes:
                    reads_since_write.append(access)

    def _add_exact_memory_dependencies(self) -> None:
        for dependency in self.exact_dependencies:
            function = str(dependency["function"])
            source_key = (function, int(dependency["source_op"]))
            target_key = (function, int(dependency["target_op"]))
            source = self.operation_by_uid.get(source_key)
            target = self.operation_by_uid.get(target_key)
            if source is None or target is None:
                raise RuntimeError(
                    "Exact dependence references an unknown deterministic op ordinal: "
                    f"{dependency}"
                )
            self._add_memory_dependence(
                source.node,
                target.node,
                str(dependency["kind"]).upper(),
                "must",
                dependency.get("distance", []),
            )

    @staticmethod
    def _loop_matches_spec(loop: LoopInfo, spec: ActionSpec) -> bool:
        if spec.kind != "loop" or spec.function != loop.function_name:
            return False
        if spec.loop_ordinal is not None and spec.loop_ordinal != loop.loop_ordinal:
            return False
        if spec.location is not None and spec.location not in loop.op_record.location:
            return False
        if spec.op_name is not None and spec.op_name != loop.op_record.op_name:
            return False
        return True

    @staticmethod
    def _loop_embedded_action_id(loop: LoopInfo) -> str | None:
        action_id = action_id_from_attributes(loop.op_record.attributes)
        if action_id is not None:
            return action_id
        raw_id = get_attribute(
            loop.op_record.operation,
            "mailohls.action_id",
            "hls.action_id",
            "action_id",
            "mailohls.label",
            "hls.label",
        )
        if raw_id is None:
            return None
        try:
            return _normalise_action_id(raw_id)
        except ValueError:
            return None

    def _add_pragma_node(
        self,
        *,
        action_id: str,
        kind: str,
        full_text: str,
        function_id: int,
        block_id: int,
        semantic_anchor: int,
    ) -> int:
        upper = kind.upper()
        node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": upper,
                "type": NODE_TYPE_PRAGMA,
                "full_text": full_text,
                "action_id": action_id,
                # One body block gives gexf_to_pt_zero exactly one type-4 anchor.
                "dependency_blocks": [block_id],
            }
        )
        attrs = {"flow": FLOW_PRAGMA, "position": PRAGMA_POSITION[upper]}
        self.graph.add_edge(semantic_anchor, node, **attrs)
        self.graph.add_edge(node, semantic_anchor, **attrs)
        return node

    def _attach_loop_actions(self) -> None:
        # Resolve the whole manifest before mutating the graph.  This catches an
        # op_name-only locator that would otherwise silently select the first of
        # several scf.for operations.
        manifest_by_loop: dict[int, ActionSpec] = {}
        for spec in [item for item in self.actions if item.kind == "loop"]:
            has_locator = any(
                locator is not None
                for locator in (spec.loop_ordinal, spec.location, spec.op_name)
            )
            if has_locator:
                matches = [
                    loop_index
                    for loop_index, loop in enumerate(self.loops)
                    if self._loop_matches_spec(loop, spec)
                ]
            else:
                matches = [
                    loop_index
                    for loop_index, loop in enumerate(self.loops)
                    if loop.function_name == spec.function
                    and self._loop_embedded_action_id(loop) == spec.action_id
                ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Loop action {spec.action_id} matched {len(matches)} MLIR loops; "
                    "use function + loop_ordinal or a preserved source location."
                )
            loop_index = matches[0]
            if loop_index in manifest_by_loop:
                other = manifest_by_loop[loop_index]
                raise RuntimeError(
                    f"MLIR loop {self.loops[loop_index].function_name}#"
                    f"{self.loops[loop_index].loop_ordinal} maps to both "
                    f"{other.action_id} and {spec.action_id}."
                )
            manifest_by_loop[loop_index] = spec

        for loop_index, loop in enumerate(self.loops):
            attr_id = self._loop_embedded_action_id(loop)
            spec = manifest_by_loop.get(loop_index)

            if attr_id is not None:
                if spec is not None and spec.action_id != attr_id:
                    raise RuntimeError(
                        f"Loop {loop.function_name}#{loop.loop_ordinal} carries "
                        f"{attr_id} in MLIR but the manifest assigns {spec.action_id}."
                    )
                action_id = attr_id
                if spec is not None:
                    directives = spec.directives
                else:
                    directive_attr = (
                        loop.op_record.attributes.get("mailohls.directives")
                        or get_attribute(
                            loop.op_record.operation,
                            "mailohls.directives",
                        )
                        or ""
                    ).lower()
                    parsed = tuple(
                        directive
                        for directive in ("pipeline", "unroll")
                        if directive in directive_attr
                    )
                    if not parsed:
                        parsed = tuple(
                            directive
                            for directive, legacy_attr in (
                                ("pipeline", "loop_pipeline_ii"),
                                ("unroll", "loop_unroll_factor"),
                            )
                            if legacy_attr in loop.op_record.attributes
                        )
                    directives = parsed or ("pipeline", "unroll")
            else:
                action_id = spec.action_id if spec else None
                directives = spec.directives if spec else ()

            if action_id is None:
                continue
            if action_id in self.attached_action_ids:
                raise RuntimeError(
                    f"Action {action_id} is attached to multiple MLIR scopes."
                )
            if spec is not None:
                spec.matched = True
            loop.action_id = action_id
            self.attached_action_ids.add(action_id)
            self.graph.nodes[loop.op_node]["action_id"] = action_id

            for directive in directives:
                if directive == "pipeline":
                    full_text = (
                        f"#pragma HLS PIPELINE II=auto{{_PIPE_{action_id}}}"
                    )
                elif directive == "unroll":
                    full_text = (
                        f"#pragma HLS UNROLL factor=auto{{_UNROLL_{action_id}}}"
                    )
                else:
                    raise AssertionError(directive)
                self._add_pragma_node(
                    action_id=action_id,
                    kind=directive,
                    full_text=full_text,
                    function_id=loop.function_id,
                    block_id=loop.scope_block,
                    semantic_anchor=loop.op_node,
                )

    def _array_argument_node(self, spec: ActionSpec) -> tuple[int, int, int]:
        function_id = self.function_name_to_id.get(spec.function)
        if function_id is None:
            raise RuntimeError(
                f"Array action {spec.action_id}: unknown function {spec.function!r}"
            )
        args = self.function_arguments.get(function_id, [])
        if spec.argument_index is not None:
            if not 0 <= spec.argument_index < len(args):
                raise RuntimeError(
                    f"Array action {spec.action_id}: argument_index={spec.argument_index} "
                    f"is outside function {spec.function}'s {len(args)} arguments."
                )
            node = args[spec.argument_index]
            if int(self.graph.nodes[node].get("is_memory", 0)) != 1:
                raise RuntimeError(
                    f"Array action {spec.action_id}: function argument "
                    f"{spec.argument_index} is not a memref/pointer value."
                )
            return function_id, node, int(self.graph.nodes[node]["block"])

        expected = str(spec.value_name)
        matches = [
            node
            for node in args
            if str(self.graph.nodes[node].get("value_name", "")) == expected
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Array action {spec.action_id}: value_name={expected!r} matched "
                f"{len(matches)} function arguments; use argument_index."
            )
        node = matches[0]
        if int(self.graph.nodes[node].get("is_memory", 0)) != 1:
            raise RuntimeError(
                f"Array action {spec.action_id}: value {expected!r} is not "
                "a memref/pointer argument."
            )
        return function_id, node, int(self.graph.nodes[node]["block"])

    def _attach_array_actions(self) -> None:
        for spec in [item for item in self.actions if item.kind == "array"]:
            if spec.action_id in self.attached_action_ids:
                raise RuntimeError(
                    f"Action {spec.action_id} is attached to multiple MLIR scopes."
                )
            function_id, argument_node, block_id = self._array_argument_node(spec)
            root = self.memory_root_by_value.get(
                next(
                    (
                        key for key, node in self.value_nodes.items()
                        if node == argument_node
                    ),
                    ("missing", -1),
                ),
                argument_node,
            )
            variable = spec.variable or spec.value_name or f"arg{spec.argument_index}"
            full_text = (
                "#pragma HLS ARRAY_PARTITION "
                f"variable={variable} "
                f"type=auto{{_ARRAY_T_{spec.action_id}}} "
                f"factor=auto{{_ARRAY_F_{spec.action_id}}} "
                f"dim=auto{{_ARRAY_D_{spec.action_id}}}"
            )

            pragma = self._new_node(
                {
                    "block": block_id,
                    "function": function_id,
                    "text": "ARRAY_PARTITION",
                    "type": NODE_TYPE_PRAGMA,
                    "full_text": full_text,
                    "action_id": spec.action_id,
                    "dependency_blocks": [block_id],
                }
            )
            scope = self._new_node(
                {
                    "block": block_id,
                    "function": function_id,
                    "text": ARRAY_SCOPE_TEXT,
                    "type": NODE_TYPE_ARRAY_SCOPE,
                    "full_text": f"array_scope<{variable}> action={spec.action_id}",
                    "action_id": spec.action_id,
                    "array_var": variable,
                    "memory_root_text": det_get_full_text(self.graph.nodes[root]),
                }
            )
            attrs = {
                "flow": FLOW_PRAGMA,
                "position": PRAGMA_POSITION["ARRAY_PARTITION"],
            }
            self.graph.add_edge(pragma, scope, **attrs)
            self.graph.add_edge(scope, pragma, **attrs)

            # Root plus every access is complete and deterministic; there is no
            # arbitrary "first eight textual matches" truncation.
            self.graph.add_edge(
                scope,
                root,
                flow=FLOW_ARRAY_SCOPE,
                position=0,
                role="array_root",
            )
            self.graph.add_edge(
                root,
                scope,
                flow=FLOW_ARRAY_SCOPE,
                position=10,
                role="array_root_reverse",
            )
            for access in sorted(
                (
                    item
                    for item in self.memory_accesses
                    if item.function_id == function_id and item.root_node == root
                ),
                key=lambda item: item.op_ordinal,
            ):
                mode_position = {"read": 1, "write": 2, "readwrite": 3}[access.mode]
                self.graph.add_edge(
                    scope,
                    access.op_node,
                    flow=FLOW_ARRAY_SCOPE,
                    position=mode_position,
                    role=f"array_{access.mode}",
                )
                self.graph.add_edge(
                    access.op_node,
                    scope,
                    flow=FLOW_ARRAY_SCOPE,
                    position=10 + mode_position,
                    role=f"array_{access.mode}_reverse",
                )
            spec.matched = True
            self.attached_action_ids.add(spec.action_id)

    def _attach_actions(self) -> None:
        self._attach_loop_actions()
        self._attach_array_actions()
        unmatched = [spec.action_id for spec in self.actions if not spec.matched]
        if unmatched and self.require_actions:
            raise RuntimeError(f"Unmatched MailoHLS actions: {unmatched}")
        if self.require_actions and not self.attached_action_ids:
            raise RuntimeError(
                "--require-actions was set but no Lk action was attached from "
                "the manifest or from MLIR operation attributes."
            )

    def _refresh_record_node_ids(self) -> None:
        op_uid_to_node = {
            str(data.get("op_uid")): int(node)
            for node, data in self.graph.nodes(data=True)
            if data.get("op_uid")
        }
        for record in self.operation_records:
            uid = f"{record.function_name}:op{record.function_ordinal}"
            if uid in op_uid_to_node:
                record.node = op_uid_to_node[uid]
        for loop in self.loops:
            loop.op_node = loop.op_record.node
        for function_id, function_name in self.functions.items():
            uid = f"{function_name}:function"
            if uid in op_uid_to_node:
                self.function_nodes[function_id] = op_uid_to_node[uid]
            self.function_arguments[function_id] = [
                int(node)
                for node, data in sorted(
                    self.graph.nodes(data=True),
                    key=lambda item: (
                        int(item[1].get("function_argument_index", 10**9)),
                        int(item[0]),
                    ),
                )
                if int(data.get("function", -1)) == function_id
                and int(data.get("function_argument_index", -1)) >= 0
            ]


# ---------------------------------------------------------------------------
# MailoHLS graph augmentation.  Unlike the first MLIR prototype, connected
# pseudo-blocks follow real MLIR block/region adjacency; they are not an O(B^2)
# all-pairs clique.
# ---------------------------------------------------------------------------

def parse_block_edges(graph: nx.MultiDiGraph) -> list[tuple[int, int, int, int]]:
    value = graph.graph.get("block_edges", [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    output = []
    for item in value:
        if len(item) != 4:
            continue
        output.append(tuple(int(part) for part in item))
    return sorted(set(output))


def add_auxiliary_nodes(
    source: nx.MultiDiGraph,
    connected: bool,
) -> nx.MultiDiGraph:
    graph = deepcopy(source)
    original_nodes = sorted(
        list(graph.nodes(data=True)),
        key=lambda item: det_node_sort_key(item[0], item[1]),
    )
    next_node = max((int(node) for node in graph.nodes()), default=-1) + 1
    pseudo_by_block: dict[tuple[int, int], int] = {}
    position_by_block: dict[tuple[int, int], int] = defaultdict(int)

    for node, data in original_nodes:
        key = (int(data.get("function", -1)), int(data.get("block", -1)))
        if key not in pseudo_by_block:
            pseudo = next_node
            next_node += 1
            pseudo_by_block[key] = pseudo
            graph.add_node(
                pseudo,
                block=key[1],
                function=key[0],
                text="pseudo_block",
                type=NODE_TYPE_PSEUDO_BLOCK,
                full_text="MLIR block scope used by MailoHLS pragma masks",
                is_mlir_block_scope=1,
            )
        pseudo = pseudo_by_block[key]
        position = position_by_block[key]
        attrs = {"flow": FLOW_PSEUDO_BLOCK, "position": position}
        graph.add_edge(node, pseudo, **attrs)
        graph.add_edge(pseudo, node, **attrs)
        position_by_block[key] += 1

    if connected:
        for function_id, source_block, target_block, position in parse_block_edges(source):
            left = pseudo_by_block.get((function_id, source_block))
            right = pseudo_by_block.get((function_id, target_block))
            if left is None or right is None or left == right:
                continue
            graph.add_edge(
                left,
                right,
                flow=FLOW_PSEUDO_CONNECTED,
                position=position,
                role="mlir_block_adjacency",
            )

    graph.graph["pseudo_scope_count"] = len(pseudo_by_block)
    return finalize_graph(graph)


def index_pseudo_blocks(graph: nx.MultiDiGraph) -> dict[tuple[int, int], int]:
    output: dict[tuple[int, int], int] = {}
    for node, data in graph.nodes(data=True):
        if int(data.get("type", -1)) != NODE_TYPE_PSEUDO_BLOCK:
            continue
        key = (int(data.get("function", -1)), int(data.get("block", -1)))
        if key in output:
            raise RuntimeError(f"Multiple pseudo nodes for MLIR block {key}")
        output[key] = int(node)
    return output


def add_loop_hierarchy(
    source: nx.MultiDiGraph,
    loops: list[LoopInfo],
    transitive: bool = False,
) -> nx.MultiDiGraph:
    graph = deepcopy(source)
    pseudo_by_block = index_pseudo_blocks(graph)
    pairs: dict[tuple[int, int], int] = {}

    for parent_index, parent in enumerate(loops):
        parent_scope = pseudo_by_block.get((parent.function_id, parent.scope_block))
        if parent_scope is None:
            continue
        for child_index in parent.children:
            child = loops[child_index]
            child_scope = pseudo_by_block.get((child.function_id, child.scope_block))
            if child_scope is not None and child_scope != parent_scope:
                pairs[(parent_scope, child_scope)] = 0

    if transitive:
        for child_index, child in enumerate(loops):
            child_scope = pseudo_by_block.get((child.function_id, child.scope_block))
            ancestor = child.parent_index
            while child_scope is not None and ancestor is not None:
                parent = loops[ancestor]
                parent_scope = pseudo_by_block.get((parent.function_id, parent.scope_block))
                if parent_scope is not None and parent_scope != child_scope:
                    pairs.setdefault((parent_scope, child_scope), 1)
                ancestor = parent.parent_index

    for (parent_scope, child_scope), position in sorted(pairs.items()):
        graph.add_edge(
            parent_scope,
            child_scope,
            flow=FLOW_LOOP_HIERARCHY,
            position=position,
            role="loop_parent",
        )
        graph.add_edge(
            child_scope,
            parent_scope,
            flow=FLOW_LOOP_HIERARCHY,
            position=10 + position,
            role="loop_child",
        )
    return finalize_graph(graph)


# ---------------------------------------------------------------------------
# Validation and reporting.
# ---------------------------------------------------------------------------

def _pragma_action_id(data: dict[str, Any]) -> str | None:
    value = data.get("action_id")
    if value:
        try:
            return _normalise_action_id(value)
        except ValueError:
            return None
    match = ACTION_ID_SEARCH_RE.search(det_get_full_text(data))
    return f"L{match.group(1)}" if match else None


def validate_graph(
    graph: nx.MultiDiGraph,
    *,
    require_actions: bool,
    require_single_loop_anchor: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    node_ids = sorted(int(node) for node in graph.nodes())
    if node_ids != list(range(len(node_ids))):
        errors.append("Node ids are not the contiguous range 0..N-1 required by data.py.")

    required_node_attrs = {"block", "function", "text", "type", "full_text"}
    for node, data in graph.nodes(data=True):
        missing = required_node_attrs - set(data)
        if missing:
            errors.append(f"Node {node} misses attributes {sorted(missing)}")
        if graph.degree(node) == 0:
            errors.append(f"Node {node} is isolated.")

    for source, target, data in graph.edges(data=True):
        if "flow" not in data or "position" not in data:
            errors.append(f"Edge {source}->{target} misses flow/position.")
            continue
        if int(data["flow"]) not in ALL_FLOWS:
            errors.append(f"Edge {source}->{target} has unknown flow={data['flow']}.")

    action_to_kinds: dict[str, set[str]] = defaultdict(set)
    loop_action_to_pseudos: dict[str, set[int]] = defaultdict(set)
    for node, data in graph.nodes(data=True):
        node_type = int(data.get("type", -1))
        if node_type == NODE_TYPE_PRAGMA:
            action_id = _pragma_action_id(data)
            if action_id is None:
                if require_actions:
                    errors.append(f"Pragma node {node} has no valid Lk action id.")
                continue
            action_to_kinds[action_id].add(str(data.get("text", "")).upper())
            if "auto{" not in det_get_full_text(data) and require_actions:
                errors.append(
                    f"Pragma node {node}/{action_id} contains a concrete value; "
                    "MailoHLS structural graphs require placeholders."
                )
            if require_single_loop_anchor and str(data.get("text", "")).upper() in {
                "PIPELINE",
                "UNROLL",
            }:
                for neighbour in set(graph.predecessors(node)) | set(graph.successors(node)):
                    if int(graph.nodes[neighbour].get("type", -1)) != NODE_TYPE_PSEUDO_BLOCK:
                        continue
                    edge_data = graph.get_edge_data(node, neighbour, default={})
                    reverse_data = graph.get_edge_data(neighbour, node, default={})
                    flows = {
                        int(attrs.get("flow", -1))
                        for mapping in (edge_data, reverse_data)
                        for attrs in mapping.values()
                    }
                    if FLOW_PSEUDO_BLOCK in flows:
                        loop_action_to_pseudos[action_id].add(int(neighbour))

        if node_type == NODE_TYPE_ARRAY_SCOPE:
            if not any(
                int(attrs.get("flow", -1)) == FLOW_PRAGMA
                for _, _, attrs in graph.edges(node, data=True)
            ):
                errors.append(f"Array scope node {node} is not attached to a pragma.")

    if require_single_loop_anchor:
        loop_actions = {
            action
            for action, kinds in action_to_kinds.items()
            if kinds & {"PIPELINE", "UNROLL"}
        }
        for action in sorted(loop_actions):
            count = len(loop_action_to_pseudos.get(action, set()))
            if count != 1:
                errors.append(
                    f"Loop action {action} maps to {count} pseudo scopes; "
                    "gexf_to_pt_zero.py requires exactly one."
                )

    if not action_to_kinds:
        warnings.append("Graph contains no MailoHLS action nodes.")

    report = {
        "schema_version": graph.graph.get("schema_version", SCHEMA_VERSION),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "functions": len(
            {
                int(data.get("function", -1))
                for _, data in graph.nodes(data=True)
                if int(data.get("function", -1)) >= 0
            }
        ),
        "actions": {
            action: sorted(kinds) for action, kinds in sorted(action_to_kinds.items())
        },
        "node_types": {
            str(node_type): sum(
                1
                for _, data in graph.nodes(data=True)
                if int(data.get("type", -1)) == node_type
            )
            for node_type in sorted(
                {int(data.get("type", -1)) for _, data in graph.nodes(data=True)}
            )
        },
        "edge_flows": {
            str(flow): sum(
                1
                for _, _, data in graph.edges(data=True)
                if int(data.get("flow", -1)) == flow
            )
            for flow in sorted(
                {int(data.get("flow", -1)) for _, _, data in graph.edges(data=True)}
            )
        },
        "warnings": warnings,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(
            "MLIR graph validation failed:\n  - " + "\n  - ".join(errors)
        )
    return report


def graph_counts(name: str, graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {"name": name, "num_node": graph.number_of_nodes(), "num_edge": graph.number_of_edges()}


def augmented_counts(
    name: str,
    previous: nx.MultiDiGraph,
    current: nx.MultiDiGraph,
) -> dict[str, Any]:
    row = {
        "name": name,
        "prev_node": previous.number_of_nodes(),
        "prev_edge": previous.number_of_edges(),
        "new_node": current.number_of_nodes(),
        "new_edge": current.number_of_edges(),
    }
    blocks = {
        (int(data.get("function", -1)), int(data.get("block", -1)))
        for _, data in current.nodes(data=True)
        if int(data.get("type", -1)) == NODE_TYPE_PSEUDO_BLOCK
    }
    if blocks:
        row["block"] = len(blocks)
    return row


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_initial_graph(
    mlir_path: Path,
    *,
    action_manifest: Path | None,
    dependence_manifest: Path | None,
    allow_unregistered_dialects: bool,
    conservative_memory_dependencies: bool,
    require_actions: bool,
) -> tuple[Any, ParseResult]:
    actions = load_action_manifest(action_manifest)
    dependencies = load_dependence_manifest(dependence_manifest)
    context, module, mlir_text = parse_mlir_module(
        mlir_path,
        allow_unregistered_dialects=allow_unregistered_dialects,
    )
    builder = MlirGraphBuilder(
        module=module,
        mlir_text=mlir_text,
        actions=actions,
        exact_dependencies=dependencies,
        conservative_memory_dependencies=conservative_memory_dependencies,
        require_actions=require_actions,
    )
    result = builder.build()
    return context, result


# ---------------------------------------------------------------------------
# CLI, deliberately close to the first MLIR prototype.
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    require_pythonhashseed()
    mlir_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    kernel = args.kernel or mlir_path.stem.split(".")[0]

    context, result = create_initial_graph(
        mlir_path,
        action_manifest=Path(args.action_manifest).resolve() if args.action_manifest else None,
        dependence_manifest=(
            Path(args.dependence_manifest).resolve()
            if args.dependence_manifest
            else None
        ),
        allow_unregistered_dialects=args.allow_unregistered_dialects,
        conservative_memory_dependencies=args.conservative_memory_dependencies,
        require_actions=args.require_actions,
    )
    try:
        initial = result.graph
        reports: dict[str, Any] = {}

        if args.mode in {"initial", "all"}:
            path = out_dir / "processed" / "original" / f"{kernel}_processed_result.gexf"
            reports["initial"] = validate_graph(
                initial,
                require_actions=args.require_actions,
                require_single_loop_anchor=False,
            )
            write_gexf_deterministic(initial, path)
            write_csv(
                out_dir / "initial.csv",
                ["name", "num_node", "num_edge"],
                [graph_counts(kernel, initial)],
            )

        if args.mode == "all":
            auxiliary = add_auxiliary_nodes(initial, connected=False)
            reports["auxiliary"] = validate_graph(
                auxiliary,
                require_actions=args.require_actions,
                require_single_loop_anchor=True,
            )
            write_gexf_deterministic(
                auxiliary,
                out_dir
                / "processed"
                / "extended-pseudo-block-base"
                / f"{kernel}_processed_result.gexf",
            )
            write_csv(
                out_dir / "auxiliary_False.csv",
                ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
                [augmented_counts(kernel, initial, auxiliary)],
            )

            connected = add_auxiliary_nodes(initial, connected=True)
            reports["connected"] = validate_graph(
                connected,
                require_actions=args.require_actions,
                require_single_loop_anchor=True,
            )
            write_gexf_deterministic(
                connected,
                out_dir
                / "processed"
                / "extended-pseudo-block-connected"
                / f"{kernel}_processed_result.gexf",
            )
            write_csv(
                out_dir / "auxiliary_True.csv",
                ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
                [augmented_counts(kernel, initial, connected)],
            )

            hierarchy = add_loop_hierarchy(
                connected,
                result.loops,
                transitive=args.hierarchy_transitive,
            )
            reports["hierarchy"] = validate_graph(
                hierarchy,
                require_actions=args.require_actions,
                require_single_loop_anchor=True,
            )
            write_gexf_deterministic(
                hierarchy,
                out_dir
                / "processed"
                / "extended-pseudo-block-connected-hierarchy"
                / f"{kernel}_processed_result.gexf",
            )
            write_csv(
                out_dir / "hierarchy.csv",
                ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
                [augmented_counts(kernel, connected, hierarchy)],
            )

        elif args.mode == "auxiliary":
            auxiliary = add_auxiliary_nodes(initial, connected=args.connected)
            reports["auxiliary"] = validate_graph(
                auxiliary,
                require_actions=args.require_actions,
                require_single_loop_anchor=True,
            )
            folder = (
                "extended-pseudo-block-connected"
                if args.connected
                else "extended-pseudo-block-base"
            )
            write_gexf_deterministic(
                auxiliary,
                out_dir / "processed" / folder / f"{kernel}_processed_result.gexf",
            )

        elif args.mode == "hierarchy":
            connected = add_auxiliary_nodes(initial, connected=True)
            hierarchy = add_loop_hierarchy(
                connected,
                result.loops,
                transitive=args.hierarchy_transitive,
            )
            reports["hierarchy"] = validate_graph(
                hierarchy,
                require_actions=args.require_actions,
                require_single_loop_anchor=True,
            )
            write_gexf_deterministic(
                hierarchy,
                out_dir
                / "processed"
                / "extended-pseudo-block-connected-hierarchy"
                / f"{kernel}_processed_result.gexf",
            )

        write_json(
            out_dir / f"{kernel}.mlir_graph_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kernel": kernel,
                "input": str(mlir_path),
                "input_sha256": initial.graph.get("input_sha256"),
                "toolchain_id": args.toolchain_id,
                "action_manifest": args.action_manifest,
                "dependence_manifest": args.dependence_manifest,
                "conservative_memory_dependencies": args.conservative_memory_dependencies,
                "hierarchy_transitive": args.hierarchy_transitive,
                "matched_actions": sorted(
                    spec.action_id for spec in result.actions if spec.matched
                ),
                "unmatched_actions": sorted(
                    spec.action_id for spec in result.actions if not spec.matched
                ),
                "reports": reports,
            },
        )
    finally:
        # Explicitly leave the context only after all wrappers have been used.
        context.__exit__(None, None, None)

    print(f"Generated deterministic MLIR graph artifacts for {kernel} under {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SCF/Affine/MemRef MLIR input")
    parser.add_argument("--kernel", default=None, help="Kernel/output name")
    parser.add_argument(
        "--out-dir",
        default="mlir_harp",
        help=(
            "Backend-specific output root.  Use a separate root from LLVM/ProGraML "
            "to prevent accidental encoder/checkpoint mixing."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "auxiliary", "hierarchy", "all"],
        default="all",
    )
    parser.add_argument(
        "--connected",
        action="store_true",
        help="For --mode auxiliary, add real MLIR block-adjacency edges.",
    )
    parser.add_argument(
        "--action-manifest",
        default=None,
        help="JSON file mapping stable MailoHLS Lk actions to MLIR scopes.",
    )
    parser.add_argument(
        "--dependence-manifest",
        default=None,
        help=(
            "Optional exact RAW/WAR/WAW edges emitted by a C++ MLIR affine/alias "
            "analysis pass, keyed by deterministic per-function op ordinals."
        ),
    )
    parser.add_argument(
        "--require-actions",
        action="store_true",
        help="Fail unless all manifest actions resolve and keep valid Lk placeholders.",
    )
    parser.add_argument(
        "--allow-unregistered-dialects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Polygeist or project-specific operations not registered in Python.",
    )
    parser.add_argument(
        "--conservative-memory-dependencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit conservative same-root may-RAW/WAR/WAW edges.",
    )
    parser.add_argument(
        "--hierarchy-transitive",
        action="store_true",
        help=(
            "Also connect all loop ancestors.  Off by default because direct "
            "parent-child hierarchy is the scientifically clean baseline."
        ),
    )
    parser.add_argument(
        "--toolchain-id",
        default="unknown",
        help="Pinned LLVM/MLIR/Polygeist commit or container digest for the manifest.",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
