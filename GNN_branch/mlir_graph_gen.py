#!/usr/bin/env python3
"""Build one MailoHLS training graph directly from a C or C++ kernel.

The script deliberately exposes a small end-to-end interface:

    C/C++ source
        -> Polygeist/cgeist at -O0
        -> Affine + SCF + MemRef + Arith + Func MLIR
        -> one deterministic, action-aligned GEXF graph

Why this MLIR level?  Affine operations retain static loop bounds and affine
array subscripts when Polygeist can prove them; SCF remains the lossless
fallback for dynamic or non-affine control; MemRef retains array shape, views,
and accesses; Arith retains typed computation; Func retains calls.  We stop
before CF/LLVM lowering because that would erase the loop, region, and array
semantics that are most useful to HLS optimization.

The final graph is the single representation expected for MLIR GNN training:
semantic MLIR nodes and edges, real block adjacency, one pseudo scope per MLIR
block, direct loop hierarchy, memory roots/accesses/dependencies, and stable
MailoHLS Lk pragma/array scopes.  It preserves the node/edge contract consumed
by the existing edge-aware TransformerConv backbone, but MLIR encoders must be
regenerated and the GNN retrained.

Actions are read from ``kernel_info.txt`` beside the labeled source.  The source
labels identify each action's exact source location.  Polygeist carries that
location into MLIR, so even same-shaped local arrays remain distinguishable;
loop order and array shape are retained only as strict compatibility fallbacks.

Example:

    PYTHONHASHSEED=0 python mlir_graph_gen.py gemv.cpp --output gemv_mlir.gexf

Use the official MLIR Python bindings built from the same LLVM revision as
cgeist.  Add their ``mlir_core`` directory to PYTHONPATH; do not install the
unrelated PyPI package named ``mlir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import gc
from collections import Counter, defaultdict
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
SCHEMA_VERSION = "mailohls-mlir-graph-v3"
ACTION_ID_RE = re.compile(r"^L([1-9][0-9]*)$")
ACTION_ID_SEARCH_RE = re.compile(r"\bL([1-9][0-9]*)\b")

# Minimal labeled-C/C++ parser used by kernel_info.txt integration.  These
# patterns intentionally recognize only the dataset contract: function bodies,
# Lk labels, for-loops, and labeled local array declarations.
# MailoHLS datasets contain both real C labels (``L1: for (...)``) and labels
# kept inside comments (``/*L1:*/ for (...)``).  The latter are common in
# MachSuite and vendor-style C++ kernels, where a real C label would interfere
# with transformations.  Treat both spellings as the same dataset contract.
SOURCE_LABEL_RE = re.compile(
    r"^\s*(?:/\*\s*)?(?P<label>L\d+)\s*:\s*(?:\*/\s*)?"
    r"(?:[A-Za-z_]\w*\s*:\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)
SOURCE_FUNCTION_RE = re.compile(
    r'\b(?:extern\s+"C"\s*)?(?:[A-Za-z_]\w*[\w:\<\>\s\*&]*\s+)+'
    r'(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{',
    re.MULTILINE,
)
SOURCE_ARRAY_RE = re.compile(
    r"^\s*[A-Za-z_][\w:\<\>\s\*&]*?\s+(?P<name>[A-Za-z_]\w*)"
    r"\s*(?:\[[^\]]+\]\s*)+\s*(?:=\s*[^;]+)?;"
)
CONTROL_WORDS = {"if", "for", "while", "switch", "else", "do"}

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
    "memref.view",
    "memref.subview",
    "memref.reshape",
    "memref.transpose",
    "memref.reinterpret_cast",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.memory_space_cast",
    "bufferization.to_memref",
    "unrealized_conversion_cast",
}

CALL_OPS = {"func.call", "llvm.call"}

# Region terminators whose operands define the parent operation's results.
# Loop iter_args are handled separately because they also define a backedge to
# the loop-body block arguments.
REGION_YIELD_OPS = {
    "scf.yield",
    "affine.yield",
    "memref.alloca_scope.return",
}

READ_OPS = {
    "memref.load",
    "affine.load",
    "affine.vector_load",
    "vector.load",
    "vector.maskedload",
    "vector.transfer_read",
    "memref.prefetch",
}

WRITE_OPS = {
    "memref.store",
    "affine.store",
    "affine.vector_store",
    "vector.store",
    "vector.maskedstore",
    "vector.transfer_write",
}

READ_WRITE_OPS = {
    "memref.atomic_rmw",
    "memref.atomic_cas",
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
# keep the structure close to the one of the first prototype.
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    action_id: str
    kind: str
    function: str
    directives: tuple[str, ...]
    loop_ordinal: int | None = None
    variable: str | None = None
    array_dimensions: tuple[int, ...] = ()
    # Source positions come from the labeled C/C++ contract.  They are matched
    # against Polygeist's native MLIR locations, not against printed MLIR text.
    source_file: str = ""
    source_line: int = 0
    source_column: int = 0
    # Exact source lines where an array variable is read or written. These are
    # used only to disambiguate multiple same-shaped physical allocations.
    source_use_lines: tuple[int, ...] = ()
    matched: bool = False


@dataclass(frozen=True, order=True)
class SourcePoint:
    """One concrete file/line/column carried by an MLIR Location."""

    filename: str
    line: int
    column: int


@dataclass
class BlockRecord:
    key: Any
    function_id: int
    block_id: int
    parent_op_node: int
    parent_op_block: int
    region_index: int
    block: Any | None = None
    arguments: list[Any] = field(default_factory=list)
    argument_keys: list[Any] = field(default_factory=list)
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
    source_points: tuple[SourcePoint, ...]
    operands: list[Any]
    results: list[Any]
    operand_keys: list[Any] = field(default_factory=list)
    result_keys: list[Any] = field(default_factory=list)


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
    """Require a fixed hash seed so graph IDs and output bytes are reproducible."""
    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "Determinism requires PYTHONHASHSEED to be set before Python starts.\n"
            "Run, for example:\n"
            "  PYTHONHASHSEED=0 python mlir_graph_gen.py ..."
        )


def det_sha_label(obj: Any) -> str:
    """Compute a deterministic sha label for the deterministic MLIR-to-MailoHLS graph pipeline."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def det_get_full_text(data: dict[str, Any]) -> str:
    """Compute a deterministic get full text for the deterministic MLIR-to-MailoHLS graph pipeline."""
    if data.get("full_text") is not None:
        return str(data["full_text"])
    features = data.get("features")
    if isinstance(features, dict):
        full_text = features.get("full_text")
        if isinstance(full_text, list) and full_text:
            return str(full_text[0])
    return ""


def det_node_sort_key(node: Any, data: dict[str, Any]) -> tuple[Any, ...]:
    """Compute a deterministic node sort key for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Compute a deterministic edge sort key for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Reinsert nodes, edges, and attributes in a canonical order for reproducible output."""
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
    """Replace temporary IDs with IDs derived from stable structural graph signatures."""
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
    """Convert attr for the deterministic MLIR-to-MailoHLS graph pipeline."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prepare_graph_for_write(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Prepare graph for write for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Write gexf deterministic for the deterministic MLIR-to-MailoHLS graph pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(prepare_graph_for_write(graph), path, prettyprint=False)


def prune_redundant_nodes(graph: nx.MultiDiGraph) -> None:
    """Remove redundant nodes for the deterministic MLIR-to-MailoHLS graph pipeline."""
    while True:
        isolated = [
            node for node in sorted(graph.nodes(), key=str)
            if node is None or graph.degree(node) == 0
        ]
        if not isolated:
            return
        graph.remove_nodes_from(isolated)


def finalize_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Run cleanup and canonical relabeling before the graph leaves this module."""
    prune_redundant_nodes(graph)
    graph = canonicalize_graph(graph)
    graph = relabel_nodes_canonically(graph, rounds=3)
    return canonicalize_graph(graph)


# ---------------------------------------------------------------------------
# Action discovery from kernel_info.txt and labeled source.
# ---------------------------------------------------------------------------

def _normalise_action_id(value: Any) -> str:
    """Normalize action ID for the deterministic MLIR-to-MailoHLS graph pipeline."""
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


def _strip_source_comments(text: str) -> str:
    """Remove C/C++ comments while preserving line and brace positions."""
    text = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"//.*", "", text)


def _source_function_spans(text: str) -> list[tuple[str, int, int]]:
    """Find function body ranges with lightweight brace matching."""
    clean = _strip_source_comments(text)
    spans: list[tuple[str, int, int]] = []
    for match in SOURCE_FUNCTION_RE.finditer(clean):
        name = match.group("name")
        if name in CONTROL_WORDS:
            continue
        brace = clean.find("{", match.end() - 1)
        depth = 0
        for position in range(brace, len(clean)):
            if clean[position] == "{":
                depth += 1
            elif clean[position] == "}":
                depth -= 1
                if depth == 0:
                    start_line = clean.count("\n", 0, match.start()) + 1
                    end_line = clean.count("\n", 0, position) + 1
                    spans.append((name, start_line, end_line))
                    break
    return spans


def _source_actions(source: Path) -> dict[str, dict[str, Any]]:
    """Map each MailoHLS Lk source label to its semantic action and location.

    Recognized forms include:
        L1: for (...)
        L1: LOOP_NAME: for (...)
        /*L1:*/ for (...)
        /*L1:*/ LOOP_NAME: for (...)
        L2: float local_array[N];

    The search is performed on the original physical source line so that the
    resulting source column remains aligned with Polygeist FileLineColLoc data.
    """
    text = source.read_text(encoding="utf-8", errors="replace")
    spans = _source_function_spans(text)
    clean_lines = _strip_source_comments(text).splitlines()

    # Do not require Lk to be the first non-whitespace token. This tolerates a
    # UTF-8 BOM and other harmless source-prefix characters while still
    # requiring an independent L<number>: token.
    label_re = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?:/\*\s*)?"
        r"(?P<label>L[1-9][0-9]*)\s*:\s*"
        r"(?:\*/\s*)?",
        re.IGNORECASE,
    )

    secondary_label_re = re.compile(
        r"\s*[A-Za-z_]\w*\s*:\s*"
    )

    actions: dict[str, dict[str, Any]] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        label_match = label_re.search(line)
        if label_match is None:
            continue

        action_id = _normalise_action_id(label_match.group("label"))

        if action_id in actions:
            raise ValueError(
                f"{source}:{line_number}: duplicate source action {action_id}"
            )

        function, function_start_line, function_end_line = next(
            (
                (name, start_line, end_line)
                for name, start_line, end_line in spans
                if start_line <= line_number <= end_line
            ),
            ("GLOBAL", 1, len(clean_lines)),
        )

        # Start immediately after Lk: or /*Lk:*/.
        semantic_start = label_match.end()
        remaining = line[semantic_start:]

        # Skip a secondary C/C++ label such as LOAD_TILE:.
        secondary_match = secondary_label_re.match(remaining)
        if secondary_match is not None:
            semantic_start += secondary_match.end()
            remaining = line[semantic_start:]

        loop_match = re.search(r"\bfor\s*\(", remaining, re.IGNORECASE)
        array_match = SOURCE_ARRAY_RE.match(remaining)

        common = {
            "function": function,
            "source_file": source.name,
            "source_line": line_number,
        }

        if loop_match is not None:
            actions[action_id] = {
                **common,
                "kind": "loop",
                # MLIR source locations use one-based columns.
                "source_column": semantic_start + loop_match.start() + 1,
            }
            continue

        if array_match is not None:
            variable = array_match.group("name")
            identifier_re = re.compile(
                rf"\b{re.escape(variable)}\b"
            )
            source_use_lines = tuple(
                candidate_line
                for candidate_line in range(
                    function_start_line,
                    function_end_line + 1,
                )
                if candidate_line != line_number
                and candidate_line <= len(clean_lines)
                and identifier_re.search(
                    clean_lines[candidate_line - 1]
                )
            )
            actions[action_id] = {
                **common,
                "kind": "array",
                "array_name": variable,
                "source_use_lines": source_use_lines,
                "source_column": (
                    semantic_start + array_match.start("name") + 1
                ),
            }
            continue

        actions[action_id] = {
            **common,
            "kind": "unknown",
            "source_column": semantic_start + 1,
        }

    return actions

def load_kernel_info_actions(source: Path, kernel_info: Path) -> tuple[str, list[ActionSpec]]:
    """Build action specifications from kernel_info.txt and labeled source."""
    lines = [
        line.strip() for line in kernel_info.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty kernel_info.txt: {kernel_info}")
    top_function = lines[0]
    source_actions = _source_actions(source)

    loop_ordinals: dict[str, int] = defaultdict(int)
    ordinal_by_label: dict[str, int] = {}
    for action_id, item in sorted(
        source_actions.items(), key=lambda value: int(value[0][1:])
    ):
        if item["kind"] == "loop":
            function = item["function"]
            ordinal_by_label[action_id] = loop_ordinals[function]
            loop_ordinals[function] += 1

    actions: list[ActionSpec] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            raise ValueError(f"{kernel_info}:{line_number}: malformed action line: {line!r}")
        action_id = _normalise_action_id(fields[0])
        kind = fields[1].lower()
        if action_id in seen:
            raise ValueError(f"{kernel_info}:{line_number}: duplicate action {action_id}")
        seen.add(action_id)
        if kind not in {"loop", "array"}:
            raise ValueError(f"{kernel_info}:{line_number}: unsupported kind {kind!r}")
        source_action = source_actions.get(action_id)
        if source_action is None:
            raise ValueError(f"{kernel_info}:{line_number}: {action_id} is absent from {source.name}")
        if source_action["kind"] != kind:
            raise ValueError(
                f"{kernel_info}:{line_number}: {action_id} is {kind}, but the source label "
                f"identifies a {source_action['kind']}"
            )
        function = source_action["function"]

        # ---------------------------------------------------------------
        # Loop action:
        #   Lk,loop,trip_count
        # ---------------------------------------------------------------
        if kind == "loop":
            if len(fields) != 3:
                raise ValueError(
                    f"{kernel_info}:{line_number}: loop syntax must be "
                    f"Lk,loop,trip_count; got {line!r}"
                )

            actions.append(
                ActionSpec(
                    action_id=action_id,
                    kind="loop",
                    function=function,
                    directives=("pipeline", "unroll"),
                    loop_ordinal=ordinal_by_label[action_id],
                    source_file=str(source_action["source_file"]),
                    source_line=int(source_action["source_line"]),
                    source_column=int(source_action["source_column"]),
                )
            )
            continue

        # ---------------------------------------------------------------
        # Array action:
        #   Lk,array,name,dim,size[,dim,size...]
        # ---------------------------------------------------------------
        if len(fields) < 5 or (len(fields) - 3) % 2 != 0:
            raise ValueError(
                f"{kernel_info}:{line_number}: array syntax must be "
                f"Lk,array,name,dim,size[,dim,size...]; got {line!r}"
            )

        variable = fields[2]

        try:
            dimension_indices = tuple(
                int(fields[index])
                for index in range(3, len(fields), 2)
            )
            dimensions = tuple(
                int(fields[index])
                for index in range(4, len(fields), 2)
            )
        except ValueError as exc:
            raise ValueError(
                f"{kernel_info}:{line_number}: array dimensions and sizes "
                f"must be integers; got {line!r}"
            ) from exc

        if any(index <= 0 for index in dimension_indices):
            raise ValueError(
                f"{kernel_info}:{line_number}: array dimension indices must "
                f"be positive; got {dimension_indices}"
            )

        if len(set(dimension_indices)) != len(dimension_indices):
            raise ValueError(
                f"{kernel_info}:{line_number}: duplicate array dimensions "
                f"in {dimension_indices}"
            )

        if any(size <= 0 for size in dimensions):
            raise ValueError(
                f"{kernel_info}:{line_number}: array sizes must be positive; "
                f"got {dimensions}"
            )

        source_variable = source_action.get("array_name")
        if source_variable and source_variable != variable:
            raise ValueError(
                f"{kernel_info}:{line_number}: variable {variable!r} "
                f"disagrees with source declaration {source_variable!r}"
            )

        actions.append(
            ActionSpec(
                action_id=action_id,
                kind="array",
                function=function,
                directives=("array_partition",),
                variable=variable,
                array_dimensions=dimensions,
                source_file=str(source_action["source_file"]),
                source_line=int(source_action["source_line"]),
                source_column=int(source_action["source_column"]),
                source_use_lines=tuple(
                    int(value)
                    for value in source_action.get(
                        "source_use_lines",
                        (),
                    )
                ),
            )
        )
    return top_function, actions


# ---------------------------------------------------------------------------
# Thin compatibility layer over MLIR Python bindings.
# ---------------------------------------------------------------------------

def import_mlir_ir() -> Any:
    """Import mlir ir for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Return the underlying operation for the deterministic MLIR-to-MailoHLS graph pipeline."""
    return getattr(operation, "operation", operation)


def object_key(obj: Any) -> tuple[str, int]:
    """Return an identity key for key for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return (type(obj).__name__, hash(obj))
    except Exception:
        return (type(obj).__name__, id(obj))


def operation_name(operation: Any) -> str:
    """Return operation name for the deterministic MLIR-to-MailoHLS graph pipeline."""
    return str(raw_operation(operation).name)


def operation_first_line(operation: Any) -> str:
    """Return operation first line for the deterministic MLIR-to-MailoHLS graph pipeline."""
    text = str(raw_operation(operation)).strip()
    return text.splitlines()[0].strip() if text else operation_name(operation)


def operation_location(operation: Any) -> str:
    """Return operation location for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return str(raw_operation(operation).location)
    except Exception:
        return "loc(unknown)"


def operation_source_points(
    ir: Any,
    operation: Any,
) -> tuple[SourcePoint, ...]:
    """Return every exact file/line/column point nested in an MLIR Location.

    Prefer the typed MLIR Python location API when available. Older MLIR Python
    packages may not expose FileLineColLoc/NameLoc/CallSiteLoc/FusedLoc
    inspection even though the underlying MLIR contains exact source locations.
    In that case, parse only the canonical printed location syntax:

        "file.cpp":line:column

    This remains an exact source-location mapping; it is not an ordinal or
    shape-based semantic fallback.
    """
    try:
        root = raw_operation(operation).location
    except Exception:
        return ()

    points: set[SourcePoint] = set()
    visited: set[str] = set()

    # Matches every exact FileLineColLoc embedded in simple, fused, named, or
    # call-site printed location syntax.
    printed_file_loc_re = re.compile(
        r'"(?P<filename>(?:\\.|[^"\\])+)":'
        r'(?P<line>[0-9]+):'
        r'(?P<column>[0-9]+)'
    )

    def normalize_filename(value: Any) -> str:
        text = str(value).strip()

        # StringAttr may print as "knn.cpp" rather than knn.cpp.
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            try:
                text = json.loads(text)
            except Exception:
                text = text[1:-1]

        return text

    def add_point(filename: Any, line: Any, column: Any) -> None:
        try:
            filename_text = normalize_filename(filename)
            line_number = int(line)
            column_number = int(column)
        except (TypeError, ValueError):
            return

        if (
            not filename_text
            or line_number <= 0
            or column_number <= 0
        ):
            return

        points.add(
            SourcePoint(
                filename=filename_text,
                line=line_number,
                column=column_number,
            )
        )

    def add_points_from_printed_location(marker: str) -> None:
        for match in printed_file_loc_re.finditer(marker):
            raw_filename = match.group("filename")

            # Decode standard MLIR/JSON-style escapes when possible.
            try:
                filename = json.loads(f'"{raw_filename}"')
            except Exception:
                filename = (
                    raw_filename
                    .replace(r"\"", '"')
                    .replace(r"\\", "\\")
                )

            add_point(
                filename,
                match.group("line"),
                match.group("column"),
            )

    def visit(location: Any) -> None:
        marker = str(location)

        if marker in visited:
            return
        visited.add(marker)

        # Always inspect the canonical printed representation. This makes the
        # function compatible with older Python bindings while retaining exact
        # source file/line/column information.
        add_points_from_printed_location(marker)

        file_loc_class = getattr(ir, "FileLineColLoc", None)
        if file_loc_class is not None:
            try:
                file_loc = file_loc_class(location)

                line = getattr(file_loc, "start_line", None)
                if line is None:
                    line = getattr(file_loc, "line", None)

                column = getattr(file_loc, "start_col", None)
                if column is None:
                    column = getattr(file_loc, "column", None)

                add_point(
                    getattr(file_loc, "filename", ""),
                    line,
                    column,
                )
                return
            except (TypeError, ValueError, AttributeError):
                pass

        name_loc_class = getattr(ir, "NameLoc", None)
        if name_loc_class is not None:
            try:
                named = name_loc_class(location)
                visit(named.child_loc)
                return
            except (TypeError, ValueError, AttributeError):
                pass

        callsite_loc_class = getattr(ir, "CallSiteLoc", None)
        if callsite_loc_class is not None:
            try:
                callsite = callsite_loc_class(location)
                visit(callsite.callee)
                visit(callsite.caller)
                return
            except (TypeError, ValueError, AttributeError):
                pass

        fused_loc_class = getattr(ir, "FusedLoc", None)
        if fused_loc_class is not None:
            try:
                fused = fused_loc_class(location)
                for child in fused.locations:
                    visit(child)
            except (TypeError, ValueError, AttributeError):
                pass

    visit(root)
    return tuple(sorted(points))

def attribute_items(operation: Any) -> dict[str, str]:
    """Handle items for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Return attribute for the deterministic MLIR-to-MailoHLS graph pipeline."""
    attrs = raw_operation(operation).attributes
    for name in names:
        try:
            return str(attrs[name])
        except Exception:
            continue
    return None


def get_raw_attribute(operation: Any, *names: str) -> Any | None:
    """Return an MLIR Attribute object instead of its printed spelling."""
    attrs = raw_operation(operation).attributes
    for name in names:
        try:
            return attrs[name]
        except Exception:
            continue
    return None


def strip_mlir_string(value: str | None) -> str:
    """Strip mlir string for the deterministic MLIR-to-MailoHLS graph pipeline."""
    if value is None:
        return ""
    text = value.strip().strip('"').strip("'")
    if text.startswith("@"):
        text = text[1:]
    return text


def operation_regions(operation: Any) -> list[Any]:
    """Return operation regions for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(raw_operation(operation).regions)
    except Exception:
        return []


def region_blocks(region: Any) -> list[Any]:
    """Return region blocks for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(region.blocks)
    except Exception:
        try:
            return list(region)
        except Exception:
            return []


def block_operations(block: Any) -> list[Any]:
    """Return block operations for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(block.operations)
    except Exception:
        return list(block)


def block_arguments(block: Any) -> list[Any]:
    """Return block arguments for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(block.arguments)
    except Exception:
        return []


def operation_operands(operation: Any) -> list[Any]:
    """Return operation operands for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(raw_operation(operation).operands)
    except Exception:
        return []


def operation_results(operation: Any) -> list[Any]:
    """Return operation results for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(raw_operation(operation).results)
    except Exception:
        return []


def operation_successors(operation: Any) -> list[Any]:
    """Return operation successors for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return list(raw_operation(operation).successors)
    except Exception:
        return []


def value_type(value: Any) -> str:
    """Return value type for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return str(value.type)
    except Exception:
        return "unknown"


def value_text(value: Any) -> str:
    """Return value text for the deterministic MLIR-to-MailoHLS graph pipeline."""
    try:
        return str(value)
    except Exception:
        return "<value>"


def canonical_type_token(type_text: str) -> str:
    """Handle type token for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Test whether memory type for the deterministic MLIR-to-MailoHLS graph pipeline."""
    compact = type_text.replace(" ", "")
    return (
        compact.startswith("memref<")
        or compact.startswith("!llvm.ptr")
        or compact.startswith("ptr")
    )


def parse_integer_attr(text: str | None) -> int | None:
    """Parse integer attr for the deterministic MLIR-to-MailoHLS graph pipeline."""
    if text is None:
        return None
    match = re.search(r"(?<![A-Za-z0-9_])-?[0-9]+", text)
    return int(match.group(0)) if match else None


def merge_memory_modes(left: str | None, right: str) -> str:
    """Join read/write effects in the small lattice used by graph edges."""
    if left is None:
        return right
    if left == right:
        return left
    return "readwrite"


def parse_mlir_module(path: Path, allow_unregistered_dialects: bool) -> tuple[Any, Any, str]:
    """Parse MLIR and return its live Context; wrappers become invalid when this Context closes."""
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
        conservative_memory_dependencies: bool = True,
        require_actions: bool = False,
    ) -> None:
        """Handle init for the deterministic MLIR-to-MailoHLS graph pipeline."""
        self.module = module
        # The official bindings define Value as either OpResult or
        # BlockArgument.  Keeping the module here lets us use those concrete
        # owners instead of the Python wrapper class/hash combination used by
        # the first prototype (generic Value and OpResult wrappers may alias the
        # same SSA value).
        self.ir = import_mlir_ir()
        self.mlir_text = mlir_text
        self.actions = actions
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
        self.value_objects: dict[Any, Any] = {}
        self.value_function: dict[Any, int] = {}
        self.value_def_op: dict[Any, int] = {}
        self.value_is_block_argument: set[Any] = set()
        self.constant_values: dict[Any, int] = {}
        self.function_argument_keys: dict[int, list[Any]] = {}

        self.loops: list[LoopInfo] = []
        self.loop_count_by_function: dict[int, int] = defaultdict(int)
        # A value can have more than one root after a select, region merge, or
        # a helper called with different actual buffers.  Sets preserve that
        # may-alias fact instead of choosing an arbitrary root.
        self.memory_roots_by_value: dict[Any, set[int]] = {}
        self.memory_alias_sources: dict[Any, set[Any]] = defaultdict(set)
        self.function_memory_effects: dict[int, dict[int, str]] = defaultdict(dict)
        self.memory_accesses: list[MemoryAccess] = []
        self.block_edges: set[tuple[int, int, int, int]] = set()
        self.attached_action_ids: set[str] = set()
        # Source-level C++ helper names may be mangled in MLIR. Exact loop
        # source locations provide a deterministic source-function -> MLIR
        # function mapping that array actions can safely reuse.
        self.source_function_ids_from_exact_loops: dict[str, set[int]] = (
            defaultdict(set)
        )

    def _value_key(self, value: Any) -> tuple[str, Any, int]:
        """Return the structural MLIR identity of an SSA value.

        MLIR's Python traversal may expose one underlying value once as a
        generic ``Value`` and elsewhere as ``OpResult``/``BlockArgument``.
        Including ``type(value).__name__`` in a key therefore splits a legal
        use-def chain.  MLIR already gives the exact identity we need: the
        defining operation plus result number, or the owning block plus
        argument number.
        """
        try:
            result = self.ir.OpResult(value)
            return (
                "op_result",
                raw_operation(result.owner),
                int(result.result_number),
            )
        except (TypeError, ValueError):
            pass
        try:
            argument = self.ir.BlockArgument(value)
            return (
                "block_argument",
                argument.owner,
                int(argument.arg_number),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"MLIR value is neither OpResult nor BlockArgument: {value_text(value)}"
            ) from exc

    def _value_descriptor(
        self,
        value: Any,
        function_id: int,
    ) -> tuple[Any, str, str, int]:
        """Return (key, stable id, SSA kind, defining position)."""
        key = self._value_key(value)
        function_name = self.functions[function_id]
        if key[0] == "op_result":
            owner_record = self.operation_by_key.get(object_key(key[1]))
            if owner_record is None:
                raise RuntimeError(
                    "Result owner was not indexed before its SSA value: "
                    f"{value_text(value)}"
                )
            position = int(key[2])
            return key, f"{function_name}:op{owner_record.function_ordinal}:r{position}", "op_result", position

        block_record = self.blocks.get(object_key(key[1]))
        if block_record is None:
            raise RuntimeError(
                "Block owner was not indexed before its argument: "
                f"{value_text(value)}"
            )
        position = int(key[2])
        return key, f"{function_name}:b{block_record.block_id}:a{position}", "block_argument", position

    def build(self) -> ParseResult:
        """Run graph-building passes in dependency order and return the graph plus loop metadata."""
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
        self._add_memory_shape_features()
        for loop in self.loops:
            self._annotate_loop_features(loop)
        self._add_affine_access_features()
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
        """Allocate one temporary node ID and insert its normalized metadata."""
        node = self.next_node_id
        self.next_node_id += 1
        self.graph.add_node(node, **attrs)
        return node

    def _discover_functions(self) -> list[tuple[str, Any]]:
        """Index all functions first so calls can resolve definitions that appear later."""
        found: list[tuple[str, Any]] = []

        def visit(operation: Any) -> None:
            """Recursively find functions even when a dialect nests them below a module."""
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
        """Assign stable block IDs before any edge records refer to those blocks."""
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
                        block=block,
                        arguments=block_arguments(block),
                    )
                    self.next_block_id += 1
                for child in block_operations(block):
                    self._assign_blocks(child, function_id)

    def _entry_block_id(self, function_operation: Any) -> int:
        """Return the entry block that anchors function arguments and helper nodes."""
        regions = operation_regions(function_operation)
        if not regions or not region_blocks(regions[0]):
            # Declaration-only functions get one synthetic deterministic block.
            block_id = self.next_block_id
            self.next_block_id += 1
            return block_id
        return self.blocks[object_key(region_blocks(regions[0])[0])].block_id

    def _index_function(self, function_id: int, function_name: str, operation: Any) -> None:
        """Create a function node, its argument values, and indexes for its nested body."""
        self._assign_blocks(operation, function_id)
        entry_block = self._entry_block_id(operation)
        attributes = attribute_items(operation)
        function_points = operation_source_points(self.ir, operation)
        source_attrs: dict[str, Any] = {}
        if function_points:
            point = function_points[0]
            source_attrs = {
                "source_file": point.filename,
                "source_line": point.line,
                "source_column": point.column,
            }
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
                **source_attrs,
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
            keys = [self._value_key(argument) for argument in args]
            self.function_argument_keys[function_id] = keys
            self.function_arguments[function_id] = [
                self.value_nodes[key] for key in keys
            ]
            for argument_index, node in enumerate(self.function_arguments[function_id]):
                self.graph.nodes[node]["function_argument_index"] = argument_index
        else:
            self.function_argument_keys[function_id] = []
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
        """Walk nested MLIR operations in deterministic preorder and record loop nesting."""
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
                record.argument_keys = [
                    self._value_key(argument) for argument in block_arguments(block)
                ]

                operations = block_operations(block)
                for block_order, operation in enumerate(operations):
                    op_name = operation_name(operation)
                    attributes = attribute_items(operation)
                    source_points = operation_source_points(self.ir, operation)
                    source_attrs: dict[str, Any] = {}
                    if source_points:
                        point = source_points[0]
                        source_attrs = {
                            "source_file": point.filename,
                            "source_line": point.line,
                            "source_column": point.column,
                        }
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
                            **source_attrs,
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
                        source_points=source_points,
                        operands=operation_operands(operation),
                        results=operation_results(operation),
                    )
                    record.operations.append(op_record)
                    self.operation_records.append(op_record)
                    self.operation_by_key[op_record.key] = op_record
                    self.operation_by_uid[(function_name, ordinal)] = op_record
                    op_record.operand_keys = [
                        self._value_key(value) for value in op_record.operands
                    ]
                    op_record.result_keys = [
                        self._value_key(value) for value in op_record.results
                    ]

                    for result_index, (result, result_key) in enumerate(
                        zip(op_record.results, op_record.result_keys)
                    ):
                        self._get_or_create_value(
                            result,
                            function_id,
                            record.block_id,
                            is_block_argument=False,
                            argument_index=result_index,
                        )
                        self.value_def_op[result_key] = node

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
        """Intern each SSA value once so definitions and uses share one graph node."""
        key, ssa_id, ssa_kind, defining_position = self._value_descriptor(
            value, function_id
        )
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
                "ssa_id": ssa_id,
                "ssa_kind": ssa_kind,
                "ssa_position": defining_position,
                "is_block_argument": 1 if is_block_argument else 0,
                "argument_index": argument_index,
                "is_memory": 1 if is_memory_type(type_text) else 0,
            }
        )
        self.value_nodes[key] = node
        self.value_objects[key] = value
        self.value_function[key] = function_id
        if is_block_argument:
            self.value_is_block_argument.add(key)
        return node

    def _add_ssa_edges(self) -> None:
        """Connect MLIR's native use-def chains and retain exact positions.

        ``Value.uses`` is the authoritative MLIR relation.  It avoids matching
        printed SSA names and, unlike the first implementation, does not depend
        on whether a traversal returned a generic Value or a concrete OpResult.
        A compatibility fallback uses the same structural keys for older
        Polygeist builds whose Python package predates ``Value.uses``.
        """
        for record in sorted(
            self.operation_records,
            key=lambda item: (item.function_id, item.function_ordinal),
        ):
            for position, result_key in enumerate(record.result_keys):
                value_node = self.value_nodes[result_key]
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
                    for result_key in record.result_keys:
                        self.constant_values[result_key] = integer

        resolved_uses: list[tuple[int, OperationRecord, int]] = []
        native_uses_available = True
        try:
            for key, value in sorted(
                self.value_objects.items(),
                key=lambda item: self.value_nodes[item[0]],
            ):
                for use in value.uses:
                    owner = raw_operation(use.owner)
                    user = self.operation_by_key.get(object_key(owner))
                    if user is None:
                        raise RuntimeError(
                            "SSA use owner is outside the indexed function graph: "
                            f"{operation_first_line(owner)}"
                        )
                    resolved_uses.append(
                        (self.value_nodes[key], user, int(use.operand_number))
                    )
        except AttributeError:
            native_uses_available = False
            resolved_uses = []

        if not native_uses_available:
            for record in self.operation_records:
                for position, (operand, key) in enumerate(
                    zip(record.operands, record.operand_keys)
                ):
                    if key not in self.value_nodes:
                        self._get_or_create_value(
                            operand,
                            record.function_id,
                            record.block_id,
                            is_block_argument=False,
                            argument_index=-1,
                        )
                    resolved_uses.append((self.value_nodes[key], record, position))

        use_count_by_value: dict[int, int] = defaultdict(int)
        for value_node, user, position in sorted(
            resolved_uses,
            key=lambda item: (item[1].function_id, item[1].function_ordinal, item[2], item[0]),
        ):
            self.graph.add_edge(
                value_node,
                user.node,
                flow=FLOW_DATA,
                position=position,
                role="operand",
            )
            use_count_by_value[value_node] += 1

        for node in self.value_nodes.values():
            self.graph.nodes[node]["ssa_use_count"] = use_count_by_value.get(node, 0)
        self.graph.graph["ssa_use_api"] = "Value.uses" if native_uses_available else "operand_fallback"

    def _add_control_and_region_edges(self) -> None:
        """Encode direct operation order, CFG successors, and region ownership."""
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
        """Handle from value for the deterministic MLIR-to-MailoHLS graph pipeline."""
        return self.constant_values.get(self._value_key(value))

    def _add_memory_shape_features(self) -> None:
        """Expose MemRef shape/layout facts that the current encoder can learn."""
        for key, value in sorted(
            self.value_objects.items(),
            key=lambda item: self.value_nodes[item[0]],
        ):
            value_node = self.value_nodes[key]
            if int(self.graph.nodes[value_node].get("is_memory", 0)) != 1:
                continue
            try:
                memref = self.ir.MemRefType(value.type)
                shape = [int(dimension) for dimension in memref.shape]
                rank = int(memref.rank)
                static_shape = bool(memref.has_static_shape)
            except (TypeError, ValueError, AttributeError):
                continue

            tokens = [f"memref_rank_{rank}"]
            tokens.append("memref_static_shape" if static_shape else "memref_dynamic_shape")
            total_elements = 1
            has_static_elements = True
            for dimension_index, dimension in enumerate(shape):
                try:
                    dynamic = bool(memref.is_dynamic_dim(dimension_index))
                except (TypeError, ValueError, AttributeError):
                    dynamic = dimension < 0
                if dynamic or dimension <= 0:
                    tokens.append(f"memref_dim_{dimension_index}_dynamic")
                    has_static_elements = False
                    continue
                total_elements *= dimension
                bucket = 0 if dimension <= 1 else int(math.ceil(math.log2(dimension)))
                tokens.append(f"memref_dim_{dimension_index}_log2_{bucket}")
            if has_static_elements:
                bucket = 0 if total_elements <= 1 else int(math.ceil(math.log2(total_elements)))
                tokens.append(f"memref_elements_log2_{bucket}")

            strides: list[int] = []
            offset: int | None = None
            try:
                raw_strides, raw_offset = memref.get_strides_and_offset()
                strides = [int(value) for value in raw_strides]
                offset = int(raw_offset)
                expected = 1
                contiguous = True
                for dimension, stride in reversed(list(zip(shape, strides))):
                    if dimension < 0 or stride != expected:
                        contiguous = False
                        break
                    expected *= dimension
                tokens.append(
                    "memref_contiguous_layout"
                    if contiguous
                    else "memref_strided_or_dynamic_layout"
                )
            except (TypeError, ValueError, AttributeError):
                tokens.append("memref_layout_unknown")

            data = self.graph.nodes[value_node]
            data["memory_shape"] = shape
            data["memory_rank"] = rank
            data["memory_static_shape"] = 1 if static_shape else 0
            data["memory_strides"] = strides
            data["memory_offset"] = offset if offset is not None else "dynamic"

            for position, token in enumerate(dict.fromkeys(tokens)):
                feature = self._new_node(
                    {
                        "block": int(data["block"]),
                        "function": int(data["function"]),
                        "text": token,
                        "type": NODE_TYPE_IMMEDIATE,
                        "full_text": str(value.type),
                        "feature_kind": "memory_shape",
                    }
                )
                self.graph.add_edge(
                    feature,
                    value_node,
                    flow=FLOW_DATA,
                    position=930 + position,
                    role="memory_shape_feature",
                )

    def _integer_attribute(self, operation: Any, *names: str) -> int | None:
        """Read an IntegerAttr through MLIR, with text only as API fallback."""
        attribute = get_raw_attribute(operation, *names)
        if attribute is None:
            return None
        try:
            return int(self.ir.IntegerAttr(attribute).value)
        except (TypeError, ValueError, AttributeError):
            return parse_integer_attr(str(attribute))

    def _affine_map_attribute(self, operation: Any, *names: str) -> Any | None:
        """Downcast one operation attribute to the official AffineMap object."""
        attribute = get_raw_attribute(operation, *names)
        if attribute is None:
            return None
        try:
            return self.ir.AffineMapAttr(attribute).value
        except (TypeError, ValueError, AttributeError):
            return None

    def _constant_affine_bound(self, operation: Any, *names: str) -> int | None:
        """Return a nullary single-result affine-map constant, if present."""
        affine_map = self._affine_map_attribute(operation, *names)
        if affine_map is None:
            return None
        try:
            if affine_map.n_dims != 0 or affine_map.n_symbols != 0:
                return None
            results = list(affine_map.results)
            if len(results) != 1:
                return None
            return int(self.ir.AffineConstantExpr(results[0]).value)
        except (TypeError, ValueError, AttributeError):
            return None

    def _affine_linear_form(
        self,
        expression: Any,
    ) -> tuple[dict[str, int], int, set[str]]:
        """Summarize an AffineExpr using its typed AST, never printed SSA text.

        The coefficient map is exact for affine add/multiply-by-constant forms.
        Mod/floor-div/ceil-div remain explicitly marked because their result is
        piecewise affine and should not be mislabelled as a unit-stride access.
        """
        def cast(name: str) -> Any | None:
            try:
                return getattr(self.ir, name)(expression)
            except (TypeError, ValueError, AttributeError):
                return None

        constant = cast("AffineConstantExpr")
        if constant is not None:
            return {}, int(constant.value), set()
        dimension = cast("AffineDimExpr")
        if dimension is not None:
            return {f"d{int(dimension.position)}": 1}, 0, set()
        symbol = cast("AffineSymbolExpr")
        if symbol is not None:
            return {f"s{int(symbol.position)}": 1}, 0, set()

        for class_name, flag in (
            ("AffineAddExpr", "add"),
            ("AffineMulExpr", "mul"),
            ("AffineModExpr", "mod"),
            ("AffineFloorDivExpr", "floordiv"),
            ("AffineCeilDivExpr", "ceildiv"),
        ):
            binary = cast(class_name)
            if binary is None:
                continue
            left_coeffs, left_constant, left_flags = self._affine_linear_form(binary.lhs)
            right_coeffs, right_constant, right_flags = self._affine_linear_form(binary.rhs)
            flags = set(left_flags) | set(right_flags) | {flag}

            if flag == "add":
                coefficients = dict(left_coeffs)
                for variable, coefficient in right_coeffs.items():
                    coefficients[variable] = coefficients.get(variable, 0) + coefficient
                return coefficients, left_constant + right_constant, flags

            if flag == "mul":
                if not left_coeffs:
                    factor = left_constant
                    return (
                        {name: factor * value for name, value in right_coeffs.items()},
                        factor * right_constant,
                        flags,
                    )
                if not right_coeffs:
                    factor = right_constant
                    return (
                        {name: factor * value for name, value in left_coeffs.items()},
                        factor * left_constant,
                        flags,
                    )
                return {}, 0, flags | {"complex"}

            # Division and modulo are intentionally not linearized.  Retain the
            # numerator coefficients only for scale bucketing and mark the
            # piecewise operation explicitly.
            return left_coeffs, left_constant, flags

        return {}, 0, {"unknown"}

    def _add_affine_access_features(self) -> None:
        """Materialize stable affine-map facts as nodes consumed by data.py.

        Numeric node attributes alone are currently ignored by MailoHLS's
        encoder.  Small categorical feature nodes let the unchanged
        TransformerConv receive access rank, map class, unit/non-unit stride,
        offsets, and piecewise-affine operators without using kernel-specific
        SSA names or an unbounded vocabulary of complete map strings.

        Notice that a coefficient of one in the last affine-map result is not
        necessarily a physical unit-stride access: an ``affine.apply`` may sit
        between the surrounding loop IV and the map operand, and a non-identity
        memref layout may change the physical stride.  We therefore name that
        bounded feature exactly (``unit_coefficient``), while preserving the
        complete SSA chain and the exact map text for a later MLIR C++
        dependence-analysis pass.  This avoids teaching the GNN a false fact.
        """
        for record in self.operation_records:
            affine_map = self._affine_map_attribute(record.operation, "map")
            if affine_map is None:
                continue
            try:
                results = list(affine_map.results)
                dimension_count = int(affine_map.n_dims)
                symbol_count = int(affine_map.n_symbols)
                is_permutation = bool(affine_map.is_permutation)
                is_projected = bool(affine_map.is_projected_permutation)
            except (TypeError, ValueError, AttributeError):
                continue

            summaries = [self._affine_linear_form(expr) for expr in results]
            all_flags = set().union(*(item[2] for item in summaries)) if summaries else set()
            all_coefficients = [
                abs(value)
                for coefficients, _, _ in summaries
                for value in coefficients.values()
                if value != 0
            ]
            max_coefficient = max(all_coefficients, default=0)
            offsets = [constant for _, constant, _ in summaries]

            if is_permutation:
                map_class = "affine_permutation"
            elif is_projected:
                map_class = "affine_projected_permutation"
            else:
                map_class = "affine_general_map"

            tokens = [
                f"affine_rank_{len(results)}",
                f"affine_dims_{dimension_count}",
                f"affine_symbols_{symbol_count}",
                map_class,
            ]
            if record.op_name in READ_OPS | WRITE_OPS | READ_WRITE_OPS:
                last_coefficients, _, last_flags = summaries[-1] if summaries else ({}, 0, set())
                last_unit = (
                    any(abs(value) == 1 for value in last_coefficients.values())
                    and not (last_flags & {"mod", "floordiv", "ceildiv", "complex", "unknown"})
                )
                tokens.append(
                    "affine_last_result_unit_coefficient"
                    if last_unit
                    else "affine_last_result_nonunit_or_piecewise"
                )
            if any(value != 0 for value in offsets):
                tokens.append("affine_has_constant_offset")
            for flag in ("mod", "floordiv", "ceildiv"):
                if flag in all_flags:
                    tokens.append(f"affine_has_{flag}")
            if max_coefficient > 1:
                bucket = int(math.ceil(math.log2(max_coefficient)))
                tokens.append(f"affine_max_coeff_log2_{bucket}")

            node_data = self.graph.nodes[record.node]
            node_data["affine_map"] = str(affine_map)
            node_data["affine_dims"] = dimension_count
            node_data["affine_symbols"] = symbol_count
            node_data["affine_results"] = len(results)
            node_data["affine_map_class"] = map_class

            for position, token in enumerate(dict.fromkeys(tokens)):
                feature = self._new_node(
                    {
                        "block": record.block_id,
                        "function": record.function_id,
                        "text": token,
                        "type": NODE_TYPE_IMMEDIATE,
                        "full_text": str(affine_map),
                        "feature_kind": "affine_access",
                    }
                )
                self.graph.add_edge(
                    feature,
                    record.node,
                    flow=FLOW_DATA,
                    position=910 + position,
                    role="affine_feature",
                )

    def _annotate_loop_features(self, loop: LoopInfo) -> None:
        """Extract loop bounds, step, depth, and static trip-count features."""
        record = loop.op_record
        lower: int | None = None
        upper: int | None = None
        step: int | None = None

        if record.op_name == "scf.for" and len(record.operands) >= 3:
            lower = self._constant_from_value(record.operands[0])
            upper = self._constant_from_value(record.operands[1])
            step = self._constant_from_value(record.operands[2])
        elif record.op_name == "affine.for":
            lower = self._constant_affine_bound(record.operation, "lowerBoundMap")
            upper = self._constant_affine_bound(record.operation, "upperBoundMap")
            step = self._integer_attribute(record.operation, "step") or 1

            # Compatibility for old Polygeist bindings that parse the dialect
            # but do not expose AffineMapAttr downcasts in Python.
            if lower is None or upper is None:
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
        """Expose iter_args and yields that carry data between loop iterations."""
        for loop in self.loops:
            record = loop.op_record
            regions = operation_regions(record.operation)
            if not regions or not region_blocks(regions[0]):
                continue

            if record.op_name == "scf.while":
                if len(regions) < 2 or not region_blocks(regions[1]):
                    continue
                before = region_blocks(regions[0])[0]
                after = region_blocks(regions[1])[0]
                before_args = block_arguments(before)
                after_args = block_arguments(after)
                before_record = self.blocks.get(object_key(before))
                after_record = self.blocks.get(object_key(after))
                condition = (
                    before_record.operations[-1]
                    if before_record is not None and before_record.operations
                    else None
                )
                yielded = (
                    after_record.operations[-1]
                    if after_record is not None and after_record.operations
                    else None
                )

                for position, (initial, argument) in enumerate(
                    zip(record.operands, before_args)
                ):
                    self.graph.add_edge(
                        self.value_nodes[self._value_key(initial)],
                        self.value_nodes[self._value_key(argument)],
                        flow=FLOW_LOOP_CARRIED,
                        position=position,
                        role="while_init",
                    )
                forwarded = (
                    condition.operands[1:]
                    if condition is not None and condition.op_name == "scf.condition"
                    else []
                )
                for position, value in enumerate(forwarded):
                    value_node = self.value_nodes[self._value_key(value)]
                    if position < len(after_args):
                        self.graph.add_edge(
                            value_node,
                            self.value_nodes[self._value_key(after_args[position])],
                            flow=FLOW_LOOP_CARRIED,
                            position=100 + position,
                            role="while_condition_to_after",
                        )
                    if position < len(record.results):
                        self.graph.add_edge(
                            value_node,
                            self.value_nodes[record.result_keys[position]],
                            flow=FLOW_LOOP_CARRIED,
                            position=200 + position,
                            role="while_result",
                        )
                back_values = (
                    yielded.operands
                    if yielded is not None and yielded.op_name == "scf.yield"
                    else []
                )
                for position, (value, argument) in enumerate(
                    zip(back_values, before_args)
                ):
                    self.graph.add_edge(
                        self.value_nodes[self._value_key(value)],
                        self.value_nodes[self._value_key(argument)],
                        flow=FLOW_LOOP_CARRIED,
                        position=300 + position,
                        role="while_backedge",
                    )
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
                initial_node = self.value_nodes[self._value_key(initial)]
                argument_node = self.value_nodes[self._value_key(argument)]
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
                yielded_node = self.value_nodes[self._value_key(yielded)]
                if position < len(iter_args):
                    iter_node = self.value_nodes[self._value_key(iter_args[position])]
                    self.graph.add_edge(
                        yielded_node,
                        iter_node,
                        flow=FLOW_LOOP_CARRIED,
                        position=100 + position,
                        role="loop_backedge",
                    )
                if position < len(loop_results):
                    result_node = self.value_nodes[self._value_key(loop_results[position])]
                    self.graph.add_edge(
                        yielded_node,
                        result_node,
                        flow=FLOW_LOOP_CARRIED,
                        position=200 + position,
                        role="loop_result",
                    )

    def _callee_name(self, record: OperationRecord) -> str:
        """Handle name for the deterministic MLIR-to-MailoHLS graph pipeline."""
        if record.op_name not in CALL_OPS:
            return ""
        for name in ("callee", "callee_name"):
            if name in record.attributes:
                return strip_mlir_string(record.attributes[name])
        line = operation_first_line(record.operation)
        match = re.search(r"@([A-Za-z_.$0-9-]+)", line)
        return match.group(1) if match else ""

    def _add_call_edges(self) -> None:
        """Connect call sites, function definitions, actual values, and formal arguments."""
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
                actual_node = self.value_nodes[self._value_key(actual)]
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
                    returned_node = self.value_nodes[self._value_key(returned)]
                    call_result_node = self.value_nodes[self._value_key(call_result)]
                    self.graph.add_edge(
                        returned_node,
                        call_result_node,
                        flow=FLOW_CALL,
                        position=100 + position,
                        role="return_to_call",
                    )

    def _intrinsic_memory_operands(
        self,
        record: OperationRecord,
    ) -> list[tuple[int, str]]:
        """Return effects defined by the operation itself, excluding calls.

        MLIR's generated dialect documentation identifies affine/memref reads
        and writes through interfaces, but the currently published Python
        ``MemoryEffectsOpInterface`` does not expose an effect-query method.
        The small dialect-semantic table below is therefore deliberately the
        only fallback; call effects are inferred from callee bodies rather than
        being hard-coded as read-write.
        """
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
        if name in CALL_OPS:
            return []
        if "load" in name or name.endswith(".read"):
            return [(memory_positions[0], "read")]
        if "store" in name or name.endswith(".write"):
            return [(memory_positions[0], "write")]
        if "atomic" in name:
            return [(memory_positions[0], "readwrite")]
        return []

    def _add_alias_source(self, source: Any, target: Any) -> None:
        """Record that a memory-typed target may refer to source storage."""
        if source == target:
            return
        if source not in self.value_nodes or target not in self.value_nodes:
            return
        if not is_memory_type(str(self.graph.nodes[self.value_nodes[source]].get("value_type", ""))):
            return
        if not is_memory_type(str(self.graph.nodes[self.value_nodes[target]].get("value_type", ""))):
            return
        self.memory_alias_sources[target].add(source)

    def _collect_memory_aliases(self) -> None:
        """Build SSA forwarding constraints for views, regions, loops, and calls."""
        for record in self.operation_records:
            memory_operand_keys = [
                key
                for value, key in zip(record.operands, record.operand_keys)
                if is_memory_type(value_type(value))
            ]
            memory_result_keys = [
                key
                for value, key in zip(record.results, record.result_keys)
                if is_memory_type(value_type(value))
            ]

            if record.op_name in VIEW_OPS and memory_operand_keys:
                if len(memory_operand_keys) == len(memory_result_keys):
                    pairs = zip(memory_operand_keys, memory_result_keys)
                elif len(memory_operand_keys) == 1:
                    pairs = (
                        (memory_operand_keys[0], target)
                        for target in memory_result_keys
                    )
                else:
                    pairs = (
                        (source, target)
                        for source in memory_operand_keys
                        for target in memory_result_keys
                    )
                for source, target in pairs:
                    self._add_alias_source(source, target)
            elif record.op_name == "arith.select" and memory_result_keys:
                # The condition is scalar; every memory operand is a possible
                # selected root.
                for source in memory_operand_keys:
                    for target in memory_result_keys:
                        self._add_alias_source(source, target)

            if record.op_name in CALL_OPS:
                callee_name = self._callee_name(record)
                callee_id = self.function_name_to_id.get(callee_name)
                if callee_id is not None:
                    for actual, formal in zip(
                        record.operand_keys,
                        self.function_argument_keys.get(callee_id, []),
                    ):
                        self._add_alias_source(actual, formal)
                    for return_record in self.function_returns.get(callee_id, []):
                        for returned, call_result in zip(
                            return_record.operand_keys,
                            record.result_keys,
                        ):
                            self._add_alias_source(returned, call_result)

            # Region results (e.g. scf.if/affine.if) alias any corresponding
            # memory value yielded by an executable region.
            if record.op_name not in LOOP_OPS and record.result_keys:
                for region in operation_regions(record.operation):
                    for block in region_blocks(region):
                        block_record = self.blocks.get(object_key(block))
                        if block_record is None or not block_record.operations:
                            continue
                        terminator = block_record.operations[-1]
                        if terminator.op_name not in REGION_YIELD_OPS:
                            continue
                        for yielded, result in zip(
                            terminator.operand_keys,
                            record.result_keys,
                        ):
                            self._add_alias_source(yielded, result)

        # Loop inits/yields forward to both iter_args and final results.  This
        # includes memref loop-carried state without pretending scalar
        # reductions are memory aliases.
        for loop in self.loops:
            record = loop.op_record
            regions = operation_regions(record.operation)
            if not regions or not region_blocks(regions[0]):
                continue
            if record.op_name == "scf.while":
                if len(regions) < 2 or not region_blocks(regions[1]):
                    continue
                before = region_blocks(regions[0])[0]
                after = region_blocks(regions[1])[0]
                before_args = block_arguments(before)
                after_args = block_arguments(after)
                before_record = self.blocks.get(object_key(before))
                after_record = self.blocks.get(object_key(after))
                condition = (
                    before_record.operations[-1]
                    if before_record is not None and before_record.operations
                    else None
                )
                yielded = (
                    after_record.operations[-1]
                    if after_record is not None and after_record.operations
                    else None
                )
                for initial, argument in zip(record.operands, before_args):
                    self._add_alias_source(
                        self._value_key(initial), self._value_key(argument)
                    )
                forwarded = (
                    condition.operands[1:]
                    if condition is not None and condition.op_name == "scf.condition"
                    else []
                )
                for position, value in enumerate(forwarded):
                    source = self._value_key(value)
                    if position < len(after_args):
                        self._add_alias_source(
                            source, self._value_key(after_args[position])
                        )
                    if position < len(record.result_keys):
                        self._add_alias_source(source, record.result_keys[position])
                back_values = (
                    yielded.operands
                    if yielded is not None and yielded.op_name == "scf.yield"
                    else []
                )
                for value, argument in zip(back_values, before_args):
                    self._add_alias_source(
                        self._value_key(value), self._value_key(argument)
                    )
                continue
            if record.op_name not in {"scf.for", "affine.for"}:
                continue
            body = region_blocks(regions[0])[0]
            arguments = block_arguments(body)
            iter_args = arguments[1:] if arguments else []
            init_values = (
                record.operands[3:]
                if record.op_name == "scf.for"
                else record.operands[-len(iter_args):] if iter_args else []
            )
            block_record = self.blocks.get(object_key(body))
            terminator = (
                block_record.operations[-1]
                if block_record is not None and block_record.operations
                else None
            )
            yielded_values = (
                terminator.operands
                if terminator is not None and terminator.op_name in REGION_YIELD_OPS
                else []
            )
            for initial, argument in zip(init_values, iter_args):
                self._add_alias_source(
                    self._value_key(initial), self._value_key(argument)
                )
            for position, yielded in enumerate(yielded_values):
                yielded_key = self._value_key(yielded)
                if position < len(iter_args):
                    self._add_alias_source(
                        yielded_key, self._value_key(iter_args[position])
                    )
                if position < len(record.result_keys):
                    self._add_alias_source(yielded_key, record.result_keys[position])

    def _compute_function_memory_effects(self) -> None:
        """Infer argument effects from callee bodies to a fixed point."""
        formal_origins: dict[Any, set[int]] = defaultdict(set)
        for function_id, keys in self.function_argument_keys.items():
            for index, key in enumerate(keys):
                if key in self.value_nodes and is_memory_type(
                    str(self.graph.nodes[self.value_nodes[key]].get("value_type", ""))
                ):
                    formal_origins[key].add(index)

        # Propagate only within one function.  Cross-function actual->formal
        # aliases are for physical roots, not for defining a callee's summary.
        changed = True
        while changed:
            changed = False
            for target, sources in self.memory_alias_sources.items():
                target_function = self.value_function.get(target)
                for source in sources:
                    if target_function != self.value_function.get(source):
                        continue
                    before = len(formal_origins[target])
                    formal_origins[target].update(formal_origins.get(source, set()))
                    changed |= len(formal_origins[target]) != before

        # Intraprocedural effects establish the base of the summary lattice.
        for record in self.operation_records:
            if record.op_name in CALL_OPS:
                continue
            for operand_index, mode in self._intrinsic_memory_operands(record):
                key = record.operand_keys[operand_index]
                for argument_index in formal_origins.get(key, set()):
                    previous = self.function_memory_effects[record.function_id].get(
                        argument_index
                    )
                    self.function_memory_effects[record.function_id][argument_index] = (
                        merge_memory_modes(previous, mode)
                    )

        # Then propagate callee summaries through callers.  This is exact for
        # direct calls in the module and converges for recursive call graphs.
        changed = True
        while changed:
            changed = False
            for record in self.operation_records:
                if record.op_name not in CALL_OPS:
                    continue
                callee_name = self._callee_name(record)
                callee_id = self.function_name_to_id.get(callee_name)
                callee_has_body = False
                if callee_id is not None:
                    operation = self.function_operations[callee_id]
                    callee_has_body = any(
                        region_blocks(region) for region in operation_regions(operation)
                    )
                for operand_index, key in enumerate(record.operand_keys):
                    if not is_memory_type(value_type(record.operands[operand_index])):
                        continue
                    if callee_id is None or not callee_has_body:
                        mode = "readwrite"
                    else:
                        mode = self.function_memory_effects[callee_id].get(operand_index)
                        if mode is None:
                            continue
                    for argument_index in formal_origins.get(key, set()):
                        previous = self.function_memory_effects[record.function_id].get(
                            argument_index
                        )
                        merged = merge_memory_modes(previous, mode)
                        if merged != previous:
                            self.function_memory_effects[record.function_id][argument_index] = merged
                            changed = True

        self.graph.graph["function_memory_effects"] = {
            self.functions[function_id]: {
                str(index): mode for index, mode in sorted(effects.items())
            }
            for function_id, effects in sorted(self.function_memory_effects.items())
        }

    def _memory_operands(self, record: OperationRecord) -> list[tuple[int, str]]:
        """Return direct effects or the inferred summary at a call site."""
        if record.op_name not in CALL_OPS:
            return self._intrinsic_memory_operands(record)

        memory_positions = [
            index
            for index, value in enumerate(record.operands)
            if is_memory_type(value_type(value))
        ]
        callee_name = self._callee_name(record)
        callee_id = self.function_name_to_id.get(callee_name)
        if callee_id is None:
            return [(position, "readwrite") for position in memory_positions]
        operation = self.function_operations[callee_id]
        if not any(region_blocks(region) for region in operation_regions(operation)):
            return [(position, "readwrite") for position in memory_positions]
        effects = self.function_memory_effects.get(callee_id, {})
        return [
            (position, effects[position])
            for position in memory_positions
            if position in effects
        ]

    def _solve_memory_roots(self) -> None:
        """Solve the may-alias constraints and add root<->view graph edges."""
        memory_keys = [
            key
            for key, node in self.value_nodes.items()
            if int(self.graph.nodes[node].get("is_memory", 0)) == 1
        ]
        roots: dict[Any, set[int]] = {
            key: (set() if self.memory_alias_sources.get(key) else {self.value_nodes[key]})
            for key in memory_keys
        }

        for _ in range(max(1, len(memory_keys))):
            changed = False
            for target in memory_keys:
                inherited = set().union(
                    *(roots.get(source, set()) for source in self.memory_alias_sources.get(target, set()))
                ) if self.memory_alias_sources.get(target) else set()
                before = len(roots[target])
                roots[target].update(inherited)
                changed |= len(roots[target]) != before
            if not changed:
                break

        # A closed recursive/select SCC without an external seed is still a
        # distinct may-alias root.  Seed it deterministically and propagate once
        # more rather than dropping its memory accesses.
        unresolved = [key for key in memory_keys if not roots[key]]
        for key in unresolved:
            roots[key].add(self.value_nodes[key])
        if unresolved:
            for _ in range(len(memory_keys)):
                changed = False
                for target in memory_keys:
                    inherited = set().union(
                        *(roots.get(source, set()) for source in self.memory_alias_sources.get(target, set()))
                    ) if self.memory_alias_sources.get(target) else set()
                    before = len(roots[target])
                    roots[target].update(inherited)
                    changed |= len(roots[target]) != before
                if not changed:
                    break

        self.memory_roots_by_value = roots
        for key, value_roots in roots.items():
            value_node = self.value_nodes[key]
            self.graph.nodes[value_node]["memory_root_count"] = len(value_roots)
            for root in sorted(value_roots):
                self.graph.nodes[root]["is_memory_root"] = 1
                if root == value_node:
                    continue
                self.graph.add_edge(
                    root,
                    value_node,
                    flow=FLOW_MEMORY_VIEW,
                    position=0,
                    role="may_alias_root",
                )
                self.graph.add_edge(
                    value_node,
                    root,
                    flow=FLOW_MEMORY_VIEW,
                    position=1,
                    role="may_alias_root_reverse",
                )

    def _build_memory_relations(self) -> None:
        """Build alias-aware roots, call effects, accesses, and dependencies."""
        self._collect_memory_aliases()
        self._compute_function_memory_effects()
        self._solve_memory_roots()
        self.graph.graph["memory_alias_model"] = (
            "ssa-views+region-yields+loop-carried+context-insensitive-calls"
        )
        self.graph.graph["call_effect_model"] = "body-summary-fixed-point"
        # The emitted memory-order edges are conservative may-dependences over
        # alias roots.  Exact affine dependence distances require the MLIR C++
        # APIs (MemRefAccess/checkMemrefAccessDependence), which are not exposed
        # by the current Python bindings.  Recording that boundary prevents a
        # downstream experiment from accidentally claiming exact dependence
        # analysis for this pure-Python graph generator.
        self.graph.graph["memory_dependence_model"] = "conservative-may-order"
        self.graph.graph["exact_affine_dependence"] = 0

        for record in sorted(
            self.operation_records,
            key=lambda item: (item.function_id, item.function_ordinal),
        ):
            for operand_index, mode in self._memory_operands(record):
                key = record.operand_keys[operand_index]
                value_node = self.value_nodes[key]
                roots = self.memory_roots_by_value.get(key, {value_node})
                for root in sorted(roots):
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

    def _add_memory_dependence(
        self,
        source: int,
        target: int,
        kind: str,
        certainty: str,
        distance: Any = None,
    ) -> None:
        """Add memory dependence for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
        """Add same-root may-depend edges when exact alias analysis is unavailable."""
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

    @staticmethod
    def _same_source_file(left: str, right: str) -> bool:
        """Compare source files while ignoring machine-specific path prefixes."""
        if not left or not right:
            return False
        # Some cgeist builds spell the single translation unit as stdin even
        # when diagnostics retain line/column information.  This script always
        # compiles exactly one source file, so that spelling is unambiguous.
        if left in {"-", "<stdin>"} or right in {"-", "<stdin>"}:
            return True
        return Path(left).name == Path(right).name

    def _source_distance(
        self,
        record: OperationRecord,
        spec: ActionSpec,
    ) -> int | None:
        """Return column distance for an operation on an action's source line."""
        distances = [
            abs(point.column - spec.source_column)
            for point in record.source_points
            if self._same_source_file(point.filename, spec.source_file)
            and point.line == spec.source_line
        ]
        return min(distances) if distances else None


    def _resolve_loop_action(self, spec: ActionSpec) -> tuple[int, str]:
        """Resolve an action by exact source location.

        MLIR symbols may be C++-mangled, so source-level function names must not
        filter exact FileLineColLoc candidates.
        """
        located: list[tuple[int, int, LoopInfo]] = []

        for loop_index, loop in enumerate(self.loops):
            distance = self._source_distance(loop.op_record, spec)
            if distance is not None:
                located.append((distance, loop_index, loop))

        if located:
            best_distance = min(item[0] for item in located)
            best = [
                item
                for item in located
                if item[0] == best_distance
            ]

            if len(best) == 1:
                return best[0][1], "source_location"

            # Ordinal is only a tie-breaker after exact file and line matching.
            tied_by_ordinal = [
                item
                for item in best
                if item[2].loop_ordinal == spec.loop_ordinal
            ]

            if len(tied_by_ordinal) == 1:
                return tied_by_ordinal[0][1], "source_location+ordinal"

            raise RuntimeError(
                f"Loop action {spec.action_id} at "
                f"{spec.source_file}:{spec.source_line}:{spec.source_column} "
                f"matches {len(best)} equally close MLIR loops."
            )

        # Compatibility fallback only. This must remain forbidden in final graphs.
        same_function = [
            (loop_index, loop)
            for loop_index, loop in enumerate(self.loops)
            if loop.function_name == spec.function
        ]

        ordinal = [
            loop_index
            for loop_index, loop in same_function
            if loop.loop_ordinal == spec.loop_ordinal
        ]

        if len(ordinal) == 1:
            return ordinal[0], "function+ordinal-fallback"

        raise RuntimeError(
            f"Loop action {spec.action_id} matched {len(ordinal)} MLIR loops; "
            f"no loop was found at {spec.source_file}:{spec.source_line}."
        )


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
        """Add pragma node for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
        """Attach every loop Lk exactly once and create its tunable directive nodes."""
        manifest_by_loop: dict[int, tuple[ActionSpec, str]] = {}
        for spec in [item for item in self.actions if item.kind == "loop"]:
            loop_index, resolution = self._resolve_loop_action(spec)
            if resolution.startswith("source_location"):
                self.source_function_ids_from_exact_loops[
                    spec.function
                ].add(
                    self.loops[loop_index].function_id
                )
            if loop_index in manifest_by_loop:
                other, _ = manifest_by_loop[loop_index]
                raise RuntimeError(
                    f"MLIR loop {self.loops[loop_index].function_name}#"
                    f"{self.loops[loop_index].loop_ordinal} maps to both "
                    f"{other.action_id} and {spec.action_id}."
                )
            manifest_by_loop[loop_index] = (spec, resolution)

        for loop_index, loop in enumerate(self.loops):
            match = manifest_by_loop.get(loop_index)
            spec = match[0] if match else None
            resolution = match[1] if match else ""
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
            self.graph.nodes[loop.op_node]["action_resolution"] = resolution
            self.graph.nodes[loop.op_node]["action_source_file"] = spec.source_file
            self.graph.nodes[loop.op_node]["action_source_line"] = spec.source_line
            self.graph.nodes[loop.op_node]["action_source_column"] = spec.source_column

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


    def _resolve_action_function_id(
        self,
        spec: ActionSpec,
    ) -> tuple[int, str]:
        """Resolve a source-level function name to one MLIR function ID.

        extern "C" functions retain their source symbol. Ordinary C++ helpers
        can be mangled, so reuse only a mapping already proven by an exact loop
        FileLineColLoc match.
        """
        direct = self.function_name_to_id.get(spec.function)
        if direct is not None:
            return direct, "function_symbol"

        candidates = sorted(
            self.source_function_ids_from_exact_loops.get(
                spec.function,
                set(),
            )
        )
        if len(candidates) == 1:
            return candidates[0], "function_from_exact_loop"

        if len(candidates) > 1:
            symbols = [
                self.functions[function_id]
                for function_id in candidates
            ]
            raise RuntimeError(
                f"Source function {spec.function!r} maps to multiple "
                f"MLIR functions through exact loop locations: {symbols}"
            )

        raise RuntimeError(
            f"Could not map source function {spec.function!r} to an "
            "MLIR function symbol. No exact source-location loop mapping "
            "was available."
        )

    def _node_memory_shape(
        self,
        node: int,
    ) -> tuple[int, ...]:
        """Return the structured MemRef shape already extracted for a node."""
        raw_shape = self.graph.nodes[node].get(
            "memory_shape",
            [],
        )

        if isinstance(raw_shape, str):
            try:
                raw_shape = json.loads(raw_shape)
            except json.JSONDecodeError:
                return ()

        if not isinstance(raw_shape, (list, tuple)):
            return ()

        try:
            return tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError):
            return ()

    def _root_access_source_lines(
        self,
        root: int,
        source_file: str,
    ) -> set[int]:
        """Return exact source lines of accesses rooted at one allocation."""
        records_by_node = {
            record.node: record
            for record in self.operation_records
        }
        lines: set[int] = set()

        for access in self.memory_accesses:
            if access.root_node != root:
                continue

            record = records_by_node.get(access.op_node)
            if record is None:
                continue

            for point in record.source_points:
                if self._same_source_file(
                    point.filename,
                    source_file,
                ):
                    lines.add(point.line)

        return lines

    def _array_argument_node(
        self,
        spec: ActionSpec,
    ) -> tuple[int, int, int, str]:
        """Resolve an array without changing the graph's action semantics.

        Priority:
          1. exact allocation declaration location;
          2. resolved source function + exact structured MemRef shape;
          3. when same-shaped allocations remain, exact source access lines.

        No declaration-order or global shape-only fallback is introduced.
        """
        all_candidates: list[tuple[OperationRecord, int]] = []

        for record in self.operation_records:
            if record.op_name not in {
                "memref.alloc",
                "memref.alloca",
                "llvm.alloca",
            }:
                continue

            for result_key in record.result_keys:
                node = self.value_nodes[result_key]
                type_text = str(
                    self.graph.nodes[node].get(
                        "value_type",
                        "",
                    )
                )
                if is_memory_type(type_text):
                    all_candidates.append((record, node))

        # Exact declaration location remains the primary key and is searched
        # globally so C++ symbol mangling cannot hide the owning helper.
        located = [
            (self._source_distance(record, spec), record, node)
            for record, node in all_candidates
        ]
        located = [
            item
            for item in located
            if item[0] is not None
        ]

        if located:
            best_distance = min(
                int(item[0])
                for item in located
            )
            best = [
                item
                for item in located
                if int(item[0]) == best_distance
            ]

            if len(best) > 1:
                shaped = [
                    item
                    for item in best
                    if self._node_memory_shape(item[2])
                    == spec.array_dimensions
                ]
                if len(shaped) == 1:
                    best = shaped

            if len(best) != 1:
                raise RuntimeError(
                    f"Array action {spec.action_id} at "
                    f"{spec.source_file}:{spec.source_line}:"
                    f"{spec.source_column} matches {len(best)} "
                    "equally close MLIR allocations."
                )

            _, record, node = best[0]
            return (
                record.function_id,
                node,
                record.block_id,
                "source_location",
            )

        function_id, function_resolution = (
            self._resolve_action_function_id(spec)
        )

        function_candidates = [
            (record, node)
            for record, node in all_candidates
            if record.function_id == function_id
        ]

        shape_candidates = [
            (record, node)
            for record, node in function_candidates
            if self._node_memory_shape(node)
            == spec.array_dimensions
        ]

        if len(shape_candidates) == 1:
            record, node = shape_candidates[0]
            return (
                function_id,
                node,
                record.block_id,
                f"{function_resolution}+unique_shape",
            )

        # Polygeist may place several equal-shaped allocas at function entry.
        # Distinguish them only when their rooted MLIR accesses overlap the
        # exact source lines where the named C/C++ array is used.
        if len(shape_candidates) > 1 and spec.source_use_lines:
            expected_lines = set(spec.source_use_lines)
            scored: list[
                tuple[
                    int,
                    set[int],
                    OperationRecord,
                    int,
                ]
            ] = []

            for record, node in shape_candidates:
                access_lines = self._root_access_source_lines(
                    node,
                    spec.source_file,
                )
                overlap = access_lines & expected_lines
                scored.append(
                    (
                        len(overlap),
                        overlap,
                        record,
                        node,
                    )
                )

            best_score = max(
                score
                for score, _, _, _ in scored
            )
            best = [
                item
                for item in scored
                if item[0] == best_score
            ]

            if best_score > 0 and len(best) == 1:
                _, overlap, record, node = best[0]
                self.graph.nodes[node][
                    "action_source_use_overlap"
                ] = sorted(overlap)
                return (
                    function_id,
                    node,
                    record.block_id,
                    (
                        f"{function_resolution}"
                        "+shape+source_access"
                    ),
                )

        candidate_report = [
            {
                "operation": record.op_name,
                "function": record.function_name,
                "location": record.location,
                "shape": self._node_memory_shape(node),
                "type": self.graph.nodes[node].get(
                    "value_type",
                    "",
                ),
                "access_source_lines": sorted(
                    self._root_access_source_lines(
                        node,
                        spec.source_file,
                    )
                ),
            }
            for record, node in function_candidates
        ]

        if not shape_candidates:
            raise RuntimeError(
                f"Array action {spec.action_id}/{spec.variable}: "
                f"no allocation with exact shape "
                f"{spec.array_dimensions} exists in resolved MLIR "
                f"function {self.functions[function_id]!r}.\n"
                f"Candidate allocations:\n"
                f"{json.dumps(candidate_report, indent=2)}"
            )

        raise RuntimeError(
            f"Array action {spec.action_id}/{spec.variable}: "
            f"shape {spec.array_dimensions} matches "
            f"{len(shape_candidates)} allocations in resolved "
            f"MLIR function {self.functions[function_id]!r}, "
            "and exact source access lines did not identify one "
            "unique physical root.\n"
            f"Candidate allocations:\n"
            f"{json.dumps(candidate_report, indent=2)}"
        )


    def _attach_array_actions(self) -> None:
        """Connect array-partition actions to their root allocation and all accesses."""
        for spec in [item for item in self.actions if item.kind == "array"]:
            if spec.action_id in self.attached_action_ids:
                raise RuntimeError(
                    f"Action {spec.action_id} is attached to multiple MLIR scopes."
                )
            function_id, argument_node, block_id, resolution = self._array_argument_node(spec)
            # A local allocation is itself the physical root.  Alias solving
            # propagates this node through casts and into callee formals, so the
            # scope below can include accesses performed inside helper
            # functions as well as the top-level call operation.
            root = argument_node
            variable = str(spec.variable)
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
                    "action_resolution": resolution,
                    "action_source_file": spec.source_file,
                    "action_source_line": spec.source_line,
                    "action_source_column": spec.source_column,
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
                    if item.root_node == root
                ),
                key=lambda item: (item.function_id, item.op_ordinal),
            ):
                mode_position = {"read": 1, "write": 2, "readwrite": 3}[access.mode]
                self.graph.add_edge(
                    scope,
                    access.op_node,
                    flow=FLOW_ARRAY_SCOPE,
                    position=mode_position,
                    role=f"array_{access.mode}",
                    access_function=access.function_id,
                )
                self.graph.add_edge(
                    access.op_node,
                    scope,
                    flow=FLOW_ARRAY_SCOPE,
                    position=10 + mode_position,
                    role=f"array_{access.mode}_reverse",
                    access_function=access.function_id,
                )
            spec.matched = True
            self.attached_action_ids.add(spec.action_id)

    def _attach_actions(self) -> None:
        """Attach all Lk actions and reject missing or duplicate semantic anchors."""
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
        """Refresh record node IDs for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Parse block edges for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Add function/block helpers required by the existing MailoHLS graph contract."""
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
    """Index pseudo blocks for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Add direct loop parent-child relationships without redundant ancestor edges."""
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
    """Handle action ID for the deterministic MLIR-to-MailoHLS graph pipeline."""
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
    """Enforce graph and action invariants before emitting training data."""
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

    # A scientifically valid program graph must preserve one node per MLIR SSA
    # value.  This catches the generic-Value/OpResult duplication bug directly
    # instead of allowing a visually plausible but disconnected graph to train.
    ssa_ids: dict[str, list[int]] = defaultdict(list)
    ssa_with_def_and_use = 0
    for node, data in graph.nodes(data=True):
        if int(data.get("type", -1)) != NODE_TYPE_VALUE:
            continue
        ssa_id = str(data.get("ssa_id", ""))
        if not ssa_id:
            errors.append(f"Value node {node} has no structural ssa_id.")
        else:
            ssa_ids[ssa_id].append(int(node))

        definition_edges = [
            attrs
            for _, _, attrs in graph.in_edges(node, data=True)
            if int(attrs.get("flow", -1)) == FLOW_DATA
            and attrs.get("role") == "result"
        ]
        use_edges = [
            attrs
            for _, _, attrs in graph.out_edges(node, data=True)
            if int(attrs.get("flow", -1)) == FLOW_DATA
            and attrs.get("role") == "operand"
        ]
        kind = str(data.get("ssa_kind", ""))
        if kind == "op_result" and len(definition_edges) != 1:
            errors.append(
                f"SSA result {ssa_id or node} has {len(definition_edges)} definitions; expected one."
            )
        if kind == "block_argument" and definition_edges:
            errors.append(f"Block argument {ssa_id or node} incorrectly has an op definition.")
        expected_uses = int(data.get("ssa_use_count", -1))
        if expected_uses >= 0 and len(use_edges) != expected_uses:
            errors.append(
                f"SSA value {ssa_id or node} has {len(use_edges)} graph uses but "
                f"MLIR reports {expected_uses}."
            )
        if definition_edges and use_edges:
            ssa_with_def_and_use += 1

    for ssa_id, nodes in sorted(ssa_ids.items()):
        if len(nodes) != 1:
            errors.append(f"Structural SSA id {ssa_id} occurs on nodes {nodes}.")

    for source, target, data in graph.edges(data=True):
        if "flow" not in data or "position" not in data:
            errors.append(f"Edge {source}->{target} misses flow/position.")
            continue
        if int(data["flow"]) not in ALL_FLOWS:
            errors.append(f"Edge {source}->{target} has unknown flow={data['flow']}.")

    action_to_kinds: dict[str, set[str]] = defaultdict(set)
    loop_action_to_pseudos: dict[str, set[int]] = defaultdict(set)
    action_resolutions: Counter[str] = Counter()
    for node, data in graph.nodes(data=True):
        node_type = int(data.get("type", -1))
        if data.get("action_id") and int(data.get("is_loop", 0)) == 1:
            resolution = str(data.get("action_resolution", ""))
            if not resolution and require_actions:
                errors.append(f"Loop action anchor {node} has no resolution provenance.")
            elif resolution:
                action_resolutions[resolution] += 1
                if resolution.endswith("fallback"):
                    warnings.append(
                        f"Loop action anchor {node} used {resolution}; check preserved MLIR locations."
                    )
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
            resolution = str(data.get("action_resolution", ""))
            if not resolution and require_actions:
                errors.append(f"Array scope node {node} has no resolution provenance.")
            elif resolution:
                action_resolutions[resolution] += 1
                if resolution.endswith("fallback"):
                    warnings.append(
                        f"Array scope node {node} used {resolution}; check preserved MLIR locations."
                    )
            if not any(
                int(attrs.get("flow", -1)) == FLOW_PRAGMA
                for _, _, attrs in graph.edges(node, data=True)
            ):
                errors.append(f"Array scope node {node} is not attached to a pragma.")
            scope_edges = [
                attrs
                for _, _, attrs in graph.out_edges(node, data=True)
                if int(attrs.get("flow", -1)) == FLOW_ARRAY_SCOPE
            ]
            if not any(attrs.get("role") == "array_root" for attrs in scope_edges):
                errors.append(f"Array scope node {node} has no physical memory root.")
            if not any(str(attrs.get("role", "")).startswith("array_read") or
                       str(attrs.get("role", "")).startswith("array_write")
                       for attrs in scope_edges):
                warnings.append(f"Array scope node {node} has no resolved memory access.")

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
        "action_resolutions": dict(sorted(action_resolutions.items())),
        "ssa": {
            "values": len(ssa_ids),
            "values_with_definition_and_use": ssa_with_def_and_use,
            "use_source": graph.graph.get("ssa_use_api", "unknown"),
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


# ---------------------------------------------------------------------------
# C/C++ -> canonical HLS-oriented MLIR frontend.
# ---------------------------------------------------------------------------

SOURCE_SUFFIXES = {".c", ".cc", ".cp", ".cpp", ".cxx", ".c++", ".C"}


def resolve_cgeist(requested: str) -> str:
    """Resolve cgeist without silently selecting a different frontend."""
    expanded = Path(requested).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        if not expanded.is_file():
            raise FileNotFoundError(f"cgeist was not found at {expanded}")
        if not os.access(expanded, os.X_OK):
            raise PermissionError(f"cgeist is not executable: {expanded}")
        return str(expanded.resolve())

    found = shutil.which(requested)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {requested!r} on PATH. Pass --cgeist /absolute/path/to/cgeist."
        )
    return str(Path(found).resolve())


def validate_cgeist_flags(flags: Sequence[str]) -> list[str]:
    """Allow preprocessing/target flags, but protect the representation level.

    Include paths, macro definitions, language-standard flags, and target flags
    are legitimate.  Optimization/lowering/output flags would silently change
    the graph semantics or overwrite the temporary MLIR and are rejected.
    """
    forbidden_exact = {
        "-O1",
        "-O2",
        "-O3",
        "-emit-llvm",
        "--emit-llvm",
        "-immediate",
        "--immediate",
        "-raise-scf-to-affine",
        "--raise-scf-to-affine",
        "-memref-fullrank",
        "--memref-fullrank",
        "-print-debug-info",
        "--print-debug-info",
        "-S",
        "-c",
    }
    forbidden_prefixes = (
        "-function=",
        "--function=",
        "-o=",
        "--output=",
        "-scal-rep=",
        "--scal-rep=",
    )
    cleaned = [str(flag) for flag in flags]
    for flag in cleaned:
        if flag in forbidden_exact or flag == "-o" or flag.startswith(forbidden_prefixes):
            raise ValueError(
                f"Conflicting --cflag={flag!r}. mlir_graph_gen.py fixes -O0, "
                "disables affine scalar replacement, preserves full-rank MemRefs, "
                "raises SCF to Affine, and fixes the kernel and output path."
            )
    return cleaned


def build_cgeist_command(
    *,
    executable: str,
    source: Path,
    kernel: str,
    mlir_output: Path,
    cflags: Sequence[str],
) -> list[str]:
    """Return the pinned frontend command used for every dataset example."""
    if not kernel or kernel == "*":
        raise ValueError("--kernel must name exactly one C/C++ top function")
    return [
        executable,
        # compile_source_to_mlir runs with cwd=source.parent.  Passing only the
        # basename prevents machine-specific absolute paths entering the IR.
        source.name,
        *validate_cgeist_flags(cflags),
        f"-function={kernel}",
        "-S",
        "-O0",
        # ARRAY_PARTITION action points must remain represented as MemRefs.
        # Polygeist enables affine scalar replacement by default even at -O0.
        "-scal-rep=0",
        # Source locations are part of the MailoHLS action-mapping contract.
        # Without this cgeist retains locations internally but omits them from
        # the serialized MLIR, forcing unsafe ordinal/shape action fallbacks.
        "-print-debug-info",
        # Preserve every statically-known C array dimension in the MemRef type.
        # Polygeist's own verification suite uses this flag for exactly that
        # contract; without it, a function parameter such as A[10][20] may lose
        # rank/shape information before the graph builder ever sees the IR.
        "-memref-fullrank",
        "-raise-scf-to-affine",
        "-o",
        str(mlir_output),
    ]


def compile_source_to_mlir(
    *,
    source: Path,
    kernel: str,
    mlir_output: Path,
    cgeist: str,
    cflags: Sequence[str],
) -> tuple[list[str], str]:
    """Run Polygeist and return (command, MLIR text), failing with context."""
    if not source.is_file():
        raise FileNotFoundError(f"C/C++ input does not exist: {source}")
    if source.suffix not in SOURCE_SUFFIXES:
        raise ValueError(
            f"Expected a C/C++ source suffix {sorted(SOURCE_SUFFIXES)}, got {source.name!r}"
        )

    executable = resolve_cgeist(cgeist)
    command = build_cgeist_command(
        executable=executable,
        source=source,
        kernel=kernel,
        mlir_output=mlir_output,
        cflags=cflags,
    )
    completed = subprocess.run(
        command,
        cwd=source.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostics"
        raise RuntimeError(
            "cgeist failed while translating the kernel.\n"
            f"Command: {shlex.join(command)}\n"
            f"Diagnostics:\n{detail}"
        )
    if not mlir_output.is_file():
        raise RuntimeError(
            "cgeist returned success but did not create the requested MLIR file:\n"
            f"  {mlir_output}"
        )

    text = mlir_output.read_text(encoding="utf-8")

    # Exact Lk attachment requires the source file/line/column to survive
    # textual MLIR serialization. Detect a frontend/printer configuration
    # error here rather than much later as nine misleading action fallbacks.
    printed_file_loc_re = re.compile(
        r'"(?P<filename>(?:\\.|[^"\\])+)":'
        r'[0-9]+:[0-9]+'
    )
    location_files = {
        match.group("filename")
        for match in printed_file_loc_re.finditer(text)
    }

    if not location_files:
        raise RuntimeError(
            f"cgeist emitted MLIR without any FileLineColLoc "
            f"information for {source.name}. The frontend command "
            "must include -print-debug-info."
        )

    source_location_found = any(
        filename in {"-", "<stdin>"}
        or Path(filename).name == source.name
        for filename in location_files
    )
    if not source_location_found:
        raise RuntimeError(
            f"cgeist emitted source locations, but none correspond "
            f"to {source.name}. Found location files: "
            f"{sorted(location_files)}"
        )

    if not re.search(r"\b(?:func\.func|llvm\.func)\b", text):
        hint = (
            "For C++, --kernel must be the mangled symbol understood by cgeist; "
            "declaring the HLS top as extern \"C\" avoids that ambiguity."
        )
        raise RuntimeError(
            f"Polygeist produced no function body for kernel {kernel!r}. {hint}"
        )
    return command, text


def create_initial_graph(
    mlir_path: Path,
    *,
    actions: list[ActionSpec],
    allow_unregistered_dialects: bool,
    conservative_memory_dependencies: bool,
    require_actions: bool,
) -> tuple[Any, ParseResult]:
    """Build the semantic graph and keep its MLIR Context alive.
    """
    context, module, mlir_text = parse_mlir_module(
        mlir_path,
        allow_unregistered_dialects=allow_unregistered_dialects,
    )

    builder: MlirGraphBuilder | None = None

    try:
        builder = MlirGraphBuilder(
            module=module,
            mlir_text=mlir_text,
            actions=actions,
            conservative_memory_dependencies=(
                conservative_memory_dependencies
            ),
            require_actions=require_actions,
        )

        result = builder.build()
        return context, result

    except BaseException:
        # Release Python wrappers carrying native MLIR handles before
        # closing the explicitly entered Context.
        builder = None
        module = None
        gc.collect()

        context.__exit__(*sys.exc_info())
        raise


# ---------------------------------------------------------------------------
# CLI orchestration.  This layer deliberately stays small: graph semantics
# belong in the builder above, while this code handles paths, subprocesses,
# validation, and the lifetime of MLIR's native Context.
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Path:
    """Compile one labeled kernel and write its validated deterministic training graph."""
    require_pythonhashseed()
    source = Path(args.source).expanduser().resolve()
    kernel_info = source.parent / "kernel_info.txt"
    if not kernel_info.is_file():
        raise FileNotFoundError(f"Expected kernel metadata beside the source: {kernel_info}")

    # kernel_info supplies the optimization contract; source labels supply the
    # semantic function/loop locations that the compact text file omits.
    metadata_kernel, actions = load_kernel_info_actions(source, kernel_info)
    kernel = args.kernel or metadata_kernel
    if args.kernel and args.kernel != metadata_kernel:
        raise ValueError(
            f"--kernel={args.kernel!r} disagrees with the first line of "
            f"{kernel_info.name}: {metadata_kernel!r}"
        )

    # Prevent cgeist from inlining helpers that own labeled loops.  Inlining can
    # duplicate loop actions and optimize away top-function local array buffers.
    helper_functions = sorted({item.function for item in actions if item.function != kernel})
    cflags = [
        *args.cflag,
        *(f"--force-attribute={name}:noinline" for name in helper_functions),
    ]
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name(f"{kernel}_mlir.gexf")
    )

    # Temporary MLIR is an implementation artifact.  The deterministic GEXF is
    # the sole persisted representation consumed by training.
    with tempfile.TemporaryDirectory(prefix="mailohls_mlir_") as directory:
        mlir_path = Path(directory) / f"{source.stem}.hls.mlir"
        frontend_command, mlir_text = compile_source_to_mlir(
            source=source,
            kernel=kernel,
            mlir_output=mlir_path,
            cgeist=args.cgeist,
            cflags=cflags,
        )

        if args.mlir_output:
            audit_mlir = Path(args.mlir_output).expanduser().resolve()
            audit_mlir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mlir_path, audit_mlir)

        context, result = create_initial_graph(
            mlir_path,
            actions=actions,
            allow_unregistered_dialects=True,
            conservative_memory_dependencies=True,
            require_actions=True,
        )
        try:
            initial = result.graph
            initial.graph.update(
                {
                    "kernel": kernel,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "action_sha256": hashlib.sha256(kernel_info.read_bytes()).hexdigest(),
                    "mlir_level": "affine+scf+memref+arith+func",
                    "frontend_policy": (
                        "cgeist:-O0,scal-rep=0,print-debug-info,"
                        "noinline-helpers,memref-fullrank,"
                        "raise-scf-to-affine"
                    ),
                    # The exact frontend binary is part of the experimental
                    # representation, so persist its content hash without
                    # leaking a machine-specific absolute path into the graph.
                    "cgeist_sha256": hashlib.sha256(
                        Path(frontend_command[0]).read_bytes()
                    ).hexdigest(),
                    "mlir_sha256": hashlib.sha256(mlir_text.encode("utf-8")).hexdigest(),
                }
            )
            connected = add_auxiliary_nodes(initial, connected=True)
            training_graph = add_loop_hierarchy(connected, result.loops, transitive=False)
            report = validate_graph(
                training_graph,
                require_actions=True,
                require_single_loop_anchor=True,
            )
            training_graph.graph["action_resolutions"] = report["action_resolutions"]
            fallbacks = {
                name: count
                for name, count in report["action_resolutions"].items()
                if "fallback" in name
            }

            if fallbacks and not args.allow_action_fallbacks:
                raise RuntimeError(
                    "Non-exact action mappings were detected: "
                    f"{fallbacks}. Final training graphs require source-location mappings. "
                    "Use --allow-action-fallbacks only for debugging."
                )
            write_gexf_deterministic(training_graph, output)
        finally:
            # Drop records containing MLIR operation/value wrappers before
            # closing their native Context.
            result = None
            gc.collect()
            context.__exit__(None, None, None)

    for warning in report["warnings"]:
        print(f"mlir_graph_gen.py: warning: {warning}", file=sys.stderr)
    resolution_text = ",".join(
        f"{name}:{count}"
        for name, count in report["action_resolutions"].items()
    )
    print(
        f"Wrote {output} "
        f"({report['nodes']} nodes, {report['edges']} edges, "
        f"{len(report['actions'])} actions; mappings={resolution_text})"
    )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the CLI; kernel_info.txt normally supplies both actions and the top function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="C or C++ source file")
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional top-function override (default: first line of kernel_info.txt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GEXF path (default: <kernel>_mlir.gexf beside the source)",
    )
    parser.add_argument(
        "--cgeist",
        default=os.environ.get("CGEIST", "cgeist"),
        help="cgeist executable (default: $CGEIST or cgeist on PATH)",
    )
    parser.add_argument(
        "--cflag",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Forward one include/define/language/target flag to cgeist; repeat as "
            "needed and use --cflag=-I/path for flags beginning with '-'."
        ),
    )
    parser.add_argument(
        "--allow-action-fallbacks",
        action="store_true",
        help=(
            "Allow function/ordinal or unique-shape action matching. "
            "Disabled by default for final training graphs."
        ),
    )
    parser.add_argument(
        "--mlir-output",
        default="",
        help="Optional path at which to preserve the emitted MLIR.",
    )
    return parser


def main() -> int:
    """Run the CLI and turn expected user errors into a concise nonzero exit status."""
    try:
        run(build_arg_parser().parse_args())
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"mlir_graph_gen.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())