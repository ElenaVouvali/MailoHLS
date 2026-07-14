#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


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
FLOW_PSEUDO_BLOCK = 4
FLOW_PSEUDO_CONNECTED = 5
FLOW_LOOP_HIERARCHY = 6
FLOW_ARRAY_SCOPE = 7
FLOW_PRAGMA = 200

ARRAY_SCOPE_TEXT = "array_scope"
ARRAY_SCOPE_MAX_TARGETS = 8
ARRAY_SCOPE_MAX_PER_BLOCK = 2

SSA_RE = re.compile(r"%[A-Za-z_.$0-9]+(?::\d+)?")
SYMBOL_RE = re.compile(r"@[A-Za-z_.$0-9-]+")
FUNC_RE = re.compile(r"\bfunc\.func\s+@([A-Za-z_.$0-9-]+)\s*\(")
OP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


@dataclass
class ArrayPartition:
    arg_index: int
    variable: str
    mode: str
    factor: str | None
    dim: str | None


@dataclass
class MlirFunction:
    name: str
    function_id: int
    entry_block: int
    args: list[str]
    arg_types: dict[str, str]
    array_partitions: list[ArrayPartition] = field(default_factory=list)


@dataclass
class LoopInfo:
    function_id: int
    op_node: int
    op_block: int
    body_block: int
    parent_index: int | None
    children: list[int] = field(default_factory=list)
    full_text: str = ""


@dataclass
class RegionFrame:
    kind: str
    function_id: int
    block_id: int
    loop_index: int | None = None


@dataclass
class ParseResult:
    graph: nx.MultiDiGraph
    functions: dict[int, MlirFunction]
    function_name_to_id: dict[str, int]
    loops: list[LoopInfo]


def require_pythonhashseed() -> None:
    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "Determinism requires PYTHONHASHSEED to be set before Python starts.\n"
            "Run like:\n"
            "  PYTHONHASHSEED=0 python elena/src/mlir_graph_gen_deterministic.py ...\n"
        )


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            current.append(char)
            continue

        if char in depths:
            depths[char] += 1
            current.append(char)
            continue

        if char in closing:
            opener = closing[char]
            depths[opener] = max(0, depths[opener] - 1)
            current.append(char)
            continue

        if char == sep and all(depth == 0 for depth in depths.values()):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def find_matching(text: str, open_index: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(open_index, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"unterminated {open_char}{close_char} expression")


def discover_function_ids(mlir_text: str) -> dict[str, int]:
    names = sorted(set(FUNC_RE.findall(mlir_text)))
    return {name: idx for idx, name in enumerate(names)}


def parse_function_header(line: str) -> tuple[str, str, list[ArrayPartition]]:
    match = FUNC_RE.search(line)
    if not match:
        raise ValueError(f"not a func.func header: {line}")

    name = match.group(1)
    args_open = line.find("(", match.end() - 1)
    args_close = find_matching(line, args_open, "(", ")")
    args_text = line[args_open + 1 : args_close]
    partitions = parse_array_partitions(line)
    return name, args_text, partitions


def parse_function_args(args_text: str) -> tuple[list[str], dict[str, str]]:
    args: list[str] = []
    arg_types: dict[str, str] = {}
    for part in split_top_level(args_text):
        match = SSA_RE.search(part)
        if not match:
            continue
        token = normalize_ssa(match.group(0))
        args.append(token)
        type_text = part[part.find(":") + 1 :].strip() if ":" in part else ""
        arg_types[token] = type_text
    return args, arg_types


def parse_array_partitions(header_line: str) -> list[ArrayPartition]:
    marker = "hls.array_partitions"
    marker_idx = header_line.find(marker)
    if marker_idx == -1:
        return []

    open_idx = header_line.find("[", marker_idx)
    if open_idx == -1:
        return []
    close_idx = find_matching(header_line, open_idx, "[", "]")
    body = header_line[open_idx + 1 : close_idx]

    records = []
    idx = 0
    while idx < len(body):
        open_rec = body.find("{", idx)
        if open_rec == -1:
            break
        close_rec = find_matching(body, open_rec, "{", "}")
        records.append(body[open_rec + 1 : close_rec])
        idx = close_rec + 1

    partitions: list[ArrayPartition] = []
    for record in records:
        fields: dict[str, str] = {}
        for part in split_top_level(record):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            value = value.strip()
            value = value.split(":", 1)[0].strip()
            value = value.strip('"')
            fields[key.strip()] = value

        if not {"arg", "variable", "mode"} <= set(fields):
            continue

        partitions.append(
            ArrayPartition(
                arg_index=int(fields["arg"]),
                variable=fields["variable"],
                mode=fields["mode"],
                factor=fields.get("factor"),
                dim=fields.get("dim"),
            )
        )
    return partitions


def normalize_ssa(token: str) -> str:
    return token.split(":", 1)[0]


def det_get_full_text(ndata: dict[str, Any]) -> str:
    if "full_text" in ndata and ndata["full_text"] is not None:
        return str(ndata["full_text"])
    features = ndata.get("features")
    if isinstance(features, dict):
        ft = features.get("full_text")
        if isinstance(ft, list) and ft:
            return str(ft[0])
    return ""


def det_node_sort_key(node: Any, data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        det_get_full_text(data),
        str(node),
    )


def det_edge_sort_key(
    u: Any,
    v: Any,
    data: dict[str, Any],
    node_rank: dict[Any, int],
) -> tuple[Any, ...]:
    return (
        node_rank.get(u, 10**18),
        node_rank.get(v, 10**18),
        int(data.get("flow", -1)),
        int(data.get("position", -1)),
        str(u),
        str(v),
    )


def det_sha_label(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relabel_nodes_canonically(G: nx.MultiDiGraph, rounds: int = 3) -> nx.MultiDiGraph:
    labels = {n: det_sha_label(det_node_sort_key(n, d)) for n, d in G.nodes(data=True)}

    for _ in range(max(0, rounds)):
        new_labels = {}
        for node in G.nodes():
            out_sig = sorted(
                (
                    "o",
                    labels.get(v, ""),
                    int(ed.get("flow", -1)),
                    int(ed.get("position", -1)),
                )
                for _, v, _, ed in G.out_edges(node, keys=True, data=True)
            )
            in_sig = sorted(
                (
                    "i",
                    labels.get(u, ""),
                    int(ed.get("flow", -1)),
                    int(ed.get("position", -1)),
                )
                for u, _, _, ed in G.in_edges(node, keys=True, data=True)
            )
            new_labels[node] = det_sha_label(
                {"self": labels.get(node, ""), "out": out_sig, "in": in_sig}
            )
        labels = new_labels

    def final_key(node: Any) -> tuple[Any, ...]:
        data = G.nodes[node]
        return (
            labels.get(node, ""),
            int(G.in_degree(node)),
            int(G.out_degree(node)),
            det_node_sort_key(node, data),
            str(node),
        )

    ordered = sorted(G.nodes(), key=final_key)
    mapping = {old: new for new, old in enumerate(ordered)}
    return nx.relabel_nodes(G, mapping, copy=True)


def canonicalize_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    nodes_sorted = sorted(G.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1]))
    for node, data in nodes_sorted:
        H.add_node(node, **deepcopy(data))

    node_rank = {node: idx for idx, (node, _) in enumerate(nodes_sorted)}
    edges = [(u, v, deepcopy(data)) for u, v, _, data in G.edges(keys=True, data=True)]
    for edge_id, (u, v, data) in enumerate(
        sorted(edges, key=lambda e: det_edge_sort_key(e[0], e[1], e[2], node_rank))
    ):
        data["id"] = edge_id
        H.add_edge(u, v, key=edge_id, **data)
    return H


def prepare_graph_for_write(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    for node, data in G.nodes(data=True):
        H.add_node(node, **{key: stringify_attr(value) for key, value in data.items()})
    for u, v, key, data in G.edges(keys=True, data=True):
        H.add_edge(u, v, key=key, **{k: stringify_attr(value) for k, value in data.items()})
    return H


def stringify_attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def write_gexf_deterministic(G: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(prepare_graph_for_write(G), path, prettyprint=False)


def prune_redundant_nodes(G: nx.MultiDiGraph) -> None:
    while True:
        remove_nodes = [node for node in sorted(G.nodes()) if node is None or G.degree(node) == 0]
        if not remove_nodes:
            return
        G.remove_nodes_from(remove_nodes)


class MlirGraphBuilder:
    def __init__(self, mlir_text: str):
        self.mlir_text = mlir_text
        self.function_name_to_id = discover_function_ids(mlir_text)
        self.graph = nx.MultiDiGraph()
        self.functions: dict[int, MlirFunction] = {}
        self.loops: list[LoopInfo] = []

        self.next_node_id = 0
        self.next_block_id = 0
        self.region_stack: list[RegionFrame] = []
        self.value_nodes: dict[tuple[int, str], int] = {}
        self.symbol_nodes: dict[tuple[int, str], int] = {}
        self.prev_op_by_block: dict[int, int] = {}
        self.block_op_positions: dict[int, int] = {}

    def parse(self) -> ParseResult:
        for raw_line in self.mlir_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if FUNC_RE.search(stripped):
                self._enter_function(stripped)
                continue

            if stripped.startswith("}"):
                self._close_region(stripped)
                continue

            if not self.region_stack:
                continue

            self._parse_operation(stripped)

        return ParseResult(
            graph=self.graph,
            functions=self.functions,
            function_name_to_id=self.function_name_to_id,
            loops=self.loops,
        )

    def _new_node(self, attrs: dict[str, Any]) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        self.graph.add_node(node_id, **attrs)
        return node_id

    def _new_block(self) -> int:
        block_id = self.next_block_id
        self.next_block_id += 1
        return block_id

    def _current_frame(self) -> RegionFrame:
        return self.region_stack[-1]

    def _enter_function(self, line: str) -> None:
        name, args_text, partitions = parse_function_header(line)
        function_id = self.function_name_to_id[name]
        entry_block = self._new_block()
        args, arg_types = parse_function_args(args_text)
        self.functions[function_id] = MlirFunction(
            name=name,
            function_id=function_id,
            entry_block=entry_block,
            args=args,
            arg_types=arg_types,
            array_partitions=partitions,
        )
        self.region_stack.append(RegionFrame("function", function_id, entry_block))

        for position, arg in enumerate(args):
            full_text = f"{arg}: {arg_types.get(arg, '')}".strip()
            node = self._get_or_create_value(function_id, arg, entry_block, full_text)
            self.graph.nodes[node]["arg_index"] = position

    def _close_region(self, line: str) -> None:
        if not self.region_stack:
            return
        frame = self.region_stack.pop()
        if frame.kind == "loop" and frame.loop_index is not None:
            self._attach_loop_pragmas(self.loops[frame.loop_index], line)

    def _parse_operation(self, line: str) -> None:
        function_id = self._current_frame().function_id
        block_id = self._current_frame().block_id
        results, op_name = parse_operation_head(line)
        if not op_name:
            return

        op_node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": op_name,
                "type": NODE_TYPE_OP,
                "full_text": line,
            }
        )
        self._add_control_edge(block_id, op_node)

        is_loop = op_name == "scf.for" and "{" in line
        loop_iv = parse_loop_iv(line) if is_loop else None
        loop_iter_args = parse_loop_iter_args(line) if is_loop else []
        result_tokens = [normalize_ssa(token) for token in results]
        excluded_operands = set(result_tokens)
        if loop_iv:
            excluded_operands.add(loop_iv)
        excluded_operands.update(arg for arg, _ in loop_iter_args)

        for position, operand in enumerate(parse_operands(line, excluded_operands)):
            operand_node = self._get_or_create_operand(function_id, operand, block_id)
            self.graph.add_edge(
                operand_node,
                op_node,
                flow=FLOW_DATA,
                position=position,
            )

        immediate = parse_arith_constant_immediate(line)
        if immediate is not None:
            imm_node = self._new_node(
                {
                    "block": block_id,
                    "function": function_id,
                    "text": immediate,
                    "type": NODE_TYPE_IMMEDIATE,
                    "full_text": immediate,
                }
            )
            self.graph.add_edge(imm_node, op_node, flow=FLOW_DATA, position=1000)

        for position, result in enumerate(result_tokens):
            value_node = self._get_or_create_value(
                function_id,
                result,
                block_id,
                f"{result} produced by {line}",
            )
            self.graph.add_edge(op_node, value_node, flow=FLOW_DATA, position=position)

        if is_loop:
            body_block = self._new_block()
            parent_loop = self._nearest_loop_index()
            loop_index = len(self.loops)
            self.loops.append(
                LoopInfo(
                    function_id=function_id,
                    op_node=op_node,
                    op_block=block_id,
                    body_block=body_block,
                    parent_index=parent_loop,
                    full_text=line,
                )
            )
            if parent_loop is not None:
                self.loops[parent_loop].children.append(loop_index)

            if loop_iv:
                iv_node = self._get_or_create_value(
                    function_id,
                    loop_iv,
                    body_block,
                    f"{loop_iv} induction variable for {line}",
                )
                self.graph.add_edge(op_node, iv_node, flow=FLOW_DATA, position=2000)

            for offset, (iter_arg, source) in enumerate(loop_iter_args):
                iter_node = self._get_or_create_value(
                    function_id,
                    iter_arg,
                    body_block,
                    f"{iter_arg} iter_arg for {line}",
                )
                self.graph.add_edge(op_node, iter_node, flow=FLOW_DATA, position=2100 + offset)
                source_node = self._get_or_create_operand(function_id, source, block_id)
                self.graph.add_edge(source_node, op_node, flow=FLOW_DATA, position=2200 + offset)

            self.region_stack.append(
                RegionFrame("loop", function_id, body_block, loop_index=loop_index)
            )
        elif "{" in line:
            body_block = self._new_block()
            self.region_stack.append(RegionFrame("generic", function_id, body_block))

    def _add_control_edge(self, block_id: int, op_node: int) -> None:
        if block_id in self.prev_op_by_block:
            position = self.block_op_positions.get(block_id, 0)
            self.graph.add_edge(
                self.prev_op_by_block[block_id],
                op_node,
                flow=FLOW_CONTROL,
                position=position,
            )
            self.block_op_positions[block_id] = position + 1
        else:
            self.block_op_positions[block_id] = 0
        self.prev_op_by_block[block_id] = op_node

    def _nearest_loop_index(self) -> int | None:
        for frame in reversed(self.region_stack):
            if frame.kind == "loop":
                return frame.loop_index
        return None

    def _get_or_create_operand(self, function_id: int, token: str, block_id: int) -> int:
        if token.startswith("%"):
            return self._get_or_create_value(function_id, token, block_id, token)
        if token.startswith("@"):
            return self._get_or_create_symbol(function_id, token, block_id)
        return self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": token,
                "type": NODE_TYPE_IMMEDIATE,
                "full_text": token,
            }
        )

    def _get_or_create_value(
        self,
        function_id: int,
        token: str,
        block_id: int,
        full_text: str,
    ) -> int:
        key = (function_id, normalize_ssa(token))
        if key in self.value_nodes:
            return self.value_nodes[key]
        node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": key[1],
                "type": NODE_TYPE_VALUE,
                "full_text": full_text,
            }
        )
        self.value_nodes[key] = node
        return node

    def _get_or_create_symbol(self, function_id: int, token: str, block_id: int) -> int:
        key = (function_id, token)
        if key in self.symbol_nodes:
            return self.symbol_nodes[key]
        node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": token,
                "type": NODE_TYPE_VALUE,
                "full_text": token,
            }
        )
        self.symbol_nodes[key] = node
        return node

    def _attach_loop_pragmas(self, loop: LoopInfo, closing_line: str) -> None:
        pipeline_match = re.search(r"\bloop_pipeline_ii\s*=\s*([0-9]+)", closing_line)
        unroll_match = re.search(r"\bloop_unroll_factor\s*=\s*([0-9]+)", closing_line)

        if pipeline_match:
            self._add_pragma_pair(
                loop.op_node,
                loop.op_block,
                loop.function_id,
                "PIPELINE",
                f"#pragma HLS PIPELINE II={pipeline_match.group(1)}",
                dependency_blocks=[loop.op_block, loop.body_block],
            )

        if unroll_match:
            self._add_pragma_pair(
                loop.op_node,
                loop.op_block,
                loop.function_id,
                "UNROLL",
                f"#pragma HLS UNROLL factor={unroll_match.group(1)}",
                dependency_blocks=[loop.op_block, loop.body_block],
            )

    def _add_pragma_pair(
        self,
        anchor_node: int,
        block_id: int,
        function_id: int,
        pragma_kind: str,
        full_text: str,
        dependency_blocks: list[int] | None = None,
    ) -> int:
        pragma_node = self._new_node(
            {
                "block": block_id,
                "function": function_id,
                "text": pragma_kind,
                "type": NODE_TYPE_PRAGMA,
                "full_text": full_text,
                "dependency_blocks": sorted(set(dependency_blocks or [block_id])),
            }
        )
        attr = {"flow": FLOW_PRAGMA, "position": PRAGMA_POSITION[pragma_kind]}
        self.graph.add_edge(anchor_node, pragma_node, **attr)
        self.graph.add_edge(pragma_node, anchor_node, **attr)
        return pragma_node


def parse_operation_head(line: str) -> tuple[list[str], str | None]:
    body = line
    results: list[str] = []

    equals_idx = find_assignment_equals(line)
    if equals_idx is not None:
        lhs = line[:equals_idx]
        rhs = line[equals_idx + 1 :].strip()
        results = [normalize_ssa(token) for token in SSA_RE.findall(lhs)]
        body = rhs

    op_name = body.split(None, 1)[0] if body.split(None, 1) else ""
    if not OP_RE.match(op_name):
        return [], None
    return results, op_name


def find_assignment_equals(line: str) -> int | None:
    in_string = False
    escaped = False
    for idx, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "=":
            lhs = line[:idx].strip()
            if is_result_lhs(lhs):
                return idx
            return None
    return None


def is_result_lhs(lhs: str) -> bool:
    if not lhs:
        return False
    parts = split_top_level(lhs)
    if not parts:
        return False
    return all(SSA_RE.fullmatch(part.strip()) for part in parts)


def parse_loop_iv(line: str) -> str | None:
    match = re.search(r"\bscf\.for\s+(%[A-Za-z_.$0-9]+)", line)
    return normalize_ssa(match.group(1)) if match else None


def parse_loop_iter_args(line: str) -> list[tuple[str, str]]:
    marker = "iter_args"
    marker_idx = line.find(marker)
    if marker_idx == -1:
        return []
    open_idx = line.find("(", marker_idx)
    if open_idx == -1:
        return []
    close_idx = find_matching(line, open_idx, "(", ")")
    body = line[open_idx + 1 : close_idx]

    out: list[tuple[str, str]] = []
    for part in split_top_level(body):
        if "=" not in part:
            continue
        left, right = part.split("=", 1)
        left_match = SSA_RE.search(left)
        right_match = SSA_RE.search(right)
        if left_match and right_match:
            out.append((normalize_ssa(left_match.group(0)), normalize_ssa(right_match.group(0))))
    return out


def parse_operands(line: str, excluded: set[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    for token in SSA_RE.findall(line):
        normalized = normalize_ssa(token)
        if normalized in excluded or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)

    for token in SYMBOL_RE.findall(line):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    return tokens


def parse_arith_constant_immediate(line: str) -> str | None:
    match = re.search(r"\barith\.constant\s+(.+?)\s*:", line)
    if not match:
        return None
    return match.group(1).strip()


def format_array_partition_pragma(partition: ArrayPartition) -> str:
    parts = [
        "#pragma HLS ARRAY_PARTITION",
        f"variable={partition.variable}",
        partition.mode,
    ]
    if partition.factor is not None:
        parts.append(f"factor={partition.factor}")
    if partition.dim is not None:
        parts.append(f"dim={partition.dim}")
    return " ".join(parts)


def add_array_partition_nodes(result: ParseResult) -> None:
    G = result.graph
    next_node_id = max(G.nodes(), default=-1) + 1

    def add_node(attrs: dict[str, Any]) -> int:
        nonlocal next_node_id
        node = next_node_id
        next_node_id += 1
        G.add_node(node, **attrs)
        return node

    for function_id in sorted(result.functions):
        function = result.functions[function_id]
        for partition in sorted(
            function.array_partitions,
            key=lambda item: (item.arg_index, item.variable, item.mode, item.factor or "", item.dim or ""),
        ):
            if partition.arg_index >= len(function.args):
                continue
            arg_token = function.args[partition.arg_index]
            arg_node = find_value_node(G, function_id, arg_token)
            if arg_node is None:
                continue

            arg_data = G.nodes[arg_node]
            block_id = int(arg_data.get("block", function.entry_block))
            pragma_line = format_array_partition_pragma(partition)
            targets = select_array_targets(G, function_id, arg_token, arg_node)
            dependency_blocks = sorted(
                {
                    int(G.nodes[target].get("block", block_id))
                    for target in targets
                }
                | {block_id}
            )

            pragma_node = add_node(
                {
                    "block": block_id,
                    "function": function_id,
                    "text": "ARRAY_PARTITION",
                    "type": NODE_TYPE_PRAGMA,
                    "full_text": pragma_line,
                    "dependency_blocks": dependency_blocks,
                }
            )
            scope_node = add_node(
                {
                    "block": block_id,
                    "function": function_id,
                    "text": ARRAY_SCOPE_TEXT,
                    "type": NODE_TYPE_ARRAY_SCOPE,
                    "full_text": f"array_scope<{partition.variable}> from pragma: {pragma_line}",
                    "array_var": partition.variable,
                    "mlir_arg": arg_token,
                    "dependency_blocks": dependency_blocks,
                }
            )

            pragma_attr = {"flow": FLOW_PRAGMA, "position": PRAGMA_POSITION["ARRAY_PARTITION"]}
            G.add_edge(pragma_node, scope_node, **pragma_attr)
            G.add_edge(scope_node, pragma_node, **pragma_attr)

            for position, target in enumerate(targets):
                attr = {"flow": FLOW_ARRAY_SCOPE, "position": position}
                G.add_edge(scope_node, target, **attr)
                G.add_edge(target, scope_node, **attr)


def find_value_node(G: nx.MultiDiGraph, function_id: int, token: str) -> int | None:
    matches = [
        node
        for node, data in G.nodes(data=True)
        if int(data.get("function", -1)) == function_id
        and int(data.get("type", -1)) == NODE_TYPE_VALUE
        and str(data.get("text", "")) == token
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda node: det_node_sort_key(node, G.nodes[node]))[0]


def select_array_targets(
    G: nx.MultiDiGraph,
    function_id: int,
    arg_token: str,
    arg_node: int,
) -> list[int]:
    candidates: list[int] = [arg_node]
    for node, data in G.nodes(data=True):
        if int(data.get("function", -1)) != function_id:
            continue
        if node == arg_node:
            continue
        ntype = int(data.get("type", -1))
        if ntype not in (NODE_TYPE_OP, NODE_TYPE_VALUE):
            continue
        full_text = det_get_full_text(data)
        if arg_token in full_text:
            candidates.append(node)

    ordered = sorted(set(candidates), key=lambda node: array_target_priority(node, G.nodes[node]))
    selected: list[int] = []
    seen_blocks: dict[int, int] = {}

    if arg_node in ordered:
        selected.append(arg_node)
        seen_blocks[int(G.nodes[arg_node].get("block", -1))] = 1

    for node in ordered:
        if node in selected:
            continue
        block_id = int(G.nodes[node].get("block", -1))
        if seen_blocks.get(block_id, 0) >= ARRAY_SCOPE_MAX_PER_BLOCK:
            continue
        selected.append(node)
        seen_blocks[block_id] = seen_blocks.get(block_id, 0) + 1
        if len(selected) >= ARRAY_SCOPE_MAX_TARGETS:
            return selected

    for node in ordered:
        if node in selected:
            continue
        selected.append(node)
        if len(selected) >= ARRAY_SCOPE_MAX_TARGETS:
            break
    return selected


def array_target_priority(node: int, data: dict[str, Any]) -> tuple[Any, ...]:
    text = str(data.get("text", ""))
    full_text = det_get_full_text(data)
    if int(data.get("type", -1)) == NODE_TYPE_VALUE and "arg_index" in data:
        rank = 0
    elif text == "memref.load":
        rank = 1
    elif text == "memref.store":
        rank = 2
    elif text in {"memref.alloc", "memref.get_global"}:
        rank = 3
    elif int(data.get("type", -1)) == NODE_TYPE_OP:
        rank = 4
    else:
        rank = 5
    return (
        rank,
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        text,
        full_text,
        str(node),
    )


def create_initial_graph(mlir_path: Path) -> ParseResult:
    builder = MlirGraphBuilder(mlir_path.read_text(encoding="utf-8"))
    result = builder.parse()
    add_array_partition_nodes(result)
    prune_redundant_nodes(result.graph)
    result.graph = relabel_nodes_canonically(result.graph, rounds=3)
    result.graph = canonicalize_graph(result.graph)
    return result


def add_auxiliary_nodes(
    source: nx.MultiDiGraph,
    connected: bool,
) -> nx.MultiDiGraph:
    G = deepcopy(source)
    next_node_id = max(G.nodes(), default=-1) + 1

    block_nodes: dict[tuple[int, int], int] = {}
    position_by_block: dict[tuple[int, int], int] = {}
    original_nodes = sorted(G.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1]))

    for node, data in original_nodes:
        function_id = int(data.get("function", -1))
        block_id = int(data.get("block", -1))
        key = (function_id, block_id)
        if key not in block_nodes:
            pseudo_id = next_node_id
            next_node_id += 1
            block_nodes[key] = pseudo_id
            position_by_block[key] = 0
            G.add_node(
                pseudo_id,
                block=block_id,
                function=function_id,
                text="pseudo_block",
                type=NODE_TYPE_PSEUDO_BLOCK,
                full_text="auxiliary node for each block",
            )

        pseudo_id = block_nodes[key]
        position = position_by_block[key]
        attr = {"flow": FLOW_PSEUDO_BLOCK, "position": position}
        G.add_edge(node, pseudo_id, **attr)
        G.add_edge(pseudo_id, node, **attr)
        position_by_block[key] = position + 1

    for node, data in original_nodes:
        function_id = int(data.get("function", -1))
        block_id = int(data.get("block", -1))
        node_key = (function_id, block_id)
        for dependency_block in parse_dependency_blocks(data.get("dependency_blocks")):
            dependency_key = (function_id, dependency_block)
            pseudo_id = block_nodes.get(dependency_key)
            if pseudo_id is None or dependency_key == node_key:
                continue
            position = position_by_block[dependency_key]
            attr = {"flow": FLOW_PSEUDO_BLOCK, "position": position}
            G.add_edge(node, pseudo_id, **attr)
            G.add_edge(pseudo_id, node, **attr)
            position_by_block[dependency_key] = position + 1

    if connected:
        ordered_blocks = sorted(block_nodes)
        position = 0
        for left_idx, left in enumerate(ordered_blocks):
            for right in ordered_blocks[left_idx + 1 :]:
                left_node = block_nodes[left]
                right_node = block_nodes[right]
                attr = {"flow": FLOW_PSEUDO_CONNECTED, "position": position}
                G.add_edge(left_node, right_node, **attr)
                G.add_edge(right_node, left_node, **attr)
                position += 1

    prune_redundant_nodes(G)
    G = relabel_nodes_canonically(G, rounds=3)
    return canonicalize_graph(G)


def parse_dependency_blocks(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        parsed = [value]

    blocks: list[int] = []
    for item in parsed:
        try:
            blocks.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(blocks))


def add_loop_hierarchy(
    source: nx.MultiDiGraph,
    loops: list[LoopInfo],
) -> nx.MultiDiGraph:
    G = deepcopy(source)
    pseudo_by_block = index_pseudo_blocks(G)
    hierarchy_pairs: set[tuple[int, int]] = set()

    for loop in loops:
        for child_index in loop.children:
            child = loops[child_index]
            parent_pseudo = pseudo_by_block.get((loop.function_id, loop.body_block))
            child_pseudo = pseudo_by_block.get((child.function_id, child.body_block))
            if parent_pseudo is None or child_pseudo is None:
                continue
            hierarchy_pairs.add(tuple(sorted((parent_pseudo, child_pseudo))))

    for loop_index, loop in enumerate(loops):
        if loop.children:
            continue
        chain = loop_ancestor_chain(loops, loop_index)
        pseudo_chain = [
            pseudo_by_block[(ancestor.function_id, ancestor.body_block)]
            for ancestor in chain
            if (ancestor.function_id, ancestor.body_block) in pseudo_by_block
        ]
        for left_idx, left_node in enumerate(pseudo_chain):
            for right_node in pseudo_chain[left_idx + 1 :]:
                hierarchy_pairs.add(tuple(sorted((left_node, right_node))))

    for position, (left_node, right_node) in enumerate(
        sorted(
            hierarchy_pairs,
            key=lambda pair: (
                det_node_sort_key(pair[0], G.nodes[pair[0]]),
                det_node_sort_key(pair[1], G.nodes[pair[1]]),
            ),
        )
    ):
        attr = {"flow": FLOW_LOOP_HIERARCHY, "position": position}
        G.add_edge(left_node, right_node, **attr)
        G.add_edge(right_node, left_node, **attr)

    prune_redundant_nodes(G)
    G = relabel_nodes_canonically(G, rounds=3)
    return canonicalize_graph(G)


def loop_ancestor_chain(loops: list[LoopInfo], loop_index: int) -> list[LoopInfo]:
    chain: list[LoopInfo] = []
    current: int | None = loop_index
    while current is not None:
        loop = loops[current]
        chain.append(loop)
        current = loop.parent_index
    return list(reversed(chain))


def index_pseudo_blocks(G: nx.MultiDiGraph) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for node, data in sorted(G.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1])):
        if int(data.get("type", -1)) != NODE_TYPE_PSEUDO_BLOCK:
            continue
        key = (int(data.get("function", -1)), int(data.get("block", -1)))
        out[key] = node
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def graph_counts(name: str, G: nx.MultiDiGraph) -> dict[str, Any]:
    return {"name": name, "num_node": G.number_of_nodes(), "num_edge": G.number_of_edges()}


def augmented_counts(name: str, before: nx.MultiDiGraph, after: nx.MultiDiGraph) -> dict[str, Any]:
    blocks = {
        (int(data.get("function", -1)), int(data.get("block", -1)))
        for _, data in after.nodes(data=True)
        if int(data.get("type", -1)) == NODE_TYPE_PSEUDO_BLOCK
    }
    row = {
        "name": name,
        "prev_node": before.number_of_nodes(),
        "prev_edge": before.number_of_edges(),
        "new_node": after.number_of_nodes(),
        "new_edge": after.number_of_edges(),
    }
    if blocks:
        row["block"] = len(blocks)
    return row


def run(args: argparse.Namespace) -> None:
    require_pythonhashseed()
    mlir_path = Path(args.input)
    out_dir = Path(args.out_dir)
    kernel = args.kernel or mlir_path.stem.split(".")[0]

    result = create_initial_graph(mlir_path)
    initial = result.graph

    if args.mode in {"initial", "all"}:
        initial_path = out_dir / "processed" / "original" / f"{kernel}_processed_result.gexf"
        write_gexf_deterministic(initial, initial_path)
        write_csv(out_dir / "initial.csv", ["name", "num_node", "num_edge"], [graph_counts(kernel, initial)])

    if args.mode == "all":
        aux_base = add_auxiliary_nodes(initial, connected=False)
        base_path = out_dir / "processed" / "extended-pseudo-block-base" / f"{kernel}_processed_result.gexf"
        write_gexf_deterministic(aux_base, base_path)
        write_csv(
            out_dir / "auxiliary_False.csv",
            ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
            [augmented_counts(kernel, initial, aux_base)],
        )

        aux_connected = add_auxiliary_nodes(initial, connected=True)
        connected_path = (
            out_dir
            / "processed"
            / "extended-pseudo-block-connected"
            / f"{kernel}_processed_result.gexf"
        )
        write_gexf_deterministic(aux_connected, connected_path)
        write_csv(
            out_dir / "auxiliary_True.csv",
            ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
            [augmented_counts(kernel, initial, aux_connected)],
        )

        hierarchy = add_loop_hierarchy(aux_connected, result.loops)
        hierarchy_path = (
            out_dir
            / "processed"
            / "extended-pseudo-block-connected-hierarchy"
            / f"{kernel}_processed_result.gexf"
        )
        write_gexf_deterministic(hierarchy, hierarchy_path)
        write_csv(
            out_dir / "hierarchy.csv",
            ["name", "prev_node", "prev_edge", "new_node", "new_edge"],
            [augmented_counts(kernel, aux_connected, hierarchy)],
        )

    elif args.mode == "auxiliary":
        aux = add_auxiliary_nodes(initial, connected=args.connected)
        folder = "extended-pseudo-block-connected" if args.connected else "extended-pseudo-block-base"
        aux_path = out_dir / "processed" / folder / f"{kernel}_processed_result.gexf"
        write_gexf_deterministic(aux, aux_path)
        write_csv(
            out_dir / f"auxiliary_{args.connected}.csv",
            ["name", "prev_node", "prev_edge", "new_node", "new_edge", "block"],
            [augmented_counts(kernel, initial, aux)],
        )

    elif args.mode == "hierarchy":
        aux_connected = add_auxiliary_nodes(initial, connected=True)
        hierarchy = add_loop_hierarchy(aux_connected, result.loops)
        hierarchy_path = (
            out_dir
            / "processed"
            / "extended-pseudo-block-connected-hierarchy"
            / f"{kernel}_processed_result.gexf"
        )
        write_gexf_deterministic(hierarchy, hierarchy_path)
        write_csv(
            out_dir / "hierarchy.csv",
            ["name", "prev_node", "prev_edge", "new_node", "new_edge"],
            [augmented_counts(kernel, aux_connected, hierarchy)],
        )

    print(f"Generated MLIR graph artifacts for {kernel} under {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Annotated SCF/memref MLIR file")
    parser.add_argument("--kernel", default=None, help="Kernel/output name; defaults to input stem")
    parser.add_argument("--out-dir", default="mlir_harp", help="Output graph directory")
    parser.add_argument(
        "--mode",
        choices=["initial", "auxiliary", "hierarchy", "all"],
        default="all",
        help="Pipeline stage to write",
    )
    parser.add_argument(
        "--connected",
        action="store_true",
        help="For --mode auxiliary, write connected pseudo-block graph",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
