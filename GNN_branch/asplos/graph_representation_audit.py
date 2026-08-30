#!/usr/bin/env python3
"""Paired ASPLOS audit of MailoHLS HARP and MLIR graph banks.

The script intentionally uses only the Python standard library so it can run
before the GPU/HLS environments are activated.  It pairs graphs by kernel,
checks action-ID agreement, extracts GEXF node/edge attributes, and writes a
paper-ready LaTeX table together with auditable CSV files.

This is a representation audit, not an accuracy comparison.  GNN ranking/QoR
and downstream Stage-2 effects must be reported by separate matched-model runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


FLOW_NAMES = {
    0: "control",
    1: "data",
    2: "call",
    4: "pseudo_block",
    5: "pseudo_connected",
    6: "loop_hierarchy",
    7: "array_scope",
    8: "region",
    9: "memory_view",
    10: "memory_access",
    11: "loop_carried",
    12: "memory_dependence",
    13: "memory_uncertainty",
    200: "pragma",
}

ACTION_RE = re.compile(r"(?:^|[^A-Za-z0-9])L(\d+)(?:[^0-9]|$)")
MLIR_METADATA_PREFIX = "mailohls-meta-v1:"


@dataclass(frozen=True)
class GraphStats:
    kernel: str
    representation: str
    path: str
    nodes: int
    edges: int
    nonpragma_relation_types: int
    actions: int
    action_ids: str
    control_edges: int
    data_edges: int
    call_edges: int
    loop_hierarchy_edges: int
    array_scope_edges: int
    region_edges: int
    memory_view_edges: int
    memory_access_edges: int
    loop_carried_edges: int
    memory_dependence_edges: int
    memory_uncertainty_edges: int
    pragma_edges: int
    compiler_dependence_queries: int
    compiler_proven_dependences: int
    compiler_unresolved_queries: int
    compiler_view_edges: int
    graph_bytes: int


def _namespace(root: ET.Element) -> dict[str, str]:
    if root.tag.startswith("{"):
        return {"g": root.tag[1:].split("}", 1)[0]}
    return {"g": ""}


def _findall(element: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    if ns["g"]:
        return list(element.findall(path, ns))
    return list(element.findall(path.replace("g:", "")))


def _attribute_map(
    root: ET.Element, ns: dict[str, str], attribute_class: str
) -> dict[str, str]:
    maps: dict[str, str] = {}
    path = f".//g:attributes[@class='{attribute_class}']/g:attribute"
    for item in _findall(root, path, ns):
        maps[item.attrib["id"]] = item.attrib.get("title", item.attrib["id"])
    return maps


def _attvalues(
    element: ET.Element, ns: dict[str, str], attributes: dict[str, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _findall(element, "./g:attvalues/g:attvalue", ns):
        key = attributes.get(item.attrib.get("for", ""), item.attrib.get("for", ""))
        result[key] = item.attrib.get("value", "")
    return result


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _action_ids(node_values: list[dict[str, str]]) -> set[str]:
    actions: set[str] = set()
    for values in node_values:
        explicit = values.get("action_id", "").strip()
        if explicit:
            actions.add(explicit.upper())
        for field in ("full_text", "features", "text"):
            for match in ACTION_RE.finditer(values.get(field, "")):
                actions.add(f"L{int(match.group(1))}")
    return actions


def _metadata(graph_name: str) -> dict[str, object]:
    if not graph_name.startswith(MLIR_METADATA_PREFIX):
        return {}
    try:
        value = json.loads(graph_name[len(MLIR_METADATA_PREFIX) :])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_gexf(path: Path, kernel: str, representation: str) -> GraphStats:
    root = ET.parse(path).getroot()
    ns = _namespace(root)
    node_attributes = _attribute_map(root, ns, "node")
    edge_attributes = _attribute_map(root, ns, "edge")

    graph_nodes = _findall(root, ".//g:nodes/g:node", ns)
    graph_edges = _findall(root, ".//g:edges/g:edge", ns)
    node_values = [_attvalues(node, ns, node_attributes) for node in graph_nodes]

    flows: Counter[int] = Counter()
    for edge in graph_edges:
        values = _attvalues(edge, ns, edge_attributes)
        flows[_as_int(values.get("flow"), -1)] += 1

    graph_elements = _findall(root, ".//g:graph", ns)
    graph_name = graph_elements[0].attrib.get("name", "") if graph_elements else ""
    metadata = _metadata(graph_name)
    compiler_coverage = metadata.get("compiler_analysis_coverage", {})
    if not isinstance(compiler_coverage, dict):
        compiler_coverage = {}

    actions = _action_ids(node_values)
    return GraphStats(
        kernel=kernel,
        representation=representation,
        path=str(path),
        nodes=len(graph_nodes),
        edges=len(graph_edges),
        nonpragma_relation_types=len(
            {flow for flow, count in flows.items() if count and flow != 200 and flow >= 0}
        ),
        actions=len(actions),
        action_ids=";".join(sorted(actions, key=lambda x: int(x[1:]))),
        control_edges=flows[0],
        data_edges=flows[1],
        call_edges=flows[2],
        loop_hierarchy_edges=flows[6],
        array_scope_edges=flows[7],
        region_edges=flows[8],
        memory_view_edges=flows[9],
        memory_access_edges=flows[10],
        loop_carried_edges=flows[11],
        memory_dependence_edges=flows[12],
        memory_uncertainty_edges=flows[13],
        pragma_edges=flows[200],
        compiler_dependence_queries=_as_int(compiler_coverage.get("dependence")),
        compiler_proven_dependences=_as_int(metadata.get("proven_dependence_edge_count")),
        compiler_unresolved_queries=_as_int(
            metadata.get("unresolved_dependence_query_count")
        ),
        compiler_view_edges=_as_int(compiler_coverage.get("view_edges")),
        graph_bytes=path.stat().st_size,
    )


def _kernel_name(path: Path, representation: str) -> str:
    stem = path.stem
    if representation == "HARP" and stem.endswith("_processed_result"):
        stem = stem[: -len("_processed_result")]
    return stem


def _graph_bank(directory: Path, representation: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.gexf")):
        kernel = _kernel_name(path, representation)
        if kernel in result:
            raise RuntimeError(f"Duplicate {representation} graph for {kernel}")
        result[kernel] = path
    return result


def _expected_action_ids(dataset_dir: Path, kernel: str) -> set[str]:
    kernel_info = dataset_dir / kernel / "kernel_info.txt"
    if not kernel_info.is_file():
        raise FileNotFoundError(f"Missing action contract: {kernel_info}")
    actions: set[str] = set()
    for line in kernel_info.read_text(encoding="utf-8").splitlines():
        first_field = line.split(",", 1)[0].strip().upper()
        if re.fullmatch(r"L\d+", first_field):
            actions.add(f"L{int(first_field[1:])}")
    if not actions:
        raise RuntimeError(f"No active actions found in {kernel_info}")
    return actions


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _median_iqr(values: list[int]) -> str:
    numeric = [float(value) for value in values]
    median = statistics.median(numeric)
    q1 = _percentile(numeric, 0.25)
    q3 = _percentile(numeric, 0.75)

    def format_value(value: float) -> str:
        return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"

    return f"{format_value(median)} [{format_value(q1)}--{format_value(q3)}]"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write_latex_table(
    path: Path,
    paired: list[tuple[GraphStats, GraphStats]],
    harp_contract_matches: int,
    mlir_contract_matches: int,
    expected_actions: int,
    harp_active_actions: int,
    mlir_active_actions: int,
    harp_extra_actions: int,
    mlir_extra_actions: int,
) -> None:
    harp = [item[0] for item in paired]
    mlir = [item[1] for item in paired]

    def values(items: list[GraphStats], field: str) -> list[int]:
        return [int(getattr(item, field)) for item in items]

    def structural_tuple(items: list[GraphStats]) -> str:
        fields = (
            "region_edges",
            "memory_access_edges",
            "loop_carried_edges",
            "memory_dependence_edges",
            "memory_uncertainty_edges",
        )
        return "/".join(_median_iqr(values(items, field)).split(" ", 1)[0] for field in fields)

    rows = [
        ("Nodes/kernel", _median_iqr(values(harp, "nodes")), _median_iqr(values(mlir, "nodes"))),
        ("Edges/kernel", _median_iqr(values(harp, "edges")), _median_iqr(values(mlir, "edges"))),
        ("Non-pragma relation types/kernel", _median_iqr(values(harp, "nonpragma_relation_types")), _median_iqr(values(mlir, "nonpragma_relation_types"))),
        ("R/M/C/D/U edges", structural_tuple(harp), structural_tuple(mlir)),
    ]

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Paired graph audit over {len(paired)} GN$\Omega$SIS kernels. Values are median [IQR]; R/M/C/D/U are region, memory-access, loop-carried, proven-dependence, and uncertainty edges (medians). Zero means not encoded as a distinct edge relation.}}",
        r"  \label{tab:representation-audit}",
        r"  \small",
        r"  \begin{tabularx}{\columnwidth}{@{}Xrr@{}}",
        r"    \toprule",
        r"    Metric & HARP-style & Structured MLIR \\",
        r"    \midrule",
    ]
    lines.extend(
        f"    {_latex_escape(metric)} & {harp_value} & {mlir_value} \\\\" 
        for metric, harp_value, mlir_value in rows
    )
    lines.extend(
        [
            r"    \midrule",
            f"    Exact active-contract match & {harp_contract_matches}/{len(paired)} & {mlir_contract_matches}/{len(paired)} \\\\ ",
            f"    Active action coverage & {harp_active_actions}/{expected_actions} & {mlir_active_actions}/{expected_actions} \\\\ ",
            f"    Extra/inactive action IDs & {harp_extra_actions} & {mlir_extra_actions} \\\\ ",
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harp-dir",
        type=Path,
        default=Path("GNN_branch/HARP_graphs"),
    )
    parser.add_argument(
        "--mlir-dir",
        type=Path,
        default=Path("GNN_branch/MLIR_graphs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asplos_results/representation_audit"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Data/ApplicationDataset"),
        help="Authoritative per-kernel kernel_info.txt action contracts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harp_bank = _graph_bank(args.harp_dir, "HARP")
    mlir_bank = _graph_bank(args.mlir_dir, "MLIR")

    missing_mlir = sorted(set(harp_bank) - set(mlir_bank))
    missing_harp = sorted(set(mlir_bank) - set(harp_bank))
    if missing_mlir or missing_harp:
        raise RuntimeError(
            "Graph banks are not paired. "
            f"Missing MLIR={missing_mlir}; missing HARP={missing_harp}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired: list[tuple[GraphStats, GraphStats]] = []
    detailed_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    harp_contract_matches = 0
    mlir_contract_matches = 0
    expected_action_count = 0
    harp_active_action_count = 0
    mlir_active_action_count = 0
    harp_extra_action_count = 0
    mlir_extra_action_count = 0

    for kernel in sorted(harp_bank):
        harp = parse_gexf(harp_bank[kernel], kernel, "HARP")
        mlir = parse_gexf(mlir_bank[kernel], kernel, "MLIR")
        paired.append((harp, mlir))
        detailed_rows.extend([asdict(harp), asdict(mlir)])

        harp_actions = set(filter(None, harp.action_ids.split(";")))
        mlir_actions = set(filter(None, mlir.action_ids.split(";")))
        expected_actions = _expected_action_ids(args.dataset_dir, kernel)
        harp_contract_match = harp_actions == expected_actions
        mlir_contract_match = mlir_actions == expected_actions
        harp_contract_matches += int(harp_contract_match)
        mlir_contract_matches += int(mlir_contract_match)
        expected_action_count += len(expected_actions)
        harp_active_action_count += len(harp_actions & expected_actions)
        mlir_active_action_count += len(mlir_actions & expected_actions)
        harp_extra_action_count += len(harp_actions - expected_actions)
        mlir_extra_action_count += len(mlir_actions - expected_actions)
        pair_rows.append(
            {
                "kernel": kernel,
                "expected_actions": ";".join(
                    sorted(expected_actions, key=lambda x: int(x[1:]))
                ),
                "harp_contract_match": harp_contract_match,
                "mlir_contract_match": mlir_contract_match,
                "harp_actions": harp.action_ids,
                "mlir_actions": mlir.action_ids,
                "harp_missing_actions": ";".join(
                    sorted(expected_actions - harp_actions, key=lambda x: int(x[1:]))
                ),
                "mlir_missing_actions": ";".join(
                    sorted(expected_actions - mlir_actions, key=lambda x: int(x[1:]))
                ),
                "harp_extra_actions": ";".join(
                    sorted(harp_actions - expected_actions, key=lambda x: int(x[1:]))
                ),
                "mlir_extra_actions": ";".join(
                    sorted(mlir_actions - expected_actions, key=lambda x: int(x[1:]))
                ),
                "harp_nodes": harp.nodes,
                "mlir_nodes": mlir.nodes,
                "node_ratio_mlir_over_harp": f"{mlir.nodes / harp.nodes:.6f}",
                "harp_edges": harp.edges,
                "mlir_edges": mlir.edges,
                "edge_ratio_mlir_over_harp": f"{mlir.edges / harp.edges:.6f}",
                "harp_relation_types": harp.nonpragma_relation_types,
                "mlir_relation_types": mlir.nonpragma_relation_types,
            }
        )

    _write_csv(args.output_dir / "graph_stats_long.csv", detailed_rows)
    _write_csv(args.output_dir / "paired_graph_stats.csv", pair_rows)
    _write_latex_table(
        args.output_dir / "representation_audit_table.tex",
        paired,
        harp_contract_matches,
        mlir_contract_matches,
        expected_action_count,
        harp_active_action_count,
        mlir_active_action_count,
        harp_extra_action_count,
        mlir_extra_action_count,
    )

    summary = {
        "paired_kernels": len(paired),
        "expected_active_actions": expected_action_count,
        "harp_exact_contract_matches": harp_contract_matches,
        "mlir_exact_contract_matches": mlir_contract_matches,
        "harp_recovered_active_actions": harp_active_action_count,
        "mlir_recovered_active_actions": mlir_active_action_count,
        "harp_extra_inactive_actions": harp_extra_action_count,
        "mlir_extra_inactive_actions": mlir_extra_action_count,
        "harp_only_kernels": missing_mlir,
        "mlir_only_kernels": missing_harp,
        "warning": (
            "This audit establishes paired structural coverage only. Do not use it "
            "to claim better prediction quality without matched GNN and Stage-2 runs."
        ),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
