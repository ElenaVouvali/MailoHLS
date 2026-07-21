#!/usr/bin/env python3
"""Apply the MailoHLS GEXF metadata persistence fix to mlir_graph_gen.py."""

from __future__ import annotations

import py_compile
from pathlib import Path


TARGET = Path("GNN_branch/mlir_graph_gen.py")

OLD_CONSTANTS = """ARRAY_SCOPE_TEXT = "array_scope"
SCHEMA_VERSION = "mailohls-mlir-graph-v3"
ACTION_ID_RE = re.compile(r"^L([1-9][0-9]*)$")
ACTION_ID_SEARCH_RE = re.compile(r"\\bL([1-9][0-9]*)\\b")
"""

NEW_CONSTANTS = """ARRAY_SCOPE_TEXT = "array_scope"
SCHEMA_VERSION = "mailohls-mlir-graph-v3"
GRAPH_METADATA_PREFIX = "mailohls-meta-v1:"
PERSISTED_GRAPH_METADATA_KEYS = (
    "kernel",
    "source_sha256",
    "action_sha256",
    "cgeist_sha256",
    "mlir_sha256",
    "mlir_level",
    "frontend_policy",
    "action_resolutions",
)
ACTION_ID_RE = re.compile(r"^L([1-9][0-9]*)$")
ACTION_ID_SEARCH_RE = re.compile(r"\\bL([1-9][0-9]*)\\b")
"""

OLD_WRITER = """def prepare_graph_for_write(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    \"\"\"Prepare graph for write for the deterministic MLIR-to-MailoHLS graph pipeline.\"\"\"
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
"""

NEW_WRITER = """def _metadata_json_value(value: Any) -> Any:
    \"\"\"Convert graph provenance to deterministic JSON-compatible values.\"\"\"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _metadata_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_metadata_json_value(item) for item in value)
    return str(value)


def encode_graph_metadata(graph: nx.MultiDiGraph) -> str:
    \"\"\"Encode provenance in the GEXF graph name.

    NetworkX's GEXF writer does not serialize arbitrary ``graph.graph`` keys.
    It does, however, round-trip the standard GEXF graph ``name`` attribute.
    Store a deterministic JSON envelope there so the batch driver can validate
    schema, source/action hashes, frontend identity, and action resolutions
    without adding a synthetic node to the training graph.
    \"\"\"
    missing = [
        key
        for key in PERSISTED_GRAPH_METADATA_KEYS
        if key not in graph.graph or graph.graph[key] in (None, "")
    ]
    if missing:
        raise RuntimeError(
            "Cannot serialize MailoHLS graph provenance; missing graph "
            f"metadata: {missing}"
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        **{
            key: _metadata_json_value(graph.graph[key])
            for key in PERSISTED_GRAPH_METADATA_KEYS
        },
    }
    return GRAPH_METADATA_PREFIX + json.dumps(
        metadata,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prepare_graph_for_write(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    \"\"\"Prepare a deterministic GEXF while preserving provenance.\"\"\"
    output = nx.MultiDiGraph()
    output.graph.update(
        {
            key: stringify_attr(value)
            for key, value in graph.graph.items()
        }
    )

    # Of the graph-level fields, NetworkX GEXF reliably round-trips ``name``.
    # The JSON envelope is ignored by GNN conversion but retained for strict
    # dataset validation.
    output.graph["name"] = encode_graph_metadata(graph)

    for node, data in graph.nodes(data=True):
        output.add_node(
            node,
            **{
                key: stringify_attr(value)
                for key, value in data.items()
            },
        )

    for source, target, key, data in graph.edges(keys=True, data=True):
        output.add_edge(
            source,
            target,
            key=key,
            **{
                name: stringify_attr(value)
                for name, value in data.items()
            },
        )

    return output
"""


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} block, found {count}. "
            "Your mlir_graph_gen.py differs from the reviewed GitHub version."
        )
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(
            f"Run this script from the MailoHLS repository root; missing {TARGET}"
        )

    original = TARGET.read_text(encoding="utf-8")
    updated = replace_exact(
        original,
        OLD_CONSTANTS,
        NEW_CONSTANTS,
        "schema constants",
    )
    updated = replace_exact(
        updated,
        OLD_WRITER,
        NEW_WRITER,
        "GEXF writer",
    )

    backup = TARGET.with_suffix(TARGET.suffix + ".before_metadata_fix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    TARGET.write_text(updated, encoding="utf-8")
    py_compile.compile(str(TARGET), doraise=True)

    print(f"Updated: {TARGET}")
    print(f"Backup:  {backup}")
    print("Syntax check: OK")


if __name__ == "__main__":
    main()