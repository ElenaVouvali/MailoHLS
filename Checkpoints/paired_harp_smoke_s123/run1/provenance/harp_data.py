#!/usr/bin/env python3
"""
HARP/GNOSIS paired-representation dataset backend for MailoHLS.

This backend deliberately reuses the *same design-point tensor bank* produced
by mlir_data.py and changes only the static program graph representation.
That makes the HARP-Rep vs Structured-MLIR comparison a representation-only
control: row identities, pragma vectors, QoR labels, device/clock conditions,
resource labels, and the family split are inherited byte-for-byte from the
canonical MLIR point cache.

The HARP GEXF graph is tensorized independently.  Dynamic pragma values are
re-routed onto HARP graph scopes using the same MailoHLS helper used by
mlir_data.py, which understands legacy HARP pragma nodes through their
``auto{_..._Lk}`` placeholders and direct pragma-neighbor fallback.

Expected use
------------
First make the small source patch supplied with this resend bundle, then run:

    python -u GNN_branch/harp_data.py \
      --dataset harp --target perf area --target_mode absolute \
      --preprocessed_csv_dir <the exact canonical GNOSIS preprocessing dir> \
      --mlir_dataset_cache_dir <paired MLIR cache> \
      --harp_graph_dir GNN_branch/HARP_graphs \
      --harp_dataset_cache_dir <paired HARP cache> \
      --split_json mailohls_runs/mailohls_final_family_split_s123.json \
      --force_regen

The canonical MLIR point cache is generated/reused through mlir_data.py.  The
HARP cache then copies its point identities and labels and replaces only the
static graph tensors plus HARP-aligned ``X_pragma_per_node`` matrices.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import torch
from torch_geometric.data import Data, Dataset

from config import FLAGS
from utils import get_root_path
import mlir_data as canonical

# Re-export split helpers so train_GNN.py can treat this module exactly like
# mlir_data.py.
get_kernel_samples = canonical.get_kernel_samples
split_dataset = canonical.split_dataset
split_dataset_resample = canonical.split_dataset_resample
split_train_val_test_kernel = canonical.split_train_val_test_kernel

ROOT = Path(get_root_path()).resolve()
SAVE_DIR = (
    Path(FLAGS.harp_dataset_cache_dir).expanduser().resolve()
    if getattr(FLAGS, "harp_dataset_cache_dir", None)
    else ROOT / "GNN_branch" / "HARP_dataset" / "all_kernels"
)
GRAPH_DIR = SAVE_DIR / "graphs"
POINT_DIR = SAVE_DIR / "points"
INDEX_PATH = SAVE_DIR / "index.pt"
ENCODER_PATH = SAVE_DIR / "encoders.pkl"
PRAGMA_DIM_PATH = SAVE_DIR / "pragma_dim.pt"
SCHEMA_PATH = SAVE_DIR / "feature_schema.json"
TARGET_CONDITION_DIM = canonical.TARGET_CONDITION_DIM
FEATURE_SCHEMA_VERSION = "mailohls-harp-features-v1-paired-gnosis"
AUTO_RE = re.compile(r"auto\{([^}]+)\}")


def _graph_root() -> Path:
    explicit = getattr(FLAGS, "harp_graph_dir", None)
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else ROOT / "GNN_branch" / "HARP_graphs"
    )
    if not path.is_dir():
        raise FileNotFoundError(f"HARP graph directory does not exist: {path}")
    return path


def _stable_node_order(graph: nx.Graph) -> tuple[list[Any], dict[Any, int]]:
    def key(node: Any):
        text = str(node)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)
    nodes = sorted(graph.nodes(), key=key)
    return nodes, {node: i for i, node in enumerate(nodes)}


def _kernel_variants(name: str) -> list[str]:
    return list(dict.fromkeys((
        name,
        name.replace("_", "-"),
        name.replace("-", "_"),
    )))


def _find_harp_graph(kernel: str) -> Path:
    root = _graph_root()
    for variant in _kernel_variants(kernel):
        for suffix in ("_processed_result.gexf", ".gexf"):
            candidate = root / f"{variant}{suffix}"
            if candidate.is_file():
                return candidate
    # Last-resort normalized-name match, deterministic and unambiguous only.
    norm = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    wanted = norm(kernel)
    matches = [
        path for path in sorted(root.glob("*.gexf"))
        if norm(path.name.replace("_processed_result", "")) == wanted
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"No unique HARP graph for kernel={kernel!r} in {root}; matches={matches}"
    )


def _cat(attrs: Mapping[str, Any], field: str, default: str = "<none>") -> str:
    value = attrs.get(field, default)
    text = str(value).strip()
    return text if text else default


def _node_token(attrs: Mapping[str, Any]) -> tuple[str, str]:
    return (_cat(attrs, "type", "-1"), _cat(attrs, "text", "<none>").lower())


def _edge_token(attrs: Mapping[str, Any]) -> str:
    return _cat(attrs, "flow", "-1")


def _fit_vocab(graphs: Iterable[nx.Graph]) -> dict[str, dict[str, int]]:
    node_types = {"<unk>"}
    node_text = {"<unk>"}
    edge_flow = {"<unk>"}
    for graph in graphs:
        for _, attrs in graph.nodes(data=True):
            node_type, text = _node_token(attrs)
            node_types.add(node_type)
            node_text.add(text)
        for *_, attrs in graph.edges(data=True):
            edge_flow.add(_edge_token(attrs))
    def table(values):
        return {value: index for index, value in enumerate(sorted(values))}
    return {
        "node_type": table(node_types),
        "node_text": table(node_text),
        "edge_flow": table(edge_flow),
    }


def _onehot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _graph_features(graph: nx.Graph, vocab: Mapping[str, Mapping[str, int]]):
    ordered, node_to_idx = _stable_node_order(graph)
    type_vocab = vocab["node_type"]
    text_vocab = vocab["node_text"]
    flow_vocab = vocab["edge_flow"]
    unk_type = type_vocab["<unk>"]
    unk_text = text_vocab["<unk>"]
    unk_flow = flow_vocab["<unk>"]

    xs = []
    for node in ordered:
        attrs = graph.nodes[node]
        type_token, text_token = _node_token(attrs)
        full = _cat(attrs, "full_text", "").lower()
        node_type = int(_float(attrs.get("type"), -1))
        numeric = [
            math.log1p(max(0.0, _float(attrs.get("block"), 0.0))),
            math.log1p(max(0.0, _float(attrs.get("function"), 0.0))),
            float(node_type == 100),
            float(node_type == 4),
            float(text_token == "icmp" or "icmp" in full or " cmp" in full),
            float(text_token in {"load", "store", "alloca"}),
            float("for.cond" in full or "loop" in full),
            float(bool(AUTO_RE.search(_cat(attrs, "full_text", "")))),
        ]
        xs.append(
            _onehot(type_vocab.get(type_token, unk_type), len(type_vocab))
            + _onehot(text_vocab.get(text_token, unk_text), len(text_vocab))
            + numeric
        )

    edge_pairs = []
    edge_features = []
    if graph.is_multigraph():
        iterator = graph.edges(keys=True, data=True)
        rows = [(s, t, k, a) for s, t, k, a in iterator]
        rows.sort(key=lambda item: (node_to_idx[item[0]], node_to_idx[item[1]], str(item[2])))
    else:
        rows = [(s, t, 0, a) for s, t, a in graph.edges(data=True)]
        rows.sort(key=lambda item: (node_to_idx[item[0]], node_to_idx[item[1]], str(item[2])))
    for source, target, _key, attrs in rows:
        flow = _edge_token(attrs)
        position = _float(attrs.get("position"), 0.0)
        edge_pairs.append([node_to_idx[source], node_to_idx[target]])
        edge_features.append(
            _onehot(flow_vocab.get(flow, unk_flow), len(flow_vocab))
            + [math.copysign(math.log1p(abs(position)), position)]
        )

    x = torch.tensor(xs, dtype=torch.float32)
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features
        else torch.empty((0, len(flow_vocab) + 1), dtype=torch.float32)
    )
    return ordered, x, edge_index, edge_attr


def _placeholder_keys(graph: nx.Graph) -> list[str]:
    keys = set()
    for _, attrs in graph.nodes(data=True):
        full_text = str(attrs.get("full_text", ""))
        for key in AUTO_RE.findall(full_text):
            if key.startswith((
                "_PIPE_", "_UNROLL_", "_ARRAY_T_", "_ARRAY_F_", "_ARRAY_D_"
            )):
                keys.add(key)
    return sorted(keys)


def _point_from_flat(keys: list[str], flat: torch.Tensor) -> dict[str, float]:
    flat = flat.detach().cpu().reshape(-1)
    if len(keys) > flat.numel():
        raise RuntimeError(
            f"HARP action contract has {len(keys)} directive scalars but canonical "
            f"pragma vector has only {flat.numel()} entries"
        )
    return {key: float(flat[index]) for index, key in enumerate(keys)}


def _build_harp_cache() -> None:
    # Generate/reuse canonical MLIR design-point cache.  This intentionally
    # fixes the point population before any HARP graph tensorization.
    canonical_dataset, pragma_dim = canonical.get_data_list()
    del canonical_dataset

    canonical_graph_dir = Path(canonical.GRAPH_DIR)
    canonical_point_dir = Path(canonical.POINT_DIR)
    canonical_index = Path(canonical.INDEX_PATH)
    if not canonical_index.is_file():
        raise FileNotFoundError(f"Canonical MLIR index is missing: {canonical_index}")

    records = torch.load(canonical_index, weights_only=False)
    graph_names = sorted({record["graph_name"] for record in records})
    graph_inputs = []
    for graph_name in graph_names:
        canonical_graph = torch.load(
            canonical_graph_dir / f"{graph_name}.pt", weights_only=False
        )
        kernel = str(canonical_graph["kernel_name"])
        path = _find_harp_graph(kernel)
        graph = nx.read_gexf(path)
        graph_inputs.append((graph_name, kernel, path, graph))

    vocab = _fit_vocab(graph for *_prefix, graph in graph_inputs)

    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    POINT_DIR.mkdir(parents=True, exist_ok=True)

    total_actions = 0
    exact_contract = 0
    extras = 0
    for graph_name, kernel, source_path, graph in graph_inputs:
        ordered, x, edge_index, edge_attr = _graph_features(graph, vocab)
        static_masks = canonical.build_scope_masks_and_dynamic_pragmas(
            graph, ordered, point=None
        )
        keys = _placeholder_keys(graph)
        total_actions += len({key.rsplit("_", 1)[-1] for key in keys})

        canonical_points = torch.load(
            canonical_point_dir / f"{graph_name}.pt", weights_only=False
        )
        new_points = dict(canonical_points)

        # Recover each unique raw directive assignment by the exact source CSV
        # row recorded in the canonical point bank.  This avoids assuming that
        # the HARP graph's placeholder list (which may contain extra inactive
        # action IDs) has the same lexical order as the canonical flat pragma
        # vector.  QoR/row identity still comes only from the canonical cache.
        csv_results = canonical.load_csv_results(kernel)
        raw_point_by_row = {int(result.row_idx): result.point for result in csv_results}
        directive_indices = canonical_points["directive_indices"].tolist()
        first_local_by_directive = {}
        for local_idx, directive_idx in enumerate(directive_indices):
            first_local_by_directive.setdefault(int(directive_idx), local_idx)

        pragma_rows = canonical_points["pragmas"]
        per_node = [None] * int(pragma_rows.shape[0])
        for directive_idx, local_idx in sorted(first_local_by_directive.items()):
            key = str(canonical_points["keys"][local_idx])
            match = re.fullmatch(r"csvrow_(\d+)", key)
            if match is None:
                raise RuntimeError(
                    f"Unexpected canonical row key {key!r} for {kernel}; "
                    "paired HARP tensorization requires csvrow_<index> provenance"
                )
            row_idx = int(match.group(1))
            if row_idx not in raw_point_by_row:
                raise RuntimeError(
                    f"Canonical point row {row_idx} for {kernel} is absent from "
                    "the exact preprocessing table used by this run"
                )
            masks = canonical.build_scope_masks_and_dynamic_pragmas(
                graph, ordered, point=raw_point_by_row[row_idx]
            )
            per_node[directive_idx] = masks["X_pragma_per_node"].float()

        missing_directives = [i for i, value in enumerate(per_node) if value is None]
        if missing_directives:
            raise RuntimeError(
                f"{kernel}: no source row found for directive indices "
                f"{missing_directives[:20]}"
            )
        new_points["X_pragma_per_node"] = torch.stack(per_node, dim=0)
        torch.save(new_points, POINT_DIR / f"{graph_name}.pt")

        graph_payload = {
            "kernel_name": kernel,
            "representation": "harp",
            "source_gexf": str(source_path),
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            **{name: value for name, value in static_masks.items()
               if name != "X_pragma_per_node"},
            "harp_placeholder_keys": keys,
        }
        torch.save(graph_payload, GRAPH_DIR / f"{graph_name}.pt")

    shutil.copy2(canonical_index, INDEX_PATH)
    # The flat pragma contract is identical by construction.
    if Path(canonical.PRAGMA_DIM_PATH).is_file():
        shutil.copy2(canonical.PRAGMA_DIM_PATH, PRAGMA_DIM_PATH)
    else:
        torch.save(pragma_dim, PRAGMA_DIM_PATH)
    with ENCODER_PATH.open("wb") as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "representation": "harp",
        "paired_point_source": str(canonical.SAVE_DIR),
        "node_feature_dim": int(
            len(vocab["node_type"]) + len(vocab["node_text"]) + 8
        ),
        "edge_feature_dim": int(len(vocab["edge_flow"]) + 1),
        "graph_count": len(graph_inputs),
        "record_count": len(records),
        "split_json": getattr(FLAGS, "split_json", None),
        "harp_graph_dir": str(_graph_root()),
        "contract": (
            "same index/point QoR/pragmas as canonical MLIR cache; "
            "static graph representation only is changed"
        ),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(
        f"[DONE] Built {len(records)} paired design points from "
        f"{len(graph_inputs)} HARP graphs -> {SAVE_DIR}"
    )


class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None):
        super().__init__(None, transform, pre_transform)
        self.records = (
            data_files if data_files is not None
            else torch.load(INDEX_PATH, weights_only=False)
        )
        self._graph_cache: dict[str, dict[str, Any]] = {}
        self._point_cache: dict[str, dict[str, Any]] = {}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.records)

    def __len__(self):
        return len(self.records)

    def _load_graph(self, graph_name: str):
        if graph_name not in self._graph_cache:
            self._graph_cache[graph_name] = torch.load(
                GRAPH_DIR / f"{graph_name}.pt", weights_only=False
            )
        return self._graph_cache[graph_name]

    def _load_points(self, graph_name: str):
        if graph_name not in self._point_cache:
            self._point_cache[graph_name] = torch.load(
                POINT_DIR / f"{graph_name}.pt", weights_only=False
            )
        return self._point_cache[graph_name]

    def get(self, index: int) -> Data:
        record = self.records[index]
        graph_name = record["graph_name"]
        local_idx = int(record["local_idx"])
        graph = self._load_graph(graph_name)
        points = self._load_points(graph_name)
        directive_idx = int(points["directive_indices"][local_idx])
        kwargs = {
            "gname": graph["kernel_name"],
            "graph_name": graph_name,
            "kernel": graph["kernel_name"],
            "key": points["keys"][local_idx],
            "target_device": points["target_devices"][local_idx],
            "target_clock_period_ns": points[
                "target_clock_period_ns"
            ][local_idx].view(1).float(),
            "target_group": points["target_groups"][local_idx],
            "target_condition": points[
                "target_condition"
            ][local_idx].view(1, TARGET_CONDITION_DIM).float(),
            "x": graph["x"].float(),
            "edge_index": graph["edge_index"],
            "edge_attr": graph["edge_attr"].float(),
            "X_contextnids": graph["X_contextnids"].float(),
            "X_pragmanids": graph["X_pragmanids"].float(),
            "X_pragmascopenids": graph["X_pragmascopenids"].float(),
            "X_pseudonids": graph["X_pseudonids"].float(),
            "X_arrayscopenids": graph["X_arrayscopenids"].float(),
            "X_pipeline_scopeids": graph["X_pipeline_scopeids"].float(),
            "X_unroll_scopeids": graph["X_unroll_scopeids"].float(),
            "X_array_partition_scopeids": graph[
                "X_array_partition_scopeids"
            ].float(),
            "X_scopenids": graph["X_scopenids"].float(),
            "X_icmpnids": graph["X_icmpnids"].float(),
            "X_pragma_per_node": points[
                "X_pragma_per_node"
            ][directive_idx].float(),
            "pragmas": points["pragmas"][directive_idx].float().unsqueeze(0),
        }
        if str(getattr(FLAGS, "task", "regression")) == "regression":
            kwargs.update({
                "perf": points["perf"][local_idx].view(1).float(),
                "actual_perf": points["actual_perf"][local_idx].view(1).float(),
                "kernel_speedup": points[
                    "kernel_speedup"
                ][local_idx].view(1).float(),
                "area": points["area"][local_idx].view(1).float(),
                "actual_area": points["actual_area"][local_idx].view(1).float(),
                "actual_effective_area": points[
                    "actual_effective_area"
                ][local_idx].view(1).float(),
                "resource_util": points[
                    "resource_util"
                ][local_idx].view(1, 4).float(),
            })
        else:
            kwargs["perf"] = points["perf"][local_idx].view(1).long()
        return Data(**kwargs)


def get_data_list():
    if getattr(FLAGS, "force_regen", False) or not INDEX_PATH.is_file():
        _build_harp_cache()
    dataset = MyOwnDataset()
    pragma_dim = None
    if PRAGMA_DIM_PATH.is_file():
        pragma_dim = torch.load(PRAGMA_DIM_PATH, weights_only=False)
    return dataset, pragma_dim


if __name__ == "__main__":
    dataset, _ = get_data_list()
    if len(dataset) == 0:
        raise RuntimeError("HARP paired dataset is empty")
    sample = dataset[0]
    print(
        json.dumps({
            "status": "PASS",
            "records": len(dataset),
            "node_dim": int(sample.x.shape[1]),
            "edge_dim": int(sample.edge_attr.shape[1]),
            "cache": str(SAVE_DIR),
        }, indent=2)
    )
