import argparse
import hashlib
import glob
import json
import os
import re
import subprocess
import sys
import torch


def parse_builder_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint_contract", required=True)
    parser.add_argument("--checkpoint_sidecar", required=True)
    parser.add_argument("--feature_schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_slots", type=int, default=64)
    parser.add_argument(
        "--embedding_mode",
        required=True,
        choices=["current_zero_scope_post_npt", "static_pre_npt"],
    )
    parser.add_argument(
        "--gexf_dir",
        default="GNN_branch/MLIR_graphs",
    )
    parser.add_argument(
        "--layerwise_out",
        default="",
    )
    return parser.parse_args(argv)


BUILD_ARGS = parse_builder_args(sys.argv[1:])
sys.argv = [sys.argv[0]]


# ----------------------------------------------------------
# Action-level structural relation schema for Stage 2.
# ----------------------------------------------------------

ACTION_RELATION_SCHEMA = "mailohls-action-relations-v1"

NODE_TYPE_OP = 0
NODE_TYPE_PSEUDO_BLOCK = 4
NODE_TYPE_PRAGMA = 100
NODE_TYPE_ARRAY_SCOPE = 104

FLOW_CONTROL = 0
FLOW_CALL = 2
FLOW_PSEUDO_BLOCK = 4
FLOW_LOOP_HIERARCHY = 6
FLOW_ARRAY_SCOPE = 7
FLOW_REGION = 8
FLOW_MEMORY_DEPENDENCE = 12

REL_SELF = 1 << 0
REL_PARENT = 1 << 1
REL_CHILD = 1 << 2
REL_ARRAY = 1 << 3
REL_DEP_FORWARD = 1 << 4
REL_DEP_REVERSE = 1 << 5


from os.path import join, basename
from torch_geometric.data import Batch, Data


def _edge_records(
    graph,
    source,
    target,
):
    mapping = graph.get_edge_data(
        source,
        target,
        default={},
    )

    return list(
        mapping.values()
    )


def _has_edge(
    graph,
    source,
    target,
    *,
    flow=None,
    role=None,
    certainty=None,
):
    for attrs in _edge_records(
        graph,
        source,
        target,
    ):
        if (
            flow is not None
            and int(
                attrs.get(
                    "flow",
                    -1,
                )
            ) != int(flow)
        ):
            continue

        if (
            role is not None
            and str(
                attrs.get(
                    "role",
                    "",
                )
            ) != str(role)
        ):
            continue

        if (
            certainty is not None
            and str(
                attrs.get(
                    "certainty",
                    "",
                )
            ) != str(certainty)
        ):
            continue

        return True

    return False


def _numeric_node_map(graph):
    """
    nx.read_gexf() normally gives string node IDs.
    MailoHLS tensor rows use their integer equivalents.
    """
    output = {}

    for node in graph.nodes():
        numeric = int(node)

        if numeric in output:
            raise RuntimeError(
                f"Duplicate numeric GEXF node {numeric}"
            )

        output[numeric] = node

    return output


def _loop_action_scope_pseudo(
    graph,
    action_id,
):
    pseudos = set()

    for node, attrs in graph.nodes(
        data=True
    ):
        if (
            int(
                attrs.get(
                    "type",
                    -1,
                )
            )
            != NODE_TYPE_PRAGMA
        ):
            continue

        if (
            str(
                attrs.get(
                    "action_id",
                    "",
                )
            )
            != action_id
        ):
            continue

        if (
            str(
                attrs.get(
                    "text",
                    "",
                )
            ).upper()
            not in {
                "PIPELINE",
                "UNROLL",
            }
        ):
            continue

        neighbours = (
            set(
                graph.predecessors(
                    node
                )
            )
            |
            set(
                graph.successors(
                    node
                )
            )
        )

        for neighbour in neighbours:

            nattrs = graph.nodes[
                neighbour
            ]

            if (
                int(
                    nattrs.get(
                        "type",
                        -1,
                    )
                )
                != NODE_TYPE_PSEUDO_BLOCK
            ):
                continue

            connected = (
                _has_edge(
                    graph,
                    node,
                    neighbour,
                    flow=FLOW_PSEUDO_BLOCK,
                )
                or
                _has_edge(
                    graph,
                    neighbour,
                    node,
                    flow=FLOW_PSEUDO_BLOCK,
                )
            )

            if connected:
                pseudos.add(
                    neighbour
                )

    if len(pseudos) != 1:
        raise RuntimeError(
            f"{action_id}: expected exactly "
            f"one loop pseudo scope, "
            f"found {len(pseudos)}"
        )

    return next(
        iter(pseudos)
    )



def _loop_region_distances(
    graph,
    loop_anchor,
):
    """
    Return nodes structurally reachable inside the loop body.

    Critically:
      - enter child regions
      - move forward inside blocks/CFG
      - optionally enter called helpers
      - NEVER follow region_exit back out of the loop
    """

    distances = {}
    queue = []

    # Enter the loop's regions only.
    for _, target, attrs in (
        graph.out_edges(
            loop_anchor,
            data=True,
        )
    ):
        if (
            int(
                attrs.get(
                    "flow",
                    -1,
                )
            )
            == FLOW_REGION
            and
            str(
                attrs.get(
                    "role",
                    "",
                )
            )
            == "region_entry"
        ):
            queue.append(
                (target, 1)
            )

    while queue:
        node, distance = (
            queue.pop(0)
        )

        previous = distances.get(
            node
        )

        if (
            previous is not None
            and previous <= distance
        ):
            continue

        distances[
            node
        ] = distance

        for _, target, attrs in (
            graph.out_edges(
                node,
                data=True,
            )
        ):
            flow = int(
                attrs.get(
                    "flow",
                    -1,
                )
            )

            role = str(
                attrs.get(
                    "role",
                    "",
                )
            )

            follow = False

            if (
                flow == FLOW_REGION
                and role
                == "region_entry"
            ):
                follow = True

            elif (
                flow == FLOW_CONTROL
                and role
                in {
                    "next_in_block",
                    "block_successor",
                }
            ):
                follow = True

            # If a loop contains a helper call,
            # include compiler-visible helper
            # operations too.
            elif (
                flow == FLOW_CALL
                and role == "calls"
            ):
                follow = True

            if follow:
                queue.append(
                    (
                        target,
                        distance + 1,
                    )
                )

    return distances



def build_action_relation_mask(
    graph,
    *,
    node_ids,
    active_mask,
    max_slots,
):
    """
    Project the compiler graph into MailoHLS action space.

    N(Lk) contains:
      1. self
      2. nearest labeled parent/child loops
      3. ARRAY_PARTITION actions accessed in the loop
      4. actions related by proven memory dependences
    """

    numeric_nodes = (
        _numeric_node_map(
            graph
        )
    )

    active_mask = (
        torch.as_tensor(
            active_mask,
            dtype=torch.bool,
        )
        .view(-1)
        .cpu()
    )

    relation_bits = torch.zeros(
        (
            max_slots,
            max_slots,
        ),
        dtype=torch.int16,
    )

    # --------------------------------------------------
    # slot -> actual GEXF semantic-anchor node
    # --------------------------------------------------

    slot_to_anchor = {}

    for slot, raw_node_id in enumerate(
        node_ids
    ):
        if not bool(
            active_mask[slot]
        ):
            continue

        if int(raw_node_id) < 0:
            raise RuntimeError(
                f"L{slot + 1}: active slot "
                "has no semantic node"
            )

        numeric_id = int(
            raw_node_id
        )

        if (
            numeric_id
            not in numeric_nodes
        ):
            raise RuntimeError(
                f"L{slot + 1}: "
                f"GEXF node {numeric_id} "
                "does not exist"
            )

        slot_to_anchor[
            slot
        ] = numeric_nodes[
            numeric_id
        ]

    # --------------------------------------------------
    # A. Self relation.
    # --------------------------------------------------

    for slot in slot_to_anchor:
        relation_bits[
            slot,
            slot,
        ] |= REL_SELF

    # --------------------------------------------------
    # Identify loop / array actions.
    # --------------------------------------------------

    loop_slots = {}
    array_slots = {}

    for slot, anchor in (
        slot_to_anchor.items()
    ):
        attrs = graph.nodes[
            anchor
        ]

        if (
            int(
                attrs.get(
                    "is_loop",
                    0,
                )
            )
            == 1
        ):
            loop_slots[
                slot
            ] = anchor

        elif (
            int(
                attrs.get(
                    "type",
                    -1,
                )
            )
            == NODE_TYPE_ARRAY_SCOPE
        ):
            array_slots[
                slot
            ] = anchor

    # --------------------------------------------------
    # B. Loop hierarchy.
    #
    # Map Lk loop actions to their block-scope pseudo.
    # --------------------------------------------------

    scope_to_slot = {}
    slot_to_scope = {}

    for slot in loop_slots:

        action_id = (
            f"L{slot + 1}"
        )

        pseudo = (
            _loop_action_scope_pseudo(
                graph,
                action_id,
            )
        )

        if (
            pseudo in scope_to_slot
            and scope_to_slot[
                pseudo
            ] != slot
        ):
            raise RuntimeError(
                "Two different Lk loop actions "
                "map to the same pseudo scope"
            )

        scope_to_slot[
            pseudo
        ] = slot

        slot_to_scope[
            slot
        ] = pseudo

    # All loop scopes, including unlabeled loops.
    parent_of = {}

    for source, target, attrs in (
        graph.edges(
            data=True
        )
    ):
        if (
            int(
                attrs.get(
                    "flow",
                    -1,
                )
            )
            == FLOW_LOOP_HIERARCHY
            and
            str(
                attrs.get(
                    "role",
                    "",
                )
            )
            == "loop_parent"
        ):
            if (
                target in parent_of
                and parent_of[
                    target
                ] != source
            ):
                raise RuntimeError(
                    "Loop pseudo scope has "
                    "multiple structural parents"
                )

            parent_of[
                target
            ] = source

    # Connect each labeled loop action to its
    # nearest labeled ancestor.
    #
    # This also handles an unlabeled loop lying
    # between two MailoHLS action loops.
    for child_slot, child_scope in (
        slot_to_scope.items()
    ):
        ancestor = parent_of.get(
            child_scope
        )

        while (
            ancestor is not None
            and ancestor
            not in scope_to_slot
        ):
            ancestor = parent_of.get(
                ancestor
            )

        if ancestor is None:
            continue

        parent_slot = (
            scope_to_slot[
                ancestor
            ]
        )

        relation_bits[
            child_slot,
            parent_slot,
        ] |= REL_PARENT

        relation_bits[
            parent_slot,
            child_slot,
        ] |= REL_CHILD

    # --------------------------------------------------
    # C. Compute operations structurally owned/reachable
    #    by every Lk loop.
    # --------------------------------------------------

    loop_distances = {}

    for slot, anchor in (
        loop_slots.items()
    ):
        loop_distances[
            slot
        ] = (
            _loop_region_distances(
                graph,
                anchor,
            )
        )

    # --------------------------------------------------
    # D. Array actions.
    #
    # array_scope already has exact edges to every
    # access of its memory root in your MLIR graph.
    # --------------------------------------------------

    array_accesses = {}

    for array_slot, anchor in (
        array_slots.items()
    ):
        accesses = set()

        for _, target, attrs in (
            graph.out_edges(
                anchor,
                data=True,
            )
        ):
            if (
                int(
                    attrs.get(
                        "flow",
                        -1,
                    )
                )
                != FLOW_ARRAY_SCOPE
            ):
                continue

            role = str(
                attrs.get(
                    "role",
                    "",
                )
            )

            if role in {
                "array_read",
                "array_write",
                "array_readwrite",
            }:
                accesses.add(
                    target
                )

        array_accesses[
            array_slot
        ] = accesses

    # Every loop action sees every ARRAY_PARTITION action
    # whose memory is actually accessed in that loop's
    # structural region.
    for loop_slot, distances in (
        loop_distances.items()
    ):
        owned_nodes = set(
            distances
        )

        for array_slot, accesses in (
            array_accesses.items()
        ):
            if not (
                owned_nodes
                & accesses
            ):
                continue

            relation_bits[
                loop_slot,
                array_slot,
            ] |= REL_ARRAY

            relation_bits[
                array_slot,
                loop_slot,
            ] |= REL_ARRAY

    # --------------------------------------------------
    # E. Proven memory dependence relations.
    #
    # Find nearest Lk loop action owning each memory
    # operation and project op->op dependence to Lk->Lk.
    # --------------------------------------------------

    def nearest_loop_owner(
        op_node,
    ):
        candidates = []

        for slot, distances in (
            loop_distances.items()
        ):
            distance = (
                distances.get(
                    op_node
                )
            )

            if distance is None:
                continue

            anchor = loop_slots[
                slot
            ]

            depth = int(
                graph.nodes[
                    anchor
                ].get(
                    "loop_depth",
                    0,
                )
            )

            candidates.append(
                (
                    -depth,
                    distance,
                    slot,
                )
            )

        if not candidates:
            return None

        candidates.sort()

        return candidates[0][2]

    for source, target, attrs in (
        graph.edges(
            data=True
        )
    ):
        if (
            int(
                attrs.get(
                    "flow",
                    -1,
                )
            )
            != FLOW_MEMORY_DEPENDENCE
        ):
            continue

        if (
            str(
                attrs.get(
                    "certainty",
                    "",
                )
            )
            != "proven"
        ):
            # Do not convert uncertain dependence
            # summaries into direct action relations.
            continue

        source_slot = (
            nearest_loop_owner(
                source
            )
        )

        target_slot = (
            nearest_loop_owner(
                target
            )
        )

        if (
            source_slot is None
            or target_slot is None
            or source_slot
            == target_slot
        ):
            continue

        relation_bits[
            source_slot,
            target_slot,
        ] |= REL_DEP_FORWARD

        relation_bits[
            target_slot,
            source_slot,
        ] |= REL_DEP_REVERSE

    # --------------------------------------------------
    # Final boolean visibility mask.
    # --------------------------------------------------

    relation_mask = (
        relation_bits != 0
    )

    active_pairs = (
        active_mask[:, None]
        &
        active_mask[None, :]
    )

    relation_mask &= (
        active_pairs
    )

    relation_bits = (
        relation_bits
        * active_pairs.to(
            relation_bits.dtype
        )
    )

    # Every active action must at minimum see itself.
    for slot in torch.where(
        active_mask
    )[0].tolist():
        if not bool(
            relation_mask[
                slot,
                slot,
            ]
        ):
            raise RuntimeError(
                f"L{slot + 1}: "
                "relation mask lost self edge"
            )

    # --------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------

    active_slots = torch.where(
        active_mask
    )[0]

    degrees = (
        relation_mask[
            active_slots
        ]
        .sum(dim=1)
        .float()
    )

    stats = {
        "active_actions":
            int(
                active_slots.numel()
            ),

        "relation_edges":
            int(
                relation_mask.sum()
                .item()
            ),

        "mean_keys_per_action":
            float(
                degrees.mean()
                .item()
            )
            if degrees.numel()
            else 0.0,

        "max_keys_per_action":
            int(
                degrees.max()
                .item()
            )
            if degrees.numel()
            else 0,

        "multi_key_action_fraction":
            float(
                (
                    degrees > 1
                )
                .float()
                .mean()
                .item()
            )
            if degrees.numel()
            else 0.0,
    }

    return (
        relation_mask,
        relation_bits,
        stats,
    )



def _disable_pragma_conditioning(data):
    """
    Disable all pragma-conditioned updates used by the fixed NPT path.
    """
    for name in (
        "X_pragmascopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
    ):
        if hasattr(data, name):
            setattr(data, name, torch.zeros_like(getattr(data, name)))

    # Extra safety: neutralize value tensors too
    if hasattr(data, "X_pragma_per_node"):
        data.X_pragma_per_node = torch.zeros_like(data.X_pragma_per_node)
    if hasattr(data, "pragmas"):
        data.pragmas = torch.zeros_like(data.pragmas)

    return data


STATIC_GRAPH_REQUIRED = (
    "x",
    "edge_index",
    "edge_attr",
    "X_contextnids",
    "X_pragmanids",
    "X_pragmascopenids",
    "X_pseudonids",
    "X_arrayscopenids",
    "X_pipeline_scopeids",
    "X_unroll_scopeids",
    "X_array_partition_scopeids",
    "X_scopenids",
    "X_icmpnids",
)


def _load_static_mlir_graph(path):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # Current MLIR_dataset/all_kernels/graphs/*.pt files are dictionaries.
    if not isinstance(payload, dict):
        raise TypeError(
            f"{path}: expected current MLIR static graph dictionary, "
            f"got {type(payload)}"
        )

    missing = [
        name for name in STATIC_GRAPH_REQUIRED
        if name not in payload
    ]
    if missing:
        raise RuntimeError(
            f"{path}: missing static MLIR fields: {missing}"
        )

    tensor_fields = {
        key: value
        for key, value in payload.items()
        if torch.is_tensor(value)
    }

    data = Data(**tensor_fields)

    # IMPORTANT: reproduce mlir_data.MyOwnDataset.get().
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()

    data.kernel = str(
        payload.get(
            "kernel_name",
            os.path.splitext(os.path.basename(path))[0],
        )
    )
    data.graph_name = str(
        payload.get(
            "graph_name",
            os.path.splitext(os.path.basename(path))[0],
        )
    )

    n = int(data.x.size(0))

    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise RuntimeError(
            f"{path}: invalid edge_index shape "
            f"{tuple(data.edge_index.shape)}"
        )

    if data.edge_attr.size(0) != data.edge_index.size(1):
        raise RuntimeError(
            f"{path}: edge_index/edge_attr mismatch: "
            f"E={data.edge_index.size(1)} vs "
            f"edge_attr={data.edge_attr.size(0)}"
        )

    # Every node-level mask must align exactly with x.
    node_fields = (
        "X_contextnids",
        "X_pragmanids",
        "X_pragmascopenids",
        "X_pseudonids",
        "X_arrayscopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
        "X_scopenids",
        "X_icmpnids",
    )

    for name in node_fields:
        tensor = getattr(data, name)
        if tensor.numel() != n:
            raise RuntimeError(
                f"{path}: {name} has {tensor.numel()} entries, "
                f"expected {n}"
            )

    if not torch.isfinite(data.x).all():
        raise RuntimeError(f"{path}: non-finite node features")
    if not torch.isfinite(data.edge_attr).all():
        raise RuntimeError(f"{path}: non-finite edge features")

    # Needed only if you ever use current_zero_scope_post_npt.
    # static_pre_npt returns before the pragma MLPs, but keeping this makes
    # both exporter modes safe.
    data.X_pragma_per_node = torch.zeros(
        (n, 5),
        dtype=torch.float32,
    )

    return data, payload


ACTION_ID_RE = re.compile(r"^L([1-9][0-9]*)$")

NODE_TYPE_PRAGMA = 100
NODE_TYPE_ARRAY_SCOPE = 104


def _action_label_id(value):
    match = ACTION_ID_RE.fullmatch(str(value or "").strip())
    return int(match.group(1)) if match else None


def _build_llm_slots_from_gexf(
    data,
    graph,
    *,
    max_slots,
):
    n = int(data.x.size(0))

    # mlir_data.py builds tensor row i from numeric GEXF node i.
    numeric_ids = sorted(int(node) for node in graph.nodes())
    expected_ids = list(range(n))

    if numeric_ids != expected_ids:
        raise RuntimeError(
            "GEXF/static-PT node ordering mismatch: "
            f"GEXF IDs are not exactly 0..{n - 1}"
        )

    scope = torch.zeros(n, dtype=torch.bool)
    category = torch.zeros(n, dtype=torch.long)
    labels = torch.full((n,), -1, dtype=torch.long)

    # Every real MailoHLS action has at least one pragma placeholder node.
    declared_labels = set()

    for _, attrs in graph.nodes(data=True):
        if int(attrs.get("type", -1)) != NODE_TYPE_PRAGMA:
            continue

        lid = _action_label_id(attrs.get("action_id"))
        if lid is not None:
            declared_labels.add(lid)

    if not declared_labels:
        raise RuntimeError(
            "GEXF contains no MailoHLS Lk pragma actions."
        )

    if max(declared_labels) > max_slots:
        raise RuntimeError(
            f"GEXF contains L{max(declared_labels)}, but "
            f"--max_slots={max_slots}"
        )

    used_labels = {}

    for node_id, attrs in graph.nodes(data=True):
        lid = _action_label_id(attrs.get("action_id"))
        if lid is None:
            continue

        node_idx = int(node_id)
        node_type = int(attrs.get("type", -1))
        is_loop = int(attrs.get("is_loop", 0)) == 1

        # ----------------------------------------------------------
        # Loop action:
        # semantic anchor is the actual MLIR affine/scf loop op.
        # ----------------------------------------------------------
        if is_loop:
            has_loop_scope = (
                bool(data.X_pipeline_scopeids[node_idx].item())
                or bool(data.X_unroll_scopeids[node_idx].item())
            )

            if not has_loop_scope:
                raise RuntimeError(
                    f"L{lid}: loop semantic anchor node {node_idx} "
                    "is not marked as PIPELINE/UNROLL scope in the "
                    "saved MLIR static tensors"
                )

            slot_cat = 1

        # ----------------------------------------------------------
        # Array action:
        # semantic anchor is the explicit array_scope node.
        # ----------------------------------------------------------
        elif node_type == NODE_TYPE_ARRAY_SCOPE:
            has_array_scope = bool(
                data.X_array_partition_scopeids[node_idx].item()
            )

            if not has_array_scope:
                raise RuntimeError(
                    f"L{lid}: array_scope node {node_idx} is not marked "
                    "as ARRAY_PARTITION scope in the saved MLIR tensors"
                )

            if not bool(data.X_arrayscopenids[node_idx].item()):
                raise RuntimeError(
                    f"L{lid}: node {node_idx} claims to be an array action "
                    "but X_arrayscopenids is false"
                )

            slot_cat = 2

        # Ignore pragma nodes and any other nodes carrying the same action_id.
        else:
            continue

        if lid in used_labels:
            raise RuntimeError(
                f"L{lid} maps to multiple semantic anchors: "
                f"{used_labels[lid]} and {node_idx}"
            )

        used_labels[lid] = node_idx

        scope[node_idx] = True
        category[node_idx] = slot_cat
        labels[node_idx] = lid

    mapped_labels = set(used_labels)

    if mapped_labels != declared_labels:
        missing = sorted(declared_labels - mapped_labels)
        extra = sorted(mapped_labels - declared_labels)

        raise RuntimeError(
            "MailoHLS action-to-structural-slot mapping is incomplete: "
            f"missing={missing}, extra={extra}"
        )

    data.X_llm_scopeids = scope
    data.X_llm_scopecat = category
    data.X_llm_labelid = labels

    return data


def _load_and_verify_matching_gexf(
    gexf_path,
    static_payload,
    expected_num_nodes,
):
    import networkx as nx
    from mlir_data import _require_compiler_analyzed_graph

    graph = nx.read_gexf(gexf_path)

    metadata = _require_compiler_analyzed_graph(
        graph,
        os.path.basename(gexf_path),
    )

    numeric_ids = sorted(int(node) for node in graph.nodes())

    if numeric_ids != list(range(expected_num_nodes)):
        raise RuntimeError(
            f"{gexf_path}: expected node IDs 0.."
            f"{expected_num_nodes - 1}, got incompatible IDs"
        )

    saved_provenance = static_payload.get(
        "graph_provenance",
        {},
    )

    if not saved_provenance:
        raise RuntimeError(
            f"{gexf_path}: corresponding static .pt has no "
            "graph_provenance"
        )

    mismatches = {}

    for key, expected in saved_provenance.items():
        actual = metadata.get(key)

        if actual != expected:
            mismatches[key] = {
                "static_pt": expected,
                "gexf": actual,
            }

    if mismatches:
        raise RuntimeError(
            f"{gexf_path}: GEXF does not match the static MLIR "
            f"tensor pack. Provenance differences:\n"
            f"{json.dumps(mismatches, indent=2, sort_keys=True)}"
        )

    return graph


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_pt_manifest_sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: os.path.basename(item)):
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _ordered_kernel_list_sha256(kernels) -> str:
    return hashlib.sha256("\n".join(sorted(kernels)).encode("utf-8")).hexdigest()


def _load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _verify_hash(path, expected, label):
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f'{label} SHA256 mismatch: actual={actual}, expected={expected}'
        )


def _checkpoint_state(payload):
    if isinstance(payload, dict) and 'model' in payload:
        return payload['model']
    if isinstance(payload, dict) and 'state_dict' in payload:
        return payload['state_dict']
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def main():
    args = BUILD_ARGS
    if args.layerwise_out and args.embedding_mode != "static_pre_npt":
        raise ValueError("--layerwise_out requires --embedding_mode static_pre_npt")

    contract = _load_json(args.checkpoint_contract)
    sidecar = _load_json(args.checkpoint_sidecar)
    if contract.get('format') != 'mailohls-gnn-checkpoint-v1':
        raise RuntimeError('Unsupported GNN checkpoint contract format.')
    if contract.get('provenance_status') not in {
        'captured_at_training', 'reconstructed'
    }:
        raise RuntimeError('Invalid GNN provenance_status.')
    if sidecar.get('provenance_status') != contract['provenance_status']:
        raise RuntimeError('Checkpoint and contract provenance status differ.')
    _verify_hash(
        args.checkpoint_contract,
        sidecar['contract_sha256'],
        'checkpoint contract',
    )
    _verify_hash(args.ckpt, sidecar['checkpoint_sha256'], 'checkpoint')
    _verify_hash(
        args.feature_schema,
        contract['feature_schema']['sha256'],
        'feature schema',
    )

    from config import FLAGS
    for name, value in contract['model_init_flags'].items():
        if not hasattr(FLAGS, name):
            raise RuntimeError(f'Unknown saved model-init flag: {name}')
        setattr(FLAGS, name, value)
    from model import Net

    os.makedirs(args.out, exist_ok=True)

    pt_files = sorted(
        glob.glob(join(args.pt_path, "*.pt")),
        key=os.path.basename,
    )
    if not pt_files:
        print(f"No files found in {args.pt_path}")
        return
    
    first_pt, _ = _load_static_mlir_graph(pt_files[0])

    num_features = first_pt.x.size(-1)

    edge_dim = (
        first_pt.edge_attr.size(-1)
        if first_pt.edge_attr is not None
        else 0
    )
    model_init = contract['model_init']
    feature_schema = _load_json(args.feature_schema)
    if (
        feature_schema.get('feature_schema_version')
        != contract['feature_schema'].get('version')
    ):
        raise RuntimeError('Feature-schema version disagrees with the contract.')
    if int(feature_schema.get('node_feature_dim', -1)) != int(
        model_init['in_channels']
    ) or int(feature_schema.get('edge_feature_dim', -1)) != int(
        model_init['edge_dim']
    ):
        raise RuntimeError(
            'Feature-schema tensor dimensions disagree with model_init.'
        )
    if num_features != int(model_init['in_channels']):
        raise RuntimeError(
            f'Node feature dimension mismatch: {num_features} != '
            f"{model_init['in_channels']}"
        )
    if edge_dim != int(model_init['edge_dim']):
        raise RuntimeError(
            f'Edge feature dimension mismatch: {edge_dim} != '
            f"{model_init['edge_dim']}"
        )

    resource_stats = (
        contract.get('resource_stats')
        if model_init.get('resource_aux_heads', False) else None
    )
    model = Net(
        num_features,
        edge_dim=edge_dim,
        init_pragma_dict=None,
        target_stats=contract.get('target_stats'),
        resource_stats=resource_stats,
    ).to(FLAGS.device)
    payload = torch.load(args.ckpt, map_location=FLAGS.device)
    state = _checkpoint_state(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)
    source_pt_manifest_sha256 = _source_pt_manifest_sha256(pt_files)
    source_gexf_files = [
        join(args.gexf_dir, f"{os.path.splitext(basename(path))[0]}.gexf")
        for path in pt_files
    ]
    missing_gexf = [path for path in source_gexf_files if not os.path.isfile(path)]
    if missing_gexf:
        raise FileNotFoundError("Static graph files are missing: " + ", ".join(missing_gexf[:12]))
    source_gexf_manifest_sha256 = _source_pt_manifest_sha256(source_gexf_files)
    git_commit = _git_commit()
    contract_sha256 = _sha256(args.checkpoint_contract)
    feature_schema_sha256 = _sha256(args.feature_schema)

    print(f"Starting processing of {len(pt_files)} files...")
    
    written_kernels = []
    layerwise_kernels = {}
    layerwise_dims = {}

    for kernel_path in pt_files:
        fname = basename(kernel_path)
        base_name = os.path.splitext(fname)[0]

        gexf_path = join(
            args.gexf_dir,
            f"{base_name}.gexf",
        )

        if not os.path.isfile(gexf_path):
            raise RuntimeError(
                f"{fname}: matching GEXF not found: {gexf_path}"
            )

        output_path = join(
            args.out,
            f"{base_name}.memory.pt",
        )

        try:
            # ------------------------------------------------------
            # 1. Load the exact static MLIR tensors used by the GNN.
            # ------------------------------------------------------
            pt_point, static_payload = _load_static_mlir_graph(
                kernel_path
            )

            # ------------------------------------------------------
            # 2. Load the corresponding GEXF and make sure that the
            #    GEXF and .pt describe exactly the same graph.
            # ------------------------------------------------------
            graph = _load_and_verify_matching_gexf(
                gexf_path,
                static_payload,
                expected_num_nodes=pt_point.x.size(0),
            )

            # ------------------------------------------------------
            # 3. Derive deterministic Lk -> structural-node mapping.
            # ------------------------------------------------------
            pt_point = _build_llm_slots_from_gexf(
                pt_point,
                graph,
                max_slots=args.max_slots,
            )

            # ------------------------------------------------------
            # 4. Sanity-check feature dimensions against checkpoint.
            # ------------------------------------------------------
            if pt_point.x.size(-1) != num_features:
                raise RuntimeError(
                    f"{fname}: inconsistent node feature dimension: "
                    f"{pt_point.x.size(-1)} != {num_features}"
                )

            point_edge_dim = (
                pt_point.edge_attr.size(-1)
                if getattr(pt_point, "edge_attr", None) is not None
                else 0
            )

            if point_edge_dim != edge_dim:
                raise RuntimeError(
                    f"{fname}: inconsistent edge feature dimension: "
                    f"{point_edge_dim} != {edge_dim}"
                )

            required = [
                "X_pipeline_scopeids",
                "X_unroll_scopeids",
                "X_array_partition_scopeids",
                "X_arrayscopenids",
                "X_llm_scopeids",
                "X_llm_scopecat",
                "X_llm_labelid",
            ]

            missing = [
                name
                for name in required
                if not hasattr(pt_point, name)
            ]

            if missing:
                raise RuntimeError(
                    f"{fname}: missing required structural fields: {missing}"
                )

            # ------------------------------------------------------
            # 5. Build a single-graph PyG batch.
            # ------------------------------------------------------
            batch = Batch.from_data_list(
                [pt_point]
            ).to(FLAGS.device)

            # Extra leakage protection.
            batch = _disable_pragma_conditioning(batch)

            # ------------------------------------------------------
            # 6. Get structural node embeddings.
            #
            # static_pre_npt is the recommended Stage-2 representation:
            # it returns before pragma-specific NPT/MLP conditioning.
            # ------------------------------------------------------
            layerwise_node_emb = None
            if args.embedding_mode == "current_zero_scope_post_npt":
                node_emb = model.forward_node_embed(batch)

            elif args.embedding_mode == "static_pre_npt":
                node_emb = model.forward_static_node_embed(batch)
                layerwise_node_emb = None

                if args.layerwise_out:
                    layerwise_node_emb = (
                        model.forward_static_node_embed_layers(
                            batch
                        )
                    )
                    required_layers = {"jkn", "conv_1"}
                    missing_layers = required_layers - set(layerwise_node_emb)
                    if missing_layers:
                        raise RuntimeError(
                            "Layerwise export must produce JKN and conv_1 together; missing "
                            + ", ".join(sorted(missing_layers))
                        )

                    torch.testing.assert_close(
                        layerwise_node_emb["jkn"],
                        node_emb,
                        rtol=1e-5,
                        atol=1e-6,
                    )

            else:
                raise AssertionError(args.embedding_mode)

            # ------------------------------------------------------
            # 7. Select the semantic node associated with each Lk.
            # ------------------------------------------------------
            scope = batch.X_llm_scopeids.bool()
            label = batch.X_llm_labelid.long()

            sel = (
                scope
                & (label > 0)
                & (label <= args.max_slots)
            )

            sel_idx = sel.nonzero(
                as_tuple=False
            ).view(-1)

            if sel_idx.numel() == 0:
                raise RuntimeError(
                    f"{fname}: no MailoHLS structural slots "
                    "were identified"
                )

            expected_slots = int(
                batch.X_llm_scopeids.bool().sum().item()
            )

            if sel_idx.numel() != expected_slots:
                raise RuntimeError(
                    f"{fname}: slot filtering dropped labels: "
                    f"mapped={expected_slots}, "
                    f"retained={sel_idx.numel()}, "
                    f"max_slots={args.max_slots}"
                )

            # ------------------------------------------------------
            # 8. Build fixed [max_slots, D] memory.
            # ------------------------------------------------------
            node_embs = torch.zeros(
                (
                    args.max_slots,
                    node_emb.size(-1),
                ),
                dtype=node_emb.dtype,
                device=node_emb.device,
            )

            node_embs_mask = torch.zeros(
                (args.max_slots,),
                dtype=torch.bool,
                device=node_emb.device,
            )

            slot_cats = torch.zeros(
                (args.max_slots,),
                dtype=torch.long,
                device=node_emb.device,
            )

            node_ids = [-1] * args.max_slots
            labels = [-1] * args.max_slots

            for ni in sel_idx.tolist():
                lid = int(label[ni].item())
                slot = lid - 1

                if node_embs_mask[slot]:
                    raise RuntimeError(
                        f"{fname}: duplicate structural mapping "
                        f"for L{lid}"
                    )

                node_embs[slot] = node_emb[ni]
                node_embs_mask[slot] = True
                node_ids[slot] = ni
                labels[slot] = lid
                slot_cats[slot] = int(
                    batch.X_llm_scopecat[ni].item()
                )

            # ------------------------------------------------------
            # 9. Move exported structural memory to CPU.
            # ------------------------------------------------------
            node_embs = node_embs.detach().cpu()
            node_embs_mask = node_embs_mask.detach().cpu()
            slot_cats = slot_cats.detach().cpu()

            node_embs = torch.nan_to_num(
                node_embs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Clip only exceptionally large vector norms.
            max_norm = 20.0
            eps = 1e-6

            norms = node_embs.norm(
                p=2,
                dim=1,
                keepdim=True,
            ).clamp(min=eps)

            scale = (
                max_norm / norms
            ).clamp(max=1.0)

            node_embs = node_embs * scale

            if not torch.isfinite(node_embs).all():
                raise RuntimeError(
                    f"{fname}: non-finite structural embeddings "
                    "after sanitization"
                )

            # ------------------------------------------------------
            # 10. Save the Stage-2 structural-memory pack.
            # ------------------------------------------------------
            action_relation_mask, action_relation_bits, action_relation_stats = build_action_relation_mask(
                graph,
                node_ids=node_ids,
                active_mask=node_embs_mask,
                max_slots=args.max_slots,
            )

            pack = {
                "pt_path": kernel_path,
                "gexf_path": gexf_path,

                "ckpt": args.ckpt,
                "embedding_mode": args.embedding_mode,
                "disable_pragma_injection": True,

                "gnn_checkpoint_sha256":
                    checkpoint_sha256,

                "gnn_contract_sha256":
                    contract_sha256,

                "feature_schema_sha256":
                    feature_schema_sha256,

                "source_pt_manifest_sha256":
                    source_pt_manifest_sha256,

                "source_gexf_manifest_sha256":
                    source_gexf_manifest_sha256,

                "source_gexf_sha256":
                    _sha256(gexf_path),

                "git_commit": git_commit,

                "gnn_dim":
                    int(node_embs.size(-1)),

                "node_embs":
                    node_embs,

                "node_embs_mask":
                    node_embs_mask,

                "max_slots":
                    args.max_slots,

                "slot_ids":
                    torch.arange(
                        1,
                        args.max_slots + 1,
                        dtype=torch.long,
                    ),

                "slot_cats":
                    slot_cats,

                "node_ids":
                    node_ids,

                "labels":
                    labels,

                "action_relation_schema":
                    ACTION_RELATION_SCHEMA,

                "action_relation_mask":
                    action_relation_mask,

                "action_relation_bits":
                    action_relation_bits,

                "action_relation_stats":
                    action_relation_stats,
            }

            torch.save(
                pack,
                output_path,
            )

            # ------------------------------------------------------
            # 11. Optionally export every pre-NPT GNN representation
            #     for layerwise action-embedding diagnostics.
            # ------------------------------------------------------
            if layerwise_node_emb is not None:

                for layer_name, node_tensor in (
                    layerwise_node_emb.items()
                ):

                    layer_slots = torch.zeros(
                        (
                            args.max_slots,
                            node_tensor.size(-1),
                        ),
                        dtype=node_tensor.dtype,
                        device=node_tensor.device,
                    )

                    for ni in sel_idx.tolist():
                        lid = int(
                            label[ni].item()
                        )

                        layer_slots[
                            lid - 1
                        ] = node_tensor[ni]

                    # Same CPU / finite-value sanitization
                    # used for the production memory bank.
                    layer_slots = (
                        layer_slots
                        .detach()
                        .cpu()
                    )

                    layer_slots = torch.nan_to_num(
                        layer_slots,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    # Match the production memory export's
                    # exceptional-vector clipping.
                    max_norm = 20.0
                    eps = 1e-6

                    norms = (
                        layer_slots
                        .norm(
                            p=2,
                            dim=1,
                            keepdim=True,
                        )
                        .clamp(min=eps)
                    )

                    scale = (
                        max_norm / norms
                    ).clamp(max=1.0)

                    layer_slots = (
                        layer_slots * scale
                    )

                    layer_dir = os.path.join(
                        args.layerwise_out,
                        layer_name,
                    )

                    os.makedirs(
                        layer_dir,
                        exist_ok=True,
                    )

                    # Preserve exactly the same:
                    #   mask
                    #   labels
                    #   slot IDs
                    #   categories
                    #   provenance
                    # as the production JKN memory pack.
                    layer_pack = dict(pack)

                    layer_pack[
                        "node_embs"
                    ] = layer_slots

                    layer_pack[
                        "gnn_dim"
                    ] = int(
                        layer_slots.size(-1)
                    )

                    layer_pack[
                        "embedding_mode"
                    ] = (
                        "static_pre_npt::"
                        + layer_name
                    )

                    layer_pack[
                        "layerwise_diagnostic"
                    ] = True

                    torch.save(
                        layer_pack,
                        os.path.join(
                            layer_dir,
                            f"{base_name}.memory.pt",
                        ),
                    )
                    layerwise_kernels.setdefault(layer_name, []).append(base_name)
                    layerwise_dims[layer_name] = int(layer_slots.size(-1))

            written_kernels.append(
                base_name
            )

            print(
                f"[OK] {fname}: "
                f"memory={tuple(node_embs.shape)}, "
                f"active_slots={int(node_embs_mask.sum())}"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to process {fname}: {exc}"
            ) from exc

    if len(written_kernels) != len(pt_files):
        raise RuntimeError(
            "Structural-memory export incomplete: "
            f"wrote {len(written_kernels)} memories from "
            f"{len(pt_files)} static MLIR graphs"
        )


    print(
        f"[DONE] Exported {len(written_kernels)} "
        f"structural memory files."
    )

    bank_manifest = {
        'schema': 'mailohls-memory-bank-manifest-v2',
        'gnn_contract_sha256': contract_sha256,
        'feature_schema_sha256': feature_schema_sha256,
        'gnn_checkpoint_sha256': checkpoint_sha256,
        'source_pt_manifest_sha256': source_pt_manifest_sha256,
        'source_gexf_manifest_sha256': source_gexf_manifest_sha256,
        'embedding_mode': args.embedding_mode,
        'exporter_git_commit': git_commit,
        'checkpoint_tag': sidecar.get('checkpoint_tag'),
        'checkpoint_epoch': sidecar.get('checkpoint_epoch'),
        'provenance_status': contract['provenance_status'],
        'kernel_count': len(written_kernels),
        'ordered_kernel_list_sha256': _ordered_kernel_list_sha256(written_kernels),
        'action_slot_schema': 'absolute-lk-v1',
        'gnn_dim': int(node_embs.size(-1)),
    }

    bank_manifest[
        "action_relation_schema"
    ] = ACTION_RELATION_SCHEMA

    bank_manifest[
        "action_relation_policy"
    ] = (
        "self+nearest-loop-hierarchy+"
        "array-access+proven-memory-dependence"
    )
    
    with open(join(args.out, 'memory_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(bank_manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')

    for layer_name, kernels in sorted(layerwise_kernels.items()):
        if sorted(kernels) != sorted(written_kernels):
            raise RuntimeError(f"Layer {layer_name} was not exported for the same kernel set")
        layer_dir = join(args.layerwise_out, layer_name)
        actual_files = {basename(path) for path in glob.glob(join(layer_dir, "*.memory.pt"))}
        expected_files = {f"{kernel}.memory.pt" for kernel in written_kernels}
        if actual_files != expected_files:
            raise RuntimeError(
                f"Layer {layer_name} contains stale/missing memory files: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"unexpected={sorted(actual_files - expected_files)}"
            )
        layer_manifest = dict(bank_manifest)
        layer_manifest.update({
            "embedding_mode": f"static_pre_npt::{layer_name}",
            "layer_name": layer_name,
            "gnn_dim": layerwise_dims[layer_name],
        })
        with open(join(layer_dir, "memory_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(layer_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[LAYER-MANIFEST] {layer_name}: kernels={len(kernels)} dim={layerwise_dims[layer_name]}")



if __name__ == "__main__":
    main()
