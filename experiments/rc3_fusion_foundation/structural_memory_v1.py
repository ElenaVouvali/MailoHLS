"""Frozen V4b GNN outputs exposed through three versioned memory interfaces."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from experiments.v22_gnn_training.v4b.models import V4bGraphModel
from .rc3_action_relations import build_action_relations

ACTION_SLOTS = 64
# RC3 GEMV has 82 typed node/edge records.  Capacity is explicitly versioned
# at 128; larger graphs fail closed and must be assigned a new capacity/schema.
EVIDENCE_CAPACITY = 128
EVIDENCE_SCHEMA = 'salt-hls-rc3-action-evidence-memory-v1'
KNOWN_FIELDS = ('class', 'proven', 'shape', 'trip_count', 'affine_map',
                'access_role', 'operation_family', 'loop_carried', 'dependence_kind')
EVIDENCE_TYPES = (
    'array_memory_root', 'array_ssa_root', 'loop_node', 'control_region',
    'affine_memory_access', 'arithmetic_operation', 'loop_carried_ssa_value',
    'memory_access_witness', 'loop_carried_witness', 'dependence_witness',
    'loop_hierarchy_witness', 'operand_witness', 'definition_witness',
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_encoder(checkpoint_path, expected_sha256):
    checkpoint_path = Path(checkpoint_path)
    actual = sha256(checkpoint_path)
    if actual != expected_sha256:
        raise ValueError(f'V4b checkpoint SHA mismatch: {actual}')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if state.get('arm') != 'compiler' or state.get('selected_step') != 1500:
        raise ValueError('Expected the authenticated corrected-v3 compiler-only selected checkpoint')
    if state.get('quality_policy') != 'reported-zero-adp-ambiguous-v1':
        raise ValueError('Unexpected V4b training policy')
    model = V4bGraphModel(hardware_messages=False)
    model.load_state_dict(state['state'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': actual,
        'arm': state['arm'], 'selected_step': state['selected_step'],
        'updates': state['updates'], 'schema': state['schema'],
        'quality_policy': state['quality_policy'],
        'protocol_fingerprint': state['protocol_fingerprint'],
        'objective_caveat': 'V4b was trained with physical-QoR/ADP-oriented preference supervision; not latency-qualified.',
        'encoder': 'same frozen two-layer V4b compiler-only encoder for A/B/C',
    }


def _target_graph(program_graph, *, target='u200', clock_ns=10.0):
    registry = overlay.load_registry()
    joint = overlay.enrich(program_graph, target, clock_ns, registry=registry)
    data = tensors.tensorize(joint, max_slots=ACTION_SLOTS)
    return joint, data


def _source_edge_tensor_index(joint, data, source, target, edge_key, relation):
    nodes = {node: i for i, node in enumerate(data.audit['node_ids'])}
    wanted_source, wanted_target = nodes[source], nodes[target]
    wanted_type = tensors.RELATIONS.index(relation)
    matching = []
    for i, key in enumerate(data.audit['source_edge_keys']):
        if str(key) != str(edge_key):
            continue
        if int(data.edge_index[0, i]) == wanted_source and int(data.edge_index[1, i]) == wanted_target \
                and int(data.edge_type[i]) == wanted_type:
            matching.append(i)
    if len(matching) != 1:
        raise ValueError(f'Compiler edge has {len(matching)} tensor witnesses: {source}->{target}/{edge_key}/{relation}')
    return matching[0]


def _evidence(graph, data, node_h, model, cap=EVIDENCE_CAPACITY):
    rows = []
    edges_by_relation = defaultdict(list)
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        if attrs.get('evidence_status') == 'proven':
            edges_by_relation[attrs['relation_type']].append((u, v, str(key), attrs))

    add_nodes = set()
    for node, attrs in graph.nodes(data=True):
        category = attrs.get('category')
        if category == 'memory_object':
            add_nodes.add((node, 'array_memory_root'))
        if category == 'ssa_value' and attrs.get('is_memory_root'):
            add_nodes.add((node, 'array_ssa_root'))
        if category == 'operation' and attrs.get('is_loop'):
            add_nodes.add((node, 'loop_node'))
        if category == 'control_region':
            add_nodes.add((node, 'control_region'))
        if category == 'operation' and attrs.get('operation_family') == 'arithmetic':
            add_nodes.add((node, 'arithmetic_operation'))

    access_ops = set()
    for u, v, key, attrs in edges_by_relation['memory_access']:
        indices = attrs.get('access_indices') or {}
        if indices.get('status') == 'compiler_recorded' and indices.get('affine_map'):
            access_ops.add(u)
    add_nodes.update((node, 'affine_memory_access') for node in access_ops)
    carried_values = set()
    for u, v, _, _ in edges_by_relation['loop_carried']:
        carried_values.update((u, v))
    add_nodes.update((node, 'loop_carried_ssa_value') for node in carried_values
                     if graph.nodes[node].get('category') == 'ssa_value')

    # Node selection is deterministic by semantic type then canonical graph ID.
    type_order = {name: i for i, name in enumerate(EVIDENCE_TYPES)}
    for node, evidence_type in sorted(add_nodes, key=lambda x: (type_order[x[1]], x[0])):
        attrs = graph.nodes[node]
        idx = data.audit['node_ids'].index(node)
        row = {
            'key': f'node:{evidence_type}:{node}', 'evidence_type': evidence_type,
            'kind': 'node', 'node_id': node,
            'evidence_status': attrs.get('root_resolution', 'proven'),
            'known': {
                'class': True,
                'proven': attrs.get('root_resolution', 'proven') == 'proven',
                'shape': attrs.get('memory_shape') is not None,
                'trip_count': attrs.get('trip_count_static') == 1,
                'affine_map': False, 'access_role': False,
                'operation_family': attrs.get('operation_family') is not None,
                'loop_carried': False, 'dependence_kind': False,
            },
            'provenance': {'graph_identity': graph.graph['graph_identity'], 'node_id': node},
            '_vector': node_h[idx].detach().float().cpu(),
        }
        rows.append(row)

    edge_classes = {
        'memory_access': 'memory_access_witness',
        'loop_carried': 'loop_carried_witness',
        'depends_on': 'dependence_witness',
        'loop_child': 'loop_hierarchy_witness',
    }
    # Preserve arithmetic operand/definition links only when both endpoints
    # are selected evidence nodes. No free-form edge IDs become learned inputs.
    selected_node_ids = {node for node, _ in add_nodes}
    for relation in ('memory_access', 'loop_carried', 'depends_on', 'loop_child', 'operand', 'defines'):
        for u, v, key, attrs in sorted(edges_by_relation[relation], key=lambda x: (x[0], x[1], x[2])):
            if relation in ('operand', 'defines') and not (u in selected_node_ids and v in selected_node_ids):
                continue
            edge_type = edge_classes.get(relation, 'operand_witness' if relation == 'operand' else 'definition_witness')
            pos = _source_edge_tensor_index(graph, data, u, v, key, relation)
            src_i = data.audit['node_ids'].index(u)
            dst_i = data.audit['node_ids'].index(v)
            # Explicit frozen endpoint/edge representation: mean endpoint
            # states, plus frozen type and edge-feature projections in two
            # fixed 16D slices. This is deterministic and trains no projection.
            vector = 0.5 * (node_h[src_i].detach().float().cpu() + node_h[dst_i].detach().float().cpu())
            with torch.inference_mode():
                edge_type_id = data.edge_type[pos:pos + 1]
                edge_attr = data.edge_attr[pos:pos + 1]
                relation_vec = model.graph.relation(edge_type_id).float().cpu()[0]
                feature_vec = model.graph.edge_input(edge_attr).float().cpu()[0]
            vector[:16] += relation_vec
            vector[16:32] += feature_vec
            known = {
                'class': True, 'proven': attrs.get('evidence_status') == 'proven',
                'shape': False, 'trip_count': False,
                'affine_map': (attrs.get('access_indices') or {}).get('status') == 'compiler_recorded',
                'access_role': attrs.get('access') in ('read', 'write', 'readwrite'),
                'operation_family': False, 'loop_carried': attrs.get('loop_carried') is not None or relation == 'loop_carried',
                'dependence_kind': attrs.get('dependence_kind') is not None,
            }
            payload = {k: attrs.get(k) for k in (
                'access', 'access_indices', 'evidence_status', 'carrying_loop',
                'dependence_kind', 'loop_carried', 'source_analysis') if k in attrs}
            rows.append({
                'key': f'edge:{edge_type}:{u}:{key}:{v}', 'evidence_type': edge_type,
                'kind': 'edge', 'source_node': u, 'target_node': v, 'source_relation': relation,
                'edge_key': key, 'evidence_status': attrs.get('evidence_status', 'not_established'),
                'known': known, 'provenance': {'graph_identity': graph.graph['graph_identity'],
                                              'source_node': u, 'target_node': v,
                                              'edge_key': key, 'relation_type': relation,
                                              'compiler_payload': payload},
                '_vector': vector,
            })

    rows.sort(key=lambda r: (type_order.get(r['evidence_type'], 99), r['key']))
    if len(rows) > cap:
        raise ValueError(f'Evidence memory overflow: {len(rows)} selected records exceeds cap={cap}; truncation forbidden')
    return rows


def _action_evidence_scopes(graph, relation_pack):
    labels = [f'L{i + 1}' for i, active in enumerate(relation_pack['absolute_slots']['valid_mask']) if active]
    targets = relation_pack['compiler_target_nodes']
    relations = relation_pack['eligibility']
    scopes = {}
    all_edges = list(graph.edges(keys=True, data=True))

    def array_bundle(label, node_scope, edge_scope, access_filter=None):
        target = targets[label]['node']
        node_scope.add(target)
        for node, attrs in graph.nodes(data=True):
            evidence = attrs.get('array_binding_evidence', {})
            if attrs.get('category') == 'ssa_value' and attrs.get('is_memory_root') \
                    and evidence.get('action_id') == label:
                node_scope.add(node)
        for u, v, key, attrs in all_edges:
            if attrs['relation_type'] != 'memory_access' or attrs.get('evidence_status') != 'proven':
                continue
            if v == target and (access_filter is None or access_filter(u, attrs)):
                node_scope.update((u, v)); edge_scope.add(str(key))

    def loop_anchor_bundle(label, node_scope, edge_scope):
        target = targets[label]['node']
        node_scope.add(target)
        for u, v, key, attrs in all_edges:
            if attrs.get('evidence_status') != 'proven':
                continue
            if attrs['relation_type'] == 'owns_region' and u == target:
                node_scope.update((u, v)); edge_scope.add(str(key))
                for a, b, key2, attrs2 in all_edges:
                    if a == v and attrs2['relation_type'] == 'contains_block' and attrs2.get('evidence_status') == 'proven':
                        node_scope.update((a, b)); edge_scope.add(str(key2))

    for query in labels:
        query_slot = int(query[1:]) - 1
        related = [f'L{j + 1}' for j, allowed in enumerate(relations[query_slot]) if allowed]
        node_scope, edge_scope = set(), set()
        query_is_loop = graph.nodes[targets[query]['node']].get('is_loop')
        direct_loop_nodes = set()
        if query_is_loop:
            target = targets[query]['node']
            direct_loop_nodes.add(target)
            node_scope.add(target)
            for node, attrs in graph.nodes(data=True):
                if attrs.get('category') == 'operation' and target in attrs.get('loop_context', []):
                    node_scope.add(node)
                    if attrs.get('is_loop'):
                        direct_loop_nodes.add(node)
            loop_anchor_bundle(query, node_scope, edge_scope)
        else:
            array_bundle(query, node_scope, edge_scope)

        # Related action keys add only that action's anchor plus query-linked
        # evidence. A related outer loop does not grant an array query all
        # sibling arrays' evidence. For a loop query, arrays are restricted
        # to proven accesses in that loop's own scope.
        for related_label in related:
            related_target = targets[related_label]['node']
            related_attrs = graph.nodes[related_target]
            if related_attrs.get('is_loop'):
                loop_anchor_bundle(related_label, node_scope, edge_scope)
                hierarchy_edge = [(u, v, key, attrs) for u, v, key, attrs in all_edges
                                  if attrs['relation_type'] == 'loop_child' and attrs.get('evidence_status') == 'proven'
                                  and (u == related_target or v == related_target)]
                for u, v, key, _ in hierarchy_edge:
                    node_scope.update((u, v)); edge_scope.add(str(key))
            elif query_is_loop:
                array_bundle(related_label, node_scope, edge_scope,
                             access_filter=lambda op, _attrs: op == targets[query]['node']
                             or targets[query]['node'] in graph.nodes[op].get('loop_context', []))
            else:
                array_bundle(related_label, node_scope, edge_scope)

        # Compiler access/SSA/dependence witnesses are attached only to the
        # direct query scope, not recursively expanded through related loops.
        for u, v, key, attrs in all_edges:
            rel = attrs['relation_type']
            if rel == 'memory_access' and attrs.get('evidence_status') == 'proven':
                if u in node_scope and (v in node_scope or graph.nodes[v].get('category') == 'memory_object'):
                    node_scope.update((u, v)); edge_scope.add(str(key))
            elif rel == 'loop_carried' and attrs.get('evidence_status') == 'proven':
                if attrs.get('carrying_loop') in direct_loop_nodes:
                    node_scope.update((u, v)); edge_scope.add(str(key))
            elif rel == 'depends_on' and attrs.get('evidence_status') == 'proven':
                if u in node_scope and v in node_scope:
                    node_scope.update((u, v)); edge_scope.add(str(key))
            elif rel == 'loop_child' and attrs.get('evidence_status') == 'proven':
                if u in direct_loop_nodes or v in direct_loop_nodes:
                    node_scope.update((u, v)); edge_scope.add(str(key))
        # Keep operand/definition witnesses inside the action's independently
        # established scope. Do not expand the scope through SSA traversal: an
        # array query must not inherit an unrelated accumulator recurrence.
        for u, v, key, attrs in all_edges:
            if attrs['relation_type'] in ('operand', 'defines', 'declares_argument') \
                    and attrs.get('evidence_status') == 'proven' and u in node_scope and v in node_scope:
                edge_scope.add(str(key))
        scopes[query] = {'nodes': node_scope, 'edges': edge_scope, 'related_actions': related}
    return scopes


def pad_evidence_batch(packs, *, capacity=EVIDENCE_CAPACITY):
    """Pad already extracted single-program evidence packs without truncation."""
    if not packs:
        raise ValueError('At least one pack is required')
    sizes = [int(p['evidence_memory'].shape[1]) for p in packs]
    width = max(sizes, default=0)
    if width > capacity:
        raise ValueError(f'Evidence batch width {width} exceeds cap={capacity}')
    device = packs[0]['evidence_memory'].device
    dtype = packs[0]['evidence_memory'].dtype
    batch = len(packs)
    memory = torch.zeros(batch, width, 128, dtype=dtype, device=device)
    valid = torch.zeros(batch, width, dtype=torch.bool, device=device)
    known = torch.zeros(batch, width, len(KNOWN_FIELDS), dtype=torch.bool, device=device)
    for i, pack in enumerate(packs):
        e = sizes[i]
        if e:
            memory[i, :e] = pack['evidence_memory'][0]
            valid[i, :e] = pack['evidence_valid_mask'][0]
            known[i, :e] = pack['evidence_known_mask'][0]
    return {'evidence_memory': memory, 'evidence_valid_mask': valid,
            'evidence_known_mask': known, 'lengths': sizes,
            'padding_policy': 'right-pad false; no truncation; maximum configured evidence capacity 64'}


def build_memory_pack(program_graph, model, *, graph_file_sha256, gnn_identity,
                      target='u200', clock_ns=10.0, max_slots=ACTION_SLOTS,
                      evidence_capacity=EVIDENCE_CAPACITY, expected_graph_identity=None):
    if max_slots != ACTION_SLOTS:
        raise ValueError('The current structural memory contract is fixed at 64 absolute action slots')
    relation_pack = build_action_relations(program_graph, max_slots=max_slots,
                                           expected_graph_identity=expected_graph_identity)
    joint, data = _target_graph(program_graph, target=target, clock_ns=clock_ns)
    if data.audit['program_graph_identity'] != program_graph.graph['graph_identity']:
        raise ValueError('Tensorized program identity differs from the authenticated input graph')
    model.eval()
    with torch.inference_mode():
        action_memory, pooled = model.encode(data)
        node_hidden = model.graph(data)
    if action_memory.shape != (1, max_slots, 128) or pooled.shape != (1, 128):
        raise ValueError('Frozen V4b action-memory shape mismatch')
    masks = relation_pack['absolute_slots']
    valid = torch.tensor([masks['valid_mask']], dtype=torch.bool)
    matched = torch.tensor([masks['matched_mask']], dtype=torch.bool)
    if not torch.equal(data.matched_mask.cpu(), matched) or not torch.equal(data.registered_mask.cpu(), torch.tensor([masks['registered_mask']])):
        raise ValueError('Graph tensorizer absolute-action masks disagree with RC3 action registrations')
    action_relation = torch.tensor([relation_pack['eligibility']], dtype=torch.bool)
    action_relation &= valid[:, :, None] & valid[:, None, :]
    for slot in valid[0].nonzero().flatten().tolist():
        if not action_relation[0, slot, slot]:
            raise ValueError(f'Valid absolute slot L{slot+1} is missing its self route')

    evidence_rows = _evidence(program_graph, data, node_hidden, model, cap=evidence_capacity)
    n_evidence = len(evidence_rows)
    evidence_memory = torch.stack([r['_vector'] for r in evidence_rows])[None] if n_evidence else torch.zeros(1, 0, 128)
    evidence_valid = torch.ones(1, n_evidence, dtype=torch.bool)
    evidence_known = torch.tensor([[bool(r['known'].get(name, False)) for name in KNOWN_FIELDS]
                                   for r in evidence_rows], dtype=torch.bool).reshape(1, n_evidence, len(KNOWN_FIELDS))
    scopes = _action_evidence_scopes(program_graph, relation_pack)
    node_to_items = defaultdict(list); edge_to_items = defaultdict(list)
    for index, row in enumerate(evidence_rows):
        if row['kind'] == 'node': node_to_items[row['node_id']].append(index)
        else: edge_to_items[row['edge_key']].append(index)
    action_evidence = torch.zeros(1, max_slots, n_evidence, dtype=torch.bool)
    for label, scope in scopes.items():
        slot = int(label[1:]) - 1
        for node in scope['nodes']:
            for idx in node_to_items.get(node, []): action_evidence[0, slot, idx] = True
        for edge in scope['edges']:
            for idx in edge_to_items.get(edge, []): action_evidence[0, slot, idx] = True
    action_evidence &= valid[:, :, None] & evidence_valid[:, None, :]
    for row in evidence_rows:
        row.pop('_vector', None)

    # A/B/C share bit-identical action vectors from this one frozen forward.
    pack = {
        'schema': 'salt-hls-rc3-structural-memory-v1',
        'action_memory': action_memory.detach().float().cpu().contiguous(),
        'action_valid_mask': valid,
        'action_registered_mask': torch.tensor([masks['registered_mask']], dtype=torch.bool),
        'action_matched_mask': matched,
        'action_unresolved_mask': torch.tensor([masks['unresolved_mask']], dtype=torch.bool),
        'action_source_only_mask': torch.tensor([masks['source_only_mask']], dtype=torch.bool),
        'action_relation_mask': action_relation,
        'relation_types': relation_pack['relation_types'],
        'relation_pack': relation_pack,
        'evidence_schema': EVIDENCE_SCHEMA,
        'evidence_memory': evidence_memory.detach().float().cpu().contiguous(),
        'evidence_valid_mask': evidence_valid,
        'evidence_known_mask': evidence_known,
        'evidence_type_ids': torch.tensor([[EVIDENCE_TYPES.index(r['evidence_type']) for r in evidence_rows]], dtype=torch.long),
        'evidence_type_names': EVIDENCE_TYPES,
        'known_field_names': KNOWN_FIELDS,
        'evidence_metadata': evidence_rows,
        'action_evidence_mask': action_evidence,
        'action_evidence_scope': {label: {'related_actions': s['related_actions'],
                                          'node_count': len(s['nodes']), 'edge_count': len(s['edges'])}
                                  for label, s in scopes.items()},
        'pooled_program_memory': pooled.detach().float().cpu(),
        'provenance': {
            'graph_identity': program_graph.graph['graph_identity'], 'graph_file_sha256': graph_file_sha256,
            'program_payload_sha256': joint.graph['program_payload_sha256'],
            'overlay_identity': joint.graph['overlay_identity'], 'tensor_audit': data.audit,
            'target_profile': target, 'clock_period_ns': float(clock_ns),
            'hardware_messages': False, 'target_conditioned_embeddings': False,
            'gnn': gnn_identity,
            'memory_contract_caveat': 'V4b ADP-oriented training; frozen diagnostic only, not a latency-qualified encoder.',
        },
    }
    # Save internal tensor dimensions as part of the audit contract.
    pack['shapes'] = {key: list(value.shape) for key, value in pack.items() if torch.is_tensor(value)}
    pack['evidence_count'] = n_evidence
    pack['evidence_capacity'] = evidence_capacity
    return pack


def adapt_for_xattn(pack, mode):
    """Return existing A/B memory conventions or versioned combined C keys."""
    if mode not in ('A', 'B', 'C'):
        raise ValueError('Memory mode must be A, B, or C')
    action = pack['action_memory']
    action_mask = pack['action_valid_mask'].bool()
    if action.shape != (1, ACTION_SLOTS, 128) or action_mask.shape != (1, ACTION_SLOTS):
        raise ValueError('Action memory/mask violates [B,64,128]/[B,64] contract')
    if mode in ('A', 'B'):
        result = {'schema': 'salt-hls-xattn-action-memory-v1',
                  'memory': action, 'memory_mask': action_mask,
                  'placeholder_slot_addressing': 'one-based Lk -> memory slot k-1',
                  'routing_mode': 'exact_slot' if mode == 'A' else 'compiler_relational',
                  'action_relation_mask': None if mode == 'A' else pack['action_relation_mask'].bool(),
                  'provenance': pack['provenance']}
        if mode == 'B' and result['action_relation_mask'].shape != (1, ACTION_SLOTS, ACTION_SLOTS):
            raise ValueError('B relation mask violates [B,64,64] contract')
        return result
    evidence = pack['evidence_memory']
    evidence_mask = pack['evidence_valid_mask'].bool()
    e = evidence.shape[1]
    memory = torch.cat((action, evidence), dim=1)
    memory_mask = torch.cat((action_mask, evidence_mask), dim=1)
    slots = ACTION_SLOTS + e
    relation = torch.zeros(1, slots, slots, dtype=torch.bool)
    relation[0, :ACTION_SLOTS, :ACTION_SLOTS] = pack['action_relation_mask'][0]
    relation[0, :ACTION_SLOTS, ACTION_SLOTS:] = pack['action_evidence_mask'][0]
    # Evidence keys are values, not action-query rows; no token can route by
    # treating a compiler evidence key as a future directive/action ID.
    relation &= memory_mask[:, :, None] & memory_mask[:, None, :]
    for slot in action_mask[0].nonzero().flatten().tolist():
        if not relation[0, slot].any():
            raise ValueError(f'C valid L{slot+1} has no legal key')
    return {
        'schema': EVIDENCE_SCHEMA + '+xattn-mask-v1', 'memory': memory,
        'memory_mask': memory_mask, 'placeholder_slot_addressing': 'one-based Lk addresses row k-1; evidence slots are keys only',
        'routing_mode': 'compiler_relational', 'action_relation_mask': relation,
        'evidence_start_slot_zero_based': ACTION_SLOTS,
        'evidence_type_ids': pack['evidence_type_ids'],
        'evidence_valid_mask': evidence_mask,
        'evidence_known_mask': pack['evidence_known_mask'],
        'provenance': pack['provenance'],
    }
