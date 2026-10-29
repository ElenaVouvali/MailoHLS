"""Experimental, pragma-free V2.2 tensor schema. Never consumes V1 weights.

Only this derived view adds reverse messages. Detailed compiler witnesses stay
in the authenticated JSON; IDs below are audit/alignment metadata, not features.
Validated in the isolated PyG 2.6.1 / PyTorch 2.6 CPU environment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool

from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_v22 as overlay
from GNN_branch import action_point_relations as queries
from GNN_branch import action_demand_v22 as demand
from GNN_branch import semantic_tensor_features_v22 as semantic

LEGACY_SCHEMA = 'salt-hls-target-tensors-v2-action-demand-experimental'
SCHEMA = 'salt-hls-target-tensors-v3-semantic-experimental'
NODE_TYPES = tuple(sorted(compiler.NODE_KINDS)) + ('hardware_target', 'hardware_resource')
FORWARD_RELATIONS = tuple(sorted(compiler.EDGE_KINDS | overlay.RELATIONS))
RELATIONS = FORWARD_RELATIONS + tuple('reverse:' + r for r in FORWARD_RELATIONS)
OP_TYPES = ('none', 'other', 'function', 'loop', 'call', 'load', 'store', 'view',
            'constant', 'add', 'sub', 'mul', 'div', 'compare', 'select', 'cast',
            'yield', 'branch', 'math')
VALUE_TYPES = ('unknown', 'integer', 'float', 'index', 'memref', 'pointer', 'vector', 'aggregate')
ROOT_TYPES = ('none', 'allocation', 'global', 'formal_parameter', 'unresolved', 'other')
RESOURCE_TYPES = ('none',) + overlay.RESOURCES
EVIDENCE = ('proven', 'possible', 'not_established', 'not_applicable')
NUMERICAL = ('lowered_width', 'memory_volume', 'memory_rank', 'source_volume',
             'loop_depth', 'trip_count', 'operand_count', 'result_count', 'use_count',
             'constant', 'physical_capacity', 'available_capacity', 'available_fraction',
             'clock', 'platform_capacity', 'requested_budget', 'action_loop', 'action_array')
EDGE_NUMERICAL = ('read', 'write', 'readwrite', 'RAW', 'WAR', 'WAW',
                  'must_alias', 'may_alias', 'partial_alias', 'no_alias',
                  'root_complete', 'root_complete_known', 'loop_carried', 'loop_carried_known',
                  'argument_position', 'argument_position_known', 'concrete', 'through_call',
                  'possible_implementation', 'documented_hardware')
NODE_DIM = 2 * len(NUMERICAL)
EDGE_DIM = len(EVIDENCE) + len(EDGE_NUMERICAL)
BASE_EDGE_DIM = EDGE_DIM
SEMANTIC_EDGE_DIM = len(semantic.FEATURE_NAMES)
EDGE_DIM += SEMANTIC_EDGE_DIM
CONTEXT_DIM = 13
ACTION_DEMAND_DIM = 2 * len(demand.NUMERIC_FEATURES)


def _op(attrs):
    name = attrs.get('text', '')
    family = attrs.get('operation_family', '')
    if attrs['category'] != 'operation': return 'none'
    if family in ('function', 'loop', 'call'): return family
    word = name.split('.')[-1]
    for token in ('load', 'store', 'constant', 'select', 'yield'):
        if token in word: return token
    for token in ('add', 'sub', 'mul', 'div'):
        if word.startswith(token): return token
    if 'cmp' in word: return 'compare'
    if 'cast' in word or word.startswith(('ext', 'trunc', 'sitof', 'fptos')): return 'cast'
    if name.startswith('math.'): return 'math'
    if family == 'memory_view': return 'view'
    if name.startswith('cf.') or word.startswith('br'): return 'branch'
    return 'other'


def _type(attrs):
    text = str(attrs.get('value_type', ''))
    if not text:
        types = attrs.get('result_types') or attrs.get('operand_types') or []
        text = str(types[0]) if types else ''
    if 'memref<' in text: kind = 'memref'
    elif 'ptr' in text: kind = 'pointer'
    elif 'vector<' in text: kind = 'vector'
    elif re.fullmatch(r'[su]?i\d+', text): kind = 'integer'
    elif re.fullmatch(r'(bf|f)\d+', text): kind = 'float'
    elif text == 'index': kind = 'index'
    elif text: kind = 'aggregate'
    else: kind = 'unknown'
    width = attrs.get('lowered_numerical_type', {}).get('element_bit_width')
    if width is None: width = attrs.get('lowered_numerical_type', {}).get('bit_width')
    if width is None:
        match = re.search(r'(?:^|[<x])(?:[su]?i|bf|f)(\d+)(?:>|$)', text)
        if match: width = int(match.group(1))
    return kind, width


def _pair(value, transform='log'):
    if value is None: return [0., 0.]
    value = float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite graph feature')
    if transform == 'log': value = math.copysign(math.log2(1 + abs(value)), value) / 24
    return [value, 1.]


def _shape_volume(shape):
    if not isinstance(shape, (tuple, list)) or any(not isinstance(x, int) or x < 0 for x in shape):
        return None
    result = 1
    for size in shape: result *= size
    return result


def _features(attrs, action_semantics):
    kind, width = _type(attrs)
    shape = attrs.get('memory_shape')
    trip = attrs.get('trip_count') if attrs.get('trip_count_static') == 1 else None
    if attrs.get('action_id') in action_semantics:
        trip = action_semantics[attrs['action_id']].get('compiler_proven_trip_count')
    value = None
    if attrs['category'] == 'operation' and attrs.get('text') == 'arith.constant':
        literal = str(attrs.get('mlir_attrs', {}).get('value', ''))
        match = re.match(r'^([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*:', literal)
        if match: value = float(match.group(1))
    prop = lambda name: attrs.get(name, {}).get('value')
    values = [width, _shape_volume(shape), len(shape) if isinstance(shape, list) else None,
              _shape_volume(attrs.get('source_array_dimensions')), attrs.get('loop_depth'), trip,
              attrs.get('operand_count'), attrs.get('result_count'), attrs.get('ssa_use_count'), value,
              prop('physical_capacity'), prop('available_capacity'), prop('available_fraction'),
              math.log2(attrs['clock_period_ns'] / 10.) if 'clock_period_ns' in attrs else None,
              prop('platform_available'), prop('requested_budget'),
              int(attrs.get('action_kind') == 'loop') if attrs['category'] == 'action_point' else None,
              int(attrs.get('action_kind') == 'array') if attrs['category'] == 'action_point' else None]
    features = []
    for name, value in zip(NUMERICAL, values):
        features.extend(_pair(value, 'raw' if name in ('available_fraction', 'clock', 'action_loop', 'action_array') else 'log'))
    return kind, features


def _edge_features(attrs):
    status = attrs.get('compiler_evidence', {}).get('status', attrs.get('evidence_status', 'not_established'))
    if status not in EVIDENCE: raise ValueError('Unknown evidence status')
    witnesses = attrs.get('compiler_evidence', {}).get('witnesses', [])
    complete = [w.get('root_candidates_complete') for w in witnesses if w.get('kind')=='memory_access' or 'root_candidates_complete' in w]
    if 'root_candidates_complete' in attrs: complete.append(attrs['root_candidates_complete'])
    complete_known = bool(complete) and all(type(v) is bool for v in complete)
    position = attrs.get('argument_index', attrs.get('operand_index'))
    return [float(status == s) for s in EVIDENCE] + [
        float(attrs.get('access') == mode) for mode in ('read', 'write', 'readwrite')
    ] + [float(attrs.get('dependence_kind') == mode) for mode in ('RAW', 'WAR', 'WAW')] + [
        float(attrs.get('alias_kind') == mode) for mode in ('must_alias','may_alias','partial_alias','no_alias')
    ] + [float(all(complete)) if complete_known else 0., float(complete_known),
        float(bool(attrs.get('loop_carried'))), float(type(attrs.get('loop_carried')) is bool),
        _pair(position)[0], float(position is not None), float(attrs.get('concrete', False)),
        float(attrs.get('access_path') == 'through_call'),
        float(attrs.get('compatibility', {}).get('status') == 'possible'),
        float(attrs.get('hardware_evidence', {}).get('status') == 'documented')]


def tensorize(joint, max_slots=64):
    """Expand compact input, select fixed features, keep negative evidence separate."""
    if joint.graph.get('schema') != overlay.SCHEMA: raise ValueError('Target overlay required')
    original = overlay.strip_overlay(joint)
    if original.graph.get('view') == 'compact':
        # The authoritative reconstruction guarantees identical semantic tensors.
        original = compiler.expand_compact(original)
        joint = overlay.enrich(original, joint.graph['target_profile_id'], joint.graph['deployment']['clock_period_ns'],
            joint.graph['deployment']['requested_resources'] and {
                k:v for k,v in joint.graph['deployment']['requested_resources'].items() if v is not None},
            platform_available={k:v for k,v in joint.graph['deployment']['platform_available_resources'].items() if v is not None})
    nodes = sorted(joint.nodes)
    index = {n:i for i,n in enumerate(nodes)}
    semantics = {label: a.get('loop_semantics', {}) for label,a in joint.graph['action_annotations'].items()}
    categorical = {name: [] for name in ('node_type', 'op_type', 'value_type', 'resource_type', 'root_type')}
    numeric = []
    for node in nodes:
        attrs = joint.nodes[node]
        kind, row = _features(attrs, semantics)
        numeric.append(row)
        categorical['node_type'].append(NODE_TYPES.index(attrs['category']))
        categorical['op_type'].append(OP_TYPES.index(_op(attrs)))
        categorical['value_type'].append(VALUE_TYPES.index(kind))
        categorical['resource_type'].append(RESOURCE_TYPES.index(attrs.get('resource_class', 'none')))
        root = attrs.get('root_kind', 'none')
        categorical['root_type'].append(ROOT_TYPES.index(root if root in ROOT_TYPES else 'other'))
    semantic_bindings, semantic_access_maps = semantic._index_context(joint)
    ends, types, features, source_keys = [], [], [], []
    semantic_rows, semantic_records = [], []
    negative_ends, negative_types, negative_features, negative_keys = [], [], [], []
    for u,v,key,attrs in sorted(joint.edges(keys=True,data=True), key=lambda e:(e[0],e[1],str(e[2]))):
        relation = attrs['relation_type']
        if relation not in FORWARD_RELATIONS: raise ValueError('Unknown graph relation: '+relation)
        feature = _edge_features(attrs) + semantic.edge_features(
            joint,u,v,attrs,semantic_bindings,semantic_access_maps)
        semantic_row = feature[BASE_EDGE_DIM:]
        if attrs.get('alias_kind') == 'no_alias':
            negative_ends.append((index[u], index[v])); negative_types.append(RELATIONS.index(relation))
            negative_features.append(feature); negative_keys.append(str(key))
            continue
        semantic_rows.append(semantic_row)
        semantic_records.append((u,v,str(key),attrs))
        for reverse in (False, True):
            ends.append((index[v],index[u]) if reverse else (index[u],index[v]))
            types.append(RELATIONS.index(('reverse:' if reverse else '') + relation))
            features.append(feature); source_keys.append(str(key))
    slots = joint.graph['action_slots']
    if len(slots['labels']) > max_slots: raise ValueError('Action slots exceed contract; truncation forbidden')
    masks = {}
    for name in ('registered_mask', 'matched_mask', 'unresolved_mask', 'source_only_mask'):
        masks[name] = torch.tensor([slots[name] + [False]*(max_slots-len(slots['labels']))], dtype=torch.bool)
    active = sorted((int(a['action_id'][1:])-1, index[n]) for n,a in joint.nodes(data=True) if a['category']=='action_point')
    descriptors = demand.static_action_descriptors(original)
    action_demand = torch.zeros(1,max_slots,ACTION_DEMAND_DIM,dtype=torch.float32)
    action_demand_mask = torch.zeros(1,max_slots,dtype=torch.bool)
    action_demand_known_mask = torch.zeros(1,max_slots,len(demand.NUMERIC_FEATURES),dtype=torch.bool)
    for label,descriptor in descriptors['actions'].items():
        if not descriptor.get('matched'):
            continue
        slot=int(label[1:])-1
        if slot>=max_slots: raise ValueError('Action-demand slot exceeds contract; truncation forbidden')
        vector=demand.action_feature_vector(descriptor)
        action_demand[0,slot]=torch.tensor(vector,dtype=torch.float32)
        action_demand_mask[0,slot]=True
        action_demand_known_mask[0,slot]=torch.tensor([bool(vector[i*2+1]) for i in range(len(demand.NUMERIC_FEATURES))])
    action_kind = torch.zeros(1,max_slots,dtype=torch.long)
    for record in original.graph['action_point_coverage']['records']:
        slot = int(record['action_id'][1:])-1
        action_kind[0,slot] = {'loop':1,'array':2}.get(record.get('source_binding',{}).get('kind'),0)
    action_semantic = torch.zeros(1,max_slots,SEMANTIC_EDGE_DIM,dtype=torch.float32)
    action_semantic_mask = torch.zeros(1,max_slots,dtype=torch.bool)
    action_semantic_witnesses = {}
    for action_node, action_attrs in joint.nodes(data=True):
        if action_attrs.get('category') != 'action_point' or action_attrs.get('action_resolution') == 'unresolved':
            continue
        slot = int(action_attrs['action_id'][1:])-1
        if slot >= max_slots:
            raise ValueError('Action semantic slot exceeds contract; truncation forbidden')
        if not masks['matched_mask'][0][slot]:
            continue
        targets = [v for _,v,e in joint.out_edges(action_node,data=True) if e.get('relation_type') == 'targets']
        if len(targets) != 1:
            continue
        vector, known, witnesses = semantic.action_edge_summary(joint,action_node,targets[0],semantic_rows,semantic_records)
        if known:
            action_semantic[0,slot] = torch.tensor(vector,dtype=torch.float32)
            action_semantic_mask[0,slot] = True
            action_semantic_witnesses[action_attrs['action_id']] = witnesses
    dep = joint.graph['deployment']; profile = joint.graph['target_profile']
    context = [math.log2(1+profile['resources'][r]['value'])/24 for r in overlay.RESOURCES]
    context += [math.log2(1+dep['available_resources'][r])/24 for r in overlay.RESOURCES]
    context += [dep['available_fractions'][r] for r in overlay.RESOURCES]
    context += [math.log2(dep['clock_period_ns']/10.)]
    data = Data(x=torch.tensor(numeric,dtype=torch.float32),
        edge_index=torch.tensor(ends,dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(types,dtype=torch.long), edge_attr=torch.tensor(features,dtype=torch.float32),
        negative_edge_index=torch.tensor(negative_ends,dtype=torch.long).reshape(-1,2).t().contiguous(),
        negative_edge_type=torch.tensor(negative_types,dtype=torch.long),
        negative_edge_attr=torch.tensor(negative_features,dtype=torch.float32).reshape(-1,EDGE_DIM),
        action_node_index=torch.tensor([n for s,n in active],dtype=torch.long),
        action_slot=torch.tensor([s for s,n in active],dtype=torch.long), action_kind_slot=action_kind,
        action_demand=action_demand, action_demand_mask=action_demand_mask,
        action_demand_known_mask=action_demand_known_mask,
        action_semantic=action_semantic, action_semantic_mask=action_semantic_mask,
        hardware_context=torch.tensor([context],dtype=torch.float32), **masks,
        **{name:torch.tensor(values,dtype=torch.long) for name,values in categorical.items()})
    data.audit = {'schema':SCHEMA, 'node_ids':nodes, 'source_edge_keys':source_keys,
                  'negative_edge_keys':negative_keys, 'program_payload_sha256':joint.graph['program_payload_sha256'],
                  'program_graph_identity':original.graph['graph_identity'],
                  'overlay_identity':joint.graph['overlay_identity'],
                  'previous_tensor_schema':LEGACY_SCHEMA,
                  'semantic_edge_schema':semantic.semantic_metadata()['schema'],
                  'semantic_edge_features':semantic.semantic_metadata(),
                  'action_semantic_witnesses':action_semantic_witnesses,
                  'tensor_implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'semantic_feature_implementation_sha256':hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
                  'action_demand_schema':descriptors['schema'],
                  'action_demand_implementation':descriptors['implementation']}
    validate_tensors(data)
    return data


def validate_tensors(data):
    if data.x.shape != (data.num_nodes,NODE_DIM) or data.edge_attr.shape != (data.edge_index.shape[1],EDGE_DIM):
        raise ValueError('Tensor shape mismatch')
    for tensor in (data.x,data.edge_attr,data.hardware_context,data.negative_edge_attr):
        if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite tensors')
    for edges in (data.edge_index,data.negative_edge_index):
        if edges.numel() and (edges.min()<0 or edges.max()>=data.num_nodes): raise ValueError('Invalid edge endpoint')
    if len(set(data.action_slot.tolist())) != data.action_slot.numel(): raise ValueError('Duplicate action slot')
    if not torch.equal(data.registered_mask & ~data.matched_mask,data.unresolved_mask): raise ValueError('Action mask mismatch')
    if data.action_node_index.numel() != int(data.matched_mask.sum()): raise ValueError('Action mapping mismatch')
    if data.action_demand.shape != (data.registered_mask.shape[0],data.registered_mask.shape[1],ACTION_DEMAND_DIM):
        raise ValueError('Action-demand tensor shape mismatch')
    if data.action_semantic.shape != (data.registered_mask.shape[0],data.registered_mask.shape[1],SEMANTIC_EDGE_DIM):
        raise ValueError('Action semantic tensor shape mismatch')
    if data.action_semantic_mask.shape != data.registered_mask.shape:
        raise ValueError('Action semantic mask shape mismatch')
    if data.action_demand_mask.shape != data.registered_mask.shape or data.action_demand_known_mask.shape != (*data.registered_mask.shape,len(demand.NUMERIC_FEATURES)):
        raise ValueError('Action-demand mask shape mismatch')
    if not torch.isfinite(data.action_demand).all(): raise ValueError('Nonfinite action-demand features')
    if not torch.isfinite(data.action_semantic).all(): raise ValueError('Nonfinite action-semantic features')
    if (data.action_demand_mask & ~data.matched_mask).any(): raise ValueError('Unresolved action has fabricated demand features')
    if (data.action_semantic_mask & ~data.matched_mask).any(): raise ValueError('Unresolved action has fabricated semantic features')
    if data.action_demand_known_mask.any(-1).logical_and(~data.action_demand_mask).any(): raise ValueError('Unmatched action has known demand features')
    if (data.action_semantic[~data.action_semantic_mask] != 0).any(): raise ValueError('Masked action semantic features must be zero')
    if data.action_node_index.numel() and not (data.node_type[data.action_node_index]==NODE_TYPES.index('action_point')).all():
        raise ValueError('Wrong action node category')
    return True


def without_hardware_messages(data):
    result = data.clone()
    hardware = result.node_type >= len(compiler.NODE_KINDS)
    keep = ~hardware[result.edge_index[0]] & ~hardware[result.edge_index[1]]
    result.edge_index = result.edge_index[:,keep]
    result.edge_type = result.edge_type[keep]
    result.edge_attr = result.edge_attr[keep]
    return result


def tensor_bytes(data):
    return sum(v.numel()*v.element_size() for key,v in data if torch.is_tensor(v))


class SmallGraphEncoder(nn.Module):
    """Two typed TransformerConv layers; hardware enters only through edges."""
    def __init__(self, hidden=64):
        super().__init__()
        vocab = (len(NODE_TYPES),len(OP_TYPES),len(VALUE_TYPES),len(RESOURCE_TYPES),len(ROOT_TYPES))
        self.embeddings = nn.ModuleList([nn.Embedding(n,8) for n in vocab])
        self.input = nn.Linear(NODE_DIM+40,hidden)
        self.action_demand_input = nn.Linear(ACTION_DEMAND_DIM,hidden,bias=False)
        self.action_semantic_input = nn.Linear(SEMANTIC_EDGE_DIM,hidden,bias=False)
        self.relation = nn.Embedding(len(RELATIONS),16)
        self.edge_input = nn.Linear(EDGE_DIM,16)
        self.layers = nn.ModuleList([TransformerConv(hidden,hidden,heads=2,concat=False,edge_dim=16,dropout=0) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.output = nn.Linear(hidden,128)

    def forward(self, data):
        cats = (data.node_type,data.op_type,data.value_type,data.resource_type,data.root_type)
        x = torch.relu(self.input(torch.cat([data.x]+[e(c) for e,c in zip(self.embeddings,cats)],dim=-1)))
        if data.action_node_index.numel():
            batch=getattr(data,'batch',None)
            if batch is None: rows=torch.zeros_like(data.action_slot)
            else: rows=batch[data.action_node_index]
            features=data.action_demand[rows,data.action_slot]
            x=x.index_add(0,data.action_node_index,self.action_demand_input(features))
            semantic_features=data.action_semantic[rows,data.action_slot]
            semantic_mask=data.action_semantic_mask[rows,data.action_slot].to(x.dtype).unsqueeze(-1)
            x=x.index_add(0,data.action_node_index,
                          self.action_semantic_input(semantic_features)*semantic_mask)
        edge = self.relation(data.edge_type) + self.edge_input(data.edge_attr)
        for conv,norm in zip(self.layers,self.norms): x = norm(x+torch.relu(conv(x,data.edge_index,edge)))
        return self.output(x)

    def memories(self, data, embeddings=None):
        x = self(data) if embeddings is None else embeddings
        batch = getattr(data,'batch',None)
        if batch is None: batch = torch.zeros(data.num_nodes,dtype=torch.long,device=x.device)
        count = int(data.registered_mask.shape[0]); slots=data.registered_mask.shape[1]
        memory = x.new_zeros((count,slots,128))
        memory[batch[data.action_node_index],data.action_slot] = x[data.action_node_index]
        program = data.node_type < len(compiler.NODE_KINDS)
        pooled = global_mean_pool(x[program],batch[program],size=count)
        return memory,pooled
