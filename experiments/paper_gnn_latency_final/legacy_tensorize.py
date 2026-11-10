"""Authenticated RC3/compiler-only adapter and existing measured-row utilities."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch
from torch_geometric.data import Data
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch.action_point_relations import action_point_views
from GNN_branch.target_graph_v22 import load_registry
from LLM_branch.common import directive_domains as domains
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
from Preprocessing.create_jsonl import build_examples, application_sources, load_actions, validate_preprocessing_manifest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'salt-paper-rc3-compiler-tensors-v1'
NUMERIC = ('loop_depth','trip_count','compiler_proven_trip_count','tcad_trip_count',
           'lower_bound','upper_bound','step','operand_count','result_count','use_count',
           'memory_rank','operand_index','dimension')
CATEGORIES = tuple(sorted(compiler.NODE_KINDS))
RELATIONS = tuple(sorted(compiler.EDGE_KINDS))
EVIDENCE = ('proven','possible','not_established','not_applicable')
RESOURCE_ORDER = ('BRAM_18K','DSP','FF','LUT')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def candidate_vector(assignment, actions):
    result = torch.zeros(64,8)
    for label, action in actions.items():
        i = int(label[1:])-1
        if not 0 <= i < 64: raise ValueError('Action truncation forbidden')
        get = lambda name: assignment['AUTO{_'+name+'_'+label+'}']
        if action.kind == 'loop':
            result[i,0] = float(get('PIPE'))
            result[i,1] = math.log2(1+int(get('UNROLL')))/12
        else:
            result[i,2+('none','complete','block','cyclic').index(get('ARRAY_T'))] = 1
            result[i,6] = math.log2(1+int(get('ARRAY_F')))/12
            result[i,7] = int(get('ARRAY_D'))/8
    return result


def tensorize(graph, vocabulary):
    if graph.graph.get('semantic_revision') != 'rc3': raise ValueError('RC3 required')
    nodes = sorted(graph.nodes)
    index = {n:i for i,n in enumerate(nodes)}
    features = []
    for node in nodes:
        a = graph.nodes[node]
        if a.get('_target_overlay') or a['category'] not in CATEGORIES:
            raise ValueError('Only original compiler nodes allowed')
        row = [float(a['category']==c) for c in CATEGORIES]
        token = str(a.get('text','')) if a['category']=='operation' else ''
        token = token if token in vocabulary else '<unknown>'
        row += [float(token==v) for v in vocabulary]
        for name in NUMERIC:
            value = a.get(name)
            known = isinstance(value,(int,float)) and math.isfinite(value)
            row += [math.copysign(math.log2(1+abs(value)),value)/24 if known else 0., float(known)]
        features.append(row)
    ends, edges = [], []
    for u,v,k,a in sorted(graph.edges(keys=True,data=True), key=lambda e:(e[0],e[1],str(e[2]))):
        relation = a['relation_type']
        if relation not in RELATIONS: raise ValueError('Noncompiler relation')
        for reverse in (False,True):
            ends.append((index[v],index[u]) if reverse else (index[u],index[v]))
            edges.append([float(relation==r and reverse==rev) for rev in (False,True) for r in RELATIONS]+
                         [float(a.get('evidence_status')==e) for e in EVIDENCE])
    active = sorted((int(a['action_id'][1:])-1,index[n]) for n,a in graph.nodes(data=True) if a['category']=='action_point')
    if len({i for i,n in active}) != len(active) or any(i>=64 or i<0 for i,n in active):
        raise ValueError('Ambiguous or truncated action identities')
    return Data(x=torch.tensor(features,dtype=torch.float32),
        edge_index=torch.tensor(ends,dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edges,dtype=torch.float32),
        action_slots=torch.tensor([i for i,n in active]), action_nodes=torch.tensor([n for i,n in active]))
