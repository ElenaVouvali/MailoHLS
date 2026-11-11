"""Authenticate existing RC3 assets and export frozen four-layer node states."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations
from experiments.rc3_fusion_foundation.structural_memory_v1 import _action_evidence_scopes
from experiments.gemv_structural_explainability.constants import TYPES, STATUS, KNOWN

ROOT = Path(__file__).resolve().parents[2]
GEMV = ROOT/'paper_figures/gemv_rc3_20261105'
GRAPH_SHA = '7d295d791f8438eeda62a6b230ab80c8d3a051ecda43b822c7a3827ec0cb43a6'
GNN_SHA = '79540c9f27e4807e0c3badbd876322bbde5badd9d4a04f601b316ee75e944ee8'
GNN = Path('/home/elvouvali/MailoHLS/artifacts/paper_gnn_20261110/latency/selected.pt')
STAGE1 = Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
STAGE1_SHA = '28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')


def authenticate():
    manifest = json.loads((GEMV/'manifest.json').read_text())
    if sha(GEMV/'gemv.canonical.json') != GRAPH_SHA:
        raise ValueError('GEMV graph file authentication failed')
    for path,expected in ((ROOT/manifest['selected_source'],manifest['source_sha256']),
                          (GEMV/'gemv.mlir',manifest['provenance']['mlir_sha256']),
                          (GEMV/'inputs/kernel_info.txt',manifest['metadata_sha256'])):
        if sha(path) != expected:
            raise ValueError('GEMV input authentication failed: '+str(path))
    graph = read_graph(GEMV/'gemv.canonical.json')
    if (graph.number_of_nodes(),graph.number_of_edges()) != (66,106):
        raise ValueError('Unexpected graph topology')
    relations = build_action_relations(graph, expected_graph_identity=manifest['graph_identity'])
    expected = {'L1':'A','L2':'x','L3':'y','L4':None,'L5':None}
    records = []
    for label,name in expected.items():
        attrs = graph.nodes['action:'+label]
        if attrs.get('array_variable') != name:
            raise ValueError('Source action mismatch: '+label)
        target = relations['compiler_target_nodes'][label]
        anchor = graph.nodes[target['node']]
        records.append(dict(action=label, slot=int(label[1:])-1,
            source_path=manifest['selected_source'],source_sha256=manifest['source_sha256'],
            source_file=attrs['source_file'],line=attrs['source_line'],column=attrs['source_column'],
            construct=name or ('outer loop' if label=='L4' else 'inner reduction loop'),
            directives=attrs['supported_directives'],compiler_anchor=target['node'],
            graph_identity=manifest['graph_identity'],graph_file_sha256=GRAPH_SHA,
            node_ids=['action:'+label,target['node']],edge_ids=[target['target_edge']],
            source_metadata=attrs,compiler_metadata=anchor,evidence_status='proven'))
    if records[2]['source_metadata']['source_array_dimensions'] != [32] or \
       records[2]['compiler_metadata']['memory_shape'] != []:
        raise ValueError('Source/compiler scalar promotion distinction lost')
    return graph,relations,records


def load_encoder(path=GNN):
    from experiments.paper_gnn.data import SCHEMA, CATEGORIES, RELATIONS, EVIDENCE, NUMERIC
    from experiments.paper_gnn.train import backend
    if sha(path) != GNN_SHA:
        raise ValueError('Only the authenticated preliminary four-layer checkpoint is accepted')
    state = torch.load(path,map_location='cpu',weights_only=False)
    schema = state['schema']
    if schema['schema'] != SCHEMA:
        raise ValueError('Tensor schema incompatible with checkpoint')
    for key,expected in [('node_categories',CATEGORIES),('relations',RELATIONS),
                         ('evidence',EVIDENCE),('numeric',NUMERIC)]:
        if tuple(schema[key]) != tuple(expected):
            raise ValueError('Tensor schema mismatch: '+key)
    if state['protocol']['architecture']['pre_directive_transformer_layers'] != 4:
        raise ValueError('Four static layers required')
    # Authenticate the implementation actually defining the encoder/tensorizer.
    for file in ('GNN_branch/model.py','GNN_branch/nn_att.py','experiments/paper_gnn/data.py'):
        if sha(ROOT/file) != state['protocol']['source_sha256'][file]:
            raise ValueError('Checkpoint implementation differs: '+file)
    Model,_ = backend()
    node_dim = len(CATEGORIES)+len(schema['vocabulary'])+2*len(NUMERIC)
    edge_dim = 2*len(RELATIONS)+len(EVIDENCE)
    model = Model(node_dim,edge_dim)
    model.load_state_dict(state['model'],strict=True)
    model.eval().requires_grad_(False)
    if len(model.conv_layers) != 3 or model.conv_first.out_channels != 64:
        raise ValueError('Incorrect TransformerConv architecture')
    identity = dict(path=str(path),sha256=sha(path),representation_schema=model.schema,
        tensor_schema=schema,epoch_zero_based=state['epoch'],updates=state['steps'],
        protocol=state['protocol'],strict_state_loading=True,previously_trained=True,
        production_qualified=False,selection='preliminary retained by completed-study decision')
    return model,schema,identity


def catalog(graph, relations, capacity=1024):
    """Individually addressable records; eligibility reuses audited RC3 scopes.

    Complete compiler payloads are retained; known bits describe particular facts,
    not the completeness of a whole node. Non-proven edges cannot open routes.
    """
    scopes = _action_evidence_scopes(graph,relations)
    rows = []
    accesses = {u for u,v,k,a in graph.edges(keys=True,data=True)
                if a['relation_type']=='memory_access' and a.get('evidence_status')=='proven'}
    carried = {n for u,v,k,a in graph.edges(keys=True,data=True)
               if a['relation_type']=='loop_carried' for n in (u,v)}
    for node,a in sorted(graph.nodes(data=True)):
        kind = None
        if a['category']=='memory_object': kind='memory_root'
        elif a['category']=='ssa_value' and a.get('is_memory_root'): kind='memory_ssa'
        elif a.get('is_loop'): kind='loop'
        elif a['category']=='control_region': kind='control'
        elif node in accesses: kind='access'
        elif a.get('operation_family')=='arithmetic': kind='arithmetic'
        elif a['category']=='ssa_value' and node in carried: kind='carried_value'
        if kind is None: continue
        known = dict.fromkeys(KNOWN,False)
        known.update({'class':True,'proven':True,'shape':a.get('memory_shape') is not None,
            'trip_count':a.get('trip_count_static')==1,
            'operation_family':a.get('operation_family') is not None})
        rows.append(dict(id='node:'+node,type=kind,kind='node',node=node,status='proven',
            known=known,payload=a,eligible_actions=[q for q,s in scopes.items() if node in s['nodes']]))
    for u,v,k,a in sorted(graph.edges(keys=True,data=True),key=lambda e:(e[0],e[1],str(e[2]))):
        relation = a['relation_type']
        if relation not in TYPES: continue
        known = dict.fromkeys(KNOWN,False)
        known.update({'class':True,'proven':a.get('evidence_status')=='proven',
            'affine_map':bool((a.get('access_indices') or {}).get('affine_map')),
            'access_role':a.get('access') in ('read','write','readwrite'),
            'loop_carried':relation=='loop_carried' or a.get('loop_carried') is not None,
            'dependence_kind':a.get('dependence_kind') is not None})
        rows.append(dict(id='edge:'+str(k),type=relation,kind='edge',source=u,target=v,
            edge_key=str(k),status=a.get('evidence_status','not_established'),known=known,
            payload=a,eligible_actions=[q for q,s in scopes.items() if str(k) in s['edges']
                                       and a.get('evidence_status')=='proven']))
    rows.sort(key=lambda r:(TYPES.index(r['type']),r['id']))
    if len(rows)>capacity: raise ValueError('Evidence overflow; no truncation')
    for row in rows:
        row['provenance'] = dict(graph_identity=graph.graph['graph_identity'],
            compiler_inputs=graph.graph['provenance'],
            node_ids=[row['node']] if row['kind']=='node' else [row['source'],row['target']],
            edge_ids=[] if row['kind']=='node' else [row['edge_key']])
    return rows


@torch.no_grad()
def export_pack(graph,model,schema):
    from experiments.paper_gnn.data import tensorize, RELATIONS, EVIDENCE
    tensor = tensorize(graph,schema['vocabulary'])
    first,jk = model.encode(tensor)
    node_h = torch.cat((first,jk),-1)
    node_ids = sorted(graph.nodes)
    index = {n:i for i,n in enumerate(node_ids)}
    relation = build_action_relations(graph)
    rows = catalog(graph,relation)
    n = len(rows)
    active = torch.zeros(1,64,dtype=torch.bool)
    active[0,tensor.action_slots] = True
    action = model.memory(tensor)[None]
    if not torch.equal(action[0,tensor.action_slots],node_h[tensor.action_nodes]):
        raise ValueError('Incorrect [first|JK] export')
    pack = dict(action=action,active=active,a=torch.zeros(1,n,128),b=torch.zeros(1,n,128),
        edge=torch.zeros(1,n,tensor.edge_attr.shape[1]),known=torch.zeros(1,n,len(KNOWN)),
        is_edge=torch.zeros(1,n,dtype=torch.bool),valid=torch.ones(1,n,dtype=torch.bool),
        types=torch.zeros(1,n,dtype=torch.long),status=torch.zeros(1,n,dtype=torch.long),
        routes=torch.zeros(1,64,n,dtype=torch.bool),records=rows,
        graph_identity=graph.graph['graph_identity'],gnn_sha256=GNN_SHA,
        edge_dim=tensor.edge_attr.shape[1],schema='salt-gemv-frozen-evidence-input-v1')
    for i,row in enumerate(rows):
        pack['types'][0,i] = TYPES.index(row['type'])
        pack['status'][0,i] = STATUS.index(row['status'])
        pack['known'][0,i] = torch.tensor([row['known'][k] for k in KNOWN])
        for label in row['eligible_actions']: pack['routes'][0,int(label[1:])-1,i]=True
        if row['kind']=='node':
            pack['a'][0,i] = node_h[index[row['node']]]
        else:
            pack['is_edge'][0,i] = True
            pack['a'][0,i],pack['b'][0,i] = node_h[index[row['source']]],node_h[index[row['target']]]
            pack['edge'][0,i,RELATIONS.index(row['type'])] = 1.
            if row['status'] in EVIDENCE:
                pack['edge'][0,i,2*len(RELATIONS)+EVIDENCE.index(row['status'])] = 1.
    for value in pack.values():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite GNN export')
    return pack,first,jk


def coverage(pack):
    rows = pack['records']
    result = {}
    for k in pack['active'][0].nonzero().flatten().tolist():
        selected = [r for r in rows if 'L'+str(k+1) in r['eligible_actions']]
        result['L'+str(k+1)] = dict(count=len(selected),by_type=dict(Counter(r['type'] for r in selected)),
            by_status=dict(Counter(r['status'] for r in selected)),
            complete_provenance_fraction=sum(bool(r['provenance']['compiler_inputs']) for r in selected)/len(selected)
                if selected else None,
            evidence_ids=[r['id'] for r in selected])
    return result
