#!/usr/bin/env python3
"""CPU-only frozen V4b representation and RC3 Memory-C audit extraction."""
from __future__ import annotations
import collections, hashlib, json, os, sys, tarfile, tempfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.architecture_representation_audit.audit_cohort import sha
from experiments.rc3_fusion_foundation.structural_memory_v1 import (
    ACTION_SLOTS, EVIDENCE_CAPACITY, EVIDENCE_TYPES, KNOWN_FIELDS,
    _target_graph, _evidence, _action_evidence_scopes, load_frozen_encoder,
)
from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations

ARCHIVE=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
CHECKPOINT=Path('/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/compiler_selected.pt')
CKSHA='ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162'
OUT=Path('/tmp/mailohls-gnn-audit-activations')

def main():
    torch.set_num_threads(2); torch.manual_seed(0); OUT.mkdir(parents=True,exist_ok=True)
    model, identity=load_frozen_encoder(CHECKPOINT,CKSHA); model.eval()
    audit=json.loads((ROOT/'experiments/architecture_representation_audit/results/cohort_manifest.json').read_text())
    rows=[r for r in audit['graphs'] if r['family_split'] in ('train','val','test')]
    layer_out={}; handles=[]
    for i,norm in enumerate(model.graph.norms):
        handles.append(norm.register_forward_hook(lambda m,a,o,i=i: layer_out.__setitem__(f'R{i+2}',o.detach().cpu())))
    handles.append(model.graph.output.register_forward_hook(lambda m,a,o: layer_out.__setitem__('R4',o.detach().cpu())))
    result=[]; evidence_totals=collections.Counter(); action_rows=[]; distances=[]; interventions=[]
    with tarfile.open(ARCHIVE,'r:gz') as tf, tempfile.TemporaryDirectory(prefix='rc3-graphs-') as temp:
      for k,row in enumerate(rows):
        raw=tf.extractfile(row['graph_member']).read(); path=Path(temp)/f'{k}.json'; path.write_bytes(raw)
        graph=read_graph(path); _,data=_target_graph(graph,target='u200',clock_ns=10.)
        # Capture after raw categorical/numerical projection and direct action-demand/semantic injections.
        layer_out.clear()
        def pre_hook(m,args): layer_out['R1']=args[0].detach().cpu()
        h=model.graph.layers[0].register_forward_pre_hook(pre_hook)
        with torch.inference_mode(): node_h=model.graph(data)
        h.remove(); r1=layer_out.pop('R1'); r2=layer_out.pop('R2'); r3=layer_out.pop('R3'); r4=layer_out.pop('R4')
        action_h=node_h[data.action_node_index].detach().cpu()
        # Diagnostic in-memory feature-group ablations. These are not valid
        # transformed programs and make no compiler-semantic claims.
        groups={
          'trip_count_and_loop_depth': ('node_pairs',(4,5)),
          'array_shape_and_volume': ('node_pairs',(1,2,3)),
          'action_demand_summary': ('action_demand',()),
          'action_semantic_summary': ('action_semantic',()),
          'compiler_edge_features': ('edge_features',()),
        }
        for group,(kind,indices) in groups.items():
          altered=data.clone()
          if kind=='node_pairs':
            for col in indices: altered.x[:,2*col:2*col+2]=0
          elif kind=='action_demand': altered.action_demand.zero_()
          elif kind=='action_semantic': altered.action_semantic.zero_()
          elif kind=='edge_features': altered.edge_attr.zero_()
          with torch.inference_mode(): changed=model.graph(altered)
          changed_actions=changed[data.action_node_index].detach().cpu()
          delta=(changed_actions-action_h).norm(dim=-1)
          interventions.append({'graph_member':row['graph_member'],'family_id':row['family_id'],'split':row['family_split'],'group':group,
            'max_action_l2':float(delta.max()) if delta.numel() else 0.,'mean_action_l2':float(delta.mean()) if delta.numel() else 0.,
            'changed_action_count':int((delta>1e-7).sum()),'action_count':int(delta.numel()),
            'interpretation':'diagnostic input ablation; not a source transformation or evidence of semantic validity'})
        try: rel=build_action_relations(graph,max_slots=ACTION_SLOTS); relation_error=None
        except Exception as e:
          rel=None; relation_error=f'{type(e).__name__}: {e}'
        try:
          ev=_evidence(graph,data,node_h,model,cap=EVIDENCE_CAPACITY); overflow=None
        except ValueError as e:
          ev=[]; overflow=str(e)
        scopes=_action_evidence_scopes(graph,rel) if ev and rel is not None else {}
        evidence_totals.update(x['evidence_type'] for x in ev)
        evidence_arr=torch.stack([x['_vector'] for x in ev]).numpy().astype('float32') if ev else __import__('numpy').zeros((0,128),dtype='float32')
        action_mask=rel['absolute_slots']['valid_mask'] if rel is not None else [False]*ACTION_SLOTS
        active=[i for i,x in enumerate(action_mask) if x]
        dists=[]
        # Undirected/effective topology distances from compiler target to compiler witnesses.
        import networkx as nx
        ug=graph.to_undirected()
        targets=rel['compiler_target_nodes'] if rel is not None else {}
        evidence_nodes={x['node_id'] for x in ev if x['kind']=='node'}
        for label in (f'L{i+1}' for i in active):
          target=targets[label]['node']; lengths=nx.single_source_shortest_path_length(ug,target)
          dd=[lengths[n] for n in evidence_nodes if n in lengths and n!=target]
          dists.extend(dd)
          action_rows.append({'graph_member':row['graph_member'],'family_id':row['family_id'],'split':row['family_split'],
             'action_id':label,'kind':next((a['kind'] for a in row['actions'] if a['action_id']==label),None),
             'target_node':target,'target_type':graph.nodes[target].get('operation_family'),
             'loop_depth':graph.nodes[target].get('loop_depth'),'trip_count':graph.nodes[target].get('trip_count_static'),
             'recurrence_positive_witnesses':sum(1 for u,v,_,e in graph.edges(keys=True,data=True) if e.get('relation_type')=='loop_carried' and e.get('evidence_status')=='proven' and e.get('carrying_loop')==target),
             'node_witness_count':sum(1 for x in ev if x['kind']=='node' and x['node_id'] in scopes.get(label,{}).get('nodes',set())),
             'edge_witness_count':sum(1 for x in ev if x['kind']=='edge' and x['edge_key'] in scopes.get(label,{}).get('edges',set())),
             'min_witness_distance':min(dd) if dd else None})
        distances.extend(dists)
        # Save large matrices only in /tmp; compact metadata/summary is committed later.
        np=__import__('numpy')
        activation_file=OUT/f'{k:02d}.npz'
        np.savez_compressed(activation_file,R0=data.x.detach().cpu().numpy().astype('float32'),R1=r1.numpy().astype('float32'),R2=r2.numpy().astype('float32'),R3=r3.numpy().astype('float32'),R4=r4.numpy().astype('float32'),R5=action_h.numpy().astype('float32'),R6=evidence_arr)
        result.append({'graph_member':row['graph_member'],'graph_sha256':sha(raw),'graph_identity':row['graph_identity'],'family_id':row['family_id'],'split':row['family_split'],'npz':f'{k:02d}.npz','activation_sha256':sha(activation_file.read_bytes()),'nodes':graph.number_of_nodes(),'edges':graph.number_of_edges(),'actions':len(active),'relation_error':relation_error,'evidence_count':len(ev),'evidence_overflow':overflow,'evidence_types':dict(collections.Counter(x['evidence_type'] for x in ev)),'action_evidence_scope':{a:{'nodes':len(s['nodes']),'edges':len(s['edges'])} for a,s in scopes.items()}})
        if (k+1)%10==0: print(f'processed {k+1}/{len(rows)}',flush=True)
    for h in handles: h.remove()
    manifest={'schema':'salt-hls-frozen-v4b-cpu-extraction-v1','checkpoint':identity,'checkpoint_sha256':sha(CHECKPOINT.read_bytes()),'device':'cpu','torch':torch.__version__,'pyg':'2.6.1','activation_root':str(OUT),'graph_count':len(result),'graphs':result,'evidence_type_totals':dict(evidence_totals),'action_rows':action_rows,'diagnostic_feature_group_interventions':interventions,'action_to_evidence_distance_summary':{'n':len(distances),'min':min(distances) if distances else None,'median':float(__import__('numpy').median(distances)) if distances else None,'max':max(distances) if distances else None}}
    out=ROOT/'experiments/architecture_representation_audit/results/gnn_extraction_manifest.json'; out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); print(json.dumps({'manifest':str(out),'activation_root':str(OUT),'graphs':len(result),'evidence_type_totals':dict(evidence_totals),'overflow':sum(bool(x['evidence_overflow']) for x in result)},indent=2))
if __name__=='__main__': main()
