"""Hash exact static encoder tensors independently of host/path names."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.paper_gnn.data import tensorize,sha
from GNN_branch import mlir_graph_v2 as compiler
from experiments.paper_gnn.train import write_json

FIELDS=('x','edge_index','edge_attr','action_slots','action_nodes')

def tensor_identity(tensors):
    result={}
    for kernel,g in sorted(tensors.items()):
        result[kernel]={name:hashlib.sha256(getattr(g,name).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name in FIELDS}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();identity=json.loads((a.root/'A/data_identity.json').read_text());schema=identity['feature_schema'];tensors={}
    for k,digest in sorted(identity['graph_sha256'].items()):
        paths=list((a.root/'cohort/authenticated_graphs').glob(k+'-*.json'));assert len(paths)==1 and sha(paths[0])==digest;graph=compiler.read_graph(paths[0])
        for label,attrs in schema['source_only_unbound_action_nodes'].get(k,{}).items():graph.add_node('source-unbound:'+label,**attrs)
        tensors[k]=tensorize(graph,schema['vocabulary'])
    write_json(tensor_identity(tensors),a.root/'static_tensor_identity.json')
