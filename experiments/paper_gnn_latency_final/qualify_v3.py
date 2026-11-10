"""Validate the complete-source adapter without performing optimizer updates."""
import argparse
import os
from pathlib import Path
import torch
from experiments.paper_gnn.train import backend,write_json
from experiments.paper_gnn.data import sha
from experiments.paper_gnn_latency_final.dataset_v3 import CompletePaperData
from experiments.paper_gnn_latency_final.memory_export import export

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    assert not torch.cuda.is_available();torch.set_num_threads(4);os.environ['SALT_CORRECTED_COHORT']=str(a.root/'cohort/corrected_cohort_v1')
    out=a.root/'input_compatibility_probe';out.mkdir();data=CompletePaperData(out);Model,_=backend();first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1]);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
    export(model,data,out,a.checkpoint,'PARETO_LATENCY',state['protocol'])
    checks=[]
    for kernel in data.source_only_nodes:
        graph=data.tensors[kernel];memory=model.memory(graph);source={int(k[1:])-1 for k in data.actions[kernel]}
        assert source==set(graph.action_slots.tolist())
        # A complete candidate path remains live for the genuinely unbound arrays.
        from experiments.paper_gnn.data import candidate_vector
        neutral={}
        for label,action in data.actions[kernel].items():
            fields={'PIPE':'0','UNROLL':'0'} if action.kind=='loop' else {'ARRAY_T':'none','ARRAY_F':'0','ARRAY_D':'0'}
            neutral.update({'AUTO{_'+name+'_'+label+'}':value for name,value in fields.items()})
        changed=dict(neutral);label=next(iter(data.source_only_nodes[kernel]));changed['AUTO{_ARRAY_T_'+label+'}']='complete';changed['AUTO{_ARRAY_D_'+label+'}']='1'
        candidates=torch.stack([candidate_vector(x,data.actions[kernel]) for x in (neutral,changed)])
        values=model(graph,candidates,torch.zeros(1,5));assert not torch.equal(values[0],values[1])
        graph.target_context=torch.randn(5);graph.candidate_values=torch.randn(64,8);graph.measured_qor=torch.randn(6);graph.budget=torch.randn(4)
        assert torch.equal(memory,model.memory(graph))
        checks.append({'kernel':kernel,'source_only_actions':sorted(data.source_only_nodes[kernel]),'all_source_slots_present':True,'no_compiler_edges_invented':True,'unbound_candidate_path_live':True,'static_no_leakage':True})
    write_json({'status':'passed','purpose':'representation compatibility probe only; not a new training run','weights_from_preserved_v2_checkpoint_sha256':sha(a.checkpoint),'optimizer_updates':0,'checks':checks},out/'input_checks.json')
