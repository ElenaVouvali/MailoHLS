"""Isolated source-only frozen GNN/GMT qualification; no optimizer or fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--kernel',required=True)
parser.add_argument('--graph',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--global-binding',type=Path)
a=parser.parse_args()
R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113'
sys.path.insert(0,str(R))
import torch
from experiments.historical_mlir_gmt_gemv import historical as hist
from experiments.historical_mlir_gmt_gemv.historical_gmt_slots import HistoricalRoleGMT,cohort_action_mapping,make_evidence,device_data,GROUPS
torch.set_num_threads(4)
runtime=hist.load()
graph,pack=hist.export_graph(a.graph,a.kernel,runtime)
manifest=json.loads((P/'sources/SOURCES_MANIFEST.json').read_text())[a.kernel]
if a.global_binding:
    from compatibility_hooks_v2 import install_gmt_scope_extension
    from experiments.historical_mlir_gmt_gemv import historical_gmt_slots as slots
    gp=pack['graph_provenance']
    record={key:gp[key] for key in ['source_sha256','action_sha256','generator_sha256','mlir_sha256','cgeist_sha256']}
    record['native_binding']=json.loads(a.global_binding.read_text())
    extension=install_gmt_scope_extension({hist.sha(a.graph):record},a.out.parent/'generated_gmt_scope_compat')
    cohort_action_mapping,make_evidence=slots.cohort_action_mapping,slots.make_evidence
    (a.out.parent/'GMT_SCOPE_EXTENSION.json').write_text(json.dumps(extension,indent=2,sort_keys=True)+'\n')
mapping=cohort_action_mapping(graph,pack)
assert set(m['action'] for m in mapping)==set(manifest['actions'])
assert pack['node_embs'].shape==(64,128) and torch.isfinite(pack['node_embs']).all()
assert int(pack['node_embs_mask'].sum())==len(manifest['actions'])
for key,value in [('source_sha256',manifest['source_sha256']),('action_sha256',manifest['metadata_sha256']),
                  ('generator_sha256','5e4f93c89ce4c275c04ff6ed4dd81a17858647a55b0abec0846b8da240da3b05')]:
    assert pack['graph_provenance'][key]==value,(key,pack['graph_provenance'][key])
records,evidence=make_evidence(graph,pack,mapping,a.graph)
assert torch.equal(records['anchors'],pack['node_embs'].float())
checkpoint=R/'artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/gmt_selected.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['steps']==600
gmt=HistoricalRoleGMT().eval().requires_grad_(False)
gmt.load_state_dict(state['model'],strict=True)
with torch.inference_mode():summary,present,_,_=gmt(device_data(records,'cpu'))
assert tuple(summary.shape)==(len(mapping),3,128) and torch.isfinite(summary).all()
assert present.any(1).all()
groups={m['action']:{GROUPS[g]:int(records['route'][i,g].sum()) for g in range(3)} for i,m in enumerate(mapping)}
result={'status':'PASS_FROZEN_GNN_GMT_SINGLE_KERNEL','kernel':a.kernel,'actions':len(mapping),
 'anchor_shape':list(pack['node_embs'].shape),'summary_shape':list(summary.shape),
 'evidence_records':len(evidence['records']),'groups':groups,
 'graph_sha256':hist.sha(a.graph),'gnn_checkpoint_sha256':hist.sha(hist.CKPT),'gmt_checkpoint_sha256':hist.sha(checkpoint),
 'source_sha256':manifest['source_sha256'],'metadata_sha256':manifest['metadata_sha256'],
 'representation':'evaluation-only compatibility, not identical historical preprocessing',
 'training_performed':False,'evaluation_labels_read':False}
assert not a.out.exists()
a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2),flush=True)
