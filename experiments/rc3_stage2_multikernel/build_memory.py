"""Build the target-independent Memory C bank for the frozen family-balanced cohort."""
from __future__ import annotations
import argparse, hashlib, json, tarfile
from pathlib import Path
import torch
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.structural_memory_v1 import (
    ACTION_SLOTS, EVIDENCE_CAPACITY, EVIDENCE_SCHEMA, adapt_for_xattn,
    build_memory_pack, load_frozen_encoder, sha256,
)
ROOT=Path(__file__).resolve().parents[2]
DATA_AUDIT=ROOT/'artifacts/rc3_stage2_multikernel/v1/audit/coverage.json'
ARCHIVE=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
GNN=Path('/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/compiler_selected.pt')
GNN_SHA='ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162'
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--audit',type=Path,default=DATA_AUDIT)
 ap.add_argument('--archive',type=Path,default=ARCHIVE)
 ap.add_argument('--checkpoint',type=Path,default=GNN)
 ap.add_argument('--out',type=Path,default=ROOT/'artifacts/rc3_stage2_multikernel/v1/memory_c')
 a=ap.parse_args()
 audit=json.loads(a.audit.read_text()); kernels=set(audit['one_kernel_per_training_family'].values())|set(audit['splits']['validation']['kernels'])
 a.out.mkdir(parents=True,exist_ok=True); graph_dir=a.out/'graphs'; graph_dir.mkdir(exist_ok=True); memory_dir=a.out/'memory'; memory_dir.mkdir(exist_ok=True)
 model,gnn_identity=load_frozen_encoder(a.checkpoint,GNN_SHA)
 archive_sha=sha256(a.archive); packs=[]; inputs=[]
 with tarfile.open(a.archive,'r:gz') as tar:
  members={m.name: m for m in tar.getmembers() if m.name.startswith('canonical/') and m.name.endswith('.json')}
  for kernel in sorted(kernels):
   rec=audit['inventory'][kernel]['graph']; member=members[rec['archive_member']]
   raw=tar.extractfile(member).read(); graph_path=graph_dir/(kernel+'.json'); graph_path.write_bytes(raw)
   file_sha=hashlib.sha256(raw).hexdigest()
   if file_sha != hashlib.sha256(tar.extractfile(member).read()).hexdigest(): raise ValueError('graph copy hash mismatch')
   graph=read_graph(graph_path)
   if graph.graph['graph_identity'] != rec['graph_identity']: raise ValueError(f'{kernel}: graph identity mismatch')
   pack=build_memory_pack(graph,model,graph_file_sha256=file_sha,gnn_identity=gnn_identity,
       target='u200',clock_ns=10.0,expected_graph_identity=rec['graph_identity'])
   # Compiler-only memory must be exactly target invariant; verify both action
   # and node/edge evidence vectors on one contrasting overlay per source.
   alternate=build_memory_pack(graph,model,graph_file_sha256=file_sha,gnn_identity=gnn_identity,
       target='zcu104',clock_ns=3.33,expected_graph_identity=rec['graph_identity'])
   for field in ('action_memory','evidence_memory','evidence_valid_mask','evidence_type_ids','action_relation_mask','action_evidence_mask'):
    if not torch.equal(pack[field],alternate[field]): raise ValueError(f'{kernel}: target changed static memory field {field}')
   c=adapt_for_xattn(pack,'C'); rel=pack['relation_pack']
   memory_file=memory_dir/(kernel+'.memory.pt')
   loader_pack={'node_embs':c['memory'][0].float().cpu().contiguous(),
     'node_embs_mask':c['memory_mask'][0].bool().cpu().contiguous(),'gnn_dim':128,'max_slots':ACTION_SLOTS,
     'memory_interface':'rc3_action_evidence_c_v1','action_slot_count':ACTION_SLOTS,
     'action_slot_mask':pack['action_valid_mask'][0].bool().cpu(),
     'evidence_start_slot':ACTION_SLOTS,'action_relation_mask':c['action_relation_mask'][0].bool().cpu().contiguous(),
     'disable_pragma_injection':True,'evidence_metadata':pack['evidence_metadata'],
     'provenance':pack['provenance'],'evidence_schema':EVIDENCE_SCHEMA,
     'relation_schema_id':rel['relation_schema_id'],'graph_identity':graph.graph['graph_identity'],
     'source_sha256':graph.graph.get('provenance',{}).get('source_sha256')}
   torch.save(loader_pack,memory_file)
   packs.append({'kernel':kernel,'file':memory_file.name,'sha256':sha256(memory_file),
     'graph_file_sha256':file_sha,'graph_identity':graph.graph['graph_identity'],
     'source_sha256':loader_pack['source_sha256'],'action_slots':int(pack['action_valid_mask'].sum()),
     'evidence_records':pack['evidence_count'],'key_count':int(c['memory'].shape[1]),
     'action_route_schema':rel['schema'],'relation_schema_id':rel['relation_schema_id']})
   inputs.append({'kernel':kernel,'archive_member':member.name,'graph_file':str(graph_path.relative_to(a.out)),
     'graph_file_sha256':file_sha,'graph_identity':graph.graph['graph_identity']})
 manifest={'schema':'salt-hls-stage2-rc3-memory-bank-v1','memory_interface':'rc3_action_evidence_c_v1',
   'action_relation_schema':'salt-hls-rc3-action-relations-v1','evidence_schema':EVIDENCE_SCHEMA,
   'evidence_capacity':EVIDENCE_CAPACITY,'max_slots':ACTION_SLOTS,'mem_dim':128,
   'compiler_graph_archive_sha256':archive_sha,'gnn_checkpoint_sha256':sha256(a.checkpoint),
   'gnn_identity':gnn_identity,'target_independent':True,'hardware_messages':False,
   'training_objective_caveat':'V4b frozen encoder used physical-QoR/ADP-oriented supervision, not latency qualification.',
   'graphs':inputs,'memory_files':packs}
 (memory_dir/'memory_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'memory_dir':str(memory_dir),'manifest':str(memory_dir/'memory_manifest.json'),'kernels':len(packs),
  'evidence_count_range':[min(x['evidence_records'] for x in packs),max(x['evidence_records'] for x in packs)],
  'max_key_count':max(x['key_count'] for x in packs),'target_invariance':'exact vectors and masks for both overlays'},indent=2))
if __name__=='__main__': main()
