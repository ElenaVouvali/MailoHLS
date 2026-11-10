"""Append an authenticated held-out GAN pack without changing the frozen bank.
Compiler analysis uses the archive's explicitly non-bit-true AP analysis headers.
"""
import argparse,json,shutil,tarfile,hashlib
from pathlib import Path
import torch
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.structural_memory_v1 import build_memory_pack,adapt_for_xattn,load_frozen_encoder,sha256,ACTION_SLOTS,EVIDENCE_SCHEMA
GNN_SHA='ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162'
ARCHIVE_SHA='feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73'
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--parent-memory',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--pristine-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert sha256(a.archive)==ARCHIVE_SHA;assert sha256(a.checkpoint)==GNN_SHA
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);graphs=a.out/'graphs';graphs.mkdir();source=a.prepared/'sources/GAN';manifest=json.loads((a.parent_memory/'memory_manifest.json').read_text());assert manifest['gnn_checkpoint_sha256']==GNN_SHA
 with tarfile.open(a.archive) as tar:
  candidates=[m for m in tar.getmembers() if m.name.startswith('canonical/GAN-') and m.name.endswith('.json')];assert len(candidates)==1;raw=tar.extractfile(candidates[0]).read()
 graphfile=graphs/'GAN.json';graphfile.write_bytes(raw);graph=read_graph(graphfile);provenance=graph.graph['provenance'];assert provenance['source_sha256']==sha256(a.pristine_source);assert provenance['metadata_sha256']==sha256(source/'kernel_info.txt')
 assert provenance['local_header_sha256']=={h.name:sha256(h) for h in source.glob('*.h')}
 coverage=graph.graph['action_point_coverage']['records'];assert len(coverage)==11 and all(r['compiler_matched'] for r in coverage)
 encoder,identity=load_frozen_encoder(a.checkpoint,GNN_SHA)
 pack=build_memory_pack(graph,encoder,graph_file_sha256=sha256(graphfile),gnn_identity=identity,target='u200',clock_ns=10.,expected_graph_identity=graph.graph['graph_identity']);alternate=build_memory_pack(graph,encoder,graph_file_sha256=sha256(graphfile),gnn_identity=identity,target='zcu104',clock_ns=3.33,expected_graph_identity=graph.graph['graph_identity'])
 for key in ('action_memory','evidence_memory','evidence_valid_mask','evidence_type_ids','action_relation_mask','action_evidence_mask'):assert torch.equal(pack[key],alternate[key]),key
 c=adapt_for_xattn(pack,'C');payload={'node_embs':c['memory'][0].float().cpu().contiguous(),'node_embs_mask':c['memory_mask'][0].bool().cpu().contiguous(),'gnn_dim':128,'max_slots':64,'memory_interface':'rc3_action_evidence_c_v1','action_slot_count':64,'action_slot_mask':pack['action_valid_mask'][0].bool().cpu(),'evidence_start_slot':64,'action_relation_mask':c['action_relation_mask'][0].bool().cpu().contiguous(),'disable_pragma_injection':True,'evidence_metadata':pack['evidence_metadata'],'provenance':pack['provenance'],'evidence_schema':EVIDENCE_SCHEMA,'relation_schema_id':pack['relation_pack']['relation_schema_id'],'graph_identity':graph.graph['graph_identity'],'source_sha256':provenance['source_sha256'],'numerical_fidelity':graph.graph['numerical_fidelity']}
 for record in manifest['memory_files']:
  old=a.parent_memory/record['file'];assert sha256(old)==record['sha256'];shutil.copyfile(old,a.out/record['file'])
 torch.save(payload,a.out/'GAN.memory.pt');shutil.copyfile(a.parent_memory/'memory_manifest.json',a.out/'training_memory_manifest.json')
 extra={'kernel':'GAN','file':'GAN.memory.pt','sha256':sha256(a.out/'GAN.memory.pt'),'graph_file_sha256':sha256(graphfile),'extension_graph_file':'graphs/GAN.json','graph_identity':graph.graph['graph_identity'],'source_sha256':provenance['source_sha256'],'action_slots':int(pack['action_valid_mask'].sum()),'evidence_records':pack['evidence_count'],'key_count':int(c['memory'].shape[1]),'action_route_schema':pack['relation_pack']['schema'],'relation_schema_id':pack['relation_pack']['relation_schema_id']}
 extension={**manifest,'schema':'salt-hls-authenticated-heldout-memory-extension-v1','extension_kind':'authenticated_heldout_programs','parent_memory_manifest_file':'training_memory_manifest.json','parent_memory_manifest_sha256':sha256(a.parent_memory/'memory_manifest.json'),'extension_kernels':['GAN'],'memory_files':manifest['memory_files']+[extra],'extension_source_receipt':{'source_sha256':provenance['source_sha256'],'metadata_sha256':provenance['metadata_sha256'],'header_sha256':provenance['local_header_sha256'],'archive_member':candidates[0].name,'archive_sha256':ARCHIVE_SHA,'hardware_exact':graph.graph['numerical_fidelity']['hardware_exact'],'numerical_limit':graph.graph['numerical_fidelity']['reason'],'source_role':'External descriptive inference only; no measured labels or hardware request in memory'},'target_invariance':'Exact action/evidence vectors and routing masks across U20010ns/ZCU1043.33ns'}
 shutil.copyfile(a.archive,a.out/'authenticated_rc3_archive.tar.gz');extension['authenticated_archive_file']='authenticated_rc3_archive.tar.gz';extension['extension_source_receipts']={'gan':dict(extension['extension_source_receipt'],prompt_source_sha256=json.loads((a.prepared/'manifest.json').read_text())['prompt_source_sha256'])}
 (a.out/'memory_manifest.json').write_text(json.dumps(extension,indent=2,sort_keys=True)+'\n');print(json.dumps({'GAN':extra,'source_receipt':extension['extension_source_receipt'],'parent_manifest_sha256':extension['parent_memory_manifest_sha256']},indent=2))
if __name__=='__main__':main()
