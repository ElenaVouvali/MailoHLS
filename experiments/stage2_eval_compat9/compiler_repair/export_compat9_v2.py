"""Verify compatibility provenance, then execute the unmodified original exporter."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from compatibility_hooks_v2 import sha,install_gmt_scope_extension,GENERATOR_SHA

R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113';W=P/'compat9_v2_work'
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=P/'export_compat9_v2')
a=parser.parse_args();out=a.out.resolve()
if out.exists():raise FileExistsError('Fresh export output required')
graphs=P/'graphs_compat9_v2'
receipt_file=graphs/'EVAL_COMPAT9_V2_RECEIPT.json'
receipt=json.loads(receipt_file.read_text())
if receipt['schema']!='mailohls-stage2-eval-compat9-v2':raise RuntimeError('Wrong graph compatibility contract')
if receipt['runtime_hook_sha256']!=sha(W/'compatibility_hooks_v2.py'):raise RuntimeError('Runtime hook changed')
if receipt['source_manifest_sha256']!=sha(P/'sources/SOURCES_MANIFEST.json'):raise RuntimeError('Source manifest changed')
original=Path('/home/elvouvali/export_stage2_eval_gmt.py')
if receipt['original_exporter_sha256']!=sha(original):raise RuntimeError('Original exporter changed')
for k,record in receipt['kernels'].items():
    if sha(graphs/(k+'.gexf'))!=record['graph_sha256']:raise RuntimeError('Graph changed: '+k)
    if record['variant']=='unchanged_seven_control' and sha(P/'graphs_retry1'/(k+'.gexf'))!=record['graph_sha256']:
        raise RuntimeError('Original seven control changed: '+k)
    if record['generator_sha256']!=GENERATOR_SHA:raise RuntimeError('Generator identity differs')
proof=receipt['kernels']['rosetta-3d-rendering']
if sha(graphs/'ROSETTA_NATIVE_BINDING.json')!=proof['native_binding_sha256']:raise RuntimeError('Native global proof changed')
sys.path.insert(0,str(R))
import torch
torch.set_num_threads(4)
extension=install_gmt_scope_extension({proof['graph_sha256']:proof},W/'generated_gmt_scope_export_v2')
print('[COMPAT9 V2] Original exporter unchanged; explicit verified Rosetta-global acceptance only',flush=True)
sys.argv=[str(original),'--repo',str(R),'--sources',str(P/'sources'),'--graphs',str(graphs),
          '--assets-root','/home/elvouvali/MailoHLS','--out',str(out)]
runpy.run_path(str(original),run_name='__main__')
manifest_path=out/'memory_B/memory_manifest.json'
manifest=json.loads(manifest_path.read_text())
expected={k+'.memory.pt' for k in receipt['kernels']}
if set(manifest['files'])!=expected or len(expected)!=9:raise RuntimeError('Nine memories required')
if {p.name for p in (out/'memory_B').glob('*.memory.pt')}!=expected:raise RuntimeError('Unexpected/missing memory files')
summaries={}
for filename,digest in manifest['files'].items():
    file=out/'memory_B'/filename
    if sha(file)!=digest:raise RuntimeError('Memory SHA mismatch: '+filename)
    pack=torch.load(file,map_location='cpu',weights_only=False)
    base=torch.load(out/'memory_A'/filename,map_location='cpu',weights_only=False)
    kernel=filename.removesuffix('.memory.pt') if hasattr(filename,'removesuffix') else filename[:-10]
    actions=receipt['kernels'][kernel]['actions']
    if pack['node_embs'].shape[1]!=128 or int(pack['action_slot_mask'].sum())!=actions:raise RuntimeError('Action memory dimensions differ')
    if not torch.equal(pack['node_embs'][:64],base['node_embs']):raise RuntimeError('GMT altered historical GNN anchors')
    if not torch.isfinite(pack['node_embs']).all():raise RuntimeError('Nonfinite memory')
    metadata=pack['evidence_metadata'];active=pack['action_slot_mask'].nonzero().flatten().tolist()
    if {m['action_slot'] for m in metadata}!=set(active):raise RuntimeError('Action lacks genuine evidence summary')
    for item in metadata:
        if item['evidence_count']<=0:raise RuntimeError('Empty evidence summary')
    if len(metadata)>actions*3:raise RuntimeError('Too many evidence groups')
    summaries[kernel]={'actions':actions,'memory_tokens':len(pack['node_embs']),'gmt_summary_keys':len(metadata),
                       'memory_sha256':digest,'anchors_preserved':True}
contract={'schema':'mailohls-stage2-eval-compat9-export-v2',
 'status':'PASS_NINE_COMPATIBILITY_MEMORIES_NOT_IDENTICAL_HISTORICAL_REPRESENTATION',
 'graph_receipt_sha256':sha(receipt_file),'source_manifest_sha256':receipt['source_manifest_sha256'],
 'original_exporter_sha256':sha(original),'export_wrapper_sha256':sha(__file__),
 'runtime_hook_sha256':sha(W/'compatibility_hooks_v2.py'),'runtime_scope_extension':extension,
 'memory_manifest_sha256':sha(manifest_path),'gnn_checkpoint_sha256':manifest['gnn_checkpoint_sha256'],
 'gmt_checkpoint_sha256':manifest['gmt_checkpoint_sha256'],'seven_control_graphs_unchanged':True,
 'source_templates_or_metadata_modified':False,'gnn_gmt_or_stage2_retrained':False,
 'memory_A_B_anchor_bytes_identical':True,'kernels':summaries,
 'representation_shift':['Kalman: opt-in source storage preservation through ordinary post-pass pipeline',
                         'Rosetta: exact static global/source-access binding and narrowly extended GMT provenance acceptance'],
 'remaining_scope_limitations':proof['scope_limitations']}
(out/'COMPAT9_V2_EXPORT_PASS.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
(out/'memory_B/EVAL_COMPATIBILITY.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
print('[9/9 COMPATIBILITY EXPORT PASS]',out/'memory_B',flush=True)
