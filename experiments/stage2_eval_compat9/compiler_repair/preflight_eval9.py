"""CPU inference-input qualification using the unchanged production memory loader.

No decoder checkpoint is loaded or executed; run the target inference program's
own preflight again after copying the new, explicitly versioned inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113'
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=R)
parser.add_argument('--sources',type=Path,default=P/'sources')
parser.add_argument('--memory',type=Path,default=P/'export_compat9_v2/memory_B')
parser.add_argument('--out',type=Path)
a=parser.parse_args();R=a.repo.resolve();sys.path.insert(0,str(R))
head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],universal_newlines=True).strip()
if head!='6ee1390611c9047a4b8e2b07200d223b93ee8d16':raise RuntimeError('Unexpected Stage-2 code version')
subprocess.run(['git','-C',str(R),'diff','--exit-code','--','LLM_branch/common/structural_memory.py'],check=True)
import torch
from LLM_branch.common import structural_memory as sm,directive_domains as dd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((a.memory/'memory_manifest.json').read_text())
contract=json.loads((a.memory/'EVAL_COMPATIBILITY.json').read_text())
source_manifest=json.loads((a.sources/'SOURCES_MANIFEST.json').read_text())
kernels={'GAN','Kalman','chstone-aes','chstone-jpeg','covariance','jacobi-2d','rosetta-3d-rendering','syr2k','trmm-opt'}
assert set(source_manifest)==set(contract['kernels'])==kernels
assert set(manifest['files'])=={k+'.memory.pt' for k in kernels}
assert {p.name for p in a.memory.glob('*.memory.pt')}==set(manifest['files'])
assert sha(a.memory/'memory_manifest.json')==contract['memory_manifest_sha256']
assert sha(a.sources/'SOURCES_MANIFEST.json')==contract['source_manifest_sha256']
assert manifest['kind']=='B_GMT' and manifest['schema']=='mailohls-matched-historical-gmt-stage2-v1'
assert manifest['gnn_checkpoint_sha256']=='34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623'
assert manifest['gmt_checkpoint_sha256']=='1d4cdf3071375349b94809062d05d30ad3e5c981bb6e2885b6f99c9e94398c12'
bank,dim=sm.load_memory_bank(str(a.memory),expected_mem_dim=128,expected_max_slots=64,require_pragma_free_memory=True)
assert dim==128
report={}
for k in sorted(kernels):
    folder=a.sources/k;source=next(folder.glob('source.*'));rec=source_manifest[k]
    assert sha(source)==rec['source_sha256'] and sha(folder/'kernel_info.txt')==rec['metadata_sha256']
    declared=dd.load_source_actions(k,a.sources)
    assert set(declared)==set(rec['actions'])
    filename=k+'.memory.pt';assert sha(a.memory/filename)==manifest['files'][filename]
    pack=torch.load(a.memory/filename,map_location='cpu',weights_only=False)
    assert pack['historical_graph_sha256']
    assert pack['gnn_checkpoint_sha256']==manifest['gnn_checkpoint_sha256']
    assert pack['gmt_checkpoint_sha256']==manifest['gmt_checkpoint_sha256']
    active=pack['action_slot_mask'].nonzero().flatten().tolist()
    assert {i+1 for i in active}=={int(action[1:]) for action in declared}
    kv,mask,relation=sm.get_structural_memory_pack_for_kernel(bank,k,64,128,'compiler_relational')
    assert torch.isfinite(kv).all()
    assert tuple(relation.shape)==(1,len(pack['node_embs']),len(pack['node_embs']))
    assert relation[0,active].any(1).all()
    assert not relation[0,~mask[0]].any() and not relation[0,:,~mask[0]].any()
    groups={action:[] for action in declared}
    for i,entry in enumerate(pack['evidence_metadata']):
        aid='L'+str(entry['action_slot']+1);assert aid in groups and entry['evidence_count']>0
        assert entry['group'] in ['memory','compute_dependence','control']
        assert relation[0,entry['action_slot'],64+i]
        groups[aid].append(entry['group'])
    assert all(1<=len(set(values))==len(values)<=3 for values in groups.values())
    report[k]={'actions':len(declared),'memory_tokens':len(pack['node_embs']),'groups':groups,
               'source_sha256':rec['source_sha256'],'memory_sha256':manifest['files'][filename]}
result={'status':'PASS_CPU_NINE_KERNEL_INFERENCE_INPUTS','scope':'Production structural memory loader + source/action/routing/hash contract; no decoder execution',
        'repo_head':head,'preflight_code_sha256':sha(__file__),
        'production_memory_loader_sha256':sha(R/'LLM_branch/common/structural_memory.py'),
        'kernel_count':9,'memory_manifest_sha256':sha(a.memory/'memory_manifest.json'),
        'source_manifest_sha256':sha(a.sources/'SOURCES_MANIFEST.json'),'kernels':report,
        'compatibility_representation_acknowledged':True,'target_server_inference_preflight_still_required':True,
        'kernel_all_invoked':False,'gpu_used':False,'model_or_weight_changes':False}
if a.out:
    if a.out.exists():raise FileExistsError('Do not overwrite a preflight receipt')
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
