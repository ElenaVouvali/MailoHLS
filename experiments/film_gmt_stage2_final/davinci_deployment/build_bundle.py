"""Assemble authenticated runtime closure, leaving tracked commit untouched."""
import hashlib,json,shutil,subprocess
from pathlib import Path

ROOT=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
HERE=Path(__file__).resolve().parent
BUNDLE=HERE/'bundle'
REMOTE='/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113'
PARENT=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
FILM=Path('/home/elvouvali/MailoHLS/worktrees/gnosis-stage1-film-ab/artifacts/gnosis_stage1_ab/stage1_ab_pilot_20261112_retry2/B')
HIST=Path('/home/elvouvali/MailoHLS')
PREFIX='artifacts/film_gmt_stage2_final_20261113_v1'
BANK=ROOT/PREFIX/'fallback_bank_original_anchors'
PREP=ROOT/PREFIX/'fallback_memory_original_anchors'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
mapping={str(ROOT):REMOTE+'/code',str(FILM):REMOTE+'/assets/film/B',str(PARENT):'/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run',str(HIST):REMOTE+'/assets/historical'}
def dest(p):
    s=str(p)
    for old,new in sorted(mapping.items(),key=lambda x:-len(x[0])):
        if s==old or s.startswith(old+'/'):
            physical=new+s[len(old):]
            if not physical.startswith(REMOTE+'/'):raise ValueError('Cannot copy reused parent')
            return BUNDLE/physical[len(REMOTE)+1:]
    raise ValueError(s)
receipt=json.loads((BANK/'final_assets.json').read_text())
identity=json.loads((PREP/'historical_identity.json').read_text())
files={Path(p):h for p,h in receipt['protected_files'].items()}
files[Path(identity['checkpoint_path'])]=identity['checkpoint_sha256']
for rec in identity['authenticated_training_files'].values():files[Path(rec['path'])]=rec['sha256']
files[HIST/'artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_normalization_stats.json']=identity['normalization_sha256']
for k,r in receipt['kernels'].items():files[HIST/'GNN_branch/MLIR_graphs'/(k+'.gexf')]=r['graph_sha256']
for folder in (BANK,PREP,FILM/'checkpoint-0600'):
    for p in folder.rglob('*'):
        if p.is_file():files.setdefault(p,sha(p))
files[FILM/'protocol.json']=sha(FILM/'protocol.json')
for p,h in files.items():
    if sha(p)!=h:raise ValueError('Local protected SHA differs: '+str(p))
    d=dest(p);d.parent.mkdir(parents=True,exist_ok=True)
    # Authenticated payloads are materialized regular files; record source links.
    shutil.copy2(p,d)
parent_files={str(p.relative_to(PARENT)):sha(p) for p in PARENT.iterdir() if p.is_file()}
cache=Path('/home/elvouvali/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912')
base={str(p.relative_to(cache)):sha(p) for p in cache.rglob('*') if p.is_file()}
write(BUNDLE/'adapter/paths.json',{'deployment':REMOTE,'mapping':mapping,'commit':'6ee1390611c9047a4b8e2b07200d223b93ee8d16','gpu_uuid':'GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'})
write(BUNDLE/'receipts/source_artifacts.json',{'files':[{'original':str(p),'relative':str(dest(p).relative_to(BUNDLE)),'sha256':h,'original_symlink':str(p.readlink()) if p.is_symlink() else None} for p,h in sorted(files.items())],
  'reused_parent':{'path':mapping[str(PARENT)],'files':parent_files},'reused_base':{'path':str(cache).replace('/home/elvouvali/','/home/evouvali/'),'revision':cache.name,'files':base}})
print(json.dumps({'authenticated_artifact_files':len(files),'payload_bytes':sum(p.stat().st_size for p in files),'parent_files':len(parent_files),'base_files':len(base)}))
