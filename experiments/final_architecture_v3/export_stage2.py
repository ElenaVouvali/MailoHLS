"""Complete a portable research checkpoint without changing tensors/contracts."""
import argparse,hashlib,json,shutil
from pathlib import Path
FILES=('adapter_model.safetensors','adapter_config.json','structural_xattn.pt','training_contract.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def export(source,registry,out):
 source,registry,out=map(Path,(source,registry,out));contract=json.loads((source/'training_contract.json').read_text());assert contract['stage']=='stage2'
 if sha(registry)!=contract['directive_domain_registry_sha256']:raise ValueError('Registry differs from saved contract')
 expected={n:sha(source/n) for n in FILES}|{'directive_domain_registry.json':sha(registry)}
 if out.exists():
  if any(not (out/n).is_file() or sha(out/n)!=h for n,h in expected.items()):raise ValueError('Existing export changed')
  return out
 out.mkdir(parents=True)
 for n in FILES:shutil.copyfile(source/n,out/n)
 shutil.copyfile(registry,out/'directive_domain_registry.json');(out/'export_manifest.json').write_text(json.dumps({'schema':'v3-checkpoint-dependency-export-v1','source':str(source.resolve()),'files':expected,'tensors_and_training_contract_unchanged':True,'reason':'Trainer checkpoint/best directories omit the registry artifact stored in run root'},indent=2)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(export(a.source,a.registry,a.out))
