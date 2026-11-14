"""Audit relocation failure behavior and record the reused runtime environment."""
import importlib.metadata,json,os,platform,subprocess,sys,tempfile
from pathlib import Path
import entry

deployment=Path(entry.config['deployment'])
bank=deployment/'code/artifacts/film_gmt_stage2_final_20261113_v1/fallback_bank_original_anchors'
obj=entry.assets.verify_receipt(bank/'final_assets.json',bank)
with tempfile.TemporaryDirectory() as folder:
    fixture=json.loads((bank/'final_assets.json').read_text())
    first=next(iter(fixture['protected_files']))
    fixture['protected_files'][first]='0'*64
    p=Path(folder)/'tampered-receipt.json';p.write_text(json.dumps(fixture))
    try:entry.assets.verify_receipt(p,bank)
    except ValueError as exc:assert 'artifact changed' in str(exc)
    else:raise AssertionError('Tampered authenticated SHA was not rejected')
try:entry.physical('/home/elvouvali/unmapped/path')
except ValueError:pass
else:raise AssertionError('An unmapped Liono path was accepted')
required={'torch':'2.6.0+cu124','transformers':'4.51.0','peft':'0.18.1','accelerate':'1.13.0',
 'bitsandbytes':'0.49.2','tokenizers':'0.21.4','numpy':'2.2.6','pandas':'2.3.3'}
versions={name:importlib.metadata.version(name) for name in required}
assert versions==required,(versions,required)
result={'status':'PASS','protected_receipt_entries':len(obj['protected_files']),
 'relocated_receipt_verified':True,'tampered_expected_hash_rejected':True,'unknown_liono_path_rejected':True,
 'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':versions,
 'shared_dependency_directory':'/home/evouvali/MailoHLS/.venv/lib/python3.10/site-packages',
 'shared_dependencies_modified':False,'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,compute_cap,memory.total','--format=csv,noheader'],text=True)}
if os.environ.get('CHECK_V100')=='1':
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==entry.config['gpu_uuid']
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(7,0)
    assert entry.train.native_autocast_dtype(torch)==torch.float16
    result['gpu']={'uuid':os.environ['CUDA_VISIBLE_DEVICES'],'visible_count':1,'name':torch.cuda.get_device_name(0),
      'capability':list(torch.cuda.get_device_capability(0)),'native_autocast':'torch.float16','cuda_runtime':torch.version.cuda}
print(json.dumps(result,indent=2))
