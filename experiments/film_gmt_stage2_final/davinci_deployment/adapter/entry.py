#!/usr/bin/env python3
"""Relocate recorded Liono filenames; execute the unchanged, pinned trainer."""
import hashlib,json,os,sys
from pathlib import Path

CONFIG=Path(__file__).with_name('paths.json')
config=json.loads(CONFIG.read_text())
deployment=Path(config['deployment'])
root=deployment/'code'
sys.path.insert(0,str(root))
from experiments.film_gmt_stage2_final import assets,train

mapping=sorted(config['mapping'].items(),key=lambda x:-len(x[0]))
def physical(value):
    text=os.fspath(value)
    for original,destination in mapping:
        if text==original or text.startswith(original+'/'):
            return Path(destination+text[len(original):])
    if text.startswith('/home/elvouvali/'):
        raise ValueError('No explicit deployment mapping for '+text)
    return Path(text)

# Only the receipt verifier's filename constructors and SHA filesystem access
# are relocated. Its control flow, original receipts and expected SHA values
# remain unchanged. Other modules retain the real pathlib.Path class.
original_sha=assets.sha
assets.Path=physical
assets.sha=lambda path:original_sha(physical(path))

def main():
    args=train.parse_args()
    head=train.get_git(root)
    if head['head']!=config['commit'] or head['status'] or head['dirty_diff']:
        raise RuntimeError('Deployment code is not the exact clean selected commit')
    if args.gpu_uuid!=config['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid']:
        raise RuntimeError('Only the explicitly selected Davinci V100 UUID is authorized')
    if args.mode!='audit':
        import torch
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(7,0):
            raise RuntimeError('Expected one visible Volta GPU, capability 7.0')
        if train.native_autocast_dtype(torch)!=torch.float16:
            raise RuntimeError('Davinci deployment requires native FP16')
        print('[DAVINCI_GPU]',json.dumps({'visible':os.environ['CUDA_VISIBLE_DEVICES'],'count':torch.cuda.device_count(),
          'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'autocast':'torch.float16'}),flush=True)
    print('[DEPLOYMENT]',json.dumps({'commit':head['head'],'path_map_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
      'core_entry_sha256':train.file_sha(root/'experiments/film_gmt_stage2_final/train.py'),
      'verifier_sha256':train.file_sha(root/'experiments/film_gmt_stage2_final/assets.py')}),flush=True)
    train.run(args)

if __name__=='__main__':main()
