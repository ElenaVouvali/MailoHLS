"""Read-only inspection, Python 3.6 compatible. Run through SSH stdin."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/home/bonus24/Application')
KERNELS={'GAN':('network.cpp','forward_propagation'),
 'serrano-kalman-filter':('kernel_placeholders.cpp','krnl_KALMAN'),
 'trmm-opt':('kernel.cpp','kernel_trmm'),'chstone-aes':('kernel_placeholders.c','chstone_aes'),
 'jacobi-2d':('kernel_placeholders.cpp','kernel_jacobi_2d'),
 'syr2k':('kernel_placeholders.cpp','kernel_syr2k'),
 'covariance':('kernel_placeholders.cpp','kernel_covariance'),
 'chstone-jpeg':('kernel_placeholders.c','chstone_jpeg'),
 'rosetta-3d-rendering':('kernel_placeholders.cpp','rendering')}
out={}
for k,(name,top) in KERNELS.items():
    folder=ROOT/k;files={};interesting={}
    for p in sorted(folder.iterdir()):
        if not p.is_file():continue
        files[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        if p.suffix in ('.c','.cpp','.h') or p.name in ('kernel_info.txt','README.md','ACTION_AUDIT.json'):
            lines=p.read_text(errors='replace').splitlines()
            if p.name in ('kernel_info.txt','README.md','ACTION_AUDIT.json'):
                interesting[p.name]=list(enumerate(lines,1))
            elif p.name in (name,'network.h','kernel.cpp','kernel.c'):
                hits=set()
                for i,line in enumerate(lines):
                    if re.search(r'^\s*#\s*(?:include|define)|(?:/\*)?\bL\d+:|pragma|\b(?:void|int)\s+(?:'+top+r')\s*\(',line):
                        hits.update(range(max(0,i-1),min(len(lines),i+3)))
                interesting[p.name]=[(i+1,lines[i]) for i in sorted(hits)]
    out[k]={'folder':str(folder),'candidate_source':name,'candidate_top':top,'files':files,'source_notes':interesting}
print(json.dumps(out,indent=2,sort_keys=True))
