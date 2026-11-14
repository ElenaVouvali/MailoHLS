"""Verify every payload and reused asset before qualification or launch."""
import hashlib,json,subprocess,sys
from pathlib import Path
deployment=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((deployment/'receipts/source_artifacts.json').read_text())
checks=[]
for r in manifest['files']:checks.append((deployment/r['relative'],r['sha256']))
for key in ('reused_parent','reused_base'):
    spec=manifest[key]
    for relative,digest in spec['files'].items():checks.append((Path(spec['path'])/relative,digest))
for path,digest in checks:
    if sha(path)!=digest:raise ValueError('Transfer/reused artifact hash differs: '+str(path))
expected=json.loads((deployment/'receipts/payload_sha256.json').read_text())
for rel,digest in expected.items():
    if sha(deployment/rel)!=digest:raise ValueError('Deployment payload differs: '+rel)
root=deployment/'code'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
if head!='6ee1390611c9047a4b8e2b07200d223b93ee8d16' or status:raise RuntimeError('Git identity/cleanliness differs: '+status)
print(json.dumps({'status':'PASS','commit':head,'payload_files':len(expected),'authenticated_asset_checks':len(checks),
 'source_artifacts_sha256':sha(deployment/'receipts/source_artifacts.json'),
 'payload_manifest_sha256':sha(deployment/'receipts/payload_sha256.json')},indent=2))
