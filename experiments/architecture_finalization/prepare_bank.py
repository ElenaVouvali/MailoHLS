"""Seal shared candidates from historical validation predictions; no model scoring."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/architecture_finalization/bank'
CACHE=ROOT/'artifacts/architecture_finalization/coroni_cache/bundle'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 old=json.loads((CACHE/'manifest.json').read_text())
 chosen=[j for j in old['jobs'] if (j['kernel']=='machsuite-viterbi' and j['clock_ns']==10) or (j['kernel']=='rodinia_pathfinder_0_baseline_0' and j['part'].startswith('xcu200'))]
 assert len(chosen)==4
 from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES
 jobs=[]
 for src in sorted(chosen,key=lambda x:(x['kernel']=='machsuite-viterbi',x['job_id'])):
  for part,clock in [('xczu7ev-ffvc1156-2-e',10.),('xczu7ev-ffvc1156-2-e',5.),('xcu200-fsgd2104-2-e',10.)]:
   j=dict(src); j['job_id']='bank_%02d'%(len(jobs)+1); j['part']=part;j['clock_ns']=clock
   j['capacities']=DEVICE_RESOURCES[part];j['budget']=j['capacities'];j['request_number']=len(jobs)+1
   j['candidate_origin_job']=src['job_id'];j['candidate_origin_manifest_sha256']=sha(CACHE/'manifest.json')
   j['assignment']=src['attributions'][0]['prediction']
   directory=OUT/'jobs'/j['job_id'];shutil.copytree(CACHE/'jobs'/src['job_id'],directory)
   assert sha(directory/j['source'])==j['materialized_source_sha256']
   assert all(sha(directory/n)==h for n,h in j['headers'].items())
   jobs.append(j)
 manifest={'schema':'architecture-switch-bank-v1','settings':old['settings'],'jobs':jobs,'source':'frozen V4 validation outputs; two distinct configs per source; complete assignments preserved','pool_scope':'two validation families, 2 configs each, 3 physical contexts; no test','timeout_seconds':300,'budget_views_rule':'full-device fractions for deployment; common componentwise ZCU104 limits for device contrast; derive one separating component limit only from exact counts before model scoring; otherwise axis unresolved','candidate_selection':'Viterbi ZCU104/10 single and multi; Pathfinder U200/10 single and multi; no new outcome selection'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 shutil.copy2(CACHE/'run_batch.py',OUT/'run_batch.py')
 # First job is the reference cost probe. Full bundle remains sealed separately.
 first=OUT.parent/'reference_probe';shutil.copytree(OUT,first)
 (first/'manifest.json').write_text(json.dumps(dict(manifest,jobs=jobs[:1]),indent=2)+'\n')
 for folder in (OUT,first):
  files=sorted(p for p in folder.rglob('*') if p.is_file())
  (folder/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(folder))+'\n' for p in files))
 print(json.dumps({'jobs':len(jobs),'first_job':jobs[0]['job_id'],'manifest_sha256':sha(OUT/'manifest.json')},indent=2))
if __name__=='__main__':main()
