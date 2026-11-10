"""Execute only the sealed development measurement plan on Coroni, sequentially."""
import argparse,json,math
from pathlib import Path
from experiments.refinement_search.coroni import CoroniOracle
from experiments.refinement_search.deployment import physical_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();m=json.loads(a.manifest.read_text());print('Sealed jobs',len(m['jobs']),'maximum300seconds each; finite anchor required per family',flush=True)
 if not a.execute:return
 a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'results.json').exists():print('PRESERVED completed measurement bank');return
 records=[];status={}
 for kernel in sorted({j['request']['kernel'] for j in m['jobs']}):
  jobs=[j for j in m['jobs'] if j['request']['kernel']==kernel];jobs.sort(key=lambda j:(not j['anchor'],j['id']));allowed=True
  for job in jobs:
   if not allowed:records.append({'id':job['id'],'status':'BLOCKED_BY_ANCHOR'});continue
   request=job['request'];template=Path(request['source_template']);oracle=CoroniOracle(request,template,[template.parent/name for name in request['headers_sha256']],m['runner'],a.out/'physical','architecture_v3_switch_bank_20261110',max_new_calls=min(8,len(m['jobs'])),prior_calls=m.get('protocol',{}).get('global_attempts_before',11),timeout=300)
   result=oracle(physical_identity(request,job['assignment']));records.append({'id':job['id'],'candidate_hash':job['candidate_hash'],'request':request,'assignment':job['assignment'],'entry':result})
   measure=result['measurement'];print('[MEASURE]',job['id'],measure,flush=True)
   if job['anchor'] and not (measure['synthesis_status']=='success' and isinstance(measure.get('latency_cycles'),int) and measure.get('resources') is not None):allowed=False;status[kernel]='BLOCKED: no finite anchor latency/exact counts'
  status.setdefault(kernel,'completed sealed comparisons')
  (a.out/'partial_results.json').write_text(json.dumps({'records':records,'families':status},indent=2)+'\n')
 ledger=a.out/'physical/physical_ledger.jsonl';result={'schema':'v3-measured-switch-bank-v1','manifest':str(a.manifest.resolve()),'manifest_sha256':__import__('hashlib').sha256(a.manifest.read_bytes()).hexdigest(),'records':records,'families':status,'physical_calls_reserved':len(ledger.read_text().splitlines()) if ledger.exists() else 0,'prior_calls_including_repeats':m.get('protocol',{}).get('global_attempts_before',11),'scope':'two development families, selected without QoR ranking; no locked test; no model superiority inferred'}
 (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',status,flush=True)
if __name__=='__main__':main()
