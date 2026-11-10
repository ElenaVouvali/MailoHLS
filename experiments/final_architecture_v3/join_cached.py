"""Exact current-generated assignment joins to authenticated historical physical receipts.
Budgets are re-evaluated from exact counts; physical labels do not change by fiat.
"""
import argparse,hashlib,json,re
from pathlib import Path
from experiments.refinement import configurations as c,synthesis as hls
from experiments.final_architecture_20261109.analyze_synthesis import batch

def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--requests',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);p.add_argument('--clock5-results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();requests={r['context_id']:r for r in c.read_jsonl(a.requests)};history=[];bundles=[]
 for bundle,result in [('checkpoint_0_150_synthesis_v3_reference','coroni_checkpoint_results'),('checkpoint_0_150_u200_pathfinder','coroni_u200_results'),('checkpoint_pathfinder_clock5',None)]:
  b=a.historical/bundle;res=a.historical/result if result else a.clock5_results;m=json.loads((b/'manifest.json').read_text())
  if any(not (res/j['job_id']/'run_hls.tcl').exists() for j in m['jobs']):
   # Read-only overlay: the earlier selective transfer omitted Tcl. The sealed bundle Tcl must exactly match its authenticated runtime receipt hash.
   overlay=Path(__file__).resolve().parents[2]/'artifacts/final_architecture_v3/cached_join_overlays'/bundle;overlay.mkdir(parents=True,exist_ok=True)
   for f in res.iterdir():
    if f.is_dir():
     dest=overlay/f.name;dest.mkdir(exist_ok=True)
     for child in f.iterdir():
      if not (dest/child.name).exists():(dest/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
     tcl=b/'jobs'/f.name/'run_hls.tcl'
     if not (dest/'run_hls.tcl').exists() and tcl.exists():(dest/'run_hls.tcl').symlink_to(tcl.resolve())
    elif not (overlay/f.name).exists():(overlay/f.name).symlink_to(f.resolve())
   res=overlay
  records=batch(b,res);history.extend((r,next(j for j in m['jobs'] if j['job_id']==r['job_id']),b,res) for r in records);bundles.append({'bundle_sha256':c.sha256(b/'manifest.json'),'results_provenance_sha256':c.sha256(res/'batch_provenance.json')})
 rows=[]
 for model in ('stage1','stage2','sft','stage3'):
  for generated in c.read_jsonl(a.generated/(model+'.jsonl')):
   req=requests[generated['context_id']];candidate=generated['candidates'][0];assignment=c.parse_assignment('\n'.join(line for line in candidate['prediction'].splitlines() if not re.fullmatch(r'<L[1-9][0-9]*>',line.strip())));material=hls.materialize(req['input'],'\n'.join(f'{k} = {v}' for k,v in assignment.items()));source_digest=hashlib.sha256(material.encode()).hexdigest();matches=[]
   for physical,job,b,res in history:
    if physical['kernel']!=req['kernel_name'] or physical['part']!=req['device'] or physical['clock_ns']!=float(req['selected_clock_period']) or job['assignment']!=assignment or physical['materialized_source_sha256']!=source_digest:continue
    source=Path(req['source_file']);metadata=source.parent/'kernel_info.txt'
    if c.sha256(metadata)!=job['metadata_sha256'] or {p.name:c.sha256(p) for p in source.parent.glob('*.h')}!=job['headers']:continue
    tcl=(res/job['job_id']/'run_hls.tcl').read_text()
    if not all(setting in tcl for setting in c.SETTINGS):continue
    # Existing parser authenticated tool2021.1/report/settings/part/clock, source and receipt/log witnesses.
    limits={k:int(req[c.mc.AVAIL_FIELD_BY_RESOURCE[k]]) for k in c.RESOURCE_KEYS};physical={**physical,'current_budgets':limits,'current_resource_feasible':all(physical['resources'][k]<=limits[k] for k in limits) if physical['resources'] else None}
    matches.append(physical)
   rows.append({'model':model,'context_id':generated['context_id'],'normalized_assignment':assignment,'materialized_source_sha256':source_digest,'exact_receipts':matches,'missing_status':'unknown' if not matches else None,'latency_regret':None,'regret_scope':'No shared measured contender pool for this generated assignment; exact receipt alone cannot establish regret.'})
 result={'schema':'v3-exact-historical-physical-joins-v1','input_bundles':bundles,'rows':rows,'matched_model_request_rows':sum(bool(r['exact_receipts']) for r in rows),'scope':'Exact source/materialization/header/action-schema/complete assignment/part/clock/tool/settings joins only;50%-budget historical U200 physical counts may be reused under new full budget after re-evaluation. No nearest-candidate substitution.'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('Exact matched rows',result['matched_model_request_rows'],'/',len(rows))
if __name__=='__main__':main()
