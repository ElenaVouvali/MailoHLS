"""Reconcile exact-count development receipts; never infer a learned policy gain."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--receipts',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 records=[];inputs={}
 for path in a.receipts:
  inputs[str(path.resolve())]=sha(path)
  for rec in json.loads(path.read_text())['records']:
   entry=rec['entry']
   for file,digest in entry['report_sha256'].items():
    if sha(file)!=digest:raise ValueError('Changed physical witness '+file)
   m=entry['measurement'];q=rec['request'];resources=m['resources'];cycles=m['latency_cycles'];assert isinstance(cycles,int) and cycles>0
   records.append({'id':rec['id'],'kernel':q['kernel'],'candidate':rec['candidate_hash'],'part':q['part'],'clock_ns':q['clock_ns'],'cycles':cycles,'nominal_ns':cycles*q['clock_ns'],'estimated_clock_ns':m['estimated_clock_ns'],'hls_timing_qualified':m['estimated_clock_ns']<=q['clock_ns'],'full_resource_feasible':all(resources[k]<=q['capacities'][k] for k in resources),'resources':resources,'normalized_headroom':{k:1-resources[k]/q['capacities'][k] for k in resources},'coroni':entry['coroni'],'physical_identity':entry['physical_identity'],'report_sha256':entry['report_sha256']})
 contrasts=[];binding=[]
 for kernel in sorted({r['kernel'] for r in records}):
  contexts={clock:[r for r in records if r['kernel']==kernel and r['clock_ns']==clock] for clock in (10.,3.33)}
  sets={clock:{r['candidate'] for r in rows} for clock,rows in contexts.items()};common=sets[10.]&sets[3.33]
  best={clock:min(rows,key=lambda r:r['cycles']) for clock,rows in contexts.items()};acceptable={clock:{r['candidate'] for r in rows if r['cycles']<=1.01*best[clock]['cycles']} for clock,rows in contexts.items()}
  same=bool(acceptable[10.]&acceptable[3.33]);cross=[]
  for clock,other in ((10.,3.33),(3.33,10.)):
   current=next(r for r in contexts[clock] if r['candidate']==best[other]['candidate']);cross.append(current['cycles']/best[clock]['cycles']-1)
  contrasts.append({'kernel':kernel,'axis':'clock','common_configuration_hashes':sorted(common),'candidate_counts':{str(k):len(v) for k,v in sets.items()},'acceptable_sets_1percent':{str(k):sorted(v) for k,v in acceptable.items()},'classification':'stay' if same else 'meaningful_reversal' if min(cross)>=.05 else 'ambiguous','directional_latency_regrets':cross,'timing_qualified_counts':{str(k):sum(r['hls_timing_qualified'] for r in v) for k,v in contexts.items()},'timing_scope':'HLS estimate only; no post-route evidence'})
  rows=sorted(contexts[10.],key=lambda r:r['cycles']);fast=rows[0]
  # Prospective extension rule: separate a >=5% faster resource-heavier contender by an integer midpoint; same physical receipts reused.
  for slow in rows[1:]:
   if slow['cycles']<1.05*fast['cycles']:continue
   for resource in ('BRAM_18K','DSP','FF','LUT'):
    lo,hi=slow['resources'][resource],fast['resources'][resource]
    if hi<=lo:continue
    limit=(lo+hi)//2;cap=fast['physical_identity']['capacities'][resource];eligible=[r for r in rows if r['resources'][resource]<=limit];winner=min(eligible,key=lambda r:r['cycles'])
    binding.append({'kernel':kernel,'axis':'component_budget','resource':resource,'absolute_limit':limit,'fraction_of_device':limit/cap,'full_winner':fast['candidate'],'limited_winner':winner['candidate'],'full_winner_excluded':True,'fast_cycles':fast['cycles'],'limited_best_cycles':winner['cycles'],'cycles_reduction_full':1-fast['cycles']/winner['cycles'],'hls_timing_qualified_both':fast['hls_timing_qualified'] and winner['hls_timing_qualified'],'classification':'measured_feasibility_boundary','independence':'same physical designs; correlated budget views','scope':'bank-derived diagnostic; below V4 training minimum25percent when fraction<.25'})
    break
   if binding and binding[-1]['kernel']==kernel:break
 result={'schema':'v3-exact-count-switch-stay-analysis-v1','inputs_sha256':inputs,'records':records,'clock_contrasts':contrasts,'binding_budget_contrasts':binding,'physical_calls':len(records),'independent_families':2,'locked_test_used':False,'selection_threshold_latency':.05,'acceptable_tolerance':.01,'policy_improvement':None,'limitations':['Only three measured contenders per source and clock; best in this pool is not global optimum.','No common timing-qualified Viterbi design at3.33ns; requested-period products are nominal.','No new device contrast; resource flip is one family and an unusually tight bank-derived budget.','HLS synthesis only; new jobs have no C/RTL functional or post-route validation.']}
 target=a.out/'measured_bank_analysis.json'
 if target.exists() and json.loads(target.read_text())!=result:raise FileExistsError('Analysis receipt differs; use a versioned output')
 target.write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'measured_bank.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['id','kernel','candidate','clock_ns','cycles','nominal_ns','estimated_clock_ns','hls_timing_qualified','BRAM_18K','DSP','FF','LUT']);writer.writeheader()
  for r in records:writer.writerow({k:r[k] for k in writer.fieldnames if k not in r['resources']}|r['resources'])
 print(json.dumps({'clock':contrasts,'binding':binding,'physical_calls':len(records)},indent=2))
if __name__=='__main__':main()
