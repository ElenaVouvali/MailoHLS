"""Measured-pool ranking audit; never substitutes likelihood ranking for generation."""
import argparse,collections,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=json.loads(a.scores.read_text());groups=collections.defaultdict(list)
 for r in data['records']:groups[(r['mode'],r['request_id'])].append(r)
 outcomes=[]
 for (mode,request),rows in sorted(groups.items()):
  budget=46 if request.endswith('bram46') else 624;clock=3.33 if ':3.33:' in request else 10.;valid=[r for r in rows if r['measured']['resources']['BRAM_18K']<=budget];qualified=[r for r in valid if r['measured']['estimated_clock_ns']<=clock];best=min((r['measured']['latency_cycles'] for r in qualified),default=None)
  choices={key:max(rows,key=lambda x:x[key]) for key in ('sum_value_logp','mean_value_logp','action_normalized_logp')};top=choices['sum_value_logp'];resource_ok=top in valid;timing_ok=top['measured']['estimated_clock_ns']<=clock;regret=top['measured']['latency_cycles']/best-1 if best and resource_ok and timing_ok else None
  outcomes.append({'mode':mode,'request':request,'chosen_case':top['case_id'],'all_score_conventions_same_choice':len({r['case_id'] for r in choices.values()})==1,'resource_feasible':resource_ok,'hls_estimated_timing_qualified':timing_ok,'qualified_pool_size':len(qualified),'conditional_measured_pool_latency_regret':regret,'chosen_cycles':top['measured']['latency_cycles'],'best_qualified_pool_cycles':best,'conventions':{k:r['case_id'] for k,r in choices.items()},'raw_scores':rows})
 report={'schema':'v3-measured-pool-score-analysis-v1','input_sha256':hashlib.sha256(a.scores.read_bytes()).hexdigest(),'runtime_seconds':data['runtime_seconds'],'hardware':data['hardware'],'outcomes':outcomes,'scope':'Restricted authenticated three-design pools, two validation families; teacher-forced each-completion causal prefixes, not generated output, not normalized whole-configuration production decoder probability. No independent confidence intervals from correlated clock/budget views. One OOD budget diagnostic; no data/test selection or model updates.','decision':'No changed contender winner across Stage1/Stage2/DPO. Context-sensitive scores do not deliver needed full-budget Viterbi preference.'}
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print([(r['mode'],r['request'],r['conditional_measured_pool_latency_regret'],r['all_score_conventions_same_choice']) for r in outcomes])
if __name__=='__main__':main()
