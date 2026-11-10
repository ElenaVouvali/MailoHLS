"""Direct request effect only while preceding generated directive values match.
Natural trajectories after their first divergent field are explicitly excluded.
"""
import argparse,hashlib,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();allrows=[]
 for model in ('stage1','stage2','sft','stage3'):
  path=a.generated/(model+'.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
  for kernel in sorted({r['kernel_name'] for r in rows}):
   group=[r for r in rows if r['kernel_name']==kernel];anchor=next(r for r in group if r['device']=='xczu7ev-ffvc1156-2-e' and r['clock_period_ns']==10.)
   for dest in group:
    if dest is anchor:continue
    sa=anchor['candidates'][0]['site_debug'];sb=dest['candidates'][0]['site_debug'];same=True;sites=[]
    for x,y in zip(sa,sb):
     assert x['lhs']==y['lhs']
     if same:
      aa={v['rhs']:v['score'] for v in x['top_candidates']};bb={v['rhs']:v['score'] for v in y['top_candidates']};common=set(aa)&set(bb);delta=[abs(aa[k]-bb[k]) for k in common]
      sites.append({'lhs':x['lhs'],'identical_previous_values':True,'anchor_rhs':x['chosen_rhs'],'destination_rhs':y['chosen_rhs'],'equal_static_candidate_count':x['static_candidate_count']==y['static_candidate_count'],'common_topk_candidates':len(common),'mean_absolute_score_change':statistics.mean(delta) if delta else None,'max_absolute_score_change':max(delta,default=None)})
     same=same and x['chosen_rhs']==y['chosen_rhs']
    allrows.append({'model':model,'kernel':kernel,'axis':'clock' if dest['device']==anchor['device'] else 'device','anchor':anchor['context_id'],'destination':dest['context_id'],'same_absolute_budgets':anchor['resource_budget']==dest['resource_budget'],'budgets_scope':'full-device fractions; device capacities differ, so device contrast does not isolate device from resource limit','changed_complete_assignment':anchor['candidates'][0]['prediction']!=dest['candidates'][0]['prediction'],'identical_prefix_decision_sites':sites,'excluded_after_first_divergence':len(sa)-len(sites),'physical_adaptation_benefit':None,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 result={'schema':'v3-request-response-prefix-qualified-v1','comparisons':allrows,'scope':'Observed generated prefixes only, not a new all-decisions fixed-prefix capture. Source constant, request clock/device changes; no measured winner or physical improvement inferred. Returned top-k intersections only.'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('Changed model/source/axis assignments:',sum(r['changed_complete_assignment'] for r in allrows),'/',len(allrows))
if __name__=='__main__':main()
