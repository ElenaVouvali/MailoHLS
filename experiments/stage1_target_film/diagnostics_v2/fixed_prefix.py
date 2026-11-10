"""Join native per-field receipts to every same-prefix legal candidate score."""
import csv,json
from pathlib import Path
from experiments.stage1_target_film.model import kind_id

def analyze(root):
 traces=[json.loads(s) for s in (root/'paired_scores.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (root/'gan_on/predictions.jsonl').read_text().splitlines()];cursor=0;results=[]
 for row in rows:
  for site in row['candidates'][0]['site_debug']:
   if site['forced_by_semantics']:continue
   candidates=site['top_candidates'];n=len(candidates);batch=traces[cursor:cursor+n];cursor+=n
   assert len(batch)==n and all(r['kind']==kind_id(site['lhs']) for r in batch)
   assert {r['candidate'].removesuffix('\n') for r in batch}=={r['rhs'] for r in candidates},'Candidate-domain/debug truncation prevents verified grouping'
   entry={'context_id':row['context_id'],'lhs':site['lhs'],'candidate_count':n,'prefix_scope':'Identical natural-ON prefix; native scorer constructs one base_input before its per-field candidate loop.'}
   for mode in ('value','off','legacy'):
    ordered=sorted(batch,key=lambda r:r['scores'][mode]['mean_logprob'],reverse=True);top=ordered[0]['scores'][mode]['mean_logprob'];entry[mode+'_top1_set']=','.join(sorted(r['candidate'].rstrip('\n') for r in ordered if r['scores'][mode]['mean_logprob']==top));entry[mode+'_margin']=top-ordered[1]['scores'][mode]['mean_logprob'] if n>1 else None
   entry['changed_top1_set_ON_OFF']=entry['value_top1_set']!=entry['off_top1_set'];entry['max_ON_OFF_candidate_score_delta']=max(abs(r['scores']['value']['mean_logprob']-r['scores']['off']['mean_logprob']) for r in batch);entry['ON_OFF_margin_change']=entry['value_margin']-entry['off_margin'];results.append(entry)
 assert cursor==len(traces),('Unmatched candidate traces',cursor,len(traces))
 return results

def save(root,out):
 results=analyze(root)
 with (out/'FIXED_PREFIX.csv').open('w') as f:
  w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
 return {'decision_sites':len(results),'changed_top1_sets':sum(r['changed_top1_set_ON_OFF'] for r in results),'max_candidate_score_delta':max(r['max_ON_OFF_candidate_score_delta'] for r in results),'max_absolute_margin_change':max(abs(r['ON_OFF_margin_change']) for r in results),'prefix_scope':'Natural ON prefixes held fixed for every paired score; natural OFF trajectories evaluated separately.'}
