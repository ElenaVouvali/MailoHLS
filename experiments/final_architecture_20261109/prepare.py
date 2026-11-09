"""Latency-specific matched-pool audit. Percentage-derived resources remain proxies."""
from collections import defaultdict,Counter
from itertools import combinations
from pathlib import Path
import argparse,csv,json,math
from .audit import OUT,save

def compare(left,right,budget_left,budget_right):
 common=sorted(set(left)&set(right))
 def eligible(pool,budget):
  return {h:r for h,r in pool.items() if all(x<=b for x,b in zip(r['physical'][1:],budget))}
 a,b=eligible(left,budget_left),eligible(right,budget_right)
 ca,cb=eligible({h:left[h] for h in common},budget_left),eligible({h:right[h] for h in common},budget_right)
 def best(pool):
  if not pool:return None,[]
  value=min(r['physical'][0] for r in pool.values())
  return value,sorted(h for h,r in pool.items() if r['physical'][0]<=value*1.01)
 va,wa=best(ca);vb,wb=best(cb)
 status='unsupported';gains=None
 if wa and wb:
  if set(wa)&set(wb):status='no_change'
  elif any(h not in cb for h in wa) or any(h not in ca for h in wb):status='feasibility_boundary_proxy'
  else:
   ha,hb=wa[0],wb[0]
   gains=[(ca[hb]['physical'][0]-va)/va,(cb[ha]['physical'][0]-vb)/vb]
   status='latency_reversal' if min(gains)>=.05 else 'near_tie'
 return {'status':status,'left_pool':len(left),'right_pool':len(right),'common_hashes':len(common),
  'left_feasible_proxy':len(a),'right_feasible_proxy':len(b),'left_common_feasible_proxy':len(ca),'right_common_feasible_proxy':len(cb),
  'left_acceptable_hashes':wa,'right_acceptable_hashes':wb,'left_min_latency_ms':va,'right_min_latency_ms':vb,
  'both_direction_transfer_regrets':gains,'left_budget':budget_left,'right_budget':budget_right,
  'resources':'utilization-times-capacity proxies; exact feasibility/timing not established',
  'left_unique_hashes':sorted(set(left)-set(right)),'right_unique_hashes':sorted(set(right)-set(left))}

def run(source,output):
 blob=json.loads(source.read_text()); groups=defaultdict(dict);conflicts=set()
 for r in blob['rows']:
  key=(r['kernel'],r['profile'],r['clock']);h=r['config_hash']
  if h in groups[key] and groups[key][h]['physical']!=r['physical']:conflicts.add((key,h))
  groups[key][h]=r
 for k,h in conflicts:del groups[k][h]
 rows=[]
 def add(k1,k2,b1,b2,axis,view):
  if k1 not in groups or k2 not in groups:return
  a,b=groups[k1],groups[k2]
  result=compare(a,b,b1,b2)
  rows.append({'kernel':k1[0],'split':next(iter(a.values()))['split'],'family':blob['family_map'][k1[0]],'axis':axis,'view':view,'left_context':list(k1),'right_context':list(k2),**result})
 for key,pool in sorted(groups.items()):
  if not pool:continue
  caps=next(iter(pool.values()))['capacities'];kernel,device,clock=key
  for f in (.5,.25,.2):add(key,key,caps,[x*f for x in caps],'budget',str(f))
  if clock==10:
   for c in (5.,3.33):add(key,(kernel,device,c),caps,caps,'clock','same absolute limits')
  if device=='zcu104':
   other=(kernel,'u200',clock)
   if other in groups:
    oc=next(iter(groups[other].values()))['capacities']
    add(key,other,caps,oc,'device','full-device fractions')
    add(key,other,caps,caps,'device','common absolute limits')
 counts={s:{a:dict(Counter(r['status'] for r in rows if r['split']==s and r['axis']==a)) for a in ('budget','clock','device')} for s in ('train','val')}
 summary={'schema':'latency-common-configuration-contrasts-v1','rows':rows,'counts':counts,'physical_records':len(blob['rows']),
  'unique_physical_designs':sum(len(g) for g in groups.values()),'conflicting_duplicates_excluded':len(conflicts),
  'independent_kernels':len({k[0] for k in groups}),'exact_resource_counts':False,'timing_qualified_contexts':0,'test_labels_loaded':False}
 save(output/'target_contrasts.json',summary)
 with (output/'target_contrasts.csv').open('w',newline='') as f:
  fields=['kernel','family','split','axis','view','status','common_hashes','left_pool','right_pool','left_min_latency_ms','right_min_latency_ms']
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 print(json.dumps(counts,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=OUT/'reaudit/non_test_records.json');p.add_argument('--output',type=Path,default=OUT/'contrasts');a=p.parse_args();run(a.source,a.output)
