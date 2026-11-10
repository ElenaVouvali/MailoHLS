"""Matched complete-candidate contrasts on frozen non-test measurements.
Integer-percent observations are bounded conservatively by +/-1 percentage point;
this is robust PROXY evidence, never integer resource counts or timing closure.
"""
import argparse,collections,hashlib,itertools,json,math,subprocess
from pathlib import Path
import numpy as np

def norm(x):return ''.join(c for c in x.lower() if c.isalnum())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);d=json.loads(a.bank.read_text());assert all(r['split'] in ('train','val') for r in d['rows']);ref='origin/research/final-architecture-v3-20261110';tip=subprocess.check_output(['git','rev-parse',ref],text=True).strip();registry=json.loads(subprocess.check_output(['git','show',tip+':Data/manifest.json']));restoration=json.loads(subprocess.check_output(['git','show',tip+':Data/provenance/restoration_20261110.json']));restored={norm(r['kernel']) for r in restoration['kernels']};eligibility={norm(r.get('kernel',r.get('kernel_name',''))):r.get('complete_source_configuration_eligible') for r in registry['kernels']};banks=collections.defaultdict(dict);excluded=collections.Counter();families={};familymap=d['family_map'];families_info=str(type(familymap))
 # Use family identities from the authenticated common-pool ledger, not raw 'family=train'.
 ledger=json.loads(Path('experiments/final_architecture_20261109/results/target_contrasts.json').read_text());families={r['kernel']:r['family'] for r in ledger['rows']}
 for r in d['rows']:
  if norm(r['kernel']) in restored:excluded['restored_source_action_identity_requires_reconstruction']+=1;continue
  if eligibility.get(norm(r['kernel'])) is not True:excluded['canonical_incomplete_source_configuration']+=1;continue
  if norm(r['kernel']).startswith('spcl'):excluded['quarantined_latency_units']+=1;continue
  q=r['physical']
  if any(v is None or not math.isfinite(float(v)) or float(v)<0 for v in q) or q[0]<=0:excluded['unknown_or_invalid_measurement']+=1;continue
  if all(float(v)==0 for v in q[1:]):excluded['all_zero_resources_unobservable']+=1;continue
  assign={k.upper():str(v) for k,v in r['assignment'].items()};h=digest(assign);key=(r['split'],r['kernel'],r['device'],float(r['clock']));v=dict(r,assignment=assign,identity=h,utilization=np.array(q[1:])/np.array(r['capacities']),family_name=families.get(r['kernel'],r['kernel']))
  v['robust_full']=bool(np.all(v['utilization']+.01<=1));v['budget_flags']={b:(bool(np.all(v['utilization']+.01<=b)),bool(np.any(np.maximum(0,v['utilization']-.01)>b)),bool(np.all(v['utilization']<=b))) for b in (.2,.25,.5)}
  old=banks[key].get(h)
  if old and old['physical']!=v['physical']:excluded['conflicting_duplicate']+=1;banks[key].pop(h,None);continue
  banks[key][h]=v
 stats=collections.defaultdict(collections.Counter);pair_sets=collections.defaultdict(set);kernel_sets=collections.defaultdict(set);family_sets=collections.defaultdict(set);examples=collections.defaultdict(list);allgroups=[]
 def record(fold,axis,status,kernel,x,y,left,right,qualification):
  key=(fold,axis,status);pair=digest([kernel,*sorted((x['identity'],y['identity']))]);stats[fold][axis+'_'+status]+=1;pair_sets[key].add(pair);kernel_sets[key].add(kernel);family_sets[key].add(x['family_name'])
  if len(examples[key])<12:examples[key].append({'kernel':kernel,'family':x['family_name'],'pair_identity':pair,'candidate_a':x['identity'],'candidate_b':y['identity'],'assignment_a':x['assignment'],'assignment_b':y['assignment'],'context_left':left,'context_right':right,'evidence':qualification,'physical_a':x['physical'],'physical_b':y['physical']})
 for key,pool in banks.items():
  fold,kernel,device,clock=key;rows=sorted(pool.values(),key=lambda r:r['physical'][0]);
  for x,y in itertools.combinations(rows,2):
   if y['physical'][0]<1.1*x['physical'][0]:continue
   # Both must be robustly full-budget feasible; no exact integer-count claim.
   if not (x['robust_full'] and y['robust_full']):continue
   for budget in (.2,.25,.5):
    if y['budget_flags'][budget][0] and x['budget_flags'][budget][1]:status='robust_proxy_switch'
    elif x['budget_flags'][budget][0] and y['budget_flags'][budget][0]:status='robust_proxy_stay'
    elif x['budget_flags'][budget][2]!=y['budget_flags'][budget][2]:status='ambiguous_proxy_boundary'
    else:continue
    record(fold,'budget',status,kernel,x,y,[device,clock,budget],[device,clock,1.],'percentage-derived interval +/-1pp, no exact counts or timing')
 for left,right in itertools.combinations(sorted(banks),2):
  if left[:2]!=right[:2]:continue
  axis='clock' if left[2]==right[2] and left[3]!=right[3] else 'device' if left[2]!=right[2] and left[3]==right[3] else None
  if not axis:continue
  common=sorted(set(banks[left])&set(banks[right]));fold,kernel=left[:2]
  for aa,bb in itertools.combinations(common,2):
   x,y=banks[left][aa],banks[left][bb];u,v=banks[right][aa],banks[right][bb]
   if any(not r['robust_full'] for r in (x,y,u,v)):continue
   la,lb,ra,rb=[r['physical'][0] for r in (x,y,u,v)];l=0 if max(la,lb)<1.1*min(la,lb) else (1 if la<lb else -1);r=0 if max(ra,rb)<1.1*min(ra,rb) else (1 if ra<rb else -1);status='near_tie' if not l or not r else 'switch' if l!=r else 'stay';record(fold,axis,status,kernel,x,y,list(left[2:]),list(right[2:]),'measured latency field, no timing qualification; device views use full-device fractions')
 result={'schema':'film-v2-non-test-matched-pairs-v1','bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),'canonical_registry_commit':tip,'excluded_records':dict(excluded),'physical_contexts':len(banks),'physical_records':sum(len(x) for x in banks.values()),'statistics':{},'examples':{},'precision_policy':'Raw CSV integer-percent reporting resolution observed. Conservative +/-1pp covers nearest/truncated integer reporting; aggregation does not establish exact resource counts. Robust proxy claims conditional on this bound.','locked_test_opened':False,'timing_qualified':False,'latency_gap_threshold':.1,'legality_scope':'Canonical complete-source eligibility checked. Exhaustive per-assignment source/tool validation is still required before optimization; these candidate-level contrast counts are not a training release.'}
 for key,views in stats.items():result['statistics'][key]=dict(views)
 for key,pairs in pair_sets.items():
  name='/'.join(key);result['examples'][name]=examples[key];result.setdefault('independence',{})[name]={'derived_views':stats[key[0]][key[1]+'_'+key[2]],'unique_candidate_pairs':len(pairs),'kernels':len(kernel_sets[key]),'families':sorted(family_sets[key])}
 write=a.out/'TARGET_CONTRAST_AUDIT.json';write.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('examples',)},indent=2))
if __name__=='__main__':main()
