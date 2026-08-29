"""Build budget-aligned AUTO cases using the locked family split."""
import argparse,json
import re
from collections import defaultdict
from pathlib import Path
from LLM_branch.common.mailohls_contract import supported_clock_periods
def num(r,*ns):
 for n in ns:
  try:
   if n in r:return float(r[n])
  except (TypeError,ValueError):pass
 return None
def build_cases(rows,budget_bank,split,objective='PARETO_ADP', policy_labels=None):
 idx_split={i:n for n in ('train','val','test') for i in split.get(f'{n}_jsonl_idx',[])}
 g=defaultdict(list)
 for i,r in enumerate(rows):
  k=r.get('kernel_name',r.get('kernel'));d=r.get('device');c=num(r,'clock_period','selected_clock_period','clock_period_ns');l=num(r,'latency','latency_ms');a=num(r,'area','area_mm2')
  if k and d and c is not None and l is not None and a is not None and any(abs(c-x)<=0.02 for x in supported_clock_periods(d)):
   score = max(l,0) if objective=='PARETO_LATENCY' else max(a,.0625) if objective=='PARETO_AREA' else max(l,0)*max(a,.0625)
   g[(k,d)].append((r,c,score,idx_split.get(i)))
 out=[]
 for (k,d),items in g.items():
  bs=[b for b in budget_bank.get('cases',[]) if b.get('kernel')==k and b.get('device')==d]
  if not bs: raise ValueError(f'Missing AUTO budget bank for {k}/{d}')
  seen=set()
  for b in bs:
   fr=b.get('fractions',[1,1,1,1]); fr=list(fr) if isinstance(fr,list) else [fr.get(n,1.0) for n in ('bram','dsp','ff','lut')]; key=(b.get('resource_budget_id'),tuple(fr))
   if key in seen: continue
   seen.add(key); adp={};feas={};dirs={}
   for c in supported_clock_periods(d):
    ok=[]
    for r,cc,x,_sp in items:
     util=[num(r,f'{n}_util_%',f'{n}_util',n) for n in ('bram','dsp','ff','lut')]
     if abs(cc-c)<=0.02 and all(v is not None and v/100<=f for v,f in zip(util,fr)):ok.append((x,r))
    feas[str(c)]=bool(ok)
    if ok:x,r=min(ok,key=lambda item:item[0]);adp[str(c)]=x;dirs[str(c)]=r.get('preprocessed_row',r.get('source_key'))
   if policy_labels:
    policy_values = {}
    for c in supported_clock_periods(d):
     key=(k,d,round(float(c),2),str(b.get('resource_budget_id','')),objective)
     if key not in policy_labels:
      raise ValueError(
       f'Missing complete policy label for kernel={k} device={d} '
       f'clock={float(c):g} budget={b.get("resource_budget_id", "")} objective={objective}'
      )
     value = float(policy_labels[key])
     if not (value > 0.0):
      raise ValueError(f'Non-positive policy QoR for {key}: {value}')
     policy_values[str(c)] = value
    adp = policy_values
    feas = {str(c): True for c in supported_clock_periods(d)}
   if not adp:continue
   gold=min(adp,key=adp.get); sp=next((s for _r,_c,_x,s in items if s),None)
   family=re.split(r'[-_](?:\d+|baseline|tiling|pipeline|unroll|doublebuffer|coalescing).*',k,1)[0]
   out.append({'kernel':k,'device':d,'family':family,'objective':objective,'frequency_mode':'auto','available_clock_periods':list(supported_clock_periods(d)),'gold_clock_period':float(gold),'gold_adp':adp[gold],'qor_by_clock':adp,'adp_by_clock':adp,'clock_feasible':feas,'best_directives_by_clock':dirs,'resource_budget':dict(zip(('bram','dsp','ff','lut'),fr)),'resource_budget_id':b.get('resource_budget_id'),'split':sp})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split_json',required=True);p.add_argument('--budget_bank',required=True);p.add_argument('--include_splits',default='train,val,test');p.add_argument('--objective',choices=('PARETO_LATENCY','PARETO_AREA','PARETO_ADP'),default='PARETO_ADP');p.add_argument('--policy_labels',default='');p.add_argument('--output_dir',required=True);a=p.parse_args();rows=[json.loads(x) for x in open(a.dataset) if x.strip()]; labels={}
 if a.policy_labels:
  for r in (json.loads(x) for x in open(a.policy_labels) if x.strip()):
   if r.get('qor') is None: continue
   key=(r.get('kernel_name',r.get('kernel')),r.get('device'),round(float(r.get('clock_period_ns',r.get('clock_period'))),2),str(r.get('resource_budget_id','')),str(r.get('objective',a.objective))); labels[key]=min(float(r['qor']),labels.get(key,float('inf')))
 cases=build_cases(rows,json.load(open(a.budget_bank)),json.load(open(a.split_json)),a.objective,labels);included={x.strip() for x in a.include_splits.split(',') if x.strip()};cases=[x for x in cases if x.get('split') in included];Path(a.output_dir).mkdir(parents=True,exist_ok=True)
 counts={n:sum(x['split']==n for x in cases) for n in ('train','val','test')}
 empty=sorted(n for n in included if counts.get(n,0)==0)
 if empty: raise ValueError(f'Requested AUTO splits produced no cases: {empty}')
 for n in ('train','val','test'):
  with open(Path(a.output_dir)/f'{n}.jsonl','w') as f:
   for x in cases:
    if x['split']==n:f.write(json.dumps(x,sort_keys=True)+'\n')
 print(json.dumps({'cases':len(cases),**counts}))
if __name__=='__main__':main()
