"""Build budget-aligned AUTO cases using the locked family split."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from LLM_branch.common.mailohls_contract import supported_clock_periods
def num(r,*ns):
 for n in ns:
  try:
   if n in r:return float(r[n])
  except (TypeError,ValueError):pass
 return None
def build_cases(rows,budget_bank,split):
 g=defaultdict(list)
 for r in rows:
  k=r.get('kernel_name',r.get('kernel'));d=r.get('device');c=num(r,'clock_period','selected_clock_period','clock_period_ns');l=num(r,'latency','latency_ms');a=num(r,'area','area_mm2')
  if k and d and c is not None and l is not None and a is not None and c in supported_clock_periods(d):g[(k,d)].append((r,c,max(l,0)*max(a,.0625)))
 out=[]
 for (k,d),items in g.items():
  bs=[b for b in budget_bank.get('cases',[]) if b.get('kernel')==k and b.get('device')==d] or [{'resource_budget_id':'full','fractions':[1,1,1,1]}]
  for b in bs:
   fr=b.get('fractions',[1,1,1,1]);adp={};feas={};dirs={}
   for c in supported_clock_periods(d):
    ok=[]
    for r,cc,x in items:
     util=[num(r,f'{n}_util_%',f'{n}_util',n) for n in ('bram','dsp','ff','lut')]
     if cc==c and all(v is not None and v/100<=f for v,f in zip(util,fr)):ok.append((x,r))
    feas[str(c)]=bool(ok)
    if ok:x,r=min(ok);adp[str(c)]=x;dirs[str(c)]=r.get('preprocessed_row',r.get('source_key'))
   if not adp:continue
   gold=min(adp,key=adp.get); sp=next((n for n in ('train','val','test') if k in split.get(f'{n}_kernels',[])),None)
   out.append({'kernel':k,'device':d,'frequency_mode':'auto','available_clock_periods':list(supported_clock_periods(d)),'gold_clock_period':float(gold),'gold_adp':adp[gold],'adp_by_clock':adp,'clock_feasible':feas,'best_directives_by_clock':dirs,'resource_budget':dict(zip(('bram','dsp','ff','lut'),fr)),'resource_budget_id':b.get('resource_budget_id'),'split':sp})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split_json',required=True);p.add_argument('--budget_bank',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args();rows=[json.loads(x) for x in open(a.dataset) if x.strip()];cases=build_cases(rows,json.load(open(a.budget_bank)),json.load(open(a.split_json)));Path(a.output_dir).mkdir(parents=True,exist_ok=True)
 for n in ('train','val','test'):
  with open(Path(a.output_dir)/f'{n}.jsonl','w') as f:
   for x in cases:
    if x['split']==n:f.write(json.dumps(x,sort_keys=True)+'\n')
 print(json.dumps({'cases':len(cases),'train':sum(x['split']=='train' for x in cases),'val':sum(x['split']=='val' for x in cases),'test':sum(x['split']=='test' for x in cases)}))
if __name__=='__main__':main()
