"""Create deterministic AUTO resource-budget inputs without using QoR labels."""
import argparse, hashlib, json, random
from collections import defaultdict
from pathlib import Path

RESOURCES = ('bram', 'dsp', 'ff', 'lut')

def budget_rng(seed, kernel, device):
    payload=f'{seed}|{kernel}|{device}'.encode()
    stable=int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')
    return random.Random(stable)

def _groups(rows, split):
    idx={i:n for n in ('train','val','test') for i in split.get(f'{n}_jsonl_idx',[])}
    out={}
    for i,row in enumerate(rows):
        k=row.get('kernel_name',row.get('kernel')); d=row.get('device')
        if k and d and i in idx: out[(k,d)]=idx[i]
    return out

def build_bank(rows, split, locked_val=None, budgets_per_train_group=32, minimum_fraction=.05, seed=123):
    groups=_groups(rows,split); cases=[]; seen=set()
    if locked_val:
        for item in locked_val.get('cases',[]):
            key=(item.get('kernel'),item.get('device'),item.get('resource_budget_id'),tuple(item.get('fractions',())))
            if key not in seen: cases.append(dict(item)); seen.add(key)
    for (kernel,device),sp in sorted(groups.items()):
        if sp=='val': continue
        n=budgets_per_train_group if sp=='train' else budgets_per_train_group
        rng=budget_rng(seed,kernel,device)
        vals=[[1.0]*4]
        vals += [[round(min(1.0,max(minimum_fraction,rng.random())),2) for _ in RESOURCES] for _ in range(max(0,n-1))]
        for frac in vals:
            bid='B'+'_'.join(f'{x:.2f}' for x in frac)
            key=(kernel,device,bid,tuple(frac))
            if key in seen: continue
            cases.append({'kernel':kernel,'device':device,'resource_budget_id':bid,'fractions':frac,'split':sp}); seen.add(key)
    return {'schema':'mailohls-clock-adapt-budget-bank-v1','seed':seed,'minimum_fraction':minimum_fraction,'resource_order':list(RESOURCES),'validation_locked':bool(locked_val),'cases':cases}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--split_json',required=True); p.add_argument('--locked_val_bank',required=True); p.add_argument('--budgets_per_train_group',type=int,default=32); p.add_argument('--minimum_fraction',type=float,default=.05); p.add_argument('--seed',type=int,default=123); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[json.loads(x) for x in open(a.dataset) if x.strip()]; split=json.load(open(a.split_json)); val=json.load(open(a.locked_val_bank)); bank=build_bank(rows,split,val,a.budgets_per_train_group,a.minimum_fraction,a.seed); Path(a.output).parent.mkdir(parents=True,exist_ok=True); json.dump(bank,open(a.output,'w'),indent=2,sort_keys=True); print(json.dumps({'cases':len(bank['cases']),'output':a.output}))
if __name__=='__main__': main()
