"""Group-aware OOF threshold calibration for the AUTO override gate."""
import argparse, json
from pathlib import Path
import torch
from .model import AutoClockOverrideSelector, AutoClockOverrideSelectorV5
from .train import auto_clock_override_loss, auto_clock_override_v5_loss, _override_qor

def fit_fold(rows, train_idx, args):
    e0 = rows[train_idx[0]]
    cls = AutoClockOverrideSelectorV5 if args.architecture == 'override_v5' else AutoClockOverrideSelector
    model = cls(e0['memory'].shape[-1], e0['candidate_context'].shape[-1], args.hidden_dim, len(e0['clocks']), args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for i in train_idx:
            e=rows[i]; override, clocks=model(e['memory'],e['memory_mask'],e['candidate_context'])
            adp=_override_qor(e).unsqueeze(0); avail=e.get('available',torch.ones(len(e['clocks']),dtype=torch.bool)).unsqueeze(0)
            if not torch.isfinite(adp).all(): raise RuntimeError(f'Non-finite AUTO QoR at fold training index {i}')
            loss_fn = auto_clock_override_v5_loss if args.architecture == 'override_v5' else auto_clock_override_loss
            loss=(loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.min_override_regret,args.slow_loss_weight) if args.architecture == 'override_v5' else loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.slow_loss_weight))['loss']
            if not torch.isfinite(override).all() or not torch.isfinite(clocks).all() or not torch.isfinite(loss): raise RuntimeError(f'Non-finite AUTO fold loss at index {i}')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model.eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train_features',required=True); p.add_argument('--folds',type=int,default=5); p.add_argument('--epochs',type=int,default=20); p.add_argument('--hidden_dim',type=int,default=64); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--slow_loss_weight',type=float,default=.35); p.add_argument('--min_override_regret',type=float,default=.02); p.add_argument('--architecture',choices=('override_v4','override_v5'),default='override_v5'); p.add_argument('--max_false_override_rate',type=float,default=.05); p.add_argument('--output_json',required=True); a=p.parse_args()
    rows=torch.load(a.train_features,weights_only=False); groups=[e.get('case',{}).get('family') or e.get('case',{}).get('kernel') or 'unknown' for e in rows]
    unique_groups=sorted(set(groups));
    if len(unique_groups) < a.folds: raise ValueError(f'Need at least {a.folds} distinct kernel groups, found {len(unique_groups)}')
    group_folds=[unique_groups[i::a.folds] for i in range(a.folds)]; oof=[]
    for fold,heldout_groups in enumerate(group_folds,1):
        te=[i for i,g in enumerate(groups) if g in heldout_groups]; tr=[i for i,g in enumerate(groups) if g not in heldout_groups]
        print(f'[AUTO-CV] fold={fold}/{a.folds} train={len(tr)} heldout={len(te)}',flush=True); m=fit_fold(rows,tr,a)
        with torch.no_grad():
            for i in te:
                e=rows[i]; gate, logits=m(e['memory'],e['memory_mask'],e['candidate_context']); prob=float(torch.sigmoid(gate)); clocks=torch.tensor(e['clocks']); fast=int(clocks.argmin()); slow=logits.clone(); slow[fast]=-float('inf'); cand=int(slow.argmax()); qor=_override_qor(e); gold=int(qor.argmin()); override=(gold!=fast); oof.append({'prob':prob,'y':override,'regret_fast':max(0.,float(qor[fast]/qor[gold]-1.)),'regret_override':max(0.,float(qor[cand]/qor[gold]-1.)),'kernel':e.get('case',{}).get('kernel',e.get('case',{}).get('family','unknown')),'device':e.get('case',{}).get('device','unknown')})
    best=None
    def score_for(t):
        do=[r['prob']>=t for r in oof]; regrets=[r['regret_override'] if pred else r['regret_fast'] for pred,r in zip(do,oof)]; grouped={}
        for r,reg in zip(oof,regrets): grouped.setdefault((r['kernel'],r['device']),[]).append(reg)
        macro=sum(sum(v)/len(v) for v in grouped.values())/max(1,len(grouped)); p90=float(torch.tensor(regrets).quantile(.9)); return do,regrets,macro+0.10*p90,macro,p90
    _,_,fast_score,_,_=score_for(1.01)
    for t in [i/100 for i in range(50,100)]+[1.01]:
        do,regrets,score,macro,p90=score_for(t); negatives=[r for r in oof if not r['y']]; fp=sum(int(pred and not r['y']) for pred,r in zip(do,oof))/max(1,len(negatives))
        if fp<=a.max_false_override_rate:
            key=(score,-t)
            if best is None or key<best[0]: best=(key,t,fp,score,macro,p90)
    result={'threshold':float(best[1] if best else 1.01),'false_override_rate':float(best[2] if best else 0.0),'score':float(best[3] if best else fast_score),'fastest_score':float(fast_score),'improvement_vs_fastest':float(fast_score-(best[3] if best else fast_score)),'macro_regret':float(best[4] if best else fast_score),'p90_regret':float(best[5] if best else 0.0),'folds':a.folds,'oof_cases':len(oof)}
    Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output_json,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__': main()
