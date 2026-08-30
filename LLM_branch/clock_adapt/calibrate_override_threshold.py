"""Group-aware OOF threshold calibration for the AUTO override gate."""
import argparse, json
from pathlib import Path
import torch
from .model import AutoClockOverrideSelector
from .train import auto_clock_override_loss, _qor

def fit_fold(rows, train_idx, args):
    e0 = rows[train_idx[0]]
    model = AutoClockOverrideSelector(e0['memory'].shape[-1], e0['candidate_context'].shape[-1], args.hidden_dim, len(e0['clocks']), args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for i in train_idx:
            e=rows[i]; override, clocks=model(e['memory'],e['memory_mask'],e['candidate_context'])
            adp=_qor(e).unsqueeze(0); avail=e.get('available',torch.ones(len(e['clocks']),dtype=torch.bool)).unsqueeze(0)
            loss=auto_clock_override_loss(override.unsqueeze(0),clocks.unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.slow_loss_weight)['loss']
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train_features',required=True); p.add_argument('--folds',type=int,default=5); p.add_argument('--epochs',type=int,default=20); p.add_argument('--hidden_dim',type=int,default=64); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--slow_loss_weight',type=float,default=.35); p.add_argument('--max_false_override_rate',type=float,default=.05); p.add_argument('--output_json',required=True); a=p.parse_args()
    rows=torch.load(a.train_features,weights_only=False); groups=[e.get('case',{}).get('kernel',e.get('case',{}).get('family','unknown')) for e in rows]
    unique_groups=sorted(set(groups));
    if len(unique_groups) < a.folds: raise ValueError(f'Need at least {a.folds} distinct kernel groups, found {len(unique_groups)}')
    group_folds=[unique_groups[i::a.folds] for i in range(a.folds)]; oof=[]
    for fold,heldout_groups in enumerate(group_folds,1):
        te=[i for i,g in enumerate(groups) if g in heldout_groups]; tr=[i for i,g in enumerate(groups) if g not in heldout_groups]
        print(f'[AUTO-CV] fold={fold}/{a.folds} train={len(tr)} heldout={len(te)}',flush=True); m=fit_fold(rows,tr,a)
        with torch.no_grad():
            for i in te:
                e=rows[i]; gate, logits=m(e['memory'],e['memory_mask'],e['candidate_context']); prob=float(torch.sigmoid(gate)); clocks=torch.tensor(e['clocks']); fast=int(clocks.argmin()); slow=logits.clone(); slow[fast]=-float('inf'); cand=int(slow.argmax()); qor=_qor(e); gold=int(qor.argmin()); override=(gold!=fast); selected=cand if prob>=.5 else fast; regret=max(0.,float(qor[selected]/qor[gold]-1.)); oof.append({'prob':prob,'y':override,'false':(not override and selected!=fast),'regret_fast':max(0.,float(qor[fast]/qor[gold]-1.)),'regret':regret})
    best=None
    for t in [i/100 for i in range(50,100)]:
        pred=[r for r in oof if r['prob']>=t]; negatives=[r for r in oof if not r['y']]; fp=sum(r['false'] for r in negatives)/max(1,len(negatives)); regrets=[r['regret'] if r['prob']>=t else r['regret_fast'] for r in oof]
        if fp<=a.max_false_override_rate:
            key=(sum(regrets)/max(1,len(regrets)),sum(1 for r in pred),-t)
            if best is None or key<best[0]: best=(key,t,fp,sum(regrets)/len(regrets))
    result={'threshold':float(best[1] if best else .99),'false_override_rate':float(best[2] if best else 1.0),'mean_regret':float(best[3] if best else 0.0),'folds':a.folds,'oof_cases':len(oof)}
    Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output_json,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__': main()
