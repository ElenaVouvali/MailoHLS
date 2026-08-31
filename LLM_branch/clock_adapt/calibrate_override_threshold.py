"""Group-aware OOF threshold calibration for the AUTO override gate."""
import argparse, json
from pathlib import Path
import torch
from .model import AutoClockOverrideSelector, AutoClockOverrideSelectorV5, AutoClockOverrideSelectorV6, AutoClockOverrideSelectorV7
from .train import auto_clock_override_loss, auto_clock_override_v5_loss, auto_clock_override_v6_loss, _override_qor, build_feasibility_targets


def _meaningful_override(e, min_regret):
    qor = _override_qor(e)
    clocks = torch.tensor(e['clocks'])
    fast = int(clocks.argmin())
    gold = int(qor.argmin())
    regret = max(0.0, float(qor[fast] / qor[gold] - 1.0))
    return int(gold != fast and regret >= float(min_regret))


def _group_folds(rows, n_folds, min_regret, strategy='stratified_group'):
    """Return family-disjoint, approximately override-stratified folds.

    Prefer sklearn's StratifiedGroupKFold when installed.  The fallback keeps
    the same family isolation without requiring an extra CPU dependency.
    """
    groups = [e.get('case', {}).get('family') or e.get('case', {}).get('kernel') or 'unknown' for e in rows]
    y = [_meaningful_override(e, min_regret) for e in rows]
    if strategy == 'legacy_round_robin':
        unique = sorted(set(groups))
        if len(unique) < n_folds:
            raise ValueError(f'Need at least {n_folds} groups, found {len(unique)}')
        heldout = [unique[i::n_folds] for i in range(n_folds)]
        return [([i for i, g in enumerate(groups) if g not in held],
                  [i for i, g in enumerate(groups) if g in held])
                for held in heldout]
    if strategy != 'stratified_group':
        raise ValueError(f'Unknown fold strategy: {strategy}')
    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=123)
        return [(tr.tolist(), te.tolist()) for tr, te in cv.split(np.zeros(len(rows)), y, groups)]
    except ImportError:
        unique = sorted(set(groups))
        if len(unique) < n_folds:
            raise ValueError(f'Need at least {n_folds} distinct groups, found {len(unique)}')
        stats = []
        for g in unique:
            idx = [i for i, x in enumerate(groups) if x == g]
            stats.append((g, len(idx), sum(y[i] for i in idx)))
        stats.sort(key=lambda x: (x[2], x[1]), reverse=True)
        fold_groups = [[] for _ in range(n_folds)]
        fold_stats = [[0, 0] for _ in range(n_folds)]
        for g, n, pos in stats:
            j = min(range(n_folds), key=lambda k: (fold_stats[k][1], fold_stats[k][0]))
            fold_groups[j].append(g); fold_stats[j][0] += pos; fold_stats[j][1] += n
        return [([i for i, g in enumerate(groups) if g not in held],
                 [i for i, g in enumerate(groups) if g in held])
                for held in fold_groups]

def fit_fold(rows, train_idx, args):
    e0 = rows[train_idx[0]]
    cls = AutoClockOverrideSelectorV7 if args.architecture == 'override_v7' else AutoClockOverrideSelectorV6 if args.architecture == 'override_v6' else AutoClockOverrideSelectorV5 if args.architecture == 'override_v5' else AutoClockOverrideSelector
    model = cls(e0['memory'].shape[-1], e0['candidate_context'].shape[-1], args.hidden_dim, len(e0['clocks']), args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for i in train_idx:
            e=rows[i]; outputs=model(e['memory'],e['memory_mask'],e['candidate_context']); override, clocks=outputs[:2]
            adp=_override_qor(e).unsqueeze(0); avail=e.get('available',torch.ones(len(e['clocks']),dtype=torch.bool)).unsqueeze(0)
            if not torch.isfinite(adp).all(): raise RuntimeError(f'Non-finite AUTO QoR at fold training index {i}')
            loss_fn = auto_clock_override_v6_loss if args.architecture in ('override_v6', 'override_v7') else auto_clock_override_v5_loss if args.architecture == 'override_v5' else auto_clock_override_loss
            loss=(loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),outputs[2].unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.min_override_regret,args.slow_loss_weight,feasibility_target=build_feasibility_targets(e).unsqueeze(0)) if args.architecture == 'override_v7' else loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),outputs[2].unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.min_override_regret,args.slow_loss_weight) if args.architecture == 'override_v6' else loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.min_override_regret,args.slow_loss_weight) if args.architecture == 'override_v5' else loss_fn(override.unsqueeze(0),clocks.unsqueeze(0),adp,avail,torch.tensor(e['clocks']),args.slow_loss_weight))['loss']
            if (not torch.isfinite(override).all() or not torch.isfinite(clocks).all()
                    or (args.architecture in ('override_v6', 'override_v7') and not torch.isfinite(outputs[2]).all())
                    or not torch.isfinite(loss)):
                raise RuntimeError(f'Non-finite AUTO fold loss at index {i}')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model.eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train_features',required=True); p.add_argument('--folds',type=int,default=5); p.add_argument('--epochs',type=int,default=20); p.add_argument('--hidden_dim',type=int,default=64); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--slow_loss_weight',type=float,default=.35); p.add_argument('--min_override_regret',type=float,default=.02); p.add_argument('--architecture',choices=('override_v4','override_v5','override_v6','override_v7'),default='override_v7'); p.add_argument('--fold_strategy',choices=('legacy_round_robin','stratified_group'),default='stratified_group'); p.add_argument('--max_false_override_rate',type=float,default=.05); p.add_argument('--output_json',required=True); a=p.parse_args()
    rows=torch.load(a.train_features,weights_only=False); groups=[e.get('case',{}).get('family') or e.get('case',{}).get('kernel') or 'unknown' for e in rows]
    folds = _group_folds(rows, a.folds, a.min_override_regret, a.fold_strategy); oof=[]
    for fold,(tr,te) in enumerate(folds,1):
        print(f'[AUTO-CV] fold={fold}/{a.folds} train={len(tr)} heldout={len(te)}',flush=True); m=fit_fold(rows,tr,a)
        with torch.no_grad():
            for i in te:
                e=rows[i]; outputs=m(e['memory'],e['memory_mask'],e['candidate_context']); gate, logits=outputs[:2]; prob=float(torch.sigmoid(gate)); clocks=torch.tensor(e['clocks']); fast=int(clocks.argmin());
                if a.architecture in ('override_v6', 'override_v7'):
                    pfeas=torch.sigmoid(outputs[2]); safe=logits+2.0*torch.log(pfeas.clamp_min(1e-6)); slow=safe.clone()
                else: slow=logits.clone()
                slow[fast]=-float('inf'); cand=int(slow.argmax()); qor=_override_qor(e); gold=int(qor.argmin()); meaningful_override=bool(_meaningful_override(e, a.min_override_regret)); oof.append({'prob':prob,'y':meaningful_override,'regret_fast':max(0.,float(qor[fast]/qor[gold]-1.)),'regret_override':max(0.,float(qor[cand]/qor[gold]-1.)),'kernel':e.get('case',{}).get('kernel',e.get('case',{}).get('family','unknown')),'family':e.get('case',{}).get('family','unknown'),'device':e.get('case',{}).get('device','unknown')})
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
    result={'threshold':float(best[1] if best else 1.01),'false_override_rate':float(best[2] if best else 0.0),'score':float(best[3] if best else fast_score),'fastest_score':float(fast_score),'improvement_vs_fastest':float(fast_score-(best[3] if best else fast_score)),'macro_regret':float(best[4] if best else fast_score),'p90_regret':float(best[5] if best else 0.0),'folds':a.folds,'fold_strategy':a.fold_strategy,'min_override_regret':a.min_override_regret,'oof_cases':len(oof)}
    Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output_json,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__': main()
