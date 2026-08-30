import argparse, json, torch
from collections import Counter, defaultdict
from .model import ClockResidualSelector, AutoClockOverrideSelector

def evaluate(features, model, switch_threshold=0.05):
    model.eval()
    rows=[]
    for e in features:
        with torch.no_grad():
            # Feature extraction stores the complete structural memory and
            # candidate-conditioned context.  Keep evaluation's call
            # signature identical to training and inference; the old
            # pre-attention ``features`` vector is no longer produced.
            raw=model(e['memory'], e['memory_mask'], e['candidate_context'])
        fast_idx=int(torch.tensor(e['clocks']).argmin())
        if isinstance(raw, tuple):
            override_logits, clock_logits = raw
            slow = clock_logits.clone()
            slow[fast_idx] = float('-inf')
            candidate = int(slow.argmax())
            pred = candidate if float(torch.sigmoid(override_logits)) >= switch_threshold else fast_idx
            pred_delta = clock_logits - clock_logits[fast_idx]
        else:
            pred_delta=raw-raw[fast_idx]; candidate=int(pred_delta.argmin()); pred=candidate if float(pred_delta[candidate]) < -switch_threshold else fast_idx
        order=torch.argsort(pred_delta, descending=False).tolist(); gold=e['label']
        pc=e['clocks'][pred]; gc=e['clocks'][gold]; adps=e['case'].get('qor_by_clock',e['case'].get('adp_by_clock',{})); ga=float(e['case'].get('gold_adp',1.0)); pa=adps.get(str(pc));
        rows.append({'predicted_clock_period_ns':pc,'reference_clock_period_ns':gc,
                     'correct':pred==gold,'rank':order.index(gold)+1,
                     'adp_regret':10.0 if pa is None else max(0.0,float(pa)/max(ga,1e-9)-1.0),
                     'feasible_adp_regret':None if pa is None else max(0.0,float(pa)/max(ga,1e-9)-1.0),
                     'predicted_infeasible':not e['case'].get('clock_feasible',{}).get(str(pc),False),
                     'candidate_clock_periods_ns':list(e['clocks']),
                     'device':e['case'].get('device'),'family':e['case'].get('family'),
                     'kernel':e['case'].get('kernel'),'qor_by_clock':adps,'adp_by_clock':adps,'gold_adp':ga,'predicted_delta':pred_delta.tolist(),'switch_threshold':switch_threshold})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--clock_adapter_dir',required=True); ap.add_argument('--output_json',required=True); a=ap.parse_args(); f=torch.load(a.features,weights_only=False); ck=torch.load(a.clock_adapter_dir+'/selector.pt',weights_only=False); cls=AutoClockOverrideSelector if ck.get('architecture') == 'override_v4' else ClockResidualSelector; m=(cls(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck.get('n_clocks',len(ck['clock_menu'])),ck['dropout']) if cls is AutoClockOverrideSelector else cls(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck['dropout'])); m.load_state_dict(ck['model']); threshold=float(ck.get('switch_threshold',0.05)); m.switch_threshold=threshold; rows=evaluate(f,m,threshold)
    by={}
    for x in rows: by.setdefault(x['reference_clock_period_ns'], []).append(x['correct'])
    balanced=sum(sum(v)/len(v) for v in by.values())/max(1,len(by))
    regs=[x['adp_regret'] for x in rows if x['adp_regret'] is not None]
    def grouped(key):
        g=defaultdict(list)
        for x in rows:g[x.get(key)].append(x)
        return {str(k):{'count':len(v),'accuracy':sum(x['correct'] for x in v)/len(v),'mean_adp_regret':sum(x['adp_regret'] for x in v)/len(v)} for k,v in g.items()}
    golds=[x['reference_clock_period_ns'] for x in rows]
    majority=Counter(golds).most_common(1)[0][0] if golds else None
    fastest_by_example=[min(x['candidate_clock_periods_ns']) for x in rows]
    def baseline(c):
        regrets=[]
        for x in rows:
            value=x['adp_by_clock'].get(str(c)); regrets.append(10.0 if value is None else max(0.,float(value)/max(x['gold_adp'],1e-9)-1.))
        return {'clock_accuracy':sum(x['reference_clock_period_ns']==c for x in rows)/max(1,len(rows)), 'mean_adp_regret':sum(regrets)/max(1,len(regrets))}
    def fastest_baseline():
        regrets=[]; correct=0
        for x,c in zip(rows,fastest_by_example):
            value=x['adp_by_clock'].get(str(c)); regrets.append(10.0 if value is None else max(0.,float(value)/max(x['gold_adp'],1e-9)-1.)); correct += x['reference_clock_period_ns']==c
        return {'clock_accuracy':correct/max(1,len(rows)), 'mean_adp_regret':sum(regrets)/max(1,len(regrets))}
    feasible_regs=[x['feasible_adp_regret'] for x in rows if x['feasible_adp_regret'] is not None]
    kd=defaultdict(list)
    for x in rows: kd[(x.get('kernel'),x.get('device'))].append(x)
    kd_metrics={f'{k[0]}::{k[1]}':{'count':len(v),'accuracy':sum(x['correct'] for x in v)/len(v),'mean_adp_regret':sum(x['adp_regret'] for x in v)/len(v)} for k,v in kd.items()}
    out={'clock_accuracy':sum(x['correct'] for x in rows)/max(1,len(rows)), 'balanced_clock_accuracy':balanced,
         'clock_mrr':sum(1/x['rank'] for x in rows)/max(1,len(rows)), 'mean_adp_regret':sum(regs)/max(1,len(regs)),
         'median_adp_regret':float(torch.tensor(regs).median()) if regs else None,
         'p90_adp_regret':float(torch.tensor(regs).quantile(.9)) if regs else None,
         'worst_adp_regret':max(regs) if regs else None,
         'mean_feasible_only_adp_regret':sum(feasible_regs)/max(1,len(feasible_regs)),
         'majority_clock_baseline':baseline(majority) if majority is not None else None,
         'fastest_clock_baseline':fastest_baseline() if rows else None,
         'per_family':grouped('family'),'per_device':grouped('device'),
         'kernel_device_macro':kd_metrics,
         'predicted_infeasible_rate':sum(x['predicted_infeasible'] for x in rows)/max(1,len(rows)),
         'rows':rows}; json.dump(out,open(a.output_json,'w'),indent=2); print(json.dumps({k:out[k] for k in ('clock_accuracy','balanced_clock_accuracy','mean_adp_regret','worst_adp_regret')}))
if __name__=='__main__': main()
