import argparse, json, torch
from .model import ClockResidualSelector

def evaluate(features, model):
    rows=[]
    for e in features:
        with torch.no_grad():
            logits=model(e['features'])
        order=torch.argsort(logits, descending=True).tolist(); pred=order[0]; gold=e['label']
        pc=e['clocks'][pred]; gc=e['clocks'][gold]; adps=e['case'].get('adp_by_clock',{}); ga=float(e['case'].get('gold_adp',1.0)); pa=adps.get(str(pc));
        rows.append({'predicted_clock_period_ns':pc,'reference_clock_period_ns':gc,
                     'correct':pred==gold,'rank':order.index(gold)+1,
                     'adp_regret':10.0 if pa is None else max(0.0,float(pa)/max(ga,1e-9)-1.0),
                     'feasible_adp_regret':None if pa is None else max(0.0,float(pa)/max(ga,1e-9)-1.0),
                     'predicted_infeasible':not e['case'].get('clock_feasible',{}).get(str(pc),False),
                     'device':e['case'].get('device'),'family':e['case'].get('family')})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--clock_adapter_dir',required=True); ap.add_argument('--output_json',required=True); a=ap.parse_args(); f=torch.load(a.features,weights_only=False); ck=torch.load(a.clock_adapter_dir+'/selector.pt',weights_only=False); m=ClockResidualSelector(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck['dropout']); m.load_state_dict(ck['model']); rows=evaluate(f,m)
    by={}
    for x in rows: by.setdefault(x['reference_clock_period_ns'], []).append(x['correct'])
    balanced=sum(sum(v)/len(v) for v in by.values())/max(1,len(by))
    regs=[x['adp_regret'] for x in rows if x['adp_regret'] is not None]
    out={'clock_accuracy':sum(x['correct'] for x in rows)/max(1,len(rows)), 'balanced_clock_accuracy':balanced,
         'clock_mrr':sum(1/x['rank'] for x in rows)/max(1,len(rows)), 'mean_adp_regret':sum(regs)/max(1,len(regs)),
         'median_adp_regret':float(torch.tensor(regs).median()) if regs else None,
         'p90_adp_regret':float(torch.tensor(regs).quantile(.9)) if regs else None,
         'worst_adp_regret':max(regs) if regs else None,
         'predicted_infeasible_rate':sum(x['predicted_infeasible'] for x in rows)/max(1,len(rows)),
         'rows':rows}; json.dump(out,open(a.output_json,'w'),indent=2); print(json.dumps({k:out[k] for k in ('clock_accuracy','balanced_clock_accuracy','mean_adp_regret','worst_adp_regret')}))
if __name__=='__main__': main()
