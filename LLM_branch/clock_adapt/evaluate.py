import argparse, json, torch
from .model import ClockResidualSelector

def evaluate(features, model):
    rows=[]
    for e in features:
        with torch.no_grad():
            logits=torch.cat([model(e['features'][i].unsqueeze(0),e['scores'][i].view(1)) for i in range(len(e['clocks']))])
        order=torch.argsort(logits, descending=True).tolist(); pred=order[0]; gold=e['label']
        rows.append({'predicted_clock':e['clocks'][pred],'gold_clock':e['clocks'][gold],
                     'correct':pred==gold,'rank':order.index(gold)+1})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--clock_adapter_dir',required=True); ap.add_argument('--output_json',required=True); a=ap.parse_args(); f=torch.load(a.features,weights_only=False); ck=torch.load(a.clock_adapter_dir+'/selector.pt',weights_only=False); m=ClockResidualSelector(ck['feature_dim']); m.load_state_dict(ck['model']); rows=evaluate(f,m)
    by={}
    for x in rows: by.setdefault(x['gold_clock'], []).append(x['correct'])
    balanced=sum(sum(v)/len(v) for v in by.values())/max(1,len(by))
    out={'clock_accuracy':sum(x['correct'] for x in rows)/max(1,len(rows)), 'balanced_clock_accuracy':balanced,
         'clock_mrr':sum(1/x['rank'] for x in rows)/max(1,len(rows)), 'rows':rows}; json.dump(out,open(a.output_json,'w'),indent=2); print(json.dumps({k:out[k] for k in ('clock_accuracy','balanced_clock_accuracy','clock_mrr')}))
if __name__=='__main__': main()
