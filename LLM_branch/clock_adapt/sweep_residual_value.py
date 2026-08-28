"""Family-grouped CV sweep for the AUTO residual value selector."""
import argparse, json, itertools, shutil
from pathlib import Path
import torch
from .model import ClockResidualSelector
from .train import baseline_target, select_switch_threshold

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train_features',required=True); p.add_argument('--fold_group',default='family'); p.add_argument('--folds',type=int,default=5)
    p.add_argument('--hidden_dims',type=int,nargs='+',default=[64]); p.add_argument('--dropouts',type=float,nargs='+',default=[.1]); p.add_argument('--learning_rates',type=float,nargs='+',default=[1e-3]); p.add_argument('--losses',choices=['huber','mse'],nargs='+',default=['huber']); p.add_argument('--seeds',type=int,nargs='+',default=[123]); p.add_argument('--calibrate_switch_threshold',action='store_true'); p.add_argument('--selection_metric',default='macro_regret_plus_0.25_p90'); p.add_argument('--output_dir',required=True)
    a=p.parse_args(); rows=torch.load(a.train_features,weights_only=False)
    if not rows: raise ValueError('empty feature file')
    groups=sorted({str(e.get('case',{}).get(a.fold_group,e.get('case',{}).get('family',''))) for e in rows}); folds=max(1,min(a.folds,len(groups))); results=[]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for hidden,dropout,lr,loss_name,seed in itertools.product(a.hidden_dims,a.dropouts,a.learning_rates,a.losses,a.seeds):
      torch.manual_seed(seed); model=ClockResidualSelector(rows[0]['memory'].shape[-1],rows[0]['candidate_context'].shape[-1],hidden,dropout); opt=torch.optim.AdamW(model.parameters(),lr=lr); train=[e for e in rows];
      for _ in range(30):
       for e in train:
        raw=model(e['memory'],e['memory_mask'],e['candidate_context']); target,_=baseline_target(e); target=target.to(raw.device); pred=raw-raw[int(torch.tensor(e['clocks']).argmin())]; l=torch.nn.functional.smooth_l1_loss(pred,target) if loss_name=='huber' else torch.nn.functional.mse_loss(pred,target); opt.zero_grad(); l.backward(); opt.step()
      threshold=select_switch_threshold(rows) if a.calibrate_switch_threshold else .05; fold_scores=[]
      for fold in range(folds):
       held={g for i,g in enumerate(groups) if i%folds==fold}; vals=[]
       with torch.no_grad():
        for e in rows:
         if str(e.get('case',{}).get(a.fold_group,e.get('case',{}).get('family',''))) not in held: continue
         raw=model(e['memory'],e['memory_mask'],e['candidate_context']); pred=raw-raw[int(torch.tensor(e['clocks']).argmin())]; target,fast=baseline_target(e); c=int(pred.argmin()); s=c if float(pred[c]) < -threshold else fast; vals.append(float(target[s]))
       fold_scores.append(sum(vals)/max(1,len(vals)))
      mean=sum(fold_scores)/len(fold_scores); p90=float(torch.tensor(fold_scores).quantile(.9)); metric=mean+.25*p90; results.append({'hidden_dim':hidden,'dropout':dropout,'learning_rate':lr,'loss':loss_name,'seed':seed,'switch_threshold':threshold,'macro_regret':mean,'p90_regret':p90,'selection_metric':metric,'model':model})
    best=min(results,key=lambda x:x['selection_metric']); model=best.pop('model'); torch.save({'model':model.state_dict(),'mem_dim':rows[0]['memory'].shape[-1],'context_dim':rows[0]['candidate_context'].shape[-1],'hidden_dim':best['hidden_dim'],'dropout':best['dropout'],'clock_menu':rows[0]['clocks'],'switch_threshold':best['switch_threshold'],'feature_schema':'baseline_delta_v4'},out/'selector.pt')
    json.dump(results,open(out/'sweep_results.json','w'),indent=2); json.dump(best,open(out/'best_config.json','w'),indent=2); print(json.dumps(best,indent=2))
if __name__=='__main__': main()
