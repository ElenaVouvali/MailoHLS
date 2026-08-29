"""Leakage-free family-grouped CV sweep for AUTO residual regression."""
import argparse,itertools,json
from pathlib import Path
import torch
from .model import ClockResidualSelector
from .train import baseline_target, has_complete_qor, selected_regret

def group(e,n): return str(e.get('case',{}).get(n,e.get('case',{}).get('family','')))
def fit(rows,h,d,lr,kind,seed,epochs=30,batch_size=64,device='cpu',regret_weight=0.1,temperature=1.0):
 torch.manual_seed(seed); dev=torch.device(device); m=ClockResidualSelector(rows[0]['memory'].shape[-1],rows[0]['candidate_context'].shape[-1],h,d).to(dev); o=torch.optim.AdamW(m.parameters(),lr=lr)
 for _ in range(epochs):
  order=torch.randperm(len(rows)).tolist()
  for start in range(0,len(order),batch_size):
   batch=[rows[i] for i in order[start:start+batch_size]]; memory=torch.stack([e['memory'] for e in batch]).to(dev); mask=torch.stack([e['memory_mask'] for e in batch]).to(dev); context=torch.stack([e['candidate_context'] for e in batch]).to(dev); target=torch.stack([baseline_target(e)[0] for e in batch]).to(dev); fast=torch.tensor([baseline_target(e)[1] for e in batch],device=dev); raw=m(memory,mask,context); pred=raw-raw.gather(1,fast[:,None]);
   if not torch.isfinite(target).all(): raise RuntimeError('Non-finite AUTO target')
   if not torch.isfinite(raw).all(): raise RuntimeError('Non-finite AUTO prediction')
   qor=torch.stack([torch.tensor([float(e['case'].get('qor_by_clock',e['case'].get('adp_by_clock',{}))[str(c)]) for c in e['clocks']],device=dev) for e in batch])
   regret=qor/qor.min(dim=1,keepdim=True).values-1.0
   probability=torch.softmax(-pred/max(float(temperature),1e-6),dim=-1)
   value_loss=torch.nn.functional.smooth_l1_loss(pred,target) if kind=='huber' else torch.nn.functional.mse_loss(pred,target)
   l=value_loss+float(regret_weight)*(probability*regret).sum(dim=-1).mean()
   if not torch.isfinite(l): raise RuntimeError('Non-finite AUTO loss')
   o.zero_grad(); l.backward(); o.step()
 return m
def threshold(oof):
 best=(float('inf'),.05)
 for x in (0,.01,.02,.05,.1,.2,.3,.5):
  rs=[]; grouped={}
  for e,p in oof:
   _,f=baseline_target(e); c=int(p.argmin()); selected=c if float(p[c]) < -x else f; r=selected_regret(e, selected); rs.append(r); key=(e.get('case',{}).get('kernel',e.get('case',{}).get('kernel_name')),e.get('case',{}).get('device')) ; grouped.setdefault(key,[]).append(r)
  macro=sum(sum(v)/len(v) for v in grouped.values())/max(1,len(grouped)); score=macro+.25*float(torch.tensor(rs).quantile(.9))
  if score<best[0]: best=(score,x)
 return best[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train_features',required=True); p.add_argument('--fold_group',default='family'); p.add_argument('--folds',type=int,default=5); p.add_argument('--hidden_dims',type=int,nargs='+',default=[64]); p.add_argument('--dropouts',type=float,nargs='+',default=[.1]); p.add_argument('--learning_rates',type=float,nargs='+',default=[1e-3]); p.add_argument('--losses',choices=['huber','mse'],nargs='+',default=['huber']); p.add_argument('--seeds',type=int,nargs='+',default=[123]); p.add_argument('--calibrate_switch_threshold',action='store_true'); p.add_argument('--selection_metric',default='macro_regret_plus_0.25_p90'); p.add_argument('--batch_size',type=int,default=64); p.add_argument('--epochs',type=int,default=30); p.add_argument('--device',default='cpu'); p.add_argument('--regret_weight',type=float,default=0.1); p.add_argument('--temperature',type=float,default=1.0); p.add_argument('--output_dir',required=True); a=p.parse_args(); raw_rows=torch.load(a.train_features,weights_only=False); rows=[e for e in raw_rows if has_complete_qor(e)]; print(f'[AUTO-DATA] total={len(raw_rows)} complete={len(rows)} dropped={len(raw_rows)-len(rows)}',flush=True); 
 if not rows: raise RuntimeError('No complete-clock AUTO cases remain')
 groups=sorted({group(e,a.fold_group) for e in rows}); n=max(1,min(a.folds,len(groups))); results=[]
 # Always-fastest is a mandatory incumbent.  A neural selector is only
 # promoted when its OOF decision metric actually beats this fallback.
 fastest_grouped={}
 for e in rows:
  _, fast=baseline_target(e); key=(e.get('case',{}).get('kernel',e.get('case',{}).get('kernel_name')),e.get('case',{}).get('device')); fastest_grouped.setdefault(key,[]).append(selected_regret(e, fast))
 fastest_values=[r for values in fastest_grouped.values() for r in values]
 fastest_macro=sum(sum(v)/len(v) for v in fastest_grouped.values())/max(1,len(fastest_grouped)); fastest_p90=float(torch.tensor(fastest_values).quantile(.9)); results.append({'model_type':'always_fastest','switch_threshold':None,'macro_regret':fastest_macro,'p90_regret':fastest_p90,'selection_metric':fastest_macro+.25*fastest_p90,'oof_count':len(rows),'kernel_device_groups':len(fastest_grouped)})
 for h,d,lr,k,seed in itertools.product(a.hidden_dims,a.dropouts,a.learning_rates,a.losses,a.seeds):
  oof=[]
  for fold in range(n):
   held={g for i,g in enumerate(groups) if i%n==fold}; tr=[e for e in rows if group(e,a.fold_group) not in held]; te=[e for e in rows if group(e,a.fold_group) in held]; print(f'[AUTO-CV] cfg={(h,d,lr,k,seed)} fold={fold+1}/{n} train={len(tr)} heldout={len(te)}',flush=True); m=fit(tr,h,d,lr,k,seed+fold,a.epochs,a.batch_size,a.device,a.regret_weight,a.temperature)
   with torch.no_grad():
    for e in te:
     raw=m(e['memory'],e['memory_mask'],e['candidate_context']); f=int(torch.tensor(e['clocks']).argmin()); oof.append((e,(raw-raw[f]).cpu()))
  th=threshold(oof) if a.calibrate_switch_threshold else .05; rs=[]
  for e,pred in oof:
   _,f=baseline_target(e); c=int(pred.argmin()); selected=c if float(pred[c]) < -th else f; rs.append(selected_regret(e, selected))
  grouped={}
  for e,pred in oof:
   _,f=baseline_target(e); c=int(pred.argmin()); selected=c if float(pred[c]) < -th else f; r=selected_regret(e, selected); key=(e.get('case',{}).get('kernel',e.get('case',{}).get('kernel_name')),e.get('case',{}).get('device')); grouped.setdefault(key,[]).append(r)
  macro=sum(sum(v)/len(v) for v in grouped.values())/max(1,len(grouped)); p90=float(torch.tensor([r for v in grouped.values() for r in v]).quantile(.9)); results.append({'model_type':'neural','hidden_dim':h,'dropout':d,'learning_rate':lr,'loss':k,'seed':seed,'switch_threshold':th,'macro_regret':macro,'p90_regret':p90,'selection_metric':macro+.25*p90,'oof_count':len(oof),'kernel_device_groups':len(grouped)})
 best=min(results,key=lambda r:r['selection_metric']); neural_best=min((r for r in results if r.get('model_type')=='neural'),key=lambda r:r['selection_metric']); m=fit(rows,neural_best['hidden_dim'],neural_best['dropout'],neural_best['learning_rate'],neural_best['loss'],neural_best['seed'],a.epochs,a.batch_size,a.device,a.regret_weight,a.temperature); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); use_fastest=best.get('model_type')=='always_fastest'; torch.save({'model':m.state_dict(),'mem_dim':rows[0]['memory'].shape[-1],'context_dim':rows[0]['candidate_context'].shape[-1],'hidden_dim':neural_best['hidden_dim'],'dropout':neural_best['dropout'],'clock_menu':rows[0]['clocks'],'switch_threshold':1e9 if use_fastest else neural_best['switch_threshold'],'force_fastest':use_fastest,'feature_schema':'baseline_delta_v4'},out/'selector.pt'); json.dump(results,open(out/'sweep_results.json','w'),indent=2,allow_nan=False); json.dump({'selected':best,'neural_checkpoint':neural_best},open(out/'best_config.json','w'),indent=2,allow_nan=False); print(json.dumps(best,indent=2))
if __name__=='__main__': main()
