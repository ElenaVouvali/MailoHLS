"""Small family-disjoint frozen-GNN probe with explicit compiler trip-count labels.
No LLM training-family states are available: their probe is reported infeasible.
"""
import argparse,collections,hashlib,json,re
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lexical(text):
 v=np.zeros(256)
 text=re.sub(r'<(?:SRC_)?L\d+>|auto\{_[^}]+\}',' ',text)
 for token in re.findall(r'[A-Za-z_]\w*|\d+',text):v[int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%256]+=1
 return np.log1p(v)
def fit_predict(x,y,z,alpha):
 mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.;a=(x-mean)/std;b=(z-mean)/std;center=y.mean();coef=np.linalg.solve(a@a.T+alpha*np.eye(len(a)),y-center)
 return b@a.T@coef+center
def choose(x,y,families):
 unique=sorted(set(families));folds={f:int(hashlib.sha256(f.encode()).hexdigest()[:8],16)%3 for f in unique};loss={}
 for alpha in (.01,.1,1.,10.,100.):
  errors=[]
  for fold in range(3):
   train=np.array([folds[f]!=fold for f in families]);test=~train
   if train.any() and test.any():errors.extend((fit_predict(x[train],y[train],x[test],alpha)-y[test])**2)
  loss[alpha]=float(np.mean(errors))
 return min(loss,key=loss.get),loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank-dir',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.bank_dir/'memory/memory_manifest.json').read_text());source={r['kernel_name']:r for r in map(json.loads,a.pilot.read_text().splitlines())};samples=[]
 for rec in manifest['memory_files']:
  kernel=rec['kernel'];gp=a.bank_dir/'graphs'/(kernel+'.json');mp=a.bank_dir/'memory'/rec['file'];assert sha(gp)==rec['graph_file_sha256'] and sha(mp)==rec['sha256'];graph=json.loads(gp.read_text());pack=torch.load(mp,map_location='cpu',weights_only=False);r=source[kernel]
  for n in graph['nodes']:
   trip=n.get('compiler_proven_trip_count')
   if n.get('category')!='action_point' or n.get('trip_count_provenance')!='mlir_affine_analysis' or not isinstance(trip,int) or trip<=0:continue
   slot=int(n['action_id'][1:])-1
   if not bool(pack['action_slot_mask'][slot]):continue
   samples.append({'kernel':kernel,'family':r['_family'],'split':r['_quality_split'],'action':n['action_id'],'trip_count':trip,'target_log2':float(np.log2(trip)),'gnn':pack['node_embs'][slot].float().numpy(),'lexical':lexical(r['input']),'graph_sha256':sha(gp),'memory_sha256':sha(mp),'compiler_node':n['id']})
 train=np.array([r['split']=='train' for r in samples]);test=~train;families=[r['family'] for r in samples];assert not ({families[i] for i in np.where(train)[0]} & {families[i] for i in np.where(test)[0]});y=np.array([r['target_log2'] for r in samples]);report={'schema':'v3-exploratory-trip-probe-v1','target':'log2 compiler-proven positive loop trip count','train_actions':int(train.sum()),'test_actions':int(test.sum()),'train_families':sorted({families[i] for i in np.where(train)[0]}),'test_families':sorted({families[i] for i in np.where(test)[0]}),'models':{},'samples':[{k:v for k,v in r.items() if k not in ('gnn','lexical')} for r in samples]}
 rng=np.random.default_rng(123)
 for name in ('mean','lexical','gnn','gnn_shuffled_labels'):
  if name=='mean':pred=np.full(test.sum(),y[train].mean());alpha=None;cv=None
  else:
   x=np.array([r['lexical' if name=='lexical' else 'gnn'] for r in samples]);yt=y[train].copy()
   if name.endswith('shuffled_labels'):rng.shuffle(yt)
   alpha,cv=choose(x[train],yt,[families[i] for i in np.where(train)[0]]);pred=fit_predict(x[train],yt,x[test],alpha)
  truth=y[test];errors=pred-truth;report['models'][name]={'alpha_train_cv':alpha,'training_fold_mse':cv,'test_mae_log2':float(np.mean(abs(errors))),'test_rmse_log2':float(np.sqrt(np.mean(errors**2))),'within_factor2_fraction':float(np.mean(abs(errors)<=1.)),'predictions':pred.tolist(),'test_family_mae':{f:float(np.mean(abs(errors[np.array([families[i]==f for i in np.where(test)[0]])]))) for f in report['test_families']}}
 report['limitations']=['Only two independent validation families; exploratory, no broad generalization claim','GNN selected historically for ADP; this probe does not select encoder or prove decision relevance','Trip count is explicitly available to compiler tensorizer; success does not establish inferred compiler reasoning','LLM/fused probe infeasible: cached states cover validation families only, no independent training-family representations','Recurrence negatives not established; absence of dependence witness is not a negative label']
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
if __name__=='__main__':main()
