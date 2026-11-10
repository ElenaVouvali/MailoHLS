"""Family-disjoint recovery from frozen Stage1 decision states versus GNN vectors.
All regularization/standardization is fitted inside training families. No policy
or architecture is selected from these descriptive probe outcomes.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from .probe_trip_count import fit_predict,choose,lexical,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--gnn-probe',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();g=json.loads(a.gnn_probe.read_text());sources={r['kernel_name']:r for r in map(json.loads,a.pilot.read_text().splitlines())};cache={};samples=[];features={};execution=json.loads((a.states/'execution.json').read_text())
 for sample in g['samples']:
  kernel,action=sample['kernel'],sample['action']
  if kernel not in cache:
   index=json.loads((a.states/(kernel+'.json')).read_text());tp=Path(index['tensor_file']);assert sha(tp)==index['tensor_sha256'] and index['graph_sha256']==sample['graph_sha256'] and index['hook_score_parity'];vectors=torch.load(tp,map_location='cpu',weights_only=True);lookup={(r['action'],r['layer'],r['boundary']):vectors[r['tensor_index']].numpy() for r in index['records']};memoryfile=a.memory/(kernel+'.memory.pt');assert sha(memoryfile)==sample['memory_sha256'];memory=torch.load(memoryfile,map_location='cpu',weights_only=False)['node_embs'].float().numpy();cache[kernel]=(lookup,memory,index)
  lookup,memory,index=cache[kernel]
  if (action,8,'post_self_attention_residual') not in lookup:continue
  text=sources[kernel]['input'];match=re.search(r'(?m)^\s*'+re.escape(action)+r':',text);local=text[match.start():].splitlines()[0] if match else ''
  point={'gnn':memory[int(action[1:])-1],'lexical_program':lexical(text),'lexical_action_line':lexical(local)}
  for layer in (8,16,24,32):
   state=lookup[(action,layer,'post_self_attention_residual')];point['llm_SA_'+str(layer)]=state;point['llm_plus_gnn_'+str(layer)]=np.concatenate((state,point['gnn']))
  point['llm_decoder_32']=lookup[(action,32,'post_decoder_residual')];point['llm_final_norm']=lookup[(action,32,'final_normalized')]
  samples.append({**sample,'position':next(r['position'] for r in index['records'] if r['action']==action),'encoding_sha256':index['encoding_sha256']})
  for k,v in point.items():features.setdefault(k,[]).append(v)
 y=np.array([r['target_log2'] for r in samples]);train=np.array([r['split']=='train' for r in samples]);test=~train;families=np.array([r['family'] for r in samples]);assert not set(families[train])&set(families[test]);report={'schema':'v3-family-disjoint-joint-probe-v1','inputs':{str(p):sha(p) for p in (a.gnn_probe,a.pilot,a.states/'execution.json')},'train_actions':int(train.sum()),'test_actions':int(test.sum()),'train_families':sorted(set(families[train])),'test_families':sorted(set(families[test])),'target':'log2 positive compiler-proven loop trip count','auxiliary_gpu':execution,'models':{},'samples':samples}
 features['mean']=None;features['llm_SA_32_shuffled_labels']=features['llm_SA_32'];rng=np.random.default_rng(123)
 for name,values in features.items():
  if name=='mean':pred=np.full(test.sum(),y[train].mean());alpha=None;cv=None
  else:
   x=np.array(values);yt=y[train].copy()
   if name.endswith('shuffled_labels'):rng.shuffle(yt)
   alpha,cv=choose(x[train],yt,families[train].tolist());pred=fit_predict(x[train],yt,x[test],alpha)
  errors=pred-y[test];per={f:float(np.mean(abs(errors[families[test]==f]))) for f in report['test_families']}
  report['models'][name]={'alpha_train_cv':alpha,'training_fold_mse':cv,'mae_log2':float(np.mean(abs(errors))),'rmse_log2':float(np.sqrt(np.mean(errors**2))),'within_factor2_fraction':float(np.mean(abs(errors)<=1)),'family_mae':per,'family_macro_mae_log2':float(np.mean(list(per.values()))),'predictions':pred.tolist()}
 report['limitations']=['13 actions from three kernels in two validation families, not13 independent programs','Frozen Stage1 A2 FP16 diagnostics; primary V100S model-generation/QoR qualification remains separate','Teacher-forced preferred previous directives may themselves carry structural correlates','Trip count is explicit compiler input; raw fact availability is not inferred reasoning','Regularization selected only in training family folds; no locked-test labels accessed','No demonstrated physical utility or fusion-layer optimum follows from probe performance']
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({name:{k:v for k,v in r.items() if k not in ('training_fold_mse','predictions')} for name,r in report['models'].items()},indent=2))
if __name__=='__main__':main()
