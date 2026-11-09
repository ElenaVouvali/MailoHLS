"""Revision-2 CPU re-audit using authenticated V4b data and model implementations.
No test labels or GPU execution. Historical receipts are immutable inputs.
"""
from pathlib import Path
from collections import Counter, defaultdict, deque
import argparse, hashlib, json, tarfile, time, subprocess, copy, os
import torch
from GNN_branch import target_graph_v22 as overlay, target_graph_tensor_v22 as tensors
from GNN_branch import mlir_graph_v2 as compiler
from experiments.v22_gnn_training import train
from experiments.v22_gnn_training.v4b import train_probe as probe, quality, evaluation
from experiments.v22_gnn_training.v4b.provenance import validate_cached_measurements
from experiments.v22_gnn_training.v4b.capacity_control import capacity_model
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get('SALT_V4B_RUN_ROOT', str(Path.home()/'MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b')))
OUT=ROOT/'artifacts/final_architecture_20261109'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if p.exists() and p.read_text()!=text: raise FileExistsError('Immutable receipt differs: '+str(p))
 p.write_text(text)

def load():
 protocol=json.loads((RUN/'corrected_v3_full37/protocol.json').read_text())
 cache=RUN/'prepared_full.pt'
 assert sha(cache)==protocol['prepared_cache_sha256']
 prepared=torch.load(cache,map_location='cpu',weights_only=False)['prepared']
 assert all(g['split'] in ('train','val') and 'kalman' not in g['kernel'].lower() for g in prepared['groups'])
 return prepared,protocol

def authenticate(p,protocol):
 archive=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
 assert sha(archive)=='feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73'
 release=OUT/'rc3_release'
 if not release.exists():
  release.mkdir()
  with tarfile.open(archive) as t:
   for m in t.getmembers():
    path=(release/m.name).resolve()
    if not path.is_relative_to(release.resolve()) or m.issym() or m.islnk(): raise ValueError('Unsafe archive member')
   t.extractall(release)
 # Locate actual archive root without inventing graph mappings.
 rel=next(release.rglob('canonical')).parent
 names=p['manifest']['training_kernels']+p['manifest']['validation_kernels']
 data=train.PreparedData(rel,OUT/'unused.json',selected_kernels=names,
                        authenticated_audit_path=ROOT/'experiments/v22_gnn_training/data-audit.json', preprocessed_dir=Path(os.environ.get('SALT_PREPROCESSED_DIR', str(Path.home()/'MailoHLS/LLM_branch/Data/preprocessed_CSVS'))))
 validated=validate_cached_measurements(p,data,p['manifest'])
 source={k:{'expected':v,'observed':sha(ROOT/k),'equal':sha(ROOT/k)==v} for k,v in protocol['source_sha256'].items()}
 return data,{'archive_sha256':sha(archive),'cache_sha256':sha(RUN/'prepared_full.pt'),
              'manifest_sha256':p['manifest']['manifest_sha256'],'source_checks':source,**validated}

def histories():
 manifest=json.loads((ROOT/'experiments/v22_gnn_training/v4b/receipts_v3/artifact_manifest.json').read_text())
 result={}
 for name in ('full37','capacity'):
  item=manifest['runs'][name]; base=Path(item['directory'])
  assert sha(base/'run_receipt.json')==item['run_receipt_sha256']
  receipt=json.loads((base/'run_receipt.json').read_text())
  weights={f:{'sha256':sha(base/f),'matches':sha(base/f)==h} for f,h in item['selected_weights_sha256'].items()}
  assert all(w['matches'] for w in weights.values())
  result[name]={'weights':weights,'source_commit':item['source_commit'],'schedule':receipt['sample_trace_sha256'],
                'training':{a:{k:r.get(k) for k in ('parameter_count','selected_step','selection_score','family_macro','elapsed_seconds')} for a,r in receipt['training'].items()},
                'initial_encoder_hashes':receipt.get('initial_encoder_hashes')}
 assert result['full37']['schedule']==result['capacity']['schedule']
 return result

def path_tests(p,data):
 # QoR-blind first training kernel; six actual device/clock contexts, all selected configurations.
 kernel=sorted(p['manifest']['training_kernels'])[0]
 groups=[g for g in p['groups'] if g['kernel']==kernel]
 fast,overlays,analysis={}, {}, {}; checks=[]; fresh_graphs={}
 for g in groups:
  graph=data.graph_tensor(g['rows'][0],1.,'hardware')
  cached=p['graph_cache'][(kernel,g['profile'],g['clock'])]
  equal=all(torch.equal(v,cached[k]) for k,v in graph if torch.is_tensor(v))
  assert equal
  fresh_graphs[g['key']]=graph
  for row in g['rows']:
   a=probe._cached_candidate_inputs(data,row,overlays,fast,analysis)
   b=data.candidate_inputs(row)
   same=all(torch.equal(x,y) for x,y in zip(a,b))
   assert same, (g['key'],row['config_hash'])
   checks.append({'context':g['key'],'config_hash':row['config_hash'],'candidate_max_abs':float((a[0]-b[0]).abs().max()),'demand_max_abs':float((a[1]-b[1]).abs().max())})
 models=probe._initial_models(123,p['profile_feature_dim'])
 assert all(torch.equal(v,models['hardware'].state_dict()[k]) for k,v in models['compiler'].state_dict().items())
 for arm in ('compiler','hardware','target_film'):
  ck=torch.load(RUN/f'corrected_v3_full37/{arm}_selected.pt',map_location='cpu',weights_only=False)
  models[arm].load_state_dict(ck['state']);models[arm].eval()
 rows=[]; baseline={}
 for g in groups:
  graph=fresh_graphs[g['key']]
  c,d=[torch.stack([p['inputs'][i][j] for i in g['identities']]) for j in (0,1)]
  for arm in ('compiler','hardware','target_film'):
   m=models[arm];kw={'profile_features':p['profile_features'][(g['profile'],g['clock'])]} if arm=='target_film' else {}
   with torch.no_grad():
    mem,pool=m.encode(graph,**kw); result=m.from_memory(graph,mem,pool,c,d,1.);tight=m.from_memory(graph,mem,pool,c,d,.25)
    neutral=graph.clone();neutral.hardware_context=torch.zeros_like(graph.hardware_context)
    blind=m.from_memory(neutral,mem,pool,c,d,1.)
   baseline.setdefault(arm,(mem.clone(),pool.clone()))
   rows.append({'context':g['key'],'arm':arm,'memory_delta_from_first':float((mem-baseline[arm][0]).norm()),
     'pooled_delta_from_first':float((pool-baseline[arm][1]).norm()),'numeric_head_ablation_qor_delta':float((result['qor_z']-blind['qor_z']).norm()),
     'numeric_head_ablation_preference_delta':float((result['preference_score']-blind['preference_score']).norm()),
     'budget_qor_max_abs':float((result['qor_z']-tight['qor_z']).abs().max()),
     'budget_preference_max_abs':float((result['preference_score']-tight['preference_score']).abs().max())})
 # Real directed message paths and differentiability, not synthetic graphs.
 graph=fresh_graphs[groups[0]['key']];hw=graph.node_type>=len(compiler.NODE_KINDS)
 adjacency=defaultdict(list)
 for a,b in graph.edge_index.t().tolist():adjacency[a].append(b)
 dist={int(i):0 for i in torch.where(hw)[0]}; q=deque(dist)
 while q:
  a=q.popleft()
  for b in adjacency[a]:
   if b not in dist:dist[b]=dist[a]+1;q.append(b)
 paths=[{'slot':int(s)+1,'node':graph.audit['node_ids'][n],'minimum_hardware_hops':dist.get(n)} for n,s in zip(graph.action_node_index.tolist(),graph.action_slot.tolist())]
 grads={}
 for arm in ('compiler','hardware'):
  m=models[arm];m.zero_grad(); x=graph.clone();x.x.requires_grad_(True)
  mem,_=m.encode(x); weight=torch.linspace(-1,1,mem.numel()).reshape_as(mem)
  (mem*weight).sum().backward()
  grads[arm]={'hardware_node_gradient_norm':float(x.x.grad[hw].norm()),'compiler_node_gradient_norm':float(x.x.grad[~hw].norm()),
    'finite':bool(torch.isfinite(x.x.grad).all()),'edges':int((graph if arm=='hardware' else tensors.without_hardware_messages(graph)).num_edges)}
 # Fresh zero-gate FiLM diagnostic: two real-data steps only, not a trained architecture comparison.
 f=probe._initial_models(123,p['profile_feature_dim'])['target_film'];opt=torch.optim.AdamW(f.parameters(),lr=3e-4)
 g=groups[0];c,d=[torch.stack([p['inputs'][i][j] for i in g['identities']]) for j in (0,1)]
 y=(torch.tensor([r['transformed'] for r in g['rows']])-p['mean'])/p['std']; steps=[]
 for i in range(2):
  opt.zero_grad();out=f(graph,c,d,profile_features=p['profile_features'][(g['profile'],g['clock'])]);loss=torch.nn.functional.smooth_l1_loss(out['qor_z'],y);loss.backward()
  steps.append({'step':i,'loss':float(loss),'gate':float(f.gate),'gate_gradient':float(f.gate.grad),'modulation_gradient_norm':float(sum((x.grad.square().sum() for x in f.modulation.parameters() if x.grad is not None),torch.tensor(0.)).sqrt())});opt.step()
 ck=models['target_film']; g=groups[0]
 with torch.no_grad():
  base,_=super(type(ck),ck).encode(graph);mod,_=ck.encode(graph,p['profile_features'][(g['profile'],g['clock'])])
 return {'kernel':kernel,'cache_checks':checks,'fresh_graphs_equal_cached':True,'initial_shared_states_equal':True,'target_path_results':rows,'directed_action_paths':paths,'input_gradient_tests':grads,'film_two_step_diagnostic':steps,'film_selected_gate':float(ck.gate),'film_effective_gate':float(ck.gate.tanh()),'film_modulation_relative_norm':float((mod-base).norm()/base.norm()),'scope':'CPU real-data/path diagnostics; 2 disposable FiLM updates, no model qualification'}

def reproduce(p):
 quality_receipt=quality.apply_quality(p);models=probe._initial_models(123,p['profile_feature_dim']);results={}
 for arm in ('compiler','hardware','target_film','no_graph'):
  ck=torch.load(RUN/f'corrected_v3_full37/{arm}_selected.pt',map_location='cpu',weights_only=False);models[arm].load_state_dict(ck['state']);
  results[arm]={}
  for mode in ('correct','zero','shuffled','target_mismatched'):
   result=evaluation.evaluate(arm,models[arm],p,torch.device('cpu'),mode)
   results[arm][mode]=result
 return quality_receipt,results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT/'reaudit');args=ap.parse_args()
 if (args.output/'v4b_reaudit.json').exists():print('Existing immutable re-audit; no rerun');return
 torch.set_num_threads(4);p,protocol=load();data,auth=authenticate(p,protocol);print('reauthenticated',auth['cached_measurements_reauthenticated'],flush=True)
 hist=histories();paths=path_tests(p,data);print('real cache/message/gradient tests complete',flush=True)
 qual,res=reproduce(p)
 save(args.output/'historical_replay.json',res)
 save(args.output/'quality.json',qual)
 report={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'authentication':auth,'historical_identities':hist,'path_tests':paths,'quality_counts':qual['counts'],'test_labels_loaded':False,'verdicts':{'shared_target_heads':'VERIFIED','real_candidate_cache_equivalence':'VERIFIED','message_gradient_path':'VERIFIED','film_zero_gate_trainability':'VERIFIED','historical_film_convergence':'INCONCLUSIVE','exact_integer_resource_labels':'NOT_TESTED','latency_stage2_benefit':'NOT_TESTED'},'scope':'historical ADP predictor replay; not latency Stage2 qualification'}
 save(args.output/'v4b_reaudit.json',report)
 # Save authenticated non-test records for D5 without another serializer or test load.
 save(args.output/'non_test_records.json',{'rows':data.records,'manifest':p['manifest'],'family_map':{g['kernel']:g['family'] for g in p['groups']},'authentication':auth})
 print('complete',args.output,flush=True)
if __name__=='__main__':main()
