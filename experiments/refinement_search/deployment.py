"""Versioned latency deployment backend for the existing causal search controller.

No Vitis subprocess runs here. The oracle must return exact-context authenticated
Coroni receipts. Missing receipts are unknown, never surrogate quality labels.
"""
import copy,dataclasses,json,math
from pathlib import Path
from experiments.refinement import configurations as c, synthesis as hls
from experiments.refinement_search import controller,policies
REQUIRED=('source_sha256','headers_sha256','action_metadata_sha256','domain_identity','part','clock_ns','capacities','budgets','tool_identity','flow_settings')

def physical_identity(request,assignment):
 return {k:request[k] for k in REQUIRED if k!='budgets'} | {'assignment':assignment}

def validate_request(request):
 if request.get('objective')!='PARETO_LATENCY':raise ValueError('Deployment backend currently qualifies latency only')
 if any(k not in request for k in REQUIRED):raise ValueError('Incomplete physical/source identity')
 if request['part'] not in ('xczu7ev-ffvc1156-2-e','xcu200-fsgd2104-2-e'):raise ValueError('Unapproved part')
 if float(request['clock_ns']) not in (10.,5.,3.33):raise ValueError('Unapproved fixed clock')
 for name in ('capacities','budgets'):
  if set(request[name])!=set(c.RESOURCE_KEYS) or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in request[name].values()):raise ValueError('Exact integer resource limits required')
 if any(request['budgets'][k]>request['capacities'][k] or request['capacities'][k]<=0 for k in c.RESOURCE_KEYS):raise ValueError('Invalid capacity/budget')

class ExactReceiptOracle:
 """No hidden rankings are disclosed to the policy, only its charged query."""
 def __init__(self,path):self.path=Path(path)
 def __call__(self,identity):
  entries=c.read_jsonl(self.path)
  hits=[e for e in entries if e['physical_identity']==identity]
  if len(hits)>1 and any(e['measurement']!=hits[0]['measurement'] for e in hits):raise ValueError('Conflicting exact receipts')
  if not hits:return None
  entry=hits[0]
  for filename,expected in entry['report_sha256'].items():
   if c.sha256(filename)!=expected:raise ValueError('Physical receipt changed')
  if not entry['report_sha256']:raise ValueError('Unbacked measurement')
  return entry

class DeploymentRuntime:
 def __init__(self,path,request,problem,oracle,budget,policy,seed,parent_stage3_sha256):
  validate_request(request)
  if policy not in ('coordinate','structural','hybrid'):raise ValueError('Unsupported deployment policy')
  if not 1<=budget<=16:raise ValueError('Logical query budget must be1..16, including baseline')
  self.path=Path(path);self.request=request;self.problem=problem;self.oracle=oracle
  if not parent_stage3_sha256 or len(parent_stage3_sha256)!=64:raise ValueError('Explicit Stage3 checkpoint identity required')
  serial={**problem,'actions':{k:dataclasses.asdict(v) for k,v in problem['actions'].items()}}
  c.policy_record(problem['baseline'],problem['baseline'],problem['actions'],problem['native_domains'],'tcad')
  data={'schema':'salt-latency-refinement-runtime-v1','request':request,'problem_sha256':c.object_hash(serial),'baseline_signature':c.signature(problem['baseline']),'max_evaluations':budget,'policy':policy,'seed':seed,'kernel':request.get('kernel','source'),'stage3_sha256':parent_stage3_sha256,'code_sha256':controller.code_hashes()}
  data=controller._sealed(data);self.path.mkdir(parents=True,exist_ok=True);f=self.path/'manifest.json'
  if f.exists():
   if json.loads(f.read_text())!=data:raise ValueError('Cannot resume a different request/model/policy')
  else:
   c.write_json(f,data);c.write_jsonl(self.path/'search_trace.jsonl',[]);c.write_jsonl(self.path/'measurements.jsonl',[])
 def load_run(self,path):
  manifest=controller._read_sealed(Path(path)/'manifest.json')
  if manifest['code_sha256']!=controller.code_hashes():raise ValueError('Refinement code changed; explicit new run required')
  return manifest,{'kernels':{manifest['kernel']:{}},'expected_tool_identity':self.request['tool_identity']}
 def problem_for(self,*_):return self.problem
 def physical_candidate(self,manifest,shared,proposal):
  return {'canonical_signature':proposal['signature'],'physical_identity':physical_identity(self.request,proposal['assignment'])}
 def read_state(self,*args):return controller.read_state(*args,runtime=self)
 def verify_result(self,record,candidate,kernel,expected_tool):
  r=record['result'];identity=candidate['physical_identity'];assert c.signature(identity['assignment'])==candidate['canonical_signature']
  if r.get('physical_identity')!=identity:raise ValueError('Cross-context physical result')
  if r.get('synthesis_status')=='success':
   metrics=hls.evaluate_metrics(r.get('latency_cycles'),r.get('resources'),self.request['capacities'],self.request['budgets'])
   timing=r.get('estimated_clock_ns');qualified=timing is not None and math.isfinite(float(timing)) and float(timing)<=float(self.request['clock_ns'])
   if not r.get('report_sha256'):raise ValueError('Success has no report witnesses')
   for path,digest in r['report_sha256'].items():
    if c.sha256(path)!=digest:raise ValueError('Report witness changed')
   if any(r.get(k)!=metrics[k] for k in ('ADP','feasible')):raise ValueError('Incorrect measured feasibility')
   if r.get('rankable')!=bool(metrics['rankable'] and qualified):raise ValueError('Unknown/missed timing must not be ranked')
  elif r.get('rankable') or r.get('feasible') is not None:raise ValueError('Failure/unknown is not feasible')
 def measure(self,manifest,shared,proposal,query_id,*args):
  identity=physical_identity(self.request,proposal['assignment']);entry=self.oracle(identity)
  if entry is None:
   result={'synthesis_status':'unavailable','ADP':None,'latency_cycles':None,'resources':None,'feasible':None,'rankable':False,'physical_identity':identity,'objective':'PARETO_LATENCY'}
  else:
   if entry['physical_identity']!=identity:raise ValueError('Oracle returned wrong physical design')
   m=entry['measurement'];metrics=hls.evaluate_metrics(m.get('latency_cycles'),m.get('resources'),self.request['capacities'],self.request['budgets']);clock=m.get('estimated_clock_ns')
   result={**metrics,**m,'synthesis_status':m['synthesis_status'],'rankable':bool(m['synthesis_status']=='success' and metrics['rankable'] and clock is not None and math.isfinite(float(clock)) and float(clock)<=self.request['clock_ns']),'objective':'PARETO_LATENCY','physical_identity':identity,'comparison_context_id':c.object_hash(self.request),'report_sha256':entry['report_sha256']}
  self.verify_result({'result':result},self.physical_candidate(manifest,shared,proposal),{},shared['expected_tool_identity'])
  new=bool(entry and entry.get('new_physical_call'));return result,{'physical_synthesis_status':'new_synthesis' if new else 'authenticated_reuse' if entry else 'unavailable','new_physical_synthesis_runs':int(new),'authenticated_measurement_reuses':int(entry is not None and not new),'physical_synthesis_runtime_seconds':float(entry.get('result',{}).get('runtime_seconds',0)) if new else 0.}
 def summarize(self,path):
  manifest,shared=self.load_run(path);_,_,history,_,terminal=self.read_state(path,manifest,shared);best=controller._best(history)
  result={'status':'verified_feasible' if best else 'unverified','assignment':best['assignment'] if best else self.problem['baseline'],'objective':'PARETO_LATENCY','objective_value':hls.objective_value(best['measurement']) if best else None,'logical_queries':len(history),'new_physical_calls':sum(r.get('new_physical_synthesis_runs',0) for r in c.read_jsonl(Path(path)/'measurements.jsonl')),'termination':terminal,'qualification':'AVAILABLE_EXPERIMENTAL','clock_convention':'cycles ranked only within fixed requested period, finite HLS estimate <= request; no post-route claim'}
  c.write_json(Path(path)/'deployment_result.json',result);return result

def optimize(request,stage3_assignment,*,problem=None,refinement_enabled=False,refinement_budget=4,refinement_policy='coordinate',run_dir=None,oracle=None,stage3_sha256=None):
 """No-refinement is exact assignment parity and performs zero oracle calls."""
 if not refinement_enabled:return {'assignment':copy.deepcopy(stage3_assignment),'refinement_enabled':False,'logical_queries':0,'new_physical_calls':0,'status':'unverified'}
 if problem is None or run_dir is None or oracle is None:raise ValueError('Refinement requires sealed source/action problem, journal and authenticated oracle')
 if stage3_assignment!=problem['baseline']:raise ValueError('Refinement must start at this Stage3 output')
 runtime=DeploymentRuntime(run_dir,request,problem,oracle,refinement_budget,refinement_policy,123,stage3_sha256)
 controller.search(run_dir,runtime=runtime)
 return {**runtime.summarize(run_dir),'refinement_enabled':True}
