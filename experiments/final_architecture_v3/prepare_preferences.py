"""Reuse authenticated non-test banks and original pair sampling, with current Memory-C coverage.
QoR resource proxies remain proxies; exact timing/resource qualification is not asserted.
"""
import argparse,ast,collections,hashlib,importlib,json,math,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from LLM_branch.train import train_SFT_xattn_new as production
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.common import mailohls_contract as contract
PARENT='3aeb704967714aa08d6ca68e4e6bf1136cb0b464'
V4='63afcfc7a54e573a544678e728cdb7d99d865981'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True);ap.add_argument('--policy-package',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 source=subprocess.check_output(['git','show',PARENT+':experiments/stage1_qor_preference/prepare.py'],text=True)
 selected=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('one_pair','balance')]
 budget=('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut');ns={'digest':digest,'defaultdict':collections.defaultdict,'Counter':collections.Counter,'core':SimpleNamespace(BUDGET=budget)};exec(compile(ast.Module(body=selected,type_ignores=[]),'pinned_preference_selection','exec'),ns)
 for f in ('directive_domains.py','mailohls_contract.py','source_tool_policy.py'):
  assert (a.policy_package/f).read_bytes()==subprocess.check_output(['git','show',V4+':LLM_branch/common/'+f])
 sys.path.insert(0,str(a.policy_package.parent));domain=importlib.import_module(a.policy_package.name+'.directive_domains');tool=importlib.import_module(a.policy_package.name+'.source_tool_policy');policy=tool.load_policy(a.policy_package/'policy.json')
 records=json.loads(a.bank.read_text());assert all(r['split'] in ('train','val') for r in records['rows'])
 pilot=[json.loads(s) for s in a.pilot.read_text().splitlines()];split=json.loads(a.split.read_text());reg=inference.load_directive_domain_registry(str(a.registry));templates={};contexts={};assignments={};exclusions=collections.Counter()
 for i,r in enumerate(pilot):
  fold='train' if i in set(split['train_jsonl_idx']) else 'val';templates[r['kernel_name']]=r
  key=(r['kernel_name'],r['device'],float(r['clock_period']),r['resource_budget_id']);contexts[key]=(fold,r)
 # Source/tool validation is independent of target. Do not alter production decoder domains.
 actions={k:domain.load_source_actions(k,Path('Data/ApplicationDataset')/k) for k in templates}
 banks=collections.defaultdict(list)
 for r in records['rows']:
  if r['kernel'] not in templates or r['kernel'].replace('-','_').startswith('spcl'):
   exclusions['outside_memory_cohort_or_quarantined_spcl']+=1;continue
  key=(r['kernel'],r['config_hash'])
  if key not in assignments:
   try:
    text=templates[r['kernel']]['input'];target='\n'.join(f'{k} = {v}' for k,v in r['assignment'].items());rhs=production.build_rhs_map_from_target(target);prior={}
    for _,lhs in production.extract_ordered_lhs_plan(text):
     legal=contract.filter_semantic_candidates(lhs,inference.get_rhs_candidates_for_lhs(r['kernel'],lhs,reg),prior,reg[inference.normalize_kname(r['kernel'])])
     if rhs[lhs.upper()] not in legal:raise ValueError('outside source domain')
     prior[lhs.upper()]=rhs[lhs.upper()]
    tool.validate_source_assignment(r['assignment'],actions[r['kernel']]);tool.validate_complete_assignment(r['assignment'],actions[r['kernel']],policy);assignments[key]=True
   except (ValueError,KeyError):assignments[key]=False
  if not assignments[key]:exclusions['source_or_tool_illegal']+=1;continue
  banks[(r['kernel'],r['device'],float(r['clock']))].append(r)
 allpairs={'train':[],'val':[]};combinations=collections.Counter()
 for key,(fold,template) in sorted(contexts.items()):
  kernel,device,clock,bid=key;fraction=[float(template[b]) for b in budget];feasible=[]
  for r in banks[(kernel,device,clock)]:
   if r['split']!=fold:raise ValueError('family membership conflict')
   physical=r['physical']
   if not all(v is not None and math.isfinite(float(v)) and float(v)>=0 for v in physical) or physical[0]<=0:continue
   # Original bank resource observations are percentage-derived proxies. Preserve their status.
   if any(float(v)>float(cap)*f+1e-9 for v,cap,f in zip(physical[1:],r['capacities'],fraction)):continue
   feasible.append({'candidate_id':r['config_hash'],'latency':physical[0],'assignment':r['assignment'],'measurement':r})
  request={'request_id':digest(key),'family':template['_family'],'physical_context':[kernel,template['source_key'],'PARETO_LATENCY',device,clock],'budget':dict(zip(budget,fraction))}
  pair,why,n=ns['one_pair'](request,feasible,123);combinations[fold]+=n
  if pair is None:exclusions[str(why)]+=1;continue
  pair['row']=template
  def target(assignment):
   lines=[]
   for _,lhs in production.extract_ordered_lhs_plan(template['input']):lines.append(f'{lhs} = {assignment[lhs.upper()]}')
   return '\n'.join(lines)
  pair.update(preferred_target=target(pair['preferred']['assignment']),rejected_target=target(pair['rejected']['assignment']),source_sha256=hashlib.sha256(template['input'].encode()).hexdigest(),resource_evidence='percentage-derived proxy; no exact count or timing qualification')
  allpairs[fold].append(pair)
 allpairs['train']=ns['balance'](allpairs['train'],8,123)
 a.out.mkdir(parents=True)
 for fold,pairs in allpairs.items():(a.out/(fold+'_pairs.jsonl')).write_text(''.join(json.dumps(p)+'\n' for p in pairs))
 report={'schema':'v3-authenticated-preferences-v1','sampler_commit':PARENT,'sampler_functions':['one_pair','balance'],'sampler_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'inputs':{str(p):sha(p) for p in (a.bank,a.pilot,a.split,a.registry)},'counts':{k:len(v) for k,v in allpairs.items()},'families':{k:dict(collections.Counter(p['request']['family'] for p in v)) for k,v in allpairs.items()},'eligible_combinations':dict(combinations),'exclusions':dict(exclusions),'resource_qualification':'proxy-derived feasibility; exact-count and timing-qualified DPO release is blocked','timing_estimates_available':False,'spcl':'excluded until extreme latency-unit discrepancy is resolved','test_accessed':False,'pair_sha256':{k:sha(a.out/(k+'_pairs.jsonl')) for k in allpairs}}
 (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
