"""Normal Stage3 constrained inference with explicit, bounded latency refinement.

All standard decoder arguments pass through unchanged. Without --refinement-enabled,
no oracle, source snapshot, SSH or Vitis request is constructed.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from LLM_branch.inference import eval_stage1_stage2_stage3 as decoder
from LLM_branch.common import directive_domains as domains,mailohls_contract as contract
from experiments.refinement import configurations as c
from experiments.refinement_search.deployment import optimize,ExactReceiptOracle

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--allow-unqualified-checkpoint',action='store_true');p.add_argument('--refinement-enabled',action='store_true');p.add_argument('--refinement-budget',type=int,default=4);p.add_argument('--refinement-policy',choices=['coordinate','structural','hybrid'],default='coordinate');p.add_argument('--refinement-request',type=Path);p.add_argument('--refinement-run-dir',type=Path);p.add_argument('--refinement-archive',type=Path);p.add_argument('--refinement-coroni-runner',type=Path);p.add_argument('--refinement-coroni-directory');p.add_argument('--refinement-max-new-calls',type=int,default=8);p.add_argument('--refinement-prior-physical-calls',type=int);a,argv=p.parse_known_args()
 if '--stage' not in argv:argv=['--stage','stage3',*argv]
 if argv[argv.index('--stage')+1]!='stage3':p.error('Normal endpoint requires Stage3; invoke the original decoder directly for Stage1/2 ablations')
 if '--adapter_dir' not in argv:p.error('Stage3 adapter directory required')
 parent=Path(argv[argv.index('--adapter_dir')+1]);checkpoint_contract=json.loads((parent/'training_contract.json').read_text());qualification=checkpoint_contract.get('stage3_preference',{}).get('qualification','NOT_QUALIFIED')
 if qualification not in ('QUALIFIED','CONSERVATIVELY_RETAINED') and not a.allow_unqualified_checkpoint:raise ValueError('Stage3 checkpoint is not qualified for default deployment; explicit --allow-unqualified-checkpoint is research-only')
 weight=parent/'structural_xattn.pt';stage3_sha=hashlib.sha256(weight.read_bytes()).hexdigest()
 if a.refinement_enabled and not all((a.refinement_request,a.refinement_run_dir)):p.error('Enabled refinement requires an exact request manifest and persistent run directory')
 original=decoder.predict_case
 def predict(**kwargs):
  result=original(**kwargs);result['checkpoint_qualification']=qualification;assignment=decoder.parse_assignment_dict(result['candidates'][0]['prediction'])
  if not a.refinement_enabled:
   result['final_assignment']=assignment;result['refinement']={'enabled':False,'logical_queries':0,'new_physical_calls':0};return result
  request=json.loads(a.refinement_request.read_text());case=kwargs['case'];platform=case.platform_row
  if request['kernel']!=case.kernel_name or request['part']!=platform['device'] or float(request['clock_ns'])!=float(platform['selected_clock_period']) or case.obj_mode!=request['objective']:raise ValueError('Refinement and decoder target contexts differ')
  actual_budget={res:int(platform[contract.AVAIL_FIELD_BY_RESOURCE[res]]) for res in c.RESOURCE_KEYS}
  if actual_budget!=request['budgets']:raise ValueError('Refinement and decoder budgets differ')
  if hashlib.sha256(case.source_text.encode()).hexdigest()!=request['prompt_source_sha256']:raise ValueError('Decoder source differs from refinement request')
  directory=Path(request['source_directory']);metadata=directory/'kernel_info.txt'
  if c.sha256(metadata)!=request['action_metadata_sha256']:raise ValueError('Action metadata changed')
  actions=domains.load_source_actions(case.kernel_name,directory);native=c.legacy_domains(actions,directory)
  problem={'baseline':assignment,'actions':actions,'native_domains':native,'relations':request.get('verified_relations',[]),'inactive_actions':request.get('inactive_actions',[])}
  # Derive relations from the authenticated graph; never trust a supplied table of claims.
  if request.get('compiler_graph_path'):
   from experiments.refinement_search.rc3_relations import from_rc3
   problem['relations']=from_rc3(request['compiler_graph_path'],request['compiler_graph_sha256'],actions)
  elif problem['relations']:raise ValueError('Relation claims require authenticated RC3 graph evidence')
  if a.refinement_coroni_runner:
   from experiments.refinement_search.coroni import CoroniOracle
   if not a.refinement_coroni_directory:p.error('Explicit run-specific Coroni directory required')
   if a.refinement_prior_physical_calls is None:p.error('Coroni execution requires the current aggregate --refinement-prior-physical-calls ledger charge')
   oracle=CoroniOracle(request,request['source_template'],[directory/n for n in request['headers_sha256']],a.refinement_coroni_runner,a.refinement_run_dir/'physical',a.refinement_coroni_directory,cache=a.refinement_archive,max_new_calls=a.refinement_max_new_calls,prior_calls=a.refinement_prior_physical_calls)
  elif a.refinement_archive:oracle=ExactReceiptOracle(a.refinement_archive)
  else:p.error('Select authenticated replay archive or the Coroni runner explicitly')
  refined=optimize(request,assignment,problem=problem,refinement_enabled=True,refinement_budget=a.refinement_budget,refinement_policy=a.refinement_policy,run_dir=a.refinement_run_dir/'policies'/a.refinement_policy,oracle=oracle,stage3_sha256=stage3_sha)
  result['final_assignment']=refined['assignment'];result['refinement']=refined;return result
 decoder.predict_case=predict;sys.argv=[sys.argv[0],*argv]
 try:decoder.main()
 finally:decoder.predict_case=original
if __name__=='__main__':main()
