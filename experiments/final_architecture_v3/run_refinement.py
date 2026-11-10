"""Apply the maintained deployment API to a sealed actual Stage3 prediction.
Dry-run first; same causal policy, opaque physical cache, hard aggregate call cap.
"""
import argparse,json,re
from pathlib import Path
from LLM_branch.common import directive_domains as domains
from experiments.refinement import configurations as c
from experiments.refinement_search.deployment import optimize
from experiments.refinement_search.coroni import CoroniOracle
from experiments.refinement_search.rc3_relations import from_rc3

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--stage3',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prior-physical-calls',type=int,required=True);p.add_argument('--max-new-calls',type=int,default=15);p.add_argument('--timeout',type=int,default=300);p.add_argument('--execute',action='store_true');a=p.parse_args()
 request=json.loads(a.request.read_text());contract=json.loads((a.stage3/'training_contract.json').read_text())
 if contract['stage']!='stage3':raise ValueError('Stage3 baseline required')
 matches=[r for r in c.read_jsonl(a.predictions) if r['kernel_name']==request['kernel'] and r['device']==request['part'] and float(r['clock_period_ns'])==request['clock_ns'] and r['resource_budget']==request['budgets']]
 if len(matches)!=1:raise ValueError('Ambiguous Stage3 context')
 prediction=matches[0]['candidates'][0];assignment=c.parse_assignment('\n'.join(line for line in prediction['prediction'].splitlines() if not re.fullmatch(r'<L[1-9][0-9]*>',line.strip())));directory=Path(request['source_directory']);actions=domains.load_source_actions(request['kernel'],directory)
 if not prediction['schema_compliant'] or not prediction['expected_key_match']:raise ValueError('Illegal/incomplete decoder output')
 relations=from_rc3(request['compiler_graph_path'],request['compiler_graph_sha256'],actions)
 problem={'baseline':assignment,'actions':actions,'native_domains':c.legacy_domains(actions,directory),'relations':relations,'inactive_actions':[]}
 c.tcad.validate_assignments(assignment,actions);c.policy_record(assignment,assignment,actions,problem['native_domains'],'tcad')
 identity={'schema':'v3-stage3-refinement-protocol-v1','request_sha256':c.sha256(a.request),'predictions_sha256':c.sha256(a.predictions),'stage3_contract_sha256':c.sha256(a.stage3/'training_contract.json'),'stage3_weights_sha256':c.sha256(a.stage3/'structural_xattn.pt'),'policies':['coordinate','structural'],'logical_budgets':[1,4,8],'max_new_calls':a.max_new_calls,'prior_physical_calls':a.prior_physical_calls,'timeout_seconds':a.timeout,'reason_for_reallocation':'Required Stage3-start latency refinement after twelve bank measurements found clock stays and one very-tight resource boundary; prioritize direct endpoint validation over unsupported architecture sweeps. Hard48-attempt cap retained.','interpretation':'One development source in Pathfinder family, selected for finite HLS coverage before evaluating policy outcomes. No locked-test tuning. Budget includes baseline. Independent policy histories; same shared physical cache reveals only queried designs.'}
 print(json.dumps(identity,indent=2),flush=True)
 if not a.execute:return
 a.out.mkdir(parents=True,exist_ok=True);manifest=a.out/'protocol.json'
 if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Refinement protocol changed')
 c.write_json(manifest,identity)
 off=optimize(request,assignment,refinement_enabled=False);assert off['assignment']==assignment and off['logical_queries']==0
 c.write_json(a.out/'disabled_parity.json',off)
 oracle=CoroniOracle(request,request['source_template'],[directory/h for h in request['headers_sha256']],a.runner,a.out/'physical','architecture_v3_stage3_refinement_20261110',max_new_calls=a.max_new_calls,prior_calls=a.prior_physical_calls,timeout=a.timeout)
 for policy in ('coordinate','structural'):
  run=a.out/policy
  try:result=optimize(request,assignment,problem=problem,refinement_enabled=True,refinement_budget=8,refinement_policy=policy,run_dir=run,oracle=oracle,stage3_sha256=identity['stage3_weights_sha256'])
  except RuntimeError as exc:
   c.write_json(a.out/(policy+'_blocked.json'),{'status':'BLOCKED','reason':str(exc),'preserved_run':str(run)});raise
  c.write_json(a.out/(policy+'_result.json'),result);print('[REFINEMENT]',policy,result['status'],result['logical_queries'],result['objective_value'],flush=True)
if __name__=='__main__':main()
