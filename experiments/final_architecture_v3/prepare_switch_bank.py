"""Seal development candidates by source legality/structural distance, never QoR.
Two validation families, common configurations at ZCU10410/3.33ns. Anchor gates
are integrity gates, not architecture-performance selection.
"""
import argparse,collections,hashlib,importlib,json,subprocess,sys
from pathlib import Path
from experiments.refinement import configurations as c
from .prepare_preferences import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--policy-package',type=Path,required=True);p.add_argument('--tool-provenance',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);bank=json.loads(a.bank.read_text());pilot=[json.loads(s) for s in a.pilot.read_text().splitlines()];sys.path.insert(0,str(a.policy_package.parent));domain=importlib.import_module(a.policy_package.name+'.directive_domains');tool=importlib.import_module(a.policy_package.name+'.source_tool_policy');policy=tool.load_policy(a.policy_package/'policy.json');provenance=json.loads(a.tool_provenance.read_text());tool_identity={k:provenance[k] for k in ('executable_sha256','version_output','vendor_headers')};jobs=[];reasons={}
 for kernel in ('machsuite-viterbi','rodinia_pathfinder_4_doublebuffer_0'):
  template=next(r for r in pilot if r['kernel_name']==kernel);source=Path(template['source_file']);metadata=source.parent/'kernel_info.txt';actions=domain.load_source_actions(kernel,source.parent);groups={clock:{r['config_hash']:r for r in bank['rows'] if r['kernel']==kernel and r['device']=='xczu7ev-ffvc1156-2-e' and float(r['clock'])==clock and r['split']=='val'} for clock in (10.,3.33)}
  common=sorted(set(groups[10.])&set(groups[3.33]));union={**groups[10.],**groups[3.33]};candidate_ids=common if len(common)>=2 else sorted(union);legal=[]
  for cid in candidate_ids:
   assignment=union[cid]['assignment']
   if any(k.startswith('AUTO{_UNROLL') and int(v)>4 for k,v in assignment.items()):continue
   if any(k.startswith('AUTO{_ARRAY_T') and v=='complete' for k,v in assignment.items()):continue
   if any(k.startswith('AUTO{_ARRAY_F') and int(v)>8 for k,v in assignment.items()):continue
   try:tool.validate_source_assignment(assignment,actions);tool.validate_complete_assignment(assignment,actions,policy)
   except ValueError:continue
   legal.append(cid)
  pairs=[]
  for i,left in enumerate(legal):
   aa=union[left]['assignment']
   for right in legal[i+1:]:
    bb=union[right]['assignment'];fields=[k for k in aa if aa[k]!=bb[k]]
    if not any(k.startswith('AUTO{_UNROLL') for k in fields):continue
    labels={k.rsplit('_',1)[1] for k in fields}
    if len(labels)>2:continue
    pairs.append((len(labels),len(fields),left,right))
  if not pairs:reasons[kernel]={'status':'NO_ELIGIBLE_STRUCTURAL_PAIR','common_configurations':len(common),'low_cost_legal':len(legal)};continue
  _,_,left,right=min(pairs);selected=[left,right];snap=a.out/'sources'/kernel;snap.mkdir(parents=True);(snap/source.name).write_text(template['input']);(snap/'kernel_info.txt').write_bytes(metadata.read_bytes())
  for header in source.parent.glob('*.h'):(snap/header.name).write_bytes(header.read_bytes())
  caps=dict(c.CAPACITIES);request={'kernel':kernel,'objective':'PARETO_LATENCY','part':c.PART,'capacities':caps,'budgets':caps,'source_sha256':sha(snap/source.name),'original_source_sha256':sha(source),'action_metadata_sha256':sha(metadata),'headers_sha256':{h.name:sha(h) for h in snap.glob('*.h')},'top':metadata.read_text().splitlines()[0].strip(),'domain_identity':policy.policy_id,'tool_identity':tool_identity,'flow_settings':list(c.SETTINGS),'experiment_arm':'development_switch_stay','split':'validation','source_template':str((snap/source.name).resolve())}
  for choice,cid in enumerate(selected):
   for clock in (10.,3.33):jobs.append({'id':kernel+':'+str(choice)+':'+str(clock),'request':dict(request,clock_ns=clock),'assignment':union[cid]['assignment'],'missing_cross_measurement_in_original_bank':cid not in groups[clock],'candidate_hash':cid,'anchor':choice==0 and clock==10.})
  reasons[kernel]={'status':'SEALED','common_configurations':len(common),'low_cost_legal':len(legal),'candidate_ids':selected,'candidate_availability':'common pool if>=2; otherwise union with missing contexts to be physically measured','choice_rule':'minimum action distance, then field distance, then hashes among low-unroll/noncomplete legal common configurations; must differ in unroll; no latency/resource ranking'}
 result={'schema':'v3-development-switch-bank-v1','source_code_sha256':sha(Path(__file__)),'jobs':jobs,'selection':reasons,'inputs':{str(p):sha(p) for p in (a.bank,a.pilot,a.tool_provenance,a.runner)},'runner':str(a.runner.resolve()),'budget':{'max_new_calls':len(jobs),'prior_calls_including_repeats':11,'timeout_per_job':300},'protocol':{'latency_reversal_gap':.05,'near_ties':'ambiguous','clock_timing':'separate finite HLS estimate <= request subset','anchor_gate':'Run one10ns candidate per family; release remaining three only if anchor has finite authenticated latency and exact resource counts. Otherwise mark family BLOCKED, not failed hardware.','budget_diagnostic':'After measuring, define one component-wise midpoint threshold only if candidates differ in exact resource counts; label bank-derived diagnostic and do not present as in-distribution budget adaptation.','no_test_access':True,'no_architecture_selection_from_HLS_until_all_comparisons_frozen':True}}
 (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(reasons,indent=2))
if __name__=='__main__':main()
