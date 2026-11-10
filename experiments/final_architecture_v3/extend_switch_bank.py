"""One prospective compiler-linked co-edit per development family.
First bank was near-tied: this is a bounded data-coverage extension, not model
selection. Labels remain unknown until Coroni measurement.
"""
import argparse,copy,importlib,json,re,sys
from pathlib import Path
from GNN_branch.mlir_graph_v2 import read_graph
from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations
from experiments.refinement import configurations as c
from .prepare_preferences import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--policy-package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 base=json.loads(a.base.read_text());prior=json.loads(a.results.read_text());assert prior['physical_calls_reserved']==8;sys.path.insert(0,str(a.policy_package.parent));domain=importlib.import_module(a.policy_package.name+'.directive_domains');tool=importlib.import_module(a.policy_package.name+'.source_tool_policy');policy=tool.load_policy(a.policy_package/'policy.json');jobs=[];exclusions={}
 for kernel in sorted({j['request']['kernel'] for j in base['jobs']}):
  j=next(x for x in base['jobs'] if x['request']['kernel']==kernel and x['anchor']);request=j['request'];template=Path(request['source_template']);actions=domain.load_source_actions(kernel,template.parent);gp=a.graphs/(kernel+'.json');graph=read_graph(gp);relations=build_action_relations(graph);candidates=[]
  for row in relations['ledger']:
   q,k=row['query_action'],row['key_action']
   if row.get('source_relation_type')=='memory_access' and row['evidence_status']=='proven' and row['routing_eligible'] and actions[q].kind=='loop' and actions[k].kind=='array':candidates.append((int(q[1:]),q,k))
  if not candidates:exclusions[kernel]='No proven loop-array eligibility; no invented co-edit';continue
  # Highest absolute slot is a deterministic tie-break, not a nesting-depth claim.
  # Include all arrays with a proven eligible access from that selected loop.
  loop=max(candidates)[1];arrays=sorted({k for _,q,k in candidates if q==loop});assignment=copy.deepcopy(j['assignment']);evidence=[r for r in relations['ledger'] if r['query_action']==loop and r['key_action'] in arrays and r.get('source_relation_type')=='memory_access' and r['evidence_status']=='proven' and r['routing_eligible']]
  assignment[c.field_key('UNROLL',loop)]='8'
  for array in arrays:
   assignment[c.field_key('ARRAY_T',array)]='cyclic';assignment[c.field_key('ARRAY_F',array)]='8';key=c.field_key('ARRAY_D',array)
   if assignment[key]=='0':assignment[key]=str(min(d for d,extent in actions[array].dimensions if extent>=8))
  try:tool.validate_source_assignment(assignment,actions);tool.validate_complete_assignment(assignment,actions,policy)
  except ValueError as error:exclusions[kernel]=str(error);continue
  for clock in (10.,3.33):jobs.append({'id':kernel+':coedit8:'+str(clock),'request':dict(request,clock_ns=clock,experiment_arm='development_compiler_coedit_extension'),'assignment':assignment,'candidate_hash':c.signature(assignment),'anchor':False,'compiler_evidence':evidence,'graph_sha256':sha(gp),'relation_schema_id':relations['relation_schema_id'],'new_design':'whole legal co-edit, not an inherited measured QoR label'})
 result={'schema':'v3-switch-bank-coedit-extension-v1','base_manifest_sha256':sha(a.base),'base_measurement_sha256':sha(a.results),'runner':base['runner'],'jobs':jobs,'exclusions':exclusions,'protocol':{'reason':'Original two candidate pairs are within1% at each measured clock; add one compiler-linked unroll/partition co-edit to test a wider resource/latency tradeoff. This is a prospective data-coverage amendment before scoring these designs.','max_additional_unique_jobs':4,'timeout_per_job_seconds':300,'global_attempts_before':19,'physical_call_cap':48,'thresholds_unchanged':True,'no_model_training_or_locked_test_use':True,'claim':'best among this small measured pool, never global optimum'}}
 a.out.mkdir(parents=True);(a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('additional jobs',len(jobs),'exclusions',exclusions)
if __name__=='__main__':main()
