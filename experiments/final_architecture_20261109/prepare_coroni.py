"""Seal one training-family clock-switch confirmation; never launch HLS here."""
import argparse,hashlib,importlib,json,shutil,subprocess,sys
from pathlib import Path
from .audit import ROOT,OUT,sha
REF='63afcfc7a54e573a544678e728cdb7d99d865981'
RUNNER_SHA='ec05f6b97479a30ebc1c3b45eea7758881889cc5c87e64234ae85b610e7f1414'

def main():
 p=argparse.ArgumentParser();p.add_argument('--runner',type=Path,required=True);p.add_argument('--output',type=Path,default=OUT/'lava_confirmation');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 assert sha(a.runner)==RUNNER_SHA
 refs=OUT/'references/salt_qualification_contract';refs.mkdir(parents=True,exist_ok=True);(refs/'__init__.py').touch()
 sources={}
 for f in ('directive_domains.py','mailohls_contract.py','source_tool_policy.py'):
  b=subprocess.check_output(['git','show',REF+':LLM_branch/common/'+f],cwd=ROOT);(refs/f).write_bytes(b);sources[f]=hashlib.sha256(b).hexdigest()
 policy=subprocess.check_output(['git','show',REF+':experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json'],cwd=ROOT);(refs/'policy.json').write_bytes(policy)
 sys.path.insert(0,str(refs.parent));domains=importlib.import_module('salt_qualification_contract.directive_domains');tool=importlib.import_module('salt_qualification_contract.source_tool_policy');contract=importlib.import_module('salt_qualification_contract.mailohls_contract')
 from GNN_branch.insert_placeholders import insert_placeholders
 from experiments.refinement.synthesis import materialize
 kernel='rodinia_lavaMD_0_baseline'; source_dir=ROOT/'Data/ApplicationDataset'/kernel;source=source_dir/'lavaMD.cpp';metadata=source_dir/'kernel_info.txt'
 actions=domains.load_source_actions(kernel,source_dir);template=''.join(insert_placeholders(str(source),allowed_labels=set(actions)))
 audit=json.loads((OUT/'contrasts/target_contrasts.json').read_text());contrast=next(r for r in audit['rows'] if r['kernel']==kernel and r['axis']=='clock' and r['left_context'][2]==10 and r['right_context'][2]==5 and r['status']=='latency_reversal')
 hashes=[contrast['left_acceptable_hashes'][0],contrast['right_acceptable_hashes'][0]]
 allrows=json.loads((OUT/'reaudit/non_test_records.json').read_text())['rows'];rows=[r for r in allrows if r['kernel']==kernel and r['profile']=='u200' and r['clock'] in (10.,5.) and r['config_hash'] in hashes]
 assert len(rows)==4 and all(r['split']=='train' for r in rows)
 jobs=[];a.output.mkdir(parents=True);shutil.copy2(a.runner,a.output/'run_batch.py')
 settings=['config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0'];profile=tool.load_policy(refs/'policy.json')
 for clock in (10.,5.):
  for h in hashes:
   row=next(r for r in rows if r['clock']==clock and r['config_hash']==h);assignment=row['assignment'];tool.validate_source_assignment(assignment,actions);tool.validate_complete_assignment(assignment,actions,profile)
   text=materialize(template,'\n'.join(k+' = '+v for k,v in assignment.items()));jid='lava_%02d'%(len(jobs)+1);d=a.output/'jobs'/jid;d.mkdir(parents=True);(d/source.name).write_text(text)
   headers={}
   for f in sorted(source_dir.glob('*.h')):shutil.copy2(f,d/f.name);headers[f.name]=sha(f)
   caps=contract.DEVICE_RESOURCES['xcu200-fsgd2104-2-e']
   j={'job_id':jid,'kernel':kernel,'arm':'common_assignment_'+h[:12],'request_number':len(jobs)+1,'source':source.name,'top':'workload','part':'xcu200-fsgd2104-2-e','clock_ns':clock,'capacities':caps,'budget':caps,'assignment':assignment,'config_hash':h,'original_source_sha256':sha(source),'metadata_sha256':sha(metadata),'materialized_source_sha256':sha(d/source.name),'headers':headers,'historical_latency_ms':row['physical'][0],'historical_resource_proxies':row['physical'][1:],'split':'train','historical_measurements_are_comparison_inputs_not_new_ground_truth':True}
   tcl='\n'.join(['open_project -reset hls_project','set_top workload','add_files -cflags {-I"/opt/Xilinx/Vitis_HLS/2021.1/include"} {'+source.name+'}','open_solution -reset solution1 -flow_target vivado','set_part {'+j['part']+'}','create_clock -period '+str(clock)+' -name default',*settings,'csynth_design','exit','']);(d/'run_hls.tcl').write_text(tcl);j['expected_tcl_sha256']=sha(d/'run_hls.tcl');jobs.append(j)
 manifest={'schema':'architecture-r2-lava-clock-confirmation-v1','jobs':jobs,'settings':settings,'objective':'PARETO_LATENCY','selection':'first audited lavaMD U20010vs5 common-pool reversal; preserve complete original configurations','source_policy':{'commit':REF,'source_sha256':sources,'policy_sha256':hashlib.sha256(policy).hexdigest(),'policy_id':profile.policy_id},'runner_sha256':RUNNER_SHA,'protocol':{'anchor_timeout_seconds':1200,'remaining_jobs_gate':'finite valid anchor latency and original-source/tool identity; otherwise stop','total_pool_jobs':4,'new_unique_HLS_cap_global':48,'prior_architecture_calls':2,'functional_validation':False,'timing':'HLS estimate only'}}
 def publish(path,m):
  (path/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(path/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(path))+'\n' for f in sorted(path.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 publish(a.output,manifest);probe=a.output.with_name(a.output.name+'_anchor');probe.mkdir();shutil.copy2(a.runner,probe/'run_batch.py');shutil.copytree(a.output/'jobs'/jobs[0]['job_id'],probe/'jobs'/jobs[0]['job_id']);publish(probe,{**manifest,'jobs':jobs[:1]})
 print(json.dumps({'bundle':str(a.output),'anchor':str(probe),'jobs':4,'anchor_manifest_sha256':sha(probe/'manifest.json')},indent=2))
if __name__=='__main__':main()
