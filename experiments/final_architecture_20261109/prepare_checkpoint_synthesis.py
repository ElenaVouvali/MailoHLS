"""Seal checkpoint0/150 complete natural predictions from matched saved evaluations.

Uses the exact same JSONL request identity at both steps; never compares the
U20050%-budget evaluation to an unrelated full-budget decode.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
from .audit import ROOT,OUT,sha

def evaluated_request(rows, selection):
 """Trainer jsonl_idx is a line number, not preserved upstream _jsonl_idx."""
 row=rows[selection['jsonl_idx']]
 if row['_quality_split']!='validation':raise ValueError('Selection crossed split')
 for key in ('kernel_name','device','resource_budget_id'):
  if row[key]!=selection[key]:raise ValueError('Selection/request mismatch: '+key)
 if float(row['selected_clock_period'])!=float(selection['selected_clock_period']):raise ValueError('Selection/request clock mismatch')
 return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT/'checkpoint_0_150_synthesis_v3');ap.add_argument('--runner',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 # Reuse the already authenticated policy snapshots from the lava preparation.
 import sys,importlib
 refs=OUT/'references';sys.path.insert(0,str(refs))
 domains=importlib.import_module('salt_qualification_contract.directive_domains');tool=importlib.import_module('salt_qualification_contract.source_tool_policy');contract=importlib.import_module('salt_qualification_contract.mailohls_contract')
 from experiments.refinement.synthesis import materialize
 from LLM_branch.inference.eval_stage1_stage2_stage3 import parse_assignment_dict
 from Preprocessing.create_jsonl import application_sources
 policy=tool.load_policy(refs/'salt_qualification_contract/policy.json')
 datafile=Path('/home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/audit/stage2_pilot.jsonl');data={i:r for i,r in enumerate(map(json.loads,datafile.read_text().splitlines()))}
 base=OUT/'davinci/static';evaluations={s:json.loads((base/f'val_selection_step_{s}.json').read_text()) for s in (0,150)}
 selected=[r for r in evaluations[0]['rows'] if float(r['selected_clock_period'])==10.]
 assert len(selected)==4
 sources=application_sources({r['kernel_name'] for r in selected});jobs=[];excluded=[];dedup={};args.output.mkdir(parents=True);shutil.copy2(args.runner,args.output/'run_batch.py')
 settings=['config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0']
 # Reference request first; one matched source pair at a time, source with known short compile first.
 for r0 in sorted(selected,key=lambda x:('pathfinder' not in x['kernel_name'],x['device']!='xczu7ev-ffvc1156-2-e')):
  row=evaluated_request(data,r0)
  source=sources[r0['kernel_name']];metadata=source.parent/'kernel_info.txt';actions=domains.load_source_actions(r0['kernel_name'],source.parent)
  for step in (0,150):
   rec=next(x for x in evaluations[step]['rows'] if x['case_id']==r0['case_id']);assert rec['jsonl_idx']==r0['jsonl_idx']
   assignment=parse_assignment_dict(rec['prediction']);h=hashlib.sha256(json.dumps(assignment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   attr={'checkpoint_step':step,'case_id':rec['case_id'],'request_jsonl_idx':r0['jsonl_idx'],'prediction':rec['prediction'],'assignment_sha256':h,'evaluation_receipt_sha256':sha(base/f'val_selection_step_{step}.json'),'request_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),'input_source_sha256':hashlib.sha256(row['input'].encode()).hexdigest(),'requested_budget':{k:row['avail_'+k.lower()] for k in ('BRAM','DSP','FF','LUT')}}
   try:tool.validate_source_assignment(assignment,actions);tool.validate_complete_assignment(assignment,actions,policy)
   except ValueError as exc:excluded.append({**attr,'reason':str(exc)});continue
   key=(r0['kernel_name'],row['device'],float(row['selected_clock_period']),h)
   if key in dedup:jobs[dedup[key]]['attributions'].append(attr);continue
   jid='cp_%02d'%(len(jobs)+1);d=args.output/'jobs'/jid;d.mkdir(parents=True)
   (d/source.name).write_text(materialize(row['input'],'\n'.join(k+' = '+v for k,v in assignment.items())));headers={}
   for f in sorted(source.parent.glob('*.h')):shutil.copy2(f,d/f.name);headers[f.name]=sha(f)
   caps=contract.DEVICE_RESOURCES[row['device']];budget={'BRAM_18K':row['avail_bram'],'DSP':row['avail_dsp'],'FF':row['avail_ff'],'LUT':row['avail_lut']}
   job={'job_id':jid,'kernel':r0['kernel_name'],'arm':'step'+str(step),'request_number':len(jobs)+1,'source':source.name,'top':'viterbi' if r0['kernel_name']=='machsuite-viterbi' else 'workload','part':row['device'],'clock_ns':float(row['selected_clock_period']),'capacities':caps,'budget':budget,'materialized_source_sha256':sha(d/source.name),'original_source_sha256':sha(source),'metadata_sha256':sha(metadata),'headers':headers,'assignment':assignment,'attributions':[attr],'split':'validation','functional_validation':False}
   tcl='\n'.join(['open_project -reset hls_project','set_top '+job['top'],'add_files -cflags {-I"/opt/Xilinx/Vitis_HLS/2021.1/include"} {'+job['source']+'}','open_solution -reset solution1 -flow_target vivado','set_part {'+job['part']+'}','create_clock -period '+str(job['clock_ns'])+' -name default',*settings,'csynth_design','exit','']);(d/'run_hls.tcl').write_text(tcl);job['expected_tcl_sha256']=sha(d/'run_hls.tcl');dedup[key]=len(jobs);jobs.append(job)
 m={'schema':'architecture-r2-checkpoint-0-150-matched-synthesis-v1','jobs':jobs,'settings':settings,'objective':'PARETO_LATENCY','data_sha256':sha(datafile),'runner_sha256':sha(args.runner),'source_tool_policy':policy.policy_id,'policy_sha256':policy.source_sha256,'excluded':excluded,'protocol':{'comparison':'same source/target/clock/budget/request at step0vs150; cached constrained selection outputs','late_checkpoint':'fixed final step150 diagnostic, not selected using HLS','timeout_seconds':1200,'scope':'two validation families; no test; HLS estimates only','go_no_go':'synthesize both model outputs even if one fails; no latency imputation','heldout_scoring':'do not use these outcomes to tune more variants without new development/confirmation protocol'}}
 def seal(d,m):
  (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(d/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(d))+'\n' for f in sorted(d.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 seal(args.output,m)
 # ZCU reference pairs for both families; no U200 release before reviewing this bounded batch.
 first=[j for j in jobs if j['part']=='xczu7ev-ffvc1156-2-e'];d=args.output.with_name(args.output.name+'_reference');d.mkdir();shutil.copy2(args.runner,d/'run_batch.py')
 for j in first:shutil.copytree(args.output/'jobs'/j['job_id'],d/'jobs'/j['job_id'])
 seal(d,{**m,'jobs':first})
 print(json.dumps({'physical_jobs':len(jobs),'reference_jobs':len(first),'excluded':excluded,'reference_manifest_sha256':sha(d/'manifest.json')},indent=2))
if __name__=='__main__':main()
