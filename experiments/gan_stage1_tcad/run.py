"""Selected V4 Stage1 inference on the eighteen TCAD GAN requests.
No training, structural fusion, reference-label loading or synthesis.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ADAPTER_SHA='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--draft',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run'));p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-uuid',default='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if sha(a.checkpoint/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('Wrong selected V4 checkpoint')
 contract=json.loads((a.checkpoint/'training_contract.json').read_text());assert contract['objective']=='PARETO_LATENCY' and contract['stage']=='stage1'
 if sha(a.checkpoint/'directive_domain_registry.json')!=contract['directive_domain_registry_sha256']:raise ValueError('Checkpoint registry changed')
 if a.out.exists():raise FileExistsError('Fresh immutable output directory required')
 a.out.mkdir(parents=True);src=ROOT/'evaluation_set/GAN/code_to_memory_outputs/meta';snapshot=a.out/'sources/GAN';snapshot.mkdir(parents=True)
 for name in ('GAN.cpp','GAN_placeholders.cpp','kernel_info.txt'):shutil.copyfile(src/name,snapshot/name)
 for h in (ROOT/'evaluation_set/GAN').glob('*.h'):shutil.copyfile(h,snapshot/h.name)
 with zipfile.ZipFile(a.draft) as z:
  section=z.read('sections/05__Detailed_Design.tex');figure=z.read('figures/02__Methodology/target_conditioning_heatmap.pdf');(a.out/'Detailed_Design.tex').write_bytes(section);(a.out/'target_conditioning_heatmap.pdf').write_bytes(figure)
 figure_text=subprocess.check_output(['pdftotext','-layout',str(a.out/'target_conditioning_heatmap.pdf'),'-'],text=True);(a.out/'figure_text.txt').write_text(figure_text)
 for label in ('ZCU104','U200','3.33','5 ns','10 ns','20%','50%','100%'):assert label in figure_text,label
 from LLM_branch.common import directive_domains as domains,mailohls_contract as mc
 template=domains.prepare_source_template('GAN',snapshot/'GAN_placeholders.cpp',a.out/'sources');actions=domains.load_source_actions('GAN',a.out/'sources');derived=domains.build_source_domain_registry([{'kernel_name':'GAN','input':template}],a.out/'sources');dump(a.out/'derived_GAN_domains.json',derived)
 # Historical checkpoint registry remains byte-identical; normal decoder adds only the absent source-derived GAN domains in memory.
 cases=[]
 for name,part in [('ZCU104','xczu7ev-ffvc1156-2-e'),('U200','xcu200-fsgd2104-2-e')]:
  for clock in (3.33,5.,10.):
   for fraction in (.2,.5,1.):
    r={'context_id':f'GAN:{name}:{clock:g}ns:{fraction*100:g}pct','kernel_name':'GAN','input':template,'obj_mode':'PARETO_LATENCY','device':part,'clock_period':clock,'selected_clock_period':clock,'frequency_mode':'specified','resource_budget_frac':fraction,'resource_budget_id':f'uniform-{round(fraction*100):03d}'}
    r.update({field:int(round(mc.DEVICE_RESOURCES[part][resource]*fraction)) for resource,field in mc.AVAIL_FIELD_BY_RESOURCE.items()});mc.target_prompt_fields(r);cases.append(r)
 requests=a.out/'requests.jsonl';requests.write_text(''.join(json.dumps(r)+'\n' for r in cases));assert len(cases)==18
 manifest={'schema':'gan-selected-stage1-tcad-requests-v1','checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':ADAPTER_SHA,'contract_sha256':sha(a.checkpoint/'training_contract.json'),'registry_sha256':sha(a.checkpoint/'directive_domain_registry.json'),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(Path(__file__)),'draft_zip_sha256':sha(a.draft),'section_sha256':hashlib.sha256(section).hexdigest(),'figure_sha256':hashlib.sha256(figure).hexdigest(),'paper_anchor':'sections/05__Detailed_Design.tex:518-549, fig:target-conditioning','requests_sha256':sha(requests),'requests':18,'source_sha256':sha(snapshot/'GAN_placeholders.cpp'),'prompt_source_sha256':hashlib.sha256(template.encode()).hexdigest(),'metadata_sha256':sha(snapshot/'kernel_info.txt'),'header_sha256':{p.name:sha(p) for p in snapshot.glob('*.h')},'decoder':'real production constrained greedy, one sample, mean RHS continuation score, debug_topk16','candidate_policy':'original selected-checkpoint registry plus normal legacy source-domain extension for absent GAN; no silent source_tool_v2 intervention','budgets':'component-wise20/50/100percent; canonical integer rounding recorded per request','objective_scope':'Selected adapter is PARETO_LATENCY; original figure checkpoint/objective attribution is not assumed','evaluation_role':'GAN external descriptive inference only; no model/checkpoint selection or training','hardware_QoR':'Not measured in this run; historical figure values not attributed to these outputs'};dump(a.out/'manifest.json',manifest)
 print('Prepared18 authenticated requests. Execute:',a.execute,flush=True)
 if not a.execute:return
 from experiments.final_architecture_20261109.run import preflight
 receipt=preflight(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;dump(a.out/'preflight.json',receipt)
 from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
 original=inference.predict_case;start=time.monotonic()
 def checkpoint_case(**kwargs):
  r=original(**kwargs)
  with (a.out/'completed_cases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  return r
 inference.predict_case=checkpoint_case
 sys.argv=[sys.argv[0],'--stage','stage1','--adapter_dir',str(a.checkpoint),'--directive_domain_registry_json',str(a.checkpoint/'directive_domain_registry.json'),'--application_dataset_dir',str(a.out/'sources'),'--input_jsonl',str(requests),'--output_jsonl',str(a.out/'predictions.jsonl'),'--score_reduction','mean','--debug_topk','16','--num_samples','1','--print_predictions']
 try:inference.main()
 finally:inference.predict_case=original
 assert sha(a.checkpoint/'adapter_model.safetensors')==ADAPTER_SHA,'Checkpoint mutated'
 dump(a.out/'execution.json',{'status':'PASS','runtime_seconds':time.monotonic()-start,'GPU':receipt,'predictions_sha256':sha(a.out/'predictions.jsonl'),'completed_cases':len((a.out/'completed_cases.jsonl').read_text().splitlines()),'checkpoint_unchanged':True,'new_HLS_calls':0,'optimizer_updates':0})
if __name__=='__main__':main()
