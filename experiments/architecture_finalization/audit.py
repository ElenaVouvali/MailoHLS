"""Authenticate architecture inputs and replay canonical requests without QoR labels."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/architecture_finalization/audit'
PROD=Path('/home/elvouvali/MailoHLS/mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1')
V4=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 refs={}
 for name in ['stage1-refinement','stage1-qor-preference','gnn-refinement','stage1-gnn-representation-audit','rc3-fusion-foundation','rc3-stage2-multikernel-c','gemv-stage1-full-depth','refinement-mechanism','rc3-structural-memory-audit']:
  ref=('origin/research/' if name in ['stage1-gnn-representation-audit','rc3-structural-memory-audit'] else 'research/')+name
  refs[ref]=subprocess.check_output(['git','rev-parse',ref],text=True).strip()
 models={}
 for name,path in [('registered_latency',PROD),('v4_single',V4/'single_reference/run'),('v4_multi',V4/'multi_reference/run')]:
  contract=json.loads((path/'training_contract.json').read_text());config=json.loads((path/'adapter_config.json').read_text())
  models[name]={'path':str(path),'files':{n:sha(path/n) if (path/n).is_file() else None for n in ['adapter_model.safetensors','adapter_config.json','training_contract.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','directive_domain_registry.json']},'objective':contract['objective'],'model':contract['model'],'revision':contract['model_revision'],'target_modules':config['target_modules'],'lora_r':config['r'],'trainable_tokens':len(config.get('trainable_token_indices') or []),'trainables':contract.get('stage1_trainable_parameter_contract'),'data_split':contract.get('split_summary')}
 assert models['v4_multi']['files']['adapter_model.safetensors']=='28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'
 from LLM_branch.common import mailohls_contract as c
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(V4/'multi_reference/run'),local_files_only=True)
 rows=[json.loads(l) for l in (V4/'multi_reference/train.jsonl').read_text().splitlines()]
 requests=[];replay=[]
 for kernel in ['machsuite-viterbi','rodinia_pathfinder_0_baseline_0']:
  row=next(r for r in rows if r['kernel_name']==kernel)
  assert row['_quality_split']=='validation'
  for part,period in [('xczu7ev-ffvc1156-2-e',10.),('xczu7ev-ffvc1156-2-e',5.),('xcu200-fsgd2104-2-e',10.)]:
   request=dict(context_id=f'{kernel}:{part}:{period}',kernel_name=kernel,input=row['input'],device=part,clock_period=period,selected_clock_period=period,frequency_mode='specified',objective='PARETO_LATENCY',obj_mode='PARETO_LATENCY',**{v:c.DEVICE_RESOURCES[part][k] for k,v in c.AVAIL_FIELD_BY_RESOURCE.items()})
   prompt=c.build_prompt(request['input'],'PARETO_LATENCY',c.target_prompt_fields(request));ids=tok(prompt,add_special_tokens=False)['input_ids']; assert len(ids)<7168
   requests.append(request);replay.append({'context_id':request['context_id'],'source_sha256':hashlib.sha256(row['input'].encode()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'token_count':len(ids),'prompt_token_ids':ids,'fields':c.target_prompt_fields(request)})
 (OUT/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
 (OUT/'prompt_replay.json').write_text(json.dumps(replay,indent=2)+'\n')
 gnn=Path('/home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37/compiler_selected.pt')
 receipt={'schema':'architecture-input-audit-v1','refs':refs,'models':models,'gnn_sha256':sha(gnn),'request_sha256':sha(OUT/'requests.jsonl'),'prompt_replay_sha256':sha(OUT/'prompt_replay.json'),'test_labels_read':0,'draft_zip':'absent; TCAD sections/figures cannot be authenticated','code_files':{n:sha(ROOT/n) for n in ['FINAL_ARTIFACTS.json','LLM_branch/common/mailohls_contract.py','LLM_branch/common/structural_memory.py','LLM_branch/common/structural_xattn.py','LLM_branch/inference/eval_stage1_stage2_stage3.py','LLM_branch/train/train_SFT_xattn_new.py','LLM_branch/train/train_DPO_harp_xattn.py']}}
 (OUT/'identity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'models':{k:v['files']['adapter_model.safetensors'] for k,v in models.items()},'requests':len(requests),'prompt_lengths':[r['token_count'] for r in replay]},indent=2))
if __name__=='__main__':main()
