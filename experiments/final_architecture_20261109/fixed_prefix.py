"""Counterfactual prefix replay using pinned helper and real production RHS scorer.
All previous fields use the same ZCU104/10ns Stage1 assignment per source.
No output from this utility is an official model prediction.
"""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
import torch
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.common import mailohls_contract as contract
from .evidence_controls import changed
HELPER_COMMIT='97ed75ed3f41ad80247d739888d9ee488881872c'
HELPER_PATH='experiments/gemv_framework_deep_dive/common.py'
source=subprocess.check_output(['git','show',HELPER_COMMIT+':'+HELPER_PATH],text=True)
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('digest_object','prompt_parts','decision_prefixes')]
assert len(nodes)==3
space={'contract':contract,'hashlib':hashlib,'json':json}
exec(compile(ast.Module(body=nodes,type_ignores=[]),HELPER_PATH,'exec'),space)
prefixes=space['decision_prefixes'];digest=space['digest_object']
references={};reference_rows={}
for line in Path(os.environ['SALT_FIXED_REFERENCE']).read_text().splitlines():
 row=json.loads(line)
 if row['device']=='xczu7ev-ffvc1156-2-e' and row['clock_period_ns']==10:
  references[row['kernel_name']]=inference.parse_assignment_dict(row['candidates'][0]['prediction']);reference_rows[row['kernel_name']]=row
prior_hashes={};domain_hashes={}


def checked(**kwargs):
 started=time.monotonic()
 case,model,tok=kwargs['case'],kwargs['model'],kwargs['tok'];platform=dict(case.platform_row);platform['input']=case.source_text;platform['objective']=case.obj_mode
 registry=kwargs['directive_domain_registry'];kernel_key=inference.normalize_kname(case.kernel_name);domains=registry[kernel_key];assignment=references[case.kernel_name]
 device=inference.get_first_real_device(model);gates=[p for n,p in model.named_parameters() if n.endswith('attn_gate')];saved=[p.detach().clone() for p in gates]
 records=[];vectors=[]
 selected={'machsuite-viterbi':('ARRAY_F_L1','ARRAY_D_L1','PIPE_L6','UNROLL_L6','PIPE_L8','UNROLL_L8'),'rodinia_pathfinder_0_baseline_0':('ARRAY_F_L3','ARRAY_F_L4','PIPE_L5','UNROLL_L5','PIPE_L6','UNROLL_L6')}[case.kernel_name]
 for mode in os.environ.get('SALT_FIXED_MODES','authentic,zero_evidence,permuted_evidence,gate_off').split(','):
  prior={};bank=changed(kwargs['mem_bank'],mode) if mode in ('zero_evidence','permuted_evidence','cross_route_evidence') else kwargs['mem_bank']
  memory,valid,relation=inference.get_real_memory_pack_for_kernel(bank,case.kernel_name,64,128,'compiler_relational')
  pack=bank.get(case.kernel_name) or bank[kernel_key]
  with torch.no_grad():
   for p,v in zip(gates,saved):p.copy_(torch.zeros_like(v) if mode=='gate_off' else v)
  try:
   model.condition_structural_memory(memory.to(device),valid.to(device),action_relation_mask=relation.to(device))
   for prefix in prefixes(platform,tok,assignment):
    lhs=prefix['field'];value=assignment[lhs.upper()]
    if not any(lhs.endswith(x+'}') for x in selected):prior[lhs.upper()]=value;continue
    choices=contract.filter_semantic_candidates(lhs,inference.get_rhs_candidates_for_lhs(case.kernel_name,lhs,registry),prior,domains)
    prior_hash=digest(prefix['prior_output_ids']);key=(case.kernel_name,lhs)
    if key in prior_hashes:assert prior_hashes[key]==prior_hash
    prior_hashes[key]=prior_hash
    if key in domain_hashes:assert domain_hashes[key]==digest(choices)
    domain_hashes[key]=digest(choices)
    ids=torch.tensor([prefix['ids']],device=device);route=len(prefix['ids'])-len(prefix['prior_output_ids'])-len(contract.selected_clock_response_token_ids(platform,tok))
    options=dict(model=model,tok=tok,base_input_ids=ids,base_attention_mask=torch.ones_like(ids),routing_start_idx=torch.tensor([route],device=device),use_structural_memory=True)
    scores=[]
    for rhs in choices:
     stats=inference.score_rhs_candidate_suffix(**options,candidate_text=rhs+'\n') if len(choices)>1 else {'mean_logprob':0.,'sum_logprob':0.,'token_count':0}
     scores.append({'rhs':rhs,'score':stats['mean_logprob'],**stats})
    if mode=='gate_off' and case.platform_row.get('context_id')==reference_rows[case.kernel_name]['context_id']:
     expected=next(x for x in reference_rows[case.kernel_name]['candidates'][0]['site_debug'] if x['lhs']==lhs)
     old={x['rhs']:x['score'] for x in expected['top_candidates']}
     assert all(abs(x['score']-old[x['rhs']])<=1e-5 for x in scores if x['rhs'] in old),'Gate-off differs from V4 baseline scores'
    scores.sort(key=lambda x:x['score'],reverse=True);rhs=scores[0]['rhs'];position=len(prefix['ids'])-1;captures={};handles=[]
    for name,module in model.named_modules():
     if not name.endswith('gated_cross_attn_layer'):continue
     layer=int(name.split('.layers.')[1].split('.')[0])+1;data={};captures[layer]=data
     def before(m,inputs,data=data):data['before']=inputs[0][0,position].detach().float().cpu().clone()
     def after(m,inputs,output,data=data):data['after']=output[0,position].detach().float().cpu().clone();data['effective_gate']=float(m.attn_gate.detach().tanh()*m.attn_gate_scale)
     def observe(attention,mask,data=data):
      weights=attention[0,:,position,:].detach().float().cpu().clone();allowed=mask[0,0,position,:].detach().cpu().bool();assert not weights[:,~allowed].any();data['attention']=weights;data['allowed']=allowed
     handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)]);module.attn._diagnostic_attention_observer=observe
    on=inference.score_rhs_candidate_suffix(**options,candidate_text=rhs+'\n')
    for handle in handles:handle.remove()
    for name,module in model.named_modules():
     if name.endswith('gated_cross_attn_layer'):module.attn._diagnostic_attention_observer=None
    off=inference.score_rhs_candidate_suffix(**options,candidate_text=rhs+'\n')
    assert all(abs(on[k]-off[k])<=1e-6 for k in ('mean_logprob','sum_logprob'))
    layer_rows=[]
    for layer,data in captures.items():
     before,after=data['before'],data['after'];index=len(vectors);vectors.append({'before':before,'after':after,'attention':data['attention'],'allowed':data['allowed']});avg=data['attention'].mean(0);eligible=torch.where(data['allowed'])[0];top=eligible[torch.argsort(avg[eligible],descending=True)[:5]];mass={};top_keys=[]
     for i in eligible.tolist():
      category='action' if i<64 else pack['evidence_metadata'][i-64]['evidence_type'];mass[category]=mass.get(category,0.)+float(avg[i])
     for i in top.tolist():
      meta={'key':'L'+str(i+1),'evidence_type':'action'} if i<64 else pack['evidence_metadata'][i-64]
      top_keys.append({'slot':i,'weight':float(avg[i]),'metadata':meta})
     layer_rows.append({'layer':layer,'tensor_index':index,'shape':list(before.shape),'relative_update':float((after-before).norm()/before.norm()),'cosine':float(torch.nn.functional.cosine_similarity(before,after,dim=0)),'before_norm':float(before.norm()),'after_norm':float(after.norm()),'effective_gate':data['effective_gate'],'eligible_keys':int(data['allowed'].sum()),'attention_entropy':float(-(avg[eligible]*avg[eligible].clamp_min(1e-30).log()).sum()),'category_mass':mass,'top_keys':top_keys})
    records.append({'mode':mode,'field':lhs,'action':prefix['action'],'prior_output_sha256':prior_hash,'full_prefix_sha256':digest(prefix['ids']),'token_index':position,'token_id':prefix['ids'][-1],'token_text':tok.decode([prefix['ids'][-1]]),'candidate_ids':{s['rhs']:tok(s['rhs']+'\n',add_special_tokens=False)['input_ids'] for s in scores},'candidate_domain_sha256':digest(choices),'semantically_forced':len(choices)==1,'scores':scores,'margin':scores[0]['score']-scores[1]['score'] if len(scores)>1 else None,'layers':layer_rows,'tracing_score_parity':True})
    prior[lhs.upper()]=value
  finally:model.clear_structural_memory()
 with torch.no_grad():
  for p,v in zip(gates,saved):p.copy_(v)
 out=Path(os.environ['SALT_FIXED_OUTPUT']);out.parent.mkdir(parents=True,exist_ok=True);name=digest(case.platform_row.get('context_id',case.kernel_name))[:12]+'.pt';tensorpath=out.parent/name
 if tensorpath.exists():raise FileExistsError(tensorpath)
 torch.save(vectors,tensorpath)
 record={'context_id':case.platform_row['context_id'],'kernel':case.kernel_name,'records':records,'tensors':str(tensorpath),'tensor_sha256':hashlib.sha256(tensorpath.read_bytes()).hexdigest(),'prefix_helper_commit':HELPER_COMMIT,'prefix_helper_sha256':hashlib.sha256(source.encode()).hexdigest(),'diagnostic_runtime_seconds':time.monotonic()-started,'scope':'controlled preceding Stage1 R1 assignments; not official predictions'}
 with out.open('a') as f:f.write(json.dumps(record)+'\n')
 return {'context_id':record['context_id'],'diagnostic':'controlled-prefix replay','records':len(records)}
if __name__=='__main__':
 from .run import preflight
 preflight(os.environ['CUDA_VISIBLE_DEVICES'])
 if Path(os.environ['SALT_FIXED_OUTPUT']).exists():raise FileExistsError('Use fresh output')
 inference.predict_case=checked;inference.main()
