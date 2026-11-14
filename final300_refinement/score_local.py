"""Causal FiLM+GMT local preference prior for a candidate vs its reference.

A HEURISTIC, not a full-design probability or QoR predictor: on the edited
fields, compare mean RHS token logps in *each completion's own causal prefix*.
All unchanged field values are still teacher-forced to reconstruct true context.
Both candidate and reference share the same compiler memory and target context.
"""
import math

def score_slate(slate,case,packed,actions,model,film,hooks,tok,registry,dtype):
    import torch
    from torch.nn import functional as F
    from LLM_branch.common import mailohls_contract as mc
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    from experiments.gnosis_stage1_ab import train_gnosis_ab as gn
    device=torch.device('cuda:0')
    fields=mc.target_prompt_fields(case.platform_row)
    sections=mc.build_prompt_sections(case.source_text,case.obj_mode,fields)
    base=[t for section in sections for t in tok(section,add_special_tokens=False)['input_ids']]
    base+=mc.selected_clock_response_token_ids(case.platform_row,tok)
    routing_idx=torch.tensor([sum(len(tok(section,add_special_tokens=False)['input_ids']) for section in sections)],device=device,dtype=torch.long)
    if len(base)>7168:raise RuntimeError('Source+clock prefix exceeds training max length')
    ctx=gn.context_features(case.platform_row).to(device)[None,None,:]
    labels=inf.extract_ordered_lhs_plan(case.source_text)
    expected={lhs.strip().upper() for _,lhs in labels}
    kv,mask,relations=(x.to(device) if x is not None else None for x in packed)

    def scored(assignment,edited):
        if set(assignment)!=expected:raise ValueError('Incomplete scorer assignment')
        prefix=list(base);chosen={};current_label=None;scores={}
        model.condition_structural_memory(kv,mask,relations)
        try:
            for label,lhs in labels:
                if label!=current_label:
                    prefix.extend(tok(f'<{label}>\n',add_special_tokens=False)['input_ids']);current_label=label
                prefix.extend(tok(f'{lhs} = ',add_special_tokens=False)['input_ids'])
                key=lhs.strip().upper();rhs=str(assignment[key]).lower()
                options=inf.get_rhs_candidates_for_lhs(case.kernel_name,lhs,registry)
                valid=mc.filter_semantic_candidates(lhs,options,chosen,registry[inf.normalize_kname(case.kernel_name)])
                if rhs not in valid:
                    # TCAD-expanded legal values can be outside this model's native vocabulary.
                    raise ValueError(f'{key}={rhs}: outside checkpoint-300 native candidates')
                rhs_ids=tok(rhs+'\n',add_special_tokens=False)['input_ids']
                if len(prefix)+len(rhs_ids)>7168:raise ValueError('Candidate too long')
                if key in edited and len(valid)>1:
                    nbase=len(prefix);ncand=len(rhs_ids)
                    ids=torch.tensor([prefix+rhs_ids],device=device,dtype=torch.long)
                    xmask=inf.make_candidate_xattn_apply_mask(full_length=nbase+ncand,base_len=nbase,candidate_len=ncand,device=device)
                    kinds=torch.zeros_like(ids,dtype=torch.long)
                    kinds[:,nbase-1:nbase-1+ncand]=gn.kind_id(lhs)
                    with torch.inference_mode(),hooks.bind(ctx,kinds,enabled=True):
                        with torch.autocast('cuda',dtype=dtype):
                            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,routing_start_idx=routing_idx,xattn_apply_mask=xmask).logits
                        selected=logits[:,nbase-1:nbase-1+ncand,:].float()
                        gold=torch.tensor(rhs_ids,dtype=torch.long,device=device)[None]
                        logps=F.log_softmax(selected,-1).gather(-1,gold[...,None]).squeeze(-1)
                        value=float(logps.mean())
                    if not math.isfinite(value):raise FloatingPointError('Nonfinite causal score')
                    scores[key]=value
                chosen[key]=rhs;prefix.extend(rhs_ids)
            mc.validate_directive_assignments(chosen)
            return scores
        finally:model.clear_structural_memory()

    out=[]
    for item in slate['candidates']:
        name=item['signature'];ref=item['reference'];cand=item['assignment']
        edited={k for k in ref if ref[k]!=cand[k]}
        try:
            # Score the SAME edited sites on BOTH complete causal histories.
            a=scored(cand,edited);b=scored(ref,edited)
            common=sorted(set(a)&set(b))
            delta=sum(a[k]-b[k] for k in common)/len(common) if common else None
            out.append({'signature':name,'delta':delta,'scored_fields':common,'unscored_reason':None if delta is not None else 'All changed fields forced'})
        except ValueError as e:
            out.append({'signature':name,'delta':None,'scored_fields':[],'unscored_reason':str(e)})
    return {'schema':'salt-final300-local-conditional-prior-v1','checkpoint_role':'frozen-stage2-300','not_a_qor_prediction':True,'candidates':out}
