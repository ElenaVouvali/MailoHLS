"""Real decoder scoring and fixed-prefix interventions; requires trained weights.

This entry delegates loading, prompting, legal domains, free-running decoding and
full causal RHS+newline scores to the maintained inference implementation.
"""
from contextlib import contextmanager
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
import time

import torch
from experiments.gemv_structural_explainability.audit import (
    ROOT,GEMV,GNN_SHA,STAGE1,STAGE1_SHA,sha,write_json)
from experiments.gemv_structural_explainability.memory_alignment import to_device,SCHEMA
from experiments.gemv_structural_explainability.interventions import plan,apply,deltas


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


@contextmanager
def gates_off(model,enabled):
    modules=[m for m in model.modules() if hasattr(m,'attn_gate')]
    saved=[m.attn_gate.detach().clone() for m in modules]
    try:
        if enabled:
            with torch.no_grad():
                for m in modules:m.attn_gate.zero_()
        yield
    finally:
        with torch.no_grad():
            for m,value in zip(modules,saved):m.attn_gate.copy_(value)


def decision_prefixes(inference,tok,kwargs,output):
    """Reconstruct the exact untruncated prefixes chosen by native decoding."""
    current=None;ids=list(kwargs['prompt_ids']);history={};records=[]
    registry=kwargs['directive_domain_registry'];kernel=kwargs['kernel_name']
    debug=output['site_debug']
    for index,(label,lhs) in enumerate(inference.extract_ordered_lhs_plan(kwargs['source_text'])):
        if label!=current:
            ids+=tok(f'{inference.target_placeholder_token(label)}\n',add_special_tokens=False)['input_ids']
            current=label
        ids+=tok(f'{lhs} = ',add_special_tokens=False)['input_ids']
        original=inference.get_rhs_candidates_for_lhs(kernel,lhs,registry)
        candidates=inference.mailohls_contract.filter_semantic_candidates(lhs,original,history,
            registry[inference.normalize_kname(kernel)])
        chosen=debug[index]['chosen_rhs']
        if chosen not in candidates:raise ValueError('Native decoder produced an illegal value')
        records.append(dict(action=label,lhs=lhs,prefix_ids=list(ids),
            prefix_text=tok.decode(ids,skip_special_tokens=False),
            preceding_directives=dict(history),candidates=list(candidates),chosen=chosen,
            routing_start_idx=int(kwargs['routing_start_idx'].item())))
        history[lhs.strip().upper()]=chosen
        ids+=tok(chosen+'\n',add_special_tokens=False)['input_ids']
    if len(records)!=13:raise ValueError('Expected all thirteen GEMV directive fields')
    inference.mailohls_contract.validate_directive_assignments(history)
    return records


class AttentionRecorder:
    """Preserves layer/head/token-level values, including exact masked zeros."""
    def __init__(self,model,key_ids):
        self.rows=[];self.context=None;self.handles=[];self.observers=[]
        self.key_ids=key_ids
        blocks=[(n,m) for n,m in model.named_modules() if n.endswith('gated_cross_attn_layer')]
        if len(blocks)!=4:raise ValueError('Expected four structural fusion layers')
        for layer,(name,block) in zip((8,16,24,32),blocks):
            state={}
            def observer(attention,mask,layer=layer,state=state,block=block):
                if self.context is None:return
                context,start,length=self.context
                weights=attention[0,:,start:start+length].detach().float().cpu()
                allowed=mask[0,0,start:start+length].detach().cpu()
                state['weights'],state['allowed']=weights,allowed
            block.attn._diagnostic_attention_observer=observer
            self.observers.append(block.attn)
            def before(module,inputs,state=state):
                if self.context is None:return
                _,start,length=self.context
                state['hidden']=inputs[0][0,start:start+length].detach().float().norm(dim=-1).cpu()
            def after(module,inputs,output,layer=layer,state=state,block=block):
                if self.context is None:return
                context,start,length=self.context
                gate=float((block.attn_gate.tanh()*block.attn_gate_scale).item())
                residual=output[0,start:start+length].detach().float().norm(dim=-1).cpu()*abs(gate)
                for h,head in enumerate(state['weights']):
                    for t,weights in enumerate(head):
                        count=int(state['allowed'][t].sum())
                        entropy=float(-(weights*weights.clamp_min(1e-30).log()).sum())
                        if weights[~state['allowed'][t]].count_nonzero():raise ValueError('Invalid/padding attention is nonzero')
                        for key,weight in enumerate(weights.tolist()):
                            self.rows.append(dict(**context,layer=layer,head=h,causal_query_position=start+t,
                                key=key,memory_id=self.key_ids[key],eligible=bool(state['allowed'][t,key]),
                                attention=weight,eligible_key_count=count,entropy=entropy,
                                normalized_entropy=entropy/__import__('math').log(count) if count>1 else 0.,
                                gate=gate,residual_hidden_ratio=float(residual[t]/state['hidden'][t].clamp_min(1e-20))))
            self.handles.extend([block.attn.register_forward_pre_hook(before),block.attn.register_forward_hook(after)])

    def close(self):
        for h in self.handles:h.remove()
        for module in self.observers:del module._diagnostic_attention_observer


@torch.no_grad()
def score_field(inference,model,tok,record,memory,*,recorder=None,model_name='',intervention='original'):
    device=inference.get_first_real_device(model)
    base=torch.tensor([record['prefix_ids']],device=device,dtype=torch.long)
    model.condition_structural_memory(memory['memory'],memory['mask'],action_relation_mask=memory['relation'])
    values={};rows=[]
    try:
        for rhs in record['candidates']:
            continuation=rhs+'\n'
            if recorder is not None:
                recorder.context=(dict(model=model_name,action=record['action'],lhs=record['lhs'],
                    rhs=rhs,intervention=intervention),len(record['prefix_ids'])-1,
                    len(tok(continuation,add_special_tokens=False)['input_ids']))
            stats=inference.score_rhs_candidate_suffix(model=model,tok=tok,base_input_ids=base,
                base_attention_mask=torch.ones_like(base),candidate_text=continuation,
                routing_start_idx=torch.tensor([record['routing_start_idx']],device=device),
                use_structural_memory=True)
            values[rhs]=stats['mean_logprob']
            rows.append(dict(model=model_name,action=record['action'],lhs=record['lhs'],
                intervention=intervention,rhs=rhs,continuation=repr(continuation),**stats,
                prefix_text=record['prefix_text'],prefix_ids=json.dumps(record['prefix_ids']),
                preceding_directives=json.dumps(record['preceding_directives'],sort_keys=True)))
    finally:
        if recorder is not None:recorder.context=None
        model.clear_structural_memory()
    ranks=deltas(values,values)
    ordered=sorted(rows,key=lambda r:(r['mean_logprob'],r['sum_logprob']),reverse=True)
    rank_by_rhs={r['rhs']:i+1 for i,r in enumerate(ordered)}
    ranks['winner']=ordered[0]['rhs']
    for row in rows:
        row.update(rank=rank_by_rhs[row['rhs']],selected=row['rhs']==ranks['winner'],
                   top_two_margin=ranks['reference_margin'])
    return values,rows


def gradient_audit(model,tok,module,args,contract,production):
    """One authenticated training-batch replay; no optimizer update or GEMV label."""
    from experiments.final_architecture_v3.supervision import corrected
    path=args.data_dir/'stage2_pilot.jsonl'
    row=next(json.loads(line) for line in path.read_text().splitlines()
             if json.loads(line)['_quality_split']=='train')
    flags=[(p,p.requires_grad) for p in model.parameters()]
    device=next(model.parameters()).device
    try:
        model.requires_grad_(False)
        for name,p in model.named_parameters():
            if 'gated_cross_attn_layer.attn.' in name or name.endswith('gated_cross_attn_layer.attn_gate'):
                p.requires_grad_(True)
        registry=production.load_directive_domain_registry(str(args.checkpoint/'directive_domain_registry.json'))
        with corrected(production):
            dataset=production.SFTDataset([row],tok,contract['max_length'],
                directive_domain_registry=registry,supervise_eos=False,
                input_only_special_ids=contract['special_token_ids'])
        sample=dataset.samples[0]
        raw=torch.load(args.raw_dir/(row['kernel_name']+'.pt'),map_location='cpu',weights_only=False)
        with torch.enable_grad():
            memory=module(to_device(raw,device))
            model.condition_structural_memory(memory['memory'],memory['mask'],action_relation_mask=memory['relation'])
            logits=model(input_ids=sample['input_ids'][None].to(device),
                attention_mask=sample['attention_mask'][None].to(device),
                routing_start_idx=sample['routing_start_idx'][None].to(device),
                xattn_apply_mask=sample['xattn_apply_mask'][None].to(device)).logits[:,:-1].float()
            labels=sample['labels'][None,1:].to(device)
            weight=sample['token_weights'][None,1:].to(device)*labels.ne(-100)
            loss=(torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),
                    ignore_index=-100,reduction='none').view_as(weight)*weight).sum()/weight.sum().clamp_min(1e-20)
            loss.backward()
        gradients={name:dict(l2=float(p.grad.float().norm()),rms=float(p.grad.float().square().mean().sqrt()),
                            finite=bool(torch.isfinite(p.grad).all()))
                   for name,p in module.named_parameters() if p.grad is not None}
        if not gradients or not all(r['finite'] for r in gradients.values()):raise ValueError('Gradient audit failed')
        write_json(args.out/'gradient_audit.json',dict(loss=float(loss.detach()),kernel=row['kernel_name'],
            corpus_sha256=sha(path),gradients=gradients,optimizer_updates=0,
            interpretation='Post-training action-normalized training-batch replay; not a per-update convergence trace'))
    finally:
        model.clear_structural_memory();model.zero_grad(set_to_none=True)
        for p,flag in flags:p.requires_grad_(flag)


def evaluate(args):
    if not args.checkpoint:raise ValueError('--checkpoint requires a completed trained arm directory')
    completion=args.checkpoint/'alignment_completion.json'
    if not completion.is_file():
        raise RuntimeError('No completed alignment receipt; refusing random/untrained retrieval results')
    completed=json.loads(completion.read_text())
    if completed.get('updates',0)<=0:raise ValueError('No authenticated positive optimizer-update count')
    contract=json.loads((args.checkpoint/'training_contract.json').read_text())
    alignment=contract['structural_alignment']
    if alignment['schema']!=SCHEMA or alignment['mode']!=args.mode:
        raise ValueError('Checkpoint alignment contract mismatch')
    if alignment['gnn_sha256']!=GNN_SHA or completed['checkpoint_sha256']!=sha(args.checkpoint/'structural_xattn.pt'):
        raise ValueError('Checkpoint identity mismatch')
    if alignment['raw_manifest_sha256']!=sha(args.raw_dir/'manifest.json'):
        raise ValueError('Raw memory bank differs from trained alignment')
    from experiments.final_architecture_20261109.run import preflight
    receipt=preflight(args.gpu_uuid,minimum=14000 if args.gpu_uuid=='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35' else 24000)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    import numpy._core.multiarray
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
    from LLM_branch.train import train_SFT_xattn_new as production
    from LLM_branch.common import directive_domains as domains
    from experiments.gemv_structural_explainability.integration import aligned_runtime
    native_quant_config=inference.BitsAndBytesConfig
    def fp16_quantization(**kwargs):
        kwargs['bnb_4bit_compute_dtype']=torch.float16
        return native_quant_config(**kwargs)
    inference.BitsAndBytesConfig=fp16_quantization
    args.out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c').read_text()
    registry=domains.build_source_domain_registry([dict(kernel_name='gemv',input=source)],
        ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta')
    write_json(args.out/'gemv_domains.json',registry)
    raw=torch.load(args.raw_dir/'gemv.pt',map_location='cpu',weights_only=False)
    pinned=torch.load(Path(__file__).parent/'results/action_memory.pt',map_location='cpu',weights_only=False)
    if not torch.equal(raw['action'][0],pinned['action_memory']):raise ValueError('Visualization/experiment embedding drift')
    candidate_rows=[];intervention_rows=[];summary_rows=[];end_to_end=[];attention_rows=[]
    old_decode=inference.constrained_decode_rhs_by_candidate_scoring
    old_score=inference.score_rhs_candidate_suffix
    old_move=inference.move_structural_modules_to_model_device
    started=time.monotonic()
    timings={}
    score_cache={};cache_stats=dict(hits=0,misses=0);scoring_state={'recorder':None}
    def cached_score(**kwargs):
        # Exact memoization is legal because native scoring recomputes the full
        # prefix and applies fusion ONLY at current causal RHS query positions.
        # Earlier prefix positions receive zero structural residual. Cache keys
        # include every visible memory value/mask, gate, prefix and continuation.
        # No approximate single-token scoring or reuse across model weights.
        model,tok=kwargs['model'],kwargs['tok']
        placeholder_ids={v:i for i,v in enumerate(tok.convert_tokens_to_ids(
            inference.TARGET_PLACEHOLDER_TOKENS))}
        ids=kwargs['base_input_ids'][0].tolist()
        slot=next(placeholder_ids[t] for t in reversed(ids) if t in placeholder_ids)
        wrapper=next(model._structural_wrappers())
        relation=wrapper.action_relation_mask[0,slot] & wrapper.structural_memory_mask[0]
        values=wrapper.structural_memory[0,relation].detach().float().cpu().contiguous()
        gates=tuple(float(m.attn_gate.detach()) for m in model.modules() if hasattr(m,'attn_gate'))
        key=(tuple(ids),kwargs['candidate_text'],int(kwargs['routing_start_idx'].item()),
             gates,tuple(relation.nonzero().flatten().tolist()),values.numpy().tobytes())
        recording=scoring_state['recorder'] is not None and scoring_state['recorder'].context is not None
        if key in score_cache and not recording:
            cache_stats['hits']+=1;return dict(score_cache[key])
        value=old_score(**kwargs)
        if key in score_cache and recording:
            delta=abs(value['mean_logprob']-score_cache[key]['mean_logprob'])
            if delta>1e-6:raise ValueError('Exact score-cache parity failed')
            cache_stats['verified_replays']=cache_stats.get('verified_replays',0)+1
            cache_stats['max_replay_mean_logp_delta']=max(cache_stats.get('max_replay_mean_logp_delta',0.),delta)
        score_cache[key]=dict(value);cache_stats['misses']+=1
        return value
    with aligned_runtime(production,args.raw_dir,args.mode) as holder:
        inference.move_structural_modules_to_model_device=production.move_structural_modules_to_model_device
        def decode(**kwargs):
            native_started=time.monotonic()
            output=old_decode(**kwargs)
            timings['native_generation_seconds']=time.monotonic()-native_started
            model,tok=kwargs['model'],kwargs['tok']
            source_actions=domains.load_source_actions('gemv',ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta')
            domains.validate_assignments({k.upper():v for k,v in inference.build_rhs_map_from_target(output['prediction']).items()},source_actions)
            module=holder['alignment']
            device=inference.get_first_real_device(model)
            p=to_device(raw,device)
            with torch.no_grad():memory=module(p)
            own=decision_prefixes(inference,tok,kwargs,output)
            write_json(args.out/'natural_prefixes.json',own)
            reference=json.loads(args.reference_prefixes.read_text()) if args.reference_prefixes else own
            key_ids=['action:L'+str(k+1) for k in range(64)]+[r['id'] for r in raw['records']]
            if args.mode=='gmt':key_ids += [f'summary:L{k+1}:{s}' for k in range(64) for s in range(2)]
            recorder=AttentionRecorder(model,key_ids)
            scoring_state['recorder']=recorder
            baseline={}
            try:
                for record in reference:
                    values,rows=score_field(inference,model,tok,record,memory,recorder=recorder,model_name=args.mode)
                    baseline[record['lhs']]=values;candidate_rows.extend(rows)
                if memory['summary_attention'] is not None:
                    weights=memory['summary_attention'].detach().float().cpu()
                    for k in raw['active'][0].nonzero().flatten().tolist():
                        for h in range(4):
                            for s in range(2):
                                for i,weight in enumerate(weights[0,k,h,s].tolist()):
                                    summary_rows.append(dict(action='L'+str(k+1),head=h,summary=s,
                                        evidence_id=raw['records'][i]['id'],evidence_type=raw['records'][i]['type'],
                                        eligible=bool(raw['routes'][0,k,i]),attention=weight))
                if args.gradient_audit and args.mode=='gmt':
                    gradient_audit(model,tok,module,args,contract,production)
                if args.stage=='intervene':
                    specs=plan(raw)
                    write_json(args.out/'intervention_plan.json',[asdict(s) for s in specs])
                    for spec in specs:
                        with torch.no_grad():changed=apply(module,p,spec)
                        with gates_off(model,spec.gate_off):
                            for record in reference:
                                if record['action']!='L'+str(spec.action+1):continue
                                values,rows=score_field(inference,model,tok,record,changed,model_name=args.mode,intervention=spec.name)
                                candidate_rows.extend(rows)
                                delta=deltas(baseline[record['lhs']],values)
                                intervention_rows.append(dict(model=args.mode,action=record['action'],lhs=record['lhs'],
                                    intervention=spec.name,control_for=spec.control_for,removed=json.dumps(spec.removed),
                                    replacements=json.dumps(spec.replacements),summary_policy=spec.summary,
                                    delta_margin=delta['delta_margin'],winner_changed=delta['winner_changed'],
                                    rank_changed=delta['rank_changed'],reference_winner=delta['reference_winner'],
                                    winner=delta['winner'],delta_logp=json.dumps(delta['delta_logp']),
                                    ranks=json.dumps(delta['ranks']),interpretation='downstream memory intervention; GNN unchanged'))
                            # Separate native free-running generation, never reused
                            # as a fixed-prefix effect estimate.
                            free=old_decode(**dict(kwargs,structural_memory=changed['memory'],
                                structural_memory_mask=changed['mask'],structural_relation_mask=changed['relation']))
                            domains.validate_assignments({k.upper():v for k,v in inference.build_rhs_map_from_target(free['prediction']).items()},source_actions)
                            end_to_end.append(dict(action='L'+str(spec.action+1),intervention=spec.name,
                                changed=free['prediction']!=output['prediction'],source_valid=True,prediction=free['prediction']))
                attention_rows.extend(recorder.rows)
            finally:
                recorder.close();scoring_state['recorder']=None
            return output
        inference.constrained_decode_rhs_by_candidate_scoring=decode
        inference.score_rhs_candidate_suffix=cached_score
        sys.argv=[sys.argv[0],'--stage','stage2','--adapter_dir',str(args.checkpoint),
            '--memory_dir',str(args.raw_dir/'memory'),'--kernel_name','gemv',
            '--code_file',str(ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta/gemv.c'),
            '--application_dataset_dir',str(ROOT/'evaluation_set/gemv/code_to_memory_outputs/meta'),
            '--directive_domain_registry_json',str(args.checkpoint/'directive_domain_registry.json'),
            '--objective','latency','--clock_period','10','--available_clock_periods','10','--avail_bram','624','--avail_dsp','1728',
            '--avail_ff','460800','--avail_lut','230400','--debug_topk','10000',
            '--score_reduction','mean','--num_samples','1','--output_json',str(args.out/'native_generation.json')]
        try:inference.main()
        finally:
            inference.constrained_decode_rhs_by_candidate_scoring=old_decode
            inference.score_rhs_candidate_suffix=old_score
            inference.move_structural_modules_to_model_device=old_move
    write_csv(args.out/'candidate_scores.csv',candidate_rows)
    write_csv(args.out/'decoder_attention.csv',attention_rows)
    write_csv(args.out/'memory_attention.csv',summary_rows)
    write_csv(args.out/'intervention_results.csv',intervention_rows)
    if end_to_end:write_json(args.out/'end_to_end_interventions.json',end_to_end)
    write_json(args.out/'evaluation_receipt.json',dict(mode=args.mode,preflight=receipt,
        checkpoint_sha256=completed['checkpoint_sha256'],seconds=time.monotonic()-started,
        **timings,
        exact_score_memoization=cache_stats,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        prefix_context='shared supplied prefixes' if args.reference_prefixes else 'own natural prefixes',
        physical_qor_improvement='NOT EVALUATED'))
