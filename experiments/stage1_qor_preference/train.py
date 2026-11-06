"""One bounded QLoRA pilot, initialized from exact V4; one shared quantized backbone."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import math
import os
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from peft import PeftModel, prepare_model_for_kbit_training

from LLM_branch.train import train_SFT_xattn_new as production
from LLM_branch.common import frozen_stage1
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from experiments.stage1_quality_sft.gpu_preflight import check
from experiments.stage1_quality_sft.support import load_registry
from .common import *
from .objective import selected_logps, pair_loss


def encode(pairs, tok, contract):
    registry = load_registry(V4/'directive_domain_registry.json')
    materialized = [dict(p['row'],target=p[side+'_target']) for p in pairs for side in ['preferred','rejected']]
    ds = production.SFTDataset(materialized, tok, max_length=contract['max_length'],
        directive_domain_registry=registry, supervision_policy='decision', supervise_eos=False,
        input_only_special_ids=contract['special_token_ids'])
    result=[]
    for i,p in enumerate(pairs):
        a,b=ds.samples[2*i:2*i+2]
        ra,rb=int(a['routing_start_idx']),int(b['routing_start_idx'])
        assert ra==rb and torch.equal(a['input_ids'][:ra],b['input_ids'][:rb]), 'Chosen/rejected prompt mismatch'
        header,code,suffix,_=production.build_prompt_sections(p['row']['input'],'PARETO_LATENCY',p['row'])
        exact=[t for text in (header,code,suffix) for t in tok(text,add_special_tokens=False)['input_ids']]
        assert a['input_ids'][:ra].tolist()==exact, 'Truncated source prompt is forbidden'
        for sample in (a,b):
            assert not sample['contrastive_sites']
            assert abs(float(sample['token_weights'].sum())-1.)<1e-5
        hashes={side:digest({k:sample[k].tolist() for k in ['input_ids','labels','token_weights']})
                for side,sample in [('preferred',a),('rejected',b)]}
        result.append(dict(pair=p,preferred=a,rejected=b,encoding_hashes=hashes))
    return result


def score(model, sample):
    device=next(model.parameters()).device
    ids=sample['input_ids'].to(device);labels=sample['labels'].to(device);weights=sample['token_weights'].to(device)
    with torch.autocast('cuda',dtype=torch.float16):
        output=model(input_ids=ids[None,:],attention_mask=torch.ones_like(ids)[None,:])
    return selected_logps(output.logits,labels,weights)


@torch.no_grad()
def evaluate(model, encoded, out_path=None):
    model.eval(); records=[]
    for i,item in enumerate(encoded):
        a,sa,na,ca=score(model,item['preferred']);b,sb,nb,cb=score(model,item['rejected'])
        p=item['pair'];r=p['request']
        records.append(dict(pair_id=p['pair_id'],family=r['family'],request_id=r['request_id'],
            device=r['physical_context'][3],clock=r['physical_context'][4],budget=r['budget'][next(iter(r['budget']))],
            preferred_logp=float(a),rejected_logp=float(b),margin=float(a-b),correct=bool(a>b),
            preferred_sft_nll=float(sa),rejected_sft_nll=float(sb),preferred_tokens=na,rejected_tokens=nb,
            preferred_token_accuracy=float(ca)/na,encoding_hashes=item['encoding_hashes']))
        if (i+1)%20==0:print(f'[SCORE] {i+1}/{len(encoded)}',flush=True)
    if out_path:write(out_path,records)
    return records


def summary(records):
    groups={}
    for key in ('family','device','clock','budget'):
        groups[key]={str(value):dict(pairs=len(sub),accuracy=float(np.mean([r['correct'] for r in sub])),
              mean_margin=float(np.mean([r['margin'] for r in sub])),sft_nll=float(np.mean([r['preferred_sft_nll'] for r in sub])))
            for value in sorted({r[key] for r in records},key=str)
            for sub in [[r for r in records if r[key]==value]]}
    return dict(pairs=len(records),ranking_accuracy=float(np.mean([r['correct'] for r in records])),
        mean_margin=float(np.mean([r['margin'] for r in records])),
        preferred_sft_nll=float(np.mean([r['preferred_sft_nll'] for r in records])),
        preferred_token_accuracy=float(np.mean([r['preferred_token_accuracy'] for r in records])),by=groups)


def train_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.eval()


def save(model,tok,parent,path,manifest,cache_sha,step):
    path.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(path,safe_serialization=True,save_embedding_layers=False)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','directive_domain_registry.json']:
        shutil.copyfile(V4/name,path/name)
    c=copy.deepcopy(parent)
    c.update(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        stage1_trainable_parameter_contract=production.assert_stage1_trainable_contract(model))
    c['stage1_supervision'].update(initialization='measured_preference_refinement_from_v4_multi_reference',
        initial_adapter_lineage=dict(path=str(V4),weights_sha256=ADAPTER_SHA,contract_sha256=CONTRACT_SHA))
    c['qor_preference']=dict(schema='stage1-qor-dpo-v1',beta=.1,lambda_dpo=.2,
        parent_adapter_sha256=ADAPTER_SHA,parent_contract_sha256=CONTRACT_SHA,
        pair_manifest_sha256=sha(OUT/'manifest.json'),reference_cache_sha256=cache_sha,
        actual_preference_pair_file_sha256=manifest['pair_files'],
        raw_family_split_sha256=manifest['raw_family_split_sha256'],
        compatibility_data_identity='generic dataset/split fields retain inherited SFT interface; actual refinement data are the sealed pair files',
        objective='standard reference DPO on summed nonforced directive VALUE token log-probabilities',
        sft_anchor='existing equal-action/equal-nonforced-field/value-token normalized SFT on preferred completion',
        masks='native decision-policy labels; excludes prompt, anchors, fixed LHS, newline, EOS, semantic singletons',
        reference='exact starting V4 adapter including trained special-token deltas; cached before any update',
        status='experimental; not a production promotion or Stage-2 qualification',successful_updates=step)
    c['runtime']=dict(hostname=__import__('socket').gethostname(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_device_name=torch.cuda.get_device_name(0),bnb_compute_dtype='torch.float16',
        argv=__import__('sys').argv,working_directory=str(ROOT))
    write(path/'training_contract.json',c)
    return dict(path=str(path),adapter_sha256=sha(path/'adapter_model.safetensors'),contract_sha256=sha(path/'training_contract.json'))


def finalize_contracts(model_out=MODEL_OUT):
    """Keep immutable SFT interface IDs separate from continued-training data IDs.

    The first live run used generic dataset/split keys for preference JSON hashes.
    Preserve that draft metadata, retain all actual pair hashes explicitly, and
    verify the existing production Stage-2 compatibility checker without changing
    any checkpoint tensor or weakening the checker.
    """
    parent=read(V4/'training_contract.json');manifest=read(OUT/'manifest.json');receipts={}
    for name in ['pilot','run']:
        path=model_out/name;contract=path/'training_contract.json';c=read(contract)
        before=sha(contract);weights=sha(path/'adapter_model.safetensors')
        if any(c[k]!=parent[k] for k in ['dataset_sha256','split_sha256']):
            backup=path/'training_contract.preference_data_keys_draft.json'
            if not backup.exists():shutil.copyfile(contract,backup)
        for k in ['dataset_sha256','split_sha256']:c[k]=parent[k]
        c['qor_preference'].update(actual_preference_pair_file_sha256=manifest['pair_files'],
            raw_family_split_sha256=manifest['raw_family_split_sha256'],
            compatibility_data_identity='generic dataset/split fields retain inherited SFT interface; actual refinement data are the sealed pair files')
        write(contract,c)
        production.require_compatible_stage1_contract(str(path),parent)
        assert sha(path/'adapter_model.safetensors')==weights
        receipts[name]=dict(path=str(path),adapter_sha256=weights,contract_sha256=sha(contract),
            draft_contract_sha256=before,existing_stage2_checker_passed=True,weights_unchanged=True)
    r=read(OUT/'training_result.json');r['checkpoint']=receipts['run'];r['stage2_contract_compatible']=True
    write(OUT/'training_result.json',r)
    write(OUT/'contract_interface_audit.json',dict(checkpoints=receipts,
        policy='Preserve inherited interface identities; record actual preference data separately. Draft metadata archived; no weight modification.'))


@torch.no_grad()
def roundtrip(model,path,sample):
    model.eval();before=score(model,sample)[0].float()
    model.load_adapter(str(path),adapter_name='roundtrip',is_trainable=False)
    model.set_adapter('roundtrip');after=score(model,sample)[0].float()
    torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)
    model.set_adapter('default');model.delete_adapter('roundtrip')
    for n,p in model.named_parameters():p.requires_grad_('lora_' in n or 'trainable_tokens_delta' in n)
    production.assert_stage1_trainable_contract(model)
    return dict(passed=True,before=float(before),after=float(after),same_backbone=True)


def run(model_out=MODEL_OUT):
    if model_out.exists() and any(model_out.iterdir()):raise FileExistsError(model_out)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    gpu=check(GPU);assert gpu['name']=='Tesla V100S-PCIE-32GB'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    parent,tok=authenticate();manifest=read(OUT/'manifest.json')
    assert manifest['training_justified_for_bounded_pilot']
    for split,h in manifest['pair_files'].items():assert sha(OUT/f'{split}_pairs.jsonl')==h
    random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    train=encode(rows(OUT/'train_pairs.jsonl'),tok,parent);val=encode(rows(OUT/'validation_pairs.jsonl'),tok,parent)
    model_out.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=123,beta=.1,lambda_dpo=.2,learning_rate=1e-5,parent_learning_rate=5e-5,
        grad_accum=4,gradient_clip=1.,gradient_checkpointing=True,pilot_successful_updates=4,
        final_successful_updates=math.ceil(len(train)/4),epochs=1,
        pilot_gate='finite updates; mean train-probe policy-reference margin increases; chosen SFT NLL <= start*1.15+0.1; roundtrip passes',
        checkpoint_selection='final predeclared one-pass endpoint; no GEMV or validation-based tuning',
        dropout='disabled for deterministic policy/reference scoring and preference training; saved LoRA config unchanged',
        gpu=gpu,pair_manifest_sha256=sha(OUT/'manifest.json'))
    write(OUT/'training_protocol.json',protocol)
    base=inference.load_base_model(parent['model'],revision=parent['model_revision'])
    base.resize_token_embeddings(len(tok))
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=inference.load_peft_adapter_strict(base,str(V4))
    for name,parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name or 'trainable_tokens_delta' in name)
    assert not getattr(model,'initialized_structural_xattn',False)
    initial=frozen_stage1.frozen_stage1_hashes(model)
    trainables=production.assert_stage1_trainable_contract(model)
    assert trainables['parameter_names_sha256']==parent['stage1_trainable_parameter_contract']['parameter_names_sha256']
    print('[REFERENCE] Caching exact frozen starting checkpoint before optimizer creation',flush=True)
    before_train=evaluate(model,train,OUT/'reference_train.json');before_val=evaluate(model,val,OUT/'reference_validation.json')
    assert frozen_stage1.frozen_stage1_hashes(model)==initial
    cache=dict(schema='stage1-qor-reference-cache-v1',adapter_sha256=ADAPTER_SHA,contract_sha256=CONTRACT_SHA,
        tokenizer_sha256=sha(V4/'tokenizer.json'),registry_sha256=parent['directive_domain_registry_sha256'],
        pair_manifest_sha256=sha(OUT/'manifest.json'),trainable_tensor_hashes=initial,
        dtype='NF4 double quantization; FP16 autocast; kbit preparation and value-token masks identical for policy/reference',
        mask_policy=sha(ROOT/'LLM_branch/train/stage1_decision_loss.py'),
        train=before_train,validation=before_val)
    write(OUT/'reference_cache.json',cache);cache_sha=sha(OUT/'reference_cache.json')
    write(OUT/'offline_baseline.json',dict(train=summary(before_train),validation=summary(before_val)))
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and marker in n],lr=1e-5,weight_decay=wd)
            for marker,wd in [('lora_',.01),('trainable_tokens_delta',0.)]]
    assert all(g['params'] for g in groups)
    optimizer=torch.optim.AdamW(groups)
    total=protocol['final_successful_updates']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min((n+1)/2,1.)*(.1+.9*(1+math.cos(math.pi*min(n,total)/total))/2))
    scaler=torch.amp.GradScaler('cuda',init_scale=64.)
    logs=[];step=0;probe=train[:min(16,len(train))];start_probe=before_train[:len(probe)]
    for offset in range(0,len(train),4):
        batch=train[offset:offset+4];optimizer.zero_grad(set_to_none=True);train_mode(model)
        parts=[]
        for index,item in enumerate(batch,offset):
            a,sft,na,_=score(model,item['preferred']);b,_,nb,_=score(model,item['rejected'])
            ref=before_train[index]
            loss,dpo=pair_loss(a,b,torch.tensor(ref['preferred_logp'],device=a.device),torch.tensor(ref['rejected_logp'],device=a.device),sft)
            assert torch.isfinite(loss)
            scaler.scale(loss/len(batch)).backward()
            parts.append(dict(loss=float(loss.detach()),dpo=float(dpo.detach()),sft=float(sft.detach()),preferred_tokens=na,rejected_tokens=nb))
        scaler.unscale_(optimizer)
        grad_groups={marker:float(sum(p.grad.detach().float().square().sum() for n,p in model.named_parameters()
                  if marker in n and p.grad is not None).sqrt()) for marker in ['lora_','trainable_tokens_delta']}
        assert all(math.isfinite(v) and v>0 for v in grad_groups.values()),grad_groups
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=old_scale,'AMP update skipped; stop bounded pilot rather than silently count it'
        scheduler.step();step+=1
        logs.append(dict(step=step,grad_norm=float(norm),gradient_groups=grad_groups,amp_scale=scaler.get_scale(),
            lr=scheduler.get_last_lr(),mean_loss=float(np.mean([p['loss'] for p in parts])),parts=parts))
        write(OUT/'training_log.json',logs)
        print(f'[UPDATE] {step}/{total} loss={logs[-1]["mean_loss"]:.6f} grad={float(norm):.4f}',flush=True)
        if step==4:
            after=evaluate(model,probe,OUT/'pilot_probe.json')
            margin_gain=float(np.mean([a['margin']-b['margin'] for a,b in zip(after,start_probe)]))
            nll=float(np.mean([a['preferred_sft_nll'] for a in after]));nll0=float(np.mean([a['preferred_sft_nll'] for a in start_probe]))
            assert margin_gain>0 and nll<=nll0*1.15+.1,('pilot failed',margin_gain,nll,nll0)
            pilot=save(model,tok,parent,model_out/'pilot',manifest,cache_sha,step)
            rt=roundtrip(model,model_out/'pilot',probe[0]['preferred'])
            write(OUT/'pilot_gate.json',dict(passed=True,margin_gain=margin_gain,initial_sft_nll=nll0,current_sft_nll=nll,roundtrip=rt,checkpoint=pilot))
    final=save(model,tok,parent,model_out/'run',manifest,cache_sha,step)
    final_rt=roundtrip(model,model_out/'run',val[0]['preferred'])
    after_train=evaluate(model,train,OUT/'refined_train.json');after_val=evaluate(model,val,OUT/'refined_validation.json')
    assert sha(V4/'adapter_model.safetensors')==ADAPTER_SHA and sha(V4/'training_contract.json')==CONTRACT_SHA
    write(OUT/'training_result.json',dict(status='TRAINING_COMPLETED',successful_updates=step,skipped_updates=0,
        checkpoint=final,roundtrip=final_rt,initial_tensor_hashes=initial,final_tensor_hashes=frozen_stage1.frozen_stage1_hashes(model),
        reference_unchanged_before_training=True,parent_files_unchanged=True,
        baseline=dict(train=summary(before_train),validation=summary(before_val)),
        refined=dict(train=summary(after_train),validation=summary(after_val)),stage2_qualified=False))
    print('[DONE] '+str(model_out/'run'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-out',type=Path,default=MODEL_OUT);a=ap.parse_args();run(a.model_out)
