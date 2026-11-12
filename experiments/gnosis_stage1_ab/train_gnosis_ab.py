"""Matched Davinci V100 Stage-1 target-conditioning A/B experiment.

A: frozen authenticated SFT parent + fresh residual LoRA.
B: same + action-position, intermediate-layer residual FiLM.
Both receive identical multi-reference GNOSIS SFT and matched-context ranking.

IMPORTANT: experiment-only adapters; no production promotion or physical QoR claim.
FiLM context remains bound through backward to prevent checkpoint recomputation
from using the context of a different candidate.
"""
from __future__ import annotations
import argparse,collections,contextlib,copy,hashlib,json,math,os,random,time,re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from peft import PeftModel,LoraConfig,prepare_model_for_kbit_training
from LLM_branch.common import device_adaptation as da,mailohls_contract as mc
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from LLM_branch.train import train_SFT_xattn_new as prod
from experiments.final_architecture_v3.supervision import corrected
from experiments.final_architecture_v3.preferences_objective import selected_logps
KINDS=('PIPE','UNROLL','ARRAY_T','ARRAY_F','ARRAY_D')
def kind_id(lhs):
    m=re.fullmatch(r'AUTO\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*\}',lhs.upper())
    if not m:raise ValueError(lhs)
    return KINDS.index(m[1])+1

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')

def load_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

FEATURE_KEYS=('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut')
def context_features(row):
    device=prod._norm_device(row['device']);clock=float(prod._clock_of(row))
    devices=('xczu7ev-ffvc1156-2-e','xcu200-fsgd2104-2-e')
    assert device in devices and clock in (3.33,5.,10.)
    caps=[float(mc.DEVICE_RESOURCES[device][k]) for k in mc.RESOURCE_KEYS]
    budget=[float(row[k]) for k in FEATURE_KEYS]
    assert all(0<v<=1. for v in budget)
    return torch.tensor([float(device==devices[0]),float(device==devices[1]),math.log(clock/10.),
        *[math.log1p(v)/16. for v in caps],*budget,
        *[math.log1p(c*b)/16. for c,b in zip(caps,budget)]],dtype=torch.float32)

class ResidualFiLM(nn.Module):
    """Context+directive kind -> bounded affine change in FP32; apply to selected causal positions.
    Unlike head-FiLM, applies in the decoder and affects subsequent computation.
    """
    def __init__(self,hidden,num_layers,layers=None,bottleneck=64):
        super().__init__()
        self.selected=tuple(layers if layers is not None else sorted(set([num_layers//4-1,num_layers//2-1,3*num_layers//4-1])))
        if any(i<0 or i>=num_layers for i in self.selected):raise ValueError('Invalid FiLM layer indices')
        self.kind=nn.Embedding(6,8)
        self.enc=nn.Sequential(nn.Linear(15+8,bottleneck),nn.SiLU())
        self.proj=nn.ModuleDict({str(i):nn.Linear(bottleneck,2*hidden) for i in self.selected})
        self.gates=nn.ParameterDict({str(i):nn.Parameter(torch.tensor(.2)) for i in self.selected})
        # Nonzero, but modest initial residual; unlike zero-gate, generator receives gradients.
        for layer in self.proj.values():
            nn.init.normal_(layer.weight,std=.008);nn.init.zeros_(layer.bias)
        self.hidden=hidden;self.state=None;self.on=True;self.last=[]
    def apply_layer(self,h,idx):
        if not self.on or self.state is None:return h
        ctx,kinds=self.state
        if h.dim()!=3 or h.shape[:2]!=kinds.shape:raise ValueError('FiLM causal positions / sequence mismatch')
        sel=kinds.gt(0)
        if not sel.any():return h
        base=h[sel].float()
        z=ctx.expand(*kinds.shape,-1)[sel]
        ab=self.proj[str(idx)](self.enc(torch.cat((z,self.kind(kinds[sel])),dim=-1)))
        gamma,beta=ab.chunk(2,-1)
        delta=torch.tanh(self.gates[str(idx)])*.1*(gamma.tanh()*F.layer_norm(base,(self.hidden,))+beta.tanh())
        result=h.clone()
        effective=(base+delta).to(h.dtype)
        result[sel]=effective
        if not torch.is_grad_enabled():
            intended=float(delta.norm().detach());surviving=float((effective.float()-base).norm().detach())
            self.last.append({'layer':idx,'intended':intended,'effective':surviving,'gate':float(self.gates[str(idx)].tanh().detach())})
        return result

class InstalledFiLM:
    def __init__(self,model,film):
        self.film=film
        core=model.get_base_model()
        stack=getattr(getattr(core,'model',None),'layers',None)
        if stack is None:raise TypeError('Unsupported backbone: need Llama-style model.layers; do not guess')
        self.handles=[]
        for i in film.selected:
            def hook(module,inp,out,index=i):
                h=out[0] if isinstance(out,tuple) else out
                modified=film.apply_layer(h,index)
                return (modified,*out[1:]) if isinstance(out,tuple) else modified
            self.handles.append(stack[i].register_forward_hook(hook))
    @contextlib.contextmanager
    def bind(self,ctx,kinds,enabled=True):
        assert self.film.state is None,'Overlapping FiLM contexts are unsafe under gradient checkpointing'
        self.film.state=(ctx,kinds);self.film.on=enabled
        try:yield
        finally:self.film.state=None

class Encoded:
    def __init__(self,tokenizer,registry,max_length=6144,capacity=256,special_ids=None):
        self.tok=tokenizer;self.registry=registry;self.max_length=max_length
        self.cache=collections.OrderedDict();self.capacity=capacity;self.special_ids=special_ids
    def get(self,row,target):
        signature=(row['kernel_name'],row['device'],float(prod._clock_of(row)),
                   tuple(float(row[k]) for k in FEATURE_KEYS),target)
        if signature in self.cache:
            self.cache.move_to_end(signature);return self.cache[signature]
        ex=dict(row,target=target)
        with corrected(prod):
            dataset=prod.SFTDataset([ex],self.tok,self.max_length,
                directive_domain_registry=self.registry,supervise_eos=False,
                input_only_special_ids=self.special_ids)
        if len(dataset.samples)!=1:raise ValueError('Truncated or unsupported completion: '+str(signature[:3]))
        sample=dataset.samples[0]
        plan=prod.build_deterministic_rhs_pack(ex['input'],prod.reorder_target_by_source_order(ex['input'],target),
                    self.tok,directive_domain_registry=self.registry,kernel_name=ex['kernel_name'])
        offset=len(sample['input_ids'])-len(plan.input_ids)
        assert offset>=0 and sample['input_ids'][offset:].tolist()==plan.input_ids
        causal=torch.zeros(len(sample['input_ids']),dtype=torch.long)
        for j,key in enumerate(plan.site_keys):
            p=offset+j
            if key is not None and p>0 and float(sample['token_weights'][p])>0:
                causal[p-1]=kind_id(key)
        # Every supervised token must be sourced from a known directive field.
        supervised=(sample['token_weights']>0).nonzero().flatten()
        assert len(supervised)>0 and torch.all(causal[supervised-1]>0)
        d={'ids':sample['input_ids'],'attn':sample['attention_mask'],
           'labels':sample['labels'],'weights':sample['token_weights'],'kinds':causal}
        self.cache[signature]=d
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return d

def validate_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Select the pinned Davinci V100 with CUDA_VISIBLE_DEVICES'
    name=torch.cuda.get_device_name(0)
    major,minor=torch.cuda.get_device_capability(0)
    capacity=torch.cuda.get_device_properties(0).total_memory/2**30
    if 'V100' not in name.upper() or major!=7 or capacity<30:
        raise RuntimeError('Expected Tesla V100 32GB, got %s / compute %s.%s / %.1f GiB'%(name,major,minor,capacity))
    print('VERIFIED CUDA DEVICE',name,torch.cuda.get_device_properties(0).total_memory,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True);p.add_argument('--steps',type=int,default=720)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--max-length',type=int,default=6144)
    p.add_argument('--train-contexts',type=int,default=240)
    p.add_argument('--val-sft',type=int,default=32);p.add_argument('--val-pairs',type=int,default=24)
    p.add_argument('--eval-every',type=int,default=40);p.add_argument('--lr',type=float,default=8e-5)
    p.add_argument('--film-lr',type=float,default=3e-4);p.add_argument('--rank-budget',type=float,default=.20)
    p.add_argument('--rank-clock',type=float,default=.05)
    a=p.parse_args();validate_gpu();torch.manual_seed(a.seed);random.seed(a.seed)
    if a.out.exists() and not a.resume_from:raise FileExistsError('Run directory exists; never silently overwrite: '+str(a.out))
    if a.resume_from and not a.out.exists():raise FileNotFoundError('Resume requires the existing output directory')
    assert a.steps>0 and a.eval_every>0
    bank=json.loads((a.bank/'manifest.json').read_text());assert bank['schema']=='gnosis-target-conditioning-ab-v1'
    for f,h in bank['outputs_sha256'].items():assert sha(a.bank/f)==h,(f,'bank corrupted')
    data={fold:{k:load_jsonl(a.bank/(fold+'_'+k+'.jsonl')) for k in ('sft','pairs')} for fold in ('train','val')}
    assert all(data[z]['sft'] for z in ('train','val'))
    assert all(data[z]['pairs'] for z in ('train','val')),'No held-out target contrasts; cannot select FiLM by target behavior'
    # Use identical, predetermined validation cases for both arms.
    def subset(rows,n):
        groups=collections.defaultdict(list)
        for x in rows:groups[x['row'].get('_family',prod.family_id_from_kernel_name(x['row']['kernel_name']))].append(x)
        rng=random.Random(a.seed+999);chosen=[]
        for family in sorted(groups):
            vv=sorted(groups[family],key=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest())
            chosen.append(vv[0])
        pool=[v for fam in sorted(groups) for v in groups[fam] if v not in chosen]
        rng.shuffle(pool)
        return (chosen+pool)[:max(n,len(chosen))]
    # Pair validation is GROUPED: never select only one of the two contexts.
    pair_groups=collections.defaultdict(list)
    for item in data['val']['pairs']:
        pair_groups[(item['row']['_family'],item['kind'],item['group_id'])].append(item)
    pair_groups={key:items for key,items in pair_groups.items() if len(items)==2}
    family_keys=collections.defaultdict(list)
    for k in pair_groups:family_keys[k[0]].append(k)
    for family in family_keys:
        family_keys[family].sort(key=lambda k:hashlib.sha256(repr(k).encode()).hexdigest())
    selected_groups=[]
    while len(selected_groups)<a.val_pairs and any(family_keys.values()):
        for family in sorted(family_keys):
            if family_keys[family]:selected_groups.append(family_keys[family].pop(0))
            if len(selected_groups)>=a.val_pairs:break
    val={'sft':subset(data['val']['sft'],a.val_sft),
         'pairs':[item for k in selected_groups for item in pair_groups[k]]}
    assert val['pairs'],'No complete validation contrast groups'
    parent_hash=sha(a.parent/'adapter_model.safetensors');contract=json.loads((a.parent/'training_contract.json').read_text())
    assert contract['stage']=='stage1' and contract['objective']=='PARETO_LATENCY' and float(contract['auto_frequency_fraction'])==0
    tokenizer=inf.load_checkpoint_tokenizer(str(a.parent),contract)
    registry_path=a.parent/'directive_domain_registry.json'
    if not registry_path.exists():registry_path=a.parent.parent/'directive_domain_registry.json'
    registry=inf.load_directive_domain_registry(str(registry_path))
    # Verify all examples before allocating expensive optimizer state.
    encode=Encoded(tokenizer,registry,max_length=a.max_length,special_ids=contract['special_token_ids'])
    for kind in ('sft','pairs'):
        for x in val[kind]:
            encode.get(x['row'],x['row']['target'] if kind=='sft' else x['preferred_target'])
            if kind=='pairs':encode.get(x['row'],x['rejected_target'])
    base=inf.load_base_model(contract['model'],revision=contract['model_revision'])
    base.resize_token_embeddings(len(tokenizer))
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=PeftModel.from_pretrained(base,str(a.parent),adapter_name='mailohls_base',is_trainable=False)
    da.isolate_parent_token_serialization(model)
    cfg=json.loads((a.parent/'adapter_config.json').read_text())
    model.add_adapter('device_adapt',LoraConfig(r=cfg['r'],lora_alpha=cfg['lora_alpha'],target_modules=cfg['target_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    da.activate(model,trainable=True);model.enable_input_require_grads()
    inventory=da.trainables_contract(model)
    # Disable frozen adapters' dropout for stable gradient replay.
    for _,m in model.named_modules():
        if m.__class__.__name__=='Dropout':m.p=0.
    device=torch.device('cuda:0')
    nlayer=len(model.get_base_model().model.layers)
    film=ResidualFiLM(model.config.hidden_size,nlayer).to(device)
    hooks=InstalledFiLM(model,film)
    trainable=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW([{'params':trainable,'lr':a.lr},
                                *([{'params':list(film.parameters()),'lr':a.film_lr}] if a.arm=='B' else [])],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=256.,growth_interval=250)
    a.out.mkdir(parents=True,exist_ok=True)
    protocol_path=a.out/'protocol.json'
    if a.resume_from and protocol_path.is_file():
        old=json.loads(protocol_path.read_text())
        if old['parent_sha256']!=parent_hash or old['bank_manifest_sha256']!=sha(a.bank/'manifest.json') or old['arm']!=a.arm or old['steps']!=a.steps:
            raise ValueError('Resume protocol mismatch')
    else:save_json(protocol_path,{'arm':a.arm,'seed':a.seed,'steps':a.steps,'bank_manifest_sha256':sha(a.bank/'manifest.json'),
          'parent_sha256':parent_hash,'parent_contract_sha256':sha(a.parent/'training_contract.json'),
          'parent_registry_sha256':sha(registry_path),'parent_model':contract['model'],'revision':contract['model_revision'],
          'initial_lora_tensor_sha256':hashlib.sha256(b''.join(v.detach().cpu().float().numpy().tobytes() for n,v in sorted(model.named_parameters()) if v.requires_grad)).hexdigest(),
          'architecture':'frozen SFT parent + residual LoRA '+('+ intermediate FiLM' if a.arm=='B' else 'alone'),
          'film_layers':film.selected if a.arm=='B' else [],'film_precision':'FP32 affine and FP32 addition before cast',
          'checkpointing':'one candidate forward/backward before changing contextual hook state',
          'selection':'family macro correct-both-contexts accuracy, then SFT NLL; held-out families only',
          'resource_qualification':'GNOSIS percentage proxy, HLS estimate, no achieved timing',
          'parent_trainables':inventory,'train_set_counts':bank['counts']['train'],
          'val_set_counts':bank['counts']['val'],'eval_sft':len(val['sft']),'eval_pairs':len(val['pairs']),
          'hyperparams':vars(a)})
    def score(sample,row,active,backward_coeff=None):
        ctx=context_features(row).to(device)[None,None,:];kinds=sample['kinds'].to(device)[None]
        with hooks.bind(ctx,kinds,enabled=active):
            with torch.autocast('cuda',dtype=torch.float16):
                o=model(input_ids=sample['ids'].to(device)[None],attention_mask=sample['attn'].to(device)[None],use_cache=False)
            # Selected-logp implementation uses correct causal shift and action-normalized weights.
            _,nll,_=selected_logps(o.logits,sample['labels'].to(device),sample['weights'].to(device))
            score=-nll
            if backward_coeff is not None:
                # CRITICAL: keep context bound through activation-checkpoint recomputation.
                scaler.scale(backward_coeff*score).backward()
            return score
    def backprop(sample,row,coeff,active):
        s=score(sample,row,active,backward_coeff=coeff)
        return float(s.detach())
    def eval_now(step):
        model.eval();film.eval();film.last.clear();results={}
        for active,label in ([(False,'off'),(True,'on')] if a.arm=='B' else [(False,'off')]):
            out={'sft':[],'pairs':[]}
            with torch.no_grad():
                for x in val['sft']:
                    row=x['row'];s=score(encode.get(row,row['target']),row,active)
                    out['sft'].append({'family':row['_family'],'nll':float(-s),'context_id':x['context_id']})
                for x in val['pairs']:
                    row=x['row'];sp=float(score(encode.get(row,x['preferred_target']),row,active))
                    sn=float(score(encode.get(row,x['rejected_target']),row,active))
                    out['pairs'].append({'family':row['_family'],'kind':x['kind'],'group_id':x['group_id'],
                                        'margin':sp-sn,'correct':sp>sn})
            fam=sorted(set(x['family'] for x in out['pairs']))
            macro=sum(sum(t['correct'] for t in out['pairs'] if t['family']==f)/
                      sum(t['family']==f for t in out['pairs']) for f in fam)/len(fam)
            bykind={k:{'n':sum(t['kind']==k for t in out['pairs']),
                    'accuracy':sum(t['correct'] for t in out['pairs'] if t['kind']==k)/max(1,sum(t['kind']==k for t in out['pairs']))}
                    for k in ('clock_switch','budget_switch')}
            bygroup=collections.defaultdict(list)
            for entry in out['pairs']:bygroup[(entry['family'],entry['kind'],entry['group_id'])].append(entry)
            pair_complete={k:len(v)==2 and all(x['correct'] for x in v) for k,v in bygroup.items()}
            family_group={f:sum(ok for (fam,_,_),ok in pair_complete.items() if fam==f)/
                              sum(fam==f for (fam,_,_) in pair_complete) for f in fam}
            group_macro=sum(family_group.values())/len(family_group)
            results[label]={'macro_pair_accuracy':macro,'macro_two_context_group_accuracy':group_macro,
                            'two_context_groups':len(pair_complete),'families_with_groups':len(fam),'pair_by_kind':bykind,
                            'sft_nll':sum(t['nll'] for t in out['sft'])/len(out['sft']),
                            'paired_results':out['pairs'],'nll_results':out['sft']}
        result={'step':step,'arms':results,'film_numerics':film.last[-min(20,len(film.last)):],
                'timestamp':time.time()};save_json(a.out/('eval_%04d.json'%step),result)
        print('[EVAL]',step,{k:(v['macro_pair_accuracy'],v['sft_nll']) for k,v in results.items()},flush=True)
        return results['on' if a.arm=='B' else 'off']
    train_sft=collections.defaultdict(list)
    for item in data['train']['sft']:train_sft[item['context_id']].append(item)
    # Stratify chosen training contexts over kernel/device/clock, then round-robin
    # budget contexts. Without this, one pass over many budgets never visits a
    # context's additional near-optimal references.
    strata=collections.defaultdict(list)
    for context,items in train_sft.items():
        row=items[0]['row'];strata[(row['kernel_name'],row['device'],float(prod._clock_of(row)))].append(context)
    for k,v in strata.items():
        def priority(c):
            row=train_sft[c][0]['row'];bud=tuple(float(row[b]) for b in FEATURE_KEYS)
            return (0 if all(b==1 for b in bud) else 1 if min(bud)<.6 else 2,
                    int(hashlib.sha256(c.encode()).hexdigest(),16))
        v.sort(key=priority)
    context_ids=[]
    while len(context_ids)<a.train_contexts and any(strata.values()):
        for k in sorted(strata):
            if strata[k]:context_ids.append(strata[k].pop(0))
            if len(context_ids)>=a.train_contexts:break
    assert len(context_ids)>0
    if a.steps < 3*len(context_ids):
        raise ValueError(f'{a.steps} steps for {len(context_ids)} selected contexts: need >=3 visits/context to cover three near-optimal references')
    rnd=random.Random(a.seed);rnd.shuffle(context_ids)
    train_pairs=data['train']['pairs'];rnd.shuffle(train_pairs)
    save_json(a.out/'training_context_coverage.json',{'selected_contexts':len(context_ids),
             'available_contexts':len(train_sft),'visits_min':a.steps//len(context_ids),
             'visits_max':math.ceil(a.steps/len(context_ids)),
             'device_clock_counts':dict(collections.Counter(str(train_sft[c][0]['row']['device'])+'|'+str(prod._clock_of(train_sft[c][0]['row'])) for c in context_ids))})
    def ckpt(step,metric):
        folder=a.out/('checkpoint-%04d'%step);folder.mkdir()
        model.save_pretrained(folder,selected_adapters=['device_adapt'],save_embedding_layers=False)
        if a.arm=='B':torch.save(film.state_dict(),folder/'film.pt')
        torch.save({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},folder/'progress.pt')
        save_json(folder/'receipt.json',{'step':step,'arm':a.arm,'parent_sha256':parent_hash,
            'bank_manifest_sha256':sha(a.bank/'manifest.json'),'metric':metric,
            'lora_sha256':sha(folder/'device_adapt'/'adapter_model.safetensors'),
            'film_sha256':sha(folder/'film.pt') if a.arm=='B' else None})
        return folder
    best_key=(-1e20,-1e20);start=time.monotonic();resume_step=0
    if a.resume_from:
        cp=a.resume_from
        receipt=json.loads((cp/'receipt.json').read_text())
        if receipt['arm']!=a.arm or receipt['parent_sha256']!=parent_hash or receipt['bank_manifest_sha256']!=sha(a.bank/'manifest.json'):
            raise ValueError('Checkpoint provenance mismatch')
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model,load_file(str(cp/'device_adapt'/'adapter_model.safetensors')),adapter_name='device_adapt')
        if a.arm=='B':film.load_state_dict(torch.load(cp/'film.pt',map_location=device,weights_only=True))
        progress=torch.load(cp/'progress.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(progress['optimizer']);scaler.load_state_dict(progress['scaler'])
        torch.set_rng_state(progress['rng']);torch.cuda.set_rng_state_all(progress['cuda_rng'])
        resume_step=int(receipt['step']);assert 0<=resume_step<a.steps
        if (a.out/'best.json').is_file():
            prev=json.loads((a.out/'best.json').read_text())['selection']
            best_key=(prev['macro_two_context_group_accuracy'],-prev['sft_nll'])
        save_json(a.out/('resume_%04d.json'%resume_step),{'source':str(cp),'step':resume_step,
                 'note':'Discard any partial log after this checkpoint; reproducibility starts at checkpoint state'})
    for step in range(resume_step,a.steps+1):
        if (step%a.eval_every==0 or step==a.steps) and not (a.resume_from and step==resume_step):
            m=eval_now(step);k=(m['macro_two_context_group_accuracy'],-m['sft_nll'])
            dest=ckpt(step,m)
            if k>best_key:
                best_key=k;save_json(a.out/'best.json',{'checkpoint':str(dest),'step':step,'selection':m})
        if step==a.steps:break
        model.train();film.train()
        rank_stat=None
        # Optimizer steps must be successful in BOTH arms; repeat a skipped
        # update with a reduced scaler instead of counting it.
        for attempt in range(5):
            optimizer.zero_grad(set_to_none=True)
            # Exact same deterministic schedule for both A and B.
            context=context_ids[step%len(context_ids)]
            eligible=train_sft[context];item=eligible[(step//len(context_ids))%len(eligible)];row=item['row']
            sample=encode.get(row,row['target']);active=a.arm=='B'
            sft=backprop(sample,row,-1.,active)
            rank_stat=None
            if train_pairs:
                pair=train_pairs[step%len(train_pairs)];r=pair['row']
                pos=encode.get(r,pair['preferred_target']);neg=encode.get(r,pair['rejected_target'])
                # Detached margins supply exact analytic coefficients. Two separate
                # grad forward/backwards avoid checkpointing under the wrong FiLM state.
                with torch.no_grad():
                    sp=float(score(pos,r,active));sn=float(score(neg,r,active))
                w=a.rank_budget if pair['kind']=='budget_switch' else a.rank_clock
                grad=w/(1+math.exp(max(-60.,min(60.,sp-sn))))
                backprop(pos,r,-grad,active);backprop(neg,r,+grad,active)
                rank_stat={'kind':pair['kind'],'margin':sp-sn,'weight':w,'correct':sp>sn}
            scaler.unscale_(optimizer)
            parameters=trainable+(list(film.parameters()) if active else [])
            norms=torch.nn.utils.clip_grad_norm_(parameters,1.)
            fi_grad=sum(float(p.grad.detach().float().square().sum()) for p in film.parameters() if p.grad is not None)**.5
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<old
            if not skipped:break
            with (a.out/'skipped_updates.jsonl').open('a') as fh:
                fh.write(json.dumps({'step':step+1,'attempt':attempt,'scale_before':old,'scale_after':scaler.get_scale()})+'\n')
        else:
            raise FloatingPointError('Five skipped updates on same batch; diagnose numerics')
        gates=[float(g.detach().tanh()) for g in film.gates.values()]
        with (a.out/'training.jsonl').open('a') as f:
            f.write(json.dumps({'step':step+1,'sft_nll':-sft,'rank':rank_stat,'grad_norm':float(norms),
                                'film_grad_norm':fi_grad,'film_gates':gates,'seconds':round(time.monotonic()-start,3)})+'\n')
        if (step+1)%10==0:print('[TRAIN]',a.arm,step+1,'sft',-sft,'rank',rank_stat,'film_grad',fi_grad,'gate',gates,flush=True)
    save_json(a.out/'completed.json',{'arm':a.arm,'steps':a.steps,'elapsed_seconds':time.monotonic()-start,
        'physical_HLS_calls':0,'parent_adapter_sha256':parent_hash,'selected':json.loads((a.out/'best.json').read_text())['step']})
    print('COMPLETE',a.arm,flush=True)

if __name__=='__main__':main()
