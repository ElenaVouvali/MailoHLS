#!/usr/bin/env python3
"""Matched U50 few-shot adaptation of frozen FiLM/GMT Stage-2 checkpoint-0300.

This is a *retrospective K-labeled-measurement* study.  Never imports U50
training measurements for final unseen-kernel inference; only the selected U50
support/validation rows are read during training.  Checkpoint LoRAs, FiLM weights,
trained structural XAttn, historical GNN/GMT and tokenizer remain frozen.

The only learned tensor is a U50-specific 5x64 action-kind offset injected into
FiLM's existing 64D bottleneck.  This is a small target-conditioned adapter, not
a new GNN or a re-trained Stage-2 model.
"""
from __future__ import annotations
import argparse, collections, contextlib, hashlib, json, math, os, random, re
from pathlib import Path
from types import SimpleNamespace

U50 = 'xcu50-fsvh2104-2-e'
CAPS = {'BRAM_18K':2688,'DSP':5952,'FF':1743360,'LUT':871680}
RESOURCES = tuple(CAPS)
KIND_COUNT = 5
BOTTLENECK = 64
SUPPORTED_SHOTS = (4, 8, 16, 22)
STAGE2_RUN = 'runs/campaign_s123_v1/final'
MODEL_ID = 'FiLM Arm B checkpoint-0600 frozen + historical GMT B frozen + gated Xattn'
TEST_KERNELS = ('GAN', 'Kalman', 'chstone-jpeg', 'covariance')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,p)


def lines(p): return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def dataset_paths(root):
    base=root/'mailohls_u50_adaptation'/'artifacts'
    return base/'u50_10ns_sft.jsonl',base/'u50_10ns_sft.sources.json'


def family(k):
    # Pure-stdlib version of the audited canonical mapping, for CPU preflight.
    z=k.strip().lower().replace('-','_')
    if z.startswith('machsuite_gemm'):return 'machsuite_gemm'
    if z.startswith('machsuite_'):
        p=z.split('_');return '_'.join(p[:-1]) if len(p)>=3 and p[-1].isdigit() else z
    if z.startswith('spcl_example'):return 'spcl_example'
    if z.startswith('rodinia_'):
        r=z[8:]
        for a in ('cfd_flux','cfd_step_factor','lc_gicov','lc_mgvf'):
            if r.startswith(a):return 'rodinia_'+a
        return 'rodinia_'+r.split('_')[0]
    return z


def check_parent(repo,deployment):
    run=deployment/STAGE2_RUN
    cp=run/'checkpoint-0300'
    receipt=json.loads((cp/'receipt.json').read_text())
    proto=json.loads((run/'protocol.json').read_text())
    if proto.get('model')!=MODEL_ID or 'compiler_relational' not in proto.get('routing',''):
        raise ValueError('Not the final compiler-relational FiLM-GMT Stage2 policy')
    if receipt.get('step')!=300 or sha(cp/'structural_xattn.pt')!=receipt['structural_state_sha256']:
        raise ValueError('Wrong or altered Stage2 checkpoint-0300')
    digest=hashlib.sha256(json.dumps(proto,sort_keys=True,default=str).encode()).hexdigest()
    if digest!=receipt['protocol_sha256']:
        raise ValueError('Stage2 protocol/receipt mismatch')
    # Do not assume the active checkout is the exact training-source version.
    if sha(repo/'experiments/film_gmt_stage2_final/train.py')!=proto['launcher_sha256']:
        raise ValueError('Current source differs from checkpoint trainer; use pinned film-gmt-stage2-final checkout')
    return dict(run=str(run),cp=str(cp),stage2_sha=receipt['structural_state_sha256'],
                protocol_sha=receipt['protocol_sha256'],frozen_sha=receipt['frozen_composite_sha256'])


def input_rows(repo):
    p,src=dataset_paths(repo)
    raw=lines(p)
    bundle=json.loads(src.read_text())
    templates=bundle['templates']
    if len(raw)!=72:raise ValueError('Expected authenticated U50 72-example SFT bank, got '+str(len(raw)))
    counts=collections.Counter(r['kernel_name'] for r in raw)
    if len(counts)!=20: raise ValueError('Expected U50 SFT bank to cover exactly 20 kernels')
    items=[]
    seen=set()
    for r in raw:
        k=r['kernel_name']
        if r.get('device')!=U50 or abs(float(r.get('clock_period',-1))-10.)>1e-6:
            raise ValueError('U50-only 10ns data required')
        if k not in templates:raise ValueError('Missing source template '+k)
        if r.get('schema')!='mailohls-sft-example-v2-compact-source':raise ValueError('Wrong U50 SFT schema')
        if not math.isfinite(float(r['latency'])) or r['latency']<=0: raise ValueError('Invalid measured latency')
        if any(not math.isfinite(float(r[t])) or not 0<=float(r[t])<100 for t in ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%')):
            raise ValueError('Invalid measured U50 utilization')
        key=(k,r['target'].strip().upper())
        if key in seen:raise ValueError('Duplicate measured U50 configuration '+k)
        seen.add(key)
        row=dict(r,input=templates[k],obj_mode='PARETO_LATENCY',frequency_mode='specified',
            selected_clock_period=10.,
            avail_bram=CAPS['BRAM_18K'],avail_dsp=CAPS['DSP'],
            avail_ff=CAPS['FF'],avail_lut=CAPS['LUT'],
            budget_frac_bram_18k=1.,budget_frac_dsp=1.,
            budget_frac_ff=1.,budget_frac_lut=1.,
            _family=family(k))
        items.append(row)
    return items,p,src


def get_eligible(repo, memorydir, inputs):
    manifest=json.loads((memorydir/'memory_manifest.json').read_text())
    if manifest.get('kind')!='B_GMT':raise ValueError('Source memory must be exact historical GMT B')
    byk=collections.defaultdict(list)
    for r in inputs:byk[r['kernel_name']].append(r)
    u50_stats=json.loads((repo/'mailohls_u50_adaptation/artifacts/u50_10ns_sft.manifest.json').read_text())
    u50_source_sha={r['kernel']:r['source_sha256'] for r in u50_stats['kernel_statistics']}
    gnosis=json.loads((repo/'Data/manifest.json').read_text())
    canon={r['kernel']:r for r in gnosis['kernels']}
    eligible,blocked={},{}
    # Only original authenticated source mappings can be used by current GMT bank.
    for k,rows in sorted(byk.items()):
        filename=k+'.memory.pt'
        file=memorydir/filename
        if filename not in manifest.get('files',{}) or not file.is_file():
            blocked[k]='missing authenticated historical GMT source/action memory'
            continue
        if sha(file)!=manifest['files'][filename]:
            blocked[k]='memory file differs from signed manifest'
            continue
        source=repo/rows[0]['source_file']
        if not source.is_file():blocked[k]='canonical source missing';continue
        if sha(source)!=u50_source_sha.get(k):
            blocked[k]='source hash differs from U50 measured bank';continue
        original=canon.get(k)
        if original is None or sha(source)!=original['files']['source']['sha256']:
            blocked[k]='source hash differs from pinned GNOSIS source';continue
        # U50 compact templates must use the same actual source that generated
        # historical GNN anchors; this check catches stale/unrelated snapshots.
        md=repo/'Data/ApplicationDataset'/k/'kernel_info.txt'
        if not md.is_file():blocked[k]='action metadata missing';continue
        try:
            from LLM_branch.common import directive_domains as domains
            actions=domains.load_source_actions(k,repo/'Data/ApplicationDataset')
            import re
            source_labels=set(re.findall(r'\b(L[1-9][0-9]*)\s*:',rows[0]['input'],flags=re.I))
            if source_labels!=set(actions):
                blocked[k]='compact source action labels differ from metadata';continue
        except Exception as exc:
            blocked[k]='source metadata invalid: '+str(exc);continue
        eligible[k]=rows
    return eligible,blocked,manifest


def sample_pairs(rows_by_kernel,shot,seed):
    """Select two measured, legal, distinguishable designs per support family.

    Family and design ordering is fixed by hash.  A statistically ambiguous
    pair (less than 0.5% relative latency difference) is *excluded*, never
    assigned a preferred/rejected label.  Selection is retrospective and the
    audit reports the filter; no claim of prospective acquisition is made.
    """
    if shot not in SUPPORTED_SHOTS:raise ValueError('Supported measured-design budgets: '+str(SUPPORTED_SHOTS))
    byfam=collections.defaultdict(list)
    for k,v in rows_by_kernel.items():
        if len(v)>=2:byfam[family(k)].append(k)
    fams=sorted(byfam,key=lambda f:hashlib.sha256(f'{seed}:family:{f}'.encode()).hexdigest())
    pairs=[]; skipped=[]
    for f in fams:
        k=min(byfam[f],key=lambda k:hashlib.sha256(f'{seed}:kernel:{k}'.encode()).hexdigest())
        choices=sorted(rows_by_kernel[k],key=lambda r:hashlib.sha256(f'{seed}:{k}:{r["target"]}'.encode()).hexdigest())[:2]
        x,y=choices
        relative_gap=abs(float(x['latency'])-float(y['latency']))/max(float(x['latency']),float(y['latency']))
        if relative_gap < 0.005:
            skipped.append({'kernel':k,'family':f,'relative_latency_gap':relative_gap})
            continue
        good,bad=(x,y) if x['latency']<y['latency'] else (y,x)
        pairs.append({'kernel':k,'family':f,'chosen':good,'rejected':bad,
                      'relative_latency_gap':relative_gap,'measurements_charged':2,
                      'selection':'two hash-selected measured U50 designs; near-equal QoR pairs excluded'})
        if len(pairs)==shot//2:break
    if len(pairs)*2!=shot:
        raise ValueError(f'Only {len(pairs)*2} qualified measured support designs for requested {shot}; '
                         f'{len(skipped)} family pairs were near-equal. Lower SHOTS or audit the inputs.')
    return pairs


def qualify_gold_domains(repo,parent,eligible):
    """Check every U50 measurement against the *exact* frozen model RHS menu.

    Exclude unsupported measured targets without relabeling, canonicalizing,
    replacing, or pretending they are additional measured configurations.
    This preflight runs on CPU before GPU allocation and applies identically
    to train and validation families.
    """
    from LLM_branch.common import mailohls_contract as mc
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    registry_path=Path(parent)/'directive_domain_registry.json'
    if not registry_path.is_file():raise FileNotFoundError(registry_path)
    registry=inf.load_directive_domain_registry(str(registry_path))
    add_domains(registry,eligible,repo)
    assign=re.compile(r'^auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*)\}\s*=\s*(\S+)\s*$',re.I)
    qualified={}; exclusions=[]
    for kernel,rows in sorted(eligible.items()):
        available=registry[inf.normalize_kname(kernel)]
        plan=inf.extract_ordered_lhs_plan(rows[0]['input'])
        valid=[]
        for row in rows:
            target={}
            for entry in row['target'].splitlines():
                if not entry.strip():continue
                m=assign.fullmatch(entry.strip())
                if not m:raise ValueError(f'Malformed measured U50 directive {kernel}: {entry!r}')
                lhs='AUTO{'+m.group(1).upper()+'}'
                if lhs in target:raise ValueError(f'Duplicate measured target directive {kernel}/{lhs}')
                target[lhs]=m.group(2).lower()
            required={lhs.upper() for _,lhs in plan}
            if set(target)!=required:
                raise ValueError(f'{kernel}: measured target/source-field inventory mismatch: '
                                 f'missing={sorted(required-set(target))}, extra={sorted(set(target)-required)}')
            chosen={};reason=None
            for _,lhs in plan:
                lhs=lhs.upper()
                legal=mc.filter_semantic_candidates(lhs,
                    inf.get_rhs_candidates_for_lhs(kernel,lhs,registry),chosen,available)
                rhs=target[lhs]
                if rhs not in legal:
                    reason=f'{lhs}={rhs} not in frozen model source domain {legal}'
                    break
                chosen[lhs]=rhs
            if reason is None:mc.validate_directive_assignments(target)
            if reason is None:valid.append(row)
            else:exclusions.append({'kernel':kernel,
                                    'target_sha256':hashlib.sha256(row['target'].encode()).hexdigest(),
                                    'reason':reason})
        if valid:qualified[kernel]=valid
    return qualified,exclusions


def audit(a):
    idn=check_parent(a.repo,a.deployment)
    data,p,src=input_rows(a.repo)
    train_names=set(json.loads((a.repo/'experiments/u50_adaptation/config.json').read_text())['train_kernels'])
    val_names=set(json.loads((a.repo/'experiments/u50_adaptation/config.json').read_text())['val_kernels'])
    if train_names & val_names:raise ValueError('Train/validation kernels overlap')
    train_rows=[r for r in data if r['kernel_name'] in train_names]
    val_rows=[r for r in data if r['kernel_name'] in val_names]
    if len(train_rows)+len(val_rows)!=len(data): raise ValueError('Unexpected SFT kernel outside prior family split')
    if {family(k) for k in train_names}&{family(k) for k in val_names}:raise ValueError('U50 train/val family leak')
    mem=Path(a.memory_dir)
    eligible,blocked,manifest=get_eligible(a.repo,mem,data)
    if getattr(a,'parent',None) is None:
        raise ValueError('Audit requires --parent to check actual frozen Stage-1 source domains')
    qualified,domain_exclusions=qualify_gold_domains(a.repo,a.parent,eligible)
    train={k:v for k,v in qualified.items() if k in train_names}
    val={k:v for k,v in qualified.items() if k in val_names}
    if len(val)<2:raise ValueError('Fewer than two available independent validation families')
    if len({family(k) for k in val})<2:raise ValueError('Validation has fewer than two families')
    sample=sample_pairs(train,a.shots,a.seed)
    sha_data={'sft':sha(p),'sources':sha(src),'memory_manifest':sha(mem/'memory_manifest.json')}
    result=dict(schema='salt-u50-fewshot-audit-v1',model=idn,
        u50_measurements=len(data),u50_sft_sha256=sha_data,
        measured_shots=a.shots,seed=a.seed,
        support_pairs=[{'kernel':r['kernel'],'family':r['family'],
                       'chosen_sha':hashlib.sha256(r['chosen']['target'].encode()).hexdigest(),
                       'rejected_sha':hashlib.sha256(r['rejected']['target'].encode()).hexdigest(),
                       'chosen_latency_msec':r['chosen']['latency'],
                       'rejected_latency_msec':r['rejected']['latency']} for r in sample],
        train_available_kernels=sorted(train),validation_available_kernels=sorted(val),
        blocked_kernels=blocked,domain_exclusions=domain_exclusions,
        pair_eligibility_min_relative_latency_gap=0.005,
        qualified_support_kernels=len(train),qualified_validation_kernels=len(val),
        u50_conditioning='existing UNKNOWN device token, measured physical capacities, two original FiLM device bits [0,0] + U50-only FiLM bottleneck residual',
        evaluation_test_kernels=list(TEST_KERNELS),
        warning='Retrospective shots selected from previously synthesized and preprocessed U50 pool; not an 8-call prospective acquisition experiment')
    if a.out:
        if Path(a.out).exists():raise FileExistsError('Audit outputs are immutable: '+str(a.out))
        dump(a.out,result)
    print(json.dumps(result,indent=2));return result,train,val,sample,manifest


def install_u50_contract():
    # This is an explicit runtime extension. It neither changes learned tokens
    # nor the repository's historical checkpoint files.
    from LLM_branch.common import mailohls_contract as mc
    from experiments.gnosis_stage1_ab import train_gnosis_ab as gn
    mc.DEVICE_RESOURCES[U50]=dict(CAPS)
    original=gn.context_features
    def u50_features(row):
        from LLM_branch.train import train_SFT_xattn_new as prod
        import torch
        dev=prod._norm_device(row['device'])
        if dev!=U50:return original(row)
        clock=float(prod._clock_of(row))
        if clock not in (3.33,5.,10.):raise ValueError('Unsupported unseen-clock extrapolation')
        fractions=[float(row[k]) for k in gn.FEATURE_KEYS]
        if any(not 0<x<=1 for x in fractions):raise ValueError('Invalid U50 resource fractions')
        cap=[float(CAPS[t]) for t in mc.RESOURCE_KEYS]
        return torch.tensor([0.,0.,math.log(clock/10.),
            *[math.log1p(v)/16. for v in cap],*fractions,
            *[math.log1p(c*b)/16. for c,b in zip(cap,fractions)]],dtype=torch.float32)
    gn.context_features=u50_features
    return original


def load_runtime(a):
    # Call the actual Stage2 builder so the model/tokenizer/FiLM/GMT contracts are
    # those of the selected checkpoint, not a new approximate implementation.
    import torch
    from experiments.film_gmt_stage2_final import train
    from LLM_branch.common import structural_memory as sm
    idn=check_parent(a.repo,a.deployment)
    cp=Path(idn['cp']);args=SimpleNamespace(parent=a.parent,film_checkpoint=a.film_checkpoint,
         seed=a.seed,lr_gate=2e-4,lr_xattn=1e-4,weight_decay=.01,max_length=7168)
    # Recheck inherited Stage1/FiLM hashes, not only Stage2 weight metadata.
    train.require_hash(a.parent/'adapter_model.safetensors',train.PARENT_SHA,'frozen V4 parent')
    train.require_hash(a.film_checkpoint/'device_adapt/adapter_model.safetensors',train.RESIDUAL_SHA,'FiLM B residual LoRA')
    train.require_hash(a.film_checkpoint/'film.pt',train.FILM_SHA,'FiLM Arm B')
    parent_contract=json.loads((a.parent/'training_contract.json').read_text())
    model,film,hooks,tok,registry,encoder,_,scaler,dtype,_=train.model_and_runtime(args,parent_contract)
    original_state=torch.load(cp/'structural_xattn.pt',map_location='cpu',weights_only=True)
    expected={n for n in model.state_dict() if 'gated_cross_attn_layer' in n}
    if set(original_state)!=expected:raise ValueError('Structural State2 tensor inventory differs')
    model.load_state_dict(original_state,strict=False)
    if train.tensors_sha(model,film)!=idn['frozen_sha']:
        raise ValueError('Loaded original V4+residual+FiLM lineage differs from checkpoint300 receipt')
    model.requires_grad_(False)
    film.requires_grad_(False)
    memories={}
    for label,directory in [('support',a.memory_dir),('evaluation',a.eval_memory_dir)]:
        if directory is None:continue
        directory=Path(directory)
        manifest=json.loads((directory/'memory_manifest.json').read_text())
        if manifest.get('kind')!='B_GMT':raise ValueError(label+' not frozen historical GMT B')
        for name,hash_ in manifest['files'].items():
            if sha(directory/name)!=hash_:raise ValueError(label+' memory hash mismatch '+name)
        bank,dim=sm.load_memory_bank(str(directory),expected_mem_dim=128,expected_max_slots=64,require_pragma_free_memory=True)
        if dim!=128:raise ValueError('Memory width differs')
        memories[label]=bank
    if 'evaluation' in memories:
        a_man=json.loads((Path(a.memory_dir)/'memory_manifest.json').read_text())
        b_man=json.loads((Path(a.eval_memory_dir)/'memory_manifest.json').read_text())
        for k in ('gnn_checkpoint_sha256','gmt_checkpoint_sha256'):
            if a_man.get(k)!=b_man.get(k):raise ValueError('External source memory from different GNN/GMT lineage')
    return model,film,hooks,tok,registry,encoder,scaler,dtype,memories,idn

class U50KindCode:
    def __init__(self,film):
        import torch
        self.film=film
        self.code=torch.nn.Parameter(torch.zeros(KIND_COUNT,BOTTLENECK,device='cuda:0',dtype=torch.float32))
        def add_code(module,inputs,output):
            state=self.film.state
            if state is None:return output
            kinds=state[1]
            selection=kinds[kinds>0].long()
            if output.ndim!=2 or output.size(0)!=selection.numel() or output.size(1)!=BOTTLENECK:
                raise RuntimeError('FiLM bottleneck hook received wrong decision sites')
            return output + .5*torch.tanh(self.code[selection-1]).to(output.dtype)
        self.handle=film.enc.register_forward_hook(add_code)
    def close(self):self.handle.remove()
    def set(self,tensor):
        import torch
        if tuple(tensor.shape)!=(KIND_COUNT,BOTTLENECK):raise ValueError('Wrong U50 adapter tensor')
        with torch.no_grad():self.code.copy_(tensor.to(self.code.device))


def add_domains(registry,rows,root):
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    # Only the source program/action metadata determine candidate domains.
    cases=[]
    for k,rs in rows.items():
        row=rs[0]
        cases.append(inf.InferenceCase(kernel_name=k,source_text=row['input'],obj_mode='PARETO_LATENCY',
           w_lat=1.0,w_area=0.0,platform_row=row))
    inf.add_missing_source_directive_domains(registry,cases,str(root/'Data/ApplicationDataset'))


def evaluate(model,film,hooks,enc,bank,dtype,val_pairs):
    import torch
    from experiments.film_gmt_stage2_final import train
    model.eval();film.eval()
    result=[]
    with torch.no_grad():
        for p in val_pairs:
            good=p['chosen'];bad=p['rejected'];row=good
            g=train.bind_sample(model,hooks,enc.get(row,good['target']),row,bank,dtype,None)
            b=train.bind_sample(model,hooks,enc.get(row,bad['target']),row,bank,dtype,None)
            result.append(dict(kernel=p['kernel'],family=p['family'],chosen_nll=-g,
                               pair_accuracy=int(g>b),score_margin=g-b))
    macro_nll=sum(x['chosen_nll'] for x in result)/len(result)
    macro_acc=sum(x['pair_accuracy'] for x in result)/len(result)
    return {'macro_chosen_nll':macro_nll,'macro_pair_accuracy':macro_acc,'families':result}


def val_pairs_for(val):
    pairs=[]
    for k,rows in sorted(val.items()):
        ranked=sorted(rows,key=lambda r:(r['latency'],r['target']))
        if len(ranked)<2:raise ValueError('Not enough domain-qualified validation U50 measurements for '+k)
        if not float(ranked[0]['latency'])<float(ranked[-1]['latency']):
            raise ValueError('Cannot create a preference label from tied validation latency: '+k)
        pairs.append({'kernel':k,'family':family(k),'chosen':ranked[0],'rejected':ranked[-1]})
    return pairs


def train_main(a):
    import torch
    a.out=Path(a.out)
    if a.out.exists():raise FileExistsError('Refusing overwrite existing fewshot run '+str(a.out))
    audit_,train_rows,val_rows,support,manifest=audit(SimpleNamespace(**{**vars(a),'out':None}))
    install_u50_contract()
    model,film,hooks,tok,registry,enc,scaler,dtype,memories,idn=load_runtime(a)
    bank=memories['support']
    add_domains(registry,{**train_rows,**val_rows},a.repo)
    calibration=U50KindCode(film)
    try:
        # Explicitly audit that frozen U50 zero-code is the unchanged policy.
        validation=val_pairs_for(val_rows)
        initial=evaluate(model,film,hooks,enc,bank,dtype,validation)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        random.seed(a.seed)
        optimizer=torch.optim.AdamW([calibration.code],lr=a.lr,weight_decay=0.)
        a.out.mkdir(parents=True)
        dump(a.out/'audit.json',audit_)
        protocol=dict(schema='salt-u50-film-gmt-fewshot-v1',shots=a.shots,seed=a.seed,
            steps=a.steps,lr=a.lr,rank_lambda=a.rank_lambda,stage2=idn,
            memory_manifest_sha=sha(Path(a.memory_dir)/'memory_manifest.json'),
            measured_sft_sha=audit_['u50_sft_sha256'],
            adapter_source_sha256=sha(Path(__file__)),
            adapter='U50-only 5x64 code at existing 64D FiLM bottleneck; parent model/FiLM/GMT/XAttn frozen',
            selected_shot_hash=hashlib.sha256(canonical(audit_['support_pairs']).encode()).hexdigest(),
            val_families=sorted(val_rows),train_families=sorted(train_rows))
        dump(a.out/'protocol.json',protocol)
        best=(initial['macro_chosen_nll'],-initial['macro_pair_accuracy'],0)
        selected=torch.zeros_like(calibration.code).cpu()
        best_step=0
        trace=[{'step':0,'validation':initial}]
        # One pair per minibatch, both completion likelihoods always use exactly
        # the same U50 prompt/context and compiler-memory mask.
        from experiments.film_gmt_stage2_final import train
        from experiments.final_architecture_v3.preferences_objective import selected_logps
        for step in range(1,a.steps+1):
            model.train();film.eval();optimizer.zero_grad(set_to_none=True)
            pair=support[(step-1)%len(support)]
            good,bad=pair['chosen'],pair['rejected']
            pos=enc.get(good,good['target']);neg=enc.get(good,bad['target'])
            if a.rank_lambda:
                model.eval()
                with torch.no_grad():
                    pg=train.bind_sample(model,hooks,pos,good,bank,dtype,None)
                    ng=train.bind_sample(model,hooks,neg,good,bank,dtype,None)
                model.train()
                coeff=a.rank_lambda/(1.+math.exp(max(-40.,min(40.,pg-ng))))
            else: coeff=0.
            train.bind_sample(model,hooks,pos,good,bank,dtype,(scaler,-1.0-coeff))
            if coeff:train.bind_sample(model,hooks,neg,good,bank,dtype,(scaler,+coeff))
            scaler.unscale_(optimizer)
            grad=float(calibration.code.grad.float().norm()) if calibration.code.grad is not None else 0.
            if not math.isfinite(grad):raise FloatingPointError('Nonfinite U50 code gradient')
            torch.nn.utils.clip_grad_norm_([calibration.code],1.)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<previous:raise FloatingPointError('AMP skipped a few-shot update; not counting it')
            if step%a.eval_interval==0 or step==a.steps:
                m=evaluate(model,film,hooks,enc,bank,dtype,validation)
                # Baseline included as candidate. NLL selection may favor the
                # unchanged policy; never force an unqualified adaptation.
                objective=(m['macro_chosen_nll'],-m['macro_pair_accuracy'],step)
                trace.append({'step':step,'validation':m,'code_l2':float(calibration.code.detach().norm()),'grad_l2':grad})
                if objective<best:
                    best=objective;selected=calibration.code.detach().cpu().clone();best_step=step
                print('[EVAL]',step,'chosen_nll',m['macro_chosen_nll'],'rank_acc',m['macro_pair_accuracy'],'best_step',best_step,flush=True)
        import io
        result=dict(schema='salt-u50-calibration-result-v1',best_step=best_step,
            selected_validation=next(x['validation'] for x in trace if x['step']==best_step),
            initial_validation=initial,trainable_params=KIND_COUNT*BOTTLENECK,
            status='NO_SELECTED_ADAPTATION' if best_step==0 else 'ADAPTED_CHECKPOINT',
            run_protocol_sha256=sha(a.out/'protocol.json'),
            note='Validation is development only; independent HLS performance must be measured on external heldout kernels')
        # Save only 384 numbers, never duplicate parent 6.7B parameters.
        torch.save({'code':selected,'parent_structural_sha256':idn['stage2_sha'],
                    'protocol_sha256':sha(a.out/'protocol.json'),'best_step':best_step},a.out/'u50_code.pt')
        dump(a.out/'result.json',result)
        dump(a.out/'training_trace.json',trace)
        print('[DONE]',a.out, 'selected',best_step,'validation',result['selected_validation'])
    finally:calibration.close()


def predict_main(a):
    import torch
    from LLM_branch.common import mailohls_contract as mc,directive_domains as domains,structural_memory as sm
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    from experiments.stage2_eval_compat9 import stage2_infer_compact_views as inference
    a.out=Path(a.out)
    if a.out.exists():raise FileExistsError('Predictions output must be a fresh directory')
    idn=check_parent(a.repo,a.deployment)
    record=torch.load(a.calibration/'u50_code.pt',map_location='cpu',weights_only=True)
    if record['parent_structural_sha256']!=idn['stage2_sha']:
        raise ValueError('U50 code belongs to a different Stage2 checkpoint')
    if record['protocol_sha256']!=sha(a.calibration/'protocol.json'):
        raise ValueError('Adaptation protocol modified')
    install_u50_contract()
    model,film,hooks,tok,registry,enc,scaler,dtype,memories,idn=load_runtime(a)
    model.requires_grad_(False);model.eval();film.eval()
    code=U50KindCode(film)
    results={'zero_shot':[],'few_shot':[]}
    try:
        if not set(a.test_kernels)<=set(TEST_KERNELS) or len(set(a.test_kernels))!=len(a.test_kernels):
            raise ValueError('Test kernels must be a unique subset of '+str(TEST_KERNELS))
        for kernel in a.test_kernels:
            folder=Path(a.eval_sources_root)/kernel
            src=inference.source_path(folder)
            prompt=domains.prepare_source_template(kernel,src,a.eval_sources_root)
            # Keep the original held-out HLS source unchanged; compact views
            # are prompt-only and must have identical directive label inventories.
            if kernel=='chstone-jpeg':
                view=a.deployment/'eval_prompt_views'/(kernel+'.c')
                if view.is_file():
                    compact=domains.prepare_source_template(kernel,view,a.eval_sources_root)
                    import re
                    keys=lambda x:set(re.findall(r'auto\{_[A-Z0-9_]+_L\d+\}',x,re.I))
                    if keys(compact)!=keys(prompt):raise ValueError('JPEG compact view changed action inventory')
                    prompt=compact
            actions=domains.load_source_actions(kernel,a.eval_sources_root)
            if len(actions)>64:raise ValueError('Test exceeds trained 64-action maximum')
            row={'kernel_name':kernel,'device':U50,'clock_period':10.,'selected_clock_period':10.,
                 'frequency_mode':'specified',
                 'avail_bram':CAPS['BRAM_18K'],'avail_dsp':CAPS['DSP'],
                 'avail_ff':CAPS['FF'],'avail_lut':CAPS['LUT'],
                 'budget_frac_bram_18k':1.,'budget_frac_dsp':1.,
                 'budget_frac_ff':1.,'budget_frac_lut':1.}
            case=inf.InferenceCase(kernel_name=kernel,source_text=prompt,obj_mode='PARETO_LATENCY',
                                  w_lat=1.,w_area=0.,platform_row=row)
            inf.add_missing_source_directive_domains(registry,[case],str(a.eval_sources_root))
            packed=sm.get_structural_memory_pack_for_kernel(memories['evaluation'],kernel,64,128,'compiler_relational')
            if packed[-1] is None:raise ValueError('Test memory missing compiler relations')
            for arm in results:
                tensor=torch.zeros_like(record['code']) if arm=='zero_shot' else record['code']
                code.set(tensor)
                with torch.inference_mode():assign,score=inference.decode(case,packed,actions,model,film,hooks,tok,registry,dtype)
                directives=inference.as_directives(assign,actions,case.source_text)
                results[arm].append({'kernel':kernel,'stage':'stage-2-gmt-u50-fewshot',
                    'objective':'LATENCY','latency':None,'resources':None,
                    'directives':directives,'device':U50,'clock_period_ns':10.,
                    'budget_percent':100.,'checkpoint_step':300,
                    'checkpoint_sha256':idn['stage2_sha'],
                    'source_sha256':sha(src),'mean_nonforced_site_logprob':score,
                    'status':'inference_only_not_synthesized',
                    'adaptation_arm':arm,'u50_code_sha256':sha(a.calibration/'u50_code.pt') if arm=='few_shot' else None,
                    'u50_calibration_step':record['best_step'] if arm=='few_shot' else 0})
                print('[PREDICT]',kernel,arm,'directives',len(directives),'score',score,flush=True)
        a.out.mkdir(parents=True)
        for arm,rows in results.items():
            (a.out/(arm+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
        dump(a.out/'prediction_manifest.json',{'schema':'salt-u50-heldout-generation-v1',
             'kernels':list(a.test_kernels),'stage2':idn,'adaptation_sha256':sha(a.calibration/'u50_code.pt'),
             'code_selected_step':record['best_step'],'U50_capacities':CAPS,
             'source_memory_manifest_sha256':sha(Path(a.eval_memory_dir)/'memory_manifest.json'),
             'policy':'same frozen Stage2 weights/decoder/unknown-device prompt/target context; zero vs U50-code only'})
        print('[PREDICT-DONE]',2*len(a.test_kernels),'rows; next run matched U50 HLS synthesis, not old Stage1 U50 results')
    finally:code.close()


def compare_main(a):
    import csv,statistics
    result={}
    for arm in ('zero_shot','few_shot'):
        p=a.results_root/arm/'summary'/'results.jsonl'
        if not p.is_file():raise FileNotFoundError(p)
        rr=lines(p)
        if len(rr)!=4:raise ValueError('Expected four HLS test results per model')
        v={}
        for row in rr:
            name=row['kernel']
            if name not in TEST_KERNELS or name in v:raise ValueError('Unexpected or duplicate kernel: '+str(name))
            ok=(row['status']=='success' and row.get('latency_cycles') is not None
                and row.get('budget_feasible') is True
                and row.get('estimated_clock_ns') is not None
                and float(row['estimated_clock_ns'])<=10.0)
            v[name]={'status':row['status'],'latency_cycles':row['latency_cycles'],
                'estimated_clock_ns':row.get('estimated_clock_ns'),
                'budget_feasible':row.get('budget_feasible'),
                'qualified':ok}
        result[arm]=v
    summary=[];speedups=[]
    for k in TEST_KERNELS:
        z=result['zero_shot'][k];f=result['few_shot'][k]
        speedup=z['latency_cycles']/f['latency_cycles'] if z['qualified'] and f['qualified'] else None
        if speedup is not None:speedups.append(speedup)
        summary.append({'kernel':k,'zero':z,'adapted':f,'speedup':speedup,
                        'case':'paired_qualified' if speedup is not None else 'not_comparable_or_feasibility_transition'})
    report={'schema':'salt-u50-external-physical-comparison-v1',
            'cases':summary,'paired_qualified':len(speedups),
            'geometric_mean_speedup':math.exp(sum(math.log(x) for x in speedups)/len(speedups)) if speedups else None,
            'caution':'Only matched, timing-qualified HLS estimates counted; do not claim improved generalization on missing or failed tests'}
    dump(a.out,report)
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode',required=True)
    for mode in ('audit','train','predict'):
        p=sub.add_parser(mode)
        p.add_argument('--repo',type=Path,required=True)
        p.add_argument('--deployment',type=Path,required=True)
        p.add_argument('--memory-dir',type=Path,required=True)
        p.add_argument('--out',type=Path,required=True)
        p.add_argument('--shots',type=int,default=8,choices=SUPPORTED_SHOTS)
        p.add_argument('--seed',type=int,default=123)
        p.add_argument('--parent',type=Path,required=True)
        if mode in ('train','predict'):
            p.add_argument('--film-checkpoint',type=Path,required=True)
        if mode=='train':
            p.add_argument('--eval-memory-dir',type=Path)
            p.add_argument('--steps',type=int,default=20)
            p.add_argument('--eval-interval',type=int,default=5)
            p.add_argument('--lr',type=float,default=.02)
            p.add_argument('--rank-lambda',type=float,default=.1)
        if mode=='predict':
            p.add_argument('--eval-sources-root',type=Path,required=True)
            p.add_argument('--eval-memory-dir',type=Path,required=True)
            p.add_argument('--calibration',type=Path,required=True)
            p.add_argument('--test-kernels',nargs='+',default=list(TEST_KERNELS),choices=TEST_KERNELS)
    p=sub.add_parser('compare')
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    for k in ('repo','deployment','memory_dir','parent','film_checkpoint','eval_memory_dir','eval_sources_root','calibration','results_root'):
        if hasattr(a,k) and getattr(a,k) is not None:setattr(a,k,getattr(a,k).resolve())
    if a.mode=='audit':audit(a)
    elif a.mode=='train':train_main(a)
    elif a.mode=='predict':predict_main(a)
    else:compare_main(a)

if __name__=='__main__':main()
