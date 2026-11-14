#!/usr/bin/env python3
"""Closed-loop budgeted latency refinement for the selected SALT FiLM/GMT Stage2-300.

Uses final300 predictions, exact frozen historical memory, original complete source,
repository's action/domain proposal API, and the authenticated Coroni batch runner.
No training, no gradient, no ROI estimate in place of HLS. Durable resume.
The 'model' policy also calls the exact FiLM/GMT frozen model as a soft prior.
"""
import argparse, collections, fcntl, hashlib, json, math, os, re, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

RES=('BRAM_18K','DSP','FF','LUT')

def hfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def read(path):return json.loads(Path(path).read_text())

def run(argv, *, stdout=None):
    print('[EXEC]', ' '.join(map(str,argv)),flush=True)
    return subprocess.run(list(map(str,argv)),stdout=stdout,check=True)

def prepare(a):
    # Delay repo imports until the user has selected the exact worktree.
    sys.path.insert(0,str(a.repo))
    from experiments.refinement import configurations as c
    from experiments.refinement_search import policies
    import torch
    candidates=[json.loads(s) for s in a.baseline_jsonl.read_text().splitlines() if s.strip()]
    rows=[row for row in candidates if row.get('kernel')==a.kernel and row.get('device') in ('xczu7ev-ffvc1156-2-e','zcu104') and abs(float(row.get('clock_period_ns',0))-a.clock_ns)<1e-5 and abs(float(row.get('budget_percent',0))-a.budget_pct)<1e-5]
    if len(rows)!=1:raise ValueError(f'Expected exactly one {a.kernel}/zcu104/{a.clock_ns}ns/{a.budget_pct}% final300 prediction; found {len(rows)}')
    row=rows[0]
    if row.get('status')!='inference_only_not_synthesized' or row.get('checkpoint_step')!=300:
        raise ValueError('Baseline is not a genuine Stage2 checkpoint-300 prediction')
    checkpoint=a.deployment/'runs/campaign_s123_v1/final/checkpoint-0300'
    receipt=read(checkpoint/'receipt.json')
    if row.get('checkpoint_sha256')!=receipt.get('structural_state_sha256') or hfile(checkpoint/'structural_xattn.pt')!=receipt['structural_state_sha256']:
        raise ValueError('Wrong final300 checkpoint/weights')
    protocol=read(checkpoint.parent/'protocol.json')
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True,default=str).encode()).hexdigest()
    if digest!=receipt['protocol_sha256']:raise ValueError('Saved final300 training protocol hash mismatch')
    if protocol['model']!='FiLM Arm B checkpoint-0600 frozen + historical GMT B frozen + gated Xattn':
        raise ValueError('Not the selected FiLM-GMT model')
    folder=a.sources_root/a.kernel
    from experiments.refinement import synthesis as hls
    if not (folder/'kernel_info.txt').is_file():raise FileNotFoundError('Missing canonical action metadata')
    from LLM_branch.common import directive_domains as domains
    actions=domains.load_source_actions(a.kernel, a.sources_root)
    if len(actions)>64:raise ValueError('Too many action slots')
    native=c.legacy_domains(actions,folder)
    initial=c.from_paper(row,actions)
    c.policy_record(initial,initial,actions,native,'tcad')
    original_source=next((p for p in [folder/x for x in ('source.cpp','source.c','source.cc','source.cxx')] if p.is_file()),None)
    if original_source is None:raise RuntimeError('Canonical source.cpp/c required (same selection as Coroni runner)')
    if hfile(original_source)!=row['source_sha256']:raise ValueError('Baseline source differs from canonical HLS source')
    manifest=read(a.memory_dir/'memory_manifest.json')
    if hfile(a.memory_dir/'memory_manifest.json')!=row['memory_manifest_sha256']:
        raise ValueError('Initial model output memory does not match selected independent-GMT bank')
    if manifest['kind']!='B_GMT':raise ValueError('Not historical B-GMT memory')
    def norm(value):return re.sub(r'[-_\s]+','',value).lower()
    names=[name for name in manifest['files'] if name.endswith('.memory.pt') and norm(name[:-10])==norm(a.kernel)]
    if len(names)!=1:raise ValueError(f'Independent Kalman memory manifest is missing or ambiguous: {names}')
    memfile=names[0]
    if hfile(a.memory_dir/memfile)!=manifest['files'][memfile]:raise ValueError('Compiler memory hash mismatch')
    data=torch.load(a.memory_dir/memfile,map_location='cpu',weights_only=False)
    if data['gnn_checkpoint_sha256']!=manifest['gnn_checkpoint_sha256'] or data['gmt_checkpoint_sha256']!=manifest['gmt_checkpoint_sha256']:
        raise ValueError('Memory encoder identity mismatch')
    rel=data['action_relation_mask'].bool();active=data['action_slot_mask'].bool()
    if rel.ndim!=2 or rel.shape[0]<64 or rel.shape[1]<64 or active.numel()!=64:
        raise ValueError('Unexpected historical memory schema')
    labels=sorted(actions, key=lambda x:int(x[1:]))
    pairs=[]
    for i,x in enumerate(labels):
        for y in labels[i+1:]:
            ix=int(x[1:])-1;iy=int(y[1:])-1
            if bool(active[ix]) and bool(active[iy]) and (bool(rel[ix,iy]) or bool(rel[iy,ix])):
                # Historical route eligibility, not a proved physical bottleneck or memory dependence.
                pairs.append((x,y))
    limits={k:int(math.floor(v*a.budget_pct/100.)) for k,v in {
        'BRAM_18K':624,'DSP':1728,'FF':460800,'LUT':230400}.items()}
    identity={'schema':'final300-closed-loop-v1','checkpoint_structural_sha256':row['checkpoint_sha256'],
        'baseline_assignment_sha256':c.signature(initial),'baseline_file_sha256':hfile(a.baseline_jsonl),
        'source_sha256':hfile(original_source),'metadata_sha256':hfile(folder/'kernel_info.txt'),
        'memory_manifest_sha256':hfile(a.memory_dir/'memory_manifest.json'),'memory_file_sha256':hfile(a.memory_dir/memfile),
        'runner_sha256':hfile(Path(__file__).with_name('coroni_batch.py')),
        'kernel':a.kernel,'part':'xczu7ev-ffvc1156-2-e','clock_ns':a.clock_ns,'budget_percent':a.budget_pct,
        'budget':a.max_evaluations,'policy':a.policy,'seed':123,'score_top':a.score_top,
        'limits':limits,'repo_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()}
    return {'c':c,'policies':policies,'hls':hls,'actions':actions,'native':native,'initial':initial,'baseline_row':row,
        'pairs':pairs,'identity':identity,'folder':folder}

def measured_ok(m):
    return m.get('status')=='success' and m.get('latency_cycles') is not None and m.get('budget_feasible') is True and m.get('timing_qualified') is True

def extract_ii(bundle,results):
    """Only consume explicit Lk matches in the CSynth XML; never invent a bottleneck."""
    try:
        manifest=read(bundle/'manifest.json');job=manifest['jobs'][0]
        p=results/job['job_id']/'hls_project/solution1/syn/report'/(job['top']+'_csynth.xml')
        root=ET.parse(p).getroot()
    except (FileNotFoundError,ET.ParseError,KeyError,IndexError):return {}
    found={}
    for node in root.findall('.//SummaryOfLoopLatency//Loop'):
        data=' '.join((node.findtext(tag) or '') for tag in ('Name','LoopName','name'))
        hit=re.search(r'(?<![A-Za-z0-9])L([1-9][0-9]*)(?![A-Za-z0-9])',data)
        if not hit:continue
        number=None
        for tag in ('PipelineII','InitiationInterval','IterationInterval','II'):
            try:number=int((node.findtext(tag) or '').strip());break
            except ValueError:pass
        if number is not None:found['L'+hit.group(1)]={'ii':number,'xml_loop':data}
    return found

def histories(out,c):
    items=[]
    for step in range(1,100):
        p=out/f'step_{step:02d}'/'query.json';m=out/f'step_{step:02d}'/'measurement.json'
        if not p.exists():break
        if not m.exists():break
        query=read(p);result=read(m)
        items.append({**query['proposal'],'measurement':result,'signature':query['signature']})
    return items

def make_pair(problem,reference,a,b,reason):
    c=problem['c'];changes={}
    for p in (a,b):
        for k in p['reference_changed_fields']:changes[k]=p['assignment'][k]
    candidate={**reference,**changes}
    try:policy=c.policy_record(candidate,problem['initial'],problem['actions'],problem['native'],'tcad')
    except (ValueError,KeyError):return None
    if candidate==reference:return None
    actions,fields=c.changed(reference,candidate,problem['actions'])
    if len(actions)!=2:return None
    return {'assignment':candidate,'signature':c.signature(candidate),
        'reference_signature':c.signature(reference),'reference_changed_actions':actions,'reference_changed_fields':fields,
        'proposal_type':'structural_pair','selection_reason':reason,'domain_classification':policy,
        'memory_evidence':'historical_GNN_action_route_eligibility_only'}

def propose(problem,history,policy):
    c=problem['c'];p=problem['policies'];base=problem['initial']
    feasible=[r for r in history if measured_ok(r['measurement'])]
    incumbent=min(feasible,key=lambda z:z['measurement']['latency_cycles'])['assignment'] if feasible else base
    observed={r['signature'] for r in history}
    singles=p.single_proposals({'baseline':base,'actions':problem['actions'],'native_domains':problem['native'],'inactive_actions':()},incumbent,123)
    singles=[v for v in singles if v['signature'] not in observed and c.admissible(v['assignment'],problem['actions'],problem['native'],'legacy')]
    if policy=='coordinate':return singles,incumbent
    by_action=collections.defaultdict(list)
    for x in singles:by_action[x['reference_changed_actions'][0]].append(x)
    pairs=[]
    for left,right in problem['pairs']:
        xs=by_action[left][:2];ys=by_action[right][:2]
        for x in xs:
            for y in ys:
                row=make_pair(problem,incumbent,x,y,f'Compiler-routing related {left}/{right}: complementary experiment, physical effect unknown')
                if row and row['signature'] not in observed and c.admissible(row['assignment'],problem['actions'],problem['native'],'legacy'):
                    pairs.append(row)
    # Compose two independently *measured* beneficial single-action edits at a
    # common reference. Do not assume additivity or combine stale measurements.
    from itertools import combinations
    by_ref=collections.defaultdict(list)
    observed_rows={r['signature']:r for r in history}
    for measured in history:
        if measured['proposal_type']!='coordinate' or not measured_ok(measured['measurement']):continue
        prior=observed_rows.get(measured['reference_signature'])
        if (prior and measured_ok(prior['measurement']) and
            measured['measurement']['latency_cycles']<prior['measurement']['latency_cycles']):
            by_ref[measured['reference_signature']].append(measured)
    compositions=[]
    for refsig,rows in by_ref.items():
        reference=observed_rows[refsig]['assignment']
        for left,right in combinations(rows,2):
            if set(left['reference_changed_actions']) & set(right['reference_changed_actions']):continue
            # Never roll back unrelated improvements in current incumbent.
            if any(incumbent[field]!=reference[field] for field in reference if field not in
                   (set(left['reference_changed_fields']) | set(right['reference_changed_fields']))):continue
            combined=make_pair(problem,reference,left,right,'Two independently measured beneficial edits at one shared reference')
            if combined and combined['signature'] not in observed and c.admissible(combined['assignment'],problem['actions'],problem['native'],'legacy'):
                combined['proposal_type']='measured_composition'
                compositions.append(combined)
    seen=set();all_candidates=[]
    for row in compositions+singles+pairs:
        if row['signature'] not in seen:seen.add(row['signature']);all_candidates.append(row)
    return all_candidates,incumbent

def heuristic(row,problem,history,incumbent):
    """Transparent feedback-only heuristic; model prior is a bounded tie-breaker."""
    kind=row['proposal_type'];actions=row['reference_changed_actions'];last=history[-1] if history else None
    evidence=1.0 if kind=='structural_pair' else 0.
    # Correlation of edits is a reason for exploration, not a QoR prediction.
    past=collections.defaultdict(list)
    for v in history:
        if v['proposal_type']=='baseline':continue
        m=v['measurement']
        if measured_ok(m):
            for k in v['reference_changed_actions']:past[k].append(float(m['latency_cycles']))
    score=0.05*evidence + (0.55 if kind=='measured_composition' else 0.)
    baseline=next((x['measurement'] for x in history if x['proposal_type']=='baseline'),{})
    if history and measured_ok(baseline):
        reference=float(baseline['latency_cycles'])
        for action in actions:
            if past[action] and min(past[action])<reference:score+=0.5
    if last:
        m=last['measurement']
        ii=m.get('loop_ii',{})
        for key in actions:
            if key in ii and ii[key]['ii']>1:score+=0.65
        # Move away from resource-heavy changes if last synthesis exceeded the budget.
        if m.get('status')=='success' and not m.get('budget_feasible',True):
            decreases=0;increases=0
            for key in row['reference_changed_fields']:
                if '_UNROLL_' in key or '_ARRAY_F_' in key:
                    try:
                        old=int(incumbent[key]);new=int(row['assignment'][key])
                        decreases+=new<old;increases+=new>old
                    except (ValueError,KeyError):pass
            score+=0.75*decreases-0.75*increases
    if kind=='structural_pair' and len(history)<3:score-=0.85
    # Avoid thrashing many unrelated actions when a measured improvement is found.
    if last and measured_ok(last['measurement']) and measured_ok(baseline):
        if last['measurement']['latency_cycles']<baseline['latency_cycles']:
            if set(last['reference_changed_actions'])&set(actions):score+=0.18
    return score

def choose(problem,history,a,out):
    c=problem['c'];slate,incumbent=propose(problem,history,a.policy)
    if not slate:return None
    scores={}
    # Limit the GPU proposals to a small promising slate, after feedback-based filtering.
    # Reserve one compiler-related experiment at step 4, if the authenticated
    # historical eligibility mask supports a legal two-action proposal.
    step=len(history)+1
    if step==4 and a.policy!='coordinate':
        structural=[p for p in slate if p['proposal_type'] in ('structural_pair','measured_composition')]
        if structural:slate=structural
    ranked=[p for _,p in sorted(enumerate(slate),key=lambda item:(-heuristic(item[1],problem,history,incumbent),item[0]))]
    if a.policy=='model':
        top=ranked[:a.score_top]
        payload={'candidates':[{'signature':p['signature'],'reference':incumbent,'assignment':p['assignment']} for p in top]}
        write(out/'scoring_slate.json',payload)
        command=['flock','-w','14400',str(a.repo/'artifacts/.salt_final_v100s_global.lock'),str(a.python),
            str(a.scorer),'--kernel',a.kernel,'--device','zcu104','--clock-ns',str(a.clock_ns),'--budget',str(a.budget_pct),
            '--checkpoint-step','300','--deployment',str(a.deployment),'--sources-root',str(a.sources_root),
            '--memory-dir',str(a.memory_dir),'--score-proposals',str(out/'scoring_slate.json'),
            '--score-out',str(out/'scored.json'), '--output',str(out/'model_scorer_unused.jsonl')]
        env=os.environ.copy();env.update({'SALT_GPU_UUID':a.gpu_uuid,'SALT_PARENT_STAGE1':str(a.parent_stage1),'PYTHONPATH':str(a.repo)+os.pathsep+str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH',''), 'TOKENIZERS_PARALLELISM':'false'})
        print('[MODEL] Frozen checkpoint-300 causal scoring; no gradients',flush=True)
        subprocess.run(command,env=env,check=True)
        scores={r['signature']:r['delta'] for r in read(out/'scored.json')['candidates'] if r['delta'] is not None}
        # Strong model preference is only a bounded tie-breaker, not a pseudo-HLS reward.
        ranked=sorted(top,key=lambda p:(-heuristic(p,problem,history,incumbent)-0.15*max(-2.,min(2.,scores.get(p['signature'],0.))),p['signature']))+ranked[a.score_top:]
    selected=ranked[0]
    return {**selected,'feedback_priority':heuristic(selected,problem,history,incumbent),
        'frozen_model_local_delta':scores.get(selected['signature']),'candidate_pool':len(slate)}

def as_directives(candidate,actions):
    output=[]
    for label,action in actions.items():
        if action.kind=='loop':output.append({'action_id':label,'kind':'loop',
            'pipeline_ii':int(candidate[f'AUTO{{_PIPE_{label}}}']),
            'unroll_factor':int(candidate[f'AUTO{{_UNROLL_{label}}}'])})
        else:output.append({'action_id':label,'kind':'array',
            'partition_type':candidate[f'AUTO{{_ARRAY_T_{label}}}'],
            'partition_factor':int(candidate[f'AUTO{{_ARRAY_F_{label}}}']),
            'partition_dim':int(candidate[f'AUTO{{_ARRAY_D_{label}}}'])})
    return output

def measure_one(problem,proposal,a,stepdir):
    c=problem['c'];batch=Path(__file__).with_name('coroni_batch.py')
    row={**problem['baseline_row'],'directives':as_directives(proposal['assignment'],problem['actions']),
        'status':'inference_only_not_synthesized' if proposal['proposal_type']=='baseline' else 'refinement_proposal_not_synthesized',
        'refinement_origin':'original_checkpoint300' if proposal['proposal_type']=='baseline' else 'adaptive_search_not_model_generated',
        'refinement_candidate_signature':proposal['signature']}
    prediction=stepdir/'proposal.jsonl'
    if prediction.exists():
        old=json.loads(prediction.read_text())
        if old!=row:raise ValueError('Existing proposal changed on resume')
    else:prediction.write_text(json.dumps(row,sort_keys=True)+'\n')
    bundle=stepdir/'bundle';results=stepdir/'results';summary=stepdir/'summary'
    if not bundle.exists():run([a.python,batch,'prepare','--predictions',prediction,'--sources-root',a.sources_root,
        '--out',bundle,'--device','zcu104','--clock-ns',a.clock_ns,'--budget-percent',a.budget_pct])
    else:
        from importlib import util
        spec=util.spec_from_file_location('ref_batch',batch);mod=util.module_from_spec(spec);spec.loader.exec_module(mod)
        m=mod.verify_bundle(bundle)
        if len(m['jobs'])!=1 or m['jobs'][0]['assignment']!=proposal['assignment']:raise ValueError('Existing HLS bundle is not same proposal')
    ident=hashlib.sha256(json.dumps(problem['identity'],sort_keys=True).encode()).hexdigest()[:16]
    remote=a.remote_root+'/'+a.policy+'/'+ident+'/'+stepdir.name
    if not re.fullmatch('[A-Za-z0-9_/-]+',remote) or remote.startswith('/') or '..' in remote:raise ValueError('Unsafe Coroni relative directory')
    ssh=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=60',a.coroni]
    run(ssh+['mkdir -p '+remote+'/bundle '+remote+'/results'])
    run(['rsync','-a',str(bundle)+'/',a.coroni+':'+remote+'/bundle/'])
    # Cooperate with existing Vitis jobs; remote lock prevents two copies of this script.
    # Prefer POSIX shell quoting rather than Python's Windows-style list2cmdline.
    import shlex
    inner="while pgrep -u $(id -un) -x vitis_hls >/dev/null; do sleep 20; done; cd " + shlex.quote(remote+'/bundle')+" && sha256sum -c SHA256SUMS >/dev/null && python3 coroni_batch.py run --bundle . --results ../results --tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls --timeout-seconds "+str(a.timeout)
    command='flock -w 14400 ~/salt_refine_final300_coroni.lock sh -c '+shlex.quote(inner)
    run(ssh+[command])
    results.mkdir(exist_ok=True)
    run(['rsync','-a',a.coroni+':'+remote+'/results/',str(results)+'/'])
    if not (summary/'results.jsonl').exists():run([a.python,batch,'collect','--bundle',bundle,'--results',results,'--summary',summary])
    rr=[json.loads(s) for s in (summary/'results.jsonl').read_text().splitlines() if s.strip()]
    if len(rr)!=1:raise ValueError('Expected one HLS result')
    m=rr[0];m['budget_feasible']=m.get('budget_feasible') if m['status'] in ('success','latency_undefined') else None
    m['timing_qualified']=bool(m['estimated_clock_ns'] is not None and math.isfinite(float(m['estimated_clock_ns'])) and float(m['estimated_clock_ns'])<=a.clock_ns) if m['status']=='success' else False
    m['loop_ii']=extract_ii(bundle,results)
    manifest=read(bundle/'manifest.json');recfile=results/manifest['jobs'][0]['job_id']/'run_receipt.json'
    if not recfile.is_file():raise ValueError('Missing authenticated Coroni completion receipt')
    m['hls_runtime_seconds']=read(recfile)['elapsed_seconds']
    m['candidate_signature']=proposal['signature']
    m['physical_query_accounted']=True  # May be an authenticated retry; do not mislabel as fresh.
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('audit','plan','score-smoke','run','report'),required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--baseline-jsonl',type=Path,required=True);p.add_argument('--sources-root',type=Path,required=True)
    p.add_argument('--memory-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kernel',default='Kalman');p.add_argument('--clock-ns',type=float,default=10.)
    p.add_argument('--budget-pct',type=float,default=100.)
    p.add_argument('--max-evaluations',type=int,default=8);p.add_argument('--limit',type=int,default=1)
    p.add_argument('--policy',choices=('coordinate','structural','model'),default='model')
    p.add_argument('--score-top',type=int,default=3)
    p.add_argument('--scorer',type=Path,default=Path(__file__).with_name('score_checkpoint300_liono.py'))
    p.add_argument('--parent-stage1',type=Path,default=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run'))
    p.add_argument('--python',type=Path,default=Path('/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python'))
    p.add_argument('--gpu-uuid',default='GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb')
    p.add_argument('--coroni',default='bonus24@coroni.microlab.ntua.gr')
    p.add_argument('--remote-root',default='salt_final300_refinement_20261114')
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    for name in ('repo','deployment','baseline_jsonl','sources_root','memory_dir','out','scorer','python','parent_stage1'):setattr(a,name,getattr(a,name).expanduser().resolve())
    if a.clock_ns not in (10.,5.,3.33) or a.kernel!='Kalman' or not 0<a.budget_pct<=100:raise ValueError('Initial pilot pins Kalman / ZCU104 and supported clock/budget')
    if not 1<=a.max_evaluations<=8 or not 1<=a.limit<=8 or not 1<=a.score_top<=8 or a.timeout<=0:raise ValueError('Invalid experiment budget/timeout')
    if not a.python.is_file():raise FileNotFoundError('Selected model Python unavailable')
    if a.policy=='model' and not a.scorer.is_file():raise FileNotFoundError('Generate isolated model scorer first via setup_scorer.py')
    problem=prepare(a);c=problem['c'];out=a.out
    print('[AUDIT] checkpoint-300 verified',problem['identity']['checkpoint_structural_sha256'])
    print('[AUDIT] historical compiler-memory routed action pairs',len(problem['pairs']))
    print('[AUDIT] no physical synthesis in audit or plan mode')
    if a.mode=='audit':
        # The real Coroni runner is also dry-run validated against the exact
        # selected Kalman baseline before any synthesis or GPU inference.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='salt_final300_preflight_') as tmp:
            tmp=Path(tmp)
            row=dict(problem['baseline_row']);row['directives']=as_directives(problem['initial'],problem['actions'])
            prediction=tmp/'one.jsonl';prediction.write_text(json.dumps(row)+'\n')
            batch=Path(__file__).with_name('coroni_batch.py')
            run([a.python,batch,'prepare','--predictions',prediction,'--sources-root',a.sources_root,
                 '--device','zcu104','--clock-ns',a.clock_ns,'--budget-percent',a.budget_pct,'--out',tmp/'bundle'])
            print('[AUDIT PASS] Authentic full-source HLS bundle created locally; no HLS invoked')
        return
    out.mkdir(parents=True,exist_ok=True)
    identityfile=out/'identity.json'
    if identityfile.exists():
        if read(identityfile)!=problem['identity']:raise ValueError('Refusing to reuse run with changed source/model/memory/protocol')
    else:write(identityfile,problem['identity'])
    if a.mode=='report':
        produce_report(out,problem);return
    if a.mode=='plan':
        hist=histories(out,c);slate,_=propose(problem,hist,a.policy)
        print(json.dumps({'history':len(hist),'next_candidates':[{'signature':x['signature'],'type':x['proposal_type'],'actions':x['reference_changed_actions']} for x in slate[:15]]},indent=2));return
    if a.mode=='score-smoke':
        if a.policy!='model':raise ValueError('Score smoke requires --policy model')
        hist=histories(out,c)
        slate,_=propose(problem,hist,a.policy)
        if not slate:raise RuntimeError('No legal proposal for scorer smoke')
        row=choose(problem,hist,a,out/'score_smoke')
        if row is None:raise RuntimeError('No scorer proposal')
        print('[SCORE SMOKE PASS] Frozen checkpoint loaded; selected proposal',row['signature']);return
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        executed=0
        for step in range(1,a.max_evaluations+1):
            stepdir=out/f'step_{step:02d}';queryfile=stepdir/'query.json';measurementfile=stepdir/'measurement.json'
            if measurementfile.exists():continue
            if executed>=a.limit:break
            stepdir.mkdir(exist_ok=True)
            hist=histories(out,c)
            if queryfile.exists():
                query=read(queryfile);proposal=query['proposal']
                print('[RESUME] existing reserved query',step,proposal['signature'])
            else:
                if step==1:
                    initial=problem['initial'];proposal={'assignment':initial,'signature':c.signature(initial),'reference_signature':c.signature(initial),
                        'reference_changed_actions':[],'reference_changed_fields':[],'proposal_type':'baseline',
                        'selection_reason':'Frozen FiLM-GMT checkpoint300 initial prediction','domain_classification':c.policy_record(initial,initial,problem['actions'],problem['native'],'tcad')}
                else:
                    proposal=choose(problem,hist,a,stepdir)
                    if proposal is None:
                        print('[STOP] legal neighborhood exhausted');break
                if step>1 and proposal['signature'] in {r['signature'] for r in hist}:raise ValueError('Candidate already queried')
                write(queryfile,{'step':step,'signature':proposal['signature'],'proposal':proposal,'paid_logical_query':True})
            # Query reserved durably BEFORE remote HLS is accessed.
            measurement=measure_one(problem,proposal,a,stepdir)
            write(measurementfile,measurement)
            executed+=1
            print('[MEASURED]',step,measurement['status'],'cycles',measurement.get('latency_cycles'),'feasible',measured_ok(measurement))
            produce_report(out,problem)
        produce_report(out,problem)

def produce_report(out,problem):
    hist=histories(out,problem['c']);successful=[r for r in hist if measured_ok(r['measurement'])]
    best=min(successful,key=lambda r:r['measurement']['latency_cycles']) if successful else None
    summary={'schema':'salt-final300-refinement-summary-v1','logical_evaluations_completed':len(hist),
        'logical_budget':problem['identity']['budget'],'selected_checkpoint_sha256':problem['identity']['checkpoint_structural_sha256'],
        'best_verified_feasible_cycles':best['measurement']['latency_cycles'] if best else None,
        'best_assignment':best['assignment'] if best else None,
        'total_hls_runtime_seconds':sum(float(r['measurement'].get('hls_runtime_seconds') or 0) for r in hist),
        'baseline_status':hist[0]['measurement']['status'] if hist else None,
        'baseline_cycles':hist[0]['measurement'].get('latency_cycles') if hist else None,
        'relative_latency_improvement_pct':(100*(hist[0]['measurement']['latency_cycles']-best['measurement']['latency_cycles'])/hist[0]['measurement']['latency_cycles'] if best and hist and measured_ok(hist[0]['measurement']) else None),
        'outcomes':[{'step':i+1,'signature':r['signature'],'type':r['proposal_type'],
            'status':r['measurement']['status'],'latency_cycles':r['measurement'].get('latency_cycles'),
            'budget_feasible':r['measurement'].get('budget_feasible'),'timing_qualified':r['measurement'].get('timing_qualified'),
            'changed_actions':r['reference_changed_actions'],
            'model_delta':r.get('frozen_model_local_delta'),
            'hls_runtime_seconds':r['measurement'].get('hls_runtime_seconds')} for i,r in enumerate(hist)]}
    write(out/'summary.json',summary)
    lines=['# Checkpoint-300 budgeted refinement','',f"Evaluations: {len(hist)}/{problem['identity']['budget']}",
       f"Best verified feasible HLS cycles: {summary['best_verified_feasible_cycles']}",
       f"Improvement versus *verified* baseline: {summary['relative_latency_improvement_pct']}%",'',
       '| Call | Proposal | Actions | Status | Cycles | Feasible |','|---:|---|---|---|---:|---|']
    for e in summary['outcomes']:
        lines.append(f"| {e['step']} | {e['type']} | {','.join(e['changed_actions']) or 'baseline'} | {e['status']} | {e['latency_cycles']} | {e['budget_feasible'] and e['timing_qualified']} |")
    lines+=['','Only HLS C-synthesis estimates; no C/RTL cosimulation or post-route timing qualification.','']
    (out/'report.md').write_text('\n'.join(lines))
    print('[REPORT]',out/'report.md')
if __name__=='__main__':main()
