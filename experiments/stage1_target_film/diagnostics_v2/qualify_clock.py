"""Qualify a bounded matched clock-switch/stay candidate bank without training."""
import collections,hashlib,importlib,itertools,json,math,sys
from pathlib import Path
from LLM_branch.common import directive_domains as domains,mailohls_contract as mc
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.train import train_SFT_xattn_new as production
from experiments.stage1_target_film.train import sha,write
ROOT=Path('artifacts/stage1_target_film/diagnostics_v2');dest=ROOT/'enriched_clock_candidate'
if dest.exists():raise FileExistsError(dest)
dest.mkdir(parents=True)
bankpath=Path('/home/elvouvali/MailoHLS-final-architecture-20261109/artifacts/final_architecture_20261109/reaudit/non_test_records.json');bank=json.loads(bankpath.read_text());audit=json.loads((ROOT/'bank_v2/TARGET_CONTRAST_AUDIT.json').read_text());excluded_names={''.join(c for c in x.lower() if c.isalnum()) for x in ('rodinia_cfd_flux_0_baseline_0','rodinia_lavaMD_0_baseline','rodinia_lavaMD_1_tiling_0','rodinia_lavaMD_1_tiling_1','rodinia_lavaMD_2_pipeline_0','rodinia_pathfinder_0_baseline_0')};norm=lambda x:''.join(c for c in x.lower() if c.isalnum())
pilotpath=Path('/home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/audit/stage2_pilot.jsonl');templates={r['kernel_name']:r for r in map(json.loads,pilotpath.read_text().splitlines())};regpath=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run/directive_domain_registry.json');registry=inference.load_directive_domain_registry(str(regpath));package=Path('/home/elvouvali/MailoHLS-final-architecture-20261109/artifacts/final_architecture_20261109/references/salt_qualification_contract');sys.path.insert(0,str(package.parent));tool=importlib.import_module(package.name+'.source_tool_policy');policy=tool.load_policy(package/'policy.json');actions={k:domains.load_source_actions(k,Path('Data/ApplicationDataset')/k) for k in templates};valid={};exclude=collections.Counter();pools=collections.defaultdict(dict)
for r in bank['rows']:
 if r['kernel'] not in templates or norm(r['kernel']) in excluded_names or norm(r['kernel']).startswith('spcl'):exclude['outside_compatible_source_cohort']+=1;continue
 key=(r['kernel'],r['config_hash']);a={k.upper():str(v) for k,v in r['assignment'].items()}
 if key not in valid:
  try:
   mc.validate_directive_assignments(a);tool.validate_source_assignment(a,actions[r['kernel']]);tool.validate_complete_assignment(a,actions[r['kernel']],policy);prior={}
   for _,lhs in production.extract_ordered_lhs_plan(templates[r['kernel']]['input']):
    legal=mc.filter_semantic_candidates(lhs,inference.get_rhs_candidates_for_lhs(r['kernel'],lhs,registry),prior,registry[inference.normalize_kname(r['kernel'])]);assert a[lhs.upper()] in legal;prior[lhs.upper()]=a[lhs.upper()]
   assert set(a)==set(prior);valid[key]=True
  except (ValueError,KeyError,AssertionError):valid[key]=False
 if not valid[key]:exclude['source_or_domain_illegal']+=1;continue
 q=r['physical'];caps=r['capacities']
 if not all(v is not None and math.isfinite(v) and v>=0 for v in q) or q[0]<=0 or all(v==0 for v in q[1:]):exclude['missing_or_unobservable_qor']+=1;continue
 if any(v/c+.01>1 for v,c in zip(q[1:],caps)):exclude['not_conservatively_full_budget_proxy_feasible']+=1;continue
 h=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest();pools[(r['split'],r['kernel'],r['device'],float(r['clock']))][h]=dict(r,assignment=a,identity=h)
contrasts=[];counts=collections.Counter()
for l,r in itertools.combinations(sorted(pools),2):
 if l[:3]!=r[:3] or l[3]==r[3]:continue
 common=set(pools[l])&set(pools[r]);bestl=min(v['physical'][0] for v in pools[l].values());bestr=min(v['physical'][0] for v in pools[r].values())
 for x,y in itertools.combinations(sorted(common),2):
  la,lb,ra,rb=[v['physical'][0] for v in (pools[l][x],pools[l][y],pools[r][x],pools[r][y])]
  if min(max(la,lb)/min(la,lb),max(ra,rb)/min(ra,rb))<1.1:counts[l[0]+'_near_tie']+=1;continue
  status='switch' if (la<lb)!=(ra<rb) else 'stay';counts[l[0]+'_'+status]+=1;family=templates[l[1]]['_family'];near=min(la,lb)<=1.01*bestl and min(ra,rb)<=1.01*bestr
  counts[l[0]+'_'+status+'_near_optimal_both_contexts']+=near
  contrasts.append({'split':l[0],'kernel':l[1],'family':family,'status':status,'near_optimal_both_contexts':near,'left':l,'right':r,'a':x,'b':y,'id':hashlib.sha256(repr((l,r,x,y)).encode()).hexdigest()})
selected=[]
for fold in ('train','val'):
 for family in sorted({v['family'] for v in contrasts if v['split']==fold}):
  for status in ('switch','stay'):
   choices=sorted([v for v in contrasts if v['split']==fold and v['family']==family and v['status']==status],key=lambda v:(not v['near_optimal_both_contexts'],v['id']))
   selected.extend(choices[:4])
for fold in ('train','val'):
 examples=[]
 for group in selected:
  if group['split']!=fold:continue
  for side in ('left','right'):
   context=tuple(group[side]);pool=pools[context];aa,bb=pool[group['a']],pool[group['b']];preferred,rejected=sorted((aa,bb),key=lambda v:v['physical'][0]);source=templates[context[1]];plan=production.extract_ordered_lhs_plan(source['input']);serialize=lambda a:'\n'.join(f'{lhs} = {a[lhs.upper()]}' for _,lhs in plan);sft=min(pool.values(),key=lambda v:(v['physical'][0],v['identity']));row=dict(source,device=context[2],clock_period=context[3],selected_clock_period=context[3],frequency_mode='specified',resource_budget_frac=1.,resource_budget_id='full',obj_mode='PARETO_LATENCY');row.update(dict(zip(('avail_bram','avail_dsp','avail_ff','avail_lut'),map(int,preferred['capacities']))));row.update(dict.fromkeys(('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut'),1.))
   examples.append({'contrast':group,'physical_context':context,'row':row,'preferred_target':serialize(preferred['assignment']),'rejected_target':serialize(rejected['assignment']),'sft_target':serialize(sft['assignment']),'sft_reference_note':'Separate complete measured best-in-pool; do not SFT-impose a dominated ranking-pair winner.','preferred_measurement':preferred,'rejected_measurement':rejected,'source_sha256':hashlib.sha256(source['input'].encode()).hexdigest(),'resource_evidence':'Conservative percentage-proxy interval; not exact counts or timing qualification'})
 (dest/(fold+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in examples))
report={'schema':'qualified-clock-contrast-candidate-v1','status':'EXPLORATORY_NOT_TRAINED','counts':dict(counts),'exclusions':dict(exclude),'selected_groups':selected,'file_sha256':{f:sha(dest/(f+'.jsonl')) for f in ('train','val')},'inputs_sha256':{str(p):sha(p) for p in (bankpath,pilotpath,regpath)},'canonical_registry_commit':audit['canonical_registry_commit'],'resource_qualification':'Conditional percentage bounds; no exact counts or timing; NOT RETRAIN_RECOMMENDED','selection':'Up to4 switch and4 stay groups per independent family; complete same-candidate pairs, source/domain legal; prefer groups near-optimal in both contexts; split immutable; validation not used to construct train rows.','locked_test_used':False,'GAN_used':False,'original_bank_modified':False};write(dest/'manifest.json',report);write(Path('experiments/stage1_target_film/diagnostics_v2/ENRICHED_BANK_MANIFEST.json'),report);print(json.dumps({k:v for k,v in report.items() if k not in ('selected_groups',)},indent=2))
