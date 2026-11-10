"""Measured budget winner transitions, separate from arbitrary pair reversals."""
import collections,hashlib,json,math
from pathlib import Path
from experiments.stage1_target_film.train import sha,write
# Reuse the source/domain qualification exactly; execute only its preparation.
source=Path(__file__).with_name('qualify_clock.py').read_text().split('contrasts=[];counts=')[0];source=source.replace('if dest.exists():raise FileExistsError(dest)','').replace('dest.mkdir(parents=True)','')
ns={'__name__':'qualified_pool_reuse'};exec(compile(source,'qualified_clock_pool_preparation','exec'),ns);pools=ns['pools'];templates=ns['templates'];production=ns['production'];root=ns['ROOT'];dest=root/'enriched_budget_candidate'
if dest.exists():raise FileExistsError(dest)
dest.mkdir(parents=True);groups=[];counts=collections.Counter();coverage=collections.defaultdict(set)
for context,pool in pools.items():
 fold,kernel,device,clock=context;family=templates[kernel]['_family'];fast=min(pool.values(),key=lambda v:(v['physical'][0],v['identity']))
 for fraction in (.2,.25,.5):
  fits=lambda r:all(float(v)/float(c)+.01<=fraction for v,c in zip(r['physical'][1:],r['capacities']))
  excluded=lambda r:any(max(0,float(v)/float(c)-.01)>fraction for v,c in zip(r['physical'][1:],r['capacities']))
  eligible=[r for r in pool.values() if fits(r)]
  if not eligible:counts[fold+'_no_conservatively_feasible_tight_design']+=1;continue
  tight=min(eligible,key=lambda v:(v['physical'][0],v['identity']))
  if fits(fast):
   options=[r for r in eligible if r['physical'][0]>=1.1*fast['physical'][0]]
   if not options:counts[fold+'_stay_no_separated_rejection']+=1;continue
   slow=min(options,key=lambda v:v['identity']);status='stay'
  elif excluded(fast) and tight['physical'][0]>=1.1*fast['physical'][0]:slow=tight;status='switch'
  else:counts[fold+'_ambiguous_or_near_tie']+=1;continue
  counts[fold+'_'+status]+=1;coverage[(fold,status)].add(family);gid=hashlib.sha256(repr((context,fast['identity'],slow['identity'],fraction)).encode()).hexdigest();groups.append({'id':gid,'context':context,'family':family,'status':status,'fraction':fraction,'fast':fast,'slow':slow})
selected=[]
for fold in ('train','val'):
 for family in sorted({g['family'] for g in groups if g['context'][0]==fold}):
  for status in ('switch','stay'):
   unique={}
   for g in sorted((g for g in groups if g['context'][0]==fold and g['family']==family and g['status']==status),key=lambda g:g['id']):unique.setdefault((g['context'][1],g['context'][2],g['context'][3],g['fast']['identity'],g['slow']['identity']),g)
   selected.extend(list(unique.values())[:4])
for fold in ('train','val'):
 examples=[]
 for g in selected:
  if g['context'][0]!=fold:continue
  _,kernel,device,clock=g['context'];source_row=templates[kernel];plan=production.extract_ordered_lhs_plan(source_row['input']);serialize=lambda r:'\n'.join(f'{lhs} = {r["assignment"][lhs.upper()]}' for _,lhs in plan)
  for budget in (g['fraction'],1.):
   chosen=g['slow'] if g['status']=='switch' and budget<1 else g['fast'];rejected=g['fast'] if chosen is g['slow'] else g['slow'];row=dict(source_row,device=device,clock_period=clock,selected_clock_period=clock,frequency_mode='specified',resource_budget_frac=budget,resource_budget_id='proxy-qualified-'+str(budget),obj_mode='PARETO_LATENCY');row.update(dict(zip(('avail_bram','avail_dsp','avail_ff','avail_lut'),(int(round(float(c)*budget)) for c in chosen['capacities']))));row.update(dict.fromkeys(('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut'),budget));examples.append({'group_id':g['id'],'family':g['family'],'status':g['status'],'row':row,'preferred_target':serialize(chosen),'rejected_target':serialize(rejected),'sft_target':serialize(chosen),'preferred_measurement':chosen,'rejected_measurement':rejected,'qualification':'Conservative +/-1pp sensitivity assumption; not exact counts or authenticated rounding contract. Tight rejection may be infeasible; keep distinct from feasible-vs-feasible rank.','source_sha256':hashlib.sha256(source_row['input'].encode()).hexdigest()})
 (dest/(fold+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in examples))
report={'schema':'qualified-budget-contrast-candidate-v1','status':'EXPLORATORY_NOT_TRAINED','counts':dict(counts),'family_coverage':{'/'.join(k):sorted(v) for k,v in coverage.items()},'selected_physical_groups':dict(collections.Counter((g['context'][0]+'_'+g['status']) for g in selected)),'selected_group_ids':[g['id'] for g in selected],'physical_pair_count':len({(g['context'],g['fast']['identity'],g['slow']['identity']) for g in groups}),'derived_budget_views':len(groups),'file_sha256':{f:sha(dest/(f+'.jsonl')) for f in ('train','val')},'parent_adapter_sha256':'28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164','parent_registry_sha256':sha(ns['regpath']),'source_pool_bank_sha256':sha(ns['bankpath']),'qualification':'Source/tool/domain qualified complete measured designs; robustness conditional on +/-1pp assumption; no exact counts or achieved timing. Candidate bank only.','split_policy':'Original family membership; val examples evaluation-only; GAN and locked test unopened.'};write(dest/'manifest.json',report);write(Path('experiments/stage1_target_film/diagnostics_v2/BUDGET_BANK_MANIFEST.json'),report);print(json.dumps(report,indent=2))
