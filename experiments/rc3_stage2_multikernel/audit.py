"""CPU-only audit of authenticated V4 train/validation rows against canonical RC3 graphs."""
from __future__ import annotations
import argparse, hashlib, json, re, tarfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=Path('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/train.jsonl')
DEFAULT_ARCHIVE=ROOT/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
EXPECTED_DATA_SHA='92ffd967652caa5e56a305625b72e03a9defa9c75c78a36cee09c259d10d0eba'
EXPECTED_ARCHIVE_SHA='feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73'
INCOMPLETE={'rodinia-hotspot-2-pipeline','rodinia-hotspot-3-unroll'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def graph_index(archive):
 out={}
 with tarfile.open(archive,'r:gz') as tar:
  for m in tar.getmembers():
   if not m.name.startswith('canonical/') or not m.name.endswith('.json'): continue
   obj=json.load(tar.extractfile(m)); md=obj['metadata']; k=m.name.rsplit('/',1)[-1][:-5].rsplit('-',1)[0]
   cov=md['action_point_coverage']['records']
   record={'archive_member':m.name,'graph_identity':md['graph_identity'],'source_sha256':md['provenance']['source_sha256'],
    'nodes':len(obj['nodes']),'edges':len(obj['edges']),
    'registered_actions':[x['action_id'] for x in cov if x.get('metadata_registered')],
    'matched_actions':[x['action_id'] for x in cov if x.get('metadata_registered') and x.get('compiler_matched')],
    'source_actions':[x['action_id'] for x in cov if x.get('source_labeled')]}
   if k in out:
    out[k].setdefault('archive_variants', []).append(record)
   else:
    out[k]=dict(record, archive_variants=[record])
 return out
def audit(data,archive):
 data,archive=Path(data),Path(archive); dh,gh=sha(data),sha(archive)
 if dh!=EXPECTED_DATA_SHA: raise ValueError('V4 training JSONL hash mismatch: '+dh)
 if gh!=EXPECTED_ARCHIVE_SHA: raise ValueError('RC3 archive hash mismatch: '+gh)
 rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
 graphs=graph_index(archive); grouped=defaultdict(list)
 for r in rows: grouped[r['kernel_name']].append(r)
 from LLM_branch.common import directive_domains as domains
 inv={}; usable=set()
 for k,rs in sorted(grouped.items()):
  r=rs[0]; g=graphs.get(k); sp=ROOT/r['source_file']; source_sha=sha(sp) if sp.is_file() else None
  if g and len(g.get('archive_variants', [])) != 1:
   raise ValueError(f'{k}: ambiguous RC3 variants require an explicit graph identity')
  actions=sorted({f'L{n}' for n in re.findall(r'\bL(\d+)\s*:',r['input'])},key=lambda x:int(x[1:]))
  try:
   source_actions=domains.load_source_actions(k,ROOT/'Data/ApplicationDataset')
   expected=set(domains.domains_from_actions(source_actions))
   placeholder_match=all(
    {m[1].upper() for m in domains.DIRECTIVE_RE.finditer(x.get('input',''))}==expected for x in rs)
  except (KeyError,ValueError,FileNotFoundError):
   placeholder_match=False
  complete=bool(g and set(actions).issubset(g['matched_actions']) and placeholder_match)
  source_match=bool(g and source_sha==g['source_sha256'])
  ok=bool(g and complete and source_match and k not in INCOMPLETE)
  if ok: usable.add(k)
  inv[k]={'family':r['_family'],'split':r['_quality_split'],'rows':len(rs),'request_count':len({x['_quality_request_id'] for x in rs}),
   'source_file':r['source_file'],'source_sha256':source_sha,'action_ids':actions,'graph':g,'source_hash_matches_graph':source_match,
   'source_action_placeholders_match':placeholder_match,'complete_action_mapping':complete,'usable':ok,
   'exclusion_reason':None if ok else ('no_rc3_graph' if not g else 'source_hash_mismatch' if not source_match else
       'source_action_placeholder_mismatch' if not placeholder_match else 'incomplete_compiler_action_mapping')}
 splits={}
 for split in ('train','validation'):
  rr=[r for r in rows if r['_quality_split']==split and r['kernel_name'] in usable]
  splits[split]={'rows':len(rr),'kernels':sorted({r['kernel_name'] for r in rr}),'families':sorted({r['_family'] for r in rr}),
   'requests':len({r['_quality_request_id'] for r in rr}),
   'device_clock_budget_rows':dict(Counter(f"{r['device']}|{r['selected_clock_period']}ns|{r['resource_budget_frac']}" for r in rr))}
 byfam=defaultdict(list)
 for k in usable:
  if inv[k]['split']=='train': byfam[inv[k]['family']].append(k)
 selected={f:max(ks,key=lambda k:(len(inv[k]['graph']['matched_actions']),inv[k]['request_count'],k)) for f,ks in sorted(byfam.items())}
 schedule=[r for r in rows if r['_quality_split']=='train' and r['kernel_name']==selected.get(r['_family'])]
 # Validate complete target assignments against the current source-aware tuple registry.
 legal_rows=0
 for r in rows:
  if r['kernel_name'] not in usable: continue
  site=domains.tcad_site_domains(r['kernel_name'],r['input'],ROOT/'Data/ApplicationDataset')
  values={}
  for line in r['target'].splitlines():
   if '=' in line:
    key,val=line.split('=',1); values[key.strip().upper()]=val.strip()
  domains.validate_assignments(values,site.actions)
  legal_rows+=1
 # Resource percentages are rounded source fields; this is a screening audit,
 # not exact vendor-resource receipt validation. Timing estimates are absent.
 def row_feasible(r):
  frac=float(r['resource_budget_frac'])
  return all(float(r[name]) <= 100.0*frac+0.01 for name in ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%'))
 resource_audit={}
 train_rows=[r for r in rows if r['_quality_split']=='train' and r['kernel_name'] in usable]
 for frac in (0.25,0.5,1.0):
  subset=[r for r in train_rows if float(r['resource_budget_frac'])==frac]
  resource_audit[str(frac)]={'rows':len(subset),'screened_feasible_rows':sum(row_feasible(r) for r in subset),
    'screening_note':'Utilization percentages are rounded and rows are curated near-optimal references; exact resource counts and omitted-candidate feasibility are unavailable.'}
 grouped_candidates=defaultdict(list)
 for r in train_rows: grouped_candidates[(r['kernel_name'],r['device'],r['selected_clock_period'],r['resource_budget_frac'])].append(r)
 multi_assignment_contexts=sum(len({x['target'] for x in rs})>1 for rs in grouped_candidates.values())
 assignment_strata=defaultdict(dict)
 for (kernel,device,clock,budget),rs in grouped_candidates.items():
  feasible=[x for x in rs if row_feasible(x)]
  if feasible:
   best=min(feasible,key=lambda x:float(x['latency']))
   assignment_strata[(kernel,device,clock)][float(budget)]=best['target']
 budget_differences={}
 for lo,hi in ((0.25,0.5),(0.5,1.0),(0.25,1.0)):
  pairs=[(strata[lo],strata[hi]) for strata in assignment_strata.values() if lo in strata and hi in strata]
  budget_differences[f'{lo}_vs_{hi}']={'matched_strata':len(pairs),'different_lowest_sampled_assignments':sum(a!=b for a,b in pairs),
   'interpretation':'lowest measured latency among the four curated exposures per request; not a global measured-candidate winner'}
 report={'schema':'salt-hls-stage2-multikernel-audit-v1',
  'inputs':{'training_jsonl':str(data),'training_jsonl_sha256':dh,'canonical_rc3_archive':str(archive),'canonical_rc3_archive_sha256':gh,
   'stage1_adapter_sha256':'28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164',
   'gnn_checkpoint_sha256':'ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162'},
  'canonical_graph_count':len(graphs),
  'canonical_graph_variant_count':sum(len(g['archive_variants']) for g in graphs.values()),
  'ambiguous_graph_names':{k:[v['graph_identity'] for v in g['archive_variants']] for k,g in graphs.items() if len(g['archive_variants'])>1},'candidate_kernel_count':len(grouped),'candidate_family_count':len({r['_family'] for r in rows}),
  'usable_kernels':len(usable),'usable_families':len({inv[k]['family'] for k in usable}),'splits':splits,
  'one_kernel_per_training_family':selected,'pilot_schedule_rows':len(schedule),
  'source_tuple_validated_rows':legal_rows,'resource_screen_audit':resource_audit,
  'training_contexts_with_multiple_sampled_assignments':multi_assignment_contexts,
  'budget_assignment_comparison':budget_differences,
  'timing_estimate_fields_present':False,
  'known_incomplete_mapping_exclusions':{k:inv[k]['exclusion_reason'] for k in sorted(INCOMPLETE)},'inventory':inv}
 return report,schedule
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=DEFAULT_DATA); p.add_argument('--graphs',type=Path,default=DEFAULT_ARCHIVE)
 p.add_argument('--out',type=Path,default=ROOT/'artifacts/rc3_stage2_multikernel/v1/audit'); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 report,schedule=audit(a.data,a.graphs)
 (a.out/'coverage.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 (a.out/'pilot_schedule.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in schedule))
 all_rows=[json.loads(x) for x in a.data.read_text().splitlines() if x.strip()]
 selected=set(report['one_kernel_per_training_family'].values())
 validation=set(report['splits']['validation']['kernels'])
 pilot=[r for r in all_rows if (r['_quality_split']=='train' and r['kernel_name'] in selected) or
        (r['_quality_split']=='validation' and r['kernel_name'] in validation)]
 dataset_path=a.out/'stage2_pilot.jsonl'
 dataset_path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in pilot))
 train_idx=[i for i,r in enumerate(pilot) if r['_quality_split']=='train']
 val_idx=[i for i,r in enumerate(pilot) if r['_quality_split']=='validation']
 split={'train_jsonl_idx':train_idx,'val_jsonl_idx':val_idx,'test_jsonl_idx':[],
        'seed':123,'train_families':sorted({r['_family'] for r in pilot if r['_quality_split']=='train'}),
        'validation_families':sorted({r['_family'] for r in pilot if r['_quality_split']=='validation'}),
        'test_families':[],'source_split':'authenticated V4 seed-123 family split; test partition not accessed'}
 (a.out/'split.json').write_text(json.dumps(split,indent=2,sort_keys=True)+'\n')
 manifest={'schema':'salt-hls-stage2-multikernel-pilot-dataset-v1','source_sha256':report['inputs']['training_jsonl_sha256'],
  'dataset_sha256':sha(dataset_path),'row_count':len(pilot),'train_rows':len(train_idx),'validation_rows':len(val_idx),
  'test_rows':0,'training_kernels':sorted(selected),'validation_kernels':sorted(validation),
  'training_families':split['train_families'],'validation_families':split['validation_families'],
  'label_policy':'reuse complete authenticated multi-reference V4 assignments with their measured fields; no relabeling',
  'timing_qualification':'unavailable: rows contain requested selected_clock_period but no achieved/estimated HLS clock per candidate',
  'limitations':['not a full measured candidate bank','utilization is reported percent, not exact counts','curated near-optimal references only']}
 (a.out/'pilot_dataset_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'kernels':report['candidate_kernel_count'],'usable':report['usable_kernels'],
   'train_families':len(split['train_families']),'train_rows':len(train_idx),'validation_families':len(split['validation_families']),
   'validation_rows':len(val_idx),'timing_qualification':'unavailable'},indent=2))
if __name__=='__main__': main()
