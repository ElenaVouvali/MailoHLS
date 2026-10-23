#!/usr/bin/env python3
"""Prepare audited ZCU winners and label-free U50 inference cases. No synthesis."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from Preprocessing import data_preprocess as dp, create_jsonl as cj
from LLM_branch.common import mailohls_contract as mc

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name('config.json')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def prepare(root):
    from LLM_branch.train import train_SFT_xattn_new as trainer
    cfg = json.loads(CONFIG.read_text())
    kernels = sum((cfg[f'{p}_kernels'] for p in ('train', 'val', 'test')), [])
    assert len(kernels) == len(set(kernels)) == 21
    families = {p: {mc.family_id_from_kernel_name(k) for k in cfg[f'{p}_kernels']} for p in ('train','val','test')}
    assert len(families['train']) == 16 and len(families['val']) == 3 and len(families['test']) == 1
    assert not (families['train'] & families['val'] or families['train'] & families['test'] or families['val'] & families['test'])
    assert not set(kernels) & set(cfg['excluded_final_kernels'])
    base = REPO / cfg['base_stage1']
    parent = json.loads((base/'training_contract.json').read_text())
    trainer.TARGET_CFG.effective_area_floor = parent['effective_area_floor']
    assert set(parent['family_split_summary']['val']['families']) == families['val']
    inputs = root/'zcu_reference_preprocessed'; inputs.mkdir(parents=True, exist_ok=True)
    audit = []
    for k in kernels:
        raw = REPO/'Data/CSVS'/f'{k}.csv'
        frame, stats = dp.preprocess_kernel(raw, 'gnn', 'xczu7ev-ffvc1156-2-e', 10.0, 0.1, 2.0)
        frame.to_csv(inputs/f'preprocessed-{k}.csv',index=False)
        audit.append({'kernel':k, 'raw_csv_sha256':sha(raw), **stats})
    examples, _, templates = cj.build_examples(inputs)
    candidates=[]; cases=[]
    for k in kernels:
        rows=[dict(r,input=templates[k]) for r in examples if r['kernel_name']==k]
        mapping=dp.load_action_definitions(k)
        table=pd.read_csv(inputs/f'preprocessed-{k}.csv').fillna('')
        seen=set()
        for objective in ('PARETO_AREA','PARETO_ADP','PARETO_LATENCY'):
            win=trainer.rank_goal_candidates(rows,objective,0.0,0.0)[0]['row']
            record=table.iloc[win['preprocessed_row']]
            directives={col:dp.canonicalize_directive(record.get(col,''),action) or 'NDIR' for col,action in mapping.items()}
            signature=hashlib.sha256(json.dumps(directives,sort_keys=True).encode()).hexdigest()
            candidates.append({'kernel':k,'origin':'zcu104_'+objective.lower(),'directives':directives,
                'target':win['target'],'signature':signature,'duplicate':signature in seen,
                'zcu_reference':{'version':str(record['Version']),'latency_msec':win['latency'],'area_pct':win['area']}})
            seen.add(signature)
        cases.append({'kernel_name':k,'input':templates[k],'device':cfg['part'],'clock_period':10.0,
            'frequency_mode':'specified','resource_budget_id':'full','obj_mode':'PARETO_LATENCY',
            **{mc.AVAIL_FIELD_BY_RESOURCE[r]:n for r,n in cfg['u50_available_resources'].items()}})
    for dirname in ('raw_csv','preprocessed','artifacts','splits','runs','sources'):
        (root/dirname).mkdir(parents=True,exist_ok=True)
    sources=cj.application_sources(set(kernels)); source_audit=[]
    for k in kernels:
        source_dir=sources[k].parent; dest=root/'sources'/k; dest.mkdir(exist_ok=True)
        files={}
        for p in source_dir.iterdir():
            if p.suffix.lower() in {'.c','.cc','.cpp','.cxx','.h','.hpp'} or p.name=='kernel_info.txt':
                shutil.copy2(p,dest/p.name); files[p.name]=sha(p)
        raw=REPO/'Data/CSVS'/f'{k}.csv'
        source_audit.append({'kernel':k,'main_source':sources[k].name,'files':files,
            'raw_header':pd.read_csv(raw,nrows=0).columns.tolist(),
            'apl':(REPO/'Data/ApplicationAPLMapping'/f'{k}.txt').read_text()})
    write(root/'artifacts/config.json',cfg)
    write(root/'u50_device_specs.json',{'devices':{cfg['part']:cfg['u50_available_resources']}})
    write(root/'artifacts/zcu_candidates.json',candidates)
    write(root/'artifacts/source_manifest.json',source_audit)
    write(root/'artifacts/preparation_audit.json',{'reference_preprocessing':audit,'families':{p:sorted(f) for p,f in families.items()},
        'parent_contract_sha256':sha(base/'training_contract.json'),'parent_adapter_sha256':sha(base/'adapter_model.safetensors'),
        'duplicates_requiring_neighbor':[{k:r[k] for k in ('kernel','origin','signature')} for r in candidates if r['duplicate']],
        'status':'awaiting_21_zero_shot_predictions_and_duplicate_resolution'})
    (root/'artifacts/u50_zero_shot_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    print(f'Prepared {len(candidates)} winner slots, {sum(r["duplicate"] for r in candidates)} duplicate slots, and {len(cases)} label-free inference cases in {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve())
