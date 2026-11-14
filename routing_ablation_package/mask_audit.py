#!/usr/bin/env python3
"""CPU audit of actual Lk-to-memory routing for the frozen GMT bank.

No surrogate memory and no architectural mutation. Identifies degenerate masks.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from LLM_branch.common import structural_memory as sm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--memory-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    bank,dim=sm.load_memory_bank(str(a.memory_dir),expected_mem_dim=128,
                                 expected_max_slots=64,require_pragma_free_memory=True)
    if dim!=128:raise ValueError('Memory feature dimension changed')
    report={'memory_manifest':str(a.memory_dir/'memory_manifest.json'),
            'kernels':{},'summary':{'kernel_count':0,'actions_checked':0,
                                   'self_only_accessible_keys':1,
                                   'compiler_vs_self_nontrivial_actions':0,
                                   'full_vs_compiler_nontrivial_actions':0}}
    seen_records=set()
    for name,rec in sorted(bank.items()):
        if id(rec) in seen_records:  # loader also exports normalized-name aliases
            continue
        seen_records.add(id(rec))
        mask=rec['mask'].bool()
        rel=rec['relation_mask']
        active=rec['action_slot_mask'][:64].bool()
        if rel is None or rel.shape!=(len(mask),len(mask)):
            raise ValueError(f'{name}: missing compiler relational matrix')
        valid=int(mask.sum())
        if valid<2:raise ValueError(f'{name}: expected multiple valid memory slots')
        nbs=[]
        for slot in active.nonzero().flatten().tolist():
            if not mask[slot]:raise ValueError(f'{name}: active action missing anchor L{slot+1}')
            row=rel[slot].bool() & mask
            count=int(row.sum())
            if not row[slot]:raise ValueError(f'{name}: compiler route omits own anchor L{slot+1}')
            if count<1 or count>valid:raise ValueError(f'{name}: bad compiler route degree')
            nbs.append({'action':f'L{slot+1}', 'self_only':1,
                        'compiler_guided':count,'full_memory':valid})
            report['summary']['actions_checked']+=1
            report['summary']['compiler_vs_self_nontrivial_actions']+=count>1
            report['summary']['full_vs_compiler_nontrivial_actions']+=count<valid
        if not nbs:raise ValueError(f'{name}: no action anchors')
        report['kernels'][name]={'memory_slots_total':int(mask.numel()),
                                'memory_slots_valid':valid,'actions':nbs}
    report['summary']['kernel_count']=len(bank)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('[MASK AUDIT]',json.dumps(report['summary']))
    if report['summary']['compiler_vs_self_nontrivial_actions']==0:
        raise RuntimeError('Compiler routing never differs from self-only')
    if report['summary']['full_vs_compiler_nontrivial_actions']==0:
        raise RuntimeError('Full-memory routing never differs from compiler routing')

if __name__=='__main__':main()
