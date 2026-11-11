"""Deterministic downstream memory interventions, never source transformations."""
from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class Intervention:
    name: str
    action: int  # zero-based absolute slot
    removed: tuple = ()
    replacements: tuple = ()  # (recipient index, content donor index)
    summary: str = 'recompute'
    gate_off: bool = False
    control_for: str = ''


def plan(pack,seed=123):
    rows=pack['records']
    specs=[]
    for k in pack['active'][0].nonzero().flatten().tolist():
        eligible=pack['routes'][0,k].nonzero().flatten().tolist()
        specs += [Intervention('original',k),Intervention('gate_off',k,gate_off=True),
                  Intervention('action_only',k,tuple(eligible),summary='remove_all'),
                  Intervention('action_evidence',k,summary='remove_all'),
                  Intervention('remove_summary_0',k,summary='remove_0'),
                  Intervention('remove_summary_1',k,summary='remove_1'),
                  Intervention('zero_summaries',k,summary='zero'),
                  Intervention('cross_action_summaries',k,summary='cross_action')]
        for family in sorted({rows[i]['type'] for i in eligible}):
            chosen=tuple(i for i in eligible if rows[i]['type']==family)
            label='remove_family:'+family
            specs.append(Intervention(label,k,chosen))
            specs.append(Intervention('key_only_family:'+family,k,chosen,summary='keep_original'))
            # Same key-removal cardinality in the same legal route. Actual ID
            # sets are recorded; chance overlap is legitimate and reported.
            rng=random.Random(f'{seed}:{k}:{family}')
            control=tuple(sorted(rng.sample(eligible,len(chosen))))
            specs.append(Intervention('random_for:'+family,k,control,control_for=label))
        # Predeclared individual-record subset: first canonical record of every
        # eligible type (no post-hoc selection by observed effect magnitude).
        individuals=[next(i for i in eligible if rows[i]['type']==family)
                     for family in sorted({rows[i]['type'] for i in eligible})]
        individuals=sorted(set(individuals)|{i for i in eligible if
            (k==4 and rows[i]['type'] in ('memory_access','access','arithmetic','loop_carried','loop'))
            or (k<3 and rows[i]['type'] in ('memory_access','access'))})
        for i in individuals:
            specs.append(Intervention('remove_record:'+rows[i]['id'],k,(i,)))
            # Content-changing cross-route control matched on type/status/all
            # knownness bits. ID and routing mask remain recipient-authorized.
            donors=[j for j,r in enumerate(rows) if j not in eligible and
                    (r['type'],r['status'],r['known']) ==
                    (rows[i]['type'],rows[i]['status'],rows[i]['known'])]
            if donors:
                specs.append(Intervention('wrong_content:'+rows[i]['id'],k,
                    replacements=((i,donors[0]),),control_for='remove_record:'+rows[i]['id']))
    return specs


def apply(module,pack,spec):
    """Recompute summaries after evidence deletion to avoid indirect leakage.

    Other action routes retain their keys. This is an action-specific model-level
    intervention: it does not erase information already encoded in action states.
    """
    changed={**pack,'routes':pack['routes'].clone()}
    for i in spec.removed: changed['routes'][:,spec.action,i]=False
    if spec.replacements:
        for field in ('a','b','edge'):
            changed[field]=pack[field].clone()
            for recipient,donor in spec.replacements:
                changed[field][:,recipient]=pack[field][:,donor]
    result=module(pack if spec.summary=='keep_original' else changed)
    result={**result,'memory':result['memory'].clone(),'relation':result['relation'].clone()}
    if spec.summary=='keep_original':
        for i in spec.removed:result['relation'][:,spec.action,64+i]=False
    n=pack['a'].shape[1]
    begin=64+n+2*spec.action
    if module.mode=='gmt':
        if spec.summary=='remove_all':result['relation'][:,spec.action,64+n:]=False
        elif spec.summary.startswith('remove_'):
            result['relation'][:,spec.action,begin+int(spec.summary[-1])]=False
        elif spec.summary=='zero':result['memory'][:,begin:begin+2]=0
        elif spec.summary=='cross_action':
            candidates=[k for k in pack['active'][0].nonzero().flatten().tolist()
                        if k!=spec.action and result['mask'][0,64+n+2*k:64+n+2*k+2].all()]
            if not candidates:raise ValueError('No valid cross-action summary donor')
            other=64+n+2*candidates[0]
            result['memory'][:,begin:begin+2]=result['memory'][:,other:other+2].clone()
    return result


def deltas(reference,changed):
    """Fixed candidates: preference shift, ranking and winner changes.

    Reference-winner margin always compares the SAME top-two continuations.
    The intervened own-top-two margin is reported separately.
    """
    if set(reference)!=set(changed):raise ValueError('Candidate domains changed')
    order=sorted(reference,key=lambda k:(-reference[k],k))
    new=sorted(changed,key=lambda k:(-changed[k],k))
    winner=order[0]
    runner=order[1] if len(order)>1 else None
    margin=reference[winner]-reference[runner] if runner else None
    new_fixed=changed[winner]-changed[runner] if runner else None
    return dict(reference_winner=winner,winner=new[0],winner_changed=winner!=new[0],
        reference_margin=margin,fixed_pair_margin=new_fixed,
        delta_margin=new_fixed-margin if runner else None,
        intervened_top_two_margin=changed[new[0]]-changed[new[1]] if runner else None,
        rank_changed=order!=new,reference_ranks={k:i+1 for i,k in enumerate(order)},
        ranks={k:i+1 for i,k in enumerate(new)},
        delta_logp={k:changed[k]-reference[k] for k in order})
