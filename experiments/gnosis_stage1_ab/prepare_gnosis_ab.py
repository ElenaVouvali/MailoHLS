"""Build immutable, family-disjoint GNOSIS Stage-1 conditional supervision.

Run inside MailoHLS worktree. This reads ONLY train/val JSONL records;
locked-test JSONL records are not parsed. Uses verified compact-source sidecars.

The derived resource budgets are HLS-estimate proxy constraints (reported percent);
not exact FPGA resource counts or achieved-timing guarantees.
"""
from __future__ import annotations
import argparse, collections, functools, hashlib, itertools, json, math, random
from pathlib import Path
from LLM_branch.train import train_SFT_xattn_new as prod

RKEYS = tuple(prod.RESOURCE_KEYS)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def dump(p,obj):
    Path(p).write_text(json.dumps(obj,sort_keys=True,indent=2,default=str)+'\n')

def jlines(p,rows):
    with Path(p).open('w') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')

def src_rows(dataset,split):
    ds=Path(dataset); split=json.loads(Path(split).read_text())
    memb={n:set(split[n+'_jsonl_idx']) for n in ('train','val','test')}
    assert all(memb[a].isdisjoint(memb[b]) for a,b in itertools.combinations(memb,2))
    sources_path=ds.with_suffix('.sources.json'); manifest_path=ds.with_suffix('.manifest.json')
    assert sources_path.exists() and manifest_path.exists(), 'Expected authenticated compact SFT sidecars'
    man=json.loads(manifest_path.read_text())
    assert man['output_sha256']==digest(ds) and man['sources_sha256']==digest(sources_path)
    templates=json.loads(sources_path.read_text())['templates']
    records={'train':[],'val':[]};seen=collections.Counter()
    with ds.open() as inp:
        for idx,line in enumerate(inp):
            # Avoid even parsing the sealed test's outcomes.
            fold='train' if idx in memb['train'] else 'val' if idx in memb['val'] else None
            if fold is None: continue
            row=json.loads(line)
            if not row.get('input'):row['input']=templates[row['source_key']]
            row['_jsonl_idx']=idx;row['_family']=prod.family_id_from_kernel_name(row['kernel_name'])
            records[fold].append(row);seen[fold]+=1
    assert all(seen[n]==len(memb[n]) for n in ('train','val')), (seen,{k:len(v) for k,v in memb.items()})
    assert {r['_family'] for r in records['train']}.isdisjoint({r['_family'] for r in records['val']})
    return records,{k:len(v) for k,v in memb.items()},man

@functools.lru_cache(maxsize=128)
def cached_ordered_lhs_plan(source):
    return tuple(prod.extract_ordered_lhs_plan(source))

def validate_row(row, eligible, registry=None):
    """Conservative row validation, returning a rejection reason or None.

    Production create_jsonl emits lowercase ``auto{...}`` target LHS keys;
    source/domain identities are compared in uppercase. Normalize BOTH sides.
    """
    try:
        if row['kernel_name'] not in eligible:
            return 'not_canonical_source_complete'
        latency=float(row['latency'])
        if not (latency>0 and math.isfinite(latency)):
            return 'invalid_latency'
        pct=[float(row[prod.UTIL_FIELD_BY_RESOURCE[k]]) for k in RKEYS]
        if any(not math.isfinite(x) or x<0 or x>100 for x in pct):
            return 'invalid_resource_percentage'
        if all(x==0 for x in pct):
            return 'unobservable_all_zero_resources'
        if not row.get('target') or not row.get('input'):
            return 'missing_source_or_completion'
        if float(prod._clock_of(row)) not in (3.33,5.,10.):
            return 'unsupported_clock'
        if prod._norm_device(row['device']) not in prod.DEVICE_RESOURCES:
            return 'unsupported_device'
        plan=cached_ordered_lhs_plan(row['input'])
        expected=[lhs.upper() for _,lhs in plan]
        parsed=prod.build_rhs_map_from_target(row['target'])
        target={lhs.upper():rhs for lhs,rhs in parsed.items()}
        if len(target)!=len(parsed):
            return 'duplicate_target_lhs_after_normalization'
        if not expected or len(expected)!=len(set(expected)) or set(expected)!=set(target):
            return 'source_completion_lhs_mismatch'
        if registry is not None:
            prior={}
            for _,lhs in plan:
                key=lhs.upper()
                _,choices=prod.constrained_rhs_candidates(row['kernel_name'],lhs,registry,prior)
                if target[key] not in choices:
                    return 'out_of_parent_legal_domain'
                prior[key]=target[key]
        return None
    except KeyError:
        return 'missing_parent_registry_or_input_field'
    except (ValueError,TypeError,AssertionError):
        return 'invalid_value_or_domain_contract'


def valid(row,eligible,registry=None):
    return validate_row(row,eligible,registry) is None

def resources(row):return tuple(float(row[prod.UTIL_FIELD_BY_RESOURCE[k]]) for k in RKEYS)

def feasible(row,budget,guard=1.):
    return all(r+guard<=100*b+1e-9 for r,b in zip(resources(row),budget))

def excluded(row,budget,guard=1.):
    return any(r-guard>100*b+1e-9 for r,b in zip(resources(row),budget))

def stable(row):
    return (float(row['latency']),prod.canonical_completion_key(row['input'],row['target']),row['_jsonl_idx'])

def bid(budget):return 'B'+ '_'.join('%03d'%round(100*v) for v in budget)

def make_row(src,budget):
    obj=prod.attach_shared_budget(src,prod.SharedResourceBudget(*budget))
    obj.update(obj_mode='PARETO_LATENCY',frequency_mode='specified',selected_clock_period=float(prod._clock_of(src)),_quality_arm='multi_reference')
    return obj

def key(row):return (row['kernel_name'],prod._norm_device(row['device']),float(prod._clock_of(row)))

def id_of(row):
    if '_canonical_id' in row:return row['_canonical_id']
    return prod.canonical_completion_key(row['input'],row['target'])

def select_budgets(items,limit):
    full=(1.,)*4
    budgets=[full,(.25,)*4,(.5,)*4,(.75,)*4]
    fast=min(items,key=stable);fast_resource=resources(fast);fast_id=id_of(fast)
    # Binding, resource-specific budgets, anchored in actual feasible alternatives.
    for j in range(4):
        possible=[]
        for alt in items:
            rf,ra=fast_resource[j],resources(alt)[j]
            if id_of(alt)==fast_id or float(alt['latency'])<float(fast['latency'])*1.10:continue
            if rf-ra < 3.:continue # robust +/-1 percentage-point interval
            frac=round((ra+rf)/200.,2)
            b=list(full);b[j]=frac;b=tuple(b)
            if min(b)<.05:continue
            if feasible(alt,b) and excluded(fast,b):possible.append((float(alt['latency']),b))
        if possible:budgets.append(min(possible)[1])
    # Optional independently chosen mixed budgets, drawn from feasible demand envelopes.
    sorted_designs=sorted(items,key=stable)[:min(24,len(items))]
    for alt in sorted_designs[::max(1,len(sorted_designs)//4)]:
        b=tuple(min(1.,max(.05,math.ceil((r+1.)/2.)/100.)) for r in resources(alt))
        budgets.append(b)
    out=[]
    for b in budgets:
        if b not in out:out.append(b)
    # Always keep full and standard; bound the rest to avoid count inflation.
    return out[:limit]

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',type=Path,required=True)
    a.add_argument('--split',type=Path,required=True)
    a.add_argument('--canonical-manifest',type=Path,default=Path('Data/manifest.json'))
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--top-k',type=int,default=3)
    a.add_argument('--registry',type=Path,required=True)
    a.add_argument('--near-opt',type=float,default=.02)
    a.add_argument('--max-budgets',type=int,default=10)
    a.add_argument('--max-pairs-per-kernel',type=int,default=8)
    z=a.parse_args();assert not z.out.exists(),'Output immutable: use fresh path'
    raw,split_counts,source_manifest=src_rows(z.dataset,z.split)
    registry=prod.load_directive_domain_registry(str(z.registry))
    cm=json.loads(z.canonical_manifest.read_text());kstates={r['kernel']:r for r in cm['kernels']}
    eligible={k for k,v in kstates.items() if v['complete_source_configuration_eligible'] and v['split']!='test'}
    out={'train':{'sft':[],'pairs':[]},'val':{'sft':[],'pairs':[]}};ledger={};cc={}
    for fold in ('train','val'):
        # Preserve fail-closed source/registry policy and give actionable diagnostics.
        rejects=collections.Counter();example_rejects={};filtered=[]
        for r in raw[fold]:
            reason=validate_row(r,eligible,registry)
            if reason is None:
                filtered.append(r)
            else:
                rejects[reason]+=1
                if reason not in example_rejects:
                    example_rejects[reason]={
                        'kernel':r.get('kernel_name'),
                        'jsonl_idx':r.get('_jsonl_idx'),
                        'example_lhs':next(iter(prod.build_rhs_map_from_target(r.get('target',''))),None),
                    }
        print('[SOURCE-DOMAIN-AUDIT]',fold,'source_rows',len(raw[fold]),
              'accepted',len(filtered),'excluded',dict(rejects),flush=True)
        if not filtered:
            raise ValueError('No '+fold+' rows passed source/domain qualification; '
                             'diagnostics='+json.dumps({'counts':dict(rejects),
                             'samples':example_rejects},sort_keys=True))
        by=collections.defaultdict(dict)
        for row in filtered:
            k=key(row);cid=id_of(row);row['_canonical_id']=cid
            prev=by[k].get(cid)
            if prev is None or stable(row)<stable(prev):by[k][cid]=row
        groups={k:list(d.values()) for k,d in by.items()}
        n_switch=n_stay=0;n_clock=0;status=collections.Counter();samples=[];pairs=[]
        for group,items in sorted(groups.items()):
            if len(items)<2:continue
            for budget in select_budgets(items,z.max_budgets):
                eligible_items=sorted((r for r in items if feasible(r,budget)),key=stable)
                if len(eligible_items)<2:continue
                best=float(eligible_items[0]['latency'])
                refs=[r for r in eligible_items if float(r['latency'])<=best*(1+z.near_opt)][:z.top_k]
                # Keep multi-reference support, weighted by context rather than row count.
                for r in refs:samples.append({'row':make_row(r,budget),'reference_count':len(refs),'context_id':str(group)+'|'+bid(budget),'kind':'multi_reference','measurement_idx':r['_jsonl_idx'],'source_family':r['_family']})
                if budget==(1.,)*4:continue
                fast=min(items,key=stable);tight=eligible_items[0]
                if id_of(fast)==id_of(tight):
                    # Nonbinding 'stay' is covered by SFT, don't invent a reversal.
                    n_stay+=1;continue
                if not excluded(fast,budget) or float(tight['latency'])<float(fast['latency'])*1.10:continue
                if status[group[0]]>=z.max_pairs_per_kernel:continue
                status[group[0]]+=1;n_switch+=1
                for budget_,winner,loser in [((1.,)*4,fast,tight),(budget,tight,fast)]:
                    pairs.append({'kind':'budget_switch','group_id':group[0]+'|'+str(group[1:])+'|'+bid(budget),
                                  'row':make_row(winner,budget_), 'preferred_target':winner['target'],
                                  'rejected_target':loser['target'], 'preferred_idx':winner['_jsonl_idx'],
                                  'rejected_idx':loser['_jsonl_idx'],'winner_latency':winner['latency'],
                                  'loser_latency':loser['latency'],
                                  'evidence':'reported utilization percentages, conservative +/-1pp; no achieved timing proof'})
        # Same source + device, two clock contexts, identical complete assignments.
        by_kd=collections.defaultdict(dict)
        for (k,dev,clk),rows in groups.items():by_kd[k,dev][clk]={id_of(r):r for r in rows}
        for (k,dev),by_clock in sorted(by_kd.items()):
            count=0
            for t1,t2 in itertools.combinations(sorted(by_clock),2):
                common=sorted(set(by_clock[t1])&set(by_clock[t2]))
                if len(common)<2:continue
                for a_id,b_id in itertools.combinations(common,2):
                    if count>=z.max_pairs_per_kernel:break
                    u,v=by_clock[t1],by_clock[t2]
                    u1,u2=u[a_id],u[b_id];v1,v2=v[a_id],v[b_id]
                    if any(not feasible(r,(1.,)*4) for r in (u1,u2,v1,v2)):continue
                    s1=float(u1['latency'])<float(u2['latency']);s2=float(v1['latency'])<float(v2['latency'])
                    if s1==s2:continue
                    if min(max(float(u1['latency']),float(u2['latency']))/min(float(u1['latency']),float(u2['latency'])),
                           max(float(v1['latency']),float(v2['latency']))/min(float(v1['latency']),float(v2['latency'])))<1.10:continue
                    group_id=k+'|'+dev+'|clock|'+str(t1)+'|'+str(t2)+'|'+str(a_id)+'|'+str(b_id)
                    for win,lose in ((u1,u2) if s1 else (u2,u1),(v1,v2) if s2 else (v2,v1)):
                        pairs.append({'kind':'clock_switch','group_id':group_id,'row':make_row(win,(1.,)*4),
                                      'preferred_target':win['target'],'rejected_target':lose['target'],
                                      'preferred_idx':win['_jsonl_idx'],'rejected_idx':lose['_jsonl_idx'],
                                      'winner_latency':win['latency'],'loser_latency':lose['latency'],
                                      'evidence':'two measured clock contexts; pair reversal, not necessarily near-optimal'})
                    count+=1;n_clock+=1
        # Cap SFT exposures per kernel/context, preserve per-context reference groups.
        # Separately cap contrast groups so many related clock pairs do not dominate.
        out[fold]['sft']=samples;out[fold]['pairs']=pairs
        ledger[fold]={'source_rows':len(raw[fold]),'qualified_rows':len(filtered),'physical_groups':len(groups),
                      'sft_exposures':len(samples),'distinct_sft_contexts':len({x['context_id'] for x in samples}),
                      'budget_switch_groups':n_switch,'clock_switch_groups':n_clock,'nonbinding_budget_contexts':n_stay,
                      'pair_exposures':len(pairs),'families':sorted({r['_family'] for r in filtered}),
                      'rejection_counts':dict(rejects),'rejection_examples':example_rejects}
    assert {x['source_family'] for x in out['train']['sft']}.isdisjoint({x['source_family'] for x in out['val']['sft']})
    if not all(out[n]['sft'] for n in ('train','val')):
        raise ValueError('No SFT exposures in one split after budget/context qualification; '
                         'inspect [SOURCE-DOMAIN-AUDIT] and per-fold source counts: '+json.dumps(ledger))
    z.out.mkdir(parents=True)
    for fold in ('train','val'):
        for kind in ('sft','pairs'):jlines(z.out/(fold+'_'+kind+'.jsonl'),out[fold][kind])
    data={'schema':'gnosis-target-conditioning-ab-v1','source':'authenticated compact SFT rows, canonical completeness filter',
          'limitations':['proxy resource feasibility from percentages','no timing closure','clock reversals need not involve near-optimal designs','test records never parsed'],
          'locked_test_read':False,'counts':ledger,'split_counts':split_counts,
          'params':{'top_k':z.top_k,'near_opt':z.near_opt,'max_budgets':z.max_budgets,'max_pairs_per_kernel':z.max_pairs_per_kernel},
          'inputs_sha256':{str(p):digest(p) for p in (z.dataset,z.dataset.with_suffix('.sources.json'),z.dataset.with_suffix('.manifest.json'),z.split,z.canonical_manifest,z.registry)},
          'outputs_sha256':{p.name:digest(p) for p in z.out.glob('*.jsonl')}}
    dump(z.out/'manifest.json',data)
    print(json.dumps(ledger,indent=2),flush=True)
    print('COMPLETE',z.out)

if __name__=='__main__':main()
