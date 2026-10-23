#!/usr/bin/env python3
"""Replace duplicate signatures with one legal, single-action neighbor."""
import argparse, hashlib, json, re
from pathlib import Path
from LLM_branch.common import directive_domains as dd, mailohls_contract as mc

ASSIGN=re.compile(r"^auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}\s*=\s*(\S+)$",re.I)
def sig(t): return hashlib.sha256('\n'.join(x.strip().upper() for x in t.splitlines() if x.strip()).encode()).hexdigest()
def parse(t):
    d={}
    for line in t.splitlines():
        if line.strip():
            m=ASSIGN.match(line.strip())
            if not m: raise ValueError(f'Malformed target: {line}')
            d[m.group(1).upper()]=m.group(2).lower()
    return d
def serial(d): return '\n'.join(f'auto{{{k}}} = {v}' for k,v in d.items())
def neighbor(target, domains, used):
    d=parse(target)
    labels=[]
    for key in d:
        label=key.rsplit('_',1)[-1]
        if label not in labels: labels.append(label)
    for label in labels:
        if f'_PIPE_{label}' in d:
            for value in domains.get(f'AUTO{{_PIPE_{label}}}',[]):
                if value==d[f'_PIPE_{label}']: continue
                x=dict(d);x[f'_PIPE_{label}']=value
                if mc._valid_action_values({'PIPE':x[f'_PIPE_{label}'],'UNROLL':x.get(f'_UNROLL_{label}','0')}) and sig(serial(x)) not in used:
                    return serial(x),f'{label}_PIPE'
        if f'_UNROLL_{label}' in d:
            for value in domains.get(f'AUTO{{_UNROLL_{label}}}',[]):
                if value==d[f'_UNROLL_{label}']: continue
                x=dict(d);x[f'_UNROLL_{label}']=value
                if mc._valid_action_values({'PIPE':x.get(f'_PIPE_{label}','0'),'UNROLL':x[f'_UNROLL_{label}']}) and sig(serial(x)) not in used:
                    return serial(x),f'{label}_UNROLL'
        if f'_ARRAY_T_{label}' in d:
            keys=[f'_ARRAY_T_{label}',f'_ARRAY_F_{label}',f'_ARRAY_D_{label}']
            for key in keys:
                for value in domains.get('AUTO{'+key+'}',[]):
                    if value==d[key]: continue
                    x=dict(d);x[key]=value
                    vals={'ARRAY_T':x[f'_ARRAY_T_{label}'],'ARRAY_F':x[f'_ARRAY_F_{label}'],'ARRAY_D':x[f'_ARRAY_D_{label}']}
                    if mc._valid_action_values(vals) and sig(serial(x)) not in used:
                        return serial(x),f'{label}_{key.split("_")[1]}'
    raise ValueError('No legal unused single-action neighbor exists')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zcu-candidates',type=Path,required=True);ap.add_argument('--zero-shot',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output-zcu',type=Path,required=True);ap.add_argument('--output-zero-shot',type=Path,required=True);a=ap.parse_args()
    cases={};
    for line in a.cases.read_text().splitlines():
        if line.strip():
            r=json.loads(line);cases[r['kernel_name']]=r['input']
    zcu=json.loads(a.zcu_candidates.read_text());zero=[json.loads(x) for x in a.zero_shot.read_text().splitlines() if x.strip()]
    all_rows={k:[r for r in zcu if r['kernel']==k] for k in cases}
    for r in zero:
        all_rows[r['kernel_name']].append({'kernel':r['kernel_name'],'origin':'u50_zero_shot_pareto_latency','target':r['target']})
    outz=[];out0=[]
    for kernel in sorted(cases):
        domains=dd.source_site_domains(kernel,cases[kernel]); used=set()
        for r in all_rows[kernel]:
            target=r['target']; origin=r['origin'];
            if sig(target) in used:
                target,site=neighbor(target,domains,used);origin += '__neighbor_'+site
            used.add(sig(target)); item={'kernel_name':kernel,'origin':origin,'device':'xcu50-fsvh2104-2-e','clock_period_ns':10.0,'target':target,'neighbor_resolved':('__neighbor_' in origin)}
            if origin.startswith('u50_zero_shot'): out0.append(item)
            else: outz.append(dict(r, target=target, origin=origin, neighbor_resolved=('__neighbor_' in origin)))
    a.output_zcu.parent.mkdir(parents=True,exist_ok=True);a.output_zero_shot.parent.mkdir(parents=True,exist_ok=True)
    a.output_zcu.write_text(json.dumps(outz,indent=2)+'\n');a.output_zero_shot.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in out0));print(f'resolved {len(outz)+len(out0)} designs')
if __name__=='__main__':main()
