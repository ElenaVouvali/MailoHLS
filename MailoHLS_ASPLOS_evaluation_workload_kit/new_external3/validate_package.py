#!/usr/bin/env python3
from pathlib import Path
import json, re, sys
ROOT=Path(__file__).resolve().parent
KERNELS=['chstone-aes','chstone-jpeg','rosetta-3d-rendering']
errors=[]
for k in KERNELS:
    d=ROOT/k
    srcs=list(d.glob('kernel.c'))+list(d.glob('kernel.cpp'))
    phs=list(d.glob('kernel_placeholders.c'))+list(d.glob('kernel_placeholders.cpp'))
    if len(srcs)!=1 or len(phs)!=1 or not (d/'kernel_info.txt').is_file():
        errors.append(f'{k}: not materialized; run materialize_upstream.py')
        continue
    src=srcs[0].read_text(encoding='utf-8')
    ph=phs[0].read_text(encoding='utf-8')
    lines=[x.strip() for x in (d/'kernel_info.txt').read_text().splitlines() if x.strip()]
    if not lines: errors.append(f'{k}: empty kernel_info'); continue
    top=lines[0]
    if not re.search(r'\b'+re.escape(top)+r'\s*\(',src): errors.append(f'{k}: top {top} missing')
    declared=[]; kinds={}
    for x in lines[1:]:
        f=[q.strip() for q in x.split(',')]
        if len(f)<3 or not re.fullmatch(r'L[1-9][0-9]*',f[0]): errors.append(f'{k}: malformed action {x}'); continue
        if f[1]=='loop' and len(f)==3 and int(f[2])>0:
            pass
        elif f[1]=='array' and len(f)>=5 and (len(f)-3)%2==0:
            pass
        else: errors.append(f'{k}: malformed action {x}'); continue
        declared.append(f[0]); kinds[f[0]]=f[1]
    expected=[f'L{i}' for i in range(1,len(declared)+1)]
    if declared!=expected: errors.append(f'{k}: labels must be contiguous absolute slots: {declared}')
    observed=re.findall(r'/\*\s*(L\d+)\s*:\s*\*/',src)
    if observed!=declared: errors.append(f'{k}: source labels {observed} != manifest {declared}')
    for L in declared:
        if kinds[L]=='loop':
            if ph.count(f'auto{{_PIPE_{L}}}')!=1: errors.append(f'{k}: PIPE count {L}')
            if ph.count(f'auto{{_UNROLL_{L}}}')!=1: errors.append(f'{k}: UNROLL count {L}')
        else:
            for kind in ('ARRAY_T','ARRAY_F','ARRAY_D'):
                if ph.count(f'auto{{_{kind}_{L}}}')!=1: errors.append(f'{k}: {kind} count {L}')
    bad=[]
    for line in src.splitlines():
        if re.match(r'^\s*#\s*pragma\s+HLS\s+(PIPELINE|UNROLL|ARRAY_PARTITION|DATAFLOW|INLINE)\b',line,re.I): bad.append(line.strip())
    if bad: errors.append(f'{k}: pre-existing optimization pragmas remain: {bad[:4]}')
    audit=json.loads((d/'ACTION_AUDIT.json').read_text())
    if audit['selected_action_count']>64: errors.append(f'{k}: >64 selected actions')
if errors:
    print('FAIL')
    print('\n'.join(' - '+e for e in errors)); sys.exit(1)
print('PASS: all three external MailoHLS source/action contracts are internally consistent.')
