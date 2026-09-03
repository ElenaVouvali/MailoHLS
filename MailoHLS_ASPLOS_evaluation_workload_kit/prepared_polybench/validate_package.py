#!/usr/bin/env python3
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parent
KERNELS=['covariance','jacobi-2d','syr2k','trmm-opt']
errors=[]
for k in KERNELS:
    d=ROOT/k
    src=(d/'kernel.cpp').read_text()
    ph=(d/'kernel_placeholders.cpp').read_text()
    lines=[x.strip() for x in (d/'kernel_info.txt').read_text().splitlines() if x.strip()]
    declared=[]
    for x in lines[1:]:
        f=x.split(',')
        if len(f)!=3 or f[1]!='loop': errors.append(f'{k}: malformed/non-loop action {x}')
        else: declared.append(f[0])
    labels=re.findall(r'/\*\s*(L\d+)\s*:\s*\*/\s*for\s*\(',src)
    if labels != declared: errors.append(f'{k}: labels {labels} != manifest {declared}')
    if '#pragma ACCEL' in src or '#pragma ACCEL' in ph: errors.append(f'{k}: upstream ACCEL pragma remains')
    for L in declared:
        if ph.count(f'auto{{_PIPE_{L}}}') != 1: errors.append(f'{k}: PIPE placeholder count for {L}')
        if ph.count(f'auto{{_UNROLL_{L}}}') != 1: errors.append(f'{k}: UNROLL placeholder count for {L}')
    observed=set(re.findall(r'auto\{_(?:PIPE|UNROLL)_(L\d+)\}',ph))
    if observed != set(declared): errors.append(f'{k}: unexpected placeholder labels {sorted(observed)}')
if errors:
    print('FAIL')
    print('\n'.join(' - '+e for e in errors))
    sys.exit(1)
print('PASS: all four MailoHLS source/action contracts are internally consistent.')
