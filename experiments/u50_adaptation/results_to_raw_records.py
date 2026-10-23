#!/usr/bin/env python3
"""Convert Coroni synthesis results to records for assemble_raw_csv.py."""
import argparse, json, re
from pathlib import Path

ASSIGN=re.compile(r"^auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}\s*=\s*(\S+)$",re.I)

def apl_map(path):
    out={}
    for line in path.read_text().splitlines():
        bits=[x.strip() for x in line.split(',')]
        if len(bits)==2 and bits[1].startswith('L'): out[bits[1].upper()]=bits[0]
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--source-csv-dir',type=Path,required=True);ap.add_argument('--repo',type=Path,default=Path('/home/elvouvali/MailoHLS/MailoHLS-next'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    records=[]
    for line in a.results.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('status')!='success': raise ValueError(f"Synthesis failed: {r.get('case_id')}")
        assigns={}
        for raw in r['target'].splitlines():
            m=ASSIGN.match(raw.strip())
            if not m: raise ValueError(f"Malformed target in {r['case_id']}: {raw}")
            assigns[m.group(1).upper()]=m.group(2).lower()
        amap=apl_map(a.repo/'Data/ApplicationAPLMapping'/f"{r['kernel_name']}.txt")
        directives={}
        for action,column in amap.items():
            if f'_PIPE_{action}' in assigns or f'_UNROLL_{action}' in assigns:
                pipe=assigns.get(f'_PIPE_{action}','0'); unroll=assigns.get(f'_UNROLL_{action}','0')
                directives[column]='pipeline' if pipe=='1' and unroll=='0' else ('unroll_'+unroll if unroll!='0' else 'NDIR')
            elif f'_ARRAY_T_{action}' in assigns:
                typ=assigns[f'_ARRAY_T_{action}']; fac=assigns.get(f'_ARRAY_F_{action}','0'); dim=assigns.get(f'_ARRAY_D_{action}','0')
                directives[column]='NDIR' if typ=='none' else (f'{typ}_{fac}_{dim}' if typ!='complete' else f'complete_{dim}')
        records.append({'kernel':r['kernel_name'],'origin':r['origin'],'directives':directives,
                        'latency_cycles':r['latency_cycles'],'resources':r['resources']})
    if len(records)!=80: raise ValueError(f'Expected 80 results, got {len(records)}')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))

if __name__=='__main__':main()
