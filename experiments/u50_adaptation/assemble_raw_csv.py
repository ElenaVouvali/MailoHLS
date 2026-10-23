#!/usr/bin/env python3
"""Materialize validated U50 measurements into GNOSIS-compatible raw CSVs.

The input JSONL is intentionally explicit: each row must contain a unique
kernel, four-way provenance label, directive columns using canonical spellings,
worst-case latency cycles, and absolute post-synthesis resources. This keeps
measurement collection separate from preprocessing and makes accidental use of
the held-out final kernels impossible.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

RESOURCES = {"BRAM_18K": "BRAM_Utilization_percentage", "DSP": "DSP_Utilization_percentage",
             "FF": "FF_Utilization_percentage", "LUT": "LUT_Utilization_percentage"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--records',type=Path,required=True)
    ap.add_argument('--source-csv-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--specs',type=Path,required=True); ap.add_argument('--expected-count',type=int,default=84); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    specs=json.loads(a.specs.read_text()); specs=specs.get('devices',specs); part, caps=next(iter(specs.items()))
    rows=[json.loads(x) for x in a.records.read_text().splitlines() if x.strip()]
    final={'GAN','chstone-jpeg','Kalman','covariance','serrano-kalman-filter'}
    seen=set(); out=[]; audit=[]
    for index,r in enumerate(rows,1):
        k=r.get('kernel'); key=(k,str(r.get('origin')),json.dumps(r.get('directives',{}),sort_keys=True))
        if k in final: raise ValueError(f'held-out final kernel supplied: {k}')
        if key in seen: raise ValueError(f'duplicate directive measurement: {k}/{r.get("origin")}')
        seen.add(key)
        source=a.source_csv_dir/f'{k}.csv'
        if not source.is_file(): raise FileNotFoundError(source)
        frame=pd.read_csv(source,nrows=0); cols=list(frame.columns)
        required={'Version','Device','Clock_Period_nsec','Latency_msec',*RESOURCES.values()}
        missing=required-set(cols)
        if missing: raise ValueError(f'{k}: source header missing {sorted(missing)}')
        cycles=float(r['latency_cycles']);
        if cycles<=0: raise ValueError(f'{k}: invalid latency cycles')
        resources=r['resources']
        vals={'Version':f'u50_adapt_{index:04d}_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]}',
              'Device':part,'Clock_Period_nsec':10.0,'Latency_msec':cycles*10.0/1e6}
        for resource,column in RESOURCES.items():
            absolute=float(resources[resource]); denominator=float(caps[resource]); pct=100*absolute/denominator
            if absolute<0 or pct>=100: raise ValueError(f'{k}: invalid {resource}={absolute}/{denominator}')
            vals[column]=pct
        directives=r['directives'];
        for col in cols:
            if col in required or col in {'Synthesis_Time','Synthesis_Time_sec'}: continue
            vals[col]=directives.get(col,'NDIR')
        out.append(pd.DataFrame([{c:vals.get(c,'NDIR') for c in cols}],columns=cols)); audit.append({'kernel':k,'origin':r['origin'],'version':vals['Version']})
    if len(rows)!=a.expected_count: raise ValueError(f'Expected exactly {a.expected_count} measurements, got {len(rows)}')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for k in sorted({r['kernel'] for r in rows}):
        if (a.output_dir/f'{k}.csv').exists() and not a.force: raise FileExistsError(a.output_dir/f'{k}.csv')
        pieces=[x for x,r in zip(out,rows) if r['kernel']==k]
        pd.concat(pieces,ignore_index=True).to_csv(a.output_dir/f'{k}.csv',index=False)
    (a.output_dir/'assembly_manifest.json').write_text(json.dumps({'schema':'mailohls-u50-raw-assembly-v1','part':part,'clock_period_ns':10.0,'measurements':audit},indent=2)+'\n')

if __name__=='__main__': main()
