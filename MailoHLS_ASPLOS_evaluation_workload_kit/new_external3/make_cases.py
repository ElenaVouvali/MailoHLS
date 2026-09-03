#!/usr/bin/env python3
import argparse, json
from pathlib import Path
KERNELS=['chstone-aes','chstone-jpeg','rosetta-3d-rendering']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device', required=True)
    p.add_argument('--clock', type=float, required=True)
    p.add_argument('--bram', type=int, required=True)
    p.add_argument('--dsp', type=int, required=True)
    p.add_argument('--ff', type=int, required=True)
    p.add_argument('--lut', type=int, required=True)
    p.add_argument('--objective', default='PARETO_ADP', choices=['PARETO_ADP','PARETO_LATENCY','PARETO_AREA'])
    p.add_argument('--frequency_mode', default='specified', choices=['specified','auto'])
    p.add_argument('--output', required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parent
    with open(a.output,'w',encoding='utf-8') as f:
        for k in KERNELS:
            ph=list((root/k).glob('kernel_placeholders.c'))+list((root/k).glob('kernel_placeholders.cpp'))
            if len(ph)!=1: raise FileNotFoundError(f'{k}: run materialize_upstream.py first')
            code=ph[0].read_text(encoding='utf-8')
            row={
              'context_id': f'external_{k}_{a.device}_{a.frequency_mode}_{a.clock:g}ns',
              'kernel_name': k, 'code': code, 'input': code,
              'obj_mode': a.objective, 'objective': a.objective,
              'device': a.device, 'frequency_mode': a.frequency_mode,
              'clock_period': a.clock, 'selected_clock_period': a.clock,
              'avail_bram': a.bram, 'avail_dsp': a.dsp,
              'avail_ff': a.ff, 'avail_lut': a.lut,
              'resource_budget_id': 'external_user_budget',
            }
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(f'Wrote {len(KERNELS)} cases -> {a.output}')
if __name__=='__main__': main()
