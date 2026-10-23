#!/usr/bin/env python3
"""Combine the prepared ZCU winners and 21 U50 zero-shot completions.

The command intentionally refuses duplicate directive signatures. Replace each
duplicate with a legal local neighbor before invoking the Coroni runner.
"""
import argparse, hashlib, json
from pathlib import Path

def signature(target):
    lines=tuple(x.strip().upper() for x in target.splitlines() if x.strip())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zcu-candidates',type=Path,required=True);ap.add_argument('--zero-shot',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    zcu=json.loads(a.zcu_candidates.read_text()); zero=[]
    for line in a.zero_shot.read_text().splitlines():
        if line.strip(): zero.append(json.loads(line))
    by_kernel={r.get('kernel_name',r.get('kernel')):r for r in zero}
    if len(by_kernel)!=21: raise ValueError(f'Expected 21 unique zero-shot kernels, got {len(by_kernel)}')
    rows=[]
    for r in zcu:
        rows.append({'kernel_name':r['kernel'],'origin':r['origin'],'device':'xcu50-fsvh2104-2-e','clock_period_ns':10.0,'target':r['target']})
    for kernel,r in sorted(by_kernel.items()):
        target=r.get('target') or r.get('canonical_prediction')
        if not target and r.get('candidates'):
            index=int(r.get('best_unique_sample_id_by_model_score', 0))
            selected=[c for c in r['candidates'] if int(c.get('sample_id', -1)) == index]
            if selected: target=selected[0].get('canonical_prediction') or selected[0].get('prediction')
        if not target: raise ValueError(f'Missing target for zero-shot {kernel}')
        rows.append({'kernel_name':kernel,'origin':'u50_zero_shot_pareto_latency','device':'xcu50-fsvh2104-2-e','clock_period_ns':10.0,'target':target})
    if len(rows)!=84: raise ValueError(f'Expected 84 rows before train/val selection, got {len(rows)}')
    seen={}; duplicates=[]
    for r in rows:
        key=(r['kernel_name'],signature(r['target']))
        if key in seen: duplicates.append({'kernel_name':r['kernel_name'],'origins':[seen[key],r['origin']]})
        seen[key]=r['origin']
    if duplicates: raise ValueError('Duplicate signatures require legal local neighbors: '+json.dumps(duplicates))
    train={'machsuite-gemm-blocked','machsuite-md-knn','machsuite-spmv-ellpack','machsuite-stencil2d','machsuite-stencil3d','rodinia-backprop-0-baseline-forward','rodinia_cfd_flux_0_baseline_0','rodinia_cfd_step_factor_4_doublebuffer_0','rodinia_dilate_2_pipeline_0','rodinia-hotspot-2-pipeline','rodinia-kmeans-1-tiling','rodinia-knn-4-doublebuffer','rodinia_lavaMD_2_pipeline_0','rodinia_lc_gicov_0_baseline_0','rodinia_lud_1_tiling_0','rodinia_streamcluster_3_doublebuffer_0'}
    val={'machsuite-sort-radix','machsuite-viterbi','rodinia_pathfinder_0_baseline_0','rodinia_pathfinder_4_doublebuffer_0'}
    out=[r for r in rows if r['kernel_name'] in train|val]
    if len(out)!=80: raise ValueError(f'Expected 80 train/val rows, got {len(out)}')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in out));print(f'wrote {len(out)} rows to {a.output}')
if __name__=='__main__':main()
