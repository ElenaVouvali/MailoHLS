"""Plot the 18 measured request cells from an authenticated synthesis summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
rows = json.loads((a.root/'request_results.json').read_text())
assert len(rows) == 18 and all(r['speedup'] is not None for r in rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout='constrained')
norm = Normalize(vmin=1, vmax=max(r['speedup'] for r in rows))
for ax, device, prefix in zip(axes, ('ZCU104', 'U200'), ('xczu', 'xcu')):
    selected = [r for r in rows if r['part'].startswith(prefix)]
    lookup = {(r['clock_ns'], r['budget_pct']): r for r in selected}
    values = np.array([[lookup[c,b]['speedup'] for b in (20,50,100)] for c in (3.33,5.,10.)])
    im = ax.imshow(values, cmap='YlGnBu', norm=norm, aspect='auto')
    for i, clock in enumerate((3.33,5.,10.)):
        for j, budget in enumerate((20,50,100)):
            r = lookup[clock,budget]
            marker = '†' if not r['prediction']['hls_estimated_timing_met'] else ''
            if not r['budget_met']: marker += '‡'
            color = 'white' if norm(r['speedup']) > .57 else '#111827'
            ax.text(j,i,'{:.2f}×{}\n{:.2f}% util.'.format(r['speedup'],marker,r['mean_normalized_utilization_pct']),
                    ha='center',va='center',fontsize=11,color=color,fontweight='medium')
    ax.set(xticks=range(3),xticklabels=['20%','50%','100%'],yticks=range(3),yticklabels=['3.33','5','10'],
           xlabel='Resource budget (fraction of device capacity)',ylabel='Requested clock period (ns)',title=device)
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3)
    ax.tick_params(which='minor',bottom=False,left=False)
fig.colorbar(im,ax=axes,label='Speedup vs. matched pragma-free baseline (cycles ratio)',shrink=.78)
fig.suptitle('GAN · Stage-1 FiLM ON · Vitis HLS 2021.1',fontsize=16)
fig.supxlabel('Mean utilization = average BRAM18K / DSP / FF / LUT usage relative to full device capacity.\n'
              '† HLS estimated clock exceeds request.  ‡ At least one resource exceeds the requested budget.\n'
              'Nominal speedup uses matched requested periods; no post-route timing or functional simulation.',fontsize=9)
fig.savefig(a.root/'request_map.png',dpi=180)
fig.savefig(a.root/'request_map.pdf')
print(a.root/'request_map.png')
