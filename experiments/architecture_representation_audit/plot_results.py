#!/usr/bin/env python3
"""Rebuild compact GNN diagnostic figures from the checked-in manifest."""
import json, re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'results/gnn_extraction_manifest.json').read_text())
fig=ROOT/'figures'; fig.mkdir(parents=True,exist_ok=True)
groups={}
for row in data['diagnostic_feature_group_interventions']:
    groups.setdefault(row['group'],[]).append(row['mean_action_l2'])
order=['action_demand_summary','action_semantic_summary','trip_count_and_loop_depth','compiler_edge_features','array_shape_and_volume']
labels=['Action demand','Action semantic','Loop depth + trip count','Compiler edge features','Array shape + volume']
fig1,ax=plt.subplots(figsize=(8.2,4.2))
vals=[groups[k] for k in order]
ax.boxplot(vals,tick_labels=labels,showfliers=False,patch_artist=True,
           boxprops={'facecolor':'#b9d6ee'},medianprops={'color':'#173b5e','linewidth':1.5})
ax.set_ylabel('Per-graph mean action embedding L2 change')
ax.set_title('Frozen V4b response to diagnostic input-group zeroing')
ax.tick_params(axis='x',rotation=22); ax.grid(axis='y',alpha=.25)
fig1.text(.01,.01,'In-memory feature ablations; not source transformations or evidence of decision benefit.',fontsize=8)
fig1.tight_layout(rect=(0,.08,1,1)); fig1.savefig(fig/'gnn_input_sensitivity.png',dpi=170); plt.close(fig1)

def count(row):
    if row.get('evidence_overflow'):
        m=re.search(r'overflow: (\d+) selected',row['evidence_overflow']); return int(m.group(1)) if m else None
    return row.get('evidence_count')
graphs=data['graphs']; y=[count(x) for x in graphs]
fig2,ax=plt.subplots(figsize=(9,4.2)); colors=['#cf5c36' if x>128 else '#3c8d73' for x in y]
ax.scatter(range(len(y)),y,c=colors,s=25,alpha=.85,edgecolors='none'); ax.axhline(128,color='#252525',ls='--',lw=1,label='Current C cap = 128')
ax.set_yscale('log'); ax.set_xlabel('Authenticated measured graph (manifest order)'); ax.set_ylabel('Selected typed evidence records (log scale)')
ax.set_title('Memory-C capacity coverage across 55 exact-source RC3 graphs'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
fig2.text(.01,.01,'Green: build fits; orange: fail-closed overflow. No truncation was applied.',fontsize=8)
fig2.tight_layout(rect=(0,.08,1,1)); fig2.savefig(fig/'memory_c_coverage.png',dpi=170); plt.close(fig2)
print('wrote',fig/'gnn_input_sensitivity.png',fig/'memory_c_coverage.png')
