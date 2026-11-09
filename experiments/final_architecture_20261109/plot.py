"""Deterministic figures from committed CPU receipts; no checkpoint loading."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/architecture_freeze/figures'
def main():
 D.mkdir(exist_ok=True);h=json.loads((ROOT/'experiments/final_architecture_20261109/results/adp_replay_summary.json').read_text())
 labels=['no_graph','compiler','hardware','target_film'];fig,ax=plt.subplots(figsize=(8,4))
 failures=[h[k]['selection_failure'] for k in labels];regrets=[h[k]['top1_regret'] for k in labels]
 ax.bar(labels,failures,label='Feasible-selection failure');ax.bar(labels,regrets,bottom=failures,label='Conditional ADP regret');ax.set_ylabel('Historical combined score components');ax.set_title('ADP predictor audit: 3 validation kernels / 2 families / 1 seed');ax.legend(fontsize=9);fig.tight_layout()
 for ext in ('png','svg'):fig.savefig(D/('historical_adp_components.'+ext),dpi=220)
 plt.close(fig)
 c=json.loads((ROOT/'experiments/final_architecture_20261109/results/target_contrasts.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
 cats=['latency_reversal','feasibility_boundary_proxy','no_change','near_tie','unsupported'];colors=['#205c82','#d29544','#75a38a','#afb5bc','#dddddd']
 for ax,split in zip(axes,['train','val']):
  base=[0]*3
  for cat,col in zip(cats,colors):
   vals=[c['counts'][split][axis].get(cat,0) for axis in ('budget','clock','device')];ax.bar(['Budget','Clock','Device'],vals,bottom=base,label=cat,color=col);base=[x+y for x,y in zip(base,vals)]
  ax.set_title(split+' (correlated views)');ax.set_ylabel('Matched contrast views')
 axes[1].legend(fontsize=7);fig.suptitle('Latency common-configuration audit; resource feasibility uses proxies');fig.tight_layout()
 for ext in ('png','svg'):fig.savefig(D/('target_contrast_coverage.'+ext),dpi=220)
 plt.close(fig)
if __name__=='__main__':main()
