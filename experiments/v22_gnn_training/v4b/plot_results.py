"""Render recorded training curves; no training or checkpoint selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-run',required=True);p.add_argument('--small-run',required=True)
    p.add_argument('--output-prefix',required=True);a=p.parse_args()
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,(label,path) in enumerate((('37 training kernels',a.full_run),('14 training kernels',a.small_run))):
        data=json.loads((Path(path)/'run_receipt.json').read_text())['training']
        for arm,report in data.items():
            h=report['history'];steps=[r['step'] for r in h]
            for col,key in enumerate(('physical_qor_loss','pairwise_loss')):
                values=np.array([r[key] if key!='pairwise_loss' or not r.get('pointwise_only') else np.nan for r in h])
                window=np.ones(100)
                mean=np.convolve(np.nan_to_num(values),window,'valid')/np.maximum(1,np.convolve(np.isfinite(values),window,'valid'))
                axes[row,col].plot(steps[99:],mean,label=arm)
            checkpoints=[r for r in h if 'validation' in r]
            axes[row,2].plot([r['step'] for r in checkpoints],[r['validation']['selection_score'] for r in checkpoints],marker='o',label=arm)
        for col,title in enumerate(('Physical SmoothL1 (rolling 100)','Pair logistic loss (rolling 100)','Validation failure + regret')):
            axes[row,col].set_title(label+'\n'+title);axes[row,col].set_xlabel('Update');axes[row,col].grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('V4b corrected v3 — reported-positive-ADP measured subpool; one seed, two validation families')
    fig.tight_layout()
    for suffix in ('.png','.pdf'):fig.savefig(a.output_prefix+suffix,dpi=160)


if __name__=='__main__':main()
