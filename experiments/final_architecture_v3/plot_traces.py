"""Plots already-computed fixed-prefix measurements, with explicit scope."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=json.loads(a.input.read_text());a.out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');keys=list(r['controls']);axes[0].bar(range(len(keys)),[r['controls'][k]['mean_max_abs_candidate_score_change'] for k in keys]);axes[0].set_xticks(range(len(keys)),['Evidence zero','Matched shuffle','Gate off']);axes[0].set_ylabel('Mean maximum |RHS score change| (log-probability)');axes[0].set_title('33 unforced field comparisons; fixed preceding directives')
 fields=sorted({x['field'] for x in r['target_contrasts']});layers=[8,16,24,32];values=np.array([[np.mean([x['attention_mean_total_variation'] for x in r['target_contrasts'] if x['field']==f and x['layer']==l]) for l in layers] for f in fields]);im=axes[1].imshow(values,aspect='auto',vmin=0);axes[1].set_xticks(range(4),layers);axes[1].set_yticks(range(len(fields)),[x.replace('auto{_','').replace('}','') for x in fields]);axes[1].set_xlabel('Transformer layer');axes[1].set_title('Target contrast: mean attention total variation');fig.colorbar(im,ax=axes[1],label='Total variation [0,1]')
 fig.suptitle('Historical checkpoint150, two validation families; no physical-quality conclusion')
 for ext in ('png','svg'):fig.savefig(a.out/('fixed_prefix_evidence.'+ext),dpi=180)
 plt.close(fig)
if __name__=='__main__':main()
