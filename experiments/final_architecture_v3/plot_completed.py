"""Plot recorded optimizer/validation metrics; no smoothed or invented outcomes."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--static',type=Path,required=True);p.add_argument('--preferences',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 state=json.loads((a.static/'checkpoint-150/trainer_state.json').read_text());loss=[r for r in state['log_history'] if 'loss' in r];val=[json.loads((a.static/f'val_selection_step_{s}.json').read_text()) for s in (0,50,100,150)];diag=rows(a.static/'xattn_diagnostics.jsonl');train={n:rows(a.preferences/n/'training.jsonl') for n in ('sft','dpo')}
 plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(2,3,figsize=(13,7))
 axes[0,0].plot([r['step'] for r in loss],[r['loss'] for r in loss],label='train');axes[0,0].plot([r['step'] for r in val],[r['eval_loss'] for r in val],marker='o',label='validation');axes[0,0].set_title('Corrected action-normalized loss');axes[0,0].legend()
 axes[0,1].plot([r['step'] for r in val],[r['mean_decision_site_accuracy'] for r in val],marker='o');axes[0,1].axvline(50,color='gray',linestyle=':');axes[0,1].set_title('Constrained validation decision accuracy')
 for layer in sorted({r['layer'] for r in diag}):
  points=[r for r in diag if r['layer']==layer];label='layer '+str(int(layer.split('.layers.')[1].split('.')[0])+1)
  axes[0,2].plot([r['step'] for r in points],[abs(r['tanh_gate']) for r in points],label=label)
  axes[1,0].plot([r['step'] for r in points],[r['gated_residual_hidden_rms_ratio'] for r in points],label=label)
  axes[1,1].plot([r['step'] for r in points],[max(r['gradient_rms'].values()) for r in points],label=label)
 axes[0,2].set_title('Effective gate magnitude');axes[0,2].legend();axes[1,0].set_title('Structural residual / hidden RMS');axes[1,1].set_title('Maximum Q/K/V/output gradient RMS')
 for name,data in train.items():axes[1,2].plot([r['step'] for r in data],[r['loss'] for r in data],label=name)
 axes[1,2].set_title('Matched Stage3 objectives (different scales)');axes[1,2].legend()
 for ax in axes.flat:ax.set_xlabel('Successful optimizer update');ax.grid(alpha=.2)
 fig.suptitle('Recorded bounded pilots; validation agreement is not measured QoR');fig.tight_layout();fig.savefig(a.out/'training_curves.png',dpi=180);fig.savefig(a.out/'training_curves.svg');plt.close(fig)
 with (a.out/'xattn_learning_curves.csv').open('w') as f:
  w=csv.DictWriter(f,['step','eval_loss','mean_decision_site_accuracy','candidate_margin_mean']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in val)
 with (a.out/'xattn_gate_stats.csv').open('w') as f:
  keys=['step','layer','tanh_gate','gated_residual_hidden_rms_ratio','attention_entropy'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in diag)
 with (a.out/'xattn_gradients.csv').open('w') as f:
  keys=['step','layer','q','k','v','out'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({'step':r['step'],'layer':r['layer'],**r['gradient_rms']} for r in diag)
if __name__=='__main__':main()
