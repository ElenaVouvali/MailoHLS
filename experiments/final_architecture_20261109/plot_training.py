"""Plot actual completed static-C curves; no model loading."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 rows=list(csv.DictReader((ROOT/'experiments/final_architecture_20261109/results/static_training_curve.csv').open()));steps=[int(r['step']) for r in rows];fig,axes=plt.subplots(1,3,figsize=(11,3.4))
 for ax,key,title in zip(axes,['teacher_forced_eval_loss','macro_decision_accuracy','candidate_margin'],['Teacher-forced validation loss','Family-macro decision accuracy','Mean candidate margin']):
  ax.plot(steps,[float(r[key]) for r in rows],marker='o',color='#255f85');ax.set_title(title,fontsize=10);ax.set_xlabel('Successful optimizer updates');ax.grid(alpha=.2)
 fig.suptitle('Static C: 12 correlated requests / 2 validation families; physical QoR pending',fontsize=11);fig.tight_layout();d=ROOT/'docs/architecture_freeze/figures'
 for ext in ('png','svg'):fig.savefig(d/('static_C_training.'+ext),dpi=220)
if __name__=='__main__':main()
