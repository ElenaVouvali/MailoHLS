"""Replay the wider-head control and compare new CPU results to sealed receipts."""
import json,torch
from .audit import ROOT,RUN,OUT,load,save
from experiments.v22_gnn_training.v4b.capacity_control import capacity_model
from experiments.v22_gnn_training.v4b.quality import apply_quality
from experiments.v22_gnn_training.v4b.evaluation import evaluate
from experiments.v22_gnn_training.v4b.train_probe import _pair_schedule,_json_sha

def main():
 torch.set_num_threads(4);p,protocol=load();apply_quality(p)
 schedule=_pair_schedule(p['groups'],1500,123,True)
 schedule_sha=_json_sha([[p['groups'][gi]['key'],f,i] for gi,f,i in schedule])
 assert schedule_sha==protocol['schedule']
 m,_,_=capacity_model(123,p['profile_feature_dim'],191667)
 ck=torch.load(RUN/'corrected_v3_capacity/compiler_capacity_selected.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state'])
 r=evaluate('compiler',m,p,torch.device('cpu'))
 replay=json.loads((OUT/'reaudit/historical_replay.json').read_text())
 original=json.loads((ROOT/'experiments/v22_gnn_training/v4b/receipts_v3/full37_validation.json').read_text())
 errors={}
 for arm,result in replay.items():
  old=original[arm]["validation"]
  errors[arm]={'selection_score_abs_error':abs(result['correct']['selection_score']-old['selection_score']),
    'selected_hash_disagreements':sum(x['selected_hash']!=y['selected_hash'] for x,y in zip(result['correct']['contexts'],old['contexts'])) if 'contexts' in old else None}
 rows=[r for g in p['groups'] if g['split']=='train' for r in g['rows']]
 positive=[r for r in rows if sum(r['physical'][1:])>0]
 all_y=torch.tensor([r['transformed'] for r in rows]);positive_y=torch.tensor([r['transformed'] for r in positive])
 save(OUT/'reaudit/supplement.json',{'capacity_replay':r,'replay_vs_historical':errors,'schedule_sha256':schedule_sha,
  'regression_label_sensitivity':{'all_train_n':len(rows),'excluding_allzero_n':len(positive),'all_mean_log_labels':all_y.mean(0).tolist(),'excluding_allzero_mean_log_labels':positive_y.mean(0).tolist(),'retrained':False},'test_labels_loaded':False})
 print(errors,'capacity',r['selection_score'])
if __name__=='__main__':main()
