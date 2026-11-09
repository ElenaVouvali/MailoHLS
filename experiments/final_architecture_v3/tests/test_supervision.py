from types import SimpleNamespace
import pytest
from experiments.final_architecture_v3.supervision import corrected, preserve_references
from LLM_branch.train.stage1_decision_loss import apply_decision_policy

class Tokens:
 def __call__(self,text,**kwargs):
  return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}

def test_action_field_value_mass_and_singleton_exclusion():
 keys=['AUTO{_PIPE_L2}','AUTO{_UNROLL_L2}','AUTO{_ARRAY_T_L7}','AUTO{_ARRAY_F_L7}']
 values=['0','16','cyclic','2']; ids=[]; sites=[]; labels=[]
 for k,v in zip(keys,values):
  encoded=[ord(c) for c in v+'\n']; ids+=encoded; sites += [k]*len(encoded);labels+=encoded if k!=keys[-1] else [-100]*len(encoded)
 p=SimpleNamespace(input_ids=ids,labels=labels,token_weights=[1.]*len(ids),xattn_target_mask=[1]*len(ids),site_keys=sites)
 before=list(p.input_ids);apply_decision_policy(p,Tokens(),dict(zip(keys,values)),'decision')
 masses={k:sum(w for s,w in zip(sites,p.token_weights) if s==k) for k in keys}
 assert masses==pytest.approx({keys[0]:.25,keys[1]:.25,keys[2]:.5,keys[3]:0.})
 assert sum(p.token_weights)==pytest.approx(1.) and p.input_ids==before
 assert all(w==0 for token,w in zip(ids,p.token_weights) if token==ord('\n'))

def test_keep_repeated_complete_exposures():
 rows=[{'_quality_arm':'multi_reference','target':v} for v in ('a','b','a')]
 selected,_=preserve_references(rows,goal_mode='PARETO_LATENCY')
 assert [r['target'] for r in selected]==['a','b','a']
 with pytest.raises(ValueError):preserve_references([{'target':'invented'}])

def test_scoped_patch_restored_on_failure():
 a,b=object(),object();p=SimpleNamespace(build_deterministic_rhs_pack=a,select_goal_rows=b)
 with pytest.raises(RuntimeError):
  with corrected(p):raise RuntimeError('deliberate')
 assert p.build_deterministic_rhs_pack is a and p.select_goal_rows is b
