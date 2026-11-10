import copy
import pytest
from experiments.refinement import configurations as c,synthesis as hls
from experiments.refinement_search import controller
from experiments.refinement_search.deployment import optimize,physical_identity
from LLM_branch.common.directive_domains import SourceAction

def fixture():
 actions={'L9':SourceAction('loop',trip_count=8)}
 base={'AUTO{_PIPE_L9}':'0','AUTO{_UNROLL_L9}':'2'}
 native={'AUTO{_PIPE_L9}':['0','1'],'AUTO{_UNROLL_L9}':['0','1','2','4','8']}
 problem={'baseline':base,'actions':actions,'native_domains':native,'relations':[],'inactive_actions':[]}
 request={'top':'test_top','objective':'PARETO_LATENCY','source_sha256':'a'*64,'headers_sha256':{},'action_metadata_sha256':'b'*64,'domain_identity':'tcad','part':'xczu7ev-ffvc1156-2-e','clock_ns':10.,'capacities':dict(c.CAPACITIES),'budgets':dict(c.CAPACITIES),'tool_identity':'synthetic-test-only','flow_settings':[]}
 return request,problem

def test_disabled_never_queries_or_validates_expensive_source():
 assignment={'an':'unchanged input'}
 r=optimize({},assignment,refinement_enabled=False)
 assert r['assignment']==assignment and r['logical_queries']==0

def test_latency_uses_cycles_not_adp():
 a={'synthesis_status':'success','feasible':True,'rankable':True,'objective':'PARETO_LATENCY','latency_cycles':100,'ADP':1.}
 b={**a,'latency_cycles':80,'ADP':100.}
 assert controller._best([{'measurement':a},{'measurement':b}])['measurement'] is b
 assert hls.objective_value({'ADP':3.})==3.

def test_causal_budget_resume_and_incumbent(tmp_path):
 request,problem=fixture();calls=[];report=tmp_path/'report.txt';report.write_text('SYNTHETIC UNIT TEST; never hardware evidence')
 def oracle(identity):
  calls.append(copy.deepcopy(identity));v=100 if len(calls)==1 else 120
  return {'physical_identity':identity,'measurement':{'latency_cycles':v,'resources':{k:1 for k in c.RESOURCE_KEYS},'estimated_clock_ns':8.,'synthesis_status':'success'},'report_sha256':{str(report):c.sha256(report)}}
 args=dict(problem=problem,refinement_enabled=True,refinement_budget=4,run_dir=tmp_path/'run',oracle=oracle,stage3_sha256='c'*64)
 r=optimize(request,problem['baseline'],**args)
 assert r['logical_queries']==4 and len(calls)==4 and r['assignment']==problem['baseline'] and r['objective_value']==100
 optimize(request,problem['baseline'],**args)
 assert len(calls)==4
 with pytest.raises(ValueError):optimize({**request,'clock_ns':5.},problem['baseline'],**args)

def test_unknown_or_missed_clock_never_ranked(tmp_path):
 request,problem=fixture();report=tmp_path/'report';report.write_text('synthetic')
 def oracle(identity):return {'physical_identity':identity,'measurement':{'latency_cycles':1,'resources':{k:1 for k in c.RESOURCE_KEYS},'estimated_clock_ns':11.,'synthesis_status':'success'},'report_sha256':{str(report):c.sha256(report)}}
 r=optimize(request,problem['baseline'],problem=problem,refinement_enabled=True,refinement_budget=1,run_dir=tmp_path/'run',oracle=oracle,stage3_sha256='d'*64)
 assert r['status']=='unverified' and r['objective_value'] is None

def test_normal_inference_without_refinement_has_exact_assignment_parity(tmp_path,monkeypatch):
 import json,sys
 from LLM_branch.inference import optimize as entry
 parent=tmp_path/'stage3';parent.mkdir();(parent/'structural_xattn.pt').write_bytes(b'synthetic-only')
 (parent/'training_contract.json').write_text(json.dumps({'stage3_preference':{'qualification':'QUALIFIED'}}))
 prediction='auto{_PIPE_L9} = 0\nauto{_UNROLL_L9} = 2';captured=[]
 def fake_prediction(**kwargs):return {'candidates':[{'prediction':prediction}]}
 def fake_main():captured.append(entry.decoder.predict_case())
 monkeypatch.setattr(entry.decoder,'predict_case',fake_prediction);monkeypatch.setattr(entry.decoder,'main',fake_main)
 monkeypatch.setattr(sys,'argv',['optimize','--adapter_dir',str(parent)])
 entry.main()
 assert captured[0]['candidates'][0]['prediction']==prediction
 assert captured[0]['final_assignment']=={'AUTO{_PIPE_L9}':'0','AUTO{_UNROLL_L9}':'2'}
 assert captured[0]['refinement']['logical_queries']==0
 assert entry.decoder.predict_case is fake_prediction
