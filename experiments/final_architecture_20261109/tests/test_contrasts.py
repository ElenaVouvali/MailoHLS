from experiments.final_architecture_20261109.prepare import compare

def row(latency,bram=1):return {'physical':[latency,bram,1,1,1]}
def test_reversal_requires_both_candidates_measured_in_both_contexts():
 a={'a':row(10),'b':row(12)};b={'a':row(15),'b':row(10)}
 assert compare(a,b,[10]*4,[10]*4)['status']=='latency_reversal'
 assert compare({'a':row(10)},{'b':row(10)},[10]*4,[10]*4)['status']=='unsupported'
def test_budget_changes_eligibility_not_physical_latency():
 a={'a':row(10,9),'b':row(20,2)}
 assert compare(a,a,[10]*4,[3]*4)['status']=='feasibility_boundary_proxy'
 assert a['a']['physical'][0]==10
 assert compare(a,a,[10]*4,[10]*4)['status']=='no_change'
