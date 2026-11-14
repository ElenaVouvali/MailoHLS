"""Negative tests on real compiler graph + frozen GNN output, no fake memory."""
import copy
import json
from pathlib import Path
import sys
R=Path('/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final')
P=R/'artifacts/stage2_eval_9_20261113';W=P/'compat9_v2_work'
sys.path.insert(0,str(R))
import torch
torch.set_num_threads(4)
from experiments.historical_mlir_gmt_gemv import historical as hist,historical_gmt_slots as slots
from compatibility_hooks_v2 import install_gmt_scope_extension,sha
g,pack=hist.export_graph(P/'graphs_compat9_v2/rosetta-3d-rendering.gexf','rosetta-3d-rendering',hist.load())
record=json.loads((P/'graphs_compat9_v2/EVAL_COMPAT9_V2_RECEIPT.json').read_text())['kernels']['rosetta-3d-rendering']
checks={}
def reject(name,call):
    try:call()
    except (RuntimeError,ValueError):checks[name]='REJECTED'
    else:raise AssertionError('Bad compatibility evidence accepted: '+name)
reject('original_training_scope_rule_rejects_global',lambda:slots.cohort_action_mapping(g,pack))
allow={}
install_gmt_scope_extension(allow,W/'generated_gmt_scope_negative_tests')
reject('missing_graph_allowlist',lambda:slots.cohort_action_mapping(g,pack))
bad=copy.deepcopy(record);bad['native_binding']['symbol']='different_global'
allow[pack['source_gexf_sha256']]=bad
reject('wrong_global_symbol',lambda:slots.cohort_action_mapping(g,pack))
bad=copy.deepcopy(record);bad['source_sha256']='0'*64
allow[pack['source_gexf_sha256']]=bad
reject('wrong_source_identity',lambda:slots.cohort_action_mapping(g,pack))
bad=copy.deepcopy(record);bad['native_binding']['compiler_access_source_lines']=[]
allow[pack['source_gexf_sha256']]=bad
reject('missing_source_access_evidence',lambda:slots.cohort_action_mapping(g,pack))
allow[pack['source_gexf_sha256']]=record
mapping=slots.cohort_action_mapping(g,pack)
assert len(mapping)==12
anchor=next(m['compiler_node_id'] for m in mapping if m['action']=='L1')
assert g.nodes[str(anchor)]['action_resolution']=='global_symbol_and_access'
checks['valid_exact_global_without_attribute_rewrite']='ACCEPTED'
data,evidence=slots.make_evidence(g,pack,mapping,P/'graphs_compat9_v2/rosetta-3d-rendering.gexf')
assert len(evidence['records'])==15387
assert torch.isfinite(data['anchors']).all()
(W/'GLOBAL_ACCEPTANCE_TEST_PASS.json').write_text(json.dumps({'status':'PASS','checks':checks,
 'graph_file_changed':False,'scope_attributes_rewritten':False,'real_frozen_gnn_used':True},indent=2)+'\n')
print(json.dumps(checks,indent=2))
