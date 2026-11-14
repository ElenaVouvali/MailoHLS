"""FINAL_ARTIFACTS lineage and real trained compiler-memory checks."""
import json
from pathlib import Path
import torch
import pytest

from experiments.gemv_structural_explainability.final_artifacts_refinement.export import SHA,sha
from experiments.gemv_structural_explainability.final_artifacts_refinement.model import SemanticMemoryAlignment
from experiments.gemv_structural_explainability.final_artifacts_refinement.facts import FACT_NAMES,NUMERIC

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'artifacts/gemv_structural_explainability/final_artifacts_cohort_v1'
RUN=ROOT/'artifacts/gemv_structural_explainability/final_artifacts_training_v1'


@pytest.fixture(scope='module')
def trained():
    torch.set_num_threads(4)
    pack=torch.load(RAW/'packs/gemv.pt',map_location='cpu',weights_only=False)
    completion=json.loads((RUN/'completion.json').read_text())
    assert completion['selected_updates']>0 and completion['gnn_checkpoint_sha256']==SHA
    assert sha(RUN/'selected.pt')==completion['selected_checkpoint_sha256']
    state=torch.load(RUN/'selected.pt',map_location='cpu',weights_only=False)
    model=SemanticMemoryAlignment();model.load_state_dict(state['model'],strict=True);model.eval()
    return pack,model


def test_correct_selected_lineage_and_corpus_authentication():
    final=json.loads((ROOT/'FINAL_ARTIFACTS.json').read_text())
    assert final['gnn']['checkpoint'].endswith('gnn_final_absolute_direct_rank_s123/run1/val_structural_rank_model_state_dict.pth')
    manifest=json.loads((RAW/'manifest.json').read_text())
    assert manifest['checkpoint_sha256']==SHA
    assert len(manifest['packs'])==55
    assert not manifest['gnn_retrained'] and not manifest['gemv_fitted']
    assert manifest['static_tensor_tree_sha256']=='75d2f40816ec47dadf8c7767d5e7fc4530289f79066e306f09478b3d088d53d7'
    for kernel,r in manifest['packs'].items():
        if kernel!='gemv':assert r['production_memory_max_abs_delta']<2e-5


def test_explicit_rank_map_and_scalarization_facts(trained):
    pack,_=trained
    def fact(node,name):
        i=next(i for i,r in enumerate(pack['records']) if r.get('node')==node)
        j=FACT_NAMES.index(name+'_value')
        return pack['facts'][0,i,j:j+2]
    for node,rank in [('memory:gemv:op3:r0',2),('memory:gemv:op2:r0',1),('memory:gemv:op1:r0',0)]:
        value,known=fact(node,'memory_rank')
        assert known==1
        assert float(value)==pytest.approx(__import__('math').log2(1+rank)/16)
    assert fact('memory:gemv:op1:r0','source_rank')[1]==1
    assert fact('memory:gemv:op1:r0','scalarized')[0]>0


def test_trained_role_routes_and_absolute_action_prefix(trained):
    pack,model=trained
    with torch.no_grad():out=model(pack)
    assert torch.equal(out['memory'][:,:64],pack['action'])
    assert not out['mask'][0,5:64].any()
    n=pack['a'].shape[1]
    assert out['mask'][0,64+n:64+n+10].all()
    for k in range(5):
        for role in range(2):
            weights=out['summary_attention'][0,k,:,role]
            legal=pack['routes'][0,k]&pack['valid'][0]&(pack['is_edge'][0] if role else ~pack['is_edge'][0])
            assert not weights[:,~legal].any()
            torch.testing.assert_close(weights.sum(-1),torch.ones(4))
            assert legal.any()


def test_evidence_basis_is_explicit_and_resolved_for_every_gemv_route(trained):
    pack,_=trained
    for k in range(5):
        for i in pack['routes'][0,k].nonzero().flatten().tolist():
            for proof in pack['records'][i]['gnn_basis'].values():
                assert proof['rule']!='unresolved' and proof['legacy_nodes']
    assert pack['node_ids'][:5]==[23,43,18,28,3]


def test_information_objective_keeps_nonzero_gradients(trained):
    pack,model=trained
    model.zero_grad(set_to_none=True)
    out=model(pack);out['auxiliary_loss'].backward()
    for parameter in (model.queries,model.fact_projection[0].weight,model.edge[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0
