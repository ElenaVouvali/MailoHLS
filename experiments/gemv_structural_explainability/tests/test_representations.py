"""Focused checks of frozen extraction, observation masks and saved evidence."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.gemv_structural_explainability.audit import authenticate,load_encoder,sha
from experiments.gemv_structural_explainability.diagnose_representations import (
    extract,mask_numeric,mask_relations,metadata_mask,numeric_columns,metrics,STAGES)
from experiments.paper_gnn.data import tensorize

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'representation_diagnostics_v1'


@pytest.fixture(scope='module')
def frozen():
    torch.set_num_threads(4)
    graph,_,_=authenticate()
    model,schema,_=load_encoder()
    tensor=tensorize(graph,schema['vocabulary'])
    rms=json.loads((OUT/'diagnostic_manifest.json').read_text())['normalization']['rms']
    return graph,model,schema,tensor,rms


def test_all_observed_layers_and_historical_memory(frozen):
    graph,model,schema,tensor,rms=frozen
    before={k:v.clone() for k,v in model.state_dict().items()}
    states=extract(model,tensor,rms)
    saved=np.load(OUT/'layerwise_embeddings.npz')
    for stage in STAGES:
        np.testing.assert_array_equal(states[stage][tensor.action_nodes],saved[stage])
    historical=torch.load(HERE/'results/action_memory.pt',weights_only=False,map_location='cpu')
    assert torch.equal(states['memory'][tensor.action_nodes],historical['action_memory'][:5])
    assert torch.equal(states['input'][tensor.action_nodes[0]],states['input'][tensor.action_nodes[1]])
    assert torch.equal(states['input'][tensor.action_nodes[3]],states['input'][tensor.action_nodes[4]])
    assert not torch.equal(states['conv1'][tensor.action_nodes[0]],states['conv1'][tensor.action_nodes[1]])
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_known_zero_rank_is_distinct_from_withheld_rank(frozen):
    graph,model,schema,tensor,rms=frozen
    index=sorted(graph.nodes).index('memory:gemv:op1:r0')
    value,known=numeric_columns(schema,['memory_rank'])
    assert tensor.x[index,value]==0 and tensor.x[index,known]==1
    masked=mask_numeric(tensor,schema,['memory_rank'],[index])
    assert masked.x[index,value]==0 and masked.x[index,known]==0
    assert tensor.x[index,known]==1


@pytest.mark.parametrize('name',['memory_shape','source_shape','affine_access','access_role',
                                'recurrence_detail','dependence_detail'])
def test_omitted_payloads_are_exact_negative_controls(frozen,name):
    graph,model,schema,tensor,rms=frozen
    changed=metadata_mask(graph,schema,name)
    for key in ('x','edge_attr','edge_index','action_nodes','action_slots'):
        assert torch.equal(getattr(changed,key),getattr(tensor,key))
    assert graph.nodes['memory:gemv:op3:r0']['memory_shape']==[32,32]


def test_relation_masks_remove_forward_and_reverse_evidence(frozen):
    graph,model,schema,tensor,rms=frozen
    changed=mask_relations(tensor,['loop_carried'])
    assert tensor.edge_index.shape[1]-changed.edge_index.shape[1]==6
    assert torch.equal(changed.x,tensor.x)
    assert changed.edge_attr[:,-4:].sum(1).eq(1).all()
    assert torch.equal(changed.action_slots,tensor.action_slots)
    for i in range(changed.edge_index.shape[1]):
        reverse=(changed.edge_index==changed.edge_index[:,i].flip(0)[:,None]).all(0)
        assert reverse.any()


def test_saved_metrics_and_provenance_are_reproducible():
    manifest=json.loads((OUT/'diagnostic_manifest.json').read_text())
    for filename,digest in manifest['files'].items():
        assert sha(OUT/filename)==digest
    assert manifest['encoder_state_unchanged'] and manifest['gemv_exact_historical_memory']
    assert not manifest['locked_test_labels_read'] and not manifest['gemv_training']
    assert manifest['device']=='cpu'
    arrays=np.load(OUT/'layerwise_embeddings.npz')
    recorded=json.loads((OUT/'layerwise_metrics.json').read_text())['stages']
    for stage in STAGES:
        actual=metrics(torch.from_numpy(arrays[stage]),['array']*3+['loop']*2)
        np.testing.assert_allclose(actual['cosine'],recorded[stage]['cosine'],atol=1e-12)
        np.testing.assert_allclose(actual['euclidean'],recorded[stage]['euclidean'],atol=1e-12)
    corpus=json.loads((OUT/'corpus_layerwise_metrics.json').read_text())['kernels']
    training={k['family'] for k in corpus if k['split']=='train'}
    development={k['family'] for k in corpus if k['split']=='val'}
    assert not training&development and len(corpus)==54
    probes=json.loads((OUT/'linear_probes.json').read_text())['results']
    assert all(not any(a.startswith('gemv:') for a in p.get('val_actions',[])) for p in probes)


def test_value_gates_and_pathway_controls_are_reported_separately():
    records=json.loads((OUT/'action_feature_ablations.json').read_text())['interventions']
    by_name={r['intervention']:r for r in records}
    for name in ('memory_rank','loop_depth','trip_counts','operand_result_use'):
        assert not by_name['gate_value_channel_'+name]['tensor_identical']
        assert 'knownness retained' in by_name['gate_value_channel_'+name]['semantics']
    for r in records:
        if r['tensor_identical']:
            assert max(r['stages']['memory']['delta_norm'])==0
    paths=json.loads((OUT/'training_path_sensitivity.json').read_text())
    assert len(paths['source_fields'])==13 and len(paths['candidates'])==71
    assert set(paths['conditions'])=={'full','graph_path_only_varies','summary_path_only_varies'}
