"""Checks run on authenticated measured checkpoint outputs, not random states."""
from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch

from experiments.gemv_structural_explainability.audit import (
    authenticate,load_encoder,export_pack,GNN_SHA,ROOT,sha)

HERE=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def exported():
    torch.set_num_threads(4)
    graph,relations,actions=authenticate()
    model,schema,identity=load_encoder()
    pack,first,jk=export_pack(graph,model,schema)
    return graph,model,pack,actions,identity


def test_trained_checkpoint_and_exact_saved_vectors(exported):
    graph,model,pack,actions,identity=exported
    assert identity['sha256']==GNN_SHA and identity['updates']==4736
    assert identity['strict_state_loading'] and not identity['production_qualified']
    assert len(model.conv_layers)==3
    saved=torch.load(HERE/'results/action_memory.pt',map_location='cpu',weights_only=False)
    assert torch.equal(pack['action'][0],saved['action_memory'])
    assert saved['tensor_schema']==identity['tensor_schema']
    assert all(not p.requires_grad for p in model.parameters())


def test_source_compiler_and_absolute_slots(exported):
    graph,model,pack,actions,identity=exported
    assert [r['slot'] for r in actions]==list(range(5))
    assert [r['construct'] for r in actions]==['A','x','y','outer loop','inner reduction loop']
    assert [r['line'] for r in actions]==[5,7,9,12,16]
    assert actions[2]['compiler_metadata']['memory_shape']==[]
    assert actions[2]['source_metadata']['source_array_dimensions']==[32]
    assert pack['active'][0,:5].all() and not pack['active'][0,5:].any()
    assert not pack['action'][0,5:].any()
    assert torch.isfinite(pack['action']).all()


def test_csv_npz_and_metric_consistency():
    saved=np.load(HERE/'results/action_embeddings.npz')
    rows=list(csv.DictReader((HERE/'results/active_action_embeddings.csv').open()))
    actual=np.array([[float(row['d'+str(k)]) for k in range(128)] for row in rows])
    np.testing.assert_array_equal(actual,saved['active_vectors'])
    np.testing.assert_array_equal(saved['active_vectors'],saved['action'][:5])
    metrics=json.loads((HERE/'results/embedding_metrics.json').read_text())
    np.testing.assert_allclose(np.linalg.norm(actual,axis=-1),metrics['norms'],rtol=1e-6)
    assert metrics['inactive_slots_exact_zero'] and metrics['nonfinite_count']==0


def test_provenance_and_figures(exported):
    graph,model,pack,actions,identity=exported
    for r in pack['records']:
        assert r['provenance']['graph_identity']==graph.graph['graph_identity']
        assert all(n in graph for n in r['provenance']['node_ids'])
        if r['kind']=='edge':
            assert graph.has_edge(r['source'],r['target'],r['edge_key'])
    manifest=json.loads((HERE/'figures/figure_manifest.json').read_text())
    assert manifest['checkpoint_sha256']==GNN_SHA
    for name,digest in manifest['inputs'].items():assert sha(HERE/'results'/name)==digest
    for name,digest in manifest['files'].items():assert sha(HERE/'figures'/name)==digest
