"""PyG-dependent tensor integration tests; skipped when PyG is unavailable."""
import hashlib
import json
from pathlib import Path
import tarfile

import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('torch_geometric')

from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from GNN_branch import mlir_graph_v2 as compiler


def _graph(tmp_path):
    root=Path(__file__).resolve().parents[1]
    archive=root/'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz'
    manifest=json.loads((root/'docs/graph-release-v22-rc3.json').read_text())
    member='canonical/serrano-kalman-filter-96b4d608969f.json'
    with tarfile.open(archive,'r:gz') as tar:
        raw=tar.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest()==manifest['files_sha256'][member]
    path=tmp_path/Path(member).name;path.write_bytes(raw)
    return compiler.read_graph(path)


def test_action_demand_tensor_is_aligned_masked_and_gradient_connected(tmp_path):
    graph=_graph(tmp_path)
    joint=overlay.enrich(graph,'zcu104',10)
    data=tensors.tensorize(joint)
    assert tensors.SCHEMA=='salt-hls-target-tensors-v2-action-demand-experimental'
    assert data.action_demand.shape==(1,64,tensors.ACTION_DEMAND_DIM)
    assert data.action_demand_known_mask.shape==(1,64,len(tensors.demand.NUMERIC_FEATURES))
    assert data.action_demand_mask[0,6].item() is False  # unresolved absolute L7 remains slot 6
    assert data.action_slot.tolist()==[i-1 for i in range(1,14) if i not in (7,8)]
    assert torch.isfinite(data.action_demand).all()
    model=tensors.SmallGraphEncoder().eval();model.zero_grad()
    action_index=int((data.action_slot==2).nonzero().item())
    node=int(data.action_node_index[action_index])
    model(data)[node].sum().backward()
    grad=model.action_demand_input.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_candidate_head_has_a_separate_optional_demand_path(tmp_path):
    from torch_geometric.data import Batch
    from experiments.target_graph_readiness.experiment import CandidatePredictor,CANDIDATE_DIM,CANDIDATE_DEMAND_DIM
    data=tensors.tensorize(overlay.enrich(_graph(tmp_path),'zcu104',10))
    batch=Batch.from_data_list([data])
    model=CandidatePredictor().eval()
    candidate=torch.zeros(1,64,CANDIDATE_DIM)
    extra=torch.zeros(1,64,CANDIDATE_DEMAND_DIM,requires_grad=True)
    model(batch,candidate,torch.tensor([0]),'enriched',candidate_demand=extra).sum().backward()
    assert extra.grad is not None and torch.isfinite(extra.grad).all() and extra.grad.abs().sum()>0

