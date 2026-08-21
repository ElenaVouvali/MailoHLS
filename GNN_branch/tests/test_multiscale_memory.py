"""Training-only normalization and matched multiscale-control regression tests."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from GNN_branch import build_multiscale_memory as multiscale


def _fixture(tmp_path):
    jkn_dir, conv1_dir = tmp_path / "jkn", tmp_path / "conv1"
    jkn_dir.mkdir()
    conv1_dir.mkdir()
    kernels = ["test-kernel", "train-kernel", "val-kernel"]
    base_manifest = {
        "schema": "mailohls-memory-bank-manifest-v2",
        "gnn_contract_sha256": "contract",
        "gnn_checkpoint_sha256": "checkpoint",
        "feature_schema_sha256": "features",
        "source_pt_manifest_sha256": "pts",
        "source_gexf_manifest_sha256": "graphs",
        "exporter_git_commit": "commit",
        "action_relation_schema": "mailohls-action-relations-v1",
        "action_slot_schema": "absolute-lk-v1",
        "kernel_count": len(kernels),
        "ordered_kernel_list_sha256": multiscale.ordered_names_sha256(kernels),
        "gnn_dim": 2,
    }
    for directory, layer in ((jkn_dir, "jkn"), (conv1_dir, "conv_1")):
        manifest = dict(base_manifest, layer_name=layer, embedding_mode=f"static_pre_npt::{layer}")
        (directory / "memory_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for kernel in kernels:
        magnitude = 1.0 if kernel == "train-kernel" else 1_000.0
        relation = torch.tensor([[False, False, False], [False, True, True], [False, True, True]])
        common = {
            "node_embs_mask": torch.tensor([False, True, True]),
            "labels": torch.tensor([-1, 2, 3]),
            "slot_ids": torch.tensor([1, 2, 3]),
            "slot_cats": torch.tensor([0, 1, 2]),
            "action_relation_mask": relation,
            "action_relation_bits": relation.long(),
            "gnn_contract_sha256": "contract",
            "gnn_checkpoint_sha256": "checkpoint",
            "feature_schema_sha256": "features",
            "source_pt_manifest_sha256": "pts",
            "source_gexf_manifest_sha256": "graphs",
            "source_gexf_sha256": kernel,
            "action_relation_schema": "mailohls-action-relations-v1",
            "max_slots": 3,
            "disable_pragma_injection": True,
            "gnn_dim": 2,
        }
        jkn = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]) * magnitude
        conv1 = torch.tensor([[0.0, 0.0], [1.0, 3.0], [5.0, 7.0]]) * magnitude
        torch.save(dict(common, node_embs=jkn), jkn_dir / f"{kernel}.memory.pt")
        torch.save(dict(common, node_embs=conv1), conv1_dir / f"{kernel}.memory.pt")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"kernel_name": name}) for name in kernels) + "\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_jsonl_idx": [1], "val_jsonl_idx": [2], "test_jsonl_idx": [0]}))
    return jkn_dir, conv1_dir, dataset, split


def _args(tmp_path, fixture, mode="aligned", stats_in=None):
    jkn_dir, conv1_dir, dataset, split = fixture
    return SimpleNamespace(
        jkn_dir=jkn_dir,
        conv1_dir=conv1_dir,
        out=tmp_path / mode,
        dataset_jsonl=dataset,
        split_json=split,
        normalization_stats_out=(tmp_path / "normalization.json") if stats_in is None else None,
        normalization_stats_in=stats_in,
        local_scale=1.0,
        local_mode=mode,
        seed=123,
    )


def test_rms_excludes_validation_and_test_kernels(tmp_path):
    fixture = _fixture(tmp_path)
    multiscale.build_memory(_args(tmp_path, fixture))
    stats = json.loads((tmp_path / "normalization.json").read_text())
    assert stats["training_kernel_count"] == 1
    assert stats["jkn_rms"] == pytest.approx(math.sqrt(7.5))
    assert stats["centered_conv1_rms"] == pytest.approx(2.0)


def test_aligned_deranged_and_zero_share_exact_normalization_and_lk_alignment(tmp_path):
    fixture = _fixture(tmp_path)
    aligned = multiscale.build_memory(_args(tmp_path, fixture))
    stats = tmp_path / "normalization.json"
    deranged = multiscale.build_memory(_args(tmp_path, fixture, "deranged", stats))
    zero = multiscale.build_memory(_args(tmp_path, fixture, "zero", stats))
    hashes = {
        manifest["multiscale"]["normalization_stats_sha256"]
        for manifest in (aligned, deranged, zero)
    }
    assert len(hashes) == 1
    aligned_pack = torch.load(tmp_path / "aligned/train-kernel.memory.pt", weights_only=False)
    deranged_pack = torch.load(tmp_path / "deranged/train-kernel.memory.pt", weights_only=False)
    zero_pack = torch.load(tmp_path / "zero/train-kernel.memory.pt", weights_only=False)
    assert not aligned_pack["node_embs_mask"][0]
    assert torch.equal(aligned_pack["slot_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(aligned_pack["action_relation_mask"], zero_pack["action_relation_mask"])
    assert torch.equal(aligned_pack["action_relation_bits"], deranged_pack["action_relation_bits"])
    assert torch.count_nonzero(zero_pack["node_embs"][:, 2:]) == 0
    assert not torch.equal(aligned_pack["node_embs"][:, 2:], deranged_pack["node_embs"][:, 2:])


def test_jkn_conv1_provenance_mismatch_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    manifest_path = fixture[1] / "memory_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gnn_checkpoint_sha256"] = "different-checkpoint"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance mismatch"):
        multiscale.build_memory(_args(tmp_path, fixture))


def test_kernel_set_mismatch_reports_both_sides(tmp_path):
    fixture = _fixture(tmp_path)
    (fixture[1] / "train-kernel.memory.pt").unlink()
    with pytest.raises(ValueError, match="missing_from_conv1=.*train-kernel"):
        multiscale.build_memory(_args(tmp_path, fixture))
