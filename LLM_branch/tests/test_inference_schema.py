import torch

from LLM_branch.inference.eval_stage1_stage2_stage3 import (
    directive_schema_signature,
    load_memory_bank,
    make_candidate_xattn_apply_mask,
)


def test_directive_schema_signature_accepts_lk_anchor():
    text = (
        "<L1>\n"
        "auto{_PIPE_L1} = 1\n"
        "auto{_UNROLL_L1} = 0"
    )

    assert directive_schema_signature(text) == [
        ("anchor", "L1"),
        ("assignment", "AUTO{_PIPE_L1}"),
        ("assignment", "AUTO{_UNROLL_L1}"),
    ]


def test_inference_candidate_mask_matches_training_causal_span():
    mask = make_candidate_xattn_apply_mask(
        full_length=9,
        base_len=5,
        candidate_len=3,
        device="cpu",
    )
    assert torch.equal(
        mask.nonzero(as_tuple=False)[:, 1],
        torch.tensor([4, 5, 6]),
    )


def test_inference_loader_preserves_sparse_absolute_slots(tmp_path):
    node_embs = torch.arange(16, dtype=torch.float32).view(4, 4)
    torch.save(
        {
            "node_embs": node_embs,
            "node_embs_mask": torch.tensor([0, 1, 0, 1], dtype=torch.bool),
            "labels": torch.tensor([1, 2, 3, 4]),
            "slot_ids": torch.tensor([1, 2, 3, 4]),
            "gnn_dim": 4,
            "max_slots": 4,
        },
        tmp_path / "kernel.memory.pt",
    )
    bank = load_memory_bank(
        str(tmp_path),
        expected_mem_dim=4,
        expected_max_slots=4,
    )
    assert torch.equal(bank["kernel"]["kv"], node_embs)
    assert torch.equal(
        bank["kernel"]["mask"],
        torch.tensor([0, 1, 0, 1], dtype=torch.bool),
    )
