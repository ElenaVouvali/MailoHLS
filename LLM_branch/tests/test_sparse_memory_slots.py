import torch

from LLM_branch.train.train_SFT_xattn_new import (
    get_structural_memory_pack_for_kernel,
    load_memory_bank,
)


def test_sparse_absolute_lk_slots_are_preserved(tmp_path):
    max_slots = 64
    dim = 64

    kv = torch.randn(max_slots, dim)

    mask = torch.zeros(
        max_slots,
        dtype=torch.bool,
    )

    # L2 ... L15 active; L1 intentionally absent.
    mask[1:15] = True

    labels = torch.full(
        (max_slots,),
        -1,
        dtype=torch.long,
    )
    labels[1:15] = torch.arange(2, 16)

    slot_cats = torch.zeros(
        max_slots,
        dtype=torch.long,
    )
    slot_cats[1:15] = 1

    pack = {
        "node_embs": kv.clone(),
        "node_embs_mask": mask.clone(),
        "labels": labels,
        "slot_ids": torch.arange(
            1,
            max_slots + 1,
            dtype=torch.long,
        ),
        "slot_cats": slot_cats,
        "gnn_dim": dim,
        "max_slots": max_slots,
        "disable_pragma_injection": True,
    }

    torch.save(
        pack,
        tmp_path / "lava.memory.pt",
    )

    bank, inferred_dim = load_memory_bank(
        str(tmp_path),
        expected_mem_dim=dim,
        expected_max_slots=max_slots,
        require_pragma_free_memory=True,
    )

    rec = bank["lava"]

    assert inferred_dim == dim

    # L1 remains empty.
    assert not rec["mask"][0]

    # L2 remains exactly at absolute slot index 1.
    assert rec["mask"][1]

    # Loader must not compact/reorder any structural embeddings.
    assert torch.equal(
        rec["kv"],
        kv.float(),
    )

    assert torch.equal(
        rec["mask"],
        mask,
    )

    assert torch.equal(
        rec["slot_cats"],
        slot_cats,
    )


    batched_kv, batched_mask = get_structural_memory_pack_for_kernel(
        bank,
        "lava",
        max_slots=max_slots,
        mem_dim=dim,
    )

    assert batched_kv.shape == (1, max_slots, dim)
    assert batched_mask.shape == (1, max_slots)
    assert torch.equal(batched_kv[0], kv.float())
    assert torch.equal(batched_mask[0], mask)
