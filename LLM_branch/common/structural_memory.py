from __future__ import annotations

import os
import re
from typing import Dict, Tuple, Optional

import torch


STRUCTURAL_ROUTING_MODES = {
    "exact_slot",
    "compiler_relational",
}


def resolve_relation_mask(
    pack: dict,
    routing_mode: str,
) -> Optional[torch.Tensor]:

    if routing_mode == "exact_slot":
        return None

    if routing_mode == "compiler_relational":
        relation = pack.get(
            "relation_mask",
            None,
        )

        if relation is None:
            raise ValueError(
                "compiler_relational routing requires "
                "action_relation_mask"
            )

        return relation.bool().contiguous()

    raise ValueError(
        f"Unsupported structural routing: "
        f"{routing_mode!r}"
    )


def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())

def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


# ==========================================
# STRUCTURAL memory bank loader (.memory.pt files)
# ==========================================

def load_memory_bank(
    memory_dir: str,
    expected_mem_dim: Optional[int] = None,
    expected_max_slots: Optional[int] = None,
    require_pragma_free_memory: bool = False,
) -> Tuple[Dict[str, dict], Optional[int]]:
    bank = {}
    inferred_mem_dim = None

    for fn in sorted(os.listdir(memory_dir)):
        if not fn.endswith(".memory.pt"):
            continue

        pack = torch.load(os.path.join(memory_dir, fn), map_location="cpu", weights_only=False)

        kv = pack["node_embs"].float()
        mask = pack["node_embs_mask"].bool()
        labels = pack.get("labels", None)
        slot_cats = pack.get("slot_cats", None)

        if kv.ndim != 2:
            raise ValueError(f"{fn}: node_embs must be [S, D], got {tuple(kv.shape)}")
        if mask.ndim != 1 or mask.numel() != kv.size(0):
            raise ValueError(f"{fn}: bad node_embs_mask shape {tuple(mask.shape)} for node_embs {tuple(kv.shape)}")

        gnn_dim = int(pack.get("gnn_dim", kv.size(1)))
        if gnn_dim != kv.size(1):
            raise ValueError(f"{fn}: gnn_dim={gnn_dim} but node_embs.size(1)={kv.size(1)}")

        if inferred_mem_dim is None:
            inferred_mem_dim = gnn_dim
        elif inferred_mem_dim != gnn_dim:
            raise ValueError(f"Inconsistent gnn_dim across memory packs: {inferred_mem_dim} vs {gnn_dim}")

        if expected_mem_dim is not None and gnn_dim != expected_mem_dim:
            raise ValueError(f"{fn}: memory dim mismatch, pack={gnn_dim}, expected={expected_mem_dim}")

        max_slots = int(pack.get("max_slots", kv.size(0)))
        if expected_max_slots is not None and max_slots != expected_max_slots:
            raise ValueError(f"{fn}: max_slots mismatch, pack={max_slots}, expected={expected_max_slots}")

        if require_pragma_free_memory and not bool(pack.get("disable_pragma_injection", False)):
            raise ValueError(f"{fn}: memory was not built with disable_pragma_injection=True")

        # Preserve absolute MailoHLS Lk slot alignment.
        #
        # build_structural_memory.py stores action Lk at slot k-1.
        # Sparse action sets are therefore valid; for example, a kernel
        # may contain active L2..L15 while slot 0 / L1 remains inactive.
        #
        # Do NOT compact active vectors to slots 0..N-1: structural_xattn
        # routes the <Lk> placeholder directly to one-based memory slot k.
        if labels is not None:
            labels_t = torch.as_tensor(
                labels,
                dtype=torch.long,
            ).view(-1)

            if labels_t.numel() != kv.size(0):
                raise ValueError(
                    f"{fn}: labels have {labels_t.numel()} entries "
                    f"for {kv.size(0)} memory slots"
                )

            expected_slots = torch.arange(
                1,
                kv.size(0) + 1,
                dtype=torch.long,
            )

            active_idx = mask.nonzero(
                as_tuple=False
            ).view(-1)

            if active_idx.numel() > 0:
                active_labels = labels_t.index_select(
                    0,
                    active_idx,
                )
                expected_active = expected_slots.index_select(
                    0,
                    active_idx,
                )

                if not torch.equal(
                    active_labels,
                    expected_active,
                ):
                    raise ValueError(
                        f"{fn}: active labels violate absolute Lk "
                        f"slot alignment: "
                        f"labels={active_labels.tolist()}, "
                        f"expected={expected_active.tolist()}"
                    )

        # slot_ids, when present, must describe the same fixed absolute
        # one-based structural address space.
        slot_ids = pack.get("slot_ids", None)

        if slot_ids is not None:
            slot_ids_t = torch.as_tensor(
                slot_ids,
                dtype=torch.long,
            ).view(-1)

            expected_slots = torch.arange(
                1,
                kv.size(0) + 1,
                dtype=torch.long,
            )

            if not torch.equal(
                slot_ids_t,
                expected_slots,
            ):
                raise ValueError(
                    f"{fn}: slot_ids must equal absolute [1..S]; "
                    f"got {slot_ids_t.tolist()}"
                )

        if slot_cats is not None:
            slot_cats = torch.as_tensor(
                slot_cats,
                dtype=torch.long,
            ).view(-1)

            if slot_cats.numel() != kv.size(0):
                raise ValueError(
                    f"{fn}: slot_cats have {slot_cats.numel()} entries "
                    f"for {kv.size(0)} memory slots"
                )

        k = fn.replace(".memory.pt", "")

        relation_mask = pack.get(
            "action_relation_mask",
            None,
        )

        relation_bits = pack.get(
            "action_relation_bits",
            None,
        )

        RELATION_ALL_BITS = (1 << 6) - 1  # self,parent,child,array,dep_fwd,dep_rev

        if relation_bits is not None:
            relation_bits = torch.as_tensor(
                relation_bits,
                dtype=torch.long,
            ).contiguous()

            expected_shape = (kv.size(0), kv.size(0))
            if tuple(relation_bits.shape) != expected_shape:
                raise ValueError(
                    f"{fn}: action_relation_bits "
                    f"{tuple(relation_bits.shape)} != {expected_shape}"
                )

            if torch.bitwise_and(
                relation_bits,
                ~RELATION_ALL_BITS,
            ).any():
                raise ValueError(f"{fn}: unknown relation bits")

            if relation_mask is None:
                raise ValueError(
                    f"{fn}: relation bits exist without relation mask"
                )

            if not torch.equal(
                relation_bits.ne(0),
                relation_mask,
            ):
                raise ValueError(
                    f"{fn}: relation bits/mask disagree"
                )

        if relation_mask is not None:

            relation_mask = (
                torch.as_tensor(
                    relation_mask,
                    dtype=torch.bool,
                )
                .contiguous()
            )

            expected_relation_shape = (
                kv.size(0),
                kv.size(0),
            )

            if tuple(
                relation_mask.shape
            ) != expected_relation_shape:
                raise ValueError(
                    f"{fn}: action_relation_mask "
                    f"{tuple(relation_mask.shape)} "
                    f"!= {expected_relation_shape}"
                )

            # Relation mask may reference only
            # active memory slots.
            active_pairs = (
                mask[:, None]
                &
                mask[None, :]
            )

            if (
                relation_mask
                & ~active_pairs
            ).any():
                raise ValueError(
                    f"{fn}: relation mask references "
                    "inactive structural slots"
                )

            active = torch.where(
                mask
            )[0]

            if (
                active.numel() > 0
                and not relation_mask[
                    active,
                    active,
                ].all()
            ):
                raise ValueError(
                    f"{fn}: active action is missing "
                    "its self relation"
                )
            
        rec = {
            "kv": kv.contiguous(),
            "mask": mask.contiguous(),
            "relation_mask":
                (
                    relation_mask
                    if relation_mask
                    is not None
                    else None
                ),
            "relation_bits": relation_bits,
            "slot_cats": slot_cats,
            "ckpt": pack.get("ckpt", ""),
            "disable_pragma_injection": bool(pack.get("disable_pragma_injection", False)),
        }
    
        for alias in dict.fromkeys((k, normalize_kname(k))):
            existing = bank.get(alias)
            if existing is not None and existing is not rec:
                raise ValueError(
                    "Structural-memory normalization collision for "
                    f"{alias!r}: {existing['_source_kernel']!r} and {k!r}"
                )
            rec["_source_kernel"] = k
            bank[alias] = rec

    return bank, inferred_mem_dim


def memory_bank_summary(memory_dir: str, bank: Dict[str, dict], required_kernels=()) -> dict:
    """Distinguish physical packs and unique records from raw/normalized aliases."""
    required = {normalize_kname(str(kernel)) for kernel in required_kernels}
    covered = {kernel for kernel in required if kernel in bank}
    return {
        "memory_files": sum(name.endswith(".memory.pt") for name in os.listdir(memory_dir)),
        "unique_memory_records": len({id(record) for record in bank.values()}),
        "lookup_aliases": len(bank),
        "required_split_kernels": len(required),
        "required_split_kernels_covered": len(covered),
    }


def print_memory_bank_summary(memory_dir: str, bank: Dict[str, dict], required_kernels=()) -> dict:
    summary = memory_bank_summary(memory_dir, bank, required_kernels)
    print(f"[INFO] Memory files: {summary['memory_files']}")
    print(f"[INFO] Unique memory records: {summary['unique_memory_records']}")
    print(f"[INFO] Lookup aliases: {summary['lookup_aliases']}")
    print(
        "[INFO] Required split kernels covered: "
        f"{summary['required_split_kernels_covered']}/{summary['required_split_kernels']}"
    )
    return summary


def get_structural_memory_pack_for_kernel(
    mem_bank,
    kernel_name,
    max_slots,
    mem_dim,
    structural_routing="exact_slot",
):
    pack = (
        mem_bank.get(kernel_name)
        or mem_bank.get(
            normalize_kname(kernel_name)
        )
    )

    if pack is None:
        raise KeyError(
            f"No structural memory found for "
            f"kernel={kernel_name!r}"
        )

    kv = pack["kv"]
    mask = pack["mask"]

    expected_kv = (
        int(max_slots),
        int(mem_dim),
    )

    if tuple(kv.shape) != expected_kv:
        raise ValueError(
            f"{kernel_name}: "
            f"{tuple(kv.shape)} != {expected_kv}"
        )

    if tuple(mask.shape) != (
        int(max_slots),
    ):
        raise ValueError(
            f"{kernel_name}: invalid memory mask"
        )

    relation = resolve_relation_mask(
        pack,
        structural_routing,
    )

    return (
        kv.unsqueeze(0).contiguous(),
        mask.unsqueeze(0).contiguous(),
        (
            relation.unsqueeze(0).contiguous()
            if relation is not None
            else None
        ),
    )
