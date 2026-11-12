"""Rebuild frozen LLVM memories when archived checkpoint-path metadata is stale."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from prepare import sha256
    root = a.runtime.resolve()
    runtime = root / "gnn_rebuild_runtime"
    (runtime / "src").mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    shutil.copyfile(repo / "GNN_branch/config.py", runtime / "src/config.py")
    os.environ["MAILOHLS_LLVM_RUNTIME_ROOT"] = str(runtime)
    sys.path.insert(0, str(repo / "GNN_branch"))
    sys.argv = [sys.argv[0], "--device", "cpu"]
    import torch
    from torch_geometric.data import Batch
    from model import Net
    from pt_to_gnn_emb import disable_pragma_conditioning

    torch.set_num_threads(4)
    checkpoint = root / "assets/gnn_val.pth"
    state = torch.load(checkpoint, map_location="cpu")
    nf = state["conv_first.lin_key.weight"].shape[1]
    ed = state["conv_first.lin_edge.weight"].shape[1]
    model = Net(nf, edge_dim=ed, init_pragma_dict=None)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint_sha256=sha256(checkpoint), strict_load=True,
        node_features=nf, edge_features=ed, memory_shape=[64, 64],
        reason="Archived path-only memory provenance failed numerical reproduction; regenerate all memories from the actual recovered weights.",
        kernels={})
    for path in sorted((root / "original_graphs").glob("*_processed_result.pt")):
        name = path.name[:-len("_processed_result.pt")]
        graph = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(graph.x.shape)[1] != nf or tuple(graph.edge_attr.shape)[1] != ed:
            raise ValueError(f"Graph incompatible with recovered checkpoint: {name}")
        batch = disable_pragma_conditioning(Batch.from_data_list([graph]))
        with torch.no_grad():
            node = model.forward_node_embed(batch)
            pooled = model.forward_embed(batch)
        if not torch.isfinite(node).all() or not torch.isfinite(pooled).all():
            raise ValueError(f"Nonfinite representation: {name}")
        vectors = torch.zeros(64, 64)
        mask = torch.zeros(64, dtype=torch.bool)
        cats = torch.zeros(64, dtype=torch.long)
        labels, node_ids = [-1] * 64, [-1] * 64
        for ni in batch.X_llm_scopeids.nonzero().flatten().tolist():
            lid = int(batch.X_llm_labelid[ni])
            if not 1 <= lid <= 64 or mask[lid-1]:
                raise ValueError(f"Ambiguous absolute slot: {name}")
            vectors[lid-1], mask[lid-1] = node[ni], True
            cats[lid-1] = batch.X_llm_scopecat[ni]
            labels[lid-1], node_ids[lid-1] = lid, ni
        reference = root / "memory_recovered_legacy"
        if not reference.exists():
            reference = root / "memory"
        old_path = reference / (name + ".memory.pt")
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        if not torch.equal(mask, old["node_embs_mask"]) or labels != old["labels"] or node_ids != old["node_ids"]:
            raise ValueError(f"Recovered graph/action identity differs from archived memory: {name}")
        pack = dict(node_embs=vectors, node_embs_mask=mask, graph_embed=pooled,
                    labels=labels, node_ids=node_ids, slot_cats=cats,
                    ckpt=str(checkpoint), checkpoint_sha256=manifest["checkpoint_sha256"],
                    pt_path=str(path), graph_sha256=sha256(path),
                    disable_pragma_injection=True, gnn_dim=64, max_slots=64)
        dst = a.output / (name + ".memory.pt")
        torch.save(pack, dst)
        manifest["kernels"][name] = dict(
            graph_sha256=pack["graph_sha256"], memory_sha256=sha256(dst),
            archived_memory_sha256=sha256(old_path), active_slots=int(mask.sum()),
            action_identity_preserved=True,
            max_abs_delta_from_archived=float((vectors-old["node_embs"]).abs().max()),
            node_ids={f"L{i+1}": node_ids[i] for i in mask.nonzero().flatten().tolist()})
        print(json.dumps({"kernel": name, **manifest["kernels"][name]}), flush=True)
        (root / "memory_rebuild_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(manifest["kernels"]) != 55:
        raise ValueError("Incomplete original graph cohort; do not activate this memory bank")


if __name__ == "__main__":
    main()
