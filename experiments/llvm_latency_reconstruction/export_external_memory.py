"""Compile held-out sources with the archived LLVM/ProGraML pipeline on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--gnn", type=Path, required=True)
    p.add_argument("--encoders", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    runtime = a.runtime.resolve()
    root = runtime / "external_llvm"
    (root / "src").mkdir(parents=True, exist_ok=True)
    for name in ("config.py", "clang_script.sh"):
        shutil.copyfile(repo / "GNN_branch" / name, root / "src" / name)
    os.environ["MAILOHLS_LLVM_RUNTIME_ROOT"] = str(root)
    sys.path.insert(0, str(repo / "GNN_branch"))
    # The archived config parses sys.argv on import.
    sys.argv = [sys.argv[0], "--device", "cpu"]
    import torch
    from torch_geometric.data import Batch
    import data as data_py
    from cpp_to_gexf_deterministic import cpp_to_gexf
    from gexf_to_pt_zero import gexf_to_pt
    from model import Net
    from pt_to_gnn_emb import disable_pragma_conditioning

    torch.set_num_threads(4)
    data_py.ENCODER_PATH = str(a.encoders.resolve())
    state = torch.load(a.gnn, map_location="cpu")
    nf = state["conv_first.lin_key.weight"].shape[1]
    ed = state["conv_first.lin_edge.weight"].shape[1]
    model = Net(nf, edge_dim=ed, init_pragma_dict=None)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    manifest = {"checkpoint_sha256": hashlib.sha256(a.gnn.read_bytes()).hexdigest(),
                "encoders_sha256": hashlib.sha256(a.encoders.read_bytes()).hexdigest(),
                "strict_load": True, "num_features": nf, "edge_dim": ed,
                "graph_pipeline": "original llvm-ir/LLVM10/ProGraML", "kernels": {},
                "representation_limit": "Preserves archived tensorizer, including its lexicographic edge relabeling convention; no MLIR substitution or encoder refit."}
    out_memory = runtime / "external_memory"
    out_memory.mkdir(exist_ok=True)
    for line in a.cases.open():
        case = json.loads(line)
        name = case["kernel_name"]
        if not re.fullmatch(r"[\w-]+", name):
            raise ValueError("Unsafe kernel name")
        # augment_graph_hierarchy resolves LLVM bodies under <root>/harp/<name>.
        folder = root / "harp" / name
        folder.mkdir(parents=True, exist_ok=True)
        src = folder / (name + ".cpp")
        src.write_text(case["code"])
        gexf = folder / (name + ".gexf")
        cpp_to_gexf(name, str(folder), "cpp", str(gexf), log=False)
        pt = folder / (name + ".pt")
        gexf_to_pt(str(gexf), "NONE", str(pt), name)
        graph = torch.load(pt, weights_only=False, map_location="cpu")
        if graph.x.shape[1] != nf or graph.edge_attr.shape[1] != ed:
            raise ValueError("Frozen encoder/checkpoint schema mismatch")
        batch = disable_pragma_conditioning(Batch.from_data_list([graph]))
        with torch.no_grad():
            node = model.forward_node_embed(batch)
            pooled = model.forward_embed(batch)
        if not torch.isfinite(node).all() or not torch.isfinite(pooled).all():
            raise ValueError("Nonfinite GNN output")
        vectors = torch.zeros(64, 64)
        mask = torch.zeros(64, dtype=torch.bool)
        labels, node_ids = [-1] * 64, [-1] * 64
        cats = torch.zeros(64, dtype=torch.long)
        for ni in batch.X_llm_scopeids.nonzero().flatten().tolist():
            lid = int(batch.X_llm_labelid[ni])
            if not 1 <= lid <= 64 or mask[lid - 1]:
                raise ValueError("Ambiguous or invalid action identity")
            vectors[lid - 1], mask[lid - 1] = node[ni], True
            labels[lid - 1], node_ids[lid - 1] = lid, ni
            cats[lid - 1] = batch.X_llm_scopecat[ni]
        expected = sorted({int(x) for x in re.findall(r"auto\{[^}]*_L(\d+)\}", case["code"])})
        if any(not mask[k - 1] for k in expected):
            raise ValueError(f"Missing authenticated LLVM action bindings for {name}")
        pack = dict(node_embs=vectors, node_embs_mask=mask, graph_embed=pooled,
                    labels=labels, node_ids=node_ids, slot_cats=cats,
                    disable_pragma_injection=True, gnn_dim=64,
                    ckpt=str(a.gnn.resolve()), checkpoint_sha256=manifest["checkpoint_sha256"])
        output = out_memory / (name + ".memory.pt")
        torch.save(pack, output)
        manifest["kernels"][name] = {
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "graph_sha256": hashlib.sha256(gexf.read_bytes()).hexdigest(),
            "tensor_sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
            "memory_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "nodes": int(graph.x.shape[0]), "edges": int(graph.edge_index.shape[1]),
            "active_actions": expected,
            "action_node_ids": {f"L{k}": node_ids[k-1] for k in expected},
        }
        print(json.dumps({"kernel": name, **manifest["kernels"][name]}), flush=True)
        (runtime / "external_memory_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
