#!/usr/bin/env python3
import os
import glob
import re
import argparse
import networkx as nx
from typing import Dict, Tuple, List

import torch
from torch_geometric.loader import DataLoader  # creates Batch with .batch :contentReference[oaicite:2]{index=2}
from torch_geometric.data import Batch
from collections import deque
import math

from model import Net
from config import FLAGS
from extract_gnn_embs import load_and_clean_graph


_AUTO_RE = re.compile(r'auto\{([^}]+)\}')

def extract_autokeys_from_full_text(full_text: str) -> List[str]:
    """
    Returns ["auto{_PIPE_L12}", ...] in the order they appear.
    """
    if not isinstance(full_text, str):
        return []
    return [f"auto{{{k}}}" for k in _AUTO_RE.findall(full_text)]


def build_nodeid_to_keys_map(gexf_path: str) -> Dict[int, List[str]]:
    g = nx.read_gexf(gexf_path, node_type=int)  # <-- important
    nodeid_to_keys = {}
    for nid, ndata in g.nodes(data=True):
        ft = ndata.get("full_text", "")
        nodeid_to_keys[int(nid)] = extract_autokeys_from_full_text(ft)
    return nodeid_to_keys


def _to_bool_mask(x: torch.Tensor) -> torch.Tensor:
    """Accept float/bool/long masks; return bool mask [N]."""
    if x is None:
        return None
    if x.dtype == torch.bool:
        return x.view(-1)
    return (x.view(-1) > 0.5)


def _lexsort_fallback(keys: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Fallback lexsort: sort by keys[0] then keys[1] where keys are 1D tensors.
    keys = (primary, secondary) - both ascending.
    """
    primary, secondary = keys
    # sort by secondary, then stable sort by primary
    idx = torch.argsort(secondary, stable=True)
    idx = idx[torch.argsort(primary[idx], stable=True)]
    return idx


def build_undirected_adj(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """
    Build an undirected adjacency list from edge_index [2, E].
    Deterministic: neighbors are sorted.
    """
    ei = edge_index.detach().cpu()
    src = ei[0].tolist()
    dst = ei[1].tolist()

    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    for u in range(num_nodes):
        adj[u] = sorted(set(adj[u]))
    return adj


def multi_source_bfs_min_dist(
    adj: List[List[int]],
    sources: List[int],
    max_hops: int,
    pseudo_mask: torch.Tensor = None,
    expand_pseudo: bool = False,
) -> List[int]:
    """
    Compute min hop distance from ANY source to every node (<=max_hops).
    Returns dist list of length N, with -1 for unreachable within max_hops.

    Key knob:
      expand_pseudo=False prevents BFS from expanding *through* pseudo nodes,
      which avoids “pseudo hubs” flooding the neighborhood.
    """
    N = len(adj)
    dist = [-1] * N
    q = deque()

    # init
    for s in sorted(set(int(x) for x in sources)):
        dist[s] = 0
        q.append(s)

    pseudo = None
    if pseudo_mask is not None:
        pseudo = (pseudo_mask.detach().cpu().view(-1) > 0.5).tolist()

    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_hops:
            continue

        # If u is a pseudo node and we don't want to expand through pseudo hubs:
        if (not expand_pseudo) and pseudo is not None and pseudo[u]:
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                if dist[v] <= max_hops:
                    q.append(v)

    return dist



def sinusoidal_slot_pe(s_max: int, d: int, dtype=torch.float32) -> torch.Tensor:
    pe = torch.zeros((s_max, d), dtype=dtype)
    position = torch.arange(0, s_max, dtype=dtype).unsqueeze(1)  # [S,1]
    div_term = torch.exp(torch.arange(0, d, 2, dtype=dtype) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe



def select_memory_tokens(
    node_emb: torch.Tensor,          # [N, D]
    edge_index: torch.Tensor,        # [2, E]
    pragmanids: torch.Tensor,        # [N]
    pseudonids: torch.Tensor,        # [N]
    contextnids: torch.Tensor,       # [N]
    s_max: int = 256,
    keep_all_pragma_nodes: bool = True,
    max_pseudo: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      tokens: [S_MAX, D] --> the selected node embeddings padded with zeros
      mask:   [S_MAX] (1 real token , 0 pad)
      token_types: [S_MAX] (0=pragma, 1=pseudo, 2=context, -1=pad)
      node_ids: [S_MAX] original node indices (or -1 for pad)
    """
    device = node_emb.device
    N, D = node_emb.shape
    prag = _to_bool_mask(pragmanids)
    pseudo = _to_bool_mask(pseudonids)
    ctx = _to_bool_mask(contextnids)

    pragma_ids = torch.nonzero(prag, as_tuple=False).view(-1)
    pseudo_ids = torch.nonzero(pseudo, as_tuple=False).view(-1)
    ctx_ids = torch.nonzero(ctx, as_tuple=False).view(-1)

    selected: List[torch.Tensor] = []
    types: List[torch.Tensor] = []

    selected.append(pragma_ids)
    selected.append(pseudo_ids)
    selected.append(ctx_ids)
    types.append(torch.full((pragma_ids.numel(),), 0, dtype=torch.long, device=device))
    types.append(torch.full((pseudo_ids.numel(),), 1, dtype=torch.long, device=device))
    types.append(torch.full((ctx_ids.numel(),), 2, dtype=torch.long, device=device))


    # 1) pragma nodes (placeholders) — strongest alignment to what you want to predict
    # if pragma_ids.numel() > 0:
    #     if keep_all_pragma_nodes:
    #         keep = pragma_ids
    #     else:
    #         keep = pragma_ids[: min(pragma_ids.numel(), s_max)]
    #     selected.append(keep)
    #     types.append(torch.full((keep.numel(),), 0, dtype=torch.long, device=device))

    # remaining budget
    # used = int(sum(t.numel() for t in selected))
    # remaining = s_max - used
    # if remaining <= 0:
        # pad/return
    #     ids = torch.cat(selected, dim=0)[:s_max]
    #     tok = node_emb[ids]
    #     mask = torch.ones((tok.size(0),), dtype=torch.long, device=device)
        # pad
    #     pad = s_max - tok.size(0)
    #     if pad > 0:
    #         tok = torch.cat([tok, torch.zeros((pad, D), device=device, dtype=tok.dtype)], dim=0)
    #         mask = torch.cat([mask, torch.zeros((pad,), device=device, dtype=torch.long)], dim=0)
    #         ttypes = torch.cat([torch.cat(types, 0), torch.full((pad,), -1, device=device, dtype=torch.long)], dim=0)
    #         node_ids = torch.cat([ids, torch.full((pad,), -1, device=device, dtype=torch.long)], dim=0)
    #     else:
    #         ttypes = torch.cat(types, 0)
    #         node_ids = ids
    #     return tok, mask, ttypes, node_ids
    
    # 2) pseudo block nodes — rank by degree and take as many as fit
    # if pseudo_ids.numel() > 0 and remaining > 0:
    #     deg = torch.bincount(edge_index.view(-1), minlength=N)  # in+out degree proxy :contentReference[oaicite:2]{index=2}
    #     pseudo_deg = deg[pseudo_ids]

        # sort pseudo by (-degree, node_id) for determinism
    #     order = torch.argsort(pseudo_ids, stable=True)
    #     pseudo_ids_sorted = pseudo_ids[order]
    #     pseudo_deg_sorted = pseudo_deg[order]
    #     order2 = torch.argsort(-pseudo_deg_sorted, stable=True)

    #     keep = pseudo_ids_sorted[order2[:remaining]]
    #     selected.append(keep)
    #     types.append(torch.full((keep.numel(),), 1, dtype=torch.long, device=device))
    #     remaining -= keep.numel()

    # 3) context nodes — pick top-K by degree to capture “important” instructions
    # if ctx_ids.numel() > 0 and remaining > 0 and pragma_ids.numel() > 0:
    #     adj = build_undirected_adj(edge_index, N)
    #     K_HOPS = 3
    #     dist = multi_source_bfs_min_dist(
    #         adj=adj,
    #         sources=pragma_ids.detach().cpu().tolist(),
    #         max_hops=K_HOPS,
    #         pseudo_mask=pseudonids,      # so we can cap pseudo expansion
    #         expand_pseudo=False,         # critical to avoid flooding
    #     )

        # Candidates: context nodes that are reachable within K hops
    #     ctx_list = ctx_ids.detach().cpu().tolist()
    #     cand = [nid for nid in ctx_list if dist[nid] != -1]

    #     if cand:
    #         deg = torch.bincount(edge_index.view(-1), minlength=N).detach().cpu().tolist()

            # Rank: smaller dist first, then higher degree, then smaller node id (deterministic)
    #         cand.sort(key=lambda n: (dist[n], -deg[n], n))

    #         keep = torch.tensor(cand[:remaining], dtype=torch.long, device=device)
    #         selected.append(keep)
    #         types.append(torch.full((keep.numel(),), 2, dtype=torch.long, device=device))
    #         remaining -= keep.numel()

    ids = torch.cat(selected, dim=0) if selected else torch.empty((0,), dtype=torch.long, device=device)
    ttypes = torch.cat(types, dim=0) if types else torch.empty((0,), dtype=torch.long, device=device)

    dist_vec = None
    if pragma_ids.numel() > 0:
        # reuse the `dist` computed above if available; else compute with K=3
        if 'dist' not in locals():
            adj = build_undirected_adj(edge_index, N)
            dist = multi_source_bfs_min_dist(adj, pragma_ids.detach().cpu().tolist(), 3, pseudonids, False)
        dist_vec = torch.tensor([dist[int(n)] for n in ids.detach().cpu().tolist()], device=device, dtype=torch.long)
    else:
        dist_vec = torch.full((ids.numel(),), -1, device=device, dtype=torch.long)

    # build outputs + pad
    tok = node_emb[ids] if ids.numel() else torch.zeros((0, D), device=device, dtype=node_emb.dtype)
    mask = torch.ones((tok.size(0),), dtype=torch.long, device=device)

    pad = s_max - tok.size(0)
    if pad > 0:
        tok = torch.cat([tok, torch.zeros((pad, D), device=device, dtype=tok.dtype)], dim=0)
        mask = torch.cat([mask, torch.zeros((pad,), device=device, dtype=torch.long)], dim=0)
        ttypes = torch.cat([ttypes, torch.full((pad,), -1, device=device, dtype=torch.long)], dim=0)
        ids = torch.cat([ids, torch.full((pad,), -1, device=device, dtype=torch.long)], dim=0)
    else:
        tok = tok[:s_max]
        mask = mask[:s_max]
        ttypes = ttypes[:s_max]
        ids = ids[:s_max]

    return tok, mask, ttypes, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", type=str, required=True, help="Directory with pragma-free kernel .pt files")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained GNN state_dict .pth")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write per-kernel memory .pt files")
    ap.add_argument("--s_max", type=int, default=256, help="Number of memory tokens per kernel (padded)")
    ap.add_argument("--gexf_dir", type=str, required=True, help="Directory with the matching kernel .gexf files (same base names as pt files)")

    args = ap.parse_args()

    pt_dir = os.path.expanduser(args.pt_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {pt_dir}")

    # Load one graph to infer in_channels/edge_dim
    sample = torch.load(pt_files[0], map_location="cpu")
    num_features = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1) if hasattr(sample, "edge_attr") and sample.edge_attr is not None else 0

    # Build model with same FLAGS you used in training
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(FLAGS.device)

    # Load checkpoint
    state = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # DataLoader creates Batch with .batch assignment vector :contentReference[oaicite:4]{index=4}
    loader = DataLoader([sample], batch_size=1, shuffle=False)

    memory_index: Dict[str, str] = {}

    for path in pt_files:
        data = load_and_clean_graph(path)
        # data = torch.load(path, map_location="cpu")
        # ensure batch exists via DataLoader
        batch = next(iter(DataLoader([data], batch_size=1, shuffle=False)))
        batch = batch.to(FLAGS.device)

        with torch.no_grad():
            node_emb = model.forward_node_embed(batch)  # [N, D]

        # Select tokens
        tokens, mask, token_types, node_ids = select_memory_tokens(
            node_emb=node_emb,
            edge_index=batch.edge_index,
            pragmanids=batch.X_pragmanids,
            pseudonids=batch.X_pseudonids,
            contextnids=batch.X_contextnids,
            s_max=args.s_max,
            keep_all_pragma_nodes=True,
            max_pseudo=min(128, args.s_max),
        )

        # kernel name
        kernel = None
        if kernel is None:
            kernel = os.path.basename(path).replace("_processed_result.pt", "")

        gexf_path = os.path.join(os.path.expanduser(args.gexf_dir), f"{kernel}_processed_result.gexf")
        if not os.path.isfile(gexf_path):
            raise FileNotFoundError(f"Missing GEXF for kernel={kernel}: {gexf_path}")

        nodeid_to_keys = build_nodeid_to_keys_map(gexf_path)

        # Build keys aligned with selected node_ids/token_types
        keys = []
        for nid, ttype, m in zip(node_ids.detach().cpu().tolist(),
                                 token_types.detach().cpu().tolist(),
                                 mask.detach().cpu().tolist()):
            if m == 0 or nid < 0:
                keys.append([])
                continue
            if ttype == 0:  # pragma node
                keys.append(nodeid_to_keys.get(nid, []))
            else:
                keys.append([])  # only pragma nodes get keys by default

        # --- slot id / slot positional encoding ---
        slot_ids = torch.arange(args.s_max, dtype=torch.long)

        # add slot positional encoding directly into token vectors (no training-script change needed)
        S, D = tokens.shape
        slot_pe = sinusoidal_slot_pe(args.s_max, D, dtype=tokens.dtype).to(FLAGS.device)  # [S_MAX, D]
        SLOT_PE_SCALE = 0.15  # start small so you don’t drown the GNN signal
        tokens = tokens + SLOT_PE_SCALE * slot_pe * mask.unsqueeze(-1).to(tokens.dtype)

        out_path = os.path.join(out_dir, f"{kernel}.memory.pt")
        torch.save(
            {
                "kernel": kernel,
                "tokens": tokens.detach().cpu(),         # [S_MAX, D_harp]
                "mask": mask.detach().cpu(),             # [S_MAX]
                "token_types": token_types.detach().cpu(),# [S_MAX]
                "slot_ids": slot_ids,
                "slot_per_scale": SLOT_PE_SCALE,
                "node_ids": node_ids.detach().cpu(),     # [S_MAX] original node indices
                "keys": keys,
                "D_harp": int(tokens.size(-1)),
                "S_MAX": int(tokens.size(0)),
            },
            out_path,
        )
        memory_index[kernel] = out_path
        print(f"[OK] {kernel} -> {out_path} | nonpad={int(mask.sum().item())} D={tokens.size(-1)}")
        data = torch.load(memory_index[kernel], map_location='cpu', weights_only=False)

        print("--- Fields in .pt file ---")
        for key in data.keys():
            print(f"Key: {key}")

        valid = int(mask.sum().item())
        n_pragma = int((token_types == 0).sum().item())
        n_pseudo = int((token_types == 1).sum().item())
        n_ctx    = int((token_types == 2).sum().item())

        # sanity: padding tokens should be all-zero vectors
        pad_pos = (mask == 0).nonzero(as_tuple=False).view(-1)
        pad_ok = True
        if pad_pos.numel() > 0:
            pad_ok = float(tokens[pad_pos].abs().sum().item()) == 0.0

        print(f"[{kernel}] valid={valid}/{args.s_max} pragma={n_pragma} pseudo={n_pseudo} ctx={n_ctx} pad_ok={pad_ok}")

        total_pragma = int((batch.X_pragmanids.view(-1) > 0.5).sum().item())
        total_pseudo = int((batch.X_pseudonids.view(-1) > 0.5).sum().item())
        print(f"    included_pragma={n_pragma}/{total_pragma} included_pseudo={n_pseudo}/{total_pseudo}")

        pragma_key_tokens = sum(1 for i in range(args.s_max) if mask[i] == 1 and token_types[i] == 0 and len(keys[i]) > 0)
        pragma_tokens = int(((mask == 1) & (token_types == 0)).sum().item())
        print(f"    pragma_tokens_with_keys={pragma_key_tokens}/{pragma_tokens}")
        keys_preview = [(i, keys[i]) for i in range(args.s_max) if len(keys[i]) > 0][:50]
        print(f"Keys list (preview) : {keys_preview}") 
        tokens_preview = tokens[:50,:]
        print(f"Memory tokens (preview) : {tokens_preview}")
        print(f"Mask : {mask}")
        print(f"slot_ids : {slot_ids}")
        print(f"node_ids : {node_ids}")
        print(f"token_types : {token_types}")


    # Save an index for convenience
    index_path = os.path.join(out_dir, "memory_index.pt")
    torch.save(memory_index, index_path)
    print(f"[OK] Saved memory index: {index_path}")




if __name__ == "__main__":
    main()

