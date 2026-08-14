import argparse
import hashlib
import glob
import json
import os
import re
import subprocess
import sys
import torch


def parse_builder_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint_contract", required=True)
    parser.add_argument("--checkpoint_sidecar", required=True)
    parser.add_argument("--feature_schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_slots", type=int, default=64)
    parser.add_argument(
        "--embedding_mode",
        required=True,
        choices=["current_zero_scope_post_npt", "static_pre_npt"],
    )
    return parser.parse_args(argv)


BUILD_ARGS = parse_builder_args(sys.argv[1:])
sys.argv = [sys.argv[0]]

from os.path import join, basename
from torch_geometric.data import Batch, Data


def _normalize_kernel_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())    # match both '-' and '_'


def _disable_pragma_conditioning(data):
    """
    Disable all pragma-conditioned updates used by the fixed NPT path.
    """
    for name in (
        "X_pragmascopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
    ):
        if hasattr(data, name):
            setattr(data, name, torch.zeros_like(getattr(data, name)))

    # Extra safety: neutralize value tensors too
    if hasattr(data, "X_pragma_per_node"):
        data.X_pragma_per_node = torch.zeros_like(data.X_pragma_per_node)
    if hasattr(data, "pragmas"):
        data.pragmas = torch.zeros_like(data.pragmas)

    return data


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_pt_manifest_sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: os.path.basename(item)):
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _verify_hash(path, expected, label):
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f'{label} SHA256 mismatch: actual={actual}, expected={expected}'
        )


def _load_graph(path):
    graph = torch.load(path, weights_only=False)
    if isinstance(graph, Data):
        for key in list(graph.keys()):
            if not isinstance(key, str):
                del graph[key]
        if hasattr(graph, 'edge_id_to_idx'):
            del graph.edge_id_to_idx
    return graph


def _checkpoint_state(payload):
    if isinstance(payload, dict) and 'model' in payload:
        return payload['model']
    if isinstance(payload, dict) and 'state_dict' in payload:
        return payload['state_dict']
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def main():
    args = BUILD_ARGS

    contract = _load_json(args.checkpoint_contract)
    sidecar = _load_json(args.checkpoint_sidecar)
    if contract.get('format') != 'mailohls-gnn-checkpoint-v1':
        raise RuntimeError('Unsupported GNN checkpoint contract format.')
    if contract.get('provenance_status') not in {
        'captured_at_training', 'reconstructed'
    }:
        raise RuntimeError('Invalid GNN provenance_status.')
    if sidecar.get('provenance_status') != contract['provenance_status']:
        raise RuntimeError('Checkpoint and contract provenance status differ.')
    _verify_hash(
        args.checkpoint_contract,
        sidecar['contract_sha256'],
        'checkpoint contract',
    )
    _verify_hash(args.ckpt, sidecar['checkpoint_sha256'], 'checkpoint')
    _verify_hash(
        args.feature_schema,
        contract['feature_schema']['sha256'],
        'feature schema',
    )

    from config import FLAGS
    for name, value in contract['model_init_flags'].items():
        if not hasattr(FLAGS, name):
            raise RuntimeError(f'Unknown saved model-init flag: {name}')
        setattr(FLAGS, name, value)
    from model import Net

    os.makedirs(args.out, exist_ok=True)

    pt_files = sorted(
        glob.glob(join(args.pt_path, "*_processed_result.pt")),
        key=lambda path: os.path.basename(path),
    )
    if not pt_files:
        print(f"No files found in {args.pt_path}")
        return
    
    first_pt = _load_graph(pt_files[0])
    num_features = first_pt.x.size(-1)
    edge_dim = first_pt.edge_attr.size(-1) if getattr(first_pt, "edge_attr", None) is not None else 0
    model_init = contract['model_init']
    feature_schema = _load_json(args.feature_schema)
    if (
        feature_schema.get('feature_schema_version')
        != contract['feature_schema'].get('version')
    ):
        raise RuntimeError('Feature-schema version disagrees with the contract.')
    if int(feature_schema.get('node_feature_dim', -1)) != int(
        model_init['in_channels']
    ) or int(feature_schema.get('edge_feature_dim', -1)) != int(
        model_init['edge_dim']
    ):
        raise RuntimeError(
            'Feature-schema tensor dimensions disagree with model_init.'
        )
    if num_features != int(model_init['in_channels']):
        raise RuntimeError(
            f'Node feature dimension mismatch: {num_features} != '
            f"{model_init['in_channels']}"
        )
    if edge_dim != int(model_init['edge_dim']):
        raise RuntimeError(
            f'Edge feature dimension mismatch: {edge_dim} != '
            f"{model_init['edge_dim']}"
        )

    resource_stats = (
        contract.get('resource_stats')
        if model_init.get('resource_aux_heads', False) else None
    )
    model = Net(
        num_features,
        edge_dim=edge_dim,
        init_pragma_dict=None,
        target_stats=contract.get('target_stats'),
        resource_stats=resource_stats,
    ).to(FLAGS.device)
    payload = torch.load(args.ckpt, map_location=FLAGS.device)
    state = _checkpoint_state(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)
    source_pt_manifest_sha256 = _source_pt_manifest_sha256(pt_files)
    git_commit = _git_commit()
    contract_sha256 = _sha256(args.checkpoint_contract)
    feature_schema_sha256 = _sha256(args.feature_schema)

    print(f"Starting processing of {len(pt_files)} files...")

    targets = ["rodinia", "machsuite", "spcl", "serrano"]
    
    for kernel_path in pt_files:
        # {x}_processed_result.pt -> {x}.memory.pt
        fname = basename(kernel_path)
        normalized_name = _normalize_kernel_name(fname)
        if not any(t in normalized_name for t in targets):
            continue
        prefix_match = re.search(r"(.+)_processed_result\.pt", fname)
        if prefix_match:
            base_name = prefix_match.group(1)

        output_path = join(args.out, f"{base_name}.memory.pt")

        try:
            pt_point = _load_graph(kernel_path)
            if pt_point.x.size(-1) != num_features:
                raise RuntimeError(f'{fname} has inconsistent node features.')
            point_edge_dim = (
                pt_point.edge_attr.size(-1)
                if getattr(pt_point, 'edge_attr', None) is not None else 0
            )
            if point_edge_dim != edge_dim:
                raise RuntimeError(f'{fname} has inconsistent edge features.')

            required = [
                "X_pipeline_scopeids",
                "X_unroll_scopeids",
                "X_array_partition_scopeids",
                "X_arrayscopenids",
                "X_llm_scopeids",
                "X_llm_scopecat",
                "X_llm_labelid",
            ]
            missing = [k for k in required if not hasattr(pt_point, k)]
            if missing:
                raise RuntimeError(
                    f"{fname} is missing required fields {missing}. "
                    "Regenerate pragma-free .pt files with the patched gexf_to_pt_zero.py."
                )

            batch = Batch.from_data_list([pt_point]).to(FLAGS.device)

            batch = _disable_pragma_conditioning(batch)

            if args.embedding_mode == "current_zero_scope_post_npt":
                node_emb = model.forward_node_embed(batch)
            elif args.embedding_mode == "static_pre_npt":
                node_emb = model.forward_static_node_embed(batch)
            else:
                raise AssertionError(args.embedding_mode)

#            # Use X_llm_scopeids and X_llm_labelid to build slot-aligned memory
            scope = batch.X_llm_scopeids.bool()
            label = batch.X_llm_labelid.long()

            # tokens that correspond to placeholders
            sel = scope & (label > 0) & (label <= args.max_slots)
            sel_idx = sel.nonzero(as_tuple=False).view(-1)

            # slot-aligned outputs
            node_embs = torch.zeros((args.max_slots, node_emb.size(-1)), dtype=node_emb.dtype, device=node_emb.device)
            node_embs_mask = torch.zeros((args.max_slots,), dtype=torch.bool, device=node_emb.device)
            slot_cats = torch.zeros((args.max_slots,), dtype=torch.long, device=node_emb.device)

            node_ids = [-1] * args.max_slots
            labels = [-1] * args.max_slots

            for ni in sel_idx.tolist():
                lid = int(label[ni].item())  # 1..max_slots
                slot = lid - 1
                node_embs[slot] = node_emb[ni]
                node_embs_mask[slot] = True
                node_ids[slot] = ni
                labels[slot] = lid
                slot_cats[slot] = int(batch.X_llm_scopecat[ni].item())

            node_embs = node_embs.detach().cpu()
            node_embs_mask = node_embs_mask.detach().cpu()
            node_embs = torch.nan_to_num(node_embs, nan=0.0, posinf=0.0, neginf=0.0)
            max_norm = 20.0
            eps = 1e-6
            norms = node_embs.norm(p=2, dim=1, keepdim=True).clamp(min=eps) # L2 normalization
            scale = (max_norm / norms).clamp(max=1.0)
            node_embs = node_embs * scale

            pack = {
                "pt_path": kernel_path,
                "ckpt": args.ckpt,
                "embedding_mode": args.embedding_mode,
                "disable_pragma_injection": True,
                "gnn_checkpoint_sha256": checkpoint_sha256,
                "gnn_contract_sha256": contract_sha256,
                "feature_schema_sha256": feature_schema_sha256,
                "source_pt_manifest_sha256": source_pt_manifest_sha256,
                "git_commit": git_commit,
                "gnn_dim": int(node_embs.size(-1)),
                "node_embs": node_embs,
                "node_embs_mask": node_embs_mask,
                "max_slots": args.max_slots,
                "slot_ids": torch.arange(1, args.max_slots + 1, dtype=torch.long),
                "slot_cats": slot_cats.detach().cpu(),
                "node_ids": node_ids,
                "labels": labels,
            }

            torch.save(pack, output_path)
            print(f"[OK] {fname} -> Node embeddings shape : {node_embs.shape}")
            
            # Select only rows where memory_mask is True
            active_memory = node_embs[node_embs_mask] 
            
            if active_memory.size(0) > 0:
                print(f"node_embs[node_embs_mask] --> shape: {active_memory.shape}")
                print(active_memory)
 #               print(f"Global graph embedding : {graph_embed}")
                print(f"node_embs_mask : {node_embs_mask}")
                print(f"node_ids : {node_ids}")
                print(f"labels : {labels}")
            else:
                print("Kernel found, but no slots were identified (All zeros).")


        except Exception as e:
            raise RuntimeError(f"Failed to process {fname}: {e}") from e

    bank_manifest = {
        'schema': 'mailohls-memory-bank-manifest-v2',
        'gnn_contract_sha256': contract_sha256,
        'feature_schema_sha256': feature_schema_sha256,
        'gnn_checkpoint_sha256': checkpoint_sha256,
        'source_pt_manifest_sha256': source_pt_manifest_sha256,
        'embedding_mode': args.embedding_mode,
        'exporter_git_commit': git_commit,
        'checkpoint_tag': sidecar.get('checkpoint_tag'),
        'checkpoint_epoch': sidecar.get('checkpoint_epoch'),
        'provenance_status': contract['provenance_status'],
    }
    with open(join(args.out, 'memory_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(bank_manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')



if __name__ == "__main__":
    main()
