"""Adapt authenticated frozen exports to the maintained Stage-2 bank ABI."""
import argparse
import json
from pathlib import Path
import os
import torch
from experiments.gemv_structural_explainability.final_artifacts_refinement.export import SHA,sha,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    source=json.loads((a.raw/'manifest.json').read_text())
    if source['checkpoint_sha256']!=SHA:raise ValueError('Wrong frozen GNN')
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Fresh bank required')
    a.out.mkdir(parents=True,exist_ok=True);bank=a.out/'memory';bank.mkdir()
    manifest=dict(schema='salt-final-artifacts-stage2-evidence-bank-v1',gnn_sha256=SHA,
        representation_schema='salt-final-artifacts-jk-centered-local-rc3-facts-v1',
        feature_schema_sha256=source['feature_schema_sha256'],edge_dim=48,packs={},
        embedding_layout=source['embedding_layout'],exported_embedding_mode='multiscale::jkn+centered_conv1',
        frozen_preprocessing=dict(normalization=source['normalization'],sha256=source['normalization_sha256']),
        source_manifest_sha256=sha(a.raw/'manifest.json'))
    for kernel,r in source['packs'].items():
        path=(a.raw/r['file']).resolve()
        if sha(path)!=r['sha256']:raise ValueError('Source pack changed')
        link=a.out/(kernel+'.pt');link.symlink_to(os.path.relpath(path,a.out.resolve()))
        manifest['packs'][kernel]=dict(file=link.name,sha256=r['sha256'])
        pack=torch.load(path,map_location='cpu',weights_only=False)
        active=pack['active'][0]
        loader=dict(schema='salt-final-artifacts-action-anchor-v1',representation_schema=manifest['representation_schema'],
            node_embs=pack['action'][0],node_embs_mask=active,gnn_dim=128,max_slots=64,
            disable_pragma_injection=True,slot_ids=torch.arange(1,65),
            action_relation_mask=torch.eye(64,dtype=torch.bool)&active[:,None]&active[None,:])
        torch.save(loader,bank/(kernel+'.memory.pt'))
    dump(a.out/'manifest.json',manifest)
    dump(bank/'memory_manifest.json',manifest)


if __name__=='__main__':main()
