"""Trainable evidence encoder and action-conditioned GMT-inspired tokenizer.

Graph IDs and source names remain sidecar provenance, never learned features.
The module consumes frozen four-layer states and preserves all 64 action slots.
"""
from __future__ import annotations

import torch
from torch import nn

SCHEMA = 'salt-gemv-structural-alignment-v1'
from experiments.gemv_structural_explainability.constants import TYPES, STATUS, KNOWN, EVIDENCE_CAPACITY


class MemoryAlignment(nn.Module):
    def __init__(self, edge_dim, mode='gmt', capacity=EVIDENCE_CAPACITY):
        super().__init__()
        if mode not in ('action', 'evidence', 'gmt'):
            raise ValueError(mode)
        self.mode, self.capacity = mode, capacity
        if mode == 'action':
            return
        self.node = nn.Linear(128, 128)
        # Ordered endpoints, relation one-hot/reverse/status features, knownness.
        # This is deliberately different from either endpoint representation.
        self.edge = nn.Sequential(nn.Linear(256 + edge_dim + len(KNOWN), 128),
                                  nn.GELU(), nn.Linear(128, 128))
        self.type_embedding = nn.Embedding(len(TYPES), 128)
        self.status_embedding = nn.Embedding(len(STATUS), 128)
        self.known = nn.Linear(len(KNOWN), 128, bias=False)
        self.norm = nn.LayerNorm(128)
        if mode == 'gmt':
            self.queries = nn.Parameter(torch.randn(2, 128) * .02)
            self.condition = nn.Linear(128, 128, bias=False)
            self.tokenizer = nn.MultiheadAttention(128, 4, batch_first=True,
                                                    dropout=0.)

    def forward(self, pack, *, evidence_keep=None, summaries=True):
        action, active = pack['action'], pack['active'].bool()
        if action.ndim != 3 or action.shape[1:] != (64, 128):
            raise ValueError('Absolute [B,64,128] action memory required')
        if active.shape != action.shape[:2]:
            raise ValueError('Action validity shape mismatch')
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite action memory')
        batch = action.shape[0]
        relation = torch.eye(64, dtype=torch.bool, device=action.device)[None].expand(batch,-1,-1)
        relation = relation & active[:,:,None] & active[:,None,:]
        if self.mode == 'action':
            return dict(memory=action, mask=active, relation=relation, summary_attention=None)
        n = pack['a'].shape[1]
        if n > self.capacity:
            raise ValueError('Evidence capacity exceeded; truncation forbidden')
        valid = pack['valid'].bool()
        if valid.shape != (batch, n) or pack['routes'].shape != (batch,64,n):
            raise ValueError('Evidence mask/routing shape mismatch')
        if pack['a'].shape != (batch,n,128) or pack['b'].shape != (batch,n,128):
            raise ValueError('Evidence endpoint shape mismatch')
        if evidence_keep is not None:
            if evidence_keep.shape != valid.shape:
                raise ValueError('Intervention mask shape mismatch')
            valid = valid & evidence_keep.bool()
        edge = self.edge(torch.cat((pack['a'], pack['b'], pack['edge'], pack['known']), -1))
        evidence = torch.where(pack['is_edge'][:,:,None], edge, self.node(pack['a']))
        evidence = self.norm(evidence + self.type_embedding(pack['types']) +
                             self.status_embedding(pack['status']) + self.known(pack['known']))
        evidence = evidence.masked_fill(~valid[:,:,None], 0.)
        routes = pack['routes'].bool() & valid[:,None,:] & active[:,:,None]
        memory, mask = torch.cat((action,evidence),1), torch.cat((active,valid),1)
        summary_attention = None
        if self.mode == 'gmt' and summaries:
            # Empty evidence sets never enter MHA (avoids all-masked NaNs).
            summary = action.new_zeros(batch,64,2,128)
            summary_mask = torch.zeros(batch,64,2,dtype=torch.bool,device=action.device)
            summary_attention = action.new_zeros(batch,64,4,2,n)
            for b in range(batch):
                for k in active[b].nonzero().flatten().tolist():
                    eligible = routes[b,k].nonzero().flatten()
                    if not eligible.numel():
                        continue
                    query = self.queries + self.condition(action[b,k])[None]
                    values = evidence[b,eligible][None]
                    output, weights = self.tokenizer(query[None], values, values,
                        need_weights=True, average_attn_weights=False)
                    summary[b,k] = output[0]
                    summary_mask[b,k] = True
                    summary_attention[b,k,:,:,eligible] = weights[0]
            memory = torch.cat((memory,summary.flatten(1,2)),1)
            mask = torch.cat((mask,summary_mask.flatten(1,2)),1)
        total = memory.shape[1]
        routing = torch.zeros(batch,total,total,dtype=torch.bool,device=action.device)
        routing[:,:64,:64] = relation
        routing[:,:64,64:64+n] = routes
        if summary_attention is not None:
            for k in range(64):
                routing[:,k,64+n+2*k:64+n+2*k+2] = summary_mask[:,k]
        routing &= mask[:,:,None] & mask[:,None,:]
        if not torch.isfinite(memory).all():
            raise ValueError('Nonfinite encoded memory')
        return dict(memory=memory, mask=mask, relation=routing,
                    summary_attention=summary_attention)


def to_device(pack, device, dtype=torch.float32):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            if torch.is_tensor(v) else v for k,v in pack.items()}
