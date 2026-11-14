"""Fact-supervised, role-constrained GMT memory refinement.

Two roles are defined by compiler record kind, not inferred human concepts:
summary 0 pools node records; summary 1 pools relation witnesses. Both pools are
intersected with the authenticated action route. Original action slots are fixed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from experiments.gemv_structural_explainability.memory_alignment import MemoryAlignment
from experiments.gemv_structural_explainability.constants import TYPES
from experiments.gemv_structural_explainability.final_artifacts_refinement.facts import FACT_NAMES

SCHEMA='salt-final-artifacts-role-fact-gmt-v1'


class SemanticMemoryAlignment(MemoryAlignment):
    def __init__(self,edge_dim=48,mode='gmt',capacity=4096):
        super().__init__(edge_dim,mode,capacity)
        self.schema=SCHEMA
        if mode=='action':return
        self.fact_projection=nn.Sequential(nn.Linear(len(FACT_NAMES),128),nn.GELU(),nn.Linear(128,128))
        if mode=='gmt':
            self.summary_norm=nn.LayerNorm(128)
            self.readout=nn.Sequential(nn.Linear(128,128),nn.GELU(),
                nn.Linear(128,2*len(TYPES)+len(FACT_NAMES)))
            self.register_buffer('fact_mean',torch.zeros(2,len(FACT_NAMES)//2))
            self.register_buffer('fact_scale',torch.ones(2,len(FACT_NAMES)//2))

    def forward(self,pack,*,evidence_keep=None,summaries=True):
        if self.mode=='action':return super().forward(pack)
        action,active=pack['action'],pack['active'].bool()
        if action.shape[1:]!=(64,128):raise ValueError('Absolute action prefix required')
        b,n=pack['a'].shape[:2]
        if n>self.capacity:raise ValueError('Evidence capacity exceeded; no truncation')
        valid=pack['valid'].bool()
        if evidence_keep is not None:valid=valid&evidence_keep.bool()
        routes=pack['routes'].bool()&valid[:,None]&active[:,:,None]
        facts=pack['facts']
        if facts.shape!=(b,n,len(FACT_NAMES)):raise ValueError('Explicit fact schema mismatch')
        supplied=facts
        if self.training:
            retain=torch.rand_like(facts[:,:,0::2])>=.15
            supplied=facts.clone();supplied[:,:,0::2]*=retain;supplied[:,:,1::2]*=retain
        edge=self.edge(torch.cat((pack['a'],pack['b'],pack['edge'],pack['known']),-1))
        values=torch.where(pack['is_edge'][:,:,None],edge,self.node(pack['a']))
        values=self.norm(values+self.type_embedding(pack['types'])+
            self.status_embedding(pack['status'])+self.known(pack['known'])+self.fact_projection(supplied))
        values=values.masked_fill(~valid[:,:,None],0.)
        memory=torch.cat((action,values),1);mask=torch.cat((active,valid),1)
        summary_attention=None;losses=[];diagnostics=[]
        if self.mode=='gmt' and summaries:
            summary=action.new_zeros(b,64,2,128)
            summary_mask=torch.zeros(b,64,2,dtype=torch.bool,device=action.device)
            summary_attention=action.new_zeros(b,64,4,2,n)
            for batch in range(b):
                for k in active[batch].nonzero().flatten().tolist():
                    for role in range(2):
                        allowed=routes[batch,k]&(pack['is_edge'][batch] if role else ~pack['is_edge'][batch])
                        selected=allowed.nonzero().flatten()
                        if not selected.numel():continue
                        query=self.queries[role]+self.condition(action[batch,k])
                        output,attention=self.tokenizer(query[None,None],values[batch,selected][None],
                            values[batch,selected][None],need_weights=True,average_attn_weights=False)
                        vector=self.summary_norm(output[0,0])
                        summary[batch,k,role]=vector;summary_mask[batch,k,role]=True
                        summary_attention[batch,k,:,role,selected]=attention[0,:,0]
                        target=self.targets(pack,batch,k,role)
                        prediction=self.readout(vector)
                        nt=len(TYPES);nf=len(FACT_NAMES)//2
                        presence=F.binary_cross_entropy_with_logits(prediction[:nt],target['presence'])
                        counts=F.smooth_l1_loss(prediction[nt:2*nt],target['counts'])
                        standardized=(target['numeric']-self.fact_mean[role])/self.fact_scale[role].clamp_min(.01)
                        numeric=(F.smooth_l1_loss(prediction[2*nt:2*nt+nf],standardized,reduction='none')*target['known']).sum()/target['known'].sum().clamp_min(1)
                        known=F.mse_loss(prediction[2*nt+nf:].sigmoid(),target['coverage'])
                        losses.append(presence+counts+numeric+.25*known)
                        diagnostics.append(dict(batch=batch,action=k,role=role,
                            prediction=prediction,target=target))
            memory=torch.cat((memory,summary.flatten(1,2)),1)
            mask=torch.cat((mask,summary_mask.flatten(1,2)),1)
            both=summary_mask.all(-1)
            cosine=F.cosine_similarity(summary[:,:,0],summary[:,:,1],dim=-1)
            diversity=F.relu(cosine[both]-.85).square().mean() if both.any() else action.sum()*0
            aux=torch.stack(losses).mean()+.05*diversity if losses else action.sum()*0
        else:
            aux=action.sum()*0;diversity=aux
        total=memory.shape[1]
        relation=torch.zeros(b,total,total,dtype=torch.bool,device=action.device)
        relation[:,:64,:64]=torch.eye(64,dtype=torch.bool,device=action.device)[None]&active[:,:,None]&active[:,None]
        relation[:,:64,64:64+n]=routes
        if summary_attention is not None:
            for k in range(64):relation[:,k,64+n+2*k:64+n+2*k+2]=summary_mask[:,k]
        relation &=mask[:,:,None]&mask[:,None]
        if not torch.isfinite(memory).all():raise ValueError('Nonfinite semantic memory')
        return dict(memory=memory,mask=mask,relation=relation,summary_attention=summary_attention,
                    auxiliary_loss=aux,diversity_loss=diversity,diagnostics=diagnostics)

    @staticmethod
    def targets(pack,batch,k,role):
        selected=pack['routes'][batch,k]&pack['valid'][batch]
        selected &= pack['is_edge'][batch] if role else ~pack['is_edge'][batch]
        facts=pack['facts'][batch,selected]
        counts=torch.bincount(pack['types'][batch,selected],minlength=len(TYPES)).float()
        known=facts[:,1::2]
        numeric=(facts[:,0::2]*known).sum(0)/known.sum(0).clamp_min(1)
        return dict(presence=counts.gt(0).float(),counts=torch.log2(1+counts)/4,
            numeric=numeric,known=known.sum(0).gt(0).float(),coverage=known.mean(0))
