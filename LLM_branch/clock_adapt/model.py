import math
import torch
from torch import nn

class ClockSelector(nn.Module):
    """One-pass candidate-conditioned selector over frozen structural memory."""
    def __init__(self, mem_dim=128, context_dim=9, hidden_dim=64, dropout=.1):
        super().__init__(); self.mem_dim=mem_dim; self.context_dim=context_dim
        self.query=nn.Sequential(nn.LayerNorm(context_dim),nn.Linear(context_dim,hidden_dim),nn.GELU())
        self.key=nn.Linear(mem_dim,hidden_dim,bias=False); self.value=nn.Linear(mem_dim,hidden_dim,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(hidden_dim+context_dim),nn.Linear(hidden_dim+context_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self, memory, memory_mask=None, candidate_context=None):
        # Also accept the cached [C, 137] representation used by extract_features.
        if candidate_context is None:
            candidate_context=memory[..., self.mem_dim:self.mem_dim+self.context_dim]
            memory=memory[..., :self.mem_dim].unsqueeze(-2)
        if memory.dim()==2: memory=memory.unsqueeze(0)
        if candidate_context.dim()==1: candidate_context=candidate_context.unsqueeze(0)
        q=self.query(candidate_context); k=self.key(memory); v=self.value(memory); score=q.unsqueeze(1).matmul(k.transpose(-1,-2))/math.sqrt(k.shape[-1])
        if memory_mask is not None: score=score.masked_fill(~memory_mask.bool().view(1,1,-1),-torch.inf)
        pooled=torch.softmax(score,dim=-1).matmul(v).squeeze(1); return self.head(torch.cat([pooled,candidate_context],dim=-1)).squeeze(-1)

ClockResidualSelector = ClockSelector
