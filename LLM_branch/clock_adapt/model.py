import math
import torch
from torch import nn

class ClockSelector(nn.Module):
    """One-pass candidate-conditioned selector over frozen structural memory."""
    def __init__(self, mem_dim=128, context_dim=12, hidden_dim=64, dropout=.1):
        super().__init__(); self.mem_dim=mem_dim; self.context_dim=context_dim
        self.query=nn.Sequential(nn.LayerNorm(context_dim),nn.Linear(context_dim,hidden_dim),nn.GELU())
        self.key=nn.Linear(mem_dim,hidden_dim,bias=False); self.value=nn.Linear(mem_dim,hidden_dim,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(hidden_dim+context_dim),nn.Linear(hidden_dim+context_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self, memory, memory_mask=None, candidate_context=None):
        if memory.ndim != 2 or candidate_context is None or candidate_context.ndim != 2:
            raise ValueError("memory must be [S,D] and candidate_context must be [C,K]")
        valid = memory_mask.bool().reshape(-1)
        if valid.numel() != memory.shape[0] or not valid.any():
            raise ValueError("Invalid or empty structural memory mask")
        memory = memory[valid]
        q=self.query(candidate_context); k=self.key(memory); v=self.value(memory)
        score=q.matmul(k.transpose(0,1))/math.sqrt(k.shape[-1])
        pooled=torch.softmax(score,dim=-1).matmul(v)
        return self.head(torch.cat([pooled,candidate_context],dim=-1)).squeeze(-1)

ClockResidualSelector = ClockSelector
