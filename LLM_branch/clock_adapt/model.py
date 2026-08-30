import math
import torch
from torch import nn


class AutoClockOverrideHead(nn.Module):
    """Binary override gate plus a conditional slower-clock head."""
    def __init__(self, hidden_dim: int, n_clocks: int, dropout: float = 0.10):
        super().__init__()
        mid = max(64, hidden_dim // 2)
        self.override_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, mid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mid, 1),
        )
        self.clock_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, mid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mid, n_clocks),
        )

    def forward(self, z):
        return self.override_gate(z).squeeze(-1), self.clock_head(z)

class ClockSelector(nn.Module):
    """One-pass candidate-conditioned selector over frozen structural memory."""
    def __init__(self, mem_dim=128, context_dim=12, hidden_dim=64, dropout=.1):
        super().__init__(); self.mem_dim=mem_dim; self.context_dim=context_dim
        self.query=nn.Sequential(nn.LayerNorm(context_dim),nn.Linear(context_dim,hidden_dim),nn.GELU())
        self.key=nn.Linear(mem_dim,hidden_dim,bias=False); self.value=nn.Linear(mem_dim,hidden_dim,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(hidden_dim+context_dim),nn.Linear(hidden_dim+context_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self, memory, memory_mask=None, candidate_context=None):
        if candidate_context is None or memory_mask is None:
            raise ValueError("memory_mask and candidate_context are required")
        squeeze = memory.ndim == 2
        if squeeze:
            memory, memory_mask, candidate_context = memory.unsqueeze(0), memory_mask.unsqueeze(0), candidate_context.unsqueeze(0)
        if memory.ndim != 3 or memory_mask.ndim != 2 or candidate_context.ndim != 3:
            raise ValueError("expected memory [B,S,D], mask [B,S], context [B,C,K]")
        if memory.shape[:2] != memory_mask.shape or memory.shape[0] != candidate_context.shape[0]:
            raise ValueError("inconsistent batched selector shapes")
        valid = memory_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Invalid or empty structural memory mask")
        q=self.query(candidate_context); k=self.key(memory); v=self.value(memory)
        scores=torch.einsum("bch,bsh->bcs",q,k)/math.sqrt(k.shape[-1])
        scores=scores.masked_fill(~valid[:,None,:],-torch.inf)
        pooled=torch.softmax(scores,dim=-1) @ v
        out=self.head(torch.cat([pooled,candidate_context],dim=-1)).squeeze(-1)
        return out.squeeze(0) if squeeze else out


class AutoClockOverrideSelector(nn.Module):
    """Candidate-aware structural encoder with an explicit override gate."""
    def __init__(self, mem_dim=128, context_dim=12, hidden_dim=64,
                 n_clocks=3, dropout=.1):
        super().__init__()
        self.mem_dim, self.context_dim, self.hidden_dim = mem_dim, context_dim, hidden_dim
        self.n_clocks = n_clocks
        self.query = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU())
        self.key = nn.Linear(mem_dim, hidden_dim, bias=False)
        self.value = nn.Linear(mem_dim, hidden_dim, bias=False)
        self.head = AutoClockOverrideHead(hidden_dim + context_dim, n_clocks, dropout)

    def forward(self, memory, memory_mask=None, candidate_context=None):
        if memory_mask is None or candidate_context is None:
            raise ValueError("memory_mask and candidate_context are required")
        squeeze = memory.ndim == 2
        if squeeze:
            memory = memory.unsqueeze(0); memory_mask = memory_mask.unsqueeze(0); candidate_context = candidate_context.unsqueeze(0)
        if memory.ndim != 3 or memory_mask.ndim != 2 or candidate_context.ndim != 3:
            raise ValueError("expected memory [B,S,D], mask [B,S], context [B,C,K]")
        valid = memory_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Invalid or empty structural memory mask")
        q, k, v = self.query(candidate_context), self.key(memory), self.value(memory)
        scores = torch.einsum("bch,bsh->bcs", q, k) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        pooled = torch.softmax(scores, dim=-1) @ v
        # Keep device/budget/objective information upstream of the gate by
        # retaining the complete context, while removing clock-specific noise
        # through a mean context representation.
        z = torch.cat([pooled.mean(dim=1), candidate_context.mean(dim=1)], dim=-1)
        override, clocks = self.head(z)
        if squeeze:
            return override.squeeze(0), clocks.squeeze(0)
        return override, clocks

ClockResidualSelector = ClockSelector
