import math
import torch
from torch import nn


def select_auto_clock_decision(
    override_logits,
    clock_logits,
    clocks,
    switch_threshold,
    feasibility_logits=None,
    feasibility_threshold=0.5,
    feasibility_log_weight=2.0,
):
    """Apply the single AUTO-clock selection contract."""
    periods = torch.as_tensor(
        clocks, dtype=clock_logits.dtype, device=clock_logits.device
    )
    fast_idx = int(periods.argmin())
    slow_mask = periods > periods[fast_idx]
    override_probability = float(torch.sigmoid(override_logits))
    if feasibility_logits is None:
        candidate_idx = (
            int(clock_logits.masked_fill(~slow_mask, -torch.inf).argmax())
            if bool(slow_mask.any()) else fast_idx
        )
        selected_idx = (
            candidate_idx
            if override_probability >= float(switch_threshold)
            else fast_idx
        )
        return {
            "selected_idx": selected_idx,
            "candidate_idx": candidate_idx,
            "fast_idx": fast_idx,
            "override_probability": override_probability,
            "risky_fast": False,
            "force_override": False,
            "no_predicted_feasible": False,
            "adjusted_logits": clock_logits,
            "feasibility_probability": None,
            "predicted_feasible_mask": None,
        }

    pfeas = torch.sigmoid(feasibility_logits)
    feasible = pfeas >= float(feasibility_threshold)
    adjusted_logits = (
        clock_logits
        + float(feasibility_log_weight)
        * torch.log(pfeas.clamp_min(1e-6))
    )
    fast_feasible = bool(feasible[fast_idx])
    feasible_slow = slow_mask & feasible
    if bool(feasible_slow.any()):
        candidate_idx = int(
            adjusted_logits.masked_fill(~feasible_slow, -torch.inf).argmax()
        )
    else:
        candidate_idx = fast_idx
    force_override = (not fast_feasible) and bool(feasible_slow.any())
    gate_override = override_probability >= float(switch_threshold)
    if force_override:
        selected_idx = candidate_idx
    elif fast_feasible and gate_override and bool(feasible_slow.any()):
        selected_idx = candidate_idx
    elif fast_feasible:
        selected_idx = fast_idx
    else:
        selected_idx = int(pfeas.argmax())
    return {
        "selected_idx": selected_idx,
        "candidate_idx": candidate_idx,
        "fast_idx": fast_idx,
        "override_probability": override_probability,
        "risky_fast": not fast_feasible,
        "force_override": force_override,
        "no_predicted_feasible": not bool(feasible.any()),
        "adjusted_logits": adjusted_logits,
        "feasibility_probability": pfeas,
    }


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


class AutoClockOverrideSelectorV5(nn.Module):
    """Candidate-preserving AUTO selector with explicit fast/slow comparison."""
    def __init__(self, mem_dim=128, context_dim=12, hidden_dim=64,
                 n_clocks=3, dropout=.1):
        super().__init__()
        self.mem_dim, self.context_dim, self.hidden_dim, self.n_clocks = mem_dim, context_dim, hidden_dim, n_clocks
        self.query = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU())
        self.key = nn.Linear(mem_dim, hidden_dim, bias=False)
        self.value = nn.Linear(mem_dim, hidden_dim, bias=False)
        zdim, mid = hidden_dim + context_dim, max(64, (hidden_dim + context_dim) // 2)
        self.clock_score = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, 1))
        self.override_gate = nn.Sequential(nn.LayerNorm(3 * zdim), nn.Linear(3 * zdim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, 1))

    def forward(self, memory, memory_mask=None, candidate_context=None):
        if memory_mask is None or candidate_context is None:
            raise ValueError("memory_mask and candidate_context required")
        squeeze = memory.ndim == 2
        if squeeze:
            memory = memory.unsqueeze(0); memory_mask = memory_mask.unsqueeze(0); candidate_context = candidate_context.unsqueeze(0)
        valid = memory_mask.bool()
        if not valid.any(dim=1).all(): raise ValueError("Invalid or empty structural memory mask")
        q, k, v = self.query(candidate_context), self.key(memory), self.value(memory)
        scores = torch.einsum("bch,bsh->bcs", q, k) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        pooled = torch.softmax(scores, dim=-1) @ v
        zc = torch.cat([pooled, candidate_context], dim=-1)
        clock_logits = self.clock_score(zc).squeeze(-1)
        fast_idx = candidate_context[..., 8].argmin(dim=-1)
        batch = torch.arange(zc.shape[0], device=zc.device)
        z_fast = zc[batch, fast_idx]
        slow_mask = torch.ones_like(clock_logits, dtype=torch.bool); slow_mask[batch, fast_idx] = False
        z_slow = (zc * slow_mask[..., None]).sum(dim=1) / slow_mask.sum(dim=1, keepdim=True).clamp_min(1)
        override = self.override_gate(torch.cat([z_fast, z_slow, z_slow - z_fast], dim=-1)).squeeze(-1)
        return (override.squeeze(0), clock_logits.squeeze(0)) if squeeze else (override, clock_logits)


class AutoClockOverrideSelectorV6(AutoClockOverrideSelectorV5):
    """V5 plus a learned per-clock feasibility safety head."""
    def __init__(self, mem_dim=128, context_dim=12, hidden_dim=64,
                 n_clocks=3, dropout=.1):
        super().__init__(mem_dim, context_dim, hidden_dim, n_clocks, dropout)
        zdim, mid = hidden_dim + context_dim, max(64, (hidden_dim + context_dim) // 2)
        self.feasibility_score = nn.Sequential(
            nn.LayerNorm(zdim), nn.Linear(zdim, mid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mid, 1),
        )

    def forward(self, memory, memory_mask=None, candidate_context=None):
        if memory_mask is None or candidate_context is None:
            raise ValueError("memory_mask and candidate_context required")
        squeeze = memory.ndim == 2
        if squeeze:
            memory = memory.unsqueeze(0); memory_mask = memory_mask.unsqueeze(0); candidate_context = candidate_context.unsqueeze(0)
        valid = memory_mask.bool()
        if not valid.any(dim=1).all(): raise ValueError("Invalid or empty structural memory mask")
        q, k, v = self.query(candidate_context), self.key(memory), self.value(memory)
        scores = torch.einsum("bch,bsh->bcs", q, k) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        pooled = torch.softmax(scores, dim=-1) @ v
        zc = torch.cat([pooled, candidate_context], dim=-1)
        clock_logits = self.clock_score(zc).squeeze(-1)
        feasibility_logits = self.feasibility_score(zc).squeeze(-1)
        fast_idx = candidate_context[..., 8].argmin(dim=-1); batch = torch.arange(zc.shape[0], device=zc.device)
        z_fast = zc[batch, fast_idx]; slow_mask = torch.ones_like(clock_logits, dtype=torch.bool); slow_mask[batch, fast_idx] = False
        z_slow = (zc * slow_mask[..., None]).sum(dim=1) / slow_mask.sum(dim=1, keepdim=True).clamp_min(1)
        override = self.override_gate(torch.cat([z_fast, z_slow, z_slow-z_fast], dim=-1)).squeeze(-1)
        if squeeze: return override.squeeze(0), clock_logits.squeeze(0), feasibility_logits.squeeze(0)
        return override, clock_logits, feasibility_logits


class AutoClockOverrideSelectorV7(AutoClockOverrideSelectorV6):
    """Corrected v6 deployment/training contract with explicit feasibility labels."""
    pass

ClockResidualSelector = ClockSelector
