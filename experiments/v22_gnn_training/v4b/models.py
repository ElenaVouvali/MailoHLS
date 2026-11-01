"""V4b candidate-independent physical memory and action-difference heads.

The graph encoder never receives candidate, budget, objective, QoR, or pair
labels. Graph tensors are constructed once at full physical capacity; budget
fractions are passed only to the separate preference head/feasibility rule.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from GNN_branch import target_graph_tensor_v22 as tensors
from experiments.target_graph_readiness.experiment import CANDIDATE_DEMAND_DIM, CANDIDATE_DIM


class V4bGraphModel(nn.Module):
    """V4 graph encoder with budget-independent QoR and budget-aware ranking."""

    def __init__(self, hardware_messages: bool):
        super().__init__()
        self.hardware_messages = bool(hardware_messages)
        self.graph = tensors.SmallGraphEncoder(hidden=64)
        self.action_kind = nn.Embedding(3, 4)
        local_input = 128 + CANDIDATE_DIM + CANDIDATE_DEMAND_DIM + 5
        self.local = nn.Sequential(nn.Linear(local_input, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU())
        shared = 128 + 32 + tensors.CONTEXT_DIM
        self.physical_qor = nn.Sequential(nn.Linear(shared, 64), nn.ReLU(), nn.Linear(64, 5))
        self.preference = nn.Sequential(
            nn.Linear(shared + 4, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, graph):
        """Return candidate-independent [1,64,128] memory and pooled program state."""
        if graph.hardware_context.shape[0] != 1:
            raise ValueError('V4b currently encodes one program/physical-target graph at a time')
        # V4 tensor context columns 8:12 are resource availability fractions.
        # Only a full-capacity physical context is accepted in the encoder.
        if graph.hardware_context.shape[-1] != tensors.CONTEXT_DIM or not torch.allclose(
                graph.hardware_context[:, 8:12], torch.ones_like(graph.hardware_context[:, 8:12]),
                atol=1e-7, rtol=0):
            raise ValueError('V4b physical graph memory must use full device capacity, not a requested budget')
        message_graph = graph if self.hardware_messages else tensors.without_hardware_messages(graph)
        return self.graph.memories(message_graph)

    def from_memory(self, graph, memory, pooled, candidates, demands, budget_fraction):
        if memory.shape != (1, graph.registered_mask.shape[1], 128):
            raise ValueError('V4b action memory violates the absolute-slot contract')
        if candidates.ndim != 3 or demands.shape[:2] != candidates.shape[:2]:
            raise ValueError('Candidate and demand tensors must be aligned by action slot')
        if candidates.shape[1] != memory.shape[1] or demands.shape[1] != memory.shape[1]:
            raise ValueError('Candidate features must preserve absolute action slots')
        n = candidates.shape[0]
        kinds = self.action_kind(graph.action_kind_slot.expand(n, -1))
        matched = graph.matched_mask.expand(n, -1).to(candidates.dtype).unsqueeze(-1)
        joined = torch.cat((memory.expand(n, -1, -1), candidates, demands, kinds, matched), dim=-1)
        action = self.local(joined)
        registered = graph.registered_mask.expand(n, -1).to(action.dtype).unsqueeze(-1)
        action = (action * registered).sum(1) / registered.sum(1).clamp_min(1.)
        context = graph.hardware_context.expand(n, -1)
        shared = torch.cat((pooled.expand(n, -1), action, context), dim=-1)
        # Budget does not enter this head. Identical P,d,D,T inputs therefore
        # produce identical physical predictions for every requested budget.
        qor_z = self.physical_qor(shared)
        budget = _budget_tensor(budget_fraction, n, candidates.device, candidates.dtype)
        preference_score = self.preference(torch.cat((shared, budget), dim=-1)).squeeze(-1)
        return {'qor_z': qor_z, 'preference_score': preference_score,
                'action_features': action}

    def forward(self, graph, candidates, demands, budget_fraction=1.0):
        memory, pooled = self.encode(graph)
        return self.from_memory(graph, memory, pooled, candidates, demands, budget_fraction)


class TargetFilmV4bModel(V4bGraphModel):
    """Compiler encoder with zero-gated authenticated physical-profile FiLM.

    `profile_features` must come from the existing capability-vector contract;
    no device-ID embedding, requested budget, objective, candidate or label is
    accepted by the memory modulation.
    """

    def __init__(self, profile_feature_dim):
        super().__init__(hardware_messages=False)
        self.profile_feature_dim = int(profile_feature_dim)
        if self.profile_feature_dim < 1:
            raise ValueError('Profile feature width must be positive')
        self.modulation = nn.Sequential(
            nn.Linear(128 + self.profile_feature_dim + tensors.ACTION_DEMAND_DIM, 128),
            nn.ReLU(), nn.Linear(128, 256))
        self.gate = nn.Parameter(torch.zeros(()))

    def encode(self, graph, profile_features=None):
        memory, pooled = super().encode(graph)
        if profile_features is None:
            raise ValueError('Target-FiLM memory requires authenticated profile features')
        features = torch.as_tensor(profile_features, dtype=memory.dtype, device=memory.device)
        if features.shape != (1, self.profile_feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Physical profile features must be finite [1,profile_feature_dim]')
        active = (graph.registered_mask & graph.matched_mask).squeeze(0)
        if not active.any():
            return memory, pooled
        relevance = graph.action_demand[0]
        target = features.expand(memory.shape[1], -1)
        raw = self.modulation(torch.cat((memory[0], target, relevance), dim=-1))
        gamma, beta = raw.chunk(2, dim=-1)
        changed = memory[0] + torch.tanh(self.gate) * (
            torch.tanh(gamma) * memory[0] + beta)
        result = memory.clone()
        result[0, active] = changed[active]
        return result, pooled

    def forward(self, graph, candidates, demands, budget_fraction=1.0,
                profile_features=None):
        memory, pooled = self.encode(graph, profile_features)
        return self.from_memory(graph, memory, pooled, candidates, demands, budget_fraction)


class NoGraphControl(nn.Module):
    """Candidate/target control with no learned structural-memory encoder.

    It shares authenticated candidate-demand descriptors and action masks with
    the graph arms, but has no compiler-graph message-passing or action-memory
    parameters. Thus the control isolates learned V4 memory beyond those
    common analytic candidate features; it is not a claim that candidate
    demand itself is graph-free.
    """

    def __init__(self, hidden=192):
        super().__init__()
        self.action_kind = nn.Embedding(3, 4)
        self.action = nn.Sequential(
            nn.Linear(CANDIDATE_DIM + CANDIDATE_DEMAND_DIM + 5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        shared = hidden + tensors.CONTEXT_DIM
        self.physical_qor = nn.Sequential(nn.Linear(shared, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        self.preference = nn.Sequential(nn.Linear(shared + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def from_graph_context(self, graph, candidates, demands, budget_fraction=1.0):
        n = candidates.shape[0]
        kind = self.action_kind(graph.action_kind_slot.expand(n, -1))
        matched = graph.matched_mask.expand(n, -1).to(candidates.dtype).unsqueeze(-1)
        action = self.action(torch.cat((candidates, demands, kind, matched), dim=-1))
        registered = graph.registered_mask.expand(n, -1).to(action.dtype).unsqueeze(-1)
        action = (action * registered).sum(1) / registered.sum(1).clamp_min(1.)
        context = graph.hardware_context.expand(n, -1)
        shared = torch.cat((action, context), dim=-1)
        qor_z = self.physical_qor(shared)
        budget = _budget_tensor(budget_fraction, n, candidates.device, candidates.dtype)
        return {'qor_z': qor_z,
                'preference_score': self.preference(torch.cat((shared, budget), -1)).squeeze(-1),
                'action_features': action}

    def forward(self, graph, candidates, demands, budget_fraction=1.0):
        return self.from_graph_context(graph, candidates, demands, budget_fraction)


class ActionDifferenceComparator(nn.Module):
    """Antisymmetric attention over candidate-conditioned absolute actions.

    Positive pair logits mean the left candidate is preferred. Attention uses
    symmetric action context and absolute candidate differences; the final
    weighted projection uses signed differences, guaranteeing antisymmetry.
    Identical candidates yield a zero logit exactly.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.kind = nn.Embedding(3, 4)
        self.action = nn.Sequential(
            nn.Linear(128 + CANDIDATE_DIM + CANDIDATE_DEMAND_DIM + 5, hidden),
            nn.ReLU(), nn.Linear(hidden, hidden))
        self.attention = nn.Sequential(nn.Linear(hidden * 2 + tensors.CONTEXT_DIM + 4, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 1))
        self.signed_projection = nn.Linear(hidden, 1, bias=False)

    def _candidate_actions(self, graph, memory, candidates, demands):
        n = candidates.shape[0]
        kind = self.kind(graph.action_kind_slot.expand(n, -1))
        matched = graph.matched_mask.expand(n, -1).to(candidates.dtype).unsqueeze(-1)
        values = torch.cat((memory.expand(n, -1, -1), candidates, demands, kind, matched), -1)
        return self.action(values)

    def forward(self, graph, memory, candidate_a, demand_a, candidate_b, demand_b,
                budget_fraction=1.0):
        if memory.shape[0] != 1:
            raise ValueError('Pair comparator expects one shared program/target memory')
        if candidate_a.shape != candidate_b.shape or demand_a.shape != demand_b.shape:
            raise ValueError('Pair candidate tensors must have matching shapes')
        ha = self._candidate_actions(graph, memory, candidate_a, demand_a)
        hb = self._candidate_actions(graph, memory, candidate_b, demand_b)
        delta = ha - hb
        symmetric = 0.5 * (ha + hb)
        n, slots, _ = delta.shape
        budget = _budget_tensor(budget_fraction, n, delta.device, delta.dtype).unsqueeze(1)
        budget = budget.expand(-1, slots, -1)
        context = graph.hardware_context.expand(n, -1).unsqueeze(1).expand(-1, slots, -1)
        # The same explicit numerical target context is supplied to every arm;
        # objective identity is intentionally absent from the GNN contract.
        logits = self.attention(torch.cat((delta.abs(), symmetric, context, budget), -1)).squeeze(-1)
        valid = graph.registered_mask.expand(n, -1).clone()
        # Only changed registered actions receive comparison attention. An
        # unresolved action can participate only when its explicit candidate
        # representation changes; no missing value is filled with a default.
        valid &= delta.detach().abs().sum(-1) > 0
        empty = ~valid.any(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        logits[empty] = 0.
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(~valid, 0.)
        aggregate = (weights.unsqueeze(-1) * delta).sum(1)
        score = self.signed_projection(aggregate).squeeze(-1)
        return torch.where(empty, torch.zeros_like(score), score), weights


def _budget_tensor(value, count, device, dtype):
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim == 0:
        result = result.repeat(count, 4)
    elif result.ndim == 1:
        if result.numel() == 1:
            result = result.repeat(count, 4)
        elif result.numel() == 4:
            result = result.unsqueeze(0).expand(count, -1)
    if result.shape != (count, 4) or not torch.isfinite(result).all():
        raise ValueError('Budget fractions must be scalar or [N,4] finite fractions')
    if (result < 0).any() or (result > 1).any():
        raise ValueError('Budget fractions must lie in [0,1]')
    return result


def physical_qor_loss(prediction, target):
    """Pointwise SmoothL1 on training-normalized log physical labels."""
    if prediction.shape != target.shape or prediction.shape[-1] != 5:
        raise ValueError('QoR prediction/target must have identical [N,5] shapes')
    return F.smooth_l1_loss(prediction, target)


def hybrid_loss(qor_prediction, qor_target, pair_logits, pair_preferred_left,
                pair_weight=0.1):
    """Physical regression plus normalized pairwise logistic preference loss."""
    if pair_weight < 0:
        raise ValueError('pair_weight must be nonnegative')
    point = physical_qor_loss(qor_prediction, qor_target)
    if pair_logits.numel() == 0:
        return point, {'physical_qor': point.detach(), 'pairwise': point.detach().new_zeros(())}
    labels = torch.as_tensor(pair_preferred_left, dtype=pair_logits.dtype,
                             device=pair_logits.device)
    if labels.shape != pair_logits.shape or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Pair labels must be aligned booleans (left preferred)')
    signed = torch.where(labels.bool(), pair_logits, -pair_logits)
    pair = F.softplus(-signed).mean()
    return point + pair_weight * pair, {'physical_qor': point.detach(), 'pairwise': pair.detach()}


def feasibility_from_physical(predicted_resources, capacities, budget_fraction):
    """Budget feasibility derived from predicted physical [BRAM,DSP,FF,LUT]."""
    resources = torch.as_tensor(predicted_resources)
    caps = torch.as_tensor(capacities, dtype=resources.dtype, device=resources.device)
    if resources.shape[-1] != 4 or caps.shape[-1] != 4 or (caps <= 0).any():
        raise ValueError('Expected four resource values and positive capacities')
    budget = torch.as_tensor(budget_fraction, dtype=resources.dtype, device=resources.device)
    if not torch.isfinite(resources).all() or not torch.isfinite(caps).all():
        raise ValueError('Nonfinite resource prediction or capacity')
    if (budget < 0).any() or (budget > 1).any():
        raise ValueError('Budget fractions must lie in [0,1]')
    return (resources <= caps * budget.unsqueeze(-1)).all(-1)


def pairwise_pool_order(pair_scores, stable_ids):
    """Rank by deterministic net wins; this is a tournament approximation.

    `pair_scores[i,j] > 0` means i is preferred to j. Nontransitive cycles are
    possible; stable_ids resolves equal net wins and must be unique.
    """
    scores = torch.as_tensor(pair_scores)
    n = len(stable_ids)
    if scores.shape != (n, n) or len(set(stable_ids)) != n:
        raise ValueError('Pair score matrix and unique stable IDs must align')
    if not torch.allclose(scores, -scores.T, atol=1e-6, rtol=0):
        raise ValueError('Pair score matrix must be antisymmetric')
    wins = (scores > 0).to(torch.int64).sum(1) + 0.5 * (scores == 0).to(torch.float32).sum(1)
    order = sorted(range(n), key=lambda i: (-float(wins[i]), str(stable_ids[i])))
    return order
