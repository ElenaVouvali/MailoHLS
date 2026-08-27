"""Whole-design Stage-3 reranker.

Unlike DPO over directive tokens, this module assigns one score to a complete
constrained Stage-2 candidate.  Candidate generation and HLS labelling are
deliberately upstream: training records must identify beam outputs and contain
either measured ADP labels or a qualified-surrogate label marked for synthesis
verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RerankerDimensions:
    structural: int
    budget: int
    directive: int
    gnn_qor: int
    hidden: int = 256


class WholeDesignRanker(nn.Module):
    """Scores complete designs without changing the Stage-2 decoder."""

    def __init__(self, dims: RerankerDimensions, dropout: float = 0.1):
        super().__init__()
        input_dim = dims.structural + dims.budget + dims.directive + 1 + dims.gnn_qor
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dims.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims.hidden, dims.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims.hidden, 1),
        )

    def forward(
        self,
        structural_memory: torch.Tensor,
        budget_features: torch.Tensor,
        candidate_directive_embeddings: torch.Tensor,
        stage2_logprobs: torch.Tensor,
        gnn_qor_features: torch.Tensor,
        structural_memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if structural_memory.ndim == candidate_directive_embeddings.ndim + 1:
            if structural_memory_mask is None:
                structural = structural_memory.mean(dim=-2)
            else:
                mask = structural_memory_mask.to(structural_memory.dtype).unsqueeze(-1)
                structural = (structural_memory * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        else:
            structural = structural_memory
        logprobs = stage2_logprobs.unsqueeze(-1) if stage2_logprobs.ndim == structural.ndim - 1 else stage2_logprobs
        features = torch.cat(
            (structural, budget_features, candidate_directive_embeddings, logprobs, gnn_qor_features),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


def constrained_beam_decode(stage2_decoder, *, beam_size: int = 4, **decode_kwargs) -> list[dict]:
    """Request complete, schema-constrained designs from a Stage-2 decoder.

    The inference adapter is expected to expose ``constrained_beam_decode``;
    keeping that interface outside the ranker prevents ranker gradients from
    changing Stage-2. Each returned record is tagged for on-policy pair audits.
    """
    if beam_size < 2:
        raise ValueError("beam_size must be at least 2 for reranking")
    decode = getattr(stage2_decoder, "constrained_beam_decode", None)
    if decode is None:
        raise TypeError("Stage-2 decoder must implement constrained_beam_decode")
    candidates = list(decode(beam_size=beam_size, **decode_kwargs))
    if len(candidates) > beam_size:
        raise ValueError("Stage-2 decoder returned more candidates than beam_size")
    for candidate in candidates:
        if not isinstance(candidate, dict) or "candidate" not in candidate:
            raise ValueError("every beam item must contain a complete 'candidate' design")
        candidate.setdefault("origin", "stage2_beam")
    return candidates


def whole_design_pair_loss(
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -F.logsigmoid((score_chosen - score_rejected) / temperature).mean()


def select_complete_design(candidates: Sequence[object], scores: torch.Tensor):
    if not candidates:
        raise ValueError("cannot select from an empty beam")
    flat_scores = scores.detach().reshape(-1)
    if len(candidates) != flat_scores.numel():
        raise ValueError("candidate and score counts differ")
    return candidates[int(flat_scores.argmax().item())]


def build_complete_design_pairs(
    candidates: Iterable[Mapping[str, object]],
    *,
    include_dataset_candidates: bool = True,
) -> list[dict]:
    """Build within-context pairs from labelled complete candidates.

    Required fields are ``context_id``, ``candidate``, ``adp``, and ``origin``.
    Beam hard negatives use ``origin='stage2_beam'``. Dataset candidates may be
    mixed in, but every surrogate-labelled item must explicitly request later
    synthesis verification.
    """
    by_context: dict[str, list[Mapping[str, object]]] = {}
    for row in candidates:
        missing = {"context_id", "candidate", "adp", "origin"} - set(row)
        if missing:
            raise ValueError(f"candidate is missing fields: {sorted(missing)}")
        if row["origin"] not in {"stage2_beam", "dataset"}:
            raise ValueError("origin must be 'stage2_beam' or 'dataset'")
        if row["origin"] == "dataset" and not include_dataset_candidates:
            continue
        label_source = row.get("label_source", "measured_hls")
        if label_source not in {"measured_hls", "qualified_surrogate"}:
            raise ValueError("label_source must be measured_hls or qualified_surrogate")
        if label_source == "qualified_surrogate" and not row.get("requires_synthesis_verification", False):
            raise ValueError("surrogate labels must require synthesis verification")
        by_context.setdefault(str(row["context_id"]), []).append(row)

    pairs: list[dict] = []
    for context_id, rows in by_context.items():
        ordered = sorted(
            rows,
            key=lambda r: (
                not bool(r.get("feasible", True)),
                float(r["adp"]),
            ),
        )
        if len(ordered) < 2:
            continue
        chosen = ordered[0]
        for rejected in ordered[1:]:
            if chosen["candidate"] == rejected["candidate"]:
                continue
            pairs.append({
                "context_id": context_id,
                "chosen": dict(chosen),
                "rejected": dict(rejected),
                "adp_improvement": float(rejected["adp"]) - float(chosen["adp"]),
                "hard_negative": rejected["origin"] == "stage2_beam",
            })
    return pairs
