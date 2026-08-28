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
import json
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RerankerDimensions:
    structural: int
    budget: int
    directive: int
    gnn_qor: int = 0  # deprecated; retained for checkpoint/config compatibility
    hidden: int = 256


class WholeDesignRanker(nn.Module):
    """Scores complete designs without changing the Stage-2 decoder."""

    def __init__(self, dims: RerankerDimensions, dropout: float = 0.1):
        super().__init__()
        # GNN memory is structural context, not a calibrated QoR oracle.
        input_dim = dims.structural + dims.budget + dims.directive + 1
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
        gnn_qor_features: torch.Tensor | None = None,
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
        features = torch.cat((structural, budget_features,
                              candidate_directive_embeddings, logprobs), dim=-1)
        if features.shape[-1] != self.network[1].in_features:
            raise ValueError("ranker input width mismatch (gnn_qor_features is disabled)")
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
    qor_chosen: torch.Tensor | None = None,
    qor_rejected: torch.Tensor | None = None,
    median_gap: float | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    loss = F.softplus(-(score_chosen - score_rejected) / temperature)
    if qor_chosen is not None or qor_rejected is not None:
        if qor_chosen is None or qor_rejected is None:
            raise ValueError("qor_chosen and qor_rejected must be provided together")
        gap = torch.log(torch.as_tensor(qor_rejected, device=loss.device) /
                        torch.as_tensor(qor_chosen, device=loss.device)).clamp_min(0)
        med = float(median_gap if median_gap is not None else gap.detach().median().item())
        weight = (gap / max(med, 1e-8)).sqrt().clamp(0.5, 2.0)
        loss = loss * weight
    return loss.mean()


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


def canonical_candidate_key(row: Mapping[str, object]) -> tuple:
    """Exact configuration identity used for measured-data joins."""
    clock = row.get("clock_period_ns", row.get("clock_period"))
    context = row.get("context_id", row.get("kernel_name", ""))
    return (str(context), str(row.get("kernel_name", "")),
            str(row.get("device", "")),
            None if clock in (None, "") else round(float(clock), 8),
            str(row.get("resource_budget_id", "")), str(row.get("objective", "")),
            str(row.get("candidate", row.get("target", ""))))


class CandidateBankWriter:
    """JSONL append writer. Existing identities are skipped for resume."""
    REQUIRED = {"context_id", "kernel_name", "device", "clock_period_ns",
                "resource_budget_id", "objective", "candidate",
                "stage2_mean_site_logprob", "stage2_sum_logprob",
                "sample_temperature", "origin"}

    def __init__(self, path: str):
        import os
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.seen = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try: self.seen.add(self.identity(json.loads(line)))
                        except (ValueError, KeyError): pass
        self.handle = open(path, "a", encoding="utf-8", buffering=1)

    @staticmethod
    def identity(row):
        return (canonical_candidate_key(row), int(row.get("sample_id", -1)))

    def write(self, row: Mapping[str, object]) -> bool:
        missing = self.REQUIRED - set(row)
        if missing: raise ValueError(f"candidate record missing fields: {sorted(missing)}")
        if row["origin"] != "stage2_sample": raise ValueError("origin must be stage2_sample")
        key = self.identity(row)
        if key in self.seen: return False
        self.handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        self.seen.add(key)
        return True

    def close(self): self.handle.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def exact_join_candidates(candidates: Iterable[Mapping[str, object]],
                         measured: Iterable[Mapping[str, object]]):
    """Join only exact configurations; unmatched promising rows form a queue."""
    measured_index = {canonical_candidate_key(r): r for r in measured}
    labelled, synthesis_queue = [], []
    for row in candidates:
        hit = measured_index.get(canonical_candidate_key(row))
        if hit is None:
            if row.get("promising", True): synthesis_queue.append(dict(row))
            continue
        out = dict(row)
        out["qor"] = hit.get("qor", hit.get("adp"))
        out["label_source"] = "measured_hls"
        labelled.append(out)
    return labelled, synthesis_queue
