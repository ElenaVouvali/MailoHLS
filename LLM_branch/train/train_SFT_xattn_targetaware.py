#!/usr/bin/env python3
"""
train_SFT_xattn_targetaware.py
==============================

Clean target-aware entry point for MailoHLS.

Place this file beside ``train_SFT_xattn_new.py`` and run this file instead.
It reuses the tested LoRA, deterministic RHS, contrastive-loss, Trainer, and
HARP/MLIR cross-attention implementation from the original script, while
replacing only the target-conditioning/data-selection layer.

Implemented changes
-------------------
1. Shared random *four-dimensional* resource budgets:
   - budgets are sampled once per (kernel, device), then applied to every
     available clock and every design point in that candidate pool;
   - this preserves valid "best feasible design under budget" supervision;
   - independent, correlated, full-device, and boundary-focused budgets are
     mixed;
   - each budget keeps a compact union of high-quality candidates for all
     objectives, avoiding an enormous intermediate expansion.

2. Optional target clock:
   - specified mode: prompt contains the requested period;
   - automatic mode: prompt contains <CLK=AUTO>;
   - the output schema always begins with:
         <CLOCK>
         selected_clock_period_ns = ...
     followed by the unchanged deterministic MailoHLS directive schema;
   - automatic examples rank candidates jointly across all available periods.

3. Stable and reproducible:
   - budget generation uses SHA-256-derived seeds, not Python's randomized hash;
   - validation/test conditions are deterministic;
   - automatic-frequency subsampling is deterministic.

4. Cross-device conditioning:
   - prompts contain both absolute available resources and percentages of the
     selected device capacity;
   - optional device-token dropout remains supported.

This is intentionally a small extension module.  The large and already tested
model/trainer code remains in ``train_SFT_xattn_new.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Load the original MailoHLS trainer from the same directory.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "train_SFT_xattn_new.py"

if not BASE_PATH.is_file():
    raise FileNotFoundError(
        f"Expected the original trainer beside this file: {BASE_PATH}"
    )

_spec = importlib.util.spec_from_file_location("mailohls_sft_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import {BASE_PATH}")

base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


# ---------------------------------------------------------------------------
# Runtime configuration populated by this entry point.
# ---------------------------------------------------------------------------

@dataclass
class TargetAwareConfig:
    budget_mode: str = "random"
    random_budgets_per_case: int = 16
    min_budget_frac: float = 0.10
    min_feasible_candidates: int = 3
    candidate_pool_per_objective: int = 24
    auto_frequency_fraction: float = 0.30
    min_auto_clock_count: int = 2
    seed: int = 123


CFG = TargetAwareConfig()


# ---------------------------------------------------------------------------
# Prompt and target-platform schema.
# ---------------------------------------------------------------------------

AUTO_PERIOD_TOKEN = "<CLK=AUTO>"
CLOCK_ANCHOR_TOKEN = "<CLOCK>"

PROMPT_TEMPLATE = """
### Role: Expert FPGA/HLS engineer.

### Task:
The kernel marks each directive site with a source marker <SRC_Lk>.
Select the clock period and directive RHS values for the optimization goal.
When Target clock period is <CLK=AUTO>, choose the best supported period.
Anchors and directive names are fixed by the source code.

### Target Platform
Device: {device_token}
Target clock period: {period_token}

Available resources:
BRAM_18K={avail_bram} ({avail_bram_pct:.1f}% of device)
DSP={avail_dsp} ({avail_dsp_pct:.1f}% of device)
FF={avail_ff} ({avail_ff_pct:.1f}% of device)
LUT={avail_lut} ({avail_lut_pct:.1f}% of device)

### Objective
{obj_token}

### Kernel
{code}

### Selected Clock and Directives
""".lstrip()


def _norm_device(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_clock(value: Any) -> float:
    return round(float(value), 2)


def _clock_of(row: Mapping[str, Any]) -> float:
    value = row.get("clock_period", row.get("Clock_Period_nsec"))
    if value in (None, ""):
        raise ValueError(
            f"Row for {row.get('kernel_name', '<unknown>')} has no clock period"
        )
    return _norm_clock(value)


def period_token_from_clock(clock_period: Any) -> str:
    cp = _norm_clock(clock_period)
    for known_cp, token in base.PERIOD_TOKEN_MAP.items():
        if abs(cp - float(known_cp)) < 0.02:
            return token

    # Preserve decimals: 4.50 ns -> <CLK=4P5NS>, never truncate to 4 ns.
    text = f"{cp:.2f}".rstrip("0").rstrip(".").replace(".", "P")
    return f"<CLK={text}NS>"


def _available_resources(row: Mapping[str, Any]) -> Dict[str, int]:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = base.DEVICE_RESOURCES.get(device)
    if caps is None:
        raise ValueError(f"Unsupported device: {device!r}")

    out: Dict[str, int] = {}
    for resource in base.RESOURCE_KEYS:
        field = base.AVAIL_FIELD_BY_RESOURCE[resource]
        raw = row.get(field)
        out[resource] = (
            int(round(float(raw)))
            if raw not in (None, "")
            else int(caps[resource])
        )
    return out


def target_prompt_fields(
    row: Optional[dict],
    device_token_dropout: float = 0.0,
) -> dict:
    row = row or {}
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = base.DEVICE_RESOURCES.get(device)
    if caps is None:
        raise ValueError(f"Unsupported device: {device!r}")

    device_token = base.DEVICE_TOKEN_MAP.get(
        device, base.UNKNOWN_DEVICE_TOKEN
    )
    if (
        device_token_dropout > 0.0
        and random.random() < float(device_token_dropout)
    ):
        device_token = base.UNKNOWN_DEVICE_TOKEN

    frequency_mode = str(row.get("frequency_mode", "specified")).lower()
    period_token = (
        AUTO_PERIOD_TOKEN
        if frequency_mode == "auto"
        else period_token_from_clock(_clock_of(row))
    )

    avail = _available_resources(row)

    def percentage(resource: str) -> float:
        return 100.0 * float(avail[resource]) / float(caps[resource])

    return {
        "device_token": device_token,
        "period_token": period_token,
        "avail_bram": avail["BRAM_18K"],
        "avail_dsp": avail["DSP"],
        "avail_ff": avail["FF"],
        "avail_lut": avail["LUT"],
        "avail_bram_pct": percentage("BRAM_18K"),
        "avail_dsp_pct": percentage("DSP"),
        "avail_ff_pct": percentage("FF"),
        "avail_lut_pct": percentage("LUT"),
    }


def build_prompt(
    code: str,
    obj_mode: str,
    row: Optional[dict] = None,
    device_token_dropout: float = 0.0,
) -> str:
    return PROMPT_TEMPLATE.format(
        code=base.replace_source_labels_with_tokens(code),
        obj_token=base.GOALS[obj_mode]["token"],
        **target_prompt_fields(
            row,
            device_token_dropout=device_token_dropout,
        ),
    )


def clock_target_text(row: Mapping[str, Any]) -> str:
    selected = row.get("selected_clock_period", _clock_of(row))
    return (
        f"{CLOCK_ANCHOR_TOKEN}\n"
        f"selected_clock_period_ns = {_norm_clock(selected):g}\n"
    )


# Install prompt globals before the base main() creates tokenizer special tokens.
base.PROMPT_TEMPLATE = PROMPT_TEMPLATE
base.AUTO_PERIOD_TOKEN = AUTO_PERIOD_TOKEN
base.CLOCK_ANCHOR_TOKEN = CLOCK_ANCHOR_TOKEN
base.period_token_from_clock = period_token_from_clock
base.target_prompt_fields = target_prompt_fields
base.build_prompt = build_prompt
base.TARGET_PLATFORM_TOKENS = (
    sorted(set(base.DEVICE_TOKEN_MAP.values()))
    + [base.UNKNOWN_DEVICE_TOKEN]
    + list(base.PERIOD_TOKEN_MAP.values())
    + [AUTO_PERIOD_TOKEN, CLOCK_ANCHOR_TOKEN]
)


# ---------------------------------------------------------------------------
# Shared random resource-budget generation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class ResourceBudget:
    bram_frac: float
    dsp_frac: float
    ff_frac: float
    lut_frac: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "BRAM_18K": self.bram_frac,
            "DSP": self.dsp_frac,
            "FF": self.ff_frac,
            "LUT": self.lut_frac,
        }


def _stable_seed(parts: Sequence[Any], seed: int) -> int:
    payload = repr((tuple(parts), int(seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _row_used_fraction(row: Mapping[str, Any], resource: str) -> float:
    field = base.UTIL_FIELD_BY_RESOURCE[resource]
    value = float(row.get(field, 0.0) or 0.0) / 100.0
    return max(0.0, value)


def _quantized_budget(values: Iterable[float]) -> ResourceBudget:
    clipped = [
        round(min(1.0, max(CFG.min_budget_frac, float(value))), 2)
        for value in values
    ]
    return ResourceBudget(*clipped)


def sample_shared_budgets(
    case_key: Tuple[str, str],
    candidates: Sequence[dict],
) -> List[ResourceBudget]:
    """
    Generate budgets shared by every design and clock in one kernel/device pool.

    Mixture:
      15% full-device,
      20% correlated scalar,
      50% independent resource fractions,
      15% close to measured feasibility boundaries.
    """
    rng = random.Random(_stable_seed(case_key, CFG.seed))
    budgets = {ResourceBudget(1.0, 1.0, 1.0, 1.0)}

    while len(budgets) < max(1, CFG.random_budgets_per_case):
        p = rng.random()

        if p < 0.15:
            values = [1.0] * 4

        elif p < 0.35:
            scalar = rng.uniform(CFG.min_budget_frac, 1.0)
            values = [scalar] * 4

        elif p < 0.85:
            values = [
                CFG.min_budget_frac
                + (1.0 - CFG.min_budget_frac)
                * rng.betavariate(2.0, 1.5)
                for _ in base.RESOURCE_KEYS
            ]

        else:
            # Sample just above a real design's measured usage.  These examples
            # teach the decision boundaries where one configuration becomes
            # feasible and another does not.
            anchor = candidates[rng.randrange(len(candidates))]
            values = [
                _row_used_fraction(anchor, resource)
                + rng.uniform(0.01, 0.15)
                for resource in base.RESOURCE_KEYS
            ]

        budgets.add(_quantized_budget(values))

    return sorted(budgets)


def design_fits_budget(
    row: Mapping[str, Any],
    budget: ResourceBudget,
) -> bool:
    fractions = budget.as_dict()
    return all(
        _row_used_fraction(row, resource)
        <= fractions[resource] + 1e-9
        for resource in base.RESOURCE_KEYS
    )


def attach_budget(row: Mapping[str, Any], budget: ResourceBudget) -> dict:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = base.DEVICE_RESOURCES.get(device)
    if caps is None:
        raise ValueError(f"Unsupported device: {device!r}")

    out = dict(row)
    fractions = budget.as_dict()

    for resource in base.RESOURCE_KEYS:
        field = base.AVAIL_FIELD_BY_RESOURCE[resource]
        out[field] = int(round(float(caps[resource]) * fractions[resource]))
        out[f"budget_frac_{resource.lower()}"] = float(fractions[resource])

    out["resource_budget_id"] = (
        f"B{budget.bram_frac:.2f}_D{budget.dsp_frac:.2f}_"
        f"F{budget.ff_frac:.2f}_L{budget.lut_frac:.2f}"
    )
    out["resource_pressure"] = max(
        _row_used_fraction(out, resource)
        / max(fractions[resource], 1e-9)
        for resource in base.RESOURCE_KEYS
    )
    return out


def _compact_candidate_union(
    feasible: Sequence[dict],
) -> List[dict]:
    """
    Keep an objective-independent union of strong candidates in O(N log N).

    The original Pareto implementation is quadratic. Running it over every raw
    candidate for every sampled budget would be unnecessarily expensive for a
    ~1 GB dataset. We therefore retain:
      * the lowest-latency candidates;
      * the lowest-area candidates;
      * the lowest-ADP candidates;
      * the exact two-objective Pareto frontier, capped deterministically.

    The original MailoHLS goal-ranking function is then applied to this compact
    candidate set, so final labels still follow the existing objective logic.
    """
    if not feasible:
        return []

    k = max(1, int(CFG.candidate_pool_per_objective))
    valid = [
        row for row in feasible
        if float(row.get("latency", 0.0)) > 0.0
        and float(row.get("area", 0.0)) > 0.0
    ]
    if not valid:
        return []

    def row_identity(row: Mapping[str, Any]) -> Tuple[int, str]:
        return (
            int(row.get("_jsonl_idx", -1)),
            base.canonical_completion_key(row["input"], row["target"]),
        )

    keep: Dict[Tuple[int, str], dict] = {}

    ordered_latency = sorted(
        valid,
        key=lambda row: (
            float(row["latency"]),
            float(row["area"]),
            row_identity(row),
        ),
    )
    ordered_area = sorted(
        valid,
        key=lambda row: (
            float(row["area"]),
            float(row["latency"]),
            row_identity(row),
        ),
    )
    ordered_adp = sorted(
        valid,
        key=lambda row: (
            math.log2(max(float(row["latency"]), 1e-12))
            + math.log2(max(float(row["area"]), 1e-12)),
            float(row["latency"]),
            float(row["area"]),
            row_identity(row),
        ),
    )

    for row in ordered_latency[:k] + ordered_area[:k] + ordered_adp[:k]:
        keep[row_identity(row)] = row

    # Exact 2-D frontier for minimization: after sorting by latency, a point is
    # non-dominated iff its area is lower than every earlier point's area.
    frontier = []
    best_area = float("inf")
    for row in ordered_latency:
        area = float(row["area"])
        if area < best_area - 1e-12:
            frontier.append(row)
            best_area = area

    # Keep the whole frontier when modest; otherwise sample it uniformly so
    # both latency and area extremes remain represented.
    frontier_cap = 3 * k
    if len(frontier) > frontier_cap:
        indices = np.linspace(
            0, len(frontier) - 1, frontier_cap, dtype=int
        ).tolist()
        frontier = [frontier[index] for index in indices]

    for row in frontier:
        keep[row_identity(row)] = row

    return list(keep.values())


def augment_rows_with_random_resource_budgets(
    rows: List[dict],
    _legacy_fractions: Optional[Sequence[float]] = None,
) -> List[dict]:
    """
    Drop-in replacement for the base fixed-zone augmentation function.

    Budgets are shared across all clocks for a kernel/device pair, which is
    required for valid automatic-frequency competition.
    """
    if CFG.budget_mode == "none":
        return [dict(row) for row in rows]

    if CFG.budget_mode == "fixed":
        # Preserve the original implementation as a controlled ablation.
        fractions = (
            list(_legacy_fractions)
            if _legacy_fractions
            else [0.10, 0.25, 0.50, 0.75, 1.00]
        )
        return ORIGINAL_FIXED_AUGMENT(rows, fractions)

    by_case: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["kernel_name"],
            _norm_device(row.get("device", row.get("Device", ""))),
        )
        by_case[key].append(row)

    augmented: List[dict] = []
    stats = Counter()

    for case_key, candidates in sorted(by_case.items()):
        budgets = sample_shared_budgets(case_key, candidates)

        for budget in budgets:
            feasible = [
                row for row in candidates
                if design_fits_budget(row, budget)
            ]
            if len(feasible) < CFG.min_feasible_candidates:
                stats["rejected_small_candidate_sets"] += 1
                continue

            compact = _compact_candidate_union(feasible)
            if len(compact) < CFG.min_feasible_candidates:
                stats["rejected_after_compaction"] += 1
                continue

            augmented.extend(attach_budget(row, budget) for row in compact)
            stats["kept_budgets"] += 1
            stats["candidate_rows"] += len(compact)

    print(
        "[RANDOM-BUDGET] "
        f"input_rows={len(rows)} output_rows={len(augmented)} "
        f"budgets_per_case={CFG.random_budgets_per_case} "
        f"stats={dict(stats)}"
    )
    return augmented


ORIGINAL_FIXED_AUGMENT = base.augment_rows_with_resource_budgets
base.augment_rows_with_resource_budgets = (
    augment_rows_with_random_resource_budgets
)


# ---------------------------------------------------------------------------
# Specified-frequency and automatic-frequency objective selection.
# ---------------------------------------------------------------------------

def target_bucket_key(row: Mapping[str, Any]) -> tuple:
    """
    Seven-field key retained for compatibility with base validation utilities.

    The third field is either a numeric requested period or the string "AUTO".
    """
    frequency_mode = str(row.get("frequency_mode", "specified")).lower()
    period_or_mode: Any = (
        "AUTO" if frequency_mode == "auto" else _clock_of(row)
    )
    avail = _available_resources(row)
    return (
        row["kernel_name"],
        _norm_device(row.get("device", row.get("Device", ""))),
        period_or_mode,
        avail["BRAM_18K"],
        avail["DSP"],
        avail["FF"],
        avail["LUT"],
    )


base.target_bucket_key = target_bucket_key


def _rank_and_select_case(
    items: Sequence[dict],
    goal_mode: str,
    top_k: int,
    domination_penalty: float,
    max_dominated_gap: float,
    score_weight_min: float,
    score_weight_power: float,
    frequency_mode: str,
) -> Tuple[List[dict], dict]:
    ranked = base.rank_goal_candidates(
        list(items),
        goal_mode=goal_mode,
        domination_penalty=domination_penalty,
        max_dominated_gap=max_dominated_gap,
    )

    unique = []
    seen = set()
    for rec in ranked:
        row = rec["row"]
        completion = base.canonical_completion_key(
            row["input"], row["target"]
        )
        key = (
            _clock_of(row) if frequency_mode == "auto" else None,
            completion,
        )
        if key in seen:
            continue
        seen.add(key)
        rec["score"] = float(
            base.goal_sort_key(
                rec,
                goal_mode,
                domination_penalty=0.0,
            )[0]
        )
        unique.append(rec)

    chosen = unique[: min(top_k, len(unique))]
    if not chosen:
        return [], {}

    scores = [float(rec["score"]) for rec in chosen]
    best_score, worst_score = min(scores), max(scores)

    selected = []
    for rank, rec in enumerate(chosen):
        out = dict(rec["row"])
        out["obj_mode"] = goal_mode
        out["frequency_mode"] = frequency_mode
        out["selected_clock_period"] = _clock_of(out)
        out["_score"] = float(rec["score"])
        out["_rank_within_kernel"] = int(rank)
        out["_sample_weight"] = float(
            base.score_gap_weight(
                score=float(rec["score"]),
                best_score=best_score,
                worst_score=worst_score,
                w_min=score_weight_min,
                power=score_weight_power,
            )
        )
        selected.append(out)

    # Build directive hard negatives from the jointly ranked alternatives.
    hard_negatives = base.build_local_hard_negative_bank(
        [{"row": rec["row"], "score": rec["score"]} for rec in unique],
        hard_neg_top_k=max(6, top_k),
    )
    hard_negatives = {
        lhs: sorted(values, key=base._rhs_sort_key)
        for lhs, values in hard_negatives.items()
    }
    for out in selected:
        out["_local_hard_negatives"] = hard_negatives

    return selected, {
        "selected": len(selected),
        "candidate_count": len(items),
        "frequency_mode": frequency_mode,
        "selected_clocks": [
            float(row["selected_clock_period"]) for row in selected
        ],
    }


def select_goal_rows_targetaware(
    rows: List[dict],
    goal_mode: str,
    top_k: int,
    domination_penalty: float,
    max_dominated_gap: float,
    score_weight_min: float = 0.6,
    score_weight_power: float = 1.0,
):
    """
    Create both tasks:
      specified: best directives at a requested period;
      auto: best (period, directives) across all available periods.
    """
    specified_buckets: Dict[tuple, List[dict]] = defaultdict(list)
    auto_buckets: Dict[tuple, List[dict]] = defaultdict(list)

    for row in rows:
        specified_key = (
            row["kernel_name"],
            _norm_device(row.get("device", row.get("Device", ""))),
            _clock_of(row),
            *_available_resources(row).values(),
        )
        auto_key = (
            row["kernel_name"],
            _norm_device(row.get("device", row.get("Device", ""))),
            *_available_resources(row).values(),
        )
        specified_buckets[specified_key].append(row)
        auto_buckets[auto_key].append(row)

    selected: List[dict] = []
    metadata: Dict[str, dict] = {}

    for key, items in sorted(specified_buckets.items()):
        chosen, info = _rank_and_select_case(
            items,
            goal_mode,
            top_k,
            domination_penalty,
            max_dominated_gap,
            score_weight_min,
            score_weight_power,
            frequency_mode="specified",
        )
        selected.extend(chosen)
        metadata[f"specified::{key!r}"] = info

    auto_selected: List[dict] = []
    for key, items in sorted(auto_buckets.items()):
        clocks = sorted({_clock_of(row) for row in items})
        if len(clocks) < CFG.min_auto_clock_count:
            continue

        chosen, info = _rank_and_select_case(
            items,
            goal_mode,
            top_k,
            domination_penalty,
            max_dominated_gap,
            score_weight_min,
            score_weight_power,
            frequency_mode="auto",
        )
        auto_selected.extend(chosen)
        info["available_clocks"] = clocks
        metadata[f"auto::{key!r}"] = info

    # Keep the requested training mixture without stochastic per-epoch labels.
    if CFG.auto_frequency_fraction > 0.0 and auto_selected:
        target_auto_count = int(
            round(
                len(selected)
                * CFG.auto_frequency_fraction
                / max(1e-9, 1.0 - CFG.auto_frequency_fraction)
            )
        )
        rng = random.Random(_stable_seed(("auto_mix", goal_mode), CFG.seed))
        rng.shuffle(auto_selected)
        auto_selected = auto_selected[:target_auto_count]
        selected.extend(auto_selected)

    rng = random.Random(_stable_seed(("selected_shuffle", goal_mode), CFG.seed))
    rng.shuffle(selected)

    mode_counts = Counter(
        row.get("frequency_mode", "specified") for row in selected
    )
    print(
        f"[CLOCK-MODE] objective={goal_mode} selected={len(selected)} "
        f"counts={dict(mode_counts)}"
    )
    return selected, metadata


base.select_goal_rows = select_goal_rows_targetaware


# ---------------------------------------------------------------------------
# Dataset: prepend supervised clock tokens to the unchanged directive pack.
# ---------------------------------------------------------------------------

class TargetAwareSFTDataset(Dataset):
    """
    Equivalent to the base SFTDataset, with a supervised clock prefix.

    The deterministic directive construction remains exactly the same.
    """

    def __init__(
        self,
        rows: List[dict],
        tok,
        max_length: int,
        value_loss_weight: float = 1.0,
        candidate_sites_per_sample: int = 0,
        candidate_negatives_per_site: int = 0,
        device_token_dropout: float = 0.0,
    ):
        self.samples = []
        self.lengths = []
        self.tok = tok
        self.max_length = max_length

        kind_loss_weights = {
            "UNROLL": 1.6,
            "ARRAY_F": 1.2,
            "PIPE": 1.2,
            "ARRAY_T": 1.0,
            "ARRAY_D": 0.8,
        }

        missing_objective = 0

        for ex in rows:
            prompt = build_prompt(
                ex["input"],
                ex["obj_mode"],
                row=ex,
                device_token_dropout=device_token_dropout,
            )
            target_core = base.reorder_target_by_source_order(
                ex["input"], ex["target"].strip()
            )

            prompt_ids = tok(
                prompt, add_special_tokens=False
            )["input_ids"]

            directive_pack = base.build_deterministic_rhs_pack(
                ex["input"],
                target_core,
                tok,
                value_w=value_loss_weight,
                kind_loss_weights=kind_loss_weights,
            )

            clock_ids = tok(
                clock_target_text(ex),
                add_special_tokens=False,
            )["input_ids"]

            # Every clock token is supervised.  The value receives the same
            # base weight as an ordinary directive RHS value.
            target_ids = clock_ids + directive_pack.input_ids
            target_labels = clock_ids + directive_pack.labels
            token_weights_target = (
                [float(value_loss_weight)] * len(clock_ids)
                + directive_pack.token_weights
            )
            xattn_target_mask = (
                [0] * len(clock_ids)
                + directive_pack.xattn_target_mask
            )

            if len(target_ids) >= max_length:
                target_ids = target_ids[:max_length]
                target_labels = target_labels[:max_length]
                token_weights_target = token_weights_target[:max_length]
                xattn_target_mask = xattn_target_mask[:max_length]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_length - len(target_ids)):]

            obj_id = tok.encode(
                base.GOALS[ex["obj_mode"]]["token"],
                add_special_tokens=False,
            )[0]
            if obj_id not in prompt_ids:
                missing_objective += 1

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_labels
            token_weights = (
                [0.0] * len(prompt_ids) + token_weights_target
            )
            full_xattn_mask = (
                [0] * len(prompt_ids) + xattn_target_mask
            )
            xattn_apply_mask = full_xattn_mask[1:] + [0]

            contrastive_sites = base.build_contrastive_sites_from_sample(
                source_text=ex["input"],
                target_text=target_core,
                prompt_ids=prompt_ids + clock_ids,
                tok=tok,
                max_length=max_length,
                local_hard_negatives=ex.get(
                    "_local_hard_negatives", {}
                ),
                candidate_sites_per_sample=candidate_sites_per_sample,
                candidate_negatives_per_site=candidate_negatives_per_site,
                kind_priority=kind_loss_weights,
            )

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(
                    len(input_ids), dtype=torch.long
                ),
                "labels": torch.tensor(labels, dtype=torch.long),
                "token_weights": torch.tensor(
                    token_weights, dtype=torch.float32
                ),
                "xattn_apply_mask": torch.tensor(
                    xattn_apply_mask, dtype=torch.float32
                ),
                "sample_weight": torch.tensor(
                    float(ex.get("_sample_weight", 1.0)),
                    dtype=torch.float32,
                ),
                "kernel_name": ex["kernel_name"],
                "routing_start_idx": torch.tensor(
                    len(prompt_ids), dtype=torch.long
                ),
                "contrastive_sites": contrastive_sites,
            })
            self.lengths.append(len(input_ids))

        print(
            f"[DATASET] samples={len(self.samples)} "
            f"missing_objective_after_truncation={missing_objective}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


base.SFTDataset = TargetAwareSFTDataset


# ---------------------------------------------------------------------------
# Keep checkpoint selection on specified-clock cases until the constrained
# decoder is extended to score the clock field jointly.
# ---------------------------------------------------------------------------

ORIGINAL_BUILD_SELECTION_CASES = base.build_selection_cases


def build_selection_cases_specified_only(
    val_rows: List[dict],
    goal_mode: str,
    max_kernels: int = 4,
    min_coverage: float = 0.85,
    min_supervised_sites: int = 4,
):
    specified = [
        row for row in val_rows
        if row.get("frequency_mode", "specified") == "specified"
    ]
    return ORIGINAL_BUILD_SELECTION_CASES(
        specified,
        goal_mode=goal_mode,
        max_kernels=max_kernels,
        min_coverage=min_coverage,
        min_supervised_sites=min_supervised_sites,
    )


base.build_selection_cases = build_selection_cases_specified_only


# ---------------------------------------------------------------------------
# CLI extension.  Unknown arguments are left for the original parser.
# ---------------------------------------------------------------------------

def parse_extension_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--resource_budget_mode",
        choices=["none", "fixed", "random"],
        default="random",
    )
    parser.add_argument(
        "--random_budgets_per_case",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--random_budget_min_frac",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--min_feasible_candidates_per_budget",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--candidate_pool_per_objective",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--auto_frequency_fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--min_auto_clock_count",
        type=int,
        default=2,
    )
    extension, remaining = parser.parse_known_args(list(argv))
    return extension, remaining


def validate_extension_args(args) -> None:
    if args.random_budgets_per_case < 1:
        raise ValueError("--random_budgets_per_case must be >= 1")
    if not 0.0 < args.random_budget_min_frac <= 1.0:
        raise ValueError("--random_budget_min_frac must be in (0, 1]")
    if args.min_feasible_candidates_per_budget < 2:
        raise ValueError(
            "--min_feasible_candidates_per_budget must be >= 2"
        )
    if args.candidate_pool_per_objective < 1:
        raise ValueError("--candidate_pool_per_objective must be >= 1")
    if not 0.0 <= args.auto_frequency_fraction < 1.0:
        raise ValueError("--auto_frequency_fraction must be in [0, 1)")
    if args.min_auto_clock_count < 2:
        raise ValueError("--min_auto_clock_count must be >= 2")


def main() -> None:
    extension, remaining = parse_extension_args(sys.argv[1:])
    validate_extension_args(extension)

    # Read the base seed if supplied; otherwise use its default.
    seed_parser = argparse.ArgumentParser(add_help=False)
    seed_parser.add_argument("--seed", type=int, default=123)
    seed_args, _ = seed_parser.parse_known_args(remaining)

    CFG.budget_mode = extension.resource_budget_mode
    CFG.random_budgets_per_case = extension.random_budgets_per_case
    CFG.min_budget_frac = extension.random_budget_min_frac
    CFG.min_feasible_candidates = (
        extension.min_feasible_candidates_per_budget
    )
    CFG.candidate_pool_per_objective = (
        extension.candidate_pool_per_objective
    )
    CFG.auto_frequency_fraction = extension.auto_frequency_fraction
    CFG.min_auto_clock_count = extension.min_auto_clock_count
    CFG.seed = seed_args.seed

    print("[TARGET-AWARE-CONFIG]", CFG)

    # The original code enters resource augmentation only with this legacy flag.
    # Add it automatically for random/fixed modes.
    if (
        CFG.budget_mode != "none"
        and "--use_resource_budgets" not in remaining
    ):
        remaining.append("--use_resource_budgets")

    # The old list is ignored in random mode but remains valid in fixed mode.
    if "--resource_budget_fracs" not in remaining:
        remaining += [
            "--resource_budget_fracs",
            "10,25,50,75,100",
        ]

    sys.argv = [str(BASE_PATH)] + remaining
    base.main()


if __name__ == "__main__":
    main()

