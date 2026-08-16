#!/usr/bin/env python3
"""Train the target-aware MailoHLS directive generator.

Stage 1 learns deterministic directive values from source/action markers,
optimization objective, FPGA resources, and clock constraints.  Stage 2 starts
from the selected Stage-1 adapter and adds cross-attention over action-aligned
GNN structural memory.  The specified-clock task is the default; automatic
selection among measured clocks is an explicit, disabled-by-default ablation.

Some internal classes and checkpoint field names retain the historical "HARP"
term for compatibility.  In the current pipeline they mean MLIR-derived GNN
structural memory, not the legacy LLVM graph representation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import gc
import shutil
import subprocess
import numpy as np
from pathlib import Path

from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Iterable, Mapping, Sequence

import torch
import peft
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import hashlib

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from LLM_branch.common import mailohls_contract, structural_xattn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_DATASET = REPOSITORY_ROOT / "artifacts" / "llm" / "mailohls_sft.jsonl"
DEFAULT_STRUCTURAL_MEMORY = (
    REPOSITORY_ROOT / "artifacts" / "llm" / "mlir_structural_memory"
)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


DEVICE_RESOURCES = mailohls_contract.DEVICE_RESOURCES
RESOURCE_KEYS = mailohls_contract.RESOURCE_KEYS
UTIL_FIELD_BY_RESOURCE = mailohls_contract.UTIL_FIELD_BY_RESOURCE
AVAIL_FIELD_BY_RESOURCE = mailohls_contract.AVAIL_FIELD_BY_RESOURCE
DEVICE_TOKEN_MAP = mailohls_contract.DEVICE_TOKEN_MAP
UNKNOWN_DEVICE_TOKEN = mailohls_contract.UNKNOWN_DEVICE_TOKEN
PERIOD_TOKEN_MAP = mailohls_contract.PERIOD_TOKEN_MAP
CLOCK_ANCHOR_TOKEN = mailohls_contract.CLOCK_ANCHOR_TOKEN
TARGET_PLATFORM_TOKENS = mailohls_contract.TARGET_PLATFORM_TOKENS


def _norm_device(device: Any) -> str:
    return str(device or "").strip().lower()


def _norm_clock(clock_period: Any) -> float:
    return round(float(clock_period), 2)


def prepend_selected_clock(
    target_text: str,
    selected_clock_period: float,
) -> str:
    period = _norm_clock(selected_clock_period)
    return (
        f"{CLOCK_ANCHOR_TOKEN}\n"
        f"selected_clock_period_ns = {period:g}\n"
        f"{target_text.strip()}"
    )


def auto_frequency_bucket_key(row: dict):
    return (
        row["kernel_name"],
        _norm_device(row.get("device", "")),
        *_avail_resource_tuple(row),
    )


def select_auto_frequency_rows(
    rows: List[dict],
    goal_mode: str,
    top_k: int,
    domination_penalty: float,
    max_dominated_gap: float,
):
    by_case = defaultdict(list)

    for row in rows:
        by_case[auto_frequency_bucket_key(row)].append(row)

    selected = []

    for case_key, candidates in by_case.items():
        # Candidates include all available clock periods.
        ranked = rank_goal_candidates(
            candidates,
            goal_mode=goal_mode,
            domination_penalty=domination_penalty,
            max_dominated_gap=max_dominated_gap,
        )

        seen = set()
        unique = []

        for rec in ranked:
            row = rec["row"]
            completion = canonical_completion_key(
                row["input"], row["target"]
            )
            key = (
                _norm_clock(
                    row.get(
                        "clock_period",
                        row.get("Clock_Period_nsec"),
                    )
                ),
                completion,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(rec)

        for rank, rec in enumerate(unique[:top_k]):
            row = dict(rec["row"])
            row["frequency_mode"] = "auto"
            row["selected_clock_period"] = _norm_clock(
                row.get(
                    "clock_period",
                    row.get("Clock_Period_nsec"),
                )
            )
            row["_rank_within_kernel"] = rank
            selected.append(row)

    return selected


def _avail_resource_tuple(row: dict) -> Tuple[int, int, int, int]:
    """
    Resource budget visible to the model. If a row has synthetic available-resource
    fields, use them; otherwise fall back to full-device capacity.
    """
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device, {})

    def get_int(field: str, res_key: str) -> int:
        if field in row and row[field] not in (None, ""):
            return int(round(float(row[field])))
        return int(caps.get(res_key, 0))

    return (
        get_int("avail_bram", "BRAM_18K"),
        get_int("avail_dsp", "DSP"),
        get_int("avail_ff", "FF"),
        get_int("avail_lut", "LUT"),
    )



# ==============================
# Objective + placeholder tokens
# ==============================

GOALS = mailohls_contract.GOALS
GOAL_ORDER = tuple(GOALS)

# Target anchors used in the generated directives
TARGET_PLACEHOLDER_TOKENS = mailohls_contract.TARGET_PLACEHOLDER_TOKENS

# Source-only structural markers used inside the kernel code
SOURCE_PLACEHOLDER_TOKENS = mailohls_contract.SOURCE_PLACEHOLDER_TOKENS


def source_placeholder_token(label: str) -> str:
    return f"<SRC_{label.upper()}>"


def target_placeholder_token(label: str) -> str:
    return f"<{label.upper()}>"



# ===============================
# Regexes
# ===============================

# source labels like L1: or /* L1: */
SOURCE_LABEL_RE = re.compile(
    r'^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)',
    re.IGNORECASE
)

# auto{_PIPE_L1}=..., auto{_UNROLL_L1}=..., auto{_ARRAY_T_L2}=...
TARGET_LINE_LABEL_RE = re.compile(
    r'auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}\s*=',
    re.IGNORECASE
)

ANCHOR_OR_ASSIGN_RE = re.compile(
    r'^\s*(<L\d+>|auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}\s*=\s*.+)$',
    re.IGNORECASE | re.MULTILINE
)

ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)
TARGET_ANCHOR_RE = re.compile(r"^<(L\d+)>$", re.IGNORECASE)


def directive_schema_signature(text: str):
    """Return the ordered anchor/assignment signature, or None if malformed."""
    signature = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        anchor_match = TARGET_ANCHOR_RE.fullmatch(line)
        if anchor_match is not None:
            signature.append(("anchor", anchor_match.group(1).upper()))
            continue
        assignment_match = ASSIGN_RE.fullmatch(line)
        if assignment_match is not None:
            signature.append(("assignment", assignment_match.group(1).upper()))
            continue
        return None
    return signature

SOURCE_PLACEHOLDER_IN_CODE_RE = re.compile(
    r'auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}',
    re.IGNORECASE
)

LHS_KIND_RE = re.compile(
    r"^auto\{_([A-Z0-9]+(?:_[A-Z0-9]+)*)_L\d+\}$",
    re.IGNORECASE,
)

AUTO_PERIOD_TOKEN = mailohls_contract.AUTO_PERIOD_TOKEN



# ===========================================
# Formatting Helpers (Target Construction)
# ===========================================

def replace_source_labels_with_tokens(text: str) -> str:
    return mailohls_contract.replace_source_labels_with_tokens(text)


def extract_source_label_order(source_text: str) -> List[str]:
    order = []
    seen = set()

    for line in source_text.splitlines():
        stripped = line.lstrip()
        m = SOURCE_LABEL_RE.match(stripped)
        if not m:
            continue
        label = (m.group(1) or m.group(2)).upper()
        if label not in seen:
            seen.add(label)
            order.append(label)

    return order


def reorder_target_by_source_order(source_text: str, target_text: str) -> str:
    """
    Reorder raw target assignment lines so that label groups follow the order
    of labels in the source code.
    """
    label_order = extract_source_label_order(source_text)

    grouped = defaultdict(list)
    extras = []

    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = TARGET_LINE_LABEL_RE.search(line)
        if m is None:
            extras.append(line)
            continue

        label = m.group(1).upper()
        grouped[label].append(line)

    out = []
    emitted = set()

    for label in label_order:
        if label in grouped:
            out.extend(grouped[label])
            emitted.add(label)

    # keep any leftover labels deterministically at the end
    for label in sorted(grouped.keys()):
        if label not in emitted:
            out.extend(grouped[label])

    out.extend(extras)
    return "\n".join(out)


def print_target_coverage(rows):
    counts = Counter(
        (
            row["kernel_name"],
            _norm_device(row.get("device", "")),
            _norm_clock(
                row.get(
                    "clock_period",
                    row.get("Clock_Period_nsec"),
                )
            ),
        )
        for row in rows
    )

    clocks_by_kernel_device = defaultdict(set)

    for kernel, device, clock in counts:
        clocks_by_kernel_device[(kernel, device)].add(clock)

    multi_clock = sum(
        len(clocks) >= 2
        for clocks in clocks_by_kernel_device.values()
    )

    print(
        f"[CLOCK-COVERAGE] multi-clock cases="
        f"{multi_clock}/{len(clocks_by_kernel_device)}"
    )

    print("[CLOCK-COVERAGE]", Counter(
        len(clocks)
        for clocks in clocks_by_kernel_device.values()
    ))


def extract_ordered_lhs_plan(source_text: str) -> List[Tuple[str, str]]:
    """
    Returns a deterministic ordered plan of directive sites from the source code:
        [("L1", "auto{_ARRAY_T_L1}"), ("L1", "auto{_ARRAY_F_L1}"), ...]
    """
    by_label = defaultdict(list)

    for line in source_text.splitlines():
        for m in SOURCE_PLACEHOLDER_IN_CODE_RE.finditer(line):
            lhs = m.group(0)              # e.g. auto{_PIPE_L3}
            label = m.group(1).upper()    # e.g. L3
            if lhs not in by_label[label]:
                by_label[label].append(lhs)

    plan = []
    for label in extract_source_label_order(source_text):
        for lhs in by_label.get(label, []):
            plan.append((label, lhs))

    return plan


def build_rhs_map_from_target(target_text: str) -> Dict[str, str]:
    """
    Maps RHS values
    Parses: auto{_PIPE_L3} = 1  into: {"auto{_PIPE_L3}": "1"}
    """
    rhs_map = {}

    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = ASSIGN_RE.match(line)
        if m is None:
            continue

        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        rhs_map[lhs] = rhs

    return rhs_map



@dataclass
class DeterministicRHSPack:
    input_ids: List[int]          # fixed target-side tokens + RHS tokens interleaved
    labels: List[int]             # -100 for fixed tokens, token id for RHS tokens
    token_weights: List[float]    # 0 for fixed tokens, value weight for RHS tokens
    xattn_target_mask: List[int]  # 1 only on RHS tokens



def build_deterministic_rhs_pack(
    source_text: str,
    target_text: str,
    tok,
    value_w: float = 1.0,
    kind_loss_weights: Optional[Dict[str, float]] = None,
    supervise_eos: bool = False,
) -> DeterministicRHSPack:
    """
    Build the deterministic target sequence for RHS-only training.
    Fixed schema tokens (<Lk> anchors and "lhs =") are kept in the input as context but are not supervised
    Only RHS value tokens receive labels, loss weights and xattn routing marks. 
    kind_loss_weights --> bias learning toward more important directive kinds 
    """
    rhs_map = build_rhs_map_from_target(target_text)
    full_plan = extract_ordered_lhs_plan(source_text)
    plan = [(label, lhs) for (label, lhs) in full_plan if lhs in rhs_map]

    input_ids, labels, token_weights, xattn_target_mask = [], [], [], []

    def add_fixed(text: str):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend([-100] * len(ids))
        token_weights.extend([0.0] * len(ids))
        xattn_target_mask.extend([0] * len(ids))

    def add_rhs(text: str, weight: float):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids)
        token_weights.extend([weight] * len(ids))
        xattn_target_mask.extend([1] * len(ids))

    current_label = None
    kind_loss_weights = kind_loss_weights or {}

    for label, lhs in plan:
        if label != current_label:
            add_fixed(f"{target_placeholder_token(label)}\n")
            current_label = label

        rhs = rhs_map[lhs].strip()
        kind = lhs_kind(lhs)

        weight = value_w
        weight *= kind_loss_weights.get(kind, 1.0)

        add_fixed(f"{lhs} = ")
        add_rhs(rhs + "\n", weight) # add per-token supervision weighting by directive kind 
                                    # (most difficult directive kind --> larger weight)

    if supervise_eos:
        eos_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids.extend(eos_ids)
        labels.extend(eos_ids)
        token_weights.extend([value_w] * len(eos_ids))
        xattn_target_mask.extend([0] * len(eos_ids))

    return DeterministicRHSPack(
        input_ids=input_ids,
        labels=labels,
        token_weights=token_weights,
        xattn_target_mask=xattn_target_mask,
    )


def build_clock_pack(
    row: Mapping[str, Any],
    tok,
    value_w: float = 1.0,
) -> DeterministicRHSPack:
    """Keep the clock schema fixed and supervise only its numeric RHS."""
    selected = row.get("selected_clock_period")
    if selected in (None, ""):
        selected = _clock_of(row)
    fixed_ids = tok(
        f"{CLOCK_ANCHOR_TOKEN}\nselected_clock_period_ns = ",
        add_special_tokens=False,
    )["input_ids"]
    value_ids = tok(
        f"{_norm_clock(selected):g}\n",
        add_special_tokens=False,
    )["input_ids"]
    return DeterministicRHSPack(
        input_ids=fixed_ids + value_ids,
        labels=[-100] * len(fixed_ids) + value_ids,
        token_weights=[0.0] * len(fixed_ids) + [value_w] * len(value_ids),
        xattn_target_mask=[0] * (len(fixed_ids) + len(value_ids)),
    )



# =============================
# Dataset Loading
# =============================

def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())


def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


def family_id_from_kernel_name(name: str) -> str:
    s = normalize_kname(name)

    if s.startswith("machsuite_gemm"):
        return "machsuite_gemm"

    if s.startswith("machsuite_"):
        parts = s.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return s

    if s.startswith("spcl_example"):
        return "spcl_example"

    if s.startswith("serrano_"):
        return "serrano_kalman_filter"

    if s.startswith("rodinia_"):
        rest = s[len("rodinia_"):]
        for special in ["cfd_flux", "cfd_step_factor", "lc_gicov", "lc_mgvf"]:
            if rest.startswith(special):
                return f"rodinia_{special}"
        algo = rest.split("_")[0]
        return f"rodinia_{algo}"

    return s


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(jsonl_path: str) -> List[dict]:
    """Load SFT rows and resolve compact source-template references.

    Rows from the current builder store ``source_key`` instead of repeating the
    full kernel text.  Historical JSONL files containing ``input`` remain
    readable for comparison experiments.
    """
    dataset_path = Path(jsonl_path)
    sources_path = dataset_path.with_suffix(".sources.json")
    manifest_path = dataset_path.with_suffix(".manifest.json")
    source_templates: Dict[str, str] = {}
    if sources_path.is_file():
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing SFT manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != "mailohls-sft-jsonl-manifest-v2-compact-source":
            raise ValueError(f"Unsupported SFT manifest schema: {manifest_path}")
        if manifest.get("output_sha256") != _file_sha256(dataset_path):
            raise ValueError(f"SFT JSONL hash mismatch: {dataset_path}")
        if manifest.get("sources_sha256") != _file_sha256(sources_path):
            raise ValueError(f"Source-template hash mismatch: {sources_path}")
        with sources_path.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        if source_payload.get("schema") != "mailohls-sft-sources-v1":
            raise ValueError(f"Unsupported source-template schema: {sources_path}")
        raw_templates = source_payload.get("templates")
        if not isinstance(raw_templates, dict) or not raw_templates:
            raise ValueError(f"No source templates in {sources_path}")
        source_templates = {
            str(key): str(value) for key, value in raw_templates.items()
            if str(key) and str(value)
        }

    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            if not str(ex.get("input", "")).strip():
                source_key = str(ex.get("source_key", ""))
                source_text = source_templates.get(source_key)
                if source_text is None:
                    raise ValueError(
                        f"JSONL row {idx} has unresolved source_key={source_key!r}"
                    )
                ex["input"] = source_text
            ex["_jsonl_idx"] = idx
            ex["_family"] = family_id_from_kernel_name(ex["kernel_name"])
            rows.append(ex)
    if source_templates and int(manifest.get("examples", -1)) != len(rows):
        raise ValueError(f"SFT row count disagrees with {manifest_path}")
    return rows


def row_used_resources_abs(row: dict) -> Dict[str, float]:
    """
    Convert measured utilization percentages into absolute resource usage
    for the row's own measured device.
    """
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device)
    if caps is None:
        return {res: 0.0 for res in RESOURCE_KEYS}

    used = {}
    for res in RESOURCE_KEYS:
        util_field = UTIL_FIELD_BY_RESOURCE[res]
        if util_field is None or util_field not in row:
            used[res] = 0.0
        else:
            used[res] = float(row[util_field]) / 100.0 * float(caps[res])
    return used



def parse_resource_budget_fracs(spec: str) -> List[float]:
    """
    Parses comma-separated resource-budget fractions.

    Examples:
        "10,25,50,75,100" -> [0.10, 0.25, 0.50, 0.75, 1.00]
        "0.1,0.25,0.5,1.0" -> [0.10, 0.25, 0.50, 1.00]
    """
    vals = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v > 1.0:
            v /= 100.0
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Invalid resource budget fraction: {part}")
        vals.append(round(v, 4))
    return sorted(set(vals))


def make_resource_conditioned_row(row: dict, budget_frac: float) -> Optional[dict]:
    """
    Create one training context where the Available resources are a fraction
    of the measured device capacity. Keep the row only if the measured design fits
    inside that resource budget.

    This creates supervision of the form:
        same kernel/device/clock/objective + smaller available resources
        -> best feasible directive configuration inside that budget.
    """
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device)
    if caps is None:
        return None

    used = row_used_resources_abs(row)
    budget = {res: float(caps[res]) * float(budget_frac) for res in RESOURCE_KEYS}

    for res in RESOURCE_KEYS:
        if used[res] > budget[res] + 1e-9:
            return None

    r2 = dict(row)
    r2["resource_budget_frac"] = float(budget_frac)

    for res in RESOURCE_KEYS:
        field = AVAIL_FIELD_BY_RESOURCE[res]
        r2[field] = int(round(budget[res]))
        r2[f"used_{res.lower()}"] = float(used[res])

    r2["resource_pressure"] = float(max(
        used[res] / max(budget[res], 1e-9)
        for res in RESOURCE_KEYS
    ))
    return r2


def augment_rows_with_resource_budgets(rows: List[dict], budget_fracs: List[float]) -> List[dict]:
    """
    Expands rows into multiple resource-budget contexts 
    (the "Available resources" field of the prompt changes).
    """
    out = []
    kept = Counter()

    for row in rows:
        for frac in budget_fracs:
            r2 = make_resource_conditioned_row(row, frac)
            if r2 is None:
                continue
            out.append(r2)
            kept[f"{int(round(frac * 100))}pct"] += 1

    print(f"[RES-BUDGET] fractions={budget_fracs}")
    print(f"[RES-BUDGET] rows before={len(rows)} after={len(out)} kept={dict(sorted(kept.items()))}")
    return out


def design_fits_budget(
    row: dict,
    budget: ResourceBudget,
    tolerance: float = 1e-9,
) -> bool:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device)

    if caps is None:
        return False

    used = row_used_resources_abs(row)
    fractions = budget.as_dict()

    for resource in RESOURCE_KEYS:
        available = float(caps[resource]) * fractions[resource]
        if used[resource] > available + tolerance:
            return False

    return True


def attach_budget(row: dict, budget: ResourceBudget) -> dict:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES[device]
    fractions = budget.as_dict()

    out = dict(row)

    for resource in RESOURCE_KEYS:
        available = float(caps[resource]) * fractions[resource]
        out[AVAIL_FIELD_BY_RESOURCE[resource]] = int(round(available))

    out["resource_budget_frac_bram"] = budget.bram_frac
    out["resource_budget_frac_dsp"] = budget.dsp_frac
    out["resource_budget_frac_ff"] = budget.ff_frac
    out["resource_budget_frac_lut"] = budget.lut_frac

    used = row_used_resources_abs(row)
    out["resource_pressure"] = max(
        used[resource]
        / max(
            float(caps[resource]) * fractions[resource],
            1e-9,
        )
        for resource in RESOURCE_KEYS
    )

    return out



@dataclass(frozen=True)
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


def base_target_key(row: dict) -> Tuple[str, str, float]:
    """A candidate pool before applying a synthetic resource budget."""
    return (
        row["kernel_name"],
        _norm_device(row.get("device", row.get("Device", ""))),
        _norm_clock(
            row.get("clock_period", row.get("Clock_Period_nsec"))
        ),
    )


def stable_case_seed(case_key: tuple, global_seed: int) -> int:
    """
    Python's built-in hash is process-dependent unless PYTHONHASHSEED is fixed.
    Use a stable digest so budget generation is reproducible.
    """
    text = repr((case_key, int(global_seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def sample_resource_budgets(
    case_key: tuple,
    num_budgets: int,
    seed: int,
    min_frac: float = 0.10,
    full_budget_probability: float = 0.15,
    correlated_probability: float = 0.20,
) -> List[ResourceBudget]:
    rng = random.Random(stable_case_seed(case_key, seed))

    budgets = {
        ResourceBudget(1.0, 1.0, 1.0, 1.0)
    }

    while len(budgets) < num_budgets:
        p = rng.random()

        if p < full_budget_probability:
            values = [1.0] * 4

        elif p < full_budget_probability + correlated_probability:
            # Retain some scalar-resource-envelope examples.
            frac = rng.uniform(min_frac, 1.0)
            values = [frac] * 4

        else:
            # Independent resource pressures.
            values = [
                min_frac + (1.0 - min_frac) * rng.betavariate(2.0, 1.5)
                for _ in range(4)
            ]

        # Quantization controls the number of unique buckets and makes
        # prompts easier for the LLM to interpolate between.
        values = [round(v, 2) for v in values]

        budgets.add(ResourceBudget(*values))

    return sorted(
        budgets,
        key=lambda b: (
            b.bram_frac,
            b.dsp_frac,
            b.ff_frac,
            b.lut_frac,
        ),
    )


def split_by_family(rows: List[dict], val_fams: set, test_fams: set):
    train, val, test = [], [], []
    for r in rows:
        fam = r["_family"]
        if fam in test_fams:
            test.append(r)
        elif fam in val_fams:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


def assert_disjoint_nonempty_kernel_splits(train, val, test) -> None:
    kernel_sets = {
        "train": {row["kernel_name"] for row in train},
        "validation": {row["kernel_name"] for row in val},
        "test": {row["kernel_name"] for row in test},
    }
    empty = [name for name, kernels in kernel_sets.items() if not kernels]
    if empty:
        raise AssertionError(
            "Family split produced empty kernel set(s): " + ", ".join(empty)
        )
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = sorted(kernel_sets[left] & kernel_sets[right])
        if overlap:
            raise AssertionError(
                f"Family split kernel overlap between {left} and {right}: {overlap}"
            )


def split_rows_random_design(
    rows: List[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_kernel: bool = True,
):
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("Require 0 <= val_ratio, test_ratio < 1 and val_ratio + test_ratio < 1")

    rng = random.Random(seed)

    def split_bucket(bucket: List[dict]):
        bucket = list(bucket)
        rng.shuffle(bucket)
        n = len(bucket)

        if n <= 2:
            return bucket, [], []

        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))

        if val_ratio > 0 and n_val == 0 and n >= 3:
            n_val = 1
        if test_ratio > 0 and n_test == 0 and n >= 4:
            n_test = 1

        while n_val + n_test >= n:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break

        val = bucket[:n_val]
        test = bucket[n_val:n_val + n_test]
        train = bucket[n_val + n_test:]
        return train, val, test

    if not stratify_by_kernel:
        return split_bucket(rows)

    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel_name"]].append(r)

    train, val, test = [], [], []
    for k in sorted(by_kernel.keys()):
        tr, va, te = split_bucket(by_kernel[k])
        train.extend(tr)
        val.extend(va)
        test.extend(te)

    return train, val, test


def save_split_spec(path: str, train_rows: List[dict], val_rows: List[dict], test_rows: List[dict]):
    dump_json(path, {
        "train_jsonl_idx": [int(r["_jsonl_idx"]) for r in train_rows],
        "val_jsonl_idx": [int(r["_jsonl_idx"]) for r in val_rows],
        "test_jsonl_idx": [int(r["_jsonl_idx"]) for r in test_rows],
    })


def load_split_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_split_spec(rows: List[dict], spec: dict):
    idx_to_row = {int(r["_jsonl_idx"]): r for r in rows}

    train_rows = [idx_to_row[i] for i in spec["train_jsonl_idx"] if i in idx_to_row]
    val_rows   = [idx_to_row[i] for i in spec["val_jsonl_idx"] if i in idx_to_row]
    test_rows  = [idx_to_row[i] for i in spec["test_jsonl_idx"] if i in idx_to_row]
    return train_rows, val_rows, test_rows


def dump_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_mailohls_adapter(model, tokenizer, output_dir, contract):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(output_dir)
    dump_json(os.path.join(output_dir, "training_contract.json"), contract)


STAGE1_COMPATIBILITY_FIELDS = (
    "model",
    "model_revision",
    "tokenizer",
    "dataset_sha256",
    "split_sha256",
    "prompt_schema_version",
    "objective",
    "tokenizer_size",
    "special_tokens",
    "special_token_ids",
    "embedding_weights_tied",
    "trainable_token_modules",
    "special_token_role",
    "supervise_eos",
    "directive_domain_registry_sha256",
    "directive_loss_weighting",
    "directive_loss_weights",
    "max_length",
    "top_k",
    "device_mode",
    "device_token_dropout",
    "resource_budget_mode",
    "resource_budget_fracs",
    "random_budgets_per_case",
    "random_budget_min_frac",
    "min_feasible_candidates_per_budget",
    "candidate_pool_per_objective",
    "auto_frequency_fraction",
    "min_auto_clock_count",
    "goal_domination_penalty",
    "goal_max_dominated_gap",
    "min_supervised_sites",
    "min_site_coverage",
    "score_weight_min",
    "score_weight_power",
)


def require_compatible_stage1_contract(adapter_dir, expected_contract):
    contract_path = Path(adapter_dir) / "training_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 adapter is missing training contract: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "mailohls-training-contract-v1":
        raise ValueError(f"Unsupported Stage-1 training contract: {contract_path}")
    if contract.get("stage") != "stage1":
        raise ValueError(
            f"--init_adapter_dir must contain a Stage-1 adapter, got "
            f"stage={contract.get('stage')!r}"
        )
    missing = [key for key in STAGE1_COMPATIBILITY_FIELDS if key not in contract]
    if missing:
        raise ValueError(f"Stage-1 training contract is missing fields: {missing}")
    mismatches = {
        key: {"stage1": contract[key], "stage2": expected_contract[key]}
        for key in STAGE1_COMPATIBILITY_FIELDS
        if contract[key] != expected_contract[key]
    }
    if mismatches:
        raise ValueError(
            "Stage-1 adapter contract is incompatible with Stage 2: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return contract


def dump_json_atomic(path: str, obj: Any):
    """Write JSON in the destination directory and publish it with os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def harp_state_manifest(model) -> dict:
    tensor_hashes = {}
    combined = hashlib.sha256()
    selected = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "gated_cross_attn_layer" in name
    )
    if not selected:
        raise ValueError("Cannot create initial HARP state manifest: no cross-attention tensors")
    for name, parameter in selected:
        tensor = parameter.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        tensor_hashes[name] = {"sha256": digest, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {"combined_sha256": combined.hexdigest(), "tensors": tensor_hashes}


def verify_and_save_initial_harp_state(model, reference_path: str, output_dir: str) -> dict:
    manifest = harp_state_manifest(model)
    reference = Path(reference_path)
    if reference.exists():
        with reference.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
        if expected.get("combined_sha256") != manifest["combined_sha256"]:
            raise ValueError(
                "Initial HARP state differs from reference: "
                f"expected {expected.get('combined_sha256')}, got {manifest['combined_sha256']}"
            )
    else:
        dump_json_atomic(str(reference), manifest)
        print(f"[INIT-STATE] Created reference atomically: {reference}")
    output_copy = os.path.join(
        output_dir, "initial_harp_state_post_sa_pre_mlp_s123.json"
    )
    dump_json_atomic(output_copy, manifest)
    print(f"[INIT-STATE] combined_sha256={manifest['combined_sha256']}")
    return manifest



# =====================================================
# Goal-aware point selection
# =====================================================

def pareto_nondominated_mask(rows: List[dict]) -> np.ndarray:
    """
    True = row is non-dominated on (latency, area), assuming both are minimized.
    """
    vals = np.array(
        [[float(r["latency"]), float(r["area"])] for r in rows],
        dtype=np.float64,
    )
    keep = np.ones(len(vals), dtype=bool)

    for i in range(len(vals)):
        dominated = (
            (vals[:, 0] <= vals[i, 0]) &
            (vals[:, 1] <= vals[i, 1]) &
            ((vals[:, 0] < vals[i, 0]) | (vals[:, 1] < vals[i, 1]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    return keep


def _kernel_normalized_qor(rows: List[dict]):
    lat_vals = np.array([float(r["latency"]) for r in rows], dtype=np.float64)
    area_vals = np.array([float(r["area"]) for r in rows], dtype=np.float64)

    lat_vals = np.log2(np.maximum(lat_vals, 1e-12))

    def minmax(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    lat_n = minmax(lat_vals)
    area_n = minmax(area_vals)
    return lat_n, area_n


def pareto_records_for_kernel(items: List[dict]) -> List[dict]:
    lat_n, area_n = _kernel_normalized_qor(items)
    out = []
    for row, ln, an in zip(items, lat_n, area_n):
        latency = max(float(row["latency"]), 1e-12)
        area = max(float(row["area"]), 1e-12)
        out.append({
            "row": row,
            "lat_n": float(ln),
            "area_n": float(an),
            "adp_log": float(math.log2(latency) + math.log2(area)),
        })

    nd_mask = pareto_nondominated_mask(items)
    for rec, keep in zip(out, nd_mask):
        rec["is_pareto"] = bool(keep)

    frontier = [x for x in out if x["is_pareto"]]
    for x in out:
        x["knee_dist"] = float(math.sqrt(x["lat_n"] ** 2 + x["area_n"] ** 2))
        if x["is_pareto"]:
            x["dom_gap"] = 0.0
            x["dom_count"] = 0
            continue

        dom_count = 0
        for y in out:
            if (
                y["lat_n"] <= x["lat_n"]
                and y["area_n"] <= x["area_n"]
                and (y["lat_n"] < x["lat_n"] or y["area_n"] < x["area_n"])
            ):
                dom_count += 1
        x["dom_count"] = int(dom_count)

        gaps = [
            max(0.0, x["lat_n"] - f["lat_n"]) + max(0.0, x["area_n"] - f["area_n"])
            for f in frontier
        ]
        x["dom_gap"] = float(min(gaps)) if gaps else float("inf")

    return out


def goal_distance_to_ideal(lat_n: float, area_n: float, goal_mode: str) -> float:
    if goal_mode == "PARETO_LATENCY":
        return lat_n
    if goal_mode == "PARETO_AREA":
        return area_n
    if goal_mode == "PARETO_ADP":
        # Handled explicitly in goal_sort_key because it uses raw log(latency * area),
        # not normalized distance-to-knee.
        return 0.0
    raise ValueError(f"Unknown goal_mode: {goal_mode}")


def goal_sort_key(rec: dict, goal_mode: str, domination_penalty: float = 0.0):
    lat_n = float(rec["lat_n"])
    area_n = float(rec["area_n"])
    dom_gap = float(rec.get("dom_gap", 0.0))

    if goal_mode == "PARETO_ADP":
        primary = float(rec["adp_log"])
    else:
        primary = goal_distance_to_ideal(lat_n, area_n, goal_mode)

    if domination_penalty > 0.0:
        primary = primary + domination_penalty * dom_gap

    if goal_mode == "PARETO_LATENCY":
        return (primary, area_n)
    if goal_mode == "PARETO_AREA":
        return (primary, lat_n)
    if goal_mode == "PARETO_ADP":
        return (primary, lat_n + area_n, abs(lat_n - area_n))
    raise ValueError(f"Unknown goal_mode: {goal_mode}")


def canonical_completion_key(source_text: str, target_text: str) -> str:
    """
    Deduplicates by directive assignment (no duplicate in our dataset but keep for robustness)
    If 2 rows have the exact same directive completion, only the best-ranked one is kept.
    """
    target_core = reorder_target_by_source_order(source_text, target_text.strip())
    rhs_map = build_rhs_map_from_target(target_core)

    parts = []
    for label, lhs in extract_ordered_lhs_plan(source_text):
        if lhs in rhs_map:
            parts.append(f"{lhs}={rhs_map[lhs].strip()}")
    return "\n".join(parts)


def score_gap_weight(
    score: float,
    best_score: float,
    worst_score: float,
    w_min: float = 0.6,
    power: float = 1.0,
) -> float:
    """
    Map a per-kernel normalized score to a mild sample weight in [w_min, 1.0].

    score:       lower is better
    best_score:  best score among the chosen top_k for this kernel
    worst_score: worst score among the chosen top_k for this kernel

    power:
      1.0  -> linear decay
      >1.0 -> more aggressive emphasis on the best few
      <1.0 -> flatter weights
    """
    w_min = float(max(0.0, min(1.0, w_min)))
    power = float(max(1e-6, power))

    if worst_score <= best_score + 1e-12:
        return 1.0

    gap = (score - best_score) / (worst_score - best_score)
    gap = max(0.0, min(1.0, gap))

    return float(w_min + (1.0 - w_min) * ((1.0 - gap) ** power))


def rank_goal_candidates(
    rows: List[dict],
    goal_mode: str,
    domination_penalty: float,
    max_dominated_gap: float,
) -> List[dict]:
    decorated = pareto_records_for_kernel(rows)
    frontier = sorted(
        [x for x in decorated if x["is_pareto"]],
        key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=0.0),
    )

    preferred = []
    fallback = []
    for x in decorated:
        if x["is_pareto"]:
            continue
        if float(x["dom_gap"]) <= float(max_dominated_gap):
            preferred.append(x)
        else:
            fallback.append(x)

    preferred.sort(key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=domination_penalty))
    fallback.sort(key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=domination_penalty))
    return frontier + preferred + fallback


def build_local_hard_negative_bank(unique_ranked, hard_neg_top_k=6):
    """
    Use nearby ranked alternatives as contrastive candidates.
    """
    best_row = unique_ranked[0]["row"]
    best_target = reorder_target_by_source_order(best_row["input"], best_row["target"].strip())
    best_rhs = build_rhs_map_from_target(best_target)

    bank = defaultdict(set)

    for rec in unique_ranked[1:hard_neg_top_k]:
        row = rec["row"]
        target_core = reorder_target_by_source_order(row["input"], row["target"].strip())
        rhs_map = build_rhs_map_from_target(target_core)

        for lhs, rhs in rhs_map.items():
            lhs = lhs.upper()
            rhs = rhs.strip()
            if best_rhs.get(lhs, None) != rhs:
                bank[lhs].add(rhs)

    return bank


def build_contrastive_sites_from_sample(
    source_text: str,
    target_text: str,
    prompt_ids: List[int],
    tok,
    max_length: int,
    local_hard_negatives: Optional[Dict[str, List[str]]] = None,
    candidate_sites_per_sample: int = 0,
    candidate_negatives_per_site: int = 0,
    kind_priority: Optional[Dict[str, float]] = None,
):
    """
    For selected directive sites, keep:
      - prefix token ids up to 'lhs = '
      - gold RHS token ids
      - local hard-negative RHS token ids
    """
    if candidate_sites_per_sample <= 0 or candidate_negatives_per_site <= 0:
        return []

    rhs_map = build_rhs_map_from_target(target_text)
    full_plan = extract_ordered_lhs_plan(source_text)
    plan = [(label, lhs) for (label, lhs) in full_plan if lhs in rhs_map]

    local_hard_negatives = local_hard_negatives or {}
    kind_priority = kind_priority or {}

    prefix_ids = list(prompt_ids)
    current_label = None
    sites = []

    for label, lhs in plan:
        if label != current_label:
            anchor_ids = tok(
                f"{target_placeholder_token(label)}\n",
                add_special_tokens=False
            )["input_ids"]

            if len(prefix_ids) + len(anchor_ids) > max_length:
                break

            prefix_ids = prefix_ids + anchor_ids
            current_label = label

        fixed_ids = tok(f"{lhs} = ", add_special_tokens=False)["input_ids"]
        gold_rhs = rhs_map[lhs].strip()
        gold_ids = tok(gold_rhs + "\n", add_special_tokens=False)["input_ids"]

        prefix_for_site = prefix_ids + fixed_ids

        if len(prefix_for_site) + len(gold_ids) > max_length:
            break

        neg_texts = []
        for neg in local_hard_negatives.get(lhs.upper(), []):
            neg = neg.strip()
            if neg and neg != gold_rhs and neg not in neg_texts:
                neg_texts.append(neg)
            if len(neg_texts) >= candidate_negatives_per_site:
                break

        if neg_texts:
            sites.append({
                "label": label,
                "lhs": lhs,
                "kind": lhs_kind(lhs),
                "prefix_ids": prefix_for_site,
                "gold_rhs": gold_rhs,
                "gold_ids": gold_ids,
                "negative_rhs": neg_texts,
                "negative_ids": [
                    tok(neg + "\n", add_special_tokens=False)["input_ids"]
                    for neg in neg_texts
                ],
            })

        prefix_ids = prefix_for_site + gold_ids

    sites.sort(
        key=lambda s: (
            -float(kind_priority.get(s["kind"], 1.0)),
            s["label"],
            s["lhs"],
        )
    )

    return sites[:candidate_sites_per_sample]



# =======================================================
# Candidate bank + validation decoding for best_stage1
# =======================================================

def build_partial_deterministic_target_text(source_text: str, raw_target: str, min_supervised_sites: int = 1) :
    """
    Build deterministic target text using ONLY directive sites that actually
    exist in the target. This mirrors build_deterministic_rhs_pack().
    """
    target_core = reorder_target_by_source_order(source_text, raw_target.strip())
    rhs_map = build_rhs_map_from_target(target_core)

    out = []
    current_label = None
    n_expected = 0
    n_supervised = 0
    missing_lhs = []

    for label, lhs in extract_ordered_lhs_plan(source_text):
        n_expected += 1
        rhs = rhs_map.get(lhs, None)

        if rhs is None:
            missing_lhs.append(lhs)
            continue

        rhs = rhs.strip()
        if rhs == "" or rhs == "?":
            missing_lhs.append(lhs)
            continue

        if label != current_label:
            out.append(target_placeholder_token(label))
            current_label = label

        out.append(f"{lhs} = {rhs}")
        n_supervised += 1

    if n_supervised < min_supervised_sites:
        raise ValueError(
            f"Too few supervised sites: kept={n_supervised}, expected={n_expected}"
        )

    return "\n".join(out).strip(), {
        "n_expected": n_expected,
        "n_supervised": n_supervised,
        "coverage": (n_supervised / n_expected) if n_expected > 0 else 0.0,
        "missing_lhs": missing_lhs,
    }


def parse_assignment_dict(text: str) -> Dict[str, str]:
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = ASSIGN_RE.match(line)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out


def canonicalize_generation(text: str) -> str:
    return mailohls_contract.canonicalize_generation(text)


def lhs_kind(lhs: str) -> str:
    m = LHS_KIND_RE.match(lhs.strip())
    if m is None:
        raise ValueError(f"Could not parse lhs kind from: {lhs}")
    return m.group(1).upper()


def _rhs_sort_key(rhs: str):
    s = rhs.strip()
    if re.fullmatch(r"-?\d+", s):
        return (0, int(s), s)
    return (1, s.lower(), s)


DIRECTIVE_DOMAIN_REGISTRY_SCHEMA = "mailohls-directive-domain-registry-v1"


def load_directive_domain_registry(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load exact legal RHS domains keyed by kernel and directive site."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "kernels" in payload:
        schema = payload.get("schema")
        if schema != DIRECTIVE_DOMAIN_REGISTRY_SCHEMA:
            raise ValueError(f"Unsupported directive registry schema: {schema!r}")
        payload = payload["kernels"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Directive domain registry must contain kernel mappings")

    registry: Dict[str, Dict[str, List[str]]] = {}
    for raw_kernel, raw_sites in payload.items():
        kernel = normalize_kname(str(raw_kernel))
        if kernel in registry:
            raise ValueError(f"Duplicate normalized kernel in directive registry: {kernel}")
        if not isinstance(raw_sites, dict) or not raw_sites:
            raise ValueError(f"Kernel {raw_kernel!r} has no directive domains")
        sites: Dict[str, List[str]] = {}
        for raw_lhs, raw_values in raw_sites.items():
            lhs = str(raw_lhs).strip()
            lhs_kind(lhs)  # validate the site syntax
            normalized_lhs = lhs.upper()
            if normalized_lhs in sites:
                raise ValueError(
                    f"Duplicate normalized directive site: {raw_kernel}/{lhs}"
                )
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(f"Directive site {raw_kernel}/{lhs} has an empty domain")
            values = [str(value).strip() for value in raw_values]
            if any(not value or value == "?" for value in values):
                raise ValueError(f"Directive site {raw_kernel}/{lhs} has an invalid RHS")
            if len(values) != len(set(values)):
                raise ValueError(f"Directive site {raw_kernel}/{lhs} has duplicate RHS values")
            sites[normalized_lhs] = sorted(values, key=_rhs_sort_key)
        registry[kernel] = sites
    return registry


def get_rhs_candidates_for_lhs(
    kernel_name: str,
    lhs: str,
    directive_domain_registry: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    kernel = normalize_kname(kernel_name)
    sites = directive_domain_registry.get(kernel)
    if sites is None:
        raise KeyError(f"No directive domains found for kernel={kernel_name!r}")
    cands = sites.get(lhs.strip().upper(), [])
    if not cands:
        raise KeyError(f"No legal RHS domain for kernel={kernel_name!r}, lhs={lhs!r}")
    return cands


DIRECTIVE_WEIGHT_MIN = 0.5
DIRECTIVE_WEIGHT_MAX = 2.0


def compute_directive_loss_weights(
    train_rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> Dict[str, float]:
    """Derive optional directive balancing from the selected training split."""
    if mode == "uniform":
        return {}
    if mode != "inverse_sqrt_frequency":
        raise ValueError(f"Unsupported directive loss weighting: {mode}")

    counts: Counter = Counter()
    for row in train_rows:
        target_core = reorder_target_by_source_order(
            row["input"], str(row["target"]).strip()
        )
        for lhs, rhs in build_rhs_map_from_target(target_core).items():
            if rhs.strip() and rhs.strip() != "?":
                counts[lhs_kind(lhs)] += 1
    if not counts:
        raise ValueError("Cannot balance directive loss: training split has no RHS targets")

    total = float(sum(counts.values()))
    return {
        kind: min(
            DIRECTIVE_WEIGHT_MAX,
            max(DIRECTIVE_WEIGHT_MIN, math.sqrt(total / float(count))),
        )
        for kind, count in sorted(counts.items())
    }


@torch.no_grad()
def append_token_ids(input_ids, attention_mask, new_ids: List[int]):
    device = input_ids.device
    new_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=device)
    new_attn = torch.ones((1, len(new_ids)), dtype=attention_mask.dtype, device=device)
    input_ids = torch.cat([input_ids, new_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, new_attn], dim=1)
    return input_ids, attention_mask


@torch.no_grad()
def score_rhs_candidate_suffix(
    *,
    model,
    tok,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    candidate_text: str,
    routing_start_idx: Optional[torch.Tensor] = None,
    use_harp: bool = False,
):
    device = base_input_ids.device
    cand_ids = tok(candidate_text, add_special_tokens=False)["input_ids"]
    if len(cand_ids) == 0:
        raise ValueError(f"Empty candidate_text tokenization: {repr(candidate_text)}")

    full_input_ids, full_attention_mask = append_token_ids(
        base_input_ids, base_attention_mask, cand_ids
    )

    base_len = int(base_input_ids.shape[1])
    cand_len = len(cand_ids)

    model_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
    }

    if use_harp:
        if routing_start_idx is None:
            raise ValueError("routing_start_idx is required when use_harp=True")

        xmask = torch.zeros(
            (1, full_input_ids.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        # apply xattn only on the candidate RHS suffix
        xmask[:, base_len:] = 1.0

        model_inputs["routing_start_idx"] = routing_start_idx
        model_inputs["xattn_apply_mask"] = xmask

    outputs = model(**model_inputs)

    cand_logits = outputs.logits[:, base_len - 1: base_len - 1 + cand_len, :].float()
    target = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(0)

    token_logprobs = F.log_softmax(cand_logits, dim=-1)
    token_logprobs = token_logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).squeeze(0)

    return {
        "sum_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
    }


@torch.no_grad()
def score_rhs_candidate_batch(
    *,
    model,
    tok,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    candidate_texts: List[str],
):
    """Score independent RHS suffixes in one non-structural model forward."""
    if not candidate_texts:
        return []
    if base_input_ids.shape[0] != 1 or base_attention_mask.shape[0] != 1:
        raise ValueError("Candidate batching expects one shared base prefix.")
    candidate_ids = [
        tok(text, add_special_tokens=False)["input_ids"]
        for text in candidate_texts
    ]
    if any(not ids for ids in candidate_ids):
        raise ValueError("A batched candidate tokenized to an empty sequence.")

    device = base_input_ids.device
    batch_size = len(candidate_ids)
    base_len = int(base_input_ids.shape[1])
    max_candidate_len = max(len(ids) for ids in candidate_ids)
    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        raise ValueError("Candidate batching requires tokenizer.pad_token_id.")

    input_ids = torch.full(
        (batch_size, base_len + max_candidate_len),
        int(pad_token_id),
        dtype=base_input_ids.dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, base_len + max_candidate_len),
        dtype=base_attention_mask.dtype,
        device=device,
    )
    input_ids[:, :base_len] = base_input_ids.expand(batch_size, -1)
    attention_mask[:, :base_len] = base_attention_mask.expand(batch_size, -1)
    for index, ids in enumerate(candidate_ids):
        length = len(ids)
        input_ids[index, base_len:base_len + length] = torch.tensor(
            ids, dtype=input_ids.dtype, device=device
        )
        attention_mask[index, base_len:base_len + length] = 1

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.float()
    results = []
    for index, ids in enumerate(candidate_ids):
        length = len(ids)
        candidate_logits = logits[
            index, base_len - 1:base_len - 1 + length, :
        ]
        target = torch.tensor(ids, dtype=torch.long, device=device)
        token_logprobs = F.log_softmax(candidate_logits, dim=-1).gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        results.append({
            "sum_logprob": float(token_logprobs.sum().item()),
            "mean_logprob": float(token_logprobs.mean().item()),
        })
    return results


@torch.no_grad()
def constrained_decode_rhs_by_candidate_scoring(
    *,
    model,
    tok,
    prompt_ids: List[int],
    source_text: str,
    kernel_name: str,
    directive_domain_registry: Dict[str, Dict[str, List[str]]],
    score_reduction: str = "mean",
    harp_x: Optional[torch.Tensor] = None,
    harp_mask: Optional[torch.Tensor] = None,
    routing_start_idx: Optional[torch.Tensor] = None,
    candidate_batch_size: int = 1,
):
    assert score_reduction in {"mean", "sum"}
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be >= 1")

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    if routing_start_idx is None:
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

    parts = []
    current_label = None

    harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)
    use_harp = harp_enabled and (harp_x is not None) and (harp_mask is not None)

    if use_harp:
        model.condition_harp(harp_x.to(device), harp_mask.to(device))

    try:
        for label, lhs in extract_ordered_lhs_plan(source_text):
            if label != current_label:
                anchor_text = f"{target_placeholder_token(label)}\n"
                anchor_ids = tok(anchor_text, add_special_tokens=False)["input_ids"]
                input_ids, attention_mask = append_token_ids(input_ids, attention_mask, anchor_ids)
                parts.append(anchor_text)
                current_label = label

            prefix_text = f"{lhs} = "
            prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, prefix_ids)
            parts.append(prefix_text)

            candidates = get_rhs_candidates_for_lhs(
                kernel_name, lhs, directive_domain_registry
            )

            scored = []
            effective_batch_size = 1 if use_harp else candidate_batch_size
            for start in range(0, len(candidates), effective_batch_size):
                rhs_batch = candidates[start:start + effective_batch_size]
                if effective_batch_size == 1:
                    batch_stats = [score_rhs_candidate_suffix(
                        model=model,
                        tok=tok,
                        base_input_ids=input_ids,
                        base_attention_mask=attention_mask,
                        candidate_text=rhs_batch[0] + "\n",
                        routing_start_idx=routing_start_idx,
                        use_harp=use_harp,
                    )]
                else:
                    batch_stats = score_rhs_candidate_batch(
                        model=model,
                        tok=tok,
                        base_input_ids=input_ids,
                        base_attention_mask=attention_mask,
                        candidate_texts=[rhs + "\n" for rhs in rhs_batch],
                    )
                for rhs, stats in zip(rhs_batch, batch_stats):
                    scored.append({
                        "rhs": rhs,
                        "score": stats["mean_logprob"] if score_reduction == "mean" else stats["sum_logprob"],
                        "mean_logprob": stats["mean_logprob"],
                        "sum_logprob": stats["sum_logprob"],
                    })

            scored.sort(key=lambda x: (x["score"], x["sum_logprob"]), reverse=True)
            best = scored[0]

            chosen_text = best["rhs"] + "\n"
            chosen_ids = tok(chosen_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, chosen_ids)
            parts.append(chosen_text)

        return "".join(parts).rstrip()

    finally:
        if hasattr(model, "clear_harp"):
            model.clear_harp()



def evaluate_prediction(reference_target: str, raw_generation: str) -> Dict[str, object]:
    pred_text = canonicalize_generation(raw_generation)
    ref_text = canonicalize_generation(reference_target)

    ref_signature = directive_schema_signature(ref_text)
    if ref_signature is None:
        raise ValueError("Reference target violates the directive schema")
    pred_signature = directive_schema_signature(pred_text)

    ref_assign = parse_assignment_dict(ref_text)
    pred_assign = parse_assignment_dict(pred_text)

    expected_keys = list(ref_assign.keys())
    expected_key_match = set(pred_assign.keys()) == set(ref_assign.keys())
    schema_compliant = pred_signature is not None and pred_signature == ref_signature
    exact_value_match_count = sum(
        (k in pred_assign) and (pred_assign[k] == ref_assign[k])
        for k in expected_keys
    )
    pragma_kind_counts = defaultdict(lambda: {"correct": 0, "expected": 0})
    for key in expected_keys:
        kind = lhs_kind(key)
        pragma_kind_counts[kind]["expected"] += 1
        pragma_kind_counts[kind]["correct"] += int(
            key in pred_assign and pred_assign[key] == ref_assign[key]
        )

    return {
        "canonical_prediction": pred_text,
        "value_accuracy_over_expected": exact_value_match_count / max(len(expected_keys), 1),
        "schema_compliant": schema_compliant,
        "expected_key_match": expected_key_match,
        "exact_design_match": schema_compliant and pred_assign == ref_assign,
        "pragma_kind_counts": {
            kind: dict(counts)
            for kind, counts in sorted(pragma_kind_counts.items())
        },
    }


@dataclass
class SelectionCase:
    kernel_name: str
    obj_mode: str
    source_text: str
    reference_target: str
    row: dict


def summarize_selection_rows(rows: List[dict]) -> Dict[str, object]:
    """Aggregate selection metrics while giving every kernel equal weight."""
    if not rows:
        raise RuntimeError("Validation selection has no cases.")
    rows_by_kernel = defaultdict(list)
    for row in rows:
        rows_by_kernel[row["kernel_name"]].append(row)
    kernel_value_acc = {
        kernel: sum(
            row["value_accuracy_over_expected"] for row in kernel_rows
        ) / len(kernel_rows)
        for kernel, kernel_rows in sorted(rows_by_kernel.items())
    }
    pragma_kind_totals = defaultdict(lambda: {"correct": 0, "expected": 0})
    for row in rows:
        for kind, counts in row["pragma_kind_counts"].items():
            pragma_kind_totals[kind]["correct"] += counts["correct"]
            pragma_kind_totals[kind]["expected"] += counts["expected"]
    return {
        "mean_value_acc": float(
            sum(row["value_accuracy_over_expected"] for row in rows) / len(rows)
        ),
        "schema_compliance": float(
            sum(row["schema_compliant"] for row in rows) / len(rows)
        ),
        "expected_key_accuracy": float(
            sum(row["expected_key_match"] for row in rows) / len(rows)
        ),
        "exact_design_accuracy": float(
            sum(row["exact_design_match"] for row in rows) / len(rows)
        ),
        "pragma_kind_accuracy": {
            kind: counts["correct"] / counts["expected"]
            for kind, counts in sorted(pragma_kind_totals.items())
            if counts["expected"] > 0
        },
        "per_kernel_accuracy": kernel_value_acc,
        "minimum_kernel_accuracy": min(kernel_value_acc.values()),
        "selection_score": float(
            sum(kernel_value_acc.values()) / len(kernel_value_acc)
        ),
    }



class StageValSelectionCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        selection_cases: List[SelectionCase],
        directive_domain_registry: Dict[str, Dict[str, List[str]]],
        output_dir: str,
        max_prompt_tokens: int = 7168,
        candidate_score_reduction: str = "mean",
        best_dir_name: str = "best_custom_stage1",
        mem_bank: Optional[Dict[str, dict]] = None,
        mem_dim: int = 32,
        max_slots: int = 64,
        training_contract: Optional[dict] = None,
        selection_eval_steps: int = 200,
        candidate_batch_size: int = 1,
    ):
        self.tok = tokenizer
        self.selection_cases = selection_cases
        self.directive_domain_registry = directive_domain_registry
        self.output_dir = output_dir
        self.max_prompt_tokens = max_prompt_tokens
        self.candidate_score_reduction = candidate_score_reduction
        self.best_dir_name = best_dir_name
        self.mem_bank = mem_bank or {}
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.training_contract = training_contract or {}
        self.selection_eval_steps = selection_eval_steps
        self.candidate_batch_size = candidate_batch_size
        best_path = Path(output_dir) / best_dir_name / "best_selection_metrics.json"

        if best_path.is_file():
            previous = json.loads(best_path.read_text(encoding="utf-8"))
            self.best_key = tuple(previous["checkpoint_key"])
            self.best_step = int(previous["step"])
            print(
                f"[VAL-SELECTION] Restored previous best: "
                f"step={self.best_step}, key={self.best_key}"
            )
        else:
            self.best_key = (float("-inf"),) * 4
            self.best_step = -1
        self.last_selection_step = None

    def _run_case(self, model, case: SelectionCase) -> dict:
        prompt = build_prompt(
            case.source_text,
            case.obj_mode,
            row=case.row,
            device_token_dropout=0.0,
        )

        enc = self.tok(prompt, add_special_tokens=False)
        prompt_ids = enc["input_ids"][-self.max_prompt_tokens:] if len(enc["input_ids"]) > self.max_prompt_tokens else enc["input_ids"]

        device = next(model.parameters()).device
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

        harp_x = None
        harp_mask = None
        if hasattr(model, "initialized_harp_flamingo") and getattr(model, "initialized_harp_flamingo", False):
            harp_x, harp_mask = get_real_memory_pack_for_kernel(
                self.mem_bank,
                case.kernel_name,
                self.max_slots,
                self.mem_dim,
            )

        pred = constrained_decode_rhs_by_candidate_scoring(
            model=model,
            tok=self.tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            kernel_name=case.kernel_name,
            directive_domain_registry=self.directive_domain_registry,
            score_reduction=self.candidate_score_reduction,
            harp_x=harp_x,
            harp_mask=harp_mask,
            routing_start_idx=routing_start_idx,
            candidate_batch_size=self.candidate_batch_size,
        )

        metrics = evaluate_prediction(case.reference_target, pred)
        return {
            "kernel_name": case.kernel_name,
            "obj_mode": case.obj_mode,
            "reference_target": case.reference_target,
            "prediction": metrics["canonical_prediction"],
            "value_accuracy_over_expected": float(metrics["value_accuracy_over_expected"]),
            "schema_compliant": bool(metrics["schema_compliant"]),
            "expected_key_match": bool(metrics["expected_key_match"]),
            "exact_design_match": bool(metrics["exact_design_match"]),
            "pragma_kind_counts": metrics["pragma_kind_counts"],
        }

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        is_final = state.global_step >= state.max_steps
        if (
            not is_final
            and state.global_step % self.selection_eval_steps != 0
        ):
            return
        if self.last_selection_step == state.global_step:
            return
        self.last_selection_step = int(state.global_step)

        model = kwargs["model"]
        was_training = model.training
        model.eval()

        try:
            rows = [self._run_case(model, case) for case in self.selection_cases]
            summary = summarize_selection_rows(rows)
            mean_value_acc = summary["mean_value_acc"]
            schema_compliance = summary["schema_compliance"]
            expected_key_accuracy = summary["expected_key_accuracy"]
            exact_design_accuracy = summary["exact_design_accuracy"]
            pragma_kind_accuracy = summary["pragma_kind_accuracy"]
            kernel_value_acc = summary["per_kernel_accuracy"]
            minimum_kernel_accuracy = summary["minimum_kernel_accuracy"]
            selection_score = summary["selection_score"]
            if schema_compliance != 1.0 or expected_key_accuracy != 1.0:
                raise RuntimeError(
                    "Constrained decoder violated its schema contract"
                )
            eval_loss = float(
                kwargs.get("metrics", {}).get("eval_loss", float("inf"))
            )
            checkpoint_key = (
                selection_score,
                minimum_kernel_accuracy,
                exact_design_accuracy,
                -eval_loss,
            )

            print("\n" + "=" * 100)
            print(f"[VAL-SELECTION] step={state.global_step}")
            print(f"[VAL-SELECTION] mean_value_acc={mean_value_acc:.6f}")
            print(f"[VAL-SELECTION] schema_compliance={schema_compliance:.6f}")
            print(f"[VAL-SELECTION] expected_key_accuracy={expected_key_accuracy:.6f}")
            print(f"[VAL-SELECTION] exact_design_accuracy={exact_design_accuracy:.6f}")
            print(f"[VAL-SELECTION] pragma_kind_accuracy={pragma_kind_accuracy}")
            print(f"[VAL-SELECTION] per_kernel_accuracy={kernel_value_acc}")
            print(f"[VAL-SELECTION] minimum_kernel_accuracy={minimum_kernel_accuracy:.6f}")
            print(f"[VAL-SELECTION] selection_score={selection_score:.6f}")
            print(f"[VAL-SELECTION] checkpoint_key={checkpoint_key}")
            print("=" * 100)

            metrics_obj = {
                "step": int(state.global_step),
                "eval_loss": eval_loss,
                "checkpoint_key": list(checkpoint_key),
                **summary,
                "rows": rows,
            }

            dump_json(
                os.path.join(self.output_dir, f"val_selection_step_{state.global_step}.json"),
                metrics_obj,
            )

            if checkpoint_key > self.best_key:
                self.best_key = checkpoint_key
                self.best_step = int(state.global_step)

                best_dir = os.path.join(self.output_dir, self.best_dir_name)
                if os.path.isdir(best_dir):
                    shutil.rmtree(best_dir)

                save_mailohls_adapter(
                    model, self.tok, best_dir, self.training_contract
                )
                if hasattr(model, "initialized_harp_flamingo") and getattr(model, "initialized_harp_flamingo", False):
                    harp_sd = get_harp_xattn_state_dict(model)
                    if harp_sd:
                        torch.save(harp_sd, os.path.join(best_dir, "harp_xattn.pt"))
                        print(f"[VAL-SELECTION] Saved best HARP xattn weights -> {os.path.join(best_dir, 'harp_xattn.pt')}")

                dump_json(
                    os.path.join(best_dir, "best_selection_metrics.json"),
                    metrics_obj,
                )

                print(f"[VAL-SELECTION] New best checkpoint at step {state.global_step} -> {best_dir}")

        finally:
            if was_training:
                model.train()


# ==========================================
# HARP memory bank loader (.memory.pt files)
# ==========================================

def load_memory_bank(
    memory_dir: str,
    expected_mem_dim: Optional[int] = None,
    expected_max_slots: Optional[int] = None,
    require_pragma_free_memory: bool = False,
) -> Tuple[Dict[str, dict], Optional[int]]:
    bank = {}
    inferred_mem_dim = None

    for fn in os.listdir(memory_dir):
        if not fn.endswith(".memory.pt"):
            continue

        pack = torch.load(os.path.join(memory_dir, fn), map_location="cpu", weights_only=False)

        kv = pack["node_embs"].float()
        mask = pack["node_embs_mask"].bool()
        labels = pack.get("labels", None)
        slot_cats = pack.get("slot_cats", None)

        if kv.ndim != 2:
            raise ValueError(f"{fn}: node_embs must be [S, D], got {tuple(kv.shape)}")
        if mask.ndim != 1 or mask.numel() != kv.size(0):
            raise ValueError(f"{fn}: bad node_embs_mask shape {tuple(mask.shape)} for node_embs {tuple(kv.shape)}")

        gnn_dim = int(pack.get("gnn_dim", kv.size(1)))
        if gnn_dim != kv.size(1):
            raise ValueError(f"{fn}: gnn_dim={gnn_dim} but node_embs.size(1)={kv.size(1)}")

        if inferred_mem_dim is None:
            inferred_mem_dim = gnn_dim
        elif inferred_mem_dim != gnn_dim:
            raise ValueError(f"Inconsistent gnn_dim across memory packs: {inferred_mem_dim} vs {gnn_dim}")

        if expected_mem_dim is not None and gnn_dim != expected_mem_dim:
            raise ValueError(f"{fn}: memory dim mismatch, pack={gnn_dim}, expected={expected_mem_dim}")

        max_slots = int(pack.get("max_slots", kv.size(0)))
        if expected_max_slots is not None and max_slots != expected_max_slots:
            raise ValueError(f"{fn}: max_slots mismatch, pack={max_slots}, expected={expected_max_slots}")

        if require_pragma_free_memory and not bool(pack.get("disable_pragma_injection", False)):
            raise ValueError(f"{fn}: memory was not built with disable_pragma_injection=True")

        # Preserve absolute MailoHLS Lk slot alignment.
        #
        # build_structural_memory.py stores action Lk at slot k-1.
        # Sparse action sets are therefore valid; for example, a kernel
        # may contain active L2..L15 while slot 0 / L1 remains inactive.
        #
        # Do NOT compact active vectors to slots 0..N-1: structural_xattn
        # routes the <Lk> placeholder directly to one-based memory slot k.
        if labels is not None:
            labels_t = torch.as_tensor(
                labels,
                dtype=torch.long,
            ).view(-1)

            if labels_t.numel() != kv.size(0):
                raise ValueError(
                    f"{fn}: labels have {labels_t.numel()} entries "
                    f"for {kv.size(0)} memory slots"
                )

            expected_slots = torch.arange(
                1,
                kv.size(0) + 1,
                dtype=torch.long,
            )

            active_idx = mask.nonzero(
                as_tuple=False
            ).view(-1)

            if active_idx.numel() > 0:
                active_labels = labels_t.index_select(
                    0,
                    active_idx,
                )
                expected_active = expected_slots.index_select(
                    0,
                    active_idx,
                )

                if not torch.equal(
                    active_labels,
                    expected_active,
                ):
                    raise ValueError(
                        f"{fn}: active labels violate absolute Lk "
                        f"slot alignment: "
                        f"labels={active_labels.tolist()}, "
                        f"expected={expected_active.tolist()}"
                    )

        # slot_ids, when present, must describe the same fixed absolute
        # one-based structural address space.
        slot_ids = pack.get("slot_ids", None)

        if slot_ids is not None:
            slot_ids_t = torch.as_tensor(
                slot_ids,
                dtype=torch.long,
            ).view(-1)

            expected_slots = torch.arange(
                1,
                kv.size(0) + 1,
                dtype=torch.long,
            )

            if not torch.equal(
                slot_ids_t,
                expected_slots,
            ):
                raise ValueError(
                    f"{fn}: slot_ids must equal absolute [1..S]; "
                    f"got {slot_ids_t.tolist()}"
                )

        if slot_cats is not None:
            slot_cats = torch.as_tensor(
                slot_cats,
                dtype=torch.long,
            ).view(-1)

            if slot_cats.numel() != kv.size(0):
                raise ValueError(
                    f"{fn}: slot_cats have {slot_cats.numel()} entries "
                    f"for {kv.size(0)} memory slots"
                )

        k = fn.replace(".memory.pt", "")
        rec = {
            "kv": kv.contiguous(),
            "mask": mask.contiguous(),
            "slot_cats": slot_cats,
            "ckpt": pack.get("ckpt", ""),
            "disable_pragma_injection": bool(pack.get("disable_pragma_injection", False)),
        }
        bank[k] = rec
        bank[normalize_kname(k)] = rec

    return bank, inferred_mem_dim


# ================================
# Structural cross-attention utilities
# ================================


def print_xattn_gate_stats(model, print_grads=True):
    attn_gates = []
    ff_gates = []

    for n, p in model.named_parameters():
        if n.endswith("attn_gate") or n.endswith("ff_gate"):
            raw = float(p.detach().cpu().item())
            tanh_val = float(p.detach().cpu().tanh().item())

            grad = None
            grad_abs = None
            if print_grads and p.grad is not None:
                grad = float(p.grad.detach().cpu().item())
                grad_abs = abs(grad)

            row = {
                "name": n,
                "raw": raw,
                "tanh": tanh_val,
                "grad": grad,
                "grad_abs": grad_abs,
            }

            if n.endswith("attn_gate"):
                attn_gates.append(row)
            else:
                ff_gates.append(row)

    print("[GATES] attn gates:")
    for row in attn_gates:
        if row["grad"] is None:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} grad=None"
            )
        else:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} "
                f"grad={row['grad']:.8e} |grad|={row['grad_abs']:.8e}"
            )

    print("[GATES] ff gates:")
    for row in ff_gates:
        if row["grad"] is None:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} grad=None"
            )
        else:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} "
                f"grad={row['grad']:.8e} |grad|={row['grad_abs']:.8e}"
            )


def print_xattn_forward_stats(model):
    found = False
    for name, module in model.named_modules():
        if isinstance(module, MaskedCrossAttention) and getattr(module, "last_debug", None):
            found = True
            dbg = module.last_debug
            print(f"[XATTN-DBG] {name}: {dbg}")
    if not found:
        print("[XATTN-DBG] no cross-attn forward stats collected yet")


def get_harp_xattn_state_dict(model):
    sd = model.state_dict()
    return {
        k: v.detach().cpu()
        for k, v in sd.items()
        if "gated_cross_attn_layer" in k
    }


def get_first_real_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0")


def move_harp_modules_to_model_device(model):
    device = get_first_real_device(model)
    moved = 0
    for module in model.modules():
        if isinstance(module, GatedCrossAttentionBlock):
            module.to(device=device)
            moved += 1
    print(f"[HARP-DEVICE] moved {moved} HARP blocks to {device}")


def load_partial_harp_xattn(model, harp_xattn_path: str, tag: str):
    if not harp_xattn_path or not os.path.isfile(harp_xattn_path):
        print(f"[{tag}] no harp_xattn.pt found at: {harp_xattn_path}")
        return

    harp_sd = torch.load(harp_xattn_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(harp_sd, strict=False)

    harp_missing = [k for k in missing if "gated_cross_attn_layer" in k]
    print(f"[{tag}] harp_missing[:10]={harp_missing[:10]}")
    print(f"[{tag}] unexpected[:10]={unexpected[:10]}")

    move_harp_modules_to_model_device(model)


# Shared implementation used by new checkpoints.
MaskedCrossAttention = structural_xattn.MaskedCrossAttention
GatedCrossAttentionBlock = structural_xattn.GatedCrossAttentionBlock
HARPLMMixin = structural_xattn.StructuralCrossAttentionMixin
extend_instance = structural_xattn.extend_instance
infer_decoder_layers_attr_name = structural_xattn.infer_decoder_layers_attr_name


class SaveMailoHLSCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, training_contract):
        self.tokenizer = tokenizer
        self.training_contract = training_contract

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_mailohls_adapter(
            model, self.tokenizer, ckpt_dir, self.training_contract
        )

        harp_sd = get_harp_xattn_state_dict(model)
        if harp_sd:
            torch.save(harp_sd, os.path.join(ckpt_dir, "harp_xattn.pt"))
            print(f"[HARP-SAVE] saved xattn weights to {ckpt_dir}/harp_xattn.pt")
        return control



# ====================================
# Dataset + Pad Collator (SFT)
# ====================================

class PadCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)

        def pad_1d(t, pad_value):
            if t.shape[0] == max_len:
                return t
            pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        input_ids = torch.stack([pad_1d(x["input_ids"], self.tok.pad_token_id) for x in batch])
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch])
        labels = torch.stack([pad_1d(x["labels"], -100) for x in batch])
        sample_weight = torch.stack([x["sample_weight"] for x in batch])
        kernel_name = [x["kernel_name"] for x in batch]
        routing_start_idx = torch.stack([x["routing_start_idx"] for x in batch])
        token_weights = torch.stack([pad_1d(x["token_weights"], 0.0) for x in batch])
        xattn_apply_mask = torch.stack([pad_1d(x["xattn_apply_mask"], 0.0) for x in batch])
        contrastive_sites = [x["contrastive_sites"] for x in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": sample_weight,
            "kernel_name": kernel_name,
            "routing_start_idx": routing_start_idx,
            "token_weights": token_weights,
            "xattn_apply_mask": xattn_apply_mask,
            "contrastive_sites": contrastive_sites,
        }



# ========================================
# Model Helpers for Stage_1 (simple SFT)
# ========================================

def get_input_embeddings_module(model):
    emb = None

    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()

    if emb is None and hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        emb = model.base_model.get_input_embeddings()

    if emb is None:
        raise ValueError("Could not access model input embeddings.")

    return emb


# =========================================
# Trainer (SFT : Stage_1 + Stage_2)
# =========================================

class LengthGroupedTrainer(Trainer):
    """
    - Length-grouped sampling + Per-sample weights
    - Conditions HARP memory per batch using kernel_name
    - Computes chunked CE to avoid giant [B*T,V] flatten allocations
    """
    def __init__(
        self,
        *args,
        group_by_length: bool = False,
        mem_bank: Optional[Dict[str, dict]] = None,
        mem_dim: int = 32,
        max_slots: int = 64,
        lr_lora: float = 2e-4,
        lr_xattn: float = 1e-4,
        lr_gate: float = 1e-3,
        lr_ff: float = 0.0,
        lr_gate_ff: float = 0.0,
        lr_embed: Optional[float] = None,
        loss_chunk_t: int = 256,
        candidate_loss_weight: float = 0.0,
        candidate_sites_per_sample: int = 0,
        candidate_negatives_per_site: int = 0,
        candidate_max_prefix_tokens: int = 1536,
        candidate_keep_head_tokens: int = 256,
        **kwargs,
    ):
        self._group_by_length = group_by_length
        self.mem_bank = mem_bank or {}
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.lr_lora = lr_lora
        self.lr_xattn = lr_xattn
        self.lr_gate = lr_gate
        self.lr_ff = lr_ff
        self.lr_gate_ff = lr_gate_ff
        self.lr_embed = lr_lora if lr_embed is None else lr_embed
        self.loss_chunk_t = loss_chunk_t
        self.candidate_loss_weight = float(candidate_loss_weight)
        self.candidate_sites_per_sample = int(candidate_sites_per_sample)
        self.candidate_negatives_per_site = int(candidate_negatives_per_site)
        self.candidate_max_prefix_tokens = int(candidate_max_prefix_tokens)
        self.candidate_keep_head_tokens = int(candidate_keep_head_tokens)
        self._last_debug_step = -1
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lora_params, embed_params = [], []
        attn_gate_params, ff_gate_params = [], []
        xattn_attn_params, xattn_ff_params = [], []
        other_trainables = []

        try:
            input_emb_param_ids = {id(p) for p in get_input_embeddings_module(self.model).parameters()}
        except Exception:
            input_emb_param_ids = set()

        try:
            output_emb = self.model.get_output_embeddings()
            output_emb_param_ids = {id(p) for p in output_emb.parameters()} if output_emb is not None else set()
        except Exception:
            output_emb_param_ids = set()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            if id(p) in input_emb_param_ids or id(p) in output_emb_param_ids:
                embed_params.append(p)
            elif "lora_" in n:
                lora_params.append(p)
            elif n.endswith("attn_gate"):
                attn_gate_params.append(p)
            elif n.endswith("ff_gate"):
                ff_gate_params.append(p)
            elif "gated_cross_attn_layer.attn." in n:
                xattn_attn_params.append(p)
            elif "gated_cross_attn_layer.ff." in n:
                xattn_ff_params.append(p)
            else:
                other_trainables.append((n, p))

        if other_trainables:
            print("[WARN] Unexpected trainable params:")
            for n, _ in other_trainables[:20]:
                print("  -", n)

        opt_groups = []
        if lora_params:
            opt_groups.append({"params": lora_params, "lr": self.lr_lora})
        if embed_params:
            opt_groups.append({"params": embed_params, "lr": self.lr_embed})
        if attn_gate_params:
            opt_groups.append({"params": attn_gate_params, "lr": self.lr_gate})
        if ff_gate_params:
            opt_groups.append({"params": ff_gate_params, "lr": self.lr_gate_ff})
        if xattn_attn_params:
            opt_groups.append({"params": xattn_attn_params, "lr": self.lr_xattn})
        if xattn_ff_params:
            opt_groups.append({"params": xattn_ff_params, "lr": self.lr_ff})
        if other_trainables:
            opt_groups.append({"params": [p for _, p in other_trainables], "lr": self.lr_lora})

        try:
            from bitsandbytes.optim import PagedAdamW8bit
            self.optimizer = PagedAdamW8bit(opt_groups, weight_decay=0.0)
        except Exception:
            self.optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.0)

        print(
            f"[OPT] param groups: "
            f"lora={sum(p.numel() for p in lora_params):,} "
            f"embed={sum(p.numel() for p in embed_params):,} "
            f"attn_gate={sum(p.numel() for p in attn_gate_params):,} "
            f"ff_gate={sum(p.numel() for p in ff_gate_params):,} "
            f"xattn_attn={sum(p.numel() for p in xattn_attn_params):,} "
            f"xattn_ff={sum(p.numel() for p in xattn_ff_params):,} "
            f"lr_lora={self.lr_lora:g} "
            f"lr_embed={self.lr_embed:g} "
            f"lr_gate={self.lr_gate:g} "
            f"lr_gate_ff={self.lr_gate_ff:g} "
            f"lr_xattn={self.lr_xattn:g} "
            f"lr_ff={self.lr_ff:g}"
        )
        return self.optimizer

    def get_train_dataloader(self):
        if not self._group_by_length:
            return super().get_train_dataloader()

        sampler = LengthGroupedSampler(
            self.args.train_batch_size,
            dataset=self.train_dataset,
            lengths=getattr(self.train_dataset, "lengths", None),
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # IMPORTANT: clear only after backward, so checkpoint recomputation
        # still sees the conditioned HARP state
        if hasattr(model, "clear_harp"):
            model.clear_harp()

        if self.accelerator.sync_gradients and self.state.global_step != self._last_debug_step:
            if self.state.global_step % 20 == 0:
                print_xattn_gate_stats(model, print_grads=True)
                if hasattr(model, "clear_harp") and not getattr(self.args, "disable_harp", False):
                    print_xattn_forward_stats(model)
                self._last_debug_step = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps
    

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs and inputs["labels"] is not None

        if has_labels:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            loss = loss.mean().detach()

            if prediction_loss_only:
                return (loss, None, None)

            logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()
            labels = inputs["labels"].detach()
            return (loss, logits, labels)

        model_inputs = dict(inputs)
        for k in (
            "sample_weight",
            "kernel_name",
            "routing_start_idx",
            "token_weights",
            "xattn_apply_mask",
            "contrastive_sites",
        ):
            model_inputs.pop(k, None)

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()
        return (None, logits, None)


    def _condition_harp_from_kernel_names(self, model, kernel_names: List[str]):
        kvs, ms = [], []
        for k in kernel_names:
            pack = self.mem_bank.get(k) or self.mem_bank.get(normalize_kname(k))
            if pack is None:
                kvs.append(torch.zeros((self.max_slots, self.mem_dim), dtype=torch.float32))
                ms.append(torch.zeros((self.max_slots,), dtype=torch.bool))
            else:
                kvs.append(pack["kv"])
                ms.append(pack["mask"])
        mem_kv = torch.stack(kvs, dim=0)  # [B, S, mem_dim]
        mem_m  = torch.stack(ms, dim=0)   # [B, S]
        device = next(model.parameters()).device
        model.condition_harp(mem_kv.to(device), mem_m.to(device))

    def truncate_scoring_prefix_preserve_target(
        self,
        prefix_ids: List[int],
        routing_start_idx: Optional[int],
        max_prefix_tokens: int,
        keep_head_tokens: int,
    ) -> Tuple[List[int], Optional[int]]:
        if (
            routing_start_idx is None
            or max_prefix_tokens <= 0
            or len(prefix_ids) <= max_prefix_tokens
        ):
            return prefix_ids, routing_start_idx

        R = int(routing_start_idx)
        prompt_ids = prefix_ids[:R]
        target_prefix_ids = prefix_ids[R:]

        # Always preserve the entire generated target prefix if possible,
        # because HARP routing depends on target anchors already emitted.
        if len(target_prefix_ids) >= max_prefix_tokens:
            kept_target = target_prefix_ids[-max_prefix_tokens:]
            return kept_target, 0

        prompt_budget = max_prefix_tokens - len(target_prefix_ids)
        if len(prompt_ids) <= prompt_budget:
            return prefix_ids, R

        keep_head = min(max(0, keep_head_tokens), max(0, prompt_budget - 1))
        keep_tail = prompt_budget - keep_head

        if keep_head <= 0:
            kept_prompt = prompt_ids[-prompt_budget:]
        else:
            kept_prompt = prompt_ids[:keep_head] + prompt_ids[-keep_tail:]

        new_prefix = kept_prompt + target_prefix_ids
        new_routing_start_idx = len(kept_prompt)
        return new_prefix, new_routing_start_idx


    def _score_candidate_sequence(
        self,
        model,
        prefix_ids: List[int],
        candidate_ids: List[int],
        routing_start_idx: Optional[int],
        use_harp: bool,
    ):
        device = next(model.parameters()).device

        effective_route_idx = int(routing_start_idx) if routing_start_idx is not None else None

        prefix_ids, effective_route_idx = self.truncate_scoring_prefix_preserve_target(
            prefix_ids=prefix_ids,
            routing_start_idx=effective_route_idx,
            max_prefix_tokens=self.candidate_max_prefix_tokens,
            keep_head_tokens=self.candidate_keep_head_tokens,
        )

        base_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        base_attention_mask = torch.ones_like(base_input_ids)

        cand_tensor = torch.tensor([candidate_ids], dtype=torch.long, device=device)

        full_input_ids = torch.cat([base_input_ids, cand_tensor], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)

        model_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        }

        if use_harp:
            if effective_route_idx is None:
                raise ValueError("effective_route_idx is required when use_harp=True")

            model_inputs["routing_start_idx"] = torch.tensor(
                [effective_route_idx],
                dtype=torch.long,
                device=device,
            )
            xmask = torch.zeros(
                (1, full_input_ids.shape[1]),
                dtype=torch.float32,
                device=device,
            )
            xmask[:, effective_route_idx:] = 1.0
            model_inputs["xattn_apply_mask"] = xmask

        outputs = model(**model_inputs)

        base_len = len(prefix_ids)
        cand_len = len(candidate_ids)

        cand_logits = outputs.logits[:, base_len - 1: base_len - 1 + cand_len, :].float()
        token_logprobs = F.log_softmax(cand_logits, dim=-1)
        token_logprobs = token_logprobs.gather(
            -1,
            cand_tensor.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

        return token_logprobs.mean()

    def _compute_candidate_loss(
        self,
        model,
        contrastive_sites,
        kernel_names,
        routing_start_idx,
        sample_weights=None,
    ):
        """
        The model should learn:
            score(gold RHS) > score(negative RHS alternatives)

        We compute:
        1) mean candidate loss per sample (over its available selected sites)
        2) weighted average across samples, matching the CE loss logic
        """
        device = next(model.parameters()).device

        per_sample_losses = []
        per_sample_weights = []

        harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)

        for b_idx, sites in enumerate(contrastive_sites):
            if not sites:
                continue

            route_idx = int(routing_start_idx[b_idx].item()) if routing_start_idx is not None else 0

            if harp_enabled:
                self._condition_harp_from_kernel_names(model, [kernel_names[b_idx]])

            site_losses = []

            for site in sites[:self.candidate_sites_per_sample]:
                neg_ids_list = site["negative_ids"][:self.candidate_negatives_per_site]
                if not neg_ids_list:
                    continue

                gold_score = self._score_candidate_sequence(
                    model=model,
                    prefix_ids=site["prefix_ids"],
                    candidate_ids=site["gold_ids"],
                    routing_start_idx=route_idx,
                    use_harp=harp_enabled,
                )

                neg_scores = [
                    self._score_candidate_sequence(
                        model=model,
                        prefix_ids=site["prefix_ids"],
                        candidate_ids=neg_ids,
                        routing_start_idx=route_idx,
                        use_harp=harp_enabled,
                    )
                    for neg_ids in neg_ids_list
                ]

                scores = torch.stack([gold_score] + neg_scores, dim=0)
                site_loss = -F.log_softmax(scores, dim=0)[0]
                site_losses.append(site_loss)

            if not site_losses:
                continue

            sample_loss = torch.stack(site_losses).mean()
            per_sample_losses.append(sample_loss)

            if sample_weights is not None:
                per_sample_weights.append(sample_weights[b_idx].to(device=device, dtype=torch.float32))
            else:
                per_sample_weights.append(torch.tensor(1.0, device=device, dtype=torch.float32))

        if not per_sample_losses:
            return torch.zeros((), device=device, dtype=torch.float32)

        per_sample_losses = torch.stack(per_sample_losses)
        per_sample_weights = torch.stack(per_sample_weights)

        return (per_sample_losses * per_sample_weights).sum() / per_sample_weights.sum().clamp(min=1e-8)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        weights = inputs.pop("sample_weight", None)
        kernel_names = inputs.pop("kernel_name", None)
        labels = inputs["labels"]
        token_weights = inputs.pop("token_weights", None)
        xattn_apply_mask = inputs.pop("xattn_apply_mask", None)
        contrastive_sites = inputs.pop("contrastive_sites", None)

        if kernel_names is not None and hasattr(model, "condition_harp"):
            self._condition_harp_from_kernel_names(model, kernel_names)

        try:
            routing_start_idx = inputs.pop("routing_start_idx", None)

            model_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

            harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)

            if harp_enabled:
                if routing_start_idx is not None:
                    model_inputs["routing_start_idx"] = routing_start_idx
                if xattn_apply_mask is not None:
                    model_inputs["xattn_apply_mask"] = xattn_apply_mask

            outputs = model(**model_inputs)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_token_weights = token_weights[:, 1:].contiguous() if token_weights is not None else None

            B, Tm1 = shift_labels.shape
            device = shift_labels.device

            chunk_t = int(self.loss_chunk_t)
            loss_sum = torch.zeros(B, device=device, dtype=torch.float32)
            tok_cnt = torch.zeros(B, device=device, dtype=torch.float32)

            for s in range(0, Tm1, chunk_t):
                e = min(Tm1, s + chunk_t)
                logits_chunk = shift_logits[:, s:e, :]
                labels_chunk = shift_labels[:, s:e]

                flat_logits = logits_chunk.reshape(-1, logits_chunk.size(-1))
                flat_labels = labels_chunk.reshape(-1)

                per_tok = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="none",
                ).view(B, -1)

                mask = labels_chunk.ne(-100)
                tok_weight = torch.ones_like(per_tok)

                # Each supervised token is weighted by : 
                # 0 for prompt / fixed schema , directive-kind-specific weight for RHS tokens , value_w for EOS
                if shift_token_weights is not None:
                    tok_weight = tok_weight * shift_token_weights[:, s:e].to(per_tok.dtype)

                weighted_mask = tok_weight * mask.to(tok_weight.dtype)
                loss_sum += (per_tok * weighted_mask).sum(dim=1)
                tok_cnt += weighted_mask.sum(dim=1)

            # Each example gets normalized by its weighted token count
            # Long targets do not automatically dominate short ones
            per_ex = loss_sum / tok_cnt.clamp(min=1.0)
 
            # Change the relative contribution to loss of each example
            if weights is not None:
                w = weights.to(device=device, dtype=per_ex.dtype)
                ce_loss = (per_ex * w).sum() / w.sum().clamp(min=1e-8)
            else:
                ce_loss = per_ex.mean()

            # contrastive loss : active only during training to make eval lighter
            cand_loss = torch.zeros((), device=device, dtype=torch.float32)
            if (
                model.training
                and self.candidate_loss_weight > 0.0
                and contrastive_sites is not None
                and self.candidate_sites_per_sample > 0
                and self.candidate_negatives_per_site > 0
            ):
                cand_loss = self._compute_candidate_loss(
                    model=model,
                    contrastive_sites=contrastive_sites,
                    kernel_names=kernel_names,
                    routing_start_idx=routing_start_idx,
                    sample_weights=weights,
                )

            # CE loss : predict the exact gold RHS tokens
            # Contrastive loss :  among plausible alternatives for the same site, prefer the gold RHS over nearby hard negatives
            loss = ce_loss + self.candidate_loss_weight * cand_loss
            return (loss, outputs) if return_outputs else loss
    
        finally:
            if hasattr(model, "clear_harp") and not model.training:
                model.clear_harp()



# ===============================================
# Stage Configs (Stage_1 / Stage_2)
# ===============================================

@dataclass
class StageRunConfig:
    name: str
    output_dir: str
    disable_harp: bool

    init_adapter_dir: str = ""
    init_harp_xattn_from: str = ""
    best_dir_name: str = "best_custom_stage1"

    value_loss_weight: float = 1.0

    lr_lora: float = 5e-5
    lr_embed: float = 5e-5
    lr_xattn: float = 0.0
    lr_gate: float = 0.0
    lr_ff: float = 0.0
    lr_gate_ff: float = 0.0

    epochs: int = 2
    max_steps: int = -1
    eval_steps: int = 100
    save_steps: int = 100


def clone_args(args):
    return argparse.Namespace(**vars(args))


def make_stage_args(base_args, cfg: StageRunConfig):
    a = clone_args(base_args)

    a.output_dir = cfg.output_dir
    a.best_dir_name = cfg.best_dir_name
    a.disable_harp = cfg.disable_harp

    a.init_adapter_dir = cfg.init_adapter_dir
    a.init_harp_xattn_from = cfg.init_harp_xattn_from
    a.resume_from_checkpoint = ""

    a.value_loss_weight = cfg.value_loss_weight

    a.lr_lora = cfg.lr_lora
    a.lr_embed = cfg.lr_embed
    a.lr_xattn = cfg.lr_xattn
    a.lr_gate = cfg.lr_gate
    a.lr_ff = cfg.lr_ff
    a.lr_gate_ff = cfg.lr_gate_ff

    a.epochs = cfg.epochs
    a.max_steps = cfg.max_steps
    a.eval_steps = cfg.eval_steps
    a.save_steps = cfg.save_steps

    return a


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_default_stage_arguments(args):
    # Stage 1: learn objective-conditioned RHS values only
    stage1 = StageRunConfig(
        name="stage1_goal_rhs_only_sft",
        output_dir=args.stage1_output_dir,
        best_dir_name="best_custom_stage1",
        disable_harp=True,

        value_loss_weight=1.0,

        lr_lora=args.lr_lora,
        lr_embed=args.lr_embed,
        lr_xattn=0.0,
        lr_gate=0.0,
        lr_ff=0.0,
        lr_gate_ff=0.0,

        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # Stage 2: keep same deterministic target-side format, enable HARP for RHS refinement
    stage2 = StageRunConfig(
        name="stage2_goal_harp_rhs_only",
        output_dir=args.stage2_output_dir,
        disable_harp=False,

        init_harp_xattn_from="",
        init_adapter_dir=os.path.join(args.stage1_output_dir, "best_custom_stage1"),
        best_dir_name="best_custom_stage2",

        value_loss_weight=1.0,

        lr_lora=0.0,
        lr_embed=0.0,
        lr_xattn=args.stage2_lr_xattn,
        lr_gate=args.stage2_lr_gate,
        lr_ff=args.stage2_lr_ff,
        lr_gate_ff=args.stage2_lr_gate_ff,

        epochs=args.stage2_epochs,
        max_steps=args.stage2_max_steps,
        eval_steps=args.stage2_eval_steps,
        save_steps=args.stage2_save_steps,
    )

    return stage1, stage2



# =================================
# Main Training
# =================================

# =============================================================================
# Unified target conditioning
# =============================================================================

DEVICE_MODES = mailohls_contract.DEVICE_MODES
ADAPTED_DEVICE_TOKEN = mailohls_contract.ADAPTED_DEVICE_TOKEN


@dataclass
class TargetAwareConfig:
    """Runtime policy shared by prompt construction and target selection."""

    device_mode: str = "known"
    adapt_device: str = ""
    budget_mode: str = "random"
    random_budgets_per_case: int = 16
    min_budget_frac: float = 0.10
    min_feasible_candidates: int = 3
    candidate_pool_per_objective: int = 24
    auto_frequency_fraction: float = 0.0
    min_auto_clock_count: int = 2
    strict_source_markers: bool = True
    seed: int = 123


TARGET_CFG = TargetAwareConfig()

def load_device_specs(path: str) -> None:
    """Extend the device-capacity registry from a small JSON file.

    Accepted shape:
      {"devices": {"part-name": {"BRAM_18K": ..., "DSP": ...,
                                  "FF": ..., "LUT": ...}}}
    The outer ``devices`` key may be omitted.
    """
    if not path:
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    specs = payload.get("devices", payload)
    if not isinstance(specs, dict) or not specs:
        raise ValueError("--device_specs_json must contain at least one device")

    for raw_name, raw_caps in specs.items():
        name = _norm_device(raw_name)
        if not name or not isinstance(raw_caps, dict):
            raise ValueError(f"Invalid device entry: {raw_name!r}")
        caps = {}
        for resource in RESOURCE_KEYS:
            value = raw_caps.get(resource)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: missing numeric {resource}")
            value = int(round(float(value)))
            if value <= 0:
                raise ValueError(f"{name}: {resource} must be positive")
            caps[resource] = value
        DEVICE_RESOURCES[name] = caps


def filter_rows_for_device_mode(rows: List[dict]) -> List[dict]:
    """Validate target metadata and select calibration rows in adapt mode."""
    checked = []
    for row in rows:
        device = _norm_device(row.get("device", row.get("Device", "")))
        if device not in DEVICE_RESOURCES:
            raise ValueError(
                f"No capacity specification for device {device!r}; "
                "provide --device_specs_json"
            )
        if TARGET_CFG.device_mode == "known" and device not in DEVICE_TOKEN_MAP:
            raise ValueError(
                f"Known-device training has no identity token for {device!r}"
            )
        for field in ("input", "target", "kernel_name"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Row is missing non-empty {field!r}")
        clock = _clock_of(row)
        if not math.isfinite(clock) or clock <= 0.0:
            raise ValueError(f"Invalid clock period in {row['kernel_name']}")
        for field in ("latency", "area"):
            value = float(row.get(field, 0.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Invalid {field} for {row['kernel_name']}: {value}"
                )
        for resource in RESOURCE_KEYS:
            field = UTIL_FIELD_BY_RESOURCE[resource]
            if field not in row:
                raise ValueError(
                    f"Missing {field} for {row['kernel_name']}"
                )
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Invalid {field} for {row['kernel_name']}: {value}"
                )
        if (
            TARGET_CFG.device_mode == "device_adapt"
            and device != TARGET_CFG.adapt_device
        ):
            continue
        checked.append(row)

    if TARGET_CFG.device_mode == "device_adapt" and not checked:
        raise ValueError(
            f"No rows found for --adapt_device={TARGET_CFG.adapt_device!r}"
        )
    return checked


def period_token_from_clock(clock_period: Any) -> str:
    return mailohls_contract.period_token_from_clock(clock_period)


def _clock_of(row: Mapping[str, Any]) -> float:
    value = row.get("clock_period", row.get("Clock_Period_nsec"))
    if value in (None, ""):
        raise ValueError(
            f"Row for {row.get('kernel_name', '<unknown>')} has no clock"
        )
    return _norm_clock(value)


def _available_resources(row: Mapping[str, Any]) -> Dict[str, int]:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device)
    if caps is None:
        raise ValueError(f"Unsupported device: {device!r}")
    result = {}
    for resource in RESOURCE_KEYS:
        raw = row.get(AVAIL_FIELD_BY_RESOURCE[resource])
        result[resource] = (
            int(round(float(raw)))
            if raw not in (None, "")
            else int(caps[resource])
        )
    return result


def target_prompt_fields(
    row: Optional[dict],
    device_token_dropout: float = 0.0,
) -> dict:
    return mailohls_contract.target_prompt_fields(
        row or {},
        device_mode=TARGET_CFG.device_mode,
        device_token_dropout=device_token_dropout,
    )


def build_prompt_sections(
    code: str,
    obj_mode: str,
    row: Optional[dict] = None,
    device_token_dropout: float = 0.0,
) -> Tuple[str, str, str, dict]:
    fields = target_prompt_fields(row, device_token_dropout)
    header, canonical_code, suffix = mailohls_contract.build_prompt_sections(
        code, obj_mode, fields
    )
    return header, canonical_code, suffix, fields


def build_prompt(
    code: str,
    obj_mode: str,
    row: Optional[dict] = None,
    device_token_dropout: float = 0.0,
) -> str:
    fields = target_prompt_fields(row, device_token_dropout)
    return mailohls_contract.build_prompt(code, obj_mode, fields)


def clock_target_text(row: Mapping[str, Any]) -> str:
    selected = row.get("selected_clock_period")
    if selected in (None, ""):
        selected = _clock_of(row)
    return (
        f"{CLOCK_ANCHOR_TOKEN}\n"
        f"selected_clock_period_ns = {_norm_clock(selected):g}\n"
    )


@dataclass(frozen=True, order=True)
class SharedResourceBudget:
    bram_frac: float
    dsp_frac: float
    ff_frac: float
    lut_frac: float

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(RESOURCE_KEYS, (
            self.bram_frac, self.dsp_frac, self.ff_frac, self.lut_frac
        )))


def _stable_seed(parts: Sequence[Any], seed: int) -> int:
    payload = repr((tuple(parts), int(seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _row_used_fraction(row: Mapping[str, Any], resource: str) -> float:
    value = float(row.get(UTIL_FIELD_BY_RESOURCE[resource], 0.0) or 0.0)
    return max(0.0, value / 100.0)


def _quantized_budget(values: Iterable[float]) -> SharedResourceBudget:
    clipped = [
        round(min(1.0, max(TARGET_CFG.min_budget_frac, float(value))), 2)
        for value in values
    ]
    return SharedResourceBudget(*clipped)


def sample_shared_budgets(
    case_key: Tuple[str, str],
    candidates: Sequence[dict],
    count: int,
    seed: int,
) -> List[SharedResourceBudget]:
    """Generate deterministic budgets shared across all clocks in a case."""
    rng = random.Random(_stable_seed(case_key, seed))
    budgets = {SharedResourceBudget(1.0, 1.0, 1.0, 1.0)}
    while len(budgets) < max(1, count):
        draw = rng.random()
        if draw < 0.15:
            values = [1.0] * 4
        elif draw < 0.35:
            scalar = rng.uniform(TARGET_CFG.min_budget_frac, 1.0)
            values = [scalar] * 4
        elif draw < 0.85:
            values = [
                TARGET_CFG.min_budget_frac
                + (1.0 - TARGET_CFG.min_budget_frac)
                * rng.betavariate(2.0, 1.5)
                for _ in RESOURCE_KEYS
            ]
        else:
            anchor = candidates[rng.randrange(len(candidates))]
            values = [
                _row_used_fraction(anchor, resource) + rng.uniform(0.01, 0.15)
                for resource in RESOURCE_KEYS
            ]
        budgets.add(_quantized_budget(values))
    return sorted(budgets)


def design_fits_shared_budget(
    row: Mapping[str, Any], budget: SharedResourceBudget
) -> bool:
    fractions = budget.as_dict()
    return all(
        _row_used_fraction(row, resource) <= fractions[resource] + 1e-9
        for resource in RESOURCE_KEYS
    )


def attach_shared_budget(
    row: Mapping[str, Any], budget: SharedResourceBudget
) -> dict:
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES[device]
    fractions = budget.as_dict()
    result = dict(row)
    for resource in RESOURCE_KEYS:
        result[AVAIL_FIELD_BY_RESOURCE[resource]] = int(
            round(caps[resource] * fractions[resource])
        )
        result[f"budget_frac_{resource.lower()}"] = fractions[resource]
    result["resource_budget_id"] = (
        f"B{budget.bram_frac:.2f}_D{budget.dsp_frac:.2f}_"
        f"F{budget.ff_frac:.2f}_L{budget.lut_frac:.2f}"
    )
    return result


def _compact_candidate_union(feasible: Sequence[dict]) -> List[dict]:
    """Retain an exact objective-extreme candidate superset before ranking."""
    valid = [
        row for row in feasible
        if float(row.get("latency", 0.0)) > 0.0
        and float(row.get("area", 0.0)) > 0.0
    ]
    if not valid:
        return []
    k = max(1, TARGET_CFG.candidate_pool_per_objective)

    def identity(row):
        return (
            int(row.get("_jsonl_idx", -1)),
            canonical_completion_key(row["input"], row["target"]),
        )

    latency = sorted(valid, key=lambda r: (float(r["latency"]), float(r["area"]), identity(r)))
    area = sorted(valid, key=lambda r: (float(r["area"]), float(r["latency"]), identity(r)))
    adp = sorted(
        valid,
        key=lambda r: (
            math.log2(max(float(r["latency"]), 1e-12))
            + math.log2(max(float(r["area"]), 1e-12)),
            identity(r),
        ),
    )
    keep = {identity(row): row for row in latency[:k] + area[:k] + adp[:k]}
    best_area = float("inf")
    for row in latency:
        value = float(row["area"])
        if value < best_area - 1e-12:
            keep[identity(row)] = row
            best_area = value
    return list(keep.values())


def augment_rows_with_random_resource_budgets(
    rows: List[dict],
    num_budgets_per_case: int,
    seed: int,
    min_feasible_candidates: int = 3,
) -> List[dict]:
    """Apply each budget to the full kernel/device multi-clock candidate pool."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row["kernel_name"], _norm_device(row.get("device", "")))].append(row)

    augmented, stats = [], Counter()
    for case_key, candidates in sorted(by_case.items()):
        for budget in sample_shared_budgets(
            case_key, candidates, num_budgets_per_case, seed
        ):
            feasible = [
                row for row in candidates
                if design_fits_shared_budget(row, budget)
            ]
            if len(feasible) < min_feasible_candidates:
                stats["rejected_small_candidate_sets"] += 1
                continue
            compact = _compact_candidate_union(feasible)
            if len(compact) < min_feasible_candidates:
                stats["rejected_after_compaction"] += 1
                continue
            augmented.extend(attach_shared_budget(row, budget) for row in compact)
            stats["kept_budgets"] += 1
            stats["candidate_rows"] += len(compact)
    print(
        f"[RANDOM-BUDGET] input={len(rows)} output={len(augmented)} "
        f"stats={dict(stats)}"
    )
    return augmented


def target_bucket_key(row: Mapping[str, Any]) -> tuple:
    mode = str(row.get("frequency_mode", "specified")).lower()
    period = "AUTO" if mode == "auto" else _clock_of(row)
    available = _available_resources(row)
    return (
        row["kernel_name"],
        _norm_device(row.get("device", row.get("Device", ""))),
        period,
        available["BRAM_18K"], available["DSP"],
        available["FF"], available["LUT"],
    )


def shared_budget_fraction(row: Mapping[str, Any]) -> float:
    device = _norm_device(row["device"])
    capacities = DEVICE_RESOURCES[device]
    available = _available_resources(row)
    return min(
        available[name] / capacities[name]
        for name in RESOURCE_KEYS
    )


def evenly_spaced_cases(cases, count):
    ordered = sorted(
        cases,
        key=lambda case: (
            shared_budget_fraction(case.row),
            target_bucket_key(case.row),
        ),
    )
    count = min(count, len(ordered))
    if count == 0:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    }
    return [ordered[index] for index in sorted(indices)]


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
    ranked = rank_goal_candidates(
        list(items), goal_mode, domination_penalty, max_dominated_gap
    )
    unique, seen = [], set()
    for record in ranked:
        row = record["row"]
        completion = canonical_completion_key(row["input"], row["target"])
        key = (_clock_of(row) if frequency_mode == "auto" else None, completion)
        if key in seen:
            continue
        seen.add(key)
        record["score"] = float(goal_sort_key(record, goal_mode, 0.0)[0])
        unique.append(record)

    chosen = unique[: min(top_k, len(unique))]
    if not chosen:
        return [], {}
    scores = [record["score"] for record in chosen]
    selected = []
    hard_negatives = build_local_hard_negative_bank(
        [{"row": r["row"], "score": r["score"]} for r in unique],
        hard_neg_top_k=max(6, top_k),
    )
    hard_negatives = {
        lhs: sorted(values, key=_rhs_sort_key)
        for lhs, values in hard_negatives.items()
    }
    for rank, record in enumerate(chosen):
        out = dict(record["row"])
        out.update({
            "obj_mode": goal_mode,
            "frequency_mode": frequency_mode,
            "selected_clock_period": _clock_of(out),
            "_score": record["score"],
            "_rank_within_kernel": rank,
            "_sample_weight": score_gap_weight(
                record["score"], min(scores), max(scores),
                score_weight_min, score_weight_power,
            ),
            "_local_hard_negatives": hard_negatives,
        })
        selected.append(out)
    return selected, {
        "selected": len(selected),
        "candidate_count": len(items),
        "frequency_mode": frequency_mode,
        "selected_clocks": [row["selected_clock_period"] for row in selected],
    }


def select_goal_rows(
    rows: List[dict],
    goal_mode: str,
    top_k: int,
    domination_penalty: float,
    max_dominated_gap: float,
    score_weight_min: float = 0.6,
    score_weight_power: float = 1.0,
):
    """Select one deterministic optimum per prompt by default.

    ``top_k > 1`` remains an explicit ablation because it creates multiple
    completions for an otherwise identical prompt.
    """
    specified, automatic = defaultdict(list), defaultdict(list)
    for row in rows:
        avail = _available_resources(row)
        base_key = (
            row["kernel_name"], _norm_device(row.get("device", "")),
            avail["BRAM_18K"], avail["DSP"], avail["FF"], avail["LUT"],
        )
        specified[(base_key[0], base_key[1], _clock_of(row), *base_key[2:])].append(row)
        automatic[base_key].append(row)

    selected, metadata = [], {}
    for key, items in sorted(specified.items()):
        chosen, info = _rank_and_select_case(
            items, goal_mode, top_k, domination_penalty, max_dominated_gap,
            score_weight_min, score_weight_power, "specified"
        )
        selected.extend(chosen)
        metadata[f"specified::{key!r}"] = info

    auto_rows = []
    for key, items in sorted(automatic.items()):
        clocks = sorted({_clock_of(row) for row in items})
        if len(clocks) < TARGET_CFG.min_auto_clock_count:
            continue
        chosen, info = _rank_and_select_case(
            items, goal_mode, top_k, domination_penalty, max_dominated_gap,
            score_weight_min, score_weight_power, "auto"
        )
        for row in chosen:
            # The model may select only among clocks represented by measured
            # candidates for this exact kernel/device/resource-budget case.
            row["available_clock_periods"] = clocks
        auto_rows.extend(chosen)
        info["available_clocks"] = clocks
        metadata[f"auto::{key!r}"] = info

    if TARGET_CFG.auto_frequency_fraction > 0.0 and auto_rows:
        count = round(
            len(selected) * TARGET_CFG.auto_frequency_fraction
            / (1.0 - TARGET_CFG.auto_frequency_fraction)
        )
        rng = random.Random(_stable_seed(("auto", goal_mode), TARGET_CFG.seed))
        rng.shuffle(auto_rows)
        selected.extend(auto_rows[:count])
    rng = random.Random(_stable_seed(("selected", goal_mode), TARGET_CFG.seed))
    rng.shuffle(selected)
    print("[CLOCK-MODE]", Counter(r.get("frequency_mode") for r in selected))
    return selected, metadata


class SFTDataset(Dataset):
    """Build supervised clock/directive samples while preserving target context."""

    def __init__(
        self,
        rows: List[dict],
        tok,
        max_length: int,
        value_loss_weight: float = 1.0,
        candidate_sites_per_sample: int = 0,
        candidate_negatives_per_site: int = 0,
        device_token_dropout: float = 0.0,
        supervise_eos: bool = False,
        input_only_special_ids: Optional[Iterable[int]] = None,
        kind_loss_weights: Optional[Dict[str, float]] = None,
    ):
        self.samples, self.lengths = [], []
        truncated = 0
        kind_weights = dict(kind_loss_weights or {})
        source_token_ids = set(tok.convert_tokens_to_ids(SOURCE_PLACEHOLDER_TOKENS))

        for example in rows:
            header, kernel, suffix, fields = build_prompt_sections(
                example["input"], example["obj_mode"], example,
                device_token_dropout,
            )
            header_ids = tok(header, add_special_tokens=False)["input_ids"]
            code_ids = tok(kernel, add_special_tokens=False)["input_ids"]
            suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]

            target_core = reorder_target_by_source_order(
                example["input"], example["target"].strip()
            )
            directives = build_deterministic_rhs_pack(
                example["input"], target_core, tok,
                value_w=value_loss_weight,
                kind_loss_weights=kind_weights,
                supervise_eos=supervise_eos,
            )
            clock = build_clock_pack(example, tok, value_w=value_loss_weight)
            target_ids = clock.input_ids + directives.input_ids
            target_labels = clock.labels + directives.labels
            target_weights = clock.token_weights + directives.token_weights
            target_xmask = clock.xattn_target_mask + directives.xattn_target_mask

            input_only_special_id_set = set(input_only_special_ids or ())
            bad = {
                int(label) for label in target_labels
                if int(label) != -100
            }.intersection(input_only_special_id_set)
            if bad:
                raise RuntimeError(
                    f"Input-only special tokens are supervised: {sorted(bad)}"
                )

            prompt_budget = max_length - len(target_ids)
            fixed_prompt = len(header_ids) + len(suffix_ids)
            if prompt_budget <= fixed_prompt:
                raise ValueError(
                    f"max_length={max_length} cannot preserve target conditioning "
                    f"for {example['kernel_name']}"
                )
            code_budget = prompt_budget - fixed_prompt
            kept_code = code_ids
            if len(code_ids) > code_budget:
                truncated += 1
                head = code_budget // 2
                kept_code = code_ids[:head] + code_ids[-(code_budget - head):]
                before = sum(token in source_token_ids for token in code_ids)
                after = sum(token in source_token_ids for token in kept_code)
                if TARGET_CFG.strict_source_markers and after != before:
                    raise ValueError(
                        f"Context truncation drops {before - after} source markers "
                        f"for {example['kernel_name']}; increase --max_length or use "
                        "--allow_source_marker_truncation only for debugging"
                    )

            prompt_ids = header_ids + kept_code + suffix_ids
            required = [
                GOALS[example["obj_mode"]]["token"],
                fields["device_token"], fields["period_token"],
            ]
            for token in required:
                token_id = tok.convert_tokens_to_ids(token)
                if token_id not in prompt_ids:
                    raise ValueError(f"Required conditioning token {token} was lost")

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_labels
            token_weights = [0.0] * len(prompt_ids) + target_weights
            full_xmask = [0] * len(prompt_ids) + target_xmask
            contrastive_sites = build_contrastive_sites_from_sample(
                example["input"], target_core, prompt_ids + clock.input_ids, tok,
                max_length,
                example.get("_local_hard_negatives", {}),
                candidate_sites_per_sample,
                candidate_negatives_per_site,
                kind_weights,
            )
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "token_weights": torch.tensor(token_weights, dtype=torch.float32),
                "xattn_apply_mask": torch.tensor(full_xmask[1:] + [0], dtype=torch.float32),
                "sample_weight": torch.tensor(float(example.get("_sample_weight", 1.0))),
                "kernel_name": example["kernel_name"],
                "routing_start_idx": torch.tensor(len(prompt_ids), dtype=torch.long),
                "contrastive_sites": contrastive_sites,
            })
            self.lengths.append(len(input_ids))
        print(f"[DATASET] samples={len(self.samples)} code_truncated={truncated}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def build_selection_cases(
    val_rows: List[dict],
    goal_mode: str,
    max_kernels: int = 0,
    cases_per_kernel_device: int = 2,
    min_coverage: float = 0.85,
    min_supervised_sites: int = 4,
) -> List[SelectionCase]:
    """Use specified-clock validation until clock decoding is evaluated jointly."""
    by_case = defaultdict(list)
    for row in val_rows:
        if (
            row["obj_mode"] == goal_mode
            and row.get("frequency_mode", "specified") == "specified"
        ):
            by_case[target_bucket_key(row)].append(row)
    candidates_by_kernel_device = defaultdict(list)
    for case_key in sorted(by_case):
        best = min(
            by_case[case_key],
            key=lambda row: (
                int(row.get("_rank_within_kernel", 10**9)),
                float(row.get("_score", 10**9)),
            ),
        )
        try:
            target, meta = build_partial_deterministic_target_text(
                best["input"], best["target"], min_supervised_sites
            )
        except ValueError:
            continue
        if meta["coverage"] < min_coverage:
            continue
        case = SelectionCase(
            kernel_name=best["kernel_name"],
            obj_mode=goal_mode,
            source_text=best["input"],
            reference_target=target,
            row=best,
        )
        candidates_by_kernel_device[
            (case.kernel_name, _norm_device(best["device"]))
        ].append(case)

    selected = []
    kernel_names = sorted({
        kernel for kernel, _ in candidates_by_kernel_device
    })
    if max_kernels > 0:
        kernel_names = kernel_names[:max_kernels]
    selected_kernel_names = set(kernel_names)
    for kernel_device in sorted(candidates_by_kernel_device):
        kernel, _ = kernel_device
        if kernel not in selected_kernel_names:
            continue
        selected.extend(
            evenly_spaced_cases(
                candidates_by_kernel_device[kernel_device],
                cases_per_kernel_device,
            )
        )
    return selected


def configure_target_policy(args) -> None:
    load_device_specs(args.device_specs_json)
    TARGET_CFG.device_mode = args.device_mode
    TARGET_CFG.adapt_device = _norm_device(args.adapt_device)
    TARGET_CFG.budget_mode = args.resource_budget_mode
    TARGET_CFG.random_budgets_per_case = args.random_budgets_per_case
    TARGET_CFG.min_budget_frac = args.random_budget_min_frac
    TARGET_CFG.min_feasible_candidates = args.min_feasible_candidates_per_budget
    TARGET_CFG.candidate_pool_per_objective = args.candidate_pool_per_objective
    TARGET_CFG.auto_frequency_fraction = args.auto_frequency_fraction
    TARGET_CFG.min_auto_clock_count = args.min_auto_clock_count
    TARGET_CFG.strict_source_markers = not args.allow_source_marker_truncation
    TARGET_CFG.seed = args.seed

    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.selection_num_val_kernels < 0:
        raise ValueError("--selection_num_val_kernels must be >= 0")
    if args.selection_cases_per_kernel_device < 1:
        raise ValueError("--selection_cases_per_kernel_device must be >= 1")
    if args.selection_eval_steps < 1:
        raise ValueError("--selection_eval_steps must be >= 1")
    if args.selection_candidate_batch_size < 1:
        raise ValueError("--selection_candidate_batch_size must be >= 1")
    if args.top_k > 1:
        print("[WARN] top_k > 1 creates conflicting completions for identical prompts")
    if not 0.0 < args.random_budget_min_frac <= 1.0:
        raise ValueError("--random_budget_min_frac must be in (0, 1]")
    if not 0.0 <= args.auto_frequency_fraction < 1.0:
        raise ValueError("--auto_frequency_fraction must be in [0, 1)")
    if args.auto_frequency_fraction > 0.0 and args.min_auto_clock_count < 2:
        raise ValueError("automatic-clock training requires at least two clocks")
    if args.device_mode == "known" and args.device_token_dropout != 0.0:
        raise ValueError("known mode requires --device_token_dropout 0")
    if (
        args.device_mode == "resource_dropout_ablation"
        and not 0.0 < args.device_token_dropout < 1.0
    ):
        raise ValueError(
            "resource_dropout_ablation requires device-token dropout in (0, 1)"
        )
    if args.device_mode == "device_adapt":
        if not TARGET_CFG.adapt_device:
            raise ValueError("device_adapt requires --adapt_device")
        if TARGET_CFG.adapt_device not in DEVICE_RESOURCES:
            raise ValueError("adapted device is absent from --device_specs_json")
        if args.run_mode != "single":
            raise ValueError("device_adapt currently requires --run_mode single")
        if not args.init_adapter_dir:
            raise ValueError("device_adapt requires --init_adapter_dir")
        if args.lr_embed != 0.0:
            raise ValueError("device_adapt keeps embeddings frozen; use --lr_embed 0")


def run_single_training(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        print("[CUDA] using:", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)
    dump_root = os.path.join(args.output_dir, "selected_debug")
    os.makedirs(dump_root, exist_ok=True)

    rows = filter_rows_for_device_mode(load_rows(args.dataset))
    print(f"[INFO] Loaded {len(rows)} raw rows from {args.dataset}")
    fam_counts = Counter(r["_family"] for r in rows)
    print("[INFO] Raw rows per family (top 15):", fam_counts.most_common(15))

    if args.split_json:
        split_spec = load_split_spec(args.split_json)
        raw_train_rows, raw_val_rows, raw_test_rows = apply_split_spec(rows, split_spec)
        print(f"[INFO] Loaded split from {args.split_json}")
    elif args.split_mode == "family":
        val_fams = {normalize_name(x) for x in args.val_families.split(";") if x.strip()}
        test_fams = {normalize_name(x) for x in args.test_families.split(";") if x.strip()}
        print("[INFO] val_families:", sorted(val_fams))
        print("[INFO] test_families:", sorted(test_fams))
        raw_train_rows, raw_val_rows, raw_test_rows = split_by_family(rows, val_fams, test_fams)
        assert_disjoint_nonempty_kernel_splits(
            raw_train_rows, raw_val_rows, raw_test_rows
        )
    else:
        raw_train_rows, raw_val_rows, raw_test_rows = split_rows_random_design(
            rows,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            stratify_by_kernel=args.stratify_by_kernel,
        )
        print(f"[INFO] random design-point split with val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, split_seed={args.split_seed}, stratify_by_kernel={args.stratify_by_kernel}")

    print(f"[INFO] Raw split sizes: train={len(raw_train_rows)} val={len(raw_val_rows)} test={len(raw_test_rows)}")
    split_payload = {
        name: sorted(int(row["_jsonl_idx"]) for row in split_rows)
        for name, split_rows in (
            ("train", raw_train_rows), ("val", raw_val_rows), ("test", raw_test_rows)
        )
    }
    split_sha256 = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    eval_seed = args.seed + 10_000

    if args.save_split_json:
        save_split_spec(args.save_split_json, raw_train_rows, raw_val_rows, raw_test_rows)
        print(f"[INFO] Saved split spec -> {args.save_split_json}")

    if args.resource_budget_mode == "fixed":
        fractions = parse_resource_budget_fracs(
            args.resource_budget_fracs
        )
        raw_train_rows = augment_rows_with_resource_budgets(
            raw_train_rows, fractions
        )
        raw_val_rows = augment_rows_with_resource_budgets(
            raw_val_rows, fractions
        )
        raw_test_rows = augment_rows_with_resource_budgets(
            raw_test_rows, fractions
        )

    elif args.resource_budget_mode == "random":
        raw_train_rows = augment_rows_with_random_resource_budgets(
            raw_train_rows,
            num_budgets_per_case=args.random_budgets_per_case,
            seed=args.seed,
            min_feasible_candidates=(
                args.min_feasible_candidates_per_budget
            ),
        )
        raw_val_rows = augment_rows_with_random_resource_budgets(
            raw_val_rows,
            num_budgets_per_case=args.random_budgets_per_case,
            seed=eval_seed,
            min_feasible_candidates=3,
        )

        raw_test_rows = augment_rows_with_random_resource_budgets(
            raw_test_rows,
            num_budgets_per_case=args.random_budgets_per_case,
            seed=eval_seed + 1,
            min_feasible_candidates=3,
        )

    objectives = mailohls_contract.resolve_objectives(args.objective)
    goal_key = "all_objectives" if args.objective == "ALL" else GOALS[args.objective]["tag"]

    def select_objectives(source_rows):
        combined, information = [], {}
        for objective in objectives:
            chosen, details = select_goal_rows(
                source_rows,
                goal_mode=objective,
                top_k=args.top_k,
                domination_penalty=args.goal_domination_penalty,
                max_dominated_gap=args.goal_max_dominated_gap,
                score_weight_min=args.score_weight_min,
                score_weight_power=args.score_weight_power,
            )
            combined.extend(chosen)
            information[objective] = details
        random.Random(
            _stable_seed(("objective_mix", goal_key), args.seed)
        ).shuffle(combined)
        return combined, information

    train_rows, train_goal_info = select_objectives(raw_train_rows)
    val_rows, val_goal_info = select_objectives(raw_val_rows)
    test_rows, test_goal_info = select_objectives(raw_test_rows)
    directive_loss_weights = compute_directive_loss_weights(
        train_rows, args.directive_loss_weighting
    )

    print(f"[INFO] Selected split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(
        f"[LOSS] directive weighting={args.directive_loss_weighting} "
        f"weights={directive_loss_weights or 'uniform'}"
    )

    dump_jsonl(os.path.join(dump_root, f"train_selected_{goal_key}.jsonl"), train_rows)
    dump_json(os.path.join(dump_root, f"train_selected_{goal_key}.indices.json"), train_goal_info)
    if val_rows:
        dump_jsonl(os.path.join(dump_root, f"val_selected_{goal_key}.jsonl"), val_rows)
        dump_json(os.path.join(dump_root, f"val_selected_{goal_key}.indices.json"), val_goal_info)
    if test_rows:
        dump_jsonl(os.path.join(dump_root, f"test_selected_{goal_key}.jsonl"), test_rows)
        dump_json(os.path.join(dump_root, f"test_selected_{goal_key}.indices.json"), test_goal_info)

    selection_cases = []
    for objective in objectives:
        selection_cases.extend(build_selection_cases(
            val_rows,
            goal_mode=objective,
            max_kernels=args.selection_num_val_kernels,
            cases_per_kernel_device=args.selection_cases_per_kernel_device,
            min_coverage=args.min_site_coverage,
            min_supervised_sites=args.min_supervised_sites,
        ))
    selection_kernel_count = len({case.kernel_name for case in selection_cases})
    selection_kernel_device_count = len({
        (case.kernel_name, _norm_device(case.row["device"]))
        for case in selection_cases
    })
    print(
        f"[INFO] Built {len(selection_cases)} validation selection cases from "
        f"{selection_kernel_count} distinct kernels and "
        f"{selection_kernel_device_count} kernel/device groups"
    )
    directive_domain_registry: Dict[str, Dict[str, List[str]]] = {}
    if selection_cases:
        if not args.directive_domain_registry_json:
            raise ValueError(
                "Production validation requires --directive_domain_registry_json; "
                "legal RHS domains must not be inferred from training rows."
            )
        directive_domain_registry = load_directive_domain_registry(
            args.directive_domain_registry_json
        )
        for case in selection_cases:
            for _, lhs in extract_ordered_lhs_plan(case.source_text):
                get_rhs_candidates_for_lhs(
                    case.kernel_name, lhs, directive_domain_registry
                )
        print(
            f"[DOMAINS] validated {len(directive_domain_registry)} kernel registries"
        )

    if args.disable_harp:
        mem_bank = {}
        print("[INFO] Structural memory disabled -> skipping memory bank loading")
    else:
        mem_bank, inferred_mem_dim = load_memory_bank(
            args.memory_dir,
            expected_mem_dim=None if args.mem_dim <= 0 else args.mem_dim,
            expected_max_slots=args.max_slots,
            require_pragma_free_memory=args.require_pragma_free_memory,
        )

        if inferred_mem_dim is not None and args.mem_dim != inferred_mem_dim:
            print(f"[INFO] Overriding --mem_dim {args.mem_dim} -> {inferred_mem_dim} from memory bank")
            args.mem_dim = inferred_mem_dim

        print(f"[INFO] Memory bank keys: {len(mem_bank)}")
        required_kernels = {
            row["kernel_name"] for row in train_rows + val_rows + test_rows
        }
        missing_memory = sorted(
            kernel
            for kernel in required_kernels
            if (
                kernel not in mem_bank
                and normalize_kname(kernel) not in mem_bank
            )
        )
        if missing_memory and not args.allow_missing_harp_memory:
            raise ValueError(
                "Missing HARP/GNN memory for kernels: "
                + ", ".join(missing_memory[:20])
            )
        if missing_memory:
            print(
                f"[WARN] {len(missing_memory)} kernels will receive zero HARP memory"
            )

        manifest_path = os.path.join(args.memory_dir, "memory_manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError(f"Stage 2 requires memory manifest: {manifest_path}")
        memory_manifest_sha256 = _file_sha256(Path(manifest_path))
        structural_config = {
            "schema": "mailohls-structural-config-v1",
            "mem_dim": args.mem_dim,
            "max_slots": args.max_slots,
            "every_n_layers": args.every_n_layers,
            "xattn_heads": args.xattn_heads,
            "xattn_dim_head": args.xattn_dim_head,
            "xattn_ff_mult": args.xattn_ff_mult,
            "xattn_enable_ff": False,
            "xattn_placement": "post_self_attn_pre_mlp",
            "xattn_gate_init": 0.0,
            "memory_manifest_sha256": memory_manifest_sha256,
        }

    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
    )

    special_token_strings = (
        [goal["token"] for goal in GOALS.values()]
        + TARGET_PLATFORM_TOKENS
        + SOURCE_PLACEHOLDER_TOKENS
        + TARGET_PLACEHOLDER_TOKENS
    )
    tok.add_special_tokens({"additional_special_tokens": special_token_strings})

    special_ids = sorted(
        set(tok.convert_tokens_to_ids(special_token_strings))
    )
    if any(token_id < 0 for token_id in special_ids):
        raise RuntimeError("A MailoHLS special token was not added correctly.")

    training_contract = {
        "schema": "mailohls-training-contract-v1",
        "stage": "stage1" if args.disable_harp else "stage2",
        "git_commit": current_git_commit(),
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer": args.model,
        "dataset_sha256": _file_sha256(Path(args.dataset)),
        "split_sha256": split_sha256,
        "prompt_schema_version": mailohls_contract.PROMPT_SCHEMA_VERSION,
        "objective": args.objective,
        "seed": args.seed,
        "tokenizer_size": len(tok),
        "special_tokens": special_token_strings,
        "special_token_ids": special_ids,
        "supervise_eos": args.supervise_eos,
        "directive_domain_registry_sha256": (
            _file_sha256(Path(args.directive_domain_registry_json))
            if args.directive_domain_registry_json else None
        ),
        "directive_loss_weighting": args.directive_loss_weighting,
        "directive_loss_weights": directive_loss_weights,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "device_mode": args.device_mode,
        "device_token_dropout": args.device_token_dropout,
        "resource_budget_mode": args.resource_budget_mode,
        "resource_budget_fracs": args.resource_budget_fracs,
        "random_budgets_per_case": args.random_budgets_per_case,
        "random_budget_min_frac": args.random_budget_min_frac,
        "min_feasible_candidates_per_budget": (
            args.min_feasible_candidates_per_budget
        ),
        "candidate_pool_per_objective": args.candidate_pool_per_objective,
        "auto_frequency_fraction": args.auto_frequency_fraction,
        "min_auto_clock_count": args.min_auto_clock_count,
        "goal_domination_penalty": args.goal_domination_penalty,
        "goal_max_dominated_gap": args.goal_max_dominated_gap,
        "min_supervised_sites": args.min_supervised_sites,
        "min_site_coverage": args.min_site_coverage,
        "score_weight_min": args.score_weight_min,
        "score_weight_power": args.score_weight_power,
        "selection_num_val_kernels": args.selection_num_val_kernels,
        "selection_cases_per_kernel_device": (
            args.selection_cases_per_kernel_device
        ),
        "selection_eval_steps": args.selection_eval_steps,
        "selection_candidate_batch_size": (
            args.selection_candidate_batch_size if args.disable_harp else 1
        ),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
    }
    if not args.disable_harp:
        training_contract["structural"] = structural_config
    resume_ckpt = os.path.abspath(args.resume_from_checkpoint) if args.resume_from_checkpoint else ""
    init_adapter_dir = os.path.abspath(args.init_adapter_dir) if args.init_adapter_dir else ""

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    native_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    compute_dtype = torch.bfloat16 if native_bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )

    base.resize_token_embeddings(len(tok))
    input_weight = base.get_input_embeddings().weight
    output_weight = base.get_output_embeddings().weight
    weights_tied = input_weight.data_ptr() == output_weight.data_ptr()

    # MailoHLS special tokens condition the prompt. They are deliberately not
    # prediction targets, so only their input-embedding rows are trainable.
    trainable_token_spec = special_ids
    trainable_token_modules = ["embed_tokens"]

    training_contract["embedding_weights_tied"] = weights_tied
    training_contract["trainable_token_modules"] = trainable_token_modules
    training_contract["special_token_role"] = "input_context_only"
    dump_json(
        os.path.join(args.output_dir, "training_contract.json"),
        training_contract,
    )
    if not args.disable_harp and init_adapter_dir:
        require_compatible_stage1_contract(init_adapter_dir, training_contract)

    print(f"[MODEL] weights_tied={weights_tied}")

    base.config.use_cache = False

    gc_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices=trainable_token_spec,
    )

    if resume_ckpt and os.path.isdir(resume_ckpt):
        model = PeftModel.from_pretrained(
            base,
            resume_ckpt,
            is_trainable=True,
        )
        print(f"[INIT] Loaded PEFT adapter from resume checkpoint: {resume_ckpt}")

    elif args.device_mode == "device_adapt":
        if not os.path.isdir(init_adapter_dir):
            raise FileNotFoundError(
                f"Base MailoHLS adapter not found: {init_adapter_dir}"
            )
        # Keep the validated MailoHLS adapter immutable and train a second,
        # device-specific residual adapter.  At step zero the new adapter is
        # neutral, so adaptation starts from the known-device model.
        model = PeftModel.from_pretrained(
            base,
            init_adapter_dir,
            adapter_name="mailohls_base",
            is_trainable=False,
        )
        model.add_adapter("device_adapt", lora_cfg)
        try:
            model.base_model.set_adapter(["mailohls_base", "device_adapt"])
        except Exception as exc:
            raise RuntimeError(
                "This PEFT version cannot compose the frozen MailoHLS adapter "
                "with a trainable device adapter"
            ) from exc

        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_("device_adapt" in name)
        print(
            "[INIT] Frozen mailohls_base + trainable device_adapt "
            f"for {TARGET_CFG.adapt_device}"
        )

    elif init_adapter_dir and os.path.isdir(init_adapter_dir):
        model = PeftModel.from_pretrained(
            base,
            init_adapter_dir,
            is_trainable=True,
        )
        print(f"[INIT] Loaded adapter from: {init_adapter_dir}")

    else:
        model = get_peft_model(base, lora_cfg)
        print("[INIT] Created fresh LoRA adapter")

    token_delta_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name and parameter.requires_grad
    )
    expected_token_delta_params = (
        len(special_ids) * int(base.config.hidden_size)
    )
    if token_delta_params != expected_token_delta_params:
        raise RuntimeError(
            "Special-token trainable parameter mismatch: "
            f"actual={token_delta_params}, expected={expected_token_delta_params}, "
            f"weights_tied={weights_tied}"
        )
    print(
        "[MODEL] special_token_trainable_params="
        f"{token_delta_params}"
    )

    if not args.disable_harp:
        if args.lr_ff != 0 or args.lr_gate_ff != 0:
            raise ValueError(
                "The post-self-attention/pre-MLP structural branch requires "
                "--lr_ff 0 and --lr_gate_ff 0"
            )
        extend_instance(model, HARPLMMixin)
        decoder_layers_attr_name = infer_decoder_layers_attr_name(model)
        model.set_decoder_layers_attr_name(decoder_layers_attr_name)

        placeholder_token_ids = tok.convert_tokens_to_ids(TARGET_PLACEHOLDER_TOKENS)
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer LM hidden size from model.config")

        model.init_structural_cross_attention(
            placeholder_token_ids=placeholder_token_ids,
            lang_hidden_size=hidden_size,
            mem_hidden_size=args.mem_dim,
            cross_attn_every_n_layers=args.every_n_layers,
            xattn_heads=args.xattn_heads,
            xattn_dim_head=args.xattn_dim_head,
            only_attend_immediate_memory=True,
            mask_mode="segment",
        )

        structural_config["selected_xattn_layers_1based"] = list(
            model.structural_xattn_layer_indices
        )

        print(f"[HARP-XATTN] decoder_layers_attr_name={decoder_layers_attr_name}")
        print(f"[HARP-XATTN] inserted gated xattn every {args.every_n_layers} decoder layers")
        move_harp_modules_to_model_device(model)
    else:
        print("[HARP-XATTN] disabled for Stage 1 format-only training")

    if resume_ckpt and os.path.isdir(resume_ckpt):
        resume_harp_xattn = os.path.join(resume_ckpt, "harp_xattn.pt")
        load_partial_harp_xattn(model, resume_harp_xattn, tag="HARP-RESUME")
    elif args.init_harp_xattn_from:
        load_partial_harp_xattn(model, args.init_harp_xattn_from, tag="HARP-INIT")

    model.print_trainable_parameters()

    inp = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight
    print("tied =", inp.data_ptr() == out.data_ptr())

    embedding_ids = {id(parameter) for parameter in model.get_input_embeddings().parameters()}
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        embedding_ids.update(id(parameter) for parameter in output_embeddings.parameters())
    frozen_zero_lr = 0
    for name, parameter in model.named_parameters():
        lr = None
        if id(parameter) in embedding_ids:
            lr = args.lr_embed
        elif "lora_" in name:
            lr = args.lr_lora
        elif name.endswith("attn_gate"):
            lr = args.lr_gate
        elif name.endswith("ff_gate"):
            lr = args.lr_gate_ff
        elif "gated_cross_attn_layer.attn." in name:
            lr = args.lr_xattn
        elif "gated_cross_attn_layer.ff." in name:
            lr = args.lr_ff
        if lr is not None and lr <= 0 and parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen_zero_lr += parameter.numel()
    print(f"[OPT] froze {frozen_zero_lr:,} parameters assigned to zero-LR groups")


    effective_candidate_sites = (
        args.candidate_sites_per_sample
        if args.candidate_loss_weight > 0.0 else 0
    )
    effective_candidate_negatives = (
        args.candidate_negatives_per_site
        if args.candidate_loss_weight > 0.0 else 0
    )
    if args.candidate_loss_weight == 0.0:
        print("[DATASET] candidate loss disabled; skipping local hard negatives")

    train_ds = SFTDataset(
        train_rows,
        tok,
        args.max_length,
        value_loss_weight=args.value_loss_weight,
        candidate_sites_per_sample=effective_candidate_sites,
        candidate_negatives_per_site=effective_candidate_negatives,
        device_token_dropout=args.device_token_dropout,
        supervise_eos=args.supervise_eos,
        input_only_special_ids=special_ids,
        kind_loss_weights=directive_loss_weights,
    )

    special_id_set = set(special_ids)
    for sample in train_ds.samples:
        supervised = {
            int(token_id)
            for token_id, label in zip(sample["input_ids"], sample["labels"])
            if int(label) != -100
        }
        leaked = special_id_set.intersection(supervised)
        if leaked:
            raise RuntimeError(
                "RHS-only supervision leaked special-token ids: "
                f"{sorted(leaked)}"
            )

    val_ds = SFTDataset(
        val_rows,
        tok,
        args.max_length,
        value_loss_weight=args.value_loss_weight,
        candidate_sites_per_sample=effective_candidate_sites,
        candidate_negatives_per_site=effective_candidate_negatives,
        device_token_dropout=0.0,
        supervise_eos=args.supervise_eos,
        input_only_special_ids=special_ids,
        kind_loss_weights=directive_loss_weights,
    ) if val_rows else None

    collator = PadCollator(tok)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.batch_size))
    total_steps = int(steps_per_epoch * args.epochs / max(1, args.grad_accum))
    effective_total_steps = args.max_steps if args.max_steps > 0 else total_steps
    warmup_steps = int(0.03 * effective_total_steps)

    bf16_ok = native_bf16
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=max(args.lr_lora, args.lr_xattn, args.lr_gate, args.lr_embed),
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        tf32=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        save_total_limit=6,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],   # <- add this
    )


    if not args.disable_harp:
        if not args.initial_state_reference:
            raise ValueError("Stage 2 requires --initial_state_reference")
        initial_manifest = verify_and_save_initial_harp_state(
            model, args.initial_state_reference, args.output_dir
        )
        structural_config["initial_harp_state_sha256"] = initial_manifest["combined_sha256"]
        dump_json(
            os.path.join(args.output_dir, "training_contract.json"),
            training_contract,
        )

    trainer = LengthGroupedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        group_by_length=args.group_by_length,
        mem_bank=mem_bank,
        mem_dim=args.mem_dim,
        max_slots=args.max_slots,
        lr_lora=args.lr_lora,
        lr_xattn=args.lr_xattn,
        lr_embed=args.lr_embed,
        lr_gate=args.lr_gate,
        lr_ff=args.lr_ff,
        lr_gate_ff=args.lr_gate_ff,
        loss_chunk_t=args.loss_chunk_t,
        candidate_loss_weight=args.candidate_loss_weight,
        candidate_sites_per_sample=effective_candidate_sites,
        candidate_negatives_per_site=effective_candidate_negatives,
        candidate_max_prefix_tokens=args.candidate_max_prefix_tokens,
        candidate_keep_head_tokens=args.candidate_keep_head_tokens,
    )

    trainer.add_callback(
        SaveMailoHLSCheckpointCallback(
            tokenizer=tok,
            training_contract=training_contract,
        )
    )

    if selection_cases:
        trainer.add_callback(
            StageValSelectionCallback(
                tokenizer=tok,
                selection_cases=selection_cases,
                directive_domain_registry=directive_domain_registry,
                output_dir=args.output_dir,
                max_prompt_tokens=args.max_length,
                candidate_score_reduction="mean",
                best_dir_name=args.best_dir_name,
                mem_bank=mem_bank,
                mem_dim=args.mem_dim,
                max_slots=args.max_slots,
                training_contract=training_contract,
                selection_eval_steps=args.selection_eval_steps,
                candidate_batch_size=(
                    args.selection_candidate_batch_size
                    if args.disable_harp else 1
                ),
            )
        )

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.")
        trainer.train()

    if val_ds is not None:
        trainer.evaluate(metric_key_prefix="final_eval")

    best_dir = os.path.join(args.output_dir, args.best_dir_name)
    if not os.path.isdir(best_dir):
        save_mailohls_adapter(model, tok, best_dir, training_contract)
        if not args.disable_harp:
            torch.save(get_harp_xattn_state_dict(model), os.path.join(best_dir, "harp_xattn.pt"))

    save_mailohls_adapter(model, tok, args.output_dir, training_contract)

    if not args.disable_harp:
        torch.save(
            get_harp_xattn_state_dict(model),
            os.path.join(args.output_dir, "harp_xattn.pt")
        )
        print(f"[DONE] Saved LoRA + HARP xattn adapters to: {args.output_dir}")
    else:
        print(f"[DONE] Saved LoRA adapter to: {args.output_dir}")

    if args.device_mode == "device_adapt":
        dump_json(
            os.path.join(args.output_dir, "device_adaptation.json"),
            {
                "schema": "mailohls-device-adaptation-v1",
                "device": TARGET_CFG.adapt_device,
                "base_adapter": os.path.abspath(args.init_adapter_dir),
                "base_harp_xattn": (
                    os.path.abspath(args.init_harp_xattn_from)
                    if args.init_harp_xattn_from else ""
                ),
                "active_adapters": ["mailohls_base", "device_adapt"],
                "device_capacities": DEVICE_RESOURCES[TARGET_CFG.adapt_device],
                "objective": args.objective,
            },
        )



def main():
    ap = argparse.ArgumentParser()

    # Data / Memory / Model
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_SFT_DATASET))
    ap.add_argument(
        "--directive_domain_registry_json",
        type=str,
        default="",
        help=(
            "Per-kernel, per-site legal RHS domains used by production "
            "constrained decoding. Required when validation selection is enabled."
        ),
    )
    ap.add_argument("--memory_dir", type=str, default=str(DEFAULT_STRUCTURAL_MEMORY))
    ap.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    ap.add_argument("--model_revision", type=str, default="main")
    ap.add_argument(
        "--objective",
        type=str,
        required=True,
        choices=GOAL_ORDER + ("ALL",),
        help="Use ALL for one objective-conditioned adapter.",
    )

    # Split Mode
    ap.add_argument("--split_mode", type=str, default="family", choices=["family", "random_design"])
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--stratify_by_kernel", action="store_true")
    ap.add_argument("--split_json", type=str, default="")
    ap.add_argument("--save_split_json", type=str, default="")

    # Goal-specific point selection
    ap.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="One target per prompt is the production setting.",
    )
    ap.add_argument("--goal_domination_penalty", type=float, default=0.25)
    ap.add_argument("--goal_max_dominated_gap", type=float, default=0.12)
    ap.add_argument("--candidate_loss_weight", type=float, default=0.0) # controls the influence of the contrastive loss in the final loss
    ap.add_argument("--candidate_sites_per_sample", type=int, default=0)
    ap.add_argument("--candidate_negatives_per_site", type=int, default=0)
    ap.add_argument("--candidate_max_prefix_tokens", type=int, default=1536)
    ap.add_argument("--candidate_keep_head_tokens", type=int, default=256)
    ap.add_argument("--val_families", type=str, default="rodinia_pathfinder;machsuite_sort_radix")
    ap.add_argument("--test_families", type=str, default="serrano-kalman-filter")
    ap.add_argument("--min_supervised_sites", type=int, default=2)
    ap.add_argument("--min_site_coverage", type=float, default=0.85)
    ap.add_argument("--score_weight_min", type=float, default=0.6)
    ap.add_argument("--score_weight_power", type=float, default=1.0)
    ap.add_argument(
        "--device_mode",
        choices=DEVICE_MODES,
        default="known",
        help=(
            "known: train the two measured devices; resource_dropout_ablation: "
            "device-token-dropout ablation; device_adapt: train a residual "
            "LoRA for one newly measured device"
        ),
    )
    ap.add_argument("--device_token_dropout", type=float, default=0.0)
    ap.add_argument("--device_specs_json", type=str, default="")
    ap.add_argument("--adapt_device", type=str, default="")
    ap.add_argument(
        "--allow_source_marker_truncation",
        action="store_true",
        help="Debug only: permit context truncation to remove action markers.",
    )
    ap.add_argument(
        "--resource_budget_mode",
        choices=["none", "fixed", "random"],
        default="random",
    )

    ap.add_argument(
        "--resource_budget_fracs",
        type=str,
        default="10,25,50,75,100",
    )

    ap.add_argument(
        "--random_budgets_per_case",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--random_budget_min_frac",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--min_feasible_candidates_per_budget",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--auto_frequency_fraction",
        type=float,
        default=0.0,
        help=(
            "Opt-in fraction of training prompts that ask the model to choose "
            "among measured clock periods. Keep 0 for the specified-clock "
            "baseline; evaluate this task separately before publication."
        ),
    )
    ap.add_argument(
        "--candidate_pool_per_objective",
        type=int,
        default=24,
    )
    ap.add_argument(
        "--min_auto_clock_count",
        type=int,
        default=2,
        help="Minimum measured clocks required to construct an automatic-clock prompt.",
    )

    # Training Params
    ap.add_argument("--max_length", type=int, default=7168)
    ap.add_argument(
        "--supervise_eos",
        action="store_true",
        help="Supervise EOS for free-generation ablations (off in production).",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument("--init_adapter_dir", type=str, default="")
    ap.add_argument("--init_harp_xattn_from", type=str, default="")
    ap.add_argument(
        "--initial_state_reference",
        type=str,
        default="",
        help=(
            "Shared initial_harp_state_post_sa_pre_mlp_s123.json; created "
            "atomically by the first arm and verified by later arms."
        ),
    )
    ap.add_argument("--value_loss_weight", type=float, default=1.0)
    ap.add_argument(
        "--directive_loss_weighting",
        choices=("uniform", "inverse_sqrt_frequency"),
        default="uniform",
        help="Uniform baseline or training-split-only inverse-sqrt balancing.",
    )

    # LoRA
    ap.add_argument("--lr_lora", type=float, default=5e-5)
    ap.add_argument("--lr_embed", type=float, default=5e-5)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # MLIR structural memory (older checkpoint fields retain the HARP name).
    ap.add_argument("--mem_dim", type=int, default=-1)   # -1 => infer from memory bank
    ap.add_argument("--require_pragma_free_memory", action="store_true")
    ap.add_argument(
        "--allow_missing_harp_memory",
        action="store_true",
        help="Debug only: use zero memory when a kernel embedding is absent.",
    )
    ap.add_argument("--max_slots", type=int, default=64)
    ap.add_argument("--every_n_layers", type=int, default=8)
    ap.add_argument("--xattn_heads", type=int, default=4)
    ap.add_argument("--xattn_dim_head", type=int, default=64)
    ap.add_argument("--xattn_ff_mult", type=int, default=1)
    ap.add_argument("--lr_xattn", type=float, default=0.0)
    ap.add_argument("--lr_gate", type=float, default=0.0)
    ap.add_argument("--lr_ff", type=float, default=0.0)
    ap.add_argument("--lr_gate_ff", type=float, default=0.0)

    # Best Checkpoint Selection
    ap.add_argument("--selection_num_val_kernels", type=int, default=0)
    ap.add_argument(
        "--selection_cases_per_kernel_device",
        "--selection_cases_per_kernel",
        dest="selection_cases_per_kernel_device",
        type=int,
        default=2,
    )
    ap.add_argument("--selection_eval_steps", type=int, default=200)
    ap.add_argument("--selection_candidate_batch_size", type=int, default=4)
    ap.add_argument("--best_dir_name", type=str, default="best_custom_stage1")

    # Trainer / pipeline
    ap.add_argument(
        "--disable_structural_memory",
        "--disable_harp",
        dest="disable_harp",
        action="store_true",
        help="Run directive-only Stage 1 without GNN structural memory.",
    )
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--loss_chunk_t", type=int, default=256)
    ap.add_argument("--run_mode", type=str, default="two_stage", choices=["single", "two_stage"])
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--stage1_output_dir", type=str, default="")
    ap.add_argument("--stage2_output_dir", type=str, default="")
    ap.add_argument("--stage2_epochs", type=int, default=4)
    ap.add_argument("--stage2_max_steps", type=int, default=-1)
    ap.add_argument("--stage2_eval_steps", type=int, default=50)
    ap.add_argument("--stage2_save_steps", type=int, default=50)
    ap.add_argument("--stage2_lr_xattn", type=float, default=1e-4)
    ap.add_argument("--stage2_lr_gate", type=float, default=2e-4)
    ap.add_argument("--stage2_lr_ff", type=float, default=0.0)
    ap.add_argument("--stage2_lr_gate_ff", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    configure_target_policy(args)
    print("[TARGET-CONDITIONING]", TARGET_CFG)

    goal_tag = (
        "all_objectives"
        if args.objective == "ALL"
        else GOALS[args.objective]["tag"]
    )
    if not args.stage1_output_dir:
        args.stage1_output_dir = f"./sft_harp_xattn_{goal_tag}_stage1"
    if not args.stage2_output_dir:
        args.stage2_output_dir = f"./sft_harp_xattn_{goal_tag}_stage2"

    if args.run_mode == "single":
        if not args.output_dir:
            args.output_dir = args.stage1_output_dir
        run_single_training(args)
        return

    stage1_cfg, stage2_cfg = build_default_stage_arguments(args)

    print("\n" + "=" * 120)
    print(f"[PIPELINE] Running {stage1_cfg.name} for {args.objective}")
    print(f"[PIPELINE] output_dir={stage1_cfg.output_dir}")
    print("=" * 120)
    stage1_args = make_stage_args(args, stage1_cfg)
    run_single_training(stage1_args)

    stage1_best_dir = os.path.join(stage1_cfg.output_dir, stage1_cfg.best_dir_name)
    if not os.path.isdir(stage1_best_dir):
        raise FileNotFoundError(
            f"Stage 1 best adapter was not created: {stage1_best_dir}"
        )

    cleanup_cuda()

    print("\n" + "=" * 120)
    print(f"[PIPELINE] Running {stage2_cfg.name} for {args.objective}")
    print(f"[PIPELINE] output_dir={stage2_cfg.output_dir}")
    print(f"[PIPELINE] init_adapter_dir={stage2_cfg.init_adapter_dir}")
    print("=" * 120)

    stage2_args = make_stage_args(args, stage2_cfg)

    if args.save_split_json:
        stage2_args.split_json = args.save_split_json
        stage2_args.save_split_json = ""

    run_single_training(stage2_args)
    


if __name__ == "__main__":
    main()
