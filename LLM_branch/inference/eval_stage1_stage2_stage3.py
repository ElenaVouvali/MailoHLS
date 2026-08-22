import argparse
import hashlib
import json
import os
import re
import warnings
import math
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from LLM_branch.common import (
    mailohls_contract,
    structural_memory as structural_memory_utils,
    structural_xattn,
)


TARGET_PLACEHOLDER_TOKENS = mailohls_contract.TARGET_PLACEHOLDER_TOKENS
SOURCE_PLACEHOLDER_TOKENS = mailohls_contract.SOURCE_PLACEHOLDER_TOKENS


# ============================================================
# Regexes
# ============================================================
SOURCE_LABEL_RE = re.compile(
    r"^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)",
    re.IGNORECASE,
)

TARGET_LINE_LABEL_RE = re.compile(
    r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}\s*=",
    re.IGNORECASE,
)

ANCHOR_OR_ASSIGN_RE = re.compile(
    r"^\s*(<L\d+>|auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}\s*=\s*.+)$",
    re.IGNORECASE | re.MULTILINE,
)

ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)
TARGET_ANCHOR_RE = re.compile(
    r"^<(L[1-9][0-9]*)>$",
    re.IGNORECASE,
)


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
    r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}",
    re.IGNORECASE,
)

LHS_KIND_RE = re.compile(
    r"^auto\{_([A-Z0-9]+(?:_[A-Z0-9]+)*)_L\d+\}$",
    re.IGNORECASE,
)


# ============================================================
# Small helpers
# ============================================================
def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())


def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


def mode_from_weights(w_lat: float, w_area: float) -> str:
    eps = 1e-9
    if abs(w_lat - 1.0) < eps and abs(w_area - 0.0) < eps:
        return "PARETO_LATENCY"
    if abs(w_lat - 0.0) < eps and abs(w_area - 1.0) < eps:
        return "PARETO_AREA"
    return "PARETO_ADP"


def normalize_weight_pair(w_lat: float, w_area: float) -> Tuple[float, float]:
    s = w_lat + w_area
    if s <= 0:
        raise ValueError("w_lat + w_area must be > 0")
    return w_lat / s, w_area / s


def source_placeholder_token(label: str) -> str:
    return f"<SRC_{label.upper()}>"


def target_placeholder_token(label: str) -> str:
    return f"<{label.upper()}>"


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


def dump_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dump_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Source / target formatting helpers
# ============================================================
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
        grouped[m.group(1).upper()].append(line)

    out = []
    emitted = set()

    for label in label_order:
        if label in grouped:
            out.extend(grouped[label])
            emitted.add(label)

    for label in sorted(grouped.keys()):
        if label not in emitted:
            out.extend(grouped[label])

    out.extend(extras)
    return "\n".join(out)


def extract_ordered_lhs_plan(source_text: str) -> List[Tuple[str, str]]:
    by_label = defaultdict(list)

    for line in source_text.splitlines():
        for m in SOURCE_PLACEHOLDER_IN_CODE_RE.finditer(line):
            lhs = m.group(0)
            label = m.group(1).upper()
            if lhs not in by_label[label]:
                by_label[label].append(lhs)

    plan = []
    for label in extract_source_label_order(source_text):
        for lhs in by_label.get(label, []):
            plan.append((label, lhs))
    return plan


def build_rhs_map_from_target(target_text: str) -> Dict[str, str]:
    rhs_map = {}
    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = ASSIGN_RE.match(line)
        if m is None:
            continue
        rhs_map[m.group(1).strip()] = m.group(2).strip()
    return rhs_map


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


def build_partial_deterministic_target_text(
    source_text: str,
    raw_target: str,
    min_supervised_sites: int = 1,
):
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


# ============================================================
# Dataset / candidate bank
# ============================================================
@dataclass
class InferenceCase:
    kernel_name: str
    source_text: str
    obj_mode: str
    w_lat: float
    w_area: float
    platform_row: dict
    reference_target: Optional[str] = None


def family_id_from_kernel_name(name: str) -> str:
    return mailohls_contract.family_id_from_kernel_name(name)


def load_rows(jsonl_path: str) -> List[dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["_jsonl_idx"] = idx
            ex["_family"] = family_id_from_kernel_name(ex["kernel_name"])
            rows.append(ex)
    return rows


def load_directive_domain_registry(path: str) -> Dict[str, Dict[str, List[str]]]:
    from LLM_branch.train.train_SFT_xattn_new import load_directive_domain_registry as load
    return load(path)


def get_rhs_candidates_for_lhs(kernel_name, lhs, directive_domain_registry):
    kernel = normalize_kname(kernel_name)
    sites = directive_domain_registry.get(kernel)
    if sites is None:
        raise KeyError(f"No directive domains found for kernel={kernel_name!r}")
    cands = sites.get(lhs.strip().upper(), [])
    if not cands:
        raise KeyError(f"No legal RHS domain for kernel={kernel_name!r}, lhs={lhs!r}")
    return cands
def objective_from_case_dict(ex: dict) -> Tuple[str, float, float]:
    if "w_lat" in ex and "w_area" in ex:
        w_lat, w_area = normalize_weight_pair(float(ex["w_lat"]), float(ex["w_area"]))
        return mode_from_weights(w_lat, w_area), w_lat, w_area

    if "obj_mode" in ex:
        obj_mode = str(ex["obj_mode"]).strip().upper()

        if obj_mode == "PARETO_LATENCY":
            return obj_mode, 1.0, 0.0
        if obj_mode == "PARETO_AREA":
            return obj_mode, 0.0, 1.0
        if obj_mode == "PARETO_ADP":
            return obj_mode, 0.5, 0.5

    if "objective" in ex:
        obj = str(ex["objective"]).strip().lower()

        if obj in {"pareto_latency_extreme", "latency_extreme", "latency", "min_lat", "min_latency"}:
            return "PARETO_LATENCY", 1.0, 0.0

        if obj in {"pareto_area_extreme", "area_extreme", "area", "min_area"}:
            return "PARETO_AREA", 0.0, 1.0

        if obj in {"pareto_adp", "pareto_knee", "adp", "knee", "balanced", "balance"}:
            return "PARETO_ADP", 0.5, 0.5

        raise ValueError(f"Unknown objective: {obj}")

    raise ValueError("Case must provide either (w_lat, w_area), obj_mode, or objective")


def load_inference_cases_jsonl(path: str) -> List[InferenceCase]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            obj_mode, w_lat, w_area = objective_from_case_dict(ex)
            source_text = ex.get("input") or ex.get("code") or ex.get("source_text")
            if not source_text:
                raise ValueError("Each case must contain one of: input, code, source_text")

            ref = ex.get("reference_target")
            if ref is None and ex.get("target"):
                ref, _ = build_partial_deterministic_target_text(
                    source_text,
                    ex["target"],
                    min_supervised_sites=1,
                )

            rows.append(
                InferenceCase(
                    kernel_name=ex["kernel_name"],
                    source_text=source_text,
                    obj_mode=obj_mode,
                    w_lat=w_lat,
                    w_area=w_area,
                    platform_row=ex,
                    reference_target=ref,
                )
            )
    return rows


# ============================================================
# STRUCTURAL memory bank
# ============================================================
def load_memory_bank(
    memory_dir: str,
    *,
    expected_mem_dim: int,
    expected_max_slots: int,
) -> Dict[str, dict]:
    """Load inference memory through the same validated Stage-2 path.

    In particular, never compact sparse absolute Lk slots: target placeholders
    route directly to their one-based compiler action addresses.
    """

    bank, inferred_mem_dim = structural_memory_utils.load_memory_bank(
        memory_dir,
        expected_mem_dim=expected_mem_dim,
        expected_max_slots=expected_max_slots,
    )
    if inferred_mem_dim != expected_mem_dim:
        raise ValueError(
            "Inference memory dimension differs from the checkpoint: "
            f"{inferred_mem_dim} != {expected_mem_dim}"
        )
    return bank


def get_real_memory_pack_for_kernel(
    mem_bank: Dict[str, dict],
    kernel_name: str,
    max_slots: int,
    mem_dim: int,
    structural_routing: str,
):
    return structural_memory_utils.get_structural_memory_pack_for_kernel(
        mem_bank,
        kernel_name,
        max_slots,
        mem_dim,
        structural_routing=structural_routing,
    )


# ============================================================
# Structural cross-attention utilities
# ============================================================
def get_first_real_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0")


def move_structural_modules_to_model_device(model):
    device = get_first_real_device(model)
    moved = 0
    for module in model.modules():
        if isinstance(module, GatedCrossAttentionBlock):
            module.to(device=device)
            moved += 1
    print(f"[STRUCTURAL-DEVICE] moved {moved} STRUCTURAL blocks to {device}")


def print_xattn_forward_stats(model):
    found = False
    for name, module in model.named_modules():
        if isinstance(module, MaskedCrossAttention) and getattr(module, "last_debug", None):
            found = True
            print(f"[XATTN-DBG] {name}: {module.last_debug}")
    if not found:
        print("[XATTN-DBG] no cross-attn forward stats collected yet")


def get_structural_xattn_state_dict(model):
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if "gated_cross_attn_layer" in k}


def load_partial_structural_xattn(model, structural_xattn_path: str, tag: str):
    if not structural_xattn_path or not os.path.isfile(structural_xattn_path):
        raise FileNotFoundError(f"[{tag}] no structural_xattn.pt found at: {structural_xattn_path}")

    structural_sd = torch.load(
        structural_xattn_path,
        map_location="cpu",
        weights_only=True,
    )
    target_keys = set(model.state_dict())
    placement_paths = (
        (
            ".post_attention_layernorm."
            "gated_cross_attn_layer."
        ),
        (
            ".structural_post_decoder_residual."
            "gated_cross_attn_layer."
        ),
        (
            ".structural_post_self_attention_residual."
            "gated_cross_attn_layer."
        ),
    )
    remapped = {}
    for key, value in structural_sd.items():
        candidates = [key]
        source_path = next(
            (path for path in placement_paths if path in key),
            None,
        )
        if source_path is not None:
            candidates.extend(
                key.replace(source_path, target_path)
                for target_path in placement_paths
                if target_path != source_path
            )
        mapped_key = next(
            (candidate for candidate in candidates if candidate in target_keys),
            key,
        )
        remapped[mapped_key] = value

    structural_sd = remapped
    missing, unexpected = model.load_state_dict(structural_sd, strict=False)
    structural_missing = [k for k in missing if "gated_cross_attn_layer" in k]
    structural_unexpected = [
        key for key in unexpected if "gated_cross_attn_layer" in key
    ]
    if structural_missing or structural_unexpected:
        raise ValueError(
            f"[{tag}] incompatible cross-attention state: "
            f"missing={structural_missing[:10]}, "
            f"unexpected={structural_unexpected[:10]}"
        )
    print(f"[{tag}] loaded all structural parameters exactly")
    move_structural_modules_to_model_device(model)


MaskedCrossAttention = structural_xattn.MaskedCrossAttention
GatedCrossAttentionBlock = structural_xattn.GatedCrossAttentionBlock
StructuralCrossAttentionMixin = structural_xattn.StructuralCrossAttentionMixin
extend_instance = structural_xattn.extend_instance
infer_decoder_layers_attr_name = structural_xattn.infer_decoder_layers_attr_name


# ============================================================
# Candidate-scoring inference
# ============================================================
@torch.no_grad()
def append_token_ids(input_ids, attention_mask, new_ids: List[int]):
    device = input_ids.device
    new_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=device)
    new_attn = torch.ones((1, len(new_ids)), dtype=attention_mask.dtype, device=device)
    input_ids = torch.cat([input_ids, new_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, new_attn], dim=1)
    return input_ids, attention_mask


def truncate_scoring_prefix(
    prefix_ids: List[int],
    max_prefix_tokens: int,
    keep_head_tokens: int,
) -> List[int]:
    if max_prefix_tokens <= 0 or len(prefix_ids) <= max_prefix_tokens:
        return prefix_ids

    keep_head = min(max(0, keep_head_tokens), max_prefix_tokens - 1)
    keep_tail = max_prefix_tokens - keep_head

    if keep_head <= 0:
        return prefix_ids[-max_prefix_tokens:]

    return prefix_ids[:keep_head] + prefix_ids[-keep_tail:]


def truncate_scoring_prefix_preserve_target(
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
    # because STRUCTURAL routing depends on target anchors already emitted.
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


def make_candidate_xattn_apply_mask(
    *,
    full_length: int,
    base_len: int,
    candidate_len: int,
    device,
):
    """Match the causal candidate-logit mask used during Stage-2 training."""

    start = base_len - 1
    end = start + candidate_len
    if start < 0 or end > full_length:
        raise ValueError(
            "Invalid causal candidate span: "
            f"full={full_length} base={base_len} candidate={candidate_len}"
        )
    mask = torch.zeros(
        (1, full_length),
        dtype=torch.float32,
        device=device,
    )
    mask[:, start:end] = 1.0
    return mask


@torch.no_grad()
def score_rhs_candidate_suffix(
    *,
    model,
    tok,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    candidate_text: str,
    routing_start_idx: Optional[torch.Tensor] = None,
    use_structural_memory: bool = False,
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

    if use_structural_memory:
        if routing_start_idx is None:
            raise ValueError("routing_start_idx is required when use_structural_memory=True")

        xmask = make_candidate_xattn_apply_mask(
            full_length=full_input_ids.shape[1],
            base_len=base_len,
            candidate_len=cand_len,
            device=device,
        )

        model_inputs["routing_start_idx"] = routing_start_idx
        model_inputs["xattn_apply_mask"] = xmask

    outputs = model(**model_inputs)
    cand_logits = outputs.logits[:, base_len - 1 : base_len - 1 + cand_len, :].float()
    target = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(0)

    token_logprobs = F.log_softmax(cand_logits, dim=-1)
    token_logprobs = token_logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).squeeze(0)

    return {
        "sum_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
        "token_count": int(cand_len),
    }


def select_candidate_from_scored(
    scored: List[Dict[str, Any]],
    *,
    decode_mode: str,
    sample_temperature: float,
    sample_top_p: float,
    sample_top_k: int,
    sample_generator: Optional[torch.Generator],
):
    """
    scored: list of dicts with keys:
        rhs, score, mean_logprob, sum_logprob, token_count
    Returns:
        chosen_candidate_dict, ordered_scored_list
    """
    ordered = sorted(scored, key=lambda x: (x["score"], x["sum_logprob"]), reverse=True)

    if len(ordered) == 1 or decode_mode == "greedy":
        chosen = copy.deepcopy(ordered[0])
        chosen["sample_prob"] = 1.0
        chosen["sample_rank"] = 1
        return chosen, ordered

    logits = torch.tensor([x["score"] for x in ordered], dtype=torch.float32)

    temp = max(float(sample_temperature), 1e-6)
    logits = logits / temp

    # top-k filter
    if sample_top_k > 0 and sample_top_k < logits.numel():
        topk_vals, topk_idx = torch.topk(logits, k=sample_top_k)
        masked = torch.full_like(logits, -float("inf"))
        masked[topk_idx] = logits[topk_idx]
        logits = masked

    # top-p / nucleus filter
    if sample_top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=0)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        keep = cumulative_probs <= float(sample_top_p)
        if keep.numel() > 0:
            keep[0] = True  # always keep the best one

        masked = torch.full_like(logits, -float("inf"))
        kept_idx = sorted_idx[keep]
        masked[kept_idx] = logits[kept_idx]
        logits = masked

    probs = torch.softmax(logits, dim=0)

    if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0:
        probs = torch.zeros_like(logits)
        probs[0] = 1.0

    chosen_idx = int(torch.multinomial(probs, num_samples=1, generator=sample_generator).item())

    probs_list = probs.tolist()
    for i, p in enumerate(probs_list):
        ordered[i]["sample_prob"] = float(p)

    chosen = copy.deepcopy(ordered[chosen_idx])
    chosen["sample_prob"] = float(probs[chosen_idx].item())
    chosen["sample_rank"] = int(chosen_idx + 1)
    return chosen, ordered


def annotate_candidate_uniqueness(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Marks candidates in-place with:
      - is_unique
      - duplicate_of_sample_id
      - score_rank_among_unique
    Returns the unique candidates sorted by model score.
    """
    seen = {}
    unique = []

    for cand in candidates:
        key = cand["canonical_prediction"]
        if key not in seen:
            cand["is_unique"] = True
            cand["duplicate_of_sample_id"] = None
            seen[key] = cand["sample_id"]
            unique.append(cand)
        else:
            cand["is_unique"] = False
            cand["duplicate_of_sample_id"] = seen[key]

    unique_sorted = sorted(
        unique,
        key=lambda x: (x["sequence_score"], x["sequence_sum_logprob"]),
        reverse=True,
    )

    for rank, cand in enumerate(unique_sorted, start=1):
        cand["score_rank_among_unique"] = rank

    for cand in candidates:
        if "score_rank_among_unique" not in cand:
            cand["score_rank_among_unique"] = None

    return unique_sorted


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
    structural_memory: Optional[torch.Tensor] = None,
    structural_memory_mask: Optional[torch.Tensor] = None,
    structural_relation_mask: Optional[torch.Tensor] = None,
    routing_start_idx: Optional[torch.Tensor] = None,
    debug_topk: int = 0,
    candidate_max_prefix_tokens: int = 0,
    candidate_keep_head_tokens: int = 0,
    decode_mode: str = "greedy",              # NEW
    sample_temperature: float = 1.0,          # NEW
    sample_top_p: float = 1.0,                # NEW
    sample_top_k: int = 0,                    # NEW
    sample_generator: Optional[torch.Generator] = None,  # NEW
):
    assert score_reduction in {"mean", "sum"}
    assert decode_mode in {"greedy", "sample"}

    device = get_first_real_device(model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    if routing_start_idx is None:
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

    parts = []
    current_label = None
    site_debug = []
    chosen_assignments = {}
    site_domains = directive_domain_registry[normalize_kname(kernel_name)]

    sequence_score = 0.0
    sequence_sum_logprob = 0.0
    site_count = 0

    structural_enabled = hasattr(model, "condition_structural_memory") and getattr(model, "initialized_structural_xattn", False)
    use_structural_memory = structural_enabled and (structural_memory is not None) and (structural_memory_mask is not None)

    if use_structural_memory:
        model.condition_structural_memory(
            structural_memory.to(device),
            structural_memory_mask.to(device),
            action_relation_mask=(
                structural_relation_mask.to(device)
                if structural_relation_mask is not None
                else None
            ),
        )

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

            original_candidates = get_rhs_candidates_for_lhs(
                kernel_name, lhs, directive_domain_registry
            )
            candidates = mailohls_contract.filter_semantic_candidates(
                lhs, original_candidates, chosen_assignments, site_domains
            )
            scored = []

            base_prefix_ids = input_ids[0].tolist()
            effective_route_idx = int(routing_start_idx.item()) if routing_start_idx is not None else None

            truncated_prefix_ids, effective_route_idx = truncate_scoring_prefix_preserve_target(
                prefix_ids=base_prefix_ids,
                routing_start_idx=effective_route_idx,
                max_prefix_tokens=candidate_max_prefix_tokens,
                keep_head_tokens=candidate_keep_head_tokens,
            )

            base_input = torch.tensor([truncated_prefix_ids], dtype=torch.long, device=device)
            base_mask = torch.ones_like(base_input)

            effective_routing_start_idx = None
            if effective_route_idx is not None:
                effective_routing_start_idx = torch.tensor(
                    [effective_route_idx],
                    dtype=torch.long,
                    device=device,
                )

            if len(candidates) == 1:
                scored.append({"rhs": candidates[0], "score": 0.0,
                               "mean_logprob": 0.0, "sum_logprob": 0.0,
                               "token_count": 0})
            for rhs in (candidates if len(candidates) > 1 else []):
                stats = score_rhs_candidate_suffix(
                    model=model,
                    tok=tok,
                    base_input_ids=base_input,
                    base_attention_mask=base_mask,
                    candidate_text=rhs + "\n",
                    routing_start_idx=effective_routing_start_idx,
                    use_structural_memory=use_structural_memory,
                )
                scored.append(
                    {
                        "rhs": rhs,
                        "score": stats["mean_logprob"] if score_reduction == "mean" else stats["sum_logprob"],
                        "mean_logprob": stats["mean_logprob"],
                        "sum_logprob": stats["sum_logprob"],
                        "token_count": stats["token_count"],
                    }
                )

            chosen, ordered = select_candidate_from_scored(
                scored,
                decode_mode=decode_mode,
                sample_temperature=sample_temperature,
                sample_top_p=sample_top_p,
                sample_top_k=sample_top_k,
                sample_generator=sample_generator,
            )

            chosen_text = chosen["rhs"] + "\n"
            chosen_assignments[lhs.strip().upper()] = chosen["rhs"]
            chosen_ids = tok(chosen_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, chosen_ids)
            parts.append(chosen_text)

            sequence_score += float(chosen["score"])
            sequence_sum_logprob += float(chosen["sum_logprob"])
            site_count += 1

            if debug_topk > 0:
                site_debug.append(
                    {
                        "label": label,
                        "lhs": lhs,
                        "chosen_rhs": chosen["rhs"],
                        "chosen_score": float(chosen["score"]),
                        "chosen_sum_logprob": float(chosen["sum_logprob"]),
                        "chosen_sample_prob": float(chosen.get("sample_prob", 1.0)),
                        "chosen_sample_rank": int(chosen.get("sample_rank", 1)),
                        "static_candidate_count": len(original_candidates),
                        "forced_by_semantics": len(candidates) == 1,
                        "top_candidates": ordered[:debug_topk],
                    }
                )

        prediction = "".join(parts).rstrip()
        mailohls_contract.validate_directive_assignments(chosen_assignments)
        canonical_prediction = canonicalize_generation(prediction)

        return {
            "prediction": prediction,
            "canonical_prediction": canonical_prediction,
            "site_debug": site_debug,
            "sequence_score": float(sequence_score),
            "sequence_sum_logprob": float(sequence_sum_logprob),
            "sequence_mean_site_score": float(sequence_score / max(site_count, 1)),
            "site_count": int(site_count),
            "decode_mode": decode_mode,
        }

    finally:
        if hasattr(model, "clear_structural_memory"):
            model.clear_structural_memory()


# ============================================================
# Metrics
# ============================================================
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
        (k in pred_assign) and (pred_assign[k] == ref_assign[k]) for k in expected_keys
    )

    return {
        "canonical_prediction": pred_text,
        "value_accuracy_over_expected": exact_value_match_count / max(len(expected_keys), 1),
        "schema_compliant": schema_compliant,
        "expected_key_match": expected_key_match,
        "exact_design_match": schema_compliant and pred_assign == ref_assign,
        "n_expected": len(expected_keys),
        "n_predicted": len(pred_assign),
    }


# ============================================================
# Model loading
# ============================================================
def build_tokenizer(
    tokenizer_source: str,
    revision: str = "main",
) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=revision,
        trust_remote_code=True,
    )
    special_tokens = (
        [spec["token"] for spec in mailohls_contract.GOALS.values()]
        + list(mailohls_contract.DEVICE_TOKEN_MAP.values())
        + [mailohls_contract.UNKNOWN_DEVICE_TOKEN, mailohls_contract.ADAPTED_DEVICE_TOKEN]
        + list(mailohls_contract.PERIOD_TOKEN_MAP.values())
        + [mailohls_contract.AUTO_PERIOD_TOKEN, mailohls_contract.CLOCK_ANCHOR_TOKEN]
        + SOURCE_PLACEHOLDER_TOKENS
        + TARGET_PLACEHOLDER_TOKENS
    )
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# def load_base_model(model_name: str, use_4bit: bool = True, device_map: str = "auto"):
#     quantization_config = None
#     if use_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=quantization_config,
#         torch_dtype=(torch.bfloat16 if use_4bit and torch.cuda.is_available() else None),
#         device_map=device_map,
#         trust_remote_code=True,
#     )

#     model.config.use_cache = False
#     return model


def load_base_model(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = "auto",
    revision: str = "main",
):
    quant_config = None
    if use_4bit:
        native_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
            and torch.cuda.is_bf16_supported()
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if native_bf16 else torch.float16
            ),
        )

    if device_map == "auto":
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    return model


def attach_structural_modules(
    model,
    tok,
    mem_dim: int,
    every_n_layers: int,
    xattn_heads: int,
    xattn_dim_head: int,
    xattn_ff_mult: int,
    xattn_enable_ff: bool,
    structural_fusion_placement: str,
    structural_gate_scale: float,
    structural_memory_value_scale: float,
):
    extend_instance(model, StructuralCrossAttentionMixin)
    decoder_layers_attr_name = infer_decoder_layers_attr_name(model)
    model.set_decoder_layers_attr_name(decoder_layers_attr_name)

    placeholder_token_ids = tok.convert_tokens_to_ids(TARGET_PLACEHOLDER_TOKENS)
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer LM hidden size from model.config")

    model.init_structural_cross_attention(
        placeholder_token_ids=placeholder_token_ids,
        lang_hidden_size=hidden_size,
        mem_hidden_size=mem_dim,
        cross_attn_every_n_layers=every_n_layers,
        xattn_heads=xattn_heads,
        xattn_dim_head=xattn_dim_head,
        only_attend_immediate_memory=True,
        mask_mode="segment",
        attn_gate_scale=structural_gate_scale,
        memory_value_scale=structural_memory_value_scale,
        structural_fusion_placement=structural_fusion_placement,
    )

    print(f"[STRUCTURAL-XATTN] decoder_layers_attr_name={decoder_layers_attr_name}")
    print(f"[STRUCTURAL-XATTN] inserted gated xattn every {every_n_layers} decoder layers")
    move_structural_modules_to_model_device(model)


def load_stage_model(args, tok):
    base = load_base_model(
        model_name=args.model,
        use_4bit=not args.no_4bit,
        device_map=args.device_map,
        revision=args.model_revision,
    )

    base.resize_token_embeddings(len(tok))
    if args.stage in {"stage1", "stage2", "stage3"}:
        input_weight = base.get_input_embeddings().weight
        output_weight = base.get_output_embeddings().weight
        actual_tied = input_weight.data_ptr() == output_weight.data_ptr()
        if actual_tied != args.embedding_weights_tied:
            raise ValueError(
                "Backbone embedding tying differs from the training contract"
            )

    lora_adapter_dir = args.lora_adapter_dir or args.adapter_dir
    if not lora_adapter_dir:
        raise ValueError("--adapter_dir (or --lora_adapter_dir) is required for stage1/stage2 inference")

    model = load_peft_adapter_strict(base, lora_adapter_dir)

    if args.stage in {"stage2", "stage3"}:
        attach_structural_modules(
            model,
            tok,
            mem_dim=args.mem_dim,
            every_n_layers=args.every_n_layers,
            xattn_heads=args.xattn_heads,
            xattn_dim_head=args.xattn_dim_head,
            xattn_ff_mult=args.xattn_ff_mult,
            xattn_enable_ff=args.xattn_enable_ff,
            structural_fusion_placement=(
                args.structural_fusion_placement
            ),
            structural_gate_scale=args.structural_gate_scale,
            structural_memory_value_scale=(
                args.structural_memory_value_scale
            ),
        )

        structural_xattn_path = args.structural_xattn_path
        if not structural_xattn_path:
            structural_xattn_path = os.path.join(args.adapter_dir, "structural_xattn.pt")

        load_partial_structural_xattn(
            model,
            structural_xattn_path,
            tag=f"STRUCTURAL-LOAD-{args.stage.upper()}",
        )

    model.eval()
    return model



def has_adapter_weights(adapter_dir: str) -> bool:
    return any(
        os.path.isfile(os.path.join(adapter_dir, fn))
        for fn in ("adapter_model.safetensors", "adapter_model.bin")
    )


def load_peft_adapter_strict(base, adapter_dir: str):
    if not adapter_dir:
        raise ValueError("Empty adapter_dir")

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    if not has_adapter_weights(adapter_dir):
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in: {adapter_dir}"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)

    missing_adapter_warnings = [
        str(w.message) for w in caught
        if "Found missing adapter keys" in str(w.message)
    ]

    if missing_adapter_warnings:
        raise RuntimeError(
            f"LoRA adapter did not load cleanly from {adapter_dir}.\n"
            "Use the self-contained adapter directory for the requested "
            "stage and its matching structural_xattn.pt."
        )

    print(f"[ADAPTER] loaded clean PEFT adapter from: {adapter_dir}")
    return model


# ============================================================
# Case prediction
# ============================================================
def predict_case(
    *,
    model,
    tok,
    case: InferenceCase,
    directive_domain_registry: Dict[str, Dict[str, List[str]]],
    stage: str,
    max_prompt_tokens: int,
    mem_bank: Optional[Dict[str, dict]],
    mem_dim: int,
    max_slots: int,
    structural_routing: str,
    score_reduction: str,
    debug_topk: int,
    candidate_max_prefix_tokens: int,
    candidate_keep_head_tokens: int,
    print_xattn_debug_flag: bool,
    num_samples: int,                 # NEW
    no_greedy_first: bool,            # NEW
    sample_temperature: float,        # NEW
    sample_top_p: float,              # NEW
    sample_top_k: int,                # NEW
    sample_seed: int,                 # NEW
) -> Dict[str, Any]:
    if str(case.platform_row.get("frequency_mode", "specified")).lower() != "specified":
        raise ValueError("Automatic-clock inference is not implemented in the locked contract")
    sections = mailohls_contract.build_prompt_sections(
        case.source_text,
        case.obj_mode,
        mailohls_contract.target_prompt_fields(case.platform_row),
    )
    base_prompt_ids = [
        token_id for section in sections
        for token_id in tok(section, add_special_tokens=False)["input_ids"]
    ]
    prompt_ids = base_prompt_ids + mailohls_contract.selected_clock_response_token_ids(
        case.platform_row, tok
    )
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError("Inference prompt and selected-clock prefix exceed --max_length")

    device = get_first_real_device(model)
    routing_start_idx = torch.tensor([len(base_prompt_ids)], dtype=torch.long, device=device)

    structural_memory = None
    structural_memory_mask = None
    structural_relation_mask = None
    if stage in {"stage2", "stage3"}:
        if mem_bank is None:
            raise ValueError("mem_bank must be provided for stage2/stage3 inference")
        (
            structural_memory,
            structural_memory_mask,
            structural_relation_mask,
        ) = get_real_memory_pack_for_kernel(
            mem_bank,
            case.kernel_name,
            max_slots=max_slots,
            mem_dim=mem_dim,
            structural_routing=structural_routing,
        )

    decode_plan = []
    if not no_greedy_first:
        decode_plan.append(("greedy", None))

    while len(decode_plan) < num_samples:
        per_sample_seed = int(sample_seed + len(decode_plan))
        decode_plan.append(("sample", per_sample_seed))

    candidates = []

    for sample_id, (decode_mode, this_seed) in enumerate(decode_plan):
        sample_generator = None
        if decode_mode == "sample":
            sample_generator = torch.Generator(device="cpu")
            sample_generator.manual_seed(int(this_seed))

        out = constrained_decode_rhs_by_candidate_scoring(
            model=model,
            tok=tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            kernel_name=case.kernel_name,
            directive_domain_registry=directive_domain_registry,
            score_reduction=score_reduction,
            structural_memory=structural_memory,
            structural_memory_mask=structural_memory_mask,
            structural_relation_mask=structural_relation_mask,
            routing_start_idx=routing_start_idx,
            debug_topk=debug_topk,
            candidate_max_prefix_tokens=candidate_max_prefix_tokens,
            candidate_keep_head_tokens=candidate_keep_head_tokens,
            decode_mode=decode_mode,
            sample_temperature=sample_temperature,
            sample_top_p=sample_top_p,
            sample_top_k=sample_top_k,
            sample_generator=sample_generator,
        )

        cand = {
            "sample_id": int(sample_id),
            "decode_mode": decode_mode,
            "sample_seed": this_seed,
            "prediction": out["canonical_prediction"],
            "canonical_prediction": out["canonical_prediction"],
            "sequence_score": float(out["sequence_score"]),
            "sequence_sum_logprob": float(out["sequence_sum_logprob"]),
            "sequence_mean_site_score": float(out["sequence_mean_site_score"]),
            "site_count": int(out["site_count"]),
            "site_debug": out["site_debug"],
        }

        if case.reference_target is not None:
            metrics = evaluate_prediction(case.reference_target, out["canonical_prediction"])
            cand.update(metrics)

        candidates.append(cand)

        if print_xattn_debug_flag and stage in {"stage2", "stage3"} and sample_id == 0:
            print_xattn_forward_stats(model)

    unique_candidates_sorted = annotate_candidate_uniqueness(candidates)

    row = {
        "kernel_name": case.kernel_name,
        "obj_mode": case.obj_mode,
        "w_lat": case.w_lat,
        "w_area": case.w_area,
        "prompt_token_count": len(prompt_ids),
        "n_generated": len(candidates),
        "n_unique": len(unique_candidates_sorted),
        "best_unique_sample_id_by_model_score": (
            int(unique_candidates_sorted[0]["sample_id"]) if unique_candidates_sorted else None
        ),
        "unique_sample_ids_by_model_score": [int(x["sample_id"]) for x in unique_candidates_sorted],
        "candidates": candidates,
    }

    if stage in {"stage2", "stage3"} and structural_memory_mask is not None:
        row["memory_active_slots"] = int(structural_memory_mask.sum().item())

    if case.reference_target is not None:
        row["reference_target"] = case.reference_target

        unique_with_ref = [x for x in unique_candidates_sorted if "value_accuracy_over_expected" in x]
        if unique_with_ref:
            best_acc = sorted(
                unique_with_ref,
                key=lambda x: (x["value_accuracy_over_expected"], x["sequence_score"]),
                reverse=True,
            )[0]
            row["best_unique_sample_id_by_value_accuracy"] = int(best_acc["sample_id"])
            row["best_value_accuracy_over_expected"] = float(best_acc["value_accuracy_over_expected"])

    return row


# ============================================================
# CLI utilities
# ============================================================
def build_single_case_from_args(args) -> InferenceCase:
    if not args.kernel_name:
        raise ValueError("--kernel_name is required for single-case inference")

    if not args.code_file:
        raise ValueError("--code_file is required for single-case inference")

    with open(args.code_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    if args.objective:
        obj = args.objective.strip().lower()

        if obj in {"pareto_latency_extreme", "latency_extreme", "latency", "min_lat", "min_latency"}:
            obj_mode, w_lat, w_area = "PARETO_LATENCY", 1.0, 0.0
        elif obj in {"pareto_area_extreme", "area_extreme", "area", "min_area"}:
            obj_mode, w_lat, w_area = "PARETO_AREA", 0.0, 1.0
        elif obj in {"pareto_adp", "pareto_knee", "adp", "knee", "balanced", "balance"}:
            obj_mode, w_lat, w_area = "PARETO_ADP", 0.5, 0.5
        else:
            raise ValueError(f"Unknown objective: {args.objective}")
    else:
        w_lat, w_area = normalize_weight_pair(args.w_lat, args.w_area)
        obj_mode = mode_from_weights(w_lat, w_area)

    return InferenceCase(
        kernel_name=args.kernel_name,
        source_text=source_text,
        obj_mode=obj_mode,
        w_lat=w_lat,
        w_area=w_area,
        platform_row={
            "kernel_name": args.kernel_name,
            "device": args.device,
            "clock_period": args.clock_period,
            "frequency_mode": args.frequency_mode,
            "available_clock_periods": args.available_clock_periods,
            "avail_bram": args.avail_bram,
            "avail_dsp": args.avail_dsp,
            "avail_ff": args.avail_ff,
            "avail_lut": args.avail_lut,
        },
        reference_target=None,
    )


def load_cases(args) -> List[InferenceCase]:
    if args.input_jsonl:
        return load_inference_cases_jsonl(args.input_jsonl)
    return [build_single_case_from_args(args)]


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # model / stage
    ap.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2", "stage3"])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--lora_adapter_dir", type=str, default="")
    ap.add_argument("--structural_xattn_path", type=str, default="")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument(
        "--model_revision",
        type=str,
        default=None,
    )

    ap.add_argument("--directive_domain_registry_json", type=str, required=True)

    # stage2 memory
    ap.add_argument("--memory_dir", type=str, default="")
    ap.add_argument("--mem_dim", type=int)
    ap.add_argument("--max_slots", type=int)
    ap.add_argument("--every_n_layers", type=int)
    ap.add_argument("--xattn_heads", type=int)
    ap.add_argument("--xattn_dim_head", type=int)
    ap.add_argument("--xattn_ff_mult", type=int)
    ap.add_argument(
        "--structural_fusion_placement",
        choices=(
            "legacy_norm_wrapper",
            "post_decoder_residual",
            "post_self_attention_residual",
        ),
        default=None,
    )
    ap.add_argument(
        "--structural_routing",
        choices=("exact_slot", "compiler_relational"),
        default=None,
    )
    ap.add_argument("--structural_gate_scale", type=float, default=None)
    ap.add_argument(
        "--structural_memory_value_scale",
        type=float,
        default=None,
    )

    # input cases
    ap.add_argument("--input_jsonl", type=str, default="")
    ap.add_argument("--kernel_name", type=str, default="")
    ap.add_argument("--code_file", type=str, default="")
    ap.add_argument("--objective", type=str, default="")
    ap.add_argument("--w_lat", type=float, default=0.5)
    ap.add_argument("--w_area", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="xczu7ev-ffvc1156-2-e")
    ap.add_argument("--clock_period", type=float, default=5.0)
    ap.add_argument("--frequency_mode", choices=["specified", "auto"], default="specified")
    ap.add_argument("--available_clock_periods", type=float, nargs="+", default=[5.0])
    ap.add_argument("--avail_bram", type=int)
    ap.add_argument("--avail_dsp", type=int)
    ap.add_argument("--avail_ff", type=int)
    ap.add_argument("--avail_lut", type=int)

    # decoding / scoring
    ap.add_argument("--max_prompt_tokens", type=int, default=7168)
    ap.add_argument("--score_reduction", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--debug_topk", type=int, default=0)
    ap.add_argument("--candidate_max_prefix_tokens", type=int, default=0)
    ap.add_argument("--candidate_keep_head_tokens", type=int, default=256)
    ap.add_argument("--print_xattn_debug", action="store_true")

    # output
    ap.add_argument("--output_jsonl", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--print_predictions", action="store_true")

    # samples
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--no_greedy_first", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.8)
    ap.add_argument("--sample_top_p", type=float, default=1.0)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=123)

    args = ap.parse_args()
    args.xattn_enable_ff = False

    if args.stage in {"stage2", "stage3"} and not args.memory_dir:
        raise ValueError("--memory_dir is required for stage2/stage3 inference")
    if args.stage in {"stage2", "stage3"}:
        if (
            args.stage == "stage3"
            and args.lora_adapter_dir
            and os.path.realpath(args.lora_adapter_dir)
            != os.path.realpath(args.adapter_dir)
        ):
            raise ValueError(
                "Stage-3 inference is self-contained; do not override its "
                "LoRA with --lora_adapter_dir"
            )
        contract_path = os.path.join(args.adapter_dir, "training_contract.json")
        if not os.path.isfile(contract_path):
            raise ValueError(
                f"{args.stage} checkpoint is missing {contract_path}"
            )
        with open(contract_path, "r", encoding="utf-8") as handle:
            training_contract = json.load(handle)
        if training_contract.get("schema") != "mailohls-training-contract-v1":
            raise ValueError("Unsupported training contract")
        if training_contract.get("stage") != args.stage:
            raise ValueError(
                f"{args.stage} inference requires a matching "
                f"{args.stage} training contract"
            )
        if args.stage == "stage3":
            parent = training_contract.get("parent_stage2")
            preference = training_contract.get("stage3_preference")
            if not isinstance(parent, dict) or not {
                "contract_sha256", "structural_xattn_sha256"
            }.issubset(parent):
                raise ValueError(
                    "Stage-3 contract has no valid parent_stage2 lineage"
                )
            if (
                not isinstance(preference, dict)
                or preference.get("schema")
                != "mailohls-stage3-preference-v1"
            ):
                raise ValueError(
                    "Stage-3 contract has no valid preference section"
                )
        trained_model = training_contract.get("model")
        if args.model and trained_model != args.model:
            raise ValueError(
                f"--model={args.model!r} conflicts with checkpoint model "
                f"{trained_model!r}"
            )
        args.model = trained_model
        trained_revision = training_contract.get("model_revision")
        if args.model_revision and trained_revision != args.model_revision:
            raise ValueError(
                f"Model revision does not match the {args.stage} contract: "
                f"{args.model_revision} != "
                f"{trained_revision}"
            )
        args.model_revision = trained_revision
        args.embedding_weights_tied = bool(
            training_contract["embedding_weights_tied"]
        )
        structural_config = training_contract.get("structural")
        if not isinstance(structural_config, dict):
            raise ValueError(
                f"{args.stage} training contract has no structural section"
            )
        required = (
            "mem_dim", "max_slots",
            "every_n_layers", "xattn_heads", "xattn_dim_head", "xattn_ff_mult", "xattn_enable_ff",
            "memory_manifest_sha256",
            "xattn_placement", "xattn_gate_init",
            "selected_xattn_layers_1based",
            "selection_eval_gate_scale",
            "selection_eval_memory_value_scale",
            "structural_routing",
        )
        missing = [key for key in required if key not in structural_config]
        if missing:
            raise ValueError(f"Structural contract is missing fields: {missing}")
        if training_contract.get("prompt_schema_version") not in {
            1, mailohls_contract.PROMPT_SCHEMA_VERSION
        }:
            raise ValueError("Unsupported prompt_schema_version")
        trained_placement = structural_config["xattn_placement"]
        supported_placements = {
            "legacy_norm_wrapper",
            "post_decoder_residual",
            "post_self_attention_residual",
        }
        if trained_placement not in supported_placements:
            raise ValueError(
                "Unsupported structural cross-attention placement: "
                f"{trained_placement!r}"
            )
        if float(structural_config["xattn_gate_init"]) != 0.0:
            raise ValueError("Unsupported xattn_gate_init; expected 0.0")
        if bool(structural_config["xattn_enable_ff"]):
            raise ValueError("Structural cross-attention FF must be disabled")
        args.selected_xattn_layers_1based = tuple(
            map(int, structural_config["selected_xattn_layers_1based"])
        )
        contract_runtime_values = {
            "structural_fusion_placement": trained_placement,
            "structural_routing": structural_config["structural_routing"],
            "structural_gate_scale": float(
                structural_config["selection_eval_gate_scale"]
            ),
            "structural_memory_value_scale": float(
                structural_config["selection_eval_memory_value_scale"]
            ),
        }
        for key, trained_value in contract_runtime_values.items():
            cli_value = getattr(args, key)
            if cli_value is not None and cli_value != trained_value:
                raise ValueError(
                    f"--{key}={cli_value!r} conflicts with checkpoint "
                    f"value {trained_value!r}"
                )
            setattr(args, key, trained_value)
        for key in (
            "mem_dim", "max_slots", "every_n_layers", "xattn_heads",
            "xattn_dim_head", "xattn_ff_mult",
        ):
            cli_value = getattr(args, key)
            trained_value = int(structural_config[key])
            if cli_value is not None and cli_value != trained_value:
                raise ValueError(
                    f"--{key}={cli_value} conflicts with checkpoint value {trained_value}"
                )
            setattr(args, key, trained_value)
        args.xattn_enable_ff = bool(structural_config["xattn_enable_ff"])
        manifest_path = os.path.join(args.memory_dir, "memory_manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError(f"Memory bank is missing {manifest_path}")
        digest = hashlib.sha256()
        with open(manifest_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != structural_config["memory_manifest_sha256"]:
            raise ValueError(
                f"Memory manifest does not match the {args.stage} checkpoint"
            )
    else:
        contract_path = os.path.join(args.adapter_dir, "training_contract.json")
        if not os.path.isfile(contract_path):
            raise ValueError(f"Stage-1 checkpoint is missing {contract_path}")
        with open(contract_path, "r", encoding="utf-8") as handle:
            training_contract = json.load(handle)
        if (training_contract.get("schema") != "mailohls-training-contract-v1"
                or training_contract.get("stage") != "stage1"):
            raise ValueError("Stage-1 inference requires its exact Stage-1 training contract")
        for field in ("model", "model_revision"):
            trained = training_contract.get(field)
            supplied = getattr(args, field)
            if not trained or (supplied and supplied != trained):
                raise ValueError(f"--{field} conflicts with the Stage-1 contract")
            setattr(args, field, trained)
        trainables = training_contract.get("stage1_trainable_parameter_contract")
        if (training_contract.get("prompt_schema_version") != 1
                and (not isinstance(trainables, dict)
                     or trainables.get("schema") != "mailohls-stage1-trainables-v1")):
            raise ValueError("Stage-1 checkpoint is missing its trainable-parameter contract")
        args.mem_dim = args.mem_dim or 0
        args.max_slots = args.max_slots or 64
        args.structural_fusion_placement = None
        args.structural_routing = "exact_slot"
        args.structural_gate_scale = 1.0
        args.structural_memory_value_scale = 1.0
        args.embedding_weights_tied = bool(training_contract["embedding_weights_tied"])

    required_policies = {
        "prompt_schema_version": mailohls_contract.PROMPT_SCHEMA_VERSION,
        "response_prefix_policy": mailohls_contract.RESPONSE_PREFIX_POLICY,
        "clock_supervision_policy": mailohls_contract.CLOCK_SUPERVISION_POLICY,
        "directive_supervision_policy": mailohls_contract.DIRECTIVE_SUPERVISION_POLICY,
        "semantic_domain_policy": mailohls_contract.SEMANTIC_DOMAIN_POLICY,
    }
    is_historical = training_contract.get("prompt_schema_version") == 1
    mismatches = {} if is_historical else {
        name: {"checkpoint": training_contract.get(name), "runtime": value}
        for name, value in required_policies.items()
        if training_contract.get(name) != value
    }
    if mismatches:
        raise ValueError("Checkpoint response contract is incompatible: "
                         + json.dumps(mismatches, sort_keys=True))
    domain_digest = hashlib.sha256()
    with open(args.directive_domain_registry_json, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            domain_digest.update(chunk)
    if domain_digest.hexdigest() != training_contract.get("directive_domain_registry_sha256"):
        raise ValueError("Directive-domain registry does not match the checkpoint contract")

    cases = load_cases(args)
    mismatched = [case.kernel_name for case in cases
                  if training_contract["objective"] != "ALL"
                  and case.obj_mode != training_contract["objective"]]
    if mismatched:
        raise ValueError(
            f"Evaluation objective must match checkpoint objective {training_contract['objective']}; "
            f"mismatched cases: {mismatched[:10]}"
        )
    for case in cases:
        mailohls_contract.target_prompt_fields(case.platform_row)
    directive_domain_registry = load_directive_domain_registry(
        args.directive_domain_registry_json
    )
    tok = build_tokenizer(
        args.model,
        revision=args.model_revision,
    )
    expected_tokens = training_contract.get("special_tokens")
    expected_ids = training_contract.get("special_token_ids")
    actual_ids = (sorted(set(tok.convert_tokens_to_ids(expected_tokens)))
                  if isinstance(expected_tokens, list) else None)
    if len(tok) != training_contract.get("tokenizer_size") or actual_ids != expected_ids:
        raise ValueError("Tokenizer does not match the checkpoint training contract")
    model = load_stage_model(args, tok)
    if args.stage in {"stage2", "stage3"}:
        actual_layers = tuple(model.structural_xattn_layer_indices)
        if actual_layers != args.selected_xattn_layers_1based:
            raise ValueError(
                "Checkpoint structural layer selection does not match the "
                f"runtime backbone: trained={args.selected_xattn_layers_1based}, "
                f"runtime={actual_layers}"
            )
    mem_bank = (
        load_memory_bank(
            args.memory_dir,
            expected_mem_dim=args.mem_dim,
            expected_max_slots=args.max_slots,
        )
        if args.stage in {"stage2", "stage3"}
        else None
    )

    for n, p in model.named_parameters():
        if n.endswith("attn_gate"):
            print(n, float(p.item()), float(p.tanh().item()))

    outputs = []
    for idx, case in enumerate(cases, start=1):
        print(f"[CASE {idx}/{len(cases)}] kernel={case.kernel_name} obj={case.obj_mode}")
        row = predict_case(
            model=model,
            tok=tok,
            case=case,
            directive_domain_registry=directive_domain_registry,
            stage=args.stage,
            max_prompt_tokens=args.max_prompt_tokens,
            mem_bank=mem_bank,
            mem_dim=args.mem_dim,
            max_slots=args.max_slots,
            structural_routing=args.structural_routing,
            score_reduction=args.score_reduction,
            debug_topk=args.debug_topk,
            candidate_max_prefix_tokens=args.candidate_max_prefix_tokens,
            candidate_keep_head_tokens=args.candidate_keep_head_tokens,
            print_xattn_debug_flag=args.print_xattn_debug,
            num_samples=args.num_samples,
            no_greedy_first=args.no_greedy_first,
            sample_temperature=args.sample_temperature,
            sample_top_p=args.sample_top_p,
            sample_top_k=args.sample_top_k,
            sample_seed=args.sample_seed,
        )
        outputs.append(row)

        if args.print_predictions:
            print("-" * 100)
            print(
                f"[CASE SUMMARY] generated={row['n_generated']} "
                f"unique={row['n_unique']} "
                f"best_unique_sample_id_by_model_score={row['best_unique_sample_id_by_model_score']}"
            )
            for cand in row["candidates"]:
                print(
                    f"[SAMPLE {cand['sample_id']}] "
                    f"mode={cand['decode_mode']} "
                    f"is_unique={cand['is_unique']} "
                    f"dup_of={cand['duplicate_of_sample_id']} "
                    f"seq_score={cand['sequence_score']:.6f} "
                    f"seq_sum_logprob={cand['sequence_sum_logprob']:.6f}"
                )
                if "value_accuracy_over_expected" in cand:
                    print(f"  [ACC] value_accuracy_over_expected={cand['value_accuracy_over_expected']:.6f}")
                print(cand["canonical_prediction"])
                print("-" * 60)
            print("-" * 100)

    if args.output_jsonl:
        dump_jsonl(args.output_jsonl, outputs)
        print(f"[DONE] wrote JSONL -> {args.output_jsonl}")

    if args.output_json:
        payload = outputs[0] if len(outputs) == 1 else outputs
        dump_json(args.output_json, payload)
        print(f"[DONE] wrote JSON -> {args.output_json}")

    if not args.output_jsonl and not args.output_json:
        payload = outputs[0] if len(outputs) == 1 else outputs
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
