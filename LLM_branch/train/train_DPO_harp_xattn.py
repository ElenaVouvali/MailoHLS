from __future__ import annotations

import argparse
import copy
import re
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import numpy as np

from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import PeftModel, prepare_model_for_kbit_training

from LLM_branch.common import frozen_stage1, mailohls_contract



def build_prompt(mod, source_text: str, obj_mode: str, row: dict) -> str:
    return mod.build_prompt(source_text, obj_mode, row=row)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_stage2_contract(stage2_adapter_dir: str) -> dict:
    adapter_dir = Path(stage2_adapter_dir).resolve()
    contract_path = adapter_dir / "training_contract.json"
    structural_path = adapter_dir / "structural_xattn.pt"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Stage-2 adapter is missing {contract_path}"
        )
    if not structural_path.is_file():
        raise FileNotFoundError(
            f"Stage-2 adapter is missing {structural_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "mailohls-training-contract-v1":
        raise ValueError("Unsupported Stage-2 training contract schema")
    if contract.get("stage") != "stage2":
        raise ValueError(
            "--stage2_adapter_dir must contain a Stage-2 contract"
        )
    structural = contract.get("structural")
    if not isinstance(structural, dict):
        raise ValueError("Stage-2 contract has no structural section")
    required = {
        "mem_dim", "max_slots", "every_n_layers", "xattn_heads",
        "xattn_dim_head", "xattn_ff_mult", "xattn_enable_ff",
        "xattn_placement",
        "memory_manifest_sha256", "selected_xattn_layers_1based",
    }
    historical_v1 = structural.get("schema") == "mailohls-structural-config-v1"
    if not historical_v1:
        required.update({
            "selection_eval_gate_scale", "selection_eval_memory_value_scale",
            "structural_routing",
        })
    missing = sorted(required - set(structural))
    if missing:
        raise ValueError(
            f"Stage-2 structural contract is missing: {missing}"
        )
    production_fields = {
        "stage1_contract_sha256", "stage1_adapter_sha256", "special_token_sha256",
        "stage1_lora_sha256", "frozen_stage1_sha256", "embedding_mode",
        "action_relation_schema", "apply_mask_policy", "xattn_gate_init",
        "loss_policy", "trainable_parameter_contract",
        "normalization_artifact_sha256",
    }
    if "trainable_parameter_contract" in structural:
        missing_production = sorted(production_fields - set(structural))
        if missing_production:
            raise ValueError(
                "Stage-2 production structural contract is missing: "
                + ", ".join(missing_production)
            )
        # Stage 3 consumes the frozen Stage-2 weights and lineage.  A
        # non-negative auxiliary candidate-ranking loss used to obtain those
        # weights does not make the checkpoint incompatible with DPO.
        loss_policy = structural["loss_policy"]
        ce_weight = float(loss_policy.get("ce_loss_weight", -1.0))
        candidate_weight = float(
            loss_policy.get("candidate_loss_weight", -1.0)
        )
        if ce_weight != 1.0 or candidate_weight < 0.0:
            raise ValueError(
                "Stage-3 requires Stage-2 RHS CE weight 1.0 and a "
                "non-negative candidate-loss weight"
            )
        top_loss = contract.get("stage2_loss")
        if isinstance(top_loss, dict) and (
            float(top_loss.get("ce_loss_weight", -1.0)) != ce_weight
            or float(top_loss.get("candidate_loss_weight", -1.0))
            != candidate_weight
        ):
            raise ValueError(
                "Stage-2 loss-policy lineage is internally inconsistent"
            )
        if structural["apply_mask_policy"] != "rhs_only_causal_logit_positions":
            raise ValueError("Stage-3 requires the Stage-2 RHS-only structural apply mask")
    if bool(structural["xattn_enable_ff"]):
        raise ValueError("Stage-3 currently requires xattn FF disabled")
    if (
        float(structural.get("selection_eval_gate_scale", 1.0)) != 1.0
        or float(structural.get("selection_eval_memory_value_scale", 1.0)) != 1.0
    ):
        raise ValueError(
            "Stage-3 requires deployment-equivalent Stage-2 scales of 1; "
            "bake diagnostic gate scaling into the checkpoint first"
        )
    return contract


def apply_parent_contract(mod, args, contract: dict) -> None:
    """Make Stage-3 preprocessing/model construction inherit Stage 2."""
    expected_objective = str(contract["objective"])
    if args.objective and args.objective != expected_objective:
        raise ValueError(
            f"--objective={args.objective!r} conflicts with Stage-2 "
            f"objective {expected_objective!r}"
        )
    args.objective = expected_objective

    for name in ("model", "model_revision"):
        trained = contract.get(name)
        supplied = getattr(args, name)
        if supplied and supplied != trained:
            raise ValueError(
                f"--{name}={supplied!r} conflicts with Stage-2 value "
                f"{trained!r}"
            )
        setattr(args, name, trained)

    args.device_mode = contract["device_mode"]
    args.resource_budget_mode = contract["resource_budget_mode"]
    args.resource_budget_fracs = contract["resource_budget_fracs"]
    args.random_budgets_per_case = int(
        contract["random_budgets_per_case"]
    )
    args.random_budget_min_frac = float(
        contract["random_budget_min_frac"]
    )
    args.min_feasible_candidates_per_budget = int(
        contract["min_feasible_candidates_per_budget"]
    )
    args.candidate_pool_per_objective = int(
        contract["candidate_pool_per_objective"]
    )
    args.auto_frequency_fraction = float(
        contract["auto_frequency_fraction"]
    )
    args.min_auto_clock_count = int(contract["min_auto_clock_count"])
    args.goal_domination_penalty = float(
        contract["goal_domination_penalty"]
    )
    args.goal_max_dominated_gap = float(
        contract["goal_max_dominated_gap"]
    )
    args.score_weight_min = float(contract["score_weight_min"])
    args.score_weight_power = float(contract["score_weight_power"])
    args.min_supervised_sites = int(contract["min_supervised_sites"])
    args.min_site_coverage = float(contract["min_site_coverage"])

    mod.TARGET_CFG.device_mode = args.device_mode
    mod.TARGET_CFG.budget_mode = args.resource_budget_mode
    mod.TARGET_CFG.random_budgets_per_case = args.random_budgets_per_case
    mod.TARGET_CFG.min_budget_frac = args.random_budget_min_frac
    mod.TARGET_CFG.min_feasible_candidates = (
        args.min_feasible_candidates_per_budget
    )
    mod.TARGET_CFG.candidate_pool_per_objective = (
        args.candidate_pool_per_objective
    )
    mod.TARGET_CFG.auto_frequency_fraction = (
        args.auto_frequency_fraction
    )
    mod.TARGET_CFG.min_auto_clock_count = args.min_auto_clock_count
    mod.TARGET_CFG.effective_area_floor = float(
        contract.get("effective_area_floor", 1e-12)
    )
    mod.TARGET_CFG.strict_source_markers = True
    # Random-budget case construction must remain tied to the Stage-2 seed;
    # the ordinary seed may vary for replication training runs.
    mod.TARGET_CFG.seed = int(getattr(args, "stage2_seed", args.seed))

    structural = contract["structural"]
    args.mem_dim = int(structural["mem_dim"])
    args.max_slots = int(structural["max_slots"])
    args.every_n_layers = int(structural["every_n_layers"])
    args.xattn_heads = int(structural["xattn_heads"])
    args.xattn_dim_head = int(structural["xattn_dim_head"])
    args.xattn_ff_mult = int(structural["xattn_ff_mult"])
    args.structural_fusion_placement = {
        "post_self_attn_pre_mlp": "post_self_attention_residual"
    }.get(structural["xattn_placement"], structural["xattn_placement"])
    args.structural_routing = structural.get("structural_routing", "exact_slot")
    args.structural_gate_scale = float(
        structural.get("selection_eval_gate_scale", 1.0)
    )
    args.structural_memory_value_scale = float(
        structural.get("selection_eval_memory_value_scale", 1.0)
    )
    args.selected_xattn_layers_1based = tuple(
        int(x) for x in structural["selected_xattn_layers_1based"]
    )
    args.stage2_structural_contract_sha256 = canonical_json_sha256(structural)


def build_stage3_contract(
    mod,
    args,
    parent_contract: dict,
    train_pairs: List[dict],
    val_pairs: List[dict],
    test_pairs: List[dict],
) -> dict:
    contract = copy.deepcopy(parent_contract)
    contract["stage"] = "stage3"
    contract["git_commit"] = mod.current_git_commit()
    adapter_dir = Path(args.stage2_adapter_dir).resolve()
    contract["parent_stage2"] = {
        "adapter_dir": str(adapter_dir),
        "contract_sha256": file_sha256(
            adapter_dir / "training_contract.json"
        ),
        "structural_xattn_sha256": file_sha256(
            adapter_dir / "structural_xattn.pt"
        ),
        "git_commit": parent_contract.get("git_commit"),
        "structural_contract_sha256": canonical_json_sha256(parent_contract["structural"]),
    }
    contract["stage3_preference"] = {
        "schema": "mailohls-stage3-preference-v1",
        "algorithm": "reference_dpo",
        "deployment_role": "final_focused_dpo_ablation",
        "production_model_stage": "stage2",
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "sft_alpha": args.sft_alpha,
        "logp_reduction": (
            "semantic_action_member_mean"
            if args.dpo_logp_reduction == "mean" and (args.dpo_pair_unit == "semantic_action" or args.dpo_semantic_action_only)
            else "changed_site_mean"
            if args.dpo_logp_reduction == "mean"
            else args.dpo_logp_reduction
        ),
        "candidate_top_k": args.top_k,
        "chosen_top_k": args.dpo_chosen_top_k,
        "hard_window": args.dpo_hard_window,
        "hard_negatives_per_chosen": (
            args.dpo_hard_negatives_per_chosen
        ),
        "medium_negatives_per_chosen": (
            args.dpo_medium_negatives_per_chosen
        ),
        "min_score_gap": args.dpo_min_score_gap,
        "hard_gap_max": args.dpo_hard_gap_max,
        "medium_gap_max": args.dpo_medium_gap_max,
        "min_primary_rel_gain": args.dpo_min_primary_rel_gain,
        "min_edit_distance": args.dpo_min_edit_distance,
        "min_edit_frac": args.dpo_min_edit_frac,
        "max_edit_frac": args.dpo_max_edit_frac,
        "require_chosen_rank0": args.dpo_require_chosen_rank0,
        "max_edit_distance": args.dpo_max_edit_distance,
        "pair_unit": args.dpo_pair_unit,
        "min_action_distance": args.dpo_min_action_distance,
        "max_action_distance": args.dpo_max_action_distance,
        "max_reference_margin": args.dpo_max_reference_margin,
        "semantic_action_scoring": (args.dpo_pair_unit == "semantic_action" or args.dpo_semantic_action_only),
        "pair_weighting": "clip(sqrt(adp_rel_gain / median_gain), 0.5, 2.0)",
        "pair_weight_median_adp_gain": float(np.median([
            max(float(p.get("adp_rel_gain", 0.0)), 1e-8) for p in train_pairs
        ])) if train_pairs else None,
        "require_same_supervised_schema": (
            args.require_same_supervised_schema
        ),
        "pair_counts": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "train_contexts": len({
            pair["context_id"] for pair in train_pairs
        }),
    }
    contract["stage3_runtime"] = {
        "train_script_sha256": file_sha256(__file__),
        "git_dirty": mod.git_is_dirty(),
        "argv": list(sys.argv),
        "hostname": os.uname().nodename,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": args.seed,
        "stage2_seed": args.stage2_seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "selection_eval_steps": args.selection_eval_steps,
        "save_steps": args.save_steps,
        "pair_file_hashes": {
            "train": file_sha256(Path(args.output_dir) / "pair_debug" / "train_pairs.jsonl"),
            "val": file_sha256(Path(args.output_dir) / "pair_debug" / "val_pairs.jsonl") if val_pairs else None,
        },
    }
    contract["stage3_trainables"] = {
        "lora": bool(args.train_lora_dpo),
        "xattn": bool(args.train_xattn_dpo),
        "attn_gate": bool(args.train_attn_gate_dpo),
        "ff_gate": bool(args.train_ff_gate_dpo),
        "special_token_embeddings": bool(
            args.train_special_token_embeddings
        ),
    }
    contract["stage3_learning_rates"] = {
        "lora": args.lr_lora,
        "xattn": args.lr_xattn,
        "gate": args.lr_gate,
        "ff": args.lr_ff,
        "ff_gate": args.lr_gate_ff,
        "embedding": args.lr_embed,
    }
    return contract

def import_module_from_path(module_path: str, module_name: str = "sft_mod"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution.  Dataclasses (Python 3.10+) resolve
    # postponed/forward annotations through sys.modules; dynamic execution
    # without this registration raises ``NoneType has no attribute __dict__``.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module

def build_selected_splits(mod, args, rows, parent_contract: dict):
    if args.split_json:
        split_spec = mod.load_split_spec(args.split_json)
        raw_train_rows, raw_val_rows, raw_test_rows = mod.apply_split_spec(rows, split_spec)
        print(f"[INFO] Loaded split from {args.split_json}")

    elif args.split_mode == "family":
        val_fams = {mod.normalize_name(x) for x in args.val_families.split(";") if x.strip()}
        test_fams = {mod.normalize_name(x) for x in args.test_families.split(";") if x.strip()}
        raw_train_rows, raw_val_rows, raw_test_rows = mod.split_by_family(rows, val_fams, test_fams)
        print("[INFO] val_families:", sorted(val_fams))
        print("[INFO] test_families:", sorted(test_fams))

    else:
        raw_train_rows, raw_val_rows, raw_test_rows = mod.split_rows_random_design(
            rows,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            stratify_by_kernel=args.stratify_by_kernel,
        )
        print(
            f"[INFO] random design-point split with "
            f"val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, "
            f"split_seed={args.split_seed}, stratify_by_kernel={args.stratify_by_kernel}"
        )

    if args.save_split_json:
        mod.save_split_spec(args.save_split_json, raw_train_rows, raw_val_rows, raw_test_rows)
        print(f"[INFO] Saved split spec -> {args.save_split_json}")

    if args.split_mode == "family":
        # Stage 3 may consume the locked Stage-1/Stage-2 family split, whose
        # sealed test set intentionally contains one family.  Preserve all
        # leakage/disjointness checks, but do not apply Stage-1's minimum-size
        # qualification to this downstream DPO preflight.
        mod.assert_family_split_contract(
            raw_train_rows,
            raw_val_rows,
            raw_test_rows,
            minimum_validation_families=1,
            minimum_test_families=1,
        )

    split_payload = {
        name: sorted(int(row["_jsonl_idx"]) for row in split_rows)
        for name, split_rows in (
            ("train", raw_train_rows),
            ("val", raw_val_rows),
            ("test", raw_test_rows),
        )
    }
    split_digest = canonical_json_sha256(split_payload)
    if split_digest != parent_contract.get("split_sha256"):
        raise ValueError(
            "Stage-3 split does not match the Stage-2 training contract: "
            f"{split_digest} != {parent_contract.get('split_sha256')}"
        )

    budget_seed = int(getattr(args, "stage2_seed", args.seed))
    eval_seed = budget_seed + 10_000
    if args.resource_budget_mode == "fixed":
        fractions = mod.parse_resource_budget_fracs(
            args.resource_budget_fracs
        )
        raw_train_rows = mod.augment_rows_with_resource_budgets(
            raw_train_rows, fractions
        )
        raw_val_rows = mod.augment_rows_with_resource_budgets(
            raw_val_rows, fractions
        )
    elif args.resource_budget_mode == "random":
        raw_train_rows = mod.augment_rows_with_random_resource_budgets(
            raw_train_rows,
            num_budgets_per_case=args.random_budgets_per_case,
            seed=budget_seed,
            min_feasible_candidates=(
                args.min_feasible_candidates_per_budget
            ),
        )
        raw_val_rows = mod.augment_rows_with_random_resource_budgets(
            raw_val_rows,
            num_budgets_per_case=args.random_budgets_per_case,
            seed=eval_seed,
            min_feasible_candidates=args.min_feasible_candidates_per_budget,
        )
    else:
        raise ValueError(
            f"Unsupported resource_budget_mode={args.resource_budget_mode!r}"
        )

    train_rows, _ = mod.select_goal_rows(
        raw_train_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    val_rows, _ = mod.select_goal_rows(
        raw_val_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    print(f"[INFO] Selected split sizes: train={len(train_rows)} val={len(val_rows)}; "
          f"held-out test rows not selected={len(raw_test_rows)}")
    return train_rows, val_rows, []


def dump_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)

def parse_target_map(target_text: str) -> Dict[str, str]:
    """
    Parse:
        auto{_PIPE_L3} = 1
    into:
        {"auto{_PIPE_L3}": "1"}
    """
    out = {}
    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = ASSIGN_RE.match(line)
        if m is None:
            continue
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        out[lhs] = rhs
    return out


def relative_improvement(old_val: float, new_val: float) -> float:
    """
    Positive means 'new_val' is better (smaller) than 'old_val'.
    """
    old_val = float(old_val)
    new_val = float(new_val)
    denom = max(abs(old_val), 1e-12)
    return (old_val - new_val) / denom


class GoalPreferencePairBuilder:
    def __init__(
        self,
        mod,
        objective: str,
        chosen_top_k: int = 3,
        hard_window: int = 8,
        hard_negatives_per_chosen: int = 2,
        medium_negatives_per_chosen: int = 1,
        min_score_gap: float = 0.02,
        hard_gap_max: float = 0.15,
        medium_gap_max: float = 0.35,
        min_primary_rel_gain: float = 0.02,
        min_edit_distance: int = 1,
        min_edit_frac: float = 0.0,
        max_edit_frac: float = 1.0,
        min_supervised_sites: int = 2,
        min_site_coverage: float = 0.85,
        require_same_supervised_schema: bool = True,
        balanced_min_sum_gain: float = 0.02,
        balanced_max_axis_loss: float = 0.25,
        balanced_min_better_axis_gain: float = 0.03,
        require_chosen_rank0: bool = False,
        max_edit_distance: int = 0,
        semantic_action_only: bool = False,
        pair_unit: str = "field",
        min_action_distance: int = 1,
        max_action_distance: int = 1,
    ):
        self.mod = mod
        self.objective = objective
        self.chosen_top_k = chosen_top_k
        self.hard_window = hard_window
        self.hard_negatives_per_chosen = hard_negatives_per_chosen
        self.medium_negatives_per_chosen = medium_negatives_per_chosen
        self.min_score_gap = float(min_score_gap)
        self.hard_gap_max = float(hard_gap_max)
        self.medium_gap_max = float(medium_gap_max)
        self.min_primary_rel_gain = float(min_primary_rel_gain)
        self.min_edit_distance = int(min_edit_distance)
        self.min_edit_frac = float(min_edit_frac)
        self.max_edit_frac = float(max_edit_frac)
        self.min_supervised_sites = int(min_supervised_sites)
        self.min_site_coverage = float(min_site_coverage)
        self.require_same_supervised_schema = bool(require_same_supervised_schema)
        self.balanced_min_sum_gain = float(balanced_min_sum_gain)
        self.balanced_max_axis_loss = float(balanced_max_axis_loss)
        self.balanced_min_better_axis_gain = float(balanced_min_better_axis_gain)
        self.require_chosen_rank0 = bool(require_chosen_rank0)
        self.max_edit_distance = int(max_edit_distance)
        self.semantic_action_only = bool(semantic_action_only)
        if pair_unit not in {"field", "semantic_action"}:
            raise ValueError("pair_unit must be field or semantic_action")
        self.pair_unit = pair_unit
        self.min_action_distance = int(min_action_distance)
        self.max_action_distance = int(max_action_distance)

    def _canonical_row(self, row: dict):
        try:
            completion, meta = self.mod.build_partial_deterministic_target_text(
                row["input"],
                row["target"],
                min_supervised_sites=self.min_supervised_sites,
            )
        except ValueError:
            return None

        if meta["coverage"] < self.min_site_coverage:
            return None

        rhs_map = parse_target_map(completion)
        schema_key = tuple(
            lhs for _, lhs in self.mod.extract_ordered_lhs_plan(row["input"])
            if lhs in rhs_map
        )

        prompt = build_prompt(
            self.mod,
            row["input"],
            self.objective,
            row,
        )
        return {
            "row": row,
            "prompt": prompt,
            "completion": completion,
            "rhs_map": rhs_map,
            "schema_key": schema_key,
            "score": float(row.get("_score", 1e9)),
            "rank": int(row.get("_rank_within_kernel", 10**9)),
            "num_sites": int(meta["n_supervised"]),
        }

    def _directive_diff(self, a: dict, b: dict):
        if a["schema_key"] != b["schema_key"]:
            return 0, 0.0
        keys = list(a["schema_key"])
        diff = sum(a["rhs_map"].get(k) != b["rhs_map"].get(k) for k in keys)
        frac = diff / max(1, len(keys))
        return int(diff), float(frac)

    def _rel_gains(self, chosen: dict, rejected: dict):
        ch_lat = float(chosen["row"]["latency"])
        rj_lat = float(rejected["row"]["latency"])
        ch_area = float(chosen["row"]["area"])
        rj_area = float(rejected["row"]["area"])

        lat_gain = (rj_lat - ch_lat) / max(abs(rj_lat), 1e-12)
        area_gain = (rj_area - ch_area) / max(abs(rj_area), 1e-12)
        area_floor = float(self.mod.TARGET_CFG.effective_area_floor)
        ch_adp = ch_lat * max(ch_area, area_floor)
        rj_adp = rj_lat * max(rj_area, area_floor)
        adp_gain = (rj_adp - ch_adp) / max(abs(rj_adp), 1e-12)
        return float(lat_gain), float(area_gain), float(adp_gain)

    def _primary_gain_ok(
        self,
        lat_gain: float,
        area_gain: float,
        adp_gain: float,
    ):
        if self.objective == "PARETO_LATENCY":
            return lat_gain >= self.min_primary_rel_gain
        if self.objective == "PARETO_AREA":
            return area_gain >= self.min_primary_rel_gain

        if self.objective != "PARETO_ADP":
            raise ValueError(f"Unsupported objective: {self.objective!r}")

        better_axis = max(lat_gain, area_gain)
        worse_axis = min(lat_gain, area_gain)

        return (
            better_axis >= self.balanced_min_better_axis_gain
            and worse_axis >= -self.balanced_max_axis_loss
            and adp_gain >= max(
                self.min_primary_rel_gain,
                self.balanced_min_sum_gain,
            )
        )

    def build(self, rows: List[dict]) -> List[dict]:
        # A DPO pair is valid only when chosen and rejected are completions for
        # the exact same prompt.  Grouping merely by kernel leaks device,
        # clock, and resource-budget changes into the preference label.
        by_context = defaultdict(list)
        for row in rows:
            rec = self._canonical_row(row)
            if rec is not None:
                by_context[rec["prompt"]].append(rec)

        pairs = []

        for prompt, context_rows in by_context.items():
            kernel_names = {rec["row"]["kernel_name"] for rec in context_rows}
            if len(kernel_names) != 1:
                raise RuntimeError(
                    "A canonical prompt unexpectedly spans multiple kernels"
                )
            kernel_name = next(iter(kernel_names))
            ranked_rows = sorted(
                context_rows,
                key=lambda rec: (rec["rank"], rec["score"]),
            )

            uniq = []
            seen = set()
            for rec in ranked_rows:
                if rec["completion"] in seen:
                    continue
                seen.add(rec["completion"])
                uniq.append(rec)

            if len(uniq) < 2:
                continue

            if self.require_chosen_rank0:
                chosen_pool = [rec for rec in uniq if rec["rank"] == 0][:1]
            else:
                chosen_pool = uniq[: min(self.chosen_top_k, len(uniq))]

            for chosen_idx, chosen in enumerate(chosen_pool):
                hard_pool = []
                medium_pool = []

                for rejected in uniq[chosen_idx + 1:]:
                    gap = float(rejected["score"] - chosen["score"])
                    if gap > self.medium_gap_max:
                        break

                    if self.require_same_supervised_schema and chosen["schema_key"] != rejected["schema_key"]:
                        continue

                    lat_gain, area_gain, adp_gain = self._rel_gains(
                        chosen, rejected
                    )
                    if not self._primary_gain_ok(
                        lat_gain, area_gain, adp_gain
                    ):
                        continue
                    if gap < self.min_score_gap:
                        continue

                    diff_count, diff_frac = self._directive_diff(chosen, rejected)
                    if diff_count < self.min_edit_distance:
                        continue
                    if self.max_edit_distance > 0 and diff_count > self.max_edit_distance:
                        continue
                    if diff_frac < self.min_edit_frac or diff_frac > self.max_edit_frac:
                        continue
                    changed_lhs = {
                        lhs for lhs in chosen["schema_key"]
                        if chosen["rhs_map"].get(lhs) != rejected["rhs_map"].get(lhs)
                    }
                    changed_actions = {
                        mailohls_contract.directive_preference_action_key(lhs)
                        for lhs in changed_lhs
                    }
                    if (self.semantic_action_only or self.pair_unit == "semantic_action") and not (
                        self.min_action_distance <= len(changed_actions) <= self.max_action_distance
                    ):
                        continue
                    changed_action = next(iter(changed_actions), None)
                    if changed_action is not None:
                        representative_lhs = next(iter(changed_lhs))
                        expected_members = {
                            x.upper() for x in mailohls_contract.directive_preference_action_members(representative_lhs)
                        }
                        schema_members = {str(x).strip().upper() for x in chosen["schema_key"]}
                        if not expected_members.issubset(schema_members):
                            continue
                        action_members = sorted(expected_members)
                    else:
                        action_members = []

                    rec = {
                        "kernel_name": kernel_name,
                        "family": chosen["row"].get("_family"),
                        "source_text": chosen["row"]["input"],
                        "obj_mode": self.objective,
                        "prompt": prompt,
                        "context_id": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "platform_row": {
                            key: chosen["row"].get(key)
                            for key in (
                                "kernel_name", "device", "Device",
                                "clock_period", "Clock_Period_nsec",
                                "selected_clock_period", "frequency_mode",
                                "available_clock_periods", "resource_budget_id",
                                "avail_bram", "avail_dsp", "avail_ff", "avail_lut",
                            )
                            if key in chosen["row"]
                        },
                        "chosen_clock_row": {
                            key: chosen["row"].get(key)
                            for key in (
                                "clock_period", "Clock_Period_nsec",
                                "selected_clock_period", "frequency_mode",
                            )
                            if key in chosen["row"]
                        },
                        "rejected_clock_row": {
                            key: rejected["row"].get(key)
                            for key in (
                                "clock_period", "Clock_Period_nsec",
                                "selected_clock_period", "frequency_mode",
                            )
                            if key in rejected["row"]
                        },
                        "chosen": chosen["completion"],
                        "rejected": rejected["completion"],
                        "chosen_score": float(chosen["score"]),
                        "rejected_score": float(rejected["score"]),
                        "score_gap": float(gap),
                        "directive_diff_count": int(diff_count),
                        "directive_diff_frac": float(diff_frac),
                        "semantic_action_diff_count": int(len(changed_actions)),
                        "changed_lhs": sorted(changed_lhs),
                        "preference_action_type": changed_action[0] if changed_action else None,
                        "preference_action_label": changed_action[1] if changed_action else None,
                        "preference_action_members": action_members,
                        "semantic_action_only": self.semantic_action_only,
                        "latency_rel_gain": float(lat_gain),
                        "area_rel_gain": float(area_gain),
                        "adp_rel_gain": float(adp_gain),
                        "chosen_rank": int(chosen["rank"]),
                        "rejected_rank": int(rejected["rank"]),
                        "num_sites": int(chosen["num_sites"]),
                    }

                    if (rejected["rank"] - chosen["rank"]) <= self.hard_window and gap <= self.hard_gap_max:
                        rec["pair_tier"] = "hard"
                        hard_pool.append(rec)
                    else:
                        rec["pair_tier"] = "medium"
                        medium_pool.append(rec)

                selected = (
                    hard_pool[: self.hard_negatives_per_chosen]
                    + medium_pool[: self.medium_negatives_per_chosen]
                )

                if not selected:
                    fallback = hard_pool if hard_pool else medium_pool
                    if fallback:
                        selected = [fallback[0]]

                pairs.extend(selected)

        dedup = []
        seen = set()
        for p in pairs:
            key = (
                p["context_id"], p["obj_mode"],
                p["chosen"], p["rejected"],
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        return dedup

    

def _q(vals, q):
    if not vals:
        return None
    return float(np.quantile(np.array(vals, dtype=np.float64), q))

def compute_dpo_metrics(eval_pred):
    predictions = np.asarray(eval_pred.predictions)
    if predictions.ndim == 1:
        # Backward compatibility with cached evaluations produced before the
        # absolute-margin diagnostics were added.
        dpo_margin = predictions.reshape(-1)
        return {
            "preference_accuracy": float((dpo_margin > 0).mean()),
            "preference_margin_mean": float(dpo_margin.mean()),
            "preference_margin_p10": float(np.quantile(dpo_margin, 0.10)),
        }
    dpo_margin = predictions[..., 0].reshape(-1)
    policy_margin = predictions[..., 1].reshape(-1)
    reference_margin = predictions[..., 2].reshape(-1)
    negative_reference = reference_margin < 0
    return {
        "preference_accuracy": float((dpo_margin > 0).mean()),
        "preference_margin_mean": float(dpo_margin.mean()),
        "preference_margin_p10": float(np.quantile(dpo_margin, 0.10)),
        "policy_pair_accuracy": float((policy_margin > 0).mean()),
        "reference_pair_accuracy": float((reference_margin > 0).mean()),
        "negative_reference_margin_count": int(negative_reference.sum()),
        "negative_reference_margins_crossed_zero": int(
            (negative_reference & (policy_margin > 0)).sum()
        ),
    }


def audit_preference_pairs(name: str, rows: List[dict]) -> None:
    print(f"\n[PAIR-AUDIT] {name}")
    if not rows:
        print("  no pairs")
        return

    by_obj = Counter(r["obj_mode"] for r in rows)
    by_kernel_obj = Counter((r["kernel_name"], r["obj_mode"]) for r in rows)
    by_tier = Counter(r.get("pair_tier", "unknown") for r in rows)

    gaps = [float(r["score_gap"]) for r in rows]
    diff_counts = [int(r["directive_diff_count"]) for r in rows]
    diff_fracs = [float(r["directive_diff_frac"]) for r in rows]
    lat_gains = [float(r["latency_rel_gain"]) for r in rows if r["obj_mode"] == "PARETO_LATENCY"]
    area_gains = [float(r["area_rel_gain"]) for r in rows if r["obj_mode"] == "PARETO_AREA"]

    print(f"  total pairs                 : {len(rows)}")
    print(f"  kernel-objective buckets    : {len(by_kernel_obj)}")
    print(f"  pairs by objective          : {dict(by_obj)}")
    print(f"  pairs by tier               : {dict(by_tier)}")

    print(f"  score_gap q10/q50/q90       : {_q(gaps, 0.10):.4f} / {_q(gaps, 0.50):.4f} / {_q(gaps, 0.90):.4f}")
    print(f"  diff_count q10/q50/q90      : {_q(diff_counts, 0.10):.1f} / {_q(diff_counts, 0.50):.1f} / {_q(diff_counts, 0.90):.1f}")
    print(f"  diff_frac q10/q50/q90       : {_q(diff_fracs, 0.10):.3f} / {_q(diff_fracs, 0.50):.3f} / {_q(diff_fracs, 0.90):.3f}")

    if lat_gains:
        print(f"  LATENCY_EXTREME latency gain q10/q50/q90 : {_q(lat_gains, 0.10):.3f} / {_q(lat_gains, 0.50):.3f} / {_q(lat_gains, 0.90):.3f}")
    if area_gains:
        print(f"  AREA_EXTREME area gain q10/q50/q90   : {_q(area_gains, 0.10):.3f} / {_q(area_gains, 0.50):.3f} / {_q(area_gains, 0.90):.3f}")


def preview_preference_pairs(rows: List[dict], n: int = 3) -> None:
    print(f"\n[PAIR-PREVIEW] showing {min(n, len(rows))} pairs")
    for i, ex in enumerate(rows[:n]):
        ch = parse_target_map(ex["chosen"])
        rj = parse_target_map(ex["rejected"])
        changed = [(k, ch.get(k), rj.get(k)) for k in sorted(ch.keys()) if ch.get(k) != rj.get(k)]

        print("\n" + "-" * 100)
        print(f"[{i}] kernel={ex['kernel_name']} obj={ex['obj_mode']}")
        print(f"score_gap={ex['score_gap']:.4f}  diff_count={ex['directive_diff_count']}  diff_frac={ex['directive_diff_frac']:.3f}")
        print(f"latency_rel_gain={ex['latency_rel_gain']:.3f}  area_rel_gain={ex['area_rel_gain']:.3f}")
        print("changed directives:")
        for k, v_ch, v_rj in changed[:12]:
            print(f"  {k}: chosen={v_ch} | rejected={v_rj}")
        if len(changed) > 12:
            print(f"  ... {len(changed) - 12} more")


class DPOPreferenceDataset(Dataset):
    def __init__(
        self,
        mod,
        rows: List[dict],
        tokenizer,
        max_length: int,
        value_weight: float = 1.0,
        directive_domain_registry: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ):
        self.mod = mod
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.value_weight = float(value_weight)
        self.directive_domain_registry = directive_domain_registry
        self.samples: List[dict] = []
        self.kernel_names: List[str] = []
        self.families: List[str] = []
        self.lengths: List[int] = []
        source_token_ids = set(
            tokenizer.convert_tokens_to_ids(mod.SOURCE_PLACEHOLDER_TOKENS)
        )

        for ex in rows:
            header, kernel, suffix, fields = mod.build_prompt_sections(
                ex["source_text"],
                ex["obj_mode"],
                row=ex["platform_row"],
            )
            if (
                header + kernel + suffix
                != ex["prompt"]
            ):
                raise RuntimeError(
                    "Stored DPO prompt is not canonical for its platform row"
                )
            header_ids = tokenizer(
                header, add_special_tokens=False
            )["input_ids"]
            code_ids = tokenizer(
                kernel, add_special_tokens=False
            )["input_ids"]
            suffix_ids = tokenizer(
                suffix, add_special_tokens=False
            )["input_ids"]

            chosen = self._pack_prompt_and_completion(
                header_ids=header_ids,
                code_ids=code_ids,
                suffix_ids=suffix_ids,
                source_token_ids=source_token_ids,
                source_text=ex["source_text"],
                completion_text=ex["chosen"],
                clock_row=ex["chosen_clock_row"],
                kernel_name=ex["kernel_name"],
                required_tokens=(
                    self.mod.GOALS[ex["obj_mode"]]["token"],
                    fields["device_token"],
                    fields["period_token"],
                ),
            )
            rejected = self._pack_prompt_and_completion(
                header_ids=header_ids,
                code_ids=code_ids,
                suffix_ids=suffix_ids,
                source_token_ids=source_token_ids,
                source_text=ex["source_text"],
                completion_text=ex["rejected"],
                clock_row=ex["rejected_clock_row"],
                kernel_name=ex["kernel_name"],
                required_tokens=(
                    self.mod.GOALS[ex["obj_mode"]]["token"],
                    fields["device_token"],
                    fields["period_token"],
                ),
            )

            # Preference credit is restricted to RHS sites whose assignment
            # actually changed.  Keep the complete RHS weighting separately
            # for the chosen-response SFT anchor.
            chosen_map = {k.strip().upper(): v.strip() for k, v in parse_target_map(ex["chosen"]).items()}
            rejected_map = {k.strip().upper(): v.strip() for k, v in parse_target_map(ex["rejected"]).items()}
            changed_lhs = {k for k, value in chosen_map.items() if rejected_map.get(k) != value}
            semantic_scoring = (ex.get("pair_unit") == "semantic_action" or ex.get("semantic_action_only", False))
            if semantic_scoring:
                scored_lhs = {str(x).strip().upper() for x in ex["preference_action_members"]}
            else:
                scored_lhs = changed_lhs
            for pack in (chosen, rejected):
                sites = pack.get("site_keys", [None] * pack["score_weights"].numel())
                pack["anchor_score_weights"] = pack["score_weights"].clone()
                base = pack["score_weights"].tolist()
                site_mass = defaultdict(float)
                for weight, site in zip(base, sites):
                    site_key = str(site).strip().upper() if site is not None else None
                    if site_key in scored_lhs and weight > 0:
                        site_mass[site_key] += float(weight)
                if not any(site_mass.values()):
                    raise RuntimeError("DPO pair has no scored changed sites")
                pack["dpo_score_weights"] = torch.tensor(
                    [float(weight) / site_mass[(str(site).strip().upper() if site is not None else None)]
                     if (str(site).strip().upper() if site is not None else None) in scored_lhs and site_mass[(str(site).strip().upper() if site is not None else None)] > 0 else 0.0
                     for weight, site in zip(base, sites)],
                    dtype=torch.float32,
                )

            self.samples.append({
                "kernel_name": ex["kernel_name"],
                "chosen": chosen,
                "rejected": rejected,
                "pair_weight": float(ex.get("pair_weight", 1.0)),
            })
            self.kernel_names.append(ex["kernel_name"])
            self.families.append(ex.get("family") or ex["kernel_name"])
            self.lengths.append(max(chosen["length"], rejected["length"]))

    def _pack_prompt_and_completion(
        self,
        *,
        header_ids,
        code_ids,
        suffix_ids,
        source_token_ids,
        source_text: str,
        completion_text: str,
        clock_row: Mapping[str, Any],
        kernel_name: str,
        required_tokens,
    ) -> Dict[str, torch.Tensor]:
        det_pack = self.mod.build_deterministic_rhs_pack(
            source_text,
            completion_text,
            self.tok,
            value_w=self.value_weight,
            directive_domain_registry=self.directive_domain_registry,
            kernel_name=kernel_name,
        )

        clock = self.mod.build_clock_pack(
            clock_row,
            self.tok,
            value_w=self.value_weight,
        )
        target_ids = clock.input_ids + det_pack.input_ids
        target_xmask = (
            clock.xattn_target_mask + det_pack.xattn_target_mask
        )
        score_clock = (
            str(clock_row.get("frequency_mode", "specified")).lower()
            == "auto"
        )
        clock_score_mask = [
            int(label != -100) if score_clock else 0
            for label in clock.labels
        ]
        target_score_weights = list(clock.token_weights) + list(det_pack.token_weights)

        prompt_budget = self.max_length - len(target_ids)
        fixed_prompt = len(header_ids) + len(suffix_ids)
        if prompt_budget <= fixed_prompt:
            raise ValueError(
                f"max_length={self.max_length} cannot preserve Stage-3 "
                f"target conditioning for {kernel_name}"
            )
        code_budget = prompt_budget - fixed_prompt
        kept_code = list(code_ids)
        if len(kept_code) > code_budget:
            head = code_budget // 2
            kept_code = (
                kept_code[:head]
                + kept_code[-(code_budget - head):]
            )
            before = sum(token in source_token_ids for token in code_ids)
            after = sum(token in source_token_ids for token in kept_code)
            if after != before:
                raise ValueError(
                    f"Context truncation drops {before - after} source "
                    f"markers for {kernel_name}; increase --max_length"
                )

        prompt_ids_kept = list(header_ids) + kept_code + list(suffix_ids)
        for token in required_tokens:
            token_id = self.tok.convert_tokens_to_ids(token)
            if token_id not in prompt_ids_kept:
                raise ValueError(
                    f"Required conditioning token {token} was lost"
                )

        input_ids = prompt_ids_kept + target_ids
        attention_mask = [1] * len(input_ids)

        # Compare only directive RHS values and, for AUTO-frequency rows,
        # the selected clock RHS.  Fixed schema and prompt tokens are excluded.
        score_weights = [0.0] * len(prompt_ids_kept) + target_score_weights

        # Same next-token routing convention as SFT.
        full_xattn_target_mask = (
            [0] * len(prompt_ids_kept) + list(target_xmask)
        )
        xattn_apply_mask = full_xattn_target_mask[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "score_weights": torch.tensor(score_weights, dtype=torch.float32),
            "site_keys": [None] * len(prompt_ids_kept) + list(clock.site_keys or []) + list(det_pack.site_keys or []),
            "xattn_apply_mask": torch.tensor(xattn_apply_mask, dtype=torch.float32),
            "routing_start_idx": torch.tensor(len(prompt_ids_kept), dtype=torch.long),
            "length": len(input_ids),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class DPOPairCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def _pad_1d(self, t: torch.Tensor, max_len: int, pad_value: int | float):
        if t.shape[0] == max_len:
            return t
        pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    def __call__(self, batch: List[dict]) -> Dict[str, Any]:
        shared_max_len = 0
        for ex in batch:
            shared_max_len = max(
                shared_max_len,
                ex["chosen"]["input_ids"].shape[0],
                ex["rejected"]["input_ids"].shape[0],
            )

        out = {
            "kernel_name": [ex["kernel_name"] for ex in batch],
        }

        for side in ["chosen", "rejected"]:
            out[f"{side}_input_ids"] = torch.stack([
                self._pad_1d(ex[side]["input_ids"], shared_max_len, self.tok.pad_token_id) for ex in batch
            ])
            out[f"{side}_attention_mask"] = torch.stack([
                self._pad_1d(ex[side]["attention_mask"], shared_max_len, 0) for ex in batch
            ])
            out[f"{side}_score_weights"] = torch.stack([
                self._pad_1d(ex[side]["dpo_score_weights"], shared_max_len, 0.0) for ex in batch
            ])
            out[f"{side}_anchor_score_weights"] = torch.stack([
                self._pad_1d(ex[side]["anchor_score_weights"], shared_max_len, 0.0) for ex in batch
            ])
            out[f"{side}_xattn_apply_mask"] = torch.stack([
                self._pad_1d(ex[side]["xattn_apply_mask"], shared_max_len, 0.0) for ex in batch
            ])
            out[f"{side}_routing_start_idx"] = torch.stack([
                ex[side]["routing_start_idx"] for ex in batch
            ])

        out["pair_weight"] = torch.tensor(
            [float(ex.get("pair_weight", 1.0)) for ex in batch], dtype=torch.float32
        )

        return out
    

class STRUCTURALDPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model,
        sft_mod,
        mem_bank: Dict[str, dict],
        mem_dim: int,
        max_slots: int,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        sft_alpha: float = 0.0,
        logp_reduction: str = "mean",
        structural_routing: str = "exact_slot",
        group_by_length: bool = False,
        lr_lora: float = 2e-5,
        lr_xattn: float = 5e-5,
        lr_gate: float = 2e-5,
        lr_ff: float = 0.0,
        lr_gate_ff: float = 0.0,
        lr_embed: float = 0.0,
        max_reference_margin: float | None = None,
        **kwargs,
    ):
        self.ref_model = ref_model
        self._sft_mod = sft_mod
        self.mem_bank = mem_bank
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.sft_alpha = float(sft_alpha)
        if logp_reduction not in {"sum", "mean"}:
            raise ValueError("logp_reduction must be 'sum' or 'mean'")
        self.logp_reduction = logp_reduction
        if structural_routing not in {"exact_slot", "compiler_relational"}:
            raise ValueError(
                f"Unsupported structural routing: {structural_routing!r}"
            )
        self.structural_routing = structural_routing
        self._group_by_length = bool(group_by_length)
        self.lr_lora = lr_lora
        self.lr_xattn = lr_xattn
        self.lr_gate = lr_gate
        self.lr_ff = lr_ff
        self.lr_gate_ff = lr_gate_ff
        self.lr_embed = lr_embed
        self.max_reference_margin = max_reference_margin
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lora_params, embed_params = [], []
        attn_gate_params, ff_gate_params = [], []
        xattn_attn_params, xattn_ff_params = [], []
        other_trainables = []

        input_emb_param_ids = set()
        output_emb_param_ids = set()

        try:
            emb = self.model.get_input_embeddings()
            input_emb_param_ids = {id(p) for p in emb.parameters()}
        except Exception:
            pass

        try:
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None:
                output_emb_param_ids = {id(p) for p in out_emb.parameters()}
        except Exception:
            pass

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in input_emb_param_ids or id(param) in output_emb_param_ids:
                embed_params.append(param)
            elif "lora_" in name:
                lora_params.append(param)
            elif name.endswith("attn_gate"):
                attn_gate_params.append(param)
            elif name.endswith("ff_gate"):
                ff_gate_params.append(param)
            elif "gated_cross_attn_layer.attn." in name:
                xattn_attn_params.append(param)
            elif "gated_cross_attn_layer.ff." in name:
                xattn_ff_params.append(param)
            else:
                other_trainables.append((name, param))

        opt_groups = []
        if lora_params and self.lr_lora > 0:
            opt_groups.append({"params": lora_params, "lr": self.lr_lora})
        if embed_params and self.lr_embed > 0:
            opt_groups.append({"params": embed_params, "lr": self.lr_embed})
        if attn_gate_params and self.lr_gate > 0:
            opt_groups.append({"params": attn_gate_params, "lr": self.lr_gate})
        if ff_gate_params and self.lr_gate_ff > 0:
            opt_groups.append({"params": ff_gate_params, "lr": self.lr_gate_ff})
        if xattn_attn_params and self.lr_xattn > 0:
            opt_groups.append({"params": xattn_attn_params, "lr": self.lr_xattn})
        if xattn_ff_params and self.lr_ff > 0:
            opt_groups.append({"params": xattn_ff_params, "lr": self.lr_ff})
        if other_trainables:
            bad_names = [n for n, _ in other_trainables]
            raise ValueError(
                "[OPT-DPO] Unexpected trainable parameters outside the allowed groups. "
                f"First 20: {bad_names[:20]}"
            )
        if not opt_groups:
            raise ValueError("[OPT-DPO] No positive-LR parameter groups")

        try:
            from bitsandbytes.optim import PagedAdamW8bit
            self.optimizer = PagedAdamW8bit(opt_groups, weight_decay=0.0)
        except Exception:
            self.optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.0)

        print(
            f"[OPT-DPO] lora={sum(p.numel() for p in lora_params):,} "
            f"embed={sum(p.numel() for p in embed_params):,} "
            f"attn_gate={sum(p.numel() for p in attn_gate_params):,} "
            f"ff_gate={sum(p.numel() for p in ff_gate_params):,} "
            f"xattn_attn={sum(p.numel() for p in xattn_attn_params):,} "
            f"xattn_ff={sum(p.numel() for p in xattn_ff_params):,}"
        )
        return self.optimizer

    def get_train_dataloader(self):
        # Kernel balancing is more important than length sorting for the
        # short, partial Stage-3 runs; length grouping otherwise reintroduces
        # pair-rich-kernel bias.
        sampler = self._kernel_weighted_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )

    def _kernel_weighted_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        dataset = self.train_dataset
        if isinstance(dataset, torch.utils.data.Subset):
            base = dataset.dataset
            indices = list(dataset.indices)
            names = [base.kernel_names[i] for i in indices]
            families = [base.families[i] for i in indices]
        else:
            names = list(getattr(dataset, "kernel_names", []))
            families = list(getattr(dataset, "families", []))
        counts = Counter(names)
        family_kernels = defaultdict(set)
        for family, kernel in zip(families, names):
            family_kernels[family].add(kernel)
        raw = torch.tensor([
            counts[kernel] ** -0.5 * len(family_kernels[family]) ** -0.5
            for family, kernel in zip(families, names)
        ], dtype=torch.double)
        weights = (raw / raw.median()).clamp_(0.25, 4.0)
        generator = torch.Generator().manual_seed(int(self.args.data_seed))
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    
    def _condition_structural_memory_from_kernel_names(self, model, kernel_names: List[str]):
        kvs, masks, relations = [], [], []
        for kernel_name in kernel_names:
            kv, mask, relation = (
                self._sft_mod.structural_memory_utils
                .get_structural_memory_pack_for_kernel(
                    self.mem_bank,
                    kernel_name,
                    self.max_slots,
                    self.mem_dim,
                    structural_routing=self.structural_routing,
                )
            )
            kvs.append(kv)
            masks.append(mask)
            relations.append(relation)

        mem_kv = torch.cat(kvs, dim=0)
        mem_m = torch.cat(masks, dim=0)
        relation_mask = None
        if self.structural_routing == "compiler_relational":
            if any(relation is None for relation in relations):
                raise RuntimeError(
                    "compiler_relational DPO received missing relations"
                )
            relation_mask = torch.cat(relations, dim=0)
        device = next(model.parameters()).device
        model.condition_structural_memory(
            mem_kv.to(device),
            mem_m.to(device),
            action_relation_mask=(
                relation_mask.to(device)
                if relation_mask is not None
                else None
            ),
        )

    def _clear_structural_memory_if_present(self, model):
        if hasattr(model, "clear_structural_memory"):
            model.clear_structural_memory()

    def _sequence_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        score_weights: torch.Tensor,
        anchor_weights: Optional[torch.Tensor],
        routing_start_idx: torch.Tensor,
        xattn_apply_mask: torch.Tensor,
        kernel_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Important:
        - For the TRAINABLE policy model under grad checkpointing, keep STRUCTURAL conditioned
        until backward finishes.
        - For the frozen ref model / eval no_grad path, clear immediately after forward.
        """
        keep_structural_memory_for_backward = bool(torch.is_grad_enabled() and model.training)

        if hasattr(model, "condition_structural_memory"):
            self._condition_structural_memory_from_kernel_names(model, kernel_names)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                routing_start_idx=routing_start_idx,
                xattn_apply_mask=xattn_apply_mask,
            )
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            weights = score_weights[:, 1:]
            anchor = anchor_weights[:, 1:] if anchor_weights is not None else weights

            log_probs = F.log_softmax(logits, dim=-1)
            token_logps = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            seq_logps = (token_logps * weights).sum(dim=-1)
            token_counts = weights.sum(dim=-1).clamp(min=1.0)
            anchor_logps = (token_logps * anchor).sum(dim=-1)
            anchor_counts = anchor.sum(dim=-1).clamp(min=1.0)
            return seq_logps, token_counts, anchor_logps, anchor_counts

        finally:
            # DO NOT clear the trainable policy model before backward checkpoint
            # recomputation has happened.
            if not keep_structural_memory_for_backward:
                self._clear_structural_memory_if_present(model)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        disable_lora_dropout_for_dpo(model)
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        try:
            if not torch.isfinite(loss.detach()).all().item():
                raise FloatingPointError("Non-finite Stage-3 loss before backward")
            loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
        finally:
            # Clear ONLY after backward so checkpoint recomputation still sees STRUCTURAL memory
            self._clear_structural_memory_if_present(model)
            self._clear_structural_memory_if_present(self.ref_model)

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kernel_names = inputs["kernel_name"]

        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_score_weights = inputs["chosen_score_weights"]
        chosen_routing_start_idx = inputs["chosen_routing_start_idx"]
        chosen_xattn_apply_mask = inputs["chosen_xattn_apply_mask"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_score_weights = inputs["rejected_score_weights"]
        rejected_routing_start_idx = inputs["rejected_routing_start_idx"]
        rejected_xattn_apply_mask = inputs["rejected_xattn_apply_mask"]
        cat_anchor_weights = torch.cat([inputs["chosen_anchor_score_weights"], inputs["rejected_anchor_score_weights"]], dim=0)

        batch_size = chosen_input_ids.shape[0]

        cat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        cat_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        cat_score_weights = torch.cat([chosen_score_weights, rejected_score_weights], dim=0)
        cat_routing_start_idx = torch.cat([chosen_routing_start_idx, rejected_routing_start_idx], dim=0)
        cat_xattn_apply_mask = torch.cat([chosen_xattn_apply_mask, rejected_xattn_apply_mask], dim=0)
        cat_kernel_names = list(kernel_names) + list(kernel_names)

        pi_logps, pi_token_counts, pi_anchor_logps, pi_anchor_token_counts = self._sequence_logps(
            model=model,
            input_ids=cat_input_ids,
            attention_mask=cat_attention_mask,
            score_weights=cat_score_weights,
            anchor_weights=cat_anchor_weights,
            routing_start_idx=cat_routing_start_idx,
            xattn_apply_mask=cat_xattn_apply_mask,
            kernel_names=cat_kernel_names,
        )

        with torch.no_grad():
            ref_logps, ref_token_counts, _, _ = self._sequence_logps(
                model=self.ref_model,
                input_ids=cat_input_ids,
                attention_mask=cat_attention_mask,
                score_weights=cat_score_weights,
                anchor_weights=None,
                routing_start_idx=cat_routing_start_idx,
                xattn_apply_mask=cat_xattn_apply_mask,
                kernel_names=cat_kernel_names,
            )

        ref_token_counts = ref_token_counts.clamp(min=1.0)
        pi_token_counts = pi_token_counts.clamp(min=1.0)

        if self.logp_reduction == "mean":
            pi_scores = pi_logps / pi_token_counts
            ref_scores = ref_logps / ref_token_counts
        else:
            pi_scores = pi_logps
            ref_scores = ref_logps

        pi_chosen, pi_rejected = pi_scores[:batch_size], pi_scores[batch_size:]
        ref_chosen, ref_rejected = ref_scores[:batch_size], ref_scores[batch_size:]
        chosen_token_counts = pi_token_counts[:batch_size]

        preference_logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
        policy_margin = pi_chosen - pi_rejected
        reference_margin = ref_chosen - ref_rejected
        if self.label_smoothing > 0.0:
            losses = (
                -(1.0 - self.label_smoothing) * F.logsigmoid(self.beta * preference_logits)
                - self.label_smoothing * F.logsigmoid(-self.beta * preference_logits)
            )
        else:
            losses = -F.logsigmoid(self.beta * preference_logits)

        pair_weights = inputs.get("pair_weight", torch.ones_like(losses)).to(losses.device)
        if self.max_reference_margin is not None:
            keep = reference_margin <= float(self.max_reference_margin)
            pair_weights = pair_weights * keep.to(pair_weights.dtype)
        loss = (pair_weights * losses).sum() / pair_weights.sum().clamp_min(1e-8)

        if self.sft_alpha > 0.0:
            chosen_nll = -(pi_anchor_logps[:batch_size] / pi_anchor_token_counts[:batch_size].clamp_min(1.0))
            loss = loss + self.sft_alpha * chosen_nll.mean()

        if return_outputs:
            outputs = {
                "losses": losses.detach(),
                "preference_logits": preference_logits.detach(),
                "dpo_margin": preference_logits.detach(),
                "policy_margin": policy_margin.detach(),
                "reference_margin": reference_margin.detach(),
                "pi_chosen": pi_chosen.detach(),
                "pi_rejected": pi_rejected.detach(),
                "ref_chosen": ref_chosen.detach(),
                "ref_rejected": ref_rejected.detach(),
            }
            return loss, outputs
        return loss


    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        is_dpo_batch = (
            isinstance(inputs, dict)
            and "chosen_input_ids" in inputs
            and "rejected_input_ids" in inputs
        )

        if not is_dpo_batch:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)

        try:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model,
                        inputs,
                        return_outputs=True,
                    )

            loss = loss.mean().detach()

            if prediction_loss_only:
                return loss, None, None

            logits = torch.stack(
                (
                    outputs["dpo_margin"],
                    outputs["policy_margin"],
                    outputs["reference_margin"],
                ),
                dim=-1,
            ).detach().float()
            labels = torch.ones(logits.shape[0], device=logits.device, dtype=torch.long)
            return loss, logits, labels

        finally:
            self._clear_structural_memory_if_present(model)
            self._clear_structural_memory_if_present(self.ref_model)


def precompute_reference_margins(trainer, dataset, collator, cache_path=None):
    """Score every pair with the frozen Stage-2 reference before training.

    The returned margins are cached with a pair fingerprint so a resumed run
    cannot accidentally apply margins computed for a different pair bank.
    """
    pair_rows = getattr(dataset, "rows", None)
    pair_hash = canonical_json_sha256(pair_rows) if pair_rows is not None else None
    if cache_path and os.path.isfile(cache_path):
        cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if cached.get("pair_hash") == pair_hash and len(cached.get("margins", [])) == len(dataset):
            return [float(x) for x in cached["margins"]]

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(trainer.args.per_device_eval_batch_size)),
        shuffle=False,
        collate_fn=collator,
    )
    margins = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = trainer._prepare_inputs(batch)
            _, outputs = trainer.compute_loss(
                trainer.model, batch, return_outputs=True
            )
            values = outputs["reference_margin"].detach().float().cpu().tolist()
            if not np.isfinite(values).all():
                raise RuntimeError("Non-finite frozen-reference margin")
            margins.extend(float(x) for x in values)
    if cache_path:
        payload = {
            "schema": "stage3-reference-margins-v1",
            "pair_hash": pair_hash,
            "margins": margins,
            "count": len(margins),
            "margin_hash": hashlib.sha256(
                json.dumps(margins, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        Path(cache_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return margins


def build_tokenizer(mod, args, parent_contract: dict):
    tok = AutoTokenizer.from_pretrained(
        args.stage2_adapter_dir,
        trust_remote_code=True,
    )
    special_tokens = (
        [g["token"] for g in mod.GOALS.values()]
        + mod.TARGET_PLATFORM_TOKENS
        + mod.SOURCE_PLACEHOLDER_TOKENS
        + mod.TARGET_PLACEHOLDER_TOKENS
    )
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    actual_ids = sorted(set(tok.convert_tokens_to_ids(special_tokens)))
    if (
        len(tok) != int(parent_contract["tokenizer_size"])
        or actual_ids != parent_contract["special_token_ids"]
        or special_tokens != parent_contract["special_tokens"]
    ):
        raise ValueError(
            "Stage-3 tokenizer does not exactly match Stage 2"
        )
    return tok


def maybe_restrict_special_token_embeddings(mod, model, tokenizer):
    special_ids = tokenizer.convert_tokens_to_ids(
        [g["token"] for g in mod.GOALS.values()]
        + mod.TARGET_PLATFORM_TOKENS
        + mod.SOURCE_PLACEHOLDER_TOKENS
        + mod.TARGET_PLACEHOLDER_TOKENS
    )

    def enable_only_selected_rows(weight: torch.nn.Parameter, token_ids):
        weight.requires_grad_(True)
        token_ids = torch.tensor(sorted(set(int(x) for x in token_ids if isinstance(x, int) and x >= 0)), dtype=torch.long)

        def grad_mask_hook(grad):
            mask = torch.zeros(grad.size(0), device=grad.device, dtype=grad.dtype)
            mask[token_ids.to(grad.device)] = 1.0
            return grad * mask.unsqueeze(1)

        weight.register_hook(grad_mask_hook)

    inp_emb = model.get_input_embeddings()
    enable_only_selected_rows(inp_emb.weight, special_ids)

    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight is not inp_emb.weight:
        enable_only_selected_rows(out_emb.weight, special_ids)



def freeze_embeddings(model):
    try:
        inp = model.get_input_embeddings()
        if inp is not None:
            inp.weight.requires_grad_(False)
    except Exception:
        pass
    try:
        out = model.get_output_embeddings()
        if out is not None:
            out.weight.requires_grad_(False)
    except Exception:
        pass



def configure_dpo_trainables(
    model,
    *,
    train_lora: bool,
    train_xattn: bool,
    train_attn_gate: bool,
    train_ff_gate: bool,
):
    """
    For stage-3 DPO, do NOT let the whole PEFT adapter move by default.
    We want to preserve the stage-2 language prior and let DPO mainly refine
    memory-conditioned routing through STRUCTURAL xattn.
    """
    if train_ff_gate:
        raise ValueError("The contracted Stage-2 structural FF branch is disabled")
    model.requires_grad_(False)

    for name, param in model.named_parameters():
        if "lora_" in name and train_lora:
            param.requires_grad_(True)
        elif "gated_cross_attn_layer.attn." in name and train_xattn:
            param.requires_grad_(True)
        elif name.endswith("attn_gate") and train_attn_gate:
            param.requires_grad_(True)
        elif name.endswith("ff_gate") and train_ff_gate:
            param.requires_grad_(True)

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not train_lora:
        frozen_stage1.assert_stage1_frozen(model)
    print(f"[DPO-TRAINABLE] groups enabled: "
          f"lora={train_lora} xattn={train_xattn} attn_gate={train_attn_gate} ff_gate={train_ff_gate}")
    print(f"[DPO-TRAINABLE] total trainable params: {sum(x[1] for x in trainable):,}")
    print(f"[DPO-TRAINABLE] first trainables: {[x[0] for x in trainable[:20]]}")


def disable_lora_dropout_for_dpo(model):
    count = 0
    for name, module in model.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.eval()
            count += 1
    return count



def build_structural_model(mod, args, tokenizer, trainable: bool):
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
    base.resize_token_embeddings(len(tokenizer))
    input_weight = base.get_input_embeddings().weight
    output_weight = base.get_output_embeddings().weight
    weights_tied = input_weight.data_ptr() == output_weight.data_ptr()
    if weights_tied != bool(args.parent_contract["embedding_weights_tied"]):
        raise ValueError(
            "Backbone embedding tying does not match the Stage-2 contract"
        )

    base.config.use_cache = False

    gc_enabled = bool(args.gradient_checkpointing and trainable)
    gc_kwargs = {"use_reentrant": False} if gc_enabled else None

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=gc_enabled,
        gradient_checkpointing_kwargs=gc_kwargs,
    )

    model = PeftModel.from_pretrained(
        base,
        os.path.abspath(args.stage2_adapter_dir),
        is_trainable=trainable,
    )

    mod.extend_instance(model, mod.StructuralCrossAttentionMixin)
    decoder_layers_attr_name = mod.infer_decoder_layers_attr_name(model)
    model.set_decoder_layers_attr_name(decoder_layers_attr_name)

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(mod.TARGET_PLACEHOLDER_TOKENS)
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
        attn_gate_scale=args.structural_gate_scale,
        memory_value_scale=args.structural_memory_value_scale,
        structural_fusion_placement=(
            args.structural_fusion_placement
        ),
    )
    mod.move_structural_modules_to_model_device(model)
    mod.load_partial_structural_xattn(
        model,
        os.path.join(args.stage2_adapter_dir, "structural_xattn.pt"),
        tag="STRUCTURAL-LOAD-STAGE2",
    )
    actual_layers = tuple(model.structural_xattn_layer_indices)
    if actual_layers != args.selected_xattn_layers_1based:
        raise ValueError(
            "Stage-3 structural layers differ from the Stage-2 contract: "
            f"{actual_layers} != {args.selected_xattn_layers_1based}"
        )

    if trainable:
        configure_dpo_trainables(
            model,
            train_lora=args.train_lora_dpo,
            train_xattn=args.train_xattn_dpo,
            train_attn_gate=args.train_attn_gate_dpo,
            train_ff_gate=args.train_ff_gate_dpo,
        )

        if args.train_special_token_embeddings:
            raise ValueError("Stage-3 production keeps special-token embeddings frozen")
        freeze_embeddings(model)
        disable_lora_dropout_for_dpo(model)
    else:
        model.requires_grad_(False)
        freeze_embeddings(model)
        model.eval()

    return model


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--directive_domain_registry_json", type=str, default="")
    ap.add_argument("--application_dataset_dir", type=str, default="")
    ap.add_argument("--memory_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--model_revision", type=str, default=None)
    ap.add_argument("--sft_script", type=str, required=True)

    ap.add_argument("--objective", type=str, default=None, choices=[
        "PARETO_LATENCY",
        "PARETO_ADP",
        "PARETO_AREA",
    ])

    ap.add_argument("--stage2_adapter_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--split_mode", type=str, default="family", choices=["family", "random_design"])
    ap.add_argument("--split_json", type=str, default="")
    ap.add_argument("--save_split_json", type=str, default="")
    ap.add_argument(
        "--save_selection_debug",
        action="store_true",
        help="Persist selected split rows under selected_debug/.",
    )
    ap.add_argument("--val_families", type=str, default="rodinia_pathfinder;machsuite_sort_radix")
    ap.add_argument("--test_families", type=str, default="serrano_kalman_filter")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--stratify_by_kernel", action="store_true")

    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--goal_domination_penalty", type=float, default=0.25)
    ap.add_argument("--goal_max_dominated_gap", type=float, default=0.12)
    ap.add_argument("--score_weight_min", type=float, default=0.6)
    ap.add_argument("--score_weight_power", type=float, default=1.0)

    ap.add_argument("--dpo_chosen_top_k", type=int, default=3)
    ap.add_argument("--dpo_hard_window", type=int, default=8)
    ap.add_argument("--dpo_hard_negatives_per_chosen", type=int, default=2)
    ap.add_argument("--dpo_medium_negatives_per_chosen", type=int, default=1)
    ap.add_argument("--dpo_min_score_gap", type=float, default=0.02)
    ap.add_argument("--dpo_hard_gap_max", type=float, default=0.15)
    ap.add_argument("--dpo_medium_gap_max", type=float, default=0.35)
    ap.add_argument("--dpo_min_primary_rel_gain", type=float, default=0.02)
    ap.add_argument("--dpo_min_edit_distance", type=int, default=1)
    ap.add_argument("--dpo_min_edit_frac", type=float, default=0.0)
    ap.add_argument("--dpo_max_edit_frac", type=float, default=1.0)
    ap.add_argument("--dpo_require_chosen_rank0", action="store_true")
    ap.add_argument("--dpo_max_edit_distance", type=int, default=0)
    ap.add_argument("--dpo_semantic_action_only", action="store_true",
                    help="Keep one semantic action per pair and score all its member fields")
    ap.add_argument("--dpo_pair_unit", choices=("field", "semantic_action"), default="field")
    ap.add_argument("--dpo_min_action_distance", type=int, default=1)
    ap.add_argument("--dpo_max_action_distance", type=int, default=1)
    ap.add_argument("--dpo_max_reference_margin", type=float, default=None,
                    help="Train only pairs whose frozen Stage-2 margin is <= epsilon")

    ap.add_argument("--min_supervised_sites", type=int, default=2)
    ap.add_argument("--min_site_coverage", type=float, default=0.85)
    ap.add_argument("--selection_num_val_kernels", type=int, default=None)
    ap.add_argument(
        "--selection_cases_per_kernel_device",
        type=int,
        default=None,
    )
    ap.add_argument("--selection_eval_steps", type=int, default=None)
    ap.add_argument("--selection_early_stopping_patience", type=int, default=1)
    ap.add_argument("--selection_early_stopping_min_step", type=int, default=20)
    ap.add_argument(
        "--selection_candidate_batch_size",
        type=int,
        default=1,
    )

    ap.add_argument("--require_same_supervised_schema", dest="require_same_supervised_schema", action="store_true")
    ap.add_argument("--allow_mismatched_supervised_schema", dest="require_same_supervised_schema", action="store_false")
    ap.set_defaults(require_same_supervised_schema=True)

    ap.add_argument("--value_loss_weight", type=float, default=1.0)
    ap.add_argument("--train_special_token_embeddings", action="store_true")
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sft_alpha", type=float, default=0.0)
    ap.add_argument(
        "--dpo_logp_reduction",
        choices=("sum", "mean"),
        default="mean",
        help=(
            "Use mean to match MailoHLS candidate-score inference; sum is "
            "the original sequence-level DPO objective."
        ),
    )

    ap.add_argument("--train_lora_dpo", action="store_true")
    ap.add_argument("--train_xattn_dpo", action="store_true")
    ap.add_argument("--train_attn_gate_dpo", action="store_true")
    ap.add_argument("--train_ff_gate_dpo", action="store_true")

    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--lr_lora", type=float, default=0.0)
    ap.add_argument("--lr_xattn", type=float, default=5e-5)
    ap.add_argument("--lr_gate", type=float, default=2e-5)
    ap.add_argument("--lr_ff", type=float, default=0.0)
    ap.add_argument("--lr_gate_ff", type=float, default=0.0)
    ap.add_argument("--lr_embed", type=float, default=0.0)

    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument(
        "--pair_build_only",
        action="store_true",
        help=(
            "Validate contracts and write preference-pair audits without "
            "loading the policy/reference models."
        ),
    )
    ap.add_argument("--reuse_pair_cache", action="store_true")
    ap.add_argument("--reuse_selection_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--stage2_seed", type=int, default=None,
        help="Seed used for reproducing Stage-2 random-budget conditioning; defaults to the parent contract seed.",
    )

    args = ap.parse_args()

    if args.beta <= 0.0:
        raise ValueError("--beta must be positive")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("--label_smoothing must be in [0, 0.5)")
    if args.sft_alpha < 0.0:
        raise ValueError("--sft_alpha must be non-negative")
    if args.train_special_token_embeddings:
        raise ValueError("Stage-3 keeps special-token embeddings frozen")
    if args.train_lora_dpo or args.lr_lora != 0.0:
        raise ValueError(
            "The final focused-DPO ablation keeps the Stage-1 LoRA frozen; "
            "do not pass --train_lora_dpo and keep --lr_lora=0"
        )
    if args.top_k < 2:
        raise ValueError("Stage-3 preference construction requires --top_k >= 2")
    if not 0.0 <= args.dpo_min_edit_frac <= args.dpo_max_edit_frac <= 1.0:
        raise ValueError(
            "DPO edit fractions must satisfy 0 <= min <= max <= 1"
        )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # if the user did not explicitly choose trainable groups,
    # train only STRUCTURAL xattn + attn_gate.
    if not any([
        args.train_lora_dpo,
        args.train_xattn_dpo,
        args.train_attn_gate_dpo,
        args.train_ff_gate_dpo,
    ]):
        args.train_xattn_dpo = True
        args.train_attn_gate_dpo = True

    enabled_lrs = {
        "lora": (args.train_lora_dpo, args.lr_lora),
        "xattn": (args.train_xattn_dpo, args.lr_xattn),
        "attn_gate": (args.train_attn_gate_dpo, args.lr_gate),
        "ff_gate": (args.train_ff_gate_dpo, args.lr_gate_ff),
        "special_token_embeddings": (
            args.train_special_token_embeddings,
            args.lr_embed,
        ),
    }
    bad_lrs = [
        name for name, (enabled, lr) in enabled_lrs.items()
        if enabled and float(lr) <= 0.0
    ]
    if bad_lrs:
        raise ValueError(
            "Enabled Stage-3 groups require positive learning rates: "
            + ", ".join(bad_lrs)
        )
    if args.train_ff_gate_dpo or args.lr_ff != 0.0:
        raise ValueError(
            "The contracted Stage-2 structural branch has xattn FF disabled"
        )

    mod = import_module_from_path(args.sft_script)
    parent_contract = load_stage2_contract(args.stage2_adapter_dir)
    args.parent_contract = parent_contract
    if args.stage2_seed is None:
        args.stage2_seed = int(parent_contract["seed"])
    apply_parent_contract(mod, args, parent_contract)
    if args.max_length is None:
        args.max_length = int(parent_contract["max_length"])
    if int(args.stage2_seed) != int(parent_contract["seed"]):
        raise ValueError(
            "--stage2_seed must match the Stage-2 contract seed so random-budget "
            "conditioning is reproduced exactly"
        )
    if args.selection_cases_per_kernel_device is None:
        args.selection_cases_per_kernel_device = int(
            parent_contract["selection_cases_per_kernel_device"]
        )
    if args.selection_num_val_kernels is None:
        args.selection_num_val_kernels = int(
            parent_contract["selection_num_val_kernels"]
        )
    if args.selection_eval_steps is None:
        args.selection_eval_steps = args.eval_steps

    if file_sha256(args.dataset) != parent_contract["dataset_sha256"]:
        raise ValueError("Dataset does not match the Stage-2 contract")
    registry_sha = parent_contract.get(
        "directive_domain_registry_sha256"
    )
    if registry_sha:
        if not args.directive_domain_registry_json:
            raise ValueError("This historical Stage-2 checkpoint requires its directive-domain registry")
        if file_sha256(args.directive_domain_registry_json) != registry_sha:
            raise ValueError("Directive-domain registry does not match Stage 2")
    elif args.directive_domain_registry_json:
        raise ValueError("Source-domain Stage-2 checkpoints must not receive a measured registry")
    manifest_path = os.path.join(args.memory_dir, "memory_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Stage-3 memory is missing {manifest_path}"
        )
    if file_sha256(manifest_path) != parent_contract["structural"][
        "memory_manifest_sha256"
    ]:
        raise ValueError("Memory manifest does not match Stage 2")
    memory_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    parent_structural = parent_contract["structural"]
    if parent_structural.get("embedding_mode") is not None and (
        memory_manifest.get("embedding_mode") != parent_structural["embedding_mode"]
    ):
        raise ValueError("Memory embedding mode does not match the Stage-2 structural contract")
    if memory_manifest.get("action_relation_schema") != parent_structural.get(
        "action_relation_schema", memory_manifest.get("action_relation_schema")
    ):
        raise ValueError("Memory relation schema does not match the Stage-2 structural contract")
    expected_stats_hash = parent_structural.get("normalization_artifact_sha256")
    actual_stats_hash = memory_manifest.get("multiscale", {}).get("normalization_stats_sha256")
    if expected_stats_hash != actual_stats_hash:
        raise ValueError("Multiscale normalization provenance does not match Stage 2")
    if expected_stats_hash:
        stats_path = Path(args.memory_dir) / memory_manifest["multiscale"].get(
            "normalization_stats_file", "normalization_stats.json"
        )
        if not stats_path.is_file() or file_sha256(stats_path) != expected_stats_hash:
            raise ValueError("Multiscale normalization artifact does not match Stage 2")

    rows = mod.filter_rows_for_device_mode(mod.load_rows(args.dataset))
    print(f"[INFO] Loaded {len(rows)} raw rows from {args.dataset}")
    fam_counts = Counter(r["_family"] for r in rows)
    print("[INFO] Raw rows per family (top 15):", fam_counts.most_common(15))

    selected_train_cache = os.path.join(args.output_dir, "selected_debug", "train_selected.jsonl")
    selected_val_cache = os.path.join(args.output_dir, "selected_debug", "val_selected.jsonl")
    if args.reuse_selection_cache and os.path.isfile(selected_train_cache):
        def load_selected(path):
            with open(path, encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        train_rows = load_selected(selected_train_cache)
        val_rows = load_selected(selected_val_cache) if os.path.isfile(selected_val_cache) else []
        test_rows = []
        print(f"[SELECTION-CACHE] Reused train={len(train_rows)} val={len(val_rows)}")
    else:
        train_rows, val_rows, test_rows = build_selected_splits(
            mod=mod, args=args, rows=rows, parent_contract=parent_contract,
        )

    if args.save_selection_debug:
        selected_debug_dir = os.path.join(args.output_dir, "selected_debug")
        os.makedirs(selected_debug_dir, exist_ok=True)
        dump_jsonl(
            os.path.join(selected_debug_dir, "train_selected.jsonl"),
            train_rows,
        )
        if val_rows:
            dump_jsonl(
                os.path.join(selected_debug_dir, "val_selected.jsonl"),
                val_rows,
            )

    print(f"[INFO] Final selected split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    pair_builder = GoalPreferencePairBuilder(
        mod=mod,
        objective=args.objective,
        chosen_top_k=args.dpo_chosen_top_k,
        hard_window=args.dpo_hard_window,
        hard_negatives_per_chosen=args.dpo_hard_negatives_per_chosen,
        medium_negatives_per_chosen=args.dpo_medium_negatives_per_chosen,
        min_score_gap=args.dpo_min_score_gap,
        hard_gap_max=args.dpo_hard_gap_max,
        medium_gap_max=args.dpo_medium_gap_max,
        min_primary_rel_gain=args.dpo_min_primary_rel_gain,
        min_edit_distance=args.dpo_min_edit_distance,
        min_edit_frac=args.dpo_min_edit_frac,
        max_edit_frac=args.dpo_max_edit_frac,
        min_supervised_sites=args.min_supervised_sites,
        min_site_coverage=args.min_site_coverage,
        require_same_supervised_schema=args.require_same_supervised_schema,
        require_chosen_rank0=args.dpo_require_chosen_rank0,
        max_edit_distance=args.dpo_max_edit_distance,
        semantic_action_only=args.dpo_semantic_action_only,
        pair_unit=args.dpo_pair_unit,
        min_action_distance=args.dpo_min_action_distance,
        max_action_distance=args.dpo_max_action_distance,
    )
    
    cache_train = os.path.join(args.output_dir, "pair_debug", "train_pairs.jsonl")
    cache_val = os.path.join(args.output_dir, "pair_debug", "val_pairs.jsonl")
    if args.reuse_pair_cache:
        if not os.path.isfile(cache_train):
            raise FileNotFoundError(f"Missing cached pair file: {cache_train}")
        def load_jsonl(path):
            with open(path, encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        train_pairs = load_jsonl(cache_train)
        val_pairs = load_jsonl(cache_val) if os.path.isfile(cache_val) else []
        print(f"[PAIR-CACHE] Reused train={len(train_pairs)} val={len(val_pairs)}")
    else:
        train_pairs = pair_builder.build(train_rows)
        val_pairs = pair_builder.build(val_rows) if val_rows else []
    test_pairs = []

    print(f"[INFO] Preference pairs: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    if len(train_pairs) == 0:
        raise ValueError("No training pairs were constructed. Relax the pair filters.")

    # Bounded square-root weighting preserves coverage while emphasizing the
    # objective that actually defines the preference pair.  The old ADP-only
    # weighting was incorrect for latency and area Stage-3 runs.
    primary_gain_field = {
        "PARETO_LATENCY": "latency_rel_gain",
        "PARETO_AREA": "area_rel_gain",
        "PARETO_ADP": "adp_rel_gain",
    }[args.objective]
    gains = [
        max(float(p.get(primary_gain_field, 0.0)), 1e-8)
        for p in train_pairs
    ]
    median_gain = float(np.median(gains))
    for pair in train_pairs + val_pairs:
        pair["pair_weight"] = float(np.clip(
            np.sqrt(max(float(pair.get(primary_gain_field, 0.0)), 1e-8) / median_gain),
            0.5, 2.0,
        ))

    print(f"[INFO] Unique train kernels with pairs: {len(set(r['kernel_name'] for r in train_pairs))}")
    print(f"[INFO] Avg pairs per kernel-objective bucket: {len(train_pairs)/max(1, len(Counter((r['kernel_name'], r['obj_mode']) for r in train_pairs))):.2f}")

    audit_preference_pairs("train", train_pairs)
    audit_preference_pairs("val", val_pairs)

    preview_preference_pairs(train_pairs, n=3)

    debug_dir = os.path.join(args.output_dir, "pair_debug")
    os.makedirs(debug_dir, exist_ok=True)
    dump_jsonl(os.path.join(debug_dir, "train_pairs.jsonl"), train_pairs)
    if val_pairs:
        dump_jsonl(os.path.join(debug_dir, "val_pairs.jsonl"), val_pairs)

    training_contract = build_stage3_contract(
        mod,
        args,
        parent_contract,
        train_pairs,
        val_pairs,
        test_pairs,
    )
    mod.dump_json(
        os.path.join(args.output_dir, "training_contract.json"),
        training_contract,
    )
    if args.pair_build_only:
        print(
            "[PAIR-BUILD-ONLY] Contract and pair audits validated; "
            "skipping model construction."
        )
        return

    tokenizer = build_tokenizer(mod, args, parent_contract)
    mem_bank, inferred_mem_dim = (
        mod.structural_memory_utils.load_memory_bank(
            args.memory_dir,
            expected_mem_dim=args.mem_dim,
            expected_max_slots=args.max_slots,
            require_pragma_free_memory=True,
        )
    )
    if inferred_mem_dim != args.mem_dim:
        raise ValueError(
            f"Memory dimension {inferred_mem_dim} != contract {args.mem_dim}"
        )
    required_kernels = {row["kernel_name"] for row in train_rows + val_rows}
    missing_memory = sorted(
        kernel for kernel in required_kernels
        if (
            kernel not in mem_bank
            and mod.normalize_kname(kernel) not in mem_bank
        )
    )
    if missing_memory:
        raise ValueError(
            "Missing Stage-3 structural memory for: "
            + ", ".join(missing_memory[:20])
        )
    mod.structural_memory_utils.print_memory_bank_summary(
        args.memory_dir, mem_bank, required_kernels
    )

    if args.directive_domain_registry_json:
        directive_domain_registry = mod.load_directive_domain_registry(
            args.directive_domain_registry_json
        )
    else:
        if parent_contract.get("directive_domain_registry_policy") != mod.directive_domains.SOURCE_DOMAIN_POLICY:
            raise ValueError("Stage-2 checkpoint does not declare source-derived directive domains")
        directive_domain_registry = mod.directive_domains.build_source_domain_registry(
            (row for split_rows in (train_rows, val_rows) for row in split_rows),
            args.application_dataset_dir or None,
        )
    train_ds = DPOPreferenceDataset(
        mod=mod,
        rows=train_pairs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        value_weight=args.value_loss_weight,
        directive_domain_registry=directive_domain_registry,
    )
    val_ds = DPOPreferenceDataset(
        mod=mod,
        rows=val_pairs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        value_weight=args.value_loss_weight,
        directive_domain_registry=directive_domain_registry,
    ) if val_pairs else None

    collator = DPOPairCollator(tokenizer)

    print("[INFO] Building policy model...")
    policy_model = build_structural_model(mod, args, tokenizer, trainable=True)
    frozen_policy_stage1 = frozen_stage1.frozen_stage1_hashes(policy_model)
    for field, actual in (
        ("stage1_lora_sha256", frozen_policy_stage1["lora_sha256"]),
        ("special_token_sha256", frozen_policy_stage1["special_token_sha256"]),
        ("frozen_stage1_sha256", frozen_policy_stage1["combined_sha256"]),
    ):
        expected = parent_contract["structural"].get(field)
        if expected is not None and expected != actual:
            raise ValueError(f"Loaded Stage-1 state does not match Stage 2: {field}")
    print("[INFO] Building frozen reference model...")
    ref_model = build_structural_model(mod, args, tokenizer, trainable=False)
    frozen_reference_stage1 = frozen_stage1.frozen_stage1_hashes(ref_model)
    if frozen_reference_stage1 != frozen_policy_stage1:
        raise ValueError("Policy/reference models do not share the exact frozen Stage-1 state")

    if hasattr(policy_model, "print_trainable_parameters"):
        policy_model.print_trainable_parameters()

    selection_cases = mod.build_selection_cases(
        val_rows,
        goal_mode=args.objective,
        max_kernels=args.selection_num_val_kernels,
        cases_per_kernel_device=(
            args.selection_cases_per_kernel_device
        ),
        min_coverage=args.min_site_coverage,
        min_supervised_sites=args.min_supervised_sites,
    )

    effective_total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(len(train_ds) / max(1, args.batch_size * args.grad_accum)) * args.epochs)
    warmup_steps = max(1, int(0.03 * effective_total_steps))

    native_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=max(args.lr_lora, args.lr_xattn, args.lr_gate, args.lr_embed, 1e-8),
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=native_bf16,
        fp16=not native_bf16,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_on_start=True,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = STRUCTURALDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        sft_mod=mod,
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_dpo_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        mem_bank=mem_bank,
        mem_dim=args.mem_dim,
        max_slots=args.max_slots,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        sft_alpha=args.sft_alpha,
        logp_reduction=args.dpo_logp_reduction,
        structural_routing=args.structural_routing,
        group_by_length=args.group_by_length,
        lr_lora=args.lr_lora,
        lr_xattn=args.lr_xattn,
        lr_gate=args.lr_gate,
        lr_ff=args.lr_ff,
        lr_gate_ff=args.lr_gate_ff,
        lr_embed=args.lr_embed,
        max_reference_margin=args.dpo_max_reference_margin,
    )

    # Freeze the Stage-2 reference filter before any optimizer/scheduler work.
    # Inactive pairs are removed entirely so they do not consume accumulation
    # or learning-rate schedule steps.
    if args.dpo_max_reference_margin is not None:
        margin_cache = os.path.join(args.output_dir, "reference_margins.json")
        reference_margins = precompute_reference_margins(
            trainer, train_ds, collator, cache_path=margin_cache
        )
        epsilon = float(args.dpo_max_reference_margin)
        active = [i for i, margin in enumerate(reference_margins) if margin <= epsilon]
        active_accuracy = float(np.mean(np.asarray(reference_margins)[active] > 0.0)) if active else 0.0
        negative_count = int(np.sum(np.asarray(reference_margins) < 0.0))
        active_negative_count = int(np.sum(np.asarray(reference_margins)[active] < 0.0)) if active else 0
        print(
            f"[REFERENCE-FILTER] total={len(reference_margins)} active={len(active)} "
            f"epsilon={epsilon:g} active_reference_accuracy={active_accuracy:.4f} "
            f"negative_margin_count={negative_count} active_negative_margin_count={active_negative_count}",
            flush=True,
        )
        train_ds = torch.utils.data.Subset(train_ds, active)
        trainer.train_dataset = train_ds
        pref = training_contract["stage3_preference"]
        pref.update({
            "active_pair_count": len(active),
            "reference_pair_count": len(reference_margins),
            "active_reference_pair_accuracy": active_accuracy,
            "reference_negative_margin_count": negative_count,
            "active_negative_reference_margin_count": active_negative_count,
            "reference_margin_cache": os.path.abspath(margin_cache),
            "reference_margin_cache_sha256": file_sha256(margin_cache),
        })
        training_contract["stage3_preference"]["pair_counts"]["train_active"] = len(active)
        mod.dump_json(os.path.join(args.output_dir, "training_contract.json"), training_contract)
        if not active:
            raise ValueError("No active training pairs remain after reference-margin filtering")

    trainer.add_callback(
        mod.SaveMailoHLSCheckpointCallback(
            tokenizer,
            training_contract,
        )
    )

    if selection_cases:
        trainer.add_callback(
            mod.StageValSelectionCallback(
                tokenizer=tokenizer,
                selection_cases=selection_cases,
                directive_domain_registry=directive_domain_registry,
                output_dir=args.output_dir,
                max_prompt_tokens=args.max_length,
                candidate_score_reduction="mean",
                best_dir_name="best_custom_stage3",
                mem_bank=mem_bank,
                mem_dim=args.mem_dim,
                max_slots=args.max_slots,
                training_contract=training_contract,
                selection_eval_steps=args.selection_eval_steps,
                candidate_batch_size=(
                    args.selection_candidate_batch_size
                ),
                structural_routing=args.structural_routing,
                early_stopping_patience=args.selection_early_stopping_patience,
                early_stopping_min_step=args.selection_early_stopping_min_step,
            )
        )

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        resume_contract_path = os.path.join(
            args.resume_from_checkpoint,
            "training_contract.json",
        )
        if not os.path.isfile(resume_contract_path):
            raise ValueError(
                f"Resume checkpoint lacks {resume_contract_path}"
            )
        with open(resume_contract_path, encoding="utf-8") as handle:
            resume_contract = json.load(handle)
        if resume_contract != training_contract:
            raise ValueError(
                "Resume checkpoint Stage-3 contract does not match this run"
            )
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    final_policy_stage1 = frozen_stage1.frozen_stage1_hashes(policy_model)
    final_reference_stage1 = frozen_stage1.frozen_stage1_hashes(ref_model)
    if final_reference_stage1 != frozen_reference_stage1:
        raise RuntimeError("Reference Stage-1 state changed during DPO")
    if final_policy_stage1["special_token_sha256"] != frozen_policy_stage1["special_token_sha256"]:
        raise RuntimeError("Special-token embeddings changed during DPO")
    if args.train_lora_dpo and final_policy_stage1["lora_sha256"] == frozen_policy_stage1["lora_sha256"]:
        raise RuntimeError("LoRA training requested but policy LoRA hash did not change")
    if not args.train_lora_dpo and final_policy_stage1 != frozen_policy_stage1:
        raise RuntimeError("Frozen policy Stage-1 state changed during DPO")
    training_contract.setdefault("stage3_runtime", {})["initial_lora_sha256"] = frozen_policy_stage1["lora_sha256"]
    training_contract["stage3_runtime"]["final_lora_sha256"] = final_policy_stage1["lora_sha256"]
    mod.dump_json(os.path.join(args.output_dir, "training_contract.json"), training_contract)

    mod.save_mailohls_adapter(
        policy_model,
        tokenizer,
        args.output_dir,
        training_contract,
    )

    torch.save(
        mod.get_structural_xattn_state_dict(policy_model),
        os.path.join(args.output_dir, "structural_xattn.pt"),
    )

    print(
        f"[DPO-STRUCTURAL-CONFIG] mem_dim={args.mem_dim} "
        f"max_slots={args.max_slots} "
        f"every_n_layers={args.every_n_layers} "
        f"xattn_heads={args.xattn_heads} "
        f"xattn_dim_head={args.xattn_dim_head} "
        f"xattn_ff_mult={args.xattn_ff_mult}"
    )

    print(f"[DONE] Saved DPO LoRA + STRUCTURAL xattn adapters to: {args.output_dir}")

    cleanup_cuda()


if __name__ == "__main__":
    main()
