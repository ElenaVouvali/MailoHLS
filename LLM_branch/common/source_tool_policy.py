"""Versioned source and compiler admissibility layered over saved domains.

The policy narrows tool-inadmissible candidates before model scoring.  Its
source-derived tuple universe may restore legal non-power-of-two factors that
the historical checkpoint registry never exposed; checkpoint files are never
modified and the original registry is reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .directive_domains import SourceAction, action_tuples


@dataclass(frozen=True)
class ToolPolicy:
    policy_id: str
    compiler: str
    version: str
    complete_partition_limit: int
    dim_zero_all_dimensions: bool
    source_sha256: str
    raw: dict


def load_policy(path: str | Path) -> ToolPolicy:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "mailohls-source-tool-admissibility-v1":
        raise ValueError("Unsupported source/tool admissibility policy schema")
    compiler = raw.get("compiler", {})
    limit = raw.get("complete_partition_dimension_element_limit")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Policy needs a positive complete-partition dimension limit")
    if not compiler.get("name") or not compiler.get("version"):
        raise ValueError("Policy must identify the compiler and version")
    return ToolPolicy(raw["policy_id"], compiler["name"], compiler["version"],
                      limit, raw.get("array_dimension_zero_semantics") == "all_dimensions",
                      hashlib.sha256(path.read_bytes()).hexdigest(), raw)


def _lhs_parts(lhs: str) -> tuple[str, str]:
    import re
    match = re.fullmatch(r"AUTO\{_(ARRAY_T|ARRAY_F|ARRAY_D)_(L\d+)\}", lhs.strip(), re.I)
    if not match:
        raise ValueError(f"Not an array directive field: {lhs!r}")
    return match.group(1).upper(), match.group(2).upper()


def _field_keys(label: str) -> tuple[str, str, str]:
    return tuple(f"AUTO{{_{kind}_{label}}}" for kind in ("ARRAY_T", "ARRAY_F", "ARRAY_D"))


def _partitioned_elements(action: SourceAction, dimension: int) -> int | None:
    extents = dict(action.dimensions)
    if not extents:
        return None
    if dimension == 0:  # HLS dim=0 means partition all dimensions.
        return __import__("math").prod(extents.values())
    return extents.get(dimension)


def _maximum_partial_factor(action: SourceAction, dimension: int) -> int | None:
    extents = dict(action.dimensions)
    if not extents:
        return None
    if dimension == 0:  # The same factor is applied to every dimension.
        return min(extents.values())
    return extents.get(dimension)


def tuple_rejection(reason_tuple: tuple[str, str, str], action: SourceAction,
                    policy: ToolPolicy) -> str | None:
    kind, factor_text, dim_text = reason_tuple
    kind = kind.lower()
    factor, dimension = int(factor_text), int(dim_text)
    if kind == "none":
        return None if factor == 0 and dimension == 0 else "invalid_none_sentinel"
    extent = _partitioned_elements(action, dimension)
    if extent is None:
        return "unknown_or_unresolved_array_dimension"
    if kind in {"block", "cyclic"}:
        if factor < 1:
            return "partial_partition_factor_must_be_positive"
        partial_extent = _maximum_partial_factor(action, dimension)
        if partial_extent is None:
            return "unknown_or_unresolved_array_dimension"
        if factor > partial_extent:
            return f"factor_exceeds_selected_dimension_extent:{factor}>{partial_extent}"
        return None
    if kind == "complete":
        if factor != 0:
            return "complete_partition_uses_zero_factor_sentinel"
        if extent > policy.complete_partition_limit:
            return ("complete_partition_dimension_exceeds_authenticated_tool_policy:"
                    f"{extent}>{policy.complete_partition_limit}")
        return None
    return "unsupported_partition_kind"


def _source_tuple_rejection(option: tuple[str, str, str], action: SourceAction,
                            *, allow_dim_zero_all: bool = False) -> str | None:
    """Check HLS tuple syntax/shape without restricting factors to a proposal grid.

    Vitis partial partition factors are counts and need not be powers of two or
    divisors of an extent (UG1399 gives a non-divisor example).  The saved
    checkpoint registry and the source-derived proposal set remain finite, but
    an already saved factor is source-valid whenever it is positive and does
    not exceed the selected dimension.
    """
    kind, factor_text, dim_text = option
    try:
        factor, dimension = int(factor_text), int(dim_text)
    except (TypeError, ValueError):
        return "malformed_array_tuple"
    if kind == "none":
        return None if (factor, dimension) == (0, 0) else "invalid_none_sentinel"
    dimensions = dict(action.dimensions)
    if kind == "complete":
        allowed_dimension = dimension in dimensions or (
            dimension == 0 and allow_dim_zero_all and bool(dimensions)
        )
        return None if factor == 0 and allowed_dimension else "invalid_complete_tuple"
    if kind in {"block", "cyclic"}:
        if dimension == 0 and allow_dim_zero_all and dimensions:
            extent = min(dimensions.values())
        else:
            extent = dimensions.get(dimension)
        if extent is None:
            return "unknown_or_unresolved_array_dimension"
        if factor < 1:
            return "partial_partition_factor_must_be_positive"
        if factor > extent:
            return f"factor_exceeds_selected_dimension_extent:{factor}>{extent}"
        return None
    return "unsupported_partition_kind"


def filter_array_candidates(lhs: str, original_candidates: list[str],
                            chosen_assignments: Mapping[str, str],
                            action: SourceAction, policy: ToolPolicy) -> tuple[list[str], dict]:
    """Return complete-tuple-supported values before language-model scoring.

    Candidate values are the union of the saved checkpoint vocabulary and
    source-derived complete tuples.  Each value is retained only when at least
    one complete source/tool-admissible tuple remains consistent with the
    fields selected earlier in this action.
    """
    field, label = _lhs_parts(lhs)
    if action.kind != "array":
        raise ValueError(f"Array field {lhs} is registered as {action.kind}")
    keys = _field_keys(label)
    members = ("ARRAY_T", "ARRAY_F", "ARRAY_D")
    member_index = members.index(field)
    assignments = {str(k).strip().upper(): str(v).strip().lower()
                   for k, v in chosen_assignments.items()}
    tuples = action_tuples(action)
    # Preserve all saved factors that are valid for the actual dimension even
    # when they are absent from the finite source proposal grid.
    proposal_tuples = set(tuples)
    saved_by_field = {name: set(map(str, original_candidates)) for name in members}
    factors = {str(option[1]) for option in tuples}
    factors.update(saved_by_field["ARRAY_F"])
    for dimension, _extent in action.dimensions:
        for kind in ("block", "cyclic"):
            proposal_tuples.update((kind, factor, str(dimension)) for factor in factors)
    # AMD UG1399 defines dim=0 as applying the directive to every dimension.
    # It is distinct from the existing none sentinel because the partition type
    # remains part of the complete action tuple.
    if policy.dim_zero_all_dimensions:
        for kind in ("block", "cyclic", "complete"):
            proposal_tuples.add((kind, "0", "0"))
        for kind in ("block", "cyclic"):
            proposal_tuples.update((kind, factor, "0") for factor in factors)
    tuples = tuple(sorted(proposal_tuples))
    source_values = sorted({str(option[member_index]) for option in tuples},
                           key=(str if field == "ARRAY_T" else int))
    expanded = sorted(set(map(str, original_candidates)) | set(source_values),
                      key=(str if field == "ARRAY_T" else int))
    legal, rejected = [], {}
    for candidate in expanded:
        possible, reasons = False, []
        for option in tuples:
            if option[member_index] != candidate:
                continue
            if any(key in assignments and assignments[key] != value.lower()
                   for key, value in zip(keys, option)):
                continue
            reason = (_source_tuple_rejection(option, action,
                                              allow_dim_zero_all=policy.dim_zero_all_dimensions)
                      or tuple_rejection(option, action, policy))
            if reason is None:
                possible = True
                break
            reasons.append(reason)
        if possible:
            legal.append(candidate)
        else:
            if not reasons:
                if candidate not in source_values:
                    if field == "ARRAY_F":
                        try:
                            factor = int(candidate)
                            selected = assignments.get(keys[2])
                            dimensions = dict(action.dimensions)
                            dims = ([int(selected)] if selected and selected != "0"
                                    else list(dimensions))
                            extents = [dimensions[d] for d in dims if d in dimensions]
                            over = [extent for extent in extents if factor > extent]
                            if factor > 0 and over:
                                reasons = [f"factor_exceeds_selected_dimension_extent:{factor}>{min(over)}"]
                        except ValueError:
                            pass
                    if not reasons:
                        reasons = ["not_in_any_complete_source_action_tuple"]
                else:
                    # The chosen prefix may have ruled out every otherwise
                    # legal tuple; retain the specific tool reason as evidence.
                    reasons = [tuple_rejection(option, action, policy)
                               for option in tuples if option[member_index] == candidate]
                    reasons = [reason for reason in reasons if reason]
                    if not reasons:
                        reasons = ["incompatible_with_previously_selected_action_fields"]
            rejected[candidate] = sorted(set(reasons))
    if not legal:
        raise ValueError(f"No source/tool-admissible candidate remains for {lhs}; rejected={rejected}")
    return legal, {
        "field": lhs.upper(),
        "saved_checkpoint_candidates": list(map(str, original_candidates)),
        "source_candidates_added": sorted(set(legal) - set(map(str, original_candidates))),
        "admissible_candidates": legal,
        "rejected_candidates": rejected,
        "policy_id": policy.policy_id,
        "compiler": f"{policy.compiler} {policy.version}",
    }


def filter_partial_tuple(lhs: str, original_candidates: list[str],
                         chosen_assignments: Mapping[str, str],
                         action: SourceAction, policy: ToolPolicy) -> tuple[list[str], dict]:
    """Reject a value unless at least one whole action remains admissible."""
    return filter_array_candidates(lhs, original_candidates, chosen_assignments,
                                   action, policy)


def validate_complete_assignment(assignments: Mapping[str, str], actions: Mapping[str, SourceAction],
                                 policy: ToolPolicy) -> None:
    from .mailohls_contract import validate_directive_assignments
    normalized = {str(k).strip().upper(): str(v).strip().lower()
                  for k, v in assignments.items()}
    expected = set()
    for label, action in actions.items():
        if action.kind == "array":
            keys = _field_keys(label)
            expected.update(keys)
            option = tuple(normalized.get(key, "") for key in keys)
            source_reason = _source_tuple_rejection(
                option, action, allow_dim_zero_all=policy.dim_zero_all_dimensions
            )
            if source_reason:
                raise ValueError(f"Source array tuple violation at {label}: {source_reason}")
            reason = tuple_rejection(option, action, policy)
            if reason:
                raise ValueError(f"Tool policy rejects array tuple at {label}: {reason}")
        elif action.kind == "loop":
            keys = (f"AUTO{{_PIPE_{label}}}", f"AUTO{{_UNROLL_{label}}}")
            expected.update(keys)
            option = tuple(normalized.get(key, "") for key in keys)
            if option not in action_tuples(action):
                raise ValueError(f"Source loop tuple violation at {label}: {option}")
        else:
            raise ValueError(f"Unsupported source action kind at {label}: {action.kind}")
    if set(normalized) != expected:
        raise ValueError("Incomplete or extra fields in final source/tool assignment")
    validate_directive_assignments(normalized, allow_dim_zero_all=policy.dim_zero_all_dimensions)


def validate_source_assignment(assignments: Mapping[str, str], actions: Mapping[str, SourceAction]) -> None:
    """Validate complete source action identity and HLS tuple shape without a tool profile."""
    from .mailohls_contract import validate_directive_assignments
    normalized = {str(k).strip().upper(): str(v).strip().lower()
                  for k, v in assignments.items()}
    expected = set()
    for label, action in actions.items():
        if action.kind == "array":
            keys = _field_keys(label)
            expected.update(keys)
            option = tuple(normalized.get(key, "") for key in keys)
            reason = _source_tuple_rejection(option, action, allow_dim_zero_all=True)
            if reason:
                raise ValueError(f"Source array tuple violation at {label}: {reason}")
        elif action.kind == "loop":
            keys = (f"AUTO{{_PIPE_{label}}}", f"AUTO{{_UNROLL_{label}}}")
            expected.update(keys)
            option = tuple(normalized.get(key, "") for key in keys)
            if option not in action_tuples(action):
                raise ValueError(f"Source loop tuple violation at {label}: {option}")
        else:
            raise ValueError(f"Unsupported source action kind at {label}: {action.kind}")
    if set(normalized) != expected:
        raise ValueError("Incomplete or extra fields in source assignment")
    validate_directive_assignments(normalized, allow_dim_zero_all=True)
