"""Typed, provenance-aware view of one policy's authenticated observations."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Mapping

from experiments.refinement import synthesis

RESOURCES = synthesis.RESOURCE_KEYS


@dataclass(frozen=True)
class Quantity:
    value: float | int | None
    unit: str
    kind: str
    provenance: str
    available: bool


@dataclass(frozen=True)
class HardwareFeedback:
    candidate_id: str
    deployment_context_id: str | None
    synthesis_status: str
    feasible: bool | None
    quantities: Mapping[str, Quantity]
    resource_headroom: Mapping[str, Quantity]
    deltas: Mapping[str, Quantity]
    diagnostics: Mapping[str, Quantity]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _q(value, unit, kind, provenance):
    if value is None:
        return Quantity(None, unit, "unknown", provenance, False)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid {kind} measurement: {value!r}")
    return Quantity(value, unit, kind, provenance, True)


def parse_feedback(candidate_id: str, measurement: Mapping[str, Any], context: Mapping[str, Any],
                   reference: Mapping[str, Any] | None = None) -> HardwareFeedback:
    """Validate authenticated measurement fields; never accepts cache inventory."""
    status = measurement.get("synthesis_status", "unknown")
    if status not in {"success", "failed", "unavailable", "timeout", "unknown"}:
        raise ValueError("Unknown synthesis status")
    capacities, budgets = context["device_capacities"], context["resource_budget"]
    resources = measurement.get("resources") or {}
    normalized = measurement.get("normalized_utilization") or {}
    values: dict[str, Quantity] = {
        "latency_cycles": _q(measurement.get("latency_cycles"), "cycles", "measured", "authenticated HLS result"),
        "area_proxy": _q(measurement.get("area_proxy"), "% mean device utilization", "derived", "existing evaluate_metrics formula"),
        "ADP": _q(measurement.get("ADP"), "cycles·%", "derived", "existing evaluate_metrics formula"),
        "estimated_clock_ns": _q(measurement.get("estimated_clock_ns"), "ns", "measured_estimate", "Vitis HLS report; not post-route timing"),
    }
    headroom: dict[str, Quantity] = {}
    for key in RESOURCES:
        count = resources.get(key)
        values[key] = _q(count, "count", "measured", "authenticated HLS result")
        values[f"{key}_utilization_pct"] = _q(normalized.get(key), "% physical capacity", "derived", "existing evaluate_metrics formula")
        budget = budgets.get(key)
        headroom[key] = _q(None if count is None or budget is None else 100.0 * (budget - count) / budget if budget else (0.0 if count else 100.0),
                           "% requested budget", "derived", "requested resource budget; zero budget handled explicitly")
    deltas: dict[str, Quantity] = {}
    for name, key in (("latency_cycles", "latency_cycles"), ("ADP", "ADP")):
        old = (reference or {}).get(key)
        new = measurement.get(key)
        deltas[name] = _q(None if new is None or old is None else new - old, "cycles" if name == "latency_cycles" else "cycles·%", "derived", "same policy reference measurement")
    old_resources = (reference or {}).get("resources") or {}
    for key in RESOURCES:
        old, new = old_resources.get(key), resources.get(key)
        deltas[key] = _q(None if old is None or new is None else new - old, "count", "derived", "same policy reference measurement")
    diagnostics = {"initiation_interval": _q(measurement.get("initiation_interval"), "cycles", "compiler_report_diagnostic", "Vitis HLS report; unavailable unless explicitly parsed"),
                   "scheduling_violation": _q(measurement.get("scheduling_violation"), "boolean", "compiler_report_diagnostic", "explicit report evidence only")}
    if all(values[k].available for k in ("latency_cycles", *RESOURCES)):
        checked = synthesis.evaluate_metrics(values["latency_cycles"].value,
            {k: values[k].value for k in RESOURCES}, capacities=capacities, budget=budgets)
        for key in ("area_proxy", "ADP"):
            if values[key].available and not math.isclose(float(values[key].value), float(checked[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Authenticated {key} does not match the existing synthesis metric formula")
    return HardwareFeedback(candidate_id, measurement.get("comparison_context_id"), status,
                            measurement.get("feasible"), values, headroom, deltas, diagnostics)


def history_feedback(history, context):
    """Return views constructed only from this policy's already revealed history."""
    by_signature = {row["signature"]: row for row in history}
    result = {}
    for row in history:
        reference = by_signature.get(row.get("reference_signature"))
        if reference and reference["measurement"].get("comparison_context_id") != row["measurement"].get("comparison_context_id"):
            reference = None
        result[row["signature"]] = parse_feedback(row["signature"], row["measurement"], context,
            reference["measurement"] if reference else None).to_dict()
    return result
