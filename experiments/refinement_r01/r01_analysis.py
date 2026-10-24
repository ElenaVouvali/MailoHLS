"""Finite-difference attribution and explicit native/extended-domain accounting."""
from __future__ import annotations

import math
from experiments.refinement import r0_candidates as candidates
from experiments.refinement.r0_metrics import RESOURCE_KEYS


def domain_classification(values, baseline, native_domains):
    outside = {key: value for key, value in values.items() if value not in native_domains[key]}
    changed = [key for key in baseline if baseline[key] != values[key]]
    return {
        "configuration_domain": "extended_domain" if outside else "native_domain",
        "outside_native_domain": outside,
        "changed_value_requires_extension": any(key in outside for key in changed),
        "changed_fields_domain": {key: {"before": baseline[key], "after": values[key],
            "before_native": baseline[key] in native_domains[key],
            "after_native": values[key] in native_domains[key]} for key in changed},
    }


def factorial_candidates(base, actions, domains, left, right):
    """Exact complete assignments; two single-field edits on distinct actions."""
    if len(left) != 1 or len(right) != 1 or set(left) & set(right):
        raise ValueError("Factorial modifications must be distinct single fields")
    lhs, rhs = next(iter(left)), next(iter(right))
    if candidates.FIELD.fullmatch(lhs)[2] == candidates.FIELD.fullmatch(rhs)[2]:
        raise ValueError("Factorial modifications must affect distinct actions")
    if base[lhs] == left[lhs] or base[rhs] == right[rhs]:
        raise ValueError("No-op factorial modification")
    result = {}
    for name, update in [("d0", {}), ("dX", left), ("dY", right), ("dXY", {**left, **right})]:
        value = {**base, **update}
        candidates.validate(value, actions, domains)
        _, fields = candidates.changed(base, value, actions)
        if set(fields) != set(update):
            raise ValueError("Unexpected factorial field change")
        result[name] = value
    return result


def factorial_analysis(measurements):
    """Every cell must be successful, feasible, authenticated and comparable."""
    names = ("d0", "dX", "dY", "dXY")
    if any(not measurements.get(name) for name in names):
        return {"status": "unavailable", "reason": "missing measurements"}
    rows = [measurements[name] for name in names]
    if any(row.get("authentication") not in {"authenticated_reuse", "authenticated_new"}
           or row.get("synthesis_status") != "success" or row.get("feasible") is not True
           or row.get("ADP") is None for row in rows):
        return {"status": "unavailable", "reason": "unverified, failed, incomplete or infeasible measurement"}
    contexts = {row.get("comparison_context_id") for row in rows}
    if None in contexts or len(contexts) != 1:
        return {"status": "unavailable", "reason": "mismatched evaluation contexts"}
    j0, jx, jy, jxy = (row["ADP"] for row in rows)
    if j0 <= 0 or any(not math.isfinite(j) for j in (j0, jx, jy, jxy)):
        return {"status": "unavailable", "reason": "invalid objective values"}
    interaction = jxy - jx - jy + j0
    # Only suppress floating-point cancellation at machine-precision scale.
    epsilon = 32 * math.ulp(float(max(abs(j) for j in (j0, jx, jy, jxy)))) if hasattr(math, "ulp") else 1e-10
    if abs(interaction) <= epsilon:
        interaction = 0.0
    return {"status": "measured", "interaction_ADP": interaction,
        "interaction_pct_of_baseline": 100 * interaction / j0,
        "interaction_class": "favorable_nonadditive" if interaction < 0 else "unfavorable_nonadditive" if interaction > 0 else "additive",
        "improvement_pct": {name: 100 * (j0 - measurements[name]["ADP"]) / j0 for name in names},
        "ADP_deltas": {name: measurements[name]["ADP"] - j0 for name in names},
        "latency_deltas": {name: measurements[name]["latency_cycles"] - rows[0]["latency_cycles"] for name in names},
        "resource_deltas": {name: {key: measurements[name]["resources"][key] - rows[0]["resources"][key]
            for key in RESOURCE_KEYS} for name in names},
        "interpretation_limit": "Finite-difference ADP attribution, not proof of a physical hardware mechanism."}
