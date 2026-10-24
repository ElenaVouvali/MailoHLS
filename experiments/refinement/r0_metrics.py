"""The TCAD mean-resource-utilization area proxy, never LUT-only area."""
from __future__ import annotations

import math
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES, RESOURCE_KEYS

PART = "xczu7ev-ffvc1156-2-e"
CAPACITIES = dict(DEVICE_RESOURCES[PART])
CLOCK_NS = 10.0


def evaluate_metrics(latency_cycles, resources, capacities=None, budget=None):
    capacities = capacities or CAPACITIES
    budget = budget or capacities
    if set(capacities) != set(RESOURCE_KEYS) or set(budget) != set(RESOURCE_KEYS):
        raise ValueError("Capacities and budgets must specify all four resources")
    if any(not math.isfinite(float(capacities[k])) or capacities[k] <= 0
           or not math.isfinite(float(budget[k])) or budget[k] < 0 for k in RESOURCE_KEYS):
        raise ValueError("Invalid capacities/budgets")
    missing = {"latency_cycles": None, "resources": resources,
               "normalized_utilization": None, "area_proxy": None,
               "ADP": None, "feasible": None, "rankable": False}
    if latency_cycles is None or resources is None or any(resources.get(k) is None for k in RESOURCE_KEYS):
        return missing
    numbers = [latency_cycles, *(resources[k] for k in RESOURCE_KEYS)]
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
           or n < 0 or int(n) != n for n in numbers) or latency_cycles <= 0:
        raise ValueError("QoR must be finite nonnegative integer counts, with positive latency")
    util = {k: 100.0 * resources[k] / capacities[k] for k in RESOURCE_KEYS}
    area = sum(util.values()) / 4.0
    feasible = all(resources[k] <= budget[k] for k in RESOURCE_KEYS)
    return {"latency_cycles": int(latency_cycles), "resources": dict(resources),
            "normalized_utilization": util, "area_proxy": area,
            "ADP": latency_cycles * area, "feasible": feasible,
            "rankable": feasible}


def best_feasible(results):
    valid = [r for r in results if r.get("synthesis_status") == "success"
             and r.get("rankable") is True and r.get("ADP") is not None]
    return min(valid, key=lambda r: (r["ADP"], r["candidate_id"])) if valid else None


def improvement_pct(baseline, candidate):
    if not baseline or not candidate or not candidate.get("rankable") or baseline.get("ADP", 0) <= 0:
        return None
    return 100.0 * (baseline["ADP"] - candidate["ADP"]) / baseline["ADP"]
