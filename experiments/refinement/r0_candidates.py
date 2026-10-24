"""Complete configurations and deterministic neighborhoods anchored to SALT outputs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import re

from LLM_branch.common import directive_domains as dd, mailohls_contract as mc

ASSIGNMENT = re.compile(r"auto\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L[1-9][0-9]*)\}\s*=\s*(\S+)", re.I)
FIELD = re.compile(r"AUTO\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L[1-9][0-9]*)\}")
KINDS = {"loop": ("PIPE", "UNROLL"), "array": ("ARRAY_T", "ARRAY_F", "ARRAY_D")}


def key(kind, label):
    return f"AUTO{{_{kind}_{label}}}"


def ordered_fields(actions):
    return [key(kind, label) for label, action in actions.items() for kind in KINDS[action.kind]]


def parse_assignment(text):
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = ASSIGNMENT.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Malformed directive assignment: {line!r}")
        kind, label, rhs = match.groups()
        lhs = key(kind.upper(), label.upper())
        if lhs in values:
            raise ValueError(f"Duplicate directive field: {lhs}")
        rhs = rhs.lower()
        if kind.upper() != "ARRAY_T" and not re.fullmatch(r"0|[1-9][0-9]*", rhs):
            raise ValueError(f"Noncanonical integer: {rhs!r}")
        values[lhs] = rhs
    if not values:
        raise ValueError("Empty directive assignment")
    mc.validate_directive_assignments(values)
    return values


def serialize(values, actions):
    expected = ordered_fields(actions)
    if set(values) != set(expected):
        raise ValueError("Incomplete assignment or action/schema mismatch")
    return "\n".join(f"auto{{{lhs[5:-1]}}} = {values[lhs]}" for lhs in expected)


def validate(values, actions, domains):
    expected = set(ordered_fields(actions))
    if set(values) != expected or set(domains) != expected:
        raise ValueError(f"Incomplete/schema mismatch: missing={sorted(expected-set(values))}, extra={sorted(set(values)-expected)}")
    mc.validate_directive_assignments(values)
    violations = {lhs: {"value": rhs, "allowed": domains[lhs]}
                  for lhs, rhs in values.items() if rhs not in domains[lhs]}
    if violations:
        raise DomainViolation(violations)


class DomainViolation(ValueError):
    def __init__(self, violations):
        self.violations = violations
        super().__init__("Published baseline violates source-derived domains: " + json.dumps(violations, sort_keys=True))


def from_paper(row, actions):
    values = {}
    seen = set()
    for directive in row["directives"]:
        label = directive["action_id"]
        if label in seen or label not in actions or directive["kind"] != actions[label].kind:
            raise ValueError(f"Duplicate, unknown, or mismatched action: {label}")
        seen.add(label)
        if directive["kind"] == "loop":
            data = {"PIPE": directive["pipeline_ii"], "UNROLL": directive["unroll_factor"]}
        else:
            data = {"ARRAY_T": directive["partition_type"], "ARRAY_F": directive["partition_factor"], "ARRAY_D": directive["partition_dim"]}
        for kind, value in data.items():
            values[key(kind, label)] = str(value).lower()
    # Parsing enforces the same canonical syntax used for all subsequent designs.
    return parse_assignment(serialize(values, actions))


def published_factor_domains(base, actions, source_domains):
    """R0-only override: preserve published factors, propose adjacent integers.

    The experiment owner explicitly waived the historical powers-of-two factor
    whitelist. Non-factor domains and tuple semantics still use SALT's contract.
    Zero disables a directive; two is the smallest nontrivial enabled factor.
    New factors stay within source extents/trip counts. Published values are
    retained verbatim even when the old proposal whitelist did not contain them.
    No QoR fields or alternative historical configurations are consulted.
    """
    domains = {lhs: list(values) for lhs, values in source_domains.items()}
    if set(base) != set(ordered_fields(actions)):
        raise ValueError("Incomplete published configuration")
    mc.validate_directive_assignments(base)
    for label, action in actions.items():
        field = key("UNROLL" if action.kind == "loop" else "ARRAY_F", label)
        factor = int(base[field])
        bound = action.trip_count if action.kind == "loop" else max(e for _, e in action.dimensions)
        local = {0, factor}
        local.update(value for value in (2, factor - 1, factor + 1) if 2 <= value <= bound)
        domains[field] = [str(value) for value in sorted(local)]
    validate(base, actions, domains)
    return domains


def signature(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def changed(base, candidate, actions):
    fields = [lhs for lhs in ordered_fields(actions) if base[lhs] != candidate[lhs]]
    labels = [label for label in actions if any(FIELD.fullmatch(lhs)[2] == label for lhs in fields)]
    return labels, fields


def alternatives(base, label, action, domains):
    """Enumerate legal tuples; index distance favors adjacent supported values."""
    kinds = KINDS[action.kind]
    fields = [key(k, label) for k in kinds]
    output = []
    for combo in itertools.product(*(domains[f] for f in fields)):
        if all(base[f] == v for f, v in zip(fields, combo)):
            continue
        if not mc._valid_action_values(dict(zip(kinds, combo))):
            continue
        updates = dict(zip(fields, combo))
        distance = sum((int(base[f] != v) if f.startswith("AUTO{_ARRAY_T_") else
                        abs(domains[f].index(base[f]) - domains[f].index(v))) for f, v in updates.items())
        output.append({"values": updates, "locality_distance": distance,
                       "changed_fields": sum(base[f] != v for f, v in updates.items())})
    return sorted(output, key=lambda a: (a["locality_distance"], a["changed_fields"], json.dumps(a["values"], sort_keys=True)))


def candidate_record(kernel, group, base, values, actions, domains, evidence=None, distance=0):
    validate(values, actions, domains)
    labels, fields = changed(base, values, actions)
    expected = {"baseline": 0, "independent": 1, "structural": 2}[group]
    if len(labels) != expected:
        raise ValueError(f"{group} must change exactly {expected} actions, got {labels}")
    if group == "structural" and (not evidence or set(labels) != {evidence["action_i"], evidence["action_j"]}):
        raise ValueError("Joint changes must exactly match their evidenced pair")
    digest = signature(values)
    return {"kernel": kernel, "candidate_id": f"{kernel.lower()}-{group}-{digest[:16]}",
            "candidate_group": group, "assignment": dict(values), "target": serialize(values, actions),
            "canonical_signature": digest, "changed_action_ids": labels,
            "changed_directive_fields": fields, "changed_action_count": len(labels),
            "changed_field_count": len(fields), "action_relation_evidence": evidence,
            "locality_distance": distance}


def generate_pools(kernel, base, actions, domains, pairs, inactive_actions=()):
    validate(base, actions, domains)
    if set(inactive_actions) - set(actions):
        raise ValueError("Unknown inactive source action")
    options = {label: alternatives(base, label, action, domains) for label, action in actions.items()
               if label not in inactive_actions}
    independent = []
    for label, alts in options.items():
        for alt in alts:
            independent.append(candidate_record(kernel, "independent", base, {**base, **alt["values"]}, actions, domains, distance=alt["locality_distance"]))
    structural = []
    # At most three nearest supported tuples per action; no QoR-guided pruning.
    for pair in pairs:
        i, j = pair["action_i"], pair["action_j"]
        if i == j or i not in options or j not in options:
            raise ValueError("Unsupported pair/action identity")
        for left, right in itertools.product(options[i][:3], options[j][:3]):
            structural.append(candidate_record(kernel, "structural", base,
                {**base, **left["values"], **right["values"]}, actions, domains,
                pair, left["locality_distance"] + right["locality_distance"]))
    def dedup(rows):
        seen = set()
        result = []
        for row in rows:
            if row["canonical_signature"] not in seen:
                seen.add(row["canonical_signature"])
                result.append(row)
        return result
    return dedup(independent), dedup(structural)


def select_diverse(pool, actions, seed=123, limit=4):
    """Round-robin sites/pairs, nearest tuples first; array coupling first."""
    buckets = defaultdict(list)
    def tie(text):
        return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()
    for row in pool:
        bucket = tuple(row["changed_action_ids"])
        buckets[bucket].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: (r["locality_distance"], r["changed_field_count"], tie(r["canonical_signature"])))
    def priority(bucket):
        row = buckets[bucket][0]
        evidence = row["action_relation_evidence"]
        if evidence:
            types = evidence["relation_type"]
            rank = 0 if "loop_array_access" in types else (1 if any("loop" in t for t in types) else 2)
        else:
            rank = 0 if actions[bucket[0]].kind == "array" else 1
        return rank, buckets[bucket][0]["locality_distance"], tie("/".join(bucket))
    # Alternate action types in the independent arm, preserving site diversity.
    order = sorted(buckets, key=priority)
    if order and len(order[0]) == 1:
        types = {kind: [b for b in order if actions[b[0]].kind == kind] for kind in KINDS}
        order = [b for pair in itertools.zip_longest(types["loop"], types["array"]) for b in pair if b is not None]
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in order:
            if buckets[bucket] and len(selected) < limit:
                row = dict(buckets[bucket].pop(0))
                row["selection_rationale"] = "nearest legal tuple(s), diverse actions/pairs; compiler loop-array priority; seeded SHA-256 tie break; no QoR input"
                selected.append(row)
    return selected
