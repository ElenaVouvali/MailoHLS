"""Shared source domains: checkpoint-compatible defaults and explicit TCAD policy.

Policy identifiers remain stable for saved contracts. Mathematical factors and
dimension-aware tuples are implemented once here, rather than in versioned files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPLICATION_DATASET_DIR = REPOSITORY_ROOT / "Data" / "ApplicationDataset"
SOURCE_DOMAIN_POLICY = "source_action_metadata_proposal_domains_v2"
MAX_UNROLL_FACTOR = 64
MAX_PARTITION_FACTOR = 1024

DIRECTIVE_RE = re.compile(
    r"(auto\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L[1-9][0-9]*)\})",
    re.IGNORECASE,
)
ACTION_ID_RE = re.compile(r"L[1-9][0-9]*", re.IGNORECASE)


@dataclass(frozen=True)
class SourceAction:
    kind: str
    trip_count: int | None = None
    dimensions: tuple[tuple[int, int], ...] = ()


def normalize_kernel_name(value: Any) -> str:
    return re.sub(r"[-\s]+", "_", str(value).strip().lower())


def _metadata_path(kernel_name: str, application_dataset_dir: str | Path | None) -> Path:
    root = Path(application_dataset_dir or DEFAULT_APPLICATION_DATASET_DIR)
    direct = root / "kernel_info.txt"
    if direct.is_file():
        return direct
    candidates = dict.fromkeys((
        str(kernel_name).strip(),
        str(kernel_name).strip().replace("_", "-"),
        str(kernel_name).strip().replace("-", "_"),
    ))
    for candidate in candidates:
        path = root / candidate / "kernel_info.txt"
        if path.is_file():
            return path
    if root.is_dir():
        matches = [
            directory / "kernel_info.txt"
            for directory in root.iterdir()
            if directory.is_dir()
            and normalize_kernel_name(directory.name) == normalize_kernel_name(kernel_name)
            and (directory / "kernel_info.txt").is_file()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous normalized source metadata for {kernel_name!r}: {matches}")
    raise FileNotFoundError(
        f"No source action metadata found for {kernel_name!r} under {root}. "
        "Provide --application_dataset_dir containing the kernel's kernel_info.txt; "
        "measured directive-domain registries are not required."
    )


def load_source_actions(
    kernel_name: str,
    application_dataset_dir: str | Path | None = None,
) -> dict[str, SourceAction]:
    path = _metadata_path(kernel_name, application_dataset_dir)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty source action manifest: {path}")
    actions: dict[str, SourceAction] = {}
    for line in lines[1:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or not ACTION_ID_RE.fullmatch(fields[0]):
            raise ValueError(f"Malformed source action in {path}: {line!r}")
        label, kind = fields[0].upper(), fields[1].lower()
        if label in actions:
            raise ValueError(f"Duplicate source action {label} in {path}")
        if kind == "loop":
            if len(fields) != 3:
                raise ValueError(f"Malformed loop action in {path}: {line!r}")
            trip_count = int(fields[2])
            if trip_count < 1:
                raise ValueError(f"Non-positive trip count for {label} in {path}")
            actions[label] = SourceAction(kind="loop", trip_count=trip_count)
        elif kind == "array":
            if len(fields) < 5 or (len(fields) - 3) % 2:
                raise ValueError(f"Malformed array action in {path}: {line!r}")
            dimensions = tuple(
                (int(fields[index]), int(fields[index + 1]))
                for index in range(3, len(fields), 2)
            )
            if any(dimension < 1 or extent < 1 for dimension, extent in dimensions):
                raise ValueError(f"Invalid array dimensions for {label} in {path}")
            if len({dimension for dimension, _ in dimensions}) != len(dimensions):
                raise ValueError(f"Duplicate array dimension for {label} in {path}")
            actions[label] = SourceAction(kind="array", dimensions=dimensions)
        else:
            raise ValueError(f"Unsupported source action kind {kind!r} in {path}")
    if not actions:
        raise ValueError(f"Source action manifest contains no actions: {path}")
    return actions


def _powers_of_two(limit: int) -> list[str]:
    values: list[str] = []
    value = 2
    while value <= limit:
        values.append(str(value))
        value *= 2
    return values


def prepare_source_template(
    kernel_name: str,
    source_path: str | Path,
    application_dataset_dir: str | Path | None = None,
) -> str:
    """Accept either a raw labeled kernel or its already prepared source template."""
    path = Path(source_path)
    source_text = path.read_text(encoding="utf-8")
    actions = load_source_actions(kernel_name, application_dataset_dir)
    if not DIRECTIVE_RE.search(source_text):
        from GNN_branch.insert_placeholders import insert_placeholders

        source_text = "".join(
            insert_placeholders(str(path), allowed_labels=set(actions))
        ).strip()
    expected = {
        f"AUTO{{_{kind}_{label}}}"
        for label, action in actions.items()
        for kind in (
            ("PIPE", "UNROLL")
            if action.kind == "loop"
            else ("ARRAY_T", "ARRAY_F", "ARRAY_D")
        )
    }
    observed = {match.group(1).upper() for match in DIRECTIVE_RE.finditer(source_text)}
    if observed != expected:
        raise ValueError(
            f"Source/action placeholder mismatch for {path}: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return source_text


def legacy_site_domains(
    kernel_name: str,
    source_text: str,
    application_dataset_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    actions = load_source_actions(kernel_name, application_dataset_dir)
    domains: dict[str, list[str]] = {}
    for match in DIRECTIVE_RE.finditer(source_text):
        lhs, kind, label = match.group(1).upper(), match.group(2).upper(), match.group(3).upper()
        action = actions.get(label)
        if action is None:
            raise ValueError(f"Directive {lhs} has no matching source action in {kernel_name!r}")
        if kind in {"PIPE", "UNROLL"} and action.kind != "loop":
            raise ValueError(f"Loop directive {lhs} refers to an {action.kind} action")
        if kind.startswith("ARRAY_") and action.kind != "array":
            raise ValueError(f"Array directive {lhs} refers to a {action.kind} action")
        if kind == "PIPE":
            values = ["0", "1"]
        elif kind == "UNROLL":
            assert action.trip_count is not None
            limit = min(action.trip_count, MAX_UNROLL_FACTOR)
            candidates = {"0", *_powers_of_two(limit)}
            if 1 < action.trip_count <= MAX_UNROLL_FACTOR:
                candidates.add(str(action.trip_count))
            values = sorted(candidates, key=int)
        elif kind == "ARRAY_T":
            values = ["block", "complete", "cyclic", "none"]
        elif kind == "ARRAY_F":
            extent = max(extent for _, extent in action.dimensions)
            rounded_extent = 1 << (extent - 1).bit_length()
            limit = min(rounded_extent, MAX_PARTITION_FACTOR)
            values = ["0", *_powers_of_two(limit)]
        else:
            values = ["0", *(str(dimension) for dimension, _ in action.dimensions)]
            values = sorted(set(values), key=int)
        if lhs in domains and domains[lhs] != values:
            raise ValueError(f"Inconsistent source-derived domain for {kernel_name}/{lhs}")
        domains[lhs] = values
    if not domains:
        raise ValueError(f"No directive placeholders found in source for {kernel_name!r}")
    return domains


def legacy_domain_registry(
    rows: Iterable[Mapping[str, Any]],
    application_dataset_dir: str | Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    registry: dict[str, dict[str, list[str]]] = {}
    sources: dict[str, str] = {}
    for row in rows:
        raw_kernel = str(row["kernel_name"])
        kernel = normalize_kernel_name(raw_kernel)
        source_text = str(row.get("input", row.get("source_text", "")))
        if not source_text:
            raise ValueError(f"Missing source text for {raw_kernel!r}")
        if kernel in registry:
            if sources[kernel] != source_text:
                raise ValueError(f"Inconsistent source text for normalized kernel {kernel!r}")
            continue
        registry[kernel] = source_site_domains(
            raw_kernel, source_text, application_dataset_dir
        )
        sources[kernel] = source_text
    if not registry:
        raise ValueError("No source-derived directive domains could be built")
    return registry

from functools import lru_cache
from itertools import product
from math import isqrt
from types import SimpleNamespace
from . import mailohls_contract as contract

TCAD_SOURCE_DOMAIN_POLICY = "tcad_source_factors_v3"

@lru_cache(maxsize=4096)
def factor_values(n):
    """Sorted immutable F(n); O(sqrt(n)) divisor enumeration, no 64/1024 caps."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('F(n) requires a resolved positive integer size')
    values = set()
    power = 2
    while power <= n:
        values.add(power)
        power *= 2
    if n > 1:
        values.add(n)
        if n & n - 1:
            for divisor in range(2, isqrt(n) + 1):
                if n % divisor == 0:
                    values.add(divisor)
                    values.add(n // divisor)
    return tuple(sorted(values))

@lru_cache(maxsize=4096)
def action_tuples(action):
    """Authoritative whole-action tuples, including disabled encodings."""
    if action.kind == 'loop':
        factors = (0, *factor_values(action.trip_count)) if action.trip_count is not None else (0,)
        return tuple(((str(pipe), str(factor)) for (pipe, factor) in product((0, 1), factors)))
    if action.kind != 'array' or not action.dimensions:
        raise ValueError('Array metadata must resolve its positive dimension extents')
    result = [('none', '0', '0')]
    for (dimension, extent) in action.dimensions:
        result.append(('complete', '0', str(dimension)))
        result.extend(
            (
                (kind, str(factor), str(dimension)) for kind in ('block', 'cyclic') for factor in factor_values(extent)
            )
        )
    return tuple(result)

class SourceDomains(dict):
    """Flat decoder vocabulary plus authoritative per-site tuple metadata."""

    def __init__(self, values, actions):
        super().__init__(values)
        self.actions = actions
        self.policy = TCAD_SOURCE_DOMAIN_POLICY

KINDS = {'loop': ('PIPE', 'UNROLL'), 'array': ('ARRAY_T', 'ARRAY_F', 'ARRAY_D')}

def domains_from_actions(actions):
    domains = {}
    for (label, action) in actions.items():
        tuples = action_tuples(action)
        for (index, kind) in enumerate(KINDS[action.kind]):
            values = {row[index] for row in tuples}
            domains[f'AUTO{{_{kind}_{label}}}'] = sorted(values, key=str if kind == 'ARRAY_T' else int)
    return SourceDomains(domains, actions)

def tcad_site_domains(kernel_name, source_text, application_dataset_dir=None):
    actions = load_source_actions(kernel_name, application_dataset_dir)
    domains = domains_from_actions(actions)
    observed = {match[1].upper() for match in DIRECTIVE_RE.finditer(source_text)}
    if observed != set(domains):
        raise ValueError('Source/action placeholders do not match the TCAD action contract')
    return SourceDomains(
        {
            match[1].upper(): domains[match[1].upper()] for match in DIRECTIVE_RE.finditer(source_text)
        },
        actions
    )

def tcad_domain_registry(rows, application_dataset_dir=None):
    (result, sources) = ({}, {})
    for row in rows:
        name = str(row['kernel_name'])
        key = normalize_kernel_name(name)
        text = str(row.get('input', row.get('source_text', '')))
        if not text or (key in sources and sources[key] != text):
            raise ValueError('Missing/inconsistent source text for TCAD registry')
        sources[key] = text
        result[key] = tcad_site_domains(name, text, application_dataset_dir)
    if not result:
        raise ValueError('No TCAD source domains')
    return result

def filter_semantic_candidates(lhs, candidates, chosen_assignments, site_domains):
    """Preserve a value iff a dimension-correct complete action can extend it."""
    if not isinstance(site_domains, SourceDomains):
        raise ValueError('TCAD filtering requires source action metadata, not a flat vocabulary alone')
    match = DIRECTIVE_RE.fullmatch(lhs)
    if not match:
        raise ValueError('Malformed directive key')
    (kind, label) = (match[2].upper(), match[3].upper())
    action = site_domains.actions[label]
    kinds = KINDS[action.kind]
    keys = [f'AUTO{{_{field}_{label}}}' for field in kinds]
    assignments = {key.upper(): str(value).strip().lower() for (key, value) in chosen_assignments.items()}
    legal = []
    for candidate in candidates:
        value = str(candidate).strip().lower()
        if value not in site_domains[lhs.upper()]:
            continue
        prefix = {key: assignments[key] for key in keys if key in assignments}
        prefix[lhs.upper()] = value
        if any(
            (
                all((prefix.get(key, entry) == entry for (key, entry) in zip(keys, option))) for option in action_tuples(action)
            )
        ):
            legal.append(value)
    if not legal:
        raise ValueError(f'No source-aware TCAD completion for {lhs}')
    return legal

def validate_assignments(values, actions, baseline=None, allow_baseline_exceptions=False):
    expected = {
        f'AUTO{{_{kind}_{label}}}' for (label, action) in actions.items() for kind in KINDS[action.kind]
    }
    if set(values) != expected:
        raise ValueError('Incomplete/extra directive fields')
    contract.validate_directive_assignments(values)
    exceptions = {}
    for (label, action) in actions.items():
        keys = [f'AUTO{{_{kind}_{label}}}' for kind in KINDS[action.kind]]
        option = tuple((values[key] for key in keys))
        if option not in action_tuples(action):
            if allow_baseline_exceptions and baseline is not None and all((values[key] == baseline[key] for key in keys)):
                exceptions[label] = {key: values[key] for key in keys}
            else:
                raise ValueError(f'Source-aware TCAD tuple violation at {label}: {option}')
    return exceptions


def source_site_domains(kernel_name, source_text, application_dataset_dir=None, *, policy="legacy"):
    """Choose source domains explicitly; existing checkpoints keep legacy defaults."""
    if policy in ("legacy", SOURCE_DOMAIN_POLICY):
        return legacy_site_domains(kernel_name, source_text, application_dataset_dir)
    if policy in ("tcad", TCAD_SOURCE_DOMAIN_POLICY):
        return tcad_site_domains(kernel_name, source_text, application_dataset_dir)
    raise ValueError("Unknown directive domain policy: " + str(policy))


def build_source_domain_registry(rows, application_dataset_dir=None, *, policy="legacy"):
    if policy in ("legacy", SOURCE_DOMAIN_POLICY):
        return legacy_domain_registry(rows, application_dataset_dir)
    if policy in ("tcad", TCAD_SOURCE_DOMAIN_POLICY):
        return tcad_domain_registry(rows, application_dataset_dir)
    raise ValueError("Unknown directive domain policy: " + str(policy))


# This view supplies the existing train/inference contract interface without
# duplicating domain implementations or relabeling historical checkpoints.
tcad = SimpleNamespace(
    SOURCE_DOMAIN_POLICY=TCAD_SOURCE_DOMAIN_POLICY,
    MAX_UNROLL_FACTOR=None, MAX_PARTITION_FACTOR=None,
    DEFAULT_APPLICATION_DATASET_DIR=DEFAULT_APPLICATION_DATASET_DIR,
    load_source_actions=load_source_actions, prepare_source_template=prepare_source_template,
    normalize_kernel_name=normalize_kernel_name, DIRECTIVE_RE=DIRECTIVE_RE,
    SourceAction=SourceAction, SourceDomains=SourceDomains,
    factor_values=factor_values, action_tuples=action_tuples,
    domains_from_actions=domains_from_actions, source_site_domains=tcad_site_domains,
    build_source_domain_registry=tcad_domain_registry,
    filter_semantic_candidates=filter_semantic_candidates,
    validate_assignments=validate_assignments,
)
