"""Build inference cases for policy/clock experiments.

This is intentionally a thin adapter around the canonical AUTO cases: it
attaches source text from the SFT dataset, preserves budget identity, and can
expand each case to the complete device clock menu.  It does not invent QoR
labels or perform nearest-neighbour matching.
"""
import argparse, json
import re
from pathlib import Path
from collections import defaultdict
from LLM_branch.common.mailohls_contract import supported_clock_periods

DIRECTIVE_RE = re.compile(
    r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}", re.IGNORECASE
)

def build_cases(dataset_rows, auto_rows, objective="PARETO_ADP", cases_per_kernel_device=4,
                expand_all_clocks=False):
    candidates = defaultdict(list)
    for row in dataset_rows:
        kernel = row.get("kernel_name", row.get("kernel"))
        source = row.get("input") or row.get("code") or row.get("source_text")
        if not source and row.get("source_file"):
            try: source = Path(row["source_file"]).read_text(encoding="utf-8")
            except OSError: pass
        if kernel and source:
            candidates[kernel].append(source)
    sources = {}
    for kernel, values in candidates.items():
        source = max(values, key=lambda value: len(DIRECTIVE_RE.findall(value)))
        if not DIRECTIVE_RE.search(source):
            raise ValueError(f"{kernel}: no directive placeholders in source")
        sources[kernel] = source
    grouped = defaultdict(list)
    for row in auto_rows:
        kernel = row.get("kernel_name", row.get("kernel"))
        device = row.get("device")
        if kernel and device and kernel in sources:
            grouped[(kernel, device)].append(row)
    out = []
    for key in sorted(grouped):
        for row in grouped[key][:max(1, int(cases_per_kernel_device))]:
            kernel, device = key
            if expand_all_clocks:
                # Policy-bank generation uses the exact specified-clock Stage-2
                # decoder once per clock; do not leave an AUTO menu without a
                # scalar clock_period for the prompt contract.
                for clock in supported_clock_periods(device):
                    case = dict(row)
                    case.update({"kernel_name": kernel, "input": sources[kernel],
                                 "objective": objective, "frequency_mode": "specified",
                                 "clock_period": float(clock),
                                 "selected_clock_period": float(clock),
                                 "selected_clock_period_ns": float(clock)})
                    out.append(case)
            else:
                case = dict(row)
                case.update({"kernel_name": kernel, "input": sources[kernel],
                             "objective": objective, "frequency_mode": "auto"})
                out.append(case)
    if not out:
        raise ValueError("no AUTO cases could be matched to source text")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--auto_cases", required=True)
    p.add_argument("--objective", choices=("PARETO_LATENCY", "PARETO_AREA", "PARETO_ADP"), default="PARETO_ADP")
    p.add_argument("--cases_per_kernel_device", type=int, default=4)
    p.add_argument("--expand_all_clocks", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in open(a.dataset, encoding="utf-8") if x.strip()]
    sources_path = Path(a.dataset).with_suffix(".sources.json")
    if sources_path.is_file():
        payload = json.loads(sources_path.read_text(encoding="utf-8"))
        templates = payload.get("templates", {})
        for row in rows:
            if not (row.get("input") or row.get("code") or row.get("source_text")):
                key = str(row.get("source_key", row.get("kernel_name", row.get("kernel", ""))))
                if key in templates:
                    row["input"] = templates[key]
    auto = [json.loads(x) for x in open(a.auto_cases, encoding="utf-8") if x.strip()]
    result = build_cases(rows, auto, a.objective, a.cases_per_kernel_device, a.expand_all_clocks)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        for row in result: f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(result), "output": a.output}))

if __name__ == "__main__": main()
