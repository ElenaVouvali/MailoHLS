"""Build oracle-free public-menu clock cases from measured JSONL rows."""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from LLM_branch.common.mailohls_contract import supported_clock_periods


def _num(row, *names):
    for name in names:
        if name in row:
            try: return float(row[name])
            except (TypeError, ValueError): pass
    return None


def build_cases(rows, budgets_per_case=16, seed=123):
    groups = defaultdict(list)
    for row in rows:
        kernel = row.get("kernel_name", row.get("kernel"))
        device = row.get("device")
        clock = _num(row, "clock_period", "selected_clock_period", "clock_period_ns")
        lat = _num(row, "latency", "latency_ms", "latency_ms_mean")
        area = _num(row, "area", "area_mm2")
        if not kernel or not device or clock is None or lat is None or area is None:
            continue
        try: menu = list(supported_clock_periods(device))
        except ValueError: continue
        if clock not in menu: continue
        groups[(kernel, device)].append((row, clock, max(lat, 0.0) * max(area, 0.0625)))
    out = []
    for (kernel, device), items in sorted(groups.items()):
        by_clock = defaultdict(list)
        for row, clock, adp in items: by_clock[clock].append((row, adp))
        # Public menu deliberately includes clocks with no feasible measurement.
        choices = [(clock, min(v for _, v in by_clock[clock])) for clock in by_clock]
        if not choices: continue
        gold_clock, gold_adp = min(choices, key=lambda x: x[1])
        base = items[0][0]
        out.append({"kernel": kernel, "device": device,
                    "frequency_mode": "auto", "available_clock_periods": list(supported_clock_periods(device)),
                    "gold_clock_period": gold_clock, "gold_adp": gold_adp,
                    "resource_budget": base.get("resource_budget", base.get("budgets", {})),
                    "source_key": base.get("source_key", kernel)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--split_json")
    ap.add_argument("--objective", default="PARETO_ADP"); ap.add_argument("--budgets_per_case", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123); ap.add_argument("--output_dir", required=True)
    a = ap.parse_args(); rows = [json.loads(x) for x in open(a.dataset) if x.strip()]
    cases = build_cases(rows, a.budgets_per_case, a.seed)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    # Keep a deterministic, auditable split; callers may replace this with the locked family split.
    cut = max(1, int(len(cases) * .8)) if cases else 0
    for name, vals in (("train", cases[:cut]), ("val", cases[cut:])):
        with open(Path(a.output_dir) / f"{name}.jsonl", "w") as f:
            for x in vals: f.write(json.dumps(x, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), "train": cut, "val": len(cases)-cut}, indent=2))


if __name__ == "__main__": main()
