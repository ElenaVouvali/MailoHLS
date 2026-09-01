#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path


ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9_]+_L\d+\})\s*=\s*(\S+)\s*$",
    re.I,
)

PH_RE = re.compile(
    r"auto\{_[A-Z0-9_]+_L\d+\}",
    re.I,
)

VAR_RE = re.compile(
    r"\bvariable=([A-Za-z_]\w*)"
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def assignment_map(text):
    result = {}

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("<L") and line.endswith(">"):
            continue

        match = ASSIGN_RE.fullmatch(line)
        if not match:
            raise ValueError(
                f"Malformed prediction line: {line!r}"
            )

        result[match.group(1).upper()] = match.group(2)

    return result


def directive_value(values, token):
    key = token.upper()

    if key not in values:
        raise KeyError(
            f"Missing directive value for {token}"
        )

    return values[key]


def materialize(template, prediction):
    values = assignment_map(prediction)
    output = []

    for raw in template.splitlines():
        line = raw
        low = line.lower()

        # PIPELINE
        if (
            "#pragma hls pipeline" in low
            and "auto{_pipe_" in low
        ):
            token = PH_RE.search(line).group(0)
            value = directive_value(values, token)

            if value == "0":
                continue

            if value != "1":
                raise ValueError(
                    f"Unsupported PIPE value "
                    f"{value} for {token}"
                )

            output.append(
                PH_RE.sub("1", line, count=1)
            )
            continue

        # UNROLL
        if (
            "#pragma hls unroll" in low
            and "auto{_unroll_" in low
        ):
            token = PH_RE.search(line).group(0)
            value = directive_value(values, token)

            if int(value) == 0:
                continue

            output.append(
                PH_RE.sub(value, line, count=1)
            )
            continue

        # ARRAY_PARTITION
        if (
            "#pragma hls array_partition" in low
            and "auto{_array_t_" in low
        ):
            tokens = [
                token.upper()
                for token in PH_RE.findall(line)
            ]

            type_token = next(
                t for t in tokens
                if "_ARRAY_T_" in t
            )
            factor_token = next(
                t for t in tokens
                if "_ARRAY_F_" in t
            )
            dim_token = next(
                t for t in tokens
                if "_ARRAY_D_" in t
            )

            partition_type = directive_value(
                values, type_token
            ).lower()

            factor = directive_value(
                values, factor_token
            )

            dim = directive_value(
                values, dim_token
            )

            var_match = VAR_RE.search(line)

            if var_match is None:
                raise ValueError(
                    f"Cannot parse array variable: {line}"
                )

            variable = var_match.group(1)
            indent = line[
                :len(line) - len(line.lstrip())
            ]

            if partition_type == "none":
                continue

            if partition_type == "complete":
                output.append(
                    f"{indent}"
                    f"#pragma HLS ARRAY_PARTITION "
                    f"variable={variable} "
                    f"type=complete dim={dim}"
                )

            elif partition_type in {
                "block",
                "cyclic",
            }:
                if int(factor) <= 0:
                    raise ValueError(
                        f"{partition_type} requires "
                        f"positive factor"
                    )

                output.append(
                    f"{indent}"
                    f"#pragma HLS ARRAY_PARTITION "
                    f"variable={variable} "
                    f"type={partition_type} "
                    f"factor={factor} "
                    f"dim={dim}"
                )

            else:
                raise ValueError(
                    "Unsupported ARRAY_PARTITION "
                    f"type {partition_type!r}"
                )

            continue

        output.append(line)

    text = "\n".join(output) + "\n"

    if "auto{" in text:
        raise RuntimeError(
            "Unmaterialized MailoHLS placeholder remains"
        )

    return text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--queue", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--top",
        default="krnl_KALMAN",
    )

    args = parser.parse_args()

    cases = {
        row["context_id"]: row
        for row in load_jsonl(args.cases)
    }

    queue = load_jsonl(args.queue)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    manifest = []

    for item in queue:
        case = cases[item["context_id"]]

        digest = hashlib.sha256(
            (
                item["context_id"]
                + "\n"
                + item["canonical_prediction"]
            ).encode()
        ).hexdigest()[:20]

        job = root / digest
        job.mkdir(parents=True, exist_ok=True)

        source = materialize(
            case["input"],
            item["canonical_prediction"],
        )

        (job / "kernel.cpp").write_text(
            source,
            encoding="utf-8",
        )

        clock = float(
            item["clock_period_ns"]
        )

        device = item["device"]

        tcl = f"""open_project -reset hls_project
set_top {args.top}
add_files kernel.cpp
open_solution "solution1" -flow_target vitis
set_part {{{device}}}
create_clock -period {clock:g} -name default
csynth_design
exit
"""

        (job / "run_hls.tcl").write_text(
            tcl,
            encoding="utf-8",
        )

        metadata = dict(item)
        metadata["job_id"] = digest
        metadata["top"] = args.top

        (job / "job.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest.append(metadata)

    with (
        root / "jobs.jsonl"
    ).open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )

    print(
        f"[DONE] {len(manifest)} jobs -> {root}"
    )


if __name__ == "__main__":
    main()