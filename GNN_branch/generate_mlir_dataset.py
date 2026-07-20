#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--app", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="GNN_branch/MLIR_Graphs")
    parser.add_argument("--cgeist", required=True)
    parser.add_argument("--cflag", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    manifest = root / "Data" / "ApplicationInformation.csv"
    data_root = root / "Data" / "ApplicationDataset"
    generator = root / "GNN_branch" / "mlir_graph_gen.py"
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.app:
        rows = [r for r in rows if r["app_name"] == args.app]
    elif not args.all:
        parser.error("Specify --app NAME or --all")

    if not rows:
        raise RuntimeError(f"No application matched {args.app!r}")

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"

    for row in rows:
        app = row["app_name"]
        source = data_root / app / row["file_name"]
        output = output_dir / f"{app}.gexf"
        mlir_output = output_dir / f"{app}.mlir"

        if not source.is_file():
            raise FileNotFoundError(source)

        command = [
            sys.executable,
            str(generator),
            str(source),
            "--kernel",
            row["top_level_function"],
            "--output",
            str(output),
            "--mlir-output",
            str(mlir_output),
            "--cgeist",
            args.cgeist,
        ]

        for flag in args.cflag:
            command.append(f"--cflag={flag}")

        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())