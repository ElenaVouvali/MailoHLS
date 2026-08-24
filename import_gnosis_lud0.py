#!/usr/bin/env python3
"""Import pinned GNΩSIS rodinia_lud_0_baseline_0 assets for Stage-1 only."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path.cwd()
PIN = "8888db17a6f344307de0a27cb431582f3f6fd093"
BASE = f"https://raw.githubusercontent.com/aferikoglou/GNWSIS/{PIN}/data"
FILES = {
    "Data/ApplicationDataset/rodinia_lud_0_baseline_0/kernel_info.txt": (
        "ApplicationDataset/rodinia_lud_0_baseline_0/kernel_info.txt",
        "fc374de8729e578d42331e0cdf571cab6950171e",
    ),
    "Data/ApplicationDataset/rodinia_lud_0_baseline_0/lud.cpp": (
        "ApplicationDataset/rodinia_lud_0_baseline_0/lud.cpp",
        "41c2f1da8b1c497bed9445fd016a216ee8f8f265",
    ),
    "Data/ApplicationDataset/rodinia_lud_0_baseline_0/lud.h": (
        "ApplicationDataset/rodinia_lud_0_baseline_0/lud.h",
        "9e48a52f919cfe93fb7c9312d1df79d8d4510f60",
    ),
    "Data/ApplicationDataset/rodinia_lud_0_baseline_0/src_info.json": (
        "ApplicationDataset/rodinia_lud_0_baseline_0/src_info.json",
        "e25fc8abcabfb4c3539d3ebafc8418c1708fe788",
    ),
    "Data/ApplicationAPLMapping/rodinia_lud_0_baseline_0.txt": (
        "ApplicationAPLMapping/rodinia_lud_0_baseline_0.txt",
        "82909ba22e152e98885a29767960c89fb454081a",
    ),
    "Data/CSVS/rodinia_lud_0_baseline_0.csv": (
        "CSVS/rodinia_lud_0_baseline_0.csv",
        "64b5c4ffbf46b7017dffd54ce4686f4de37271bd",
    ),
}


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def fetch(relative: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
        return response.read()


def main() -> None:
    if not (ROOT / ".git").exists():
        raise SystemExit("Run this script from the MailoHLS repository root.")

    for destination, (source, expected_sha) in FILES.items():
        path = ROOT / destination
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing imported asset: {path}")
        data = fetch(source)
        actual = git_blob_sha(data)
        if actual != expected_sha:
            raise RuntimeError(
                f"Pinned GNΩSIS blob mismatch for {source}: {actual} != {expected_sha}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"[IMPORT] {destination} blob={actual}")

    table = ROOT / "Data/ApplicationInformation.csv"
    row = "rodinia_lud_0_baseline_0,workload,lud.cpp,cpp"
    text = table.read_text(encoding="utf-8")
    if row not in text.splitlines():
        if not text.endswith("\n"):
            text += "\n"
        table.write_text(text + row + "\n", encoding="utf-8")
        print("[IMPORT] appended Data/ApplicationInformation.csv row")

    print(f"Imported rodinia_lud_0_baseline_0 from GNΩSIS commit {PIN}")


if __name__ == "__main__":
    main()
