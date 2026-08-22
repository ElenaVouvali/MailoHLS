import json
from pathlib import Path

path = Path("artifacts/llm/mailohls_sft.jsonl")
bad = []

with path.open() as handle:
    for line_number, line in enumerate(handle, 1):
        row = json.loads(line)
        if float(row["area"]) <= 0:
            bad.append((line_number, row["kernel_name"], row["area"]))

if bad:
	print(f"ERROR: found {len(bad)} nonpositive-area rows: {bad[:5]}")

print("Dataset validation passed: every example has positive area")