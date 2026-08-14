"""Versioned prompt and directive-serialization contract for MailoHLS."""

import re
from collections import defaultdict

PROMPT_SCHEMA_VERSION = 1

PROMPT_TEMPLATE = """
### Role: Expert FPGA/HLS engineer.

### Task:
The kernel marks each directive site with a source marker <SRC_Lk>.
Select the directive RHS values for the optimization goal and target platform.
If the clock is <CLK=AUTO>, also select the best supported clock period.
Anchors and directive names are fixed by the source code.

### Target Platform
Device class: {device_token}
Device name: {device_name}
Target clock period: {period_token}
Supported measured clock periods: {supported_clock_periods}

Available resources:
BRAM_18K={avail_bram} ({avail_bram_pct:.1f}% of device)
DSP={avail_dsp} ({avail_dsp_pct:.1f}% of device)
FF={avail_ff} ({avail_ff_pct:.1f}% of device)
LUT={avail_lut} ({avail_lut_pct:.1f}% of device)

### Objective
{obj_token}

### Kernel
""".lstrip()

PROMPT_SUFFIX = "\n\n### Selected Clock and Directives\n"
GOALS = {
    "PARETO_LATENCY_EXTREME": {"token": "<OBJ=PARETO_LATENCY_EXTREME>", "tag": "pareto_latency_extreme"},
    "PARETO_ADP": {"token": "<OBJ=PARETO_ADP>", "tag": "pareto_adp"},
    "PARETO_AREA_EXTREME": {"token": "<OBJ=PARETO_AREA_EXTREME>", "tag": "pareto_area_extreme"},
}
DEVICE_TOKEN_MAP = {
    "xczu7ev-ffvc1156-2-e": "<DEV=XCZU7EV_FFVC1156_2E>",
    "xcu200-fsgd2104-2-e": "<DEV=XCU200_FSGD2104_2E>",
}
UNKNOWN_DEVICE_TOKEN = "<DEV=UNKNOWN>"
PERIOD_TOKEN_MAP = {10.0: "<CLK=10NS>", 5.0: "<CLK=5NS>", 3.33: "<CLK=3P33NS>"}
CLOCK_ANCHOR_TOKEN = "<CLOCK>"
AUTO_PERIOD_TOKEN = "<CLK=AUTO>"
TARGET_PLACEHOLDER_TOKENS = [f"<L{i}>" for i in range(1, 65)]
SOURCE_PLACEHOLDER_TOKENS = [f"<SRC_L{i}>" for i in range(1, 65)]

SOURCE_LABEL_RE = re.compile(r"^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)", re.IGNORECASE)
TARGET_LINE_LABEL_RE = re.compile(r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}\s*=", re.IGNORECASE)
ANCHOR_OR_ASSIGN_RE = re.compile(
    r"^\s*(<L\d+>|auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}\s*=\s*.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def replace_source_labels_with_tokens(text: str) -> str:
    if not isinstance(text, str):
        return text
    output = []
    for line in text.splitlines():
        stripped = line.lstrip()
        match = SOURCE_LABEL_RE.match(stripped)
        if not match:
            output.append(line)
            continue
        label = (match.group(1) or match.group(2)).upper()
        rest = stripped[match.end():].lstrip()
        token = f"<SRC_{label}>"
        output.append(f"{line[:len(line) - len(stripped)]}{token}{' ' + rest if rest else ''}")
    return "\n".join(output)


def canonical_pragma_serialization(source_text: str, target_text: str) -> str:
    order = []
    for line in source_text.splitlines():
        match = SOURCE_LABEL_RE.match(line.lstrip())
        if match:
            label = (match.group(1) or match.group(2)).upper()
            if label not in order:
                order.append(label)
    grouped = defaultdict(list)
    extras = []
    for raw in target_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = TARGET_LINE_LABEL_RE.search(line)
        (grouped[match.group(1).upper()] if match else extras).append(line)
    result = []
    for label in order + sorted(set(grouped) - set(order)):
        result.extend(grouped[label])
    result.extend(extras)
    return "\n".join(result)


def canonicalize_generation(text: str) -> str:
    return "\n".join(match.group(0).strip() for match in ANCHOR_OR_ASSIGN_RE.finditer(text)).strip()


def build_prompt(code: str, obj_mode: str, prompt_fields: dict) -> str:
    return (
        PROMPT_TEMPLATE.format(obj_token=GOALS[obj_mode]["token"], **prompt_fields)
        + replace_source_labels_with_tokens(code)
        + PROMPT_SUFFIX
    )
