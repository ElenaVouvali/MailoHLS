"""Versioned prompt and directive-serialization contract for MailoHLS."""

import random
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
ADAPTED_DEVICE_TOKEN = "<DEV=ADAPTED>"
PERIOD_TOKEN_MAP = {10.0: "<CLK=10NS>", 5.0: "<CLK=5NS>", 3.33: "<CLK=3P33NS>"}
CLOCK_ANCHOR_TOKEN = "<CLOCK>"
AUTO_PERIOD_TOKEN = "<CLK=AUTO>"
TARGET_PLACEHOLDER_TOKENS = [f"<L{i}>" for i in range(1, 65)]
SOURCE_PLACEHOLDER_TOKENS = [f"<SRC_L{i}>" for i in range(1, 65)]
DEVICE_RESOURCES = {
    "xczu7ev-ffvc1156-2-e": {"BRAM_18K": 624, "DSP": 1728, "FF": 460800, "LUT": 230400},
    "xcu200-fsgd2104-2-e": {"BRAM_18K": 4320, "DSP": 6840, "FF": 2364480, "LUT": 1182240},
}
RESOURCE_KEYS = ("BRAM_18K", "DSP", "FF", "LUT")
AVAIL_FIELD_BY_RESOURCE = {
    "BRAM_18K": "avail_bram", "DSP": "avail_dsp", "FF": "avail_ff", "LUT": "avail_lut"
}

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


def _norm_device(value) -> str:
    return str(value or "").strip().lower()


def _norm_clock(value) -> float:
    return round(float(value), 2)


def period_token_from_clock(clock_period) -> str:
    clock = _norm_clock(clock_period)
    for known, token in PERIOD_TOKEN_MAP.items():
        if abs(clock - known) < 0.02:
            return token
    return f"<CLK={f'{clock:.2f}'.rstrip('0').rstrip('.').replace('.', 'P')}NS>"


def target_prompt_fields(row: dict, device_mode: str = "standard", device_token_dropout: float = 0.0) -> dict:
    """Return the complete, versioned target-platform prompt fields for a dataset row."""
    row = row or {}
    device = _norm_device(row.get("device", row.get("Device", "")))
    caps = DEVICE_RESOURCES.get(device)
    if caps is None:
        raise ValueError(f"Unsupported device: {device!r}")
    device_token = ADAPTED_DEVICE_TOKEN if device_mode == "device_adapt" else DEVICE_TOKEN_MAP.get(device, UNKNOWN_DEVICE_TOKEN)
    if device_mode == "resource_dropout_ablation" and device_token_dropout > 0 and random.random() < device_token_dropout:
        device_token = UNKNOWN_DEVICE_TOKEN
    clock_value = row.get("clock_period", row.get("Clock_Period_nsec"))
    if clock_value in (None, ""):
        raise ValueError(f"Row for {row.get('kernel_name', '<unknown>')} has no clock")
    selected_clock = _norm_clock(clock_value)
    frequency_mode = str(row.get("frequency_mode", "specified")).lower()
    if frequency_mode == "auto":
        raw_supported = row.get("available_clock_periods")
        if not isinstance(raw_supported, (list, tuple)) or not raw_supported:
            raise ValueError("Automatic-clock rows require available_clock_periods")
        supported = sorted({_norm_clock(value) for value in raw_supported})
        if selected_clock not in supported:
            raise ValueError("Selected clock is absent from available_clock_periods")
    else:
        supported = [selected_clock]
    available = {
        resource: int(round(float(row.get(field)))) if row.get(field) not in (None, "") else caps[resource]
        for resource, field in AVAIL_FIELD_BY_RESOURCE.items()
    }
    pct = lambda resource: 100.0 * available[resource] / float(caps[resource])
    return {
        "device_name": device, "device_token": device_token,
        "period_token": AUTO_PERIOD_TOKEN if frequency_mode == "auto" else period_token_from_clock(selected_clock),
        "supported_clock_periods": ", ".join(f"{value:g} ns" for value in supported),
        "avail_bram": available["BRAM_18K"], "avail_dsp": available["DSP"],
        "avail_ff": available["FF"], "avail_lut": available["LUT"],
        "avail_bram_pct": pct("BRAM_18K"), "avail_dsp_pct": pct("DSP"),
        "avail_ff_pct": pct("FF"), "avail_lut_pct": pct("LUT"),
    }


def build_prompt(code: str, obj_mode: str, prompt_fields: dict) -> str:
    return (
        PROMPT_TEMPLATE.format(obj_token=GOALS[obj_mode]["token"], **prompt_fields)
        + replace_source_labels_with_tokens(code)
        + PROMPT_SUFFIX
    )
