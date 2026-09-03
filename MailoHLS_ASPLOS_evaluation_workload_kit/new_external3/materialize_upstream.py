#!/usr/bin/env python3
"""Materialize three pinned external HLS workloads into the MailoHLS source/action contract.

The script intentionally derives the prepared sources from pinned upstream benchmark
files instead of maintaining a hand-copied fork.  It verifies each upstream Git blob,
flattens benchmark-local includes into one translation unit, removes pre-existing HLS
optimization pragmas (interface pragmas are preserved), identifies statically bounded
loop/local-array action sites without QoR information, and emits:

  kernel.{c,cpp}
  kernel_info.txt
  kernel_placeholders.{c,cpp}
  code_to_memory_outputs/meta/*
  ACTION_AUDIT.json

Run on any machine with HTTPS access, or pass --source-cache to an already populated
cache directory.  No MailoHLS training/validation result is consulted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
MAX_SLOTS = 64

CHSTONE_COMMIT = "2b7e20ffd3365016faf1e4e2b86496a5c95445fb"
ROSETTA_COMMIT = "2feed1ce02d871603bf1fc344a65051837ac780f"

# path -> git blob SHA at the pinned commit
SPECS = {
    "chstone-aes": {
        "repo": "ferrandi/CHStone",
        "commit": CHSTONE_COMMIT,
        "root": "aes/aes.c",
        "top": "chstone_aes",
        "language": "c",
        "files": {
            "aes/aes.c": "a0cdf2d50b880c312a2004ec79bcfd3524e7ef2f",
            "aes/aes.h": "792d08f0a425d78cb8c7bb84eb88000969e0681d",
            "aes/aes_dec.c": "25173f36bbb8082cc7dea3d9e05456c5d6cd244d",
            "aes/aes_enc.c": "edeb665dd0702f8d9cf90283eca3c15b8cfaf932",
            "aes/aes_func.c": "9450dddcfd37282c3fbfe20d4bdd909a77da7258",
            "aes/aes_key.c": "cac8412c9cb6a7428664a8eb3c7d86861fa568d5",
        },
        "rename_main": True,
    },
    "chstone-jpeg": {
        "repo": "ferrandi/CHStone",
        "commit": CHSTONE_COMMIT,
        "root": "jpeg/main.c",
        "top": "chstone_jpeg",
        "language": "c",
        "files": {
            "jpeg/chenidct.c": "1e0278e0a02b3876e8a823dc64392a425f57e74d",
            "jpeg/decode.c": "a6703817ec727b685ae9cd4606504f81cebcba89",
            "jpeg/decode.h": "bb7f5af264de5777f7366ad990bc5264634f0897",
            "jpeg/global.h": "267a24c35c13b01de924a09fe74245cd48d86c6e",
            "jpeg/huffman.c": "bf1528feb68fc80c1665aa105690a6e8a74ebe02",
            "jpeg/huffman.h": "7c0b4fea06e41558d2251f868701ac01f842a4b1",
            "jpeg/init.h": "2cd639a8857e6346e32c28f41b6b9bfc20934b90",
            "jpeg/jfif_read.c": "c23734474d36e88eaa5d9ab6cd7957a2ae075d43",
            "jpeg/jpeg2bmp.c": "b71c98d1f828b1174d1e1e4a92ff6856eede1b1f",
            "jpeg/main.c": "91696ff2b910eafc5d41d785e00c69ba77edb55f",
            "jpeg/marker.c": "48802652b54dca296c1703c7461e3c697923091b",
        },
        "rename_main": True,
    },
    "rosetta-3d-rendering": {
        "repo": "cornell-zhang/rosetta",
        "commit": ROSETTA_COMMIT,
        "root": "3d-rendering/src/ocl/rendering.cpp",
        "top": "rendering",
        "language": "cpp",
        "files": {
            "3d-rendering/src/ocl/rendering.cpp": "2f3cf6b1da26e655fd0dd471dbb1c6866697a093",
            "3d-rendering/src/host/typedefs.h": "4227d2d17617c3d244eddb4c603370a45d3a984c",
        },
        "include_aliases": {
            # rendering.cpp lives in src/ocl but includes host/typedefs.h relative to src.
            "3d-rendering/src/ocl/host/typedefs.h": "3d-rendering/src/host/typedefs.h",
        },
        "rename_main": False,
    },
}

OPT_PRAGMA_RE = re.compile(
    r"^\s*#\s*pragma\s+HLS\s+(PIPELINE|UNROLL|ARRAY_PARTITION|DATAFLOW|INLINE|LOOP_FLATTEN|LOOP_TRIPCOUNT)\b",
    re.IGNORECASE,
)
LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$")
CONST_INT_RE = re.compile(
    r"^\s*(?:static\s+)?const\s+(?:unsigned\s+)?(?:long\s+)?int\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*;\s*$"
)
FOR_RE = re.compile(r"\bfor\s*\(([^;]*);([^;]*);([^)]*)\)")
ARRAY_RE = re.compile(
    r"^\s*(?!typedef\b)(?!return\b)(?:static\s+)?(?:const\s+)?"
    r"[A-Za-z_][A-Za-z0-9_:\s<>\*&]*?\s+([A-Za-z_]\w*)\s*"
    r"((?:\[[^\]]+\])+)(?:\s*=.*)?;\s*$"
)
DIM_RE = re.compile(r"\[([^\]]+)\]")
FUNC_HEAD_RE = re.compile(
    r"(?ms)^[ \t]*(?:extern\s+\"C\"\s*)?(?:[A-Za-z_][\w:<>,\s\*&]*?\s+)?"
    r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MailoHLS-ASPLOS-external-workload-materializer/1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def acquire(spec: dict, cache_root: Path, offline: bool, local_root: Path | None = None) -> dict[str, str]:
    texts = {}
    for path, expected_blob in spec["files"].items():
        local = cache_root / spec["repo"].replace("/", "__") / spec["commit"] / path
        if local.is_file():
            data = local.read_bytes()
        elif local_root is not None and (local_root / path).is_file():
            data = (local_root / path).read_bytes()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        else:
            if offline:
                raise FileNotFoundError(
                    f"Offline source miss: cache={local}; local_root={local_root}"
                )
            data = fetch_bytes(raw_url(spec["repo"], spec["commit"], path))
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        actual = git_blob_sha(data)
        if actual != expected_blob:
            raise RuntimeError(f"Pinned upstream blob mismatch for {path}: {actual} != {expected_blob}")
        texts[path] = data.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    return texts


def resolve_include(current: str, include: str, spec: dict, texts: dict[str, str]) -> str | None:
    cur = PurePosixPath(current)
    candidate = str(cur.parent / include)
    aliases = spec.get("include_aliases", {})
    candidate = aliases.get(candidate, candidate)
    if candidate in texts:
        return candidate
    # CHStone files include siblings from the benchmark directory.
    alt = str(PurePosixPath(spec["root"]).parent / include)
    alt = aliases.get(alt, alt)
    return alt if alt in texts else None


def flatten(path: str, spec: dict, texts: dict[str, str], visited: set[str]) -> str:
    if path in visited:
        return f"/* MailoHLS flatten: already included {path} */\n"
    visited.add(path)
    out = [f"/* ===== BEGIN UPSTREAM FILE: {path} ===== */\n"]
    for raw in texts[path].splitlines(keepends=True):
        m = LOCAL_INCLUDE_RE.match(raw.rstrip("\n"))
        if m:
            resolved = resolve_include(path, m.group(1), spec, texts)
            if resolved is not None:
                out.append(flatten(resolved, spec, texts, visited))
                continue
        out.append(raw)
    out.append(f"\n/* ===== END UPSTREAM FILE: {path} ===== */\n")
    return "".join(out)


def strip_optimization_pragmas(source: str) -> tuple[str, list[str]]:
    kept, removed = [], []
    for line in source.splitlines(keepends=True):
        if OPT_PRAGMA_RE.match(line):
            removed.append(line.strip())
            kept.append("/* MailoHLS: removed pre-existing HLS optimization pragma */\n")
        else:
            kept.append(line)
    return "".join(kept), removed


def rename_main(source: str, top: str, language: str) -> str:
    # Both CHStone roots use `int\nmain ()` and their Vitis scripts compile with
    # -Dmain=chstone_main.  Rename only the function definition and expose C linkage.
    pat = re.compile(r"\bint\s+main\s*\(\s*\)\s*\{")
    linkage = 'extern \"C\" ' if language == 'cpp' else ''
    repl = f'{linkage}int {top} ()\n{{'
    out, count = pat.subn(repl, source, count=1)
    if count != 1:
        raise RuntimeError("Could not uniquely rename CHStone main()")
    return out


# ---- small constant-expression evaluator ---------------------------------
ALLOWED_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b if b and a % b == 0 else a / b,
    ast.Mod: lambda a, b: a % b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
}
ALLOWED_UN = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval_node(node: ast.AST, constants: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in constants:
        return int(constants[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BIN:
        return int(ALLOWED_BIN[type(node.op)](_eval_node(node.left, constants), _eval_node(node.right, constants)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UN:
        return int(ALLOWED_UN[type(node.op)](_eval_node(node.operand, constants)))
    raise ValueError(ast.dump(node))


def eval_const(expr: str, constants: dict[str, int]) -> int | None:
    expr = re.sub(r"/\*.*?\*/", "", expr)
    expr = re.sub(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", expr)
    expr = expr.strip()
    if not expr or any(token in expr for token in ("[", "]", "?", ":", "sizeof")):
        return None
    # Strip a few scalar casts common in C benchmark code.
    expr = re.sub(r"\((?:unsigned\s+)?(?:char|short|int|long|bit\d+)\)", "", expr)
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree.body, constants)
        return int(value)
    except Exception:
        return None


def collect_constants(source: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    pending: list[tuple[str, str]] = []
    for line in source.splitlines():
        m = DEFINE_RE.match(line)
        if m and "(" not in m.group(1):
            expr = m.group(2).split("//", 1)[0].strip()
            pending.append((m.group(1), expr))
            continue
        m = CONST_INT_RE.match(line)
        if m:
            pending.append((m.group(1), m.group(2)))
    changed = True
    while changed:
        changed = False
        rest = []
        for name, expr in pending:
            v = eval_const(expr, constants)
            if v is None:
                rest.append((name, expr))
            else:
                constants[name] = v
                changed = True
        pending = rest
    return constants


def function_spans(source: str) -> list[tuple[int, int, str]]:
    spans = []
    for m in FUNC_HEAD_RE.finditer(source):
        name = m.group(1)
        if name in {"if", "for", "while", "switch"}:
            continue
        open_pos = source.find("{", m.start(), m.end())
        if open_pos < 0:
            continue
        depth = 0
        in_str = in_chr = False
        esc = False
        i = open_pos
        while i < len(source):
            ch = source[i]
            if esc:
                esc = False
            elif ch == "\\" and (in_str or in_chr):
                esc = True
            elif ch == '"' and not in_chr:
                in_str = not in_str
            elif ch == "'" and not in_str:
                in_chr = not in_chr
            elif not in_str and not in_chr:
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append((m.start(), i + 1, name))
                        break
            i += 1
    # Remove obvious nested false-positive spans by keeping the smallest named
    # span for each start and allowing true functions only at brace depth zero.
    spans.sort()
    top = []
    last_end = -1
    for s, e, n in spans:
        if s >= last_end:
            top.append((s, e, n)); last_end = e
    return top


def line_offsets(source: str) -> tuple[list[str], list[int]]:
    lines = source.splitlines(keepends=True)
    starts, p = [], 0
    for line in lines:
        starts.append(p); p += len(line)
    return lines, starts


def containing_function(offset: int, spans: list[tuple[int, int, str]]) -> str | None:
    for s, e, name in spans:
        if s <= offset < e:
            return name
    return None


def infer_for_bound(line: str, constants: dict[str, int]) -> int | None:
    m = FOR_RE.search(line)
    if not m:
        return None
    init, cond, step = (x.strip() for x in m.groups())
    # Identify the induction variable from the condition and evaluate the other side.
    cm = re.match(r"\s*([A-Za-z_]\w*)\s*(<=|<|>=|>)\s*(.+?)\s*$", cond)
    reverse = False
    if not cm:
        cm2 = re.match(r"\s*(.+?)\s*(<=|<|>=|>)\s*([A-Za-z_]\w*)\s*$", cond)
        if not cm2:
            return None
        bound_expr, op, var = cm2.group(1), cm2.group(2), cm2.group(3)
        reverse = True
    else:
        var, op, bound_expr = cm.group(1), cm.group(2), cm.group(3)
    bound = eval_const(bound_expr, constants)
    if bound is None:
        return None
    # MailoHLS uses this field to construct a finite unroll proposal domain;
    # matching the existing dataset convention, store a positive static bound
    # rather than claiming a dynamic exact trip count.
    if bound <= 1:
        return None
    return int(bound)


def array_dims(line: str, constants: dict[str, int]) -> tuple[str, list[int]] | None:
    m = ARRAY_RE.match(line)
    if not m:
        return None
    name = m.group(1)
    dims = []
    for expr in DIM_RE.findall(m.group(2)):
        v = eval_const(expr, constants)
        if v is None or v < 2:
            return None
        dims.append(v)
    return (name, dims) if dims else None


def code_lines_without_comments(lines: list[str]) -> list[str]:
    """Mask comments while preserving one output record per physical source line."""
    cleaned=[]
    in_block=False
    for line in lines:
        i=0; out=[]; in_str=False; in_chr=False; esc=False
        while i < len(line):
            if in_block:
                j=line.find('*/', i)
                if j < 0:
                    i=len(line); continue
                in_block=False; i=j+2; continue
            ch=line[i]
            nxt=line[i:i+2]
            if esc:
                out.append(ch); esc=False; i+=1; continue
            if (in_str or in_chr) and ch=='\\\\':
                out.append(ch); esc=True; i+=1; continue
            if not in_chr and ch=='"':
                in_str=not in_str; out.append(ch); i+=1; continue
            if not in_str and ch=="'":
                in_chr=not in_chr; out.append(ch); i+=1; continue
            if not in_str and not in_chr and nxt=='/*':
                in_block=True; i+=2; continue
            if not in_str and not in_chr and nxt=='//':
                break
            out.append(ch); i+=1
        cleaned.append(''.join(out))
    return cleaned


def derive_actions(source: str) -> tuple[str, list[dict], dict]:
    constants = collect_constants(source)
    spans = function_spans(source)
    lines, starts = line_offsets(source)
    scan_lines = code_lines_without_comments(lines)
    candidates = []
    skipped_dynamic_loops = 0
    for idx, line in enumerate(lines):
        scan_line = scan_lines[idx]
        fn = containing_function(starts[idx], spans)
        if fn is None:
            continue
        fm = FOR_RE.search(scan_line)
        if fm:
            bound = infer_for_bound(scan_line, constants)
            if bound is None:
                skipped_dynamic_loops += 1
            else:
                candidates.append({"line": idx, "kind": "loop", "bound": bound, "function": fn})
        arr = array_dims(scan_line, constants)
        if arr is not None:
            name, dims = arr
            candidates.append({"line": idx, "kind": "array", "name": name, "dims": dims, "function": fn})

    candidates.sort(key=lambda x: (x["line"], 0 if x["kind"] == "array" else 1))
    total = len(candidates)
    selected = candidates[:MAX_SLOTS]
    for i, action in enumerate(selected, start=1):
        action["label"] = f"L{i}"

    by_line = {a["line"]: a for a in selected}
    out_lines = []
    for idx, line in enumerate(lines):
        action = by_line.get(idx)
        if action is None:
            out_lines.append(line); continue
        indent = line[:len(line) - len(line.lstrip())]
        out_lines.append(f"{indent}/*{action['label']}:*/ {line.lstrip()}")
    audit = {
        "constant_count": len(constants),
        "function_count": len(spans),
        "statically_bounded_action_candidates": total,
        "selected_action_count": len(selected),
        "selected_loop_actions": sum(a["kind"] == "loop" for a in selected),
        "selected_array_actions": sum(a["kind"] == "array" for a in selected),
        "skipped_dynamic_loop_headers": skipped_dynamic_loops,
        "max_slots": MAX_SLOTS,
        "truncated": total > MAX_SLOTS,
        "selection_policy": "all statically bounded local-array and for-loop sites in lexical order; cap at absolute L1..L64",
    }
    return "".join(out_lines), selected, audit


def kernel_info(top: str, actions: list[dict]) -> str:
    rows = [top]
    for a in actions:
        if a["kind"] == "loop":
            rows.append(f"{a['label']},loop,{a['bound']}")
        else:
            fields = [a["label"], "array", a["name"]]
            for d, extent in enumerate(a["dims"], start=1):
                fields += [str(d), str(extent)]
            rows.append(",".join(fields))
    return "\n".join(rows) + "\n"


def placeholders(labeled_source: str, actions: list[dict]) -> str:
    action_by_label = {a["label"]: a for a in actions}
    out = []
    label_re = re.compile(r"^(\s*)/\*\s*(L\d+)\s*:\s*\*/\s*(.*)$")
    for raw in labeled_source.splitlines(keepends=True):
        m = label_re.match(raw.rstrip("\n"))
        if not m:
            out.append(raw); continue
        indent, label, rest = m.groups()
        newline = "\n" if raw.endswith("\n") else ""
        out.append(f"{indent}/*{label}:*/ {rest}{newline}")
        a = action_by_label[label]
        if a["kind"] == "loop":
            out.append(f"{indent}#pragma HLS pipeline II=auto{{_PIPE_{label}}}\n")
            out.append(f"{indent}#pragma HLS unroll factor=auto{{_UNROLL_{label}}}\n")
        else:
            out.append(
                f"{indent}#pragma HLS array_partition variable={a['name']} "
                f"type=auto{{_ARRAY_T_{label}}} factor=auto{{_ARRAY_F_{label}}} "
                f"dim=auto{{_ARRAY_D_{label}}}\n"
            )
    return "".join(out)


def write_kernel(name: str, spec: dict, texts: dict[str, str], cache_root: Path) -> dict:
    flattened = flatten(spec["root"], spec, texts, set())
    flattened, removed = strip_optimization_pragmas(flattened)
    if spec.get("rename_main"):
        flattened = rename_main(flattened, spec["top"], spec["language"])
    header = (
        "/* MailoHLS ASPLOS external-evaluation source.\n"
        f" * Benchmark: {name}\n"
        f" * Upstream: {spec['repo']} @ {spec['commit']}\n"
        " * Benchmark-local includes were flattened mechanically.\n"
        " * Existing HLS optimization pragmas were removed; interface pragmas are preserved.\n"
        " * Lk sites are derived QoR-blind from statically bounded source actions.\n"
        " */\n\n"
    )
    source0 = header + flattened
    labeled, actions, audit = derive_actions(source0)
    if not actions:
        raise RuntimeError(f"{name}: no statically bounded action sites found")
    info = kernel_info(spec["top"], actions)
    prompt = placeholders(labeled, actions)

    outdir = ROOT / name
    ext = ".cpp" if spec["language"] == "cpp" else ".c"
    src = outdir / f"kernel{ext}"
    ph = outdir / f"kernel_placeholders{ext}"
    src.write_text(labeled, encoding="utf-8")
    ph.write_text(prompt, encoding="utf-8")
    (outdir / "kernel_info.txt").write_text(info, encoding="utf-8")
    audit.update({
        "kernel": name,
        "top_function": spec["top"],
        "removed_upstream_optimization_pragmas": removed,
        "source_sha256": sha256(labeled.encode()),
        "placeholder_sha256": sha256(prompt.encode()),
        "kernel_info_sha256": sha256(info.encode()),
        "actions": actions,
    })
    (outdir / "ACTION_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    meta = outdir / "code_to_memory_outputs" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, meta / f"{name}{ext}")
    shutil.copy2(ph, meta / f"{name}_placeholders{ext}")
    shutil.copy2(outdir / "kernel_info.txt", meta / "kernel_info.txt")
    for sub in ("gexf", "pt", "memory"):
        p = outdir / "code_to_memory_outputs" / sub / "README.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "Generated checkpoint-dependent artifact destination. Do not fabricate/copy another kernel's artifacts.\n",
            encoding="utf-8",
        )
    return audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", type=Path, default=ROOT / "_upstream_cache")
    ap.add_argument("--offline", action="store_true", help="Require every pinned upstream blob to exist in --source-cache or a provided local checkout")
    ap.add_argument("--chstone-root", type=Path, default=None, help="Optional local ferrandi/CHStone checkout root at the pinned commit")
    ap.add_argument("--rosetta-root", type=Path, default=None, help="Optional local cornell-zhang/rosetta checkout root at the pinned commit")
    ap.add_argument("--kernels", nargs="*", choices=sorted(SPECS), default=sorted(SPECS))
    args = ap.parse_args()
    args.source_cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in args.kernels:
        print(f"[FETCH/MATERIALIZE] {name}")
        spec = SPECS[name]
        local_root = args.chstone_root if spec["repo"] == "ferrandi/CHStone" else args.rosetta_root
        texts = acquire(spec, args.source_cache, args.offline, local_root=local_root)
        records[name] = write_kernel(name, spec, texts, args.source_cache)
        print(
            f"  actions={records[name]['selected_action_count']} "
            f"loops={records[name]['selected_loop_actions']} "
            f"arrays={records[name]['selected_array_actions']} "
            f"truncated={records[name]['truncated']}"
        )
    (ROOT / "MATERIALIZATION.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("[DONE] Prepared MailoHLS sources, action manifests, and prompt templates.")
    print("Next: python validate_package.py")


if __name__ == "__main__":
    main()
