#!/usr/bin/env python3
"""
mlir_emit_mailohls.py

Purpose
-------
First-stage MailoHLS MLIR frontend helper:
  1) Parse an already-labeled / placeholderized C/C++ HLS source and build an
     action-point sidecar schema for L1..LN loops/arrays.
  2) Create a compiler-friendly copy of the source, optionally stripping HLS
     pragma-placeholder lines so cgeist/Clang does not need to preserve them.
  3) Invoke Polygeist/cgeist to emit MLIR, first in SCF/MemRef form and then,
     when requested, in Affine form.
  4) Optionally run mlir-opt cleanup / affine-raise-from-memref on the emitted IR.

This script intentionally does NOT build the final GNN/PyG graph yet.  It is the
safe milestone before mlir_graph_gen.py: it verifies that each kernel can be
converted to useful MLIR and that MailoHLS action points are recovered in a
stable sidecar schema.

Expected downstream use
-----------------------
  C/C++ with Lk labels + placeholders
      -> ap_schema.json              (directive/action-point contract)
      -> raw_scf.mlir / affine.mlir  (compiler structural IR)
      -> later: MLIR + ap_schema -> GEXF/PyG graph with X_llm_* masks

Requirements
------------
  - Polygeist cgeist in PATH or pass --cgeist /path/to/cgeist
  - Optional mlir-opt in PATH or pass --mlir-opt /path/to/mlir-opt

Example
-------
  python3 mlir_emit_mailohls.py \
      --source knn.cpp \
      --include-dir . \
      --out-dir mlir_knn \
      --top-func workload \
      --target-device xczu7ev-ffvc1156-2-e \
      --clock-ns 10.0 \
      --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ----------------------------- Regexes -----------------------------

LABEL_PREFIX_RE = re.compile(
    r"^(?P<indent>\s*)(?P<label>L\d+)\s*:\s*(?:(?P<named>[A-Za-z_]\w*)\s*:\s*)?(?P<rest>.*)$",
    re.IGNORECASE,
)

FOR_RE = re.compile(r"\bfor\s*\((?P<header>.*)\)", re.IGNORECASE)

PRAGMA_RE = re.compile(r"^\s*#\s*pragma\s+HLS\b(?P<body>.*)$", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"auto\{(?P<key>_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+)\}", re.IGNORECASE)

# Conservative array declaration recognizer for labeled local arrays:
#   L5: float local_inputQuery[NUM_FEATURE];
#   L6: const int buf[N][M] = {...};
ARRAY_DECL_RE = re.compile(
    r"""
    ^\s*
    (?P<type>[A-Za-z_][\w:\<\>\s\*&]*?)
    \s+
    (?P<name>[A-Za-z_]\w*)
    \s*
    (?P<dims>(?:\[[^\]]+\]\s*)+)
    (?:=\s*[^;]+)?
    ;
    """,
    re.VERBOSE,
)
DIM_RE = re.compile(r"\[([^\]]+)\]")

FUNC_DEF_RE = re.compile(
    r"\b(?:extern\s+\"C\"\s*)?(?:[A-Za-z_]\w*[\w:\<\>\s\*&]*\s+)+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
    re.MULTILINE,
)
CONTROL_WORDS = {"if", "for", "while", "switch", "else", "do"}

DEFINE_RE = re.compile(r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)\s+(?P<expr>.+?)\s*$")
CONST_ASSIGN_RE = re.compile(
    r"""
    ^\s*
    (?:static\s+)?(?:constexpr\s+)?(?:const\s+)?
    (?:unsigned\s+|signed\s+|long\s+|short\s+)*
    (?:int|size_t|uint\d+_t|int\d+_t)
    \s+
    (?P<name>[A-Za-z_]\w*)
    \s*=\s*
    (?P<expr>[^;]+)
    ;
    """,
    re.VERBOSE,
)


# ----------------------------- Data classes -----------------------------

@dataclass
class FunctionSpan:
    name: str
    start_line: int   # 1-based inclusive
    end_line: int     # 1-based inclusive


@dataclass
class ActionPoint:
    label: str
    kind: str                     # "loop" or "array" or "unknown"
    function: str
    line: int                     # 1-based source line
    named_label: Optional[str]
    source_text: str
    placeholders: List[str]
    pragmas: List[str]
    loop_bound_expr: Optional[str] = None
    loop_bound_value: Optional[int] = None
    induction_var: Optional[str] = None
    array_name: Optional[str] = None
    array_type: Optional[str] = None
    array_dims_expr: Optional[List[str]] = None
    array_dims_value: Optional[List[Optional[int]]] = None


# ----------------------------- Utilities -----------------------------

def which_or_fail(path_or_name: str, tool_name: str) -> str:
    p = shutil.which(path_or_name) if not os.path.isabs(path_or_name) else path_or_name
    if p and Path(p).is_file() and os.access(p, os.X_OK):
        return str(p)
    raise FileNotFoundError(
        f"Could not find executable for {tool_name}: {path_or_name}. "
        f"Install it or pass --{tool_name.replace('-', '_')} /path/to/{tool_name}."
    )


def run_cmd(cmd: List[str], cwd: Optional[Path], log_path: Path, verbose: bool) -> int:
    if verbose:
        print("[RUN]", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write("--- STDOUT ---\n")
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
    if verbose:
        print(f"[EXIT {proc.returncode}] log: {log_path}")
    return int(proc.returncode)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------- Constants -----------------------------

def _strip_c_comments_keep_lines(text: str) -> str:
    # Preserve line count for easier source-line diagnostics.
    def repl_block(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    text = re.sub(r"/\*.*?\*/", repl_block, text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


class SafeExprEval(ast.NodeVisitor):
    """Tiny integer expression evaluator for #define constants and dimensions."""
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Div: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b,
        ast.BitOr: lambda a, b: a | b,
        ast.BitAnd: lambda a, b: a & b,
    }
    allowed_unops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def __init__(self, env: Dict[str, int]):
        self.env = env

    def visit_Expression(self, node: ast.Expression) -> int:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int:
        if isinstance(node.value, int):
            return int(node.value)
        raise ValueError("non-integer constant")

    def visit_Name(self, node: ast.Name) -> int:
        if node.id in self.env:
            return int(self.env[node.id])
        raise ValueError(f"unknown name {node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> int:
        op_t = type(node.op)
        if op_t not in self.allowed_binops:
            raise ValueError(f"unsupported op {op_t}")
        return int(self.allowed_binops[op_t](self.visit(node.left), self.visit(node.right)))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        op_t = type(node.op)
        if op_t not in self.allowed_unops:
            raise ValueError(f"unsupported unary op {op_t}")
        return int(self.allowed_unops[op_t](self.visit(node.operand)))

    def generic_visit(self, node):
        raise ValueError(f"unsupported expression node {type(node).__name__}")


def normalize_int_expr(expr: str) -> str:
    expr = expr.strip()
    # Remove common integer suffixes and simple casts.
    expr = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", expr)
    expr = re.sub(r"\(\s*(?:int|size_t|unsigned|long|short)\s*\)", "", expr)
    return expr


def safe_eval_int(expr: str, constants: Dict[str, int]) -> Optional[int]:
    expr = normalize_int_expr(expr)
    try:
        tree = ast.parse(expr, mode="eval")
        return int(SafeExprEval(constants).visit(tree))
    except Exception:
        return None


def collect_constants(paths: List[Path]) -> Dict[str, int]:
    env: Dict[str, int] = {}
    # Iterate a few times so dependent constants can resolve.
    candidates: List[Tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        text = _strip_c_comments_keep_lines(read_text(path))
        for line in text.splitlines():
            m = DEFINE_RE.match(line)
            if m and "(" not in m.group("name"):
                candidates.append((m.group("name"), m.group("expr")))
                continue
            m = CONST_ASSIGN_RE.match(line)
            if m:
                candidates.append((m.group("name"), m.group("expr")))
    changed = True
    while changed:
        changed = False
        for name, expr in candidates:
            if name in env:
                continue
            val = safe_eval_int(expr, env)
            if val is not None:
                env[name] = val
                changed = True
    return env


# ----------------------------- Function / AP parsing -----------------------------

def find_function_spans(source_text: str) -> List[FunctionSpan]:
    # A pragmatic brace-matching parser. Good enough for kernels; falls back to GLOBAL.
    clean = _strip_c_comments_keep_lines(source_text)
    spans: List[FunctionSpan] = []
    for m in FUNC_DEF_RE.finditer(clean):
        name = m.group("name")
        if name in CONTROL_WORDS:
            continue
        brace_pos = clean.find("{", m.end() - 1)
        if brace_pos < 0:
            continue
        depth = 0
        end_pos = None
        for idx in range(brace_pos, len(clean)):
            ch = clean[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = idx
                    break
        if end_pos is None:
            continue
        start_line = clean.count("\n", 0, m.start()) + 1
        end_line = clean.count("\n", 0, end_pos) + 1
        spans.append(FunctionSpan(name=name, start_line=start_line, end_line=end_line))
    spans.sort(key=lambda s: s.start_line)
    return spans or [FunctionSpan(name="GLOBAL", start_line=1, end_line=len(source_text.splitlines()))]


def function_for_line(spans: List[FunctionSpan], line_no: int) -> str:
    best = "GLOBAL"
    for sp in spans:
        if sp.start_line <= line_no <= sp.end_line:
            best = sp.name
    return best


def collect_following_pragmas(lines: List[str], start_idx: int) -> List[str]:
    pragmas: List[str] = []
    j = start_idx + 1
    while j < len(lines):
        s = lines[j].strip()
        if not s:
            j += 1
            continue
        if PRAGMA_RE.match(s):
            pragmas.append(s)
            j += 1
            continue
        break
    return pragmas


def extract_placeholders(pragmas: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for pr in pragmas:
        for m in PLACEHOLDER_RE.finditer(pr):
            key = m.group("key")
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def parse_loop_header(rest: str, constants: Dict[str, int]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    m = FOR_RE.search(rest)
    if not m:
        return None, None, None
    header = m.group("header")
    # Simple forms:
    #   int i = 0; i < N; ++i
    #   int i(0); i < N; ++i
    parts = [p.strip() for p in header.split(";")]
    induction = None
    bound_expr = None
    bound_val = None
    if len(parts) >= 2:
        init = parts[0]
        mi = re.search(r"\b(?:int|size_t|unsigned|long|short)?\s*([A-Za-z_]\w*)\s*(?:=|\()", init)
        if mi:
            induction = mi.group(1)
        cond = parts[1]
        # Prefer var < expr or var <= expr.
        if induction:
            mb = re.search(rf"\b{re.escape(induction)}\s*<\s*(.+)$", cond)
            if mb:
                bound_expr = mb.group(1).strip()
                bound_val = safe_eval_int(bound_expr, constants)
            else:
                mb = re.search(rf"\b{re.escape(induction)}\s*<=\s*(.+)$", cond)
                if mb:
                    bound_expr = f"({mb.group(1).strip()}) + 1"
                    bound_val = safe_eval_int(bound_expr, constants)
    return induction, bound_expr, bound_val


def parse_array_decl(rest: str, constants: Dict[str, int]) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[List[Optional[int]]]]:
    m = ARRAY_DECL_RE.match(rest)
    if not m:
        return None, None, None, None
    name = m.group("name")
    typ = " ".join(m.group("type").split())
    dims_expr = [d.strip() for d in DIM_RE.findall(m.group("dims"))]
    dims_val = [safe_eval_int(d, constants) for d in dims_expr]
    return name, typ, dims_expr, dims_val


def parse_action_points(source_path: Path, header_paths: List[Path], target: Dict[str, Any]) -> Dict[str, Any]:
    text = read_text(source_path)
    lines = text.splitlines()
    constants = collect_constants([source_path] + header_paths)
    spans = find_function_spans(text)

    aps: List[ActionPoint] = []
    for idx, line in enumerate(lines):
        m = LABEL_PREFIX_RE.match(line)
        if not m:
            continue
        label = m.group("label").upper()
        rest = m.group("rest").strip()
        named = m.group("named")
        line_no = idx + 1
        func = function_for_line(spans, line_no)
        pragmas = collect_following_pragmas(lines, idx)
        placeholders = extract_placeholders(pragmas)

        if FOR_RE.search(rest):
            induction, bound_expr, bound_val = parse_loop_header(rest, constants)
            aps.append(ActionPoint(
                label=label, kind="loop", function=func, line=line_no,
                named_label=named, source_text=rest, placeholders=placeholders,
                pragmas=pragmas, loop_bound_expr=bound_expr,
                loop_bound_value=bound_val, induction_var=induction,
            ))
            continue

        arr_name, arr_type, dims_expr, dims_val = parse_array_decl(rest, constants)
        if arr_name:
            aps.append(ActionPoint(
                label=label, kind="array", function=func, line=line_no,
                named_label=named, source_text=rest, placeholders=placeholders,
                pragmas=pragmas, array_name=arr_name, array_type=arr_type,
                array_dims_expr=dims_expr, array_dims_value=dims_val,
            ))
            continue

        aps.append(ActionPoint(
            label=label, kind="unknown", function=func, line=line_no,
            named_label=named, source_text=rest, placeholders=placeholders,
            pragmas=pragmas,
        ))

    aps.sort(key=lambda ap: int(ap.label[1:]))
    return {
        "source_file": str(source_path),
        "headers": [str(p) for p in header_paths],
        "functions": [asdict(s) for s in spans],
        "constants": constants,
        "target": target,
        "action_points": [asdict(ap) for ap in aps],
        "notes": [
            "This sidecar is the authoritative MailoHLS action-point schema.",
            "MLIR conversion is used for structure; placeholders/pragmas are attached back later using this schema.",
        ],
    }


# ----------------------------- Source cleanup -----------------------------

def make_mlir_source(source_path: Path, out_path: Path, strip_hls_pragmas: bool) -> None:
    lines = read_text(source_path).splitlines(True)
    out: List[str] = []
    for line in lines:
        if strip_hls_pragmas and PRAGMA_RE.match(line.strip()):
            # Preserve line count for debugging/source location stability.
            out.append("\n")
        else:
            out.append(line)
    write_text(out_path, "".join(out))


# ----------------------------- MLIR emission -----------------------------


def tool_has_option(tool: str, option: str) -> bool:
    proc = subprocess.run(
        [tool, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return option.lstrip("-") in proc.stdout


def build_cgeist_base_cmd(cgeist: str, src: Path, args: argparse.Namespace) -> List[str]:
    cmd = [cgeist, str(src), "-S"]
    for inc in args.include_dir:
        cmd.extend(["-I", str(Path(inc))])
    for define in args.define:
        cmd.append(f"-D{define}")
    # Unknown HLS pragmas should not be fatal if they survive cleanup.
    # cmd.append("-Wno-unknown-pragmas")
    # Allow user to pass frontend-specific flags after --extra-arg.
    for ea in args.extra_arg:
        cmd.append(ea)
    if args.top_func:
        # cgeist accepts -function=<name> per Polygeist docs.
        cmd.append(f"-function={args.top_func}")
    return cmd


def emit_mlir(args: argparse.Namespace, mlir_src: Path, out_dir: Path) -> Dict[str, Any]:
    cgeist = which_or_fail(args.cgeist, "cgeist")
    mlir_opt = shutil.which(args.mlir_opt) if args.mlir_opt else None

    logs = out_dir / "logs"
    raw_scf = out_dir / "raw_scf.mlir"
    affine = out_dir / "affine.mlir"
    opt_affine = out_dir / "affine_opt.mlir"

    base = build_cgeist_base_cmd(cgeist, mlir_src, args)

    # Raw SCF/MemRef MLIR.
    raw_cmd = base + ["-o", str(raw_scf)]
    raw_rc = run_cmd(raw_cmd, cwd=None, log_path=logs / "cgeist_raw_scf.log", verbose=args.verbose)

    # Affine attempt directly from cgeist.
    affine_rc = None
    if args.emit_affine:
        aff_cmd = base + ["-raise-scf-to-affine", "-o", str(affine)]
        affine_rc = run_cmd(aff_cmd, cwd=None, log_path=logs / "cgeist_affine.log", verbose=args.verbose)

    # Optional mlir-opt cleanup. This is deliberately best-effort across MLIR versions.
    opt_rc = None
    if args.run_mlir_opt:
        if not mlir_opt:
            print("[WARN] --run-mlir-opt requested but mlir-opt not found; skipping.", file=sys.stderr)
        else:
            input_mlir = affine if affine.exists() and affine_rc == 0 else raw_scf
            opt_cmd = [mlir_opt, str(input_mlir), "--canonicalize", "--cse"]
            if args.try_affine_raise_from_memref:
                if tool_has_option(mlir_opt, "affine-raise-from-memref"):
                    opt_cmd.append("-affine-raise-from-memref")
                else:
                    print("[WARN] mlir-opt does not support affine-raise-from-memref; skipping.", file=sys.stderr)
            opt_cmd += ["-o", str(opt_affine)]
            opt_rc = run_cmd(opt_cmd, cwd=None, log_path=logs / "mlir_opt.log", verbose=args.verbose)

    return {
        "cgeist": cgeist,
        "mlir_opt": mlir_opt,
        "raw_scf": str(raw_scf) if raw_scf.exists() else None,
        "affine": str(affine) if affine.exists() else None,
        "affine_opt": str(opt_affine) if opt_affine.exists() else None,
        "return_codes": {
            "raw_scf": raw_rc,
            "affine": affine_rc,
            "mlir_opt": opt_rc,
        },
    }


# ----------------------------- Main -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit MLIR + MailoHLS AP sidecar from C/C++ HLS kernel.")
    p.add_argument("--source", required=True, help="Input .c/.cc/.cpp/.cxx source file, labeled with Lk action points.")
    p.add_argument("--header", action="append", default=[], help="Header file to parse for constants. Can be repeated.")
    p.add_argument("--include-dir", action="append", default=[], help="Include directory for cgeist. Can be repeated.")
    p.add_argument("--define", action="append", default=[], help="Preprocessor definition, e.g. NUM_FEATURE=2. Can be repeated.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--top-func", default=None, help="Optional cgeist -function=<top>; omit to emit all possible functions.")

    p.add_argument("--cgeist", default="cgeist", help="Path/name of Polygeist cgeist executable.")
    p.add_argument("--mlir-opt", default="mlir-opt", help="Path/name of mlir-opt executable.")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed to cgeist. Can be repeated.")

    p.add_argument("--emit-affine", action="store_true", default=True, help="Try cgeist -raise-scf-to-affine. Default: true.")
    p.add_argument("--no-emit-affine", dest="emit_affine", action="store_false")
    p.add_argument("--run-mlir-opt", action="store_true", help="Run best-effort mlir-opt canonicalize/cse after cgeist.")
    p.add_argument("--try-affine-raise-from-memref", action="store_true", help="With --run-mlir-opt, try --affine-raise-from-memref.")

    p.add_argument("--strip-hls-pragmas", action="store_true", default=True, help="Strip #pragma HLS lines in MLIR source copy. Default: true.")
    p.add_argument("--keep-hls-pragmas", dest="strip_hls_pragmas", action="store_false")

    p.add_argument("--target-device", default=None, help="Optional FPGA part, saved in sidecar.")
    p.add_argument("--clock-ns", type=float, default=None, help="Optional target clock period in ns, saved in sidecar.")
    p.add_argument("--resource-caps-json", default=None, help="Optional JSON string or path with BRAM/DSP/FF/LUT/URAM caps.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_resource_caps(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    path = Path(s)
    if path.exists():
        return json.loads(read_text(path))
    return json.loads(s)


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(source)

    # If no headers are explicitly supplied, include headers next to the source.
    headers = [Path(h).resolve() for h in args.header]
    if not headers:
        headers = sorted(source.parent.glob("*.h")) + sorted(source.parent.glob("*.hpp"))

    target = {
        "target_device": args.target_device,
        "clock_period_ns": args.clock_ns,
        "resource_caps": load_resource_caps(args.resource_caps_json),
    }

    schema = parse_action_points(source, headers, target=target)
    dump_json(out_dir / "ap_schema.json", schema)

    mlir_src = out_dir / f"{source.stem}.mlir_input{source.suffix}"
    make_mlir_source(source, mlir_src, strip_hls_pragmas=args.strip_hls_pragmas)

    meta = {
        "input_source": str(source),
        "mlir_input_source": str(mlir_src),
        "out_dir": str(out_dir),
        "strip_hls_pragmas": args.strip_hls_pragmas,
        "headers": [str(h) for h in headers],
    }

    try:
        meta["mlir"] = emit_mlir(args, mlir_src, out_dir)
    except FileNotFoundError as e:
        meta["mlir_error"] = str(e)
        print(f"[ERROR] {e}", file=sys.stderr)
        print("[INFO] ap_schema.json and cleaned MLIR input source were still written.", file=sys.stderr)
        dump_json(out_dir / "run_metadata.json", meta)
        return 2

    dump_json(out_dir / "run_metadata.json", meta)

    print("[OK] Wrote:")
    print("  ", out_dir / "ap_schema.json")
    print("  ", mlir_src)
    for k, v in meta.get("mlir", {}).items():
        if isinstance(v, str) and v:
            print("  ", v)
    print("[NEXT] Inspect raw_scf.mlir / affine.mlir and verify every Lk in ap_schema.json can be mapped to a loop or array/memref anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
