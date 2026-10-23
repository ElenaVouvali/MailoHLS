#!/usr/bin/env python3
"""
Single-kernel pipeline for building MailoHLS/HARP memory packs from an
application folder.

Extended preprocessing:
  0) inspect the source + headers
  1) infer / validate Lk labels on loop and local-array declarations
  2) generate kernel_info.txt from the labeled source and resolved constants
  3) stage the application under <project_root>/harp/<kernel_name>
  4) insert pragma placeholders into the staged labeled source
  5) build the deterministic extended-pseudo-block-connected-hierarchy GEXF
  6) convert the final GEXF to a pragma-free PT
  7) extract slot-aligned node embeddings and save <kernel_name>.memory.pt

Expected application folder contents:
  - exactly one source file (.c/.cc/.cpp/.cxx), or pass --source-file
  - any required header files (.h/.hh/.hpp/.hxx)

Important:
  - For strict determinism, launch with PYTHONHASHSEED=0 before Python starts.
  - This script is intended to live next to the existing project scripts
    (graph_gen_deterministic.py, insert_placeholders.py, gexf_to_pt_zero.py,
    pt_to_gnn_emb.py, model.py, config.py).

Notes on the new preprocessing step:
  - If all loop / local-array declarations already have contiguous labels
    L1..LN in source order, the labels are preserved.
  - Otherwise, all relevant declarations are relabeled contiguously in source
    order so the downstream memory format stays compatible with the existing
    X_llm_labelid / memory-bank assumptions.
  - The original source in --app-dir is NOT overwritten. The generated labeled
    source and generated kernel_info.txt are written into the staged directory
    and copied to the output bundle.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter

PLACEHOLDER_KEY_RE = re.compile(r"auto\{(_(?:PIPE|UNROLL|ARRAY_[TFD])_L\d+)\}")

SRC_EXTS = {".c", ".cc", ".cpp", ".cxx"}
HDR_EXTS = {".h", ".hh", ".hpp", ".hxx"}
CONTROL_KEYWORDS = {"if", "for", "while", "switch", "else", "do", "case"}
PREFERRED_TOP_FUNC_NAMES = ("workload", "kernel", "top", "top_func")

LABEL_PREFIX_RE = re.compile(
    r"^(?P<indent>\s*)(?:/\*\s*(?P<label1>L\d+)\s*:\s*\*/|(?P<label2>L\d+)\s*:)",
    re.IGNORECASE,
)

FUNCTION_DEF_RE = re.compile(
    r"\b\w[\w\s\*:&<>]*\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
    re.DOTALL,
)

FOR_HEAD_START_RE = re.compile(r"\bfor\s*\(", re.IGNORECASE)

CONST_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)(?!\()\s*(.+?)\s*$",
    re.MULTILINE,
)

CONST_ASSIGN_RE = re.compile(
    r"""
    (?m)
    ^\s*
    (?:static\s+)?
    (?:constexpr\s+)?
    (?:const\s+)?
    (?:unsigned\s+|signed\s+|long\s+|short\s+)*
    (?:int|size_t|ptrdiff_t|ssize_t|char|bool|ap_uint<[^>]+>|ap_int<[^>]+>|uint\d+_t|int\d+_t)
    \s+
    ([A-Za-z_]\w*)
    \s*=\s*
    ([^;]+)
    ;
    """,
    re.VERBOSE,
)

L_SUFFIX_RE = re.compile(r"\b(\d+)(?:[uUlL]+)\b")
NAMED_LABEL_PREFIX_RE = re.compile(r"^\s*(?:(?:[A-Za-z_]\w*)\s*:\s*)+")

SIZEOF_RE = re.compile(r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)")

SIMPLE_SIZEOF_BYTES = {
    "char": 1,
    "bool": 1,
    "short": 2,
    "int": 4,
    "long": 8,
    "float": 4,
    "double": 8,
}


def _replace_simple_sizeof(expr: str) -> str:
    def repl(m):
        ty = m.group(1)
        if ty in SIMPLE_SIZEOF_BYTES:
            return str(SIMPLE_SIZEOF_BYTES[ty])
        raise ValueError(f"Unsupported sizeof type in constant expression: {ty}")
    return SIZEOF_RE.sub(repl, expr)


def _source_already_has_placeholders(text: str) -> bool:
    return bool(PLACEHOLDER_KEY_RE.search(text))

def _assert_no_duplicate_placeholder_keys(text: str, what: str) -> None:
    counts = Counter(PLACEHOLDER_KEY_RE.findall(text))
    dups = {k: v for k, v in counts.items() if v > 1}
    if dups:
        sample = ", ".join(f"{k}x{v}" for k, v in sorted(dups.items())[:10])
        raise RuntimeError(f"{what} contains duplicate placeholder keys: {sample}")


@dataclass
class FunctionSpan:
    name: str
    start_line: int
    end_line: int
    start_pos: int
    end_pos: int


@dataclass
class TargetDecl:
    kind: str                    # "loop" or "array"
    line_idx: int                # 0-based line index in source
    end_line_idx: int            # 0-based end line index for parsed statement/header
    function_name: str
    existing_label: Optional[str]
    final_label: Optional[str]
    loop_bound: Optional[int] = None
    array_name: Optional[str] = None
    array_dims: Optional[List[int]] = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _print(msg: str) -> None:
    print(f"[code_to_memory_pt] {msg}")


@contextlib.contextmanager
def _sanitized_argv() -> Iterable[None]:
    """
    Several project modules parse argparse flags at import time (notably config.py).
    We temporarily clear argv to make those imports safe.
    """
    saved = list(sys.argv)
    try:
        sys.argv = [saved[0] if saved else "python"]
        yield
    finally:
        sys.argv = saved


@contextlib.contextmanager
def _pushd(path: Path) -> Iterable[None]:
    saved = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_source_file(app_dir: Path, user_source: str) -> Path:
    if user_source:
        p = Path(user_source)
        if not p.is_absolute():
            p = app_dir / p
        _require_file(p, "source file")
        if p.suffix.lower() not in SRC_EXTS:
            raise ValueError(f"Unsupported source extension for {p}. Expected one of {sorted(SRC_EXTS)}")
        return p

    candidates = [
        p for p in sorted(app_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in SRC_EXTS
        and not p.name.endswith("_placeholders" + p.suffix)
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No source file found in {app_dir}. Expected one of: {sorted(SRC_EXTS)}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple source files found. Pass --source-file explicitly. Found: "
            + ", ".join(p.name for p in candidates)
        )
    return candidates[0]


def _find_header_files(app_dir: Path) -> Tuple[Path, ...]:
    return tuple(
        p for p in sorted(app_dir.iterdir())
        if p.is_file() and p.suffix.lower() in HDR_EXTS
    )


def _kernel_name_from_args(app_dir: Path, source_path: Path, kernel_name: str) -> str:
    if kernel_name:
        return kernel_name
    return app_dir.name or source_path.stem


def _purge_kernel_products(project_root: Path, kernel_name: str) -> None:
    """
    Remove only the per-kernel staging directory.
    Do NOT touch <project_root>/harp/processed/*.
    """
    stage_dir = project_root / "harp" / kernel_name
    if stage_dir.is_dir():
        shutil.rmtree(stage_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Import project modules safely
# -----------------------------------------------------------------------------

def _import_project_modules(script_dir: Path, auto_set_pythonhashseed: bool):
    """
    Import the project modules only after argv has been sanitized.

    graph_gen_deterministic.py checks PYTHONHASHSEED during import. If the user
    did not start Python with it, we can only set the env var late and proceed
    functionally; strict determinism still requires launching the interpreter
    with PYTHONHASHSEED already set.
    """
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    if not os.environ.get("PYTHONHASHSEED"):
        if auto_set_pythonhashseed:
            os.environ["PYTHONHASHSEED"] = "0"
            _print(
                "PYTHONHASHSEED was not set at interpreter startup. "
                "Setting it to 0 now so graph_gen_deterministic can import, "
                "but strict determinism still requires launching with: "
                "PYTHONHASHSEED=0 python code_to_memory_pt.py ..."
            )
        else:
            raise RuntimeError(
                "PYTHONHASHSEED is not set. Run like:\n"
                "  PYTHONHASHSEED=0 python code_to_memory_pt.py ..."
            )

    with _sanitized_argv(), _pushd(script_dir):
        insert_placeholders_mod = importlib.import_module("insert_placeholders")
        graph_gen_mod = importlib.import_module("graph_gen_deterministic")
        gexf_to_pt_mod = importlib.import_module("gexf_to_pt_zero")
        pt_to_gnn_emb_mod = importlib.import_module("pt_to_gnn_emb")
        config_mod = importlib.import_module("config")
        model_mod = importlib.import_module("model")
        torch_geometric_data = importlib.import_module("torch_geometric.data")

    return {
        "insert_placeholders_mod": insert_placeholders_mod,
        "graph_gen_mod": graph_gen_mod,
        "gexf_to_pt_mod": gexf_to_pt_mod,
        "pt_to_gnn_emb_mod": pt_to_gnn_emb_mod,
        "config_mod": config_mod,
        "model_mod": model_mod,
        "Batch": torch_geometric_data.Batch,
    }


# -----------------------------------------------------------------------------
# Source preprocessing: labels + kernel_info.txt generation
# -----------------------------------------------------------------------------

def _strip_comments_keep_layout(text: str) -> str:
    """
    Remove comments while preserving character positions and line count as much as
    possible. This is used only for parsing, not for rewriting.
    """
    out: List[str] = []
    i = 0
    n = len(text)
    in_sl = False
    in_ml = False
    in_sq = False
    in_dq = False
    escape = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_sl:
            if ch == "\n":
                in_sl = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue

        if in_ml:
            if ch == "*" and nxt == "/":
                out.extend([" ", " "])
                in_ml = False
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue

        if in_sq:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_sq = False
            i += 1
            continue

        if in_dq:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_dq = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_sl = True
            out.extend([" ", " "])
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_ml = True
            out.extend([" ", " "])
            i += 2
            continue

        if ch == "'":
            in_sq = True
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _remove_line_comments(expr: str) -> str:
    return re.sub(r"//.*", "", expr).strip()


def _clean_int_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _remove_line_comments(expr)
    expr = expr.rstrip(",")
    expr = _replace_simple_sizeof(expr)
    expr = L_SUFFIX_RE.sub(r"\1", expr)
    return expr


class _SafeIntEvaluator(ast.NodeVisitor):
    def __init__(self, symbols: Dict[str, int]):
        self.symbols = symbols

    def visit_Expression(self, node: ast.Expression) -> int:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int:
        if isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node.value, int):
            return int(node.value)
        raise ValueError(f"Unsupported constant in integer expression: {node.value!r}")

    def visit_Name(self, node: ast.Name) -> int:
        if node.id not in self.symbols:
            raise NameError(node.id)
        return int(self.symbols[node.id])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        v = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")

    def visit_BinOp(self, node: ast.BinOp) -> int:
        l = self.visit(node.left)
        r = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return l + r
        if isinstance(op, ast.Sub):
            return l - r
        if isinstance(op, ast.Mult):
            return l * r
        if isinstance(op, ast.FloorDiv):
            return l // r
        if isinstance(op, ast.Div):
            return l // r
        if isinstance(op, ast.Mod):
            return l % r
        if isinstance(op, ast.Pow):
            return l ** r
        if isinstance(op, ast.LShift):
            return l << r
        if isinstance(op, ast.RShift):
            return l >> r
        if isinstance(op, ast.BitOr):
            return l | r
        if isinstance(op, ast.BitAnd):
            return l & r
        if isinstance(op, ast.BitXor):
            return l ^ r
        raise ValueError(f"Unsupported binary operator: {ast.dump(op)}")

    def visit_Paren(self, node):  # pragma: no cover
        return self.visit(node.value)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def _eval_int_expr(expr: str, symbols: Dict[str, int]) -> int:
    expr = _clean_int_expr(expr)
    if not expr:
        raise ValueError("Empty integer expression")
    tree = ast.parse(expr, mode="eval")
    return int(_SafeIntEvaluator(symbols).visit(tree))


def _collect_constant_exprs(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    stripped = _strip_comments_keep_layout(text)

    for name, expr in CONST_DEFINE_RE.findall(stripped):
        out[name] = _clean_int_expr(expr)

    for name, expr in CONST_ASSIGN_RE.findall(stripped):
        out[name] = _clean_int_expr(expr)

    return out


def _resolve_symbol_table(raw_exprs: Dict[str, str]) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    pending = dict(raw_exprs)

    while pending:
        progress = False
        for name in list(pending.keys()):
            expr = pending[name]
            try:
                resolved[name] = _eval_int_expr(expr, resolved)
            except NameError:
                # dependency not resolved yet
                continue
            except Exception:
                # keep it pending; maybe another pass or future normalization helps
                continue
            else:
                pending.pop(name, None)
                progress = True

        if not progress:
            break

    return resolved


def _build_constant_table(header_files: Sequence[Path], source_text: str) -> Dict[str, int]:
    raw: Dict[str, str] = {}
    for hdr in header_files:
        try:
            raw.update(_collect_constant_exprs(hdr.read_text(encoding="utf-8", errors="ignore")))
        except FileNotFoundError:
            continue
    raw.update(_collect_constant_exprs(source_text))
    return _resolve_symbol_table(raw)


def _find_function_spans(source_text: str) -> List[FunctionSpan]:
    stripped = _strip_comments_keep_layout(source_text)
    spans: List[FunctionSpan] = []

    for m in FUNCTION_DEF_RE.finditer(stripped):
        name = m.group(1)
        if name in CONTROL_KEYWORDS:
            continue

        brace_pos = stripped.find("{", m.start())
        if brace_pos < 0:
            continue

        depth = 0
        end_pos = None
        for i in range(brace_pos, len(stripped)):
            ch = stripped[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos is None:
            continue

        start_line = stripped.count("\n", 0, brace_pos) + 1
        end_line = stripped.count("\n", 0, end_pos) + 1
        spans.append(
            FunctionSpan(
                name=name,
                start_line=start_line,
                end_line=end_line,
                start_pos=brace_pos,
                end_pos=end_pos,
            )
        )

    return spans


def _validate_top_func_override(top_func_override: str, spans: Sequence[FunctionSpan]) -> None:
    if not top_func_override:
        return

    names = {sp.name for sp in spans}
    if top_func_override not in names:
        raise RuntimeError(
            f"--top-func '{top_func_override}' was not found in source functions: "
            f"{sorted(names)}"
        )


def _infer_top_function(source_text: str, spans: Sequence[FunctionSpan], top_func_override: str) -> Optional[str]:
    if top_func_override:
        return top_func_override
    if not spans:
        return None

    # 1) prefer the unique function containing HLS INTERFACE pragmas
    hls_interface_hits = []
    for sp in spans:
        body = source_text[sp.start_pos:sp.end_pos + 1]
        if re.search(r"#\s*pragma\s+HLS\s+INTERFACE\b", body):
            hls_interface_hits.append(sp.name)
    if len(hls_interface_hits) == 1:
        return hls_interface_hits[0]

    # 2) prefer common top-level names
    lower_to_name = {sp.name.lower(): sp.name for sp in spans}
    for pref in PREFERRED_TOP_FUNC_NAMES:
        if pref in lower_to_name:
            return lower_to_name[pref]

    # 3) if there is a single function, use it
    if len(spans) == 1:
        return spans[0].name

    # 4) fallback: last function in the file (common in HLS app datasets)
    return spans[-1].name


def _line_to_function_map(spans: Sequence[FunctionSpan]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for sp in spans:
        for ln in range(sp.start_line, sp.end_line + 1):
            out[ln] = sp.name
    return out


def _get_existing_label(line: str) -> Optional[str]:
    m = LABEL_PREFIX_RE.match(line)
    if not m:
        return None
    return (m.group("label1") or m.group("label2") or "").upper()


def _strip_leading_source_label(line: str) -> str:
    m = LABEL_PREFIX_RE.match(line)
    if not m:
        return line
    return line[m.end():]


def _replace_or_insert_source_label(line: str, new_label: str) -> str:
    """
    Always emit compiler-safe comment labels:
      /*L1:*/ ...
    Never raw C/C++ labels like:
      L1: ...
    """
    nl = "\n" if line.endswith("\n") else ""
    body = line[:-1] if nl else line
    comment_label = f"/*{new_label}:*/"

    m = LABEL_PREFIX_RE.match(body)
    if m:
        indent = m.group("indent") or ""
        rest = body[m.end():].lstrip()
        return f"{indent}{comment_label}\t{rest}{nl}"

    indent_len = len(body) - len(body.lstrip(" \t"))
    indent = body[:indent_len]
    rest = body[indent_len:]
    return f"{indent}{comment_label}\t{rest}{nl}"


def _collect_for_header_text(lines: Sequence[str], start_idx: int) -> Tuple[str, int]:
    """
    Collect a possibly multi-line for-header starting at start_idx until the
    matching ')' of the first for(...).
    """
    text = ""
    found = False
    paren_depth = 0
    end_idx = start_idx

    for j in range(start_idx, len(lines)):
        frag = _strip_comments_keep_layout(_strip_leading_source_label(lines[j]))
        text += frag
        if not found:
            m = FOR_HEAD_START_RE.search(text)
            if not m:
                if j > start_idx:
                    break
                continue
            found = True
            k = m.end() - 1
            paren_depth = 1
            while k + 1 < len(text):
                k += 1
                ch = text[k]
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        return text, j
        else:
            for ch in frag:
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        return text, j
        end_idx = j

    return text, end_idx


def _collect_statement_until_semicolon(lines: Sequence[str], start_idx: int) -> Tuple[str, int]:
    """
    Collect a declaration statement possibly spanning multiple lines.
    Stops at the first semicolon outside (), {}, and string/comment noise.
    """
    text = ""
    depth_paren = 0
    depth_brace = 0
    end_idx = start_idx

    for j in range(start_idx, len(lines)):
        frag = _strip_comments_keep_layout(_strip_leading_source_label(lines[j]))
        text += frag
        end_idx = j
        for ch in frag:
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
            elif ch == ";" and depth_paren == 0 and depth_brace == 0:
                return text, end_idx
    return text, end_idx


def _split_top_level_semicolons(text: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _extract_for_paren_body(header_text: str) -> str:
    m = FOR_HEAD_START_RE.search(header_text)
    if not m:
        raise ValueError("No for(...) found in header text")
    i = m.end() - 1  # at '('
    depth = 1
    body_chars: List[str] = []
    while i + 1 < len(header_text):
        i += 1
        ch = header_text[i]
        if ch == "(":
            depth += 1
            body_chars.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
            body_chars.append(ch)
        else:
            body_chars.append(ch)
    if depth != 0:
        raise ValueError(f"Unbalanced for-header parentheses: {header_text!r}")
    return "".join(body_chars)


def _parse_loop_init(init_text: str, symbols: Dict[str, int]) -> Tuple[str, int]:
    s = init_text.strip()
    s = re.sub(r"\b(?:const|volatile|static|register|unsigned|signed|long|short|int|size_t|ptrdiff_t|ssize_t|char|bool)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # int i = 0, i = 0
    m = re.search(r"\b([A-Za-z_]\w*)\s*=\s*(.+)$", s)
    if m:
        return m.group(1), _eval_int_expr(m.group(2), symbols)

    # int i(0)
    m = re.search(r"\b([A-Za-z_]\w*)\s*\(\s*(.+)\s*\)\s*$", s)
    if m:
        return m.group(1), _eval_int_expr(m.group(2), symbols)

    raise ValueError(f"Unsupported for-init form: {init_text!r}")


def _parse_loop_step(step_text: str, var: str, symbols: Dict[str, int]) -> int:
    s = step_text.strip()
    if re.fullmatch(rf"\+\+\s*{re.escape(var)}|{re.escape(var)}\s*\+\+", s):
        return 1
    if re.fullmatch(rf"--\s*{re.escape(var)}|{re.escape(var)}\s*--", s):
        return -1

    m = re.fullmatch(rf"{re.escape(var)}\s*\+=\s*(.+)", s)
    if m:
        return _eval_int_expr(m.group(1), symbols)

    m = re.fullmatch(rf"{re.escape(var)}\s*-=\s*(.+)", s)
    if m:
        return -_eval_int_expr(m.group(1), symbols)

    m = re.fullmatch(rf"{re.escape(var)}\s*=\s*{re.escape(var)}\s*\+\s*(.+)", s)
    if m:
        return _eval_int_expr(m.group(1), symbols)

    m = re.fullmatch(rf"{re.escape(var)}\s*=\s*{re.escape(var)}\s*-\s*(.+)", s)
    if m:
        return -_eval_int_expr(m.group(1), symbols)

    raise ValueError(f"Unsupported for-step form: {step_text!r}")


def _compute_tripcount(start: int, op: str, end: int, step: int) -> int:
    if step == 0:
        raise ValueError("Loop step evaluated to 0")

    if step > 0:
        if op == "<":
            span = end - start
            return 0 if span <= 0 else (span + step - 1) // step
        if op == "<=":
            span = end - start
            return 0 if span < 0 else span // step + 1
    else:
        step_abs = -step
        if op == ">":
            span = start - end
            return 0 if span <= 0 else (span + step_abs - 1) // step_abs
        if op == ">=":
            span = start - end
            return 0 if span < 0 else span // step_abs + 1

    raise ValueError(f"Incompatible loop condition / step: start={start}, op={op}, end={end}, step={step}")


def _parse_loop_bound_from_header(header_text: str, symbols: Dict[str, int]) -> int:
    body = _extract_for_paren_body(header_text)
    parts = _split_top_level_semicolons(body)
    if len(parts) < 3:
        raise ValueError(f"Unexpected for-header shape: {header_text!r}")

    init_text, cond_text, step_text = parts[0].strip(), parts[1].strip(), parts[2].strip()
    var, start_val = _parse_loop_init(init_text, symbols)
    step_val = _parse_loop_step(step_text, var, symbols)

    cond = cond_text.strip()
    cond_patterns = [
        (rf"^{re.escape(var)}\s*(<=|<|>=|>)\s*(.+)$", False),
        (rf"^(.+)\s*(<=|<|>=|>)\s*{re.escape(var)}$", True),
    ]

    for pat, reversed_side in cond_patterns:
        m = re.match(pat, cond)
        if not m:
            continue
        if reversed_side:
            end_expr, op = m.group(1), m.group(2)
            # EXPR > i  == i < EXPR ; EXPR >= i == i <= EXPR ; etc.
            op_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
            op = op_map[op]
        else:
            op, end_expr = m.group(1), m.group(2)

        end_val = _eval_int_expr(end_expr, symbols)
        return _compute_tripcount(start_val, op, end_val, step_val)

    raise ValueError(f"Unsupported loop condition form: {cond_text!r}")


def _parse_array_dims_from_statement(stmt_text: str, symbols: Dict[str, int]) -> Tuple[str, List[int]]:
    stmt = _strip_comments_keep_layout(_strip_leading_source_label(stmt_text)).strip()
    stmt = NAMED_LABEL_PREFIX_RE.sub("", stmt)

    if not stmt.endswith(";"):
        raise ValueError("Array declaration statement does not end with ';'")

    if "(" in stmt and stmt.find("(") < stmt.find("["):
        raise ValueError("Likely not a local array declaration")

    first_lb = stmt.find("[")
    if first_lb < 0:
        raise ValueError("No '[' found in array declaration")

    prefix = stmt[:first_lb]
    m_name = re.search(r"([A-Za-z_]\w*)\s*$", prefix)
    if not m_name:
        raise ValueError(f"Could not infer array variable name from: {stmt!r}")

    varname = m_name.group(1)
    dim_exprs = re.findall(r"\[([^\]]+)\]", stmt)
    dims = [_eval_int_expr(expr, symbols) for expr in dim_exprs]

    return varname, dims


def _split_top_level_commas(s: str) -> List[str]:
    parts = []
    start = 0
    paren = bracket = brace = angle = 0
    in_sq = in_dq = False
    escape = False

    for i, ch in enumerate(s):
        if in_sq:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_sq = False
            continue

        if in_dq:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_dq = False
            continue

        if ch == "'":
            in_sq = True
            continue
        if ch == '"':
            in_dq = True
            continue

        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "," and paren == bracket == brace == angle == 0:
            parts.append(s[start:i].strip())
            start = i + 1

    parts.append(s[start:].strip())
    return parts


def _try_split_multi_array_decl_line(line: str) -> Optional[List[str]]:
    """
    Split simple C/C++ multi-array declarations in the staged source.

    Example:
        float a[N], b[M];
    becomes:
        float a[N];
        float b[M];

    Returns None if the line is not a supported multi-array declaration.
    """
    nl = "\n" if line.endswith("\n") else ""
    raw = line[:-1] if nl else line

    indent = raw[: len(raw) - len(raw.lstrip(" \t"))]
    body = raw[len(indent):]

    # Preserve an existing /*Lk:*/ label only on the first split declaration.
    label_prefix = ""
    m_label = LABEL_PREFIX_RE.match(body)
    if m_label:
        label_prefix = body[:m_label.end()]
        body = body[m_label.end():].lstrip()

    stripped = _strip_comments_keep_layout(body).strip()
    if not stripped.endswith(";"):
        return None
    if "[" not in stripped or "," not in stripped:
        return None
    if "(" in stripped and stripped.find("(") < stripped.find("["):
        return None

    stmt = stripped[:-1].strip()
    parts = _split_top_level_commas(stmt)

    array_parts = [p for p in parts if re.search(r"\b[A-Za-z_]\w*\s*(?:\[[^\]]+\])+", p)]
    if len(array_parts) <= 1 or len(array_parts) != len(parts):
        return None

    first = parts[0]
    first_lb = first.find("[")
    if first_lb < 0:
        return None

    m_name = re.search(r"([A-Za-z_]\w*)\s*$", first[:first_lb])
    if not m_name:
        return None

    # Type prefix is everything before the first declarator name.
    type_prefix = first[:m_name.start(1)]
    first_decl = first[m_name.start(1):].strip()

    # Avoid unsafe pointer/reference splitting such as: float *a[N], *b[M]
    if type_prefix.rstrip().endswith(("*", "&")):
        _print(f"Warning: not auto-splitting pointer/reference multi-array declaration: {raw.strip()}")
        return None

    out = []
    for i, part in enumerate(parts):
        decl = first_decl if i == 0 else part.strip()

        if i == 0 and label_prefix:
            out.append(f"{indent}{label_prefix} {type_prefix}{decl};{nl}")
        else:
            out.append(f"{indent}{type_prefix}{decl};{nl}")

    return out


def _normalize_multi_array_declarations(source_text: str) -> str:
    """
    Rewrite staged source so that each local array declaration becomes its own
    source action point candidate.
    """
    lines = source_text.splitlines(keepends=True)
    out = []

    for line in lines:
        split_lines = _try_split_multi_array_decl_line(line)
        if split_lines is None:
            out.append(line)
        else:
            _print(
                "Split multi-array declaration into separate declarations: "
                + " | ".join(x.strip() for x in split_lines)
            )
            out.extend(split_lines)

    return "".join(out)


def _scan_targets_in_source(source_text: str, symbols: Dict[str, int]) -> Tuple[List[TargetDecl], List[FunctionSpan]]:
    lines = source_text.splitlines(keepends=True)
    spans = _find_function_spans(source_text)
    line_to_func = _line_to_function_map(spans)

    decls: List[TargetDecl] = []
    i = 0
    while i < len(lines):
        line_no = i + 1
        function_name = line_to_func.get(line_no)
        if function_name is None:
            i += 1
            continue

        raw_line = lines[i]
        existing_label = _get_existing_label(raw_line)
        analysis_line = _strip_comments_keep_layout(_strip_leading_source_label(raw_line))
        analysis_stripped = analysis_line.strip()

        # Skip preprocessor / pragma / braces / empty lines
        if (
            not analysis_stripped
            or analysis_stripped.startswith("#")
            or "#pragma" in analysis_stripped
            or analysis_stripped in {"{", "}", "};"}
        ):
            i += 1
            continue

        # 1) loops
        if FOR_HEAD_START_RE.search(analysis_line):
            header_text, end_idx = _collect_for_header_text(lines, i)
            try:
                tripcount = _parse_loop_bound_from_header(header_text, symbols)
                decls.append(
                    TargetDecl(
                        kind="loop",
                        line_idx=i,
                        end_line_idx=end_idx,
                        function_name=function_name,
                        existing_label=existing_label,
                        final_label=None,
                        loop_bound=tripcount,
                    )
                )
                i = end_idx + 1
                continue
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resolve loop bound at line {line_no} in function '{function_name}'.\n"
                    f"Header: {header_text.strip()}\n"
                    f"Reason: {e}"
                ) from e

        # 2) local arrays
        if _looks_like_local_array_decl_start(raw_line):
            try:
                stmt_text, end_idx = _collect_statement_until_semicolon(lines, i)
                stmt_probe = _strip_comments_keep_layout(_strip_leading_source_label(stmt_text)).strip()
                stmt_probe = NAMED_LABEL_PREFIX_RE.sub("", stmt_probe)

                if stmt_probe.startswith("#") or "#pragma" in stmt_probe:
                    i += 1
                    continue

                varname, dims = _parse_array_dims_from_statement(stmt_text, symbols)
                decls.append(
                    TargetDecl(
                        kind="array",
                        line_idx=i,
                        end_line_idx=end_idx,
                        function_name=function_name,
                        existing_label=existing_label,
                        final_label=None,
                        array_name=varname,
                        array_dims=dims,
                    )
                )
                i = end_idx + 1
                continue
            except Exception:
                pass

        i += 1

    return decls, spans


def _looks_like_local_array_decl_start(line: str) -> bool:
    s = _strip_comments_keep_layout(_strip_leading_source_label(line)).strip()

    if not s:
        return False
    if s.startswith("#"):
        return False
    if "#pragma" in s:
        return False
    if s in {"{", "}", "};"}:
        return False
    if re.match(r"^(?:return|if|for|while|switch|else|do)\b", s):
        return False
    
    # Reject obvious assignments to existing arrays, but allow declarations with initializers.
    if "=" in s:
        left = s.split("=", 1)[0]
        if "[" not in left:
            return False
    if "(" in s and s.find("(") < s.find("[") if "[" in s else False:
        return False

    return bool(re.match(
        r"^(?:static\s+|const\s+|constexpr\s+|volatile\s+)*"
        r"(?:unsigned\s+|signed\s+|long\s+|short\s+)*"
        r"[A-Za-z_]\w*(?:\s*<[^;{}()]+>)?(?:\s+[A-Za-z_]\w*(?:\s*<[^;{}()]+>)?)*"
        r"(?:\s*[*&]\s*)*"
        r"\s+[A-Za-z_]\w*\s*(?:\[[^\]]+\])+",
        s
    ))


def _labels_are_complete_and_contiguous(decls: Sequence[TargetDecl]) -> bool:
    labels = [d.existing_label for d in decls]
    if any(lbl is None for lbl in labels):
        return False
    expected = [f"L{i}" for i in range(1, len(decls) + 1)]
    return [str(x).upper() for x in labels] == expected


def _assign_final_labels(decls: List[TargetDecl]) -> None:
    if _labels_are_complete_and_contiguous(decls):
        for d in decls:
            d.final_label = d.existing_label
        return

    for idx, d in enumerate(decls, start=1):
        d.final_label = f"L{idx}"


def _rewrite_source_with_labels(source_text: str, decls: Sequence[TargetDecl]) -> str:
    lines = source_text.splitlines(keepends=True)
    line_to_decl: Dict[int, TargetDecl] = {d.line_idx: d for d in decls}

    for idx, decl in line_to_decl.items():
        assert decl.final_label is not None
        lines[idx] = _replace_or_insert_source_label(lines[idx], decl.final_label)

    return "".join(lines)


def _kernel_info_text(top_func: Optional[str], decls: Sequence[TargetDecl]) -> str:
    if not top_func:
        top_func = "top"
    out = [top_func]
    for d in decls:
        assert d.final_label is not None
        if d.kind == "loop":
            out.append(f"{d.final_label},loop,{int(d.loop_bound)}")
        elif d.kind == "array":
            dims = d.array_dims or []
            tail = []
            for idx, dim in enumerate(dims, start=1):
                tail.extend([str(idx), str(int(dim))])
            out.append(
                ",".join([d.final_label, "array", str(d.array_name)] + tail)
            )
        else:
            raise ValueError(f"Unknown declaration kind: {d.kind}")
    return "\n".join(out) + "\n"


def _prepare_labeled_source_and_kernel_info(
    source_path: Path,
    header_files: Sequence[Path],
    top_func_override: str,
) -> Dict[str, object]:
    source_text = source_path.read_text(encoding="utf-8", errors="ignore")

    # Normalize before building symbols/scanning so labels and line indices match final staged source.
    source_text = _normalize_multi_array_declarations(source_text)

    symbols = _build_constant_table(header_files, source_text)

    decls, spans = _scan_targets_in_source(source_text, symbols)

    _validate_top_func_override(top_func_override, spans)

    if not decls:
        raise RuntimeError(
            "No loop / local-array declarations were identified in the source file. "
            "This pipeline expects HLS-relevant arrays and for-loops in the source."
        )

    top_func = _infer_top_function(source_text, spans, top_func_override=top_func_override)

    _assign_final_labels(decls)
    labeled_source_text = _rewrite_source_with_labels(source_text, decls)
    kernel_info_text = _kernel_info_text(top_func, decls)

    return {
        "top_function": top_func,
        "symbols": symbols,
        "decls": decls,
        "labeled_source_text": labeled_source_text,
        "kernel_info_text": kernel_info_text,
    }


# -----------------------------------------------------------------------------
# Staging + graph / PT / memory pipeline
# -----------------------------------------------------------------------------

def _stage_application(
    app_dir: Path,
    stage_dir: Path,
    kernel_name: str,
    source_path: Path,
    header_files: Tuple[Path, ...],
    labeled_source_text: str,
    kernel_info_text: str,
    insert_placeholders_fn,
) -> Dict[str, Path]:
    _ensure_dir(stage_dir)

    staged_src = stage_dir / f"{kernel_name}{source_path.suffix.lower()}"
    staged_kernel_info = stage_dir / "kernel_info.txt"
    placeholder_src = stage_dir / f"{kernel_name}_placeholders{source_path.suffix.lower()}"

    # Start from labeled source
    staged_src.write_text(labeled_source_text, encoding="utf-8")
    staged_kernel_info.write_text(kernel_info_text, encoding="utf-8")

    for hdr in header_files:
        shutil.copy2(hdr, stage_dir / hdr.name)

    staged_text = staged_src.read_text(encoding="utf-8", errors="ignore")

    if _source_already_has_placeholders(staged_text):
        _print(
            "Detected existing pragma placeholders in staged source; "
            "skipping insert_placeholders.py."
        )
        final_placeholder_text = staged_text
    else:
        placeholder_lines = insert_placeholders_fn(str(staged_src))
        final_placeholder_text = "".join(placeholder_lines)

    _assert_no_duplicate_placeholder_keys(final_placeholder_text, "placeholderized source")

    # Critical:
    # graph_gen_deterministic reads <kernel_name>.cpp, so staged_src itself must
    # be the final placeholderized source.
    staged_src.write_text(final_placeholder_text, encoding="utf-8")
    placeholder_src.write_text(final_placeholder_text, encoding="utf-8")

    if "auto{" not in final_placeholder_text:
        _print(
            "Warning: the placeholder source does not contain any auto{...} placeholders. "
            "This usually means the labeled source still did not expose loop / array action points in the form expected by insert_placeholders.py."
        )

    return {
        "staged_src": staged_src,
        "staged_kernel_info": staged_kernel_info,
        "placeholder_src": placeholder_src,
    }


def _build_hierarchy_graph(
    graph_gen_mod,
    kernel_name: str,
    stage_dir: Path,
    src_ext: str,
    top_func: Optional[str],
    gexf_root: Path,
) -> Path:
    """
    Build the exact same 3-step graph pipeline as graph_gen_deterministic.py:

        graph_generator()
        -> add_auxiliary_nodes()
        -> augment_graph_hierarchy()

    but do it inside an isolated temporary processed root so that:
      - <project_root>/harp/processed is never modified
      - only the final hierarchy GEXF is copied into gexf_root

    Output layout:
        <gexf_root>/<kernel>_processed_result.gexf
    """
    final_hierarchy_dir = gexf_root.resolve()
    _ensure_dir(final_hierarchy_dir)

    final_gexf = final_hierarchy_dir / f"{kernel_name}_processed_result.gexf"
    final_gexf.unlink(missing_ok=True)

    # Build inside a temp processed tree under the local output bundle.
    with tempfile.TemporaryDirectory(prefix=f"{kernel_name}_gbuild_", dir=str(gexf_root)) as tmpdir:
        tmp_processed_root = Path(tmpdir)
        tmp_original_dir = tmp_processed_root / "original"
        tmp_connected_dir = tmp_processed_root / "extended-pseudo-block-connected"
        tmp_hierarchy_dir = tmp_processed_root / "extended-pseudo-block-connected-hierarchy"

        for p in (tmp_processed_root, tmp_original_dir, tmp_connected_dir, tmp_hierarchy_dir):
            _ensure_dir(p)

        # graph_generator() internally writes to graph_gen_mod.processed_gexf_folder
        # through process_graph(), so override that global temporarily.
        old_processed_gexf_folder = graph_gen_mod.processed_gexf_folder
        try:
            graph_gen_mod.processed_gexf_folder = str(tmp_processed_root)

            _print("Generating deterministic ProGraML/pragma graph ...")
            graph_gen_mod.graph_generator(
                kernel_name,
                str(stage_dir),
                kernel_name,
                src_ext=src_ext,
                generate_programl=True,
                csv_dict=None,
                top_func=top_func,
            )

            _print("Adding connected pseudo-block auxiliary nodes ...")
            graph_gen_mod.add_auxiliary_nodes(
                kernel_name,
                str(tmp_processed_root),   # contains original/
                str(tmp_connected_dir),
                csv_dict=None,
                node_type="block",
                connected=True,
            )

            _print("Adding hierarchy edges ...")
            for_blocks_info = graph_gen_mod.get_for_blocks_info(kernel_name, str(stage_dir))
            graph_gen_mod.augment_graph_hierarchy(
                kernel_name,
                for_blocks_info,
                src_path=str(tmp_connected_dir),
                dst_path=str(tmp_hierarchy_dir),
                csv_dict=None,
                node_type="block",
            )

        finally:
            graph_gen_mod.processed_gexf_folder = old_processed_gexf_folder

        tmp_final_gexf = tmp_hierarchy_dir / f"{kernel_name}_processed_result.gexf"
        _require_file(tmp_final_gexf, "temporary final hierarchy GEXF")

        shutil.copy2(tmp_final_gexf, final_gexf)

    _require_file(final_gexf, "final hierarchy GEXF")
    return final_gexf


def _build_pragma_free_pt(gexf_to_pt_mod, final_gexf: Path, out_pt: Path, max_pragma_length: int) -> Path:
    _ensure_dir(out_pt.parent)
    _print("Building pragma-free PT ...")
    gexf_to_pt_mod.gexf_to_pt(
        gexf_path=str(final_gexf),
        point_json="NONE",
        out_pt=str(out_pt),
        key_name="pragma_free",
        perf=0.0,
        area=0.0,
        max_pragma_length=max_pragma_length,
    )
    _require_file(out_pt, "pragma-free PT")
    return out_pt


def _disable_pragma_conditioning(data):
    for name in (
        "X_pragmascopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
    ):
        if hasattr(data, name):
            setattr(data, name, getattr(data, name).new_zeros(getattr(data, name).shape))

    if hasattr(data, "X_pragma_per_node"):
        data.X_pragma_per_node = data.X_pragma_per_node.new_zeros(data.X_pragma_per_node.shape)
    if hasattr(data, "pragmas"):
        data.pragmas = data.pragmas.new_zeros(data.pragmas.shape)
    return data


def _build_memory_pack(
    pt_to_gnn_emb_mod,
    model_mod,
    Batch,
    config_mod,
    pt_path: Path,
    ckpt_path: Path,
    out_memory: Path,
    max_slots: int,
    disable_pragma_injection: bool,
    device_override: str,
    expected_num_labels: Optional[int] = None,
) -> Path:
    _ensure_dir(out_memory.parent)
    _print("Extracting slot-aligned memory pack ...")

    if device_override:
        config_mod.FLAGS.device = device_override

    pt_point = pt_to_gnn_emb_mod.load_and_clean_graph(str(pt_path))

    required = [
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
        "X_arrayscopenids",
        "X_llm_scopeids",
        "X_llm_scopecat",
        "X_llm_labelid",
    ]
    missing = [k for k in required if not hasattr(pt_point, k)]
    if missing:
        raise RuntimeError(
            f"{pt_path.name} is missing required fields {missing}. "
            "Regenerate the PT with the patched gexf_to_pt_zero.py."
        )

    num_features = pt_point.x.size(-1)
    edge_dim = pt_point.edge_attr.size(-1) if getattr(pt_point, "edge_attr", None) is not None else 0

    model = model_mod.Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(config_mod.FLAGS.device)
    state = __import__("torch").load(str(ckpt_path), map_location=config_mod.FLAGS.device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    batch = Batch.from_data_list([pt_point]).to(config_mod.FLAGS.device)
    if disable_pragma_injection:
        batch = _disable_pragma_conditioning(batch)

    with __import__("torch").no_grad():
        graph_embed = model.forward_embed(batch)
        node_emb = model.forward_node_embed(batch)

        scope = batch.X_llm_scopeids.bool()
        label = batch.X_llm_labelid.long()
        sel = scope & (label > 0) & (label <= max_slots)
        sel_idx = sel.nonzero(as_tuple=False).view(-1)

        torch_mod = __import__("torch")
        node_embs = torch_mod.zeros((max_slots, node_emb.size(-1)), dtype=node_emb.dtype, device=node_emb.device)
        node_embs_mask = torch_mod.zeros((max_slots,), dtype=torch_mod.bool, device=node_emb.device)
        slot_cats = torch_mod.zeros((max_slots,), dtype=torch_mod.long, device=node_emb.device)
        node_ids = [-1] * max_slots
        labels = [-1] * max_slots

        for ni in sel_idx.tolist():
            lid = int(label[ni].item())
            slot = lid - 1
            node_embs[slot] = node_emb[ni]
            node_embs_mask[slot] = True
            slot_cats[slot] = int(batch.X_llm_scopecat[ni].item())
            node_ids[slot] = ni
            labels[slot] = lid

        active_labels = sorted(int(x) for x in labels if int(x) > 0)

        if expected_num_labels is not None:
            expected = list(range(1, expected_num_labels + 1))
            if active_labels != expected:
                raise RuntimeError(
                    f"Memory anchor mismatch. Expected active labels {expected}, "
                    f"but got {active_labels}. "
                    "This means some placeholder sites did not map to pseudo-block/array-scope anchors."
                )

        node_embs = torch_mod.nan_to_num(node_embs.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        node_embs_mask = node_embs_mask.detach().cpu()
        graph_embed = torch_mod.nan_to_num(graph_embed.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

        # Same clipping as build_harp_memory.py
        max_norm = 20.0
        eps = 1e-6
        norms = node_embs.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        scale = (max_norm / norms).clamp(max=1.0)
        node_embs = node_embs * scale

        pack = {
            "pt_path": str(pt_path),
            "ckpt": str(ckpt_path),
            "disable_pragma_injection": bool(disable_pragma_injection),
            "gnn_dim": int(node_embs.size(-1)),
            "node_embs": node_embs,
            "node_embs_mask": node_embs_mask,
            "graph_embed": graph_embed,
            "max_slots": max_slots,
            "slot_ids": torch_mod.arange(1, max_slots + 1, dtype=torch_mod.long),
            "slot_cats": slot_cats.detach().cpu(),
            "node_ids": node_ids,
            "labels": labels,
        }

        torch_mod.save(pack, str(out_memory))

    _require_file(out_memory, "memory pack")
    return out_memory


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate labels + kernel_info.txt, then build the hierarchy GEXF, "
            "pragma-free PT, and memory.pt for one application folder."
        )
    )
    ap.add_argument("--app-dir", required=True, help="Application folder containing the source file and headers")
    ap.add_argument("--source-file", default="", help="Optional explicit source file inside --app-dir")
    ap.add_argument("--kernel-name", default="", help="Logical kernel/application name. Defaults to the application folder name")
    ap.add_argument("--top-func", default="", help="Optional explicit top-level HLS function name for the first line of kernel_info.txt")
    ap.add_argument("--gnn-ckpt", required=True, help="Frozen GNN checkpoint used to build the memory pack")
    ap.add_argument("--out-dir", default="", help="Output directory for copied final artifacts. Default: <app-dir>/code_to_memory_pt_outputs/<kernel-name>")
    ap.add_argument("--max-slots", type=int, default=64)
    ap.add_argument("--max-pragma-length", type=int, default=93)
    ap.add_argument("--device", default="", help="Optional override for FLAGS.device during memory extraction, e.g. cuda:0 or cpu")
    ap.add_argument("--keep-stage", action="store_true", help="Keep the staged kernel directory under <project_root>/harp/<kernel-name>")
    ap.add_argument("--overwrite", action="store_true", help="Remove stale per-kernel products before rebuilding")
    ap.add_argument("--allow-pragma-injection", action="store_true", help="Use pragma-conditioned GNN updates when extracting memory (default: disabled, recommended)")
    ap.add_argument("--auto-set-pythonhashseed", action="store_true", help="If PYTHONHASHSEED was not exported before launch, set it to 0 before importing graph_gen_deterministic so the script can proceed")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    app_dir = Path(args.app_dir).resolve()
    if not app_dir.is_dir():
        raise NotADirectoryError(f"--app-dir is not a directory: {app_dir}")

    source_path = _find_source_file(app_dir, args.source_file)
    header_files = _find_header_files(app_dir)
    _require_file(Path(args.gnn_ckpt).resolve(), "GNN checkpoint")

    kernel_name = _kernel_name_from_args(app_dir, source_path, args.kernel_name)

    pre = _prepare_labeled_source_and_kernel_info(
        source_path=source_path,
        header_files=header_files,
        top_func_override=args.top_func,
    )

    script_dir = Path(__file__).resolve().parent
    mods = _import_project_modules(script_dir, auto_set_pythonhashseed=args.auto_set_pythonhashseed)

    graph_gen_mod = mods["graph_gen_mod"]
    project_root = Path(graph_gen_mod.get_root_path()).resolve()

    stage_dir = project_root / "harp" / kernel_name
    out_dir = app_dir / "code_to_memory_outputs"
    gexf_out_dir = out_dir / "gexf"
    pt_out_dir = out_dir / "pt"
    mem_out_dir = out_dir / "memory"
    info_out_dir = out_dir / "meta"
    for p in (gexf_out_dir, pt_out_dir, mem_out_dir, info_out_dir):
        _ensure_dir(p)

    if args.overwrite:
        _print(f"Overwriting stale products for kernel '{kernel_name}'")
        _purge_kernel_products(project_root, kernel_name)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
            for p in (gexf_out_dir, pt_out_dir, mem_out_dir, info_out_dir):
                _ensure_dir(p)

    _print(f"Project root     : {project_root}")
    _print(f"Application dir  : {app_dir}")
    _print(f"Kernel name      : {kernel_name}")
    _print(f"Top function     : {pre['top_function']}")
    _print(f"Source           : {source_path.name}")
    _print(f"Headers          : {', '.join(h.name for h in header_files) if header_files else '(none)'}")
    _print(f"Resolved decls   : {len(pre['decls'])}")
    _print(f"Stage dir        : {stage_dir}")
    _print(f"Output dir       : {out_dir}")

    staged = _stage_application(
        app_dir=app_dir,
        stage_dir=stage_dir,
        kernel_name=kernel_name,
        source_path=source_path,
        header_files=header_files,
        labeled_source_text=str(pre["labeled_source_text"]),
        kernel_info_text=str(pre["kernel_info_text"]),
        insert_placeholders_fn=mods["insert_placeholders_mod"].insert_placeholders,
    )

    final_gexf = _build_hierarchy_graph(
        graph_gen_mod=graph_gen_mod,
        kernel_name=kernel_name,
        stage_dir=stage_dir,
        src_ext=source_path.suffix.lower().lstrip("."),
        top_func=str(pre["top_function"] or ""),
        gexf_root=gexf_out_dir,
    )

    pt_path = _build_pragma_free_pt(
        gexf_to_pt_mod=mods["gexf_to_pt_mod"],
        final_gexf=final_gexf,
        out_pt=pt_out_dir / f"{kernel_name}_processed_result.pt",
        max_pragma_length=args.max_pragma_length,
    )

    memory_path = _build_memory_pack(
        pt_to_gnn_emb_mod=mods["pt_to_gnn_emb_mod"],
        model_mod=mods["model_mod"],
        Batch=mods["Batch"],
        config_mod=mods["config_mod"],
        pt_path=pt_path,
        ckpt_path=Path(args.gnn_ckpt).resolve(),
        out_memory=mem_out_dir / f"{kernel_name}.memory.pt",
        max_slots=args.max_slots,
        disable_pragma_injection=not args.allow_pragma_injection,
        device_override=args.device,
        expected_num_labels=len(pre["decls"]),
    )

    # Copy the final hierarchy graph and generated intermediates into the output bundle.
    placeholder_copy = info_out_dir / staged["placeholder_src"].name
    shutil.copy2(staged["placeholder_src"], placeholder_copy)
    staged_src_copy = info_out_dir / staged["staged_src"].name
    shutil.copy2(staged["staged_src"], staged_src_copy)
    staged_kernel_info_copy = info_out_dir / "kernel_info.txt"
    shutil.copy2(staged["staged_kernel_info"], staged_kernel_info_copy)

    summary = {
        "app_dir": str(app_dir),
        "project_root": str(project_root),
        "kernel_name": kernel_name,
        "top_function": pre["top_function"],
        "source_file": str(source_path),
        "header_files": [str(h) for h in header_files],
        "staged_source": str(staged["staged_src"]),
        "generated_kernel_info": str(staged["staged_kernel_info"]),
        "placeholder_source": str(staged["placeholder_src"]),
        "final_hierarchy_gexf": str(final_gexf),
        "pragma_free_pt": str(pt_path),
        "memory_pt": str(memory_path),
        "copied_final_hierarchy_gexf": str(final_gexf),
        "copied_placeholder_source": str(placeholder_copy),
        "copied_labeled_source": str(staged_src_copy),
        "copied_kernel_info": str(staged_kernel_info_copy),
        "max_slots": args.max_slots,
        "max_pragma_length": args.max_pragma_length,
        "disable_pragma_injection": not args.allow_pragma_injection,
        "decls": [
            {
                "kind": d.kind,
                "function": d.function_name,
                "line_idx": d.line_idx,
                "existing_label": d.existing_label,
                "final_label": d.final_label,
                "loop_bound": d.loop_bound,
                "array_name": d.array_name,
                "array_dims": d.array_dims,
            }
            for d in pre["decls"]
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print("Done.")
    _print(f"Generated labeled source : {staged_src_copy}")
    _print(f"Generated kernel_info    : {staged_kernel_info_copy}")
    _print(f"Final hierarchy GEXF    : {final_gexf}")
    _print(f"Pragma-free PT          : {pt_path}")
    _print(f"Memory pack             : {memory_path}")
    _print(f"Summary                 : {out_dir / 'summary.json'}")

    if not args.keep_stage:
        _print(f"Cleaning staged directory: {stage_dir}")
        shutil.rmtree(stage_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
