#!/usr/bin/env python3
"""Conservative MailoHLS action-point discovery for previously unlabeled kernels.

Emits (without changing the input):
  <stem>.c/.cpp         pragma-free, /*Lk:*/-annotated source for MLIR
  <stem>_placeholders.* LLM source with existing MailoHLS auto{...} markers
  kernel_info.txt       repository-compatible action registry
  directive_domains.json  real MailoHLS domains + legal complete action tuples
  action_manifest.json  recorded AST location/identity and exclusions

Requires Clang with JSON AST dumping, plus this repository on PYTHONPATH.
Only canonical, statically resolved for-loops and fixed-size local C arrays
are auto-accepted. Unknown, ambiguous, or unsupported sites are reported and
skipped; this is not a general C/C++ or HLS-legality prover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Site:
    offset: int
    line: int
    col: int
    kind: str
    detail: str
    extent: int | None = None
    dimensions: tuple[int, ...] = ()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def children(node: dict) -> list[dict]:
    return [c for c in node.get("inner", ()) if isinstance(c, dict) and c.get("kind")]


def walk(node: dict):
    yield node
    for c in children(node):
        yield from walk(c)


def start(node: dict) -> dict:
    return node.get("range", {}).get("begin", {}) or node.get("loc", {})


def offset_of(point: dict) -> int | None:
    # A macro spelling location is not the token we can safely annotate.
    if "spellingLoc" in point or "expansionLoc" in point:
        return None
    x = point.get("offset")
    return x if isinstance(x, int) else None


def source_line(raw: bytes, pos: int) -> tuple[int, int, bytes]:
    line_start = raw.rfind(b"\n", 0, pos) + 1
    line_end = raw.find(b"\n", pos)
    if line_end < 0:
        line_end = len(raw)
    return raw.count(b"\n", 0, pos) + 1, pos - line_start + 1, raw[line_start:line_end]


def only_declaration_start(raw: bytes, pos: int) -> bool:
    # Existing placeholders are deliberately not accepted. Nothing executable
    # may precede an inserted source-marker comment on its physical line.
    return raw[raw.rfind(b"\n", 0, pos) + 1:pos].strip() == b""


def strip_cast(expr: dict) -> dict:
    while expr.get("kind") in {
        "ImplicitCastExpr", "CStyleCastExpr", "CXXStaticCastExpr",
        "ParenExpr", "ConstantExpr", "ExprWithCleanups",
    }:
        kids = children(expr)
        if len(kids) != 1:
            break
        expr = kids[0]
    return expr


def integer(expr: dict) -> int | None:
    expr = strip_cast(expr)
    if expr.get("kind") == "IntegerLiteral":
        try:
            return int(str(expr["value"]), 0)
        except (KeyError, ValueError):
            # Clang's JSON emits decimal integers such as '08' and '-10'.
            try:
                return int(str(expr["value"]), 10)
            except (KeyError, ValueError):
                return None
    if expr.get("kind") == "UnaryOperator" and expr.get("opcode") in {"+", "-"}:
        kids = children(expr)
        n = integer(kids[0]) if len(kids) == 1 else None
        return (n if expr["opcode"] == "+" else -n) if n is not None else None
    return None


def refers_to(expr: dict, name: str, decl_id: str | None) -> bool:
    expr = strip_cast(expr)
    if expr.get("kind") != "DeclRefExpr":
        return False
    ref = expr.get("referencedDecl", {})
    return ref.get("name") == name and (
        decl_id is None or str(ref.get("id")) == str(decl_id)
    )


def body_modifies_induction(body: dict, name: str, decl_id: str | None) -> bool:
    for n in walk(body):
        kind = n.get("kind")
        if kind in {"BreakStmt", "GotoStmt", "ReturnStmt", "ContinueStmt", "CXXThrowExpr"}:
            return True
        kids = children(n)
        if kind in {"UnaryOperator", "BinaryOperator", "CompoundAssignOperator"} and kids:
            op = n.get("opcode")
            if op in {"++", "--", "=", "+=", "-=", "*=", "/=", "%="} and refers_to(kids[0], name, decl_id):
                return True
            if op == "&" and refers_to(kids[0], name, decl_id):
                return True
    return False


def canonical_trip_count(loop: dict) -> tuple[int | None, str]:
    # Clang ForStmt has 5 fixed slots: init, condition-var, condition,
    # increment, body. The second slot is frequently an empty dict.
    parts = loop.get("inner", [])
    if len(parts) != 5:
        return None, "noncanonical-for-AST"
    init, _cond_var, cond, inc, body = parts
    if init.get("kind") != "DeclStmt":
        return None, "requires-single-integer-declaration-init"
    vars_ = children(init)
    if len(vars_) != 1 or vars_[0].get("kind") != "VarDecl":
        return None, "multiple-or-nonvariable-initializers"
    var = vars_[0]
    name, decl_id = var.get("name"), var.get("id")
    if not name or not re.fullmatch(r"(?:signed\s+)?(?:char|short|int|long|long long)(?:\s+int)?", var.get("type", {}).get("qualType", "")):
        return None, "induction-variable-not-supported-signed-integer"
    init_children = children(var)
    lo = integer(init_children[-1]) if init_children else None
    if lo is None:
        return None, "nonconstant-loop-start"

    if cond.get("kind") != "BinaryOperator" or cond.get("opcode") not in {"<", "<=", ">", ">=", "!="}:
        return None, "unsupported-loop-condition"
    cond_children = children(cond)
    if len(cond_children) != 2 or not refers_to(cond_children[0], name, decl_id):
        return None, "condition-does-not-compare-induction-variable"
    hi = integer(cond_children[1])
    if hi is None:
        return None, "nonconstant-loop-bound"

    ik = children(inc)
    step = None
    if len(ik) == 1 and inc.get("kind") == "UnaryOperator" and refers_to(ik[0], name, decl_id):
        step = 1 if inc.get("opcode") == "++" else (-1 if inc.get("opcode") == "--" else None)
    elif len(ik) == 2 and inc.get("kind") == "CompoundAssignOperator" and refers_to(ik[0], name, decl_id):
        delta = integer(ik[1])
        if delta is not None and inc.get("opcode") in {"+=", "-="}:
            step = delta if inc["opcode"] == "+=" else -delta
    if step is None or step == 0:
        return None, "unsupported-or-zero-loop-step"
    if body_modifies_induction(body, name, decl_id):
        return None, "loop-body-has-control-exit-or-induction-write"

    cmp = cond["opcode"]
    if cmp in {"<", "<="} and step < 0 or cmp in {">", ">="} and step > 0:
        return None, "potentially-nonterminating-loop"
    if cmp == "!=":
        if (hi - lo) * step <= 0 or (hi - lo) % step != 0:
            return None, "inequality-loop-bound-not-reachable"
        count = abs((hi - lo) // step)
    else:
        end = hi + 1 if cmp == "<=" else hi - 1 if cmp == ">=" else hi
        count = len(range(lo, end, step))
    if not 0 < count <= 2**31 - 1:
        return None, "empty-or-too-large-loop"
    qtype = var.get("type", {}).get("qualType", "")
    bits = 8 if "char" in qtype else 16 if "short" in qtype else (64 if "long" in qtype else 32)
    minimum, maximum = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    after_last = lo + count * step  # the final increment also executes
    if any(not minimum <= number <= maximum for number in (lo, hi, after_last)):
        return None, "signed-induction-overflow-or-bound-conversion-risk"
    return count, "verified-header-trip-count"


def array_shape(node: dict) -> tuple[int, ...] | None:
    typ = node.get("type", {}).get("qualType", "")
    # Clang canonicalizes macros such as N=32 into float[32]. No VLAs,
    # pointer parameters, references, std::array, or zero extents.
    suffix = re.search(r"((?:\[\d+\])+)$", typ)
    if not suffix or "*" in typ[:suffix.start()] or "&" in typ[:suffix.start()]:
        return None
    nums = tuple(int(x) for x in re.findall(r"\[(\d+)\]", suffix.group(1)))
    if not nums or any(n < 1 for n in nums):
        return None
    return nums


def clang_tree(source: Path, clang: str, flags: list[str]) -> dict:
    executable = shutil.which(clang)
    if not executable:
        raise RuntimeError(f"Clang executable not found: {clang}")
    language = "c++" if source.suffix.lower() in {".cc", ".cpp", ".cxx", ".c++"} else "c"
    std = "-std=c++17" if language == "c++" else "-std=c11"
    command = [executable, "-x", language, std, "-Xclang", "-ast-dump=json", "-fsyntax-only", *flags, str(source)]
    p = subprocess.run(command, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("Clang failed; pass the same -I/-D options as your HLS frontend:\n" + p.stderr[-8000:])
    return json.loads(p.stdout)


def discover(ast: dict, raw: bytes, top: str) -> tuple[list[Site], list[dict]]:
    funcs = [x for x in walk(ast) if x.get("kind") == "FunctionDecl" and x.get("name") == top
             and any(c.get("kind") == "CompoundStmt" for c in children(x))]
    if len(funcs) != 1:
        raise RuntimeError(f"Expected one concrete top function '{top}', found {len(funcs)}. Use the exact definition name.")
    body = next(c for c in children(funcs[0]) if c.get("kind") == "CompoundStmt")
    sites: list[Site] = []
    rejected: list[dict] = []
    var_lines: dict[int, int] = {}
    all_nodes = list(walk(body))
    for n in all_nodes:
        if n.get("kind") == "VarDecl" and array_shape(n):
            pos = offset_of(start(n))
            if pos is not None and pos < len(raw):
                line, _, _ = source_line(raw, pos)
                var_lines[line] = var_lines.get(line, 0) + 1

    for n in all_nodes:
        kind = n.get("kind")
        if kind in {"WhileStmt", "DoStmt", "CXXForRangeStmt"}:
            pos = offset_of(start(n))
            line = source_line(raw, pos)[0] if pos is not None and pos < len(raw) else None
            rejected.append({"line": line, "kind": kind,
                             "reason": "not-supported-by-existing-for-loop-placeholder-contract"})
            continue
        if kind == "VarDecl" and "[" in n.get("type", {}).get("qualType", "") and not array_shape(n):
            pos = offset_of(start(n))
            line = source_line(raw, pos)[0] if pos is not None and pos < len(raw) else None
            rejected.append({"line": line, "kind": kind,
                             "reason": "nonconstant-or-unsupported-C-array-type"})
            continue
        if kind not in {"ForStmt", "VarDecl"}:
            continue
        pos = offset_of(start(n))
        if pos is None or pos >= len(raw) or not only_declaration_start(raw, pos):
            if kind == "ForStmt" or (kind == "VarDecl" and array_shape(n)):
                rejected.append({"kind": kind, "reason": "macro-or-non-leading-source-token"})
            continue
        line, col, physical = source_line(raw, pos)
        tail = raw[pos:raw.find(b"\n", pos) if b"\n" in raw[pos:] else len(raw)].decode("utf8", errors="replace")
        if kind == "ForStmt":
            if not re.match(r"^for\s*\(.*\)\s*\{\s*(?://.*)?$", tail):
                # The existing insert_placeholders.py inserts directives on
                # the NEXT physical line. An inline body or brace on the next
                # line would put them at the wrong source location.
                rejected.append({"line": line, "kind": kind,
                                 "reason": "requires-for-header-ending-with-opening-brace"})
                continue
            trip, reason = canonical_trip_count(n)
            if trip is None:
                rejected.append({"line": line, "kind": kind, "reason": reason})
                continue
            sites.append(Site(pos, line, col, "loop", "for", extent=trip))
        else:
            dims, name = array_shape(n), n.get("name", "")
            if not dims:
                continue
            if var_lines.get(line) != 1 or not name:
                rejected.append({"line": line, "kind": "array", "reason": "multiple-declarations-on-line"})
                continue
            # Must be the simple declaration grammar supported by source
            # analyzer + existing insert_placeholders.py (one-line only).
            if not re.match(r"^[A-Za-z_][\w:\<\>\s\*&]*?\s+" + re.escape(name)
                            + r"\s*(?:\[[^\]]+\]\s*)+\s*(?:=\s*[^;]+)?;", tail):
                rejected.append({"line": line, "kind": "array", "reason": "unsupported-declaration-syntax"})
                continue
            sites.append(Site(pos, line, col, "array", name, dimensions=dims))
    sites.sort(key=lambda s: (s.offset, 0 if s.kind == "array" else 1))
    if len({s.line for s in sites}) != len(sites):
        raise RuntimeError("Two discovered action sites occupy one source line. Split statements across lines first.")
    return sites, rejected


def annotate(raw: bytes, sites: list[Site]) -> tuple[bytes, list[dict]]:
    records = []
    for k, site in enumerate(sites, 1):
        records.append({"id": f"L{k}", "kind": site.kind, "line": site.line,
                        "column": site.col, "offset": site.offset, "name": site.detail,
                        "trip_count": site.extent, "dimensions": list(site.dimensions)})
    annotated = raw
    for item in reversed(records):
        pos = item["offset"]
        annotated = annotated[:pos] + f"/*{item['id']}:*/ ".encode() + annotated[pos:]
    return annotated, records


def write_new(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() == content:
            return
        raise RuntimeError(f"Refusing to overwrite a non-identical artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def execute(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    if not (root / "GNN_branch" / "insert_placeholders.py").is_file():
        raise RuntimeError("Install as GNN_branch/deterministic_analyzer.py in the MailoHLS repository")
    source = args.source.resolve()
    if source.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".c++"}:
        raise ValueError("Supported input extensions: .c, .cc, .cpp, .cxx")
    raw = source.read_bytes()
    raw.decode("utf-8")
    if re.search(rb"\bL[1-9]\d*\s*:|auto\{_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+\}", raw):
        raise ValueError("Source already contains MailoHLS labels/placeholders; refusing to relabel")
    if re.search(rb"(?im)^\s*#\s*pragma\s+HLS\b", raw):
        raise ValueError("Source already contains HLS pragmas; supply a pragma-free input")
    ast = clang_tree(source, args.clang, args.clang_arg)
    sites, rejected = discover(ast, raw, args.top)
    if args.strict and rejected:
        raise RuntimeError("--strict: skipped candidate sites: " + json.dumps(rejected))
    if not sites:
        raise RuntimeError("No supported actions found. Review exclusions or supply manual labels/metadata.")
    if len(sites) > args.max_actions:
        raise RuntimeError(f"Detected {len(sites)} actions; model contract supports {args.max_actions} absolute slots")

    out = args.output.resolve() if args.output else source.parent / f"mailohls_generated_{source.stem}"
    if out == source.parent:
        raise RuntimeError("Use a separate output directory to protect the original source")
    labeled, records = annotate(raw, sites)
    suffix = source.suffix
    labeled_path = out / (source.stem + suffix)
    placeholders_path = out / (source.stem + "_placeholders" + suffix)
    metadata_path = out / "kernel_info.txt"
    domains_path = out / "directive_domains.json"
    manifest_path = out / "action_manifest.json"
    lines = [args.top]
    for x in records:
        if x["kind"] == "loop":
            lines.append(f"{x['id']},loop,{x['trip_count']}")
        else:
            nums = ",".join(f"{i},{n}" for i, n in enumerate(x["dimensions"], start=1))
            lines.append(f"{x['id']},array,{x['name']},{nums}")
    info = "\n".join(lines) + "\n"
    write_new(labeled_path, labeled)
    write_new(metadata_path, info.encode())

    # Reuse the actual repo implementations; never quietly invent a second
    # directive-domain definition. TCAD domains include complete legal tuples.
    sys.path.insert(0, str(root))
    from LLM_branch.common import directive_domains as domains
    top, actions = None, None
    try:
        from GNN_branch.mlir_graph_gen import load_kernel_info_actions
        top, actions = load_kernel_info_actions(labeled_path, metadata_path)
        if top != args.top or len(actions) != len(records):
            raise RuntimeError("MLIR action registry mismatch")
    except ImportError as exc:
        # The MLIR graph module may require optional native dependencies.
        # LLM-side metadata and placeholder checks remain mandatory.
        rejected.append({"kind": "contract_check", "reason": f"MLIR registry import unavailable: {exc}"})
    template = domains.prepare_source_template(args.top, labeled_path, out)
    write_new(placeholders_path, (template.rstrip() + "\n").encode())
    action_map = domains.load_source_actions(args.top, out)
    if len(action_map) != len(records):
        raise RuntimeError("Incomplete LLM action registry")
    if args.policy == "tcad":
        entries = domains.tcad.source_site_domains(args.top, template, out)
        tuples = {label: [list(row) for row in domains.tcad.action_tuples(action)]
                  for label, action in action_map.items()}
    else:
        entries = domains.legacy_site_domains(args.top, template, out)
        tuples = None
    payload = {"policy": args.policy, "flat_directive_domains": dict(entries),
               "legal_action_tuples": tuples, "note": "Locally admissible proposals; HLS synthesis feasibility is not guaranteed"}
    write_new(domains_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    report = {"source": str(source), "source_sha256": digest(raw),
              "labeled_source_sha256": digest(labeled), "top": args.top,
              "clang": args.clang, "clang_args": args.clang_arg,
              "actions": records, "skipped_candidates": rejected,
              "location_policy": "source-order indices stable for identical source and flags, not across inserted sites",
              "scope": "canonical constant-bound for-loops and fixed-size local C arrays",
              "output_paths": {"labeled_source": str(labeled_path),
                               "llm_template": str(placeholders_path),
                               "kernel_info": str(metadata_path),
                               "directive_domains": str(domains_path)}}
    write_new(manifest_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="Original *unlabeled* C/C++ file")
    p.add_argument("--top", required=True, help="Exact C/C++ top-function name")
    p.add_argument("--output", type=Path, help="New output directory (default: sibling generated directory)")
    p.add_argument("--clang", default="clang", help="Clang binary with JSON AST support")
    p.add_argument("--clang-arg", action="append", default=[], help="Repeat for compiler args, e.g. --clang-arg=-I/path or --clang-arg=-DN=64")
    p.add_argument("--policy", choices=("tcad", "legacy"), default="tcad")
    p.add_argument("--max-actions", type=int, default=64)
    p.add_argument("--strict", action="store_true", help="Reject all skipped candidates instead of reporting them")
    args = p.parse_args()
    try:
        report = execute(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        p.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({"actions": len(report["actions"]),
                      "skipped": len(report["skipped_candidates"]),
                      "outputs": report["output_paths"]}, indent=2))
    if report["skipped_candidates"]:
        print("WARNING: some candidates were skipped; inspect action_manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
