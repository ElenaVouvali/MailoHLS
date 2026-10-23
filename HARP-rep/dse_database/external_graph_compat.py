"""
Compatibility helpers for external HARP graph generation.

Does NOT modify HARP's graph/model/search logic.
Only extends HARP's constant loop-bound parser to understand
C/C++ integer casts emitted by the Merlin DS Generator.
"""

import re

# C integer casts that may occur in constant loop bounds.
_C_INTEGER_CAST = re.compile(
    r'\(\s*'
    r'(?:(?:unsigned|signed)\s+)?'
    r'(?:long\s+long|long|int|short|char)'
    r'\s*\)'
)

_ALLOWED_EXPR = re.compile(r'^[0-9()+\-*/%]+$')


def external_get_tc_for_loop(for_loop_text):
    """
    Equivalent to HARP get_tc_for_loop(), but accepts constant C/C++
    integer casts such as:

        ((unsigned long)262144) /
        (((unsigned long)256) / (4 * ((unsigned long)8)))

    The HARP kernels use positive integer loop bounds, so '/' is
    evaluated using integer division to preserve C integer semantics.
    """

    comp = for_loop_text.split(';')[1].strip()

    delims = ['<=', '>=', '<', '>', '--']

    delim = None
    for d in delims:
        if d in comp:
            delim = d
            break

    if delim is None:
        raise RuntimeError(
            "No comparison sign found in loop: " + for_loop_text
        )

    if delim == '--':
        return 0

    rhs = comp.split(delim, 1)[1].strip()

    # Remove C/C++ integral casts only.
    rhs = _C_INTEGER_CAST.sub('', rhs)

    # Remove common integer literal suffixes if present.
    rhs = re.sub(r'(?<=\d)[uUlL]+\b', '', rhs)

    expr = rhs.replace(' ', '')

    if not _ALLOWED_EXPR.fullmatch(expr):
        raise RuntimeError(
            "Unsupported external constant loop-bound expression:\n"
            f"  source: {for_loop_text}\n"
            f"  parsed: {expr}"
        )

    # Positive loop-bound arithmetic: C integer division.
    expr = expr.replace('/', '//')

    value = eval(expr, {"__builtins__": {}}, {})

    return int(value)


def install(graph_gen):
    graph_gen.get_tc_for_loop = external_get_tc_for_loop
    print("[HARP-EXTERNAL] installed C/C++ constant-bound compatibility parser")
