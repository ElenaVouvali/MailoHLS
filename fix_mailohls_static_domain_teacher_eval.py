#!/usr/bin/env python3
from pathlib import Path
import ast
import os
import tempfile

ROOT = Path.cwd()
TRAIN = ROOT / "LLM_branch/train/train_SFT_xattn_new.py"
TEST = ROOT / "LLM_branch/tests/test_stage1_final_contract.py"

if not TRAIN.is_file() or not TEST.is_file():
    raise SystemExit("Run this script from the MailoHLS repository root.")

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)

train = TRAIN.read_text(encoding="utf-8")
test = TEST.read_text(encoding="utf-8")

old_train = '''            trace.append({
                "label": label,
                "lhs": lhs,
                "static_candidate_count": len(original_candidates),
                "forced_by_semantics": len(candidates) == 1,
                "candidates": [dict(record) for record in scored],
            })
'''
new_train = '''            trace.append({
                "label": label,
                "lhs": lhs,
                "static_candidate_count": len(candidates),
                # Dynamic cross-directive semantic pruning is intentionally
                # disabled in the final Stage-1 contract. A singleton static
                # proposal domain is not "forced by semantics".
                "forced_by_semantics": False,
                "candidates": [dict(record) for record in scored],
            })
'''
train = replace_once(
    train, old_train, new_train,
    "teacher-forced trace static candidate metadata"
)

old_test = '''    assert trace[0]["candidates"][0]["rhs"] == "0"
    # PIPE is teacher-forced to 1 for the prefix, therefore UNROLL becomes
    # semantically forced to 0 even though the model preferred PIPE=0.
    assert trace[1]["forced_by_semantics"] is True
    assert trace[1]["candidates"][0]["rhs"] == "0"
'''
new_test = '''    assert trace[0]["candidates"][0]["rhs"] == "0"
    # The reference PIPE value is appended to the teacher-forced prefix, but
    # it must NOT delete source-supported UNROLL proposals. The local scorer
    # therefore remains free to prefer UNROLL=2.
    assert trace[1]["forced_by_semantics"] is False
    assert trace[1]["static_candidate_count"] == 2
    assert trace[1]["candidates"][0]["rhs"] == "2"
'''
test = replace_once(
    test, old_test, new_test,
    "teacher-forced static-domain regression test"
)

ast.parse(train, filename=str(TRAIN))
ast.parse(test, filename=str(TEST))

def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

atomic_write(TRAIN, train)
atomic_write(TEST, test)

print("Applied teacher-forced static-domain hotfix.")
print("Run next:")
print("  python -m pytest -q LLM_branch/tests/test_stage1_final_contract.py")
print("  python -m pytest -q GNN_branch/tests")
print("  git diff --check -- LLM_branch GNN_branch")
