#!/usr/bin/env python3
"""Add progress visibility to historical Stage-2 selection callback only.

Run from the historical MailoHLS worktree, with no active historical launcher.
Does not change candidate scoring, model, metrics, or selected checkpoint logic.
"""
from pathlib import Path
p = Path('LLM_branch/train/train_SFT_xattn_new.py')
if not p.is_file():
    raise SystemExit(f'Run from historical worktree; missing {p}')
s = p.read_text(encoding='utf-8')
old = '            rows = [self._run_case(model, case) for case in self.selection_cases]'
new = '''            import time
            rows = []
            for case_idx, case in enumerate(self.selection_cases, 1):
                start_t = time.perf_counter()
                print(
                    f"[VAL-CASE] step={state.global_step} "
                    f"case={case_idx}/{len(self.selection_cases)} "
                    f"kernel={case.kernel_name} START",
                    flush=True,
                )
                rows.append(self._run_case(model, case))
                print(
                    f"[VAL-CASE] step={state.global_step} "
                    f"case={case_idx}/{len(self.selection_cases)} "
                    f"kernel={case.kernel_name} "
                    f"seconds={time.perf_counter() - start_t:.1f} END",
                    flush=True,
                )'''
if new in s:
    print('Already patched; no changes made')
elif s.count(old) != 1:
    raise SystemExit(f'Expected exactly one original callback site, found {s.count(old)}; refusing edit')
else:
    p.write_text(s.replace(old, new), encoding='utf-8')
    print(f'Installed per-case progress logging in {p}')
