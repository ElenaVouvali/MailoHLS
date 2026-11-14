#!/usr/bin/env python3
"""Build an isolated runner copy that can use compact LLM-only CHStone views.

Canonical sources and memory hashes remain untouched; original runner unchanged.
Works against Davinci's 2026-11-13 stage2_infer.py anchor as inspected via CLI.
"""
import argparse
from pathlib import Path
import py_compile

ANCHOR='prepared = domains.prepare_source_template(kernel, code, a.sources_root)'
PATCH='''\n        # Prompt-only compact view: source checksum, compiler memory and HLS code\n        # continue to reference the untouched authenticated original source.\n        if kernel in ("chstone-aes", "chstone-jpeg"):\n            import hashlib as _view_hashlib\n            import re as _view_re\n            _view_path = a.deployment / "eval_prompt_views" / (kernel + ".c")\n            if _view_path.is_file():\n                _source_keys = set(_view_re.findall(r"auto\\{_[A-Z0-9_]+_L\\d+\\}", prepared, _view_re.I))\n                _view_text = domains.prepare_source_template(kernel, _view_path, a.sources_root)\n                _view_keys = set(_view_re.findall(r"auto\\{_[A-Z0-9_]+_L\\d+\\}", _view_text, _view_re.I))\n                if _view_keys != _source_keys:\n                    raise RuntimeError("Prompt-only CHStone view changes directive action fields")\n                prepared = _view_text\n                print("[PROMPT-ONLY-VIEW]", kernel, _view_path, "sha256",\n                      _view_hashlib.sha256(_view_path.read_bytes()).hexdigest(), flush=True)\n'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=a.runner.read_text(encoding='utf-8')
    if original.count(ANCHOR)!=1:raise RuntimeError('Expected single historical source-template anchor; refusing patch')
    if a.output.exists():raise FileExistsError('Runner copy already exists')
    # Never mutate /home/evouvali/MailoHLS/inference/stage2_infer.py
    patched=original.replace(ANCHOR,ANCHOR+PATCH)
    a.output.write_text(patched,encoding='utf-8')
    try:py_compile.compile(str(a.output),doraise=True)
    except Exception:
        a.output.unlink()
        raise
    print('Created isolated runner:',a.output)
    print('Original preserved:',a.runner)
if __name__=='__main__':main()
