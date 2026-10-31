"""Legacy entry point; implementation is maintained in LLM_branch/inference/inference_qLoRA.py."""
from pathlib import Path as _Path

# Execute in the legacy module namespace to retain pickle identities and CLI guards.
# Keep the caller's sys.path so LLVM-era sibling imports retain their meaning.
_source = _Path(__file__).resolve().parents[1] / 'LLM_branch/inference/inference_qLoRA.py'
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
