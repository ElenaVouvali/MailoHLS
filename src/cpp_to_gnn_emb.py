"""Legacy entry point; implementation is maintained in GNN_branch/cpp_to_gnn_emb.py."""
from pathlib import Path as _Path

# Execute in the legacy module namespace to retain pickle identities and CLI guards.
# Keep the caller's sys.path so LLVM-era sibling imports retain their meaning.
_source = _Path(__file__).resolve().parents[1] / 'GNN_branch/cpp_to_gnn_emb.py'
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
