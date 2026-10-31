"""Legacy entry point; implementation is maintained in GNN_branch/gexf_to_pt_zero.py."""
from pathlib import Path as _Path

# Preserve the legacy module namespace, pickle identities and caller sys.path.
_source = _Path(__file__).resolve().parents[1] / 'GNN_branch/gexf_to_pt_zero.py'
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
