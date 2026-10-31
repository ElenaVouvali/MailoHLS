"""Legacy entry point; implementation is maintained in Preprocessing/filter_pt.py."""
from pathlib import Path as _Path

# Preserve the legacy module namespace, pickle identities and caller sys.path.
_source = _Path(__file__).resolve().parents[1] / 'Preprocessing/filter_pt.py'
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
