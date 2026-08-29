"""Test bootstrap for source-tree execution.

The project is intentionally runnable directly from a checkout rather than
requiring an installation step.  Pytest otherwise places only the tests
directory on ``sys.path`` when invoked through some environment wrappers.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
