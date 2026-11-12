"""Clang-AST unit tests for GNN_branch/deterministic_analyzer.py.

Install in tests/test_deterministic_analyzer.py, or run directly next to
script (adjust its path if needed). Requires Clang, no GPU or Vitis.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent / "deterministic_analyzer.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "GNN_branch" / "deterministic_analyzer.py"
_spec = importlib.util.spec_from_file_location("salt_analyzer_under_test", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


@unittest.skipUnless(shutil.which("clang"), "requires clang")
class SourceASTTests(unittest.TestCase):
    def analyze(self, text: str, suffix: str = ".c"):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / ("kernel" + suffix)
            p.write_text(text)
            data = p.read_bytes()
            sites, skipped = mod.discover(mod.clang_tree(p, "clang", []), data, "kernel")
            labeled, records = mod.annotate(data, sites)
            return sites, skipped, labeled.decode(), records

    def test_c_macro_trip_counts_and_array_dimensions(self):
        sites, skipped, source, records = self.analyze("""\
#define N 32
void kernel(float *out) {
  float bank[4][8];
  for (int i=0; i<N; ++i) {
    out[i] = 0;
  }
  for (int j=1; j<=7; j+=2) {
    out[j] = 0;
  }
}
""")
        self.assertEqual(skipped, [])
        self.assertEqual([r["id"] for r in records], ["L1", "L2", "L3"])
        self.assertEqual([r["kind"] for r in records], ["array", "loop", "loop"])
        self.assertEqual(records[0]["dimensions"], [4, 8])
        self.assertEqual([r["trip_count"] for r in records[1:]], [32, 4])
        self.assertIn("/*L1:*/ float bank[4][8];", source)
        self.assertIn("/*L2:*/ for", source)

    def test_cpp_example(self):
        _, skipped, source, records = self.analyze("""\
void kernel(float *o) {
  int a[4];
  for (int i=0; i<12; i++) {
    o[i] = i;
  }
}
""", ".cpp")
        self.assertEqual(skipped, [])
        self.assertEqual(len(records), 2)
        self.assertIn("/*L1:*/ int a[4];", source)

    def test_unsupported_dynamic_and_inline_loops_reported(self):
        _, skipped, source, records = self.analyze("""\
void kernel(int *a, int n) {
  int bank[8];
  for (int i=0; i<n; ++i) {
    a[i] = 0;
  }
  for (int j=0; j<8; ++j) { a[j] = 0; }
}
""")
        self.assertEqual(len(records), 1)
        self.assertEqual(len(skipped), 2)
        self.assertTrue(any("nonconstant-loop-bound" in x["reason"] for x in skipped))
        self.assertTrue(any("requires-for-header" in x["reason"] for x in skipped))
        self.assertIn("/*L1:*/ int bank[8];", source)

    def test_reproducible_identity(self):
        sample = """void kernel(int *a) {
  for (int i=0; i<8; ++i) {
    a[i]++;
  }
}
"""
        one = self.analyze(sample)
        two = self.analyze(sample)
        self.assertEqual(one[2], two[2])
        self.assertEqual(one[3], two[3])


if __name__ == "__main__":
    unittest.main()
