"""Dependency-light checks for compiler/source-derived directive domains."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from LLM_branch.common import directive_domains


ROOT = Path(__file__).resolve().parents[2]


class SourceDirectiveDomainTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        kernel = self.root / "example-kernel"
        kernel.mkdir()
        (kernel / "kernel_info.txt").write_text(
            "example\nL1,loop,63\nL2,array,buffer,1,100,2,4\n",
            encoding="utf-8",
        )
        self.source = "\n".join((
            "auto{_PIPE_L1} = ?",
            "auto{_UNROLL_L1} = ?",
            "auto{_ARRAY_T_L2} = ?",
            "auto{_ARRAY_F_L2} = ?",
            "auto{_ARRAY_D_L2} = ?",
        ))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_source_domains_include_noop_and_source_trip_count(self):
        domains = directive_domains.source_site_domains(
            "example-kernel", self.source, self.root
        )
        self.assertEqual(domains["AUTO{_PIPE_L1}"], ["0", "1"])
        self.assertEqual(
            domains["AUTO{_UNROLL_L1}"], ["0", "2", "4", "8", "16", "32", "63"]
        )
        self.assertEqual(domains["AUTO{_ARRAY_D_L2}"], ["0", "1", "2"])
        self.assertIn("none", domains["AUTO{_ARRAY_T_L2}"])
        self.assertIn("128", domains["AUTO{_ARRAY_F_L2}"])

    def test_normalized_aliases_resolve_public_metadata(self):
        domains = directive_domains.source_site_domains(
            "example_kernel", self.source, self.root
        )
        self.assertIn("AUTO{_UNROLL_L1}", domains)

    def test_mixed_separator_aliases_resolve_public_metadata(self):
        mixed = self.root / "rodinia-lavamd_0-baseline"
        mixed.mkdir()
        (mixed / "kernel_info.txt").write_text("mixed\nL1,loop,8\n")
        domains = directive_domains.source_site_domains(
            "rodinia_lavamd_0_baseline", "auto{_UNROLL_L1} = ?", self.root
        )
        self.assertIn("8", domains["AUTO{_UNROLL_L1}"])

    def test_registry_does_not_touch_unlisted_test_kernel(self):
        registry = directive_domains.build_source_domain_registry(
            [{"kernel_name": "example-kernel", "input": self.source}], self.root
        )
        self.assertEqual(set(registry), {"example_kernel"})

    def test_inconsistent_kernel_source_is_rejected(self):
        rows = [
            {"kernel_name": "example-kernel", "input": self.source},
            {"kernel_name": "example_kernel", "input": self.source + "\n"},
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent source text"):
            directive_domains.build_source_domain_registry(rows, self.root)

    def test_directive_kind_must_match_source_action(self):
        with self.assertRaisesRegex(ValueError, "Loop directive"):
            directive_domains.source_site_domains(
                "example-kernel", "auto{_UNROLL_L2} = ?", self.root
            )

    def test_raw_labeled_source_receives_placeholders_automatically(self):
        source_path = self.root / "example.c"
        source_path.write_text("L1: for (;;) {}\nL2: int buffer[100][4];\n")
        module = ModuleType("GNN_branch.insert_placeholders")
        module.insert_placeholders = lambda path, allowed_labels: [self.source]
        with patch.dict("sys.modules", {"GNN_branch.insert_placeholders": module}):
            template = directive_domains.prepare_source_template(
                "example-kernel", source_path, self.root
            )
        self.assertEqual(template, self.source)

    def test_stage3_normalizes_loss_before_backward(self):
        source = (ROOT / "LLM_branch" / "train" / "train_DPO_harp_xattn.py").read_text()
        tree = ast.parse(source)
        methods = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "training_step"
        ]
        self.assertEqual(len(methods), 1)
        method = methods[0]
        division = next(
            node.lineno for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Div)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "loss"
        )
        backward = next(
            node.lineno for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
        )
        self.assertLess(division, backward)


if __name__ == "__main__":
    unittest.main()
