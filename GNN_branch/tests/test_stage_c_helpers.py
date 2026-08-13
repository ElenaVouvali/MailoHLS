import ast
import csv
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_reference_functions():
    source = (ROOT / "reference_delta.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_split_names", "_source_tree_sha256", "load_reference_baselines"
        }
    ]
    namespace = {
        "Path": Path,
        "csv": csv,
        "math": math,
        "hashlib": __import__("hashlib"),
        "REQUIRED_COLUMNS": {
            "kernel", "status", "source_sha256", "toolchain_id", "device",
            "clock_period_ns", "baseline_latency_ms", "baseline_area_score",
        },
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), "reference_delta.py", "exec"), namespace)
    return namespace


class StageCHelperTests(unittest.TestCase):
    def make_manifest(self, path, kernels):
        fields = [
            "kernel", "status", "source_sha256", "toolchain_id", "device",
            "clock_period_ns", "baseline_latency_ms", "baseline_area_score",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for kernel in kernels:
                writer.writerow({
                    "kernel": kernel,
                    "status": "success",
                    "source_sha256": "a" * 64,
                    "toolchain_id": "Vitis HLS 2021.1",
                    "device": "xczu7ev-ffvc1156-2-e",
                    "clock_period_ns": "10.0",
                    "baseline_latency_ms": "8.0",
                    "baseline_area_score": "4.0",
                })

    def test_manifest_converts_physical_reference_to_log2(self):
        functions = load_reference_functions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self.make_manifest(path, ["train-kernel"])
            result = functions["load_reference_baselines"](
                path,
                required_kernels={"train-kernel"},
                expected_device="xczu7ev-ffvc1156-2-e",
                expected_clock_period_ns=10.0,
                expected_toolchain_version="2021.1",
                epsilon=1e-6,
                verify_source_hashes=False,
            )
        self.assertTrue(np.isclose(result["train-kernel"]["perf"], math.log2(8.000001)))
        self.assertTrue(np.isclose(result["train-kernel"]["area"], math.log2(4.000001)))

    def test_manifest_rejects_locked_test_measurement(self):
        functions = load_reference_functions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self.make_manifest(path, ["train-kernel", "test-kernel"])
            with self.assertRaisesRegex(RuntimeError, "Held-out test kernel"):
                functions["load_reference_baselines"](
                    path,
                    required_kernels={"train-kernel"},
                    forbidden_kernels={"test-kernel"},
                    expected_device="xczu7ev-ffvc1156-2-e",
                    expected_clock_period_ns=10.0,
                    expected_toolchain_version="2021.1",
                    epsilon=1e-6,
                    verify_source_hashes=False,
                )

    def test_smooth_l1_matches_definition(self):
        source = (ROOT / "train_GNN.py").read_text(encoding="utf-8")
        node = next(
            item for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef)
            and item.name == "_point_loss_from_error"
        )
        namespace = {
            "np": np,
            "FLAGS": SimpleNamespace(loss="smooth_l1", smooth_l1_beta=0.5),
        }
        exec(compile(ast.Module([node], type_ignores=[]), "train_GNN.py", "exec"), namespace)
        values = namespace["_point_loss_from_error"](np.array([0.25, 1.0]))
        self.assertTrue(np.allclose(values, [0.0625, 0.75]))

    def test_vitis_xml_is_converted_to_mailohls_units(self):
        spec = importlib.util.spec_from_file_location(
            "neutral_generator", ROOT / "generate_neutral_baselines.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        xml = """<Report>
          <PerformanceEstimates><SummaryOfOverallLatency>
            <Worst-caseLatency>1000</Worst-caseLatency>
          </SummaryOfOverallLatency></PerformanceEstimates>
          <AreaEstimates>
            <Resources><BRAM_18K>0</BRAM_18K><DSP>20</DSP><FF>100</FF><LUT>200</LUT></Resources>
            <AvailableResources><BRAM_18K>100</BRAM_18K><DSP>100</DSP><FF>10000</FF><LUT>10000</LUT></AvailableResources>
          </AreaEstimates>
        </Report>"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.xml"
            report.write_text(xml, encoding="utf-8")
            values = module.parse_csynth_report(report, 10.0)
        self.assertTrue(np.isclose(values["baseline_latency_ms"], 0.01))
        # BRAM is floored from 0% to 1%, matching data_preprocess.py.
        self.assertTrue(np.isclose(values["baseline_area_score"], 6.0))


if __name__ == "__main__":
    unittest.main()
