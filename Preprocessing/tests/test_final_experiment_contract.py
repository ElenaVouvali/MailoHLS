"""Dependency-light regression checks for the final shared experiment contract."""

from __future__ import annotations

import ast
import json
import math
import runpy
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from Preprocessing.build_family_split import build_family_split
from Preprocessing.data_preprocess import ActionDefinition, preprocess_kernel


ROOT = Path(__file__).resolve().parents[2]


def load_gnn_helper(name: str, namespace: dict):
    source = ROOT / "GNN_branch" / "mlir_data.py"
    node = next(
        item for item in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    environment = dict(namespace)
    exec(compile(ast.Module([node], type_ignores=[]), str(source), "exec"), environment)
    return environment[name]


def load_grouped_sampler():
    source = ROOT / "GNN_branch" / "train_GNN.py"
    node = next(
        item for item in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(item, ast.ClassDef) and item.name == "KernelGroupedBatchSampler"
    )
    environment = {
        "Sampler": object,
        "defaultdict": defaultdict,
        "np": np,
        "_as_int_seed": int,
        "_sample_kernel": lambda sample: sample.kernel,
        "_sample_target_group": lambda sample: sample.target_group,
    }
    exec(compile(ast.Module([node], type_ignores=[]), str(source), "exec"), environment)
    return environment["KernelGroupedBatchSampler"]


class FinalExperimentContractTests(unittest.TestCase):
    def test_explicit_holdouts_and_area_floor_use_training_rows_only(self):
        rows = [
            {"kernel_name": "machsuite-gemm-blocked", "area": 2.0},
            {"kernel_name": "machsuite-gemm-blocked", "area": 0.0},
            {"kernel_name": "machsuite-viterbi", "area": 0.02},
            {"kernel_name": "serrano-kalman-filter", "area": 0.001},
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            split = build_family_split(
                dataset,
                val_kernels=("machsuite-viterbi",),
                test_kernels=("serrano-kalman-filter",),
            )
        self.assertEqual(split["train_kernels"], ["machsuite-gemm-blocked"])
        self.assertEqual(split["test_kernels"], ["serrano-kalman-filter"])
        self.assertEqual(split["effective_area_floor"], 1.0)

    def test_split_rejects_partial_holdout_family(self):
        rows = [
            {"kernel_name": "rodinia_pathfinder_0_baseline_0", "area": 2.0},
            {"kernel_name": "rodinia_pathfinder_4_doublebuffer_0", "area": 3.0},
            {"kernel_name": "machsuite-gemm-blocked", "area": 1.0},
            {"kernel_name": "serrano-kalman-filter", "area": 4.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "splits a kernel family"):
                build_family_split(
                    dataset,
                    val_kernels=("rodinia_pathfinder_0_baseline_0",),
                    test_kernels=("serrano-kalman-filter",),
                )

    def test_gnn_retains_zero_area_without_logarithmic_singularity(self):
        flags = SimpleNamespace(
            invalid=False, min_allowed_latency=0.0, epsilon=1e-6,
            effective_area_floor=0.0625, norm_method="log2",
        )
        keep = load_gnn_helper("_keep_result", {"math": math, "FLAGS": flags})
        normalize = load_gnn_helper(
            "normalize_targets", {"math": math, "FLAGS": flags}
        )
        point = SimpleNamespace(perf=2.0, area=0.0)
        self.assertTrue(keep(point))
        self.assertAlmostEqual(
            normalize(2.0, 0.0, 2.0)[1], math.log2(0.0625 + 1e-6)
        )

    def test_target_condition_distinguishes_devices_and_clocks(self):
        function = load_gnn_helper(
            "target_condition_vector",
            {
                "math": math,
                "TARGET_DEVICE_CAPACITIES": {
                    "device-a": (624, 1728, 460800, 230400),
                    "device-b": (4320, 6840, 2364480, 1182240),
                },
            },
        )
        device_a = function("device-a", 10.0)
        device_b = function("device-b", 10.0)
        faster = function("device-a", 5.0)
        self.assertEqual(len(device_a), 5)
        self.assertNotEqual(device_a[:4], device_b[:4])
        self.assertEqual(device_a[:4], faster[:4])
        self.assertEqual(faster[4], -1.0)

    def test_gnn_preprocessing_keeps_all_targets_and_zero_utilization(self):
        rows = []
        for device, clock in (
            ("xczu7ev-ffvc1156-2-e", 5.0),
            ("xczu7ev-ffvc1156-2-e", 10.0),
            ("xcu200-fsgd2104-2-e", 10.0),
        ):
            rows.append({
                "Version": "v1", "Device": device, "Clock_Period_nsec": clock,
                "Latency_msec": 1.0, "BRAM_Utilization_percentage": 0.0,
                "DSP_Utilization_percentage": 0.0, "FF_Utilization_percentage": 1.0,
                "LUT_Utilization_percentage": 1.0, "loop": "pipeline",
            })
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "kernel.csv"
            pd.DataFrame(rows).to_csv(table, index=False)
            with patch(
                "Preprocessing.data_preprocess.load_action_definitions",
                return_value={"loop": ActionDefinition("L1", "loop", trip_count=8)},
            ):
                frame, summary = preprocess_kernel(
                    table, "gnn", "xczu7ev-ffvc1156-2-e", 10.0, 0.1, 2.0,
                    all_targets=True,
                )
        self.assertEqual(summary["targets"], 3)
        self.assertTrue((frame["BRAM_Utilization_percentage"] == 0.0).all())

    def test_directive_tensors_are_reused_across_measured_targets(self):
        source = (ROOT / "GNN_branch" / "mlir_data.py").read_text(encoding="utf-8")
        self.assertIn('directive_to_index[directive_key] = directive_idx', source)
        self.assertIn('"directive_indices": torch.tensor(directive_indices', source)
        self.assertIn('][directive_idx].float()', source)

    def test_ranking_batches_never_mix_device_clock_targets(self):
        samples = [
            SimpleNamespace(kernel=kernel, target_group=f"{kernel}|{target}")
            for kernel in ("kernel-a", "kernel-b")
            for target in ("device-a|5", "device-b|10")
            for _ in range(3)
        ]
        sampler = load_grouped_sampler()(
            samples, kernels_per_batch=2, points_per_kernel=2,
            samples_per_kernel_per_epoch=4, seed=123,
        )
        for batch in sampler:
            for kernel in ("kernel-a", "kernel-b"):
                groups = {
                    samples[index].target_group
                    for index in batch if samples[index].kernel == kernel
                }
                self.assertEqual(len(groups), 1)

    def test_final_gnn_launcher_options_resolve_shared_split(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "split.json"
            split.write_text(json.dumps({
                "train_kernels": ["machsuite-gemm-blocked"],
                "val_kernels": ["machsuite-viterbi"],
                "test_kernels": ["serrano-kalman-filter"],
                "effective_area_floor": 0.0625,
            }), encoding="utf-8")
            torch_stub = ModuleType("torch")
            torch_stub.cuda = SimpleNamespace(is_available=lambda: True)
            utils_stub = ModuleType("utils")
            utils_stub.get_user = lambda: "test"
            utils_stub.get_host = lambda: "test"
            utils_stub.get_root_path = lambda: str(ROOT)
            arguments = [
                "config.py", "--dataset", "mlir", "--subtask", "train",
                "--device", "cuda:0", "--split_json", str(split),
                "--force_regen", "--multi_target_qor", "--target", "perf", "area",
                "--target_mode", "absolute", "--checkpoint_objective", "absolute",
                "--D", "64", "--num_layers", "4",
                "--graph_attention_heads", "4", "--graph_residual_beta",
                "--graph_layer_norm", "--dropout", "0.10", "--standardize_targets",
                "--kernel_balanced_loss", "--kernel_grouped_sampling", "--batch_size", "8",
                "--kernels_per_batch", "4", "--points_per_kernel", "2",
                "--samples_per_kernel_per_epoch", "64", "--rank_aux_weight", "0.05",
                "--rank_temperature", "0.5", "--rank_tie_epsilon", "0.02",
                "--resource_aux_weight", "0.10", "--loss", "smooth_l1",
                "--smooth_l1_beta", "0.5", "--lr", "0.0005", "--weight_decay", "0.0001",
                "--scheduler", "plateau", "--warmup_epochs", "5", "--plateau_patience", "5",
                "--early_stopping_patience", "20", "--epoch_num", "150", "--num_workers", "2",
                "--eval_num_workers", "0", "--random_seed", "123",
                "--experiment_name", "gnn_final_multitarget_s123",
            ]
            with patch.dict(sys.modules, {"torch": torch_stub, "utils": utils_stub}):
                with patch.object(sys, "argv", arguments):
                    flags = runpy.run_path(str(ROOT / "GNN_branch" / "config.py"))["FLAGS"]
        self.assertEqual(flags.val_kernels, "machsuite-viterbi")
        self.assertEqual(flags.test_kernels, "serrano-kalman-filter")
        self.assertEqual(flags.effective_area_floor, 0.0625)
        self.assertEqual(flags.graph_attention_heads, 4)


if __name__ == "__main__":
    unittest.main()
