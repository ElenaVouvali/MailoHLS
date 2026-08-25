import ast
import __future__
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.preprocessing import OneHotEncoder


TRAIN_SOURCE = (
    Path(__file__).resolve().parents[1] / "train_GNN.py"
).read_text(encoding="utf-8")
DATA_SOURCE = (
    Path(__file__).resolve().parents[1] / "mlir_data.py"
).read_text(encoding="utf-8")
MODEL_SOURCE = (
    Path(__file__).resolve().parents[1] / "model.py"
).read_text(encoding="utf-8")
UTILS_SOURCE = (
    Path(__file__).resolve().parents[1] / "utils.py"
).read_text(encoding="utf-8")


def _load_function(name, globals_dict):
    tree = ast.parse(TRAIN_SOURCE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(globals_dict)
    exec(compile(module, "train_GNN.py", "exec"), namespace)
    return namespace[name]


def _load_data_functions(names):
    tree = ast.parse(DATA_SOURCE)
    nodes = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    namespace = {
        "np": np,
        "OneHotEncoder": OneHotEncoder,
        "Iterable": Iterable,
        "Sequence": Sequence,
        "Mapping": Mapping,
        "Any": Any,
        "UNKNOWN_CATEGORY": "<unknown>",
        "TARGET_DEVICE_CAPACITIES": {
            "xczu7ev-ffvc1156-2-e": (624, 1728, 460800, 230400),
        },
        "QOR_REFERENCE_DEVICE": "xczu7ev-ffvc1156-2-e",
    }
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            "mlir_data.py",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def _load_model_function(name, globals_dict):
    tree = ast.parse(MODEL_SOURCE)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = dict(globals_dict)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "model.py", "exec"),
        namespace,
    )
    return namespace[name]


def _load_utils_function(name, globals_dict):
    tree = ast.parse(UTILS_SOURCE)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "utils.py", "exec"), namespace)
    return namespace[name]


class StageBHelperTests(unittest.TestCase):
    def test_shared_initialization_hash_excludes_resource_state(self):
        hash_state = _load_utils_function("hash_state_dict", {
            "hashlib": __import__("hashlib"),
            "torch": torch,
        })
        common = {"encoder.weight": torch.arange(6.0).reshape(2, 3)}
        r0 = dict(common)
        r1 = {
            **common,
            "resource_heads.bram.weight": torch.randn(1, 3),
            "resource_mean": torch.randn(4),
            "resource_std": torch.randn(4),
        }
        kwargs = {
            "exclude_prefixes": ("resource_heads.",),
            "exclude_names": ("resource_mean", "resource_std"),
        }
        self.assertEqual(hash_state(r0, **kwargs), hash_state(r1, **kwargs))
        changed = {"encoder.weight": common["encoder.weight"] + 1}
        self.assertNotEqual(
            hash_state(r0, **kwargs), hash_state(changed, **kwargs)
        )

    def test_shared_initialization_hash_accepts_scalar_state(self):
        hash_state = _load_utils_function("hash_state_dict", {
            "hashlib": __import__("hashlib"),
            "torch": torch,
        })
        first = hash_state({"gate": torch.tensor(0.0)})
        second = hash_state({"gate": torch.tensor(0.0)})
        changed = hash_state({"gate": torch.tensor(1.0)})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_resource_training_diagnostics_capture_variation_and_baseline(self):
        diagnose = _load_function("resource_training_diagnostics", {
            "np": np,
            "torch": torch,
            "defaultdict": __import__("collections").defaultdict,
            "RESOURCE_NAMES": ("bram", "dsp", "ff", "lut"),
            "_sample_kernel": lambda sample: sample.kernel,
            "json": __import__("json"),
            "saver": SimpleNamespace(log_info=lambda message: None),
        })
        train = [
            SimpleNamespace(kernel="a", resource_util=torch.tensor([[0., .2, .1, .3]])),
            SimpleNamespace(kernel="a", resource_util=torch.tensor([[0., .2, .3, .5]])),
            SimpleNamespace(kernel="b", resource_util=torch.tensor([[.4, .2, .2, .4]])),
            SimpleNamespace(kernel="b", resource_util=torch.tensor([[.4, .2, .2, .4]])),
        ]
        stats = diagnose(train)
        self.assertEqual(stats["bram"]["fraction_zero"], 0.5)
        self.assertEqual(
            stats["dsp"]["kernels_with_two_or_more_distinct_values"], 0
        )
        self.assertEqual(
            stats["ff"]["kernels_with_two_or_more_distinct_values"], 1
        )
        self.assertEqual(
            stats["dsp"]["train_mean_baseline_macro_kendall_tau"], 0.0
        )
        self.assertGreater(stats["ff"]["train_mean_baseline_mae"], 0.0)

    def test_paired_comparison_requires_all_three_hashes(self):
        require = _load_function("require_paired_comparison_contract", {
            "json": __import__("json"),
        })
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump({
                "resolved_flags": {
                    "resource_aux_weight": 0.0,
                    "rank_aux_weight": 0.0,
                },
                "dataset_manifest_sha256": "dataset",
                "split_sha256": "split",
                "shared_initialization_sha256": "init",
            }, handle)
            handle.flush()
            require(
                handle.name,
                dataset_manifest_sha256="dataset",
                split_sha256="split",
                shared_initialization_sha256="init",
            )
            with self.assertRaisesRegex(RuntimeError, "Invalid paired"):
                require(
                    handle.name,
                    dataset_manifest_sha256="dataset",
                    split_sha256="different",
                    shared_initialization_sha256="init",
                )

    def test_resource_csv_order_and_percentage_conversion(self):
        functions = _load_data_functions({
            "_as_float", "resource_utilization_from_csv_row"
        })
        values = functions["resource_utilization_from_csv_row"]({
            "BRAM_Utilization_percentage": "10",
            "DSP_Utilization_percentage": "20",
            "FF_Utilization_percentage": "30",
            "LUT_Utilization_percentage": "40",
        })
        self.assertEqual(values, [0.1, 0.2, 0.3, 0.4])

    def test_resource_counts_are_preferred_without_flooring_genuine_zeros(self):
        functions = _load_data_functions({
            "_as_float", "resource_utilization_from_csv_row"
        })
        values = functions["resource_utilization_from_csv_row"]({
            "Device": "xczu7ev-ffvc1156-2-e",
            "BRAM_18K_Used": "0",
            "DSP_Used": "17",
            "FF_Used": "4608",
            "LUT_Used": "2304",
            "BRAM_Utilization_percentage": "1",
            "DSP_Utilization_percentage": "1",
            "FF_Utilization_percentage": "1",
            "LUT_Utilization_percentage": "1",
        })
        np.testing.assert_allclose(values, [0.0, 17 / 1728, 0.01, 0.01])

    def test_resource_statistics_are_log1p_and_train_only(self):
        fit = _load_function("fit_resource_statistics", {
            "np": np,
            "torch": torch,
            "Counter": __import__("collections").Counter,
            "RESOURCE_NAMES": ("bram", "dsp", "ff", "lut"),
            "FLAGS": SimpleNamespace(kernel_balanced_loss=True),
            "_sample_kernel": lambda sample: sample.kernel,
            "saver": SimpleNamespace(log_info=lambda message: None),
        })
        train = [
            SimpleNamespace(kernel="a", resource_util=torch.tensor([[0., 1., 2., 3.]])),
            SimpleNamespace(kernel="a", resource_util=torch.tensor([[2., 3., 4., 5.]])),
            SimpleNamespace(kernel="b", resource_util=torch.tensor([[4., 5., 6., 7.]])),
        ]
        stats = fit(train)
        matrix = np.log1p(np.asarray([
            [0., 1., 2., 3.], [2., 3., 4., 5.], [4., 5., 6., 7.]
        ]))
        expected_mean = np.sum(matrix * np.asarray([0.25, 0.25, 0.5])[:, None], axis=0)
        np.testing.assert_allclose(stats["mean"], expected_mean)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["transform"], "log1p")

    def test_resource_weight_controls_head_architecture_and_roundtrip(self):
        maybe_fit = _load_function("maybe_fit_resource_statistics", {
            "fit_resource_statistics": lambda dataset: {"dataset": dataset},
        })
        self.assertIsNone(maybe_fit("train", 0.0))
        stats = maybe_fit("train", 0.1)
        self.assertEqual(stats, {"dataset": "train"})

        class TinyMLP(nn.Module):
            def __init__(self, in_dim, out_dim, **kwargs):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
            def forward(self, value):
                return self.linear(value)

        make_heads = _load_model_function("make_resource_heads", {
            "nn": nn,
            "MLP": TinyMLP,
            "FLAGS": SimpleNamespace(activation="elu"),
            "RESOURCE_NAMES": ("bram", "dsp", "ff", "lut"),
        })

        class ResourceModel(nn.Module):
            def __init__(self, resource_stats):
                super().__init__()
                self.resource_heads = make_heads(16, 32, resource_stats)

        disabled = ResourceModel(None)
        self.assertIsNone(disabled.resource_heads)
        disabled_reload = ResourceModel(None)
        disabled_reload.load_state_dict(disabled.state_dict(), strict=True)

        enabled = ResourceModel(stats)
        self.assertEqual(set(enabled.resource_heads), {"bram", "dsp", "ff", "lut"})
        enabled_reload = ResourceModel(stats)
        enabled_reload.load_state_dict(enabled.state_dict(), strict=True)

        check_contract = _load_function(
            "assert_resource_contract_matches_state_dict", {}
        )
        check_contract(
            {"model_init": {"resource_aux_heads": False}},
            disabled.state_dict(),
        )
        check_contract(
            {"model_init": {"resource_aux_heads": True}},
            enabled.state_dict(),
        )
        with self.assertRaisesRegex(RuntimeError, "contract/state mismatch"):
            check_contract(
                {"model_init": {"resource_aux_heads": False}},
                enabled.state_dict(),
            )

    def test_joint_resource_boundary_uses_max_constraint_margin(self):
        metrics_for = _load_function(
            "resource_feasibility_metrics", {"np": np}
        )
        actual = np.asarray([
            [0.50, 0.90, 0.10, 0.10],
            [0.49, 0.20, 0.20, 0.20],
            [0.80, 0.80, 0.80, 0.80],
        ])
        predicted = np.asarray([
            [0.40, 0.40, 0.40, 0.40],
            [0.40, 0.40, 0.40, 0.40],
            [0.90, 0.90, 0.90, 0.90],
        ])
        metrics = metrics_for(actual, predicted, np.full(4, 0.5), 0.02)
        np.testing.assert_array_equal(
            metrics["boundary_mask"], [False, True, False]
        )
        np.testing.assert_allclose(metrics["joint_margin"], [0.4, -0.01, 0.3])
        self.assertAlmostEqual(metrics["false_feasible_fdr"], 0.5)
        self.assertAlmostEqual(metrics["false_feasible_rate"], 1.0 / 3.0)

    def test_per_kernel_target_ratios_detect_local_failure(self):
        ratios_for = _load_function(
            "compute_per_kernel_target_baseline_ratios",
            {
                "np": np,
                "FLAGS": SimpleNamespace(
                    target_mode="reference_delta",
                    standardize_targets=False,
                    loss="mse",
                ),
                "_point_loss_from_error": lambda error: np.square(error),
            },
        )
        ratios = ratios_for(
            {
                "perf": {
                    "kernel": ["a", "a", "b", "b"],
                    "pred": [
                        (101.0, 100.0), (102.0, 100.0),
                        (51.0, 53.0), (51.0, 53.0),
                    ],
                    "baseline_log2": [100.0, 100.0, 50.0, 50.0],
                    "actual_delta_log2": [1.0, 2.0, 1.0, 1.0],
                    "predicted_delta_log2": [0.0, 0.0, 3.0, 3.0],
                }
            },
            {"perf": {"mean": 123.0, "std": 1.0}},
        )
        self.assertTrue(np.isclose(ratios["a/perf"], 1.0))
        self.assertTrue(np.isclose(ratios["b/perf"], 4.0))

    def test_delta_pair_regression_preserves_zero_error_without_margin_inflation(self):
        pair_loss = _load_model_function(
            "within_kernel_delta_pair_loss", {"torch": torch, "F": F}
        )
        measured = torch.tensor([[0.0], [1.0], [2.0], [0.0], [0.01]])
        kernels = ["a", "a", "a", "b", "b"]
        exact = pair_loss(
            measured, measured, kernels,
            tie_target=measured, tie_epsilon=0.05, beta=0.5,
        )
        exaggerated = pair_loss(
            measured * 2.0, measured, kernels,
            tie_target=measured, tie_epsilon=0.05, beta=0.5,
        )
        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(exaggerated), 0.0)

    def test_anchored_head_is_exactly_zero_for_neutral_change(self):
        anchored = _load_model_function(
            "anchored_head_response", {"torch": torch}
        )
        head = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
        )
        neutral = torch.randn(5, 4)
        response = anchored(head, neutral, neutral)
        torch.testing.assert_close(response, torch.zeros_like(response))

    def test_anchored_head_cancels_static_bias_but_keeps_context(self):
        anchored = _load_model_function(
            "anchored_head_response", {"torch": torch}
        )
        head = torch.nn.Linear(2, 1)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2.0, 3.0]]))
            head.bias.fill_(17.0)
        full = torch.tensor([[4.0, 5.0]])
        neutral = torch.tensor([[4.0, 0.0]])
        torch.testing.assert_close(
            anchored(head, full, neutral), torch.tensor([[15.0]])
        )

    def test_pairwise_delta_weight_waits_for_reference_calibration(self):
        schedule = _load_function("scheduled_pairwise_delta_weight", {
            "FLAGS": SimpleNamespace(
                pairwise_delta_weight=0.05,
                pairwise_delta_start_epoch=3,
                pairwise_delta_ramp_epochs=2,
            ),
        })
        np.testing.assert_allclose(
            [schedule(epoch) for epoch in range(6)],
            [0.0, 0.0, 0.0, 0.025, 0.05, 0.05],
        )

    def test_exact_stage1_budget_bank_controls_resource_feasibility(self):
        flags = SimpleNamespace(
            resource_boundary_tolerance=0.02,
            target_device="xczu7ev-ffvc1156-2-e",
            clock_period_ns=10.0,
            random_seed=123,
        )
        namespace = {
            "np": np,
            "json": json,
            "Path": Path,
            "hashlib": __import__("hashlib"),
            "math": __import__("math"),
            "random": __import__("random"),
            "defaultdict": __import__("collections").defaultdict,
            "RESOURCE_NAMES": ("bram", "dsp", "ff", "lut"),
            "FLAGS": flags,
            "kendalltau": kendalltau,
            "saver": SimpleNamespace(log_info=lambda message: None),
        }
        for name in (
            "_resource_case_identity", "_load_stage1_resource_budget_bank",
            "_generated_stage1_style_budgets",
            "_feasibility_outcome_metrics", "resource_feasibility_metrics",
            "report_resource_metrics",
        ):
            namespace[name] = _load_function(name, namespace)
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "budgets.json"
            bank_path.write_text(json.dumps({
                "schema": "mailohls-stage1-validation-resource-budget-bank-v1",
                "resource_order": ["bram", "dsp", "ff", "lut"],
                "cases": [{
                    "kernel": "kernel-a",
                    "device": flags.target_device,
                    "clock_period_ns": 10.0,
                    "fractions": [0.5, 0.2, 0.8, 0.3],
                }],
            }))
            flags.resource_budget_bank = str(bank_path)
            diagnostics = {
                name: {"train_mean_baseline": 0.9}
                for name in namespace["RESOURCE_NAMES"]
            }
            report = namespace["report_resource_metrics"]([
                {
                    "kernel": "kernel-a", "device": flags.target_device,
                    "clock_period_ns": 10.0,
                    "actual": np.asarray([0.51, 0.10, 0.20, 0.10]),
                    "predicted": np.asarray([0.49, 0.10, 0.20, 0.10]),
                },
                {
                    "kernel": "kernel-a", "device": flags.target_device,
                    "clock_period_ns": 10.0,
                    "actual": np.asarray([0.49, 0.10, 0.20, 0.10]),
                    "predicted": np.asarray([0.49, 0.10, 0.20, 0.10]),
                },
            ], "validation", diagnostics)
            unseen = [{
                "kernel": "held-out-test-kernel", "device": flags.target_device,
                "clock_period_ns": 10.0,
                "actual": np.asarray([0.51, 0.10, 0.20, 0.10]),
                "predicted": np.asarray([0.49, 0.10, 0.20, 0.10]),
            }]
            with self.assertRaisesRegex(RuntimeError, "lacks GNN validation case"):
                namespace["report_resource_metrics"](
                    unseen, "validation", diagnostics
                )
            test_report = namespace["report_resource_metrics"](
                unseen, "test", diagnostics, allow_unseen_cases=True
            )
        self.assertEqual(report["budget_policy"], "exact_stage1_validation_bank")
        self.assertAlmostEqual(
            report["independent_budget_summary"]["false_feasible_fdr"], 0.5
        )
        self.assertAlmostEqual(
            report["independent_budget_summary"]["boundary_balanced_accuracy"], 0.5
        )
        self.assertTrue(report["all_resource_heads_beat_constant_baseline"])
        self.assertEqual(test_report["generated_unseen_case_count"], 1)
        self.assertEqual(
            test_report["budget_policy"],
            "deterministic_stage1_style_independent_budgets_for_unseen_test_cases",
        )

    def test_qualified_lexicographic_selection_rejects_uncalibrated_heads(self):
        flags = SimpleNamespace(
            max_kernel_zero_baseline_ratio=1.10,
            min_rank_tau=0.20,
        )
        qualify = _load_function("qualified_lexicographic_metrics", {
            "FLAGS": flags, "np": np,
        })
        report = {
            "resources": [
                {"resource": name, "mae_baseline_ratio": 0.8}
                for name in ("bram", "dsp", "ff", "lut")
            ],
            "independent_budget_summary": {
                "boundary_balanced_accuracy": 0.75,
                "false_feasible_fdr": 0.10,
            },
        }
        qualified = qualify(
            {"perf": 0.9, "area": 0.8},
            {"a/perf": 1.05, "a/area": 1.5},
            0.25,
            report,
        )
        self.assertTrue(qualified["qualified"])
        report["resources"][0]["mae_baseline_ratio"] = 1.01
        rejected = qualify(
            {"perf": 0.9, "area": 0.8},
            {"a/perf": 1.05},
            0.25,
            report,
        )
        self.assertFalse(rejected["qualified"])
        self.assertFalse(rejected["resource_qualified"])

    def test_rank_checkpoint_requires_every_absolute_baseline(self):
        update = _load_function(
            "should_update_qualified_rank", {"np": np}
        )
        self.assertTrue(update(
            {"perf": 0.8, "area": 0.9},
            {"kernel-a/perf": 0.9, "kernel-a/area": 1.1},
            0.2, 0.1, 1e-4, 0.0, 1.1,
        ))
        self.assertFalse(update(
            {"perf": 1.0, "area": 0.1},
            {"kernel-a/perf": 0.9},
            0.9, 0.1, 1e-4, 0.0, 1.1,
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9},
            {"kernel-a/perf": 0.9},
            0.1, 0.1, 1e-4, 0.0, 1.1,
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9},
            {"kernel-a/perf": 1.1001},
            0.9, 0.1, 1e-4, 0.0, 1.1,
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9},
            {"kernel-a/perf": 0.9},
            -0.01, -0.2, 1e-4, 0.0, 1.1,
        ))

    def test_embedding_rank_uses_aggregate_guardrails_and_tau(self):
        update = _load_function(
            "should_update_embedding_rank", {"np": np}
        )
        self.assertTrue(update(
            {"perf": 0.8, "area": 0.9},
            0.30, 0.20, 1e-4, 0.20,
        ))
        self.assertFalse(update(
            {"perf": 1.01, "area": 0.5},
            0.90, 0.20, 1e-4, 0.20,
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9},
            0.19, 0.10, 1e-4, 0.20,
        ))

    def test_macro_rank_is_equal_kernel_and_worst_target(self):
        score = _load_function(
            "compute_macro_ranking_score",
            {
                "np": np,
                "set_target_list": lambda: (["perf", "area"], {}),
            },
        )
        metrics = pd.DataFrame([
            {"target": "latency_ms", "aggregation": "kernel", "tau": 0.6},
            {"target": "latency_ms", "aggregation": "kernel", "tau": 0.2},
            {"target": "area_score", "aggregation": "kernel", "tau": 0.8},
            {"target": "area_score", "aggregation": "kernel", "tau": np.nan},
        ])
        # NaN means constant/no ranking skill. Both target means are 0.4.
        self.assertTrue(np.isclose(score(metrics), 0.4))

    def test_model_keeps_linear_final_outputs(self):
        utils_source = (
            Path(__file__).resolve().parents[1] / "utils.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(utils_source)
        mlp = next(
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "MLP"
        )
        forward = next(
            item
            for item in mlp.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        )
        final_branch = next(
            node for node in ast.walk(forward)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "layer"
        )
        self.assertFalse(any(
            isinstance(node, ast.Attribute)
            and node.attr == "activation"
            for statement in final_branch.body
            for node in ast.walk(statement)
        ))

    def test_unseen_category_uses_explicit_oov_column(self):
        functions = _load_data_functions({
            "_make_onehot_encoder",
            "_fit_onehot",
            "_encode_categorical_columns",
        })
        encoder = functions["_fit_onehot"](["known"])
        matrix = functions["_encode_categorical_columns"](
            [["never-seen"]], ["field"], {"field": encoder}
        )[0].toarray()
        categories = list(encoder.categories_[0])
        unknown_index = categories.index("<unknown>")
        self.assertEqual(float(matrix.sum()), 1.0)
        self.assertEqual(float(matrix[0, unknown_index]), 1.0)


if __name__ == "__main__":
    unittest.main()
