import ast
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class StageBHelperTests(unittest.TestCase):
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
                    "pred": [(1.0, 0.0), (2.0, 0.0), (1.0, 3.0), (1.0, 3.0)],
                }
            },
            {"perf": {"mean": 123.0, "std": 1.0}},
        )
        self.assertTrue(np.isclose(ratios["a/perf"], 1.0))
        self.assertTrue(np.isclose(ratios["b/perf"], 4.0))

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
