import ast
from pathlib import Path
import unittest
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


TRAIN_SOURCE = (
    Path(__file__).resolve().parents[1] / "train_GNN.py"
).read_text(encoding="utf-8")
DATA_SOURCE = (
    Path(__file__).resolve().parents[1] / "mlir_data.py"
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
        "UNKNOWN_CATEGORY": "<unknown>",
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "mlir_data.py", "exec"),
        namespace,
    )
    return namespace


class StageBHelperTests(unittest.TestCase):
    def test_rank_checkpoint_requires_every_absolute_baseline(self):
        update = _load_function(
            "should_update_qualified_rank", {"np": np}
        )
        self.assertTrue(update(
            {"perf": 0.8, "area": 0.9}, 0.2, 0.1, 1e-4
        ))
        self.assertFalse(update(
            {"perf": 1.0, "area": 0.1}, 0.9, 0.1, 1e-4
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9}, 0.1, 0.1, 1e-4
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
