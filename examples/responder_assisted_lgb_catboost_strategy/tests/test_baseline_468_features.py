from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from temporal_features import TemporalFeatureBuilder, temporal_column_names

try:
    import lightgbm  # noqa: F401
except ModuleNotFoundError:
    lightgbm_stub = types.ModuleType("lightgbm")
    lightgbm_stub.Sequence = object
    sys.modules["lightgbm"] = lightgbm_stub

from train import DEFAULT_RESPONDERS, target_experiment_spec


class Baseline468FeatureTests(unittest.TestCase):
    def test_plan_has_exact_468_base_columns(self):
        payload = json.loads(
            (STRATEGY_DIR / "baseline_468_feature_plan.json").read_text(
                encoding="utf-8"
            )
        )
        history_features = payload["history_features"]
        recipes = payload["recipes"]

        self.assertEqual(len(history_features), 48)
        self.assertEqual(len(set(history_features)), 48)
        self.assertEqual(list(recipes), history_features)
        self.assertTrue(
            all(value == ["lag1", "diff1", "rmean5"] for value in recipes.values())
        )
        self.assertEqual(len(temporal_column_names(history_features, recipes)), 144)
        self.assertEqual(323 + 144 + 1, payload["source_model_feature_count"])
        self.assertEqual(323 + 144 + 1 + 4, 472)

        metadata = {
            "feature_columns": [f"feature_{index:03d}" for index in range(323)],
            "temporal_feature_columns": temporal_column_names(
                history_features, recipes
            ),
        }
        spec = target_experiment_spec(
            metadata, "LGB468_C4", list(DEFAULT_RESPONDERS)
        )
        self.assertEqual(len(spec["base_indices"]), 468)
        self.assertEqual(len(spec["responder_indices"]), 4)
        self.assertEqual(len(spec["feature_names"]), 472)

    def test_history_transforms_match_baseline_cold_start_and_rolling_mean(self):
        recipes = {"feature_000": ["lag1", "diff1", "rmean5"]}
        builder = TemporalFeatureBuilder(
            1, feature_names=["feature_000"], recipes=recipes
        )
        values = np.arange(1, 7, dtype=np.float32).reshape(-1, 1)
        actual = builder.transform(np.ones(6, dtype=np.int64), values)
        expected = np.asarray(
            [
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.5],
                [2.0, 1.0, 2.0],
                [3.0, 1.0, 2.5],
                [4.0, 1.0, 3.0],
                [5.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_history_is_independent_per_asset_and_cleans_missing_values(self):
        recipes = {"feature_000": ["lag1", "diff1", "rmean5"]}
        builder = TemporalFeatureBuilder(
            1, feature_names=["feature_000"], recipes=recipes
        )
        asset_ids = np.asarray([1, 2, 1, 1, 2, 1], dtype=np.int64)
        values = np.asarray([[1.0], [10.0], [3.0], [np.nan], [14.0], [7.0]])
        actual = builder.transform(asset_ids, values)
        expected = np.asarray(
            [
                [0.0, 1.0, 1.0],
                [0.0, 10.0, 10.0],
                [1.0, 2.0, 2.0],
                [3.0, -3.0, 4.0 / 3.0],
                [10.0, 4.0, 12.0],
                [0.0, 7.0, 2.75],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
