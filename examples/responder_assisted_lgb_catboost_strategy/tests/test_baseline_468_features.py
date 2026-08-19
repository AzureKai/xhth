from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from temporal_features import (
    TEMPORAL_SUFFIXES,
    TemporalFeatureBuilder,
    temporal_column_names,
)

try:
    import lightgbm  # noqa: F401
except ModuleNotFoundError:
    lightgbm_stub = types.ModuleType("lightgbm")
    lightgbm_stub.Sequence = object
    sys.modules["lightgbm"] = lightgbm_stub

from train import (
    DEFAULT_TEMPORAL_PLAN_PATH,
    DEFAULT_RESPONDERS,
    ShardSequence,
    TARGET_PARAM_PROFILES,
    TIER_RESPONDERS,
    build_cold_start_prefix,
    clipping_diagnostics,
    matrix_for_segments,
    low_risk_lgb_params,
    registered_selection_candidates,
    select_temporal_plan,
    session_patch,
    target_experiment_spec,
    temporal_session_warmup,
)


class Baseline468FeatureTests(unittest.TestCase):
    def test_fixed_smoothed_profile_has_stronger_regularization(self):
        self.assertEqual(
            list(TARGET_PARAM_PROFILES),
            ["smoothed"],
        )
        args = types.SimpleNamespace(seed=2026, threads=8)
        params = low_risk_lgb_params(args, seed=2027, profile="smoothed")
        self.assertEqual(params["num_leaves"], 47)
        self.assertEqual(params["max_depth"], 10)
        self.assertEqual(params["min_data_in_leaf"], 5000)
        self.assertEqual(params["feature_fraction"], 0.8)
        self.assertEqual(params["feature_fraction_bynode"], 0.8)
        self.assertEqual(params["lambda_l1"], 2.0)
        self.assertEqual(params["lambda_l2"], 30.0)
        self.assertEqual(params["path_smooth"], 150.0)
        self.assertEqual(params["min_gain_to_split"], 0.01)
        self.assertEqual(params["data_random_seed"], 2026)
        self.assertEqual(params["bagging_seed"], 2027)
        self.assertEqual(params["histogram_pool_size"], 8192.0)
        self.assertNotIn("regularization_rank", params)

    def test_compact_plan_and_responder_tier_are_production_defaults(self):
        self.assertEqual(
            DEFAULT_TEMPORAL_PLAN_PATH.name,
            "long_horizon_468_feature_plan.json",
        )
        self.assertNotIn("responder_22", TIER_RESPONDERS)
        self.assertNotIn("responder_23", TIER_RESPONDERS)
        self.assertEqual(len(TIER_RESPONDERS), 10)
        self.assertEqual(
            DEFAULT_RESPONDERS,
            ["responder_03", "responder_02"],
        )

    def test_long_horizon_plan_has_exact_468_base_columns(self):
        payload = json.loads(
            (STRATEGY_DIR / "long_horizon_468_feature_plan.json").read_text(
                encoding="utf-8"
            )
        )
        history_features = payload["history_features"]
        recipes = payload["recipes"]

        self.assertEqual(len(history_features), 48)
        self.assertEqual(len(set(history_features)), 48)
        self.assertEqual(list(recipes), history_features)
        self.assertTrue(payload["exact_recipes"])
        self.assertTrue(all("rmean5" in value for value in recipes.values()))
        self.assertEqual(sum("lag1" in value for value in recipes.values()), 8)
        self.assertEqual(sum("diff1" in value for value in recipes.values()), 7)
        for transform in (
            "historical_zscore20", "minus_ema20", "rolling_std20"
        ):
            self.assertEqual(
                sum(transform in value for value in recipes.values()), 27
            )
        self.assertFalse(
            any("historical_zscore60" in value for value in recipes.values())
        )
        self.assertFalse(any("minus_ema60" in value for value in recipes.values()))
        self.assertFalse(any("rolling_std60" in value for value in recipes.values()))
        self.assertEqual(len(temporal_column_names(history_features, recipes)), 144)
        self.assertEqual(323 + 144 + 1, payload["source_model_feature_count"])
        self.assertEqual(323 + 144 + 1 + 2, 470)

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
        self.assertEqual(len(spec["responder_indices"]), 2)
        self.assertNotIn("responder_28", spec["responders"])
        self.assertNotIn("responder_29", spec["responders"])
        self.assertEqual(len(spec["feature_names"]), 470)

    def test_exact_plan_loader_does_not_add_60_step_transforms(self):
        features = [f"feature_{index:03d}" for index in range(323)]
        plan_path = STRATEGY_DIR / "long_horizon_468_feature_plan.json"
        selected, recipes = select_temporal_plan(
            features, 48, importance_path=None, plan_path=plan_path
        )
        self.assertEqual(len(selected), 48)
        self.assertEqual(sum(map(len, recipes.values())), 144)
        flattened = {value for transforms in recipes.values() for value in transforms}
        self.assertFalse(
            flattened.intersection(
                {"historical_zscore60", "minus_ema60", "rolling_std60"}
            )
        )
        # minus_ema20 is recursive: exact validation cold starts must rebuild
        # the complete session rather than only the first 20 time steps.
        self.assertEqual(temporal_session_warmup(recipes), -1)

    def test_all_feature_plan_routes_every_raw_feature_and_nests_compact_control(self):
        payload = json.loads(
            (STRATEGY_DIR / "all_feature_long_horizon_plan.json").read_text(
                encoding="utf-8"
            )
        )
        history_features = payload["history_features"]
        recipes = payload["recipes"]
        required = {
            "historical_zscore20", "minus_ema20", "rolling_std20"
        }
        self.assertEqual(len(history_features), 323)
        self.assertEqual(len(set(history_features)), 323)
        self.assertTrue(all(required.issubset(value) for value in recipes.values()))
        self.assertEqual(sum(map(len, recipes.values())), 1032)
        self.assertEqual(payload["source_model_feature_count"], 1356)

        temporal_columns = temporal_column_names(history_features, recipes)
        metadata = {
            "feature_columns": [f"feature_{index:03d}" for index in range(323)],
            "temporal_feature_columns": temporal_columns,
        }
        compact = target_experiment_spec(
            metadata, "LGB468_C4", list(DEFAULT_RESPONDERS)
        )
        expanded_control = target_experiment_spec(
            metadata, "LGB1356", list(DEFAULT_RESPONDERS)
        )
        expanded = target_experiment_spec(
            metadata, "LGB1356_C4", list(DEFAULT_RESPONDERS)
        )
        baseline = target_experiment_spec(
            metadata, "A", list(DEFAULT_RESPONDERS)
        )
        self.assertEqual(len(compact["feature_names"]), 470)
        self.assertEqual(len(expanded_control["feature_names"]), 1356)
        self.assertEqual(len(expanded["feature_names"]), 1358)
        self.assertTrue(compact["selection_candidate"])
        self.assertTrue(expanded["selection_candidate"])
        self.assertFalse(expanded_control["selection_candidate"])
        self.assertFalse(baseline["selection_candidate"])

    def test_experimental_all_feature_plan_resolves_to_declared_width(self):
        features = [f"feature_{index:03d}" for index in range(323)]
        selected, recipes = select_temporal_plan(
            features,
            48,
            importance_path=None,
            plan_path=STRATEGY_DIR / "all_feature_long_horizon_plan.json",
        )
        self.assertEqual(selected, features)
        self.assertEqual(sum(map(len, recipes.values())), 1032)
        self.assertEqual(323 + 1032 + 1, 1356)
        self.assertEqual(temporal_session_warmup(recipes), -1)

    def test_only_c4_lineage_models_are_formal_selection_candidates(self):
        metadata = {
            "feature_columns": [f"feature_{index:03d}" for index in range(323)],
            "temporal_feature_columns": temporal_column_names(
                *self._all_feature_plan_parts()
            ),
        }
        variants = [
            "A", "C4", "LGB468", "LGB468_C4",
            "LGB1356", "LGB1356_C4",
        ]
        specs = {
            name: target_experiment_spec(
                metadata, name, list(DEFAULT_RESPONDERS)
            )
            for name in variants
        }
        self.assertEqual(
            registered_selection_candidates(variants, specs),
            ["LGB468_C4", "LGB1356_C4"],
        )
        controls = ["A", "C4", "LGB468", "LGB1356"]
        with self.assertRaisesRegex(ValueError, "only control models"):
            registered_selection_candidates(controls, specs)
        self.assertEqual(
            registered_selection_candidates(
                controls, specs, allow_control_deployment=True
            ),
            controls,
        )

    @staticmethod
    def _all_feature_plan_parts():
        payload = json.loads(
            (STRATEGY_DIR / "all_feature_long_horizon_plan.json").read_text(
                encoding="utf-8"
            )
        )
        return payload["history_features"], payload["recipes"]

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

    def test_long_horizon_transforms_use_only_previous_observations(self):
        recipes = {
            "feature_000": [
                "rmean5", "historical_zscore20", "minus_ema20",
                "rolling_std20",
            ]
        }
        builder = TemporalFeatureBuilder(
            1, feature_names=["feature_000"], recipes=recipes
        )
        actual = builder.transform(
            np.ones(3, dtype=np.int64),
            np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        )
        expected = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.5, 0.0, 1.0, 0.0],
                [2.0, 3.0, 3.0 - (1.0 + 2.0 / 21.0), 0.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_recipe_routing_matches_full_temporal_matrix(self):
        feature_names = ["feature_000", "feature_001"]
        recipes = {
            "feature_000": ["historical_zscore20", "lag1"],
            "feature_001": ["rmean5", "xs_rank"],
        }
        asset_ids = np.asarray([1, 2, 1, 2], dtype=np.int64)
        values = np.asarray(
            [[1.0, 10.0], [4.0, 8.0], [3.0, 12.0], [7.0, 6.0]],
            dtype=np.float32,
        )
        routed = TemporalFeatureBuilder(
            2, feature_names=feature_names, recipes=recipes
        ).transform(asset_ids, values)
        full = TemporalFeatureBuilder(2).transform(asset_ids, values)
        indices = [
            feature_index * len(TEMPORAL_SUFFIXES)
            + TEMPORAL_SUFFIXES.index(transform)
            for feature_index, feature in enumerate(feature_names)
            for transform in recipes[feature]
        ]
        np.testing.assert_allclose(routed, full[:, indices], atol=1e-6)

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

    def test_validation_session_restarts_temporal_history(self):
        recipes = {"feature_000": ["lag1", "diff1", "rmean5"]}
        values = np.arange(1, 8, dtype=np.float32).reshape(-1, 1)
        global_builder = TemporalFeatureBuilder(
            1, feature_names=["feature_000"], recipes=recipes
        )
        temporal = np.vstack([
            global_builder.transform(
                np.asarray([1]), values[index:index + 1]
            )
            for index in range(len(values))
        ])
        matrix = np.column_stack([
            values, temporal, np.ones(len(values), dtype=np.float32)
        ]).astype(np.float32)
        metadata = {
            "feature_columns": ["feature_000"],
            "temporal_features": ["feature_000"],
            "temporal_recipes": recipes,
            "temporal_feature_columns": temporal_column_names(
                ["feature_000"], recipes
            ),
        }
        self.assertEqual(temporal_session_warmup(recipes), 5)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            np.save(cache_dir / "shard_00000_x.npy", matrix)
            np.save(
                cache_dir / "shard_00000_time.npy",
                np.arange(len(values), dtype=np.int64),
            )
            segments = [(0, 2, 7)]
            prefix = build_cold_start_prefix(cache_dir, metadata, segments)
            patched = matrix_for_segments(
                cache_dir, segments, patches=session_patch(prefix)
            )
            sequence = ShardSequence(
                cache_dir, segments, patches=session_patch(prefix)
            )
            streamed = sequence[:]

        self.assertEqual(prefix.shape, (5, 5))
        self.assertEqual(matrix[2, 1], 2.0)
        self.assertEqual(patched[0, 1], 0.0)
        np.testing.assert_allclose(streamed, patched, atol=1e-6)
        np.testing.assert_allclose(
            patched[:, 1:4],
            np.asarray([
                [0.0, 3.0, 3.0],
                [3.0, 1.0, 3.5],
                [4.0, 1.0, 4.0],
                [5.0, 1.0, 4.5],
                [6.0, 1.0, 5.0],
            ]),
            atol=1e-6,
        )

    def test_clipping_report_uses_fixed_bounds(self):
        y = np.asarray([0.0, 0.1, 0.2, 0.3])
        pred = np.asarray([0.0, 0.1, 0.2, 10.0])
        report = clipping_diagnostics(y, pred, np.ones(4), (0.0, 0.3))
        self.assertEqual(report["clip_min"], 0.0)
        self.assertEqual(report["clip_max"], 0.3)
        self.assertGreater(report["clipping_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
