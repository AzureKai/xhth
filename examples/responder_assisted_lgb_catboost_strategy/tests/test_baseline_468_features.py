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
from main import normalized_model_weights

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
    SlicedSequence,
    TARGET_PARAM_PROFILES,
    TIER_RESPONDERS,
    build_cold_start_prefix,
    clipping_diagnostics,
    cross_fold_feature_stability,
    matrix_for_segments,
    low_risk_lgb_params,
    registered_selection_candidates,
    select_oof_blend_weight,
    select_temporal_plan,
    selected_experiments,
    session_patch,
    subset_target_experiment_spec,
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

    def test_cross_fold_stability_keeps_repeated_and_protected_features(self):
        names = [
            "feature_000", "ts_lag1_feature_000", "feature_noise",
            "asset_id", "responder_03_hat", "responder_02_hat",
        ]
        gain = np.asarray([
            [10, 4, 5, 1, 0, 0],
            [12, 3, 0, 1, 0, 0],
            [11, 2, 0, 1, 0, 0],
            [9, 0, 0, 1, 0, 0],
        ], dtype=np.float64)
        split = (gain > 0).astype(np.float64)
        report, selected, summary = cross_fold_feature_stability(
            names, gain, split, min_fold_rate=0.75,
            min_count=5, max_count=5,
            protected_features=[
                "asset_id", "responder_03_hat", "responder_02_hat",
            ],
        )
        self.assertEqual(len(selected), 5)
        self.assertIn("feature_000", selected)
        self.assertIn("ts_lag1_feature_000", selected)
        self.assertNotIn("feature_noise", selected)
        self.assertEqual(summary["stable_eligible_count"], 3)
        reasons = report.set_index("feature")["selection_reason"].to_dict()
        self.assertEqual(reasons["responder_03_hat"], "protected")
        self.assertEqual(reasons["feature_noise"], "excluded")

    def test_stable_subset_preserves_matrix_order_and_required_columns(self):
        spec = {
            "name": "LGB468_C4",
            "base_indices": [10, 11, 12],
            "responders": ["responder_03", "responder_02"],
            "responder_indices": [0, 1],
            "feature_names": [
                "feature_000", "feature_001", "asset_id",
                "responder_03_hat", "responder_02_hat",
            ],
            "temporal_groups": ["compact_468"],
            "shuffle_within_time": False,
            "deployable": True,
            "selection_candidate": True,
            "stable_source": "LGB468_C4",
        }
        subset = subset_target_experiment_spec(
            spec,
            [
                "feature_000", "asset_id",
                "responder_03_hat", "responder_02_hat",
            ],
            "LGB468_C4_STABLE",
        )
        self.assertEqual(subset["base_indices"], [10, 12])
        self.assertEqual(subset["responder_indices"], [0, 1])
        self.assertEqual(
            subset["feature_names"],
            [
                "feature_000", "asset_id",
                "responder_03_hat", "responder_02_hat",
            ],
        )
        self.assertTrue(subset["stable_selection_applied"])

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

    def test_default_suite_includes_stable_improvement_and_source(self):
        args = types.SimpleNamespace(
            ablation_mode="all", target_experiments="",
            experiment_suite="next-step",
        )
        self.assertEqual(
            selected_experiments(args, list(DEFAULT_RESPONDERS)),
            [
                "A", "C4", "LGB468", "LGB468_C4",
                "LGB468_C4_STABLE",
                "LGB468_C4_STABLE_RECENT50",
                "LGB468_C4_STABLE_BLEND",
            ],
        )

    def test_recent_and_blend_specs_keep_oof_protocol_as_model_variants(self):
        metadata = {
            "feature_columns": [f"feature_{index:03d}" for index in range(323)],
            "temporal_feature_columns": temporal_column_names(
                *self._all_feature_plan_parts()
            ),
        }
        recent = target_experiment_spec(
            metadata, "LGB468_C4_STABLE_RECENT50", list(DEFAULT_RESPONDERS)
        )
        blend = target_experiment_spec(
            metadata, "LGB468_C4_STABLE_BLEND", list(DEFAULT_RESPONDERS)
        )
        self.assertEqual(recent["train_window_fraction"], 0.5)
        self.assertFalse(recent["virtual_blend"])
        self.assertTrue(blend["virtual_blend"])
        self.assertTrue(blend["selection_candidate"])
        self.assertEqual(
            blend["stable_feature_parent"], "LGB468_C4_STABLE"
        )

    def test_explicit_blend_adds_all_training_dependencies(self):
        args = types.SimpleNamespace(
            ablation_mode="all",
            target_experiments="LGB468_C4_STABLE_BLEND",
            experiment_suite="next-step",
        )
        self.assertEqual(
            selected_experiments(args, list(DEFAULT_RESPONDERS)),
            [
                "LGB468_C4", "LGB468_C4_STABLE",
                "LGB468_C4_STABLE_RECENT50",
                "LGB468_C4_STABLE_BLEND",
            ],
        )

    def test_oof_blend_selects_best_preregistered_recent_weight(self):
        targets = [
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 4.0]),
        ]
        weights = [np.ones(2), np.ones(2)]
        full = [np.zeros(2), np.zeros(2)]
        recent = [targets[0].copy(), targets[1].copy()]
        report, predictions = select_oof_blend_weight(
            targets, weights, full, recent, [0.25, 0.5, 0.75]
        )
        self.assertEqual(report["recent_weight"], 0.75)
        self.assertEqual(len(report["weight_search"]), 3)
        np.testing.assert_allclose(predictions[0], targets[0] * 0.75)

    def test_sliced_sequence_does_not_change_row_values(self):
        source = np.arange(30, dtype=np.float64).reshape(10, 3)
        sliced = SlicedSequence(source, 4)
        self.assertEqual(len(sliced), 6)
        np.testing.assert_array_equal(sliced[:], source[4:])
        np.testing.assert_array_equal(sliced[-1], source[-1])

    def test_inference_model_weights_are_explicit_and_normalized(self):
        actual = normalized_model_weights([0.25, 0.25, 0.5], 3)
        np.testing.assert_allclose(actual, [0.25, 0.25, 0.5])
        np.testing.assert_allclose(
            normalized_model_weights(None, 3),
            np.full(3, 1.0 / 3.0),
        )
        with self.assertRaisesRegex(ValueError, "matching target_models"):
            normalized_model_weights([1.0, -1.0], 2)

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
            "LGB468_C4_STABLE", "LGB1356", "LGB1356_C4",
        ]
        specs = {
            name: target_experiment_spec(
                metadata, name, list(DEFAULT_RESPONDERS)
            )
            for name in variants
        }
        self.assertEqual(
            registered_selection_candidates(variants, specs),
            ["LGB468_C4", "LGB468_C4_STABLE", "LGB1356_C4"],
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
