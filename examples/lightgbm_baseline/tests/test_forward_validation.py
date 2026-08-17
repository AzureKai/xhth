from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


STRATEGY_DIR = Path(__file__).resolve().parents[1]
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from features import prepare_model_frame
from main import DEFAULT_PREDICT_THREADS, _predict_thread_count
from train import (
    BASE_PARAMS,
    DATA_RANDOM_SEED,
    LEGACY_PARAM_CANDIDATES,
    PARAM_CANDIDATES,
    _candidate_params,
    _prepare_inference_session,
)
import train_low_memory as low_memory
from train_low_memory import build_cold_start_patch, build_dataset, row_offsets_from_counts, train_fixed
from validation import make_validation_plan


class ForwardValidationTests(unittest.TestCase):
    def test_prediction_threads_are_isolated_from_training_threads(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIGHTGBM_BASELINE_PREDICT_THREADS", None)
            self.assertEqual(_predict_thread_count(), DEFAULT_PREDICT_THREADS)
        with patch.dict(os.environ, {"LIGHTGBM_BASELINE_PREDICT_THREADS": "2"}):
            self.assertEqual(_predict_thread_count(), 2)
        with patch.dict(os.environ, {"LIGHTGBM_BASELINE_PREDICT_THREADS": "0"}):
            with self.assertRaisesRegex(ValueError, "at least 1"):
                _predict_thread_count()

    def test_low_risk_candidates_keep_the_old_winner_as_control(self) -> None:
        self.assertEqual(len(PARAM_CANDIDATES), 4)
        self.assertEqual(len({item["name"] for item in PARAM_CANDIDATES}), 4)
        reference = PARAM_CANDIDATES[0]
        self.assertEqual(reference["name"], "leaves63_reference")
        self.assertEqual(reference["num_leaves"], 63)
        self.assertEqual(reference["min_data_in_leaf"], 2000)
        self.assertEqual(reference["feature_fraction"], 0.80)
        self.assertEqual(reference["bagging_fraction"], 0.80)
        self.assertEqual(reference["lambda_l2"], 10.0)
        self.assertEqual([item["name"] for item in LEGACY_PARAM_CANDIDATES], [
            "leaves31_regular",
            "leaves63_regular",
            "leaves31_strong",
            "leaves63_strong",
        ])

    def test_bin_seed_is_fixed_while_ensemble_sampling_seeds_change(self) -> None:
        first = _candidate_params(2026, PARAM_CANDIDATES[2], num_threads=24)
        second = _candidate_params(2027, PARAM_CANDIDATES[2], num_threads=24)
        self.assertEqual(first["data_random_seed"], DATA_RANDOM_SEED)
        self.assertEqual(second["data_random_seed"], DATA_RANDOM_SEED)
        self.assertNotEqual(first["bagging_seed"], second["bagging_seed"])
        self.assertNotEqual(first["feature_fraction_seed"], second["feature_fraction_seed"])
        self.assertTrue(first["deterministic"])
        self.assertTrue(first["force_col_wise"])
        self.assertEqual(first["device_type"], "cpu")
        self.assertEqual(first["histogram_pool_size"], 8192.0)
        self.assertEqual(first["max_bin"], 255)
        self.assertEqual(first["path_smooth"], PARAM_CANDIDATES[2]["path_smooth"])
        self.assertEqual(BASE_PARAMS["data_sample_strategy"], "bagging")

    def test_folds_are_strictly_historical_and_purged_by_observed_steps(self) -> None:
        # Deliberate gaps ensure purge_steps counts observed time ids, not the
        # numeric distance between anonymous ids.
        times = np.arange(120, dtype=np.int64) * 10
        plan = make_validation_plan(
            times,
            n_splits=3,
            holdout_fraction=0.10,
            purge_steps=2,
            min_train_fraction=0.40,
        )

        self.assertEqual(plan.cv_scheme, "purged_walk_forward")
        previous_valid_end = -1
        for fold in plan.folds:
            self.assertLess(int(fold.train_time_ids[-1]), int(fold.valid_time_ids[0]))
            self.assertEqual(np.intersect1d(fold.train_time_ids, fold.valid_time_ids).size, 0)
            valid_start = int(np.searchsorted(plan.development_time_ids, fold.valid_time_ids[0]))
            train_end = int(np.searchsorted(plan.development_time_ids, fold.train_time_ids[-1]))
            self.assertEqual(valid_start - train_end - 1, 2)
            self.assertGreater(int(fold.valid_time_ids[0]), previous_valid_end)
            previous_valid_end = int(fold.valid_time_ids[-1])

    def test_target_based_feature_fit_prefix_never_overlaps_validation(self) -> None:
        plan = make_validation_plan(np.arange(1_000), purge_steps=30)
        for fold in plan.folds:
            self.assertEqual(np.intersect1d(plan.feature_fit_time_ids, fold.valid_time_ids).size, 0)
            self.assertLess(int(plan.feature_fit_time_ids[-1]), int(fold.valid_time_ids[0]))

    def test_validation_session_restarts_causal_history(self) -> None:
        frame = self._example_frame()
        global_prepared, model_cols = prepare_model_frame(
            frame,
            raw_features=["feature_000"],
            history_features=["feature_000"],
            rolling_windows=(3,),
        )
        validation_source = global_prepared.loc[global_prepared["time_id"] >= 2]
        session = _prepare_inference_session(
            validation_source,
            raw_features=["feature_000"],
            history_features=["feature_000"],
            rolling_windows=(3,),
            expected_model_cols=model_cols,
        )

        first_time = session["time_id"] == 2
        self.assertTrue(np.all(session.loc[first_time, "lag1_feature_000"].to_numpy() == 0.0))
        self.assertTrue(
            np.allclose(
                session.loc[first_time, "diff1_feature_000"],
                session.loc[first_time, "feature_000"],
            )
        )
        self.assertFalse(
            np.allclose(
                global_prepared.loc[global_prepared["time_id"] == 2, "lag1_feature_000"],
                session.loc[first_time, "lag1_feature_000"],
            )
        )

    def test_low_memory_patch_matches_cold_session_features(self) -> None:
        frame = self._example_frame()
        global_prepared, model_cols = prepare_model_frame(
            frame,
            raw_features=["feature_000"],
            history_features=["feature_000"],
            rolling_windows=(3,),
        )
        matrix = global_prepared.loc[:, model_cols].to_numpy(dtype=np.float32)
        unique_times, counts = np.unique(frame["time_id"].to_numpy(), return_counts=True)
        offsets = row_offsets_from_counts(counts)
        session_times = unique_times[2:]

        with tempfile.TemporaryDirectory() as directory:
            matrix_path = Path(directory) / "matrix.npy"
            np.save(matrix_path, matrix)
            patches = build_cold_start_patch(
                matrix_path,
                unique_times=unique_times,
                time_counts=counts,
                row_offsets=offsets,
                session_time_ids=session_times,
                raw_features=["feature_000"],
                history_features=["feature_000"],
                rolling_windows=(3,),
                model_cols=model_cols,
            )

        start, end, patched = patches[0]
        expected, _ = prepare_model_frame(
            frame.loc[frame["time_id"].isin(session_times[:3])],
            raw_features=["feature_000"],
            history_features=["feature_000"],
            rolling_windows=(3,),
        )
        self.assertEqual((start, end), (4, 10))
        self.assertTrue(np.allclose(patched, expected.loc[:, model_cols].to_numpy(dtype=np.float32)))

    def test_low_memory_cache_uses_the_frozen_rolling_window_name(self) -> None:
        frame = self._example_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "train.parquet"
            cache_dir = root / "cache"
            cache_dir.mkdir()
            frame.to_parquet(parquet_path, index=False)
            with patch.object(low_memory, "ROLLING_WINDOWS", (3,)):
                _, _, _, model_cols = low_memory.materialize_model_data(
                    [parquet_path],
                    cache_dir,
                    len(frame),
                    ["feature_000"],
                    ["feature_000"],
                )

        self.assertIn("rmean3_feature_000", model_cols)
        self.assertNotIn("rmean5_feature_000", model_cols)

    def test_low_memory_dataset_can_train_multiple_ensemble_seeds(self) -> None:
        rows = 240
        asset = np.tile(np.arange(4, dtype=np.float32), rows // 4)
        feature = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
        matrix = np.column_stack([asset, feature]).astype(np.float32)
        target = (0.2 * feature + 0.01 * np.sin(np.arange(rows))).astype(np.float32)
        weight = np.ones(rows, dtype=np.float32)
        candidate = {
            "name": "seed_contract_smoke",
            "num_leaves": 7,
            "max_depth": 4,
            "min_data_in_leaf": 5,
            "feature_fraction": 1.0,
            "feature_fraction_bynode": 1.0,
            "bagging_fraction": 0.9,
            "lambda_l1": 0.0,
            "lambda_l2": 1.0,
            "path_smooth": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.npy"
            target_path = root / "target.npy"
            weight_path = root / "weight.npy"
            np.save(matrix_path, matrix)
            np.save(target_path, target)
            np.save(weight_path, weight)
            dataset = build_dataset(
                matrix_path,
                matrix.shape,
                target_path,
                weight_path,
                [(0, rows)],
                ["asset_id", "feature_000"],
                construction_overrides={"min_data_in_leaf": 5},
            )
            first = train_fixed(dataset, candidate, 2026, 3, 2)
            second = train_fixed(dataset, candidate, 2027, 3, 2)
            first_params = dict(first.params)
            second_params = dict(second.params)
            del second, first, dataset
            gc.collect()

        self.assertEqual(first_params["data_random_seed"], DATA_RANDOM_SEED)
        self.assertEqual(second_params["data_random_seed"], DATA_RANDOM_SEED)
        self.assertNotEqual(first_params["bagging_seed"], second_params["bagging_seed"])

    @staticmethod
    def _example_frame() -> pd.DataFrame:
        rows = []
        for time_id, values in enumerate(((1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0), (5.0, 50.0))):
            for asset_id, value in enumerate(values):
                rows.append(
                    {
                        "row_id": len(rows),
                        "time_id": time_id,
                        "asset_id": asset_id,
                        "weight": 1.0,
                        "target": 0.0,
                        "feature_000": value,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
