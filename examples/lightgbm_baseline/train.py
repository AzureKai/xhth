from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from data_utils import (
    feature_columns_from_path,
    load_train_frame,
    manifest_files,
    sample_by_time,
)
from features import prepare_model_frame, select_history_features
from preprocess import apply_preprocess, fit_feature_schema
from validation import (
    evaluate_gates,
    fit_prediction_scale,
    make_validation_plan,
    weighted_zero_mean_r2,
)

# Dataset binning must stay fixed across CV candidates and final ensemble seeds.
# Bagging / feature seeds still change per model, so the ensemble remains diverse.
DATA_RANDOM_SEED = 2026
HISTOGRAM_POOL_SIZE_MB = 8_192.0

# Shared, pre-registered CPU settings. These are deliberately conservative:
# keep the score-aligned L2 objective and the default 255 bins, force the
# memory-friendlier histogram layout for this wide dataset, and cap histogram
# cache growth so Windows retains operating headroom.
BASE_PARAMS = {
    "objective": "regression",
    "metric": "None",
    "boosting_type": "gbdt",
    "data_sample_strategy": "bagging",
    "learning_rate": 0.03,
    "bagging_freq": 1,
    "max_bin": 255,
    "device_type": "cpu",
    "deterministic": True,
    "force_col_wise": True,
    "histogram_pool_size": HISTOGRAM_POOL_SIZE_MB,
    "verbosity": -1,
}

# The old v1 candidates are retained only so resume_low_memory.py can reproduce
# already-created legacy artifacts. New training uses PARAM_CANDIDATES below.
LEGACY_BASE_PARAMS = {
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.03,
    "bagging_freq": 1,
    "verbosity": -1,
}

LEGACY_PARAM_CANDIDATES: tuple[dict, ...] = (
    {
        "name": "leaves31_regular",
        "num_leaves": 31,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "lambda_l2": 10.0,
        "regularization_rank": 0,
        "logic": "较浅树 + 中等正则。",
    },
    {
        "name": "leaves63_regular",
        "num_leaves": 63,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "lambda_l2": 10.0,
        "regularization_rank": 0,
        "logic": "较深树 + 中等正则。",
    },
    {
        "name": "leaves31_strong",
        "num_leaves": 31,
        "min_data_in_leaf": 5000,
        "feature_fraction": 1.00,
        "bagging_fraction": 0.80,
        "lambda_l2": 20.0,
        "regularization_rank": 1,
        "logic": "较浅树 + 更强叶样本量与 L2 正则。",
    },
    {
        "name": "leaves63_strong",
        "num_leaves": 63,
        "min_data_in_leaf": 5000,
        "feature_fraction": 1.00,
        "bagging_fraction": 0.80,
        "lambda_l2": 20.0,
        "regularization_rank": 1,
        "logic": "较深树 + 强正则约束。",
    },
)

CANDIDATE_PARAM_KEYS = (
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "feature_fraction",
    "feature_fraction_bynode",
    "bagging_fraction",
    "lambda_l1",
    "lambda_l2",
    "path_smooth",
)

# Low-risk v3 candidates. The exact old winner remains as the control; the
# other three change only a small, interpretable set of regularization knobs.
PARAM_CANDIDATES: tuple[dict, ...] = (
    {
        "name": "leaves63_reference",
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.80,
        "feature_fraction_bynode": 1.00,
        "bagging_fraction": 0.80,
        "lambda_l1": 0.0,
        "lambda_l2": 10.0,
        "path_smooth": 0.0,
        "regularization_rank": 0,
        "logic": "保留旧 CV 冠军参数，作为所有内部优化的对照组。",
    },
    {
        "name": "leaves63_balanced",
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 3500,
        "feature_fraction": 0.90,
        "feature_fraction_bynode": 1.00,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.5,
        "lambda_l2": 15.0,
        "path_smooth": 0.0,
        "regularization_rank": 1,
        "logic": "在两组旧 63 叶配置之间插值，并加入轻量 L1。",
    },
    {
        "name": "leaves63_smoothed",
        "num_leaves": 63,
        "max_depth": 12,
        "min_data_in_leaf": 3500,
        "feature_fraction": 0.90,
        "feature_fraction_bynode": 0.90,
        "bagging_fraction": 0.85,
        "lambda_l1": 1.0,
        "lambda_l2": 15.0,
        "path_smooth": 100.0,
        "regularization_rank": 2,
        "logic": "增加深度保护、按节点特征采样和温和路径平滑。",
    },
    {
        "name": "leaves95_guarded",
        "num_leaves": 95,
        "max_depth": 14,
        "min_data_in_leaf": 5000,
        "feature_fraction": 0.90,
        "feature_fraction_bynode": 0.90,
        "bagging_fraction": 0.80,
        "lambda_l1": 1.0,
        "lambda_l2": 20.0,
        "path_smooth": 100.0,
        "regularization_rank": 2,
        "logic": "只小幅增加容量，并以叶样本、深度和路径平滑约束。",
    },
)


def _candidate_model_params(candidate: dict) -> dict:
    required = {"num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "lambda_l2"}
    missing = sorted(required.difference(candidate))
    if missing:
        raise ValueError(f"candidate {candidate.get('name', '<unnamed>')} is missing: {missing}")
    return {key: candidate[key] for key in CANDIDATE_PARAM_KEYS if key in candidate}


def lgb_zero_mean_r2(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    weight = dataset.get_weight()
    if weight is None:
        weight = np.ones_like(labels)
    denominator = np.sum(weight * labels * labels)
    score = 0.0 if denominator <= 0 else 1.0 - np.sum(weight * (labels - preds) ** 2) / denominator
    return "weighted_zero_mean_r2", float(score), True


def _candidate_params(
    seed: int,
    candidate: dict,
    *,
    num_threads: int = -1,
    extra_overrides: dict | None = None,
) -> dict:
    params = {
        **BASE_PARAMS,
        **_candidate_model_params(candidate),
        "num_threads": int(num_threads),
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "extra_seed": int(seed),
        "data_random_seed": DATA_RANDOM_SEED,
    }
    if extra_overrides:
        params.update(extra_overrides)
    return params


def _legacy_candidate_params(seed: int, candidate: dict, *, num_threads: int = -1) -> dict:
    """Exact pre-v3 parameters used only by the historical resume helper."""
    return {
        **LEGACY_BASE_PARAMS,
        "num_leaves": int(candidate["num_leaves"]),
        "min_data_in_leaf": int(candidate["min_data_in_leaf"]),
        "feature_fraction": float(candidate["feature_fraction"]),
        "bagging_fraction": float(candidate["bagging_fraction"]),
        "lambda_l2": float(candidate["lambda_l2"]),
        "num_threads": int(num_threads),
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "data_random_seed": int(seed),
    }


def _xy(frame: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = frame.loc[:, model_cols]
    y = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    w = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    return x, y, w


def _train_es(
    x_train,
    y_train,
    w_train,
    x_valid,
    y_valid,
    w_valid,
    *,
    seed: int,
    candidate: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=["asset_id"], free_raw_data=False)
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        weight=w_valid,
        categorical_feature=["asset_id"],
        reference=train_set,
        free_raw_data=False,
    )
    return lgb.train(
        _candidate_params(seed, candidate, num_threads=num_threads, extra_overrides=extra_overrides),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        feval=lgb_zero_mean_r2,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _train_fixed(
    x_train,
    y_train,
    w_train,
    *,
    seed: int,
    candidate: dict,
    num_boost_round: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=["asset_id"], free_raw_data=False)
    return lgb.train(
        _candidate_params(seed, candidate, num_threads=num_threads, extra_overrides=extra_overrides),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set],
        valid_names=["train"],
        feval=lgb_zero_mean_r2,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _mask_times(frame: pd.DataFrame, time_ids: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame["time_id"].isin(set(map(int, time_ids)))].copy()


def _prepare_inference_session(
    frame: pd.DataFrame,
    *,
    raw_features: list[str],
    history_features: list[str],
    rolling_windows: tuple[int, ...],
    expected_model_cols: list[str],
) -> pd.DataFrame:
    """Rebuild causal history from an empty state for one validation session.

    ``prepare_model_frame`` is vectorized, but its per-asset lag and rolling
    calculations are identical to calling ``Model.predict`` one increasing
    ``time_id`` at a time. Passing only the validation block prevents history
    from before the simulated API session from leaking into its first steps.
    """
    session, model_cols = prepare_model_frame(
        frame,
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
    )
    if model_cols != expected_model_cols:
        raise ValueError("sequential validation feature schema mismatch")
    return session


def _select_winning_candidate(candidate_results: list[dict]) -> dict:
    """Higher mean fold score wins; ties -> stronger regularization, then fewer rounds."""
    ordered = sorted(
        candidate_results,
        key=lambda item: (
            -float(item["mean_fold_score"]),
            -int(item["regularization_rank"]),
            int(item["mean_iterations"]),
        ),
    )
    return ordered[0]


def _evaluate_candidate_cv(
    *,
    prepared: pd.DataFrame,
    model_cols: list[str],
    raw_features: list[str],
    history_features: list[str],
    rolling_windows: tuple[int, ...],
    plan,
    candidate: dict,
    cv_seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    max_train_rows: int,
    max_valid_rows: int,
    num_threads: int,
    extra_overrides: dict | None,
) -> dict:
    oof_pred = np.zeros(len(prepared), dtype=np.float64)
    oof_mask = np.zeros(len(prepared), dtype=bool)
    fold_best_iterations: list[int] = []
    fold_scores: list[dict] = []

    for fold in plan.folds:
        train_part = sample_by_time(_mask_times(prepared, fold.train_time_ids), max_train_rows, seed=cv_seed)
        valid_source = sample_by_time(
            _mask_times(prepared, fold.valid_time_ids),
            max_valid_rows,
            seed=cv_seed + 1,
        )
        valid_part = _prepare_inference_session(
            valid_source,
            raw_features=raw_features,
            history_features=history_features,
            rolling_windows=rolling_windows,
            expected_model_cols=model_cols,
        )
        x_tr, y_tr, w_tr = _xy(train_part, model_cols)
        x_va, y_va, w_va = _xy(valid_part, model_cols)
        model = _train_es(
            x_tr,
            y_tr,
            w_tr,
            x_va,
            y_va,
            w_va,
            seed=cv_seed,
            candidate=candidate,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            num_threads=num_threads,
            extra_overrides=extra_overrides,
        )
        best_iteration = int(model.best_iteration or num_boost_round)
        fold_best_iterations.append(best_iteration)
        valid_full = _prepare_inference_session(
            _mask_times(prepared, fold.valid_time_ids),
            raw_features=raw_features,
            history_features=history_features,
            rolling_windows=rolling_windows,
            expected_model_cols=model_cols,
        )
        valid_indices = valid_full.index.to_numpy(dtype=np.int64, copy=False)
        preds = model.predict(valid_full.loc[:, model_cols], num_iteration=best_iteration)
        oof_pred[valid_indices] = preds
        oof_mask[valid_indices] = True
        fold_score = weighted_zero_mean_r2(
            valid_full["target"].to_numpy(dtype=np.float64),
            preds,
            valid_full["weight"].to_numpy(dtype=np.float64),
        )
        fold_scores.append(
            {
                "fold_id": fold.fold_id,
                "best_iteration": best_iteration,
                "valid_raw": fold_score,
                "train_rows": int(len(train_part)),
                "valid_rows": int(len(valid_part)),
                "train_time_start": int(fold.train_time_ids[0]),
                "train_time_end": int(fold.train_time_ids[-1]),
                "valid_time_start": int(fold.valid_time_ids[0]),
                "valid_time_end": int(fold.valid_time_ids[-1]),
            }
        )
        print(
            f"[cv] candidate={candidate['name']} fold={fold.fold_id} "
            f"best_iteration={best_iteration} valid_raw={fold_score:.6g}",
            flush=True,
        )

    score_values = np.asarray([item["valid_raw"] for item in fold_scores], dtype=np.float64)
    mean_fold_score = float(np.mean(score_values))
    mean_iterations = max(1, int(round(float(np.mean(fold_best_iterations)))))
    oof_frame = prepared.loc[oof_mask]
    oof_raw = weighted_zero_mean_r2(
        oof_frame["target"].to_numpy(dtype=np.float64),
        oof_pred[oof_mask],
        oof_frame["weight"].to_numpy(dtype=np.float64),
    )
    return {
        "name": candidate["name"],
        "logic": candidate.get("logic", ""),
        "regularization_rank": int(candidate.get("regularization_rank", 0)),
        "params": _candidate_model_params(candidate),
        "fold_scores": fold_scores,
        "fold_best_iterations": fold_best_iterations,
        "mean_fold_score": mean_fold_score,
        "std_fold_score": float(np.std(score_values)),
        "min_fold_score": float(np.min(score_values)),
        "latest_fold_score": float(score_values[-1]),
        "mean_iterations": mean_iterations,
        "oof_raw": oof_raw,
        "oof_pred": oof_pred,
        "oof_mask": oof_mask,
    }


def run_baseline_training(
    train_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    feature_cols: list[str] | None = None,
    top_k_history: int = 48,
    rolling_windows: tuple[int, ...] = (5,),
    seeds: tuple[int, ...] = (2026, 2027, 2028),
    n_splits: int = 5,
    holdout_fraction: float = 0.15,
    purge_steps: int = 30,
    min_train_fraction: float = 0.40,
    num_boost_round: int = 700,
    early_stopping_rounds: int = 80,
    num_threads: int = -1,
    max_train_rows: int = 0,
    max_valid_rows: int = 0,
    corr_sample_rows: int = 200_000,
    param_candidates: tuple[dict, ...] | list[dict] | None = None,
    param_overrides: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(param_candidates) if param_candidates is not None else PARAM_CANDIDATES
    if not candidates:
        raise ValueError("param_candidates must be non-empty")

    if feature_cols is None:
        feature_cols = [col for col in train_frame.columns if str(col).startswith("feature_")]
    if not feature_cols:
        raise ValueError("no feature_* columns found")

    plan = make_validation_plan(
        train_frame["time_id"],
        n_splits=n_splits,
        holdout_fraction=holdout_fraction,
        purge_steps=purge_steps,
        min_train_fraction=min_train_fraction,
    )

    # Freeze all target-dependent feature choices on the initial historical prefix.
    schema_source = _mask_times(train_frame, plan.feature_fit_time_ids)
    if schema_source.empty:
        raise ValueError("first fold train is empty; cannot fit preprocess schema")
    schema = fit_feature_schema(schema_source, feature_cols)
    raw_features = list(schema.raw_features)
    cleaned = apply_preprocess(train_frame, schema)

    history_source = sample_by_time(schema_source, corr_sample_rows, seed=seeds[0])
    history_source = apply_preprocess(history_source, schema)
    history_features = select_history_features(
        history_source,
        raw_features,
        top_k=top_k_history,
        sample_rows=corr_sample_rows,
        seed=seeds[0],
    )

    prepared, model_cols = prepare_model_frame(
        cleaned,
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
    )
    prepared = prepared.reset_index(drop=True)

    cv_seed = int(seeds[0])
    candidate_results: list[dict] = []
    for candidate in candidates:
        print(f"[cv] start candidate={candidate['name']}", flush=True)
        result = _evaluate_candidate_cv(
            prepared=prepared,
            model_cols=model_cols,
            raw_features=raw_features,
            history_features=history_features,
            rolling_windows=rolling_windows,
            plan=plan,
            candidate=candidate,
            cv_seed=cv_seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            max_train_rows=max_train_rows,
            max_valid_rows=max_valid_rows,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        )
        candidate_results.append(result)
        print(
            f"[cv] done candidate={candidate['name']} "
            f"mean_fold_score={result['mean_fold_score']:.6g} mean_iterations={result['mean_iterations']}",
            flush=True,
        )

    winner = _select_winning_candidate(candidate_results)
    winning_candidate = next(item for item in candidates if item["name"] == winner["name"])
    mean_iterations = int(winner["mean_iterations"])
    oof_mask = winner["oof_mask"]
    oof_pred = winner["oof_pred"]
    oof_frame = prepared.loc[oof_mask]
    fitted_oof_scale = fit_prediction_scale(
        oof_frame["target"].to_numpy(dtype=np.float64),
        oof_pred[oof_mask],
        oof_frame["weight"].to_numpy(dtype=np.float64),
    )
    oof_raw = float(winner["oof_raw"])

    holdout_part = _prepare_inference_session(
        _mask_times(prepared, plan.holdout_time_ids),
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
        expected_model_cols=model_cols,
    )
    development_part = sample_by_time(
        prepared.loc[prepared["time_id"].isin(set(map(int, plan.development_time_ids)))],
        max_train_rows,
        seed=cv_seed,
    )
    x_dev, y_dev, w_dev = _xy(development_part, model_cols)
    print(f"[holdout] train on development with mean_iterations={mean_iterations}", flush=True)
    holdout_model = _train_fixed(
        x_dev,
        y_dev,
        w_dev,
        seed=cv_seed,
        candidate=winning_candidate,
        num_boost_round=mean_iterations,
        num_threads=num_threads,
        extra_overrides=param_overrides,
    )
    holdout_pred = holdout_model.predict(holdout_part.loc[:, model_cols], num_iteration=mean_iterations)
    holdout_raw = weighted_zero_mean_r2(
        holdout_part["target"].to_numpy(dtype=np.float64),
        holdout_pred,
        holdout_part["weight"].to_numpy(dtype=np.float64),
    )

    gates = evaluate_gates(
        oof_raw_score=oof_raw,
        holdout_raw_score=holdout_raw,
        fitted_oof_scale=fitted_oof_scale,
    )

    # Final fit uses all labeled train rows, including holdout.
    final_train = sample_by_time(prepared, max_train_rows, seed=seeds[0])
    x_all, y_all, w_all = _xy(final_train, model_cols)
    model_files: list[str] = []
    best_iterations: list[int] = []
    for seed in seeds:
        print(f"[final] seed={seed} rounds={mean_iterations} candidate={winner['name']}", flush=True)
        booster = _train_fixed(
            x_all,
            y_all,
            w_all,
            seed=int(seed),
            candidate=winning_candidate,
            num_boost_round=mean_iterations,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        )
        name = "model.txt" if len(seeds) == 1 else f"model_seed{seed}.txt"
        booster.save_model(str(output_dir / name))
        model_files.append(name)
        best_iterations.append(mean_iterations)

    report = {
        "strategy": "lightgbm_baseline",
        "schema_version": 3,
        "optimization_profile": "lgbm_low_risk_v1",
        "tuning_policy": "purged_walk_forward_pre_registered_candidates_no_test_tuning",
        "scale_policy": "diagnostic_only_never_apply",
        "rows": {
            "train_all": int(len(prepared)),
            "oof": int(oof_mask.sum()),
            "holdout": int(len(holdout_part)),
            "development": int(len(development_part)),
            "final_train_sample": int(len(final_train)),
            "final_train_includes_holdout": True,
        },
        "validation": {
            "cv_scheme": plan.cv_scheme,
            "n_splits": n_splits,
            "holdout_fraction": holdout_fraction,
            "purge_steps": purge_steps,
            "min_train_fraction": min_train_fraction,
            "feature_fit_time_count": int(len(plan.feature_fit_time_ids)),
            "inference_simulation": "cold_start_causal_time_order",
            "rounds_aggregation": "mean",
            "selection_metric": "mean_fold_score",
            "tie_break": ["stronger_regularization", "fewer_mean_iterations"],
            "candidates": [
                {
                    "name": item["name"],
                    "logic": item["logic"],
                    "params": item["params"],
                    "regularization_rank": item["regularization_rank"],
                    "mean_fold_score": item["mean_fold_score"],
                    "std_fold_score": item["std_fold_score"],
                    "min_fold_score": item["min_fold_score"],
                    "latest_fold_score": item["latest_fold_score"],
                    "mean_iterations": item["mean_iterations"],
                    "oof_raw": item["oof_raw"],
                    "fold_best_iterations": item["fold_best_iterations"],
                    "fold_scores": item["fold_scores"],
                }
                for item in candidate_results
            ],
            "selected_candidate": winner["name"],
            "fold_scores": winner["fold_scores"],
            "fold_best_iterations": winner["fold_best_iterations"],
            "mean_iterations": mean_iterations,
            "mean_fold_score": winner["mean_fold_score"],
            "std_fold_score": winner["std_fold_score"],
            "min_fold_score": winner["min_fold_score"],
            "latest_fold_score": winner["latest_fold_score"],
            "oof_raw": oof_raw,
            "holdout_raw": holdout_raw,
            "fitted_oof_scale": fitted_oof_scale,
            "gates": gates,
        },
        "features": {
            "selected_raw_features": raw_features,
            "history_features": history_features,
            "rolling_windows": list(rolling_windows),
            "model_feature_count": len(model_cols),
        },
        "seeds": list(map(int, seeds)),
        "model_files": model_files,
        "best_iteration": mean_iterations,
        "best_iterations": best_iterations,
        "prediction_scale": 1.0,
        "fitted_oof_scale": fitted_oof_scale,
        "gates_passed": gates["gates_passed"],
        "selected_candidate": winner["name"],
        "num_threads": int(num_threads),
        "lgbm_params": _candidate_params(
            seeds[0],
            winning_candidate,
            num_threads=num_threads,
            extra_overrides=param_overrides,
        ),
    }
    (output_dir / "lightgbm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM baseline (purged walk-forward gated).")
    parser.add_argument("--release-root", required=True, help="Release root containing manifest.json and train/.")
    parser.add_argument(
        "--model-dir",
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Directory to write model files and reports.",
    )
    parser.add_argument("--top-k-history", type=int, default=48)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--max-valid-rows", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--min-train-fraction", type=float, default=0.40)
    parser.add_argument("--purge-steps", type=int, default=30)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help="LightGBM num_threads; -1 uses all available cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_root = Path(args.release_root)
    train_paths = manifest_files(release_root, "train")
    feature_cols = feature_columns_from_path(train_paths[0])
    load_cols = ["row_id", "time_id", "asset_id", "weight", "target", *feature_cols]
    train_frame = load_train_frame(release_root, columns=load_cols)
    report = run_baseline_training(
        train_frame,
        output_dir=args.output_dir,
        feature_cols=feature_cols,
        top_k_history=args.top_k_history,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        n_splits=args.n_splits,
        holdout_fraction=args.holdout_fraction,
        min_train_fraction=args.min_train_fraction,
        purge_steps=args.purge_steps,
        num_threads=args.num_threads,
    )
    print(
        json.dumps(
            {
                "gates_passed": report["gates_passed"],
                "selected_candidate": report["selected_candidate"],
                "mean_iterations": report["best_iteration"],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
