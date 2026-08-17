from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeFold:
    fold_id: int
    train_time_ids: np.ndarray
    valid_time_ids: np.ndarray


@dataclass(frozen=True)
class ValidationPlan:
    folds: tuple[TimeFold, ...]
    development_time_ids: np.ndarray
    holdout_time_ids: np.ndarray
    feature_fit_time_ids: np.ndarray
    purge_steps: int
    cv_scheme: str = "purged_walk_forward"
    min_train_fraction: float = 0.40


def make_validation_plan(
    time_ids,
    *,
    n_splits: int = 5,
    holdout_fraction: float = 0.15,
    purge_steps: int = 30,
    min_train_fraction: float = 0.40,
) -> ValidationPlan:
    """Expanding-window validation with a trailing untouched holdout.

    The first ``min_train_fraction`` of development time ids is the initial
    training prefix. The remaining development ids are split into consecutive
    validation blocks. Fold ``i`` trains only on ids strictly before its
    validation block, with ``purge_steps`` observed time ids removed from the
    training tail. No fold can train on its validation future.

    Feature schema and target-based feature selection must use
    ``feature_fit_time_ids``. This is the first fold's historical training
    prefix and therefore precedes every validation block.
    """
    unique = np.unique(np.asarray(time_ids, dtype=np.int64))
    if len(unique) < n_splits + 2:
        raise ValueError("not enough unique time_id values for validation")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if purge_steps < 0:
        raise ValueError("purge_steps must be non-negative")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not 0.0 < min_train_fraction < 1.0:
        raise ValueError("min_train_fraction must be between 0 and 1")

    holdout_count = max(1, int(np.ceil(len(unique) * holdout_fraction)))
    development, holdout = unique[:-holdout_count], unique[-holdout_count:]
    initial_count = max(1, int(np.ceil(len(development) * min_train_fraction)))
    remaining = development[initial_count:]
    if len(remaining) < n_splits:
        raise ValueError("development tail is too short for the requested forward folds")

    blocks = tuple(np.asarray(block, dtype=np.int64) for block in np.array_split(remaining, n_splits))
    folds: list[TimeFold] = []
    for fold_id, valid in enumerate(blocks):
        if valid.size == 0:
            raise ValueError("validation plan produced an empty valid block")
        valid_start = int(np.searchsorted(development, valid[0]))
        train_end = valid_start - int(purge_steps)
        train = development[: max(train_end, 0)]
        if train.size == 0:
            raise ValueError("validation plan produced an empty train fold after purge")
        if int(train[-1]) >= int(valid[0]):
            raise AssertionError("walk-forward fold contains non-historical training ids")
        folds.append(TimeFold(fold_id, train.copy(), valid))

    feature_fit = folds[0].train_time_ids.copy()
    return ValidationPlan(
        tuple(folds),
        development,
        holdout,
        feature_fit,
        int(purge_steps),
        cv_scheme="purged_walk_forward",
        min_train_fraction=float(min_train_fraction),
    )


def make_purged_kfold_plan(
    time_ids,
    *,
    n_splits: int = 5,
    holdout_fraction: float = 0.15,
    purge_steps: int = 30,
) -> ValidationPlan:
    """Legacy symmetric plan kept only for reproducing pre-v2 artifacts."""
    unique = np.unique(np.asarray(time_ids, dtype=np.int64))
    if len(unique) < n_splits + 2:
        raise ValueError("not enough unique time_id values for validation")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if purge_steps < 0:
        raise ValueError("purge_steps must be non-negative")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    holdout_count = max(1, int(np.ceil(len(unique) * holdout_fraction)))
    development, holdout = unique[:-holdout_count], unique[-holdout_count:]
    blocks = tuple(np.asarray(block, dtype=np.int64) for block in np.array_split(development, n_splits))
    folds: list[TimeFold] = []
    for fold_id, valid in enumerate(blocks):
        train_parts = [blocks[idx] for idx in range(n_splits) if idx != fold_id]
        candidate_train = np.concatenate(train_parts)
        v_min = int(valid.min())
        v_max = int(valid.max())
        keep = (candidate_train <= v_min - purge_steps - 1) | (candidate_train >= v_max + purge_steps + 1)
        train = candidate_train[keep]
        if train.size == 0:
            raise ValueError("validation plan produced an empty train fold after purge")
        folds.append(TimeFold(fold_id, np.sort(train), valid))
    return ValidationPlan(
        tuple(folds),
        development,
        holdout,
        folds[0].train_time_ids.copy(),
        int(purge_steps),
        cv_scheme="purged_kfold",
        min_train_fraction=0.0,
    )


def weighted_zero_mean_r2(y_true, y_pred, weight) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(w * y * y))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(1.0 - np.sum(w * (y - p) ** 2) / denominator)


def fit_prediction_scale(y_true, prediction, weight) -> float:
    """Closed-form amplitude diagnostic; must not be applied as a calibrator."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(w * p * p))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 1.0
    scale = float(np.sum(w * y * p) / denominator)
    return scale if np.isfinite(scale) else 1.0


def evaluate_gates(
    *,
    oof_raw_score: float,
    holdout_raw_score: float,
    fitted_oof_scale: float,
    scale_low: float = 0.75,
    scale_high: float = 1.25,
) -> dict:
    checks = {
        "oof_raw_positive": bool(oof_raw_score > 0.0),
        "holdout_raw_positive": bool(holdout_raw_score > 0.0),
        "scale_in_range": bool(scale_low <= fitted_oof_scale <= scale_high),
    }
    return {
        **checks,
        "gates_passed": bool(all(checks.values())),
        "scale_low": scale_low,
        "scale_high": scale_high,
        "prediction_scale_applied": 1.0,
        "fitted_oof_scale": float(fitted_oof_scale),
    }
