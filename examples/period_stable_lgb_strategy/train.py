from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a period-stable LightGBM strategy.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--num-periods", type=int, default=5)
    parser.add_argument("--max-rows-per-period", type=int, default=200_000)
    parser.add_argument("--max-valid-rows", type=int, default=300_000)
    parser.add_argument("--stable-feature-count", type=int, default=120)
    parser.add_argument("--min-periods-for-stable", type=int, default=2)
    parser.add_argument("--weighted-feature-count", type=int, default=None)
    parser.add_argument("--period-weighting", choices=["exp_recent", "linear_recent", "equal"], default="exp_recent")
    parser.add_argument("--period-weight-decay", type=float, default=0.7)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.2)
    parser.add_argument("--ema-feature-count", type=int, default=50)
    parser.add_argument("--ema-halflives", default="5,20,60")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--probe-rounds", type=int, default=500)
    parser.add_argument("--final-rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=80)
    parser.add_argument("--alpha-grid", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Only inspect schema/time_id split plan; do not train models.")
    return parser.parse_args()


def require_package(package: str) -> None:
    if importlib.util.find_spec(package) is None:
        raise ImportError(f"missing dependency '{package}'")


def manifest(data_root: Path) -> dict[str, Any]:
    path = data_root / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def manifest_files(data_root: Path, split: str) -> list[Path]:
    files = manifest(data_root).get("files", {}).get(split, [])
    if files:
        return [data_root / str(file) for file in files]
    return sorted((data_root / split).glob("*.parquet"))


def n_assets(data_root: Path) -> int:
    return int(manifest(data_root).get("counts", {}).get("n_assets", 15))


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def feature_columns(files: list[Path]) -> list[str]:
    if not files:
        raise ValueError("no parquet files found")
    features = [col for col in parquet_columns(files[0]) if str(col).startswith("feature_")]
    if not features:
        raise ValueError("no feature_* columns found")
    return features


def read_unique_times(files: list[Path], batch_size: int) -> np.ndarray:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    chunks = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["time_id"]):
            chunks.append(np.unique(batch.column(0).to_numpy(zero_copy_only=False)))
    if not chunks:
        raise ValueError("no time_id values found")
    return np.unique(np.concatenate(chunks)).astype(np.int64)


def split_periods(
    unique_times: np.ndarray,
    valid_time_fraction: float,
    num_periods: int,
    max_rows_per_period: int,
    max_valid_rows: int,
    assets: int,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    if num_periods <= 0:
        raise ValueError("num-periods must be positive")
    if not 0.0 < valid_time_fraction < 1.0:
        raise ValueError("valid-time-fraction must be between 0 and 1")
    valid_count = max(1, int(round(len(unique_times) * valid_time_fraction)))
    valid_count = min(valid_count, len(unique_times) - 1)
    cutoff = int(unique_times[-valid_count])
    train_times = unique_times[unique_times < cutoff]
    valid_times = unique_times[unique_times >= cutoff]
    if len(train_times) < num_periods:
        raise ValueError("not enough training time_id values for requested num-periods")

    period_chunks = [chunk.astype(np.int64) for chunk in np.array_split(train_times, num_periods) if len(chunk) > 0]
    if max_rows_per_period > 0:
        cap = max(1, max_rows_per_period // max(1, assets))
        period_chunks = [chunk[-cap:].astype(np.int64) for chunk in period_chunks]
    if max_valid_rows > 0:
        valid_cap = max(1, max_valid_rows // max(1, assets))
        valid_times = valid_times[:valid_cap]
    if len(valid_times) == 0:
        raise ValueError("validation time set is empty")
    return period_chunks, valid_times.astype(np.int64), cutoff


def split_plan(periods: list[np.ndarray], valid_times: np.ndarray, cutoff: int, assets: int) -> dict[str, Any]:
    period_rows = []
    for period_idx, period_times in enumerate(periods):
        period_rows.append(
            {
                "period": period_idx,
                "time_start": int(period_times[0]),
                "time_end": int(period_times[-1]),
                "time_count": int(len(period_times)),
                "estimated_rows": int(len(period_times) * assets),
            }
        )
    return {
        "num_periods": len(periods),
        "valid_cutoff_time_id": int(cutoff),
        "periods": period_rows,
        "validation": {
            "time_start": int(valid_times[0]),
            "time_end": int(valid_times[-1]),
            "time_count": int(len(valid_times)),
            "estimated_rows": int(len(valid_times) * assets),
        },
    }


def coerce_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col.startswith("feature_"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["asset_id"] = pd.to_numeric(frame["asset_id"], errors="coerce").fillna(0).astype("int16")
    frame["time_id"] = pd.to_numeric(frame["time_id"], errors="coerce").fillna(0).astype("int64")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).astype("float32")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).astype("float32")
    return frame


def load_time_set(files: list[Path], feature_subset: list[str], selected_times: np.ndarray, batch_size: int) -> pd.DataFrame:
    import pyarrow.parquet as pq

    columns = ["time_id", "asset_id", "weight", "target", *feature_subset]
    selected = set(int(value) for value in selected_times.tolist())
    frames: list[pd.DataFrame] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            chunk = batch.to_pandas()
            mask = np.fromiter((int(value) in selected for value in chunk["time_id"].to_numpy()), dtype=bool, count=len(chunk))
            if mask.any():
                frames.append(coerce_frame(chunk.loc[mask].copy(), columns))
    if not frames:
        raise ValueError("selected time set produced no rows")
    return pd.concat(frames, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)


def split_frame_by_time_tail(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("inner-valid-fraction must be between 0 and 1")
    times = np.sort(frame["time_id"].unique())
    if len(times) < 2:
        midpoint = max(1, int(len(frame) * (1.0 - fraction)))
        return frame.iloc[:midpoint].copy(), frame.iloc[midpoint:].copy()
    valid_count = max(1, int(round(len(times) * fraction)))
    valid_count = min(valid_count, len(times) - 1)
    cutoff = times[-valid_count]
    fit = frame.loc[frame["time_id"] < cutoff].copy()
    valid = frame.loc[frame["time_id"] >= cutoff].copy()
    if len(fit) == 0 or len(valid) == 0:
        midpoint = max(1, int(len(frame) * (1.0 - fraction)))
        fit = frame.iloc[:midpoint].copy()
        valid = frame.iloc[midpoint:].copy()
    return fit, valid


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    if any(value <= 0 for value in values):
        raise ValueError("all numeric list values must be positive")
    return values


def ema_column_names(ema_features: list[str], halflives: list[float]) -> list[str]:
    labels = [str(int(value)) if float(value).is_integer() else str(value).replace(".", "p") for value in halflives]
    columns: list[str] = []
    for label in labels:
        for feature in ema_features:
            columns.append(f"ema_gap_h{label}_{feature}")
    if len(labels) >= 2:
        for feature in ema_features:
            columns.append(f"ema_spread_h{labels[0]}_h{labels[-1]}_{feature}")
    return columns


def ema_matrix(frame: pd.DataFrame, ema_features: list[str], halflives: list[float]) -> np.ndarray:
    if not ema_features:
        return np.empty((len(frame), 0), dtype=np.float32)

    values = frame.loc[:, ema_features].to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    asset_ids = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
    unique_assets = sorted(set(int(asset) for asset in asset_ids.tolist()))
    asset_to_idx = {asset: idx for idx, asset in enumerate(unique_assets)}
    alphas = np.asarray([1.0 - 0.5 ** (1.0 / float(half_life)) for half_life in halflives], dtype=np.float32)
    state = np.zeros((len(unique_assets), len(halflives), len(ema_features)), dtype=np.float32)
    initialized = np.zeros(len(unique_assets), dtype=bool)
    out = np.empty((len(frame), len(ema_column_names(ema_features, halflives))), dtype=np.float32)

    for row_idx, asset_id in enumerate(asset_ids):
        asset_idx = asset_to_idx[int(asset_id)]
        current = values[row_idx]
        previous = state[asset_idx].copy()
        if not initialized[asset_idx]:
            previous[:] = current
            state[asset_idx] = previous
            initialized[asset_idx] = True
        pieces = [current - previous[half_idx] for half_idx in range(len(halflives))]
        if len(halflives) >= 2:
            pieces.append(previous[0] - previous[-1])
        out[row_idx] = np.concatenate(pieces).astype(np.float32)
        for half_idx, alpha in enumerate(alphas):
            state[asset_idx, half_idx] = alpha * current + (1.0 - alpha) * state[asset_idx, half_idx]
    return out


def input_columns(features: list[str], ema_features: list[str], halflives: list[float]) -> list[str]:
    return [*features, *ema_column_names(ema_features, halflives), "asset_id"]


def build_matrix(
    frame: pd.DataFrame,
    features: list[str],
    ema_features: list[str] | None = None,
    halflives: list[float] | None = None,
) -> np.ndarray:
    ema_features = ema_features or []
    halflives = halflives or []
    x = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    asset = frame["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
    return np.hstack([x, ema_matrix(frame, ema_features, halflives), asset])


def weighted_l2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    denominator = np.sum(weight)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if denominator <= 0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.sum(weight * (y_true - y_pred) ** 2) / denominator)


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def parse_alpha_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("alpha-grid must contain values")
    return values


def optimize_prediction_scale(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, alphas: list[float]) -> tuple[float, float, float]:
    best_alpha = float(alphas[0])
    best_l2 = weighted_l2(y_true, y_pred * best_alpha, weight)
    best_score = weighted_zero_mean_r2(y_true, y_pred * best_alpha, weight)
    for alpha in alphas[1:]:
        scaled = y_pred * float(alpha)
        score = weighted_zero_mean_r2(y_true, scaled, weight)
        if score > best_score:
            best_alpha = float(alpha)
            best_l2 = weighted_l2(y_true, scaled, weight)
            best_score = score
    return best_alpha, best_l2, best_score


def lgb_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "num_threads": int(args.threads),
        "seed": int(args.seed),
    }


def train_lgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    rounds: int,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, float]:
    require_package("lightgbm")
    import lightgbm as lgb

    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_set, free_raw_data=True)
    model = lgb.train(
        lgb_params(args),
        train_set,
        num_boost_round=int(rounds),
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(int(args.early_stopping), verbose=False), lgb.log_evaluation(period=0)],
    )
    pred = np.asarray(model.predict(x_valid, num_iteration=model.best_iteration), dtype=np.float64)
    return model, pred, weighted_zero_mean_r2(y_valid, pred, w_valid)


def train_lgb_fixed_rounds(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    rounds: int,
    args: argparse.Namespace,
) -> Any:
    require_package("lightgbm")
    import lightgbm as lgb

    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=True)
    return lgb.train(
        lgb_params(args),
        train_set,
        num_boost_round=max(1, int(rounds)),
        callbacks=[lgb.log_evaluation(period=0)],
    )


def model_importance(model: Any, features: list[str]) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain").astype(np.float64)
    split = model.feature_importance(importance_type="split").astype(np.float64)
    return pd.DataFrame(
        {
            "feature": [*features, "asset_id"],
            "importance_gain": gain,
            "importance_split": split,
            "importance": gain,
        }
    )


def period_weights(num_periods: int, weighting: str, decay: float) -> np.ndarray:
    if num_periods <= 0:
        raise ValueError("num_periods must be positive")
    if weighting == "equal":
        weights = np.ones(num_periods, dtype=np.float64)
    elif weighting == "linear_recent":
        weights = np.arange(1, num_periods + 1, dtype=np.float64)
    elif weighting == "exp_recent":
        if not 0.0 < decay <= 1.0:
            raise ValueError("period-weight-decay must be in (0, 1] for exp_recent")
        ages = np.arange(num_periods - 1, -1, -1, dtype=np.float64)
        weights = np.power(float(decay), ages)
    else:
        raise ValueError(f"unknown period weighting: {weighting}")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("period weights sum to zero")
    return weights / total


def weighted_importance(
    period_frames: list[pd.DataFrame],
    feature_count: int,
    weighting: str,
    decay: float,
) -> pd.DataFrame:
    rows = []
    for period_idx, frame in enumerate(period_frames):
        current = frame.copy()
        if "period" not in current.columns:
            current["period"] = period_idx
        rows.append(current)
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.loc[raw["feature"] != "asset_id"].copy()
    raw["period"] = raw["period"].astype(int)
    num_periods = int(raw["period"].max()) + 1
    weights = period_weights(num_periods, weighting, decay)
    weight_frame = pd.DataFrame({"period": np.arange(num_periods, dtype=int), "period_weight": weights})
    weighted = raw.merge(weight_frame, on="period", how="left")
    weighted["weighted_piece"] = weighted["importance"].astype(float) * weighted["period_weight"].astype(float)

    last_period = int(weight_frame["period"].max())
    first_period = int(weight_frame["period"].min())
    last_importance = weighted.loc[weighted["period"] == last_period, ["feature", "importance"]].rename(
        columns={"importance": "importance_last"}
    )
    first_importance = weighted.loc[weighted["period"] == first_period, ["feature", "importance"]].rename(
        columns={"importance": "importance_first"}
    )

    summary = (
        weighted.groupby("feature", as_index=False)
        .agg(
            weighted_importance=("weighted_piece", "sum"),
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_max=("importance", "max"),
            period_weight_sum=("period_weight", "sum"),
            period_count=("period", "count"),
        )
    )
    summary = summary.merge(last_importance, on="feature", how="left").merge(first_importance, on="feature", how="left")
    summary["importance_std"] = summary["importance_std"].fillna(0.0)
    summary["importance_last"] = summary["importance_last"].fillna(0.0)
    summary["importance_first"] = summary["importance_first"].fillna(0.0)
    summary["importance_trend"] = summary["importance_last"] - summary["importance_first"]
    summary = summary.sort_values("weighted_importance", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    if feature_count > 0:
        summary["selected"] = summary["rank"] <= int(feature_count)
    else:
        summary["selected"] = True
    return summary


def prediction_clip_bounds(pred: np.ndarray) -> tuple[float, float]:
    finite = pred[np.isfinite(pred)]
    if len(finite) == 0:
        return -1.0, 1.0
    lower, upper = np.quantile(finite, [0.001, 0.999])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return -1.0, 1.0
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    files = manifest_files(data_root, "train")
    all_features = feature_columns(files)
    unique_times = read_unique_times(files, args.batch_size)
    assets = n_assets(data_root)
    periods, valid_times, cutoff = split_periods(
        unique_times,
        args.valid_time_fraction,
        args.num_periods,
        args.max_rows_per_period,
        args.max_valid_rows,
        assets,
    )
    plan = split_plan(periods, valid_times, cutoff, assets)
    if args.dry_run:
        print(json.dumps({"strategy": "period_stable_lgb_strategy", "feature_count": len(all_features), **plan}, indent=2))
        return
    valid_full = load_time_set(files, all_features, valid_times, args.batch_size)
    x_valid_full = build_matrix(valid_full, all_features)
    y_valid = valid_full["target"].to_numpy(dtype=np.float32, copy=False)
    w_valid = np.maximum(valid_full["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)

    period_metrics = []
    period_importances = []
    for period_idx, period_times in enumerate(periods):
        train_frame = load_time_set(files, all_features, period_times, args.batch_size)
        probe_fit, probe_valid = split_frame_by_time_tail(train_frame, args.inner_valid_fraction)
        x_probe_fit = build_matrix(probe_fit, all_features)
        x_probe_valid = build_matrix(probe_valid, all_features)
        y_probe_fit = probe_fit["target"].to_numpy(dtype=np.float32, copy=False)
        w_probe_fit = np.maximum(probe_fit["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
        y_probe_valid = probe_valid["target"].to_numpy(dtype=np.float32, copy=False)
        w_probe_valid = np.maximum(probe_valid["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
        probe_model, inner_pred, inner_score = train_lgb(
            x_probe_fit,
            y_probe_fit,
            w_probe_fit,
            x_probe_valid,
            y_probe_valid,
            w_probe_valid,
            args.probe_rounds,
            args,
        )
        x_period_full = build_matrix(train_frame, all_features)
        y_period_full = train_frame["target"].to_numpy(dtype=np.float32, copy=False)
        w_period_full = np.maximum(train_frame["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
        model = train_lgb_fixed_rounds(
            x_period_full,
            y_period_full,
            w_period_full,
            int(probe_model.best_iteration or args.probe_rounds),
            args,
        )
        future_pred = np.asarray(model.predict(x_valid_full), dtype=np.float64)
        future_l2 = weighted_l2(y_valid, future_pred, w_valid)
        future_score = weighted_zero_mean_r2(y_valid, future_pred, w_valid)
        period_metrics.append(
            {
                "period": period_idx,
                "time_start": int(period_times[0]),
                "time_end": int(period_times[-1]),
                "time_count": int(len(period_times)),
                "train_rows": int(len(train_frame)),
                "inner_train_rows": int(len(probe_fit)),
                "inner_valid_rows": int(len(probe_valid)),
                "inner_valid_l2": weighted_l2(y_probe_valid, inner_pred, w_probe_valid),
                "inner_valid_score": inner_score,
                "best_iteration": int(probe_model.best_iteration or args.probe_rounds),
                "future_valid_l2": future_l2,
                "future_valid_score": future_score,
            }
        )
        importance = model_importance(model, all_features)
        importance["period"] = period_idx
        period_importances.append(importance)
        del train_frame, probe_fit, probe_valid, x_probe_fit, x_probe_valid, x_period_full

    period_metrics_df = pd.DataFrame(period_metrics)
    period_importance_df = pd.concat(period_importances, ignore_index=True)
    weighted_feature_count = int(args.weighted_feature_count) if args.weighted_feature_count is not None else int(args.stable_feature_count)
    weighted_df = weighted_importance(
        period_importances,
        weighted_feature_count,
        args.period_weighting,
        args.period_weight_decay,
    )
    selected_features = weighted_df.loc[weighted_df["selected"], "feature"].astype(str).tolist()
    if not selected_features:
        selected_features = all_features[: min(len(all_features), max(1, weighted_feature_count))]
    ema_halflives = parse_float_list(args.ema_halflives)
    ema_feature_count = max(0, int(args.ema_feature_count))
    ema_features = selected_features[: min(len(selected_features), ema_feature_count)]

    train_final_frames = [load_time_set(files, selected_features, period_times, args.batch_size) for period_times in periods]
    train_final = pd.concat(train_final_frames, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    final_fit, final_inner_valid = split_frame_by_time_tail(train_final, args.inner_valid_fraction)
    x_train_final = build_matrix(final_fit, selected_features, ema_features, ema_halflives)
    x_inner_final = build_matrix(final_inner_valid, selected_features, ema_features, ema_halflives)
    x_valid_final = build_matrix(valid_full, selected_features, ema_features, ema_halflives)
    y_train_final = final_fit["target"].to_numpy(dtype=np.float32, copy=False)
    w_train_final = np.maximum(final_fit["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
    y_inner_final = final_inner_valid["target"].to_numpy(dtype=np.float32, copy=False)
    w_inner_final = np.maximum(final_inner_valid["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
    final_selector_model, final_inner_pred, final_inner_score = train_lgb(
        x_train_final,
        y_train_final,
        w_train_final,
        x_inner_final,
        y_inner_final,
        w_inner_final,
        args.final_rounds,
        args,
    )
    x_train_full_final = build_matrix(train_final, selected_features, ema_features, ema_halflives)
    y_train_full_final = train_final["target"].to_numpy(dtype=np.float32, copy=False)
    w_train_full_final = np.maximum(train_final["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
    final_best_iteration = int(final_selector_model.best_iteration or args.final_rounds)
    final_model = train_lgb_fixed_rounds(
        x_train_full_final,
        y_train_full_final,
        w_train_full_final,
        final_best_iteration,
        args,
    )
    final_pred_raw = np.asarray(final_model.predict(x_valid_final), dtype=np.float64)
    prediction_scale, final_l2, final_score = optimize_prediction_scale(
        y_valid,
        final_pred_raw,
        w_valid,
        parse_alpha_grid(args.alpha_grid),
    )
    scaled_pred = final_pred_raw * prediction_scale
    clip_min, clip_max = prediction_clip_bounds(scaled_pred)

    final_model.save_model(str(model_dir / "final_lightgbm.txt"))
    period_metrics_df.to_csv(model_dir / "period_metrics.csv", index=False)
    period_importance_df.to_csv(model_dir / "period_feature_importance.csv", index=False)
    weighted_df.to_csv(model_dir / "weighted_feature_importance.csv", index=False)
    weighted_df.to_csv(model_dir / "stable_feature_importance.csv", index=False)

    metadata = {
        "strategy": "period_stable_lgb_strategy",
        "feature_selection_method": "weighted_period_importance",
        "feature_columns": selected_features,
        "ema_feature_columns": ema_features,
        "ema_halflives": ema_halflives,
        "input_columns": input_columns(selected_features, ema_features, ema_halflives),
        "model_file": "final_lightgbm.txt",
        "num_periods": int(args.num_periods),
        "valid_time_fraction": float(args.valid_time_fraction),
        "inner_valid_fraction": float(args.inner_valid_fraction),
        "valid_cutoff_time_id": int(cutoff),
        "split_plan": plan,
        "weighted_feature_count": weighted_feature_count,
        "period_weighting": args.period_weighting,
        "period_weight_decay": float(args.period_weight_decay),
        "stable_feature_count": int(args.stable_feature_count),
        "min_periods_for_stable": int(args.min_periods_for_stable),
        "period_metrics_file": "period_metrics.csv",
        "period_feature_importance_file": "period_feature_importance.csv",
        "weighted_feature_importance_file": "weighted_feature_importance.csv",
        "stable_feature_importance_file": "weighted_feature_importance.csv",
        "train_rows": int(len(train_final)),
        "final_inner_train_rows": int(len(final_fit)),
        "final_inner_valid_rows": int(len(final_inner_valid)),
        "final_best_iteration": final_best_iteration,
        "valid_rows": int(len(valid_full)),
        "zero_l2": weighted_l2(y_valid, np.zeros_like(y_valid, dtype=np.float64), w_valid),
        "zero_score": weighted_zero_mean_r2(y_valid, np.zeros_like(y_valid, dtype=np.float64), w_valid),
        "final_l2_raw": weighted_l2(y_valid, final_pred_raw, w_valid),
        "final_score_raw": weighted_zero_mean_r2(y_valid, final_pred_raw, w_valid),
        "final_inner_l2": weighted_l2(y_inner_final, final_inner_pred, w_inner_final),
        "final_inner_score": final_inner_score,
        "prediction_scale": prediction_scale,
        "final_l2": final_l2,
        "final_score": final_score,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "top_weighted_features": json.loads(weighted_df.head(50).to_json(orient="records")),
        "top_stable_features": json.loads(weighted_df.head(50).to_json(orient="records")),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "strategy": metadata["strategy"],
                "train_rows": metadata["train_rows"],
                "valid_rows": metadata["valid_rows"],
                "selected_feature_count": len(selected_features),
                "ema_feature_count": len(ema_features),
                "feature_selection_method": metadata["feature_selection_method"],
                "period_weighting": metadata["period_weighting"],
                "period_future_score_mean": float(period_metrics_df["future_valid_score"].mean()),
                "period_future_score_min": float(period_metrics_df["future_valid_score"].min()),
                "final_inner_score": metadata["final_inner_score"],
                "final_score_raw": metadata["final_score_raw"],
                "prediction_scale": metadata["prediction_scale"],
                "final_score": metadata["final_score"]
                #"top_stable_features": metadata["top_stable_features"][:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
