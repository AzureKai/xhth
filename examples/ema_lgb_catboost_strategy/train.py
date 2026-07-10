from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EMA-enhanced LightGBM/CatBoost strategy.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--importance-path", default="examples/lgb_catboost_strategy/model/feature_importance.csv")
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--max-valid-rows", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ema-feature-count", type=int, default=50)
    parser.add_argument("--ema-features", default="")
    parser.add_argument("--ema-halflives", default="5,20,60")
    parser.add_argument("--train-lightgbm", type=int, default=1)
    parser.add_argument("--train-catboost", type=int, default=0)
    parser.add_argument("--lgb-rounds", type=int, default=1200)
    parser.add_argument("--lgb-early-stopping", type=int, default=80)
    parser.add_argument("--catboost-iterations", type=int, default=800)
    parser.add_argument("--catboost-early-stopping", type=int, default=80)
    parser.add_argument("--alpha-grid", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def require_package(package: str) -> None:
    if importlib.util.find_spec(package) is None:
        raise ImportError(f"missing dependency '{package}'")


def manifest(data_root: Path) -> dict[str, Any]:
    path = data_root / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def manifest_files(data_root: Path, split: str) -> list[Path]:
    data = manifest(data_root)
    files = data.get("files", {}).get(split, [])
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
        raise ValueError("no train parquet files found")
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


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def select_ema_features(args: argparse.Namespace, features: list[str]) -> list[str]:
    count = int(args.ema_feature_count)
    if count <= 0:
        return []
    if args.ema_features.strip():
        selected = [item.strip() for item in args.ema_features.split(",") if item.strip()]
        missing = [feature for feature in selected if feature not in features]
        if missing:
            raise ValueError(f"unknown ema-features: {missing[:5]}")
        return selected[:count]

    importance_path = Path(args.importance_path)
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        selected = [str(feature) for feature in importance["feature"].tolist() if str(feature) in features]
        if selected:
            return selected[:count]
    return features[:count]


def split_time_blocks(
    unique_times: np.ndarray,
    valid_time_fraction: float,
    max_train_rows: int,
    max_valid_rows: int,
    assets: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 < valid_time_fraction < 1.0:
        raise ValueError("valid-time-fraction must be between 0 and 1")
    valid_count = max(1, int(round(len(unique_times) * valid_time_fraction)))
    valid_count = min(valid_count, len(unique_times) - 1)
    cutoff = int(unique_times[-valid_count])
    train_times = unique_times[unique_times < cutoff]
    valid_times = unique_times[unique_times >= cutoff]

    if max_train_rows > 0:
        train_time_cap = max(1, max_train_rows // max(1, assets))
        train_times = train_times[-train_time_cap:]
    if max_valid_rows > 0:
        valid_time_cap = max(1, max_valid_rows // max(1, assets))
        valid_times = valid_times[:valid_time_cap]
    if len(train_times) == 0 or len(valid_times) == 0:
        raise ValueError("time block sampling produced an empty train or validation time set")
    return train_times.astype(np.int64), valid_times.astype(np.int64), cutoff


def coerce_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["asset_id"] = pd.to_numeric(frame["asset_id"], errors="coerce").fillna(0).astype("int16")
    frame["time_id"] = pd.to_numeric(frame["time_id"], errors="coerce").fillna(0).astype("int64")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).astype("float32")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).astype("float32")
    return frame


def load_selected_frames(
    files: list[Path],
    features: list[str],
    train_times: np.ndarray,
    valid_times: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyarrow.parquet as pq

    columns = ["time_id", "asset_id", "weight", "target", *features]
    train_set = set(int(value) for value in train_times.tolist())
    valid_set = set(int(value) for value in valid_times.tolist())
    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            chunk = batch.to_pandas()
            time_values = chunk["time_id"].to_numpy()
            train_mask = np.fromiter((int(value) in train_set for value in time_values), dtype=bool, count=len(chunk))
            valid_mask = np.fromiter((int(value) in valid_set for value in time_values), dtype=bool, count=len(chunk))
            if train_mask.any():
                train_frames.append(coerce_frame(chunk.loc[train_mask].copy(), features))
            if valid_mask.any():
                valid_frames.append(coerce_frame(chunk.loc[valid_mask].copy(), features))
    if not train_frames or not valid_frames:
        raise ValueError("selected time blocks produced empty train or validation data")
    train = pd.concat(train_frames, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    valid = pd.concat(valid_frames, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    return train, valid


def ema_column_names(ema_features: list[str], halflives: list[float]) -> list[str]:
    columns: list[str] = []
    labels = [str(int(h)) if float(h).is_integer() else str(h).replace(".", "p") for h in halflives]
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
    initialized = np.zeros((len(unique_assets), len(ema_features)), dtype=bool)
    out = np.empty((len(frame), len(ema_column_names(ema_features, halflives))), dtype=np.float32)

    for row_idx, asset_id in enumerate(asset_ids):
        asset_idx = asset_to_idx[int(asset_id)]
        current = values[row_idx]
        first = ~initialized[asset_idx]
        if first.any():
            state[asset_idx, :, first] = current[first]
            initialized[asset_idx, first] = True
        previous = state[asset_idx].copy()
        pieces = []
        for half_idx in range(len(halflives)):
            pieces.append(current - previous[half_idx])
        if len(halflives) >= 2:
            pieces.append(previous[0] - previous[-1])
        out[row_idx] = np.concatenate(pieces).astype(np.float32)
        for half_idx, alpha in enumerate(alphas):
            state[asset_idx, half_idx] = alpha * current + (1.0 - alpha) * state[asset_idx, half_idx]
    return out


def build_matrix(frame: pd.DataFrame, features: list[str], ema_features: list[str], halflives: list[float]) -> np.ndarray:
    raw = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    asset = frame["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
    return np.hstack([raw, ema_matrix(frame, ema_features, halflives), asset])


def input_columns(features: list[str], ema_features: list[str], halflives: list[float]) -> list[str]:
    return [*features, *ema_column_names(ema_features, halflives), "asset_id"]


def weighted_l2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    denominator = np.sum(weight)
    if denominator <= 0:
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return float(np.sum(weight * (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2) / denominator)


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


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


def fit_lightgbm(x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, w_valid: np.ndarray, args: argparse.Namespace) -> tuple[Any, np.ndarray, float]:
    require_package("lightgbm")
    import lightgbm as lgb

    params = {
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
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_set, free_raw_data=True)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(args.lgb_rounds),
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(int(args.lgb_early_stopping), verbose=True), lgb.log_evaluation(period=50)],
    )
    pred = np.asarray(model.predict(x_valid, num_iteration=model.best_iteration), dtype=np.float64)
    return model, pred, weighted_zero_mean_r2(y_valid, pred, w_valid)


def fit_catboost(x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, w_valid: np.ndarray, args: argparse.Namespace) -> tuple[Any, np.ndarray, float]:
    require_package("catboost")
    from catboost import CatBoostRegressor, Pool

    train_pool = Pool(x_train, label=y_train, weight=w_train)
    valid_pool = Pool(x_valid, label=y_valid, weight=w_valid)
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=int(args.catboost_iterations),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10.0,
        random_seed=int(args.seed),
        thread_count=int(args.threads),
        od_type="Iter",
        od_wait=int(args.catboost_early_stopping),
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    pred = np.asarray(model.predict(valid_pool), dtype=np.float64)
    return model, pred, weighted_zero_mean_r2(y_valid, pred, w_valid)


def ensemble_weights(model_specs: list[dict[str, Any]]) -> list[float]:
    positive = np.asarray([max(float(spec["valid_score"]), 0.0) for spec in model_specs], dtype=np.float64)
    if positive.sum() <= 0:
        return [1.0 / len(model_specs)] * len(model_specs)
    return (positive / positive.sum()).tolist()


def lightgbm_importance(model: Any, columns: list[str]) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain").astype(np.float64)
    split = model.feature_importance(importance_type="split").astype(np.float64)
    return pd.DataFrame({"feature": columns, "model": "lightgbm", "importance_gain": gain, "importance_split": split, "importance": gain})


def catboost_importance(model: Any, columns: list[str]) -> pd.DataFrame:
    importance = np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64)
    return pd.DataFrame({"feature": columns, "model": "catboost", "importance_gain": np.nan, "importance_split": np.nan, "importance": importance})


def save_feature_importance(model_dir: Path, frames: list[pd.DataFrame]) -> list[dict[str, object]]:
    if not frames:
        return []
    raw = pd.concat(frames, ignore_index=True)
    raw.sort_values(["model", "importance"], ascending=[True, False]).to_csv(model_dir / "feature_importance_by_model.csv", index=False)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_max=("importance", "max"),
            importance_gain_mean=("importance_gain", "mean"),
            importance_split_mean=("importance_split", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(model_dir / "feature_importance.csv", index=False)
    return json.loads(summary.head(50).to_json(orient="records"))


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

    train_files = manifest_files(data_root, "train")
    features = feature_columns(train_files)
    ema_features = select_ema_features(args, features)
    halflives = parse_float_list(args.ema_halflives)
    unique_times = read_unique_times(train_files, args.batch_size)
    train_times, valid_times, cutoff_time = split_time_blocks(
        unique_times,
        args.valid_time_fraction,
        args.max_train_rows,
        args.max_valid_rows,
        n_assets(data_root),
    )
    train, valid = load_selected_frames(train_files, features, train_times, valid_times, args.batch_size)

    columns = input_columns(features, ema_features, halflives)
    x_train = build_matrix(train, features, ema_features, halflives)
    x_valid = build_matrix(valid, features, ema_features, halflives)
    y_train = train["target"].to_numpy(dtype=np.float32, copy=False)
    y_valid = valid["target"].to_numpy(dtype=np.float32, copy=False)
    w_train = np.maximum(train["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
    w_valid = np.maximum(valid["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)

    zero_pred = np.zeros_like(y_valid, dtype=np.float64)
    model_specs: list[dict[str, Any]] = []
    valid_predictions: list[np.ndarray] = []
    importance_frames: list[pd.DataFrame] = []

    if args.train_lightgbm:
        model, pred, score = fit_lightgbm(x_train, y_train, w_train, x_valid, y_valid, w_valid, args)
        model_file = "lightgbm.txt"
        model.save_model(str(model_dir / model_file))
        model_specs.append({"type": "lightgbm", "file": model_file, "valid_score": score, "valid_l2": weighted_l2(y_valid, pred, w_valid)})
        valid_predictions.append(pred)
        importance_frames.append(lightgbm_importance(model, columns))

    if args.train_catboost:
        model, pred, score = fit_catboost(x_train, y_train, w_train, x_valid, y_valid, w_valid, args)
        model_file = "catboost.cbm"
        model.save_model(str(model_dir / model_file))
        model_specs.append({"type": "catboost", "file": model_file, "valid_score": score, "valid_l2": weighted_l2(y_valid, pred, w_valid)})
        valid_predictions.append(pred)
        importance_frames.append(catboost_importance(model, columns))

    if not model_specs:
        raise ValueError("no models were trained")

    weights = ensemble_weights(model_specs)
    ensemble_pred = np.zeros(len(valid), dtype=np.float64)
    for weight, pred in zip(weights, valid_predictions):
        ensemble_pred += weight * pred
    for spec, weight in zip(model_specs, weights):
        spec["weight"] = float(weight)

    raw_l2 = weighted_l2(y_valid, ensemble_pred, w_valid)
    raw_score = weighted_zero_mean_r2(y_valid, ensemble_pred, w_valid)
    prediction_scale, ensemble_l2, ensemble_score = optimize_prediction_scale(
        y_valid,
        ensemble_pred,
        w_valid,
        parse_float_list(args.alpha_grid),
    )
    scaled_pred = ensemble_pred * prediction_scale
    clip_min, clip_max = prediction_clip_bounds(scaled_pred)
    top_importance = save_feature_importance(model_dir, importance_frames)

    metadata = {
        "strategy": "ema_lgb_catboost_strategy",
        "feature_columns": features,
        "ema_feature_columns": ema_features,
        "ema_halflives": halflives,
        "input_columns": columns,
        "models": model_specs,
        "valid_time_fraction": float(args.valid_time_fraction),
        "valid_cutoff_time_id": int(cutoff_time),
        "train_time_count": int(len(train_times)),
        "valid_time_count": int(len(valid_times)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "zero_l2": weighted_l2(y_valid, zero_pred, w_valid),
        "zero_score": weighted_zero_mean_r2(y_valid, zero_pred, w_valid),
        "prediction_scale": prediction_scale,
        "ensemble_l2_raw": raw_l2,
        "ensemble_score_raw": raw_score,
        "ensemble_l2": ensemble_l2,
        "ensemble_score": ensemble_score,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "feature_importance_files": {
            "summary": "feature_importance.csv",
            "by_model": "feature_importance_by_model.csv",
        },
        "top_feature_importance": top_importance,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "strategy": metadata["strategy"],
        "train_rows": metadata["train_rows"],
        "valid_rows": metadata["valid_rows"],
        "ema_feature_columns": metadata["ema_feature_columns"],
        "ema_halflives": metadata["ema_halflives"],
        "zero_l2": metadata["zero_l2"],
        "models": model_specs,
        "prediction_scale": metadata["prediction_scale"],
        "ensemble_l2_raw": metadata["ensemble_l2_raw"],
        "ensemble_score_raw": metadata["ensemble_score_raw"],
        "ensemble_l2": metadata["ensemble_l2"],
        "ensemble_score": metadata["ensemble_score"],
        "clip_min": metadata["clip_min"],
        "clip_max": metadata["clip_max"],
        "top_feature_importance": metadata["top_feature_importance"][:20],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
