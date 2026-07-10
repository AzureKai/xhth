from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight LightGBM/CatBoost baseline strategy.")
    parser.add_argument("--data-root", required=True, help="Release data root containing manifest.json.")
    parser.add_argument("--model-dir", required=True, help="Directory where model files will be written.")
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Row sample fraction after time split.")
    parser.add_argument("--max-train-rows", type=int, default=1000_000, help="Maximum sampled train rows; use 0 for no cap.")
    parser.add_argument("--max-valid-rows", type=int, default=300_000, help="Maximum sampled validation rows; use 0 for no cap.")
    parser.add_argument("--batch-size", type=int, default=65_536, help="Parquet streaming batch size.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-lightgbm", type=int, default=1)
    parser.add_argument("--train-catboost", type=int, default=1)
    parser.add_argument("--lgb-rounds", type=int, default=1200)
    parser.add_argument("--lgb-early-stopping", type=int, default=80)
    parser.add_argument("--catboost-iterations", type=int, default=800)
    parser.add_argument("--catboost-early-stopping", type=int, default=80)
    parser.add_argument("--alpha-grid", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5")
    parser.add_argument("--xs-feature-count", type=int, default=50, help="Number of features used for cross-sectional rank/demean.")
    parser.add_argument("--xs-features", default="", help="Comma-separated explicit feature list for cross-sectional features.")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def require_package(package: str, install_name: str | None = None) -> None:
    if importlib.util.find_spec(package) is None:
        name = install_name or package
        raise ImportError(f"missing optional dependency '{name}'. Install it or disable the related model.")


def manifest_files(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [data_root / str(file) for file in files]
    return sorted((data_root / split).glob("*.parquet"))


def manifest_train_rows(data_root: Path) -> int | None:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", {}).get("train")
    return int(rows) if rows is not None else None


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return pd.read_parquet(path).columns.tolist()


def feature_columns(files: list[Path]) -> list[str]:
    if not files:
        raise ValueError("no parquet files found")
    columns = parquet_columns(files[0])
    features = [col for col in columns if str(col).startswith("feature_")]
    if not features:
        raise ValueError("no feature_* columns found")
    return features


def coerce_train_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["asset_id"] = pd.to_numeric(frame["asset_id"], errors="coerce").fillna(0).astype("int16")
    frame["time_id"] = pd.to_numeric(frame["time_id"], errors="coerce").fillna(0).astype("int64")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).astype("float32")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).astype("float32")
    return frame


def read_unique_times(files: list[Path], batch_size: int) -> np.ndarray:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    chunks = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["time_id"]):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            chunks.append(np.unique(values))
    if not chunks:
        raise ValueError("no time_id values found")
    return np.unique(np.concatenate(chunks)).astype(np.int64)


def time_split_cutoff(files: list[Path], valid_time_fraction: float, batch_size: int) -> int:
    if not 0.0 < valid_time_fraction < 1.0:
        raise ValueError("valid-time-fraction must be between 0 and 1")
    unique_times = read_unique_times(files, batch_size=batch_size)
    if len(unique_times) < 2:
        raise ValueError("at least two time_id values are required for validation split")
    valid_count = max(1, int(round(len(unique_times) * valid_time_fraction)))
    valid_count = min(valid_count, len(unique_times) - 1)
    return int(unique_times[-valid_count])


def sample_probability(sample_frac: float, max_rows: int, estimated_rows: int | None) -> float:
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError("sample-frac must be in (0, 1]")
    probability = float(sample_frac)
    if max_rows > 0 and estimated_rows and estimated_rows > max_rows:
        probability = min(probability, max_rows / float(estimated_rows))
    return max(probability, 0.0)


def maybe_cap_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_train_valid_frames(
    data_root: Path,
    features: list[str],
    valid_time_fraction: float,
    sample_frac: float,
    max_train_rows: int,
    max_valid_rows: int,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float, float]:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    files = manifest_files(data_root, "train")
    if not files:
        raise ValueError(f"no train parquet files found under {data_root}")

    cutoff_time = time_split_cutoff(files, valid_time_fraction, batch_size=batch_size)
    total_rows = manifest_train_rows(data_root)
    estimated_train_rows = int(total_rows * (1.0 - valid_time_fraction)) if total_rows else None
    estimated_valid_rows = int(total_rows * valid_time_fraction) if total_rows else None
    train_probability = sample_probability(sample_frac, max_train_rows, estimated_train_rows)
    valid_probability = sample_probability(sample_frac, max_valid_rows, estimated_valid_rows)

    columns = ["time_id", "asset_id", "weight", "target", *features]
    rng = np.random.default_rng(seed)
    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []

    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            chunk = batch.to_pandas()
            train_mask = chunk["time_id"].to_numpy() < cutoff_time
            if train_mask.any() and train_probability > 0.0:
                part = chunk.loc[train_mask]
                if train_probability < 1.0:
                    keep = rng.random(len(part)) < train_probability
                    part = part.loc[keep]
                if not part.empty:
                    train_frames.append(coerce_train_frame(part.copy(), features))
            valid_mask = ~train_mask
            if valid_mask.any() and valid_probability > 0.0:
                part = chunk.loc[valid_mask]
                if valid_probability < 1.0:
                    keep = rng.random(len(part)) < valid_probability
                    part = part.loc[keep]
                if not part.empty:
                    valid_frames.append(coerce_train_frame(part.copy(), features))

    if not train_frames or not valid_frames:
        raise ValueError("streaming load produced an empty train or validation frame")

    train = pd.concat(train_frames, ignore_index=True)
    valid = pd.concat(valid_frames, ignore_index=True)
    train = maybe_cap_rows(train, max_train_rows, seed)
    valid = maybe_cap_rows(valid, max_valid_rows, seed + 1)
    if train.empty or valid.empty:
        raise ValueError("row sampling produced an empty train or validation frame")
    return train, valid, cutoff_time, train_probability, valid_probability


def select_xs_features(model_dir: Path, features: list[str], count: int, explicit: str) -> list[str]:
    if count <= 0:
        return []
    if explicit.strip():
        selected = [item.strip() for item in explicit.split(",") if item.strip()]
        missing = [feature for feature in selected if feature not in features]
        if missing:
            raise ValueError(f"xs-features contains unknown feature columns: {missing[:5]}")
        return selected[:count]

    importance_path = model_dir / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        selected = [
            str(feature)
            for feature in importance.get("feature", pd.Series(dtype=str)).tolist()
            if str(feature) in features
        ]
        if selected:
            return selected[:count]
    return features[:count]


def input_columns(features: list[str], xs_features: list[str], use_asset_id: bool = True) -> list[str]:
    columns = list(features)
    for feature in xs_features:
        columns.append(f"xs_demean_{feature}")
        columns.append(f"xs_rank_{feature}")
    if use_asset_id:
        columns.append("asset_id")
    return columns


def cross_section_matrix(frame: pd.DataFrame, xs_features: list[str]) -> np.ndarray:
    if not xs_features:
        return np.empty((len(frame), 0), dtype=np.float32)

    values = frame.loc[:, xs_features].astype("float32")
    means = values.groupby(frame["time_id"], sort=False).transform("mean")
    demean = (values - means).to_numpy(dtype=np.float32, copy=True)
    rank = values.groupby(frame["time_id"], sort=False).rank(method="average", pct=True).to_numpy(dtype=np.float32, copy=True)
    demean = np.nan_to_num(demean, nan=0.0, posinf=0.0, neginf=0.0)
    rank = np.nan_to_num(rank, nan=0.5, posinf=1.0, neginf=0.0)

    matrix = np.empty((len(frame), len(xs_features) * 2), dtype=np.float32)
    matrix[:, 0::2] = demean
    matrix[:, 1::2] = rank
    return matrix


def build_matrix(
    frame: pd.DataFrame,
    features: list[str],
    xs_features: list[str],
    use_asset_id: bool = True,
) -> np.ndarray:
    x = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    parts = [x]
    if xs_features:
        parts.append(cross_section_matrix(frame, xs_features))
    if use_asset_id:
        asset = frame["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        parts.append(asset)
    return np.hstack(parts)


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def weighted_l2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(weight)
    if denominator <= 0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.sum(weight * (y_true - y_pred) ** 2) / denominator)


def parse_alpha_grid(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("alpha-grid must contain at least one numeric value")
    return values


def optimize_prediction_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, float, float]:
    best_alpha = float(alpha_grid[0])
    best_l2 = weighted_l2(y_true, y_pred * best_alpha, weight)
    best_score = weighted_zero_mean_r2(y_true, y_pred * best_alpha, weight)
    for alpha in alpha_grid[1:]:
        current_pred = y_pred * float(alpha)
        current_l2 = weighted_l2(y_true, current_pred, weight)
        current_score = weighted_zero_mean_r2(y_true, current_pred, weight)
        if current_score > best_score:
            best_alpha = float(alpha)
            best_l2 = current_l2
            best_score = current_score
    return best_alpha, best_l2, best_score


def prediction_clip_bounds(predictions: list[np.ndarray]) -> tuple[float, float]:
    if not predictions:
        return -1.0, 1.0
    merged = np.concatenate([np.asarray(pred, dtype=np.float64) for pred in predictions])
    merged = merged[np.isfinite(merged)]
    if len(merged) == 0:
        return -1.0, 1.0
    lower, upper = np.quantile(merged, [0.001, 0.999])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return -1.0, 1.0
    return float(lower), float(upper)


def fit_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, float]:
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
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "num_threads": int(args.threads),
        "seed": int(args.seed),
    }
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_set, free_raw_data=True)
    callbacks = [
        lgb.early_stopping(stopping_rounds=int(args.lgb_early_stopping), verbose=True),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(args.lgb_rounds),
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    pred_valid = model.predict(x_valid, num_iteration=model.best_iteration)
    score = weighted_zero_mean_r2(y_valid, pred_valid, w_valid)
    return model, np.asarray(pred_valid, dtype=np.float64), score


def fit_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, float]:
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
        l2_leaf_reg=8.0,
        random_seed=int(args.seed),
        thread_count=int(args.threads),
        od_type="Iter",
        od_wait=int(args.catboost_early_stopping),
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    pred_valid = model.predict(valid_pool)
    score = weighted_zero_mean_r2(y_valid, pred_valid, w_valid)
    return model, np.asarray(pred_valid, dtype=np.float64), score


def ensemble_weights(model_specs: list[dict[str, Any]]) -> list[float]:
    if not model_specs:
        return []
    positive = np.asarray([max(float(spec["valid_score"]), 0.0) for spec in model_specs], dtype=np.float64)
    if positive.sum() <= 0:
        return [1.0 / len(model_specs)] * len(model_specs)
    weights = positive / positive.sum()
    return weights.tolist()


def lightgbm_importance(model: Any, input_columns: list[str]) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain").astype(np.float64)
    split = model.feature_importance(importance_type="split").astype(np.float64)
    return pd.DataFrame(
        {
            "feature": input_columns,
            "model": "lightgbm",
            "importance_gain": gain,
            "importance_split": split,
            "importance": gain,
        }
    )


def catboost_importance(model: Any, input_columns: list[str]) -> pd.DataFrame:
    importance = np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64)
    return pd.DataFrame(
        {
            "feature": input_columns,
            "model": "catboost",
            "importance_gain": np.nan,
            "importance_split": np.nan,
            "importance": importance,
        }
    )


def save_feature_importance(
    model_dir: Path,
    importance_frames: list[pd.DataFrame],
    top_n: int = 50,
) -> list[dict[str, object]]:
    if not importance_frames:
        return []

    raw = pd.concat(importance_frames, ignore_index=True)
    raw_path = model_dir / "feature_importance_by_model.csv"
    raw.sort_values(["model", "importance"], ascending=[True, False]).to_csv(raw_path, index=False)

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
    summary_path = model_dir / "feature_importance.csv"
    summary.to_csv(summary_path, index=False)

    top = summary.head(top_n).copy()
    return json.loads(top.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_files = manifest_files(data_root, "train")
    features = feature_columns(train_files)
    xs_features = select_xs_features(model_dir, features, args.xs_feature_count, args.xs_features)
    train, valid, cutoff_time, train_probability, valid_probability = load_train_valid_frames(
        data_root=data_root,
        features=features,
        valid_time_fraction=args.valid_time_fraction,
        sample_frac=args.sample_frac,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    input_cols = input_columns(features, xs_features, use_asset_id=True)
    x_train = build_matrix(train, features, xs_features, use_asset_id=True)
    x_valid = build_matrix(valid, features, xs_features, use_asset_id=True)
    y_train = train["target"].to_numpy(dtype=np.float32, copy=False)
    y_valid = valid["target"].to_numpy(dtype=np.float32, copy=False)
    w_train = np.maximum(train["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)
    w_valid = np.maximum(valid["weight"].to_numpy(dtype=np.float32, copy=False), 0.0)

    zero_pred = np.zeros_like(y_valid, dtype=np.float64)
    zero_l2 = weighted_l2(y_valid, zero_pred, w_valid)
    zero_score = weighted_zero_mean_r2(y_valid, zero_pred, w_valid)
    model_specs: list[dict[str, Any]] = []
    valid_predictions: list[np.ndarray] = []
    importance_frames: list[pd.DataFrame] = []

    if args.train_lightgbm:
        lgb_model, lgb_pred, lgb_score = fit_lightgbm(x_train, y_train, w_train, x_valid, y_valid, w_valid, args)
        lgb_file = "lightgbm.txt"
        lgb_model.save_model(str(model_dir / lgb_file))
        model_specs.append(
            {"type": "lightgbm", "file": lgb_file, "valid_score": lgb_score, "valid_l2": weighted_l2(y_valid, lgb_pred, w_valid)}
        )
        valid_predictions.append(lgb_pred)
        importance_frames.append(lightgbm_importance(lgb_model, input_cols))

    if args.train_catboost:
        cat_model, cat_pred, cat_score = fit_catboost(x_train, y_train, w_train, x_valid, y_valid, w_valid, args)
        cat_file = "catboost.cbm"
        cat_model.save_model(str(model_dir / cat_file))
        model_specs.append(
            {"type": "catboost", "file": cat_file, "valid_score": cat_score, "valid_l2": weighted_l2(y_valid, cat_pred, w_valid)}
        )
        valid_predictions.append(cat_pred)
        importance_frames.append(catboost_importance(cat_model, input_cols))

    if not model_specs:
        raise ValueError("no models were trained; enable LightGBM and/or CatBoost")

    weights = ensemble_weights(model_specs)
    ensemble_pred = np.zeros(len(valid), dtype=np.float64)
    for weight, pred in zip(weights, valid_predictions):
        ensemble_pred += weight * pred
    ensemble_l2_raw = weighted_l2(y_valid, ensemble_pred, w_valid)
    ensemble_score_raw = weighted_zero_mean_r2(y_valid, ensemble_pred, w_valid)
    prediction_scale, ensemble_l2, ensemble_score = optimize_prediction_scale(
        y_valid,
        ensemble_pred,
        w_valid,
        parse_alpha_grid(args.alpha_grid),
    )
    scaled_ensemble_pred = ensemble_pred * prediction_scale
    clip_min, clip_max = prediction_clip_bounds([scaled_ensemble_pred])

    for spec, weight in zip(model_specs, weights):
        spec["weight"] = float(weight)
    top_feature_importance = save_feature_importance(model_dir, importance_frames)

    metadata = {
        "strategy": "lgb_catboost_strategy",
        "feature_columns": features,
        "xs_feature_columns": xs_features,
        "use_asset_id": True,
        "models": model_specs,
        "valid_time_fraction": float(args.valid_time_fraction),
        "valid_cutoff_time_id": int(cutoff_time),
        "sample_frac": float(args.sample_frac),
        "train_sample_probability": float(train_probability),
        "valid_sample_probability": float(valid_probability),
        "max_train_rows": int(args.max_train_rows),
        "max_valid_rows": int(args.max_valid_rows),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "zero_l2": zero_l2,
        "zero_score": zero_score,
        "alpha_grid": parse_alpha_grid(args.alpha_grid),
        "prediction_scale": prediction_scale,
        "ensemble_l2_raw": ensemble_l2_raw,
        "ensemble_score_raw": ensemble_score_raw,
        "ensemble_l2": ensemble_l2,
        "ensemble_score": ensemble_score,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "input_columns": input_cols,
        "feature_importance_files": {
            "summary": "feature_importance.csv",
            "by_model": "feature_importance_by_model.csv",
        },
        "top_feature_importance": top_feature_importance,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = {
        "strategy": metadata["strategy"],
        "train_rows": metadata["train_rows"],
        "valid_rows": metadata["valid_rows"],
        "valid_cutoff_time_id": metadata["valid_cutoff_time_id"],
        "train_sample_probability": metadata["train_sample_probability"],
        "valid_sample_probability": metadata["valid_sample_probability"],
        "xs_feature_columns": metadata["xs_feature_columns"],
        "zero_l2": metadata["zero_l2"],
        "zero_score": metadata["zero_score"],
        "models": model_specs,
        "prediction_scale": metadata["prediction_scale"],
        "ensemble_l2_raw": metadata["ensemble_l2_raw"],
        "ensemble_score_raw": metadata["ensemble_score_raw"],
        "ensemble_l2": metadata["ensemble_l2"],
        "ensemble_score": metadata["ensemble_score"],
        "clip_min": metadata["clip_min"],
        "clip_max": metadata["clip_max"],
        #"top_feature_importance": metadata["top_feature_importance"][:20],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
