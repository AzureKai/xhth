from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_RESPONDERS = ["responder_02", "responder_03"]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit unusually predictable responder labels")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predict-sample-rows", type=int, default=100_000)
    parser.add_argument(
        "--max-rows-per-split", type=int, default=1_000_000,
        help="Rows used for per-feature moments in each of train and validation; 0 means all.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_array(cache_dir: Path, shard_id: int, suffix: str):
    return np.load(cache_dir / f"shard_{shard_id:05d}_{suffix}.npy", mmap_mode="r")


def feature_names(metadata):
    return [
        *metadata["feature_columns"],
        *metadata["temporal_feature_columns"],
        "asset_id",
    ]


def empty_moments(feature_count):
    return {
        "w": 0.0,
        "x": np.zeros(feature_count),
        "y": 0.0,
        "xx": np.zeros(feature_count),
        "xy": np.zeros(feature_count),
        "yy": 0.0,
        "rows": 0,
    }


def update_moments(moment, x, y, weight):
    valid = np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    weight = np.asarray(weight[valid], dtype=np.float64)
    moment["w"] += float(weight.sum())
    moment["x"] += np.sum(weight[:, None] * x, axis=0)
    moment["y"] += float(np.sum(weight * y))
    moment["xx"] += np.sum(weight[:, None] * x * x, axis=0)
    moment["xy"] += np.sum(weight[:, None] * x * y[:, None], axis=0)
    moment["yy"] += float(np.sum(weight * y * y))
    moment["rows"] += int(valid.sum())


def correlations(moment):
    weight = moment["w"]
    covariance = moment["xy"] - moment["x"] * moment["y"] / weight
    x_variance = moment["xx"] - moment["x"] ** 2 / weight
    y_variance = moment["yy"] - moment["y"] ** 2 / weight
    denominator = np.sqrt(np.maximum(x_variance, 0.0) * max(y_variance, 0.0))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)


def single_feature_linear_r2(train, valid):
    train_w = train["w"]
    x_mean = train["x"] / train_w
    y_mean = train["y"] / train_w
    covariance = train["xy"] - train["x"] * train["y"] / train_w
    variance = train["xx"] - train["x"] ** 2 / train_w
    slope = np.divide(covariance, variance, out=np.zeros_like(covariance), where=variance > 0)
    intercept = y_mean - slope * x_mean
    sse = (
        valid["yy"]
        + intercept * intercept * valid["w"]
        + slope * slope * valid["xx"]
        - 2.0 * intercept * valid["y"]
        - 2.0 * slope * valid["xy"]
        + 2.0 * intercept * slope * valid["x"]
    )
    constant_sse = (
        valid["yy"] + y_mean * y_mean * valid["w"] - 2.0 * y_mean * valid["y"]
    )
    zero_mean_r2 = 1.0 - sse / valid["yy"]
    incremental_r2 = 1.0 - sse / constant_sse
    constant_zero_mean_r2 = 1.0 - constant_sse / valid["yy"]
    return zero_mean_r2, incremental_r2, float(constant_zero_mean_r2)


def asset_baseline(cache_dir, metadata, responder_column, cutoff):
    sums = {}
    for shard in metadata["shards"]:
        shard_id = int(shard["id"])
        times = load_array(cache_dir, shard_id, "time")
        train_end = int(np.searchsorted(times, cutoff, side="left"))
        if train_end <= 0:
            continue
        x = load_array(cache_dir, shard_id, "x")[:train_end]
        asset = np.asarray(x[:, -1], dtype=np.int64)
        y = load_array(cache_dir, shard_id, "responder")[:train_end, responder_column]
        w = load_array(cache_dir, shard_id, "weight")[:train_end]
        valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
        for asset_id in np.unique(asset[valid]):
            mask = valid & (asset == asset_id)
            current = sums.setdefault(int(asset_id), [0.0, 0.0])
            current[0] += float(np.sum(w[mask] * y[mask]))
            current[1] += float(np.sum(w[mask]))
    means = {asset: total / weight for asset, (total, weight) in sums.items() if weight > 0}
    numerator = denominator = 0.0
    for shard in metadata["shards"]:
        shard_id = int(shard["id"])
        times = load_array(cache_dir, shard_id, "time")
        start = int(np.searchsorted(times, cutoff, side="left"))
        if start >= len(times):
            continue
        x = load_array(cache_dir, shard_id, "x")[start:]
        asset = np.asarray(x[:, -1], dtype=np.int64)
        y = load_array(cache_dir, shard_id, "responder")[start:, responder_column]
        w = load_array(cache_dir, shard_id, "weight")[start:]
        prediction = np.asarray([means.get(int(value), 0.0) for value in asset])
        valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
        numerator += float(np.sum(w[valid] * (y[valid] - prediction[valid]) ** 2))
        denominator += float(np.sum(w[valid] * y[valid] ** 2))
    return 1.0 - numerator / denominator if denominator > 0 else 0.0


def model_negative_controls(cache_dir, metadata, model_dir, responder, responder_column,
                            cutoff, max_rows, seed):
    try:
        import lightgbm as lgb
    except ImportError:
        return {"available": False, "reason": "lightgbm is not installed"}
    x_parts, y_parts, w_parts = [], [], []
    remaining = int(max_rows)
    for shard in metadata["shards"]:
        if remaining <= 0:
            break
        shard_id = int(shard["id"])
        times = load_array(cache_dir, shard_id, "time")
        start = int(np.searchsorted(times, cutoff, side="left"))
        take = min(remaining, len(times) - start)
        if take <= 0:
            continue
        x_parts.append(np.asarray(load_array(cache_dir, shard_id, "x")[start:start + take]))
        y_parts.append(np.asarray(load_array(cache_dir, shard_id, "responder")[start:start + take, responder_column]))
        w_parts.append(np.asarray(load_array(cache_dir, shard_id, "weight")[start:start + take]))
        remaining -= take
    x, y, weight = np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)
    model = lgb.Booster(model_file=str(model_dir / f"{responder}.txt"))
    prediction = model.predict(x)
    valid = np.isfinite(y) & np.isfinite(weight) & (weight > 0)

    def score(target):
        denominator = np.sum(weight[valid] * target[valid] ** 2)
        return float(1.0 - np.sum(weight[valid] * (target[valid] - prediction[valid]) ** 2) / denominator)

    rng = np.random.default_rng(seed)
    shuffled = y.copy()
    shuffled_values = shuffled[valid].copy()
    rng.shuffle(shuffled_values)
    shuffled[valid] = shuffled_values
    return {
        "available": True,
        "rows": len(y),
        "actual_score": score(y),
        "shuffled_label_score": score(shuffled),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
    }


def main():
    args = parse_args()
    work_dir, model_dir, output_dir = map(Path, (args.work_dir, args.model_dir, args.output_dir))
    cache_dir = work_dir / "cache"
    metadata = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    model_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    cutoff = int(model_metadata["valid_cutoff_time_id"])
    names = feature_names(metadata)
    if len(names) != load_array(cache_dir, 0, "x").shape[1]:
        raise ValueError("cache feature names do not match matrix width")
    compatibility_checks = {
        "cache_schema_version": (
            metadata.get("cache_schema_version")
            == model_metadata.get("cache_schema_version")
        ),
        "input_files": (
            metadata.get("input_files", [])
            == model_metadata.get("input_files", [])
        ),
        "temporal_engine_version": (
            metadata.get("temporal_engine_version")
            == model_metadata.get("temporal_engine_version")
        ),
        "temporal_plan_hash": (
            metadata.get("temporal_plan_hash", "")
            == model_metadata.get("temporal_plan_hash", "")
        ),
        "feature_columns": (
            metadata.get("feature_columns", [])
            == model_metadata.get("feature_columns", [])
        ),
        "temporal_feature_columns": (
            metadata.get("temporal_feature_columns", [])
            == model_metadata.get("temporal_feature_columns", [])
        ),
    }
    model_cache_compatible = all(compatibility_checks.values())
    if not model_cache_compatible:
        print(
            "WARNING: model metadata and cache schema do not match; "
            f"checks={compatibility_checks}",
            flush=True,
        )

    responder_indices = {name: metadata["responders"].index(name) for name in AUDIT_RESPONDERS}
    moments = {
        name: {"train": empty_moments(len(names)), "valid": empty_moments(len(names))}
        for name in AUDIT_RESPONDERS
    }
    boundary_checks = []
    sampled_rows = {"train": 0, "valid": 0}
    for shard_index, shard in enumerate(metadata["shards"], start=1):
        shard_id = int(shard["id"])
        x = load_array(cache_dir, shard_id, "x")
        time_id = load_array(cache_dir, shard_id, "time")
        responder = load_array(cache_dir, shard_id, "responder")
        weight = load_array(cache_dir, shard_id, "weight")
        boundary_checks.append(bool(np.all(time_id[:-1] <= time_id[1:])))
        for start in range(0, len(time_id), 16_384):
            end = min(start + 16_384, len(time_id))
            current_split = "train" if int(time_id[start]) < cutoff else "valid"
            # A chunk can cross the cutoff only once; split it explicitly.
            boundary = int(np.searchsorted(time_id[start:end], cutoff, side="left"))
            chunk_ranges = []
            if boundary > 0:
                chunk_ranges.append(("train", start, start + boundary))
            if boundary < end - start:
                chunk_ranges.append(("valid", start + boundary, end))
            for split_name, chunk_start, chunk_end in chunk_ranges:
                if args.max_rows_per_split > 0:
                    remaining = args.max_rows_per_split - sampled_rows[split_name]
                    if remaining <= 0:
                        continue
                    chunk_end = min(chunk_end, chunk_start + remaining)
                sampled_rows[split_name] += chunk_end - chunk_start
                for name, column in responder_indices.items():
                    update_moments(
                        moments[name][split_name],
                        x[chunk_start:chunk_end],
                        responder[chunk_start:chunk_end, column],
                        weight[chunk_start:chunk_end],
                    )
        if shard_index == 1 or shard_index == len(metadata["shards"]) or shard_index % 10 == 0:
            print(f"processed shard {shard_index}/{len(metadata['shards'])}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "cache_feature_count": len(names),
        "raw_feature_count": len(metadata["feature_columns"]),
        "temporal_feature_count": len(metadata["temporal_feature_columns"]),
        "time_monotonic_in_all_shards": all(boundary_checks),
        "valid_cutoff_time_id": cutoff,
        "model_cache_compatible": model_cache_compatible,
        "compatibility_checks": compatibility_checks,
        "cache_schema_version": metadata.get("cache_schema_version"),
        "cache_temporal_engine_version": metadata.get(
            "temporal_engine_version"
        ),
        "model_temporal_engine_version": model_metadata.get(
            "temporal_engine_version"
        ),
        "cache_temporal_plan_hash": metadata.get("temporal_plan_hash", ""),
        "model_temporal_plan_hash": model_metadata.get(
            "temporal_plan_hash", ""
        ),
        "responders": {},
    }
    for name, column in responder_indices.items():
        train_corr = correlations(moments[name]["train"])
        valid_corr = correlations(moments[name]["valid"])
        linear_r2, incremental_r2, constant_score = single_feature_linear_r2(
            moments[name]["train"], moments[name]["valid"]
        )
        table = pd.DataFrame(
            {
                "feature": names,
                "train_weighted_corr": train_corr,
                "valid_weighted_corr": valid_corr,
                "single_feature_zero_mean_r2": linear_r2,
                "single_feature_incremental_r2": incremental_r2,
            }
        ).sort_values("single_feature_incremental_r2", ascending=False)
        table.to_csv(output_dir / f"{name}_feature_audit.csv", index=False)
        report["responders"][name] = {
            "train_rows": moments[name]["train"]["rows"],
            "valid_rows": moments[name]["valid"]["rows"],
            "max_abs_valid_feature_corr": float(np.max(np.abs(valid_corr))),
            "constant_mean_baseline_r2": constant_score,
            "best_single_feature": table.iloc[0].to_dict(),
            "asset_mean_baseline_r2": asset_baseline(cache_dir, metadata, column, cutoff),
            "negative_control": (
                model_negative_controls(
                    cache_dir, metadata, model_dir, name, column, cutoff,
                    args.predict_sample_rows, args.seed,
                )
                if model_cache_compatible
                else {
                    "available": False,
                    "reason": "model metadata and cache schema do not match",
                }
            ),
            "top_features": table.head(20).to_dict(orient="records"),
        }
    (output_dir / "responder_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
