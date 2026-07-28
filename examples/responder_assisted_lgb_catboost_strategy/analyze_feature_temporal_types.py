from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RECIPE_MAP = {
    "trend": ["lag1", "lag5", "lag20", "ema5", "ema20", "ema60"],
    "difference": ["delta5", "delta20", "historical_zscore20", "historical_zscore60"],
    "volatility": ["rolling_std20", "rolling_std60", "historical_zscore20", "historical_zscore60"],
    "shock": ["historical_zscore20", "historical_zscore60", "minus_ema20", "minus_ema60"],
    "mean_reversion": ["minus_ema20", "minus_ema60", "historical_zscore20", "historical_zscore60"],
    "cross_section": ["xs_rank"],
}
SCORE_COLUMNS = [f"{name}_score" for name in RECIPE_MAP]


def parse_args():
    parser = argparse.ArgumentParser(description="Route anonymous features to temporal transforms")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-primary-features", type=int, default=120)
    parser.add_argument("--secondary-threshold", type=float, default=0.70)
    return parser.parse_args()


def files_for(root: Path):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get("train", [])
        if files:
            return [root / str(value) for value in files]
    return sorted((root / "train").glob("*.parquet"))


def load_recent_complete_times(files, columns, max_rows, batch_size):
    groups = deque()
    rows = 0
    carry = None
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
            last_time = frame["time_id"].iloc[-1]
            ready = frame.loc[frame["time_id"] != last_time]
            carry = frame.loc[frame["time_id"] == last_time].copy()
            for _, group in ready.groupby("time_id", sort=False):
                groups.append(group)
                rows += len(group)
                while rows > max_rows and len(groups) > 1:
                    rows -= len(groups.popleft())
    if carry is not None and not carry.empty:
        groups.append(carry)
    return pd.concat(groups, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort")


def corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    xv, yv = x[valid], y[valid]
    if np.std(xv) <= 0 or np.std(yv) <= 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def percentile(series):
    return series.rank(pct=True).fillna(0.0)


def analyze(frame, features):
    rows = []
    report_every = max(1, len(features) // 20)
    for feature_index, feature in enumerate(features, start=1):
        asset_metrics = []
        for _, group in frame[["asset_id", "time_id", feature]].groupby("asset_id", sort=False):
            x = group[feature].to_numpy(dtype=np.float64)
            delta = np.diff(x)
            finite_x = x[np.isfinite(x)]
            if len(finite_x) < 30:
                continue
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median))
            robust_scale = 1.4826 * mad
            period_means = [np.nanmean(chunk) for chunk in np.array_split(x, 8)]
            ema = np.empty_like(x)
            ema[0] = x[0]
            for index in range(1, len(x)):
                if not np.isfinite(x[index]):
                    ema[index] = ema[index - 1]
                elif not np.isfinite(ema[index - 1]):
                    ema[index] = x[index]
                else:
                    ema[index] = 0.1 * x[index] + 0.9 * ema[index - 1]
            asset_metrics.append(
                [
                    corr(x[1:], x[:-1]), corr(x[5:], x[:-5]), corr(x[20:], x[:-20]),
                    corr(delta[1:], delta[:-1]), corr(np.abs(delta[1:]), np.abs(delta[:-1])),
                    corr(delta[1:] ** 2, delta[:-1] ** 2),
                    np.nanmean(np.sign(delta[1:]) == np.sign(delta[:-1])),
                    np.nanstd(period_means) / (np.nanstd(x) + 1e-12),
                    np.nanmean(((x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)) ** 4),
                    np.nanmean(np.abs(x - median) > 4.0 * robust_scale) if robust_scale > 0 else 0.0,
                    corr((x[:-1] - ema[:-1]), delta),
                ]
            )
        values = np.nanmean(np.asarray(asset_metrics, dtype=np.float64), axis=0)
        rows.append([feature, *values])
        if (
            feature_index == 1
            or feature_index == len(features)
            or feature_index % report_every == 0
        ):
            print(
                f"[temporal screening] {feature_index}/{len(features)} "
                f"({100.0 * feature_index / len(features):.1f}%) {feature}",
                flush=True,
            )

    result = pd.DataFrame(rows, columns=[
        "feature", "level_acf1", "level_acf5", "level_acf20", "delta_acf1",
        "abs_delta_acf1", "squared_delta_acf1", "direction_persistence",
        "period_drift_ratio", "kurtosis", "shock_rate", "mean_reversion_corr",
    ])

    # Approximate cross-sectional rank persistence while retaining complete time slices.
    sums = {name: np.zeros(len(features), dtype=np.float64) for name in ("x", "y", "xx", "yy", "xy", "n")}
    previous = {}
    for _, group in frame.groupby("time_id", sort=True):
        values = group.loc[:, features].to_numpy(dtype=np.float64)
        order = np.argsort(np.argsort(np.nan_to_num(values, nan=np.inf), axis=0), axis=0)
        ranks = order.astype(np.float64) / max(len(group) - 1, 1)
        ranks[~np.isfinite(values)] = np.nan
        for row_index, asset in enumerate(group["asset_id"].to_numpy()):
            old = previous.get(int(asset))
            if old is not None:
                x, y = old, ranks[row_index]
                valid = np.isfinite(x) & np.isfinite(y)
                sums["x"][valid] += x[valid]; sums["y"][valid] += y[valid]
                sums["xx"][valid] += x[valid] ** 2; sums["yy"][valid] += y[valid] ** 2
                sums["xy"][valid] += x[valid] * y[valid]; sums["n"][valid] += 1
            previous[int(asset)] = ranks[row_index]
    n = np.maximum(sums["n"], 1)
    covariance = sums["xy"] - sums["x"] * sums["y"] / n
    denominator = np.sqrt(
        np.maximum(sums["xx"] - sums["x"] ** 2 / n, 0)
        * np.maximum(sums["yy"] - sums["y"] ** 2 / n, 0)
    )
    result["xs_rank_persistence"] = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)

    result["trend_score"] = (
        percentile(result[["level_acf1", "level_acf5", "level_acf20"]].mean(axis=1))
        + percentile(result["direction_persistence"])
    ) / 2
    result["difference_score"] = (
        percentile(result["period_drift_ratio"]) + percentile(result["level_acf1"] - result["delta_acf1"].abs())
    ) / 2
    result["volatility_score"] = (
        percentile(result["abs_delta_acf1"]) + percentile(result["squared_delta_acf1"])
    ) / 2
    result["shock_score"] = (percentile(result["kurtosis"]) + percentile(result["shock_rate"])) / 2
    result["mean_reversion_score"] = (
        percentile(-result["delta_acf1"]) + percentile(-result["mean_reversion_corr"])
    ) / 2
    result["cross_section_score"] = percentile(result["xs_rank_persistence"])
    result["primary_score"] = result[SCORE_COLUMNS].max(axis=1)
    result["primary_type"] = (
        result[SCORE_COLUMNS]
        .idxmax(axis=1)
        .str.removesuffix("_score")
    )
    return result


def route(result, max_primary, secondary_threshold):
    score_columns = SCORE_COLUMNS
    ranked = result.copy()
    ranked = ranked.sort_values("primary_score", ascending=False).head(max_primary)
    recipes = {}
    route_rows = []
    for _, row in ranked.iterrows():
        ordered = sorted(((column.removesuffix("_score"), float(row[column])) for column in score_columns), key=lambda item: item[1], reverse=True)
        categories = [ordered[0][0]]
        categories.extend(name for name, score in ordered[1:] if score >= secondary_threshold and score >= ordered[0][1] - 0.15)
        transforms = []
        for category in categories:
            for transform in RECIPE_MAP[category]:
                if transform not in transforms:
                    transforms.append(transform)
        recipes[str(row["feature"])] = transforms
        route_rows.append({"feature": row["feature"], "categories": ",".join(categories), "transforms": ",".join(transforms), "primary_score": ordered[0][1]})
    return recipes, pd.DataFrame(route_rows)


def main():
    args = parse_args()
    root, output_dir = Path(args.data_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = files_for(root)
    columns = list(pq.read_schema(files[0]).names)
    features = [column for column in columns if column.startswith("feature_")]
    print(f"loading the most recent {args.max_rows:,} rows with complete time slices", flush=True)
    frame = load_recent_complete_times(files, ["time_id", "asset_id", *features], args.max_rows, args.batch_size)
    print(f"analyzing {len(features)} features over {len(frame):,} rows", flush=True)
    result = analyze(frame, features)
    recipes, routes = route(result, args.max_primary_features, args.secondary_threshold)
    result.sort_values("primary_score", ascending=False).to_csv(output_dir / "feature_temporal_statistics.csv", index=False)
    routes.to_csv(output_dir / "feature_temporal_routes.csv", index=False)
    payload = {"rows": len(frame), "features_analyzed": len(features), "features_selected": len(recipes), "recipes": recipes}
    (output_dir / "temporal_feature_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "recipes"}, indent=2))


if __name__ == "__main__":
    main()
