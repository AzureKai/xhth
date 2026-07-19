from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze responder-to-target strength and temporal stability."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2_000_000,
        help="Maximum sampled rows; use 0 to retain all rows.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=8,
        help="Number of contiguous time periods used for stability analysis.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def manifest_files(data_root: Path, split: str = "train") -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [data_root / str(file) for file in files]
    return sorted((data_root / split).glob("*.parquet"))


def manifest_rows(data_root: Path, split: str = "train") -> int | None:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("rows", {}).get(split)
    return int(value) if value is not None else None


def schema_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def load_analysis_frame(
    data_root: Path,
    files: list[Path],
    responders: list[str],
    batch_size: int,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    import pyarrow.parquet as pq

    total_rows = manifest_rows(data_root)
    probability = 1.0
    if max_rows > 0 and total_rows and total_rows > max_rows:
        probability = max_rows / float(total_rows)

    rng = np.random.default_rng(seed)
    columns = ["time_id", "asset_id", "weight", "target", *responders]
    frames: list[pd.DataFrame] = []
    retained = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            if probability < 1.0:
                frame = frame.loc[rng.random(len(frame)) < probability]
            if frame.empty:
                continue
            frames.append(frame)
            retained += len(frame)
            if max_rows > 0 and retained >= int(max_rows * 1.1):
                break
        if max_rows > 0 and retained >= int(max_rows * 1.1):
            break

    if not frames:
        raise ValueError("sampling produced an empty analysis frame")

    result = pd.concat(frames, ignore_index=True)
    if max_rows > 0 and len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=seed)
    numeric_columns = ["time_id", "asset_id", "weight", "target", *responders]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["weight"] = result["weight"].fillna(0.0).clip(lower=0.0)
    result = result.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    return result, probability


def weighted_corr(x: pd.Series, y: pd.Series, weight: pd.Series) -> float:
    xv = x.to_numpy(dtype=np.float64, copy=False)
    yv = y.to_numpy(dtype=np.float64, copy=False)
    wv = weight.to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(wv) & (wv > 0)
    if valid.sum() < 2:
        return np.nan
    xv, yv, wv = xv[valid], yv[valid], wv[valid]
    weight_sum = wv.sum()
    x_centered = xv - np.sum(wv * xv) / weight_sum
    y_centered = yv - np.sum(wv * yv) / weight_sum
    denominator = np.sqrt(
        np.sum(wv * x_centered * x_centered)
        * np.sum(wv * y_centered * y_centered)
    )
    if denominator <= 0:
        return np.nan
    return float(np.sum(wv * x_centered * y_centered) / denominator)


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return np.nan
    return float(x.loc[valid].corr(y.loc[valid], method=method))


def cross_section_ic(frame: pd.DataFrame, responder: str) -> pd.Series:
    values = {}
    for time_id, group in frame.groupby("time_id", sort=True, observed=True):
        values[time_id] = safe_corr(
            group[responder], group["target"], method="spearman"
        )
    return pd.Series(values, dtype=np.float64, name="spearman_ic")


def period_correlations(
    frame: pd.DataFrame,
    responders: list[str],
    time_bins: int,
) -> pd.DataFrame:
    unique_times = np.sort(frame["time_id"].dropna().unique())
    bin_count = min(max(time_bins, 1), len(unique_times))
    time_chunks = np.array_split(unique_times, bin_count)
    time_to_period = {
        time_id: period
        for period, chunk in enumerate(time_chunks)
        for time_id in chunk
    }
    periods = frame["time_id"].map(time_to_period)
    rows = []
    for period in range(bin_count):
        current = frame.loc[periods == period]
        for responder in responders:
            rows.append(
                {
                    "period": period,
                    "time_id_min": int(current["time_id"].min()),
                    "time_id_max": int(current["time_id"].max()),
                    "rows": int(len(current)),
                    "responder": responder,
                    "weighted_pearson": weighted_corr(
                        current[responder], current["target"], current["weight"]
                    ),
                    "spearman": safe_corr(
                        current[responder], current["target"], method="spearman"
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze(
    frame: pd.DataFrame,
    responders: list[str],
    time_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_time_demean = frame["target"] - frame.groupby("time_id")["target"].transform("mean")
    target_asset_demean = frame["target"] - frame.groupby("asset_id")["target"].transform("mean")
    period = period_correlations(frame, responders, time_bins)
    ic_frames = []
    summary_rows = []

    for responder in responders:
        ic = cross_section_ic(frame[["time_id", responder, "target"]], responder)
        ic_frame = ic.rename("spearman_ic").reset_index()
        ic_frame.insert(1, "responder", responder)
        ic_frames.append(ic_frame)

        responder_time_demean = (
            frame[responder] - frame.groupby("time_id")[responder].transform("mean")
        )
        responder_asset_demean = (
            frame[responder] - frame.groupby("asset_id")[responder].transform("mean")
        )
        valid_ic = ic.dropna()
        ic_mean = float(valid_ic.mean()) if len(valid_ic) else np.nan
        ic_std = float(valid_ic.std(ddof=1)) if len(valid_ic) > 1 else np.nan
        icir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
        sign_ratio = (
            float((np.sign(valid_ic) == np.sign(ic_mean)).mean())
            if len(valid_ic) and ic_mean != 0
            else np.nan
        )

        current_periods = period.loc[period["responder"] == responder, "weighted_pearson"].dropna()
        period_mean = float(current_periods.mean()) if len(current_periods) else np.nan
        period_std = float(current_periods.std(ddof=1)) if len(current_periods) > 1 else np.nan
        period_sign_ratio = (
            float((np.sign(current_periods) == np.sign(period_mean)).mean())
            if len(current_periods) and period_mean != 0
            else np.nan
        )
        summary_rows.append(
            {
                "responder": responder,
                "rows": int((frame[responder].notna() & frame["target"].notna()).sum()),
                "weighted_pearson": weighted_corr(
                    frame[responder], frame["target"], frame["weight"]
                ),
                "pearson": safe_corr(frame[responder], frame["target"]),
                "spearman": safe_corr(frame[responder], frame["target"], method="spearman"),
                "time_demeaned_pearson": safe_corr(
                    responder_time_demean, target_time_demean
                ),
                "asset_demeaned_pearson": safe_corr(
                    responder_asset_demean, target_asset_demean
                ),
                "mean_cross_section_spearman_ic": ic_mean,
                "cross_section_ic_std": ic_std,
                "cross_section_icir": icir,
                "cross_section_sign_stability": sign_ratio,
                "period_weighted_corr_mean": period_mean,
                "period_weighted_corr_std": period_std,
                "period_sign_stability": period_sign_ratio,
                "recent_period_weighted_corr": (
                    float(current_periods.iloc[-1]) if len(current_periods) else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    stability = summary[
        ["cross_section_sign_stability", "period_sign_stability"]
    ].mean(axis=1, skipna=True)
    summary["screening_score"] = (
        0.30 * summary["weighted_pearson"].abs().fillna(0.0)
        + 0.20 * summary["spearman"].abs().fillna(0.0)
        + 0.25 * summary["mean_cross_section_spearman_ic"].abs().fillna(0.0)
        + 0.15 * summary["recent_period_weighted_corr"].abs().fillna(0.0)
        + 0.10 * stability.fillna(0.0)
    )
    summary = summary.sort_values(
        ["screening_score", "weighted_pearson"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, period, pd.concat(ic_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = manifest_files(data_root)
    if not files:
        raise ValueError(f"no train parquet files found under {data_root}")
    columns = schema_columns(files[0])
    responders = sorted(column for column in columns if column.startswith("responder_"))
    if not responders:
        raise ValueError("no responder_* columns found")

    frame, sample_probability = load_analysis_frame(
        data_root,
        files,
        responders,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    summary, periods, time_ic = analyze(frame, responders, args.time_bins)
    summary.to_csv(output_dir / "responder_summary.csv", index=False)
    periods.to_csv(output_dir / "responder_period_correlations.csv", index=False)
    time_ic.to_csv(output_dir / "responder_time_ic.csv", index=False)

    report = {
        "rows": int(len(frame)),
        "time_id_count": int(frame["time_id"].nunique()),
        "asset_id_count": int(frame["asset_id"].nunique()),
        "responder_count": len(responders),
        "sample_probability": sample_probability,
        "time_bins": int(min(args.time_bins, frame["time_id"].nunique())),
        "top_responders": json.loads(
            summary.head(10).to_json(orient="records")
        ),
        "outputs": {
            "summary": "responder_summary.csv",
            "period_correlations": "responder_period_correlations.csv",
            "time_ic": "responder_time_ic.csv",
        },
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
