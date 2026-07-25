from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Screen every responder_* by feature predictability and held-out "
            "target residual improvement."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--early-stopping", type=int, default=25)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


START = time.perf_counter()


def progress(message: str):
    print(f"[screen {time.perf_counter() - START:8.1f}s] {message}", flush=True)


def manifest_files(root: Path) -> list[Path]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = manifest.get("files", {}).get("train", [])
        if values:
            return [root / value for value in values]
    return sorted((root / "train").glob("*.parquet"))


def weighted_mean(values, weight):
    valid = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
    denominator = float(np.sum(weight[valid]))
    return (
        float(np.sum(weight[valid] * values[valid]) / denominator)
        if denominator > 0 else 0.0
    )


def weighted_corr(left, right, weight):
    valid = (
        np.isfinite(left) & np.isfinite(right)
        & np.isfinite(weight) & (weight > 0)
    )
    if valid.sum() < 2:
        return 0.0
    left, right, weight = left[valid], right[valid], weight[valid]
    left = left - np.sum(weight * left) / np.sum(weight)
    right = right - np.sum(weight * right) / np.sum(weight)
    numerator = np.sum(weight * left * right)
    denominator = np.sqrt(
        np.sum(weight * left * left) * np.sum(weight * right * right)
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def weighted_r2(y, prediction, weight, centered=False):
    valid = (
        np.isfinite(y) & np.isfinite(prediction)
        & np.isfinite(weight) & (weight > 0)
    )
    y, prediction, weight = y[valid], prediction[valid], weight[valid]
    if not len(y):
        return 0.0
    reference = weighted_mean(y, weight) if centered else 0.0
    denominator = np.sum(weight * (y - reference) ** 2)
    numerator = np.sum(weight * (y - prediction) ** 2)
    return float(1.0 - numerator / denominator) if denominator > 0 else 0.0


def load_sample(files, columns, max_rows, batch_size, seed):
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    probability = min(1.0, max_rows * 1.08 / max(total_rows, 1))
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in columns}
    selected_rows = 0
    for file_index, path in enumerate(files, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=columns, use_threads=True
        ):
            mask = rng.random(batch.num_rows) < probability
            if not mask.any():
                continue
            frame = batch.to_pandas()
            positions = np.flatnonzero(mask)
            for name in columns:
                parts[name].append(frame[name].to_numpy(copy=False)[positions])
            selected_rows += len(positions)
        progress(
            f"sampled parquet {file_index}/{len(files)}; "
            f"selected_rows={selected_rows:,}"
        )
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    row_count = len(arrays["time_id"])
    if row_count > max_rows:
        keep = np.sort(rng.choice(row_count, size=max_rows, replace=False))
        arrays = {name: values[keep] for name, values in arrays.items()}
    order = np.argsort(arrays["time_id"], kind="stable")
    arrays = {name: values[order] for name, values in arrays.items()}
    return arrays, total_rows


def time_slices(time_ids):
    unique_times = np.unique(time_ids)
    if len(unique_times) < 10:
        raise ValueError("not enough sampled time_id values for temporal screening")
    first_cutoff = unique_times[int(len(unique_times) * 0.60)]
    second_cutoff = unique_times[int(len(unique_times) * 0.80)]
    first = int(np.searchsorted(time_ids, first_cutoff, side="left"))
    second = int(np.searchsorted(time_ids, second_cutoff, side="left"))
    return slice(0, first), slice(first, second), slice(second, len(time_ids))


def fit_model(x_train, y_train, w_train, x_valid, y_valid, w_valid, args):
    train_set = lgb.Dataset(
        x_train, label=y_train, weight=w_train, free_raw_data=True
    )
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, weight=w_valid, reference=train_set,
        free_raw_data=True,
    )
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "num_threads": args.threads,
        "seed": args.seed,
        "verbosity": -1,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def clean_label(values, weight):
    values = np.asarray(values, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32).copy()
    valid = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
    weight[~valid] = 0.0
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), weight


def main():
    args = parse_args()
    data_root, output_dir = Path(args.data_root), Path(args.output_dir)
    files = manifest_files(data_root)
    if not files:
        raise ValueError(f"no training parquet files under {data_root}")
    schema = pq.read_schema(files[0]).names
    features = [name for name in schema if name.startswith("feature_")]
    responders = sorted(name for name in schema if name.startswith("responder_"))
    if not responders:
        raise ValueError("no responder_* columns found")
    progress(
        f"screening {len(responders)} responders with {len(features)} raw features"
    )
    columns = ["time_id", "weight", "target", *features, *responders]
    arrays, total_rows = load_sample(
        files, columns, args.max_rows, args.batch_size, args.seed
    )
    x = np.column_stack([arrays[name] for name in features]).astype(
        np.float32, copy=False
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    for name in features:
        del arrays[name]
    weight = np.maximum(
        np.nan_to_num(arrays["weight"], nan=0.0), 0.0
    ).astype(np.float32)
    target, target_weight = clean_label(arrays["target"], weight)
    train_slice, calibration_slice, evaluation_slice = time_slices(
        arrays["time_id"]
    )
    progress(
        f"sample rows={len(x):,}/{total_rows:,}; "
        f"train={train_slice.stop:,}, "
        f"calibration={calibration_slice.stop - calibration_slice.start:,}, "
        f"evaluation={evaluation_slice.stop - evaluation_slice.start:,}"
    )

    baseline = fit_model(
        x[train_slice], target[train_slice], target_weight[train_slice],
        x[calibration_slice], target[calibration_slice],
        target_weight[calibration_slice], args,
    )
    baseline_cal = baseline.predict(x[calibration_slice])
    baseline_eval = baseline.predict(x[evaluation_slice])
    baseline_score = weighted_r2(
        target[evaluation_slice], baseline_eval,
        target_weight[evaluation_slice],
    )
    progress(f"target baseline evaluation R2={baseline_score:.8f}")

    rows = []
    for index, name in enumerate(responders, start=1):
        responder, responder_weight = clean_label(arrays[name], weight)
        model = fit_model(
            x[train_slice], responder[train_slice],
            responder_weight[train_slice],
            x[calibration_slice], responder[calibration_slice],
            responder_weight[calibration_slice], args,
        )
        hat_cal = np.asarray(model.predict(x[calibration_slice]), dtype=np.float64)
        hat_eval = np.asarray(model.predict(x[evaluation_slice]), dtype=np.float64)
        calibration_weight = target_weight[calibration_slice].astype(np.float64)
        evaluation_weight = target_weight[evaluation_slice].astype(np.float64)
        hat_mean = weighted_mean(hat_cal, calibration_weight)
        centered_cal = hat_cal - hat_mean
        centered_eval = hat_eval - hat_mean
        residual_cal = target[calibration_slice] - baseline_cal
        denominator = np.sum(calibration_weight * centered_cal * centered_cal)
        alpha = (
            float(np.sum(
                calibration_weight * centered_cal * residual_cal
            ) / denominator)
            if denominator > 0 else 0.0
        )
        augmented_eval = baseline_eval + alpha * centered_eval
        augmented_score = weighted_r2(
            target[evaluation_slice], augmented_eval, evaluation_weight
        )
        true_eval = responder[evaluation_slice]
        responder_eval_weight = responder_weight[evaluation_slice]
        joint_evaluation_weight = np.minimum(
            evaluation_weight, responder_eval_weight
        )
        row = {
            "responder": name,
            "best_iteration": int(
                model.best_iteration or model.current_iteration()
            ),
            "responder_zero_mean_r2": weighted_r2(
                true_eval, hat_eval, responder_eval_weight
            ),
            "responder_centered_r2": weighted_r2(
                true_eval, hat_eval, responder_eval_weight, centered=True
            ),
            "responder_prediction_corr": weighted_corr(
                true_eval, hat_eval, responder_eval_weight
            ),
            "true_responder_target_corr": weighted_corr(
                true_eval, target[evaluation_slice], joint_evaluation_weight
            ),
            "hat_target_residual_corr": weighted_corr(
                centered_eval,
                target[evaluation_slice] - baseline_eval,
                evaluation_weight,
            ),
            "residual_coefficient": alpha,
            "target_baseline_r2": baseline_score,
            "target_augmented_r2": augmented_score,
            "target_incremental_r2": augmented_score - baseline_score,
        }
        rows.append(row)
        progress(
            f"{index}/{len(responders)} {name}: "
            f"centered_R2={row['responder_centered_r2']:.6f}, "
            f"target_delta={row['target_incremental_r2']:+.8f}"
        )
        del model
        gc.collect()

    result = pd.DataFrame(rows).sort_values(
        ["target_incremental_r2", "responder_centered_r2"],
        ascending=False,
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    eligible = result.loc[
        (result["target_incremental_r2"] > 0)
        & (result["responder_centered_r2"] > 0)
    ].head(args.candidate_count)
    candidates = eligible["responder"].tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "all_responder_potential.csv", index=False)
    report = {
        "method": "temporal_train_calibration_evaluation_screen",
        "total_rows": total_rows,
        "sample_rows": len(x),
        "feature_count": len(features),
        "responder_count": len(responders),
        "target_baseline_r2": baseline_score,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_results": result.head(args.candidate_count).to_dict(
            orient="records"
        ),
        "outputs": {
            "ranking": "all_responder_potential.csv",
            "candidates": "responder_candidates.json",
        },
    }
    (output_dir / "responder_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    progress(f"selected candidates: {candidates}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
