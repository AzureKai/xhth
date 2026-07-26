from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from screen_all_responders import (
    clean_label,
    fit_model,
    load_sample,
    manifest_files,
    progress,
    weighted_mean,
    weighted_r2,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Robust all-responder screening with expanding time folds, "
            "multi-output Ridge, OOF clustering, and LightGBM refinement."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-centered-r2", type=float, default=0.01)
    parser.add_argument("--min-positive-folds", type=int, default=3)
    parser.add_argument("--cluster-correlation", type=float, default=0.95)
    parser.add_argument("--refine-count", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-lightgbm-refine", action="store_true")
    return parser.parse_args()


def temporal_fold_slices(time_ids, folds, warmup_fraction):
    unique_times = np.unique(time_ids)
    warmup = max(1, int(len(unique_times) * warmup_fraction))
    blocks = np.array_split(unique_times[warmup:], folds)
    result = []
    for fold, block in enumerate(blocks):
        if len(block) < 2:
            continue
        midpoint = len(block) // 2
        calibration_times = block[:midpoint]
        evaluation_times = block[midpoint:]
        train_stop = int(np.searchsorted(
            time_ids, calibration_times[0], side="left"
        ))
        calibration_start = train_stop
        calibration_stop = int(np.searchsorted(
            time_ids, calibration_times[-1], side="right"
        ))
        evaluation_start = calibration_stop
        evaluation_stop = int(np.searchsorted(
            time_ids, evaluation_times[-1], side="right"
        ))
        result.append(
            {
                "fold": fold,
                "train": slice(0, train_stop),
                "calibration": slice(calibration_start, calibration_stop),
                "evaluation": slice(evaluation_start, evaluation_stop),
                "time_start": int(block[0]),
                "time_end": int(block[-1]),
            }
        )
    if len(result) != folds:
        raise ValueError(f"requested {folds} folds but constructed {len(result)}")
    return result


def fit_multi_output_ridge(x, y, weight, alpha):
    valid = (
        np.all(np.isfinite(x), axis=1)
        & np.all(np.isfinite(y), axis=1)
        & np.isfinite(weight) & (weight > 0)
    )
    x = np.asarray(x[valid], dtype=np.float32)
    y = np.asarray(y[valid], dtype=np.float32)
    weight = np.asarray(weight[valid], dtype=np.float32)
    weight_sum = float(np.sum(weight))
    normalized_weight = weight / max(weight_sum, 1e-12)
    x_mean = np.sum(normalized_weight[:, None] * x, axis=0)
    x_centered = x - x_mean
    x_variance = np.sum(
        normalized_weight[:, None] * x_centered * x_centered, axis=0
    )
    x_scale = np.sqrt(np.maximum(x_variance, 1e-8)).astype(np.float32)
    x_standard = x_centered / x_scale
    y_mean = np.sum(normalized_weight[:, None] * y, axis=0)
    y_centered = y - y_mean
    weighted_x = normalized_weight[:, None] * x_standard
    gram = x_standard.T @ weighted_x
    right = x_standard.T @ (normalized_weight[:, None] * y_centered)
    gram.flat[::len(gram) + 1] += float(alpha)
    coefficients = np.linalg.solve(
        np.asarray(gram, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    ).astype(np.float32)
    return x_mean, x_scale, y_mean, coefficients


def ridge_predict(model, x):
    x_mean, x_scale, y_mean, coefficients = model
    return (
        (np.asarray(x, dtype=np.float32) - x_mean) / x_scale
    ) @ coefficients + y_mean


def residual_increment(
    target_cal, baseline_cal, hat_cal, weight_cal,
    target_eval, baseline_eval, hat_eval, weight_eval,
):
    hat_mean = weighted_mean(hat_cal, weight_cal)
    centered_cal = hat_cal - hat_mean
    centered_eval = hat_eval - hat_mean
    residual_cal = target_cal - baseline_cal
    denominator = np.sum(weight_cal * centered_cal * centered_cal)
    coefficient = (
        float(np.sum(weight_cal * centered_cal * residual_cal) / denominator)
        if denominator > 0 else 0.0
    )
    baseline_score = weighted_r2(target_eval, baseline_eval, weight_eval)
    augmented_score = weighted_r2(
        target_eval, baseline_eval + coefficient * centered_eval, weight_eval
    )
    return coefficient, baseline_score, augmented_score, augmented_score - baseline_score


def summarize_fold_rows(rows, responder):
    current = [row for row in rows if row["responder"] == responder]
    deltas = np.asarray([row["target_incremental_r2"] for row in current])
    centered = np.asarray([row["responder_centered_r2"] for row in current])
    return {
        "responder": responder,
        "median_centered_r2": float(np.median(centered)),
        "minimum_centered_r2": float(np.min(centered)),
        "mean_delta_r2": float(np.mean(deltas)),
        "median_delta_r2": float(np.median(deltas)),
        "std_delta_r2": float(np.std(deltas)),
        "minimum_delta_r2": float(np.min(deltas)),
        "last_delta_r2": float(deltas[-1]),
        "positive_folds": int(np.sum(deltas > 0)),
        "robust_score": float(np.median(deltas) - 0.5 * np.std(deltas)),
        "fold_deltas": deltas.tolist(),
        "fold_centered_r2": centered.tolist(),
    }


def cluster_candidates(names, correlation, threshold, ranking):
    parent = list(range(len(names)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if abs(correlation[left, right]) >= threshold:
                union(left, right)
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(find(index), []).append(name)
    score = {
        row["responder"]: row["robust_score"] for row in ranking
    }
    clusters = []
    for members in groups.values():
        representative = max(members, key=lambda name: score[name])
        clusters.append(
            {
                "representative": representative,
                "members": sorted(members),
                "size": len(members),
            }
        )
    return sorted(
        clusters, key=lambda item: score[item["representative"]], reverse=True
    )


def lightgbm_refinement(
    x, target, target_weight, responder_values, candidates, folds, args
):
    fold_rows = []
    for fold_info in folds:
        fold = fold_info["fold"]
        train = fold_info["train"]
        calibration = fold_info["calibration"]
        evaluation = fold_info["evaluation"]
        baseline = fit_model(
            x[train], target[train], target_weight[train],
            x[calibration], target[calibration],
            target_weight[calibration], args,
        )
        baseline_cal = baseline.predict(x[calibration])
        baseline_eval = baseline.predict(x[evaluation])
        for index, name in enumerate(candidates, start=1):
            responder, responder_weight = clean_label(
                responder_values[name], target_weight
            )
            model = fit_model(
                x[train], responder[train], responder_weight[train],
                x[calibration], responder[calibration],
                responder_weight[calibration], args,
            )
            hat_cal = model.predict(x[calibration])
            hat_eval = model.predict(x[evaluation])
            coefficient, baseline_score, augmented_score, delta = (
                residual_increment(
                    target[calibration], baseline_cal, hat_cal,
                    target_weight[calibration],
                    target[evaluation], baseline_eval, hat_eval,
                    target_weight[evaluation],
                )
            )
            centered_r2 = weighted_r2(
                responder[evaluation], hat_eval,
                responder_weight[evaluation], centered=True,
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "responder": name,
                    "responder_centered_r2": centered_r2,
                    "residual_coefficient": coefficient,
                    "target_baseline_r2": baseline_score,
                    "target_augmented_r2": augmented_score,
                    "target_incremental_r2": delta,
                }
            )
            progress(
                f"LightGBM refine fold {fold + 1}/{len(folds)} "
                f"{index}/{len(candidates)} {name}: delta={delta:+.8f}"
            )
    ranking = [
        summarize_fold_rows(fold_rows, responder) for responder in candidates
    ]
    return sorted(ranking, key=lambda row: row["robust_score"], reverse=True), fold_rows


def main():
    args = parse_args()
    data_root, output_dir = Path(args.data_root), Path(args.output_dir)
    files = manifest_files(data_root)
    if not files:
        raise ValueError(f"no training parquet files under {data_root}")
    schema = pq.read_schema(files[0]).names
    features = [name for name in schema if name.startswith("feature_")]
    responders = sorted(name for name in schema if name.startswith("responder_"))
    columns = ["time_id", "weight", "target", *features, *responders]
    arrays, total_rows = load_sample(
        files, columns, args.max_rows, args.batch_size, args.seed
    )
    x = np.column_stack([arrays.pop(name) for name in features]).astype(
        np.float32, copy=False
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    weight = np.maximum(
        np.nan_to_num(arrays["weight"], nan=0.0), 0.0
    ).astype(np.float32)
    target, target_weight = clean_label(arrays["target"], weight)
    responder_matrix = np.column_stack(
        [np.nan_to_num(arrays[name], nan=0.0) for name in responders]
    ).astype(np.float32)
    all_outputs = np.column_stack([target, responder_matrix])
    folds = temporal_fold_slices(
        arrays["time_id"], args.folds, args.warmup_fraction
    )
    progress(
        f"Ridge screening {len(responders)} responders, rows={len(x):,}, "
        f"folds={len(folds)}"
    )

    ridge_fold_rows = []
    evaluation_hats = []
    for fold_info in folds:
        train = fold_info["train"]
        calibration = fold_info["calibration"]
        evaluation = fold_info["evaluation"]
        ridge = fit_multi_output_ridge(
            x[train], all_outputs[train], target_weight[train],
            args.ridge_alpha,
        )
        prediction_cal = ridge_predict(ridge, x[calibration])
        prediction_eval = ridge_predict(ridge, x[evaluation])
        evaluation_hats.append(prediction_eval[:, 1:])
        for column, name in enumerate(responders):
            coefficient, baseline_score, augmented_score, delta = (
                residual_increment(
                    target[calibration], prediction_cal[:, 0],
                    prediction_cal[:, column + 1], target_weight[calibration],
                    target[evaluation], prediction_eval[:, 0],
                    prediction_eval[:, column + 1], target_weight[evaluation],
                )
            )
            centered_r2 = weighted_r2(
                responder_matrix[evaluation, column],
                prediction_eval[:, column + 1],
                target_weight[evaluation], centered=True,
            )
            ridge_fold_rows.append(
                {
                    "fold": fold_info["fold"],
                    "time_start": fold_info["time_start"],
                    "time_end": fold_info["time_end"],
                    "responder": name,
                    "responder_centered_r2": centered_r2,
                    "residual_coefficient": coefficient,
                    "target_baseline_r2": baseline_score,
                    "target_augmented_r2": augmented_score,
                    "target_incremental_r2": delta,
                }
            )
        progress(
            f"Ridge fold {fold_info['fold'] + 1}/{len(folds)} complete"
        )

    ridge_ranking = [
        summarize_fold_rows(ridge_fold_rows, responder)
        for responder in responders
    ]
    ridge_ranking.sort(key=lambda row: row["robust_score"], reverse=True)
    eligible = [
        row for row in ridge_ranking
        if row["median_centered_r2"] >= args.min_centered_r2
        and row["positive_folds"] >= args.min_positive_folds
        and row["last_delta_r2"] > 0
        and row["robust_score"] > 0
    ]
    hats = np.vstack(evaluation_hats)
    hat_correlation = np.corrcoef(hats, rowvar=False)
    eligible_names = [row["responder"] for row in eligible]
    eligible_indices = [responders.index(name) for name in eligible_names]
    eligible_correlation = (
        hat_correlation[np.ix_(eligible_indices, eligible_indices)]
        if eligible_indices else np.empty((0, 0))
    )
    clusters = cluster_candidates(
        eligible_names, eligible_correlation,
        args.cluster_correlation, eligible,
    ) if eligible_names else []
    representatives = [
        item["representative"] for item in clusters[:args.refine_count]
    ]

    lightgbm_ranking = []
    lightgbm_fold_rows = []
    if representatives and not args.skip_lightgbm_refine:
        responder_values = {
            name: arrays[name] for name in representatives
        }
        lightgbm_ranking, lightgbm_fold_rows = lightgbm_refinement(
            x, target, target_weight, responder_values,
            representatives, folds, args,
        )
    final_candidates = [
        row["responder"] for row in lightgbm_ranking
        if row["median_centered_r2"] >= args.min_centered_r2
        and row["positive_folds"] >= args.min_positive_folds
        and row["last_delta_r2"] > 0
        and row["robust_score"] > 0
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ridge_ranking).to_csv(
        output_dir / "multifold_ridge_ranking.csv", index=False
    )
    pd.DataFrame(ridge_fold_rows).to_csv(
        output_dir / "multifold_ridge_folds.csv", index=False
    )
    pd.DataFrame(
        hat_correlation, index=responders, columns=responders
    ).to_csv(output_dir / "multifold_hat_correlations.csv")
    if lightgbm_ranking:
        pd.DataFrame(lightgbm_ranking).to_csv(
            output_dir / "multifold_lightgbm_ranking.csv", index=False
        )
        pd.DataFrame(lightgbm_fold_rows).to_csv(
            output_dir / "multifold_lightgbm_folds.csv", index=False
        )
    report = {
        "method": "expanding_multifold_ridge_cluster_lightgbm",
        "total_rows": total_rows,
        "sample_rows": len(x),
        "feature_count": len(features),
        "responder_count": len(responders),
        "folds": len(folds),
        "ridge_eligible": eligible_names,
        "clusters": clusters,
        "lightgbm_representatives": representatives,
        "final_candidates": final_candidates,
        "ridge_top": ridge_ranking[:20],
        "lightgbm_ranking": lightgbm_ranking,
    }
    (output_dir / "multifold_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    progress(f"final robust candidates: {final_candidates}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
