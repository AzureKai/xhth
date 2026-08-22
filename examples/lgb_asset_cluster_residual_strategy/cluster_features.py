from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(target, prediction, weight) -> float:
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0.0)
    denominator = float(np.sum(w[valid] * y[valid] * y[valid]))
    if denominator <= 0.0:
        return 0.0
    error = y[valid] - pred[valid]
    return float(1.0 - np.sum(w[valid] * error * error) / denominator)


def center_by_time(values, time_ids) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64).copy()
    times = np.asarray(time_ids, dtype=np.int64)
    if len(prediction) != len(times):
        raise ValueError("values and time_ids must have the same length")
    if len(prediction) == 0:
        return prediction
    if np.any(times[1:] < times[:-1]):
        raise ValueError("time_ids must be sorted")
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(times)])
    means = np.add.reduceat(prediction, starts) / counts
    prediction -= np.repeat(means, counts)
    return prediction


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    absolute = np.abs(values - median)
    scale = 1.4826 * np.nanmedian(absolute, axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.nan_to_num((values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def asset_profiles(
    features: np.ndarray,
    asset_ids: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.nan_to_num(
        np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    assets = np.asarray(asset_ids, dtype=np.int64)
    r = np.nan_to_num(np.asarray(residual, dtype=np.float64), nan=0.0)
    w = np.nan_to_num(np.asarray(weight, dtype=np.float64), nan=0.0)
    if x.ndim != 2 or len(x) != len(assets) or len(r) != len(assets):
        raise ValueError("profile inputs are not row-aligned")
    unique_assets, inverse = np.unique(assets, return_inverse=True)
    groups = len(unique_assets)
    safe_w = np.where(w > 0.0, w, 0.0)
    sum_w = np.bincount(inverse, weights=safe_w, minlength=groups)
    sum_w = np.maximum(sum_w, 1e-12)
    mean_r = np.bincount(
        inverse, weights=safe_w * r, minlength=groups
    ) / sum_w
    second_r = np.bincount(
        inverse, weights=safe_w * r * r, minlength=groups
    ) / sum_w
    var_r = np.maximum(second_r - mean_r * mean_r, 0.0)
    columns = [mean_r, np.sqrt(var_r)]
    profile_weights = [0.50, 0.50]
    correlation_columns = []
    for column in range(x.shape[1]):
        value = np.clip(x[:, column], -20.0, 20.0)
        mean_x = np.bincount(
            inverse, weights=safe_w * value, minlength=groups
        ) / sum_w
        second_x = np.bincount(
            inverse, weights=safe_w * value * value, minlength=groups
        ) / sum_w
        cross = np.bincount(
            inverse, weights=safe_w * value * r, minlength=groups
        ) / sum_w
        var_x = np.maximum(second_x - mean_x * mean_x, 0.0)
        covariance = cross - mean_x * mean_r
        correlation = covariance / np.sqrt(np.maximum(var_x * var_r, 1e-16))
        columns.extend([mean_x, np.sqrt(var_x), np.clip(correlation, -1.0, 1.0)])
        # Distribution moments identify broad asset regimes, while the
        # correlation carries the feature-target mapping this experiment tests.
        profile_weights.extend([0.20, 0.20, 3.00])
        correlation_columns.append(len(columns) - 1)
    raw_profile = np.column_stack(columns)
    profile = robust_standardize(raw_profile)
    profile *= np.asarray(profile_weights, dtype=np.float64)
    # Do not inflate near-zero noisy correlations to unit variance across only
    # a few dozen assets; retain their absolute statistical magnitude.
    profile[:, correlation_columns] = 3.0 * raw_profile[:, correlation_columns]
    return unique_assets, profile


def deterministic_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    seed: int = 2026,
    restarts: int = 12,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < clusters or clusters < 2:
        raise ValueError("k-means requires 2 <= clusters <= rows")
    best = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart * 104729)
        first = int(rng.integers(len(x)))
        chosen_indices = [first]
        centers = [x[first]]
        distance = np.sum((x - centers[0]) ** 2, axis=1)
        for _ in range(1, clusters):
            total = float(distance.sum())
            if total <= 1e-16:
                candidates = [
                    index for index in range(len(x))
                    if index not in chosen_indices
                ]
                selected = int(candidates[0])
            else:
                selected = int(rng.choice(len(x), p=distance / total))
            chosen_indices.append(selected)
            centers.append(x[selected])
            distance = np.minimum(
                distance, np.sum((x - x[selected]) ** 2, axis=1)
            )
        centers_array = np.asarray(centers, dtype=np.float64)
        labels = np.zeros(len(x), dtype=np.int16)
        for _ in range(max_iterations):
            distances = np.sum(
                (x[:, None, :] - centers_array[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(distances, axis=1).astype(np.int16)
            new_centers = centers_array.copy()
            for cluster in range(clusters):
                members = x[new_labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    new_centers[cluster] = x[farthest]
                    new_labels[farthest] = cluster
            if np.array_equal(labels, new_labels) and np.allclose(
                centers_array, new_centers, rtol=0.0, atol=1e-10
            ):
                labels = new_labels
                centers_array = new_centers
                break
            labels = new_labels
            centers_array = new_centers
        inertia = float(np.sum((x - centers_array[labels]) ** 2))
        candidate = (inertia, labels.copy(), centers_array.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def cluster_mapping(
    features: np.ndarray,
    asset_ids: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    clusters: int,
    *,
    seed: int = 2026,
) -> tuple[dict[int, int], dict]:
    assets, profile = asset_profiles(features, asset_ids, residual, weight)
    labels, _, inertia = deterministic_kmeans(profile, clusters, seed=seed)
    mapping = {
        int(asset): int(label) for asset, label in zip(assets, labels)
    }
    sizes = {
        str(cluster): int(np.sum(labels == cluster)) for cluster in range(clusters)
    }
    return mapping, {
        "asset_count": int(len(assets)),
        "profile_columns": int(profile.shape[1]),
        "inertia": inertia,
        "cluster_sizes": sizes,
        "asset_clusters": {str(key): value for key, value in mapping.items()},
    }


def assigned_clusters(asset_ids: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    assets = np.asarray(asset_ids, dtype=np.int64)
    missing = sorted(set(map(int, np.unique(assets))) - set(mapping))
    if missing:
        raise ValueError(f"cluster mapping is missing assets: {missing}")
    return np.fromiter(
        (mapping[int(asset)] for asset in assets), dtype=np.int16, count=len(assets)
    )


def cocluster_agreement(left: dict[int, int], right: dict[int, int]) -> float:
    assets = sorted(set(left) & set(right))
    if len(assets) < 2:
        return 1.0
    matches = 0
    pairs = 0
    for offset, first in enumerate(assets[:-1]):
        for second in assets[offset + 1:]:
            matches += int(
                (left[first] == left[second]) == (right[first] == right[second])
            )
            pairs += 1
    return float(matches / pairs)


def select_residual_scale(folds: list[dict], candidates: list[float]) -> dict:
    reports = []
    for scale in candidates:
        scores = [
            weighted_zero_mean_r2(
                fold["target"],
                fold["base"] + scale * fold["residual_prediction"],
                fold["weight"],
            )
            for fold in folds
        ]
        base_scores = [float(fold["base_score"]) for fold in folds]
        deltas = np.asarray(scores) - np.asarray(base_scores)
        reports.append({
            "residual_scale": float(scale),
            "fold_scores": list(map(float, scores)),
            "fold_deltas": list(map(float, deltas)),
            "mean_fold_score": float(np.mean(scores)),
            "std_fold_score": float(np.std(scores)),
            "min_fold_score": float(np.min(scores)),
            "latest_fold_score": float(scores[-1]),
            "positive_folds": int(np.sum(deltas > 0.0)),
        })
    reports.sort(key=lambda item: (
        -item["mean_fold_score"],
        -item["latest_fold_score"],
        item["residual_scale"],
    ))
    selected = dict(reports[0])
    selected["scale_search"] = sorted(
        reports, key=lambda item: item["residual_scale"]
    )
    return selected
