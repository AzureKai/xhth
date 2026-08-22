from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from cluster_features import (
    assigned_clusters,
    center_by_time,
    cluster_mapping,
    cocluster_agreement,
    select_residual_scale,
    weighted_zero_mean_r2,
)


STRATEGY_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = STRATEGY_DIR.parent
DEFAULT_C4_DIR = EXAMPLES_DIR / "responder_assisted_lgb_catboost_strategy"
SCHEMA_VERSION = 1
START = time.perf_counter()
FINAL_SEEDS = (2026,)


def progress(message: str) -> None:
    print(f"[cluster {time.perf_counter() - START:8.1f}s] {message}", flush=True)


def progress_bar(label: str, current: int, total: int, detail: str = "") -> None:
    width = 28
    fraction = min(max(current / max(total, 1), 0.0), 1.0)
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{label}] [{bar}] {current}/{total} {detail}",
        end="\n" if current >= total else "",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-safe asset-cluster residual experts over C4 OOF predictions."
    )
    parser.add_argument(
        "--c4-strategy-dir", default=str(DEFAULT_C4_DIR)
    )
    parser.add_argument(
        "--c4-model-dir", default=str(DEFAULT_C4_DIR / "model")
    )
    parser.add_argument(
        "--c4-cache-dir", default=str(DEFAULT_C4_DIR / "work" / "cache")
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--feature-count", type=int, default=24)
    parser.add_argument("--profile-feature-count", type=int, default=12)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--min-assets-per-cluster", type=int, default=3)
    parser.add_argument("--walk-forward-blocks", type=int, default=5)
    parser.add_argument("--purge-steps", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--residual-scales", default="0,0.25,0.50,0.75,1.0,1.25"
    )
    parser.add_argument("--required-positive-fold-rate", type=float, default=0.75)
    parser.add_argument("--min-cluster-stability", type=float, default=0.65)
    parser.add_argument("--skip-global-control", action="store_true")
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_scales(value: str) -> list[float]:
    scales = sorted(set(
        float(item.strip()) for item in value.split(",") if item.strip()
    ))
    if not scales or scales[0] < 0.0 or scales[-1] > 2.0:
        raise ValueError("residual scales must be in [0, 2]")
    if 0.0 not in scales:
        scales.insert(0, 0.0)
    return scales


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def signature(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_prediction_artifact(path: Path, development: bool) -> dict:
    required = {"time_id", "asset_id", "target", "weight", "prediction"}
    if development:
        required.add("fold_id")
    with np.load(path) as source:
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"missing prediction artifact fields {missing}: {path}")
        result = {name: np.asarray(source[name]) for name in required}
    rows = len(result["time_id"])
    if rows < 1 or any(len(value) != rows for value in result.values()):
        raise ValueError(f"unaligned prediction artifact: {path}")
    if np.any(np.diff(result["time_id"].astype(np.int64)) < 0):
        raise ValueError(f"prediction artifact is not sorted: {path}")
    return result


def select_raw_features(
    importance_path: Path,
    variant: str,
    raw_feature_names: list[str],
    count: int,
) -> list[str]:
    gains: dict[str, float] = {}
    with importance_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("feature", "")
            if row.get("variant") != variant or name not in raw_feature_names:
                continue
            gains[name] = gains.get(name, 0.0) + float(row["importance_gain"])
    ranked = sorted(gains, key=lambda name: (-gains[name], name))
    if len(ranked) < count:
        raise ValueError(
            f"only {len(ranked)} raw features have importance for {variant}"
        )
    return ranked[:count]


def build_feature_stage(
    cache_dir: Path,
    cache_metadata: dict,
    artifact: dict,
    feature_indices: list[int],
    output_path: Path,
    label: str,
) -> None:
    rows = len(artifact["time_id"])
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32,
        shape=(rows, len(feature_indices)),
    )
    lower = int(artifact["time_id"][0])
    upper = int(artifact["time_id"][-1])
    offset = 0
    shards = cache_metadata["shards"]
    relevant = [
        shard for shard in shards
        if int(shard["time_max"]) >= lower and int(shard["time_min"]) <= upper
    ]
    for index, shard in enumerate(relevant, start=1):
        shard_id = int(shard["id"])
        prefix = cache_dir / f"shard_{shard_id:05d}"
        times = np.load(str(prefix) + "_time.npy", mmap_mode="r")
        start = int(np.searchsorted(times, lower, side="left"))
        end = int(np.searchsorted(times, upper, side="right"))
        if end <= start:
            continue
        count = end - start
        expected_end = offset + count
        if expected_end > rows:
            raise ValueError(f"{label} cache rows exceed prediction artifact")
        cache_times = np.asarray(times[start:end], dtype=np.int64)
        if not np.array_equal(cache_times, artifact["time_id"][offset:expected_end]):
            raise ValueError(f"{label} cache time_id alignment failed")
        matrix = np.load(str(prefix) + "_x.npy", mmap_mode="r")
        cache_assets = np.asarray(matrix[start:end, -1], dtype=np.int64)
        if not np.array_equal(
            cache_assets, artifact["asset_id"][offset:expected_end].astype(np.int64)
        ):
            raise ValueError(f"{label} cache asset_id alignment failed")
        output[offset:expected_end] = np.asarray(
            matrix[start:end, feature_indices], dtype=np.float32
        )
        offset = expected_end
        progress_bar(
            f"{label} feature stage", index, len(relevant), f"rows={offset:,}"
        )
        del matrix, times
    output.flush()
    del output
    if offset != rows:
        raise ValueError(f"{label} feature stage has {offset:,}/{rows:,} rows")


def walk_forward_slices(
    time_ids: np.ndarray,
    blocks: int,
    purge_steps: int,
) -> list[dict]:
    unique_times = np.unique(np.asarray(time_ids, dtype=np.int64))
    if blocks < 4 or len(unique_times) < blocks * 2:
        raise ValueError("walk-forward blocks are insufficient")
    time_blocks = np.array_split(unique_times, blocks)
    result = []
    for index in range(1, blocks):
        valid_times = time_blocks[index]
        valid_start = int(np.searchsorted(time_ids, valid_times[0], side="left"))
        valid_end = int(np.searchsorted(time_ids, valid_times[-1], side="right"))
        valid_time_index = int(np.searchsorted(unique_times, valid_times[0]))
        train_time_index = max(0, valid_time_index - purge_steps)
        train_end_time = unique_times[train_time_index - 1] if train_time_index else -1
        train_end = int(np.searchsorted(time_ids, train_end_time, side="right"))
        if train_end < 1:
            raise ValueError("purged training prefix is empty")
        result.append({
            "fold": index,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "train_time_end": int(time_ids[train_end - 1]),
            "valid_time_start": int(time_ids[valid_start]),
            "valid_time_end": int(time_ids[valid_end - 1]),
        })
    return result


def model_params(seed: int, threads: int) -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 20.0,
        "max_bin": 127,
        "verbosity": -1,
        "num_threads": threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "deterministic": True,
        "force_col_wise": True,
    }


def matrix_for_rows(
    features: np.ndarray,
    base_prediction: np.ndarray,
    rows,
) -> np.ndarray:
    selected = np.asarray(features[rows], dtype=np.float32)
    base = np.asarray(base_prediction[rows], dtype=np.float32).reshape(-1, 1)
    return np.hstack([selected, base])


def train_residual_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    valid_x: np.ndarray | None,
    valid_y: np.ndarray | None,
    valid_w: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
    label: str,
    rounds: int | None = None,
) -> lgb.Booster:
    progress(f"training {label}: rows={len(train_y):,}, features={train_x.shape[1]}")
    train_set = lgb.Dataset(
        train_x, label=train_y, weight=train_w, free_raw_data=True
    )
    valid_sets = None
    callbacks = [lgb.log_evaluation(50)]
    if valid_x is not None and len(valid_x):
        valid_set = lgb.Dataset(
            valid_x, label=valid_y, weight=valid_w,
            reference=train_set, free_raw_data=True,
        )
        valid_sets = [valid_set]
        callbacks.insert(0, lgb.early_stopping(args.early_stopping))
    return lgb.train(
        model_params(seed, args.threads), train_set,
        num_boost_round=rounds or args.rounds, valid_sets=valid_sets,
        callbacks=callbacks,
    )


def predict(model: lgb.Booster, matrix: np.ndarray) -> np.ndarray:
    rounds = int(model.best_iteration or model.current_iteration())
    return np.asarray(model.predict(matrix, num_iteration=rounds), dtype=np.float64)


def train_cluster_set(
    features: np.ndarray,
    artifact: dict,
    train_rows,
    valid_rows,
    mapping: dict[int, int],
    args: argparse.Namespace,
    label: str,
    save_dir: Path | None = None,
    seeds: tuple[int, ...] = (2026,),
) -> tuple[np.ndarray, list[lgb.Booster]]:
    train_assets = artifact["asset_id"][train_rows]
    valid_assets = artifact["asset_id"][valid_rows]
    train_clusters = assigned_clusters(train_assets, mapping)
    valid_clusters = assigned_clusters(valid_assets, mapping)
    residual_target = artifact["target"] - artifact["prediction"]
    output = np.zeros(len(valid_assets), dtype=np.float64)
    models = []
    jobs = args.clusters * len(seeds)
    completed = 0
    for cluster in range(args.clusters):
        train_mask = train_clusters == cluster
        valid_mask = valid_clusters == cluster
        if np.count_nonzero(train_mask) < 1000 or np.count_nonzero(valid_mask) < 1:
            raise ValueError(f"cluster {cluster} has insufficient train/valid rows")
        train_indices = np.flatnonzero(train_mask)
        valid_indices = np.flatnonzero(valid_mask)
        absolute_train = train_indices if isinstance(train_rows, slice) else np.asarray(train_rows)[train_indices]
        absolute_valid = valid_indices if isinstance(valid_rows, slice) else np.asarray(valid_rows)[valid_indices]
        if isinstance(train_rows, slice):
            start = train_rows.start or 0
            absolute_train = absolute_train + start
        if isinstance(valid_rows, slice):
            start = valid_rows.start or 0
            absolute_valid = absolute_valid + start
        train_x = matrix_for_rows(features, artifact["prediction"], absolute_train)
        valid_x = matrix_for_rows(features, artifact["prediction"], absolute_valid)
        cluster_prediction = np.zeros(len(valid_indices), dtype=np.float64)
        for seed in seeds:
            model = train_residual_model(
                train_x,
                residual_target[absolute_train], artifact["weight"][absolute_train],
                valid_x,
                residual_target[absolute_valid], artifact["weight"][absolute_valid],
                args, seed + cluster * 1009,
                f"{label} cluster={cluster} seed={seed}",
            )
            cluster_prediction += predict(model, valid_x) / len(seeds)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(save_dir / f"cluster_{cluster}_seed{seed}.txt"))
            models.append(model)
            completed += 1
            progress_bar(label, completed, jobs, f"cluster={cluster} seed={seed}")
        output[valid_indices] = cluster_prediction
        del train_x, valid_x
        gc.collect()
    return output, models


def fit_global_control(
    features: np.ndarray,
    artifact: dict,
    train_slice: slice,
    valid_slice: slice,
    args: argparse.Namespace,
    seed: int,
    label: str,
    rounds: int | None = None,
) -> tuple[np.ndarray, lgb.Booster]:
    train_x = matrix_for_rows(features, artifact["prediction"], train_slice)
    valid_x = matrix_for_rows(features, artifact["prediction"], valid_slice)
    residual = artifact["target"] - artifact["prediction"]
    model = train_residual_model(
        train_x, residual[train_slice], artifact["weight"][train_slice],
        valid_x, residual[valid_slice], artifact["weight"][valid_slice],
        args, seed, label, rounds=rounds,
    )
    prediction = (
        predict(model, valid_x) if len(valid_x) else np.empty(0, dtype=np.float64)
    )
    del train_x, valid_x
    gc.collect()
    return prediction, model


def fold_report(fold: dict, artifact: dict, residual_prediction: np.ndarray) -> dict:
    valid = slice(fold["valid_start"], fold["valid_end"])
    target = artifact["target"][valid]
    base = artifact["prediction"][valid]
    weight = artifact["weight"][valid]
    return {
        "fold_id": int(fold["fold"]),
        "target": target,
        "base": base,
        "weight": weight,
        "residual_prediction": residual_prediction,
        "base_score": weighted_zero_mean_r2(target, base, weight),
    }


def serializable_fold(report: dict, fold: dict) -> dict:
    return {
        "fold_id": report["fold_id"],
        "train_time_end": fold["train_time_end"],
        "valid_time_start": fold["valid_time_start"],
        "valid_time_end": fold["valid_time_end"],
        "train_rows": fold["train_end"],
        "valid_rows": fold["valid_end"] - fold["valid_start"],
        "base_score": report["base_score"],
    }


def main() -> None:
    args = parse_args()
    if args.feature_count < 4 or not 2 <= args.profile_feature_count <= args.feature_count:
        raise ValueError("feature counts must satisfy 4 <= profile <= feature")
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")
    if args.min_assets_per_cluster < 2:
        raise ValueError("--min-assets-per-cluster must be at least 2")
    if not 0.0 < args.required_positive_fold_rate <= 1.0:
        raise ValueError("--required-positive-fold-rate must be in (0,1]")
    if not 0.0 <= args.min_cluster_stability <= 1.0:
        raise ValueError("--min-cluster-stability must be in [0,1]")
    scales = configured_scales(args.residual_scales)
    c4_strategy_dir = Path(args.c4_strategy_dir)
    c4_model_dir = Path(args.c4_model_dir)
    cache_dir = Path(args.c4_cache_dir)
    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = c4_model_dir / "metadata.json"
    c4_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if c4_metadata.get("target_variant") != "LGB468_C4_STABLE":
        raise ValueError("asset-cluster experiment requires LGB468_C4_STABLE")
    cache_metadata_path = cache_dir / "cache.json"
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    raw_feature_names = list(cache_metadata["feature_columns"])
    importance_path = c4_model_dir / c4_metadata["target_feature_importance"]
    selected_features = select_raw_features(
        importance_path, c4_metadata["target_variant"],
        raw_feature_names, args.feature_count,
    )
    feature_indices = [raw_feature_names.index(name) for name in selected_features]
    development_path = c4_model_dir / c4_metadata["development_oof_predictions"]
    holdout_path = c4_model_dir / c4_metadata["validation_predictions"]
    source_fingerprints = {
        "c4_metadata": fingerprint(metadata_path),
        "importance": fingerprint(importance_path),
        "cache_metadata": fingerprint(cache_metadata_path),
        "development_oof": fingerprint(development_path),
        "holdout": fingerprint(holdout_path),
    }
    training_config = {
        "feature_count": args.feature_count,
        "profile_feature_count": args.profile_feature_count,
        "clusters": args.clusters,
        "min_assets_per_cluster": args.min_assets_per_cluster,
        "walk_forward_blocks": args.walk_forward_blocks,
        "purge_steps": args.purge_steps,
        "rounds": args.rounds,
        "early_stopping": args.early_stopping,
        "residual_scales": scales,
        "required_positive_fold_rate": args.required_positive_fold_rate,
        "min_cluster_stability": args.min_cluster_stability,
        "global_control": not args.skip_global_control,
        "selected_features": selected_features,
    }
    output_metadata_path = model_dir / "metadata.json"
    if args.skip_existing_models and output_metadata_path.exists():
        existing = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        model_files = [model_dir / name for name in existing.get("cluster_models", [])]
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("source_artifacts") == source_fingerprints
            and existing.get("training_config") == training_config
            and all(path.exists() for path in model_files)
        ):
            progress("compatible final cluster experiment exists; training skipped")
            print(output_metadata_path.read_text(encoding="utf-8"))
            return

    progress("loading C4 development OOF and frozen terminal holdout")
    development = load_prediction_artifact(development_path, development=True)
    holdout = load_prediction_artifact(holdout_path, development=False)
    stage_payload = {
        "source_artifacts": source_fingerprints,
        "selected_features": selected_features,
        "development_rows": len(development["time_id"]),
        "holdout_rows": len(holdout["time_id"]),
    }
    stage_signature = signature(stage_payload)
    stage_path = work_dir / "feature_stage.json"
    development_feature_path = work_dir / "development_features.npy"
    holdout_feature_path = work_dir / "holdout_features.npy"
    stage_valid = False
    if stage_path.exists() and development_feature_path.exists() and holdout_feature_path.exists():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        stage_valid = stage.get("signature") == stage_signature
        if stage_valid:
            stage_valid = (
                tuple(np.load(development_feature_path, mmap_mode="r").shape)
                == (len(development["time_id"]), len(selected_features))
                and tuple(np.load(holdout_feature_path, mmap_mode="r").shape)
                == (len(holdout["time_id"]), len(selected_features))
            )
    if stage_valid:
        progress("loading compatible staged expert features")
    else:
        progress("building aligned development expert feature stage")
        build_feature_stage(
            cache_dir, cache_metadata, development, feature_indices,
            development_feature_path, "development",
        )
        progress("building aligned holdout expert feature stage")
        build_feature_stage(
            cache_dir, cache_metadata, holdout, feature_indices,
            holdout_feature_path, "holdout",
        )
        stage_path.write_text(json.dumps({
            "signature": stage_signature, **stage_payload,
        }, indent=2), encoding="utf-8")
    development_features = np.load(development_feature_path, mmap_mode="r")
    holdout_features = np.load(holdout_feature_path, mmap_mode="r")
    folds = walk_forward_slices(
        development["time_id"], args.walk_forward_blocks, args.purge_steps
    )

    cv_dir = work_dir / "cv_models"
    cv_dir.mkdir(exist_ok=True)
    cluster_folds = []
    global_folds = []
    fold_mappings = []
    cluster_cv_rounds = []
    global_cv_rounds = []
    fold_audits = []
    residual = development["target"] - development["prediction"]
    for fold_index, fold in enumerate(folds, start=1):
        train_slice = slice(0, fold["train_end"])
        valid_slice = slice(fold["valid_start"], fold["valid_end"])
        progress(
            f"walk-forward fold {fold_index}/{len(folds)}: "
            f"train<= {fold['train_time_end']}, "
            f"valid={fold['valid_time_start']}-{fold['valid_time_end']}"
        )
        mapping, mapping_audit = cluster_mapping(
            np.asarray(
                development_features[train_slice, :args.profile_feature_count],
                dtype=np.float32,
            ),
            development["asset_id"][train_slice], residual[train_slice],
            development["weight"][train_slice], args.clusters,
            seed=2026 + fold_index,
        )
        fold_mappings.append(mapping)
        fold_dir = cv_dir / f"fold_{fold_index}"
        cluster_prediction, models = train_cluster_set(
            development_features, development,
            train_slice, valid_slice, mapping, args,
            f"fold {fold_index} cluster experts", save_dir=fold_dir,
        )
        cluster_prediction = center_by_time(
            cluster_prediction, development["time_id"][valid_slice]
        )
        cluster_report = fold_report(fold, development, cluster_prediction)
        cluster_folds.append(cluster_report)
        fold_cluster_rounds = [
            int(model.best_iteration or args.rounds) for model in models
        ]
        cluster_cv_rounds.append(fold_cluster_rounds)
        del models
        if not args.skip_global_control:
            global_prediction, global_model = fit_global_control(
                development_features, development,
                train_slice, valid_slice, args, 3000 + fold_index,
                f"fold {fold_index} global residual control",
            )
            global_prediction = center_by_time(
                global_prediction, development["time_id"][valid_slice]
            )
            global_folds.append(fold_report(fold, development, global_prediction))
            global_cv_rounds.append(
                int(global_model.best_iteration or args.rounds)
            )
            global_model.save_model(str(fold_dir / "global_control.txt"))
            del global_model
        fold_audits.append({
            **serializable_fold(cluster_report, fold),
            "mapping": mapping_audit,
            "cluster_best_iterations": fold_cluster_rounds,
            "global_best_iteration": (
                global_cv_rounds[-1] if global_cv_rounds else None
            ),
        })
        gc.collect()

    cluster_selection = select_residual_scale(cluster_folds, scales)
    global_selection = (
        select_residual_scale(global_folds, scales)
        if global_folds else None
    )
    base_fold_scores = [fold["base_score"] for fold in cluster_folds]
    cluster_stabilities = [
        cocluster_agreement(left, right)
        for left, right in zip(fold_mappings[:-1], fold_mappings[1:])
    ]
    mean_cluster_stability = float(np.mean(cluster_stabilities))
    progress(
        f"OOF selected cluster scale={cluster_selection['residual_scale']:.2f}, "
        f"mean_R2={cluster_selection['mean_fold_score']:.8f}, "
        f"mapping_stability={mean_cluster_stability:.3f}"
    )
    cluster_holdout_rounds = [
        max(1, int(round(np.mean([
            fold_rounds[cluster] for fold_rounds in cluster_cv_rounds
        ]))))
        for cluster in range(args.clusters)
    ]
    global_holdout_rounds = (
        max(1, int(round(np.mean(global_cv_rounds))))
        if global_cv_rounds else args.rounds
    )

    progress("training development-only experts for frozen terminal holdout")
    full_mapping, holdout_mapping_audit = cluster_mapping(
        np.asarray(
            development_features[:, :args.profile_feature_count], dtype=np.float32
        ),
        development["asset_id"], residual, development["weight"],
        args.clusters, seed=4040,
    )
    # Holdout features live in a separate memmap, so train and predict explicitly.
    holdout_cluster_prediction = np.zeros(len(holdout["target"]), dtype=np.float64)
    development_clusters = assigned_clusters(development["asset_id"], full_mapping)
    holdout_clusters = assigned_clusters(holdout["asset_id"], full_mapping)
    holdout_models = []
    holdout_best_rounds = []
    for cluster in range(args.clusters):
        train_rows = np.flatnonzero(development_clusters == cluster)
        valid_rows = np.flatnonzero(holdout_clusters == cluster)
        train_x = matrix_for_rows(
            development_features, development["prediction"], train_rows
        )
        valid_x = matrix_for_rows(holdout_features, holdout["prediction"], valid_rows)
        model = train_residual_model(
            train_x, residual[train_rows], development["weight"][train_rows],
            None, None, None, args, 5000 + cluster,
            f"holdout cluster={cluster}",
            rounds=cluster_holdout_rounds[cluster],
        )
        holdout_cluster_prediction[valid_rows] = predict(model, valid_x)
        holdout_models.append(model)
        holdout_best_rounds.append(cluster_holdout_rounds[cluster])
        progress_bar("holdout cluster experts", cluster + 1, args.clusters)
        del train_x, valid_x
        gc.collect()
    holdout_cluster_prediction = center_by_time(
        holdout_cluster_prediction, holdout["time_id"]
    )
    cluster_scale = float(cluster_selection["residual_scale"])
    holdout_base_score = weighted_zero_mean_r2(
        holdout["target"], holdout["prediction"], holdout["weight"]
    )
    holdout_cluster_score = weighted_zero_mean_r2(
        holdout["target"],
        holdout["prediction"] + cluster_scale * holdout_cluster_prediction,
        holdout["weight"],
    )
    holdout_global_score = holdout_base_score
    if not args.skip_global_control:
        global_holdout_prediction, global_holdout_model = fit_global_control(
            development_features, development,
            slice(0, len(development["target"])), slice(0, 0),
            args, 6000, "holdout global residual control",
            rounds=global_holdout_rounds,
        )
        # The helper cannot use a second feature store for prediction.
        holdout_global_x = matrix_for_rows(
            holdout_features, holdout["prediction"], slice(0, len(holdout["target"]))
        )
        global_holdout_prediction = center_by_time(
            predict(global_holdout_model, holdout_global_x), holdout["time_id"]
        )
        global_scale = float(global_selection["residual_scale"])
        holdout_global_score = weighted_zero_mean_r2(
            holdout["target"],
            holdout["prediction"] + global_scale * global_holdout_prediction,
            holdout["weight"],
        )
        del global_holdout_model, holdout_global_x
    del holdout_models
    gc.collect()

    required_positive = int(math.ceil(
        len(folds) * args.required_positive_fold_rate
    ))
    mapping_cluster_sizes = [
        int(size)
        for audit in [
            *(item["mapping"] for item in fold_audits),
            holdout_mapping_audit,
        ]
        for size in audit["cluster_sizes"].values()
    ]
    minimum_cluster_assets = min(mapping_cluster_sizes)
    base_mean = float(np.mean(base_fold_scores))
    base_latest = float(base_fold_scores[-1])
    global_mean = (
        float(global_selection["mean_fold_score"])
        if global_selection else base_mean
    )
    global_latest = (
        float(global_selection["latest_fold_score"])
        if global_selection else base_latest
    )
    gates = {
        "positive_residual_scale": bool(cluster_scale > 0.0),
        "mean_oof_beats_c4": bool(cluster_selection["mean_fold_score"] > base_mean),
        "mean_oof_beats_global_control": bool(
            cluster_selection["mean_fold_score"] > global_mean
        ),
        "latest_oof_beats_c4": bool(
            cluster_selection["latest_fold_score"] > base_latest
        ),
        "latest_oof_beats_global_control": bool(
            cluster_selection["latest_fold_score"] > global_latest
        ),
        "enough_positive_folds": bool(
            cluster_selection["positive_folds"] >= required_positive
        ),
        "cluster_mapping_stable": bool(
            mean_cluster_stability >= args.min_cluster_stability
        ),
        "no_identity_sized_cluster": bool(
            minimum_cluster_assets >= args.min_assets_per_cluster
        ),
        "holdout_beats_c4": bool(holdout_cluster_score > holdout_base_score),
        "holdout_beats_global_control": bool(
            holdout_cluster_score > holdout_global_score
        ),
    }
    gates["passed"] = bool(all(gates.values()))
    deployment_scale = cluster_scale if gates["passed"] else 0.0

    cluster_model_files = []
    final_mapping_audit = holdout_mapping_audit
    if gates["passed"]:
        progress("promotion passed; fitting final cluster ensemble on all labeled rows")
        combined_residual = np.concatenate([
            residual, holdout["target"] - holdout["prediction"]
        ])
        # Freeze routing before examining holdout labels; holdout is added only
        # to the final expert fits after every promotion gate has passed.
        final_mapping = full_mapping
        development_final_clusters = assigned_clusters(
            development["asset_id"], final_mapping
        )
        holdout_final_clusters = assigned_clusters(holdout["asset_id"], final_mapping)
        importance_rows = []
        for cluster in range(args.clusters):
            dev_rows = np.flatnonzero(development_final_clusters == cluster)
            val_rows = np.flatnonzero(holdout_final_clusters == cluster)
            train_x = np.concatenate([
                matrix_for_rows(development_features, development["prediction"], dev_rows),
                matrix_for_rows(holdout_features, holdout["prediction"], val_rows),
            ])
            train_y = np.concatenate([residual[dev_rows], combined_residual[len(residual):][val_rows]])
            train_w = np.concatenate([
                development["weight"][dev_rows], holdout["weight"][val_rows]
            ])
            for seed in FINAL_SEEDS:
                model = train_residual_model(
                    train_x, train_y, train_w,
                    None, None, None, args, seed + cluster * 1009,
                    f"final cluster={cluster} seed={seed}",
                    rounds=holdout_best_rounds[cluster],
                )
                filename = f"cluster_{cluster}_seed{seed}.txt"
                model.save_model(str(model_dir / filename))
                cluster_model_files.append(filename)
                for name, gain, split in zip(
                    [*selected_features, "c4_prediction"],
                    model.feature_importance(importance_type="gain"),
                    model.feature_importance(importance_type="split"),
                ):
                    importance_rows.append((cluster, seed, name, float(gain), int(split)))
                del model
            del train_x, train_y, train_w
            progress_bar("final cluster ensemble", cluster + 1, args.clusters)
            gc.collect()
        with (model_dir / "feature_importance.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "cluster", "seed", "feature", "importance_gain", "importance_split"
            ])
            writer.writerows(importance_rows)
    else:
        progress("promotion failed; deployment falls back to unmodified C4")
        for path in model_dir.glob("cluster_*_seed*.txt"):
            path.unlink()

    report = {
        "strategy": "lgb_asset_cluster_residual_strategy",
        "schema_version": SCHEMA_VERSION,
        "prediction_formula": "C4 + residual_scale * centered_cluster_expert",
        "base_strategy_dir": os.path.relpath(c4_strategy_dir.resolve(), STRATEGY_DIR),
        "base_model_dir": os.path.relpath(c4_model_dir.resolve(), STRATEGY_DIR),
        "selected_features": selected_features,
        "profile_features": selected_features[:args.profile_feature_count],
        "asset_id_usage": "routing_only_not_model_feature",
        "cluster_count": args.clusters,
        "asset_clusters": final_mapping_audit["asset_clusters"],
        "cluster_sizes": final_mapping_audit["cluster_sizes"],
        "cluster_models": cluster_model_files,
        "cluster_model_seeds": list(FINAL_SEEDS),
        "cluster_final_rounds": holdout_best_rounds,
        "cluster_cv_mean_rounds": cluster_holdout_rounds,
        "global_cv_mean_rounds": global_holdout_rounds,
        "selected_residual_scale": cluster_scale,
        "deployment_residual_scale": deployment_scale,
        "deployment_source": "cluster_experts" if gates["passed"] else "c4_fallback",
        "base_oof": {
            "fold_scores": list(map(float, base_fold_scores)),
            "mean_fold_score": base_mean,
            "latest_fold_score": base_latest,
        },
        "cluster_oof": cluster_selection,
        "global_control_oof": global_selection,
        "folds": fold_audits,
        "cluster_stability": {
            "adjacent_fold_agreements": cluster_stabilities,
            "mean_agreement": mean_cluster_stability,
            "required": args.min_cluster_stability,
            "minimum_cluster_assets": minimum_cluster_assets,
            "required_minimum_cluster_assets": args.min_assets_per_cluster,
        },
        "holdout": {
            "c4_score": holdout_base_score,
            "global_control_score": holdout_global_score,
            "cluster_score": holdout_cluster_score,
            "delta_vs_c4": holdout_cluster_score - holdout_base_score,
            "delta_vs_global_control": holdout_cluster_score - holdout_global_score,
            "time_start": int(holdout["time_id"][0]),
            "time_end": int(holdout["time_id"][-1]),
            "rows": int(len(holdout["target"])),
        },
        "promotion_gates": gates,
        "source_artifacts": source_fingerprints,
        "training_config": training_config,
        "feature_stage": "feature_stage.json",
        "feature_importance": "feature_importance.csv" if gates["passed"] else None,
    }
    report["signature"] = signature({
        "source_artifacts": source_fingerprints,
        "training_config": training_config,
    })
    output_metadata_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (work_dir / "cluster_experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    progress(
        f"complete: OOF={cluster_selection['mean_fold_score']:.8f}, "
        f"holdout={holdout_cluster_score:.8f}, "
        f"scale={deployment_scale:.2f}, gates={gates['passed']}"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
