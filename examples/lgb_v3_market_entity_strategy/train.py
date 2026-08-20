from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parent
BASELINE_DIR = STRATEGY_DIR.parent / "lightgbm_baseline"
SOURCE_STRATEGY_DIR = STRATEGY_DIR.parent / "lgb_v3_regime_residual_strategy"
_THIS_TRAIN_MODULE = sys.modules.pop("train", None) if __name__ == "train" else None
sys.path.insert(0, str(BASELINE_DIR))
import train_low_memory as baseline  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402
sys.path.pop(0)
if _THIS_TRAIN_MODULE is not None:
    sys.modules["train"] = _THIS_TRAIN_MODULE

from decomposition_features import (  # noqa: E402
    MarketEntityFeatureBuilder,
    center_cross_section,
    decompose_residual,
    entity_feature_names,
    entity_model_indices,
    market_feature_names,
    market_features_from_entity_matrix,
)


START = time.perf_counter()
FINAL_SEEDS = (2026, 2027, 2028)


def progress(message: str) -> None:
    print(f"[progress {time.perf_counter() - START:9.1f}s] {message}", flush=True)


def progress_bar(label: str, current: int, total: int, detail: str = "") -> None:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    width = 28
    filled = int(width * current / total)
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{label:<22}] [{'#' * filled}{'-' * (width - filled)}] "
        f"{100.0 * current / total:6.2f}% ({current:,}/{total:,}){suffix}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict V3 market/common and entity/idiosyncratic residual decomposition."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument(
        "--source-work-dir",
        default=str(SOURCE_STRATEGY_DIR / "work"),
        help="Existing strict V3 OOF/state artifacts from lgb_v3_regime_residual_strategy.",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state-feature-count", type=int, default=16)
    parser.add_argument(
        "--component-weights",
        default="0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0",
    )
    parser.add_argument("--market-rounds", type=int, default=500)
    parser.add_argument("--entity-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_weights(value: str) -> list[float]:
    weights = sorted(set(
        float(item.strip()) for item in value.split(",") if item.strip()
    ))
    if not weights or any(item < 0.0 or item > 1.0 for item in weights):
        raise ValueError("component weights must be in [0, 1]")
    if 0.0 not in weights:
        weights.insert(0, 0.0)
    return weights


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def portable_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"name": path.name, "size": int(stat.st_size)}


def signature(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_vector_spans(path: Path, spans: list[tuple[int, int]]) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    if len(spans) == 1:
        start, end = spans[0]
        return np.asarray(source[start:end]).copy()
    return np.concatenate([
        np.asarray(source[start:end]) for start, end in spans
    ])


class ColumnSequence(lgb.Sequence):
    batch_size = baseline.SEQUENCE_BATCH_ROWS

    def __init__(self, source, columns: list[int]):
        self.source = source
        self.columns = np.asarray(columns, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index):
        values = np.asarray(self.source[index])
        return values[self.columns] if values.ndim == 1 else values[:, self.columns]


def build_entity_dataset(
    feature_path: Path,
    label_path: Path,
    weight_path: Path,
    spans: list[tuple[int, int]],
    feature_names: list[str],
    columns: list[int],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    shape = tuple(np.load(feature_path, mmap_mode="r").shape)
    source = baseline.SpannedMemmapSequence(feature_path, shape, spans)
    sequence = ColumnSequence(source, columns)
    labels = read_vector_spans(label_path, spans).astype(np.float32)
    weights = read_vector_spans(weight_path, spans).astype(np.float32)
    names = [feature_names[index] for index in columns]
    dataset = lgb.Dataset(
        sequence,
        label=labels,
        weight=weights,
        reference=reference,
        feature_name=names,
        categorical_feature=["asset_id"] if "asset_id" in names else [],
        params={
            "data_random_seed": 2026,
            "min_data_in_leaf": 5000,
            "max_bin": 255,
            "force_col_wise": True,
            "verbosity": -1,
        },
        free_raw_data=True,
    )
    dataset.construct()
    return dataset


def component_params(component: str, seed: int, threads: int) -> dict:
    is_market = component == "market"
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 15 if is_market else 31,
        "max_depth": 6 if is_market else 9,
        "min_data_in_leaf": 1000 if is_market else 5000,
        "feature_fraction": 0.85 if is_market else 0.80,
        "feature_fraction_bynode": 0.85 if is_market else 0.80,
        "bagging_fraction": 0.85 if is_market else 0.80,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 30.0 if is_market else 40.0,
        "path_smooth": 100.0 if is_market else 150.0,
        "min_gain_to_split": 0.005 if is_market else 0.01,
        "deterministic": True,
        "force_col_wise": True,
        "histogram_pool_size": 4096.0,
        "verbosity": -1,
        "num_threads": int(threads),
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "data_random_seed": 2026,
    }


def train_component(
    component: str,
    train_set: lgb.Dataset,
    valid_set: lgb.Dataset | None,
    rounds: int,
    early_stopping: int,
    threads: int,
    seed: int = 2026,
) -> lgb.Booster:
    callbacks = [lgb.log_evaluation(50)]
    valid_sets = [train_set]
    valid_names = ["train"]
    if valid_set is not None:
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.insert(0, lgb.early_stopping(early_stopping, verbose=False))
    return lgb.train(
        component_params(component, seed, threads),
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_entity(
    model: lgb.Booster,
    feature_path: Path,
    spans: list[tuple[int, int]],
    columns: list[int],
    rounds: int,
) -> np.ndarray:
    matrix = np.load(feature_path, mmap_mode="r")
    parts = []
    for span_start, span_end in spans:
        for start in range(span_start, span_end, baseline.PREDICT_BATCH_ROWS):
            end = min(start + baseline.PREDICT_BATCH_ROWS, span_end)
            values = np.asarray(matrix[start:end, columns], dtype=np.float32)
            parts.append(np.asarray(
                model.predict(values, num_iteration=rounds, num_threads=1),
                dtype=np.float64,
            ))
    return np.concatenate(parts)


def time_positions(unique_times: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(unique_times, np.asarray(time_ids, dtype=np.int64))
    if np.any(positions >= len(unique_times)) or not np.array_equal(
        unique_times[positions], np.asarray(time_ids, dtype=np.int64)
    ):
        raise ValueError("validation plan contains unknown time ids")
    return positions.astype(np.int64, copy=False)


def expand_market_prediction(
    prediction: np.ndarray,
    positions: np.ndarray,
    row_offsets: np.ndarray,
) -> np.ndarray:
    counts = row_offsets[positions + 1] - row_offsets[positions]
    return np.repeat(np.asarray(prediction, dtype=np.float64), counts)


def center_entity_by_time(
    prediction: np.ndarray,
    positions: np.ndarray,
    row_offsets: np.ndarray,
) -> np.ndarray:
    output = np.asarray(prediction, dtype=np.float64).copy()
    counts = row_offsets[positions + 1] - row_offsets[positions]
    cursor = 0
    for count in counts:
        end = cursor + int(count)
        output[cursor:end] = center_cross_section(output[cursor:end])
        cursor = end
    if cursor != len(output):
        raise ValueError("entity prediction rows do not match time positions")
    return output


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    x = x[finite]
    y = y[finite]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_market_dataset(
    feature_path: Path,
    target_path: Path,
    base_path: Path,
    weight_path: Path,
    positions: np.ndarray,
    feature_names: list[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    features = np.load(feature_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    weights = np.load(weight_path, mmap_mode="r")
    dataset = lgb.Dataset(
        np.asarray(features[positions], dtype=np.float64),
        label=np.asarray(target[positions] - base[positions], dtype=np.float32),
        weight=np.asarray(weights[positions], dtype=np.float32),
        reference=reference,
        feature_name=feature_names,
        params={
            "data_random_seed": 2026,
            "min_data_in_leaf": 1000,
            "max_bin": 255,
            "force_col_wise": True,
            "verbosity": -1,
        },
        free_raw_data=True,
    )
    dataset.construct()
    return dataset


def score_weight_grid(folds: list[dict], weights: list[float]) -> dict:
    reports = []
    base_scores = np.asarray([fold["base_score"] for fold in folds])
    for alpha in weights:
        for beta in weights:
            scores = [
                weighted_zero_mean_r2(
                    fold["target"],
                    fold["base"]
                    + alpha * fold["market_prediction"]
                    + beta * fold["entity_prediction"],
                    fold["weight"],
                )
                for fold in folds
            ]
            reports.append({
                "market_weight": float(alpha),
                "entity_weight": float(beta),
                "fold_scores": list(map(float, scores)),
                "fold_deltas": list(map(float, np.asarray(scores) - base_scores)),
                "mean_fold_score": float(np.mean(scores)),
                "std_fold_score": float(np.std(scores)),
                "min_fold_score": float(np.min(scores)),
                "latest_fold_score": float(scores[-1]),
            })
    reports.sort(key=lambda item: (
        -item["mean_fold_score"],
        item["market_weight"] + item["entity_weight"],
        item["entity_weight"],
    ))
    market_only = next(
        item for item in sorted(
            (item for item in reports if item["entity_weight"] == 0.0),
            key=lambda item: (-item["mean_fold_score"], item["market_weight"]),
        )
    )
    entity_only = next(
        item for item in sorted(
            (item for item in reports if item["market_weight"] == 0.0),
            key=lambda item: (-item["mean_fold_score"], item["entity_weight"]),
        )
    )
    return {
        **reports[0],
        "market_only_selection": {
            key: value for key, value in market_only.items()
        },
        "entity_only_selection": {
            key: value for key, value in entity_only.items()
        },
        "weight_search": reports,
    }


def component_ablation(
    folds: list[dict],
    alpha: float,
    beta: float,
    market_only_weight: float,
    entity_only_weight: float,
) -> dict:
    variants = {
        "BASE": (0.0, 0.0),
        "MARKET_ONLY": (market_only_weight, 0.0),
        "ENTITY_ONLY": (0.0, entity_only_weight),
        "MARKET_ENTITY": (alpha, beta),
    }
    output = {}
    for name, (market_weight, entity_weight) in variants.items():
        scores = [
            weighted_zero_mean_r2(
                fold["target"],
                fold["base"]
                + market_weight * fold["market_prediction"]
                + entity_weight * fold["entity_prediction"],
                fold["weight"],
            )
            for fold in folds
        ]
        output[name] = {
            "market_weight": market_weight,
            "entity_weight": entity_weight,
            "fold_scores": list(map(float, scores)),
            "mean_fold_score": float(np.mean(scores)),
        }
    return output


def require_source_artifacts(source_work_dir: Path) -> dict[str, Path]:
    paths = {
        "base_stage": source_work_dir / "base_oof_stage.json",
        "base_prediction": source_work_dir / "base_oof_prediction.npy",
        "feature_stage": source_work_dir / "residual_feature_stage.json",
        "entity_features": source_work_dir / "residual_features.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing strict V3 source artifacts; first train "
            "lgb_v3_regime_residual_strategy with --skip-existing-models: "
            + ", ".join(missing)
        )
    return paths


def main() -> None:
    args = parse_args()
    if args.state_feature_count < 1:
        raise ValueError("--state-feature-count must be positive")
    weights = configured_weights(args.component_weights)
    data_root = Path(args.data_root)
    base_model_dir = Path(args.base_model_dir)
    base_cache_dir = Path(args.base_cache_dir)
    source_work_dir = Path(args.source_work_dir)
    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    source_paths = require_source_artifacts(source_work_dir)
    base_report_path = base_model_dir / "lightgbm_report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    requested_config = {
        "state_feature_count": args.state_feature_count,
        "component_weights": weights,
        "market_rounds": args.market_rounds,
        "entity_rounds": args.entity_rounds,
        "early_stopping": args.early_stopping,
    }
    source_fingerprints = {
        name: portable_fingerprint(path) for name, path in source_paths.items()
    }
    metadata_path = model_dir / "metadata.json"
    if args.skip_existing_models and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_files = [
            model_dir / filename
            for key in ("market_models", "entity_models")
            for filename in existing.get(key, [])
        ]
        if (
            all(existing.get("training_config", {}).get(key) == value
                for key, value in requested_config.items())
            and existing.get("source_artifacts") == source_fingerprints
            and all(path.exists() for path in model_files)
            and (model_dir / "component_feature_importance.csv").exists()
        ):
            progress("all compatible final artifacts exist; training skipped")
            print(metadata_path.read_text(encoding="utf-8"))
            return

    progress("preparing/reusing V3 low-memory cache and validation plan")
    cache = baseline.prepare_cache(data_root, base_cache_dir)
    if cache["raw_features"] != base_report["features"]["selected_raw_features"]:
        raise ValueError("base cache raw feature schema differs from V3 report")
    if cache["history_features"] != base_report["features"]["history_features"]:
        raise ValueError("base cache history feature schema differs from V3 report")

    axis = np.load(base_cache_dir / cache["time_axis_file"])
    unique_times = axis["unique_times"]
    time_counts = axis["time_counts"]
    row_offsets = baseline.row_offsets_from_counts(time_counts)
    validation = base_report["validation"]
    plan = make_validation_plan(
        unique_times,
        n_splits=int(validation["n_splits"]),
        holdout_fraction=float(validation["holdout_fraction"]),
        purge_steps=int(validation["purge_steps"]),
        min_train_fraction=float(validation["min_train_fraction"]),
    )
    target_path = base_cache_dir / cache["target_file"]
    weight_path = base_cache_dir / cache["weight_file"]
    base_prediction_path = source_paths["base_prediction"]
    entity_feature_path = source_paths["entity_features"]
    total_rows = int(cache["total_rows"])
    if tuple(np.load(base_prediction_path, mmap_mode="r").shape) != (total_rows,):
        raise ValueError("source base OOF prediction shape mismatch")

    state_features = list(cache["history_features"][:args.state_feature_count])
    all_entity_names = entity_feature_names(state_features)
    source_feature_stage = json.loads(
        source_paths["feature_stage"].read_text(encoding="utf-8")
    )
    if source_feature_stage.get("state_features") != state_features:
        raise ValueError("source state feature list differs from requested configuration")
    if source_feature_stage.get("feature_names") != all_entity_names:
        raise ValueError("source entity feature schema mismatch")
    entity_columns = entity_model_indices(all_entity_names)
    all_market_names = market_feature_names(state_features)

    fold_positions = [
        time_positions(unique_times, fold.valid_time_ids) for fold in plan.folds
    ]
    holdout_positions = time_positions(unique_times, plan.holdout_time_ids)
    fold_spans = [
        baseline.spans_from_time_ids(unique_times, row_offsets, fold.valid_time_ids)
        for fold in plan.folds
    ]
    holdout_spans = baseline.spans_from_time_ids(
        unique_times, row_offsets, plan.holdout_time_ids
    )
    relevant_positions = np.concatenate([*fold_positions, holdout_positions])

    market_feature_path = work_dir / "market_features.npy"
    market_target_path = work_dir / "market_target.npy"
    market_base_path = work_dir / "market_base_prediction.npy"
    market_weight_path = work_dir / "market_weight.npy"
    entity_label_path = work_dir / "entity_residual_target.npy"
    decomposition_stage_path = work_dir / "decomposition_stage.json"
    decomposition_signature_payload = {
        "source_artifacts": source_fingerprints,
        "state_features": state_features,
        "entity_feature_names": all_entity_names,
        "market_feature_names": all_market_names,
        "method_version": 1,
        "market_definition": "unweighted within-time mean",
        "entity_constraint": "unweighted within-time zero mean",
    }
    decomposition_signature = signature(decomposition_signature_payload)
    reuse_decomposition = False
    if decomposition_stage_path.exists():
        stage = json.loads(decomposition_stage_path.read_text(encoding="utf-8"))
        reuse_decomposition = bool(
            args.skip_existing_models
            and stage.get("signature") == decomposition_signature
            and all(path.exists() for path in (
                market_feature_path,
                market_target_path,
                market_base_path,
                market_weight_path,
                entity_label_path,
            ))
        )
    if reuse_decomposition:
        progress("reusing market/entity decomposition memmaps")
    else:
        progress(
            f"materializing decomposition: times={len(relevant_positions):,}, "
            f"market_features={len(all_market_names)}"
        )
        entity_matrix = np.load(entity_feature_path, mmap_mode="r")
        base_prediction = np.load(base_prediction_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")
        row_weight = np.load(weight_path, mmap_mode="r")
        market_features = np.lib.format.open_memmap(
            market_feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(unique_times), len(all_market_names)),
        )
        market_target = np.lib.format.open_memmap(
            market_target_path, mode="w+", dtype=np.float32,
            shape=(len(unique_times),),
        )
        market_base = np.lib.format.open_memmap(
            market_base_path, mode="w+", dtype=np.float32,
            shape=(len(unique_times),),
        )
        market_weight = np.lib.format.open_memmap(
            market_weight_path, mode="w+", dtype=np.float32,
            shape=(len(unique_times),),
        )
        entity_label = np.lib.format.open_memmap(
            entity_label_path, mode="w+", dtype=np.float32,
            shape=(total_rows,),
        )
        market_features[:] = 0.0
        market_target[:] = 0.0
        market_base[:] = 0.0
        market_weight[:] = 0.0
        entity_label[:] = 0.0
        report_every = max(1, len(relevant_positions) // 100)
        for completed, position in enumerate(relevant_positions, start=1):
            start = int(row_offsets[position])
            end = int(row_offsets[position + 1])
            current_base = np.asarray(base_prediction[start:end], dtype=np.float64)
            current_target = np.asarray(target[start:end], dtype=np.float64)
            if not np.all(np.isfinite(current_base)):
                raise ValueError(f"missing strict base prediction at time position {position}")
            market_residual, current_entity_label = decompose_residual(
                current_target, current_base
            )
            market_features[position] = market_features_from_entity_matrix(
                entity_matrix[start:end], all_entity_names, state_features
            )
            market_base[position] = np.float32(np.mean(current_base))
            market_target[position] = np.float32(
                market_base[position] + market_residual
            )
            market_weight[position] = np.float32(
                np.sum(row_weight[start:end], dtype=np.float64)
            )
            entity_label[start:end] = current_entity_label
            if (
                completed == 1
                or completed == len(relevant_positions)
                or completed % report_every == 0
            ):
                progress_bar(
                    "decomposition",
                    completed,
                    len(relevant_positions),
                    f"rows={end:,}",
                )
        for memmap in (
            market_features, market_target, market_base,
            market_weight, entity_label,
        ):
            memmap.flush()
        decomposition_stage_path.write_text(json.dumps({
            "signature": decomposition_signature,
            **decomposition_signature_payload,
        }, indent=2), encoding="utf-8")
        del (
            entity_matrix, base_prediction, target, row_weight,
            market_features, market_target, market_base,
            market_weight, entity_label,
        )
        gc.collect()

    cv_dir = work_dir / "component_cv_models"
    cv_dir.mkdir(exist_ok=True)
    folds = []
    market_iterations = []
    entity_iterations = []
    for eval_index in range(1, len(fold_spans)):
        train_positions = np.concatenate(fold_positions[:eval_index])
        valid_positions = fold_positions[eval_index]
        train_spans = [span for group in fold_spans[:eval_index] for span in group]
        valid_spans = fold_spans[eval_index]
        component_predictions = {}
        component_iterations = {}
        market_time_prediction = None
        for component, requested_rounds in (
            ("market", args.market_rounds),
            ("entity", args.entity_rounds),
        ):
            model_path = cv_dir / f"{component}_fold{eval_index}.txt"
            model_meta_path = model_path.with_suffix(".json")
            model_signature_payload = {
                "decomposition_signature": decomposition_signature,
                "component": component,
                "eval_index": eval_index,
                "rounds": requested_rounds,
                "early_stopping": args.early_stopping,
                "params": component_params(component, 2026, args.threads),
            }
            model_signature = signature(model_signature_payload)
            can_load = False
            if (
                args.skip_existing_models
                and model_path.exists()
                and model_meta_path.exists()
            ):
                model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
                can_load = model_meta.get("signature") == model_signature
            if can_load:
                model = lgb.Booster(model_file=str(model_path))
                rounds = int(model_meta["best_iteration"])
                expected_count = (
                    len(all_market_names) if component == "market"
                    else len(entity_columns)
                )
                can_load = model.num_feature() == expected_count
            if can_load:
                progress(
                    f"loading {component} CV model: fold={eval_index}, rounds={rounds}"
                )
            else:
                if component == "market":
                    train_set = build_market_dataset(
                        market_feature_path, market_target_path,
                        market_base_path, market_weight_path,
                        train_positions, all_market_names,
                    )
                    valid_set = build_market_dataset(
                        market_feature_path, market_target_path,
                        market_base_path, market_weight_path,
                        valid_positions, all_market_names, reference=train_set,
                    )
                else:
                    train_set = build_entity_dataset(
                        entity_feature_path, entity_label_path, weight_path,
                        train_spans, all_entity_names, entity_columns,
                    )
                    valid_set = build_entity_dataset(
                        entity_feature_path, entity_label_path, weight_path,
                        valid_spans, all_entity_names, entity_columns,
                        reference=train_set,
                    )
                progress(
                    f"{component} CV: fold={eval_index}, "
                    f"train_rows={len(train_set.get_label()):,}"
                )
                model = train_component(
                    component, train_set, valid_set, requested_rounds,
                    args.early_stopping, args.threads,
                )
                rounds = int(model.best_iteration or requested_rounds)
                model.save_model(str(model_path))
                model_meta_path.write_text(json.dumps({
                    "signature": model_signature,
                    "signature_payload": model_signature_payload,
                    "best_iteration": rounds,
                }, indent=2), encoding="utf-8")
                del train_set, valid_set
            if component == "market":
                market_x = np.load(market_feature_path, mmap_mode="r")
                time_prediction = np.asarray(model.predict(
                    np.asarray(market_x[valid_positions], dtype=np.float32),
                    num_iteration=rounds,
                    num_threads=1,
                ), dtype=np.float64)
                component_predictions[component] = expand_market_prediction(
                    time_prediction, valid_positions, row_offsets
                )
                market_time_prediction = time_prediction
            else:
                raw_entity_prediction = predict_entity(
                    model, entity_feature_path, valid_spans,
                    entity_columns, rounds,
                )
                component_predictions[component] = center_entity_by_time(
                    raw_entity_prediction, valid_positions, row_offsets
                )
            component_iterations[component] = rounds
            del model
            gc.collect()
        target_values = read_vector_spans(target_path, valid_spans)
        weight_values = read_vector_spans(weight_path, valid_spans)
        base_values = read_vector_spans(base_prediction_path, valid_spans)
        base_score = weighted_zero_mean_r2(
            target_values, base_values, weight_values
        )
        market_target_source = np.load(market_target_path, mmap_mode="r")
        market_base_source = np.load(market_base_path, mmap_mode="r")
        true_market_residual = np.asarray(
            market_target_source[valid_positions]
            - market_base_source[valid_positions],
            dtype=np.float64,
        )
        true_entity_residual = read_vector_spans(
            entity_label_path, valid_spans
        )
        folds.append({
            "fold_id": int(plan.folds[eval_index].fold_id),
            "target": target_values,
            "weight": weight_values,
            "base": base_values,
            "base_score": base_score,
            "market_prediction": component_predictions["market"],
            "entity_prediction": component_predictions["entity"],
            "diagnostics": {
                "market_target_std": float(np.std(true_market_residual)),
                "market_prediction_std": float(np.std(market_time_prediction)),
                "market_prediction_correlation": safe_correlation(
                    true_market_residual, market_time_prediction
                ),
                "entity_target_std": float(np.std(true_entity_residual)),
                "entity_prediction_std": float(np.std(
                    component_predictions["entity"]
                )),
                "entity_prediction_correlation": safe_correlation(
                    true_entity_residual, component_predictions["entity"]
                ),
            },
        })
        market_iterations.append(component_iterations["market"])
        entity_iterations.append(component_iterations["entity"])
        progress_bar(
            "component CV folds", eval_index, len(fold_spans) - 1,
            f"base_R2={base_score:.8f}",
        )

    selected = score_weight_grid(folds, weights)
    alpha = float(selected["market_weight"])
    beta = float(selected["entity_weight"])
    market_only_weight = float(
        selected["market_only_selection"]["market_weight"]
    )
    entity_only_weight = float(
        selected["entity_only_selection"]["entity_weight"]
    )
    mean_market_rounds = max(1, int(round(np.mean(market_iterations))))
    mean_entity_rounds = max(1, int(round(np.mean(entity_iterations))))
    ablation = component_ablation(
        folds, alpha, beta, market_only_weight, entity_only_weight
    )
    progress(
        f"selected OOF weights: market={alpha:.4f}, entity={beta:.4f}, "
        f"mean_R2={selected['mean_fold_score']:.8f}"
    )

    development_positions = np.concatenate(fold_positions)
    development_spans = [span for group in fold_spans for span in group]
    holdout_models = {}
    holdout_predictions = {}
    for component, rounds in (
        ("market", mean_market_rounds),
        ("entity", mean_entity_rounds),
    ):
        if component == "market":
            train_set = build_market_dataset(
                market_feature_path, market_target_path, market_base_path,
                market_weight_path, development_positions, all_market_names,
            )
        else:
            train_set = build_entity_dataset(
                entity_feature_path, entity_label_path, weight_path,
                development_spans, all_entity_names, entity_columns,
            )
        progress(f"terminal {component} holdout fit: rounds={rounds}")
        model = train_component(
            component, train_set, None, rounds,
            args.early_stopping, args.threads,
        )
        if component == "market":
            market_x = np.load(market_feature_path, mmap_mode="r")
            time_prediction = np.asarray(model.predict(
                np.asarray(market_x[holdout_positions], dtype=np.float32),
                num_iteration=rounds,
                num_threads=1,
            ), dtype=np.float64)
            prediction = expand_market_prediction(
                time_prediction, holdout_positions, row_offsets
            )
        else:
            prediction = center_entity_by_time(
                predict_entity(
                    model, entity_feature_path, holdout_spans,
                    entity_columns, rounds,
                ),
                holdout_positions,
                row_offsets,
            )
        holdout_models[component] = model
        holdout_predictions[component] = prediction
        del train_set

    with (model_dir / "component_feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "feature", "importance_gain", "importance_split"])
        for component, names in (
            ("market", all_market_names),
            ("entity", [all_entity_names[index] for index in entity_columns]),
        ):
            model = holdout_models[component]
            for name, gain, split in zip(
                names,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
            ):
                writer.writerow([component, name, float(gain), int(split)])

    holdout_target = read_vector_spans(target_path, holdout_spans)
    holdout_weight = read_vector_spans(weight_path, holdout_spans)
    holdout_base = read_vector_spans(base_prediction_path, holdout_spans)
    holdout_base_score = weighted_zero_mean_r2(
        holdout_target, holdout_base, holdout_weight
    )

    def holdout_score(market_weight: float, entity_weight: float) -> float:
        return weighted_zero_mean_r2(
            holdout_target,
            holdout_base
            + market_weight * holdout_predictions["market"]
            + entity_weight * holdout_predictions["entity"],
            holdout_weight,
        )

    holdout_scores = {
        "BASE": holdout_base_score,
        "MARKET_ONLY": holdout_score(market_only_weight, 0.0),
        "ENTITY_ONLY": holdout_score(0.0, entity_only_weight),
        "MARKET_ENTITY": holdout_score(alpha, beta),
    }
    selected_holdout_score = holdout_scores["MARKET_ENTITY"]
    fold_deltas = np.asarray(selected["fold_deltas"], dtype=np.float64)
    gates = {
        "component_weight_positive": bool(alpha > 0.0 or beta > 0.0),
        "mean_cv_delta_positive": bool(np.mean(fold_deltas) > 0.0),
        "all_cv_fold_deltas_positive": bool(np.all(fold_deltas > 0.0)),
        "latest_cv_delta_positive": bool(fold_deltas[-1] > 0.0),
        "holdout_delta_positive": bool(selected_holdout_score > holdout_base_score),
    }
    gates["passed"] = bool(all(gates.values()))
    deployment_alpha = alpha if gates["passed"] else 0.0
    deployment_beta = beta if gates["passed"] else 0.0

    market_model_files = []
    entity_model_files = []
    if gates["passed"]:
        final_positions = np.concatenate([development_positions, holdout_positions])
        final_spans = [*development_spans, *holdout_spans]
        for component, rounds, deployment_weight in (
            ("market", mean_market_rounds, deployment_alpha),
            ("entity", mean_entity_rounds, deployment_beta),
        ):
            if deployment_weight <= 0.0:
                continue
            if component == "market":
                final_set = build_market_dataset(
                    market_feature_path, market_target_path, market_base_path,
                    market_weight_path, final_positions, all_market_names,
                )
            else:
                final_set = build_entity_dataset(
                    entity_feature_path, entity_label_path, weight_path,
                    final_spans, all_entity_names, entity_columns,
                )
            for index, seed in enumerate(FINAL_SEEDS, start=1):
                progress(
                    f"final {component} fit {index}/{len(FINAL_SEEDS)}: "
                    f"seed={seed}, rounds={rounds}"
                )
                model = train_component(
                    component, final_set, None, rounds,
                    args.early_stopping, args.threads, seed,
                )
                filename = f"{component}_seed{seed}.txt"
                model.save_model(str(model_dir / filename))
                if component == "market":
                    market_model_files.append(filename)
                else:
                    entity_model_files.append(filename)
                del model
            del final_set

    for model in holdout_models.values():
        del model
    metadata = {
        "strategy": "lgb_v3_market_entity_strategy",
        "base_strategy_dir": os.path.relpath(BASELINE_DIR, STRATEGY_DIR),
        "base_model_dir": os.path.relpath(base_model_dir.resolve(), STRATEGY_DIR),
        "base_report": file_fingerprint(base_report_path),
        "source_strategy_dir": os.path.relpath(SOURCE_STRATEGY_DIR, STRATEGY_DIR),
        "source_artifacts": source_fingerprints,
        "validation_protocol": (
            "V3 purged walk-forward OOF; each component fold uses only earlier OOF blocks"
        ),
        "decomposition": {
            "market_definition": "unweighted target/base mean within current time_id",
            "entity_definition": "target/base deviations from their current-time means",
            "inference_constraint": "entity correction is centered to zero each time_id",
            "prediction_formula": "V3 + market_weight * market_residual + entity_weight * centered_entity_residual",
        },
        "state_features": state_features,
        "market_feature_names": all_market_names,
        "entity_feature_names": all_entity_names,
        "entity_model_feature_indices": entity_columns,
        "entity_model_feature_names": [
            all_entity_names[index] for index in entity_columns
        ],
        "market_models": market_model_files,
        "entity_models": entity_model_files,
        "market_weight": deployment_alpha,
        "entity_weight": deployment_beta,
        "selected_oof_market_weight": alpha,
        "selected_oof_entity_weight": beta,
        "market_rounds": mean_market_rounds,
        "entity_rounds": mean_entity_rounds,
        "selected_weight_report": selected,
        "component_ablation": ablation,
        "cv_folds": [{
            "fold_id": fold["fold_id"],
            "base_score": fold["base_score"],
            "selected_score": selected["fold_scores"][index],
            "selected_delta": selected["fold_deltas"][index],
            "market_best_iteration": market_iterations[index],
            "entity_best_iteration": entity_iterations[index],
            "component_diagnostics": fold["diagnostics"],
        } for index, fold in enumerate(folds)],
        "holdout": {
            "scores": holdout_scores,
            "selected_delta": selected_holdout_score - holdout_base_score,
            "market_only_weight": market_only_weight,
            "entity_only_weight": entity_only_weight,
            "component_diagnostics": {
                "market_prediction_std": float(np.std(
                    holdout_predictions["market"]
                )),
                "entity_prediction_std": float(np.std(
                    holdout_predictions["entity"]
                )),
                "entity_prediction_mean": float(np.mean(
                    holdout_predictions["entity"]
                )),
            },
        },
        "promotion_gates": gates,
        "feature_importance": "component_feature_importance.csv",
        "training_config": {
            **requested_config,
            "threads": args.threads,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    progress(
        f"complete: market={deployment_alpha:.4f}, entity={deployment_beta:.4f}, "
        f"holdout_delta={selected_holdout_score - holdout_base_score:.8f}, "
        f"gates={gates['passed']}"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
