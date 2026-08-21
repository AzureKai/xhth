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
    center_cross_section,
    cross_sectional_z,
    entity_feature_names,
    entity_residual_target,
)


START = time.perf_counter()
FINAL_SEEDS = (2026, 2027, 2028)
SCHEMA_VERSION = 2


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
        description="Strict V3 entity residual with expanded cross-sectional z-scores."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument(
        "--source-work-dir",
        default=str(SOURCE_STRATEGY_DIR / "work"),
        help="Strict V3 OOF/state artifacts from lgb_v3_regime_residual_strategy.",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state-feature-count", type=int, default=16)
    parser.add_argument("--extra-cross-z-count", type=int, default=8)
    parser.add_argument(
        "--entity-weights",
        default="0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0",
    )
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
        raise ValueError("entity weights must be in [0, 1]")
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


class CombinedEntitySequence(lgb.Sequence):
    batch_size = baseline.SEQUENCE_BATCH_ROWS

    def __init__(
        self,
        source_feature_path: Path,
        source_columns: list[int],
        extra_feature_path: Path,
        spans: list[tuple[int, int]],
    ) -> None:
        source_shape = tuple(np.load(source_feature_path, mmap_mode="r").shape)
        extra_shape = tuple(np.load(extra_feature_path, mmap_mode="r").shape)
        self.source = baseline.SpannedMemmapSequence(
            source_feature_path, source_shape, spans
        )
        self.extra = baseline.SpannedMemmapSequence(
            extra_feature_path, extra_shape, spans
        )
        self.source_columns = np.asarray(source_columns, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index):
        source = np.asarray(self.source[index])
        extra = np.asarray(self.extra[index])
        selected = (
            source[self.source_columns]
            if source.ndim == 1
            else source[:, self.source_columns]
        )
        axis = 0 if selected.ndim == 1 else 1
        return np.concatenate([selected, extra], axis=axis)


def build_entity_dataset(
    source_feature_path: Path,
    source_columns: list[int],
    extra_feature_path: Path,
    label_path: Path,
    weight_path: Path,
    spans: list[tuple[int, int]],
    feature_names: list[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    sequence = CombinedEntitySequence(
        source_feature_path, source_columns, extra_feature_path, spans
    )
    labels = read_vector_spans(label_path, spans).astype(np.float32)
    weights = read_vector_spans(weight_path, spans).astype(np.float32)
    dataset = lgb.Dataset(
        sequence,
        label=labels,
        weight=weights,
        reference=reference,
        feature_name=feature_names,
        categorical_feature=["asset_id"],
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


def entity_params(seed: int, threads: int) -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 9,
        "min_data_in_leaf": 5000,
        "feature_fraction": 0.80,
        "feature_fraction_bynode": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 40.0,
        "path_smooth": 150.0,
        "min_gain_to_split": 0.01,
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


def train_entity(
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
        entity_params(seed, threads),
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_entity(
    model: lgb.Booster,
    source_feature_path: Path,
    source_columns: list[int],
    extra_feature_path: Path,
    spans: list[tuple[int, int]],
    rounds: int,
) -> np.ndarray:
    source = np.load(source_feature_path, mmap_mode="r")
    extra = np.load(extra_feature_path, mmap_mode="r")
    parts = []
    for span_start, span_end in spans:
        for start in range(span_start, span_end, baseline.PREDICT_BATCH_ROWS):
            end = min(start + baseline.PREDICT_BATCH_ROWS, span_end)
            values = np.column_stack([
                np.asarray(source[start:end, source_columns], dtype=np.float32),
                np.asarray(extra[start:end], dtype=np.float32),
            ])
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


def select_entity_weight(folds: list[dict], candidates: list[float]) -> dict:
    reports = []
    base_scores = np.asarray([fold["base_score"] for fold in folds])
    for beta in candidates:
        scores = [
            weighted_zero_mean_r2(
                fold["target"],
                fold["base"] + beta * fold["entity_prediction"],
                fold["weight"],
            )
            for fold in folds
        ]
        reports.append({
            "entity_weight": float(beta),
            "fold_scores": list(map(float, scores)),
            "fold_deltas": list(map(float, np.asarray(scores) - base_scores)),
            "mean_fold_score": float(np.mean(scores)),
            "std_fold_score": float(np.std(scores)),
            "min_fold_score": float(np.min(scores)),
            "latest_fold_score": float(scores[-1]),
        })
    reports.sort(key=lambda item: (-item["mean_fold_score"], item["entity_weight"]))
    return {**reports[0], "weight_search": reports}


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
            "lgb_v3_regime_residual_strategy: " + ", ".join(missing)
        )
    return paths


def main() -> None:
    args = parse_args()
    if args.state_feature_count < 1:
        raise ValueError("--state-feature-count must be positive")
    if args.extra_cross_z_count < 1:
        raise ValueError("--extra-cross-z-count must be positive")
    weights = configured_weights(args.entity_weights)
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
    source_fingerprints = {
        name: portable_fingerprint(path) for name, path in source_paths.items()
    }
    requested_config = {
        "state_feature_count": args.state_feature_count,
        "extra_cross_z_count": args.extra_cross_z_count,
        "entity_weights": weights,
        "entity_rounds": args.entity_rounds,
        "early_stopping": args.early_stopping,
    }
    metadata_path = model_dir / "metadata.json"
    if args.skip_existing_models and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_files = [
            model_dir / filename for filename in existing.get("entity_models", [])
        ]
        if (
            int(existing.get("schema_version", 0)) == SCHEMA_VERSION
            and all(existing.get("training_config", {}).get(key) == value
                    for key, value in requested_config.items())
            and existing.get("source_artifacts") == source_fingerprints
            and all(path.exists() for path in model_files)
            and (model_dir / "entity_feature_importance.csv").exists()
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

    history_features = list(cache["history_features"])
    needed = args.state_feature_count + args.extra_cross_z_count
    if len(history_features) < needed:
        raise ValueError(
            f"V3 has only {len(history_features)} ranked history features; need {needed}"
        )
    state_features = history_features[:args.state_feature_count]
    extra_cross_z_features = history_features[
        args.state_feature_count:needed
    ]
    base_entity_names = entity_feature_names(state_features)
    extra_cross_z_names = [
        f"cross_z_{name}" for name in extra_cross_z_features
    ]
    model_feature_names = [*base_entity_names, *extra_cross_z_names]

    source_feature_stage = json.loads(
        source_paths["feature_stage"].read_text(encoding="utf-8")
    )
    source_names = list(source_feature_stage.get("feature_names", []))
    missing_source = [name for name in base_entity_names if name not in source_names]
    if missing_source:
        raise ValueError(
            f"source entity feature schema is missing: {missing_source[:5]}"
        )
    source_columns = [source_names.index(name) for name in base_entity_names]

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
    matrix_path = base_cache_dir / cache["matrix_file"]
    matrix_columns = list(cache["model_cols"])
    extra_raw_columns = [matrix_columns.index(name) for name in extra_cross_z_features]
    target_path = base_cache_dir / cache["target_file"]
    weight_path = base_cache_dir / cache["weight_file"]
    base_prediction_path = source_paths["base_prediction"]
    source_feature_path = source_paths["entity_features"]
    total_rows = int(cache["total_rows"])
    if tuple(np.load(base_prediction_path, mmap_mode="r").shape) != (total_rows,):
        raise ValueError("source base OOF prediction shape mismatch")

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

    extra_feature_path = work_dir / "extra_cross_z_features.npy"
    entity_label_path = work_dir / "entity_residual_target.npy"
    feature_stage_path = work_dir / "entity_feature_stage.json"
    feature_signature_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": source_fingerprints,
        "state_features": state_features,
        "extra_cross_z_features": extra_cross_z_features,
        "model_feature_names": model_feature_names,
        "target": "within-time centered target minus centered V3 prediction",
        "method_version": 1,
    }
    feature_signature = signature(feature_signature_payload)
    reuse_features = False
    if feature_stage_path.exists():
        stage = json.loads(feature_stage_path.read_text(encoding="utf-8"))
        reuse_features = bool(
            args.skip_existing_models
            and stage.get("signature") == feature_signature
            and extra_feature_path.exists()
            and entity_label_path.exists()
        )
    if reuse_features:
        progress("reusing entity target and expanded cross-z memmaps")
    else:
        progress(
            f"materializing entity-only additions: times={len(relevant_positions):,}, "
            f"extra_cross_z={len(extra_cross_z_features)}"
        )
        matrix = np.load(matrix_path, mmap_mode="r")
        base_prediction = np.load(base_prediction_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")
        extra_output = np.lib.format.open_memmap(
            extra_feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, len(extra_cross_z_features)),
        )
        entity_label = np.lib.format.open_memmap(
            entity_label_path, mode="w+", dtype=np.float32,
            shape=(total_rows,),
        )
        extra_output[:] = 0.0
        entity_label[:] = 0.0
        report_every = max(1, len(relevant_positions) // 100)
        for completed, position in enumerate(relevant_positions, start=1):
            start = int(row_offsets[position])
            end = int(row_offsets[position + 1])
            current_base = np.asarray(base_prediction[start:end], dtype=np.float64)
            if not np.all(np.isfinite(current_base)):
                raise ValueError(f"missing strict base prediction at time position {position}")
            entity_label[start:end] = entity_residual_target(
                np.asarray(target[start:end], dtype=np.float64),
                current_base,
            )
            extra_output[start:end] = cross_sectional_z(
                np.asarray(matrix[start:end, extra_raw_columns], dtype=np.float32)
            )
            if (
                completed == 1
                or completed == len(relevant_positions)
                or completed % report_every == 0
            ):
                progress_bar(
                    "entity additions", completed, len(relevant_positions),
                    f"rows={end:,}",
                )
        extra_output.flush()
        entity_label.flush()
        feature_stage_path.write_text(json.dumps({
            "signature": feature_signature,
            **feature_signature_payload,
        }, indent=2), encoding="utf-8")
        del matrix, base_prediction, target, extra_output, entity_label
        gc.collect()

    cv_dir = work_dir / "entity_cv_models_v2"
    cv_dir.mkdir(exist_ok=True)
    folds = []
    best_iterations = []
    for eval_index in range(1, len(fold_spans)):
        train_spans = [span for group in fold_spans[:eval_index] for span in group]
        valid_spans = fold_spans[eval_index]
        valid_positions = fold_positions[eval_index]
        model_path = cv_dir / f"entity_fold{eval_index}.txt"
        model_meta_path = model_path.with_suffix(".json")
        model_signature_payload = {
            "feature_signature": feature_signature,
            "eval_index": eval_index,
            "rounds": args.entity_rounds,
            "early_stopping": args.early_stopping,
            "params": entity_params(2026, args.threads),
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
            can_load = model.num_feature() == len(model_feature_names)
        if can_load:
            progress(f"loading entity CV model: fold={eval_index}, rounds={rounds}")
        else:
            train_set = build_entity_dataset(
                source_feature_path, source_columns, extra_feature_path,
                entity_label_path, weight_path, train_spans,
                model_feature_names,
            )
            valid_set = build_entity_dataset(
                source_feature_path, source_columns, extra_feature_path,
                entity_label_path, weight_path, valid_spans,
                model_feature_names, reference=train_set,
            )
            progress(
                f"entity CV: fold={eval_index}, "
                f"train_rows={len(train_set.get_label()):,}"
            )
            model = train_entity(
                train_set, valid_set, args.entity_rounds,
                args.early_stopping, args.threads,
            )
            rounds = int(model.best_iteration or args.entity_rounds)
            model.save_model(str(model_path))
            model_meta_path.write_text(json.dumps({
                "signature": model_signature,
                "signature_payload": model_signature_payload,
                "best_iteration": rounds,
            }, indent=2), encoding="utf-8")
            del train_set, valid_set
        raw_prediction = predict_entity(
            model, source_feature_path, source_columns,
            extra_feature_path, valid_spans, rounds,
        )
        entity_prediction = center_entity_by_time(
            raw_prediction, valid_positions, row_offsets
        )
        target_values = read_vector_spans(target_path, valid_spans)
        weight_values = read_vector_spans(weight_path, valid_spans)
        base_values = read_vector_spans(base_prediction_path, valid_spans)
        true_entity = read_vector_spans(entity_label_path, valid_spans)
        base_score = weighted_zero_mean_r2(
            target_values, base_values, weight_values
        )
        folds.append({
            "fold_id": int(plan.folds[eval_index].fold_id),
            "target": target_values,
            "weight": weight_values,
            "base": base_values,
            "base_score": base_score,
            "entity_prediction": entity_prediction,
            "diagnostics": {
                "entity_target_std": float(np.std(true_entity)),
                "entity_prediction_std": float(np.std(entity_prediction)),
                "entity_prediction_correlation": safe_correlation(
                    true_entity, entity_prediction
                ),
            },
        })
        best_iterations.append(rounds)
        del model
        gc.collect()
        progress_bar(
            "entity CV folds", eval_index, len(fold_spans) - 1,
            f"rounds={rounds}, base_R2={base_score:.8f}",
        )

    selected = select_entity_weight(folds, weights)
    beta = float(selected["entity_weight"])
    mean_rounds = max(1, int(round(np.mean(best_iterations))))
    progress(
        f"selected OOF entity weight={beta:.4f}, "
        f"mean_R2={selected['mean_fold_score']:.8f}"
    )

    development_spans = [span for group in fold_spans for span in group]
    development_set = build_entity_dataset(
        source_feature_path, source_columns, extra_feature_path,
        entity_label_path, weight_path, development_spans,
        model_feature_names,
    )
    progress(f"terminal entity holdout fit: rounds={mean_rounds}")
    holdout_model = train_entity(
        development_set, None, mean_rounds,
        args.early_stopping, args.threads,
    )
    holdout_prediction = center_entity_by_time(
        predict_entity(
            holdout_model, source_feature_path, source_columns,
            extra_feature_path, holdout_spans, mean_rounds,
        ),
        holdout_positions,
        row_offsets,
    )
    with (model_dir / "entity_feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance_gain", "importance_split"])
        for name, gain, split in zip(
            model_feature_names,
            holdout_model.feature_importance(importance_type="gain"),
            holdout_model.feature_importance(importance_type="split"),
        ):
            writer.writerow([name, float(gain), int(split)])

    holdout_target = read_vector_spans(target_path, holdout_spans)
    holdout_weight = read_vector_spans(weight_path, holdout_spans)
    holdout_base = read_vector_spans(base_prediction_path, holdout_spans)
    holdout_base_score = weighted_zero_mean_r2(
        holdout_target, holdout_base, holdout_weight
    )
    holdout_score = weighted_zero_mean_r2(
        holdout_target,
        holdout_base + beta * holdout_prediction,
        holdout_weight,
    )
    fold_deltas = np.asarray(selected["fold_deltas"], dtype=np.float64)
    gates = {
        "entity_weight_positive": bool(beta > 0.0),
        "mean_cv_delta_positive": bool(np.mean(fold_deltas) > 0.0),
        "all_cv_fold_deltas_positive": bool(np.all(fold_deltas > 0.0)),
        "latest_cv_delta_positive": bool(fold_deltas[-1] > 0.0),
        "holdout_delta_positive": bool(holdout_score > holdout_base_score),
    }
    gates["passed"] = bool(all(gates.values()))
    deployment_beta = beta if gates["passed"] else 0.0

    entity_model_files = []
    if deployment_beta > 0.0:
        final_spans = [*development_spans, *holdout_spans]
        final_set = build_entity_dataset(
            source_feature_path, source_columns, extra_feature_path,
            entity_label_path, weight_path, final_spans,
            model_feature_names,
        )
        for index, seed in enumerate(FINAL_SEEDS, start=1):
            progress(
                f"final entity fit {index}/{len(FINAL_SEEDS)}: "
                f"seed={seed}, rounds={mean_rounds}"
            )
            model = train_entity(
                final_set, None, mean_rounds,
                args.early_stopping, args.threads, seed,
            )
            filename = f"entity_seed{seed}.txt"
            model.save_model(str(model_dir / filename))
            entity_model_files.append(filename)
            del model
        del final_set

    del development_set, holdout_model
    metadata = {
        "strategy": "lgb_v3_entity_residual_strategy",
        "schema_version": SCHEMA_VERSION,
        "base_strategy_dir": os.path.relpath(BASELINE_DIR, STRATEGY_DIR),
        "base_model_dir": os.path.relpath(base_model_dir.resolve(), STRATEGY_DIR),
        "base_report": file_fingerprint(base_report_path),
        "source_strategy_dir": os.path.relpath(SOURCE_STRATEGY_DIR, STRATEGY_DIR),
        "source_artifacts": source_fingerprints,
        "validation_protocol": (
            "V3 purged walk-forward OOF; entity fold uses only earlier OOF blocks"
        ),
        "prediction_formula": (
            "V3 + entity_weight * within-time-centered entity residual"
        ),
        "state_features": state_features,
        "extra_cross_z_features": extra_cross_z_features,
        "entity_model_feature_names": model_feature_names,
        "entity_models": entity_model_files,
        "entity_weight": deployment_beta,
        "selected_oof_entity_weight": beta,
        "entity_rounds": mean_rounds,
        "selected_weight_report": selected,
        "cv_folds": [{
            "fold_id": fold["fold_id"],
            "base_score": fold["base_score"],
            "selected_score": selected["fold_scores"][index],
            "selected_delta": selected["fold_deltas"][index],
            "entity_best_iteration": best_iterations[index],
            "diagnostics": fold["diagnostics"],
        } for index, fold in enumerate(folds)],
        "holdout": {
            "base_score": holdout_base_score,
            "entity_score": holdout_score,
            "delta": holdout_score - holdout_base_score,
            "entity_prediction_std": float(np.std(holdout_prediction)),
            "entity_prediction_mean": float(np.mean(holdout_prediction)),
        },
        "promotion_gates": gates,
        "feature_importance": "entity_feature_importance.csv",
        "training_config": {
            **requested_config,
            "threads": args.threads,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    progress(
        f"complete: entity={deployment_beta:.4f}, "
        f"holdout_delta={holdout_score - holdout_base_score:.8f}, "
        f"gates={gates['passed']}"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
