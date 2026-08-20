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
_THIS_TRAIN_MODULE = (
    sys.modules.pop("train", None) if __name__ == "train" else None
)
sys.path.insert(0, str(BASELINE_DIR))
import train_low_memory as baseline  # noqa: E402
from validation import make_validation_plan, weighted_zero_mean_r2  # noqa: E402
sys.path.pop(0)
if _THIS_TRAIN_MODULE is not None:
    sys.modules["train"] = _THIS_TRAIN_MODULE

from regime_features import (  # noqa: E402
    REGIME_FEATURE_NAMES,
    RegimeEntityFeatureBuilder,
    residual_feature_names,
    state_only_indices,
)


START = time.perf_counter()
RESIDUAL_SEEDS = (2026, 2027, 2028)


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
        f"{100.0 * current / total:6.2f}% ({current:,}/{total:,})"
        f"{suffix}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 OOF residual LightGBM with entity state and regimes."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state-feature-count", type=int, default=16)
    parser.add_argument("--base-oof-seeds", default="2026,2027,2028")
    parser.add_argument(
        "--residual-weights", default="0,0.05,0.10,0.15,0.25"
    )
    parser.add_argument("--residual-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_weights(value: str) -> list[float]:
    weights = list(dict.fromkeys(
        float(item.strip()) for item in value.split(",") if item.strip()
    ))
    if not weights or any(item < 0.0 or item > 0.5 for item in weights):
        raise ValueError("residual weights must be in [0, 0.5]")
    if 0.0 not in weights:
        weights.insert(0, 0.0)
    return weights


def configured_seeds(value: str) -> list[int]:
    seeds = list(dict.fromkeys(
        int(item.strip()) for item in value.split(",") if item.strip()
    ))
    if not seeds:
        raise ValueError("base OOF seeds must not be empty")
    if 2026 in seeds:
        seeds.remove(2026)
    seeds.insert(0, 2026)
    return seeds


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def signature(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def apply_patches(values: np.ndarray, start: int, end: int, patches) -> np.ndarray:
    if not patches:
        return values
    output = values.copy()
    for patch_start, patch_end, patch in patches:
        left, right = max(start, patch_start), min(end, patch_end)
        if left < right:
            output[left - start:right - start] = patch[
                left - patch_start:right - patch_start
            ]
    return output


def predict_spans(
    model: lgb.Booster,
    matrix_path: Path,
    spans: list[tuple[int, int]],
    output: np.ndarray,
    rounds: int,
    patches=None,
    label: str = "prediction",
    add_weight: float | None = None,
) -> None:
    matrix = np.load(matrix_path, mmap_mode="r")
    total = baseline.span_length(spans)
    completed = 0
    for span_start, span_end in spans:
        for start in range(span_start, span_end, baseline.PREDICT_BATCH_ROWS):
            end = min(start + baseline.PREDICT_BATCH_ROWS, span_end)
            values = apply_patches(
                np.asarray(matrix[start:end]), start, end, patches
            )
            prediction = model.predict(
                values, num_iteration=rounds, num_threads=1
            )
            if add_weight is None:
                output[start:end] = prediction
            else:
                output[start:end] += float(add_weight) * prediction
            completed += end - start
            if completed == total or completed % 1_000_000 < end - start:
                progress_bar(label, completed, total)


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


def read_vector_spans(path: Path, spans: list[tuple[int, int]]) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    return np.concatenate([
        np.asarray(source[start:end]) for start, end in spans
    ])


def build_residual_dataset(
    feature_path: Path,
    target_path: Path,
    weight_path: Path,
    base_prediction_path: Path,
    spans: list[tuple[int, int]],
    feature_names: list[str],
    columns: list[int],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    source = baseline.SpannedMemmapSequence(
        feature_path,
        tuple(np.load(feature_path, mmap_mode="r").shape),
        spans,
    )
    sequence = ColumnSequence(source, columns)
    target = read_vector_spans(target_path, spans).astype(np.float32)
    base_prediction = read_vector_spans(
        base_prediction_path, spans
    ).astype(np.float32)
    if not np.all(np.isfinite(base_prediction)):
        raise ValueError("residual training spans contain missing base predictions")
    weight = read_vector_spans(weight_path, spans).astype(np.float32)
    names = [feature_names[index] for index in columns]
    categorical = [
        name for name in ("asset_id", "regime_id") if name in names
    ]
    dataset = lgb.Dataset(
        sequence,
        label=target - base_prediction,
        weight=weight,
        reference=reference,
        feature_name=names,
        categorical_feature=categorical,
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


def residual_params(seed: int, threads: int) -> dict:
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


def train_residual(
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
        residual_params(seed, threads),
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_residual(
    model: lgb.Booster,
    feature_path: Path,
    spans: list[tuple[int, int]],
    columns: list[int],
    rounds: int,
    shuffle_regime: bool = False,
) -> np.ndarray:
    matrix = np.load(feature_path, mmap_mode="r")
    parts = []
    rng = np.random.default_rng(2026)
    regime_local = None
    for span_start, span_end in spans:
        for start in range(span_start, span_end, baseline.PREDICT_BATCH_ROWS):
            end = min(start + baseline.PREDICT_BATCH_ROWS, span_end)
            values = np.asarray(matrix[start:end, columns], dtype=np.float32)
            if shuffle_regime:
                if regime_local is None:
                    regime_local = np.asarray([
                        local for local, global_index in enumerate(columns)
                        if global_index >= matrix.shape[1] - len(REGIME_FEATURE_NAMES) - 1
                    ], dtype=np.int64)
                if len(regime_local):
                    permutation = rng.permutation(len(values))
                    values[:, regime_local] = values[permutation][:, regime_local]
            parts.append(np.asarray(
                model.predict(values, num_iteration=rounds, num_threads=1),
                dtype=np.float32,
            ))
    return np.concatenate(parts)


def fold_arrays(target_path, weight_path, base_path, spans):
    return (
        read_vector_spans(target_path, spans),
        read_vector_spans(weight_path, spans),
        read_vector_spans(base_path, spans),
    )


def select_residual_weight(folds: list[dict], candidates: list[float]) -> dict:
    reports = []
    for beta in candidates:
        scores = [
            weighted_zero_mean_r2(
                fold["target"],
                fold["base"] + beta * fold["residual_prediction"],
                fold["weight"],
            )
            for fold in folds
        ]
        reports.append({
            "residual_weight": float(beta),
            "fold_scores": list(map(float, scores)),
            "fold_deltas": list(map(float, np.asarray(scores) - np.asarray([
                fold["base_score"] for fold in folds
            ]))),
            "mean_fold_score": float(np.mean(scores)),
            "std_fold_score": float(np.std(scores)),
            "min_fold_score": float(np.min(scores)),
            "latest_fold_score": float(scores[-1]),
        })
    reports.sort(key=lambda item: (-item["mean_fold_score"], item["residual_weight"]))
    return {**reports[0], "weight_search": reports}


def main() -> None:
    args = parse_args()
    if args.state_feature_count < 1:
        raise ValueError("--state-feature-count must be positive")
    residual_weights = configured_weights(args.residual_weights)
    base_oof_seeds = configured_seeds(args.base_oof_seeds)
    data_root = Path(args.data_root)
    base_model_dir = Path(args.base_model_dir)
    base_cache_dir = Path(args.base_cache_dir)
    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    base_report_path = base_model_dir / "lightgbm_report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    requested_training_config = {
        "state_feature_count": args.state_feature_count,
        "base_oof_seeds": base_oof_seeds,
        "residual_weights": residual_weights,
        "residual_rounds": args.residual_rounds,
        "early_stopping": args.early_stopping,
    }
    existing_metadata_path = model_dir / "metadata.json"
    if args.skip_existing_models and existing_metadata_path.exists():
        existing = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
        existing_config = existing.get("training_config", {})
        compatible_config = all(
            existing_config.get(key) == value
            for key, value in requested_training_config.items()
        )
        existing_models = [
            model_dir / name for name in existing.get("residual_models", [])
        ]
        required_reports = [model_dir / "residual_feature_importance.csv"]
        if (
            compatible_config
            and existing.get("base_report") == file_fingerprint(base_report_path)
            and all(path.exists() for path in existing_models)
            and all(path.exists() for path in required_reports)
        ):
            progress("all compatible final artifacts exist; training skipped")
            print(existing_metadata_path.read_text(encoding="utf-8"))
            return
    progress("preparing/reusing V3 low-memory feature cache")
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
    matrix_path = base_cache_dir / cache["matrix_file"]
    target_path = base_cache_dir / cache["target_file"]
    weight_path = base_cache_dir / cache["weight_file"]
    matrix_shape = tuple(cache["matrix_shape"])
    model_cols = list(cache["model_cols"])
    candidate = next(
        item for item in baseline.PARAM_CANDIDATES
        if item["name"] == base_report["selected_candidate"]
    )

    base_signature_payload = {
        "base_report": file_fingerprint(base_report_path),
        "cache_input_files": cache["input_files"],
        "candidate": candidate["name"],
        "base_oof_seeds": base_oof_seeds,
        "validation": {
            key: validation[key] for key in (
                "n_splits", "holdout_fraction", "purge_steps",
                "min_train_fraction",
            )
        },
        "method_version": 1,
    }
    base_signature = signature(base_signature_payload)
    base_stage_path = work_dir / "base_oof_stage.json"
    base_prediction_path = work_dir / "base_oof_prediction.npy"
    base_stage = None
    if base_stage_path.exists():
        base_stage = json.loads(base_stage_path.read_text(encoding="utf-8"))
    reuse_base = bool(
        args.skip_existing_models
        and base_stage
        and base_stage.get("signature") == base_signature
        and base_prediction_path.exists()
    )
    fold_spans = [
        baseline.spans_from_time_ids(
            unique_times, row_offsets, fold.valid_time_ids
        ) for fold in plan.folds
    ]
    holdout_spans = baseline.spans_from_time_ids(
        unique_times, row_offsets, plan.holdout_time_ids
    )
    if reuse_base:
        progress("reusing strict V3 OOF and holdout predictions")
        base_records = base_stage["folds"]
        base_iterations = [int(item["best_iteration"]) for item in base_records]
    else:
        base_prediction = np.lib.format.open_memmap(
            base_prediction_path, mode="w+", dtype=np.float32,
            shape=(int(cache["total_rows"]),),
        )
        base_prediction[:] = np.nan
        base_records = []
        base_iterations = []
        base_fold_dir = work_dir / "base_fold_models"
        base_fold_dir.mkdir(exist_ok=True)
        for index, fold in enumerate(plan.folds, start=1):
            train_spans = baseline.spans_from_time_ids(
                unique_times, row_offsets, fold.train_time_ids
            )
            valid_spans = fold_spans[index - 1]
            patches = baseline.build_cold_start_patch(
                matrix_path,
                unique_times=unique_times,
                time_counts=time_counts,
                row_offsets=row_offsets,
                session_time_ids=fold.valid_time_ids,
                raw_features=cache["raw_features"],
                history_features=cache["history_features"],
                rolling_windows=tuple(cache["rolling_windows"]),
                model_cols=model_cols,
            )
            train_set = baseline.build_dataset(
                matrix_path, matrix_shape, target_path, weight_path,
                train_spans, model_cols,
            )
            valid_set = baseline.build_dataset(
                matrix_path, matrix_shape, target_path, weight_path,
                valid_spans, model_cols, reference=train_set, patches=patches,
            )
            progress(f"V3 OOF fold {index}/{len(plan.folds)} training")
            model = baseline.train_early_stopping(
                train_set, valid_set, candidate, args.threads
            )
            rounds = int(model.best_iteration or baseline.NUM_BOOST_ROUND)
            for start, end in valid_spans:
                base_prediction[start:end] = 0.0
            fold_model_files = []
            for seed_index, seed in enumerate(base_oof_seeds, start=1):
                seed_model = (
                    model if seed_index == 1 else baseline.train_fixed(
                        train_set, candidate, seed, rounds, args.threads
                    )
                )
                filename = f"fold_{fold.fold_id}_seed{seed}.txt"
                seed_model.save_model(str(base_fold_dir / filename))
                fold_model_files.append(filename)
                predict_spans(
                    seed_model, matrix_path, valid_spans, base_prediction,
                    rounds, patches,
                    f"V3 OOF fold {index} seed {seed}",
                    add_weight=1.0 / len(base_oof_seeds),
                )
                if seed_model is not model:
                    del seed_model
            base_prediction.flush()
            target, weight, prediction = fold_arrays(
                target_path, weight_path, base_prediction_path, valid_spans
            )
            score = weighted_zero_mean_r2(target, prediction, weight)
            base_records.append({
                "fold_id": int(fold.fold_id),
                "best_iteration": rounds,
                "seeds": base_oof_seeds,
                "model_files": fold_model_files,
                "score": score,
                "train_rows": baseline.span_length(train_spans),
                "valid_rows": baseline.span_length(valid_spans),
                "valid_time_start": int(fold.valid_time_ids[0]),
                "valid_time_end": int(fold.valid_time_ids[-1]),
            })
            base_iterations.append(rounds)
            del model, train_set, valid_set
            gc.collect()
            progress_bar("V3 OOF folds", index, len(plan.folds), f"R2={score:.8f}")

        development_spans = baseline.spans_from_time_ids(
            unique_times, row_offsets, plan.development_time_ids
        )
        development_set = baseline.build_dataset(
            matrix_path, matrix_shape, target_path, weight_path,
            development_spans, model_cols,
        )
        mean_base_rounds = max(1, int(round(np.mean(base_iterations))))
        holdout_patches = baseline.build_cold_start_patch(
            matrix_path,
            unique_times=unique_times,
            time_counts=time_counts,
            row_offsets=row_offsets,
            session_time_ids=plan.holdout_time_ids,
            raw_features=cache["raw_features"],
            history_features=cache["history_features"],
            rolling_windows=tuple(cache["rolling_windows"]),
            model_cols=model_cols,
        )
        for start, end in holdout_spans:
            base_prediction[start:end] = 0.0
        for seed in base_oof_seeds:
            holdout_model = baseline.train_fixed(
                development_set, candidate, seed, mean_base_rounds,
                args.threads,
            )
            holdout_model.save_model(str(
                work_dir / f"base_holdout_seed{seed}.txt"
            ))
            predict_spans(
                holdout_model, matrix_path, holdout_spans, base_prediction,
                mean_base_rounds, holdout_patches,
                f"V3 holdout seed {seed}",
                add_weight=1.0 / len(base_oof_seeds),
            )
            del holdout_model
        base_prediction.flush()
        base_stage = {
            "signature": base_signature,
            "signature_payload": base_signature_payload,
            "folds": base_records,
            "mean_iterations": mean_base_rounds,
        }
        base_stage_path.write_text(
            json.dumps(base_stage, indent=2), encoding="utf-8"
        )
        del development_set, base_prediction
        gc.collect()

    state_features = list(cache["history_features"][:args.state_feature_count])
    all_residual_names = residual_feature_names(state_features)
    residual_feature_path = work_dir / "residual_features.npy"
    feature_stage_path = work_dir / "residual_feature_stage.json"
    feature_signature_payload = {
        "base_signature": base_signature,
        "state_features": state_features,
        "feature_names": all_residual_names,
        "method_version": 2,
    }
    feature_signature = signature(feature_signature_payload)
    reuse_features = False
    if feature_stage_path.exists() and residual_feature_path.exists():
        feature_stage = json.loads(feature_stage_path.read_text(encoding="utf-8"))
        reuse_features = bool(
            args.skip_existing_models
            and feature_stage.get("signature") == feature_signature
        )
    if reuse_features:
        progress("reusing entity/regime residual feature matrix")
    else:
        progress(
            f"materializing entity/regime state: features={len(state_features)}, "
            f"output_columns={len(all_residual_names)}"
        )
        source = np.load(matrix_path, mmap_mode="r")
        base_prediction = np.load(base_prediction_path, mmap_mode="r")
        output = np.lib.format.open_memmap(
            residual_feature_path, mode="w+", dtype=np.float32,
            shape=(int(cache["total_rows"]), len(all_residual_names)),
        )
        raw_positions = [
            model_cols.index(name) for name in state_features
        ]
        output[:] = 0.0
        sessions = [
            *(fold.valid_time_ids for fold in plan.folds),
            plan.holdout_time_ids,
        ]
        total_session_times = sum(len(session) for session in sessions)
        completed_times = 0
        report_every = max(1, total_session_times // 100)
        for session_index, session_times in enumerate(sessions, start=1):
            # Every OOF block and the terminal holdout simulate a fresh API
            # session, exactly like the frozen V3 validation protocol.
            builder = RegimeEntityFeatureBuilder(state_features)
            positions = np.searchsorted(unique_times, session_times)
            for time_index in positions:
                start = int(row_offsets[time_index])
                end = int(row_offsets[time_index + 1])
                output[start:end] = builder.transform_time(
                    np.asarray(source[start:end, 0], dtype=np.int64),
                    np.asarray(
                        source[start:end, raw_positions], dtype=np.float32
                    ),
                    np.asarray(base_prediction[start:end], dtype=np.float32),
                )
                completed_times += 1
                if (
                    completed_times == 1
                    or completed_times == total_session_times
                    or completed_times % report_every == 0
                ):
                    progress_bar(
                        "state materialization", completed_times,
                        total_session_times,
                        f"session={session_index}/{len(sessions)}, rows={end:,}",
                    )
        output.flush()
        feature_stage_path.write_text(json.dumps({
            "signature": feature_signature,
            **feature_signature_payload,
        }, indent=2), encoding="utf-8")
        del source, base_prediction, output
        gc.collect()

    variant_columns = {
        "ENTITY_STATE": state_only_indices(all_residual_names),
        "REGIME_ENTITY": list(range(len(all_residual_names))),
    }
    residual_cv_dir = work_dir / "residual_cv_models"
    residual_cv_dir.mkdir(exist_ok=True)
    variant_reports = {}
    for variant_index, (variant, columns) in enumerate(variant_columns.items(), start=1):
        folds = []
        best_iterations = []
        for eval_index in range(1, len(fold_spans)):
            train_spans = [span for group in fold_spans[:eval_index] for span in group]
            valid_spans = fold_spans[eval_index]
            model_path = residual_cv_dir / f"{variant}_fold{eval_index}.txt"
            fold_metadata_path = model_path.with_suffix(".json")
            fold_signature_payload = {
                "feature_signature": feature_signature,
                "variant": variant,
                "columns": columns,
                "eval_index": eval_index,
                "residual_rounds": args.residual_rounds,
                "early_stopping": args.early_stopping,
                "params": residual_params(2026, args.threads),
            }
            fold_signature = signature(fold_signature_payload)
            can_load = False
            if (
                args.skip_existing_models
                and model_path.exists()
                and fold_metadata_path.exists()
            ):
                fold_metadata = json.loads(
                    fold_metadata_path.read_text(encoding="utf-8")
                )
                can_load = fold_metadata.get("signature") == fold_signature
            if can_load:
                model = lgb.Booster(model_file=str(model_path))
                rounds = int(fold_metadata["best_iteration"])
                if model.num_feature() != len(columns):
                    can_load = False
            if can_load:
                progress(
                    f"loading residual CV model: variant={variant}, "
                    f"fold={eval_index}, rounds={rounds}"
                )
            else:
                train_set = build_residual_dataset(
                    residual_feature_path, target_path, weight_path,
                    base_prediction_path, train_spans, all_residual_names,
                    columns,
                )
                valid_set = build_residual_dataset(
                    residual_feature_path, target_path, weight_path,
                    base_prediction_path, valid_spans, all_residual_names,
                    columns, reference=train_set,
                )
                progress(
                    f"residual CV: variant={variant}, fold={eval_index}, "
                    f"train_rows={len(train_set.get_label()):,}"
                )
                model = train_residual(
                    train_set, valid_set, args.residual_rounds,
                    args.early_stopping, args.threads,
                )
                rounds = int(model.best_iteration or args.residual_rounds)
                model.save_model(str(model_path))
                fold_metadata_path.write_text(json.dumps({
                    "signature": fold_signature,
                    "signature_payload": fold_signature_payload,
                    "best_iteration": rounds,
                }, indent=2), encoding="utf-8")
                del train_set, valid_set
            residual_prediction = predict_residual(
                model, residual_feature_path, valid_spans, columns, rounds
            )
            target, weight, base_prediction = fold_arrays(
                target_path, weight_path, base_prediction_path, valid_spans
            )
            base_score = weighted_zero_mean_r2(
                target, base_prediction, weight
            )
            folds.append({
                "fold_id": int(plan.folds[eval_index].fold_id),
                "target": target,
                "weight": weight,
                "base": base_prediction,
                "base_score": base_score,
                "residual_prediction": residual_prediction,
            })
            best_iterations.append(rounds)
            del model
            gc.collect()
            progress_bar(
                f"residual {variant}", eval_index,
                len(fold_spans) - 1, f"rounds={rounds}",
            )
        selected_weight = select_residual_weight(folds, residual_weights)
        variant_reports[variant] = {
            **selected_weight,
            "mean_iterations": max(1, int(round(np.mean(best_iterations)))),
            "feature_count": len(columns),
            "folds": [{
                "fold_id": fold["fold_id"],
                "base_score": fold["base_score"],
                "score": selected_weight["fold_scores"][index],
                "delta": selected_weight["fold_deltas"][index],
                "best_iteration": best_iterations[index],
            } for index, fold in enumerate(folds)],
        }
        progress_bar(
            "residual variants", variant_index, len(variant_columns),
            f"{variant}, mean={selected_weight['mean_fold_score']:.8f}",
        )

    selected_variant = max(
        variant_reports,
        key=lambda name: variant_reports[name]["mean_fold_score"],
    )
    selected_report = variant_reports[selected_variant]
    selected_columns = variant_columns[selected_variant]
    oof_train_spans = [span for group in fold_spans for span in group]
    development_residual_set = build_residual_dataset(
        residual_feature_path, target_path, weight_path,
        base_prediction_path, oof_train_spans, all_residual_names,
        selected_columns,
    )
    progress(f"terminal residual holdout: variant={selected_variant}")
    holdout_residual_model = train_residual(
        development_residual_set, None,
        selected_report["mean_iterations"], args.early_stopping,
        args.threads,
    )
    holdout_rounds = int(selected_report["mean_iterations"])
    holdout_residual_prediction = predict_residual(
        holdout_residual_model, residual_feature_path, holdout_spans,
        selected_columns, holdout_rounds,
    )
    holdout_target, holdout_weight, holdout_base = fold_arrays(
        target_path, weight_path, base_prediction_path, holdout_spans
    )
    beta = float(selected_report["residual_weight"])
    selected_feature_names = [
        all_residual_names[index] for index in selected_columns
    ]
    with (model_dir / "residual_feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "feature", "importance_gain", "importance_split"])
        for name, gain, split in zip(
            selected_feature_names,
            holdout_residual_model.feature_importance(importance_type="gain"),
            holdout_residual_model.feature_importance(importance_type="split"),
        ):
            writer.writerow([selected_variant, name, float(gain), int(split)])
    holdout_base_score = weighted_zero_mean_r2(
        holdout_target, holdout_base, holdout_weight
    )
    holdout_score = weighted_zero_mean_r2(
        holdout_target,
        holdout_base + beta * holdout_residual_prediction,
        holdout_weight,
    )
    fold_deltas = np.asarray(selected_report["fold_deltas"])
    gates = {
        "residual_weight_positive": bool(beta > 0.0),
        "mean_cv_delta_positive": bool(np.mean(fold_deltas) > 0.0),
        "all_cv_fold_deltas_positive": bool(np.all(fold_deltas > 0.0)),
        "latest_cv_delta_positive": bool(fold_deltas[-1] > 0.0),
        "holdout_delta_positive": bool(holdout_score > holdout_base_score),
    }

    shuffled_holdout_score = None
    if selected_variant == "REGIME_ENTITY":
        shuffled_prediction = predict_residual(
            holdout_residual_model, residual_feature_path, holdout_spans,
            selected_columns, holdout_rounds, shuffle_regime=True,
        )
        shuffled_holdout_score = weighted_zero_mean_r2(
            holdout_target,
            holdout_base + beta * shuffled_prediction,
            holdout_weight,
        )
        gates["regime_better_than_shuffled"] = bool(
            holdout_score > shuffled_holdout_score
        )
    gates["passed"] = bool(all(gates.values()))
    deployment_beta = beta if gates["passed"] else 0.0

    regime_counts = {str(index): 0 for index in range(4)}
    regime_column = all_residual_names.index("regime_id")
    residual_matrix = np.load(residual_feature_path, mmap_mode="r")
    for start, end in [*oof_train_spans, *holdout_spans]:
        values, counts = np.unique(
            np.asarray(residual_matrix[start:end, regime_column], dtype=np.int64),
            return_counts=True,
        )
        for value, count in zip(values, counts):
            regime_counts[str(int(value))] = (
                regime_counts.get(str(int(value)), 0) + int(count)
            )

    residual_model_files = []
    final_rounds = max(
        1, int(round(np.mean([
            selected_report["mean_iterations"], holdout_rounds
        ])))
    )
    if deployment_beta > 0.0:
        final_spans = [*oof_train_spans, *holdout_spans]
        final_set = build_residual_dataset(
            residual_feature_path, target_path, weight_path,
            base_prediction_path, final_spans, all_residual_names,
            selected_columns,
        )
        for index, seed in enumerate(RESIDUAL_SEEDS, start=1):
            progress(
                f"final residual fit {index}/{len(RESIDUAL_SEEDS)}: "
                f"seed={seed}, rounds={final_rounds}"
            )
            model = train_residual(
                final_set, None, final_rounds, args.early_stopping,
                args.threads, seed,
            )
            filename = f"residual_seed{seed}.txt"
            model.save_model(str(model_dir / filename))
            residual_model_files.append(filename)
            del model
        del final_set

    base_strategy_dir = BASELINE_DIR
    metadata = {
        "strategy": "lgb_v3_regime_residual_strategy",
        "base_strategy_dir": os.path.relpath(base_strategy_dir, STRATEGY_DIR),
        "base_model_dir": os.path.relpath(base_model_dir.resolve(), STRATEGY_DIR),
        "base_report": file_fingerprint(base_report_path),
        "validation_protocol": (
            "V3 purged walk-forward OOF; residual fold uses only earlier OOF blocks"
        ),
        "state_features": state_features,
        "residual_feature_names": all_residual_names,
        "residual_model_feature_indices": selected_columns,
        "residual_model_feature_names": [
            all_residual_names[index] for index in selected_columns
        ],
        "residual_feature_importance": "residual_feature_importance.csv",
        "regime_row_counts": regime_counts,
        "selected_variant": selected_variant,
        "residual_models": residual_model_files,
        "residual_seeds": list(RESIDUAL_SEEDS),
        "residual_weight": deployment_beta,
        "selected_oof_weight": beta,
        "final_rounds": final_rounds,
        "base_oof_folds": base_records,
        "variant_reports": variant_reports,
        "holdout": {
            "base_score": holdout_base_score,
            "residual_score": holdout_score,
            "delta": holdout_score - holdout_base_score,
            "shuffled_regime_score": shuffled_holdout_score,
            "shuffled_regime_delta_vs_model": (
                None if shuffled_holdout_score is None
                else shuffled_holdout_score - holdout_score
            ),
        },
        "promotion_gates": gates,
        "training_config": {
            **requested_training_config,
            "threads": args.threads,
        },
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    progress(
        f"complete: variant={selected_variant}, beta={deployment_beta:.4f}, "
        f"holdout_delta={holdout_score - holdout_base_score:.8f}, "
        f"gates={gates['passed']}"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
