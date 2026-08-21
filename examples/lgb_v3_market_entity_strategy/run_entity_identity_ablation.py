from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import train as strategy_train


START = time.perf_counter()
ABLATION_SCHEMA_VERSION = 1
DEFAULT_VARIANTS = ("full", "no_asset_id", "frozen_prior")
FROZEN_PRIOR_NAME = "frozen_entity_residual_prior"


def progress(message: str) -> None:
    elapsed = time.perf_counter() - START
    print(f"[ablation {elapsed:9.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time-safe entity identity ablation for the V3 residual strategy."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument(
        "--source-work-dir",
        default=str(strategy_train.SOURCE_STRATEGY_DIR / "work"),
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated subset of full,no_asset_id,frozen_prior.",
    )
    parser.add_argument("--prior-shrinkage", type=float, default=100.0)
    parser.add_argument("--entity-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_variants(value: str) -> list[str]:
    requested = []
    for item in value.split(","):
        name = item.strip()
        if name and name not in requested:
            requested.append(name)
    unknown = sorted(set(requested) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    if not requested:
        raise ValueError("at least one ablation variant is required")
    return requested


class AblationEntitySequence(lgb.Sequence):
    batch_size = strategy_train.baseline.SEQUENCE_BATCH_ROWS

    def __init__(
        self,
        source_feature_path: Path,
        source_columns: list[int],
        extra_feature_path: Path,
        spans: list[tuple[int, int]],
        appended_feature_path: Path | None = None,
    ) -> None:
        source_shape = tuple(np.load(source_feature_path, mmap_mode="r").shape)
        extra_shape = tuple(np.load(extra_feature_path, mmap_mode="r").shape)
        self.source = strategy_train.baseline.SpannedMemmapSequence(
            source_feature_path, source_shape, spans
        )
        self.extra = strategy_train.baseline.SpannedMemmapSequence(
            extra_feature_path, extra_shape, spans
        )
        self.source_columns = np.asarray(source_columns, dtype=np.int64)
        self.appended = (
            np.load(appended_feature_path, mmap_mode="r")
            if appended_feature_path is not None
            else None
        )
        if self.appended is not None and self.appended.shape != (len(self.source),):
            raise ValueError(
                "appended ablation feature length does not match selected spans"
            )

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
        parts = [selected, extra]
        if self.appended is not None:
            prior = np.asarray(self.appended[index])
            if selected.ndim == 1:
                prior = prior.reshape(1)
            else:
                prior = prior.reshape(-1, 1)
            parts.append(prior)
        return np.concatenate(parts, axis=axis)


def build_dataset(
    source_feature_path: Path,
    source_columns: list[int],
    extra_feature_path: Path,
    label_path: Path,
    weight_path: Path,
    spans: list[tuple[int, int]],
    feature_names: list[str],
    appended_feature_path: Path | None = None,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    sequence = AblationEntitySequence(
        source_feature_path,
        source_columns,
        extra_feature_path,
        spans,
        appended_feature_path,
    )
    labels = strategy_train.read_vector_spans(label_path, spans).astype(np.float32)
    weights = strategy_train.read_vector_spans(weight_path, spans).astype(np.float32)
    categorical = ["asset_id"] if "asset_id" in feature_names else []
    dataset = lgb.Dataset(
        sequence,
        label=labels,
        weight=weights,
        reference=reference,
        feature_name=feature_names,
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


def predict(
    model: lgb.Booster,
    source_feature_path: Path,
    source_columns: list[int],
    extra_feature_path: Path,
    spans: list[tuple[int, int]],
    rounds: int,
    appended_feature_path: Path | None = None,
) -> np.ndarray:
    source = np.load(source_feature_path, mmap_mode="r")
    extra = np.load(extra_feature_path, mmap_mode="r")
    appended = (
        np.load(appended_feature_path, mmap_mode="r")
        if appended_feature_path is not None
        else None
    )
    parts = []
    local_cursor = 0
    for span_start, span_end in spans:
        for start in range(span_start, span_end, strategy_train.baseline.PREDICT_BATCH_ROWS):
            end = min(start + strategy_train.baseline.PREDICT_BATCH_ROWS, span_end)
            values = [
                np.asarray(source[start:end, source_columns], dtype=np.float32),
                np.asarray(extra[start:end], dtype=np.float32),
            ]
            if appended is not None:
                count = end - start
                values.append(
                    np.asarray(
                        appended[local_cursor:local_cursor + count],
                        dtype=np.float32,
                    ).reshape(-1, 1)
                )
                local_cursor += count
            matrix = np.column_stack(values)
            parts.append(np.asarray(
                model.predict(matrix, num_iteration=rounds, num_threads=1),
                dtype=np.float64,
            ))
    if appended is not None and local_cursor != len(appended):
        raise ValueError("appended prediction feature was not fully consumed")
    return np.concatenate(parts)


def _write_prior_rows(
    output: np.memmap,
    cursor: int,
    assets: np.ndarray,
    sums: dict[int, float],
    counts: dict[int, int],
    shrinkage: float,
) -> int:
    unique, inverse = np.unique(assets, return_inverse=True)
    mapped = np.asarray([
        sums.get(int(asset), 0.0) / (counts.get(int(asset), 0) + shrinkage)
        for asset in unique
    ], dtype=np.float32)
    end = cursor + len(assets)
    output[cursor:end] = mapped[inverse]
    return end


def _update_prior_state(
    assets: np.ndarray,
    labels: np.ndarray,
    sums: dict[int, float],
    counts: dict[int, int],
) -> None:
    unique, inverse = np.unique(assets, return_inverse=True)
    label_sums = np.bincount(inverse, weights=labels, minlength=len(unique))
    label_counts = np.bincount(inverse, minlength=len(unique))
    for index, asset in enumerate(unique):
        key = int(asset)
        sums[key] = sums.get(key, 0.0) + float(label_sums[index])
        counts[key] = counts.get(key, 0) + int(label_counts[index])


def materialize_frozen_prior(
    source_feature_path: Path,
    asset_column: int,
    label_path: Path,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    row_offsets: np.ndarray,
    shrinkage: float,
    train_path: Path,
    valid_path: Path,
) -> None:
    source = np.load(source_feature_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    train_rows = int(np.sum(
        row_offsets[train_positions + 1] - row_offsets[train_positions]
    ))
    valid_rows = int(np.sum(
        row_offsets[valid_positions + 1] - row_offsets[valid_positions]
    ))
    train_output = np.lib.format.open_memmap(
        train_path, mode="w+", dtype=np.float32, shape=(train_rows,)
    )
    valid_output = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=np.float32, shape=(valid_rows,)
    )
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    cursor = 0
    report_every = max(1, len(train_positions) // 20)
    for completed, position in enumerate(train_positions, start=1):
        start = int(row_offsets[position])
        end = int(row_offsets[position + 1])
        assets = np.asarray(source[start:end, asset_column], dtype=np.int64)
        cursor = _write_prior_rows(
            train_output, cursor, assets, sums, counts, shrinkage
        )
        _update_prior_state(
            assets,
            np.asarray(labels[start:end], dtype=np.float64),
            sums,
            counts,
        )
        if completed == len(train_positions) or completed % report_every == 0:
            strategy_train.progress_bar(
                "frozen prior train", completed, len(train_positions),
                f"entities={len(counts)}",
            )
    if cursor != train_rows:
        raise ValueError("causal prior train row count mismatch")
    cursor = 0
    for position in valid_positions:
        start = int(row_offsets[position])
        end = int(row_offsets[position + 1])
        assets = np.asarray(source[start:end, asset_column], dtype=np.int64)
        cursor = _write_prior_rows(
            valid_output, cursor, assets, sums, counts, shrinkage
        )
    if cursor != valid_rows:
        raise ValueError("frozen prior validation row count mismatch")
    train_output.flush()
    valid_output.flush()
    del source, labels, train_output, valid_output
    gc.collect()


def ensure_frozen_prior(
    variant_dir: Path,
    split_name: str,
    signature_payload: dict,
    source_feature_path: Path,
    asset_column: int,
    label_path: Path,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    row_offsets: np.ndarray,
    shrinkage: float,
    skip_existing: bool,
) -> tuple[Path, Path]:
    train_path = variant_dir / f"{split_name}_train_prior.npy"
    valid_path = variant_dir / f"{split_name}_valid_prior.npy"
    stage_path = variant_dir / f"{split_name}_prior.json"
    prior_signature = strategy_train.signature(signature_payload)
    if skip_existing and train_path.exists() and valid_path.exists() and stage_path.exists():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("signature") == prior_signature:
            progress(f"reusing frozen prior: {split_name}")
            return train_path, valid_path
    progress(f"materializing time-safe frozen prior: {split_name}")
    materialize_frozen_prior(
        source_feature_path,
        asset_column,
        label_path,
        train_positions,
        valid_positions,
        row_offsets,
        shrinkage,
        train_path,
        valid_path,
    )
    stage_path.write_text(json.dumps({
        "signature": prior_signature,
        "signature_payload": signature_payload,
    }, indent=2), encoding="utf-8")
    return train_path, valid_path


def source_columns_for_variant(
    variant: str,
    source_names: list[str],
    production_names: list[str],
) -> tuple[list[int], list[str]]:
    base_names = [
        name for name in production_names if not name.startswith("cross_z_")
        or name in source_names
    ]
    # Extra cross-z columns are stored in a separate memmap and follow source names.
    if variant in {"no_asset_id", "frozen_prior"}:
        base_names = [name for name in base_names if name != "asset_id"]
    columns = [source_names.index(name) for name in base_names]
    return columns, base_names


def compact_fold_report(
    fold: dict,
    selected: dict,
    index: int,
    best_iteration: int,
) -> dict:
    return {
        "fold_id": fold["fold_id"],
        "base_score": fold["base_score"],
        "selected_score": selected["fold_scores"][index],
        "selected_delta": selected["fold_deltas"][index],
        "entity_best_iteration": int(best_iteration),
        "diagnostics": fold["diagnostics"],
    }


def existing_full_report(metadata: dict) -> dict:
    selected = metadata["selected_weight_report"]
    return {
        "variant": "full",
        "description": "categorical asset_id plus causal feature state",
        "reused_production_result": True,
        "selected_entity_weight": float(metadata["selected_oof_entity_weight"]),
        "mean_cv_score": float(selected["mean_fold_score"]),
        "mean_cv_delta": float(np.mean(selected["fold_deltas"])),
        "min_cv_delta": float(np.min(selected["fold_deltas"])),
        "latest_cv_delta": float(selected["fold_deltas"][-1]),
        "all_cv_deltas_positive": bool(np.all(
            np.asarray(selected["fold_deltas"]) > 0.0
        )),
        "cv_folds": metadata["cv_folds"],
        "holdout": metadata["holdout"],
    }


def main() -> None:
    args = parse_args()
    variants = configured_variants(args.variants)
    if args.prior_shrinkage <= 0.0:
        raise ValueError("--prior-shrinkage must be positive")
    data_root = Path(args.data_root)
    base_model_dir = Path(args.base_model_dir)
    base_cache_dir = Path(args.base_cache_dir)
    source_work_dir = Path(args.source_work_dir)
    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    output_dir = work_dir / "entity_identity_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != strategy_train.SCHEMA_VERSION:
        raise ValueError("production entity model must use schema_version=2")
    production_config = metadata.get("training_config", {})
    expected_config = {
        "entity_rounds": args.entity_rounds,
        "early_stopping": args.early_stopping,
    }
    mismatched = {
        key: (production_config.get(key), value)
        for key, value in expected_config.items()
        if production_config.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "ablation must use the production training budget; "
            f"metadata/request mismatches: {mismatched}"
        )
    source_paths = strategy_train.require_source_artifacts(source_work_dir)
    feature_stage_path = work_dir / "entity_feature_stage.json"
    feature_stage = json.loads(feature_stage_path.read_text(encoding="utf-8"))
    source_stage = json.loads(
        source_paths["feature_stage"].read_text(encoding="utf-8")
    )
    source_names = list(source_stage["feature_names"])
    production_names = list(metadata["entity_model_feature_names"])
    extra_names = [
        f"cross_z_{name}" for name in metadata["extra_cross_z_features"]
    ]
    extra_feature_path = work_dir / "extra_cross_z_features.npy"
    entity_label_path = work_dir / "entity_residual_target.npy"
    if not extra_feature_path.exists() or not entity_label_path.exists():
        raise FileNotFoundError(
            "missing production feature cache; run train.py first"
        )

    base_report = json.loads(
        (base_model_dir / "lightgbm_report.json").read_text(encoding="utf-8")
    )
    cache = strategy_train.baseline.prepare_cache(data_root, base_cache_dir)
    axis = np.load(base_cache_dir / cache["time_axis_file"])
    unique_times = axis["unique_times"]
    row_offsets = strategy_train.baseline.row_offsets_from_counts(axis["time_counts"])
    validation = base_report["validation"]
    plan = strategy_train.make_validation_plan(
        unique_times,
        n_splits=int(validation["n_splits"]),
        holdout_fraction=float(validation["holdout_fraction"]),
        purge_steps=int(validation["purge_steps"]),
        min_train_fraction=float(validation["min_train_fraction"]),
    )
    fold_positions = [
        strategy_train.time_positions(unique_times, fold.valid_time_ids)
        for fold in plan.folds
    ]
    holdout_positions = strategy_train.time_positions(
        unique_times, plan.holdout_time_ids
    )
    fold_spans = [
        strategy_train.baseline.spans_from_time_ids(
            unique_times, row_offsets, fold.valid_time_ids
        )
        for fold in plan.folds
    ]
    holdout_spans = strategy_train.baseline.spans_from_time_ids(
        unique_times, row_offsets, plan.holdout_time_ids
    )
    source_feature_path = source_paths["entity_features"]
    weight_path = base_cache_dir / cache["weight_file"]
    target_path = base_cache_dir / cache["target_file"]
    base_prediction_path = source_paths["base_prediction"]
    asset_column = source_names.index("asset_id")
    weights = [
        float(item["entity_weight"])
        for item in metadata["selected_weight_report"]["weight_search"]
    ]

    reports = []
    if "full" in variants:
        reports.append(existing_full_report(metadata))
        progress("reused full production CV/holdout result")

    for variant in [item for item in variants if item != "full"]:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        source_columns, base_feature_names = source_columns_for_variant(
            variant, source_names, production_names
        )
        feature_names = [*base_feature_names, *extra_names]
        if variant == "frozen_prior":
            feature_names.append(FROZEN_PRIOR_NAME)
        folds = []
        best_iterations = []
        for eval_index in range(1, len(fold_spans)):
            train_spans = [
                span for group in fold_spans[:eval_index] for span in group
            ]
            train_positions = np.concatenate(fold_positions[:eval_index])
            valid_spans = fold_spans[eval_index]
            valid_positions = fold_positions[eval_index]
            train_prior_path = None
            valid_prior_path = None
            prior_signature_payload = None
            if variant == "frozen_prior":
                prior_signature_payload = {
                    "schema_version": ABLATION_SCHEMA_VERSION,
                    "production_feature_signature": feature_stage["signature"],
                    "split": f"fold{eval_index}",
                    "train_positions": [
                        int(train_positions[0]), int(train_positions[-1]),
                        int(len(train_positions)),
                    ],
                    "valid_positions": [
                        int(valid_positions[0]), int(valid_positions[-1]),
                        int(len(valid_positions)),
                    ],
                    "prior_shrinkage": args.prior_shrinkage,
                    "method": "causal_train_then_frozen_validation_v1",
                }
                train_prior_path, valid_prior_path = ensure_frozen_prior(
                    variant_dir,
                    f"fold{eval_index}",
                    prior_signature_payload,
                    source_feature_path,
                    asset_column,
                    entity_label_path,
                    train_positions,
                    valid_positions,
                    row_offsets,
                    args.prior_shrinkage,
                    args.skip_existing_models,
                )
            model_path = variant_dir / f"fold{eval_index}.txt"
            model_meta_path = model_path.with_suffix(".json")
            model_signature_payload = {
                "schema_version": ABLATION_SCHEMA_VERSION,
                "production_feature_signature": feature_stage["signature"],
                "variant": variant,
                "feature_names": feature_names,
                "eval_index": eval_index,
                "rounds": args.entity_rounds,
                "early_stopping": args.early_stopping,
                "prior": prior_signature_payload,
                "params": strategy_train.entity_params(2026, args.threads),
            }
            model_signature = strategy_train.signature(model_signature_payload)
            can_load = False
            if args.skip_existing_models and model_path.exists() and model_meta_path.exists():
                model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
                can_load = model_meta.get("signature") == model_signature
            if can_load:
                model = lgb.Booster(model_file=str(model_path))
                rounds = int(model_meta["best_iteration"])
                can_load = list(model.feature_name()) == feature_names
            if can_load:
                progress(f"reusing {variant} fold={eval_index}, rounds={rounds}")
            else:
                train_set = build_dataset(
                    source_feature_path,
                    source_columns,
                    extra_feature_path,
                    entity_label_path,
                    weight_path,
                    train_spans,
                    feature_names,
                    train_prior_path,
                )
                valid_set = build_dataset(
                    source_feature_path,
                    source_columns,
                    extra_feature_path,
                    entity_label_path,
                    weight_path,
                    valid_spans,
                    feature_names,
                    valid_prior_path,
                    train_set,
                )
                progress(
                    f"training {variant} fold={eval_index}, "
                    f"train_rows={len(train_set.get_label()):,}"
                )
                model = strategy_train.train_entity(
                    train_set,
                    valid_set,
                    args.entity_rounds,
                    args.early_stopping,
                    args.threads,
                )
                rounds = int(model.best_iteration or args.entity_rounds)
                model.save_model(str(model_path))
                model_meta_path.write_text(json.dumps({
                    "signature": model_signature,
                    "signature_payload": model_signature_payload,
                    "best_iteration": rounds,
                }, indent=2), encoding="utf-8")
                del train_set, valid_set
            raw_prediction = predict(
                model,
                source_feature_path,
                source_columns,
                extra_feature_path,
                valid_spans,
                rounds,
                valid_prior_path,
            )
            entity_prediction = strategy_train.center_entity_by_time(
                raw_prediction, valid_positions, row_offsets
            )
            target_values = strategy_train.read_vector_spans(target_path, valid_spans)
            weight_values = strategy_train.read_vector_spans(weight_path, valid_spans)
            base_values = strategy_train.read_vector_spans(
                base_prediction_path, valid_spans
            )
            true_entity = strategy_train.read_vector_spans(
                entity_label_path, valid_spans
            )
            base_score = strategy_train.weighted_zero_mean_r2(
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
                    "entity_prediction_correlation": strategy_train.safe_correlation(
                        true_entity, entity_prediction
                    ),
                },
            })
            best_iterations.append(rounds)
            del model
            gc.collect()
            strategy_train.progress_bar(
                f"{variant} CV",
                eval_index,
                len(fold_spans) - 1,
                f"rounds={rounds}",
            )

        selected = strategy_train.select_entity_weight(folds, weights)
        mean_rounds = max(1, int(round(np.mean(best_iterations))))
        development_spans = [span for group in fold_spans for span in group]
        development_positions = np.concatenate(fold_positions)
        train_prior_path = None
        holdout_prior_path = None
        holdout_prior_signature = None
        if variant == "frozen_prior":
            holdout_prior_signature = {
                "schema_version": ABLATION_SCHEMA_VERSION,
                "production_feature_signature": feature_stage["signature"],
                "split": "holdout",
                "train_positions": [
                    int(development_positions[0]),
                    int(development_positions[-1]),
                    int(len(development_positions)),
                ],
                "valid_positions": [
                    int(holdout_positions[0]), int(holdout_positions[-1]),
                    int(len(holdout_positions)),
                ],
                "prior_shrinkage": args.prior_shrinkage,
                "method": "causal_train_then_frozen_validation_v1",
            }
            train_prior_path, holdout_prior_path = ensure_frozen_prior(
                variant_dir,
                "holdout",
                holdout_prior_signature,
                source_feature_path,
                asset_column,
                entity_label_path,
                development_positions,
                holdout_positions,
                row_offsets,
                args.prior_shrinkage,
                args.skip_existing_models,
            )
        development_set = build_dataset(
            source_feature_path,
            source_columns,
            extra_feature_path,
            entity_label_path,
            weight_path,
            development_spans,
            feature_names,
            train_prior_path,
        )
        progress(f"training {variant} terminal holdout, rounds={mean_rounds}")
        holdout_model = strategy_train.train_entity(
            development_set,
            None,
            mean_rounds,
            args.early_stopping,
            args.threads,
        )
        raw_holdout = predict(
            holdout_model,
            source_feature_path,
            source_columns,
            extra_feature_path,
            holdout_spans,
            mean_rounds,
            holdout_prior_path,
        )
        holdout_prediction = strategy_train.center_entity_by_time(
            raw_holdout, holdout_positions, row_offsets
        )
        holdout_target = strategy_train.read_vector_spans(target_path, holdout_spans)
        holdout_weight = strategy_train.read_vector_spans(weight_path, holdout_spans)
        holdout_base = strategy_train.read_vector_spans(
            base_prediction_path, holdout_spans
        )
        holdout_base_score = strategy_train.weighted_zero_mean_r2(
            holdout_target, holdout_base, holdout_weight
        )
        beta = float(selected["entity_weight"])
        holdout_score = strategy_train.weighted_zero_mean_r2(
            holdout_target,
            holdout_base + beta * holdout_prediction,
            holdout_weight,
        )
        importance_path = variant_dir / "feature_importance.csv"
        with importance_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "importance_gain", "importance_split"])
            for name, gain, split in zip(
                feature_names,
                holdout_model.feature_importance(importance_type="gain"),
                holdout_model.feature_importance(importance_type="split"),
            ):
                writer.writerow([name, float(gain), int(split)])
        fold_deltas = np.asarray(selected["fold_deltas"], dtype=np.float64)
        report = {
            "variant": variant,
            "description": (
                "asset_id removed"
                if variant == "no_asset_id"
                else "asset_id replaced by train-only frozen shrunken residual prior"
            ),
            "reused_production_result": False,
            "feature_count": len(feature_names),
            "selected_entity_weight": beta,
            "entity_rounds": mean_rounds,
            "mean_cv_score": float(selected["mean_fold_score"]),
            "mean_cv_delta": float(np.mean(fold_deltas)),
            "min_cv_delta": float(np.min(fold_deltas)),
            "latest_cv_delta": float(fold_deltas[-1]),
            "all_cv_deltas_positive": bool(np.all(fold_deltas > 0.0)),
            "selected_weight_report": selected,
            "cv_folds": [
                compact_fold_report(fold, selected, index, best_iterations[index])
                for index, fold in enumerate(folds)
            ],
            "holdout": {
                "base_score": float(holdout_base_score),
                "entity_score": float(holdout_score),
                "delta": float(holdout_score - holdout_base_score),
                "entity_prediction_std": float(np.std(holdout_prediction)),
                "entity_prediction_mean": float(np.mean(holdout_prediction)),
            },
            "feature_importance": str(importance_path.relative_to(output_dir)),
            "prior_shrinkage": (
                args.prior_shrinkage if variant == "frozen_prior" else None
            ),
        }
        (variant_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        reports.append(report)
        del development_set, holdout_model
        gc.collect()
        progress(
            f"complete {variant}: mean_cv_delta={report['mean_cv_delta']:.8f}, "
            f"holdout_delta={report['holdout']['delta']:.8f}"
        )

    full = next((item for item in reports if item["variant"] == "full"), None)
    for report in reports:
        if full is None:
            report["mean_cv_score_vs_full"] = None
            report["holdout_score_vs_full"] = None
        else:
            report["mean_cv_score_vs_full"] = (
                report["mean_cv_score"] - full["mean_cv_score"]
            )
            report["holdout_score_vs_full"] = (
                report["holdout"]["entity_score"]
                - full["holdout"]["entity_score"]
            )
    summary = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "experiment": "entity_identity_ablation",
        "time_safety": (
            "prior training rows use only earlier time labels; validation and "
            "holdout use frozen state estimated from their training periods"
        ),
        "variants": reports,
        "config": {
            "requested_variants": variants,
            "prior_shrinkage": args.prior_shrinkage,
            "entity_rounds": args.entity_rounds,
            "early_stopping": args.early_stopping,
            "threads": args.threads,
            "production_feature_signature": feature_stage["signature"],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "variant", "selected_entity_weight", "mean_cv_score",
            "mean_cv_delta", "min_cv_delta", "latest_cv_delta",
            "all_cv_deltas_positive", "holdout_score", "holdout_delta",
            "mean_cv_score_vs_full", "holdout_score_vs_full",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({
                "variant": report["variant"],
                "selected_entity_weight": report["selected_entity_weight"],
                "mean_cv_score": report["mean_cv_score"],
                "mean_cv_delta": report["mean_cv_delta"],
                "min_cv_delta": report["min_cv_delta"],
                "latest_cv_delta": report["latest_cv_delta"],
                "all_cv_deltas_positive": report["all_cv_deltas_positive"],
                "holdout_score": report["holdout"]["entity_score"],
                "holdout_delta": report["holdout"]["delta"],
                "mean_cv_score_vs_full": report["mean_cv_score_vs_full"],
                "holdout_score_vs_full": report["holdout_score_vs_full"],
            })
    progress(f"ablation summary written: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
