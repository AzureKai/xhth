from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import run_entity_identity_ablation as identity
import train as strategy_train


START = time.perf_counter()
SCHEMA_VERSION = 1
VARIANT_SPECS = {
    "full_global_prior": [
        ("frozen_entity_residual_prior", None),
    ],
    "full_ema50_prior": [
        ("frozen_entity_residual_ema50", 50.0),
    ],
    "full_multiscale_prior": [
        ("frozen_entity_residual_ema50", 50.0),
        ("frozen_entity_residual_ema200", 200.0),
    ],
}


def progress(message: str) -> None:
    print(f"[prior-ablation {time.perf_counter() - START:9.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-safe residual priors added on top of categorical asset_id."
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
        default="full," + ",".join(VARIANT_SPECS),
        help=(
            "Comma-separated subset of full,full_global_prior,"
            "full_ema50_prior,full_multiscale_prior."
        ),
    )
    parser.add_argument("--prior-shrinkage", type=float, default=100.0)
    parser.add_argument("--entity-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_variants(value: str) -> list[str]:
    allowed = {"full", *VARIANT_SPECS}
    result = []
    for item in value.split(","):
        name = item.strip()
        if name and name not in result:
            result.append(name)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"unknown prior ablation variants: {unknown}")
    if not result:
        raise ValueError("at least one prior ablation variant is required")
    return result


def _prior_values(
    assets: np.ndarray,
    specs: list[tuple[str, float | None]],
    sums: dict[int, float],
    counts: dict[int, int],
    emas: dict[float, dict[int, float]],
    shrinkage: float,
) -> np.ndarray:
    unique, inverse = np.unique(assets, return_inverse=True)
    mapped = np.empty((len(unique), len(specs)), dtype=np.float32)
    for row, asset in enumerate(unique):
        key = int(asset)
        count = counts.get(key, 0)
        reliability = count / (count + shrinkage)
        for column, (_, half_life) in enumerate(specs):
            if half_life is None:
                value = sums.get(key, 0.0) / (count + shrinkage)
            else:
                value = reliability * emas[half_life].get(key, 0.0)
            mapped[row, column] = np.float32(value)
    return mapped[inverse]


def _update_prior_state(
    assets: np.ndarray,
    labels: np.ndarray,
    specs: list[tuple[str, float | None]],
    sums: dict[int, float],
    counts: dict[int, int],
    emas: dict[float, dict[int, float]],
) -> None:
    unique, inverse = np.unique(assets, return_inverse=True)
    label_sums = np.bincount(inverse, weights=labels, minlength=len(unique))
    label_counts = np.bincount(inverse, minlength=len(unique))
    half_lives = sorted({half_life for _, half_life in specs if half_life is not None})
    for row, asset in enumerate(unique):
        key = int(asset)
        observation = float(label_sums[row] / max(label_counts[row], 1))
        sums[key] = sums.get(key, 0.0) + float(label_sums[row])
        previous_count = counts.get(key, 0)
        counts[key] = previous_count + int(label_counts[row])
        for half_life in half_lives:
            state = emas[half_life]
            if previous_count == 0:
                state[key] = observation
            else:
                alpha = 1.0 - math.exp(math.log(0.5) / half_life)
                state[key] = (1.0 - alpha) * state[key] + alpha * observation


def materialize_temporal_priors(
    source_feature_path: Path,
    asset_column: int,
    label_path: Path,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    row_offsets: np.ndarray,
    specs: list[tuple[str, float | None]],
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
        train_path,
        mode="w+",
        dtype=np.float32,
        shape=(train_rows, len(specs)),
    )
    valid_output = np.lib.format.open_memmap(
        valid_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_rows, len(specs)),
    )
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    emas = {
        half_life: {}
        for _, half_life in specs
        if half_life is not None
    }
    cursor = 0
    report_every = max(1, len(train_positions) // 20)
    for completed, position in enumerate(train_positions, start=1):
        start = int(row_offsets[position])
        end = int(row_offsets[position + 1])
        assets = np.asarray(source[start:end, asset_column], dtype=np.int64)
        next_cursor = cursor + len(assets)
        train_output[cursor:next_cursor] = _prior_values(
            assets, specs, sums, counts, emas, shrinkage
        )
        _update_prior_state(
            assets,
            np.asarray(labels[start:end], dtype=np.float64),
            specs,
            sums,
            counts,
            emas,
        )
        cursor = next_cursor
        if completed == len(train_positions) or completed % report_every == 0:
            strategy_train.progress_bar(
                "temporal prior train",
                completed,
                len(train_positions),
                f"entities={len(counts)}, features={len(specs)}",
            )
    if cursor != train_rows:
        raise ValueError("temporal prior train row count mismatch")
    cursor = 0
    for position in valid_positions:
        start = int(row_offsets[position])
        end = int(row_offsets[position + 1])
        assets = np.asarray(source[start:end, asset_column], dtype=np.int64)
        next_cursor = cursor + len(assets)
        valid_output[cursor:next_cursor] = _prior_values(
            assets, specs, sums, counts, emas, shrinkage
        )
        cursor = next_cursor
    if cursor != valid_rows:
        raise ValueError("temporal prior validation row count mismatch")
    train_output.flush()
    valid_output.flush()
    del source, labels, train_output, valid_output
    gc.collect()


def ensure_temporal_priors(
    variant_dir: Path,
    split_name: str,
    signature_payload: dict,
    source_feature_path: Path,
    asset_column: int,
    label_path: Path,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    row_offsets: np.ndarray,
    specs: list[tuple[str, float | None]],
    shrinkage: float,
    skip_existing: bool,
) -> tuple[Path, Path]:
    train_path = variant_dir / f"{split_name}_train_prior.npy"
    valid_path = variant_dir / f"{split_name}_valid_prior.npy"
    stage_path = variant_dir / f"{split_name}_prior.json"
    current_signature = strategy_train.signature(signature_payload)
    if skip_existing and train_path.exists() and valid_path.exists() and stage_path.exists():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("signature") == current_signature:
            progress(f"reusing temporal prior: {variant_dir.name}/{split_name}")
            return train_path, valid_path
    progress(f"materializing temporal prior: {variant_dir.name}/{split_name}")
    materialize_temporal_priors(
        source_feature_path,
        asset_column,
        label_path,
        train_positions,
        valid_positions,
        row_offsets,
        specs,
        shrinkage,
        train_path,
        valid_path,
    )
    stage_path.write_text(json.dumps({
        "signature": current_signature,
        "signature_payload": signature_payload,
    }, indent=2), encoding="utf-8")
    return train_path, valid_path


def _prior_signature_payload(
    feature_signature: str,
    split_name: str,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    specs: list[tuple[str, float | None]],
    shrinkage: float,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "production_feature_signature": feature_signature,
        "split": split_name,
        "train_positions": [
            int(train_positions[0]), int(train_positions[-1]),
            int(len(train_positions)),
        ],
        "valid_positions": [
            int(valid_positions[0]), int(valid_positions[-1]),
            int(len(valid_positions)),
        ],
        "prior_specs": [
            {"feature": name, "half_life": half_life}
            for name, half_life in specs
        ],
        "prior_shrinkage": shrinkage,
        "method": "causal_train_then_frozen_validation_v2",
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
    output_dir = work_dir / "entity_prior_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if int(metadata.get("schema_version", 0)) != strategy_train.SCHEMA_VERSION:
        raise ValueError("production entity model must use schema_version=2")
    production_config = metadata.get("training_config", {})
    mismatched = {
        key: (production_config.get(key), value)
        for key, value in {
            "entity_rounds": args.entity_rounds,
            "early_stopping": args.early_stopping,
        }.items()
        if production_config.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "prior ablation must use the production training budget; "
            f"metadata/request mismatches: {mismatched}"
        )

    source_paths = strategy_train.require_source_artifacts(source_work_dir)
    feature_stage = json.loads(
        (work_dir / "entity_feature_stage.json").read_text(encoding="utf-8")
    )
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
        raise FileNotFoundError("missing production feature cache; run train.py first")

    source_columns, base_feature_names = identity.source_columns_for_variant(
        "full", source_names, production_names
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
    target_path = base_cache_dir / cache["target_file"]
    weight_path = base_cache_dir / cache["weight_file"]
    base_prediction_path = source_paths["base_prediction"]
    asset_column = source_names.index("asset_id")
    weights = [
        float(item["entity_weight"])
        for item in metadata["selected_weight_report"]["weight_search"]
    ]

    reports = []
    if "full" in variants:
        full_report = identity.existing_full_report(metadata)
        full_report["description"] = "categorical asset_id without residual prior"
        reports.append(full_report)
        progress("reused full production CV/holdout result")

    for variant in [name for name in variants if name != "full"]:
        specs = VARIANT_SPECS[variant]
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        feature_names = [
            *base_feature_names,
            *extra_names,
            *(name for name, _ in specs),
        ]
        folds = []
        best_iterations = []
        for eval_index in range(1, len(fold_spans)):
            train_spans = [
                span for group in fold_spans[:eval_index] for span in group
            ]
            train_positions = np.concatenate(fold_positions[:eval_index])
            valid_spans = fold_spans[eval_index]
            valid_positions = fold_positions[eval_index]
            prior_payload = _prior_signature_payload(
                feature_stage["signature"],
                f"fold{eval_index}",
                train_positions,
                valid_positions,
                specs,
                args.prior_shrinkage,
            )
            train_prior_path, valid_prior_path = ensure_temporal_priors(
                variant_dir,
                f"fold{eval_index}",
                prior_payload,
                source_feature_path,
                asset_column,
                entity_label_path,
                train_positions,
                valid_positions,
                row_offsets,
                specs,
                args.prior_shrinkage,
                args.skip_existing_models,
            )
            model_path = variant_dir / f"fold{eval_index}.txt"
            model_meta_path = model_path.with_suffix(".json")
            model_signature_payload = {
                "schema_version": SCHEMA_VERSION,
                "production_feature_signature": feature_stage["signature"],
                "variant": variant,
                "feature_names": feature_names,
                "eval_index": eval_index,
                "rounds": args.entity_rounds,
                "early_stopping": args.early_stopping,
                "prior": prior_payload,
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
                train_set = identity.build_dataset(
                    source_feature_path,
                    source_columns,
                    extra_feature_path,
                    entity_label_path,
                    weight_path,
                    train_spans,
                    feature_names,
                    train_prior_path,
                )
                valid_set = identity.build_dataset(
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
            raw_prediction = identity.predict(
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
        holdout_payload = _prior_signature_payload(
            feature_stage["signature"],
            "holdout",
            development_positions,
            holdout_positions,
            specs,
            args.prior_shrinkage,
        )
        train_prior_path, holdout_prior_path = ensure_temporal_priors(
            variant_dir,
            "holdout",
            holdout_payload,
            source_feature_path,
            asset_column,
            entity_label_path,
            development_positions,
            holdout_positions,
            row_offsets,
            specs,
            args.prior_shrinkage,
            args.skip_existing_models,
        )
        development_set = identity.build_dataset(
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
        holdout_prediction = strategy_train.center_entity_by_time(
            identity.predict(
                holdout_model,
                source_feature_path,
                source_columns,
                extra_feature_path,
                holdout_spans,
                mean_rounds,
                holdout_prior_path,
            ),
            holdout_positions,
            row_offsets,
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
            "description": "categorical asset_id plus time-safe residual priors",
            "prior_specs": [
                {"feature": name, "half_life": half_life}
                for name, half_life in specs
            ],
            "prior_shrinkage": args.prior_shrinkage,
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
                identity.compact_fold_report(
                    fold, selected, index, best_iterations[index]
                )
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
        report["mean_cv_score_vs_full"] = (
            None if full is None
            else report["mean_cv_score"] - full["mean_cv_score"]
        )
        report["holdout_score_vs_full"] = (
            None if full is None
            else report["holdout"]["entity_score"]
            - full["holdout"]["entity_score"]
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "entity_prior_ablation",
        "time_safety": (
            "training priors use only earlier labels; validation and holdout "
            "use states frozen at the end of their training periods"
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
    progress(f"prior ablation summary written: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
