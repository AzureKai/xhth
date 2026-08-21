from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = STRATEGY_DIR.parent
DEFAULT_C4_DIR = EXAMPLES_DIR / "responder_assisted_lgb_catboost_strategy"
DEFAULT_ENTITY_DIR = EXAMPLES_DIR / "lgb_v3_market_entity_strategy"
SCHEMA_VERSION = 1
START = time.perf_counter()


def progress(message: str) -> None:
    print(f"[blend {time.perf_counter() - START:8.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict aligned OOF blend of C4 stable and V3 entity models."
    )
    parser.add_argument(
        "--c4-model-dir", default=str(DEFAULT_C4_DIR / "model")
    )
    parser.add_argument(
        "--entity-model-dir", default=str(DEFAULT_ENTITY_DIR / "model")
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--entity-weights",
        default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,"
        "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0",
    )
    parser.add_argument("--required-positive-segment-rate", type=float, default=0.70)
    parser.add_argument("--skip-existing-models", action="store_true")
    return parser.parse_args()


def configured_weights(value: str) -> list[float]:
    result = sorted(set(
        float(item.strip()) for item in value.split(",") if item.strip()
    ))
    if not result or result[0] < 0.0 or result[-1] > 1.0:
        raise ValueError("blend entity weights must be in [0, 1]")
    for endpoint in (0.0, 1.0):
        if endpoint not in result:
            result.append(endpoint)
    return sorted(result)


def fingerprint(path: Path) -> dict:
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


def weighted_zero_mean_r2(
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> float:
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0.0)
    denominator = float(np.sum(w[valid] * y[valid] * y[valid]))
    if denominator <= 0.0:
        return 0.0
    error = y[valid] - pred[valid]
    return float(1.0 - np.sum(w[valid] * error * error) / denominator)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return None
    x = x[valid]
    y = y[valid]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_prediction_artifact(path: Path, *, development: bool) -> dict:
    required = {"time_id", "asset_id", "target", "weight", "prediction"}
    if development:
        required.add("fold_id")
    with np.load(path) as source:
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"prediction artifact is missing fields {missing}: {path}")
        result = {name: np.asarray(source[name]) for name in required}
    rows = len(result["time_id"])
    if rows < 1 or any(len(values) != rows for values in result.values()):
        raise ValueError(f"prediction artifact arrays are not aligned: {path}")
    if np.any(np.diff(result["time_id"].astype(np.int64)) < 0):
        raise ValueError(f"prediction artifact time_id is not sorted: {path}")
    return result


def aligned_common_rows(left: dict, right: dict) -> tuple[dict, dict]:
    start = max(int(left["time_id"][0]), int(right["time_id"][0]))
    end = min(int(left["time_id"][-1]), int(right["time_id"][-1]))
    if start > end:
        raise ValueError("prediction artifacts have no common time range")

    def subset(source: dict) -> dict:
        times = source["time_id"].astype(np.int64, copy=False)
        lower = int(np.searchsorted(times, start, side="left"))
        upper = int(np.searchsorted(times, end, side="right"))
        return {name: values[lower:upper] for name, values in source.items()}

    aligned_left = subset(left)
    aligned_right = subset(right)
    if not np.array_equal(aligned_left["time_id"], aligned_right["time_id"]):
        raise ValueError("common prediction time_id arrays differ")
    if not np.array_equal(aligned_left["asset_id"], aligned_right["asset_id"]):
        raise ValueError("common prediction asset_id arrays differ")
    if not np.allclose(
        aligned_left["target"], aligned_right["target"], rtol=0.0, atol=1e-6
    ):
        raise ValueError("common prediction targets differ")
    if not np.allclose(
        aligned_left["weight"], aligned_right["weight"], rtol=0.0, atol=1e-6
    ):
        raise ValueError("common prediction weights differ")
    return aligned_left, aligned_right


def union_segments(c4: dict, entity: dict) -> list[tuple[int, int]]:
    rows = len(c4["time_id"])
    if rows != len(entity["time_id"]):
        raise ValueError("aligned OOF row counts differ")
    changes = np.flatnonzero(
        (c4["fold_id"][1:] != c4["fold_id"][:-1])
        | (entity["fold_id"][1:] != entity["fold_id"][:-1])
    ) + 1
    boundaries = np.r_[0, changes, rows]
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]


def score_segments(
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    times: np.ndarray,
    c4_folds: np.ndarray,
    entity_folds: np.ndarray,
    segments: list[tuple[int, int]],
) -> list[dict]:
    reports = []
    for index, (start, end) in enumerate(segments, start=1):
        reports.append({
            "segment": index,
            "time_start": int(times[start]),
            "time_end": int(times[end - 1]),
            "rows": int(end - start),
            "c4_fold": int(c4_folds[start]),
            "entity_fold": int(entity_folds[start]),
            "score": weighted_zero_mean_r2(
                target[start:end], prediction[start:end], weight[start:end]
            ),
        })
    return reports


def candidate_report(
    entity_weight: float,
    target: np.ndarray,
    weight: np.ndarray,
    c4_prediction: np.ndarray,
    entity_prediction: np.ndarray,
    times: np.ndarray,
    c4_folds: np.ndarray,
    entity_folds: np.ndarray,
    segments: list[tuple[int, int]],
) -> dict:
    prediction = (
        (1.0 - entity_weight) * c4_prediction
        + entity_weight * entity_prediction
    )
    segment_reports = score_segments(
        target, prediction, weight, times, c4_folds, entity_folds, segments
    )
    scores = np.asarray([item["score"] for item in segment_reports])
    return {
        "entity_weight": float(entity_weight),
        "c4_weight": float(1.0 - entity_weight),
        "global_oof_score": weighted_zero_mean_r2(target, prediction, weight),
        "mean_segment_score": float(np.mean(scores)),
        "std_segment_score": float(np.std(scores)),
        "min_segment_score": float(np.min(scores)),
        "latest_segment_score": float(scores[-1]),
        "segment_scores": list(map(float, scores)),
        "segments": segment_reports,
    }


def comparison(current: dict, reference: dict) -> dict:
    deltas = np.asarray(current["segment_scores"]) - np.asarray(
        reference["segment_scores"]
    )
    return {
        "mean_segment_delta": float(np.mean(deltas)),
        "global_oof_delta": float(
            current["global_oof_score"] - reference["global_oof_score"]
        ),
        "latest_segment_delta": float(deltas[-1]),
        "min_segment_delta": float(np.min(deltas)),
        "positive_segments": int(np.sum(deltas > 0.0)),
        "segment_deltas": list(map(float, deltas)),
    }


def main() -> None:
    args = parse_args()
    weights = configured_weights(args.entity_weights)
    if not 0.0 < args.required_positive_segment_rate <= 1.0:
        raise ValueError("--required-positive-segment-rate must be in (0, 1]")
    c4_model_dir = Path(args.c4_model_dir)
    entity_model_dir = Path(args.entity_model_dir)
    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    c4_metadata_path = c4_model_dir / "metadata.json"
    entity_metadata_path = entity_model_dir / "metadata.json"
    c4_metadata = json.loads(c4_metadata_path.read_text(encoding="utf-8"))
    entity_metadata = json.loads(entity_metadata_path.read_text(encoding="utf-8"))
    if c4_metadata.get("target_variant") != "LGB468_C4_STABLE":
        raise ValueError(
            "fusion is registered for deployed LGB468_C4_STABLE; "
            f"found {c4_metadata.get('target_variant')}"
        )
    if entity_metadata.get("strategy") != "lgb_v3_entity_residual_strategy":
        raise ValueError("entity source metadata is not the registered strategy")
    source_paths = {
        "c4_development": c4_model_dir / c4_metadata.get(
            "development_oof_predictions", "development_oof_predictions.npz"
        ),
        "c4_holdout": c4_model_dir / c4_metadata.get(
            "validation_predictions", "validation_predictions.npz"
        ),
        "entity_development": entity_model_dir / entity_metadata.get(
            "development_oof_predictions", "development_oof_predictions.npz"
        ),
        "entity_holdout": entity_model_dir / entity_metadata.get(
            "validation_predictions", "validation_predictions.npz"
        ),
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing aligned OOF artifacts; rerun both source trainers: "
            + ", ".join(missing)
        )
    source_fingerprints = {
        name: fingerprint(path) for name, path in source_paths.items()
    }
    requested_config = {
        "entity_weights": weights,
        "required_positive_segment_rate": args.required_positive_segment_rate,
        "c4_target_variant": c4_metadata["target_variant"],
        "entity_strategy": entity_metadata["strategy"],
    }
    metadata_path = model_dir / "metadata.json"
    if args.skip_existing_models and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("training_config") == requested_config
            and existing.get("source_artifacts") == source_fingerprints
        ):
            progress("compatible fusion report exists; calibration skipped")
            print(metadata_path.read_text(encoding="utf-8"))
            return

    progress("loading and aligning strict development OOF predictions")
    c4_oof, entity_oof = aligned_common_rows(
        load_prediction_artifact(source_paths["c4_development"], development=True),
        load_prediction_artifact(
            source_paths["entity_development"], development=True
        ),
    )
    segments = union_segments(c4_oof, entity_oof)
    if len(segments) < 4:
        raise ValueError("aligned OOF has too few independent time segments")
    target = c4_oof["target"].astype(np.float64)
    sample_weight = c4_oof["weight"].astype(np.float64)
    c4_prediction = c4_oof["prediction"].astype(np.float64)
    entity_prediction = entity_oof["prediction"].astype(np.float64)
    candidates = [
        candidate_report(
            candidate,
            target,
            sample_weight,
            c4_prediction,
            entity_prediction,
            c4_oof["time_id"],
            c4_oof["fold_id"],
            entity_oof["fold_id"],
            segments,
        )
        for candidate in weights
    ]
    candidates.sort(key=lambda item: (
        -item["mean_segment_score"],
        -item["global_oof_score"],
        abs(item["entity_weight"] - 0.5),
    ))
    selected = candidates[0]
    c4_reference = next(item for item in candidates if item["entity_weight"] == 0.0)
    entity_reference = next(
        item for item in candidates if item["entity_weight"] == 1.0
    )
    selected_vs_c4 = comparison(selected, c4_reference)
    selected_vs_entity = comparison(selected, entity_reference)
    better_parent = (
        entity_reference
        if entity_reference["mean_segment_score"]
        >= c4_reference["mean_segment_score"]
        else c4_reference
    )
    better_parent_name = (
        "entity" if better_parent is entity_reference else "c4"
    )
    selected_vs_better_parent = comparison(selected, better_parent)
    required_positive = int(np.ceil(
        len(segments) * args.required_positive_segment_rate
    ))

    progress("evaluating frozen terminal holdout")
    c4_holdout, entity_holdout = aligned_common_rows(
        load_prediction_artifact(source_paths["c4_holdout"], development=False),
        load_prediction_artifact(source_paths["entity_holdout"], development=False),
    )
    holdout_target = c4_holdout["target"].astype(np.float64)
    holdout_weight = c4_holdout["weight"].astype(np.float64)
    holdout_c4_prediction = c4_holdout["prediction"].astype(np.float64)
    holdout_entity_prediction = entity_holdout["prediction"].astype(np.float64)
    selected_entity_weight = float(selected["entity_weight"])
    holdout_blend_prediction = (
        (1.0 - selected_entity_weight) * holdout_c4_prediction
        + selected_entity_weight * holdout_entity_prediction
    )
    holdout_scores = {
        "c4": weighted_zero_mean_r2(
            holdout_target, holdout_c4_prediction, holdout_weight
        ),
        "entity": weighted_zero_mean_r2(
            holdout_target, holdout_entity_prediction, holdout_weight
        ),
        "selected_blend": weighted_zero_mean_r2(
            holdout_target, holdout_blend_prediction, holdout_weight
        ),
    }
    holdout_scores["delta_vs_c4"] = (
        holdout_scores["selected_blend"] - holdout_scores["c4"]
    )
    holdout_scores["delta_vs_entity"] = (
        holdout_scores["selected_blend"] - holdout_scores["entity"]
    )
    gates = {
        "interior_weight": bool(0.0 < selected_entity_weight < 1.0),
        "mean_segment_beats_both": bool(
            selected["mean_segment_score"]
            > max(
                c4_reference["mean_segment_score"],
                entity_reference["mean_segment_score"],
            )
        ),
        "global_oof_beats_both": bool(
            selected["global_oof_score"]
            > max(
                c4_reference["global_oof_score"],
                entity_reference["global_oof_score"],
            )
        ),
        "latest_segment_beats_both": bool(
            selected["latest_segment_score"]
            > max(
                c4_reference["latest_segment_score"],
                entity_reference["latest_segment_score"],
            )
        ),
        "enough_positive_segments_vs_better_parent": bool(
            selected_vs_better_parent["positive_segments"] >= required_positive
        ),
        "holdout_beats_both": bool(
            holdout_scores["selected_blend"]
            > max(holdout_scores["c4"], holdout_scores["entity"])
        ),
    }
    gates["passed"] = bool(all(gates.values()))
    if gates["passed"]:
        deployment_entity_weight = selected_entity_weight
        deployment_source = "selected_blend"
    elif entity_reference["mean_segment_score"] >= c4_reference["mean_segment_score"]:
        deployment_entity_weight = 1.0
        deployment_source = "entity_fallback"
    else:
        deployment_entity_weight = 0.0
        deployment_source = "c4_fallback"

    diagnostics = {
        "prediction_correlation": safe_correlation(
            c4_prediction, entity_prediction
        ),
        "error_correlation": safe_correlation(
            target - c4_prediction, target - entity_prediction
        ),
        "holdout_prediction_correlation": safe_correlation(
            holdout_c4_prediction, holdout_entity_prediction
        ),
        "common_oof_rows": int(len(target)),
        "common_oof_time_start": int(c4_oof["time_id"][0]),
        "common_oof_time_end": int(c4_oof["time_id"][-1]),
        "common_holdout_rows": int(len(holdout_target)),
        "common_holdout_time_start": int(c4_holdout["time_id"][0]),
        "common_holdout_time_end": int(c4_holdout["time_id"][-1]),
        "union_segment_count": int(len(segments)),
        "required_positive_segments": required_positive,
    }
    c4_strategy_dir = c4_model_dir.resolve().parent
    entity_strategy_dir = entity_model_dir.resolve().parent
    metadata = {
        "strategy": "lgb_c4_entity_blend_strategy",
        "schema_version": SCHEMA_VERSION,
        "prediction_formula": (
            "(1 - entity_weight) * C4_stable + entity_weight * V3_entity"
        ),
        "c4_strategy_dir": os.path.relpath(c4_strategy_dir, STRATEGY_DIR),
        "entity_strategy_dir": os.path.relpath(entity_strategy_dir, STRATEGY_DIR),
        "selected_oof_entity_weight": selected_entity_weight,
        "deployment_entity_weight": deployment_entity_weight,
        "deployment_source": deployment_source,
        "selection_metric": "mean score over union of strict source OOF boundaries",
        "selected_report": selected,
        "c4_reference": c4_reference,
        "entity_reference": entity_reference,
        "selected_vs_c4": selected_vs_c4,
        "selected_vs_entity": selected_vs_entity,
        "better_parent": better_parent_name,
        "selected_vs_better_parent": selected_vs_better_parent,
        "weight_search": sorted(
            candidates, key=lambda item: item["entity_weight"]
        ),
        "holdout": holdout_scores,
        "diagnostics": diagnostics,
        "promotion_gates": gates,
        "source_artifacts": source_fingerprints,
        "source_metadata": {
            "c4": fingerprint(c4_metadata_path),
            "entity": fingerprint(entity_metadata_path),
        },
        "training_config": requested_config,
    }
    metadata["signature"] = signature({
        "source_artifacts": source_fingerprints,
        "training_config": requested_config,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (work_dir / "fusion_report.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    progress(
        f"complete: selected_entity_weight={selected_entity_weight:.2f}, "
        f"deployment_entity_weight={deployment_entity_weight:.2f}, "
        f"holdout={holdout_scores['selected_blend']:.8f}, "
        f"gates={gates['passed']}"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
