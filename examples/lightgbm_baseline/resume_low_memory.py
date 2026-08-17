from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from train import (
    LEGACY_PARAM_CANDIDATES as PARAM_CANDIDATES,
    _legacy_candidate_params,
    _select_winning_candidate,
)
from train_low_memory import (
    HOLDOUT_FRACTION,
    N_SPLITS,
    PURGE_STEPS,
    SEEDS,
    add_stats,
    build_dataset,
    log,
    prediction_statistics,
    row_offsets_from_counts,
    score_from_stats,
    span_length,
    spans_from_time_ids,
    train_early_stopping,
    train_fixed,
)
from validation import evaluate_gates, make_purged_kfold_plan


CV_PATTERN = re.compile(
    r"CV done candidate=(?P<candidate>\w+) fold=(?P<fold>\d+) "
    r"best_iteration=(?P<iteration>\d+) valid_raw=(?P<score>[-+0-9.eE]+)"
)
SELECTED_PATTERN = re.compile(
    r"selected candidate=(?P<candidate>\w+) mean_fold_score=(?P<mean>[-+0-9.eE]+) "
    r"rounds=(?P<rounds>\d+) oof_raw=(?P<oof>[-+0-9.eE]+)"
)
HOLDOUT_PATTERN = re.compile(r"holdout_raw=(?P<score>[-+0-9.eE]+)")
LEGACY_CONSTRUCTION_OVERRIDES = {"force_col_wise": False}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_completed_run_log(
    log_path: Path,
    unique_times: np.ndarray,
    row_offsets: np.ndarray,
    plan,
) -> tuple[list[dict], dict, float, float]:
    text = log_path.read_text(encoding="utf-8")
    parsed: dict[str, dict[int, tuple[int, float]]] = {
        candidate["name"]: {} for candidate in PARAM_CANDIDATES
    }
    for match in CV_PATTERN.finditer(text):
        name = match.group("candidate")
        fold_id = int(match.group("fold"))
        value = (int(match.group("iteration")), float(match.group("score")))
        if name not in parsed:
            raise ValueError(f"unknown candidate in training log: {name}")
        previous = parsed[name].get(fold_id)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting CV results in training log for {name} fold={fold_id}")
        parsed[name][fold_id] = value
    expected_fold_ids = set(range(N_SPLITS))
    for name, folds in parsed.items():
        if set(folds) != expected_fold_ids:
            raise ValueError(f"training log lacks complete CV results for {name}: {sorted(folds)}")

    candidate_results: list[dict] = []
    for candidate in PARAM_CANDIDATES:
        fold_scores: list[dict] = []
        iterations: list[int] = []
        for fold in plan.folds:
            iteration, score = parsed[candidate["name"]][fold.fold_id]
            train_spans = spans_from_time_ids(unique_times, row_offsets, fold.train_time_ids)
            valid_spans = spans_from_time_ids(unique_times, row_offsets, fold.valid_time_ids)
            iterations.append(iteration)
            fold_scores.append(
                {
                    "fold_id": int(fold.fold_id),
                    "best_iteration": iteration,
                    "valid_raw": score,
                    "train_rows": span_length(train_spans),
                    "valid_rows": span_length(valid_spans),
                }
            )
        candidate_results.append(
            {
                "name": candidate["name"],
                "logic": candidate.get("logic", ""),
                "regularization_rank": int(candidate.get("regularization_rank", 0)),
                "params": {
                    "num_leaves": int(candidate["num_leaves"]),
                    "min_data_in_leaf": int(candidate["min_data_in_leaf"]),
                    "feature_fraction": float(candidate["feature_fraction"]),
                    "bagging_fraction": float(candidate["bagging_fraction"]),
                    "lambda_l2": float(candidate["lambda_l2"]),
                },
                "fold_scores": fold_scores,
                "fold_best_iterations": iterations,
                "mean_fold_score": float(np.mean([item["valid_raw"] for item in fold_scores])),
                "mean_iterations": max(1, int(round(float(np.mean(iterations))))),
            }
        )

    selected_matches = list(SELECTED_PATTERN.finditer(text))
    holdout_matches = list(HOLDOUT_PATTERN.finditer(text))
    if not selected_matches or not holdout_matches:
        raise ValueError("training log does not contain selected-candidate and holdout results")
    selected = selected_matches[-1]
    logged_selection = {
        "candidate": selected.group("candidate"),
        "mean_fold_score": float(selected.group("mean")),
        "rounds": int(selected.group("rounds")),
        "oof_raw": float(selected.group("oof")),
    }
    holdout_raw = float(holdout_matches[-1].group("score"))

    winner = _select_winning_candidate(candidate_results)
    if winner["name"] != logged_selection["candidate"]:
        raise ValueError(f"parsed winner mismatch: {winner['name']} != {logged_selection['candidate']}")
    if int(winner["mean_iterations"]) != logged_selection["rounds"]:
        raise ValueError("parsed mean iteration count differs from the original run")
    if abs(float(winner["mean_fold_score"]) - logged_selection["mean_fold_score"]) > 5e-10:
        raise ValueError("parsed mean fold score differs from the original run")
    return candidate_results, winner, logged_selection["oof_raw"], holdout_raw


def validate_existing_model(
    path: Path,
    model_cols: list[str],
    expected_iterations: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    booster = lgb.Booster(model_file=str(path))
    if list(booster.feature_name()) != model_cols:
        raise ValueError(f"existing model feature schema mismatch: {path}")
    if booster.current_iteration() != expected_iterations:
        raise ValueError(
            f"existing model iteration mismatch: {booster.current_iteration()} != {expected_iterations}"
        )


def run_resume(
    *,
    model_dir: Path,
    cache_dir: Path,
    original_log: Path,
    num_threads: int,
) -> dict:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise ValueError("low-memory feature cache is incomplete")
    axis = np.load(cache_dir / metadata["time_axis_file"])
    unique_times = axis["unique_times"]
    time_counts = axis["time_counts"]
    row_offsets = row_offsets_from_counts(time_counts)
    # This helper exists only to reproduce the already-completed legacy run.
    plan = make_purged_kfold_plan(
        unique_times,
        n_splits=N_SPLITS,
        holdout_fraction=HOLDOUT_FRACTION,
        purge_steps=PURGE_STEPS,
    )

    matrix_path = cache_dir / metadata["matrix_file"]
    target_path = cache_dir / metadata["target_file"]
    weight_path = cache_dir / metadata["weight_file"]
    matrix_shape = tuple(int(value) for value in metadata["matrix_shape"])
    model_cols = list(metadata["model_cols"])
    candidate_results, winner, logged_oof_raw, holdout_raw = parse_completed_run_log(
        original_log,
        unique_times,
        row_offsets,
        plan,
    )
    winning_candidate = next(item for item in PARAM_CANDIDATES if item["name"] == winner["name"])
    mean_iterations = int(winner["mean_iterations"])
    log(
        f"resume verified original CV: candidate={winner['name']} "
        f"mean_fold_score={winner['mean_fold_score']:.8g} rounds={mean_iterations}"
    )

    state_path = cache_dir / "resume_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("winner") != winner["name"] or state.get("mean_iterations") != mean_iterations:
            raise ValueError("resume checkpoint does not match the parsed training result")
    else:
        state = {
            "winner": winner["name"],
            "mean_iterations": mean_iterations,
            "oof_fold_stats": {},
            "model_files": ["model_seed2026.txt"] if (model_dir / "model_seed2026.txt").exists() else [],
        }
        atomic_json(state_path, state)

    total_stats = {"y2": 0.0, "residual": 0.0, "yp": 0.0, "p2": 0.0}
    for fold in plan.folds:
        key = str(fold.fold_id)
        if key in state["oof_fold_stats"]:
            stats = {name: float(value) for name, value in state["oof_fold_stats"][key].items()}
            log(f"resume reusing OOF stats for fold={fold.fold_id}")
        else:
            train_spans = spans_from_time_ids(unique_times, row_offsets, fold.train_time_ids)
            valid_spans = spans_from_time_ids(unique_times, row_offsets, fold.valid_time_ids)
            log(f"resume reconstructing winner OOF fold={fold.fold_id}")
            train_set = build_dataset(
                matrix_path,
                matrix_shape,
                target_path,
                weight_path,
                train_spans,
                model_cols,
                data_random_seed=SEEDS[0],
                construction_overrides=LEGACY_CONSTRUCTION_OVERRIDES,
            )
            valid_set = build_dataset(
                matrix_path,
                matrix_shape,
                target_path,
                weight_path,
                valid_spans,
                model_cols,
                reference=train_set,
                data_random_seed=SEEDS[0],
                construction_overrides=LEGACY_CONSTRUCTION_OVERRIDES,
            )
            model = train_early_stopping(
                train_set,
                valid_set,
                winning_candidate,
                num_threads,
                param_builder=_legacy_candidate_params,
            )
            best_iteration = int(model.best_iteration or 700)
            expected_iteration = int(winner["fold_best_iterations"][fold.fold_id])
            if best_iteration != expected_iteration:
                raise ValueError(
                    f"winner reconstruction iteration mismatch on fold={fold.fold_id}: "
                    f"{best_iteration} != {expected_iteration}"
                )
            stats = prediction_statistics(
                model,
                matrix_path,
                target_path,
                weight_path,
                valid_spans,
                best_iteration,
            )
            score = score_from_stats(stats)
            expected_score = float(winner["fold_scores"][fold.fold_id]["valid_raw"])
            if abs(score - expected_score) > 5e-8:
                raise ValueError(
                    f"winner reconstruction score mismatch on fold={fold.fold_id}: "
                    f"{score} != {expected_score}"
                )
            state["oof_fold_stats"][key] = stats
            atomic_json(state_path, state)
            log(f"resume saved OOF stats fold={fold.fold_id} score={score:.8g}")
            del model, valid_set, train_set
            gc.collect()
        add_stats(total_stats, stats)

    recomputed_oof_raw = score_from_stats(total_stats)
    if abs(recomputed_oof_raw - logged_oof_raw) > 5e-8:
        raise ValueError(f"aggregate OOF mismatch: {recomputed_oof_raw} != {logged_oof_raw}")
    fitted_oof_scale = 1.0 if total_stats["p2"] <= 0.0 else float(total_stats["yp"] / total_stats["p2"])
    gates = evaluate_gates(
        oof_raw_score=recomputed_oof_raw,
        holdout_raw_score=holdout_raw,
        fitted_oof_scale=fitted_oof_scale,
    )
    log(
        f"resume OOF restored: raw={recomputed_oof_raw:.8g} "
        f"scale={fitted_oof_scale:.8g} gates_passed={gates['gates_passed']}"
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    all_spans = [(0, int(metadata["total_rows"]))]
    model_files: list[str] = []
    for seed in SEEDS:
        name = f"model_seed{seed}.txt"
        path = model_dir / name
        if path.exists():
            validate_existing_model(path, model_cols, mean_iterations)
            log(f"resume validated existing final model seed={seed}")
            model_files.append(name)
            continue
        log(f"resume constructing all-train Dataset for seed={seed}")
        all_set = build_dataset(
            matrix_path,
            matrix_shape,
            target_path,
            weight_path,
            all_spans,
            model_cols,
            data_random_seed=int(seed),
            construction_overrides=LEGACY_CONSTRUCTION_OVERRIDES,
        )
        log(f"resume final fit seed={seed} rounds={mean_iterations}")
        booster = train_fixed(
            all_set,
            winning_candidate,
            int(seed),
            mean_iterations,
            num_threads,
            param_builder=_legacy_candidate_params,
        )
        booster.save_model(str(path))
        validate_existing_model(path, model_cols, mean_iterations)
        model_files.append(name)
        state["model_files"] = model_files
        atomic_json(state_path, state)
        del booster, all_set
        gc.collect()

    development_spans = spans_from_time_ids(unique_times, row_offsets, plan.development_time_ids)
    holdout_spans = spans_from_time_ids(unique_times, row_offsets, plan.holdout_time_ids)
    # Every candidate used the same validation rows and weights, so its aggregate
    # OOF R2 can be recovered from the logged per-fold R2 values and each fold's
    # weighted target sum of squares: sum(score_i * y2_i) / sum(y2_i).
    # Keep the winning value from its reconstructed predictions at full precision.
    fold_y2 = {
        int(fold_id): float(stats["y2"])
        for fold_id, stats in state["oof_fold_stats"].items()
    }
    total_oof_y2 = float(sum(fold_y2.values()))
    for item in candidate_results:
        if item["name"] == winner["name"]:
            item["oof_raw"] = recomputed_oof_raw
        else:
            item["oof_raw"] = float(
                sum(
                    fold_y2[int(fold_score["fold_id"])] * float(fold_score["valid_raw"])
                    for fold_score in item["fold_scores"]
                )
                / total_oof_y2
            )
    report = {
        "strategy": "lightgbm_baseline",
        "schema_version": 1,
        "tuning_policy": "purged_kfold_pre_registered_candidates_no_test_tuning",
        "scale_policy": "diagnostic_only_never_apply",
        "execution": {
            "data_backend": "disk_memmap_lightgbm_sequence",
            "semantics": "full_readme_defaults",
            "cache_dir": str(cache_dir.resolve()),
            "resumed_after_seed_dataset_compatibility_fix": True,
            "non_winner_oof_aggregates": "reconstructed_from_logged_fold_scores",
        },
        "rows": {
            "train_all": int(metadata["total_rows"]),
            "oof": span_length(development_spans),
            "holdout": span_length(holdout_spans),
            "development": span_length(development_spans),
            "final_train_sample": int(metadata["total_rows"]),
            "final_train_includes_holdout": True,
        },
        "validation": {
            "cv_scheme": plan.cv_scheme,
            "n_splits": N_SPLITS,
            "holdout_fraction": HOLDOUT_FRACTION,
            "purge_steps": PURGE_STEPS,
            "rounds_aggregation": "mean",
            "selection_metric": "mean_fold_score",
            "tie_break": ["stronger_regularization", "fewer_mean_iterations"],
            "candidates": candidate_results,
            "selected_candidate": winner["name"],
            "fold_scores": winner["fold_scores"],
            "fold_best_iterations": winner["fold_best_iterations"],
            "mean_iterations": mean_iterations,
            "mean_fold_score": winner["mean_fold_score"],
            "oof_raw": recomputed_oof_raw,
            "holdout_raw": holdout_raw,
            "fitted_oof_scale": fitted_oof_scale,
            "gates": gates,
        },
        "features": {
            "selected_raw_features": metadata["raw_features"],
            "history_features": metadata["history_features"],
            "rolling_windows": metadata["rolling_windows"],
            "model_feature_count": len(model_cols),
        },
        "seeds": list(SEEDS),
        "model_files": model_files,
        "best_iteration": mean_iterations,
        "best_iterations": [mean_iterations] * len(model_files),
        "prediction_scale": 1.0,
        "fitted_oof_scale": fitted_oof_scale,
        "gates_passed": gates["gates_passed"],
        "selected_candidate": winner["name"],
        "num_threads": int(num_threads),
        "lgbm_params": _legacy_candidate_params(SEEDS[0], winning_candidate, num_threads=num_threads),
    }
    atomic_json(model_dir / "lightgbm_report.json", report)
    log(f"resume complete; report and {len(model_files)} models are ready")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the full low-memory baseline after final seed fit interruption.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--original-log", required=True)
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_resume(
        model_dir=Path(args.model_dir),
        cache_dir=Path(args.cache_dir),
        original_log=Path(args.original_log),
        num_threads=int(args.num_threads),
    )
    print(
        json.dumps(
            {
                "gates_passed": report["gates_passed"],
                "selected_candidate": report["selected_candidate"],
                "mean_iterations": report["best_iteration"],
                "model_files": report["model_files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
