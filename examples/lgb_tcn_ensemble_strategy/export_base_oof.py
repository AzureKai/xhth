from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


START = time.perf_counter()


def progress(message):
    print(f"[base-oof {time.perf_counter() - START:9.1f}s] {message}", flush=True)


def round_progress(fold, total_folds, rounds):
    report_every = max(1, rounds // 10)

    def callback(environment):
        current = environment.iteration + 1
        if current == 1 or current == rounds or current % report_every == 0:
            print(
                f"[base-oof fold {fold}/{total_folds}] "
                f"{current}/{rounds} rounds "
                f"({100.0 * current / max(rounds, 1):.1f}%)",
                flush=True,
            )

    callback.order = 20
    callback.before_iteration = False
    return callback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export expanding-window OOF predictions for the selected LGB target model."
    )
    parser.add_argument("--base-work-dir", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    import lightgbm as lgb

    strategy_dir = Path(__file__).resolve().parent.parent / (
        "responder_assisted_lgb_catboost_strategy"
    )
    sys.path.insert(0, str(strategy_dir))
    from train import (  # noqa: PLC0415
        all_times,
        load_array,
        matrix_for_segments,
        segments_for_range,
        vector_for_segments,
    )

    work_dir = Path(args.base_work_dir)
    cache_dir = work_dir / "cache"
    model_dir = Path(args.base_model_dir)
    cache = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    valid_cutoff = int(metadata["valid_cutoff_time_id"])
    times_all = all_times(cache_dir, cache)
    train_times = times_all[times_all < valid_cutoff]
    warmup_index = max(
        1, int(len(train_times) * float(metadata["warmup_fraction"]))
    )
    oof_start = int(train_times[warmup_index])
    segments = segments_for_range(
        cache_dir, cache, oof_start, valid_cutoff
    )
    times = vector_for_segments(cache_dir, segments, "time").astype(np.int64)
    target = vector_for_segments(cache_dir, segments, "target").astype(np.float32)
    weight = vector_for_segments(cache_dir, segments, "weight").astype(np.float32)
    asset = vector_for_segments(
        cache_dir, segments, "x", column=-1
    ).astype(np.int64)
    responders = list(metadata["responders"])
    target_responders = list(metadata["target_responders"])
    responder_indices = [responders.index(name) for name in target_responders]
    oof_hat_path = work_dir / "oof_responder_hat.dat"
    oof_hat = np.memmap(
        oof_hat_path, dtype="float32", mode="r",
        shape=(len(times), len(responders)),
    )
    extra = (
        np.asarray(oof_hat[:, responder_indices], dtype=np.float32)
        if responder_indices else None
    )
    base_indices = np.asarray(metadata["target_base_indices"], dtype=np.int64)
    progress(
        f"materializing target matrix: rows={len(times):,}, "
        f"base_features={len(base_indices)}, responders={target_responders}"
    )
    x = matrix_for_segments(
        cache_dir, segments, extra=extra, base_indices=base_indices
    )
    unique_times = np.unique(times)
    screening_warmup = max(1, int(len(unique_times) * args.warmup_fraction))
    blocks = np.array_split(unique_times[screening_warmup:], args.folds)
    prediction = np.full(len(times), np.nan, dtype=np.float32)
    variant = metadata["target_variant"]
    rounds = int(
        metadata["ablation_scores"][variant]["best_iteration"]
    )
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "num_threads": args.threads,
        "seed": args.seed,
        "verbosity": -1,
    }
    for fold, block in enumerate(blocks, start=1):
        if len(block) == 0:
            continue
        fold_start = int(block[0])
        fold_end = int(block[-1])
        train_stop = int(np.searchsorted(times, fold_start, side="left"))
        pred_stop = int(np.searchsorted(times, fold_end, side="right"))
        progress(
            f"fold {fold}/{args.folds}: train_rows={train_stop:,}, "
            f"prediction_rows={pred_stop - train_stop:,}, rounds={rounds}"
        )
        dataset = lgb.Dataset(
            x[:train_stop], label=target[:train_stop],
            weight=weight[:train_stop], free_raw_data=False,
        )
        model = lgb.train(
            params, dataset, num_boost_round=max(rounds, 1),
            callbacks=[
                round_progress(fold, args.folds, max(rounds, 1)),
                lgb.log_evaluation(50),
            ],
        )
        prediction[train_stop:pred_stop] = np.clip(
            model.predict(x[train_stop:pred_stop]),
            float(metadata.get("clip_min", -np.inf)),
            float(metadata.get("clip_max", np.inf)),
        )
        progress(f"fold {fold}/{args.folds} complete")
    valid = np.isfinite(prediction)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time_id=times[valid],
        asset_id=asset[valid],
        target=target[valid],
        weight=weight[valid],
        prediction=prediction[valid],
    )
    progress(f"saved {int(valid.sum()):,} strict OOF target predictions: {output}")


if __name__ == "__main__":
    main()
