from __future__ import annotations

import argparse
import json
import shutil
import time
from bisect import bisect_right
from pathlib import Path
from typing import Sequence as TypingSequence

import numpy as np
import lightgbm as lgb


RESPONDERS = ["responder_03", "responder_28", "responder_29", "responder_02"]


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-core responder-stacked LightGBM")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.25)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--responder-rounds", type=int, default=500)
    parser.add_argument("--target-rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--skip-existing-models",
        action="store_true",
        help="Load model files that already exist and train only missing models.",
    )
    return parser.parse_args()


START_TIME = time.perf_counter()


def progress(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[progress {elapsed:9.1f}s] {message}", flush=True)


def manifest_files(root: Path) -> list[Path]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get("train", [])
        if files:
            return [root / str(item) for item in files]
    return sorted((root / "train").glob("*.parquet"))


def iter_time_frames(files: list[Path], columns: list[str], batch_size: int):
    import pandas as pd
    import pyarrow.parquet as pq

    carry = None
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
                carry = None
            last_time = frame["time_id"].iloc[-1]
            complete = frame["time_id"] != last_time
            ready = frame.loc[complete]
            carry = frame.loc[~complete].copy()
            for time_id, group in ready.groupby("time_id", sort=False):
                yield int(time_id), group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        yield int(carry["time_id"].iloc[0]), carry.reset_index(drop=True)


def build_cache(data_root: Path, cache_dir: Path, shard_rows: int, batch_size: int):
    import pyarrow.parquet as pq

    files = manifest_files(data_root)
    if not files:
        raise ValueError("no training parquet files")
    columns = list(pq.read_schema(files[0]).names)
    features = [name for name in columns if name.startswith("feature_")]
    missing = [name for name in ["target", "weight", *RESPONDERS] if name not in columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    read_columns = ["time_id", "asset_id", "weight", "target", *RESPONDERS, *features]
    buffers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    buffered_rows = 0
    shards = []

    def flush():
        nonlocal buffers, buffered_rows
        if not buffers:
            return
        shard_id = len(shards)
        x = np.concatenate([item[0] for item in buffers])
        time_id = np.concatenate([item[1] for item in buffers])
        target = np.concatenate([item[2] for item in buffers])
        weight = np.concatenate([item[3] for item in buffers])
        responder = np.concatenate([item[4] for item in buffers])
        prefix = cache_dir / f"shard_{shard_id:05d}"
        np.save(str(prefix) + "_x.npy", x, allow_pickle=False)
        np.save(str(prefix) + "_time.npy", time_id, allow_pickle=False)
        np.save(str(prefix) + "_target.npy", target, allow_pickle=False)
        np.save(str(prefix) + "_weight.npy", weight, allow_pickle=False)
        np.save(str(prefix) + "_responder.npy", responder, allow_pickle=False)
        shards.append({"id": shard_id, "rows": len(x), "time_min": int(time_id[0]), "time_max": int(time_id[-1])})
        total_rows = sum(int(item["rows"]) for item in shards)
        progress(
            f"cache shard {shard_id + 1} written: {len(x):,} rows; "
            f"total={total_rows:,}, time_id={int(time_id[0])}..{int(time_id[-1])}"
        )
        buffers, buffered_rows = [], 0

    for _, frame in iter_time_frames(files, read_columns, batch_size):
        raw = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        asset = frame["asset_id"].to_numpy(dtype=np.float32).reshape(-1, 1)
        x = np.hstack([raw, asset])
        time_id = frame["time_id"].to_numpy(dtype=np.int64)
        target = frame["target"].to_numpy(dtype=np.float32)
        weight = np.maximum(frame["weight"].to_numpy(dtype=np.float32), 0.0)
        responder = frame.loc[:, RESPONDERS].to_numpy(dtype=np.float32)
        buffers.append((x, time_id, target, weight, responder))
        buffered_rows += len(frame)
        if buffered_rows >= shard_rows:
            flush()
    flush()
    metadata = {"feature_columns": features, "responders": RESPONDERS, "shards": shards}
    (cache_dir / "cache.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_array(cache_dir: Path, shard_id: int, suffix: str, mmap=True):
    return np.load(cache_dir / f"shard_{shard_id:05d}_{suffix}.npy", mmap_mode="r" if mmap else None)


def all_times(cache_dir: Path, metadata: dict) -> np.ndarray:
    values = []
    for shard in metadata["shards"]:
        values.append(np.unique(load_array(cache_dir, shard["id"], "time")))
    return np.unique(np.concatenate(values))


def segments_for_range(cache_dir: Path, metadata: dict, lower: int | None, upper: int | None):
    segments = []
    for shard in metadata["shards"]:
        times = load_array(cache_dir, shard["id"], "time")
        start = 0 if lower is None else int(np.searchsorted(times, lower, side="left"))
        end = len(times) if upper is None else int(np.searchsorted(times, upper, side="left"))
        if end > start:
            segments.append((int(shard["id"]), start, end))
    return segments


class ShardSequence(lgb.Sequence):
    """LightGBM Sequence-compatible view over disk-backed NumPy shards."""

    batch_size = 8192

    def __init__(self, cache_dir: Path, segments, extra: np.memmap | None = None):
        self.cache_dir = cache_dir
        self.segments = list(segments)
        self.extra = extra
        self.lengths = [end - start for _, start, end in self.segments]
        self.offsets = np.cumsum([0, *self.lengths]).tolist()

    def __len__(self):
        return self.offsets[-1]

    def _rows(self, start: int, stop: int):
        parts = []
        while start < stop:
            segment_index = bisect_right(self.offsets, start) - 1
            shard_id, shard_start, _ = self.segments[segment_index]
            local = start - self.offsets[segment_index]
            count = min(stop - start, self.lengths[segment_index] - local)
            # LightGBM's Sequence sampling path requires rows to be float64
            # ("double"), even when the disk-backed cache is float32.
            # Convert only the requested batch so the cache remains compact.
            base = np.asarray(
                load_array(self.cache_dir, shard_id, "x")[
                    shard_start + local:shard_start + local + count
                ],
                dtype=np.float64,
            )
            if self.extra is not None:
                base = np.hstack(
                    [
                        base,
                        np.asarray(
                            self.extra[start:start + count], dtype=np.float64
                        ),
                    ]
                )
            parts.append(base)
            start += count
        return parts[0] if len(parts) == 1 else np.vstack(parts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            data = self._rows(start, stop)
            return data if step == 1 else data[::step]
        if isinstance(index, (list, np.ndarray)):
            return np.vstack([self[int(item)] for item in index])
        value = int(index)
        if value < 0:
            value += len(self)
        return self._rows(value, value + 1)[0]


def vector_for_segments(cache_dir: Path, segments, suffix: str, column: int | None = None):
    parts = []
    for shard_id, start, end in segments:
        values = np.asarray(load_array(cache_dir, shard_id, suffix)[start:end])
        parts.append(values if column is None else values[:, column])
    return np.concatenate(parts)


def predict_sequence(model, sequence: ShardSequence, label: str = "prediction") -> np.ndarray:
    output = np.empty(len(sequence), dtype=np.float32)
    total_batches = max(1, (len(sequence) + sequence.batch_size - 1) // sequence.batch_size)
    report_every = max(1, total_batches // 20)
    for batch_index, start in enumerate(range(0, len(sequence), sequence.batch_size), start=1):
        stop = min(start + sequence.batch_size, len(sequence))
        output[start:stop] = model.predict(sequence[start:stop])
        if batch_index == 1 or batch_index == total_batches or batch_index % report_every == 0:
            progress(
                f"{label}: {stop:,}/{len(sequence):,} rows "
                f"({100.0 * stop / max(len(sequence), 1):.1f}%)"
            )
    return output


def train_lgb(sequence, label, weight, valid_sequence, valid_label, valid_weight, rounds, args):
    params = {
        "objective": "regression", "metric": "l2", "learning_rate": 0.03,
        "num_leaves": 64, "min_data_in_leaf": 500, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "num_threads": args.threads, "seed": args.seed, "verbosity": -1,
    }
    train_set = lgb.Dataset(sequence, label=label, weight=weight, free_raw_data=False)
    valid_set = lgb.Dataset(valid_sequence, label=valid_label, weight=valid_weight, reference=train_set, free_raw_data=False)
    return lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)])


def train_lgb_fixed(sequence, label, weight, rounds, args):
    params = {
        "objective": "regression", "metric": "l2", "learning_rate": 0.03,
        "num_leaves": 64, "min_data_in_leaf": 500, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "num_threads": args.threads, "seed": args.seed, "verbosity": -1,
    }
    train_set = lgb.Dataset(sequence, label=label, weight=weight, free_raw_data=False)
    return lgb.train(params, train_set, num_boost_round=max(1, int(rounds)),
                     callbacks=[lgb.log_evaluation(50)])


def weighted_r2(y, pred, weight):
    y, pred, weight = map(lambda value: np.asarray(value, dtype=np.float64), (y, pred, weight))
    denominator = np.sum(weight * y * y)
    return float(1.0 - np.sum(weight * (y - pred) ** 2) / denominator) if denominator > 0 else 0.0


def main():
    args = parse_args()
    data_root, work_dir, model_dir = Path(args.data_root), Path(args.work_dir), Path(args.model_dir)
    cache_dir = work_dir / "cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_files = [model_dir / f"{name}.txt" for name in RESPONDERS]
    final_files.extend([model_dir / "target_lightgbm.txt", model_dir / "metadata.json"])
    if args.skip_existing_models and all(path.exists() for path in final_files):
        progress("all final model files already exist; training skipped")
        print((model_dir / "metadata.json").read_text(encoding="utf-8"))
        return

    progress(
        f"starting training: responders={RESPONDERS}, "
        f"skip_existing_models={args.skip_existing_models}"
    )
    if args.rebuild_cache and cache_dir.exists():
        progress(f"removing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    if (cache_dir / "cache.json").exists():
        progress(f"loading existing cache metadata: {cache_dir / 'cache.json'}")
        metadata = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    else:
        progress("building disk-backed training cache")
        metadata = build_cache(data_root, cache_dir, args.shard_rows, args.batch_size)

    progress(f"scanning time_id from {len(metadata['shards'])} cache shards")
    times = all_times(cache_dir, metadata)
    valid_count = max(1, int(round(len(times) * args.valid_time_fraction)))
    valid_cutoff = int(times[-valid_count])
    train_times = times[times < valid_cutoff]
    warmup_index = max(1, int(len(train_times) * args.warmup_fraction))
    oof_boundaries = np.linspace(warmup_index, len(train_times), args.oof_folds + 1, dtype=int)
    progress(
        f"split ready: total_times={len(times):,}, train_times={len(train_times):,}, "
        f"valid_cutoff={valid_cutoff}, oof_folds={args.oof_folds}"
    )

    oof_segments = segments_for_range(cache_dir, metadata, int(train_times[warmup_index]), valid_cutoff)
    oof_rows = sum(end - start for _, start, end in oof_segments)
    oof_path = work_dir / "oof_responder_hat.dat"
    oof_hat = np.memmap(oof_path, dtype="float32", mode="w+", shape=(oof_rows, len(RESPONDERS)))
    oof_cursor = 0
    responder_best_iterations: dict[str, list[int]] = {name: [] for name in RESPONDERS}
    oof_model_dir = work_dir / "oof_models"
    oof_model_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.oof_folds):
        fold_start = int(train_times[oof_boundaries[fold]])
        fold_end = valid_cutoff if fold == args.oof_folds - 1 else int(train_times[oof_boundaries[fold + 1]])
        fit_segments = segments_for_range(cache_dir, metadata, None, fold_start)
        pred_segments = segments_for_range(cache_dir, metadata, fold_start, fold_end)
        fit_x, pred_x = ShardSequence(cache_dir, fit_segments), ShardSequence(cache_dir, pred_segments)
        fit_w = vector_for_segments(cache_dir, fit_segments, "weight")
        pred_w = vector_for_segments(cache_dir, pred_segments, "weight")
        responders_fit = vector_for_segments(cache_dir, fit_segments, "responder")
        responders_pred = vector_for_segments(cache_dir, pred_segments, "responder")
        progress(
            f"OOF fold {fold + 1}/{args.oof_folds}: train_rows={len(fit_x):,}, "
            f"predict_rows={len(pred_x):,}, time_id={fold_start}..{fold_end - 1}"
        )
        for column, name in enumerate(RESPONDERS):
            fold_model_path = oof_model_dir / f"fold_{fold:02d}_{name}.txt"
            if args.skip_existing_models and fold_model_path.exists():
                progress(f"loading existing OOF model: {fold_model_path.name}")
                model = lgb.Booster(model_file=str(fold_model_path))
            else:
                progress(
                    f"training OOF model {column + 1}/{len(RESPONDERS)}: "
                    f"fold={fold + 1}, responder={name}"
                )
                model = train_lgb(fit_x, responders_fit[:, column], fit_w, pred_x,
                                  responders_pred[:, column], pred_w, args.responder_rounds, args)
                model.save_model(str(fold_model_path))
                progress(f"saved OOF model: {fold_model_path.name}")
            oof_hat[oof_cursor:oof_cursor + len(pred_x), column] = predict_sequence(
                model, pred_x, f"OOF fold {fold + 1} {name}"
            )
            best_iteration = int(model.best_iteration)
            if best_iteration <= 0:
                best_iteration = int(model.current_iteration())
            if best_iteration <= 0:
                best_iteration = int(args.responder_rounds)
            responder_best_iterations[name].append(best_iteration)
        oof_cursor += len(pred_x)
        oof_hat.flush()
        progress(f"OOF fold {fold + 1}/{args.oof_folds} complete")

    valid_segments = segments_for_range(cache_dir, metadata, valid_cutoff, None)
    train_segments = segments_for_range(cache_dir, metadata, None, valid_cutoff)
    train_x, valid_x = ShardSequence(cache_dir, train_segments), ShardSequence(cache_dir, valid_segments)
    train_w = vector_for_segments(cache_dir, train_segments, "weight")
    valid_w = vector_for_segments(cache_dir, valid_segments, "weight")
    train_responders = vector_for_segments(cache_dir, train_segments, "responder")
    valid_hat = np.empty((len(valid_x), len(RESPONDERS)), dtype=np.float32)
    responder_files = {}
    for column, name in enumerate(RESPONDERS):
        final_rounds = int(np.median(responder_best_iterations[name]))
        filename = f"{name}.txt"
        model_path = model_dir / filename
        if args.skip_existing_models and model_path.exists():
            progress(f"loading existing final responder model: {filename}")
            model = lgb.Booster(model_file=str(model_path))
        else:
            progress(
                f"training final responder {column + 1}/{len(RESPONDERS)}: "
                f"{name}, rounds={final_rounds}, rows={len(train_x):,}"
            )
            model = train_lgb_fixed(
                train_x, train_responders[:, column], train_w, final_rounds, args
            )
            model.save_model(str(model_path))
            progress(f"saved final responder model: {filename}")
        valid_hat[:, column] = predict_sequence(model, valid_x, f"validation {name}")
        responder_files[name] = filename

    target_train_x = ShardSequence(cache_dir, oof_segments, oof_hat)
    target_valid_x = ShardSequence(cache_dir, valid_segments, valid_hat)
    y_train = vector_for_segments(cache_dir, oof_segments, "target")
    w_train = vector_for_segments(cache_dir, oof_segments, "weight")
    y_valid = vector_for_segments(cache_dir, valid_segments, "target")
    target_path = model_dir / "target_lightgbm.txt"
    if args.skip_existing_models and target_path.exists():
        progress("loading existing target model: target_lightgbm.txt")
        target_model = lgb.Booster(model_file=str(target_path))
    else:
        progress(
            f"training target model: train_rows={len(target_train_x):,}, "
            f"valid_rows={len(target_valid_x):,}"
        )
        target_model = train_lgb(target_train_x, y_train, w_train, target_valid_x,
                                 y_valid, valid_w, args.target_rounds, args)
        target_model.save_model(str(target_path))
        progress("saved target model: target_lightgbm.txt")
    valid_pred = predict_sequence(target_model, target_valid_x, "target validation")
    clip_min, clip_max = np.quantile(valid_pred[np.isfinite(valid_pred)], [0.001, 0.999])
    output = {
        "strategy": "responder_assisted_lgb_catboost_strategy",
        "feature_columns": metadata["feature_columns"], "responders": RESPONDERS,
        "responder_models": responder_files, "target_model": "target_lightgbm.txt",
        "valid_cutoff_time_id": valid_cutoff, "oof_folds": args.oof_folds,
        "warmup_fraction": args.warmup_fraction, "target_train_rows": len(target_train_x),
        "responder_best_iterations": {
            name: int(np.median(values)) for name, values in responder_best_iterations.items()
        },
        "valid_rows": len(target_valid_x), "valid_score": weighted_r2(y_valid, valid_pred, valid_w),
        "prediction_scale": 1.0, "clip_min": float(clip_min), "clip_max": float(clip_max),
    }
    (model_dir / "metadata.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    progress(f"training pipeline complete; valid_score={output['valid_score']:.8f}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
