from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from bisect import bisect_right
from pathlib import Path
from typing import Sequence as TypingSequence

import numpy as np
import lightgbm as lgb
import pandas as pd

from temporal_features import TemporalFeatureBuilder, temporal_column_names


RESPONDERS = ["responder_03", "responder_28", "responder_29", "responder_02"]
TEMPORAL_ENGINE_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-core responder-stacked LightGBM")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.25)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--temporal-feature-count", type=int, default=30)
    parser.add_argument(
        "--feature-importance",
        default="",
        help="Optional feature_importance.csv used to select temporal raw features.",
    )
    parser.add_argument(
        "--temporal-plan",
        default="",
        help="temporal_feature_plan.json produced by analyze_feature_temporal_types.py",
    )
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--training-data-mode",
        choices=["out-of-core", "in-memory"],
        default="out-of-core",
        help=(
            "out-of-core streams cached shards through lightgbm.Sequence; "
            "in-memory concatenates each train/validation split before fitting."
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--responder-rounds", type=int, default=500)
    parser.add_argument("--target-rounds", type=int, default=1200)
    parser.add_argument(
        "--ablation-mode",
        choices=["all", "A", "B", "C", "D"],
        default="all",
        help="A=raw, B=raw+temporal, C=raw+responder_hat, D=all features.",
    )
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


def file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_config_hash(plan_path: Path | None, importance_path: Path | None, count: int) -> str:
    if plan_path is not None and plan_path.exists():
        return file_sha256(plan_path)
    payload = f"fallback:{int(count)}:{file_sha256(importance_path)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def select_temporal_features(features: list[str], count: int, importance_path: Path | None):
    if count <= 0:
        return []
    if importance_path is not None and importance_path.exists():
        import pandas as pd

        importance = pd.read_csv(importance_path)
        selected = [str(value) for value in importance.get("feature", []) if str(value) in features]
        if selected:
            progress(f"selected temporal features from {importance_path}")
            return selected[:count]
    progress("feature importance unavailable; using the first raw feature columns")
    return features[:count]


def select_temporal_plan(features: list[str], count: int, importance_path: Path | None,
                         plan_path: Path | None):
    if plan_path is not None and plan_path.exists():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        raw_recipes = payload.get("recipes", {})
        recipes = {
            str(feature): [str(value) for value in transforms]
            for feature, transforms in raw_recipes.items()
            if str(feature) in features and transforms
        }
        if recipes:
            from temporal_features import TEMPORAL_SUFFIXES

            for feature, transforms in recipes.items():
                migrated = [
                    value for value in transforms
                    if value not in {"delta1", "xs_rank_delta1"}
                    and value in TEMPORAL_SUFFIXES
                ]
                additions = []
                if "lag5" in migrated or "ema20" in migrated:
                    additions.extend(["lag20", "ema60"])
                if "delta5" in migrated:
                    additions.append("delta20")
                if "rolling_std20" in migrated:
                    additions.append("rolling_std60")
                if "historical_zscore20" in migrated:
                    additions.append("historical_zscore60")
                if "minus_ema20" in migrated:
                    additions.append("minus_ema60")
                for value in additions:
                    if value not in migrated:
                        migrated.append(value)
                recipes[feature] = migrated
            progress(
                f"loaded temporal routing plan: {plan_path}; "
                f"features={len(recipes)}, derived={sum(map(len, recipes.values()))}"
            )
            return list(recipes), recipes
    selected = select_temporal_features(features, count, importance_path)
    from temporal_features import TEMPORAL_SUFFIXES

    return selected, {feature: list(TEMPORAL_SUFFIXES) for feature in selected}


def build_cache(data_root: Path, cache_dir: Path, shard_rows: int, batch_size: int,
                temporal_feature_count: int, importance_path: Path | None,
                plan_path: Path | None):
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
    temporal_features, temporal_recipes = select_temporal_plan(
        features, temporal_feature_count, importance_path, plan_path
    )
    temporal_indices = [features.index(name) for name in temporal_features]
    temporal_builder = TemporalFeatureBuilder(
        len(temporal_features), feature_names=temporal_features, recipes=temporal_recipes
    )
    progress(
        f"temporal features: {len(temporal_features)} routed raw columns, "
        f"{sum(map(len, temporal_recipes.values()))} derived columns"
    )
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
        asset_values = frame["asset_id"].to_numpy(dtype=np.int64)
        temporal = temporal_builder.transform(asset_values, raw[:, temporal_indices])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        asset = asset_values.astype(np.float32).reshape(-1, 1)
        x = np.hstack([raw, temporal, asset])
        time_id = frame["time_id"].to_numpy(dtype=np.int64)
        target = frame["target"].to_numpy(dtype=np.float32)
        weight = np.maximum(frame["weight"].to_numpy(dtype=np.float32), 0.0)
        responder = frame.loc[:, RESPONDERS].to_numpy(dtype=np.float32)
        buffers.append((x, time_id, target, weight, responder))
        buffered_rows += len(frame)
        if buffered_rows >= shard_rows:
            flush()
    flush()
    metadata = {
        "cache_schema_version": 4,
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_plan_hash": temporal_config_hash(
            plan_path, importance_path, temporal_feature_count
        ),
        "feature_columns": features,
        "temporal_features": temporal_features,
        "temporal_recipes": temporal_recipes,
        "temporal_feature_columns": temporal_column_names(temporal_features, temporal_recipes),
        "responders": RESPONDERS,
        "shards": shards,
    }
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

    def __init__(self, cache_dir: Path, segments, extra=None,
                 base_indices: np.ndarray | None = None):
        self.cache_dir = cache_dir
        self.segments = list(segments)
        self.extra = extra
        self.base_indices = base_indices
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
            if self.base_indices is not None:
                base = base[:, self.base_indices]
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


def matrix_for_segments(cache_dir: Path, segments, extra=None,
                        base_indices: np.ndarray | None = None) -> np.ndarray:
    """Materialize one split as a contiguous float32 matrix."""
    parts = []
    cursor = 0
    for shard_id, start, end in segments:
        base = np.asarray(load_array(cache_dir, shard_id, "x")[start:end], dtype=np.float32)
        if base_indices is not None:
            base = base[:, base_indices]
        if extra is not None:
            count = end - start
            base = np.hstack((base, np.asarray(extra[cursor:cursor + count], dtype=np.float32)))
            cursor += count
        parts.append(base)
    if not parts:
        raise ValueError("cannot materialize an empty set of cache segments")
    return np.ascontiguousarray(parts[0] if len(parts) == 1 else np.vstack(parts))


def training_matrix(args, cache_dir: Path, segments, extra=None,
                    base_indices: np.ndarray | None = None):
    if args.training_data_mode == "in-memory":
        matrix = matrix_for_segments(cache_dir, segments, extra, base_indices)
        progress(
            f"materialized matrix: rows={matrix.shape[0]:,}, features={matrix.shape[1]:,}, "
            f"memory={matrix.nbytes / 1024 ** 3:.2f} GiB"
        )
        return matrix
    return ShardSequence(cache_dir, segments, extra, base_indices)


def vector_for_segments(cache_dir: Path, segments, suffix: str, column: int | None = None):
    parts = []
    for shard_id, start, end in segments:
        values = np.asarray(load_array(cache_dir, shard_id, suffix)[start:end])
        parts.append(values if column is None else values[:, column])
    return np.concatenate(parts)


def predict_sequence(model, sequence, label: str = "prediction") -> np.ndarray:
    output = np.empty(len(sequence), dtype=np.float32)
    batch_size = getattr(sequence, "batch_size", 65_536)
    total_batches = max(1, (len(sequence) + batch_size - 1) // batch_size)
    report_every = max(1, total_batches // 20)
    for batch_index, start in enumerate(range(0, len(sequence), batch_size), start=1):
        stop = min(start + batch_size, len(sequence))
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
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weight) & (weight > 0)
    y, pred, weight = y[valid], pred[valid], weight[valid]
    denominator = np.sum(weight * y * y)
    return float(1.0 - np.sum(weight * (y - pred) ** 2) / denominator) if denominator > 0 else 0.0


def target_variant_sequences(args, cache_dir, train_segments, valid_segments,
                             oof_hat, valid_hat, raw_count, cached_base_count,
                             variant):
    raw_indices = np.asarray([*range(raw_count), cached_base_count - 1], dtype=np.int64)
    specs = {
        "A": (
            None, raw_indices,
        ),
        "B": (
            None, None,
        ),
        "C": (
            (oof_hat, valid_hat), raw_indices,
        ),
        "D": (
            (oof_hat, valid_hat), None,
        ),
    }
    extras, indices = specs[variant]
    train_extra, valid_extra = (None, None) if extras is None else extras
    return (
        training_matrix(args, cache_dir, train_segments, train_extra, indices),
        training_matrix(args, cache_dir, valid_segments, valid_extra, indices),
    )


def target_variant_names(metadata, variant):
    raw = [*metadata["feature_columns"], "asset_id"]
    temporal = [*metadata["feature_columns"], *metadata["temporal_feature_columns"], "asset_id"]
    responder_hat = [f"{name}_hat" for name in RESPONDERS]
    return {"A": raw, "B": temporal, "C": [*raw, *responder_hat], "D": [*temporal, *responder_hat]}[variant]


def main():
    args = parse_args()
    data_root, work_dir, model_dir = Path(args.data_root), Path(args.work_dir), Path(args.model_dir)
    cache_dir = work_dir / "cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_files = [model_dir / f"{name}.txt" for name in RESPONDERS]
    final_files.extend([model_dir / "target_lightgbm.txt", model_dir / "metadata.json"])
    existing_model_metadata = None
    if (model_dir / "metadata.json").exists():
        existing_model_metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
    requested_plan_path = Path(args.temporal_plan) if args.temporal_plan else (
        Path(__file__).resolve().parent / "analysis" / "temporal_feature_plan.json"
    )
    requested_importance_path = Path(args.feature_importance) if args.feature_importance else (
        Path(__file__).resolve().parent.parent / "lgb_catboost_strategy" / "model" / "feature_importance.csv"
    )
    requested_plan_hash = temporal_config_hash(
        requested_plan_path, requested_importance_path, args.temporal_feature_count
    )
    model_metadata_matches = bool(
        existing_model_metadata
        and existing_model_metadata.get("temporal_recipes")
        and int(existing_model_metadata.get("temporal_engine_version", 0))
        == TEMPORAL_ENGINE_VERSION
        and existing_model_metadata.get("temporal_plan_hash", "") == requested_plan_hash
    )
    if args.skip_existing_models and model_metadata_matches and all(path.exists() for path in final_files):
        progress("all compatible final model files already exist; training skipped")
        print((model_dir / "metadata.json").read_text(encoding="utf-8"))
        return

    progress(
        f"starting training: responders={RESPONDERS}, "
        f"training_data_mode={args.training_data_mode}, "
        f"skip_existing_models={args.skip_existing_models}"
    )
    if args.rebuild_cache and cache_dir.exists():
        progress(f"removing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        if (work_dir / "oof_models").exists():
            progress("removing OOF models because the cache is being rebuilt")
            shutil.rmtree(work_dir / "oof_models")
    importance_path = Path(args.feature_importance) if args.feature_importance else (
        Path(__file__).resolve().parent.parent / "lgb_catboost_strategy" / "model" / "feature_importance.csv"
    )
    plan_path = Path(args.temporal_plan) if args.temporal_plan else (
        Path(__file__).resolve().parent / "analysis" / "temporal_feature_plan.json"
    )
    cache_metadata_path = cache_dir / "cache.json"
    if cache_metadata_path.exists():
        existing_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        cache_matches = (
            int(existing_metadata.get("cache_schema_version", 0)) == 4
            and int(existing_metadata.get("temporal_engine_version", 0))
            == TEMPORAL_ENGINE_VERSION
            and bool(existing_metadata.get("temporal_recipes"))
            and existing_metadata.get("temporal_plan_hash", "")
            == temporal_config_hash(plan_path, importance_path, args.temporal_feature_count)
        )
        if not cache_matches:
            progress("existing cache has a different temporal feature schema; rebuilding it")
            shutil.rmtree(cache_dir)
            if (work_dir / "oof_models").exists():
                shutil.rmtree(work_dir / "oof_models")
    if cache_metadata_path.exists():
        progress(f"loading existing cache metadata: {cache_dir / 'cache.json'}")
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    else:
        progress("building disk-backed training cache")
        metadata = build_cache(
            data_root, cache_dir, args.shard_rows, args.batch_size,
            args.temporal_feature_count, importance_path, plan_path,
        )

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
    responder_diagnostics = {
        name: {"folds": [], "oof_squared_error": 0.0, "oof_zero_denominator": 0.0}
        for name in RESPONDERS
    }
    oof_model_dir = work_dir / "oof_models"
    oof_signature_payload = {
        "feature_columns": metadata["feature_columns"],
        "temporal_features": metadata["temporal_features"],
        "temporal_recipes": metadata["temporal_recipes"],
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_plan_hash": metadata.get("temporal_plan_hash", ""),
        "valid_cutoff": valid_cutoff,
        "oof_folds": args.oof_folds,
        "warmup_fraction": args.warmup_fraction,
    }
    oof_signature = hashlib.sha256(
        json.dumps(oof_signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    oof_config_path = oof_model_dir / "config.json"
    if oof_config_path.exists():
        old_config = json.loads(oof_config_path.read_text(encoding="utf-8"))
        if old_config.get("signature") != oof_signature:
            progress("OOF model schema changed; removing incompatible OOF models")
            shutil.rmtree(oof_model_dir)
    elif oof_model_dir.exists() and any(oof_model_dir.glob("*.txt")):
        progress("legacy OOF models have no schema signature; removing them")
        shutil.rmtree(oof_model_dir)
    oof_model_dir.mkdir(parents=True, exist_ok=True)
    oof_config_path.write_text(
        json.dumps({"signature": oof_signature, **oof_signature_payload}, indent=2),
        encoding="utf-8",
    )

    for fold in range(args.oof_folds):
        fold_start = int(train_times[oof_boundaries[fold]])
        fold_end = valid_cutoff if fold == args.oof_folds - 1 else int(train_times[oof_boundaries[fold + 1]])
        fit_segments = segments_for_range(cache_dir, metadata, None, fold_start)
        pred_segments = segments_for_range(cache_dir, metadata, fold_start, fold_end)
        fit_x = training_matrix(args, cache_dir, fit_segments)
        pred_x = training_matrix(args, cache_dir, pred_segments)
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
            current_hat = np.asarray(
                oof_hat[oof_cursor:oof_cursor + len(pred_x), column],
                dtype=np.float64,
            )
            current_true = np.asarray(responders_pred[:, column], dtype=np.float64)
            current_weight = np.asarray(pred_w, dtype=np.float64)
            fold_score = weighted_r2(current_true, current_hat, current_weight)
            responder_diagnostics[name]["folds"].append(
                {
                    "fold": fold,
                    "time_start": fold_start,
                    "time_end": fold_end,
                    "rows": len(pred_x),
                    "score": fold_score,
                }
            )
            diagnostic_valid = (
                np.isfinite(current_true) & np.isfinite(current_hat)
                & np.isfinite(current_weight) & (current_weight > 0)
            )
            responder_diagnostics[name]["oof_squared_error"] += float(
                np.sum(current_weight[diagnostic_valid] * (
                    current_true[diagnostic_valid] - current_hat[diagnostic_valid]
                ) ** 2)
            )
            responder_diagnostics[name]["oof_zero_denominator"] += float(
                np.sum(current_weight[diagnostic_valid] * current_true[diagnostic_valid] ** 2)
            )
            progress(f"OOF fold {fold + 1} {name}: zero-mean R2={fold_score:.8f}")
            best_iteration = int(model.best_iteration)
            if best_iteration <= 0:
                best_iteration = int(model.current_iteration())
            if best_iteration <= 0:
                best_iteration = int(args.responder_rounds)
            responder_best_iterations[name].append(best_iteration)
        oof_cursor += len(pred_x)
        oof_hat.flush()
        if args.training_data_mode == "in-memory":
            del fit_x, pred_x
            gc.collect()
        progress(f"OOF fold {fold + 1}/{args.oof_folds} complete")

    valid_segments = segments_for_range(cache_dir, metadata, valid_cutoff, None)
    train_segments = segments_for_range(cache_dir, metadata, None, valid_cutoff)
    train_x = training_matrix(args, cache_dir, train_segments)
    valid_x = training_matrix(args, cache_dir, valid_segments)
    train_w = vector_for_segments(cache_dir, train_segments, "weight")
    valid_w = vector_for_segments(cache_dir, valid_segments, "weight")
    train_responders = vector_for_segments(cache_dir, train_segments, "responder")
    valid_hat = np.empty((len(valid_x), len(RESPONDERS)), dtype=np.float32)
    responder_files = {}
    for column, name in enumerate(RESPONDERS):
        final_rounds = int(np.median(responder_best_iterations[name]))
        filename = f"{name}.txt"
        model_path = model_dir / filename
        if args.skip_existing_models and model_metadata_matches and model_path.exists():
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
        valid_true = vector_for_segments(cache_dir, valid_segments, "responder", column)
        responder_diagnostics[name]["valid_score"] = weighted_r2(
            valid_true, valid_hat[:, column], valid_w
        )
        denominator = responder_diagnostics[name].pop("oof_zero_denominator")
        squared_error = responder_diagnostics[name].pop("oof_squared_error")
        responder_diagnostics[name]["oof_score"] = (
            1.0 - squared_error / denominator if denominator > 0 else 0.0
        )
        progress(
            f"final {name}: OOF R2={responder_diagnostics[name]['oof_score']:.8f}, "
            f"validation R2={responder_diagnostics[name]['valid_score']:.8f}"
        )
        responder_files[name] = filename

    if args.training_data_mode == "in-memory":
        del train_x, valid_x, train_responders
        gc.collect()

    y_train = vector_for_segments(cache_dir, oof_segments, "target")
    w_train = vector_for_segments(cache_dir, oof_segments, "weight")
    y_valid = vector_for_segments(cache_dir, valid_segments, "target")
    cached_base_count = len(metadata["feature_columns"]) + len(metadata["temporal_feature_columns"]) + 1
    variants = ["A", "B", "C", "D"] if args.ablation_mode == "all" else [args.ablation_mode]
    ablation_scores = {}
    importance_frames = []
    target_models = {}
    valid_predictions = {}
    for variant in variants:
        target_train_x, target_valid_x = target_variant_sequences(
            args, cache_dir, oof_segments, valid_segments, oof_hat, valid_hat,
            len(metadata["feature_columns"]), cached_base_count, variant,
        )
        variant_path = model_dir / f"target_{variant}.txt"
        if args.skip_existing_models and model_metadata_matches and variant_path.exists():
            progress(f"loading existing target ablation model: {variant_path.name}")
            target_model = lgb.Booster(model_file=str(variant_path))
        else:
            progress(
                f"training target variant {variant}: features={target_train_x[0].shape[0]}, "
                f"train_rows={len(target_train_x):,}, valid_rows={len(target_valid_x):,}"
            )
            target_model = train_lgb(
                target_train_x, y_train, w_train, target_valid_x,
                y_valid, valid_w, args.target_rounds, args,
            )
            target_model.save_model(str(variant_path))
        prediction = predict_sequence(target_model, target_valid_x, f"target {variant} validation")
        score = weighted_r2(y_valid, prediction, valid_w)
        best_iteration = int(target_model.best_iteration)
        if best_iteration <= 0:
            best_iteration = int(target_model.current_iteration())
        ablation_scores[variant] = {
            "score": score,
            "features": target_model.num_feature(),
            "best_iteration": best_iteration,
            "file": variant_path.name,
        }
        names = target_variant_names(metadata, variant)
        importance_frames.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "feature": names,
                    "importance_gain": target_model.feature_importance(importance_type="gain"),
                    "importance_split": target_model.feature_importance(importance_type="split"),
                }
            )
        )
        target_models[variant] = target_model
        valid_predictions[variant] = prediction
        progress(f"target variant {variant}: zero-mean R2={score:.8f}")
        if args.training_data_mode == "in-memory":
            del target_train_x, target_valid_x
            gc.collect()

    best_variant = max(ablation_scores, key=lambda name: ablation_scores[name]["score"])
    target_model = target_models[best_variant]
    valid_pred = valid_predictions[best_variant]
    target_path = model_dir / "target_lightgbm.txt"
    target_model.save_model(str(target_path))
    pd.concat(importance_frames, ignore_index=True).to_csv(
        model_dir / "target_feature_importance.csv", index=False
    )
    (model_dir / "ablation_report.json").write_text(
        json.dumps(
            {"best_variant": best_variant, "scores": ablation_scores,
             "responder_diagnostics": responder_diagnostics},
            indent=2,
        ),
        encoding="utf-8",
    )
    clip_min, clip_max = np.quantile(valid_pred[np.isfinite(valid_pred)], [0.001, 0.999])
    output = {
        "strategy": "responder_assisted_lgb_catboost_strategy",
        "feature_columns": metadata["feature_columns"], "responders": RESPONDERS,
        "temporal_features": metadata["temporal_features"],
        "temporal_recipes": metadata["temporal_recipes"],
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_feature_columns": metadata["temporal_feature_columns"],
        "temporal_plan_hash": metadata.get("temporal_plan_hash", ""),
        "responder_models": responder_files, "target_model": "target_lightgbm.txt",
        "target_variant": best_variant,
        "ablation_scores": ablation_scores,
        "responder_diagnostics": responder_diagnostics,
        "target_feature_importance": "target_feature_importance.csv",
        "ablation_report": "ablation_report.json",
        "valid_cutoff_time_id": valid_cutoff, "oof_folds": args.oof_folds,
        "warmup_fraction": args.warmup_fraction, "target_train_rows": len(y_train),
        "responder_best_iterations": {
            name: int(np.median(values)) for name, values in responder_best_iterations.items()
        },
        "valid_rows": len(y_valid), "valid_score": weighted_r2(y_valid, valid_pred, valid_w),
        "prediction_scale": 1.0, "clip_min": float(clip_min), "clip_max": float(clip_max),
    }
    (model_dir / "metadata.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    progress(f"training pipeline complete; valid_score={output['valid_score']:.8f}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
