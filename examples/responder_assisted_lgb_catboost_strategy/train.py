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


DEFAULT_RESPONDERS = [
    "responder_03", "responder_28", "responder_29", "responder_02"
]
TIER_RESPONDERS = [
    "responder_14", "responder_09", "responder_08", "responder_10",
    "responder_22", "responder_23", "responder_21", "responder_42",
    "responder_07", "responder_15", "responder_41", "responder_24",
]
TEMPORAL_ENGINE_VERSION = 3
LGB_PROFILE_VERSION = "low_risk_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-core responder-stacked LightGBM")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--valid-time-fraction", type=float, default=0.15)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--purge-steps", type=int, default=30)
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
        "--responders",
        default="",
        help=(
            "Comma-separated responder columns. The single-responder suite "
            "defaults to the first/second-tier screening candidates."
        ),
    )
    parser.add_argument(
        "--ablation-mode",
        choices=["all", "A", "B", "C", "D"],
        default="all",
        help="Run one legacy variant, or let --experiment-suite select a suite.",
    )
    parser.add_argument(
        "--experiment-suite",
        choices=[
            "legacy", "next-step", "responder", "c4-mechanism",
            "single-responder", "all",
        ],
        default="next-step",
        help=(
            "legacy runs A/B/C/D; next-step runs the compact target suite; "
            "responder isolates the default four; c4-mechanism adds leave-one-"
            "out and within-time shuffled controls; single-responder runs one "
            "full OOF target experiment per screened candidate."
        ),
    )
    parser.add_argument(
        "--target-experiments",
        default="",
        help=(
            "Optional comma-separated experiment names overriding the suite, "
            "for example C2,T60,T20_60."
        ),
    )
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--target-seeds", default="2026,2027,2028",
        help="Comma-separated seeds for the final full-data target ensemble.",
    )
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


def progress_bar(label: str, current: int, total: int, detail: str = "") -> None:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - START_TIME
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{label:<22}] [{bar}] {100.0 * current / total:6.2f}% "
        f"({current:,}/{total:,}) {elapsed:9.1f}s{suffix}",
        flush=True,
    )


def lightgbm_progress(label: str, total_rounds: int):
    interval = max(1, int(total_rounds) // 20)

    def callback(environment):
        current = int(environment.iteration) + 1
        if current == 1 or current == total_rounds or current % interval == 0:
            progress_bar(label, current, total_rounds, "boosting rounds")

    callback.order = 20
    callback.before_iteration = False
    return callback


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
    from temporal_features import DEFAULT_TEMPORAL_SUFFIXES

    return selected, {
        feature: list(DEFAULT_TEMPORAL_SUFFIXES) for feature in selected
    }


def build_cache(data_root: Path, cache_dir: Path, shard_rows: int, batch_size: int,
                temporal_feature_count: int, importance_path: Path | None,
                plan_path: Path | None, responders: list[str]):
    import pyarrow.parquet as pq

    files = manifest_files(data_root)
    if not files:
        raise ValueError("no training parquet files")
    columns = list(pq.read_schema(files[0]).names)
    expected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    features = [name for name in columns if name.startswith("feature_")]
    missing = [
        name for name in ["target", "weight", *responders]
        if name not in columns
    ]
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
    read_columns = [
        "time_id", "asset_id", "weight", "target", *responders, *features
    ]
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
        progress_bar(
            "cache preprocessing", total_rows, expected_rows,
            f"shards={len(shards)}",
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
        responder = frame.loc[:, responders].to_numpy(dtype=np.float32)
        buffers.append((x, time_id, target, weight, responder))
        buffered_rows += len(frame)
        if buffered_rows >= shard_rows:
            flush()
    flush()
    metadata = {
        "cache_schema_version": 5,
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_plan_hash": temporal_config_hash(
            plan_path, importance_path, temporal_feature_count
        ),
        "feature_columns": features,
        "temporal_features": temporal_features,
        "temporal_recipes": temporal_recipes,
        "temporal_feature_columns": temporal_column_names(temporal_features, temporal_recipes),
        "responders": responders,
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
        if value < 0 or value >= len(self):
            raise IndexError(value)
        return self._rows(value, value + 1)[0]


class ConcatenatedSequence(lgb.Sequence):
    """Present several matrices or Sequences as one LightGBM Sequence."""

    batch_size = 8192

    def __init__(self, *sources):
        self.sources = [source for source in sources if len(source)]
        if not self.sources:
            raise ValueError("cannot concatenate empty training sources")
        self.lengths = [len(source) for source in self.sources]
        self.offsets = np.cumsum([0, *self.lengths]).tolist()

    def __len__(self):
        return self.offsets[-1]

    def _rows(self, start: int, stop: int):
        if start >= stop:
            feature_count = int(self.sources[0][0:1].shape[1])
            return np.empty((0, feature_count), dtype=np.float64)
        parts = []
        while start < stop:
            source_index = bisect_right(self.offsets, start) - 1
            local = start - self.offsets[source_index]
            count = min(stop - start, self.lengths[source_index] - local)
            parts.append(np.asarray(
                self.sources[source_index][local:local + count],
                dtype=np.float64,
            ))
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
        if value < 0 or value >= len(self):
            raise IndexError(value)
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


def concatenate_training_matrices(args, *sources):
    if args.training_data_mode == "in-memory":
        matrix = np.ascontiguousarray(np.vstack(sources), dtype=np.float32)
        progress(
            f"materialized refit matrix: rows={matrix.shape[0]:,}, "
            f"features={matrix.shape[1]:,}, "
            f"memory={matrix.nbytes / 1024 ** 3:.2f} GiB"
        )
        return matrix
    return ConcatenatedSequence(*sources)


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
            progress_bar(label, stop, len(sequence), "prediction rows")
    return output


def low_risk_lgb_params(args, seed=None):
    seed = int(args.seed if seed is None else seed)
    return {
        "objective": "regression", "metric": "l2", "learning_rate": 0.03,
        "boosting_type": "gbdt", "data_sample_strategy": "bagging",
        "num_leaves": 63, "max_depth": 12, "min_data_in_leaf": 3500,
        "feature_fraction": 0.9, "feature_fraction_bynode": 0.9,
        "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l1": 1.0, "lambda_l2": 15.0, "path_smooth": 100.0,
        "max_bin": 255, "deterministic": True, "force_col_wise": True,
        "num_threads": args.threads, "seed": seed,
        "bagging_seed": seed, "feature_fraction_seed": seed,
        "extra_seed": seed, "data_random_seed": int(args.seed),
        "verbosity": -1,
    }


def train_lgb(sequence, label, weight, valid_sequence, valid_label, valid_weight,
              rounds, args, progress_label="LightGBM", categorical_feature=None):
    params = low_risk_lgb_params(args)
    categorical_feature = categorical_feature or "auto"
    train_set = lgb.Dataset(
        sequence, label=label, weight=weight,
        categorical_feature=categorical_feature, free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_sequence, label=valid_label, weight=valid_weight,
        categorical_feature=categorical_feature,
        reference=train_set, free_raw_data=False,
    )
    return lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
                     callbacks=[
                         lgb.early_stopping(args.early_stopping),
                         lightgbm_progress(progress_label, rounds),
                         lgb.log_evaluation(50),
                     ])


def train_lgb_fixed(sequence, label, weight, rounds, args,
                    progress_label="LightGBM", categorical_feature=None,
                    seed=None):
    params = low_risk_lgb_params(args, seed=seed)
    train_set = lgb.Dataset(
        sequence, label=label, weight=weight,
        categorical_feature=categorical_feature or "auto",
        free_raw_data=False,
    )
    return lgb.train(params, train_set, num_boost_round=max(1, int(rounds)),
                     callbacks=[
                         lightgbm_progress(progress_label, max(1, int(rounds))),
                         lgb.log_evaluation(50),
                     ])


def weighted_r2(y, pred, weight):
    y, pred, weight = map(lambda value: np.asarray(value, dtype=np.float64), (y, pred, weight))
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weight) & (weight > 0)
    y, pred, weight = y[valid], pred[valid], weight[valid]
    denominator = np.sum(weight * y * y)
    return float(1.0 - np.sum(weight * (y - pred) ** 2) / denominator) if denominator > 0 else 0.0


def segmented_validation_scores(time_ids, y, pred, weight, parts: int = 4):
    """Score contiguous validation-time blocks and report temporal stability."""
    time_ids = np.asarray(time_ids)
    unique_times = np.unique(time_ids)
    blocks = []
    for index, block_times in enumerate(np.array_split(unique_times, parts), start=1):
        if len(block_times) == 0:
            continue
        start = int(np.searchsorted(time_ids, block_times[0], side="left"))
        stop = int(np.searchsorted(time_ids, block_times[-1], side="right"))
        blocks.append(
            {
                "part": index,
                "time_start": int(block_times[0]),
                "time_end": int(block_times[-1]),
                "rows": stop - start,
                "score": weighted_r2(
                    y[start:stop], pred[start:stop], weight[start:stop]
                ),
            }
        )
    values = np.asarray([item["score"] for item in blocks], dtype=np.float64)
    return {
        "parts": blocks,
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "std": float(np.std(values)) if len(values) else 0.0,
        "minimum": float(np.min(values)) if len(values) else 0.0,
        "positive_parts": int(np.sum(values > 0.0)),
    }


def c4_mechanism_summary(scores):
    required = {
        "A", "C4", "R02", "R03", "R28", "R29",
        "C4_NO_R02", "C4_NO_R03", "C4_NO_R28", "C4_NO_R29",
        "C4_SHUFFLED",
    }
    if not required.issubset(scores):
        return None

    def paired_delta(left, right):
        left_parts = left["segmented_validation"]["parts"]
        right_parts = right["segmented_validation"]["parts"]
        return {
            "overall": float(left["score"] - right["score"]),
            "segments": [
                float(a["score"] - b["score"])
                for a, b in zip(left_parts, right_parts)
            ],
        }

    c4 = scores["C4"]
    baseline = scores["A"]
    shuffled = scores["C4_SHUFFLED"]
    member_suffixes = ("R02", "R03", "R28", "R29")
    return {
        "interpretation": {
            "c4_vs_baseline": (
                "Total value of the four responder_hat features."
            ),
            "c4_vs_shuffled": (
                "Sample-level information beyond within-time distributions."
            ),
            "leave_one_out": (
                "Positive means C4 became worse after removing that responder."
            ),
            "single_responder": (
                "Independent value of each responder_hat versus baseline."
            ),
        },
        "c4_vs_baseline": paired_delta(c4, baseline),
        "c4_vs_shuffled": paired_delta(c4, shuffled),
        "shuffled_vs_baseline": paired_delta(shuffled, baseline),
        "leave_one_out": {
            suffix: paired_delta(c4, scores[f"C4_NO_{suffix}"])
            for suffix in member_suffixes
        },
        "single_responder": {
            suffix: paired_delta(scores[suffix], baseline)
            for suffix in member_suffixes
        },
    }


TARGET_EXPERIMENTS = {
    "A": {"temporal_groups": (), "responders": ()},
    "B": {"temporal_groups": None, "responders": ()},
    "C": {"temporal_groups": (), "responders": tuple(DEFAULT_RESPONDERS)},
    "D": {"temporal_groups": None, "responders": tuple(DEFAULT_RESPONDERS)},
    "C4": {"temporal_groups": (), "responders": tuple(DEFAULT_RESPONDERS)},
    "C2": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_02"),
    },
    "R02": {
        "temporal_groups": (),
        "responders": ("responder_02",),
    },
    "R03": {
        "temporal_groups": (),
        "responders": ("responder_03",),
    },
    "R28": {
        "temporal_groups": (),
        "responders": ("responder_28",),
    },
    "R29": {
        "temporal_groups": (),
        "responders": ("responder_29",),
    },
    "C4_NO_R02": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_28", "responder_29"),
    },
    "C4_NO_R03": {
        "temporal_groups": (),
        "responders": ("responder_28", "responder_29", "responder_02"),
    },
    "C4_NO_R28": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_29", "responder_02"),
    },
    "C4_NO_R29": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_28", "responder_02"),
    },
    "C4_SHUFFLED": {
        "temporal_groups": (),
        "responders": tuple(DEFAULT_RESPONDERS),
        "shuffle_within_time": True,
        "deployable": False,
    },
    "C2_R28": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_02", "responder_28"),
    },
    "C2_R29": {
        "temporal_groups": (),
        "responders": ("responder_03", "responder_02", "responder_29"),
    },
    "T60": {
        "temporal_groups": ("rolling_std60", "minus_ema60"),
        "responders": ("responder_03", "responder_02"),
    },
    "T20_60": {
        "temporal_groups": (
            "rolling_std20", "rolling_std60",
            "minus_ema20", "minus_ema60",
        ),
        "responders": ("responder_03", "responder_02"),
    },
    "TZ": {
        "temporal_groups": (
            "rolling_std20", "rolling_std60",
            "minus_ema20", "minus_ema60",
            "historical_zscore20", "historical_zscore60",
        ),
        "responders": ("responder_03", "responder_02"),
    },
}


def configured_responders(args) -> list[str]:
    if args.responders:
        values = [
            value.strip() for value in args.responders.split(",")
            if value.strip()
        ]
    elif args.experiment_suite == "single-responder":
        values = list(TIER_RESPONDERS)
    else:
        values = list(DEFAULT_RESPONDERS)
    if not values:
        raise ValueError("at least one responder must be configured")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate responder names: {values}")
    invalid = [value for value in values if not value.startswith("responder_")]
    if invalid:
        raise ValueError(f"invalid responder names: {invalid}")
    return values


def configured_target_seeds(args) -> list[int]:
    values = [
        int(value.strip()) for value in args.target_seeds.split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError("--target-seeds must contain at least one integer")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate target seeds: {values}")
    return values


def selected_experiments(args, responders: list[str]) -> list[str]:
    if args.ablation_mode != "all":
        return [args.ablation_mode]
    if args.target_experiments:
        names = [
            value.strip() for value in args.target_experiments.split(",")
            if value.strip()
        ]
    elif args.experiment_suite == "legacy":
        names = ["A", "B", "C", "D"]
    elif args.experiment_suite == "next-step":
        names = ["A", "C4", "C2", "T60", "T20_60", "TZ"]
    elif args.experiment_suite == "responder":
        names = [
            "A", "R02", "R03", "R28", "R29",
            "C2", "C2_R28", "C2_R29", "C4",
        ]
    elif args.experiment_suite == "c4-mechanism":
        names = [
            "A", "C4", "R02", "R03", "R28", "R29",
            "C4_NO_R02", "C4_NO_R03", "C4_NO_R28", "C4_NO_R29",
            "C4_SHUFFLED",
        ]
    elif args.experiment_suite == "single-responder":
        names = ["A", *(f"S_{name}" for name in responders)]
    else:
        names = [
            "A", "B", "C", "D",
            "R02", "R03", "R28", "R29",
            "C2", "C2_R28", "C2_R29", "C4",
            "T60", "T20_60", "TZ",
        ]
    unknown = [
        name for name in names
        if name not in TARGET_EXPERIMENTS and not name.startswith("S_responder_")
    ]
    if unknown:
        raise ValueError(
            f"unknown target experiments: {unknown}; "
            f"available={list(TARGET_EXPERIMENTS)}"
        )
    return list(dict.fromkeys(names))


def target_experiment_spec(metadata: dict, name: str,
                           active_responders: list[str]) -> dict:
    if name.startswith("S_responder_"):
        definition = {
            "temporal_groups": (),
            "responders": (name.removeprefix("S_"),),
        }
    else:
        definition = TARGET_EXPERIMENTS[name]
    raw_count = len(metadata["feature_columns"])
    temporal_columns = list(metadata["temporal_feature_columns"])
    base_names = [
        *metadata["feature_columns"], *temporal_columns, "asset_id"
    ]
    temporal_groups = definition["temporal_groups"]
    if temporal_groups is None:
        temporal_offsets = list(range(len(temporal_columns)))
    else:
        temporal_offsets = [
            index
            for index, column in enumerate(temporal_columns)
            if any(
                column.startswith(f"ts_{group}_")
                for group in temporal_groups
            )
        ]
    base_indices = [
        *range(raw_count),
        *(raw_count + index for index in temporal_offsets),
        len(base_names) - 1,
    ]
    responder_names = list(definition["responders"])
    missing = [
        value for value in responder_names if value not in active_responders
    ]
    if missing:
        raise ValueError(
            f"experiment {name} needs responders not present in cache: {missing}"
        )
    responder_indices = [
        active_responders.index(value) for value in responder_names
    ]
    feature_names = [
        *(base_names[index] for index in base_indices),
        *(f"{value}_hat" for value in responder_names),
    ]
    return {
        "name": name,
        "temporal_groups": (
            ["all"] if temporal_groups is None else list(temporal_groups)
        ),
        "base_indices": base_indices,
        "responders": responder_names,
        "responder_indices": responder_indices,
        "feature_names": feature_names,
        "shuffle_within_time": bool(
            definition.get("shuffle_within_time", False)
        ),
        "deployable": bool(definition.get("deployable", True)),
    }


def shuffled_within_time(values, time_ids, seed):
    """Break row correspondence while preserving each time block's values."""
    shuffled = np.asarray(values, dtype=np.float32).copy()
    time_ids = np.asarray(time_ids)
    rng = np.random.default_rng(seed)
    start = 0
    while start < len(time_ids):
        stop = int(np.searchsorted(
            time_ids, time_ids[start], side="right"
        ))
        for column in range(shuffled.shape[1]):
            shuffled[start:stop, column] = shuffled[
                start:stop, column
            ][rng.permutation(stop - start)]
        start = stop
    return shuffled


def target_experiment_matrices(args, cache_dir, train_segments, valid_segments,
                               oof_hat, valid_hat, spec,
                               train_times=None, valid_times=None):
    responder_indices = spec["responder_indices"]
    train_extra = (
        np.asarray(oof_hat[:, responder_indices], dtype=np.float32)
        if responder_indices else None
    )
    valid_extra = (
        np.asarray(valid_hat[:, responder_indices], dtype=np.float32)
        if responder_indices else None
    )
    if spec["shuffle_within_time"]:
        if train_times is None or valid_times is None:
            raise ValueError("shuffled experiment requires train and valid times")
        name_seed = int(hashlib.sha256(
            spec["name"].encode("utf-8")
        ).hexdigest()[:8], 16)
        train_extra = shuffled_within_time(
            train_extra, train_times, args.seed ^ name_seed
        )
        valid_extra = shuffled_within_time(
            valid_extra, valid_times, args.seed ^ name_seed ^ 0x5A5A5A5A
        )
    base_indices = np.asarray(spec["base_indices"], dtype=np.int64)
    return (
        training_matrix(
            args, cache_dir, train_segments, train_extra, base_indices
        ),
        training_matrix(
            args, cache_dir, valid_segments, valid_extra, base_indices
        ),
    )


def main():
    args = parse_args()
    responders = configured_responders(args)
    target_seeds = configured_target_seeds(args)
    requested_target_experiments = selected_experiments(args, responders)
    data_root, work_dir, model_dir = Path(args.data_root), Path(args.work_dir), Path(args.model_dir)
    cache_dir = work_dir / "cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_files = [model_dir / f"{name}.txt" for name in responders]
    final_files.extend([
        model_dir / "target_lightgbm.txt",
        *(model_dir / f"target_final_seed{seed}.txt" for seed in target_seeds),
        model_dir / "metadata.json",
        model_dir / "validation_predictions.npz",
    ])
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
    requested_training_config = {
        "valid_time_fraction": args.valid_time_fraction,
        "oof_folds": args.oof_folds,
        "warmup_fraction": args.warmup_fraction,
        "purge_steps": args.purge_steps,
        "responder_rounds": args.responder_rounds,
        "target_rounds": args.target_rounds,
        "early_stopping": args.early_stopping,
        "seed": args.seed,
    }
    model_metadata_matches = bool(
        existing_model_metadata
        and existing_model_metadata.get("temporal_recipes")
        and int(existing_model_metadata.get("temporal_engine_version", 0))
        == TEMPORAL_ENGINE_VERSION
        and existing_model_metadata.get("temporal_plan_hash", "") == requested_plan_hash
        and existing_model_metadata.get("responders") == responders
        and existing_model_metadata.get("lgb_profile_version")
        == LGB_PROFILE_VERSION
        and existing_model_metadata.get("target_seeds") == target_seeds
        and int(existing_model_metadata.get("purge_steps", -1))
        == args.purge_steps
        and existing_model_metadata.get("training_config")
        == requested_training_config
    )
    target_suite_matches = bool(
        existing_model_metadata
        and existing_model_metadata.get("trained_target_experiments")
        == requested_target_experiments
    )
    if (
        args.skip_existing_models
        and model_metadata_matches
        and target_suite_matches
        and all(path.exists() for path in final_files)
    ):
        progress("all compatible final model files already exist; training skipped")
        print((model_dir / "metadata.json").read_text(encoding="utf-8"))
        return

    progress(
        f"starting training: responders={responders}, "
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
            int(existing_metadata.get("cache_schema_version", 0)) == 5
            and int(existing_metadata.get("temporal_engine_version", 0))
            == TEMPORAL_ENGINE_VERSION
            and bool(existing_metadata.get("temporal_recipes"))
            and existing_metadata.get("temporal_plan_hash", "")
            == temporal_config_hash(plan_path, importance_path, args.temporal_feature_count)
            and existing_metadata.get("responders") == responders
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
            responders,
        )

    progress(f"scanning time_id from {len(metadata['shards'])} cache shards")
    times = all_times(cache_dir, metadata)
    valid_count = max(1, int(round(len(times) * args.valid_time_fraction)))
    valid_cutoff = int(times[-valid_count])
    train_times = times[times < valid_cutoff]
    if args.purge_steps < 0 or args.purge_steps >= len(train_times):
        raise ValueError(
            f"purge_steps must be in [0, {len(train_times) - 1}]"
        )
    warmup_index = max(1, int(len(train_times) * args.warmup_fraction))
    oof_boundaries = np.linspace(warmup_index, len(train_times), args.oof_folds + 1, dtype=int)
    progress(
        f"split ready: total_times={len(times):,}, train_times={len(train_times):,}, "
        f"valid_cutoff={valid_cutoff}, oof_folds={args.oof_folds}"
    )

    oof_segments = segments_for_range(cache_dir, metadata, int(train_times[warmup_index]), valid_cutoff)
    oof_rows = sum(end - start for _, start, end in oof_segments)
    oof_path = work_dir / "oof_responder_hat.dat"
    oof_hat = np.memmap(
        oof_path, dtype="float32", mode="w+",
        shape=(oof_rows, len(responders)),
    )
    oof_cursor = 0
    responder_best_iterations: dict[str, list[int]] = {
        name: [] for name in responders
    }
    responder_diagnostics = {
        name: {"folds": [], "oof_squared_error": 0.0, "oof_zero_denominator": 0.0}
        for name in responders
    }
    oof_model_dir = work_dir / "oof_models"
    oof_signature_payload = {
        "feature_columns": metadata["feature_columns"],
        "temporal_features": metadata["temporal_features"],
        "temporal_recipes": metadata["temporal_recipes"],
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_plan_hash": metadata.get("temporal_plan_hash", ""),
        "responders": responders,
        "valid_cutoff": valid_cutoff,
        "oof_folds": args.oof_folds,
        "warmup_fraction": args.warmup_fraction,
        "purge_steps": args.purge_steps,
        "lgb_profile_version": LGB_PROFILE_VERSION,
        "seed": args.seed,
        "responder_rounds": args.responder_rounds,
        "early_stopping": args.early_stopping,
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
        fit_stop_index = max(1, int(oof_boundaries[fold]) - args.purge_steps)
        fit_end = int(train_times[fit_stop_index])
        fit_segments = segments_for_range(cache_dir, metadata, None, fit_end)
        pred_segments = segments_for_range(cache_dir, metadata, fold_start, fold_end)
        fit_x = training_matrix(args, cache_dir, fit_segments)
        pred_x = training_matrix(args, cache_dir, pred_segments)
        fit_w = vector_for_segments(cache_dir, fit_segments, "weight")
        pred_w = vector_for_segments(cache_dir, pred_segments, "weight")
        responders_fit = vector_for_segments(cache_dir, fit_segments, "responder")
        responders_pred = vector_for_segments(cache_dir, pred_segments, "responder")
        progress(
            f"OOF fold {fold + 1}/{args.oof_folds}: train_rows={len(fit_x):,}, "
            f"predict_rows={len(pred_x):,}, purge_steps={args.purge_steps}, "
            f"time_id={fold_start}..{fold_end - 1}"
        )
        for column, name in enumerate(responders):
            fold_model_path = oof_model_dir / f"fold_{fold:02d}_{name}.txt"
            if args.skip_existing_models and fold_model_path.exists():
                progress(f"loading existing OOF model: {fold_model_path.name}")
                model = lgb.Booster(model_file=str(fold_model_path))
            else:
                progress(
                    f"training OOF model {column + 1}/{len(responders)}: "
                    f"fold={fold + 1}, responder={name}"
                )
                model = train_lgb(fit_x, responders_fit[:, column], fit_w, pred_x,
                                  responders_pred[:, column], pred_w,
                                  args.responder_rounds, args,
                                  f"OOF {fold + 1} {name}",
                                  categorical_feature=[len(metadata["feature_columns"])
                                      + len(metadata["temporal_feature_columns"])])
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
            progress_bar(
                "OOF responder models",
                fold * len(responders) + column + 1,
                args.oof_folds * len(responders),
                f"fold={fold + 1}, responder={name}",
            )
        oof_cursor += len(pred_x)
        oof_hat.flush()
        if args.training_data_mode == "in-memory":
            del fit_x, pred_x
            gc.collect()
        progress(f"OOF fold {fold + 1}/{args.oof_folds} complete")

    valid_segments = segments_for_range(cache_dir, metadata, valid_cutoff, None)
    target_train_upper = (
        int(train_times[-args.purge_steps])
        if args.purge_steps > 0 else valid_cutoff
    )
    train_segments = segments_for_range(
        cache_dir, metadata, None, target_train_upper
    )
    train_x = training_matrix(args, cache_dir, train_segments)
    valid_x = training_matrix(args, cache_dir, valid_segments)
    train_w = vector_for_segments(cache_dir, train_segments, "weight")
    valid_w = vector_for_segments(cache_dir, valid_segments, "weight")
    train_responders = vector_for_segments(cache_dir, train_segments, "responder")
    valid_hat = np.empty((len(valid_x), len(responders)), dtype=np.float32)
    responder_files = {}
    selection_responder_dir = (
        work_dir / "selection_responder_models" / oof_signature[:16]
    )
    selection_responder_dir.mkdir(parents=True, exist_ok=True)
    for column, name in enumerate(responders):
        final_rounds = int(np.median(responder_best_iterations[name]))
        filename = f"{name}.txt"
        model_path = selection_responder_dir / filename
        if args.skip_existing_models and model_path.exists():
            progress(f"loading existing selection responder model: {filename}")
            model = lgb.Booster(model_file=str(model_path))
        else:
            progress(
                f"training selection responder {column + 1}/{len(responders)}: "
                f"{name}, rounds={final_rounds}, rows={len(train_x):,}"
            )
            model = train_lgb_fixed(
                train_x, train_responders[:, column], train_w, final_rounds,
                args, f"selection {name}",
                categorical_feature=[len(metadata["feature_columns"])
                    + len(metadata["temporal_feature_columns"])],
            )
            model.save_model(str(model_path))
            progress(f"saved selection responder model: {model_path}")
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
        progress_bar(
            "selection responders", column + 1, len(responders), name
        )

    if args.training_data_mode == "in-memory":
        del train_x, valid_x, train_responders
        gc.collect()

    target_train_segments = segments_for_range(
        cache_dir, metadata, int(train_times[warmup_index]),
        target_train_upper,
    )
    target_train_rows = sum(
        end - start for _, start, end in target_train_segments
    )
    target_oof_hat = oof_hat[:target_train_rows]
    y_train = vector_for_segments(cache_dir, target_train_segments, "target")
    w_train = vector_for_segments(cache_dir, target_train_segments, "weight")
    variants = requested_target_experiments
    experiment_specs = {
        name: target_experiment_spec(metadata, name, responders)
        for name in variants
    }
    target_train_times = (
        vector_for_segments(cache_dir, target_train_segments, "time")
        if any(spec["shuffle_within_time"] for spec in experiment_specs.values())
        else None
    )
    y_valid = vector_for_segments(cache_dir, valid_segments, "target")
    valid_times = vector_for_segments(cache_dir, valid_segments, "time")
    progress(f"target experiments: {variants}")
    ablation_scores = {}
    importance_frames = []
    best_variant = None
    best_score = -np.inf
    best_target_model = None
    best_valid_prediction = None
    for variant in variants:
        spec = experiment_specs[variant]
        target_train_x, target_valid_x = target_experiment_matrices(
            args, cache_dir, target_train_segments, valid_segments,
            target_oof_hat, valid_hat,
            spec, target_train_times, valid_times,
        )
        variant_path = model_dir / f"target_{variant}.txt"
        old_spec = (
            existing_model_metadata.get("target_experiment_specs", {}).get(variant)
            if existing_model_metadata else None
        )
        can_load = bool(
            args.skip_existing_models
            and model_metadata_matches
            and variant_path.exists()
            and old_spec
            and old_spec.get("base_indices") == spec["base_indices"]
            and old_spec.get("responders") == spec["responders"]
            and old_spec.get("shuffle_within_time", False)
            == spec["shuffle_within_time"]
        )
        if can_load:
            candidate = lgb.Booster(model_file=str(variant_path))
            can_load = candidate.num_feature() == len(spec["feature_names"])
        if can_load:
            progress(f"loading existing target experiment: {variant_path.name}")
            target_model = candidate
        else:
            progress(
                f"training target experiment {variant}: "
                f"features={len(spec['feature_names'])}, "
                f"temporal_groups={spec['temporal_groups']}, "
                f"responders={spec['responders']}, "
                f"train_rows={len(target_train_x):,}, valid_rows={len(target_valid_x):,}"
            )
            target_model = train_lgb(
                target_train_x, y_train, w_train, target_valid_x,
                y_valid, valid_w, args.target_rounds, args,
                f"target {variant}",
                categorical_feature=[spec["feature_names"].index("asset_id")],
            )
            target_model.save_model(str(variant_path))
        prediction = predict_sequence(target_model, target_valid_x, f"target {variant} validation")
        score = weighted_r2(y_valid, prediction, valid_w)
        segmented = segmented_validation_scores(
            valid_times, y_valid, prediction, valid_w
        )
        best_iteration = int(target_model.best_iteration)
        if best_iteration <= 0:
            best_iteration = int(target_model.current_iteration())
        ablation_scores[variant] = {
            "score": score,
            "features": target_model.num_feature(),
            "best_iteration": best_iteration,
            "file": variant_path.name,
            "temporal_groups": spec["temporal_groups"],
            "responders": spec["responders"],
            "segmented_validation": segmented,
        }
        importance_frames.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "feature": spec["feature_names"],
                    "importance_gain": target_model.feature_importance(importance_type="gain"),
                    "importance_split": target_model.feature_importance(importance_type="split"),
                }
            )
        )
        if spec["deployable"] and score > best_score:
            best_variant = variant
            best_score = score
            best_target_model = target_model
            best_valid_prediction = prediction.copy()
        part_scores = ", ".join(
            f"P{item['part']}={item['score']:.8f}"
            for item in segmented["parts"]
        )
        progress(
            f"target experiment {variant}: zero-mean R2={score:.8f}, "
            f"segment_std={segmented['std']:.8f}; {part_scores}"
        )
        progress_bar(
            "target experiments",
            variants.index(variant) + 1,
            len(variants),
            f"{variant}, R2={score:.8f}",
        )
        if args.training_data_mode == "in-memory":
            del target_train_x, target_valid_x
            gc.collect()

    if "A" in ablation_scores:
        baseline_score = float(ablation_scores["A"]["score"])
        baseline_parts = ablation_scores["A"]["segmented_validation"]["parts"]
        for result in ablation_scores.values():
            result["delta_vs_A"] = float(result["score"] - baseline_score)
            result_parts = result["segmented_validation"]["parts"]
            result["segment_delta_vs_A"] = [
                float(current["score"] - baseline["score"])
                for current, baseline in zip(result_parts, baseline_parts)
            ]

    if best_variant is None or best_target_model is None:
        raise RuntimeError("no target experiment was trained")
    valid_pred = best_valid_prediction
    best_spec = experiment_specs[best_variant]
    best_rounds = int(ablation_scores[best_variant]["best_iteration"])

    progress(
        f"refitting selected target {best_variant} with validation labels: "
        f"rounds={best_rounds}, seeds={target_seeds}"
    )
    refit_oof_x, refit_valid_x = target_experiment_matrices(
        args, cache_dir, oof_segments, valid_segments, oof_hat, valid_hat,
        best_spec,
    )
    target_refit_x = concatenate_training_matrices(
        args, refit_oof_x, refit_valid_x
    )
    target_refit_y = np.concatenate([
        vector_for_segments(cache_dir, oof_segments, "target"), y_valid
    ])
    target_refit_w = np.concatenate([
        vector_for_segments(cache_dir, oof_segments, "weight"), valid_w
    ])
    target_refit_rows = len(target_refit_y)
    target_model_files = []
    for model_index, seed in enumerate(target_seeds, start=1):
        filename = f"target_final_seed{seed}.txt"
        target_seed_path = model_dir / filename
        can_load = bool(
            args.skip_existing_models
            and model_metadata_matches
            and target_seed_path.exists()
            and existing_model_metadata.get("target_variant") == best_variant
            and existing_model_metadata.get("target_feature_columns")
            == best_spec["feature_names"]
        )
        if can_load:
            target_model = lgb.Booster(model_file=str(target_seed_path))
            can_load = (
                target_model.num_feature() == len(best_spec["feature_names"])
            )
        if can_load:
            progress(f"loading existing final target model: {filename}")
        else:
            progress(
                f"training final target {model_index}/{len(target_seeds)}: "
                f"seed={seed}, rows={len(target_refit_x):,}"
            )
            target_model = train_lgb_fixed(
                target_refit_x, target_refit_y, target_refit_w,
                best_rounds, args, f"final target seed={seed}",
                categorical_feature=[
                    best_spec["feature_names"].index("asset_id")
                ],
                seed=seed,
            )
            target_model.save_model(str(target_seed_path))
        target_model_files.append(filename)
        progress_bar(
            "final target ensemble", model_index, len(target_seeds),
            f"seed={seed}",
        )
    # Keep the historical filename as an alias for older tooling.
    target_model = lgb.Booster(
        model_file=str(model_dir / target_model_files[0])
    )
    target_model.save_model(str(model_dir / "target_lightgbm.txt"))
    del target_refit_y, target_refit_w
    if args.training_data_mode == "in-memory":
        del refit_oof_x, refit_valid_x, target_refit_x
        gc.collect()

    progress("refitting deployment responder models on every available row")
    all_segments = segments_for_range(cache_dir, metadata, None, None)
    all_x = training_matrix(args, cache_dir, all_segments)
    all_w = vector_for_segments(cache_dir, all_segments, "weight")
    all_responders = vector_for_segments(cache_dir, all_segments, "responder")
    asset_categorical = [
        len(metadata["feature_columns"])
        + len(metadata["temporal_feature_columns"])
    ]
    for column, name in enumerate(responders):
        filename = responder_files[name]
        model_path = model_dir / filename
        final_rounds = int(np.median(responder_best_iterations[name]))
        can_load = bool(
            args.skip_existing_models
            and model_metadata_matches
            and model_path.exists()
        )
        if can_load:
            progress(f"loading existing deployment responder model: {filename}")
        else:
            progress(
                f"training deployment responder {column + 1}/{len(responders)}: "
                f"{name}, rounds={final_rounds}, rows={len(all_x):,}"
            )
            model = train_lgb_fixed(
                all_x, all_responders[:, column], all_w, final_rounds,
                args, f"deployment {name}",
                categorical_feature=asset_categorical,
            )
            model.save_model(str(model_path))
        progress_bar(
            "deployment responders", column + 1, len(responders), name
        )
    if args.training_data_mode == "in-memory":
        del all_x
        gc.collect()
    pd.concat(importance_frames, ignore_index=True).to_csv(
        model_dir / "target_feature_importance.csv", index=False
    )
    mechanism_summary = c4_mechanism_summary(ablation_scores)
    mechanism_path = model_dir / "c4_mechanism_report.json"
    if mechanism_summary is not None:
        mechanism_path.write_text(
            json.dumps(mechanism_summary, indent=2), encoding="utf-8"
        )
    else:
        mechanism_path.unlink(missing_ok=True)
    (model_dir / "ablation_report.json").write_text(
        json.dumps(
            {"best_variant": best_variant, "scores": ablation_scores,
             "target_experiment_specs": experiment_specs,
             "responder_diagnostics": responder_diagnostics,
             "c4_mechanism": mechanism_summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    clip_min, clip_max = np.quantile(valid_pred[np.isfinite(valid_pred)], [0.001, 0.999])
    valid_assets = vector_for_segments(
        cache_dir, valid_segments, "x", column=-1
    ).astype(np.int64)
    np.savez_compressed(
        model_dir / "validation_predictions.npz",
        time_id=np.asarray(valid_times, dtype=np.int64),
        asset_id=valid_assets,
        target=np.asarray(y_valid, dtype=np.float32),
        weight=np.asarray(valid_w, dtype=np.float32),
        prediction=np.asarray(
            np.clip(valid_pred, clip_min, clip_max), dtype=np.float32
        ),
    )
    output = {
        "strategy": "responder_assisted_lgb_catboost_strategy",
        "feature_columns": metadata["feature_columns"], "responders": responders,
        "temporal_features": metadata["temporal_features"],
        "temporal_recipes": metadata["temporal_recipes"],
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "temporal_feature_columns": metadata["temporal_feature_columns"],
        "temporal_plan_hash": metadata.get("temporal_plan_hash", ""),
        "responder_models": responder_files,
        "target_model": "target_lightgbm.txt",
        "target_models": target_model_files,
        "target_seeds": target_seeds,
        "target_variant": best_variant,
        "target_base_indices": best_spec["base_indices"],
        "target_responders": best_spec["responders"],
        "target_temporal_groups": best_spec["temporal_groups"],
        "target_feature_columns": best_spec["feature_names"],
        "target_experiment_specs": experiment_specs,
        "trained_target_experiments": variants,
        "ablation_scores": ablation_scores,
        "responder_diagnostics": responder_diagnostics,
        "target_feature_importance": "target_feature_importance.csv",
        "ablation_report": "ablation_report.json",
        "valid_cutoff_time_id": valid_cutoff, "oof_folds": args.oof_folds,
        "purge_steps": args.purge_steps,
        "lgb_profile_version": LGB_PROFILE_VERSION,
        "lgb_params": low_risk_lgb_params(args),
        "categorical_features": ["asset_id"],
        "training_config": requested_training_config,
        "warmup_fraction": args.warmup_fraction, "target_train_rows": len(y_train),
        "target_refit_rows": target_refit_rows,
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
