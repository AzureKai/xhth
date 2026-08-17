from __future__ import annotations

import argparse
import gc
import json
import numbers
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from data_utils import feature_columns_from_path, manifest_files
from features import prepare_model_frame, select_history_features
from preprocess import PreprocessSpec, apply_preprocess
from train import (
    BASE_PARAMS,
    DATA_RANDOM_SEED,
    PARAM_CANDIDATES,
    _candidate_model_params,
    _candidate_params,
    _select_winning_candidate,
    lgb_zero_mean_r2,
)
from validation import evaluate_gates, make_validation_plan


CACHE_VERSION = 2
BATCH_ROWS = 65_536
SEQUENCE_BATCH_ROWS = 65_536
PREDICT_BATCH_ROWS = 100_000
TOP_K_HISTORY = 48
ROLLING_WINDOWS = (5,)
SEEDS = (2026, 2027, 2028)
N_SPLITS = 5
HOLDOUT_FRACTION = 0.15
PURGE_STEPS = 30
MIN_TRAIN_FRACTION = 0.40
NUM_BOOST_ROUND = 700
EARLY_STOPPING_ROUNDS = 80
CORR_SAMPLE_ROWS = 200_000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _file_fingerprints(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(path.resolve()),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    ]


def scan_time_axis(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    unique_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    last_time: int | None = None
    total_rows = 0
    for file_id, path in enumerate(paths, start=1):
        file_times: list[np.ndarray] = []
        file_counts: list[np.ndarray] = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=262_144, columns=["time_id"]):
            values = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if values.size == 0:
                continue
            if np.any(values[1:] < values[:-1]):
                raise ValueError(f"time_id is not sorted inside {path}")
            if last_time is not None and int(values[0]) < last_time:
                raise ValueError(f"time_id is not globally sorted at {path}")
            current_times, current_counts = np.unique(values, return_counts=True)
            if file_times and int(current_times[0]) == int(file_times[-1][-1]):
                file_counts[-1][-1] += current_counts[0]
                current_times = current_times[1:]
                current_counts = current_counts[1:]
            if current_times.size:
                file_times.append(current_times.astype(np.int64, copy=False))
                file_counts.append(current_counts.astype(np.int64, copy=False))
            last_time = int(values[-1])
            total_rows += len(values)
        if file_times:
            times = np.concatenate(file_times)
            counts = np.concatenate(file_counts)
            if unique_parts and int(times[0]) == int(unique_parts[-1][-1]):
                count_parts[-1][-1] += counts[0]
                times = times[1:]
                counts = counts[1:]
            if times.size:
                unique_parts.append(times)
                count_parts.append(counts)
        log(f"time scan {file_id}/{len(paths)}: {path.name}")
    unique_times = np.concatenate(unique_parts)
    counts = np.concatenate(count_parts)
    if int(counts.sum()) != total_rows:
        raise RuntimeError("time_id counts do not add up to the parquet row count")
    return unique_times, counts


def row_offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def spans_from_time_ids(
    unique_times: np.ndarray,
    row_offsets: np.ndarray,
    selected_times: np.ndarray,
) -> list[tuple[int, int]]:
    selected = np.asarray(selected_times, dtype=np.int64)
    if selected.size == 0:
        return []
    positions = np.searchsorted(unique_times, selected)
    if np.any(positions >= len(unique_times)) or np.any(unique_times[positions] != selected):
        raise ValueError("validation plan contains unknown time_id values")
    positions = np.sort(positions)
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    runs = np.split(positions, breaks)
    return [(int(row_offsets[run[0]]), int(row_offsets[run[-1] + 1])) for run in runs]


def span_length(spans: list[tuple[int, int]]) -> int:
    return int(sum(end - start for start, end in spans))


def _path_row_ranges(paths: list[Path]) -> list[tuple[Path, int, int]]:
    result: list[tuple[Path, int, int]] = []
    offset = 0
    for path in paths:
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        result.append((path, offset, offset + rows))
        offset += rows
    return result


def _overlap_slices(
    batch_start: int,
    batch_end: int,
    spans: list[tuple[int, int]],
) -> list[slice]:
    slices: list[slice] = []
    for start, end in spans:
        left = max(batch_start, start)
        right = min(batch_end, end)
        if left < right:
            slices.append(slice(left - batch_start, right - batch_start))
    return slices


def _record_batch_matrix(batch, start_column: int, n_columns: int) -> np.ndarray:
    columns = [
        batch.column(start_column + idx).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        for idx in range(n_columns)
    ]
    return np.column_stack(columns)


def fit_schema_streaming(
    paths: list[Path],
    feature_cols: list[str],
    schema_spans: list[tuple[int, int]],
) -> list[str]:
    finite_counts = np.zeros(len(feature_cols), dtype=np.int64)
    minima = np.full(len(feature_cols), np.inf, dtype=np.float64)
    maxima = np.full(len(feature_cols), -np.inf, dtype=np.float64)
    seen_rows = 0
    columns = ["time_id", *feature_cols]
    path_ranges = _path_row_ranges(paths)
    for file_id, (path, file_start, file_end) in enumerate(path_ranges, start=1):
        if not _overlap_slices(file_start, file_end, schema_spans):
            continue
        row_offset = file_start
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            batch_end = row_offset + batch.num_rows
            slices = _overlap_slices(row_offset, batch_end, schema_spans)
            if slices:
                matrix = _record_batch_matrix(batch, 1, len(feature_cols))
                selected = matrix[slices[0]] if len(slices) == 1 else np.concatenate([matrix[item] for item in slices])
                finite = np.isfinite(selected)
                finite_counts += finite.sum(axis=0, dtype=np.int64)
                np.nan_to_num(selected, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                minima = np.minimum(minima, selected.min(axis=0).astype(np.float64))
                maxima = np.maximum(maxima, selected.max(axis=0).astype(np.float64))
                seen_rows += len(selected)
            row_offset = batch_end
        log(f"schema scan {file_id}/{len(paths)}: {path.name}")
    if seen_rows != span_length(schema_spans):
        raise RuntimeError(f"schema scan row mismatch: {seen_rows} != {span_length(schema_spans)}")
    finite_ratio = finite_counts.astype(np.float64) / max(seen_rows, 1)
    keep = (finite_ratio >= 0.01) & (minima != maxima)
    raw_features = [name for name, selected in zip(feature_cols, keep) if bool(selected)]
    if not raw_features:
        raise ValueError("no usable feature columns after streaming health check")
    return raw_features


def _time_membership(values: np.ndarray, selected_times: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(selected_times, values)
    valid = positions < len(selected_times)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = selected_times[positions[valid]] == values[valid]
    return result


def select_history_features_streaming(
    paths: list[Path],
    unique_times: np.ndarray,
    time_counts: np.ndarray,
    schema_times: np.ndarray,
    raw_features: list[str],
) -> list[str]:
    positions = np.searchsorted(unique_times, schema_times)
    schema_rows = int(time_counts[positions].sum())
    rows_per_time = max(schema_rows / max(len(schema_times), 1), 1.0)
    n_times = max(1, min(len(schema_times), int(CORR_SAMPLE_ROWS / rows_per_time)))
    rng = np.random.default_rng(SEEDS[0])
    chosen_times = np.sort(rng.choice(schema_times, size=n_times, replace=False)).astype(np.int64)
    chosen_positions = np.searchsorted(unique_times, chosen_times)
    capacity = int(time_counts[chosen_positions].sum())

    raw = np.empty((capacity, len(raw_features)), dtype=np.float32)
    target = np.empty(capacity, dtype=np.float32)
    weight = np.empty(capacity, dtype=np.float32)
    time_id = np.empty(capacity, dtype=np.int64)
    asset_id = np.empty(capacity, dtype=np.int8)
    columns = ["time_id", "asset_id", "weight", "target", *raw_features]
    cursor = 0
    for file_id, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        stats = parquet.metadata.row_group(0).column(1).statistics
        if stats is not None and (int(stats.max) < int(chosen_times[0]) or int(stats.min) > int(chosen_times[-1])):
            continue
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns):
            times = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            mask = _time_membership(times, chosen_times)
            count = int(mask.sum())
            if count == 0:
                continue
            end = cursor + count
            time_id[cursor:end] = times[mask]
            asset_id[cursor:end] = batch.column(1).to_numpy(zero_copy_only=False)[mask].astype(np.int8, copy=False)
            weight[cursor:end] = batch.column(2).to_numpy(zero_copy_only=False)[mask].astype(np.float32, copy=False)
            target[cursor:end] = batch.column(3).to_numpy(zero_copy_only=False)[mask].astype(np.float32, copy=False)
            raw[cursor:end] = _record_batch_matrix(batch, 4, len(raw_features))[mask]
            cursor = end
        log(f"correlation sample scan {file_id}/{len(paths)}: {path.name}")
    if cursor != capacity:
        raise RuntimeError(f"correlation sample row mismatch: {cursor} != {capacity}")

    data: dict[str, np.ndarray] = {
        "time_id": time_id,
        "asset_id": asset_id,
        "weight": weight,
        "target": target,
    }
    data.update({name: raw[:, idx] for idx, name in enumerate(raw_features)})
    sample = pd.DataFrame(data, copy=False)
    if len(sample) > CORR_SAMPLE_ROWS:
        sample = sample.sample(n=CORR_SAMPLE_ROWS, random_state=SEEDS[0]).sort_values(["time_id", "asset_id"])
    sample = apply_preprocess(sample, PreprocessSpec(raw_features=tuple(raw_features)))
    selected = select_history_features(
        sample,
        raw_features,
        top_k=TOP_K_HISTORY,
        sample_rows=CORR_SAMPLE_ROWS,
        seed=SEEDS[0],
    )
    del sample, data, raw
    gc.collect()
    return selected


def _history_batch(
    values: np.ndarray,
    asset_ids: np.ndarray,
    state: dict[int, tuple[int, np.ndarray, np.ndarray]],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    lag = np.zeros_like(values, dtype=np.float32)
    rolling = np.zeros_like(values, dtype=np.float32)
    for asset_id in np.unique(asset_ids):
        indices = np.flatnonzero(asset_ids == asset_id)
        current = values[indices]
        existing = state.get(int(asset_id))
        if existing is None:
            previous_count = 0
            previous_raw = np.zeros(values.shape[1], dtype=np.float32)
            # Prefix P_0. Keeping prefix states reproduces the official float32
            # cumulative sum even when a parquet batch cuts an asset's history.
            prefix_tail = np.zeros((1, values.shape[1]), dtype=np.float32)
        else:
            previous_count, previous_raw, prefix_tail = existing

        if previous_count > 0:
            lag[indices[0]] = previous_raw
        if len(current) > 1:
            lag[indices[1:]] = current[:-1]

        continued = np.vstack([prefix_tail[-1], current])
        new_prefix = np.cumsum(continued, axis=0, dtype=np.float32)[1:]
        all_prefix = np.vstack([prefix_tail, new_prefix])
        prefix_start_count = max(0, previous_count - window)
        positions = previous_count + np.arange(1, len(current) + 1, dtype=np.int64)
        denominator_counts = np.maximum(0, positions - window)
        denominator_indices = denominator_counts - prefix_start_count
        numerators = new_prefix.astype(np.float64)
        denominators = all_prefix[denominator_indices].astype(np.float64)
        counts = np.minimum(positions, window).reshape(-1, 1)
        rolling[indices] = ((numerators - denominators) / counts).astype(np.float32)

        new_count = previous_count + len(current)
        keep_prefixes = min(new_count, window) + 1
        state[int(asset_id)] = (new_count, current[-1].copy(), all_prefix[-keep_prefixes:].copy())
    return lag, rolling


def materialize_model_data(
    paths: list[Path],
    cache_dir: Path,
    total_rows: int,
    raw_features: list[str],
    history_features: list[str],
) -> tuple[Path, Path, Path, list[str]]:
    if len(ROLLING_WINDOWS) != 1 or int(ROLLING_WINDOWS[0]) <= 0:
        raise ValueError("low-memory materialization requires exactly one positive rolling window")
    rolling_window = int(ROLLING_WINDOWS[0])
    engineered = [f"lag1_{name}" for name in history_features]
    engineered += [f"diff1_{name}" for name in history_features]
    engineered += [f"rmean{rolling_window}_{name}" for name in history_features]
    model_cols = ["asset_id", *raw_features, *engineered]
    matrix_path = cache_dir / "model_matrix.npy"
    target_path = cache_dir / "target.npy"
    weight_path = cache_dir / "weight.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, len(model_cols)),
    )
    target_out = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=(total_rows,))
    weight_out = np.lib.format.open_memmap(weight_path, mode="w+", dtype=np.float32, shape=(total_rows,))
    history_positions = [raw_features.index(name) for name in history_features]
    columns = ["time_id", "asset_id", "weight", "target", *raw_features]
    state: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    cursor = 0
    last_time: int | None = None
    for file_id, path in enumerate(paths, start=1):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            end = cursor + batch.num_rows
            times = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if last_time is not None and int(times[0]) < last_time:
                raise ValueError(f"time_id order changed while materializing {path}")
            last_time = int(times[-1])
            assets = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
            weights = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            targets = batch.column(3).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            raw = _record_batch_matrix(batch, 4, len(raw_features))
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            selected = raw[:, history_positions]
            lag, rolling = _history_batch(selected, assets, state, rolling_window)

            matrix[cursor:end, 0] = assets.astype(np.float32)
            raw_end = 1 + len(raw_features)
            matrix[cursor:end, 1:raw_end] = raw
            lag_end = raw_end + len(history_features)
            diff_end = lag_end + len(history_features)
            matrix[cursor:end, raw_end:lag_end] = lag
            matrix[cursor:end, lag_end:diff_end] = selected - lag
            matrix[cursor:end, diff_end:] = rolling
            target_out[cursor:end] = np.where(np.isnan(targets), 0.0, targets)
            weight_out[cursor:end] = np.maximum(np.where(np.isnan(weights), 0.0, weights), 0.0)
            cursor = end
        matrix.flush()
        target_out.flush()
        weight_out.flush()
        log(f"feature cache {file_id}/{len(paths)}: {path.name}; rows={cursor:,}/{total_rows:,}")
    if cursor != total_rows:
        raise RuntimeError(f"feature cache row mismatch: {cursor} != {total_rows}")
    del matrix, target_out, weight_out
    gc.collect()
    return matrix_path, target_path, weight_path, model_cols


MatrixPatch = tuple[int, int, np.ndarray]


def build_cold_start_patch(
    matrix_path: Path,
    *,
    unique_times: np.ndarray,
    time_counts: np.ndarray,
    row_offsets: np.ndarray,
    session_time_ids: np.ndarray,
    raw_features: list[str],
    history_features: list[str],
    rolling_windows: tuple[int, ...],
    model_cols: list[str],
) -> list[MatrixPatch]:
    """Patch the first rolling window so a validation API session starts empty.

    The disk cache is materialized once over the full chronology. Later rows in
    a validation block are already identical to causal sequential inference;
    only the first ``max(rolling_windows)`` time ids can depend on history from
    before the simulated session. Rebuilding those few rows keeps the large
    memmap reusable while matching ``main.Model`` cold-start semantics.
    """
    if len(session_time_ids) == 0:
        return []
    warmup_times = np.asarray(session_time_ids[: max(rolling_windows)], dtype=np.int64)
    positions = np.searchsorted(unique_times, warmup_times)
    if np.any(positions >= len(unique_times)) or np.any(unique_times[positions] != warmup_times):
        raise ValueError("session contains unknown time ids")
    start = int(row_offsets[positions[0]])
    end = int(row_offsets[positions[-1] + 1])
    repeated_times = np.repeat(warmup_times, time_counts[positions])
    if len(repeated_times) != end - start:
        raise RuntimeError("cold-start patch row count mismatch")

    source = np.load(matrix_path, mmap_mode="r")
    raw = np.asarray(source[start:end, 1 : 1 + len(raw_features)], dtype=np.float32)
    assets = np.asarray(source[start:end, 0], dtype=np.int8)
    payload: dict[str, np.ndarray] = {
        "row_id": np.arange(end - start, dtype=np.int64),
        "time_id": repeated_times,
        "asset_id": assets,
        "weight": np.ones(end - start, dtype=np.float32),
        "target": np.zeros(end - start, dtype=np.float32),
    }
    payload.update({name: raw[:, idx] for idx, name in enumerate(raw_features)})
    session, rebuilt_cols = prepare_model_frame(
        pd.DataFrame(payload, copy=False),
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
    )
    if rebuilt_cols != model_cols:
        raise ValueError("cold-start patch feature schema mismatch")
    values = session.loc[:, model_cols].to_numpy(dtype=np.float32, copy=True)
    return [(start, end, values)]


class SpannedMemmapSequence(lgb.Sequence):
    batch_size = SEQUENCE_BATCH_ROWS

    def __init__(
        self,
        matrix_path: Path,
        shape: tuple[int, int],
        spans: list[tuple[int, int]],
        patches: list[MatrixPatch] | None = None,
    ) -> None:
        self.matrix = np.load(matrix_path, mmap_mode="r")
        if tuple(self.matrix.shape) != tuple(shape):
            raise ValueError(f"matrix shape mismatch: {self.matrix.shape} != {shape}")
        self.spans = tuple((int(start), int(end)) for start, end in spans)
        lengths = np.asarray([end - start for start, end in self.spans], dtype=np.int64)
        self.ends = np.cumsum(lengths)
        self.starts = np.r_[0, self.ends[:-1]]
        self.total = int(self.ends[-1]) if len(self.ends) else 0
        self.patches = tuple(patches or ())

    def __len__(self) -> int:
        return self.total

    def _global_indices(self, local: np.ndarray) -> np.ndarray:
        span_ids = np.searchsorted(self.ends, local, side="right")
        span_starts = np.asarray([self.spans[int(idx)][0] for idx in span_ids], dtype=np.int64)
        return span_starts + local - self.starts[span_ids]

    def _patched_slice(self, start: int, end: int) -> np.ndarray:
        values = np.asarray(self.matrix[start:end], dtype=np.float64)
        if not self.patches:
            return values
        values = values.copy()
        for patch_start, patch_end, patch in self.patches:
            left = max(start, patch_start)
            right = min(end, patch_end)
            if left < right:
                values[left - start : right - start] = patch[left - patch_start : right - patch_start]
        return values

    def _patched_rows(self, indices: np.ndarray) -> np.ndarray:
        values = np.asarray(self.matrix[indices], dtype=np.float64)
        if not self.patches:
            return values
        values = values.copy()
        for patch_start, patch_end, patch in self.patches:
            mask = (indices >= patch_start) & (indices < patch_end)
            if np.any(mask):
                values[mask] = patch[indices[mask] - patch_start]
        return values

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            local = int(idx)
            if local < 0:
                local += self.total
            if local < 0 or local >= self.total:
                raise IndexError(local)
            span_id = int(np.searchsorted(self.ends, local, side="right"))
            global_idx = self.spans[span_id][0] + local - int(self.starts[span_id])
            return self._patched_slice(global_idx, global_idx + 1)[0]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total)
            if step != 1:
                local = np.arange(start, stop, step, dtype=np.int64)
                return self._patched_rows(self._global_indices(local))
            if stop <= start:
                return np.empty((0, self.matrix.shape[1]), dtype=np.float64)
            pieces: list[np.ndarray] = []
            cursor = start
            while cursor < stop:
                span_id = int(np.searchsorted(self.ends, cursor, side="right"))
                available = min(stop, int(self.ends[span_id])) - cursor
                global_start = self.spans[span_id][0] + cursor - int(self.starts[span_id])
                pieces.append(self._patched_slice(global_start, global_start + available))
                cursor += available
            return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            local = np.asarray(idx, dtype=np.int64)
            return self._patched_rows(self._global_indices(local))
        raise TypeError(f"unsupported sequence index type: {type(idx).__name__}")


def _read_vector_spans(path: Path, spans: list[tuple[int, int]]) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    if len(spans) == 1:
        start, end = spans[0]
        return np.asarray(source[start:end]).copy()
    return np.concatenate([np.asarray(source[start:end]) for start, end in spans])


def build_dataset(
    matrix_path: Path,
    matrix_shape: tuple[int, int],
    target_path: Path,
    weight_path: Path,
    spans: list[tuple[int, int]],
    model_cols: list[str],
    *,
    reference: lgb.Dataset | None = None,
    data_random_seed: int = DATA_RANDOM_SEED,
    patches: list[MatrixPatch] | None = None,
    construction_overrides: dict | None = None,
) -> lgb.Dataset:
    sequence = SpannedMemmapSequence(matrix_path, matrix_shape, spans, patches=patches)
    labels = _read_vector_spans(target_path, spans)
    weights = _read_vector_spans(weight_path, spans)
    construction_params = {
        "data_random_seed": int(data_random_seed),
        "min_data_in_leaf": min(int(item["min_data_in_leaf"]) for item in PARAM_CANDIDATES),
        "max_bin": int(BASE_PARAMS["max_bin"]),
        "force_col_wise": bool(BASE_PARAMS["force_col_wise"]),
        "verbosity": -1,
    }
    if construction_overrides:
        construction_params.update(construction_overrides)
    dataset = lgb.Dataset(
        sequence,
        label=labels,
        weight=weights,
        reference=reference,
        feature_name=model_cols,
        categorical_feature=["asset_id"],
        params=construction_params,
        free_raw_data=True,
    )
    dataset.construct()
    return dataset


def train_early_stopping(
    train_set: lgb.Dataset,
    valid_set: lgb.Dataset,
    candidate: dict,
    num_threads: int,
    *,
    param_builder=_candidate_params,
) -> lgb.Booster:
    return lgb.train(
        param_builder(SEEDS[0], candidate, num_threads=num_threads),
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        feval=lgb_zero_mean_r2,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def train_fixed(
    train_set: lgb.Dataset,
    candidate: dict,
    seed: int,
    rounds: int,
    num_threads: int,
    *,
    param_builder=_candidate_params,
) -> lgb.Booster:
    return lgb.train(
        param_builder(seed, candidate, num_threads=num_threads),
        train_set,
        num_boost_round=rounds,
        valid_sets=[train_set],
        valid_names=["train"],
        feval=lgb_zero_mean_r2,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def prediction_statistics(
    model: lgb.Booster,
    matrix_path: Path,
    target_path: Path,
    weight_path: Path,
    spans: list[tuple[int, int]],
    best_iteration: int,
    patches: list[MatrixPatch] | None = None,
) -> dict[str, float]:
    matrix = np.load(matrix_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    weight = np.load(weight_path, mmap_mode="r")
    patches = list(patches or ())
    stats = {"y2": 0.0, "residual": 0.0, "yp": 0.0, "p2": 0.0}
    for span_start, span_end in spans:
        for start in range(span_start, span_end, PREDICT_BATCH_ROWS):
            end = min(start + PREDICT_BATCH_ROWS, span_end)
            y = np.asarray(target[start:end], dtype=np.float64)
            w = np.asarray(weight[start:end], dtype=np.float64)
            values = np.asarray(matrix[start:end])
            if patches:
                values = values.copy()
                for patch_start, patch_end, patch in patches:
                    left = max(start, patch_start)
                    right = min(end, patch_end)
                    if left < right:
                        values[left - start : right - start] = patch[left - patch_start : right - patch_start]
            prediction = np.asarray(
                model.predict(values, num_iteration=best_iteration),
                dtype=np.float64,
            )
            stats["y2"] += float(np.sum(w * y * y))
            stats["residual"] += float(np.sum(w * (y - prediction) ** 2))
            stats["yp"] += float(np.sum(w * y * prediction))
            stats["p2"] += float(np.sum(w * prediction * prediction))
    return stats


def score_from_stats(stats: dict[str, float]) -> float:
    return 0.0 if stats["y2"] <= 0.0 else float(1.0 - stats["residual"] / stats["y2"])


def add_stats(total: dict[str, float], current: dict[str, float]) -> None:
    for key in total:
        total[key] += current[key]


def prepare_cache(release_root: Path, cache_dir: Path) -> dict:
    paths = manifest_files(release_root, "train")
    if not paths:
        raise ValueError("no train parquet files found")
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    fingerprints = _file_fingerprints(paths)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_files = [cache_dir / name for name in ("time_axis.npz", "model_matrix.npy", "target.npy", "weight.npy")]
        if (
            metadata.get("complete") is True
            and metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("input_files") == fingerprints
            and all(path.exists() for path in expected_files)
        ):
            log("reusing completed low-memory feature cache")
            return metadata
        raise RuntimeError(
            f"cache exists but is incomplete or stale: {cache_dir}. "
            "Move it aside or remove it before retrying."
        )

    feature_cols = feature_columns_from_path(paths[0])
    unique_times, time_counts = scan_time_axis(paths)
    total_rows = int(time_counts.sum())
    np.savez(cache_dir / "time_axis.npz", unique_times=unique_times, time_counts=time_counts)
    row_offsets = row_offsets_from_counts(time_counts)
    plan = make_validation_plan(
        unique_times,
        n_splits=N_SPLITS,
        holdout_fraction=HOLDOUT_FRACTION,
        purge_steps=PURGE_STEPS,
        min_train_fraction=MIN_TRAIN_FRACTION,
    )
    schema_times = plan.feature_fit_time_ids
    schema_spans = spans_from_time_ids(unique_times, row_offsets, schema_times)
    log(f"fitting frozen schema on {span_length(schema_spans):,} rows")
    raw_features = fit_schema_streaming(paths, feature_cols, schema_spans)
    log(f"frozen schema kept {len(raw_features)}/{len(feature_cols)} raw features")
    history_features = select_history_features_streaming(
        paths,
        unique_times,
        time_counts,
        schema_times,
        raw_features,
    )
    log(f"selected {len(history_features)} history features")
    matrix_path, target_path, weight_path, model_cols = materialize_model_data(
        paths,
        cache_dir,
        total_rows,
        raw_features,
        history_features,
    )
    metadata = {
        "complete": True,
        "cache_version": CACHE_VERSION,
        "input_files": fingerprints,
        "total_rows": total_rows,
        "raw_features": raw_features,
        "history_features": history_features,
        "rolling_windows": list(ROLLING_WINDOWS),
        "model_cols": model_cols,
        "matrix_shape": [total_rows, len(model_cols)],
        "matrix_file": matrix_path.name,
        "target_file": target_path.name,
        "weight_file": weight_path.name,
        "time_axis_file": "time_axis.npz",
        "validation_scheme": plan.cv_scheme,
        "min_train_fraction": MIN_TRAIN_FRACTION,
        "feature_fit_time_count": int(len(schema_times)),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    log("low-memory feature cache completed")
    return metadata


def run_training(
    release_root: Path,
    model_dir: Path,
    cache_dir: Path,
    num_threads: int,
) -> dict:
    metadata = prepare_cache(release_root, cache_dir)
    axis = np.load(cache_dir / metadata["time_axis_file"])
    unique_times = axis["unique_times"]
    time_counts = axis["time_counts"]
    row_offsets = row_offsets_from_counts(time_counts)
    plan = make_validation_plan(
        unique_times,
        n_splits=N_SPLITS,
        holdout_fraction=HOLDOUT_FRACTION,
        purge_steps=PURGE_STEPS,
        min_train_fraction=MIN_TRAIN_FRACTION,
    )
    matrix_path = cache_dir / metadata["matrix_file"]
    target_path = cache_dir / metadata["target_file"]
    weight_path = cache_dir / metadata["weight_file"]
    matrix_shape = tuple(int(value) for value in metadata["matrix_shape"])
    model_cols = list(metadata["model_cols"])
    raw_features = list(metadata["raw_features"])
    history_features = list(metadata["history_features"])
    rolling_windows = tuple(int(value) for value in metadata["rolling_windows"])

    candidate_results: list[dict] = []
    accumulators: dict[str, dict[str, float]] = {}
    for candidate in PARAM_CANDIDATES:
        candidate_results.append(
            {
                "name": candidate["name"],
                "logic": candidate.get("logic", ""),
                "regularization_rank": int(candidate.get("regularization_rank", 0)),
                "params": _candidate_model_params(candidate),
                "fold_scores": [],
                "fold_best_iterations": [],
            }
        )
        accumulators[candidate["name"]] = {"y2": 0.0, "residual": 0.0, "yp": 0.0, "p2": 0.0}

    for fold in plan.folds:
        train_spans = spans_from_time_ids(unique_times, row_offsets, fold.train_time_ids)
        valid_spans = spans_from_time_ids(unique_times, row_offsets, fold.valid_time_ids)
        valid_patches = build_cold_start_patch(
            matrix_path,
            unique_times=unique_times,
            time_counts=time_counts,
            row_offsets=row_offsets,
            session_time_ids=fold.valid_time_ids,
            raw_features=raw_features,
            history_features=history_features,
            rolling_windows=rolling_windows,
            model_cols=model_cols,
        )
        log(
            f"constructing fold={fold.fold_id} datasets: "
            f"train_rows={span_length(train_spans):,}, valid_rows={span_length(valid_spans):,}"
        )
        train_set = build_dataset(
            matrix_path,
            matrix_shape,
            target_path,
            weight_path,
            train_spans,
            model_cols,
        )
        valid_set = build_dataset(
            matrix_path,
            matrix_shape,
            target_path,
            weight_path,
            valid_spans,
            model_cols,
            reference=train_set,
            patches=valid_patches,
        )
        log(f"fold={fold.fold_id} datasets constructed")
        for candidate, result in zip(PARAM_CANDIDATES, candidate_results):
            log(f"CV start candidate={candidate['name']} fold={fold.fold_id}")
            model = train_early_stopping(train_set, valid_set, candidate, num_threads)
            best_iteration = int(model.best_iteration or NUM_BOOST_ROUND)
            stats = prediction_statistics(
                model,
                matrix_path,
                target_path,
                weight_path,
                valid_spans,
                best_iteration,
                patches=valid_patches,
            )
            fold_score = score_from_stats(stats)
            add_stats(accumulators[candidate["name"]], stats)
            result["fold_best_iterations"].append(best_iteration)
            result["fold_scores"].append(
                {
                    "fold_id": int(fold.fold_id),
                    "best_iteration": best_iteration,
                    "valid_raw": fold_score,
                    "train_rows": span_length(train_spans),
                    "valid_rows": span_length(valid_spans),
                    "train_time_start": int(fold.train_time_ids[0]),
                    "train_time_end": int(fold.train_time_ids[-1]),
                    "valid_time_start": int(fold.valid_time_ids[0]),
                    "valid_time_end": int(fold.valid_time_ids[-1]),
                }
            )
            log(
                f"CV done candidate={candidate['name']} fold={fold.fold_id} "
                f"best_iteration={best_iteration} valid_raw={fold_score:.8g}"
            )
            del model
            gc.collect()
        del valid_set, train_set
        gc.collect()

    for result in candidate_results:
        iterations = result["fold_best_iterations"]
        score_values = np.asarray([item["valid_raw"] for item in result["fold_scores"]], dtype=np.float64)
        result["mean_fold_score"] = float(np.mean(score_values))
        result["std_fold_score"] = float(np.std(score_values))
        result["min_fold_score"] = float(np.min(score_values))
        result["latest_fold_score"] = float(score_values[-1])
        result["mean_iterations"] = max(1, int(round(float(np.mean(iterations)))))
        result["oof_raw"] = score_from_stats(accumulators[result["name"]])
    winner = _select_winning_candidate(candidate_results)
    winning_candidate = next(item for item in PARAM_CANDIDATES if item["name"] == winner["name"])
    mean_iterations = int(winner["mean_iterations"])
    winner_stats = accumulators[winner["name"]]
    fitted_oof_scale = 1.0 if winner_stats["p2"] <= 0.0 else float(winner_stats["yp"] / winner_stats["p2"])
    oof_raw = float(winner["oof_raw"])
    log(
        f"selected candidate={winner['name']} mean_fold_score={winner['mean_fold_score']:.8g} "
        f"rounds={mean_iterations} oof_raw={oof_raw:.8g}"
    )

    development_spans = spans_from_time_ids(unique_times, row_offsets, plan.development_time_ids)
    holdout_spans = spans_from_time_ids(unique_times, row_offsets, plan.holdout_time_ids)
    holdout_patches = build_cold_start_patch(
        matrix_path,
        unique_times=unique_times,
        time_counts=time_counts,
        row_offsets=row_offsets,
        session_time_ids=plan.holdout_time_ids,
        raw_features=raw_features,
        history_features=history_features,
        rolling_windows=rolling_windows,
        model_cols=model_cols,
    )
    log(f"constructing development dataset with {span_length(development_spans):,} rows")
    development_set = build_dataset(
        matrix_path,
        matrix_shape,
        target_path,
        weight_path,
        development_spans,
        model_cols,
    )
    holdout_model = train_fixed(
        development_set,
        winning_candidate,
        SEEDS[0],
        mean_iterations,
        num_threads,
    )
    holdout_stats = prediction_statistics(
        holdout_model,
        matrix_path,
        target_path,
        weight_path,
        holdout_spans,
        mean_iterations,
        patches=holdout_patches,
    )
    holdout_raw = score_from_stats(holdout_stats)
    log(f"holdout_raw={holdout_raw:.8g}")
    del holdout_model, development_set
    gc.collect()

    gates = evaluate_gates(
        oof_raw_score=oof_raw,
        holdout_raw_score=holdout_raw,
        fitted_oof_scale=fitted_oof_scale,
    )

    all_spans = [(0, int(metadata["total_rows"]))]
    log(f"constructing final all-train dataset with {metadata['total_rows']:,} rows")
    all_set = build_dataset(
        matrix_path,
        matrix_shape,
        target_path,
        weight_path,
        all_spans,
        model_cols,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_files: list[str] = []
    best_iterations: list[int] = []
    for seed in SEEDS:
        log(f"final fit seed={seed} rounds={mean_iterations} candidate={winner['name']}")
        booster = train_fixed(all_set, winning_candidate, seed, mean_iterations, num_threads)
        name = f"model_seed{seed}.txt"
        booster.save_model(str(model_dir / name))
        model_files.append(name)
        best_iterations.append(mean_iterations)
        del booster
        gc.collect()
    del all_set
    gc.collect()

    report = {
        "strategy": "lightgbm_baseline",
        "schema_version": 3,
        "optimization_profile": "lgbm_low_risk_v1",
        "tuning_policy": "purged_walk_forward_pre_registered_candidates_no_test_tuning",
        "scale_policy": "diagnostic_only_never_apply",
        "execution": {
            "data_backend": "disk_memmap_lightgbm_sequence",
            "semantics": "full_readme_defaults",
            "cache_dir": str(cache_dir.resolve()),
        },
        "rows": {
            "train_all": int(metadata["total_rows"]),
            "oof": int(sum(span_length(spans_from_time_ids(unique_times, row_offsets, fold.valid_time_ids)) for fold in plan.folds)),
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
            "min_train_fraction": MIN_TRAIN_FRACTION,
            "feature_fit_time_count": int(len(plan.feature_fit_time_ids)),
            "inference_simulation": "cold_start_causal_time_order",
            "rounds_aggregation": "mean",
            "selection_metric": "mean_fold_score",
            "tie_break": ["stronger_regularization", "fewer_mean_iterations"],
            "candidates": candidate_results,
            "selected_candidate": winner["name"],
            "fold_scores": winner["fold_scores"],
            "fold_best_iterations": winner["fold_best_iterations"],
            "mean_iterations": mean_iterations,
            "mean_fold_score": winner["mean_fold_score"],
            "std_fold_score": winner["std_fold_score"],
            "min_fold_score": winner["min_fold_score"],
            "latest_fold_score": winner["latest_fold_score"],
            "oof_raw": oof_raw,
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
        "best_iterations": best_iterations,
        "prediction_scale": 1.0,
        "fitted_oof_scale": fitted_oof_scale,
        "gates_passed": gates["gates_passed"],
        "selected_candidate": winner["name"],
        "num_threads": int(num_threads),
        "lgbm_params": _candidate_params(SEEDS[0], winning_candidate, num_threads=num_threads),
    }
    (model_dir / "lightgbm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"training complete; models written to {model_dir}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-bounded execution of the official LightGBM baseline.")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_training(
        Path(args.release_root),
        Path(args.model_dir),
        Path(args.cache_dir),
        int(args.num_threads),
    )
    print(
        json.dumps(
            {
                "gates_passed": report["gates_passed"],
                "selected_candidate": report["selected_candidate"],
                "mean_iterations": report["best_iteration"],
                "model_dir": args.model_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
