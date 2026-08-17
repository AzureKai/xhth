from __future__ import annotations

import numpy as np


TEMPORAL_SUFFIXES = [
    "lag1",
    "lag5",
    "lag20",
    "delta1",
    "delta5",
    "delta20",
    "ema5",
    "ema20",
    "ema60",
    "minus_ema20",
    "minus_ema60",
    "rolling_std20",
    "rolling_std60",
    "historical_zscore20",
    "historical_zscore60",
    "xs_rank",
    "xs_rank_delta1",
    # Baseline-compatible transforms. Keep these at the end so persisted
    # indices for the older temporal transforms remain stable.
    "diff1",
    "rmean5",
]

# Keep the historical fallback unchanged; the baseline-compatible transforms
# are enabled explicitly by baseline_468_feature_plan.json. Keep the complete
# suffix registry so old metadata and explicitly routed plans remain readable.
DEFAULT_TEMPORAL_SUFFIXES = [
    value for value in TEMPORAL_SUFFIXES
    if value not in {"delta1", "xs_rank_delta1", "diff1", "rmean5"}
]


def temporal_column_names(features: list[str], recipes: dict[str, list[str]] | None = None) -> list[str]:
    return [
        f"ts_{suffix}_{feature}"
        for feature in features
        for suffix in (recipes.get(feature, []) if recipes is not None else TEMPORAL_SUFFIXES)
    ]


def cross_section_rank(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    result = np.full((rows, columns), 0.5, dtype=np.float32)
    finite = np.isfinite(values)
    for column in range(columns):
        valid = finite[:, column]
        count = int(valid.sum())
        if count == 0:
            continue
        if count == 1:
            result[valid, column] = 1.0
            continue
        current = values[valid, column]
        order = np.argsort(current, kind="mergesort")
        sorted_values = current[order]
        sorted_ranks = np.empty(count, dtype=np.float32)
        start = 0
        while start < count:
            end = start + 1
            while end < count and sorted_values[end] == sorted_values[start]:
                end += 1
            sorted_ranks[start:end] = (((start + 1.0) + end) * 0.5) / count
            start = end
        ranks = np.empty(count, dtype=np.float32)
        ranks[order] = sorted_ranks
        result[valid, column] = ranks
    return result


class TemporalFeatureBuilder:
    """Maintain strictly historical per-asset state for selected raw features."""

    def __init__(self, feature_count: int, window: int = 60,
                 feature_names: list[str] | None = None,
                 recipes: dict[str, list[str]] | None = None):
        self.feature_count = int(feature_count)
        self.window = int(window)
        self.states: dict[int, dict[str, np.ndarray | int]] = {}
        self.feature_names = feature_names
        self.recipes = recipes
        self.output_indices = None
        if feature_names is not None and recipes is not None:
            self.output_indices = np.asarray(
                [
                    feature_index * len(TEMPORAL_SUFFIXES) + TEMPORAL_SUFFIXES.index(suffix)
                    for feature_index, feature in enumerate(feature_names)
                    for suffix in recipes.get(feature, [])
                ],
                dtype=np.int64,
            )

    def _new_state(self):
        return {
            "buffer": np.full((self.window, self.feature_count), np.nan, dtype=np.float32),
            "position": 0,
            "steps": 0,
            "sum20": np.zeros(self.feature_count, dtype=np.float64),
            "sum_sq20": np.zeros(self.feature_count, dtype=np.float64),
            "count20": np.zeros(self.feature_count, dtype=np.int32),
            "sum60": np.zeros(self.feature_count, dtype=np.float64),
            "sum_sq60": np.zeros(self.feature_count, dtype=np.float64),
            "count60": np.zeros(self.feature_count, dtype=np.int32),
            # Sum of the previous four cleaned observations. Together with
            # the current observation this reproduces baseline rmean5.
            "sum4": np.zeros(self.feature_count, dtype=np.float64),
            "ema5": np.full(self.feature_count, np.nan, dtype=np.float32),
            "ema20": np.full(self.feature_count, np.nan, dtype=np.float32),
            "ema60": np.full(self.feature_count, np.nan, dtype=np.float32),
            "xs_rank": np.full(self.feature_count, np.nan, dtype=np.float32),
        }

    def transform(self, asset_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[1] != self.feature_count:
            raise ValueError("temporal feature count does not match state")
        ranks = cross_section_rank(values)
        full_width = self.feature_count * len(TEMPORAL_SUFFIXES)
        output_width = full_width if self.output_indices is None else len(self.output_indices)
        output = np.zeros((len(values), output_width), dtype=np.float32)

        for row, asset_value in enumerate(asset_ids):
            asset_id = int(asset_value)
            state = self.states.get(asset_id)
            if state is None:
                state = self._new_state()
                self.states[asset_id] = state
            buffer = state["buffer"]
            position = int(state["position"])
            steps = int(state["steps"])
            current = values[row]
            lag1 = buffer[(position - 1) % self.window].copy() if steps >= 1 else np.full(self.feature_count, np.nan)
            lag5 = buffer[(position - 5) % self.window].copy() if steps >= 5 else np.full(self.feature_count, np.nan)
            lag20 = buffer[(position - 20) % self.window].copy() if steps >= 20 else np.full(self.feature_count, np.nan)
            count20 = state["count20"].astype(np.float64)
            mean20 = np.divide(state["sum20"], count20, out=np.zeros(self.feature_count), where=count20 > 0)
            variance20 = np.divide(state["sum_sq20"], count20, out=np.zeros(self.feature_count), where=count20 > 0) - mean20 * mean20
            std20 = np.sqrt(np.maximum(variance20, 0.0))
            count60 = state["count60"].astype(np.float64)
            mean60 = np.divide(state["sum60"], count60, out=np.zeros(self.feature_count), where=count60 > 0)
            variance60 = np.divide(state["sum_sq60"], count60, out=np.zeros(self.feature_count), where=count60 > 0) - mean60 * mean60
            std60 = np.sqrt(np.maximum(variance60, 0.0))
            ema5 = state["ema5"].copy()
            ema20 = state["ema20"].copy()
            ema60 = state["ema60"].copy()
            previous_rank = state["xs_rank"].copy()
            clean_current = np.nan_to_num(
                current, nan=0.0, posinf=0.0, neginf=0.0
            )
            clean_lag1 = np.nan_to_num(
                lag1, nan=0.0, posinf=0.0, neginf=0.0
            )
            rmean5 = (
                state["sum4"] + clean_current
            ) / float(min(steps, 4) + 1)

            matrix = np.column_stack(
                [
                    lag1,
                    lag5,
                    lag20,
                    current - lag1,
                    current - lag5,
                    current - lag20,
                    ema5,
                    ema20,
                    ema60,
                    current - ema20,
                    current - ema60,
                    std20,
                    std60,
                    np.divide(current - mean20, std20, out=np.zeros(self.feature_count), where=std20 > 1e-6),
                    np.divide(current - mean60, std60, out=np.zeros(self.feature_count), where=std60 > 1e-6),
                    ranks[row],
                    ranks[row] - previous_rank,
                    clean_current - clean_lag1,
                    rmean5,
                ]
            )
            row_output = np.nan_to_num(
                matrix, nan=0.0, posinf=0.0, neginf=0.0
            ).reshape(-1)
            output[row] = (
                row_output
                if self.output_indices is None
                else row_output[self.output_indices]
            )

            old60 = buffer[position].copy()
            old60_valid = np.isfinite(old60) & (steps >= self.window)
            state["sum60"][old60_valid] -= old60[old60_valid]
            state["sum_sq60"][old60_valid] -= old60[old60_valid] * old60[old60_valid]
            state["count60"][old60_valid] -= 1
            old20 = buffer[(position - 20) % self.window].copy()
            old20_valid = np.isfinite(old20) & (steps >= 20)
            state["sum20"][old20_valid] -= old20[old20_valid]
            state["sum_sq20"][old20_valid] -= old20[old20_valid] * old20[old20_valid]
            state["count20"][old20_valid] -= 1
            if steps >= 4:
                old4 = buffer[(position - 4) % self.window].copy()
                state["sum4"] -= np.nan_to_num(
                    old4, nan=0.0, posinf=0.0, neginf=0.0
                )
            state["sum4"] += clean_current
            current_valid = np.isfinite(current)
            buffer[position] = current
            for suffix in ("20", "60"):
                state[f"sum{suffix}"][current_valid] += current[current_valid]
                state[f"sum_sq{suffix}"][current_valid] += current[current_valid] * current[current_valid]
                state[f"count{suffix}"][current_valid] += 1
            state["position"] = (position + 1) % self.window
            state["steps"] = steps + 1
            for key, alpha in (("ema5", 2.0 / 6.0), ("ema20", 2.0 / 21.0), ("ema60", 2.0 / 61.0)):
                previous = state[key]
                initialize = current_valid & ~np.isfinite(previous)
                update = current_valid & np.isfinite(previous)
                previous[initialize] = current[initialize]
                previous[update] = alpha * current[update] + (1.0 - alpha) * previous[update]
            state["xs_rank"] = ranks[row].copy()
        return output
