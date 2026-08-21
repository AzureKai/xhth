from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def entity_feature_names(state_features: list[str]) -> list[str]:
    return [
        "asset_id",
        *(f"raw_{name}" for name in state_features),
        "base_prediction",
        *(f"entity_z20_{name}" for name in state_features),
        *(f"entity_ema_gap20_60_{name}" for name in state_features),
        *(f"cross_z_{name}" for name in state_features),
        "entity_history_log",
    ]


def cross_sectional_z(values: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if matrix.ndim != 2:
        raise ValueError("cross-sectional values must be a matrix")
    if len(matrix) == 0:
        return matrix.copy()
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.maximum(np.std(matrix, axis=0, dtype=np.float64), 1e-3)
    return np.clip((matrix - mean) / scale, -8.0, 8.0).astype(np.float32)


def entity_residual_target(
    target: np.ndarray,
    base_prediction: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray(target, dtype=np.float64)
    base_values = np.asarray(base_prediction, dtype=np.float64)
    if target_values.shape != base_values.shape or target_values.ndim != 1:
        raise ValueError("target and base_prediction must be aligned vectors")
    if len(target_values) == 0:
        raise ValueError("cannot center an empty cross-section")
    result = (
        target_values - np.mean(target_values, dtype=np.float64)
        - base_values + np.mean(base_values, dtype=np.float64)
    )
    return result.astype(np.float32)


def center_cross_section(values: np.ndarray) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.ndim != 1:
        raise ValueError("cross-sectional prediction must be a vector")
    if len(prediction) == 0:
        return prediction.copy()
    return prediction - np.mean(prediction, dtype=np.float64)


@dataclass
class EntityState:
    count: int
    ema20: np.ndarray
    ema60: np.ndarray
    second20: np.ndarray


class EntityFeatureBuilder:
    """Causal per-asset state plus current cross-sectional normalization."""

    def __init__(self, state_features: list[str]):
        if not state_features:
            raise ValueError("state_features must not be empty")
        self.state_features = list(state_features)
        self.feature_names = entity_feature_names(self.state_features)
        self.states: dict[int, EntityState] = {}
        self.alpha20 = np.float32(2.0 / 21.0)
        self.alpha60 = np.float32(2.0 / 61.0)

    def transform_time(
        self,
        asset_ids: np.ndarray,
        raw_values: np.ndarray,
        base_prediction: np.ndarray,
    ) -> np.ndarray:
        assets = np.asarray(asset_ids, dtype=np.int64)
        raw = np.nan_to_num(
            np.asarray(raw_values, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base = np.nan_to_num(
            np.asarray(base_prediction, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if raw.ndim != 2 or raw.shape != (len(assets), len(self.state_features)):
            raise ValueError("raw_values shape does not match state_features")
        if base.shape != (len(assets),):
            raise ValueError("base_prediction shape does not match asset_ids")

        entity_z = np.zeros_like(raw)
        entity_gap = np.zeros_like(raw)
        history_log = np.zeros(len(raw), dtype=np.float32)
        pending: list[tuple[int, np.ndarray, EntityState | None]] = []
        for row, (asset, current) in enumerate(zip(assets, raw)):
            previous = self.states.get(int(asset))
            pending.append((int(asset), current.copy(), previous))
            if previous is None:
                continue
            variance = np.maximum(
                previous.second20 - previous.ema20 * previous.ema20,
                1e-4,
            )
            scale = np.sqrt(variance).astype(np.float32)
            reliability = np.float32(min(previous.count / 20.0, 1.0))
            entity_z[row] = reliability * np.clip(
                (current - previous.ema20) / scale, -8.0, 8.0
            )
            entity_gap[row] = reliability * np.clip(
                (previous.ema20 - previous.ema60) / scale, -8.0, 8.0
            )
            history_log[row] = np.float32(np.log1p(previous.count))

        output = np.column_stack([
            assets.astype(np.float32),
            raw,
            base,
            entity_z,
            entity_gap,
            cross_sectional_z(raw),
            history_log,
        ]).astype(np.float32, copy=False)

        for asset, current, previous in pending:
            if previous is None:
                self.states[asset] = EntityState(
                    count=1,
                    ema20=current.copy(),
                    ema60=current.copy(),
                    second20=current * current,
                )
                continue
            previous.ema20 = (
                (1.0 - self.alpha20) * previous.ema20
                + self.alpha20 * current
            )
            previous.ema60 = (
                (1.0 - self.alpha60) * previous.ema60
                + self.alpha60 * current
            )
            previous.second20 = (
                (1.0 - self.alpha20) * previous.second20
                + self.alpha20 * current * current
            )
            previous.count += 1
        return output
