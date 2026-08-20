from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGIME_FEATURE_NAMES = [
    "regime_entity_abs_z",
    "regime_trend_strength",
    "regime_cross_dispersion",
    "regime_shock_rate",
    "regime_base_abs_mean",
    "regime_base_std",
]


def residual_feature_names(state_features: list[str]) -> list[str]:
    return [
        "asset_id",
        *(f"raw_{name}" for name in state_features),
        "base_prediction",
        *(f"entity_z20_{name}" for name in state_features),
        *(f"entity_ema_gap20_60_{name}" for name in state_features),
        *(f"cross_z_{name}" for name in state_features),
        "entity_history_log",
        *REGIME_FEATURE_NAMES,
        "regime_id",
    ]


def state_only_indices(names: list[str]) -> list[int]:
    regime = {*REGIME_FEATURE_NAMES, "regime_id"}
    return [index for index, name in enumerate(names) if name not in regime]


@dataclass
class EntityState:
    count: int
    ema20: np.ndarray
    ema60: np.ndarray
    second20: np.ndarray


class RegimeEntityFeatureBuilder:
    """Causal observable state shared by training and sequential inference."""

    def __init__(self, state_features: list[str]):
        if not state_features:
            raise ValueError("state_features must not be empty")
        self.state_features = list(state_features)
        self.feature_names = residual_feature_names(self.state_features)
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
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        base = np.nan_to_num(
            np.asarray(base_prediction, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
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

        cross_mean = np.mean(raw, axis=0, dtype=np.float64)
        cross_scale = np.maximum(
            np.std(raw, axis=0, dtype=np.float64), 1e-3
        )
        cross_z = np.clip(
            (raw - cross_mean) / cross_scale, -8.0, 8.0
        ).astype(np.float32)

        entity_abs_z = float(np.mean(np.abs(entity_z)))
        trend_strength = float(np.mean(np.abs(entity_gap)))
        cross_dispersion = float(np.mean(np.abs(cross_z)))
        shock_rate = float(np.mean(np.abs(entity_z) >= 2.0))
        base_abs_mean = float(np.mean(np.abs(base)))
        base_std = float(np.std(base, dtype=np.float64))
        regime_values = np.asarray([
            entity_abs_z,
            trend_strength,
            cross_dispersion,
            shock_rate,
            base_abs_mean,
            base_std,
        ], dtype=np.float32)
        if shock_rate >= 0.15 or entity_abs_z >= 1.50:
            regime_id = 3
        elif trend_strength >= 0.75:
            regime_id = 2
        elif base_std >= max(1e-6, 1.25 * base_abs_mean):
            regime_id = 1
        else:
            regime_id = 0

        repeated_regime = np.repeat(
            regime_values.reshape(1, -1), len(raw), axis=0
        )
        output = np.column_stack([
            assets.astype(np.float32),
            raw,
            base,
            entity_z,
            entity_gap,
            cross_z,
            history_log,
            repeated_regime,
            np.full(len(raw), regime_id, dtype=np.float32),
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
