from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Model:
    def __init__(self):
        strategy_dir = Path(__file__).resolve().parent
        model_dir = strategy_dir / "model"
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = list(metadata["feature_columns"])
        self.ema_feature_columns = list(metadata.get("ema_feature_columns", []))
        self.ema_halflives = [float(value) for value in metadata.get("ema_halflives", [])]
        self.input_columns = list(metadata["input_columns"])
        self.prediction_scale = float(metadata.get("prediction_scale", 1.0))
        self.clip_min = float(metadata.get("clip_min", -np.inf))
        self.clip_max = float(metadata.get("clip_max", np.inf))
        self.alphas = np.asarray([1.0 - 0.5 ** (1.0 / value) for value in self.ema_halflives], dtype=np.float32)
        self.state: dict[int, np.ndarray] = {}
        self.initialized: dict[int, np.ndarray] = {}
        self.models: list[tuple[float, str, Any]] = []

        for spec in metadata.get("models", []):
            weight = float(spec.get("weight", 0.0))
            if weight == 0.0:
                continue
            model_type = str(spec["type"])
            model_path = model_dir / str(spec["file"])
            if model_type == "lightgbm":
                import lightgbm as lgb

                model = lgb.Booster(model_file=str(model_path))
            elif model_type == "catboost":
                from catboost import CatBoostRegressor

                model = CatBoostRegressor()
                model.load_model(str(model_path))
            else:
                raise ValueError(f"unsupported model type: {model_type}")
            self.models.append((weight, model_type, model))
        if not self.models:
            raise ValueError("metadata does not define any enabled models")
        self.last_time_id: int | None = None

    def _ema_matrix(self, test) -> np.ndarray:
        if not self.ema_feature_columns:
            return np.empty((len(test), 0), dtype=np.float32)

        values = test.loc[:, self.ema_feature_columns].to_numpy(dtype=np.float32, copy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        width = len(self.ema_feature_columns)
        out_width = width * len(self.ema_halflives) + (width if len(self.ema_halflives) >= 2 else 0)
        out = np.empty((len(test), out_width), dtype=np.float32)

        for row_idx, asset_id_raw in enumerate(asset_ids):
            asset_id = int(asset_id_raw)
            current = values[row_idx]
            if asset_id not in self.state:
                self.state[asset_id] = np.tile(current, (len(self.ema_halflives), 1)).astype(np.float32)
                self.initialized[asset_id] = np.ones(width, dtype=bool)
            previous = self.state[asset_id].copy()
            pieces = []
            for half_idx in range(len(self.ema_halflives)):
                pieces.append(current - previous[half_idx])
            if len(self.ema_halflives) >= 2:
                pieces.append(previous[0] - previous[-1])
            out[row_idx] = np.concatenate(pieces).astype(np.float32)
            for half_idx, alpha in enumerate(self.alphas):
                self.state[asset_id][half_idx] = alpha * current + (1.0 - alpha) * self.state[asset_id][half_idx]
        return out

    def _build_matrix(self, test):
        missing = [col for col in self.feature_columns if col not in test.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")
        raw = test.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        asset = test["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        return np.hstack([raw, self._ema_matrix(test), asset])

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        x = self._build_matrix(test)
        pred = np.zeros(len(test), dtype=np.float64)
        total_weight = 0.0
        for weight, model_type, model in self.models:
            current = model.predict(x)
            pred += weight * np.asarray(current, dtype=np.float64)
            total_weight += weight
        if total_weight > 0:
            pred /= total_weight
        pred *= self.prediction_scale
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(pred, self.clip_min, self.clip_max)
