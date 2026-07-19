from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Model:
    def __init__(self):
        model_dir = Path(__file__).resolve().parent / "model"
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing model metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self.ema_feature_columns = list(self.metadata.get("ema_feature_columns", []))
        self.ema_halflives = [float(value) for value in self.metadata.get("ema_halflives", [])]
        self.prediction_scale = float(self.metadata.get("prediction_scale", 1.0))
        self.clip_min = float(self.metadata.get("clip_min", -np.inf))
        self.clip_max = float(self.metadata.get("clip_max", np.inf))
        self.alphas = np.asarray([1.0 - 0.5 ** (1.0 / value) for value in self.ema_halflives], dtype=np.float32)
        self.state: dict[int, np.ndarray] = {}
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(model_dir / self.metadata.get("model_file", "final_lightgbm.txt")))
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
            previous = self.state[asset_id].copy()
            pieces = [current - previous[half_idx] for half_idx in range(len(self.ema_halflives))]
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
        x = test.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        asset = test["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        return np.hstack([x, self._ema_matrix(test), asset])

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id
        pred = np.asarray(self.model.predict(self._build_matrix(test)), dtype=np.float64)
        pred *= self.prediction_scale
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(pred, self.clip_min, self.clip_max)
