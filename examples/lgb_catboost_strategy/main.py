from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Model:
    def __init__(self):
        strategy_dir = Path(__file__).resolve().parent
        model_dir = strategy_dir / "model"
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing model metadata: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self.use_asset_id = bool(self.metadata.get("use_asset_id", True))
        self.input_columns = list(self.feature_columns)
        if self.use_asset_id:
            self.input_columns.append("asset_id")

        self.clip_min = float(self.metadata.get("clip_min", -np.inf))
        self.clip_max = float(self.metadata.get("clip_max", np.inf))
        self.models: list[tuple[float, str, Any]] = []

        for spec in self.metadata.get("models", []):
            if float(spec.get("weight", 0.0)) == 0.0:
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
            self.models.append((float(spec["weight"]), model_type, model))

        if not self.models:
            raise ValueError("metadata.json does not define any enabled models")

        self.last_time_id: int | None = None

    def _build_matrix(self, test):
        missing = [col for col in self.feature_columns if col not in test.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")

        matrix = test.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_asset_id:
            asset = test["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
            matrix = np.hstack([matrix, asset])
        return matrix

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        x = self._build_matrix(test)
        prediction = np.zeros(len(test), dtype=np.float64)
        total_weight = 0.0
        for weight, model_type, model in self.models:
            if model_type == "lightgbm":
                current = model.predict(x)
            elif model_type == "catboost":
                current = model.predict(x)
            else:
                raise ValueError(f"unsupported model type: {model_type}")
            prediction += weight * np.asarray(current, dtype=np.float64)
            total_weight += weight

        if total_weight > 0:
            prediction /= total_weight
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(prediction, self.clip_min, self.clip_max)
