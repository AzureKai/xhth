from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Model:
    def __init__(self):
        import lightgbm as lgb

        model_dir = Path(__file__).resolve().parent / "model"
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.features = list(self.metadata["feature_columns"])
        self.responders = list(self.metadata["responders"])
        self.responder_models = [
            lgb.Booster(model_file=str(model_dir / self.metadata["responder_models"][name]))
            for name in self.responders
        ]
        self.target_model = lgb.Booster(model_file=str(model_dir / self.metadata["target_model"]))
        self.scale = float(self.metadata.get("prediction_scale", 1.0))
        self.clip_min = float(self.metadata.get("clip_min", -np.inf))
        self.clip_max = float(self.metadata.get("clip_max", np.inf))
        self.last_time_id: int | None = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase")
        self.last_time_id = time_id

        x = test.loc[:, self.features].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        asset = test["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        base = np.hstack([x, asset])
        responder_hat = np.column_stack([model.predict(base) for model in self.responder_models]).astype(np.float32)
        target_x = np.hstack([base, responder_hat])
        prediction = np.asarray(self.target_model.predict(target_x), dtype=np.float64) * self.scale
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(prediction, self.clip_min, self.clip_max)
