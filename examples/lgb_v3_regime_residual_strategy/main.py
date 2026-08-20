from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from regime_features import RegimeEntityFeatureBuilder


def load_v3_model(strategy_dir: Path, model_dir: Path):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location(
        "_regime_residual_v3_base", main_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import V3 strategy: {main_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    old_model_dir = os.environ.get("LIGHTGBM_BASELINE_MODEL_DIR")
    sys.path.insert(0, str(strategy_dir))
    os.environ["LIGHTGBM_BASELINE_MODEL_DIR"] = str(model_dir)
    try:
        spec.loader.exec_module(module)
        return module.Model()
    finally:
        sys.path = old_path
        if old_model_dir is None:
            os.environ.pop("LIGHTGBM_BASELINE_MODEL_DIR", None)
        else:
            os.environ["LIGHTGBM_BASELINE_MODEL_DIR"] = old_model_dir


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        model_dir = root / "model"
        self.metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        base_strategy_dir = (root / self.metadata["base_strategy_dir"]).resolve()
        base_model_dir = (root / self.metadata["base_model_dir"]).resolve()
        self.base = load_v3_model(base_strategy_dir, base_model_dir)
        self.state_features = list(self.metadata["state_features"])
        self.builder = RegimeEntityFeatureBuilder(self.state_features)
        self.feature_indices = np.asarray(
            self.metadata["residual_model_feature_indices"], dtype=np.int64
        )
        self.residual_weight = float(self.metadata.get("residual_weight", 0.0))
        model_files = list(self.metadata.get("residual_models", []))
        self.residual_models = [
            lgb.Booster(model_file=str(model_dir / filename))
            for filename in model_files
        ]
        if self.residual_weight > 0.0 and not self.residual_models:
            raise ValueError("positive residual_weight requires residual models")
        expected_names = list(self.metadata["residual_model_feature_names"])
        for model in self.residual_models:
            if list(model.feature_name()) != expected_names:
                raise ValueError("residual model feature schema mismatch")

    def predict(self, test):
        base_prediction = np.asarray(
            self.base.predict(test), dtype=np.float64
        )
        if self.residual_weight <= 0.0:
            return base_prediction
        raw = test.loc[:, self.state_features].to_numpy(
            dtype=np.float32, copy=True
        )
        assets = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        residual_features = self.builder.transform_time(
            assets, raw, base_prediction.astype(np.float32)
        )
        residual_x = residual_features[:, self.feature_indices]
        residual_prediction = np.mean(
            np.asarray([
                model.predict(residual_x) for model in self.residual_models
            ], dtype=np.float64),
            axis=0,
        )
        prediction = base_prediction + self.residual_weight * residual_prediction
        return np.nan_to_num(
            prediction, nan=0.0, posinf=0.0, neginf=0.0
        )
