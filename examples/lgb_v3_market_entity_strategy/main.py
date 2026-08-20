from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from decomposition_features import (
    MarketEntityFeatureBuilder,
    center_cross_section,
)


def load_v3_model(strategy_dir: Path, model_dir: Path):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location(
        "_market_entity_v3_base", main_path
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
        self.builder = MarketEntityFeatureBuilder(self.state_features)
        self.market_weight = float(self.metadata.get("market_weight", 0.0))
        self.entity_weight = float(self.metadata.get("entity_weight", 0.0))
        self.entity_indices = np.asarray(
            self.metadata["entity_model_feature_indices"], dtype=np.int64
        )
        self.market_models = [
            lgb.Booster(model_file=str(model_dir / filename))
            for filename in self.metadata.get("market_models", [])
        ]
        self.entity_models = [
            lgb.Booster(model_file=str(model_dir / filename))
            for filename in self.metadata.get("entity_models", [])
        ]
        if self.market_weight > 0.0 and not self.market_models:
            raise ValueError("positive market_weight requires market models")
        if self.entity_weight > 0.0 and not self.entity_models:
            raise ValueError("positive entity_weight requires entity models")
        expected_market = list(self.metadata["market_feature_names"])
        expected_entity = list(self.metadata["entity_model_feature_names"])
        for model in self.market_models:
            if list(model.feature_name()) != expected_market:
                raise ValueError("market model feature schema mismatch")
        for model in self.entity_models:
            if list(model.feature_name()) != expected_entity:
                raise ValueError("entity model feature schema mismatch")

    def predict(self, test):
        base_prediction = np.asarray(
            self.base.predict(test), dtype=np.float64
        )
        if self.market_weight <= 0.0 and self.entity_weight <= 0.0:
            return base_prediction

        raw = test.loc[:, self.state_features].to_numpy(
            dtype=np.float32, copy=True
        )
        assets = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        entity_features = self.builder.transform_time(
            assets, raw, base_prediction.astype(np.float32)
        )
        prediction = base_prediction.copy()
        if self.market_weight > 0.0:
            market_x = self.builder.market_features(entity_features).reshape(1, -1)
            market_correction = float(np.mean([
                model.predict(market_x)[0] for model in self.market_models
            ]))
            prediction += self.market_weight * market_correction
        if self.entity_weight > 0.0:
            entity_x = entity_features[:, self.entity_indices]
            entity_correction = np.mean(np.asarray([
                model.predict(entity_x) for model in self.entity_models
            ], dtype=np.float64), axis=0)
            prediction += self.entity_weight * center_cross_section(
                entity_correction
            )
        return np.nan_to_num(
            prediction, nan=0.0, posinf=0.0, neginf=0.0
        )
