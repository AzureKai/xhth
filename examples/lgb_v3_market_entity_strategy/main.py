from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from decomposition_features import (
    EntityFeatureBuilder,
    center_cross_section,
    cross_sectional_z,
)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def load_v3_model(strategy_dir: Path, model_dir: Path):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location(
        "_entity_residual_v3_base", main_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import V3 strategy: {main_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    old_model_dir = os.environ.get("LIGHTGBM_BASELINE_MODEL_DIR")
    old_threads = os.environ.get("LIGHTGBM_BASELINE_PREDICT_THREADS")
    sys.path.insert(0, str(strategy_dir))
    os.environ["LIGHTGBM_BASELINE_MODEL_DIR"] = str(model_dir)
    os.environ.setdefault("LIGHTGBM_BASELINE_PREDICT_THREADS", "1")
    try:
        spec.loader.exec_module(module)
        return module.Model()
    finally:
        sys.path = old_path
        if old_model_dir is None:
            os.environ.pop("LIGHTGBM_BASELINE_MODEL_DIR", None)
        else:
            os.environ["LIGHTGBM_BASELINE_MODEL_DIR"] = old_model_dir
        if old_threads is None:
            os.environ.pop("LIGHTGBM_BASELINE_PREDICT_THREADS", None)
        else:
            os.environ["LIGHTGBM_BASELINE_PREDICT_THREADS"] = old_threads


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        model_dir = root / "model"
        self.metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if int(self.metadata.get("schema_version", 0)) != 2:
            raise ValueError("entity-only model artifacts must be retrained")
        base_strategy_dir = (root / self.metadata["base_strategy_dir"]).resolve()
        base_model_dir = (root / self.metadata["base_model_dir"]).resolve()
        self.base = load_v3_model(base_strategy_dir, base_model_dir)
        self.state_features = list(self.metadata["state_features"])
        self.extra_cross_z_features = list(
            self.metadata["extra_cross_z_features"]
        )
        self.builder = EntityFeatureBuilder(self.state_features)
        self.entity_weight = float(self.metadata.get("entity_weight", 0.0))
        self.predict_threads = _positive_int_env(
            "LIGHTGBM_ENTITY_PREDICT_THREADS", 1
        )
        self.entity_models = [
            lgb.Booster(model_file=str(model_dir / filename))
            for filename in self.metadata.get("entity_models", [])
        ]
        if self.entity_weight > 0.0 and not self.entity_models:
            raise ValueError("positive entity_weight requires entity models")
        expected_names = list(self.metadata["entity_model_feature_names"])
        for model in self.entity_models:
            if list(model.feature_name()) != expected_names:
                raise ValueError("entity model feature schema mismatch")

    def predict(self, test):
        base_prediction = np.asarray(
            self.base.predict(test), dtype=np.float64
        )
        if self.entity_weight <= 0.0:
            return base_prediction

        state_raw = test.loc[:, self.state_features].to_numpy(
            dtype=np.float32, copy=True
        )
        assets = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        entity_features = self.builder.transform_time(
            assets, state_raw, base_prediction.astype(np.float32)
        )
        if self.extra_cross_z_features:
            extra_raw = test.loc[:, self.extra_cross_z_features].to_numpy(
                dtype=np.float32, copy=True
            )
            entity_features = np.column_stack([
                entity_features,
                cross_sectional_z(extra_raw),
            ]).astype(np.float32, copy=False)
        entity_prediction = np.mean(np.asarray([
            model.predict(entity_features, num_threads=self.predict_threads)
            for model in self.entity_models
        ], dtype=np.float64), axis=0)
        prediction = (
            base_prediction
            + self.entity_weight * center_cross_section(entity_prediction)
        )
        return np.nan_to_num(
            prediction, nan=0.0, posinf=0.0, neginf=0.0
        )
