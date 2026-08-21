from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_strategy_model(strategy_dir: Path, module_name: str):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import source strategy: {main_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(strategy_dir))
    try:
        spec.loader.exec_module(module)
        return module.Model()
    finally:
        sys.path = old_path


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        metadata = json.loads(
            (root / "model" / "metadata.json").read_text(encoding="utf-8")
        )
        if int(metadata.get("schema_version", 0)) != 1:
            raise ValueError("C4/entity fusion metadata must use schema_version=1")
        self.entity_weight = float(metadata["deployment_entity_weight"])
        if not 0.0 <= self.entity_weight <= 1.0:
            raise ValueError("deployment_entity_weight must be in [0, 1]")
        c4_dir = (root / metadata["c4_strategy_dir"]).resolve()
        entity_dir = (root / metadata["entity_strategy_dir"]).resolve()
        self.c4 = (
            load_strategy_model(c4_dir, "_fusion_c4_source")
            if self.entity_weight < 1.0 else None
        )
        self.entity = (
            load_strategy_model(entity_dir, "_fusion_entity_source")
            if self.entity_weight > 0.0 else None
        )

    def predict(self, test):
        if self.entity_weight <= 0.0:
            return np.asarray(self.c4.predict(test), dtype=np.float64)
        if self.entity_weight >= 1.0:
            return np.asarray(self.entity.predict(test), dtype=np.float64)
        c4_prediction = np.asarray(self.c4.predict(test), dtype=np.float64)
        entity_prediction = np.asarray(self.entity.predict(test), dtype=np.float64)
        prediction = (
            (1.0 - self.entity_weight) * c4_prediction
            + self.entity_weight * entity_prediction
        )
        return np.nan_to_num(
            prediction, nan=0.0, posinf=0.0, neginf=0.0
        )
