from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch


def load_base_model(strategy_dir, model_dir):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location("_tcn_base_strategy", main_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import base strategy from {main_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(strategy_dir))
    try:
        spec.loader.exec_module(module)
        return module.Model(model_dir=model_dir)
    finally:
        sys.path = old_path


class Model:
    def __init__(self):
        root = Path(__file__).resolve().parent
        model_dir = root / "model"
        metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        base_strategy = (root / metadata["base_strategy_dir"]).resolve()
        base_model_dir = Path(metadata["base_model_dir"])
        if not base_model_dir.is_absolute():
            base_model_dir = (root / base_model_dir).resolve()
        self.base = load_base_model(base_strategy, base_model_dir)
        self.features = list(metadata["feature_columns"])
        self.mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(metadata["feature_scale"], dtype=np.float32)
        self.sequence_length = int(metadata["sequence_length"])
        self.alpha = float(metadata["fusion_weight"])
        self.tcn = torch.jit.load(
            str(model_dir / metadata["tcn_model"]), map_location="cpu"
        ).eval()
        self.history = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )

    def predict(self, test):
        base_prediction = np.asarray(self.base.predict(test), dtype=np.float64)
        raw = test.loc[:, self.features].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = np.clip(
            (raw - self.mean) / self.scale, -8.0, 8.0
        ).astype(np.float32)
        assets = test["asset_id"].to_numpy(dtype=np.int64)
        batch = np.zeros(
            (len(test), len(self.features) + 1, self.sequence_length),
            dtype=np.float32,
        )
        for row, (asset, values) in enumerate(zip(assets, normalized)):
            history = self.history[int(asset)]
            history.append(values)
            stacked = np.asarray(history, dtype=np.float32)
            length = len(stacked)
            batch[row, :-1, -length:] = stacked.T
            batch[row, -1, -length:] = 1.0
        with torch.no_grad():
            temporal = self.tcn(torch.from_numpy(batch)).numpy()
        return base_prediction + self.alpha * (temporal - base_prediction)
