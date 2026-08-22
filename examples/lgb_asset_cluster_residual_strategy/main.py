from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from cluster_features import center_by_time


def load_base_model(strategy_dir: Path, model_dir: Path):
    main_path = strategy_dir / "main.py"
    spec = importlib.util.spec_from_file_location(
        "_asset_cluster_c4_base", main_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import C4 strategy: {main_path}")
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
        self.metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if int(self.metadata.get("schema_version", 0)) != 1:
            raise ValueError("asset-cluster artifacts must use schema_version=1")
        base_strategy_dir = (root / self.metadata["base_strategy_dir"]).resolve()
        base_model_dir = (root / self.metadata["base_model_dir"]).resolve()
        self.base = load_base_model(base_strategy_dir, base_model_dir)
        self.features = list(self.metadata["selected_features"])
        self.clusters = int(self.metadata["cluster_count"])
        self.residual_scale = float(self.metadata["deployment_residual_scale"])
        if not 0.0 <= self.residual_scale <= 2.0:
            raise ValueError("deployment residual scale is outside [0,2]")
        self.asset_clusters = {
            int(asset): int(cluster)
            for asset, cluster in self.metadata["asset_clusters"].items()
        }
        self.cluster_models: dict[int, list] = {
            cluster: [] for cluster in range(self.clusters)
        }
        if self.residual_scale > 0.0:
            import lightgbm as lgb

            for filename in self.metadata.get("cluster_models", []):
                cluster = int(filename.split("_")[1])
                self.cluster_models[cluster].append(
                    lgb.Booster(model_file=str(model_dir / filename))
                )
            if any(not models for models in self.cluster_models.values()):
                raise ValueError("one or more deployed clusters have no models")
            expected_features = len(self.features) + 1
            for models in self.cluster_models.values():
                for model in models:
                    if model.num_feature() != expected_features:
                        raise ValueError(
                            f"cluster model expects {model.num_feature()} features; "
                            f"metadata defines {expected_features}"
                        )
        self.calls = 0
        self.started = time.perf_counter()
        self.progress_every = max(
            1, int(os.environ.get("CLUSTER_INFERENCE_PROGRESS_EVERY", "1000"))
        )

    def predict(self, test):
        base_prediction = np.asarray(self.base.predict(test), dtype=np.float64)
        if self.residual_scale <= 0.0:
            return base_prediction
        raw = test.loc[:, self.features].to_numpy(dtype=np.float32, copy=True)
        residual_x = np.hstack([
            np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0),
            base_prediction.astype(np.float32).reshape(-1, 1),
        ])
        assets = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        missing = sorted(set(map(int, np.unique(assets))) - set(self.asset_clusters))
        if missing:
            raise ValueError(f"cluster mapping is missing assets: {missing}")
        routing = np.fromiter(
            (self.asset_clusters[int(asset)] for asset in assets),
            dtype=np.int16,
            count=len(assets),
        )
        residual = np.zeros(len(test), dtype=np.float64)
        for cluster, models in self.cluster_models.items():
            rows = np.flatnonzero(routing == cluster)
            if not len(rows):
                continue
            residual[rows] = np.mean(
                [model.predict(residual_x[rows]) for model in models], axis=0
            )
        times = test["time_id"].to_numpy(dtype=np.int64, copy=False)
        residual = center_by_time(residual, times)
        prediction = base_prediction + self.residual_scale * residual
        self.calls += 1
        if self.calls % self.progress_every == 0:
            elapsed = time.perf_counter() - self.started
            print(
                f"[cluster inference] calls={self.calls:,} "
                f"time_id={int(times[-1])} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
