from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporal_features import TemporalFeatureBuilder


class Model:
    def __init__(self):
        import lightgbm as lgb

        model_dir = Path(__file__).resolve().parent / "model"
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.features = list(self.metadata["feature_columns"])
        self.responders = list(self.metadata["responders"])
        self.temporal_features = list(self.metadata.get("temporal_features", []))
        self.temporal_recipes = dict(self.metadata.get("temporal_recipes", {}))
        if not self.temporal_recipes and self.temporal_features:
            # Compatibility with metadata produced before temporal_recipes was
            # persisted. Reconstruct the exact routing from the ordered output
            # column names, e.g. ts_lag1_feature_010.
            temporal_columns = list(
                self.metadata.get("temporal_feature_columns", [])
            )
            reconstructed = {feature: [] for feature in self.temporal_features}
            for feature in self.temporal_features:
                suffix = f"_{feature}"
                for column in temporal_columns:
                    if column.startswith("ts_") and column.endswith(suffix):
                        transform = column[3:-len(suffix)]
                        reconstructed[feature].append(transform)
            self.temporal_recipes = {
                feature: transforms
                for feature, transforms in reconstructed.items()
                if transforms
            }
            if len(self.temporal_recipes) != len(self.temporal_features):
                raise ValueError(
                    "metadata is missing temporal_recipes and its temporal "
                    "columns cannot reconstruct the routing"
                )
        self.temporal_indices = [
            self.features.index(name) for name in self.temporal_features
        ]
        self.temporal_builder = TemporalFeatureBuilder(
            len(self.temporal_features),
            feature_names=self.temporal_features,
            recipes=self.temporal_recipes or None,
        )
        self.target_model = lgb.Booster(model_file=str(model_dir / self.metadata["target_model"]))
        self.target_variant = str(self.metadata.get("target_variant", "D"))
        self.base_feature_count = (
            len(self.features)
            + len(self.metadata.get("temporal_feature_columns", []))
            + 1
        )
        self.target_base_indices = self.metadata.get("target_base_indices")
        self.target_responders = self.metadata.get("target_responders")
        if self.target_base_indices is None or self.target_responders is None:
            if self.target_variant not in {"A", "B", "C", "D"}:
                raise ValueError(f"unsupported legacy target_variant: {self.target_variant}")
            raw_count = len(self.features)
            raw_indices = [*range(raw_count), self.base_feature_count - 1]
            self.target_base_indices = {
                "A": raw_indices,
                "B": list(range(self.base_feature_count)),
                "C": raw_indices,
                "D": list(range(self.base_feature_count)),
            }[self.target_variant]
            self.target_responders = (
                self.responders if self.target_variant in {"C", "D"} else []
            )
        self.target_base_indices = np.asarray(
            self.target_base_indices, dtype=np.int64
        )
        self.target_responders = list(self.target_responders)
        unknown_responders = [
            name for name in self.target_responders
            if name not in self.metadata["responder_models"]
        ]
        if unknown_responders:
            raise ValueError(
                f"target references unknown responders: {unknown_responders}"
            )
        self.responder_models = [
            lgb.Booster(
                model_file=str(
                    model_dir / self.metadata["responder_models"][name]
                )
            )
            for name in self.target_responders
        ]
        for name, model in zip(self.target_responders, self.responder_models):
            if model.num_feature() != self.base_feature_count:
                raise ValueError(
                    f"{name} model expects {model.num_feature()} features, "
                    f"but metadata defines {self.base_feature_count}"
                )
        if (
            len(self.target_base_indices)
            and (
                int(self.target_base_indices.min()) < 0
                or int(self.target_base_indices.max()) >= self.base_feature_count
            )
        ):
            raise ValueError("target_base_indices are outside the base feature matrix")
        expected_target_features = (
            len(self.target_base_indices) + len(self.target_responders)
        )
        if self.target_model.num_feature() != expected_target_features:
            raise ValueError(
                f"target model expects {self.target_model.num_feature()} features, "
                f"but metadata defines {expected_target_features}"
            )
        self.scale = float(self.metadata.get("prediction_scale", 1.0))
        self.clip_min = float(self.metadata.get("clip_min", -np.inf))
        self.clip_max = float(self.metadata.get("clip_max", np.inf))
        self.last_time_id: int | None = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase")
        self.last_time_id = time_id

        raw = test.loc[:, self.features].to_numpy(dtype=np.float32, copy=True)
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64, copy=False)
        temporal = self.temporal_builder.transform(asset_ids, raw[:, self.temporal_indices])
        x = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        asset = test["asset_id"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        base = np.hstack([x, temporal, asset])
        if base.shape[1] != self.base_feature_count:
            raise ValueError(
                f"inference built {base.shape[1]} base features; "
                f"expected {self.base_feature_count}"
            )
        if self.target_responders:
            responder_hat = np.column_stack(
                [model.predict(base) for model in self.responder_models]
            ).astype(np.float32)
        else:
            responder_hat = np.empty((len(base), 0), dtype=np.float32)
        target_x = np.hstack(
            [base[:, self.target_base_indices], responder_hat]
        )
        prediction = np.asarray(self.target_model.predict(target_x), dtype=np.float64) * self.scale
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(prediction, self.clip_min, self.clip_max)
