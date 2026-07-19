from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check period-stable strategy training artifacts.")
    parser.add_argument("--model-dir", default="examples/period_stable_lgb_strategy/model")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    require_file(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_file = model_dir / metadata.get("model_file", "final_lightgbm.txt")
    feature_table_name = metadata.get(
        "weighted_feature_importance_file",
        metadata.get("stable_feature_importance_file", "stable_feature_importance.csv"),
    )
    feature_table_path = model_dir / feature_table_name
    period_metrics_path = model_dir / metadata.get("period_metrics_file", "period_metrics.csv")
    period_importance_path = model_dir / metadata.get("period_feature_importance_file", "period_feature_importance.csv")

    for path in [model_file, feature_table_path, period_metrics_path, period_importance_path]:
        require_file(path)

    feature_table = pd.read_csv(feature_table_path)
    metrics = pd.read_csv(period_metrics_path)
    period_importance = pd.read_csv(period_importance_path)
    selected = list(metadata.get("feature_columns", []))
    ema_features = list(metadata.get("ema_feature_columns", []))
    ema_halflives = list(metadata.get("ema_halflives", []))
    selected_from_table = feature_table.loc[feature_table["selected"].astype(bool), "feature"].astype(str).tolist()

    if not selected:
        raise ValueError("metadata feature_columns is empty")
    if selected != selected_from_table:
        raise ValueError("metadata feature_columns does not match selected feature importance rows")
    if "asset_id" in selected:
        raise ValueError("asset_id should be an input column, not a selected feature")
    if metadata.get("feature_selection_method") == "weighted_period_importance" and "weighted_importance" not in feature_table.columns:
        raise ValueError("weighted feature table must include weighted_importance")
    if any(feature not in selected for feature in ema_features):
        raise ValueError("ema_feature_columns must be a subset of selected feature_columns")
    if ema_features and not ema_halflives:
        raise ValueError("ema_halflives must be present when EMA features are enabled")
    if len(metrics) != int(metadata.get("num_periods", -1)):
        raise ValueError("period_metrics row count does not match metadata num_periods")
    if period_importance["period"].nunique() != int(metadata.get("num_periods", -1)):
        raise ValueError("period_feature_importance period count does not match metadata num_periods")
    if not {"future_valid_score", "inner_valid_score"}.issubset(metrics.columns):
        raise ValueError("period_metrics must include inner and future validation scores")

    print(
        json.dumps(
            {
                "ok": True,
                "strategy": metadata.get("strategy"),
                "num_periods": int(metadata["num_periods"]),
                "feature_selection_method": metadata.get("feature_selection_method", "legacy_stable_importance"),
                "selected_feature_count": len(selected),
                "ema_feature_count": len(ema_features),
                "final_score": metadata.get("final_score"),
                "final_l2": metadata.get("final_l2"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
