from __future__ import annotations

import argparse
import json
from pathlib import Path


LONG_TRANSFORMS = (
    "historical_zscore20",
    "minus_ema20",
    "rolling_std20",
)


def parse_args() -> argparse.Namespace:
    strategy_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Expand the compact 468 plan so every raw feature receives the "
            "three strictly historical 20-step transforms."
        )
    )
    parser.add_argument(
        "--compact-plan",
        default=str(strategy_dir / "long_horizon_468_feature_plan.json"),
    )
    parser.add_argument(
        "--output",
        default=str(strategy_dir / "all_feature_long_horizon_plan.json"),
    )
    parser.add_argument("--raw-feature-count", type=int, default=323)
    return parser.parse_args()


def build_plan(compact_plan: dict, raw_feature_count: int) -> dict:
    if raw_feature_count <= 0:
        raise ValueError("--raw-feature-count must be positive")
    compact_recipes = dict(compact_plan["recipes"])
    features = [f"feature_{index:03d}" for index in range(raw_feature_count)]
    missing = [feature for feature in compact_recipes if feature not in features]
    if missing:
        raise ValueError(f"compact plan references unavailable features: {missing}")

    recipes = {}
    for feature in features:
        transforms = list(LONG_TRANSFORMS)
        for transform in compact_recipes.get(feature, []):
            if transform not in transforms:
                transforms.append(transform)
        recipes[feature] = transforms

    derived_count = sum(map(len, recipes.values()))
    base_feature_count = raw_feature_count + derived_count + 1
    return {
        "description": (
            "All-feature long-horizon plan: every raw feature receives "
            "historical_zscore20, minus_ema20 and rolling_std20; the compact "
            "468-plan rmean5 and high-importance lag1/diff1 columns are retained "
            "as a nested control subset."
        ),
        "source_model_feature_count": base_feature_count,
        "raw_feature_count": raw_feature_count,
        "derived_feature_count": derived_count,
        "exact_recipes": True,
        "all_feature_transforms": list(LONG_TRANSFORMS),
        "compact_control_plan": "long_horizon_468_feature_plan.json",
        "compact_control_derived_feature_count": sum(
            map(len, compact_recipes.values())
        ),
        "history_features": features,
        "recipes": recipes,
    }


def main() -> None:
    args = parse_args()
    compact_plan = json.loads(
        Path(args.compact_plan).read_text(encoding="utf-8")
    )
    plan = build_plan(compact_plan, args.raw_feature_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: raw={plan['raw_feature_count']}, "
        f"derived={plan['derived_feature_count']}, "
        f"base={plan['source_model_feature_count']}"
    )


if __name__ == "__main__":
    main()
