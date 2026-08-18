from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SHORT_TRANSFORMS = ("lag1", "diff1")
SOURCE_TRANSFORMS = ("lag1", "diff1", "rmean5")
LONG_TRANSFORMS = (
    "historical_zscore20",
    "minus_ema20",
    "rolling_std20",
)


def parse_args() -> argparse.Namespace:
    strategy_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-width long-horizon temporal plan from the previous "
            "LGB468 feature-importance report."
        )
    )
    parser.add_argument(
        "--importance",
        default=str(strategy_dir / "model" / "target_feature_importance.csv"),
    )
    parser.add_argument(
        "--base-plan",
        default=str(strategy_dir / "baseline_468_feature_plan.json"),
    )
    parser.add_argument(
        "--output",
        default=str(strategy_dir / "long_horizon_468_feature_plan.json"),
    )
    parser.add_argument(
        "--variants", default="LGB468,LGB468_C4",
        help="Comma-separated old variants used to make the frozen selection.",
    )
    parser.add_argument(
        "--short-top-per-variant", type=int, default=5,
        help="Keep the top N lag1 and diff1 features from each old variant.",
    )
    parser.add_argument(
        "--long-feature-count", type=int, default=27,
        help="Number of source features receiving all three 20-step transforms.",
    )
    return parser.parse_args()


def load_importance(
    path: Path, variants: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    result = {
        variant: {transform: {} for transform in SOURCE_TRANSFORMS}
        for variant in variants
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row.get("variant", "")
            if variant not in result:
                continue
            column = row.get("feature", "")
            for transform in SOURCE_TRANSFORMS:
                prefix = f"ts_{transform}_"
                if column.startswith(prefix):
                    result[variant][transform][column[len(prefix):]] = float(
                        row["importance_gain"]
                    )
                    break
    missing = [
        f"{variant}/{transform}"
        for variant in variants
        for transform in SOURCE_TRANSFORMS
        if not result[variant][transform]
    ]
    if missing:
        raise ValueError(f"importance report is missing groups: {missing}")
    return result


def build_plan(
    base_plan: dict,
    importance: dict[str, dict[str, dict[str, float]]],
    short_top_per_variant: int,
    long_feature_count: int,
) -> dict:
    history_features = list(base_plan["history_features"])
    if len(history_features) != 48 or len(set(history_features)) != 48:
        raise ValueError("the source baseline plan must contain 48 unique features")

    retained: dict[str, set[str]] = {name: set() for name in SHORT_TRANSFORMS}
    for groups in importance.values():
        for transform in SHORT_TRANSFORMS:
            ranked = sorted(
                groups[transform].items(), key=lambda item: (-item[1], item[0])
            )
            retained[transform].update(
                feature for feature, _ in ranked[:short_top_per_variant]
            )

    # Normalize within each old model before combining so the different total
    # tree gains of LGB468 and LGB468_C4 receive equal weight.
    source_scores: dict[str, float] = defaultdict(float)
    for groups in importance.values():
        totals = {
            feature: sum(
                groups[transform].get(feature, 0.0)
                for transform in SOURCE_TRANSFORMS
            )
            for feature in history_features
        }
        denominator = sum(totals.values())
        if denominator <= 0.0:
            raise ValueError("temporal importance gain must be positive")
        for feature, value in totals.items():
            source_scores[feature] += value / denominator
    long_features = [
        feature
        for feature, _ in sorted(
            source_scores.items(), key=lambda item: (-item[1], item[0])
        )[:long_feature_count]
    ]
    long_feature_set = set(long_features)

    recipes = {}
    for feature in history_features:
        transforms = ["rmean5"]
        if feature in long_feature_set:
            transforms.extend(LONG_TRANSFORMS)
        for transform in SHORT_TRANSFORMS:
            if feature in retained[transform]:
                transforms.append(transform)
        recipes[feature] = transforms

    derived_count = sum(map(len, recipes.values()))
    if derived_count != 144:
        raise ValueError(
            "selection does not preserve the 468-column contract: "
            f"derived={derived_count}, expected=144; adjust the selection counts"
        )
    return {
        "description": (
            "Fixed-width long-horizon revision of the baseline 468 plan: retain "
            "rmean5 for all 48 sources, retain only top old lag1/diff1 features, "
            "and route 27 sources through three strictly historical 20-step transforms."
        ),
        "source_model_feature_count": 468,
        "exact_recipes": True,
        "selection": {
            "importance_variants": list(importance),
            "short_top_per_variant": short_top_per_variant,
            "retained_lag1": sorted(retained["lag1"]),
            "retained_diff1": sorted(retained["diff1"]),
            "long_feature_count": long_feature_count,
            "long_features_ranked": long_features,
            "long_transforms": list(LONG_TRANSFORMS),
            "derived_feature_count": derived_count,
        },
        "history_features": history_features,
        "recipes": recipes,
    }


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not variants:
        raise ValueError("--variants must not be empty")
    base_plan = json.loads(Path(args.base_plan).read_text(encoding="utf-8"))
    importance = load_importance(Path(args.importance), variants)
    plan = build_plan(
        base_plan,
        importance,
        args.short_top_per_variant,
        args.long_feature_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selection = plan["selection"]
    print(
        f"wrote {output}: history={len(plan['history_features'])}, "
        f"derived={selection['derived_feature_count']}, "
        f"lag1={len(selection['retained_lag1'])}, "
        f"diff1={len(selection['retained_diff1'])}, "
        f"long={selection['long_feature_count']}x{len(LONG_TRANSFORMS)}"
    )


if __name__ == "__main__":
    main()
