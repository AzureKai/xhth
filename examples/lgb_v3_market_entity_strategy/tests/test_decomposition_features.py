from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from decomposition_features import (  # noqa: E402
    MarketEntityFeatureBuilder,
    center_cross_section,
    decompose_residual,
    entity_model_indices,
    market_feature_names,
)


class DecompositionFeatureTests(unittest.TestCase):
    def test_residual_decomposition_is_exact_and_entity_is_centered(self):
        target = np.asarray([1.0, 3.0, -2.0])
        base = np.asarray([0.5, 1.0, -1.0])
        market_residual, entity_residual = decompose_residual(target, base)
        reconstructed = base + market_residual + entity_residual
        np.testing.assert_allclose(reconstructed, target)
        self.assertAlmostEqual(float(np.mean(entity_residual)), 0.0, places=7)

    def test_center_cross_section_removes_only_common_level(self):
        values = np.asarray([1.0, 2.0, 6.0])
        centered = center_cross_section(values)
        self.assertAlmostEqual(float(np.mean(centered)), 0.0, places=12)
        np.testing.assert_allclose(
            np.diff(centered), np.diff(values)
        )

    def test_builder_market_schema_and_cold_state(self):
        builder = MarketEntityFeatureBuilder(["feature_001"])
        entity = builder.transform_time(
            np.asarray([1, 2]),
            np.asarray([[10.0], [20.0]], dtype=np.float32),
            np.asarray([0.1, -0.2], dtype=np.float32),
        )
        market = builder.market_features(entity)
        self.assertEqual(
            len(market), len(market_feature_names(["feature_001"]))
        )
        names = builder.feature_names
        np.testing.assert_array_equal(
            entity[:, names.index("entity_z20_feature_001")],
            np.zeros(2),
        )
        self.assertAlmostEqual(
            float(market[market_feature_names(["feature_001"]).index(
                "market_base_mean"
            )]),
            -0.05,
            places=6,
        )

    def test_entity_model_does_not_receive_market_regime_columns(self):
        builder = MarketEntityFeatureBuilder(["feature_001"])
        selected = [
            builder.feature_names[index]
            for index in entity_model_indices(builder.feature_names)
        ]
        self.assertIn("asset_id", selected)
        self.assertIn("base_prediction", selected)
        self.assertNotIn("regime_id", selected)
        self.assertFalse(any(name.startswith("regime_") for name in selected))


if __name__ == "__main__":
    unittest.main()
