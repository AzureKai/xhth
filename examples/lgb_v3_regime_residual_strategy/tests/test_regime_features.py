from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from regime_features import (  # noqa: E402
    REGIME_FEATURE_NAMES,
    RegimeEntityFeatureBuilder,
    residual_feature_names,
    state_only_indices,
)


class RegimeEntityFeatureTests(unittest.TestCase):
    def test_first_observation_has_cold_entity_state(self):
        builder = RegimeEntityFeatureBuilder(["feature_001"])
        output = builder.transform_time(
            np.asarray([1, 2]),
            np.asarray([[10.0], [20.0]], dtype=np.float32),
            np.asarray([0.1, -0.2], dtype=np.float32),
        )
        names = builder.feature_names
        self.assertEqual(output[:, names.index("entity_z20_feature_001")].tolist(), [0.0, 0.0])
        self.assertEqual(output[:, names.index("entity_history_log")].tolist(), [0.0, 0.0])
        np.testing.assert_allclose(
            output[:, names.index("cross_z_feature_001")], [-1.0, 1.0]
        )

    def test_entity_history_is_independent_and_uses_only_prior_values(self):
        builder = RegimeEntityFeatureBuilder(["feature_001"])
        builder.transform_time(
            np.asarray([1, 2]),
            np.asarray([[10.0], [100.0]], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        output = builder.transform_time(
            np.asarray([1, 2]),
            np.asarray([[11.0], [98.0]], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        z = output[:, builder.feature_names.index("entity_z20_feature_001")]
        self.assertGreater(z[0], 0.0)
        self.assertLess(z[1], 0.0)
        expected_history = np.log1p(1.0)
        np.testing.assert_allclose(
            output[:, builder.feature_names.index("entity_history_log")],
            expected_history,
        )

    def test_regime_columns_are_removed_from_state_control(self):
        names = residual_feature_names(["feature_001", "feature_002"])
        selected = [names[index] for index in state_only_indices(names)]
        self.assertIn("entity_z20_feature_001", selected)
        self.assertIn("base_prediction", selected)
        self.assertNotIn("regime_id", selected)
        self.assertTrue(set(REGIME_FEATURE_NAMES).isdisjoint(selected))

    def test_regime_id_responds_to_cross_entity_shock(self):
        builder = RegimeEntityFeatureBuilder(["feature_001"])
        assets = np.arange(10)
        for _ in range(20):
            builder.transform_time(
                assets,
                np.ones((10, 1), dtype=np.float32),
                np.zeros(10, dtype=np.float32),
            )
        output = builder.transform_time(
            assets,
            np.full((10, 1), 10.0, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        )
        regime_id = output[:, builder.feature_names.index("regime_id")]
        np.testing.assert_array_equal(regime_id, np.full(10, 3.0))


if __name__ == "__main__":
    unittest.main()
