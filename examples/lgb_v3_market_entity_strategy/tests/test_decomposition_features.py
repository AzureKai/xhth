from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from decomposition_features import (  # noqa: E402
    EntityFeatureBuilder,
    center_cross_section,
    cross_sectional_z,
    entity_feature_names,
    entity_residual_target,
)


class EntityFeatureTests(unittest.TestCase):
    def test_entity_target_is_centered_and_reconstructs_relative_residual(self):
        target = np.asarray([1.0, 3.0, -2.0])
        base = np.asarray([0.5, 1.0, -1.0])
        residual = entity_residual_target(target, base)
        expected = (
            target - np.mean(target) - base + np.mean(base)
        )
        np.testing.assert_allclose(residual, expected)
        self.assertAlmostEqual(float(np.mean(residual)), 0.0, places=7)

    def test_center_cross_section_removes_only_common_level(self):
        values = np.asarray([1.0, 2.0, 6.0])
        centered = center_cross_section(values)
        self.assertAlmostEqual(float(np.mean(centered)), 0.0, places=12)
        np.testing.assert_allclose(np.diff(centered), np.diff(values))

    def test_cross_sectional_z_is_columnwise_and_finite(self):
        values = np.asarray([
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, np.nan],
        ], dtype=np.float32)
        normalized = cross_sectional_z(values)
        self.assertTrue(np.all(np.isfinite(normalized)))
        np.testing.assert_allclose(
            np.mean(normalized, axis=0), np.zeros(2), atol=1e-6
        )

    def test_builder_schema_and_cold_state(self):
        builder = EntityFeatureBuilder(["feature_001"])
        output = builder.transform_time(
            np.asarray([1, 2]),
            np.asarray([[10.0], [20.0]], dtype=np.float32),
            np.asarray([0.1, -0.2], dtype=np.float32),
        )
        self.assertEqual(
            output.shape[1], len(entity_feature_names(["feature_001"]))
        )
        names = builder.feature_names
        np.testing.assert_array_equal(
            output[:, names.index("entity_z20_feature_001")],
            np.zeros(2),
        )
        np.testing.assert_allclose(
            output[:, names.index("cross_z_feature_001")],
            [-1.0, 1.0],
        )

    def test_builder_uses_only_prior_entity_state(self):
        builder = EntityFeatureBuilder(["feature_001"])
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
        names = builder.feature_names
        z = output[:, names.index("entity_z20_feature_001")]
        self.assertGreater(z[0], 0.0)
        self.assertLess(z[1], 0.0)


if __name__ == "__main__":
    unittest.main()
