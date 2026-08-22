from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from cluster_features import (  # noqa: E402
    assigned_clusters,
    center_by_time,
    cluster_mapping,
    cocluster_agreement,
    deterministic_kmeans,
    select_residual_scale,
)


class ClusterFeatureTests(unittest.TestCase):
    def test_kmeans_separates_two_groups_deterministically(self):
        values = np.asarray([
            [-3.0, -2.9], [-2.8, -3.1], [3.0, 3.1], [2.9, 2.8]
        ])
        first, _, _ = deterministic_kmeans(values, 2, seed=7)
        second, _, _ = deterministic_kmeans(values, 2, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        self.assertNotEqual(first[0], first[2])

    def test_cocluster_agreement_ignores_label_permutation(self):
        left = {0: 0, 1: 0, 2: 1, 3: 1}
        right = {0: 1, 1: 1, 2: 0, 3: 0}
        self.assertEqual(cocluster_agreement(left, right), 1.0)

    def test_center_by_time_removes_each_cross_section_mean(self):
        centered = center_by_time(
            [1.0, 3.0, -1.0, 1.0], [10, 10, 11, 11]
        )
        np.testing.assert_allclose(centered, [-1.0, 1.0, -1.0, 1.0])

    def test_asset_mapping_routes_every_known_asset(self):
        features = np.asarray([
            [-2.0], [-1.5], [2.0], [1.5],
            [-2.2], [-1.7], [2.2], [1.7],
        ])
        assets = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        residual = np.asarray([-1.0, -0.8, 1.0, 0.8, -1.1, -0.7, 1.1, 0.7])
        mapping, audit = cluster_mapping(
            features, assets, residual, np.ones(8), 2, seed=11
        )
        self.assertEqual(audit["asset_count"], 2)
        routed = assigned_clusters(assets, mapping)
        self.assertEqual(len(routed), len(assets))
        with self.assertRaises(ValueError):
            assigned_clusters(np.asarray([9]), mapping)

    def test_profiles_cluster_opposite_feature_slopes(self):
        rng = np.random.default_rng(19)
        assets = np.repeat(np.arange(8), 300)
        feature = rng.normal(size=(len(assets), 2))
        slope = np.where(assets < 4, 1.0, -1.0)
        residual = slope * feature[:, 0] + rng.normal(0.0, 0.05, len(assets))
        mapping, _ = cluster_mapping(
            feature, assets, residual, np.ones(len(assets)), 2, seed=31
        )
        first = {mapping[asset] for asset in range(4)}
        second = {mapping[asset] for asset in range(4, 8)}
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second)

    def test_scale_selection_rewards_complementary_residual(self):
        folds = [{
            "target": np.asarray([1.0, -1.0]),
            "base": np.asarray([0.5, -0.5]),
            "weight": np.ones(2),
            "residual_prediction": np.asarray([0.5, -0.5]),
            "base_score": 0.75,
        }]
        selected = select_residual_scale(folds, [0.0, 0.5, 1.0])
        self.assertEqual(selected["residual_scale"], 1.0)
        self.assertAlmostEqual(selected["mean_fold_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
