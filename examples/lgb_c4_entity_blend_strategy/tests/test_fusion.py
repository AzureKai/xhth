from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from train import (  # noqa: E402
    aligned_common_rows,
    candidate_report,
    configured_weights,
    union_segments,
)


def artifact(times, assets, prediction, folds=None):
    result = {
        "time_id": np.asarray(times, dtype=np.int64),
        "asset_id": np.asarray(assets, dtype=np.int16),
        "target": np.asarray([1.0, -1.0, 0.5, -0.5][:len(times)]),
        "weight": np.ones(len(times), dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.float32),
    }
    if folds is not None:
        result["fold_id"] = np.asarray(folds, dtype=np.int8)
    return result


class FusionTests(unittest.TestCase):
    def test_alignment_uses_only_common_time_range(self):
        left = artifact(
            [1, 2, 3, 4], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4],
            [1, 1, 2, 2],
        )
        right = artifact(
            [2, 3, 4], [0, 0, 0], [0.2, 0.3, 0.4], [4, 4, 5]
        )
        right["target"] = left["target"][1:]
        aligned_left, aligned_right = aligned_common_rows(left, right)
        np.testing.assert_array_equal(aligned_left["time_id"], [2, 3, 4])
        np.testing.assert_array_equal(
            aligned_left["time_id"], aligned_right["time_id"]
        )

    def test_union_segments_split_when_either_source_fold_changes(self):
        c4 = artifact(
            [1, 2, 3, 4], [0, 0, 0, 0], [0.0] * 4, [1, 1, 2, 2]
        )
        entity = artifact(
            [1, 2, 3, 4], [0, 0, 0, 0], [0.0] * 4, [3, 4, 4, 4]
        )
        self.assertEqual(union_segments(c4, entity), [(0, 1), (1, 2), (2, 4)])

    def test_candidate_report_finds_complementary_average(self):
        target = np.asarray([1.0, -1.0, 1.0, -1.0])
        c4_prediction = np.asarray([1.2, -0.8, 0.6, -1.4])
        entity_prediction = np.asarray([0.8, -1.2, 1.4, -0.6])
        report = candidate_report(
            0.5,
            target,
            np.ones(4),
            c4_prediction,
            entity_prediction,
            np.asarray([1, 2, 3, 4]),
            np.asarray([1, 1, 2, 2]),
            np.asarray([3, 3, 4, 4]),
            [(0, 2), (2, 4)],
        )
        self.assertAlmostEqual(report["global_oof_score"], 1.0)
        self.assertEqual(report["segment_scores"], [1.0, 1.0])

    def test_weight_parser_adds_endpoints(self):
        self.assertEqual(configured_weights("0.25,0.75"), [0.0, 0.25, 0.75, 1.0])
        with self.assertRaises(ValueError):
            configured_weights("-0.1,0.5")


if __name__ == "__main__":
    unittest.main()
