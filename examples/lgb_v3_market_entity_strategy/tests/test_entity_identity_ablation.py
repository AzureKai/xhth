from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from run_entity_identity_ablation import (  # noqa: E402
    AblationEntitySequence,
    _update_prior_state,
    _write_prior_rows,
    configured_variants,
)


class EntityIdentityAblationTests(unittest.TestCase):
    def test_combined_sequence_aligns_spans_and_appended_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.arange(15, dtype=np.float32).reshape(5, 3)
            extra = (100 + np.arange(10, dtype=np.float32)).reshape(5, 2)
            prior = np.asarray([0.1, 0.2, 0.4, 0.5], dtype=np.float32)
            np.save(root / "source.npy", source)
            np.save(root / "extra.npy", extra)
            np.save(root / "prior.npy", prior)
            sequence = AblationEntitySequence(
                root / "source.npy",
                [0, 2],
                root / "extra.npy",
                [(0, 2), (3, 5)],
                root / "prior.npy",
            )

            self.assertEqual(len(sequence), 4)
            expected = np.column_stack([
                source[[0, 1, 3, 4]][:, [0, 2]],
                extra[[0, 1, 3, 4]],
                prior,
            ])
            np.testing.assert_allclose(sequence[:], expected)

    def test_combined_sequence_accepts_multiple_prior_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.arange(12, dtype=np.float32).reshape(4, 3)
            extra = np.arange(8, dtype=np.float32).reshape(4, 2)
            priors = np.asarray([
                [0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4],
            ], dtype=np.float32)
            np.save(root / "source.npy", source)
            np.save(root / "extra.npy", extra)
            np.save(root / "priors.npy", priors)
            sequence = AblationEntitySequence(
                root / "source.npy",
                [0, 2],
                root / "extra.npy",
                [(0, 4)],
                root / "priors.npy",
            )

            expected = np.column_stack([source[:, [0, 2]], extra, priors])
            np.testing.assert_allclose(sequence[:], expected)

    def test_prior_uses_only_previous_times_and_shrinks_to_zero(self):
        output = np.empty(4, dtype=np.float32)
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}

        cursor = _write_prior_rows(
            output,
            0,
            np.asarray([1, 2]),
            sums,
            counts,
            shrinkage=2.0,
        )
        np.testing.assert_array_equal(output[:2], np.zeros(2))
        _update_prior_state(
            np.asarray([1, 2]),
            np.asarray([3.0, -6.0]),
            sums,
            counts,
        )
        cursor = _write_prior_rows(
            output,
            cursor,
            np.asarray([1, 2]),
            sums,
            counts,
            shrinkage=2.0,
        )

        self.assertEqual(cursor, 4)
        np.testing.assert_allclose(output[2:], [1.0, -2.0])

    def test_same_time_rows_do_not_update_each_other(self):
        output = np.empty(2, dtype=np.float32)
        _write_prior_rows(
            output,
            0,
            np.asarray([7, 7]),
            {},
            {},
            shrinkage=1.0,
        )
        np.testing.assert_array_equal(output, np.zeros(2))

    def test_variant_parser_rejects_unknown_names(self):
        self.assertEqual(
            configured_variants("full,no_asset_id,full"),
            ["full", "no_asset_id"],
        )
        with self.assertRaises(ValueError):
            configured_variants("full,leaky_prior")


if __name__ == "__main__":
    unittest.main()
