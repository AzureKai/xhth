from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


STRATEGY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY_DIR))

from run_entity_prior_ablation import (  # noqa: E402
    _prior_values,
    _update_prior_state,
    configured_variants,
)


class EntityPriorAblationTests(unittest.TestCase):
    def test_global_and_ema_priors_are_causal_and_shrunk(self):
        specs = [
            ("frozen_entity_residual_prior", None),
            ("frozen_entity_residual_ema50", 50.0),
        ]
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        emas: dict[float, dict[int, float]] = {50.0: {}}
        assets = np.asarray([1, 2])

        cold = _prior_values(
            assets, specs, sums, counts, emas, shrinkage=1.0
        )
        np.testing.assert_array_equal(cold, np.zeros((2, 2)))
        _update_prior_state(
            assets,
            np.asarray([4.0, -2.0]),
            specs,
            sums,
            counts,
            emas,
        )
        warm = _prior_values(
            assets, specs, sums, counts, emas, shrinkage=1.0
        )
        np.testing.assert_allclose(warm, [[2.0, 2.0], [-1.0, -1.0]])

    def test_ema_updates_after_the_complete_time_slice(self):
        specs = [("frozen_entity_residual_ema50", 50.0)]
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        emas: dict[float, dict[int, float]] = {50.0: {}}
        assets = np.asarray([7, 7])
        before = _prior_values(
            assets, specs, sums, counts, emas, shrinkage=1.0
        )
        np.testing.assert_array_equal(before, np.zeros((2, 1)))
        _update_prior_state(
            assets,
            np.asarray([2.0, 4.0]),
            specs,
            sums,
            counts,
            emas,
        )
        after = _prior_values(
            np.asarray([7]), specs, sums, counts, emas, shrinkage=1.0
        )
        np.testing.assert_allclose(after, [[2.0]])

    def test_variant_parser_rejects_unknown_names(self):
        self.assertEqual(
            configured_variants("full,full_ema50_prior,full"),
            ["full", "full_ema50_prior"],
        )
        with self.assertRaises(ValueError):
            configured_variants("full,ema5")


if __name__ == "__main__":
    unittest.main()
