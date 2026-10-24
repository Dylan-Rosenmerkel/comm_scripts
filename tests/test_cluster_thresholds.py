"""Tests of --cluster-threshold-e auto in src/autoconfig_panaSKImg.py (no data needed).

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import autoconfig_panaSKImg as A  # noqa: E402


class AutoThresholds(unittest.TestCase):
    def thr(self, lam, resolved=True):
        return A.auto_cluster_thresholds([dict(ext="EXT1", resolved=resolved, lam_all_median=lam)])["EXT1"]

    def test_poisson_tail_rule(self):
        """k is the smallest k >= 2 with P(Poisson(lambda) >= k) <= alpha; pixel threshold k - 0.5."""
        for lam in (0.05, 0.2, 0.5, 0.8, 1.05, 1.2, 1.7, 2.6):
            pix, seed, lam_used, k = self.thr(lam)
            self.assertEqual(pix, k - 0.5)
            self.assertLessEqual(poisson.sf(k - 1, lam), A.AUTO_ALPHA)
            if k > 2:
                self.assertGreater(poisson.sf(k - 2, lam), A.AUTO_ALPHA)

    def test_values_on_the_test_data(self):
        """The dark charges of the 2025-08-27 DM09 / DM10 Image 5s give 2 / 3 / 6 e-."""
        self.assertEqual(self.thr(0.25)[3], 2)
        self.assertEqual(self.thr(1.05)[3], 3)
        self.assertEqual(self.thr(2.62)[3], 6)

    def test_seed_stays_above_the_pixel_threshold(self):
        for lam in (0.2, 3.0, 6.0):
            pix, seed, _, _ = self.thr(lam)
            self.assertGreater(seed, pix)

    def test_unresolved_ccd_keeps_reference(self):
        self.assertIsNone(self.thr(0.5, resolved=False))
        self.assertIsNone(self.thr(float("nan")))

    def test_apply_keeps_reference_for_unresolved(self):
        j = {"process": {"ClusterFinder": {"spr_threshold": {"EXT1": [None, 10, 30], "EXT2": [None, 10, 30]}}}}
        res = [dict(ext="EXT1", resolved=True, lam_all_median=1.0), dict(ext="EXT2", resolved=False)]
        A.apply_cluster_thresholds(j, res, "auto")
        thr = j["process"]["ClusterFinder"]["spr_threshold"]
        self.assertEqual(thr["EXT1"], [1.0, 2.5, 9.5])
        self.assertEqual(thr["EXT2"], [None, 10, 30])


if __name__ == "__main__":
    unittest.main()
