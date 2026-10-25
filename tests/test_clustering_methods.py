"""Fast checks of the clustering methods in clustering_tests/ on a simulated image (no data files needed):
an Fe-55 cluster, a diffuse Fe-55 cluster, a muon and two 5.9 keV X-rays 4 px apart on a skipper-like background
(Poisson dark charge 0.01 e-/pix + 0.15 e- read noise).

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clustering_tests"))
import agent_methods as AM  # noqa: E402
import methods2 as M2  # noqa: E402
import mhc as MH  # noqa: E402

NE = 1 / AM.E_PER_E


def near(cl, y, x, r):
    return [c for c in cl if np.isfinite(c["E"]) and np.hypot(c["y"] - y, c["x"] - x) < r]


class ClusteringMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        H, W = 300, 600
        Q = rng.poisson(0.01, (H, W)) + rng.normal(0, 0.15, (H, W))
        cls.truth = {}

        def add(name, y, x, E):
            np.add.at(Q, (np.round(y).astype(int), np.round(x).astype(int)), 1.0)
            cls.truth[name] = E

        n = 1580
        add("fe55", 80 + rng.normal(0, .5, n), 100 + rng.normal(0, .5, n), n / NE)
        add("fe55_diffuse", 80 + rng.normal(0, 1.6, n), 250 + rng.normal(0, 1.6, n), n / NE)
        L, n = 40, 40 * 250
        t = rng.uniform(0, 1, n); s = 0.2 + t
        add("muon", 200 + (t - .5) * L * np.sin(.5) + rng.normal(0, 1, n) * s,
            150 + (t - .5) * L * np.cos(.5) + rng.normal(0, 1, n) * s, n / NE)
        for k, x0 in enumerate((448, 452)):
            add(f"pair{k}", 200 + rng.normal(0, .5, 1580), x0 + rng.normal(0, .5, 1580), 1580 / NE)
        cls.Q = Q
        cls.valid = np.zeros((H, W), bool); cls.valid[3:-3, 3:-3] = True
        cls.bg = AM.background_map(Q, cls.valid)
        cls.mf = AM.mf_detect(Q, cls.valid, cls.bg)[0]
        cls.hyb = M2.mfdg_hybrid(Q, cls.valid, cls.bg, ext_big=14)
        cls.mhc = MH.mhc(Q, cls.valid, cls.bg)

    def test_mfdg_energy_unbiased(self):
        for name, (y, x) in (("fe55", (80, 100)), ("fe55_diffuse", (80, 250))):
            c = near(self.mf, y, x, 3)
            self.assertEqual(len(c), 1)
            self.assertLess(abs(c[0]["E"] / self.truth[name] - 1), 0.02)

    def test_mfdg_alone_splits_the_muon(self):
        self.assertGreaterEqual(len(near(self.mf, 200, 150, 22)), 2)

    def test_track_merged_and_pair_resolved(self):
        for name, cl in (("hybrid v1", self.hyb), ("MHC", self.mhc)):
            with self.subTest(method=name):
                mu = near(cl, 200, 150, 22)
                self.assertEqual(len(mu), 1)
                self.assertLess(abs(mu[0]["E"] / self.truth["muon"] - 1), 0.05)
                pr = near(cl, 200, 450, 4)
                self.assertEqual(len(pr), 2)
                for c in pr:
                    self.assertLess(abs(c["E"] / self.truth["pair0"] - 1), 0.10)

    def test_mhc_classes(self):
        self.assertEqual(near(self.mhc, 80, 100, 3)[0]["cls"], "x-ray")
        self.assertEqual(near(self.mhc, 80, 250, 3)[0]["cls"], "x-ray diffuse")
        self.assertIn(near(self.mhc, 200, 150, 22)[0]["cls"], ("muon", "worm"))


class GMXQualityFlags(unittest.TestCase):
    """GMX quality flags (ped, peakfrac, chi2) and the satellite limit (a real X-ray next to a big deposit is not
    merged into it)."""

    @classmethod
    def setUpClass(cls):
        import gmx as GX
        rng = np.random.default_rng(2)
        H, W = 200, 300
        Q = rng.poisson(0.01, (H, W)) + rng.normal(0, 0.15, (H, W))

        def add(n, y, x, s):
            np.add.at(Q, (np.round(y + rng.normal(0, s, n)).astype(int), np.round(x + rng.normal(0, s, n)).astype(int)), 1.0)
        add(1580, 60, 60, 0.5)                       # isolated Fe-55
        add(160000, 120, 180, 2.0)                   # ~600 keV blob
        add(1580, 120, 187, 0.4)                     # Fe-55 7 px from the blob centre
        valid = np.zeros((H, W), bool); valid[3:-3, 3:-3] = True
        cls.cl = GX.general_cluster(Q, valid)

    def test_flags_isolated(self):
        c = near(self.cl, 60, 60, 3)[0]
        self.assertLess(abs(c["ped"]), 0.5)
        self.assertGreater(c["peakfrac"], 0.95)
        self.assertLess(c["chi2"], 5)

    def test_xray_next_to_big_deposit_not_merged(self):
        c = near(self.cl, 120, 187, 2)
        self.assertEqual(len(c), 1)
        self.assertLess(abs(c[0]["E"] / (1580 / NE) - 1), 0.15)


if __name__ == "__main__":
    unittest.main()
