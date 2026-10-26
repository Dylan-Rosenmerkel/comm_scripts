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
        L, n = 40, 40 * 1100                         # minimum ionising: ~4 keV (1100 e-) per 15-um pixel
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


class AGMXEdgesAndFlags(unittest.TestCase):
    """A_GMX on the mask border and the flags of every cluster (2026-10-25 audit): an X-ray centred on the first
    valid row used to lose its seed to the masked pixels next to it and came out as a flag-less 'low-E' cluster."""

    @classmethod
    def setUpClass(cls):
        import adaptive as AD
        cls.AD = AD
        rng = np.random.default_rng(5)
        H, W = 80, 120
        Q = rng.poisson(0.01, (H, W)) + rng.normal(0, 0.15, (H, W))

        def add(n, y, x, s):
            np.add.at(Q, (np.round(y + rng.normal(0, s, n)).astype(int), np.round(x + rng.normal(0, s, n)).astype(int)), 1.0)
        add(1500, 3.0, 60, 0.3)                      # X-ray on the first valid row
        add(1500, 40, 30, 0.3)                       # interior X-ray
        add(4, 40, 90, 0.3)                          # few-electron event
        valid = np.zeros((H, W), bool); valid[3:-3, 3:-3] = True
        cls.cl = AD.adaptive_cluster(np.where(valid, Q, 0.0), valid, engine="gmx", return_pixels=True)

    def test_border_xray_is_a_fitted_edge_cluster(self):
        c = [c for c in near(self.cl, 3, 60, 2) if c["E"] > 1]
        self.assertEqual(len(c), 1)
        self.assertGreaterEqual(c[0]["n_comp"], 1)
        self.assertNotEqual(c[0]["cls"], "low-E")
        self.assertTrue(c[0]["edge"])

    def test_every_cluster_has_flags_and_conserves_charge(self):
        for c in self.cl:
            for k in ("edge", "blend", "ped", "peakfrac", "chi2"):
                self.assertIn(k, c)
            self.assertAlmostEqual(c["_pix"][2].sum() * AM.E_PER_E, c["E"], places=6)
        self.assertFalse(near(self.cl, 40, 30, 2)[0]["edge"])

    def test_bad_arguments_are_rejected(self):
        Q = np.zeros((40, 40)); valid = np.ones((40, 40), bool)
        with self.assertRaises(ValueError):
            self.AD.adaptive_cluster(Q, valid, engine="GMX")
        with self.assertRaises(ValueError):
            self.AD.adaptive_cluster(Q, valid, engine="mhc", return_pixels=True)


class PhysicsAuditFixes(unittest.TestCase):
    """the physics fixes of the 2026-10-25 audit (docs/aGMX, 'Physics audit'), on simulated images"""

    @classmethod
    def setUpClass(cls):
        import adaptive as AD
        import gmx as GX
        cls.AD, cls.GX = AD, GX
        H, W = 200, 1500
        cls.valid = np.zeros((H, W), bool); cls.valid[3:-3, 3:-3] = True
        cls.shape = (H, W)

    def blank(self, lam, s, seed):
        rng = np.random.default_rng(seed)
        return np.where(self.valid, rng.poisson(lam, self.shape) + rng.normal(0, s, self.shape), 0.0)

    def test_gaussian_noise_is_not_resolved_as_a_comb(self):
        """A1 + I1: pure Gaussian noise of 1.5 e- was fitted as ~0.4 e- read noise (thousands of fake clusters)"""
        Q = np.where(self.valid, np.random.default_rng(1).normal(0, 1.5, self.shape), 0.0)
        self.assertAlmostEqual(AM.read_noise(Q, self.valid), 1.5, delta=0.1)
        cl = self.AD.adaptive_cluster(Q, self.valid, engine="gmx")
        self.assertLess(len(cl), 5)

    def test_no_counting_when_electrons_are_not_resolved(self):
        """C2: at 0.45 e- read noise rounding miscounts made hundreds of fake few-electron clusters"""
        cl = self.AD.adaptive_cluster(self.blank(0.01, 0.45, 2), self.valid, engine="gmx")
        self.assertEqual(sum(c.get("cls") == "low-E" for c in cl), 0)

    def test_hot_column_is_dark_charge(self):
        """B1: a column with +2 e-/pix was not flagged and gave dozens of fake low-E clusters"""
        Q = self.blank(0.01, 0.16, 3)
        Q[:, 700] += np.random.default_rng(4).poisson(2.0, self.shape[0]) * self.valid[:, 700]
        cl, info = self.GX.general_cluster(Q, self.valid, return_info=True)
        self.assertIn(700, info["hot_columns"])
        self.assertEqual(sum(abs(c["x"] - 700) < 1.5 for c in cl), 0)

    def test_dark_charge_not_biased_by_rounding(self):
        """A2: rounding miscounts made lam ~ lam + P(noise > 0.5 e-) (0.015 instead of 0.01 at s = 0.2)"""
        bg = AM.background_map(self.blank(0.01, 0.2, 5), self.valid)
        self.assertLess(abs(np.median(bg.lam[self.valid]) - 0.01), 0.002)

    def test_width_is_unbiased(self):
        """D2: the reported width (second moment of the shared charge) is unbiased to a few % (EM width was low)"""
        rng = np.random.default_rng(6)
        Q = self.blank(0.01, 0.16, 7); pos = [(100 + rng.uniform(-40, 40), 60 + 70 * j) for j in range(20)]
        for y0, x0 in pos:
            np.add.at(Q, (np.floor(y0 + rng.normal(0, 1.2, 1580) + .5).astype(int), np.floor(x0 + rng.normal(0, 1.2, 1580) + .5).astype(int)), 1.0)
        cl = self.GX.general_cluster(Q, self.valid)
        w = [near(cl, y, x, 2)[0]["sxy"] for y, x in pos]
        self.assertLess(abs(np.median(w) / 1.2 - 1), 0.03)

    def test_muon_needs_minimum_ionising_energy(self):
        """G1: a straight track is a muon only above ~2.5 keV per projected pixel (MIP ~4 keV per 15-um pixel)"""
        sh = dict(width=1.0, length=20.0, angle=30.0, ratio=0.1, sk_straight=0.3)
        self.assertEqual(MH.classify_ext(20 * 4.0, sh, 60), "muon")
        self.assertEqual(MH.classify_ext(20 * 0.9, sh, 60), "extended")


if __name__ == "__main__":
    unittest.main()
