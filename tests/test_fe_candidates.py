"""clustering_tests/fe_candidates.py: the few-electron candidate catalogue that runs after A_GMX (simulated images).

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clustering_tests"))
import adaptive as AD  # noqa: E402
import agent_methods as AM  # noqa: E402
import fe_candidates as FE  # noqa: E402

H, W = 400, 2000


def image(lam, seed, events=(), big=None):
    rng = np.random.default_rng(seed)
    Q = rng.poisson(lam, (H, W)) + rng.normal(0, 0.15, (H, W))
    for y0, x0, ne in events:
        np.add.at(Q, (np.round(y0 + rng.normal(0, 0.35, ne)).astype(int), np.round(x0 + rng.normal(0, 0.35, ne)).astype(int)), 1.0)
    if big is not None:
        y0, x0, n = big
        np.add.at(Q, (np.round(y0 + rng.normal(0, 1.0, n)).astype(int), np.round(x0 + rng.normal(0, 1.0, n)).astype(int)), 1.0)
    valid = np.zeros((H, W), bool); valid[5:-5, 5:-5] = True
    return np.where(valid, Q, 0.0), valid


class Catalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ev = [(40 + 30 * (i % 11), 150 + 160 * (i // 11), 3) for i in range(33)]
        cls.ev = ev
        Q, valid = image(0.001, 1, ev, big=(200, 60, 16000))
        Q[200:202, 75:76] += 2                     # 4 e- 15 px from a ~60 keV deposit: inside its halo
        bg = AM.background_map(Q, valid)
        cl = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True)
        cls.cands, cls.summ = FE.candidates(Q, valid, bg, cl, n_blanks=10)

    def test_injected_events_listed(self):
        xy = np.array([(c["y"], c["x"]) for c in self.cands])
        found = [np.hypot(xy[:, 0] - y, xy[:, 1] - x).min() < 1.5 for y, x, _ in self.ev]
        self.assertGreater(np.mean(found), 0.9)

    def test_halo_flag(self):
        near = [c for c in self.cands if abs(c["y"] - 200.5) < 3 and abs(c["x"] - 75) < 3]
        self.assertTrue(near and all(c["mask_halo"] for c in near))
        self.assertFalse(any(c["clean"] for c in near))

    def test_summary_and_write(self):
        self.assertIn("expected_clean", self.summ)
        self.assertGreater(self.summ["single_e_rate"], 0)
        with tempfile.TemporaryDirectory() as d:
            csv_f, js = FE.write(self.cands, self.summ, os.path.join(d, "x_ext1"))
            self.assertTrue(os.path.getsize(csv_f) > 0 and os.path.getsize(js) > 0)

    def test_blank_observed_matches_expected(self):
        """on a blank, the clean 2-e- groups agree with the expected dark-charge background (Poisson, 4 sigma)"""
        Q, valid = image(0.002, 2)
        bg = AM.background_map(Q, valid)
        _, summ = FE.candidates(Q, valid, bg, [], n_blanks=10)
        obs = sum(v for k, v in summ["observed_clean"].items() if k.startswith("2e"))
        exp = sum(v["mean"] for k, v in summ["expected_clean"].items() if k.startswith("2e"))
        self.assertLess(abs(obs - exp), 4 * np.sqrt(exp) + 2)


if __name__ == "__main__":
    unittest.main()
