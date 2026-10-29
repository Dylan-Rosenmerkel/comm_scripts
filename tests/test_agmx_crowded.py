"""A_GMX crowded mode (clustering_tests/crowded.py, adaptive.adaptive_cluster(crowded=...)) on simulated images:
the trigger, a muon through crossing tracks, a chain of X-ray blobs, a serial-register streak, the background
override, and that images below the trigger are clustered exactly as before.

    python -m unittest tests.test_agmx_crowded -v
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "clustering_tests"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adaptive as AD  # noqa: E402
import agent_methods as AM  # noqa: E402
import crowded as CR  # noqa: E402
import crowded_shapes as CS  # noqa: E402


def line(img, rng, y0, x0, y1, x1, e_per_px, s=0.7):
    """a straight track: e_per_px electrons per px of length, Gaussian diffusion s px"""
    L = np.hypot(y1 - y0, x1 - x0); n = rng.poisson(e_per_px * L)
    t = rng.random(n)
    y = y0 + t * (y1 - y0) + rng.normal(0, s, n); x = x0 + t * (x1 - x0) + rng.normal(0, s, n)
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    ok = (iy >= 0) & (iy < img.shape[0]) & (ix >= 0) & (ix < img.shape[1])
    T = np.zeros(img.shape); np.add.at(T, (iy[ok], ix[ok]), 1.0)
    img += T
    return T


def blob(img, rng, y, x, n, s=0.5):
    np.add.at(img, (np.floor(y + rng.normal(0, s, n)).astype(int), np.floor(x + rng.normal(0, s, n)).astype(int)), 1.0)


def noise(rng, shape, lam=0.5, s=0.16):
    return rng.poisson(lam, shape) + rng.normal(0, s, shape)


def owner(cl, shape):
    """dominant cluster index (1-based) of every pixel"""
    own = np.zeros(shape, int); best = np.zeros(shape)
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; m = q > best[ys, xs]; own[ys[m], xs[m]] = k; best[ys[m], xs[m]] = q[m]
    return own


class Trigger(unittest.TestCase):
    def test_quiet_image_is_clustered_as_before(self):
        rng = np.random.default_rng(3)
        Q = noise(rng, (200, 400), lam=0.01, s=0.15)
        blob(Q, rng, 60, 80, 1580); blob(Q, rng, 120, 300, 1580, s=1.2)
        line(Q, rng, 30, 150, 170, 230, 1100)
        valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
        a, ia = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_info=True, crowded="auto")
        b, ib = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_info=True, crowded="off")
        self.assertFalse(ia["crowded_mode"])
        self.assertLess(ia["core_fill"], CR.CROWD_FILL)
        self.assertEqual(len(a), len(b))
        for ca, cb in zip(a, b):
            self.assertEqual((ca["E"], ca["y"], ca["x"]), (cb["E"], cb["y"], cb["x"]))

    def test_dense_image_switches_crowded_mode_on(self):
        rng = np.random.default_rng(4)
        Q = noise(rng, (200, 400), lam=5.0)
        for _ in range(70):
            y0, x0 = rng.uniform(0, 200), rng.uniform(0, 400); a = rng.uniform(0, np.pi); L = rng.uniform(40, 150)
            line(Q, rng, y0, x0, y0 + L * np.sin(a), x0 + L * np.cos(a), 1100, s=rng.uniform(0.5, 1.5))
        valid = np.ones(Q.shape, bool)
        self.assertGreater(CR.core_fill(Q, valid, AM.background_map(Q, valid)), CR.CROWD_FILL)

    def test_bad_setting_rejected(self):
        Q = np.zeros((20, 20)); valid = np.ones(Q.shape, bool)
        with self.assertRaises(ValueError):
            AD.adaptive_cluster(Q, valid, engine="gmx", crowded="sometimes")


class StraightTracks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        H, W = 260, 420
        Q = noise(rng, (H, W), lam=3.0)
        cls.mu = line(Q, rng, 20, 40, 240, 380, 1100, s=0.8)          # the muon: ~400 px long
        # six short tracks crossing it at ~90 deg
        cls.cross = []
        for f in np.linspace(0.15, 0.85, 6):
            yc, xc = 20 + f * 220, 40 + f * 340
            cls.cross.append(line(Q, rng, yc - 25, xc + 16, yc + 25, xc - 16, 900, s=0.7))
        cls.Q, cls.valid = Q, np.ones(Q.shape, bool)
        cls.bg = AM.background_map(Q, cls.valid)
        cls.cl, cls.info = AD.adaptive_cluster(Q, cls.valid, cls.bg, engine="gmx", return_pixels=True, return_info=True, crowded="on")

    def test_finder_sees_the_muon(self):
        self.assertTrue(self.info["crowded_mode"])
        self.assertGreaterEqual(self.info["n_hints"], 1)

    def test_muon_is_one_cluster(self):
        own = owner(self.cl, self.Q.shape)
        ys, xs = np.nonzero(self.mu); q = self.mu[ys, xs]
        share = np.bincount(own[ys, xs], q); share[0] = 0
        self.assertGreaterEqual(share.max() / q.sum(), 0.85)
        best = self.cl[int(share.argmax()) - 1]
        self.assertEqual(best["cls"], "muon")
        # crossing tracks keep their own clusters: the muon's cluster is not much more than the muon (crowded="off"
        # merges all six crossing tracks into it: 1.7x the muon's charge)
        self.assertLess(best["E"] / AM.E_PER_E, 1.25 * q.sum())

    def test_crossing_tracks_are_not_swallowed(self):
        own = owner(self.cl, self.Q.shape)
        mu_k = np.bincount(own[self.mu > 0], self.mu[self.mu > 0]); mu_k[0] = 0; mu_k = int(mu_k.argmax())
        for T in self.cross:
            ys, xs = np.nonzero(T); q = T[ys, xs]
            sh = np.bincount(own[ys, xs], q, len(self.cl) + 1); sh[0] = 0
            self.assertLess(sh[mu_k] / q.sum(), 0.5)

    def test_blob_chain_is_not_a_track(self):
        rng = np.random.default_rng(9)
        Q = noise(rng, (200, 300), lam=3.0)
        for k in range(14):
            blob(Q, rng, 30 + 10 * k, 60 + 13 * k, 1600, s=0.6)        # X-rays in a row, 16 px apart
        valid = np.ones(Q.shape, bool)
        self.assertEqual(CR.find_hints(Q, valid, AM.background_map(Q, valid)), [])


class SerialStreak(unittest.TestCase):
    def test_streak_is_its_own_cluster(self):
        rng = np.random.default_rng(11)
        Q = noise(rng, (150, 300), lam=3.0)
        Q[70, 40:200] += rng.poisson(60, 160)                           # one row, no vertical spread
        line(Q, rng, 10, 120, 140, 130, 1100)                          # a track through it
        valid = np.ones(Q.shape, bool)
        cl = AD.adaptive_cluster(Q, valid, AM.background_map(Q, valid), engine="gmx", return_pixels=True, crowded="on")
        st = [c for c in cl if c.get("regime") == "serial"]
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0]["cls"], "artifact-row")
        self.assertTrue(np.all(st[0]["_pix"][0] == 70))
        self.assertGreater(st[0]["_pix"][1].max() - st[0]["_pix"][1].min(), 140)


class BackgroundOverride(unittest.TestCase):
    def test_never_worse_than_the_row_fit(self):
        # uniform 6 e-/pix: the row fit occasionally takes a pedestal 1 e- too high (the ambiguity the override is for);
        # the override may correct such rows (it need not find all of them) but must never move a pedestal away from
        # the truth (0) nor make the background mean worse
        for seed in (13, 21):
            rng = np.random.default_rng(seed)
            Q = noise(rng, (120, 256), lam=6.0)
            valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
            bg2, rep = CR.regional_background(Q, valid, bg)
            self.assertTrue(np.all(np.abs(bg2.o[:, 0]) <= np.abs(bg.o[:, 0]) + 0.05))     # no row moves away from 0
            self.assertLessEqual(np.abs(bg2.mean - 6.0).mean(), np.abs(bg.mean - 6.0).mean() + 0.02)

    def test_low_dark_charge_is_left_alone(self):
        rng = np.random.default_rng(15)                                 # Am-241-like: lam 2 e-/pix < lam_min
        Q = noise(rng, (120, 256), lam=2.0)
        valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
        o = bg.o.copy(); o[40:50] += 3.0
        bad = AM.Background(o, np.maximum(bg.mean - o, AM.LAM_FLOOR), bg.s2, bg.mask, bg.rows)
        bg2, rep = CR.regional_background(Q, valid, bad)
        self.assertIs(bg2, bad)
        self.assertFalse(rep["applied"])

    def test_wrong_pedestal_rows_are_restored(self):
        rng = np.random.default_rng(14)                                 # L2-like: lam 6 e-/pix
        Q = noise(rng, (120, 256), lam=6.0)
        valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
        o = bg.o.copy(); o[40:50] += 3.0                              # ten rows with a pedestal 3 e- too high
        bad = AM.Background(o, np.maximum(bg.mean - o, AM.LAM_FLOOR), bg.s2, bg.mask, bg.rows)
        bg2, rep = CR.regional_background(Q, valid, bad)
        self.assertTrue(rep["applied"])
        self.assertGreaterEqual(rep["bad_rows"], 10)
        self.assertLess(np.abs(bg2.o[40:50, 0] - bg.o[40:50, 0]).max(), 0.5)
        self.assertLess(abs(np.median(bg2.lam[40:50]) - 6.0), 0.5)


class CrowdedShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        S = StraightTracks; S.setUpClass()
        cls.S = S
        cls.cl, cls.info = AD.adaptive_cluster(S.Q, S.valid, S.bg, engine="gmx", return_pixels=True, return_info=True,
                                               crowded="on", crowded_shapes="on")

    def test_muon_is_one_line_component(self):
        self.assertTrue(self.info["crowded_shapes"]); self.assertGreaterEqual(self.info["n_lines"], 1)
        own = owner(self.cl, self.S.Q.shape)
        ys, xs = np.nonzero(self.S.mu); q = self.S.mu[ys, xs]
        sh = np.bincount(own[ys, xs], q, len(self.cl) + 1); sh[0] = 0
        best = self.cl[int(sh.argmax()) - 1]
        self.assertEqual(best["regime"], "line")
        self.assertGreaterEqual(sh.max() / q.sum(), 0.85)
        self.assertLess(best["E"] / AM.E_PER_E, 1.15 * q.sum())          # the crossing tracks are not in it

    def test_crossing_tracks_keep_their_charge(self):
        own = owner(self.cl, self.S.Q.shape)
        m = self.S.mu > 0; mk = np.bincount(own[m], self.S.mu[m]); mk[0] = 0; mk = int(mk.argmax())
        for T in self.S.cross:
            ys, xs = np.nonzero(T); q = T[ys, xs]
            sh = np.bincount(own[ys, xs], q, len(self.cl) + 1); sh[0] = 0
            self.assertLess(sh[mk] / q.sum(), 0.3)

    def test_topology_neck_splits_two_touching_deposits(self):
        import gmx as GX
        mix = GX.Mix([2000.0, 400.0, 2000.0], [[50.0, 50.0], [50.0, 53.0], [50.0, 56.0]], [[0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.4, 0.4, 0.0]])
        sub = CS.split_cluster(mix.mu, mix.A, mix.S)
        self.assertNotEqual(sub[0], sub[2])                               # two X-rays 6 px apart with a weak bridge

    def test_topology_keeps_a_straight_track(self):
        mu = np.c_[np.full(15, 40.0), np.arange(15) * 2.0 + 10]
        sub = CS.split_cluster(mu, np.full(15, 2000.0), np.tile([0.8, 0.8, 0.0], (15, 1)))
        self.assertTrue(np.all(sub == 0))


class HotColumnsAndGlow(unittest.TestCase):
    def test_hot_column_band_is_added_to_the_background(self):
        rng = np.random.default_rng(21)
        Q = noise(rng, (256, 300), lam=6.0)
        Q[:, 120:150] += rng.poisson(8.0, (256, 30))                    # a 30-column hot band, +8 e-/pix
        valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
        bg2, rep = CR.column_background(Q, valid, bg)
        self.assertTrue(rep["applied"])
        inside = (Q - bg2.mean)[:, 125:145].mean(); outside = (Q - bg2.mean)[:, 20:100].mean()
        self.assertLess(abs(inside), 1.0)                                # the band is in the model
        self.assertLess(abs(outside), 0.5)
        self.assertTrue(np.all(bg2.lam[:, 20:100] == bg.lam[:, 20:100]))  # other columns untouched

    def test_uniform_image_untouched(self):
        rng = np.random.default_rng(22)
        Q = noise(rng, (256, 300), lam=6.0)
        valid = np.ones(Q.shape, bool); bg = AM.background_map(Q, valid)
        bg2, rep = CR.column_background(Q, valid, bg)
        self.assertIs(bg2, bg)

    def test_glow_flag(self):
        rng = np.random.default_rng(23)
        bg = AM.background_map(noise(rng, (128, 256), lam=9.0), np.ones((128, 256), bool))
        ys, xs = np.mgrid[10:40, 10:30]; ys, xs = ys.ravel(), xs.ravel()
        faint = dict(E=600 * 5.0 * AM.E_PER_E, _pix=(ys, xs, np.full(ys.size, 5.0)))        # 600 px at 5 e-/px
        track = dict(E=200 * 900.0 * AM.E_PER_E, _pix=(ys[:200], xs[:200], np.full(200, 900.0)))
        self.assertEqual(CR.glow_flag([faint, track], bg), 1)
        self.assertEqual(faint["cls"], "glow"); self.assertNotIn("cls", track)


class AgmxRootRecord(unittest.TestCase):
    def test_codes_and_config(self):
        try:
            import agmx_root as AR
        except Exception as e:                                           # needs ROOT / pysimdamicm
            self.skipTest(f"agmx_root not importable: {e}")
        self.assertEqual(AR.REGIME_CODES["serial"], 3)
        self.assertEqual(AR.CLS_CODES["glow"], 11)
        self.assertEqual(AR.CLS_CODES["muon"], 4)
        r = AR.crowded_summary(dict(core_fill=0.29, crowded_mode=True, bg_override=dict(applied=True, bad_rows=3, cells_changed=5),
                                    n_hints=12, n_streaks=4, n_linked=7))
        self.assertTrue(r["on"]); self.assertEqual(r["straight_tracks"], 12); self.assertEqual(r["streaks"], 4)
        self.assertFalse(AR.crowded_summary(dict(core_fill=0.02, crowded_mode=False))["on"])


if __name__ == "__main__":
    unittest.main()
