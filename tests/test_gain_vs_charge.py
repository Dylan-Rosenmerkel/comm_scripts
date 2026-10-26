"""Tests of the g(n) models in src/gain_vs_charge.py: converting ADU to electrons must undo A(n) = integral of g.

These are the functions that rewrite the cluster energies (--gain-correction), so a round trip
electrons -> ADU -> electrons must give the electrons back, for every model form, below and above n_cap.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import gain_vs_charge as GVC  # noqa: E402

# One model of each form, shaped like the measured AMtest curves (about 3 % lower gain at 1000 e-)
LINEAR = {"form": "linear", "a": 1.30, "b": -3.0e-5, "n_cap": 3000.0}
BROKEN = {"form": "broken", "a": 1.30, "b1": -1.0e-5, "b2": -4.0e-5, "n_knot": 400.0, "n_cap": 3000.0}
INTERP = {"form": "interpolated", "a": 1.30, "n_pts": [25, 50, 100, 200, 400, 800, 1000],
          "g_pts": [1.300, 1.2995, 1.299, 1.297, 1.292, 1.280, 1.272], "slope_end": -4.0e-5, "n_cap": 3000.0}
MODELS = {"linear": LINEAR, "broken": BROKEN, "interpolated": INTERP}


class Forms(unittest.TestCase):
    def test_model_form(self):
        for name, m in MODELS.items():
            self.assertEqual(GVC.model_form(m), name)

    def test_gain_falls_and_is_constant_above_n_cap(self):
        for name, m in MODELS.items():
            g = GVC.g_of(np.array([0.0, 500.0, 1000.0, 3000.0, 5000.0]), m)
            self.assertTrue(np.all(np.diff(g[:4]) <= 0), name)             # falls with charge
            self.assertAlmostEqual(g[3], g[4], places=12, msg=name)         # held constant above n_cap

    def test_linear_closed_form_matches_formula(self):
        a, b, nc = LINEAR["a"], LINEAR["b"], LINEAR["n_cap"]
        self.assertAlmostEqual(float(GVC.adu_of_n(1000.0, a, b, nc)), a * 1000 + 0.5 * b * 1000 ** 2, places=9)


class RoundTrip(unittest.TestCase):
    """electrons_of_adu(A(n)) == n for every model form."""

    def check(self, model, tol):
        n = np.array([0.5, 1.0, 10.0, 137.0, 999.0, 2999.0, 3500.0, 8000.0])
        adu = np.array([GVC._adu(v, model) for v in n])
        back = GVC.electrons_of_adu(adu, model)
        np.testing.assert_allclose(back, n, rtol=0, atol=tol)

    def test_linear(self):
        self.check(LINEAR, 1e-9)

    def test_broken(self):
        self.check(BROKEN, 1e-6)

    def test_interpolated(self):
        self.check(INTERP, 1e-6)

    def test_negative_adu_uses_low_charge_gain(self):
        """Noise around the zero peak (ADU <= 0) is divided by the low-charge gain g(0)."""
        for name, m in MODELS.items():
            self.assertAlmostEqual(float(GVC.electrons_of_adu(np.array([-2.6]), m)[0]),
                                   -2.6 / float(GVC.g_of(0.0, m)), places=12, msg=name)


class Periodogram(unittest.TestCase):
    def test_finds_comb_spacing(self):
        """Pixel values on a comb with spacing 1.3 ADU (plus noise) peak at the true spacing."""
        rng = np.random.default_rng(0)
        v = 1.3 * rng.integers(0, 40, 20000) + rng.normal(0, 0.15, 20000)
        gs = np.linspace(1.2, 1.4, 401)
        p = GVC.periodogram(v, gs)
        self.assertAlmostEqual(gs[np.argmax(p)], 1.3, delta=0.002)


class CoherentPooling(unittest.TestCase):
    """_coherent_points: images converted with their own curves are pooled into one comb."""

    @staticmethod
    def image(seed, scale, n_pix=60000):
        """A simulated image: electrons spread over 20-1500, response scale * A(n) of LINEAR, noise 0.15 e-.
        Returns (pixels converted with the image's own, exact curve, that curve)."""
        rng = np.random.default_rng(seed)
        n = rng.integers(20, 1500, n_pix).astype(float)
        model = dict(LINEAR, a=LINEAR["a"] * scale, b=LINEAR["b"] * scale)
        x = GVC.adu_of_n(n, model["a"], model["b"], model["n_cap"]) + rng.normal(0, 0.15 * model["a"], n.size)
        return GVC.electrons_of_adu(x, model), model

    def test_recovers_the_gain_and_spacing_one(self):
        """With exact curves the tooth spacing in electrons is 1 and g(n) is the curves' mean gain."""
        imgs = [self.image(s, k) for s, k in ((1, 1.000), (2, 1.002), (3, 0.998))]
        pts = GVC._coherent_points([u for u, _ in imgs], [m for _, m in imgs])
        good = [p for p in pts if p["significant"]]
        self.assertGreaterEqual(len(good), 8)
        for p in good:
            self.assertAlmostEqual(p["spacing_vs_curves"], 1.0, delta=3 * p["err"] / p["g"] + 1e-3)
            gbar = np.mean([float(GVC.g_of(p["n"], m)) for _, m in imgs])
            self.assertAlmostEqual(p["g"], gbar, delta=3 * p["err"] + 1e-3 * gbar)

    def test_offsets_between_images_are_aligned(self):
        """Teeth shifted by +0.3 / -0.3 e- in two of the images (an integration error of their curves)
        are aligned before pooling: the pooled visibility stays that of one image, not washed out."""
        imgs = [self.image(s, 1.0) for s in (1, 2, 3)]
        us = [imgs[0][0], imgs[1][0] + 0.3, imgs[2][0] - 0.3]
        pts = GVC._coherent_points(us, [m for _, m in imgs])
        one = GVC._coherent_points([imgs[0][0]], [imgs[0][1]])
        for p, q in zip(pts[:6], one[:6]):
            self.assertGreater(p["visibility"], 0.9 * q["visibility"])

    def test_independent_of_image_order(self):
        imgs = [self.image(s, k) for s, k in ((1, 1.000), (2, 1.002), (3, 0.998))]
        a = GVC._coherent_points([u for u, _ in imgs], [m for _, m in imgs])
        b = GVC._coherent_points([u for u, _ in imgs[::-1]], [m for _, m in imgs[::-1]])
        for p, q in zip(a, b):
            self.assertAlmostEqual(p["g"], q["g"], places=6)
            self.assertAlmostEqual(p["visibility"], q["visibility"], places=6)


class ModelFiles(unittest.TestCase):
    """load_model / electrons_with_anchor: shared by gain_vs_charge.apply and the A_GMX writer"""

    def test_load_model(self):
        m = dict(a=1.3, b=-1e-5, n_cap=2000.0)
        ext = {"EXT1": dict(ok=True, model=m), "EXT2": dict(ok=True, model=repr(m)), "EXT3": dict(ok=False, model=m)}
        self.assertEqual(GVC.load_model(ext, 1), m)
        self.assertEqual(GVC.load_model(ext, "EXT2"), m)          # older files: the model as its repr
        self.assertIsNone(GVC.load_model(ext, 3))                  # curve not reliable
        self.assertIsNone(GVC.load_model(ext, 4))                  # no entry

    def test_anchor(self):
        m = dict(a=1.3, b=0.0, n_cap=2000.0)                        # constant gain 1.3 ADU/e-
        adu = np.array([13.0, 130.0])
        np.testing.assert_allclose(GVC.electrons_with_anchor(adu, m, 1.25, "comb"), adu / 1.3)
        np.testing.assert_allclose(GVC.electrons_with_anchor(adu, m, 1.25, "fit"), adu / 1.25)


class ReadoutMemory(unittest.TestCase):
    """remove_memory / fit_memory_coefficients: a leak of the pixel read before is found and removed."""

    def test_remove_memory_first_order_inverse(self):
        x = np.array([[0.0, 1000.0, 1000.0, 0.0]])
        y = GVC.remove_memory(x, 3e-4, 1e-4)
        np.testing.assert_allclose(y, [[-0.1, 1000.0 - 0.1, 1000.0 - 0.3, -0.3]])

    def test_fit_recovers_injected_leak(self):
        rng = np.random.default_rng(1)
        data = []
        for _ in range(3):
            s = rng.integers(0, 1600, (150, 300)).astype(float)               # true electrons per pixel
            x = s + rng.normal(0, 0.15, s.shape)
            x[:, 1:] += 3.0e-4 * s[:, :-1]                                    # leak from the pixel read before
            data.append((x[:, 1:-1].ravel(), x[:, :-2].ravel(), x[:, 2:].ravel()))
        kL, kR, p0, p1 = GVC.fit_memory_coefficients(data)
        self.assertAlmostEqual(kL, 3.0e-4, delta=0.2e-4)
        self.assertAlmostEqual(kR, 0.0, delta=0.2e-4)
        self.assertGreater(p1, 1.5 * p0)                                      # the comb gets sharper


if __name__ == "__main__":
    unittest.main()
