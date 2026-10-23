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


if __name__ == "__main__":
    unittest.main()
