"""Tests of the Am-241 note's fit model (docs/am241_note/amfit.py) on synthetic spectra - no data needed.

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "am241_note"))
import amfit  # noqa: E402
import common  # noqa: E402


class LineList(unittest.TestCase):
    def test_relative_intensities_sum_to_one(self):
        """Inside each Np group the relative intensities are normalised, so A_g is the group's total counts."""
        m = amfit.Model(None)
        for g, lines in m.groups:
            self.assertAlmostEqual(sum(r for _, r in lines), 1.0, places=12, msg=g)

    def test_compton_edge(self):
        """The Compton edge of the 59.54 keV gamma ray: E * 2E / (m_e c^2 + 2E) = 11.25 keV."""
        E, me = 59.54, 511.0
        self.assertAlmostEqual(E * 2 * E / (me + 2 * E), amfit.E_C, places=2)


class ModelShape(unittest.TestCase):
    def test_group_area_equals_amplitude(self):
        """A group placed well inside the range integrates to its amplitude (counts) when summed over bins."""
        m = amfit.Model(None, lo=5.0, hi=40.0, bw=0.05)
        x = np.arange(m.lo + m.bw / 2, m.hi, m.bw)
        p = np.zeros(len(m.names)); p[0], p[1], p[2], p[7] = 1.0, 0.2, 0.003, 1.0
        i = 8 + [g for g, _ in m.groups].index("Lb")
        p[i] = 1000.0
        self.assertAlmostEqual(m(x, p).sum(), 1000.0, delta=1.0)


class FitRecovery(unittest.TestCase):
    def test_recovers_energy_scale(self):
        """Fit a Poisson sample drawn from the model itself: the energy scale k comes back within its uncertainty."""
        rng = np.random.default_rng(1)
        m = amfit.Model(None)
        x = np.arange(m.lo + m.bw / 2, m.hi, m.bw)
        p_true = np.array([0.985, 0.1, 0.006, 30.0, -1.0, 0.0, 300.0, 1.2] + [50, 400, 30, 2000, 1500, 1200], float)
        counts = rng.poisson(m(x, p_true))
        # Model.fit takes energies, so turn the bin counts back into energies at the bin centres
        E = np.repeat(x, counts)
        r = m.fit(E, 0.97)                         # start away from the truth: the fit has to move k
        k, dk = r["p"][0], r["err"][0]
        self.assertLess(dk, 0.003)                  # a real constraint (a fit that does nothing has no small error)
        self.assertLess(abs(k - 0.985), 4 * dk + 1e-4)
        self.assertLess(r["devndf"], 1.6)


class Common(unittest.TestCase):
    def test_load_clusters_derived_fields(self):
        """load_clusters adds the cluster width sigma_xy = sqrt((sx^2 + sy^2)/2), Qmax/E and the bounding box."""
        with tempfile.TemporaryDirectory() as d:
            n = 3
            arr = dict(Energy=np.array([10.0, 20.0, 30.0]), E_orig=np.ones(n), fSTD_X=np.array([0.3, 0.4, 0.0]),
                       fSTD_Y=np.array([0.4, 0.3, 0.6]), Qmax=np.array([5.0, 5.0, 3.0]), minX=np.zeros(n),
                       maxX=np.array([1.0, 2.0, 0.0]), minY=np.zeros(n), maxY=np.array([0.0, 1.0, 3.0]), img=np.zeros(n))
            np.savez(common.cluster_table(d, 4), **arr)
            c = common.load_clusters(d, 4)
            np.testing.assert_allclose(c["sxy"], np.sqrt(0.5 * (arr["fSTD_X"] ** 2 + arr["fSTD_Y"] ** 2)))
            np.testing.assert_allclose(c["qfrac"], [0.5, 0.25, 0.1])
            np.testing.assert_array_equal(c["nx"], [2, 3, 1])
            np.testing.assert_array_equal(c["ny"], [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
