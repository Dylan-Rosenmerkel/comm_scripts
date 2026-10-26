"""src/agmx_root.py (A_GMX clusters written as panaSKImg cluster files) on a simulated CCD: three 2000 e- X-rays on a
skipper-like background, a minimal panaSKImg reference file (info / process_config trees) and final JSON made here.
Checks the file structure, the panaSKImg conventions (1-based pixels, keV, Energy = sum pixels_E) and the agmx_*
branches. Skipped without PyROOT / pysimdamicm / astropy.

    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from array import array

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
try:
    import ROOT  # noqa: F401
    import pysimdamicm  # noqa: F401  (availability check)
    from astropy.io import fits
    HAVE = True
except ImportError:
    HAVE = False

H, W, GAIN, E2EV, RUNID = 120, 220, 1.5, 3.74, 7
XRAYS = [(40, 60), (40, 150), (90, 100)]
N_E = 2000


def make_inputs(d):
    """calibrated image (e-), reference panaSKImg file and final JSON for EXT1 of image 'sim'"""
    rng = np.random.default_rng(3)
    Q = rng.poisson(0.01, (H, W)) + rng.normal(0, 0.15, (H, W))
    for y, x in XRAYS:
        np.add.at(Q, (np.round(y + rng.normal(0, .6, N_E)).astype(int), np.round(x + rng.normal(0, .6, N_E)).astype(int)), 1.0)
    fits.PrimaryHDU(Q.astype(np.float32)).writeto(os.path.join(d, "sim_EXT1_waders.fits"))
    os.makedirs(os.path.join(d, "root")); os.makedirs(os.path.join(d, "json", "sim", "final"))
    json.dump({"input": {"image": {"id_row_start": 0, "n_cols_prescan": 5, "ccd_cols": 210}}},
              open(os.path.join(d, "json", "sim", "final", "sim_EXT1.json"), "w"))
    f = ROOT.TFile(os.path.join(d, "root", "panaSKImg_clustersRec_sim_ext1.root"), "RECREATE")
    t = ROOT.TTree("clustersRec", "clustersRec")
    run, ncl = array("i", [RUNID]), array("i", [0])
    t.Branch("RUNID", run, "RUNID/I"); t.Branch("Nclusters", ncl, "Nclusters/I"); t.Fill(); t.Write()
    info = ROOT.TTree("info", "info")
    g = array("d", [GAIN]); mu = ROOT.std.vector("std::vector<double>")(); row = ROOT.std.vector("double")()
    for _ in range(H):
        row.push_back(1000.0)
    mu.push_back(row)
    sm, idccd, nf = array("d", [0.15 * GAIN]), array("i", [106]), array("i", [4])     # 0.15 e- noise
    info.Branch("ext1_gainext", g, "ext1_gainext/D"); info.Branch("EXT1_pedestal_mu_row_ADU", mu)
    info.Branch("EXT1_pedestal_sigma_row_mean_ADU", sm, "EXT1_pedestal_sigma_row_mean_ADU/D")
    info.Branch("idccd", idccd, "idccd/I"); info.Branch("nfile", nf, "nfile/I"); info.Fill(); info.Write()
    pc = ROOT.TTree("process_config", "process_config")
    e = array("d", [E2EV]); pc.Branch("e2eV", e, "e2eV/D"); pc.Fill(); pc.Write()
    f.Close()


@unittest.skipUnless(HAVE, "needs PyROOT, pysimdamicm and astropy")
class AGMXRootFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import agmx_root as AR
        cls.AR = AR
        cls.d = tempfile.mkdtemp()
        make_inputs(cls.d)
        cls.files, cls.failed = AR.run(cls.d, os.path.join(cls.d, "root_agmx"), jobs=1, log=lambda m: None)
        f = ROOT.TFile.Open(cls.files[0])
        cls.keys = sorted(k.GetName() for k in f.GetListOfKeys())
        t = f.Get("clustersRec"); t.GetEntry(0)
        cls.t = {b.GetName(): getattr(t, b.GetName()) for b in t.GetListOfBranches()}
        cls.v = {k: ([list(x) for x in v] if len(v) and hasattr(v[0], "__len__") else list(v)) if hasattr(v, "__len__") else v
                 for k, v in cls.t.items()}
        cls.cfg = json.loads(f.Get("agmx_config").GetTitle())
        f.Close()

    def test_file_structure(self):
        self.assertEqual(self.failed, [])
        self.assertEqual(len(self.files), 1)
        self.assertEqual(self.keys, ["agmx_config", "clustersRec", "info", "process_config"])
        for b in ("Energy", "PosX", "PosY", "pixels_x", "pixels_y", "pixels_E", "fSTD_X", "wSTD_X", "Npix", "has_seed", "ccd", "sigma_seed_eV", "qmin_cluster_seed",
                  "closest_cls_dist", "Energy_PS", "Energy_AVG", "agmx_cls", "agmx_blend", "agmx_ped", "agmx_regime"):
            self.assertIn(b, self.v)
        self.assertEqual(self.v["RUNID"], RUNID)
        self.assertEqual(self.cfg["version"], self.AR.AGMX_VERSION)
        # electron-counting image (0.15 e- noise): the few-electron candidate catalogue is written next to the file
        fe = os.path.join(self.d, "root_agmx", "fe_candidates", "sim_ext1_fe_candidates.json")
        self.assertTrue(os.path.exists(fe))
        # no WADERs clusters in the reference: ccd / seed taken from its info tree (as panaSKImg)
        self.assertEqual(set(self.v["ccd"]), {106})
        self.assertAlmostEqual(self.v["sigma_seed_eV"][0], 0.15 * E2EV, places=6)
        self.assertAlmostEqual(self.v["qmin_cluster_seed"][0], 30 * 0.15 * E2EV * 1e-3, places=8)

    def test_xrays_found_with_panaskimg_conventions(self):
        E, X, Y = np.array(self.v["Energy"]), np.array(self.v["PosX"]), np.array(self.v["PosY"])
        for y, x in XRAYS:
            # PosX/PosY are 1-based like the pixel lists
            i = np.argmin(np.hypot(X - 1 - x, Y - 1 - y))
            self.assertLess(np.hypot(X[i] - 1 - x, Y[i] - 1 - y), 0.3)
            self.assertLess(abs(E[i] / (N_E * E2EV * 1e-3) - 1), 0.02)
            self.assertAlmostEqual(sum(self.v["pixels_E"][i]), E[i], places=6)
            self.assertGreaterEqual(min(self.v["pixels_x"][i]), 1)
            # Energy_PS = sum of the pedestal-subtracted ADU of the cluster's pixels; Energy_AVG adds the row pedestal
            self.assertAlmostEqual(self.v["Energy_AVG"][i] - self.v["Energy_PS"][i], 1000.0 * self.v["Npix"][i], places=2)
            self.assertEqual(self.v["agmx_cls"][i], self.AR.CLS_CODES["x-ray"])
            self.assertLess(self.v["agmx_blend"][i], 0.02)
            self.assertLess(self.v["agmx_ped"][i], 10)

    def test_closest_cluster(self):
        X, Y = np.array(self.v["PosX"]), np.array(self.v["PosY"])
        seeded = np.array(self.v["has_seed"], bool)
        for i in np.where(seeded)[0][:20]:
            d = np.hypot(X - X[i], Y - Y[i]); d[i] = np.inf; d[~seeded] = np.inf
            d[(X == X[i]) | (Y == Y[i])] = np.inf          # panaSKImg's rule for excluding itself
            if np.isfinite(d.min()):
                self.assertAlmostEqual(self.v["closest_cls_dist"][i], d.min(), places=4)


@unittest.skipUnless(HAVE, "needs PyROOT, pysimdamicm and astropy")
class AGMXRootEdgeCases(unittest.TestCase):
    def test_reference_without_cluster_tree(self):
        """a panaSKImg file with info / process_config but no clustersRec tree"""
        import agmx_root as AR
        d = tempfile.mkdtemp()
        make_inputs(d)
        ref = os.path.join(d, "root", "panaSKImg_clustersRec_sim_ext1.root")
        f = ROOT.TFile(ref, "UPDATE"); f.Delete("clustersRec;*"); f.Close()
        r = AR.read_reference(ref, 1, os.path.join(d, "json", "sim", "final", "sim_EXT1.json"))
        self.assertEqual(r["ccd"], 106)
        self.assertAlmostEqual(r["sigma_seed_eV"], 0.15 * E2EV, places=6)

    def test_valid_mask_binned_columns(self):
        """ccd_cols is in unbinned columns: with NSBIN = 2 the overscan starts at ccd_cols / 2"""
        import agmx_root as AR
        d = tempfile.mkdtemp(); fj = os.path.join(d, "x.json")
        json.dump({"input": {"image": {"id_row_start": 0, "n_cols_prescan": 0, "ccd_cols": 200}}}, open(fj, "w"))
        self.assertEqual(np.nonzero(AR.valid_mask((50, 150), fj, nsbin=1)[10])[0].max(), 147)
        self.assertEqual(np.nonzero(AR.valid_mask((50, 150), fj, nsbin=2)[10])[0].max(), 97)


if __name__ == "__main__":
    unittest.main()
