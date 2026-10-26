"""Check that src/agmx_root.py writes panaSKImg's format exactly: the clusters of an original panaSKImg file are fed
through the writer (their pixels and charges as if A_GMX had found them) and every branch of the result is compared
with the original file.   python agmx_root_check.py PANASKIMG_ROOT WADERS_FITS EXT

Differences are allowed only in numerically degenerate cases, where panaSKImg's own result depends on the last bits
of the inputs (the file stores pixels_E as float, so a replay cannot reproduce them bit for bit):
  fSTD_X / fSTD_Y  Gaussian fit of a profile 1-3 bins wide (sigma not constrained)
  pcaTheta         0 vs +-pi: the same axis, eigenvector sign (pcaAxis*, pcaU, pcaV: original = -new)
  pcaElongation, pcaThetaAnalytic, ... : 1-2 pixel clusters (lambda ~ 0)
  closest_cls_*    a neighbour in the same row/column (panaSKImg excludes 'itself' by PosX != AND PosY !=)"""
import os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import agmx_root as AR
import ROOT


def branches(path):
    tf = ROOT.TFile.Open(path); t = tf.Get("clustersRec"); t.GetEntry(0)
    out = {}
    for b in t.GetListOfBranches():
        v = getattr(t, b.GetName())
        try:
            out[b.GetName()] = [list(x) for x in v] if len(v) and hasattr(v[0], "__len__") else list(v)
        except TypeError:
            out[b.GetName()] = v
    keys = sorted(k.GetName() for k in tf.GetListOfKeys())
    tf.Close()
    return out, keys


def main(ref_root, fits_f, ext, nmax=10 ** 9):
    from astropy.io import fits
    ref = AR.read_reference(ref_root, ext)
    cal = fits.open(fits_f)[0].data.astype(float)
    orig, okeys = branches(ref_root)
    n = min(int(orig["Nclusters"]), nmax)
    clusters = []
    for i in range(n):
        px = np.array(orig["pixels_x"][i]) - 1; py = np.array(orig["pixels_y"][i]) - 1
        q = np.array(orig["pixels_E"][i]) / (ref["e2eV"] * 1e-3)            # keV -> e-
        clusters.append(dict(_pix=(py, px, q), cls="x-ray"))
    cls = AR.build_clusters(clusters, ref, cal * ref["gain"], ref["gain"], *cal.shape)
    out = os.path.join(tempfile.mkdtemp(), "check.root")
    AR.write_root(out, cls, ref, ref_root, {"check": True})
    new, nkeys = branches(out)
    print(f"trees: original {okeys}, written {nkeys}")
    miss = sorted(set(orig) - set(new)); extra = sorted(set(new) - set(orig))
    print(f"branches missing in the written file: {miss}\nextra branches: {extra}")
    npix = np.array(orig["Npix"][:n]); span = lambda P: np.array([len(set(p)) for p in P[:n]])
    sx, sy = span(orig["pixels_x"]), span(orig["pixels_y"])
    axis = lambda d: np.isclose(np.array(d["closest_cls_dist_X"][:n], float), 0) | np.isclose(np.array(d["closest_cls_dist_Y"][:n], float), 0)
    explained = {"fSTD_X": sx <= 3, "fSTD_Y": sy <= 3, "fPosX": sx <= 3, "fPosY": sy <= 3,
                 "pcaTheta": (sy == 1) | (sx == 1) | (npix <= 2), "closest": axis(orig) | axis(new)}
    bad, deg = [], []
    for k in sorted(set(orig) & set(new)):
        if k in ("RUNID", "Nclusters"):
            continue
        a, b = orig[k][:n], new[k][:n]
        if a and isinstance(a[0], list):
            diff = np.array([len(x) != len(y) or (len(x) and not np.allclose(np.array(x, float), np.array(y, float), rtol=1e-4, atol=1e-3)) for x, y in zip(a, b)])
        elif isinstance(a, list):
            A, B = np.array(a, float), np.array(b, float)
            diff = ~(np.isclose(A, B, rtol=1e-4, atol=1e-4) | (np.isnan(A) & np.isnan(B)))
            if k.startswith("Energy_"):
                diff = ~np.isclose(A, B, rtol=1e-4, atol=1.0)                # ADU sums of float images
        else:
            diff = np.array([a != b]); 
        if not diff.any():
            continue
        if k.startswith(("pcaAxis", "pcaU", "pcaV")):                    # eigenvector sign: original = -new
            neg = (lambda x, y: len(x) == len(y) and np.allclose(np.array(x, float), -np.array(y, float), rtol=1e-4, atol=1e-3)) if isinstance(a[0], list) \
                else (lambda x, y: np.isclose(x, -y, rtol=1e-4, atol=1e-4))
            diff = diff & ~np.array([bool(neg(x, y)) for x, y in zip(a, b)])
            if not diff.any():
                deg.append((k + " (sign)", 0)); continue
        if k in ("pcaTheta", "pcaThetaAnalytic"):                        # the same axis: theta and theta +- pi
            A, B = np.array(a, float), np.array(b, float)
            diff = diff & ~np.isclose(np.abs(A - B), np.pi, atol=1e-4) & ~np.isclose(np.abs(A - B), 2 * np.pi, atol=1e-4)
            if not diff.any():
                deg.append((k + " (+-pi)", 0)); continue
        ok = explained.get("closest" if k.startswith("closest") else k, (npix <= 2) if k.startswith("pca") else np.zeros(n, bool))
        (deg if np.all(ok[diff] if diff.size == n else False) else bad).append((k, int(diff.sum())))
    print(f"{n} clusters compared")
    print(f"  identical except numerically degenerate cases (branch, clusters): {deg}")
    print(f"  real differences: {bad if bad else 'none'}")
    return not bad and not miss


if __name__ == "__main__":
    ok = main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    sys.exit(0 if ok else 1)
