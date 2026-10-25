"""MF disk+grow (MFD+G) on the Am-241 (AMTEst, 10 Image 8s, board 106) images vs WADERs default and B (auto).
Real lines, injection of 17.75 / 59.54 keV clusters, and how tracks are split."""
import argparse, json, os, re, sys
import numpy as np
from astropy.io import fits
from scipy import optimize
from multiprocessing import Pool
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import agent_methods as AM
import cluster_methods as CM
import gain_vs_charge as GVC
import autoconfig_panaSKImg as A

AMT = os.path.expanduser("~/Documents/JH1_Testing/AMTEst")
RUN = os.path.join(AMT, "WADERs_analysis_fz_files")
CUR = json.load(open(os.path.join(P, "coherent_test", "coherent_vs_incoherent.json")))["AMT"]["coherent"]
IMGS = [os.path.basename(l.strip()).split(".")[0] for l in open(os.path.join(AMT, "fz_files.txt")) if l.strip()]
LINES = [("Np La", 13.93), ("Np Lb", 17.75), ("Np Lg", 20.80), ("g 26.3", 26.34), ("g 59.5", 59.54)]


def load(stem, ext):
    cal = fits.open(os.path.join(RUN, f"{stem}_EXT{ext}_waders.fits.bz2"))[0].data.astype(float)
    import ROOT
    tf = ROOT.TFile.Open(os.path.join(RUN, "root", f"panaSKImg_clustersRec_{stem}_ext{ext}.root"))
    info = tf.Get("info"); info.GetEntry(0); gfit = float(getattr(info, f"ext{ext}_gainext")); tf.Close()
    Q = GVC.electrons_of_adu(cal * gfit, CUR[f"EXT{ext}"]["model"]).reshape(cal.shape)
    h = fits.open(os.path.join(AMT, stem + ".fz"))
    ref = json.loads(A.REFERENCE_JSON)
    _, cfg, _, _ = A.geometry(h[0].header, h, ref["input"]["image"], argparse.Namespace(ccd_cols=None))
    valid = np.zeros_like(Q, bool)
    valid[cfg["id_row_start"] + 3:Q.shape[0] - 2, cfg["n_cols_prescan"] + 2:min(cfg["ccd_cols"], Q.shape[1]) - 2] = True
    Q = np.where(valid, Q, 0.0)
    log = open(os.path.join(RUN, "logs", f"{stem}_EXT{ext}_panaSKImg.log")).read()
    m = re.search(r"Sigma for SPR: ([0-9.eE+-]+) ADUs \(([0-9.eE+-]+) ADU/e-", log)
    sw = float(m.group(1)) / float(m.group(2))
    return Q, valid, sw


def methods(Q, valid, sw, bg):
    lam_med = float(np.median(bg.lam[valid]))
    k = AM.poisson_tail_k(lam_med, 0.10)[0]
    k = max(int(k), 2)
    return {
        "WADERs default": lambda Qi: CM.wader(Qi, sw, 10, 30)[0],
        f"B auto (pixel >= {k} e-)": lambda Qi: CM.topo(Qi, 0.0, 9.5, k - 0.5, 0),
        "MFD+G": lambda Qi: AM.mf_detect(Qi, valid, bg)[0],
    }, lam_med, k


def fit_line(E, E0):
    """Gaussian + linear background around E0 (as docs/report_scripts/make_gain_correction_Am241.py)."""
    low = E0 < 40; bw = 0.4 if low else 0.15; w0 = 0.03 * E0 if low else 0.006 * E0
    h, ed = np.histogram(E, bins=np.arange(0.85 * E0, 1.06 * E0, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    if h.sum() < 60: return None
    pk = c[np.argmax(np.convolve(h, np.ones(3) / 3, mode="same"))]
    lo, hi = pk - (2.5 if low else 3.5) * w0, pk + (2.5 if low else 3.5) * w0
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - pk)
    try:
        p, cov = optimize.curve_fit(f, c, h, p0=[max(h.max() - h.min(), 1), pk, w0, h.min(), 0], sigma=np.sqrt(np.maximum(h, 1)),
                                    maxfev=20000, bounds=([0, lo, 0.2 if low else bw / 2, -np.inf, -np.inf], [np.inf, hi, 3 * w0, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h - f(c, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 5, 1)
    err = np.sqrt(cov[1, 1]) * np.sqrt(max(chi, 1))
    if err > 0.02 * E0 or p[0] / np.sqrt(max(p[3], 1)) < 5 or (low and abs(p[2]) > 1.0): return None
    return float(p[1]), float(err), float(abs(p[2]))


def job(args):
    stem, ext, do_inject = args
    Q, valid, sw = load(stem, ext)
    bg = AM.background_map(Q, valid)
    meth, lam_med, k = methods(Q, valid, sw, bg)
    out = dict(stem=stem, ext=ext, lam_med=lam_med, k=k, sigma_wader=sw, real={}, inject={}, tracks={})
    found = {}
    for name, f in meth.items():
        cl = f(Q)
        found[name] = cl
        out["real"][name] = [float(c["E"]) for c in cl if np.isfinite(c.get("E", np.nan))]
    # tracks: WADERs clusters above 80 keV -> how many MFD+G clusters have their centre inside the WADERs cluster pixels
    _, lab = CM.wader(Q, sw, 10, 30)
    big = [c for c in found["WADERs default"] if c["E"] > 80]
    nsplit = []
    for c in big:
        L = lab[int(round(c["y"])), int(round(c["x"]))]
        if L <= 0: continue
        n = sum(1 for m in found["MFD+G"] if 0 <= int(round(m["y"])) < lab.shape[0] and 0 <= int(round(m["x"])) < lab.shape[1]
                and lab[int(round(m["y"])), int(round(m["x"]))] == L)
        nsplit.append(n)
    out["tracks"] = dict(n_big=len(big), split_counts=nsplit)
    if do_inject:
        from scipy import ndimage
        rng = np.random.default_rng(11)
        busy = ndimage.binary_dilation(Q > 8.0, iterations=12) | ~ndimage.binary_erosion(valid, iterations=14)
        ys, xs = np.nonzero(~busy)
        for E0 in (17.75, 59.54):
            ne0 = int(round(E0 / CM.E_PER_E))
            for s in (0.3, 0.8, 1.4):
                idx = rng.permutation(ys.size); pos = []
                for i in idx:
                    y, x = ys[i], xs[i]
                    if all((y - a) ** 2 + (x - b) ** 2 > 28 ** 2 for a, b in pos):
                        pos.append((y + rng.uniform(-.5, .5), x + rng.uniform(-.5, .5)))
                    if len(pos) >= 60: break
                Qi, counts = CM.inject(Q, rng, ne0, s, pos)
                bgi = AM.background_map(Qi, valid)
                mi, _, _ = methods(Qi, valid, sw, bgi)
                r = {}
                for name, f in mi.items():
                    fl = f(Qi); rat = []
                    for (y, x), ne in zip(pos, counts):
                        best, bd = None, 2.5
                        for c in fl:
                            d = np.hypot(c["y"] - y, c["x"] - x)
                            if d < bd: best, bd = c, d
                        if best is not None and np.isfinite(best["E"]): rat.append(best["E"] / (ne * CM.E_PER_E))
                    rat = np.array(rat)
                    r[name] = dict(eff=len(rat) / len(pos), med=float(np.median(rat)) if rat.size else np.nan,
                                   rms=float(1.4826 * np.median(np.abs(rat - np.median(rat)))) if rat.size else np.nan)
                out["inject"][f"{E0}|{s}"] = r
    json.dump(out, open(os.path.join(HERE, "results", f"{stem}_EXT{ext}.json"), "w"))
    print(stem[-12:], ext, "done", flush=True)
    return True


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    jobs = [(s, e, i < 3) for i, s in enumerate(IMGS) for e in (1, 3, 4)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(HERE, "results", f"{j[0]}_EXT{j[1]}.json"))]
    with Pool(8) as p:
        p.map(job, jobs, chunksize=1)
