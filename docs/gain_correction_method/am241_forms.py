"""Am-241 lines (AMTEst, 10 Image 8s, board 106) with the gain correction in its three model forms.
The comb is measured once (gain_vs_charge.measure, all 10 images, incoherent combination); the three
forms are fitted to the same points (gain_vs_charge.fit_model) and applied in memory exactly as
gain_vs_charge.apply (anchor comb). All clusters, as in docs/gain_correction_Am241.pdf.

Purpose: the numbers and spectra behind the Am-241 part of gain_correction_method.tex, i.e. how
well each model form of g(n) (gain vs electrons in a pixel) puts the Np L lines and the 26.3 and
59.5 keV gammas at their true energies.

Inputs (defaults in brackets, see -h):
    --image-list   list of the 10 .fz images            [<amt-dir>/fz_files.txt]
    --root-dir     their run_WADERs cluster files       [<amt-dir>/WADERs_analysis_fz_files/root]
    --data-dir     amt_measure.json, the cached comb measurement; made here (slow) if missing
                                                        [data/ next to this script]
Outputs (in --data-dir):
    am241_forms.json    per model form and extension: gain drop at 1000 e-, nmax, g(0), model;
                        fitted line positions per (extension | form | line); brightest-pixel
                        percentiles of the 59.5 keV clusters
    am241_spectra.json  cluster-energy histograms (0-70 keV, 0.1 keV bins) per extension and form,
                        read by make_figures.py
    and a printed table: measured / true energy per line and form.

Run:  python am241_forms.py            (needs PyROOT, scipy and comm_scripts/src/gain_vs_charge.py)

How to read this file
    clusters    pixel energies of every cluster of one extension, with the image's fitted gain
    fit_line    Gaussian + linear background fit of one line (same as make_gain_correction_Am241.py)
    main        measure (or load) the comb, fit the three forms, re-compute the cluster
                energies with each, fit the lines, write the JSON files, print the table
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy import optimize
import ROOT
ROOT.gROOT.SetBatch(True)
sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/src"))
import gain_vs_charge as GVC

EXTS = (1, 3, 4)
LINES = [("Np La", 13.93), ("Np Lb", 17.75), ("Np Lg", 20.80), ("g 26.3", 26.34), ("g 59.5", 59.54)]


def clusters(ext, rootd):
    """Every cluster of extension ext in the cluster files of rootd, as (pixel energies, gain).

    pixel energies: float array [keV] of the cluster's pixels as panaSKImg wrote them (pixels_E),
    i.e. converted with one gain per image; gain: that image's fitted gain (info tree,
    ext<N>_gainext, ADU/e-). Both are needed to go back to ADU and re-convert with g(n).
    Files without a clustersRec tree are skipped. Returns a list of (array, float).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(rootd, f"panaSKImg_clustersRec_*_ext{ext}.root"))):
        tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec"); info = tf.Get("info")
        if not isinstance(t, ROOT.TTree):
            tf.Close(); continue
        info.GetEntry(0); gfit = float(getattr(info, f"ext{ext}_gainext"))
        for i in range(t.GetEntries()):
            t.GetEntry(i)
            for k in range(t.pixels_E.size()):
                out.append((np.array(list(t.pixels_E[k]), float), gfit))
        tf.Close()
    return out


def fit_line(E, E0):
    """same as docs/report_scripts/make_gain_correction_Am241.py

    Peak near E0 [keV] in the cluster energies E: maximum in [0.85, 1.06] x E0, then Gaussian +
    linear background around it. Returns (mu, err, sigma) in keV, or None unless the peak is clear
    (>= 60 counts, signal/sqrt(background) >= 5, error <= 2% of E0, width < 1 keV below 40 keV).
    The error is scaled up by sqrt(chi2/ndf) when the fit is worse than the counting errors.
    """
    low = E0 < 40
    bw = 0.4 if low else 0.15
    w0 = 0.03 * E0 if low else 0.006 * E0
    h, ed = np.histogram(E, bins=np.arange(0.85 * E0, 1.06 * E0, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    if h.sum() < 60:
        return None
    pk = c[np.argmax(np.convolve(h, np.ones(3) / 3, mode="same"))]
    lo, hi = pk - (2.5 if low else 3.5) * w0, pk + (2.5 if low else 3.5) * w0
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - pk)
    try:
        p, cov = optimize.curve_fit(f, c, h, p0=[max(h.max() - h.min(), 1), pk, w0, h.min(), 0],
                                    sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                    bounds=([0, lo, 0.2 if low else bw / 2, -np.inf, -np.inf], [np.inf, hi, 3 * w0, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h - f(c, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 5, 1)
    err = np.sqrt(cov[1, 1]) * np.sqrt(max(chi, 1))
    bg = max(p[3], 1)
    if err > 0.02 * E0 or p[0] / np.sqrt(bg) < 5 or (low and abs(p[2]) > 1.0):
        return None
    return float(p[1]), float(err), float(abs(p[2]))


def main():
    """Measure/load the comb, fit the forms, correct the cluster energies, fit the lines, write and print.

    Each cluster is corrected as in gain_vs_charge.apply: pixel keV -> ADU with the image's gain,
    ADU -> electrons with g(n) (electrons_of_adu), summed and converted back to keV.
    """
    amt = os.path.expanduser("~/Documents/JH1_Testing/AMTEst")
    ap = argparse.ArgumentParser(description="Am-241 line positions with the three gain-vs-charge model forms "
                                             "(writes am241_forms.json and am241_spectra.json).")
    ap.add_argument("--amt-dir", default=amt, help=f"AMTEst folder (default: {amt})")
    ap.add_argument("--image-list", default=None, help="list of the .fz images (default: <amt-dir>/fz_files.txt)")
    ap.add_argument("--root-dir", default=None,
                    help="run_WADERs cluster files (default: <amt-dir>/WADERs_analysis_fz_files/root)")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                    help="folder of the amt_measure.json cache and of the output JSON files "
                         "(default: data/ next to this script)")
    args = ap.parse_args()
    HERE = args.data_dir
    AMT = args.amt_dir
    ROOTD = args.root_dir or os.path.join(AMT, "WADERs_analysis_fz_files", "root")
    IMAGES = [l.strip() for l in open(args.image_list or os.path.join(AMT, "fz_files.txt")) if l.strip()]

    # ---- the comb measurement (cached) and the three model forms fitted to the same points
    mpath = os.path.join(HERE, "amt_measure.json")
    if not os.path.exists(mpath):
        json.dump(GVC.measure(IMAGES, list(EXTS), verbose=True, model="linear"), open(mpath, "w"), indent=1)
    meas = json.load(open(mpath))
    models = {}
    for form in GVC.MODELS:
        models[form] = {}
        for e in EXTS:
            r = meas[f"EXT{e}"]
            good = [p for p in r["points"] if p["significant"]]
            m, lin, nmax, rms = GVC.fit_model(good, form, 3.0)
            models[form][e] = dict(model=m, nmax=nmax, drop=float(1 - GVC.g_of(1000.0, m) / GVC.g_of(0.0, m)),
                                   ok=r["ok"])

    # ---- corrected energies, line fits and spectra per extension
    res = {"models": {f: {str(e): dict(drop=models[f][e]["drop"], nmax=models[f][e]["nmax"],
                                        g0=float(GVC.g_of(0.0, models[f][e]["model"])), model=models[f][e]["model"])
                          for e in EXTS} for f in GVC.MODELS}, "lines": {}, "maxpix": {}}
    spectra = {}
    for e in EXTS:
        cl = clusters(e, ROOTD)
        E = {"none": np.array([c.sum() for c, _ in cl])}
        for form in GVC.MODELS:
            m = models[form][e]["model"]
            E[form] = np.array([GVC.electrons_of_adu(c * 1e3 / GVC.E_PER_E * g, m).sum() * GVC.E_PER_E * 1e-3 for c, g in cl])
        # brightest pixel of the 59.5 keV clusters [e-], to see how far beyond the measured range they reach
        sel = (E["none"] > 55) & (E["none"] < 62)
        qmax = np.array([c.max() for c, _ in cl])[sel] * 1e3 / GVC.E_PER_E
        res["maxpix"][str(e)] = [float(np.percentile(qmax, q)) for q in (10, 50, 90)] if qmax.size else None
        for k, v in E.items():
            for name, E0 in LINES:
                res["lines"][f"{e}|{k}|{name}"] = fit_line(v, E0)
        spectra[e] = {k: np.histogram(v, bins=np.arange(0, 70.001, 0.1))[0].tolist() for k, v in E.items()}
    json.dump(res, open(os.path.join(HERE, "am241_forms.json"), "w"), indent=1)
    json.dump(spectra, open(os.path.join(HERE, "am241_spectra.json"), "w"))

    # ---- printed table: measured / true energy per line and form
    for e in EXTS:
        print(f"EXT{e}: max pixel of 59.5 keV clusters p10/50/90 [e-] {np.round(res['maxpix'][str(e)] or [], 0)}; "
              + ", ".join(f"{f}: drop {models[f][e]['drop'] * 100:.2f}% nmax {models[f][e]['nmax']:.0f}" for f in GVC.MODELS))
        for name, E0 in LINES:
            print(f"   {name:7s} " + "  ".join(
                f"{k}: {r[0] / E0:.4f}" if (r := res['lines'][f'{e}|{k}|{name}']) else f"{k}:   -   "
                for k in ["none"] + list(GVC.MODELS)))


if __name__ == "__main__":
    main()
