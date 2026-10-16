#!/usr/bin/env python3
"""Gain as a function of the charge in a pixel, g(n), from skipper-CCD images.

Why: the usual gain is the spacing of the 0e- and 1e- peaks. The amplifier gain falls
slowly as the charge in the pixel grows (~3% lower at ~1000 e- in the AMT images), so
cluster energies converted with the 1e- gain come out too low at keV energies.

How: pixel values in a skipper image form a comb, one peak per electron count. Wherever
pixels hold many electrons (X-ray clusters, tracks) the comb continues; its LOCAL spacing
at n electrons is the local gain g(n) = dADU/dn. It is measured with a periodogram of
the pixel values in windows of charge (Rayleigh power |<exp(2 pi i x / g)>| maximised over
trial g). Only "clean" rows are used: where the readout has just seen a lot of charge the
comb is blurred (checked per block of rows with the 1-30 e- comb).

Fit: g(n) = a + b * min(n, n_cap)   (a = low-charge gain, b < 0). Over the measured range
(up to ~1000 e-) the data are linear; the curvature cannot be constrained. Above the
highest measured charge n_max the line is extrapolated up to n_cap = 3 * n_max
(--extrap-factor) and held constant beyond: on the AMT Am-241 data this gave 3.746-3.748
eV/e- for the 59.5 keV line on all three CCDs (unlimited extrapolation overcorrects CCDs
with very compact clusters). The ADU of a pixel with n electrons is A(n) = integral_0^n g;
the correction converts ADU -> electrons with A^-1.

Usage:
  gain_vs_charge.py measure IMAGES... -o OUTDIR [--name NAME] [--ext 1 2 3 4]
      -> OUTDIR/NAME_gain_vs_charge.json (per extension: g(n) points, fit, clean rows)
         OUTDIR/NAME_gain_vs_charge.png
  gain_vs_charge.py apply MODEL.json CLUSTERS.root... -o OUTDIR
      -> OUTDIR/<same name>.root with pixels_E and Energy recomputed with g(n)
(`measure` is the default: gain_vs_charge.py IMAGES... -o OUTDIR also works.)

Images of the same CCDs (same module / ACM) can be given together for more statistics.
Needs numpy, scipy, astropy (measure) and PyROOT (apply); uses autoconfig_panaSKImg.py
from the same folder for the geometry and the low-charge gain.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

E_PER_E = 3.74          # eV per electron used by panaSKImg (pysimdamicm units.e2eV)
BLOCK = 25              # rows per block for the clean-row test
N_EDGES = np.r_[20, 35, 55, 80, 110, 150, 200, 270, 360, 480, 640, 850, 1130, 1500, 2000, 2700, 3600, 5000, 7000]


# =============================================================================
# model
# =============================================================================
def g_model(n, a, b, n_cap):
    """Local gain [ADU/e-] at n electrons: linear, held constant above n_cap."""
    return a + b * np.minimum(np.asarray(n, float), n_cap)


def adu_of_n(n, a, b, n_cap):
    """A(n) = integral_0^n g(m) dm (closed form)."""
    n = np.asarray(n, float)
    m = np.minimum(n, n_cap)
    return a * m + 0.5 * b * m * m + (a + b * n_cap) * np.maximum(n - n_cap, 0)


def n_of_adu(adu, model):
    """Electrons of pixels with the given ADU (inverse of A; negative ADU: /a)."""
    a, b, ncap = model["a"], model["b"], model["n_cap"]
    adu = np.asarray(adu, float)
    out = adu / a
    pos = adu > 0
    A_cap = adu_of_n(ncap, a, b, ncap)
    low = pos & (adu <= A_cap)
    if b != 0:
        out[low] = (-a + np.sqrt(np.maximum(a * a + 2 * b * adu[low], 0))) / b
    else:
        out[low] = adu[low] / a
    high = pos & (adu > A_cap)
    out[high] = ncap + (adu[high] - A_cap) / (a + b * ncap)
    return out


# =============================================================================
# measurement
# =============================================================================
def periodogram(v, gs):
    ph = 2j * np.pi * v
    return np.array([np.abs(np.mean(np.exp(ph / g))) for g in gs])


def best_period(v, g0, lo=0.85, hi=1.10):
    gs = np.linspace(g0 * lo, g0 * hi, 351)
    p = periodogram(v, gs)
    i = int(np.argmax(p))
    fine = np.linspace(gs[max(i - 3, 0)], gs[min(i + 3, len(gs) - 1)], 121)
    pf = periodogram(v, fine)
    j = int(np.argmax(pf))
    return fine[j], pf[j]


def ext_pixels(data, cfg, g0):
    """Pedestal-subtracted, comb-aligned active pixels (rows x cols) of one extension."""
    r0 = cfg["id_row_start"] + 3                     # skip the first rows (transient)
    a0, a1 = max(cfg["n_cols_prescan"], 1), cfg["ccd_cols"]
    d = data[r0:].astype(float)
    if d.shape[1] - (a1 + 2) >= 3:
        x = d[:, a0:a1] - np.median(d[:, a1 + 2:], axis=1, keepdims=True)
    else:
        # no overscan (windowed readout): row median of the active region (zero-electron peak)
        x = d[:, a0:a1] - np.median(d[:, a0:a1], axis=1, keepdims=True)
    near = np.where(np.abs(x) < 3 * g0, x, np.nan)
    with np.errstate(all="ignore"):
        ph = np.nan_to_num(np.angle(np.nanmean(np.exp(2j * np.pi * near / g0), axis=1, keepdims=True)))
    return x - g0 * ph / (2 * np.pi)


def clean_rows(x, g0, frac=0.6):
    """Boolean mask of rows in blocks whose 1-30 e- comb is clearly visible."""
    nb = int(math.ceil(x.shape[0] / BLOCK))
    vis = np.zeros(nb)
    for b in range(nb):
        v = x[b * BLOCK:(b + 1) * BLOCK]
        v = v[(v > 0.5 * g0) & (v < 30 * g0)]
        vis[b] = np.abs(np.mean(np.exp(2j * np.pi * v / g0))) if v.size > 300 else 0.0
    good = vis >= frac * vis.max() if vis.max() > 0 else vis > 1
    mask = np.repeat(good, BLOCK)[:x.shape[0]]
    return mask, vis


def measure(images, exts=None, verbose=True, extrap_factor=3.0):
    """{ext: result} from a list of images of the same CCDs."""
    import argparse as _ap
    from astropy.io import fits
    import autoconfig_panaSKImg as A

    ref = json.loads(A.REFERENCE_JSON)
    hdul0 = fits.open(images[0])
    exts = exts or [e for e in ref["input"]["image"].get("extensions", [1, 2, 3, 4]) if e < len(hdul0)]
    _, cfg, _, _ = A.geometry(hdul0[0].header, hdul0, ref["input"]["image"], _ap.Namespace(ccd_cols=None))
    sys_argv = sys.argv
    sys.argv = ["x", images[0], "-o", "unused.json"]
    aargs = A.parse_args()
    sys.argv = sys_argv
    gp = A.gain_priors(ref, [f"EXT{e}" for e in exts], aargs)

    results = {}
    for e in exts:
        name = f"EXT{e}"
        r = A.analyse_extension(hdul0[e].data, name, cfg, aargs, gp.get(name, 1.0))
        if r.get("dead") or not r.get("resolved"):
            results[name] = dict(ok=False, reason="no resolved electron peaks (dead, flooded or too noisy)")
            if verbose:
                print(f"  {name}: skipped ({results[name]['reason']})")
            continue
        g0 = float(r["gain"])
        vals, nrows_clean, nrows = [], 0, 0
        for f in images:
            x = ext_pixels(fits.open(f)[e].data, cfg, g0)
            mask, _ = clean_rows(x, g0)
            nrows += x.shape[0]
            nrows_clean += int(mask.sum())
            v = x[mask]
            vals.append(v[v > N_EDGES[0] * 0.9 * g0])
        v_all = np.concatenate(vals) if vals else np.array([])
        # refine the low-charge gain on the clean pixels with the same method
        pts = []
        misses = 0
        for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
            v = v_all[(v_all > lo * g0) & (v_all < hi * g0)]
            if v.size < 400:
                break
            if v.size > 300000:
                v = np.random.default_rng(0).choice(v, 300000, replace=False)
            g, R = best_period(v, g0)
            sig = R > 4 / np.sqrt(v.size)
            err = g / (2 * np.pi * ((hi - lo) / np.sqrt(12)) * R * np.sqrt(v.size)) if R > 0 else float("nan")
            pts.append(dict(n=float(np.sqrt(lo * hi)), n_lo=int(lo), n_hi=int(hi), g=float(g), err=float(err),
                            visibility=float(R), npix=int(v.size), significant=bool(sig)))
            misses = 0 if sig else misses + 1
            if misses >= 2:
                break
        good = [p for p in pts if p["significant"]]
        res = dict(ok=False, g_1e=g0, clean_rows=nrows_clean, rows=nrows, points=pts)
        if len(good) >= 4:
            n = np.array([p["n"] for p in good]); g = np.array([p["g"] for p in good])
            (b, a), cov = np.polyfit(n, g, 1, cov=True)
            nmax = float(max(pp["n_hi"] for pp in good))
            ncap = extrap_factor * nmax
            resid = g - (a + b * n)
            res.update(ok=True, model=dict(form="a + b*min(n, n_cap)", a=float(a), b=float(b), n_cap=float(ncap),
                                           a_err=float(np.sqrt(cov[1, 1])), b_err=float(np.sqrt(cov[0, 0]))),
                       rms=float(resid.std()), n_max_measured=nmax,
                       drop_at_1000=float(1 - g_model(1000, a, b, ncap) / a))
            # not trustworthy: the comb barely extends to high charge, or the gain seems to rise
            if nmax < 400 or b > 0:
                res.update(ok=False, reason=f"unreliable curve (measured to {nmax:.0f} e-, slope "
                                            f"{b * 1e5:+.1f}e-5/e-): no correction")
        else:
            res["reason"] = f"comb visible in only {len(good)} charge bins (need 4)"
        results[name] = res
        if verbose:
            if res["ok"]:
                m = res["model"]
                print(f"  {name}: g(n) = {m['a']:.4f} {m['b'] * 1e5:+.2f}e-5 * min(n, {m['n_cap']:.0f})   measured up to "
                      f"{res['n_max_measured']:.0f} e-, {res['drop_at_1000'] * 100:.1f}% lower at 1000 e-  "
                      f"({nrows_clean}/{nrows} clean rows; 1e- gain {g0:.4f})")
            else:
                print(f"  {name}: no g(n) fit ({res.get('reason')})")
    return results


def plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (name, r) in enumerate(sorted(results.items())):
        if not r.get("points"):
            continue
        col = f"C{k}"
        good = [p for p in r["points"] if p["significant"]]
        n = np.array([p["n"] for p in good]); g = np.array([p["g"] for p in good])
        norm = r["model"]["a"] if r.get("ok") else (g[0] if g.size else 1)
        for a_, s in ((ax[0], 1), (ax[1], norm)):
            a_.plot(n, g / s, "o", color=col, label=name)
            if r.get("ok"):
                grid = np.geomspace(10, max(8000, r["n_max_measured"]), 200)
                m = r["model"]
                a_.plot(grid, g_model(grid, m["a"], m["b"], m["n_cap"]) / s, "-", color=col, alpha=0.6)
                a_.axvline(r["n_max_measured"], color=col, ls=":", lw=0.8)
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlabel("electrons in the pixel"); a_.grid(alpha=0.3)
    ax[0].set_ylabel("local gain [ADU/e-]"); ax[1].set_ylabel("gain / low-charge gain")
    ax[0].set_title(title); ax[1].set_title("relative (dotted: highest measured charge)")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


# =============================================================================
# correction of cluster files
# =============================================================================
def apply(model_json, rootfiles, outdir, verbose=True, anchor="comb"):
    """Recompute pixels_E and Energy of panaSKImg cluster files with g(n).

    panaSKImg: pixel energy [keV] = ADU / g_fit * 3.74e-3 (g_fit = info ext<N>_gainext),
    Energy = sum of pixels_E. Corrected: ADU -> electrons with A^-1 of the g(n) model.
    anchor="comb" (default): the whole conversion ADU -> electrons uses the comb's g(n),
      including its low-charge gain a (one curve for all images of a CCD). In the DM24-26
      Image 5 test this aligned the front-side Fe-55 K-alpha of 11 CCDs to 5.864 +- 0.034
      keV (original: 5.834 +- 0.144).
    anchor="fit": only the charge dependence is corrected; at low charge the image's own
      fitted gain is kept (g(n) scaled so that g(0) = g_fit). Overshot (+0.8%) in that test,
      because the per-image 0e-/1e- gain fits were off by 1-4% on many CCDs.
    The extension is taken from the file name (_ext<N>.root) or, for a file with all
    extensions, from each cluster's row (needs 'rows_per_ext' in the model file).
    Returns [(input, output, n_clusters, median ratio new/old for E > 1 keV)].
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)
    models = json.load(open(model_json))["extensions"]
    os.makedirs(outdir, exist_ok=True)
    out = []
    for path in rootfiles:
        base = os.path.basename(path)
        import re
        m = re.search(r"_ext(\d)\.root$", base)
        if not m:
            print(f"  {base}: no _ext<N> in the name, skipped (run panaSKImg per extension, --ext N)")
            continue
        ext = f"EXT{m.group(1)}"
        mod = models.get(ext, {})
        fin = ROOT.TFile.Open(path)
        tin = fin.Get("clustersRec")
        info = fin.Get("info")
        if not tin or not isinstance(tin, ROOT.TTree):
            print(f"  {base}: no clusters, skipped")
            fin.Close()
            continue
        gfit = None
        if info:
            info.GetEntry(0)
            br = f"ext{m.group(1)}_gainext"
            if info.GetBranch(br):
                gfit = float(getattr(info, br))
        outpath = os.path.join(outdir, base)
        fout = ROOT.TFile(outpath, "RECREATE")
        tout = tin.CloneTree(0)
        ratios = []
        for i in range(tin.GetEntries()):
            tin.GetEntry(i)
            if mod.get("ok") and gfit:
                pe, en = tin.pixels_E, tin.Energy
                for k in range(pe.size()):
                    v = np.array(pe[k], float)
                    adu = v * 1e3 / E_PER_E * gfit
                    if anchor == "fit":
                        adu = adu * mod["model"]["a"] / gfit     # same curve shape, g(0) = g_fit
                    vn = n_of_adu(adu, mod["model"]) * E_PER_E * 1e-3
                    for j in range(len(vn)):
                        pe[k][j] = float(vn[j])
                    old = en[k]
                    en[k] = float(vn.sum())
                    if old > 1.0:
                        ratios.append(en[k] / old)
            tout.Fill()
        tout.Write()
        for name in ("info", "process_config"):
            t = fin.Get(name)
            if t:
                t.CloneTree(-1).Write()
        fout.Close()
        fin.Close()
        med = float(np.median(ratios)) if ratios else float("nan")
        out.append((path, outpath, len(ratios), med))
        if verbose:
            note = "" if mod.get("ok") else "  (no g(n) model for this extension: copied unchanged)"
            print(f"  {base}: {ext}, fit gain {gfit}, median E_new/E_old (E > 1 keV) = {med:.4f}{note}")
    return out


# =============================================================================
# CLI
# =============================================================================
def main():
    argv = sys.argv[1:]
    if argv and argv[0] not in ("measure", "apply", "-h", "--help"):
        argv = ["measure"] + argv
    ap = argparse.ArgumentParser(description="Gain as a function of electrons per pixel, g(n), from the "
                                             "electron comb; and the cluster-energy correction.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="images -> g(n) points, fit, plot")
    m.add_argument("images", nargs="+", help="images of the same CCDs (.fz, .fits)")
    m.add_argument("-o", "--outdir", required=True)
    m.add_argument("--name", default=None, help="output name (default: first image name)")
    m.add_argument("--ext", type=int, nargs="+", default=None, help="extensions (default: all in the reference)")
    m.add_argument("--extrap-factor", type=float, default=3.0,
                   help="extrapolate the linear g(n) up to this x the highest measured charge, constant above (default 3)")
    a = sub.add_parser("apply", help="correct cluster ROOT files with a measured g(n)")
    a.add_argument("model", help="*_gain_vs_charge.json from 'measure'")
    a.add_argument("rootfiles", nargs="+", help="panaSKImg_clustersRec_*_extN.root")
    a.add_argument("-o", "--outdir", required=True)
    a.add_argument("--anchor", choices=("comb", "fit"), default="comb",
                   help="low-charge gain: the comb's gain a (default; one g(n) for all images) or the "
                        "image's fitted gain (only the charge dependence corrected)")
    args = ap.parse_args(argv)

    if args.cmd == "measure":
        os.makedirs(args.outdir, exist_ok=True)
        name = args.name or os.path.basename(args.images[0]).split(".")[0]
        print(f"g(n) from {len(args.images)} image(s):")
        res = measure(args.images, args.ext, extrap_factor=args.extrap_factor)
        jpath = os.path.join(args.outdir, f"{name}_gain_vs_charge.json")
        with open(jpath, "w") as fh:
            json.dump({"images": [os.path.abspath(i) for i in args.images], "e_per_electron_eV": E_PER_E,
                       "extensions": res}, fh, indent=1)
        plot(res, os.path.join(args.outdir, f"{name}_gain_vs_charge.png"), name)
        print(f"-> {jpath}")
    else:
        apply(args.model, args.rootfiles, args.outdir, anchor=args.anchor)


if __name__ == "__main__":
    main()
