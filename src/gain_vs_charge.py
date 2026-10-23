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

Model of g(n) (--model):
  interpolated (default): monotone cubic (PCHIP) interpolation through the measured points;
      below the first point (~25 e-) g is held at the first point's value, above the last
      point it continues with the slope of a straight line through the last 4 points.
  broken: two straight lines joined at a fitted knot, g = a + b1*min(n, nk) + b2*max(n - nk, 0).
  linear: g = a + b*n (the original model).
On 9 Image 5s (DM07/DM09/DM10, 2025-08-27) a straight line does not describe the measured
points (chi2/ndf > 30; the same curvature in every image of a CCD, and on DM09 EXT2 the gain
even rises by ~4% from 25 to 200 e- before it falls), while the interpolated and broken
forms follow them. For the front-side Fe-55 K-alpha of DM09/DM10 the three forms gave the
same result (within 4 eV, mean 5.860 keV), since Fe-55 pixels sit where the forms agree.
For every form, above the highest measured charge n_max the curve continues linearly up to
n_cap = 3 * n_max (--extrap-factor) and is held constant beyond (on the AMT Am-241 data,
with the linear form, this gave 3.746-3.748 eV/e- for the 59.5 keV line; unlimited
extrapolation overcorrects CCDs with very compact clusters). The ADU of a pixel with n
electrons is A(n) = integral_0^n g; the correction converts ADU -> electrons with A^-1.
Model files written before the --model option (form "a + b*min(n, n_cap)") are linear.

Usage:
  gain_vs_charge.py measure IMAGES... -o OUTDIR [--name NAME] [--ext 1 2 3 4] [--model interpolated|broken|linear]
      -> OUTDIR/NAME_gain_vs_charge.json (per extension: g(n) points, fit, clean rows, nonlinearity)
         OUTDIR/NAME_gain_vs_charge.png
         OUTDIR/NAME_nonlinearity.png   measured charge - peak number vs measured charge,
                                        reference gain g(0) of the curve (A(n)/g(0) - n)
  gain_vs_charge.py apply MODEL.json CLUSTERS.root... -o OUTDIR
      -> OUTDIR/<same name>.root with pixels_E and Energy recomputed with g(n)
(`measure` is the default: gain_vs_charge.py IMAGES... -o OUTDIR also works.)

Images of the same CCDs (same module / ACM) can be given together: each image's comb is
measured on its own and the comb powers are added (the combs of different images need not
line up; for one image this is the plain periodogram).
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


MODELS = ("interpolated", "broken", "linear")
_TABLES = {}


def model_form(model):
    f = model.get("form", "linear")
    return "linear" if f.startswith("a + b") else f


def g_of(n, model):
    """Local gain [ADU/e-] at n electrons for any model form (held constant above n_cap)."""
    f = model_form(model)
    n = np.minimum(np.asarray(n, float), model["n_cap"])
    if f == "linear":
        return model["a"] + model["b"] * n
    if f == "broken":
        nk = model["n_knot"]
        return model["a"] + model["b1"] * np.minimum(n, nk) + model["b2"] * np.maximum(n - nk, 0)
    if f == "interpolated":
        from scipy.interpolate import PchipInterpolator
        scalar = n.ndim == 0
        n = np.atleast_1d(n)
        xs, ys = np.asarray(model["n_pts"], float), np.asarray(model["g_pts"], float)
        out = np.interp(n, xs, ys)                          # replaced inside the points below
        inside = (n >= xs[0]) & (n <= xs[-1])
        out[inside] = PchipInterpolator(xs, ys)(n[inside])
        out[n < xs[0]] = ys[0]
        above = n > xs[-1]
        out[above] = ys[-1] + model["slope_end"] * (n[above] - xs[-1])
        return out[0] if scalar else out
    raise ValueError(f"unknown g(n) model form {f!r}")


def _table(model):
    """A(n) on a 1 e- grid up to n_cap (cached per model)."""
    key = json.dumps(model, sort_keys=True)
    if key not in _TABLES:
        n = np.arange(0.0, float(model["n_cap"]) + 1.0)
        g = g_of(n, model)
        A = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]))])
        _TABLES[key] = (n, A, float(g[-1]))
    return _TABLES[key]


def electrons_of_adu(adu, model):
    """Electrons of pixels with the given ADU for any model form (negative ADU: / g(0))."""
    if model_form(model) == "linear":
        return n_of_adu(adu, model)                         # closed form, as before
    adu = np.asarray(adu, float)
    n, A, g_cap = _table(model)
    out = adu / float(g_of(0.0, model))
    pos = adu > 0
    inside = pos & (adu <= A[-1])
    out[inside] = np.interp(adu[inside], A, n)
    high = pos & (adu > A[-1])
    out[high] = n[-1] + (adu[high] - A[-1]) / g_cap
    return out


def fit_model(good, form, extrap_factor):
    """g(n) model of the given form from the significant points; also the straight-line fit
    (used by the reliability check for every form)."""
    n = np.array([p["n"] for p in good]); g = np.array([p["g"] for p in good])
    e = np.array([p["err"] for p in good])
    (b, a), cov = np.polyfit(n, g, 1, cov=True)
    nmax = float(max(pp["n_hi"] for pp in good))
    ncap = extrap_factor * nmax
    lin = dict(form="linear", a=float(a), b=float(b), n_cap=float(ncap),
               a_err=float(np.sqrt(cov[1, 1])), b_err=float(np.sqrt(cov[0, 0])))
    if form == "linear":
        model = lin
    elif form == "broken":
        from scipy import optimize
        fun = lambda p, x: p[0] + p[1] * np.minimum(x, p[3]) + p[2] * np.maximum(x - p[3], 0)
        res = lambda p: (fun(p, n) - g) / e
        hi_k = n.max() * 0.8
        best = None
        for nk in [k for k in (80, 150, 250, 400, 600) if n.min() < k < hi_k] or [0.5 * (n.min() + hi_k)]:
            r = optimize.least_squares(res, [a, b, b, nk], bounds=([0.1, -1, -1, n.min()], [10, 1, 1, hi_k]))
            if best is None or r.cost < best.cost:
                best = r
        a0, b1, b2, nk = (float(v) for v in best.x)
        model = dict(form="broken", a=a0, b1=b1, b2=b2, n_knot=nk, n_cap=float(ncap))
    elif form == "interpolated":
        order = np.argsort(n)
        xs, ys = n[order], g[order]
        k = min(4, len(xs))
        slope = float(np.polyfit(xs[-k:], ys[-k:], 1)[0])
        model = dict(form="interpolated", a=float(ys[0]), n_pts=[float(v) for v in xs],
                     g_pts=[float(v) for v in ys], slope_end=slope, n_cap=float(ncap))
    else:
        raise ValueError(f"unknown g(n) model {form!r} (choose from {', '.join(MODELS)})")
    resid = g - g_of(n, model)
    return model, lin, nmax, float(resid.std())


# =============================================================================
# measurement
# =============================================================================
def periodogram(v, gs):
    ph = 2j * np.pi * v
    return np.array([np.abs(np.mean(np.exp(ph / g))) for g in gs])


def periodogram_multi(vs, gs):
    """Comb visibility of several images together: the comb power of each image is added
    (sqrt(sum_i |S_i|^2 / sum_i N_i^2), S_i = sum of exp(2 pi i v / g) over image i), so the
    images need not have their comb at the same place. Same as periodogram for one image."""
    if len(vs) == 1:
        return periodogram(vs[0], gs)
    n2 = sum(float(v.size) ** 2 for v in vs)
    return np.array([np.sqrt(sum(np.abs(np.sum(np.exp(2j * np.pi * v / g))) ** 2 for v in vs) / n2) for g in gs])


def best_period(v, g0, lo=0.85, hi=1.10):
    """Local comb spacing near g0; v is one array of pixel values or a list (one per image)."""
    vs = v if isinstance(v, list) else [v]
    gs = np.linspace(g0 * lo, g0 * hi, 351)
    p = periodogram_multi(vs, gs)
    i = int(np.argmax(p))
    fine = np.linspace(gs[max(i - 3, 0)], gs[min(i + 3, len(gs) - 1)], 121)
    pf = periodogram_multi(vs, fine)
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


def measure(images, exts=None, verbose=True, extrap_factor=3.0, model="interpolated"):
    """{ext: result} from a list of images of the same CCDs; model: interpolated, broken or linear."""
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
        # Several images: each image's comb is measured separately and the comb powers are
        # added. Pooling all pixels into one comb would need every image's comb to sit at the
        # same place, A(n) = integral of g, to << 1 e-: a 0.1% gain difference between images
        # shifts it by ~1 e- at 850 e- and washes the pooled comb out at high charge.
        pts = []
        misses = 0
        cap = 300000 // max(len(vals), 1)
        for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
            vs = [v[(v > lo * g0) & (v < hi * g0)] for v in vals]
            if sum(v.size for v in vs) < 400:
                break
            vs = [np.random.default_rng(0).choice(v, cap, replace=False) if v.size > cap else v for v in vs]
            vs = [v for v in vs if v.size >= (50 if len(vals) > 1 else 1)]
            ntot = sum(v.size for v in vs)
            g, R = best_period(vs, g0)
            noise = np.sqrt(sum(float(v.size) for v in vs) / sum(float(v.size) ** 2 for v in vs))
            sig = R > 4 * noise                     # = 4 / sqrt(N) for one image
            err = g / (2 * np.pi * ((hi - lo) / np.sqrt(12)) * R * np.sqrt(ntot)) if R > 0 else float("nan")
            pts.append(dict(n=float(np.sqrt(lo * hi)), n_lo=int(lo), n_hi=int(hi), g=float(g), err=float(err),
                            visibility=float(R), npix=int(ntot), significant=bool(sig)))
            misses = 0 if sig else misses + 1
            if misses >= 2:
                break
        good = [p for p in pts if p["significant"]]
        res = dict(ok=False, g_1e=g0, clean_rows=nrows_clean, rows=nrows, points=pts)
        if len(good) >= 4:
            mdl, lin, nmax, rms = fit_model(good, model, extrap_factor)
            b = lin["b"]
            res.update(ok=True, model=mdl, linear=lin, rms=rms, n_max_measured=nmax,
                       drop_at_1000=float(1 - g_of(1000.0, mdl) / g_of(0.0, mdl)),
                       nonlinearity=nonlinearity(good, mdl))
            # not trustworthy: the comb barely extends to high charge, or the gain seems to rise
            # (slope of the straight-line fit, for every model form)
            if nmax < 400 or b > 0:
                res.update(ok=False, reason=f"unreliable curve (measured to {nmax:.0f} e-, slope "
                                            f"{b * 1e5:+.1f}e-5/e-): no correction")
        else:
            res["reason"] = f"comb visible in only {len(good)} charge bins (need 4)"
        results[name] = res
        if verbose:
            if res["ok"]:
                m = res["model"]
                print(f"  {name}: g(n) {model_form(m)}, g(0) = {m['a']:.4f} ADU/e-, measured up to "
                      f"{res['n_max_measured']:.0f} e-, {res['drop_at_1000'] * 100:.1f}% lower at 1000 e-  "
                      f"({nrows_clean}/{nrows} clean rows; 1e- gain {g0:.4f})")
            else:
                print(f"  {name}: no g(n) fit ({res.get('reason')})")
    return results


def _adu(n, model):
    """A(n) for any model form (closed form for linear, table otherwise)."""
    if model_form(model) == "linear":
        return float(adu_of_n(n, model["a"], model["b"], model["n_cap"]))
    nt, A, gcap = _table(model)
    return float(np.interp(n, nt, A)) if n <= nt[-1] else float(A[-1] + (n - nt[-1]) * gcap)


def nonlinearity(points, model):
    """Nonlinearity curve with the curve's low-charge gain g(0) as reference:
    n_meas = A(n) / g(0), Delta(n) = n_meas - n (measured charge minus true electron number).
    Points: A(n) as the trapezoid integral of the measured local gains (g held at the first point
    below it), with errors of the bins added in quadrature; and a quadratic c2 x^2 + c1 x through
    the origin fitted to Delta vs n_meas (x = measured charge)."""
    n = np.array([p["n"] for p in points]); g = np.array([p["g"] for p in points])
    e = np.array([p["err"] for p in points]); w = np.array([p["n_hi"] - p["n_lo"] for p in points], float)
    g0 = float(g_of(0.0, model))
    A = np.empty_like(n); A[0] = g[0] * n[0]
    A[1:] = A[0] + np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(n))
    x, d, ed = A / g0, A / g0 - n, np.sqrt(np.cumsum((e * w) ** 2)) / g0
    c2, c1 = np.linalg.lstsq(np.c_[x ** 2, x], d, rcond=None)[0]
    return dict(g_ref=g0, n_meas=[float(v) for v in x], delta=[float(v) for v in d],
                delta_err=[float(v) for v in ed], c2=float(c2), c1=float(c1),
                delta_at_1000=float(_adu(1000.0, model) / g0 - 1000.0))


def plot_nonlinearity(results, path, title):
    """One panel per extension: measured charge - peak number vs measured charge, g_ref = g(0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(results)
    fig, axs = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.9), squeeze=False)
    for ax, name in zip(axs[0], names):
        r = results[name]
        nl = r.get("nonlinearity")
        if not r.get("ok") or not nl:
            ax.text(0.5, 0.5, "no reliable curve", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
            continue
        m = r["model"]
        x, d, ed = np.array(nl["n_meas"]), np.array(nl["delta"]), np.array(nl["delta_err"])
        ax.errorbar(x, d, ed, fmt="o", ms=4, color="#1f5fbf", label="from the measured local gains")
        n, A, _ = _table(m) if model_form(m) != "linear" else (None, None, None)
        grid = np.linspace(0, r["n_max_measured"], 300)
        Ag = np.interp(grid, n, A) if n is not None else adu_of_n(grid, m["a"], m["b"], m["n_cap"])
        ax.plot(Ag / nl["g_ref"], Ag / nl["g_ref"] - grid, color="0.3", lw=1.2, label=f"{model_form(m)} model")
        xx = np.linspace(0, x.max() * 1.05, 200)
        ax.plot(xx, nl["c2"] * xx ** 2 + nl["c1"] * xx, color="#d1495b", lw=1.2,
                label=f"${nl['c2']:.3e}\\,x^2 {nl['c1']:+.4f}\\,x$".replace("e-0", "e-"))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{name}: $g_{{ref}}$ = g(0) = {nl['g_ref']:.4f}", fontsize=10)
        ax.set_xlabel("measured pixel charge $A(n)/g(0)$ [e$^-$]")
        ax.set_ylabel("measured charge $-$ peak number [e$^-$]")
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"Nonlinearity curves: {title}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(path, dpi=90)
    plt.close(fig)


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
                a_.plot(grid, g_of(grid, m) / s, "-", color=col, alpha=0.6)
                a_.axvline(r["n_max_measured"], color=col, ls=":", lw=0.8)
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlabel("electrons in the pixel"); a_.grid(alpha=0.3)
    ax[0].set_ylabel("local gain [ADU/e-]"); ax[1].set_ylabel("gain / low-charge gain")
    forms = sorted({model_form(r["model"]) for r in results.values() if r.get("ok")})
    ax[0].set_title(f"{title} ({', '.join(forms) or 'no model'})"); ax[1].set_title("relative (dotted: highest measured charge)")
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
                    vn = electrons_of_adu(adu, mod["model"]) * E_PER_E * 1e-3
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
                   help="continue g(n) linearly up to this x the highest measured charge, constant above (default 3)")
    m.add_argument("--model", choices=MODELS, default="interpolated",
                   help="form of g(n): interpolated (default; monotone interpolation through the measured "
                        "points), broken (two straight lines, fitted knot) or linear (straight line)")
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
        res = measure(args.images, args.ext, extrap_factor=args.extrap_factor, model=args.model)
        jpath = os.path.join(args.outdir, f"{name}_gain_vs_charge.json")
        with open(jpath, "w") as fh:
            json.dump({"images": [os.path.abspath(i) for i in args.images], "e_per_electron_eV": E_PER_E,
                       "extensions": res}, fh, indent=1)
        plot(res, os.path.join(args.outdir, f"{name}_gain_vs_charge.png"), name)
        plot_nonlinearity(res, os.path.join(args.outdir, f"{name}_nonlinearity.png"), name)
        print(f"-> {jpath}")
    else:
        apply(args.model, args.rootfiles, args.outdir, anchor=args.anchor)


if __name__ == "__main__":
    main()
