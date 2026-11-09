#!/usr/bin/env python3
"""Gain as a function of the charge in a pixel, g(n), from skipper-CCD images.

Place in the pipeline (bin/run_WADERs): raw image -> autoconfig_panaSKImg.py /
gain_dc_split.py (one panaSKImg JSON per CCD, gain from the 0e-/1e-... peaks) -> panaSKImg
-> cluster ROOT files -> [this script: measure g(n) on the images, then rewrite the cluster
energies] -> MakeComposite. The step is optional; run_WADERs calls measure(), plot(),
plot_nonlinearity() and apply() directly.

Why: the usual gain is the spacing of the 0e- and 1e- peaks. The amplifier gain falls
slowly as the charge in the pixel grows (~3% lower at ~1000 e- in the AMT images), so
cluster energies converted with the 1e- gain come out too low at keV energies. (The
output stage is not perfectly linear: the more charge sits on the sense node, the fewer
ADU each further electron adds. At low charge this is invisible; summed over the
hundreds to thousands of electrons of an X-ray pixel it is a percent-level energy shift.)

How: pixel values in a skipper image form a comb, one peak per electron count. Wherever
pixels hold many electrons (X-ray clusters, tracks) the comb continues; its LOCAL spacing
at n electrons is the local gain g(n) = dADU/dn. It is measured with a periodogram of
the pixel values in windows of charge (Rayleigh power |<exp(2 pi i x / g)>| maximised over
trial g). Only "clean" rows are used: where the readout has just seen a lot of charge the
comb is blurred (checked per block of rows with the 1-30 e- comb).

Model of g(n) (--model):
  smooth (default since 2026-11-02): below 200 e- as interpolated; above it windows of the same width overlapping
      (4x the points) and a weighted cubic in log n through them; above the last point a straight line
      through the points of the last 4 window widths. Stored as an interpolated curve (fit "smooth").
      On 141 Fe-55 CCD-images (DM11-26) K-alpha equal to interpolated within 1 eV on average (max 5.5 eV);
      on the Am-241 59.5 keV line (AMtest EXT4, per gain period) -0.32 % instead of -0.57 %; reaches
      higher charge.
  interpolated: monotone cubic (PCHIP) interpolation through the measured points;
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
  gain_vs_charge.py measure IMAGES... -o OUTDIR [--name NAME] [--ext 1 2 3 4] [--model smooth|interpolated|broken|linear]
      -> OUTDIR/NAME_gain_vs_charge.json (per extension: g(n) points, fit, clean rows, nonlinearity)
         OUTDIR/NAME_gain_vs_charge.png
         OUTDIR/NAME_nonlinearity.png   measured charge - peak number vs measured charge,
                                        reference gain g(0) of the curve (A(n)/g(0) - n)
         OUTDIR/NAME_adu_of_n.png       A(n) = integral of g (the curve the correction inverts) and A(n) - g(0) n
  gain_vs_charge.py apply MODEL.json CLUSTERS.root... -o OUTDIR
      -> OUTDIR/<same name>.root with pixels_E and Energy recomputed with g(n)
(`measure` is the default: gain_vs_charge.py IMAGES... -o OUTDIR also works.)

Images of the same CCDs (same module / ACM) can be given together (--combine):
  incoherent (default): the comb powers of the images are added (comb search started at the mean
      low-charge gain of the images; independent of the image order).
  coherent: each image is measured on its own and converted to electrons with its own
      curve; the remaining tooth offset of each image (a few tenths of an electron) is removed, and
      all pixels are pooled into one comb. This reaches higher charge than one image (the noise
      level falls with the total number of pixels) and does not depend on the order of the images.
      Not the default: on the Am-241 lines (AMtest EXT4, 2026-10-26) it split the two gain periods
      by 0.6 %; on Fe-55 it was as good or slightly better.
Needs numpy, scipy, astropy (measure) and PyROOT (apply); uses autoconfig_panaSKImg.py
from the same folder for the geometry and the low-charge gain.

Main entry points: measure(images, exts) -> {ext: result}; plot() and plot_nonlinearity()
draw it; apply(model_json, rootfiles, outdir) corrects cluster files; E_PER_E is the
eV per electron of the corrected energies, epsilon(T) from pair_energy.py (3.7486 eV at the default 130 K);
panaSKImg's own energies use PANASKIMG_EV = 3.74 eV.

How to read this file:
  model        g(n) and its integral A(n) for the three forms, ADU -> electrons (A^-1),
               fit_model() (measured points -> model dict)
  measurement  periodogram / best_period (comb spacing), ext_pixels (pedestal-subtracted,
               comb-aligned pixels), clean_rows, measure() and its steps (_measure_*),
               nonlinearity() and the two plots
  correction   apply(): rewrite pixels_E and Energy of panaSKImg cluster files
  CLI          main()
A model is a plain dict (stored as-is in the JSON): "form", "a" = g(0) [ADU/e-], "n_cap"
[e-], plus "b" (linear), "b1", "b2", "n_knot" (broken) or "n_pts", "g_pts", "slope_end"
(interpolated).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pair_energy import PANASKIMG_EV, pair_energy_eV, temperature_K
E_PER_E = pair_energy_eV()   # eV per electron for the corrected energies: epsilon(T), 3.7486 eV at the default 130 K
                             # (pair_energy.py); panaSKImg's own energies use PANASKIMG_EV = 3.74 eV
BLOCK = 25              # rows per block for the clean-row test
# Edges [e-] of the charge windows in which the local comb spacing is measured: roughly
# geometric, so every window holds enough pixels while the curve is sampled densely at low
# charge. Each window gives one point g(n) at n = sqrt(lo * hi).
N_EDGES = np.r_[20, 35, 55, 80, 110, 150, 200, 270, 360, 480, 640, 850, 1130, 1500, 2000, 2700, 3600, 5000, 7000]
# --model smooth: windows of the same width, but overlapping (a new window every 1/OVERLAP_STEP of a window), and a
# weighted fit of g(n) as a polynomial of degree SMOOTH_DEG in log n through all their points (fit_model) instead of a
# curve through every point. On DM09 (simulation with a known A(n), and the real images cross-validated by rows) this
# reduced the error of the curve (DM09 simulation 0.49 -> 0.47 e- rms; real images, held-out rows: tooth offset
# 0.21 -> 0.14 e-), and the comb is measured to higher charge (AMTest EXT3 1130 -> 1491 e-). Overlap alone helped
# less, narrower windows were worse. Only above SMOOTH_FROM, see there.
OVERLAP_STEP = 4
SMOOTH_DEG = 3
SMOOTH_FROM = 200.0     # e-: 'smooth' uses the standard-window points (and the single-electron anchor) below this
# charge and the smooth fit of the overlapping windows above it. Below ~40 e- the overlapping points scatter (by up to
# +0.9 % of g on AMTest EXT4), so smoothing them would move the low-charge gain; above, it helps (Am-241 59.54 keV per
# gain period: EXT3 -0.05 -> -0.01 %, EXT4 -0.57 -> -0.32 %; DM09 Fe-55 K-alpha unchanged).


def window_pairs(model="interpolated"):
    """(lo, hi) charge windows [e-] in which the comb spacing is measured: N_EDGES for the interpolated, broken and
    linear models; for 'smooth' windows of the mean N_EDGES width starting every 1/OVERLAP_STEP of a window."""
    if model != "smooth":
        return list(zip(N_EDGES[:-1], N_EDGES[1:]))
    r = float(np.exp(np.mean(np.diff(np.log(N_EDGES)))))
    pairs, j = [], 0
    while N_EDGES[0] * r ** (j / OVERLAP_STEP + 1) <= N_EDGES[-1]:
        lo = N_EDGES[0] * r ** (j / OVERLAP_STEP)
        pairs.append((float(lo), float(lo * r)))
        j += 1
    return pairs


# =============================================================================
# model
# =============================================================================
# The linear form has closed-form expressions (adu_of_n, n_of_adu). The other
# forms go through g_of() and a numerical table of A(n) (_table, electrons_of_adu, _adu).
def adu_of_n(n, a, b, n_cap):
    """A(n) = integral_0^n g(m) dm (closed form) for the linear model [ADU].

    Below n_cap: a*n + b*n^2/2; above it the gain is constant, g(n_cap) = a + b*n_cap, so A
    grows linearly. n may be a scalar or an array.
    """
    n = np.asarray(n, float)
    m = np.minimum(n, n_cap)
    return a * m + 0.5 * b * m * m + (a + b * n_cap) * np.maximum(n - n_cap, 0)


def n_of_adu(adu, model):
    """Electrons of pixels with the given ADU (inverse of A; negative ADU: /a).

    Linear model only. Below A(n_cap), A = a*n + b*n^2/2 is solved for n (the root
    n = (-a + sqrt(a^2 + 2*b*A)) / b, which goes to A/a as b -> 0); above it n grows with the
    constant gain a + b*n_cap. Pixels with ADU <= 0 (noise around the zero peak) are divided
    by a. Returns a float array of the same shape as adu [e-].
    """
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


MODELS = ("interpolated", "broken", "linear", "smooth")
_TABLES = {}            # cache of _table(): A(n) on a 1 e- grid, keyed by the model dict


def model_form(model):
    """Form of a model dict: "interpolated", "broken" or "linear".

    Model files written before the --model option have no "form" key or the old form string
    "a + b*min(n, n_cap)"; both mean linear.
    """
    f = model.get("form", "linear")
    return "linear" if f.startswith("a + b") else f


def g_of(n, model):
    """Local gain [ADU/e-] at n electrons for any model form (held constant above n_cap).

    linear: a + b*n. broken: a + b1*min(n, nk) + b2*max(n - nk, 0) (two straight lines that
    meet at the knot nk). interpolated: PCHIP (monotone cubic, no overshoot between points)
    through the measured points, constant g_pts[0] below the first point and a straight line
    with slope_end above the last one. n may be a scalar (returns a scalar) or an array.
    """
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
    """A(n) on a 1 e- grid up to n_cap (cached per model).

    A is the trapezoid integral of g_of() on n = 0, 1, ..., n_cap. Returns (n grid [e-],
    A [ADU], g at the last grid point [ADU/e-], the constant gain used beyond the grid).
    """
    key = json.dumps(model, sort_keys=True)
    if key not in _TABLES:
        n = np.arange(0.0, float(model["n_cap"]) + 1.0)
        g = g_of(n, model)
        A = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]))])
        _TABLES[key] = (n, A, float(g[-1]))
    return _TABLES[key]


def load_model(source, ext):
    """The g(n) model of one extension, or None if there is no reliable curve for it.
    source: a *_gain_vs_charge.json path or its "extensions" dict; ext: 1 / "1" / "EXT1"."""
    if isinstance(source, str):
        with open(source) as fh:
            source = json.load(fh)["extensions"]
    key = ext if str(ext).startswith("EXT") else f"EXT{ext}"
    res = source.get(key, {})
    if res.get("ok") not in (True, "True"):
        return None
    model = res.get("model")
    if isinstance(model, str):                    # older files stored the model as its repr
        import ast
        model = ast.literal_eval(model)
    return model


def electrons_with_anchor(adu, model, gain_fit, anchor="comb"):
    """Electrons of pixels with the given ADU (the image's ADU = keV / PANASKIMG_EV x gain_fit): anchor 'comb' uses the
    comb's g(n) throughout; 'fit' keeps the image's fitted gain at low charge (same curve shape, g(0) = gain_fit)."""
    adu = np.asarray(adu, float)
    if anchor == "fit":
        adu = adu * model["a"] / gain_fit
    return electrons_of_adu(adu, model)


def electrons_of_adu(adu, model):
    """Electrons of pixels with the given ADU for any model form (negative ADU: / g(0)).

    This is the energy correction: a pixel's ADU is A(n) = integral_0^n g, so its electron
    number is A^-1(ADU). Linear models use the closed form (n_of_adu); the other forms invert
    the tabulated A(n) by linear interpolation, and continue with the constant gain g(n_cap)
    above the table. Returns a float array of the same shape as adu [e-].
    """
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


def fit_model(good, form, extrap_factor, low=None):
    """g(n) model of the given form from the significant points; also the straight-line fit
    (used by the reliability check for every form).

    good: list of point dicts from measure() (keys n, g, err, n_hi, ...), all significant.
    form: "interpolated", "broken" or "linear". extrap_factor: n_cap = extrap_factor * n_max,
    n_max = upper edge of the highest measured charge window.
    broken: least-squares fit (weighted by the point errors) of the two-line form, started
    from several knot positions (80 ... 600 e-, those inside the measured range); the best
    one is kept. interpolated: the points themselves, sorted by n, plus the end slope from a
    straight line through the last 4 points.
    Returns (model dict, linear-fit dict with a_err/b_err, n_max [e-], rms of the points
    around the model [ADU/e-]).
    """
    n = np.array([p["n"] for p in good]); g = np.array([p["g"] for p in good])
    e = np.array([p["err"] for p in good])
    (b, a), cov = np.polyfit(n, g, 1, cov=True)
    nmax = float(max(pp["n_hi"] for pp in good))
    ncap = extrap_factor * nmax
    lin = dict(form="linear", a=float(a), b=float(b), n_cap=float(ncap),
               a_err=float(np.sqrt(cov[1, 1])), b_err=float(np.sqrt(cov[0, 0])))
    if form == "smooth":
        # the overlapping windows sample low charge densely, so the unweighted slope over-weights it (on DM09 EXT2,
        # whose gain rises up to ~230 e- and then falls, it turned positive); the reliability check of 'smooth' uses
        # the slope weighted by the point errors
        lin["b_weighted"] = float(np.polyfit(n, g, 1, w=1 / e)[0])
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
    elif form == "smooth":
        # above SMOOTH_FROM: weighted polynomial in log n through the overlapping-window points (fitted from one window
        # below SMOOTH_FROM); below it: the standard-window points `low` (as 'interpolated'). Stored as an
        # interpolated curve, so that every user of the model (g_of, A(n), apply, GainNonlinearityProcess) is unchanged.
        r = float(np.exp(np.mean(np.diff(np.log(N_EDGES)))))
        start = SMOOTH_FROM if low else float(n.min())
        sel = n >= start / r
        if sel.sum() < 6:                        # too few points above: smooth over the whole range
            start, sel = float(n.min()), np.ones(n.size, bool)
        ns, gs_, es = n[sel], g[sel], e[sel]
        deg = min(SMOOTH_DEG, len(ns) - 2)
        c = np.polyfit(np.log(ns), gs_, deg, w=1 / es)
        grid = np.geomspace(max(start, ns.min()), ns.max(), 30)
        below = sorted([p for p in (low or []) if p["n"] < grid[0]], key=lambda p: p["n"])
        xs = np.r_[[p["n"] for p in below], grid]
        ys = np.r_[[p["g"] for p in below], np.polyval(c, np.log(grid))]
        # continuation above the measured range (most of the energy of bright clusters, e.g. 59.5 keV, comes from
        # there): a straight line through the points of the last 4 window widths, as for 'interpolated'. The
        # polynomial's own end slope was less stable (Am-241 EXT4 59.5 keV -0.65 % instead of -0.16 %).
        last = n >= n.max() / r ** 3
        slope = float(np.polyfit(n[last], g[last], 1)[0]) if last.sum() >= 2 else \
            float(np.polyval(np.polyder(c), np.log(grid[-1])) / grid[-1])
        model = dict(form="interpolated", fit="smooth", a=float(ys[0]), n_pts=[float(v) for v in xs],
                     g_pts=[float(v) for v in ys], slope_end=slope, n_cap=float(ncap),
                     smooth=dict(deg=int(deg), coeffs_log_n=[float(v) for v in c], n_points=int(sel.sum()),
                                 overlap_step=OVERLAP_STEP, from_n=float(grid[0]), standard_points_below=len(below)))
    else:
        raise ValueError(f"unknown g(n) model {form!r} (choose from {', '.join(MODELS)})")
    resid = g - g_of(n, model)
    return model, lin, nmax, float(resid.std())


# =============================================================================
# measurement
# =============================================================================
def periodogram(v, gs):
    """Rayleigh periodogram of the pixel values v [ADU] at the trial periods gs [ADU/e-].

    For a trial period g every pixel value becomes a phase 2 pi v / g. If the values sit on
    a comb with spacing g, all phases agree and |mean of exp(i phase)| is large (1 for sharp
    peaks, smaller for peaks broadened by noise); for values spread at random it is ~1/sqrt(N).
    The maximum over g is the comb spacing, i.e. the local gain. Returns an array like gs.
    """
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
    """Local comb spacing near g0; v is one array of pixel values or a list (one per image).

    Coarse scan of 351 trial periods from lo*g0 to hi*g0, then a fine scan of 121 periods
    around the best one. Returns (spacing [ADU/e-], comb visibility R at that spacing).
    """
    vs = v if isinstance(v, list) else [v]
    gs = np.linspace(g0 * lo, g0 * hi, 351)
    p = periodogram_multi(vs, gs)
    i = int(np.argmax(p))
    fine = np.linspace(gs[max(i - 3, 0)], gs[min(i + 3, len(gs) - 1)], 121)
    pf = periodogram_multi(vs, fine)
    j = int(np.argmax(pf))
    return fine[j], pf[j]


def ext_pixels(data, cfg, g0):
    """Pedestal-subtracted, comb-aligned active pixels (rows x cols) of one extension.

    data: the raw 2D image of the extension [ADU]; cfg: geometry from autoconfig_panaSKImg
    (id_row_start, n_cols_prescan, ccd_cols); g0: low-charge gain [ADU/e-].
    The pedestal of each row is the median of its overscan (or, without overscan, of the
    active pixels, which is the zero-electron peak). A remaining offset of a fraction of an
    electron is removed by aligning the phase of the low-charge comb (|x| < 3 g0) of each row
    to zero, so that n electrons sit at n * g0. Returns a float array [ADU].
    """
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
    """Boolean mask of rows in blocks whose 1-30 e- comb is clearly visible.

    After the readout has seen a lot of charge (a large cluster or track) the following
    pixels are disturbed (baseline shifts, charge left behind) and the comb is blurred; such
    rows would wash out the high-charge comb too. The rows are cut into blocks of BLOCK rows;
    a block is clean if the visibility of its 0.5-30 e- comb at the period g0 is at least
    frac times that of the best block (blocks with <= 300 such pixels count as not clean).
    Returns (mask per row, visibility per block).
    """
    nb = int(math.ceil(x.shape[0] / BLOCK))
    vis = np.zeros(nb)
    for b in range(nb):
        v = x[b * BLOCK:(b + 1) * BLOCK]
        v = v[(v > 0.5 * g0) & (v < 30 * g0)]
        vis[b] = np.abs(np.mean(np.exp(2j * np.pi * v / g0))) if v.size > 300 else 0.0
    good = vis >= frac * vis.max() if vis.max() > 0 else vis > 1
    mask = np.repeat(good, BLOCK)[:x.shape[0]]
    return mask, vis


def _measure_setup(images, exts, autoconfig_args=()):
    """Geometry, extensions and gain priors from autoconfig_panaSKImg, for measure().

    Uses the reference JSON (autoconfig's built-in one, or --reference from autoconfig_args) and the
    first image (its header and extensions); autoconfig_args are the same settings as for the panaSKImg
    JSONs (run_WADERs --autoconfig-args, e.g. --ccd-cols, --gain, --gain-min/--gain-max, --lam-max), so
    the comb is measured with the geometry and gain range the run used. Returns (autoconfig module,
    astropy fits module, opened first image, extensions, geometry cfg, autoconfig arguments,
    {EXTn: gain prior}).
    """
    from astropy.io import fits
    import autoconfig_panaSKImg as A

    aargs = A.parse_args([images[0], "-o", "unused.json", *autoconfig_args])
    if aargs.reference:
        with open(aargs.reference) as fh:
            ref = json.load(fh)
    else:
        ref = json.loads(A.REFERENCE_JSON)
    hdul0 = fits.open(images[0])
    exts = exts or [e for e in ref["input"]["image"].get("extensions", [1, 2, 3, 4]) if e < len(hdul0)]
    _, cfg, _, _ = A.geometry(hdul0[0].header, hdul0, ref["input"]["image"], aargs)
    gp = A.gain_priors(ref, [f"EXT{e}" for e in exts], aargs)
    return A, fits, hdul0, exts, cfg, aargs, gp


MEMORY_WINDOW = (300, 1100)      # e-: charge range where the readout-memory coefficients are fitted


def remove_memory(x, kL, kR):
    """Remove the horizontal readout memory from pixel values x (rows x cols, pedestal-subtracted, ADU).

    The video signal of a pixel keeps a small fraction of the pixel read just before it (and, much less, the one
    after it): x_j = s_j + kL s_(j-1) + kR s_(j+1). Inside a cluster bright pixels follow bright pixels, so this
    adds about kL x 1000 e- = 0.3 e- to a 1000 e- pixel and blurs the electron comb at high charge; measured on
    the AMtest and DM24-26 CCDs kL = 1.6-3.8e-4, kR < 1e-4. First-order inverse: s_j = x_j - kL x_(j-1) - kR x_(j+1)
    (a missing neighbour at the row ends counts as 0). Returns a new array.
    """
    y = np.array(x, float)
    if kL:
        y[:, 1:] -= kL * x[:, :-1]
    if kR:
        y[:, :-1] -= kR * x[:, 1:]
    return y


def _memory_power(data, kL, kR, lo=MEMORY_WINDOW[0], hi=MEMORY_WINDOW[1]):
    """Comb power of pixels with lo-hi electrons after removing (kL, kR): sum over images of |sum exp(2 pi i u)|^2,
    divided by the number of pixels (about 1 for no comb). data: per image (u, u_left, u_right) in electrons,
    converted with that image's own g(n); each image keeps its own phase (powers added)."""
    num, n = 0.0, 0
    for u, L, R in data:
        v = u - kL * L - kR * R
        m = (v > lo) & (v < hi)
        if m.sum() >= 20:
            num += abs(np.sum(np.exp(2j * np.pi * v[m]))) ** 2
            n += int(m.sum())
    return num / max(n, 1)


def fit_memory_coefficients(data):
    """(kL, kR, comb power without, with) maximising the comb power at MEMORY_WINDOW (see _memory_power).

    Coarse grid of -6e-4..6e-4 in both coefficients, then Nelder-Mead. data: per image (u, u_left, u_right).
    """
    from scipy import optimize
    grid = np.linspace(-6e-4, 6e-4, 13)
    V = np.array([[_memory_power(data, a, b) for b in grid] for a in grid])
    i, j = np.unravel_index(int(np.argmax(V)), V.shape)
    r = optimize.minimize(lambda q: -_memory_power(data, q[0], q[1]), [grid[i], grid[j]], method="Nelder-Mead",
                          options=dict(xatol=2e-6, fatol=1e-3))
    return float(r.x[0]), float(r.x[1]), _memory_power(data, 0.0, 0.0), _memory_power(data, *r.x)


def _fit_memory(images, exts, extrap_factor, model, autoconfig_args):
    """{EXTn: dict(kL, kR, power_off, power_on, images)} for each extension with >= 1 image whose own g(n) is ok.

    Each image is measured alone without the correction; its clean-row pixels are converted to electrons with its
    own curve (so the comb is straight at every charge) and paired with their left and right neighbours.
    """
    A, fits, hdul0, exts, cfg, aargs, gp = _measure_setup(images, exts, autoconfig_args)
    per = [measure([f], exts, verbose=False, extrap_factor=extrap_factor, model=model, autoconfig_args=autoconfig_args)
           for f in images]
    out = {}
    for e in exts:
        name = f"EXT{e}"
        data = []
        for f, r in zip(images, per):
            rr = r.get(name, {})
            if not rr.get("ok"):
                continue
            x = ext_pixels(fits.open(f)[e].data, cfg, rr["g_1e"])
            mask, _ = clean_rows(x, rr["g_1e"])
            u = electrons_of_adu(x, rr["model"])[mask]
            cur, L, R = u[:, 1:-1].ravel(), u[:, :-2].ravel(), u[:, 2:].ravel()
            keep = cur > 20
            data.append((cur[keep], L[keep], R[keep]))
        if data:
            kL, kR, p0, p1 = fit_memory_coefficients(data)
            out[name] = dict(kL=kL, kR=kR, power_off=p0, power_on=p1, images=len(data))
    # a CCD without a usable comb of its own borrows the median of the others: the memory is a property of the
    # readout and agreed to ~20 % across CCDs (AMtest, DM24-26), while the comb of such a CCD may still be usable
    # once the images are combined
    if out:
        mL = float(np.median([d["kL"] for d in out.values()])); mR = float(np.median([d["kR"] for d in out.values()]))
        for e in exts:
            out.setdefault(f"EXT{e}", dict(kL=mL, kR=mR, borrowed=f"median of {', '.join(sorted(out))}"))
    return out


def _measure_clean_values(fits, images, e, cfg, g0, memory=None, low=None):
    """Pixel values [ADU] of the clean rows of extension e, one array per image.

    Only pixels above 0.9 * N_EDGES[0] electrons are kept (the lowest charge window).
    memory: (kL, kR) readout-memory coefficients to remove first (remove_memory), or None.
    low: (lo, hi) e- (LOW_CHECK) to also return the pixels in that window, one array per image.
    Returns (list of arrays, number of clean rows, number of rows), rows summed over images,
    and with low the list of low-window arrays as a fourth item.
    """
    vals, lows, nrows_clean, nrows = [], [], 0, 0
    for f in images:
        x = ext_pixels(fits.open(f)[e].data, cfg, g0)
        if memory:
            x = remove_memory(x, *memory)
        mask, _ = clean_rows(x, g0)
        nrows += x.shape[0]
        nrows_clean += int(mask.sum())
        v = x[mask]
        vals.append(v[v > N_EDGES[0] * 0.9 * g0])
        if low:
            lows.append(v[(v > low[0] * g0) & (v < low[1] * g0)])
    return (vals, nrows_clean, nrows, lows) if low else (vals, nrows_clean, nrows)


def _measure_comb_points(vals, g0, pairs=None):
    """Local comb spacing g(n) in each charge window of N_EDGES, from low to high charge.

    vals: clean pixel values [ADU], one array per image; g0: low-charge gain [ADU/e-].
    Per window (lo, hi) e-: at most 300000 / (number of images) pixels per image (random
    subset, fixed seed; with several images, images with < 50 pixels in the window are left
    out), then the best period g and its visibility R (best_period). The point is significant if R is above
    4 times the visibility expected from pure noise (4 / sqrt(N) for one image). The error
    of g follows from the phase lever arm: the pixels of a window spread over (hi - lo)/sqrt(12)
    electrons (rms), and a comb of visibility R from N pixels fixes the phase to ~1/(R sqrt(N)).
    The scan stops when a window has fewer than 400 pixels or after two non-significant
    windows in a row. Returns a list of point dicts (n = sqrt(lo * hi), n_lo, n_hi, g, err,
    visibility, npix, significant).
    """
    # Several images (the incoherent combination): each image's comb is measured separately
    # and the comb powers are added. Pooling the raw ADU of all images into one comb would need
    # every image's comb at the same place, A(n) = integral of g, to << 1 e- (a 0.1% gain
    # difference shifts it by ~1 e- at 850 e-); the coherent combination (_measure_coherent)
    # first converts each image with its own curve, which lines the combs up.
    pts = []
    misses = 0
    cap = 300000 // max(len(vals), 1)
    for lo, hi in (pairs or window_pairs()):
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
    return pts


LOW_RISE = 0.005    # low-charge anchor: g_1e this fraction (or more) below the first comb point -> add (1 e-, g_1e)
LOW_CHECK = (3, 18)  # e-: comb window below the first one, measured to confirm that drop before anchoring


def low_charge_anchor(mdl, g0, rise=LOW_RISE, check=None):
    """The interpolated model holds g constant below its first comb point (~26 e-, the windows start at 20 e-). On
    most CCDs the gain is flat there, but on some it changes steeply at low charge (DM09 EXT2: 1.22 ADU/e- at 1 e-,
    1.24 at 7 e-, 1.29 at 26 e-), and then pixels below ~26 e- get the wrong number of electrons (diffuse back-side
    Fe-55 -1.4 %). Returns (model, applied).

    check: the comb point measured in LOW_CHECK (3-18 e-, n = sqrt(3 * 18), _measure_comb_points), or None.
    - check significant and within rise / 2 of the first comb point: the gain is flat below the first window, the
      model is unchanged (g_1e can be biased low: VR-3 VDD-20 dH+2, ACM 125 EXT1, g_1e 1.2955 while the found
      peaks at 1-20 e- are evenly spaced at the first comb point, 1.346).
    - check significant and further away: the measured point (n_check, g_check) is added; (1 e-, g_1e) is added
      below it as well when g_1e continues the same trend (g_1e < g_check). g_1e can also be biased high (DM27 EXT3,
      ACM 109: g_1e 1.270, comb 1.238 at 3-18 e-, 1.2547 at 26 e-), then only the measured point is used.
      Ka on 2025-09-23 Image_8 / PD06 (A_GMX, all / back-side events, eV): DM27 EXT3 -22 -> -12 / -50 -> -20,
      PD06 A EXT4 -13 -> -10 / -36 -> -23, DM07 EXT2 -13 -> -11 / -24 -> -17; DM09 EXT2 (both points) and the
      other changed CCDs within 1-2 eV; widths unchanged.
    - no usable check: as before, (1 e-, g_1e) is added when g_1e is more than `rise` below the first point (DM09
      EXT2 back - front -1.40 -> +0.02 %, DM04-06 ACM109 EXT2 -1.03 -> -0.72 %, ACM110 EXT1 -1.18 -> -0.87 %); adding
      it everywhere made flat-gain CCDs slightly worse."""
    if model_form(mdl) != "interpolated" or mdl["n_pts"][0] <= 1.0 or not np.isfinite(g0):
        return mdl, False
    first = mdl["g_pts"][0]
    if check and check.get("significant") and np.isfinite(check.get("g", np.nan)):
        if abs(check["g"] / first - 1.0) <= rise / 2:
            return mdl, False
        n_pts = [float(check["n"])] + list(mdl["n_pts"])
        g_pts = [float(check["g"])] + list(mdl["g_pts"])
        if g0 < check["g"]:
            n_pts, g_pts = [1.0] + n_pts, [float(g0)] + g_pts
        return dict(mdl, n_pts=n_pts, g_pts=g_pts, a=g_pts[0]), True
    if g0 >= (1.0 - rise) * first:
        return mdl, False
    return dict(mdl, n_pts=[1.0] + list(mdl["n_pts"]), g_pts=[float(g0)] + list(mdl["g_pts"]), a=float(g0)), True


def _measure_fit(pts, g0, nrows_clean, nrows, model, extrap_factor, low_pts=None, check=None):
    """Result dict of one extension: the points and, with >= 4 significant points, the model.

    The curve is marked not ok (no correction) when the comb was measured only below 400 e-
    or the straight-line fit has a rising gain (b > 0): neither is a trustworthy curve.
    """
    good = [p for p in pts if p["significant"]]
    res = dict(ok=False, g_1e=g0, clean_rows=nrows_clean, rows=nrows, points=pts)
    if low_pts is not None:                      # 'smooth': the standard-window points, used below SMOOTH_FROM
        res["points_standard"] = low_pts
    if len(good) >= 4:
        low_good = [p for p in low_pts if p["significant"]] if low_pts is not None else None
        mdl, lin, nmax, rms = fit_model(good, model, extrap_factor, low=low_good)
        n_before = len(mdl.get("n_pts", []))
        mdl, anchored = low_charge_anchor(mdl, g0, check=check)
        if check:
            res["low_charge_check"] = dict(check, anchored=bool(anchored))
        if anchored:                                 # g_first / n_first: the first comb point (above the added points)
            k = len(mdl["n_pts"]) - n_before
            res["low_charge_anchor"] = dict(g_1e=float(g0), g_first=float(mdl["g_pts"][k]), n_first=float(mdl["n_pts"][k]),
                                            added=[[float(n), float(g)] for n, g in zip(mdl["n_pts"][:k], mdl["g_pts"][:k])])
        b = lin.get("b_weighted", lin["b"])
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
    return res


def _measure_print(name, res, g0, nrows_clean, nrows):
    """One-line report of one extension's result (verbose measure())."""
    if res["ok"]:
        m = res["model"]
        print(f"  {name}: g(n) {m.get('fit', model_form(m))}, g(0) = {m['a']:.4f} ADU/e-, measured up to "
              f"{res['n_max_measured']:.0f} e-, {res['drop_at_1000'] * 100:.1f}% lower at 1000 e-  "
              f"({nrows_clean}/{nrows} clean rows; 1e- gain {g0:.4f})")
    else:
        print(f"  {name}: no g(n) fit ({res.get('reason')})")


COMBINE = ("coherent", "incoherent")


def _tooth_offsets(u):
    """Where one image's comb teeth sit, in electrons, relative to the integers, as a function of charge.

    u: the image's clean pixels converted to electrons with its own curve. In a perfect conversion
    every tooth is at an integer; in practice the integral A(n) of the curve is off by a few tenths
    of an electron (it accumulates the small errors of g at low charge), by a different amount in
    each image. In each charge window where this image alone shows a significant comb
    (R > 4/sqrt(N)), the offset is the mean tooth position arg(sum exp(2 pi i u)) / 2 pi; the values
    are unwrapped from window to window (they drift slowly). Returns (n, offset) at the window centres,
    from the first window up to the last significant one before a non-significant window.
    """
    ns, offs = [], []
    for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
        v = u[(u > lo) & (u < hi)]
        if v.size < 50:
            break
        S = np.sum(np.exp(2j * np.pi * v))
        if abs(S) / v.size <= 4 / np.sqrt(v.size):
            break
        ns.append(float(np.sqrt(lo * hi)))
        offs.append(float(np.angle(S)))
    return np.array(ns), np.unwrap(np.array(offs)) / (2 * np.pi)


def _offset_function(ns, offs):
    """Offset d(n) [e-] and its slope d'(n): linear interpolation between the measured offsets, held
    constant below the first one, continued with the slope of the last two above the last one."""
    if ns.size == 0:
        return (lambda n: np.zeros_like(np.asarray(n, float))), (lambda n: np.zeros_like(np.asarray(n, float)))
    s_end = (offs[-1] - offs[-2]) / (ns[-1] - ns[-2]) if ns.size >= 2 else 0.0

    def d(n):
        n = np.asarray(n, float)
        return np.where(n > ns[-1], offs[-1] + s_end * (n - ns[-1]), np.interp(n, ns, offs))

    def dslope(n):
        n = np.asarray(n, float)
        if ns.size < 2:
            return np.zeros_like(n)
        seg = np.clip(np.searchsorted(ns, n) - 1, 0, ns.size - 2)
        sl = np.diff(offs) / np.diff(ns)
        return np.where(n > ns[-1], s_end, np.where(n < ns[0], 0.0, sl[seg]))
    return d, dslope


def _coherent_points(us, models, pairs=None):
    """Local gain in each charge window from several images, pooled COHERENTLY.

    us: per image, the clean pixel values already converted to electrons with that image's own
    curve (u = A_i^-1(x)); models: those curves. In these units every image's comb teeth sit at
    (nearly) integer u, so the pixels of all images can be pooled into one comb: for N pixels
    the noise level of the visibility is 1/sqrt(N) of ALL pixels, where adding comb powers
    (incoherent) only reaches ~1/sqrt(N) of one image.

    Alignment: the teeth of image i actually sit at integer + d_i(n), a slowly drifting offset of
    a few tenths of an electron that differs between images (_tooth_offsets); unless it is removed,
    the pooled teeth partly cancel. Each image is shifted by its own d_i (measured where that image
    alone sees the comb, continued linearly above), u -> u - d_i(u).

    Each window is then scanned for the tooth spacing p of the pooled, aligned pixels (p = 1 where
    the curves are right; p != 1 e.g. above an image's own n_max, where its curve is extrapolated).
    The local gain is g(n) = p * mean_i[(1 + d_i'(n)) g_i(n)]: the spacing in the original units u is
    p (1 + d_i'), and one electron in u is g_i(n) ADU. Windows are in electrons (N_EDGES), at most
    300000 / (number of images) pixels per image (random subset, fixed seed); significance
    R > 4 / sqrt(N); error as in _measure_comb_points; the scan stops at < 400 pixels or after two
    non-significant windows in a row.
    """
    shifts = [_offset_function(*_tooth_offsets(u)) for u in us]
    aligned = [u - d(u) for u, (d, _) in zip(us, shifts)]
    pts, misses = [], 0
    cap = 300000 // max(len(us), 1)
    for lo, hi in (pairs or window_pairs()):
        vs = [u[(u > lo) & (u < hi)] for u in aligned]
        if sum(v.size for v in vs) < 400:
            break
        vs = [np.random.default_rng(0).choice(v, cap, replace=False) if v.size > cap else v for v in vs]
        v = np.concatenate(vs)
        p, R = best_period(v, 1.0, lo=0.95, hi=1.05)
        nc = float(np.sqrt(lo * hi))
        gbar = float(np.mean([(1 + float(ds(nc))) * float(g_of(nc, m)) for m, (_, ds) in zip(models, shifts)]))
        sig = R > 4 / np.sqrt(v.size)
        err = gbar / (2 * np.pi * ((hi - lo) / np.sqrt(12)) * R * np.sqrt(v.size)) if R > 0 else float("nan")
        pts.append(dict(n=nc, n_lo=int(lo), n_hi=int(hi), g=float(p * gbar), err=float(err), visibility=float(R),
                        npix=int(v.size), significant=bool(sig), spacing_vs_curves=float(p)))
        misses = 0 if sig else misses + 1
        if misses >= 2:
            break
    return pts


def _measure_coherent(images, exts, verbose, extrap_factor, model, autoconfig_args=(), memory=None):
    """measure() for several images, combined coherently (see _coherent_points).

    1. Every image is measured on its own (its own g0, row alignment, clean rows and curve), so
       nothing depends on which image comes first.
    2. Per extension, the images with a usable curve of their own (at least 2 needed) are converted
       to electrons with that curve and pooled; the points from the pooled comb are fitted as usual.
       The result reaches as far as the pooled comb is significant, which can be beyond every single
       image's n_max (there the comb checks, and corrects, the images' extrapolated curves).
    3. With fewer than 2 usable images the extension falls back to the incoherent combination.
    Extra result keys: combine, images_used, per_image_n_max; points carry spacing_vs_curves (p).
    memory: {EXTn: (kL, kR)} readout-memory coefficients removed before every measurement, or None.
    """
    images = sorted(images)                     # same result whatever the order of the input
    per = [measure([f], exts, verbose=False, extrap_factor=extrap_factor, model=model, autoconfig_args=autoconfig_args,
                   memory=memory) for f in images]
    A, fits, hdul0, exts, cfg, aargs, gp = _measure_setup(images, exts, autoconfig_args)
    results, fallback = {}, None
    for e in exts:
        name = f"EXT{e}"
        usable = [(f, r[name]) for f, r in zip(images, per) if r.get(name, {}).get("ok")]
        if len(usable) < 2:
            if fallback is None:
                fallback = measure(images, exts, verbose=False, extrap_factor=extrap_factor, model=model,
                                   combine="incoherent", autoconfig_args=autoconfig_args, memory=memory)
            res = dict(fallback[name], combine="incoherent",
                       combine_note=f"only {len(usable)} image(s) with a usable curve of their own")
        else:
            us, nclean, nrows = [], 0, 0
            for f, r in usable:
                vals, nc, nr = _measure_clean_values(fits, [f], e, cfg, r["g_1e"], (memory or {}).get(name))
                us.append(electrons_of_adu(vals[0], r["model"]))
                nclean += nc
                nrows += nr
            g0 = float(np.mean([r["g_1e"] for _, r in usable]))
            models_ = [r["model"] for _, r in usable]
            low = _coherent_points(us, models_) if model == "smooth" else None
            res = _measure_fit(_coherent_points(us, models_, window_pairs(model)), g0, nclean, nrows,
                               model, extrap_factor, low)
            res.update(combine="coherent", images_used=[os.path.basename(f) for f, _ in usable],
                       per_image_n_max=[r["n_max_measured"] for _, r in usable])
        results[name] = res
        if verbose:
            _measure_print(name, res, res.get("g_1e", float("nan")), res.get("clean_rows", 0), res.get("rows", 0))
            if res.get("combine") == "coherent":
                print(f"         coherent combination of {len(res['images_used'])} images "
                      f"(single images measured to {', '.join(f'{v:.0f}' for v in res['per_image_n_max'])} e-)")
    return results


PERIOD_MIN_STEP = 0.01      # gain periods: smallest gain step between two periods (fraction)
PERIOD_MIN_SIG = 5.0        # ... and its significance against the image-to-image scatter within the periods


def gain_periods(rel_gain, err, min_step=PERIOD_MIN_STEP, min_sig=PERIOD_MIN_SIG, min_n=2):
    """Gain periods of one module: (period label per image, outlier flag per image).

    rel_gain: per image, in time order, ln(gain / module median gain), averaged over the CCDs (a change of the
    readout or bias settings shifts all CCDs together); err: its error. Pooling images of two periods (e.g.
    AMtest 106 before / after 2026-05-21, gain 2.5-3.5 % apart) puts two combs on top of each other: g(n) and
    every line come out doubled. Method:
      1. single-image outliers (a poor 0e-/1e- fit) are set aside: an image that differs from both of its
         neighbours by more than max(min_step, 5 sigma) (never the first or last image);
      2. binary segmentation of the rest: the split that minimises the absolute deviations from the median of
         each side, kept if the step between the two medians is > min_step and > min_sig times its error
         (scatter from the MAD of the deviations, at least the fit errors and 0.3 %); then each side again.
    Each side keeps at least min_n images. Outliers get the period of their place in time.
    On simulated series (0.4 % scatter, 10 % outlier fits) a 2.5 % step is found at the right image (+-1) in
    83 % of cases (69 % for a 2-image period), and a series without a step is split in 7 %.
    """
    x, s = np.asarray(rel_gain, float), np.asarray(err, float)
    bad = np.zeros(x.size, bool)
    if x.size >= 4:
        for i in range(1, x.size - 1):
            bad[i] = all(abs(x[i] - x[j]) > max(min_step, 5 * np.hypot(s[i], s[j])) for j in (i - 1, i + 1))
    idx = np.flatnonzero(~bad)
    xs, ss = x[idx], s[idx]

    def split(lo, hi):
        ks = range(lo + min_n, hi - min_n + 1)
        if not ks:
            return []
        k = min(ks, key=lambda k: np.abs(xs[lo:k] - np.median(xs[lo:k])).sum()
                + np.abs(xs[k:hi] - np.median(xs[k:hi])).sum())
        a, b = xs[lo:k], xs[k:hi]
        dev = np.r_[a - np.median(a), b - np.median(b)]
        sd = max(1.4826 * np.median(np.abs(dev)), np.median(ss[lo:hi]), 0.003)
        step = np.median(b) - np.median(a)
        if abs(step) > min_step and abs(step) > min_sig * sd * np.sqrt(1 / a.size + 1 / b.size):
            return split(lo, k) + [k] + split(k, hi)
        return []

    cuts = [idx[k] for k in split(0, idx.size)]
    return np.searchsorted(cuts, np.arange(x.size), side="right"), bad


def measure(images, exts=None, verbose=True, extrap_factor=3.0, model="smooth", combine="incoherent",
            autoconfig_args=(), memory="off"):
    """{ext: result} from a list of images of the same CCDs; model: smooth (default), interpolated, broken or linear.

    combine (several images only): "incoherent" (default; comb powers of the images added) or
    "coherent" (each image converted with its own curve, then all pixels pooled into one comb, see
    _measure_coherent). Both are independent of the order of the images.

    images: paths of .fz/.fits images of the same CCDs; exts: extension numbers (default:
    those of the reference JSON present in the first image); extrap_factor: n_cap / n_max;
    autoconfig_args: autoconfig_panaSKImg settings (as run_WADERs --autoconfig-args: geometry,
    gain prior and search range, ...), used for the geometry and the low-charge gain.
    Per extension: the low-charge gain g0 (the starting point of the comb search) = mean of
    autoconfig's gain over the images whose electron peaks are resolved, the clean pixels of
    every image, the comb spacing in each charge window and the g(n) fit. Result keys ("EXTn"):
    ok, g_1e (= g0), g_1e_images (images in that mean), clean_rows, rows, points and, with a fit,
    model, linear, rms, n_max_measured, drop_at_1000 (fractional drop of g from 0 to 1000 e-),
    nonlinearity; reason when not ok. Extensions without resolved electron peaks in any image
    give dict(ok=False, reason=...).

    memory: horizontal readout memory removed from the pixels before the comb is measured (remove_memory):
      "off" (default) - not removed;
      "auto" - coefficients fitted per extension from these images (_fit_memory), result key 'memory'
               = dict(kL, kR, power_off, power_on, images); an extension where no image has its own curve
               borrows the median of the others (key 'borrowed'); none fitted at all -> no correction;
      a number k - kL = k, kR = 0 for every extension;
      {EXTn: (kL, kR)} - given per extension (used internally).
    The correction sharpens the comb at high charge (on the AMtest and DM24-26 CCDs the comb power at
    300-1100 e- rose by 25-100 %), so the curve is measured to higher charge.
    """
    images = sorted(images)                     # same result whatever the order of the input
    if memory == "auto" or isinstance(memory, (int, float)) and not isinstance(memory, bool):
        if memory == "auto":
            fitted = _fit_memory(images, exts, extrap_factor, model, autoconfig_args)
        else:
            exts_ = _measure_setup(images, exts, autoconfig_args)[3]
            fitted = {f"EXT{e}": dict(kL=float(memory), kR=0.0) for e in exts_}
        res = measure(images, exts, verbose, extrap_factor, model, combine, autoconfig_args,
                      memory={n: (d["kL"], d["kR"]) for n, d in fitted.items()})
        for n, d in fitted.items():
            if n in res:
                res[n]["memory"] = d
                if verbose:
                    print(f"  {n}: readout memory removed, kL {d['kL']:+.2e}, kR {d['kR']:+.2e}"
                          + (f" (comb power at {MEMORY_WINDOW[0]}-{MEMORY_WINDOW[1]} e-: {d['power_off']:.0f} -> "
                             f"{d['power_on']:.0f})" if "power_on" in d else "")
                          + (f" ({d['borrowed']}: no image of this CCD has a usable comb of its own)" if "borrowed" in d else ""))
        return res
    if memory == "off":
        memory = None
    if len(images) > 1 and combine == "coherent":
        return _measure_coherent(images, exts, verbose, extrap_factor, model, autoconfig_args, memory)
    A, fits, hdul0, exts, cfg, aargs, gp = _measure_setup(images, exts, autoconfig_args)
    hduls = [hdul0] + [fits.open(f) for f in images[1:]]

    results = {}
    for e in exts:
        name = f"EXT{e}"
        # low-charge gain from every image, not only the first: the incoherent result must not depend on
        # which image comes first (a single image gives exactly its own g0, as before)
        g0s = []
        for h in hduls:
            r = A.analyse_extension(h[e].data, name, cfg, aargs, gp.get(name, 1.0))
            if not r.get("dead") and r.get("resolved"):
                g0s.append(float(r["gain"]))
        if not g0s:
            results[name] = dict(ok=False, reason="no resolved electron peaks (dead, flooded or too noisy)")
            if verbose:
                print(f"  {name}: skipped ({results[name]['reason']})")
            continue
        g0 = float(np.mean(g0s))
        vals, nrows_clean, nrows, lows = _measure_clean_values(fits, images, e, cfg, g0, (memory or {}).get(name),
                                                               low=LOW_CHECK)
        pts = _measure_comb_points(vals, g0, window_pairs(model))
        low = _measure_comb_points(vals, g0) if model == "smooth" else None
        check = _measure_comb_points(lows, g0, [LOW_CHECK])
        res = _measure_fit(pts, g0, nrows_clean, nrows, model, extrap_factor, low, check[0] if check else None)
        res["g_1e_images"] = len(g0s)
        results[name] = res
        if verbose:
            _measure_print(name, res, g0, nrows_clean, nrows)
    for h in hduls:
        h.close()
    return results


def _adu(n, model):
    """A(n) for any model form (closed form for linear, table otherwise); scalar n [e-] -> ADU."""
    if model_form(model) == "linear":
        return float(adu_of_n(n, model["a"], model["b"], model["n_cap"]))
    nt, A, gcap = _table(model)
    return float(np.interp(n, nt, A)) if n <= nt[-1] else float(A[-1] + (n - nt[-1]) * gcap)


def nonlinearity(points, model):
    """Nonlinearity curve with the curve's low-charge gain g(0) as reference:
    n_meas = A(n) / g(0), Delta(n) = n_meas - n (measured charge minus true electron number).
    Points: A(n) as the trapezoid integral of the measured local gains (g held at the first point
    below it), with errors of the bins added in quadrature; and a quadratic c2 x^2 + c1 x through
    the origin fitted to Delta vs n_meas (x = measured charge).

    This is the familiar way to show a nonlinearity: how many electrons a pixel seems to
    hold if converted with one constant gain, minus how many it holds. Returns dict(g_ref,
    n_meas, delta, delta_err, c2, c1, delta_at_1000 = Delta at 1000 e- from the model).
    """
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
    """One panel per extension: measured charge - peak number vs measured charge, g_ref = g(0).

    results: output of measure(); writes the PNG to path. Shows the points from the measured
    local gains, the model's curve and the quadratic fit of nonlinearity().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False      # pysimdamicm's plot style turns LaTeX on; not needed (or installed)
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
        ax.plot(Ag / nl["g_ref"], Ag / nl["g_ref"] - grid, color="0.3", lw=1.2, label=f"{m.get('fit', model_form(m))} model")
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


def plot_adu(results, path, title):
    """One column per extension: A(n), the ADU of a pixel with n electrons (top), and A(n) - g(0) n (bottom).

    results: output of measure(); writes the PNG to path. Points: A(n) as the trapezoid integral of the
    measured local gains (as nonlinearity()); line: the model's A(n) (the table the correction inverts),
    dashed: g(0) n (a constant gain); dotted vertical: the highest measured charge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False      # pysimdamicm's plot style turns LaTeX on; not needed (or installed)
    names = sorted(results)
    fig, axs = plt.subplots(2, len(names), figsize=(4.4 * len(names), 6.6), squeeze=False, sharex="col",
                            gridspec_kw=dict(height_ratios=(1.4, 1)))
    for k, name in enumerate(names):
        top, bot = axs[0, k], axs[1, k]
        r = results[name]
        if not r.get("ok"):
            top.text(0.5, 0.5, "no reliable curve", ha="center", va="center", transform=top.transAxes)
            top.set_title(name)
            for a_ in (top, bot):
                a_.set_xticks([]); a_.set_yticks([])
            continue
        m = r["model"]
        g0 = float(g_of(0.0, m))
        nmax = float(r["n_max_measured"])
        grid = np.linspace(0, 1.3 * nmax, 400)
        Ag = np.array([_adu(v, m) for v in grid])
        good = [p for p in r["points"] if p["significant"]]
        n = np.array([p["n"] for p in good]); g = np.array([p["g"] for p in good])
        if n.size:
            Ap = np.empty_like(n); Ap[0] = g[0] * n[0]
            Ap[1:] = Ap[0] + np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(n))
            top.plot(n, Ap, "o", ms=4, color="#1f5fbf", label="from the measured local gains")
            bot.plot(n, Ap - g0 * n, "o", ms=4, color="#1f5fbf")
        top.plot(grid, Ag, color="0.2", lw=1.3, label=f"A(n), {m.get('fit', model_form(m))} model")
        top.plot(grid, g0 * grid, color="#d1495b", ls="--", lw=1.0, label=f"g(0) n, g(0) = {g0:.4f}")
        bot.plot(grid, Ag - g0 * grid, color="0.2", lw=1.3)
        bot.axhline(0, color="#d1495b", ls="--", lw=1.0)
        for a_ in (top, bot):
            a_.axvline(nmax, color="0.5", ls=":", lw=0.8)
            a_.grid(alpha=0.3)
        top.set_title(f"{name}: A(1000) = {_adu(1000.0, m):.1f} ADU", fontsize=10)
        top.set_ylabel("A(n) [ADU]")
        bot.set_ylabel("A(n) $-$ g(0) n [ADU]")
        bot.set_xlabel("electrons in the pixel n")
        top.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"ADU of a pixel with n electrons, A(n) = $\\int_0^n g$: {title}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=90)
    plt.close(fig)


def plot(results, path, title):
    """Plot g(n) of every extension: absolute [ADU/e-] (left) and relative to g(0) (right).

    results: output of measure(); writes the PNG to path. Points: significant comb spacings;
    lines: the model (for reliable curves), dotted: the highest measured charge, above which
    the curve is extrapolated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False      # pysimdamicm's plot style turns LaTeX on; not needed (or installed)
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
    forms = sorted({r["model"].get("fit", model_form(r["model"])) for r in results.values() if r.get("ok")})
    ax[0].set_title(f"{title} ({', '.join(forms) or 'no model'})"); ax[1].set_title("relative (dotted: highest measured charge)")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


# =============================================================================
# correction of cluster files
# =============================================================================
def apply(model_json, rootfiles, outdir, verbose=True, anchor="comb", temperature=None):
    """Recompute pixels_E and Energy of panaSKImg cluster files with g(n).

    panaSKImg: pixel energy [keV] = ADU / g_fit * 3.74e-3 (g_fit = info ext<N>_gainext; PANASKIMG_EV),
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

    Each pixel's ADU is recovered from pixels_E with the fitted gain, converted to electrons
    with the model and to keV with epsilon(temperature) (pair_energy.pair_energy_eV; default 130 K or $COMM_TEMPERATURE_K).
    Output: outdir/<same file name>, with the
    clustersRec tree rewritten and the info and process_config trees copied. Files without a
    fit gain or for an extension without a reliable model are copied unchanged; files without
    _ext<N> in the name are skipped.
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)
    eps = pair_energy_eV(temperature)          # eV per electron of the corrected energies
    with open(model_json) as fh:
        models = json.load(fh)["extensions"]
    os.makedirs(outdir, exist_ok=True)
    out = []
    for path in rootfiles:
        base = os.path.basename(path)
        import re
        m = re.search(r"_ext(\d+)\.root$", base)
        if not m:
            print(f"  {base}: no _ext<N> in the name, skipped (run panaSKImg per extension, --ext N)")
            continue
        ext = f"EXT{m.group(1)}"
        model = load_model(models, ext)
        fin = ROOT.TFile.Open(path)
        tin = fin.Get("clustersRec")
        info = fin.Get("info")
        if not tin or not isinstance(tin, ROOT.TTree):
            print(f"  {base}: no clusters, skipped")
            fin.Close()
            continue
        # gain panaSKImg used to convert ADU to keV for this extension
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
            if model and gfit:
                # Energy and pixels_E hold one element per cluster of this tree entry
                # (pixels_E: the pixel energies [keV]); they are edited in place before Fill()
                pe, en = tin.pixels_E, tin.Energy
                for k in range(pe.size()):
                    v = np.array(pe[k], float)
                    vn = electrons_with_anchor(v * 1e3 / PANASKIMG_EV * gfit, model, gfit, anchor) * eps * 1e-3
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
            note = "" if model else "  (no g(n) model for this extension: copied unchanged)"
            print(f"  {base}: {ext}, fit gain {gfit}, median E_new/E_old (E > 1 keV) = {med:.4f}{note}")
    return out


# =============================================================================
# CLI
# =============================================================================
def main():
    """Command line: `measure` (images -> JSON and plots) or `apply` (model + cluster files).

    A first argument that is not a subcommand means `measure`. measure writes
    OUTDIR/NAME_gain_vs_charge.json, NAME_gain_vs_charge.png and NAME_nonlinearity.png.
    """
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
    m.add_argument("--combine", choices=COMBINE, default="incoherent",
                   help="several images: incoherent (default; comb powers added) or coherent (each image converted "
                        "with its own curve, then all pixels pooled; reaches higher charge)")
    m.add_argument("--autoconfig-args", default="", metavar='"ARGS"',
                   help="autoconfig_panaSKImg settings for the geometry and the gain prior/range, in one quoted string "
                        "(as run_WADERs --autoconfig-args, e.g. \"--ccd-cols 3352 --gain-min 1.0\")")
    m.add_argument("--memory", default="off", metavar="off|auto|K",
                   help="remove the horizontal readout memory (a pixel keeps ~3e-4 of the pixel read before it) "
                        "before measuring: off (default), auto (fit per CCD) or a fixed coefficient K")
    m.add_argument("--model", choices=MODELS, default="smooth",
                   help="form of g(n): smooth (default; overlapping windows and a weighted fit above 200 e-, "
                        "the standard points below), interpolated (monotone interpolation through the measured "
                        "points), broken (two straight lines, fitted knot) or linear (straight line)")
    a = sub.add_parser("apply", help="correct cluster ROOT files with a measured g(n)")
    a.add_argument("model", help="*_gain_vs_charge.json from 'measure'")
    a.add_argument("rootfiles", nargs="+", help="panaSKImg_clustersRec_*_extN.root")
    a.add_argument("-o", "--outdir", required=True)
    a.add_argument("--anchor", choices=("comb", "fit"), default="comb",
                   help="low-charge gain: the comb's gain a (default; one g(n) for all images) or the "
                        "image's fitted gain (only the charge dependence corrected)")
    a.add_argument("--temperature", type=float, default=None, metavar="K",
                   help="operating temperature for the energy per electron epsilon(T) (default: $COMM_TEMPERATURE_K or 130 K)")
    args = ap.parse_args(argv)

    if args.cmd == "measure":
        os.makedirs(args.outdir, exist_ok=True)
        from naming import image_stem
        name = args.name or image_stem(args.images[0])
        print(f"g(n) from {len(args.images)} image(s):")
        import shlex
        mem = args.memory if args.memory in ("off", "auto") else float(args.memory)
        res = measure(args.images, args.ext, extrap_factor=args.extrap_factor, model=args.model, combine=args.combine,
                      autoconfig_args=shlex.split(args.autoconfig_args), memory=mem)
        jpath = os.path.join(args.outdir, f"{name}_gain_vs_charge.json")
        with open(jpath, "w") as fh:
            json.dump({"images": [os.path.abspath(i) for i in args.images], "e_per_electron_eV": E_PER_E, "temperature_K": temperature_K(),
                       "extensions": res}, fh, indent=1)
        plot(res, os.path.join(args.outdir, f"{name}_gain_vs_charge.png"), name)
        plot_nonlinearity(res, os.path.join(args.outdir, f"{name}_nonlinearity.png"), name)
        plot_adu(res, os.path.join(args.outdir, f"{name}_adu_of_n.png"), name)
        print(f"-> {jpath}")
    else:
        apply(args.model, args.rootfiles, args.outdir, anchor=args.anchor, temperature=args.temperature)


if __name__ == "__main__":
    main()
