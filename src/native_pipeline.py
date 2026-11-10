#!/usr/bin/env python3
"""Native skipper-CCD processing: raw (skip-averaged) image -> cluster files in panaSKImg format, without panaSKImg
or pysimdamicm.

Place in the pipeline: replaces bin/run_WADERs steps 1-3d (autoconfig + gain_dc_split + panaSKImg runs + g(n)
correction + agmx_root) for the part that produces cluster files. Its output files
    OUT/panaSKImg_clustersRec_<image>_ext<n>.root   (trees clustersRec, info, process_config; TNamed native_config)
have the names, trees and branches (same meaning and units) that MakeComposite and the comm_scripts analysis code
read, so they can go straight into MakeComposite_Fe55_Clusters_Extensions.cc.

Per image the file is read ONCE and every CCD (FITS extension) is processed in the same process:
  1. measure    autoconfig_panaSKImg.analyse_extension (comb gain, noise, per-row dark charge, bad columns, rows
                to use) and the fit settings autoconfig would write (fit_settings + set_dark_current_fit)
  2. pedestal   per-row median of the active region (or of the overscan, when the image has one) after masking
                pixels above median + 1.5 x MAD (MAD unscaled) of the whole region - panaSKImg's
                PedestalSubtractionProcess "median"; the per-row std of the kept pixels is the pedestal sigma
  3. gain, noise, dark current   the chi2 fit of panaSKImg's FitDarkCurrentProcess (Gaussian peaks of width sigma,
                spacing gain, weighted by a Poisson of mean lambda) done twice as gain_dc_split: "all peaks" (gain)
                and the Poisson-like peaks (dark current); gain_dc_split.choose_gain decides, a held gain is fixed in
                a final fit
  4. calibrate  e- = (pixel - pedestal) / gain
  5. g(n)       optional (curve=): ADU -> electrons with gain_vs_charge's A^-1 (electrons_with_anchor), energies
                with epsilon(T) (pair_energy.py) as gain_vs_charge.apply / GainNonlinearityProcess
  6. clusters   'threshold': WADERs-like hysteresis clustering (pixel > T_pix, 8-connected, kept if a pixel >= T_seed)
                'agmx'     : A_GMX (clustering_tests/adaptive.py engine 'gmx') on the calibrated (or g(n)) image
  7. per-cluster quantities (native_kernels.py) and the ROOT files (native_rootio.py)

Thresholds of the 'threshold' clusterer (thresholds=):
  'auto'   (default) in electrons from the CCD's dark charge and read noise, the rule of autoconfig
           --cluster-threshold-e auto / ClusterFinder spr_threshold_auto: pixel > k - 0.5 e- with k the smallest
           number >= 2 with P(Poisson(lambda) >= k) <= 10 % (lambda = median per-row dark charge), seed > max(9.5,
           k + 0.5) e-. Why: in a skipper image the pixels next to a cluster hold dark-current ELECTRONS, not
           Gaussian noise; the read noise (0.14-0.25 e- here) only blurs each electron count. A threshold in units
           of the pedestal sigma (panaSKImg) scales with the width of the 0e-+1e- mixture, i.e. with the dark charge
           (0.13 e- at lambda = 0.17, ~0.75 e- at lambda ~0.5), which puts the pixel threshold anywhere between 1.3
           and 7.5 e- and cuts the tails of diffuse clusters on high-dark-charge CCDs. Resolved peaks mean noise <
           0.3 e-, so k - 0.5 >= 1.5 e- is also >= 5 x the read noise. CCDs whose peaks are not resolved (no electron
           counts, no lambda) fall back to 'sigma0'.
  'sigma0' panaSKImg's rule: pixel > 10 sigma0, seed >= 30 sigma0, sigma0 = mean per-row pedestal sigma / gain.
  (pix_e, seed_e)  fixed thresholds in electrons.

Usage:
    python native_pipeline.py IMAGES... -o OUT [-j N] [--clusterer threshold|agmx|both] [--curve model.json|DIR]
           [--thresholds auto|sigma0|PIX,SEED] [--anchor comb|fit] [--panaskimg-json DIR] [--save-fits]
    (IMAGES: .fz / .fits / .fits.bz2, or a .txt list; -j: images in parallel, at most 2 recommended)
Library: process_image(path, out_dir, curve=None, clusterer='threshold', ...) -> summary dict.

How to read this file: reading and geometry -> pedestal -> dark-current fit and gain choice -> clustering ->
per-cluster table -> ROOT output -> process_image (driver) -> CLI.
Units: ADU after pedestal subtraction unless the name says e- or keV; gain ADU/e-; lambda in ADU in the fit and the
info tree (as panaSKImg), e-/pix elsewhere.
"""
import argparse
import copy
import datetime
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage, optimize
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))              # comm_scripts/src
CS = os.path.dirname(HERE)                                      # comm_scripts
for _p in (os.path.join(CS, "clustering_tests"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import autoconfig_panaSKImg as A      # noqa: E402  measurements and fit settings
import gain_dc_split as GDS           # noqa: E402  choose_gain
import gain_vs_charge as GVC          # noqa: E402  g(n)
import naming                         # noqa: E402
from pair_energy import PANASKIMG_EV, pair_energy_eV   # noqa: E402
import native_kernels as K            # noqa: E402

VERSION = "native-0.1 (2026-11-10)"
EIGHT = np.ones((3, 3), bool)
SQ2PI = math.sqrt(2 * math.pi)
FIT_PARAMS = ["norm", "mu", "sigma", "lam", "gain"]     # order of fit_quality.PARAMS / panaSKImg's fit function
AGMX_CLS = {"x-ray": 1, "x-ray diffuse": 2, "low-E": 3, "muon": 4, "worm": 5, "alpha": 6, "big deposit": 7,
            "artifact-row": 8, "artifact-column": 9, "extended": 10, "glow": 11}      # = agmx_root.CLS_CODES
AGMX_REGIME = {"sparse": 0, "crowded": 1, "unmodelled": 2, "serial": 3, "line": 4}   # = agmx_root.REGIME_CODES
NOT_COMPUTED = -999.0


# =============================================================================
# reading and geometry
# =============================================================================
def read_image(path, autoconfig_args=()):
    """All CCDs of one image, read once. Returns dict(h0 = primary header, exts = [(n, data float64)],
    geo = autoconfig geometry (id_row_start, n_cols_prescan, ccd_cols = first overscan column, no_overscan),
    args = autoconfig argparse namespace, ref = autoconfig's reference JSON)."""
    from astropy.io import fits
    args = A.parse_args([path, "-o", os.devnull] + list(autoconfig_args))
    ref = json.loads(A.REFERENCE_JSON)
    with fits.open(path) as hdul:
        h0 = hdul[0].header.copy()
        exts = [(i, np.asarray(h.data, dtype=np.float64)) for i, h in enumerate(hdul)
                if i > 0 and h.data is not None and np.ndim(h.data) == 2]
        if not exts:
            raise ValueError(f"{path}: no 2-D image extensions")
        _, geo, ncol, nsbin = A.geometry(h0, hdul, ref["input"]["image"], args)
    nskips = int(h0.get("NDCM", h0.get("NSKIPS", 1)) or 1)
    if exts[0][1].shape[1] > ncol:
        raise ValueError(f"{path}: {exts[0][1].shape[1]} columns > NCOL={ncol}: looks like a raw skip image; "
                         f"only skip-averaged images are supported")
    return dict(h0=h0, exts=exts, geo=geo, args=args, ref=ref, nskips=nskips, ncol=ncol, nsbin=nsbin)


def regions(shape, geo):
    """Boolean masks (active, overscan) of one CCD as panaSKImg: all rows; active columns n_cols_prescan .. first
    overscan column; overscan the columns after it (empty if the image has none)."""
    act = np.zeros(shape, bool)
    act[:, geo["n_cols_prescan"]:geo["ccd_cols"]] = True
    ovs = np.zeros(shape, bool)
    ovs[:, geo["ccd_cols"]:] = True
    return act, ovs


def image_ids(stem):
    """(ccd, nfile) as panaSKImg takes them from an 'avg_..._<ACM>_<date>_<time>_<n>' name: the ACM id and the
    image number (1 and 0 otherwise)."""
    acm = naming.acm_id(stem)
    try:
        nfile = int(stem.split("_")[-1]) if stem.startswith("avg_") else 0
    except ValueError:
        nfile = 0
    return (int(acm) if acm else 1), nfile


def readout_times(h0):
    """(readout start, end, exposure + readout time) [s] from DATEINI / DATEEND / EXPOSURE, as panaSKImg (naive
    timestamps, interpreted in the local time zone); None if missing."""
    try:
        import dateutil.parser as dp
        t0 = dp.parse(h0["DATEINI"]).timestamp()
        t1 = dp.parse(h0["DATEEND"]).timestamp()
    except (KeyError, ValueError, ImportError):
        return None, None, None
    return t0, t1, float(h0.get("EXPOSURE", 0.0)) + (t1 - t0)


# =============================================================================
# pedestal
# =============================================================================
def pedestal_rows(raw, region, n_sigma_mask=1.5):
    """Per-row pedestal (median) and sigma (std) as panaSKImg's PedestalSubtractionProcess method "median", axis
    "row": pixels of `region` more than n_sigma_mask x MAD above the region's median are masked first (MAD NOT
    scaled to a Gaussian sigma; only the right tail is cut, as cut_right_tail), so tracks and most charged pixels do
    not pull the row median. The per-row std of what is left is the pedestal sigma (it includes the 0e-/1e- mixture,
    i.e. it grows with the dark charge). Rows with < 2 pixels get nan. Returns (mu_row, sigma_row) [ADU]."""
    v = raw[region]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    keep = region & ((raw - med) <= n_sigma_mask * mad) if n_sigma_mask > 0 else region
    x = np.where(keep, raw, np.nan)
    cnt = keep.sum(axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmedian(x, axis=1)
            sig = np.nanstd(x, axis=1)
    mu[cnt < 2] = np.nan
    sig[cnt < 2] = np.nan
    return mu, sig


# =============================================================================
# dark-current fit and gain choice
# =============================================================================
def ext_settings(dcp, ext):
    """FitDarkCurrentProcess block (per-ext dicts or global values) -> flat settings of one extension."""
    g = lambda k, d=None: (dcp.get(k, d)[ext] if isinstance(dcp.get(k, d), dict) else dcp.get(k, d))
    return dict(rows_to_mask=list(g("rows_to_mask", []) or []), cols_to_mask=list(g("cols_to_mask", []) or []),
                n_peaks=int(g("n_peaks")), x_min=float(dcp["x_min"]), x_max=float(dcp["x_max"]),
                binning_size=float(dcp["binning_size"]), mu_min=float(g("mu_min")), mu_max=float(g("mu_max")),
                sigma_min=float(g("sigma_min")), sigma_max=float(g("sigma_max")), dc_min=float(g("dc_min")),
                dc_max=float(g("dc_max")), gain_min=float(g("gain_min")), gain_max=float(g("gain_max")),
                calibration=float(dcp.get("calibration", 1.0) or 1.0), do_calibration=bool(dcp.get("do_calibration", True)))


def autoconfig_fit_settings(results, args):
    """The per-CCD FitDarkCurrentProcess settings autoconfig --per-ext writes, for the two fits of gain_dc_split:
    {'gain': {ext: settings}, 'dc': {ext: settings}} ('gain' = --peaks all, 'dc' = --peaks auto, i.e. the
    Poisson-like peaks with the module-wide number of peaks of choose_peaks)."""
    out = {}
    for mode, peaks in (("gain", "all"), ("dc", args.peaks)):
        rs = copy.deepcopy(results)
        a = copy.copy(args)
        a.peaks = peaks
        A.choose_peaks(rs, a)
        out[mode] = {}
        for r in rs:
            r["fit"] = A.fit_settings(r)
            j = json.loads(A.REFERENCE_JSON)
            A.set_dark_current_fit(j, [r])
            out[mode][r["ext"]] = ext_settings(j["process"]["FitDarkCurrentProcess"], r["ext"])
    return out


def fit_mask(shape, active, s):
    """Pixels used in the dark-current fit: the active region without rows_to_mask / cols_to_mask (panaSKImg
    syntax: [a, b] = a..b-1, an int = one row/column)."""
    m = active.copy()
    for r in s["rows_to_mask"]:
        if isinstance(r, list) and len(r) == 2:
            m[r[0]:r[1], :] = False
        else:
            m[int(r), :] = False
    for c in s["cols_to_mask"]:
        if isinstance(c, list) and len(c) == 2:
            m[:, c[0]:c[1]] = False
        else:
            m[:, int(c)] = False
    return m


def comb_model(c, p, npk):
    """panaSKImg's fit function at bin centres c [ADU]: norm x sum_{i<npk} Gaus(c/g; i + mu/g, sigma/g, normalised)
    x Poisson(i; lambda/g). p = (norm, mu, sigma, lambda, gain), all in ADU (lambda/g = dark charge in e-/pix)."""
    N, mu, sg, lam, g = p
    i = np.arange(npk)
    se = sg / g
    le = max(lam / g, 1e-300)
    gau = np.exp(-0.5 * ((c[:, None] / g - i - mu / g) / se) ** 2) / (SQ2PI * se)
    poi = np.exp(i * np.log(le) - le - gammaln(i + 1))
    return N * (gau * poi).sum(axis=1)


def fit_dark_current(ps, mask, s, start, gain_fixed=None):
    """chi2 fit of the pixel-charge histogram, as FitDarkCurrentProcess (Neyman chi2, empty bins left out, model at
    the bin centres, limits from the settings s). Same model, window and binning; scipy's bounded least squares
    instead of Minuit (same minimum; errors from the Jacobian, i.e. the Minuit/HESSE errors of a chi2 fit, not
    scaled by chi2/ndf). gain_fixed: hold the gain exactly (gain_dc_split holds it within +-0.001 %).
    start: dict(gain, sigma, lam [e-/pix]) from autoconfig. Returns a fit dict in fit_quality's format
    ({param: (value, error)}, chi2, ndf) plus chi2_L (Baker-Cousins likelihood chi2 / ndf, what panaSKImg stores),
    included_pixels, total_pixels, n_bins."""
    data = ps[mask]
    x0, x1 = s["x_min"], s["x_max"]
    nb = int(abs(x1 - x0) / s["binning_size"])
    sel = data[(data > x0) & (data < x1)]
    y, edges = np.histogram(sel, nb, (x0, x1))
    c = 0.5 * (edges[1:] + edges[:-1])
    use = y > 0
    cu, yu = c[use], y[use].astype(float)
    npk = s["n_peaks"]
    lo = [0.0, s["mu_min"], s["sigma_min"], s["dc_min"], s["gain_min"]]
    hi = [np.inf, s["mu_max"], s["sigma_max"], s["dc_max"], s["gain_max"]]
    g0 = gain_fixed if gain_fixed is not None else float(np.clip(start["gain"], s["gain_min"], s["gain_max"]))
    clip = lambda v, i: float(np.clip(v, lo[i] + 1e-6 * (hi[i] - lo[i]), hi[i] - 1e-6 * (hi[i] - lo[i])))
    p0 = [max(sel.size * (x1 - x0) / nb / g0, 1.0), clip(0.5 * (s["mu_min"] + s["mu_max"]), 1),
          clip(start["sigma"], 2), clip(max(start["lam"], 1e-3) * g0, 3), g0]
    free = [0, 1, 2, 3] + ([] if gain_fixed is not None else [4])

    def full(q):
        p = np.array(p0, float)
        p[free] = q
        return p

    def resid(q):
        return (yu - comb_model(cu, full(q), npk)) / np.sqrt(yu)

    q0 = np.array([p0[i] for i in free])
    if gain_fixed is None:
        q0[-1] = clip(q0[-1], 4)
    res = optimize.least_squares(resid, q0, bounds=([lo[i] for i in free], [hi[i] for i in free]),
                                 x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
    p = full(res.x)
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J)
        err_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err_free = np.full(len(free), np.nan)
    err = np.zeros(5)
    err[free] = err_free
    chi2 = float(np.sum(res.fun ** 2))
    ndf = int(use.sum()) - 5          # ROOT counts a parameter held by limits as free
    f = comb_model(c, p, npk)
    with np.errstate(divide="ignore", invalid="ignore"):
        bc = 2 * np.sum(f - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / f), 0.0))
    out = {k: (float(p[i]), float(err[i])) for i, k in enumerate(FIT_PARAMS)}
    out.update(chi2=chi2, ndf=ndf, chi2_L=float(bc / max(ndf, 1)), included_pixels=int(sel.size),
               total_pixels=int(mask.size), n_bins=nb, binning_size_used=float((x1 - x0) / nb), ok=bool(res.success),
               settings=s)
    return out


def gain_noise_dc(ps, active, ext, r, fs, mimic=None):
    """Gain, noise, dark current of one CCD (gain_dc_split logic with native fits).

    fs: autoconfig_fit_settings() result; r: autoconfig measurement of this CCD (start values). Without mimic: the
    'gain' (all peaks) and 'dc' (Poisson-like peaks) fits, gain_dc_split.choose_gain, and a final fit with the gain
    fixed where it is held. mimic: settings of one panaSKImg JSON (FitDarkCurrentProcess of the final JSON) -> one
    fit with exactly those settings (a gain window narrower than 1e-4 x gain = held, fixed at its centre).
    Returns (final fit dict, notes, the fits {'gain', 'dc'}, held (gain, error) or None)."""
    start = dict(gain=r["gain"], sigma=r.get("sigma_raw", r["sigma"]), lam=r.get("lam01") or r.get("lam") or 0.1)
    if mimic is not None:
        s = mimic
        held = None
        if s["gain_max"] - s["gain_min"] < 1e-4 * s["gain_max"]:
            held = (0.5 * (s["gain_min"] + s["gain_max"]), None)
        fit = fit_dark_current(ps, fit_mask(ps.shape, active, s), s, start, gain_fixed=held[0] if held else None)
        return fit, "settings of the panaSKImg JSON" + (" (gain held)" if held else ""), {}, held
    fits = {}
    for mode in ("gain", "dc"):
        s = fs[mode][ext]
        fits[mode] = fit_dark_current(ps, fit_mask(ps.shape, active, s), s, start)
    gcfg = {"gain_min": {ext: fs["gain"][ext]["gain_min"]}, "gain_max": {ext: fs["gain"][ext]["gain_max"]}}
    held, note = GDS.choose_gain(ext, {ext: fits["gain"]}, {ext: fits["dc"]}, gcfg)
    if held:
        s = fs["dc"][ext]
        final = fit_dark_current(ps, fit_mask(ps.shape, active, s), s, start, gain_fixed=held[0])
        final["gain"] = (held[0], held[1])        # the error of the held value (many-peak fit, scaled)
    else:
        final = fits["dc"]
    return final, note, fits, held


# =============================================================================
# clustering
# =============================================================================
def thresholds_for(mode, r, sigma0_e, mimic_thr=None):
    """(pixel threshold, seed threshold [e-], sigma_seed [e-], n_seed, description) for the 'threshold' clusterer.
    sigma_seed / n_seed only feed the panaSKImg bookkeeping branches (sigma_seed_eV, qmin_cluster_seed, seed_nsig_*):
    as panaSKImg, sigma_seed = 1 e- for thresholds given in electrons."""
    if mimic_thr is not None:                       # ClusterFinder spr_threshold of a panaSKImg JSON
        sig, npix, nseed = mimic_thr[:3]
        if sig is None:
            return npix * sigma0_e, nseed * sigma0_e, sigma0_e, nseed, f"panaSKImg JSON: {npix}/{nseed} x sigma0"
        return npix * sig, nseed * sig, float(sig), nseed, f"panaSKImg JSON: {npix}/{nseed} x {sig} e-"
    if isinstance(mode, (tuple, list)):
        return float(mode[0]), float(mode[1]), 1.0, float(mode[1]), f"fixed: pixel > {mode[0]} e-, seed >= {mode[1]} e-"
    if mode == "auto":
        thr = A.auto_cluster_thresholds([r]).get(r["ext"])
        if thr is not None:
            pix, seed, lam, k = thr
            return pix, seed, 1.0, seed, (f"auto: dark charge {lam:.3f} e-/pix (median of rows) -> pixels >= {k} e- "
                                         f"(> {pix:g}), seed >= {seed:g} e- (read noise {r['sigma'] / r['gain']:.3f} e-)")
        note = "auto: peaks not resolved -> sigma0 rule (10/30 sigma0)"
    elif mode == "sigma0":
        note = "sigma0: 10/30 x sigma0 (panaSKImg)"
    else:
        raise ValueError(f"thresholds {mode!r}: use 'auto', 'sigma0' or (pixel_e, seed_e)")
    return 10 * sigma0_e, 30 * sigma0_e, sigma0_e, 30.0, note + f", sigma0 = {sigma0_e:.4f} e-"


def threshold_clusters(e_img, region, t_pix, t_seed):
    """WADERs ClusterFinder ('waders', method 0) on the calibrated image [e-]: pixels outside `region` set to 0,
    pixels <= t_pix set to 0, 8-connected groups of the rest, kept if one pixel is >= t_seed. Cluster order and
    pixel order as panaSKImg (clusters in the iteration order of the Python set of seed labels, pixels row-major),
    so cluster_id / closest_cls_id match. Returns (rows, cols, offsets) of the kept clusters' pixels."""
    img = np.where(region, e_img, 0.0)
    sel = img >= t_pix
    seed = sel & (img >= t_seed)
    img[img <= t_pix] = 0.0
    lab, n = ndimage.label(img > 0.0, structure=EIGHT)
    order = [int(v) for v in list(set(lab[seed])) if v > 0]
    fl = lab.ravel()
    nz = np.flatnonzero(fl)
    labs = fl[nz]
    srt = np.argsort(labs, kind="stable")
    nz, labs = nz[srt], labs[srt]
    if not order:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(1, np.int64)
    o = np.array(order)
    a = np.searchsorted(labs, o, "left")
    b = np.searchsorted(labs, o, "right")
    idx = np.concatenate([nz[i:j] for i, j in zip(a, b)])
    off = np.concatenate([[0], np.cumsum(b - a)]).astype(np.int64)
    return idx // e_img.shape[1], idx % e_img.shape[1], off


def agmx_valid(shape, active, first_row):
    """A_GMX's active pixels as agmx_root.valid_mask / comm_clustering: rows from first_row + 3 to the second-last-but-one,
    columns of the active region without a 2-pixel border."""
    cols = np.flatnonzero(active.any(axis=0))
    v = np.zeros(shape, bool)
    v[first_row + 3:shape[0] - 2, cols[0] + 2:cols[-1] + 1 - 2] = True
    return v & active


def agmx_clusters(Q, valid, crowded="auto", crowded_shapes="on"):
    """A_GMX (adaptive_cluster engine 'gmx') on the image Q [e-]; returns (rows, cols, charge [e-], offsets, extras
    {agmx_*: array}, info). Clusters without pixels or a finite energy are dropped (as agmx_root)."""
    import adaptive as AD
    AD.CROWDED, AD.CROWDED_SHAPES = crowded, crowded_shapes     # also what the crowded-mode record reports
    Qv = np.where(valid, Q, 0.0)
    try:
        bg = AD.AM.background_map(Qv, valid)
    except Exception:
        bg = None
    cl, info = AD.adaptive_cluster(Qv, valid, bg, engine="gmx", return_pixels=True, return_info=True,
                                   crowded=crowded, crowded_shapes=crowded_shapes)
    cl = [c for c in cl if np.isfinite(c.get("E", np.nan)) and "_pix" in c and len(c["_pix"][0])]
    ys = [np.asarray(c["_pix"][0], np.int64) for c in cl]
    xs = [np.asarray(c["_pix"][1], np.int64) for c in cl]
    qs = [np.asarray(c["_pix"][2], float) for c in cl]
    off = np.concatenate([[0], np.cumsum([len(v) for v in ys])]).astype(np.int64)
    cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
    ex = {"agmx_cls": np.array([AGMX_CLS.get(c.get("cls"), 0) for c in cl], np.int64),
          "agmx_regime": np.array([AGMX_REGIME.get(c.get("regime"), -1) for c in cl], np.int64)}
    for k in ("blend", "ped", "peakfrac", "chi2", "sig"):
        v = [c.get(k, np.nan) for c in cl]
        ex[f"agmx_{k}"] = np.array([float(x) if x is not None and np.isfinite(x) else NOT_COMPUTED for x in v], float)
    return cat(ys, np.int64), cat(xs, np.int64), cat(qs, float), off, ex, info


def crowded_record(info, crowded):
    """crowded-mode record for the agmx_config TNamed (as agmx_root.crowded_summary)."""
    r = dict(setting=crowded, core_fill=info.get("core_fill"), on=bool(info.get("crowded_mode", False)))
    if r["on"]:
        bo, bc = info.get("bg_override") or {}, info.get("bg_columns") or {}
        r.update(hot_columns=int(bc.get("columns", 0)), glow=int(info.get("n_glow", 0)),
                 shapes=bool(info.get("crowded_shapes", False)), lines=int(info.get("n_lines", 0)),
                 topology_splits=int(info.get("n_topology_splits", 0)), bg_override=bool(bo.get("applied")),
                 bad_rows=int(bo.get("bad_rows", 0)), cells_changed=int(bo.get("cells_changed", 0)),
                 straight_tracks=int(info.get("n_hints", 0)), streaks=int(info.get("n_streaks", 0)),
                 linked=int(info.get("n_linked", 0)))
    return r


# =============================================================================
# per-cluster table (the clustersRec branches)
# =============================================================================
def cluster_table(rows, cols, off, E_props, E_final, E_seed, ps, raw, sigma_seed_eV, n_seed, ccd, nfile, times,
                  extras=None):
    """All clustersRec branches of one CCD as {name: (kind, value)} (kinds as native_rootio).

    rows, cols: 0-based pixel indices; off: cluster offsets; E_props: pixel energies [keV] the shape quantities are
    computed from; E_final: pixels_E / Energy written; E_seed: pixel energies [keV] for has_seed and seed_nsig_*
    (panaSKImg: the energies before any g(n) correction). ps / raw: pedestal-subtracted / raw image (ADU) for
    pixels_E_PS / pixels_E_AVG. sigma_seed_eV, n_seed: the seed noise unit and multiple (qmin_cluster_seed =
    n_seed x sigma_seed). times: (readout_start, readout_end) or (None, None)."""
    n = off.size - 1
    x = cols.astype(np.float64) + 1.0
    y = rows.astype(np.float64) + 1.0
    xi, yi = cols + 1, rows + 1
    M = K.moments(x, y, E_props, off)
    col = {nm: M[:, i] for i, nm in enumerate(K.MOMENT_NAMES)}
    # principal axes of the weighted covariance (np.linalg.eigh per 2x2 matrix, as pysimdamicm: same LAPACK routine,
    # same eigenvector signs); eigenvalues ascending -> long axis = column 1
    cxx, cyy, cxy = col["wSTD_X"] ** 2, col["wSTD_Y"] ** 2, col["pcaCovXY"]
    C = np.empty((n, 2, 2))
    C[:, 0, 0], C[:, 0, 1], C[:, 1, 0], C[:, 1, 1] = cxx, cxy, cxy, cyy
    # eigh also on matrices with a nan (wSTD of a cluster with negative pixel charges): LAPACK returns finite
    # eigenvalues and nan vectors there, which pysimdamicm keeps
    w, V = np.linalg.eigh(C) if n else (np.zeros((0, 2)), np.zeros((0, 2, 2)))
    # pysimdamicm orders with np.argsort(eigenvalues)[::-1]: long axis = column 1, except when only w[0] is nan
    # (argsort puts nan last)
    il = np.where(np.isnan(w[:, 0]) & ~np.isnan(w[:, 1]), 0, 1)
    it = 1 - il
    r_ = np.arange(n)
    l1, l2 = np.maximum(w[r_, il], 0.0), np.maximum(w[r_, it], 0.0)
    lx, ly, tx, ty = V[r_, 0, il], V[r_, 1, il], V[r_, 0, it], V[r_, 1, it]
    eps = 1e-12
    s_long, s_trans = np.sqrt(l1), np.sqrt(l2)
    u, v, L, W = K.pca_projections(x, y, off, col["PosX"], col["PosY"], lx, ly, tx, ty)
    px, offx, py, offy, fpx, fsx, fpy, fsy = K.profiles(xi, yi, E_props, off, col["minX"], col["maxX"],
                                                        col["minY"], col["maxY"])
    e1x, e1y, e2x, e2y = K.track_endpoints(xi, yi, off, col["_iqmax"].astype(np.int64))
    seg = np.repeat(np.arange(n), np.diff(off))
    qmin_keV = sigma_seed_eV * n_seed / 1000.0              # eV -> keV as pysimdamicm (/ u.keV)
    has_seed = np.zeros(n, np.int64)
    np.maximum.at(has_seed, seg, (E_seed > qmin_keV).astype(np.int64))
    nsig = E_seed / (sigma_seed_eV / 1000.0)
    nsmin = np.full(n, np.inf); nsmax = np.full(n, -np.inf)
    np.minimum.at(nsmin, seg, nsig); np.maximum.at(nsmax, seg, nsig)
    d, ddx, ddy, cid = K.closest(col["PosX"], col["PosY"], has_seed.astype(np.bool_))
    psv, rawv = ps[rows, cols], raw[rows, cols]
    energy = K.seg_sums(E_final, off)
    eps_ = K.seg_sums(psv, off)
    eavg = K.seg_sums(rawv, off)
    projx_cols = np.concatenate([np.linspace(a - 1.5, b + 1.5, int(b - a) + 3) for a, b in zip(col["minX"], col["maxX"])]) if n else np.zeros(0)
    projy_rows = np.concatenate([np.linspace(a - 1.5, b + 1.5, int(b - a) + 3) for a, b in zip(col["minY"], col["maxY"])]) if n else np.zeros(0)
    t0, t1 = times
    B = {
        "DX": col["DX"], "DY": col["DY"], "Energy": energy, "Energy_AVG": eavg, "Energy_PS": eps_,
        "Npix": col["Npix"].astype(np.int64), "PosX": col["PosX"], "PosY": col["PosY"],
        "Qmax": col["Qmax"], "QmaxX": col["QmaxX"], "QmaxY": col["QmaxY"], "RMSX": col["RMSX"], "RMSY": col["RMSY"],
        "STD_X": col["STD_X"], "STD_XY": col["STD_XY"], "STD_Y": col["STD_Y"],
        "ccd": np.full(n, ccd, np.int64),
        "closest_cls_dist": d, "closest_cls_dist_X": ddx, "closest_cls_dist_Y": ddy, "closest_cls_id": cid,
        "cluster_id": np.arange(1, n + 1, dtype=np.int64),
        "fPosX": fpx, "fPosY": fpy, "fSTD_X": fsx, "fSTD_Y": fsy, "has_seed": has_seed,
        "maxX": col["maxX"], "maxY": col["maxY"], "meanX": col["meanX"], "meanY": col["meanY"],
        "minX": col["minX"], "minY": col["minY"], "nfile": np.full(n, nfile, np.int64),
        "pcaAxisLongX": lx, "pcaAxisLongY": ly, "pcaAxisTransX": tx, "pcaAxisTransY": ty,
        "pcaCovXX": cxx, "pcaCovXY": cxy, "pcaCovYY": cyy,
        "pcaEccentricity": 1.0 - l2 / np.maximum(l1, eps), "pcaElongation": s_long / np.maximum(s_trans, eps),
        "pcaLambdaLong": l1, "pcaLambdaTrans": l2, "pcaLength95": L,
        "pcaLinearity": s_trans / np.maximum(s_long, eps), "pcaSigmaLong": s_long, "pcaSigmaTrans": s_trans,
        "pcaTheta": np.arctan2(ly, lx), "pcaThetaAnalytic": 0.5 * np.arctan2(2.0 * cxy, cxx - cyy),
        "pcaU": (u, off), "pcaV": (v, off), "pcaWidth68": W, "pcaWidthOverLength": W / np.maximum(L, eps),
        "pixels_E": (E_final, off), "pixels_E_AVG": (rawv, off), "pixels_E_PS": (psv, off),
        "pixels_x": (xi, off), "pixels_y": (yi, off), "pixels_z": (-np.ones(x.size), off),
        "projectionX": (px, offx), "projectionX_cols": (projx_cols, offx),
        "projectionY": (py, offy), "projectionY_rows": (projy_rows, offy),
        "qmin_cluster_seed": np.full(n, qmin_keV),
        "seed_nsig_delta": nsmax - nsmin, "seed_nsig_max": nsmax, "seed_nsig_min": nsmin,
        "seed_nsig_pixels": (nsig, off), "sigma_seed_eV": np.full(n, sigma_seed_eV),
        "track_endpoint1_X": e1x, "track_endpoint1_Y": e1y, "track_endpoint2_X": e2x, "track_endpoint2_Y": e2y,
        "wSTD_X": col["wSTD_X"], "wSTD_XY": col["wSTD_XY"], "wSTD_Y": col["wSTD_Y"],
    }
    if t0 is not None:
        B["readout_start"], B["readout_end"] = np.full(n, t0), np.full(n, t1)
    for k, a in (extras or {}).items():
        B[k] = a
    INT = {"Npix", "ccd", "cluster_id", "has_seed", "nfile", "agmx_cls", "agmx_regime"}
    out = {}
    for k in sorted(B):
        val = B[k]
        if isinstance(val, tuple):
            out[k] = ("VVI" if k in ("pixels_x", "pixels_y") else "VVF", val)
        else:
            out[k] = ("VI" if k in INT else "VF", val)
    return out


# =============================================================================
# ROOT output
# =============================================================================
def write_ccd(path, table, n, info, pconf, named):
    """One panaSKImg-format file: clustersRec (only if there are clusters, as panaSKImg), info, process_config."""
    import native_rootio as rootio
    trees = []
    if n:
        trees.append(("clustersRec", [("RUNID", "I", 0), ("Nclusters", "I", n)] + [(k, kd, v) for k, (kd, v) in table.items()]))
    trees.append(("info", info))
    trees.append(("process_config", pconf))
    rootio.write_file(path, trees, named)


def info_branches(ext_n, mu_row, sig_row, fit, ccd, nfile, h0):
    """the info tree of one CCD, panaSKImg names and units (ADU for the fit results, as FitDarkCurrentProcess)."""
    t0, t1, tot = readout_times(h0)
    E, e = f"EXT{ext_n}", f"ext{ext_n}"
    b = [("RUNID", "I", 0),
         (f"{E}_pedestal_mu_row_mean_ADU", "D", float(np.mean(mu_row))),
         (f"{E}_pedestal_sigma_row_mean_ADU", "D", float(np.mean(sig_row)))]
    for name, key in (("dcext", "lam"), ("gainext", "gain"), ("mu0ext", "mu"), ("sigext", "sigma")):
        b += [(f"{e}_{name}", "D", fit[key][0]), (f"{e}_{name}_err", "D", fit[key][1] or 0.0)]
    b += [("idccd", "I", ccd), ("nfile", "I", nfile)]
    if t0 is not None:
        b += [("readout_end", "D", t1), ("readout_start", "D", t0), ("total_exposure", "D", tot)]
    n = mu_row.size
    b += [(f"{E}_pedestal_mu_row_ADU", "VVF", (mu_row, np.array([0, n]))),
          (f"{E}_pedestal_sigma_row_ADU", "VVF", (sig_row, np.array([0, n])))]
    if t0 is not None:
        b += [("readout_time_row", "VVF", (np.linspace(0, 1, n), np.array([0, n])))]
    return b


def config_branches(fit, thr, ev_per_e):
    """process_config: the settings and fit results that matter downstream (subset of panaSKImg's; e2eV = 3.74 is
    the eV/e- of UNCORRECTED energies, native_eV_per_e the one used for Energy / pixels_E)."""
    s = fit["settings"]
    return [("id", "I", 0), ("e2eV", "D", PANASKIMG_EV), ("native_eV_per_e", "D", ev_per_e),
            ("FitDarkCurrentProcess_chi2", "D", fit["chi2_L"]),
            ("FitDarkCurrentProcess_chi2_neyman", "D", fit["chi2"]), ("FitDarkCurrentProcess_ndf", "I", fit["ndf"]),
            ("FitDarkCurrentProcess_dc_norm", "D", fit["norm"][0]), ("FitDarkCurrentProcess_dc_mu0", "D", fit["mu"][0]),
            ("FitDarkCurrentProcess_dc_sigma", "D", fit["sigma"][0]), ("FitDarkCurrentProcess_dc_lambda", "D", fit["lam"][0]),
            ("FitDarkCurrentProcess_dc_gain", "D", fit["gain"][0]),
            ("FitDarkCurrentProcess_included_pixels", "I", fit["included_pixels"]),
            ("FitDarkCurrentProcess_total_pixels", "I", fit["total_pixels"]),
            ("FitDarkCurrentProcess_n_bins_used", "I", fit["n_bins"]),
            ("FitDarkCurrentProcess_binning_size_used", "D", fit["binning_size_used"]),
            ("FitDarkCurrentProcess_x_min", "D", s["x_min"]), ("FitDarkCurrentProcess_x_max", "D", s["x_max"]),
            ("FitDarkCurrentProcess_n_peaks", "I", s["n_peaks"]),
            ("ClusterFinder_pixel_threshold_e", "D", thr[0]), ("ClusterFinder_seed_threshold_e", "D", thr[1]),
            ("native_version", "S", VERSION)]


# =============================================================================
# driver
# =============================================================================
def mimic_settings(json_dir, stem, ext):
    """(FitDarkCurrentProcess settings, spr_threshold, pedestal block, image block) of a panaSKImg JSON
    json_dir/<stem>_EXTn.json or json_dir/<stem>/final/<stem>_EXTn.json (json_dir = a run_WADERs json/ folder), or
    None if absent."""
    p = os.path.join(json_dir, f"{stem}_{ext}.json")
    if not os.path.exists(p):                         # a run_WADERs json/ folder: json/<image>/final/
        p = os.path.join(json_dir, stem, "final", f"{stem}_{ext}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    pr = j["process"]
    thr = pr.get("ClusterFinder", {}).get("spr_threshold")
    thr = thr.get(ext) if isinstance(thr, dict) else thr
    return dict(fit=ext_settings(pr["FitDarkCurrentProcess"], ext), thr=thr,
                ped=pr.get("PedestalSubtractionProcess", {}), image=j["input"]["image"])


def autoconfig_pedestal(r):
    """PedestalSubtractionProcess settings autoconfig --per-ext writes for one CCD (set_pedestal on that CCD alone):
    the per-row median of the overscan (or of the active region without one), the lowest-peak finder (is_highDC)
    for a charged overscan, or the reference Gaussian fit when the peaks are not resolved."""
    j = json.loads(A.REFERENCE_JSON)
    A.set_pedestal(j, [r])
    return j["process"]["PedestalSubtractionProcess"]


def process_image(path, out_dir, curve=None, clusterer="threshold", thresholds="auto", anchor="comb",
                  ev_per_e=None, exts=None, autoconfig_args=(), panaskimg_json=None, props_from=None,
                  crowded="auto", crowded_shapes="on", first_row=None, save_fits=False, write=True, verbose=True,
                  keep_arrays=False, gain_override=None, outputs=None, all_ext=True):
    """Process every CCD of one image (read once) and write OUT/panaSKImg_clustersRec_<stem>_ext<n>.root.

    outputs: [(key, clusterer, folder, curve)] -> one cluster file per CCD for each entry, all from the same pedestal
    and fits (run_WADERs: root/, root_gaincorr/, root_agmx/, ...); [] = fits only, no cluster files. Default (None):
    from curve / clusterer / out_dir below. all_ext: also process CCDs autoconfig flags as unresolved outliers
    (flooded among good ones; run_WADERs skips them unless --all-ext).
    curve: g(n) model JSON or run_WADERs gain_vs_charge/ folder (None: no correction); clusterer: 'threshold',
    'agmx' or 'both' (then OUT/root/ and OUT/root_agmx/); thresholds: 'auto' | 'sigma0' | (pix_e, seed_e);
    anchor: g(n) anchor ('comb' | 'fit', as gain_vs_charge); ev_per_e: eV per electron of the written energies
    (default: epsilon(T) with a g(n) curve, 3.74 eV without, as panaSKImg / gain_vs_charge.apply); exts: CCD numbers
    (default all); autoconfig_args: extra autoconfig options (geometry, gain prior, ...); panaskimg_json: folder
    with panaSKImg JSONs <stem>_EXTn.json -> pedestal, fit and spr_threshold settings taken from them (validation
    mode; also sets props_from='uncorrected'); props_from: 'final' (default) = cluster shapes / Qmax from the written
    energies, 'uncorrected' = from the energies before g(n) as panaSKImg; first_row: A_GMX's first row (default
    id_row_start of autoconfig's reference, 2); save_fits: also write <stem>_EXT<n>_native.fits (calibrated image
    [e-] and cluster mask); gain_override: {ext number: gain} replaces the fitted gain (validation only: with
    panaSKImg's own gain the cluster quantities can be compared to the last bit). Returns a summary dict (per CCD: fit, thresholds, n clusters, timings)."""
    T = {"start": time.time()}
    stem = naming.image_stem(path)
    img = read_image(path, autoconfig_args)
    T["read"] = time.time()
    h0, geo, args = img["h0"], img["geo"], img["args"]
    ccd, nfile = image_ids(stem)
    times = readout_times(h0)[:2]
    names = [f"EXT{n}" for n, _ in img["exts"]]
    gp = A.gain_priors(img["ref"], names, args)
    if props_from is None:
        props_from = "uncorrected" if panaskimg_json else "final"
    if outputs is None:
        clus = ["threshold", "agmx"] if clusterer == "both" else [clusterer]
        outdirs = ({"threshold": os.path.join(out_dir, "root"), "agmx": os.path.join(out_dir, "root_agmx")}
                   if clusterer == "both" else {clusterer: out_dir})
        outputs = [(cl, cl, outdirs[cl], curve) for cl in clus]
    if write:
        for _, _, d, _ in outputs:
            os.makedirs(d, exist_ok=True)
    first_row = geo["id_row_start"] if first_row is None else first_row

    # ---- 1. autoconfig measurements of every CCD (needed for the module-wide number of fitted peaks)
    meas = {}
    for (n, data), name in zip(img["exts"], names):
        if exts and n not in exts:
            continue
        r = A.analyse_extension(data, name, geo, args, gp.get(name, 1.0))
        meas[n] = r
    live = [r for r in meas.values() if not r["dead"]]
    fs = autoconfig_fit_settings(live, args) if live else {}
    # unresolved outliers (a flooded CCD among good ones), as autoconfig's measured.json 'skip'
    outliers = []
    if live:
        rs = copy.deepcopy(live)
        A.choose_peaks(rs, args)
        for r in rs:
            r["fit"] = A.fit_settings(r)
        outliers = A.set_dark_current_fit(json.loads(A.REFERENCE_JSON), rs) or []
    T["measure"] = time.time()

    summary = dict(image=path, stem=stem, version=VERSION, clusterer=clusterer, thresholds=str(thresholds),
                   curve=curve, ccds={}, files=[])
    for n, raw in img["exts"]:
        if n not in meas:
            continue
        r, ext = meas[n], f"EXT{n}"
        rec = dict(ext=ext, resolved=bool(r["resolved"]), lam01=r.get("lam01") if r["resolved"] else None)
        summary["ccds"][ext] = rec
        if r["dead"]:
            rec["skipped"] = "no CCD signal (autoconfig)"
            continue
        rec["outlier"] = bool(ext in outliers and not r["resolved"] and not getattr(args, "cluster_all", False))
        if rec["outlier"] and not all_ext:
            rec["skipped"] = "unresolved outlier CCD (flooded among good ones; --all-ext to run it)"
            continue
        t_ext = time.time()
        mim = mimic_settings(panaskimg_json, stem, ext) if panaskimg_json else None
        active, ovs = regions(raw.shape, geo)
        # ---- 2. pedestal: the settings autoconfig --per-ext chooses for this CCD (or the panaSKImg JSON's);
        #         overscan if they ask for it and there is one (>= 3 columns)
        ped_cfg = mim["ped"] if mim else autoconfig_pedestal(r)
        use_ovs = bool(ped_cfg.get("in_overscan", False)) and ovs.any(axis=0).sum() >= 3 and not r.get("ped_active")
        high_dc = ped_cfg.get("is_highDC")
        high_dc = bool(high_dc.get(ext)) if isinstance(high_dc, dict) else bool(high_dc)
        rec["pedestal"] = dict(method=ped_cfg.get("method", "median"), lowest_peak=high_dc,
                               region="overscan" if use_ovs else "active")
        if ped_cfg.get("method", "median") != "median" or high_dc:
            rec["pedestal_note"] = (f"pedestal method {ped_cfg.get('method')}{' (lowest peak)' if high_dc else ''} "
                                    f"not implemented natively: median used")
        mu_row, sig_row = pedestal_rows(raw, ovs if use_ovs else active, float(ped_cfg.get("n_sigma_to_mask", 1.5)))
        ps = raw - np.nan_to_num(mu_row, nan=0.0)[:, None]
        t_ped = time.time()
        # ---- 3. gain, noise, dark current
        fit, note, fits, held = gain_noise_dc(ps, active, ext, r, fs, mim["fit"] if mim else None)
        gain = fit["gain"][0]
        if gain_override and n in gain_override:
            gain = float(gain_override[n])
            fit = dict(fit, gain=(gain, fit["gain"][1]))
            note += f" (gain overridden: {gain})"
        t_fit = time.time()
        # ---- 4. calibration (panaSKImg's CalibrationProcess divides by the fitted gain ROUNDED to 4 decimals; done
        #         here only in the panaSKImg-mimic mode: 7e-5 relative, but it moves every pixel energy)
        g_cal = float(np.round(gain, 4)) if panaskimg_json else gain
        e_img = ps / g_cal
        sigma0_e = float(np.nanmean(sig_row)) / g_cal
        rec.update(gain=gain, gain_err=fit["gain"][1], noise_e=fit["sigma"][0] / gain, dc_e=fit["lam"][0] / gain,
                   mu0=fit["mu"][0], chi2ndf=fit["chi2"] / max(fit["ndf"], 1), gain_note=note, sigma0_e=sigma0_e,
                   lam_rows_median=r.get("lam_all_median"), read_noise_autoconfig_e=r["sigma"] / r["gain"],
                   pedestal_region="overscan" if use_ovs else "active", held=held,
                   fits=dict({k: {p: v[p] for p in FIT_PARAMS + ["chi2", "ndf"]} for k, v in fits.items()},
                             final={p: fit[p] for p in FIT_PARAMS + ["chi2", "ndf"]}))
        info = info_branches(n, mu_row, sig_row, fit, ccd, nfile, h0)
        t_cal = time.time()
        # ---- 5. / 6. one cluster file per output (clusterer x g(n) curve), from the same calibrated image
        thr = thresholds_for(thresholds, r, sigma0_e, mim["thr"] if mim else None)
        rec["thresholds"] = dict(pixel_e=thr[0], seed_e=thr[1], sigma_seed_e=thr[2], n_seed=thr[3], note=thr[4])
        sigma_seed_eV = thr[2] * PANASKIMG_EV          # panaSKImg's unit: sigma [e-] x e2eV
        models = {}                                    # curve -> (g(n) model or None, description)
        thr_clusters = None                            # the threshold clusters do not depend on the curve
        for key, cl, odir, cv in outputs:
            if cv not in models:
                model, cdesc = None, None
                if cv is not None:
                    import agmx_root as AR            # curve_file only (no ROOT / pysimdamicm at import)
                    cpath = AR.curve_file(cv, stem)
                    model = GVC.load_model(cpath, n) if cpath else None
                    cdesc = cpath if model else f"{cpath}: no reliable g(n) for {ext} -> fitted gain kept"
                models[cv] = (model, cdesc)
            model, cdesc = models[cv]
            eV = ev_per_e if ev_per_e else (pair_energy_eV() if model is not None else PANASKIMG_EV)
            if key == outputs[0][0]:                   # as before: the first output's curve / eV in the record
                rec.update(curve=cdesc, ev_per_e=eV)
            t_c0 = time.time()
            if cl == "threshold":
                if thr_clusters is None:
                    thr_clusters = threshold_clusters(e_img, active, thr[0], thr[1])
                rows, cols, off = thr_clusters
                e_pix = e_img[rows, cols]
                E_unc = e_pix * PANASKIMG_EV / 1000.0         # (e- x eV/e-) / 1000, the order of pysimdamicm
                if model is not None:
                    # ADU of each pixel, then A^-1. Mimic mode: back from the pixel energy as GainNonlinearityProcess
                    # (keV -> e- -> ADU with the rounded gain, last bits as panaSKImg); else the pedestal-subtracted ADU
                    adu = (E_unc * 1000.0) / PANASKIMG_EV * g_cal if panaskimg_json else ps[rows, cols]
                    E_fin = GVC.electrons_with_anchor(adu, model, g_cal, anchor) * eV / 1000.0
                else:
                    E_fin = e_pix * eV / 1000.0
                E_props = E_unc if props_from == "uncorrected" else E_fin
                extras, named = None, {}
                E_seed = E_unc
            elif cl == "agmx":
                adu_img = e_img * g_cal if panaskimg_json else ps      # mimic: as comm_clustering (calibrated x gain)
                Q = (GVC.electrons_with_anchor(adu_img.ravel(), model, g_cal, anchor).reshape(ps.shape)
                     if model is not None else e_img)
                valid = agmx_valid(ps.shape, active, first_row)
                rows, cols, q, off, extras, ainfo = agmx_clusters(Q, valid, crowded, crowded_shapes)
                E_fin = q * eV / 1000.0
                E_props = E_seed = E_fin
                named = {"agmx_config": json.dumps({ext: dict(
                    method="A_GMX (comm_scripts clustering_tests/adaptive.py engine 'gmx'), native pipeline",
                    gain_corrected=model is not None, gain_anchor=anchor if model is not None else None,
                    parameters=dict(crowded=crowded, crowded_shapes=crowded_shapes), cls_codes=AGMX_CLS,
                    regime_codes=AGMX_REGIME, not_computed=NOT_COMPUTED, crowded_mode=crowded_record(ainfo, crowded),
                    n_clusters=int(off.size - 1), eV_per_e=eV,
                    recommended_selection="agmx_blend < 0.02 and agmx_ped < 10 (and fSTD/sigma cut of the analysis)")})}
            else:
                raise ValueError(f"clusterer {cl!r}: use 'threshold', 'agmx' or 'both'")
            t_c1 = time.time()
            ncl = off.size - 1
            table = cluster_table(rows, cols, off, E_props, E_fin, E_seed, ps, raw, sigma_seed_eV, thr[3], ccd, nfile,
                                  times, extras) if ncl else {}
            t_c2 = time.time()
            cfg = dict(version=VERSION, image=path, ext=ext, clusterer=cl, thresholds=rec["thresholds"],
                       gain_note=note, curve=cdesc, anchor=anchor if model is not None else None,
                       eV_per_e=eV, props_from=props_from, pedestal=rec["pedestal"],
                       panaskimg_json=panaskimg_json, fit_settings=fit["settings"])
            named = dict(named, native_config=json.dumps(cfg))
            fname = os.path.join(odir, f"panaSKImg_clustersRec_{stem}_ext{n}.root")
            if write:
                write_ccd(fname, table, ncl, info, config_branches(fit, thr, eV), named)
                summary["files"].append(fname)
            t_c3 = time.time()
            rec[key] = dict(n_clusters=int(ncl), file=fname if write else None, curve=cdesc,
                            t_cluster=t_c1 - t_c0, t_props=t_c2 - t_c1, t_write=t_c3 - t_c2)
            if keep_arrays:
                rec[key]["table"] = table
        if keep_arrays:
            rec.update(ps=ps, mu_row=mu_row, sig_row=sig_row, e_img=e_img)
        if save_fits and write and outputs:
            save_calibrated(os.path.join(out_dir, f"{stem}_{ext}_native.fits"), e_img, rows, cols)
        rec["t"] = dict(pedestal=t_ped - t_ext, fit=t_fit - t_ped, calibrate=t_cal - t_fit, total=time.time() - t_ext)
        if verbose:
            cl_txt = ", ".join(f"{k} {rec[k]['n_clusters']} clusters" for k, _, _, _ in outputs) or "no clusters"
            print(f"  {stem} {ext}: gain {gain:.4f} ADU/e-, noise {rec['noise_e']:.3f} e-, DC {rec['dc_e']:.4f} e-/pix,"
                  f" {cl_txt} ({rec['t']['total']:.1f} s)", flush=True)
    summary["t"] = dict(read=T["read"] - T["start"], measure=T["measure"] - T["read"], total=time.time() - T["start"])
    return summary


def save_calibrated(path, e_img, rows, cols):
    """calibrated image [e-] and a cluster mask (cluster pixels dilated by one pixel; not panaSKImg's
    BuildClusterMask with its per-direction margins)."""
    from astropy.io import fits
    m = np.zeros(e_img.shape, bool)
    m[rows, cols] = True
    m = ndimage.binary_dilation(m, structure=EIGHT)
    fits.HDUList([fits.PrimaryHDU(e_img.astype(np.float32)),
                  fits.ImageHDU(m.astype(np.uint8), name="fullmask")]).writeto(path, overwrite=True)


# =============================================================================
# CLI
# =============================================================================
def _job(a):
    path, out, kw = a
    try:
        return process_image(path, out, **kw), None
    except Exception as e:                       # report and go on with the other images
        import traceback
        return None, f"{path}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def expand_inputs(items):
    """image paths; a .txt holds one path per line (relative to the .txt; # comments and empty lines skipped)"""
    out = []
    for it in items:
        if it.endswith(".txt"):
            base = os.path.dirname(os.path.abspath(it))
            for line in open(it):
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(line if os.path.isabs(line) else os.path.join(base, line))
        else:
            out.append(it)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="raw skip-averaged images (.fz/.fits/.fits.bz2) or .txt lists")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="images in parallel (processes; keep <= 2)")
    ap.add_argument("--clusterer", choices=("threshold", "agmx", "both"), default="threshold")
    ap.add_argument("--thresholds", default="auto", help="auto | sigma0 | PIX,SEED (electrons)")
    ap.add_argument("--curve", default=None, help="g(n) model JSON or gain_vs_charge/ folder (gain_vs_charge measure)")
    ap.add_argument("--anchor", choices=("comb", "fit"), default="comb")
    ap.add_argument("--ev-per-e", type=float, default=None, help="eV per electron of the energies (default: "
                    "epsilon(T) with --curve, 3.74 without)")
    ap.add_argument("--ext", type=int, nargs="+", default=None, help="CCD numbers (default: all)")
    ap.add_argument("--panaskimg-json", default=None, help="folder with panaSKImg JSONs <image>_EXTn.json: use their "
                    "pedestal, fit and threshold settings (validation)")
    ap.add_argument("--crowded", default="auto", help="A_GMX crowded mode auto|on|off")
    ap.add_argument("--save-fits", action="store_true", help="also write the calibrated image + cluster mask")
    ap.add_argument("--autoconfig-args", default="", help='extra autoconfig options, e.g. "--ccd-cols 6200"')
    a = ap.parse_args(argv)
    thr = a.thresholds
    if "," in thr:
        thr = tuple(float(v) for v in thr.split(","))
    kw = dict(curve=a.curve, clusterer=a.clusterer, thresholds=thr, anchor=a.anchor, ev_per_e=a.ev_per_e,
              exts=a.ext, autoconfig_args=a.autoconfig_args.split(), panaskimg_json=a.panaskimg_json,
              crowded=a.crowded, save_fits=a.save_fits)
    imgs = expand_inputs(a.images)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    results, failed = [], []
    tasks = [(p, a.out, kw) for p in imgs]
    if a.jobs > 1 and len(tasks) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(a.jobs, len(tasks))) as pool:
            for s, err in pool.imap_unordered(_job, tasks):
                (results.append(s) if s else failed.append(err))
    else:
        for t in tasks:
            s, err = _job(t)
            (results.append(s) if s else failed.append(err))
    files = sorted(f for s in results for f in s["files"])
    with open(os.path.join(a.out, "root_files.txt"), "w") as fh:
        fh.write("".join(f + "\n" for f in files))
    with open(os.path.join(a.out, "native_summary.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    for f in failed:
        print("FAILED", f, file=sys.stderr)
    print(f"native_pipeline: {len(results)} image(s), {len(files)} files in {a.out} ({time.time() - t0:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
