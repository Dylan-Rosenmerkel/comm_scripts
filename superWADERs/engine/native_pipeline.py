#!/usr/bin/env python3
"""Native skipper-CCD processing: raw (skip-averaged) image -> cluster files in panaSKImg format, without panaSKImg
or pysimdamicm.

The engine of run_superWADERs (superWADERs/, independent of bin/run_WADERs and src/: every module it uses is in
superWADERs/engine, clustering, commissioning). Started from src/native_pipeline.py (commit be14496) and the any-image
front end (wip/native_pipeline, 7980f24): clusterer 'auto' (characterise.py: data form, geometry, artefacts, gain
source, noise / dark charge / counting, crowding -> one clustering chain per CCD; the default of run_superWADERs),
--gain / --gain-file for CCDs without a comb, raw skip files compressed first, A_GMX's Bayesian few-electron switch
(fewe, run-pooled dark charge), clustersRec branches finder / p_event / N_post / sigma_post, TNamed
'characterisation'.

Its output files
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
  6. clusters   'auto'     : (run_superWADERs' default) one chain per CCD, chosen by characterise.decide_ccd on the
                             calibrated image: Bayesian few-electron switch inside A_GMX (sparse, low dark charge),
                             A_GMX, A_GMX crowded mode (core fill >= 0.10), A_GMX 'unmodelled' (hysteresis; read
                             noise >= 1 e- or a structured background), threshold clusterer (flooded / not counting)
                'threshold': WADERs-like hysteresis clustering (pixel > T_pix, 8-connected, kept if a pixel >= T_seed)
                'agmx'     : A_GMX (clustering/adaptive.py engine 'gmx') on the calibrated (or g(n)) image
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
    python native_pipeline.py IMAGES... -o OUT [-j N] [--clusterer auto|threshold|agmx|both] [--curve model.json|DIR]
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

HERE = os.path.dirname(os.path.abspath(__file__))              # superWADERs/engine
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import autoconfig_panaSKImg as A      # noqa: E402  measurements and fit settings
import gain_dc_split as GDS           # noqa: E402  choose_gain
import gain_vs_charge as GVC          # noqa: E402  g(n)
import naming                         # noqa: E402
from pair_energy import PANASKIMG_EV, pair_energy_eV   # noqa: E402
import native_kernels as K            # noqa: E402

VERSION = "superWADERs-0.2 (2026-11-12)"
EIGHT = np.ones((3, 3), bool)
SQ2PI = math.sqrt(2 * math.pi)
FIT_PARAMS = ["norm", "mu", "sigma", "lam", "gain"]     # order of fit_quality.PARAMS / panaSKImg's fit function
AGMX_CLS = {"x-ray": 1, "x-ray diffuse": 2, "low-E": 3, "muon": 4, "worm": 5, "alpha": 6, "big deposit": 7,
            "artifact-row": 8, "artifact-column": 9, "extended": 10, "glow": 11}      # = agmx_root.CLS_CODES
AGMX_REGIME = {"sparse": 0, "crowded": 1, "unmodelled": 2, "serial": 3, "line": 4}   # = agmx_root.REGIME_CODES
AGMX_CLS["few-e"] = 12            # Bayesian few-electron events (agmx_switch; not in agmx_root.CLS_CODES yet)
AGMX_REGIME["bayes"] = 5
FINDER = {"threshold": 0, "agmx": 1, "bayes": 2}
MIN_AGMX_ROWS = 0                 # no padding: gmx.detection_bg works on any number of rows (fixed 2026-11-13; the
                                  # padding broke the Bayesian switch's run-pooled dark-charge map of short images)     # clustersRec 'finder'
BAYES_DEFAULTS = dict(p_event=1.0, N_post=-999.0, sigma_post=-999.0)   # non-Bayes clusters (threshold, A_GMX)
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


def _nbins(x, method, manual):
    """number of histogram bins of panaSKImg's fit_gaussian (absp.py): sturge, freedman, default or manual"""
    span = float(x.max() - x.min())
    if method == "sturge":
        return int(1 + np.ceil(np.log2(x.size)))
    if method == "freedman":
        iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
        return int(np.ceil(span / (2 * iqr / x.size ** (1 / 3)))) if iqr > 0 else 10
    if method == "manual":
        return int(manual)
    return int(0.5 * span) if span > 5 else 10


_PEAK_ID = [0]


class _quiet_fd1:
    """Silence file descriptor 1 (C-level printf, e.g. TMinuit) inside a with block."""
    def __enter__(self):
        sys.stdout.flush()
        self.saved = os.dup(1)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 1)

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self.saved, 1)
        os.close(self.saved)
        os.close(self.null)


def _root_first_peak(x, nb):
    """(centre of the bin, centre of the bin before, bin width) of the first local maximum that panaSKImg's
    fit_gaussian (is_highDC) finds: the first bin where TF1::Derivative of TH1::Interpolate changes sign from + to -.
    Computed with ROOT itself: between two bins of equal content the derivative is 0 up to floating-point residue,
    whose sign decides; a Python replay of the arithmetic agrees on only ~80 % of the rows, ROOT's own call on all.
    None if there is no such bin."""
    import ROOT
    _PEAK_ID[0] += 1
    lo, hi = float(x.min()), float(x.max())
    th = ROOT.TH1F(f"native_ped_{_PEAK_ID[0]}", "", int(nb), lo, hi)
    th.SetDirectory(0)
    for q in x:
        th.Fill(float(q))
    dh = ROOT.TF1(f"native_dped_{_PEAK_ID[0]}", lambda xx, p: th.Interpolate(xx[0]), lo, hi, 0)
    prev, out = 0.0, None
    for i in range(1, int(th.GetEntries()) + 1):
        dv = dh.Derivative(th.GetBinCenter(i))
        if dv < 0 and prev > 0:
            out = (th.GetBinCenter(i), th.GetBinCenter(i - 1), th.GetBinWidth(1))
            break
        prev = dv
    del dh, th
    return out


def _root_gaus_fit(x, nb, p0, mu_lim, sd_lim):
    """(mu, sigma) of panaSKImg's fit_gaussian done with ROOT: TH1F of nb bins over [min, max], TF1 'gaus' from p0
    with the mu / sigma limits, Fit 'Q EM B'."""
    import ROOT
    _PEAK_ID[0] += 1
    th = ROOT.TH1F(f"native_pedfit_{_PEAK_ID[0]}", "", int(nb), float(x.min()), float(x.max()))
    th.SetDirectory(0)
    for q in x:
        th.Fill(float(q))
    ff = ROOT.TF1(f"native_gaus_{_PEAK_ID[0]}", "gaus")
    ff.SetParameters(float(p0[0]), float(p0[1]), float(p0[2]))
    ff.SetParLimits(1, float(mu_lim[0]), float(mu_lim[1]))
    ff.SetParLimits(2, float(sd_lim[0]), float(sd_lim[1]))
    with _quiet_fd1():                     # Minuit's IMPROVE ('M') prints "NEW MINIMUM FOUND" banners despite 'Q'
        th.Fit(ff, "Q EM B N")
    out = float(ff.GetParameter(1)), float(ff.GetParameter(2))
    del ff, th
    return out


def gaus_pedestal(x, nb_method="default", nb_manual=0, lowest_peak=False):
    """(mu, sigma) [ADU] of one row's pixels as panaSKImg's fit_gaussian (method gaus_fit): histogram of nb bins
    between the smallest and the largest value (ROOT: the largest value itself falls in the overflow bin), Gaussian
    chi2 fit (Neyman errors, empty bins left out, model at the bin centres; ROOT 'Q EM B'), start norm = n / nb.
    Limits: mu in the 10-90 % quantiles, sigma in 0.5-1.5 x std, start mu = median, sigma = std. lowest_peak
    (is_highDC): the first local maximum of the histogram (the derivative of the linear interpolation between bin
    centres changes sign) starts the fit instead: mu between the smallest value and that bin + 3 bins, sigma
    (peak - smallest value) / 4 in [/10, /2] - the zero-electron peak when the overscan holds charge. Falls back to
    the median / std if the histogram or the limits are degenerate."""
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    nb = _nbins(x, nb_method, nb_manual)
    sd = float(np.std(x))
    if nb < 3 or hi <= lo or sd <= 0:
        return float(np.median(x)), sd
    y, e = np.histogram(x[x < hi], nb, (lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    w = e[1] - e[0]
    q10, q90 = np.percentile(x, [10, 90])
    mu0, pk = None, None
    if lowest_peak:
        pk = _root_first_peak(x, nb)
        if pk is not None:
            ci, cprev, wr = pk
            mu0, mu_lo, mu_hi = 0.5 * (ci + cprev), lo, ci + 3 * wr
            sd, sd_lo, sd_hi = (ci - lo) / 4, (ci - lo) / 10, (ci - lo) / 2
    if mu0 is None:
        mu0, mu_lo, mu_hi = float(np.median(x)), float(q10), float(q90)
        sd_lo, sd_hi = 0.5 * sd, 1.5 * sd
    use = y > 0
    if use.sum() < 3 or not (mu_hi > mu_lo and sd_hi > sd_lo):
        return float(np.median(x)), float(np.std(x))
    if lowest_peak and pk is not None:
        # a few hundred overscan pixels in 150 bins: the chi2 has several minima, and which one panaSKImg reports is
        # where Minuit stops from this start -> the same fit with ROOT (as panaSKImg: 'Q EM B', same start and limits)
        return _root_gaus_fit(x, nb, (x.size / nb, mu0, sd), (mu_lo, mu_hi), (sd_lo, sd_hi))
    cu, yu = c[use], y[use].astype(float)
    p0 = [x.size / nb, float(np.clip(mu0, mu_lo, mu_hi)), float(np.clip(sd, sd_lo, sd_hi))]
    resid = lambda p: (yu - p[0] * np.exp(-0.5 * ((cu - p[1]) / p[2]) ** 2)) / np.sqrt(yu)
    try:
        res = optimize.least_squares(resid, p0, bounds=([-np.inf, mu_lo, sd_lo], [np.inf, mu_hi, sd_hi]),
                                     x_scale="jac", max_nfev=2000)
        return float(res.x[1]), float(res.x[2])
    except ValueError:
        return float(np.median(x)), float(np.std(x))


def pedestal(raw, active, ovs, cfg, ext):
    """Per-row pedestal and sigma [ADU] with panaSKImg's PedestalSubtractionProcess settings cfg (axis 'row'):
    region = overscan (in_overscan and >= 3 overscan columns) or active region; pixels more than n_sigma_to_mask x
    MAD (unscaled) from the region's median are masked on the side(s) cut_right_tail / cut_left_tail (none if
    n_sigma_to_mask <= 0; 100 when the overscan is too small and it was < 0); then per row: median or mean (sigma =
    std, or MAD with use_mad), or the Gaussian fit (gaus_fit; is_highDC[ext] -> lowest-peak start).
    Returns (mu_row, sigma_row, description)."""
    method = str(cfg.get("method", "gaus_fit")).lower()
    method = "gaus_fit" if method in ("gauss_fit", "gaussian_fit", "gaus_fit") else method
    nsig = float(cfg.get("n_sigma_to_mask", -1))
    in_ovs = bool(cfg.get("in_overscan", True))
    if in_ovs and ovs.any(axis=0).sum() < 3:
        in_ovs = False
        if nsig < 0:
            nsig = 100.0
    region = ovs if in_ovs else active
    high = cfg.get("is_highDC", False)
    high = bool(high.get(ext, False)) if isinstance(high, dict) else bool(high)
    if method == "median" and not cfg.get("use_mad", True) and bool(cfg.get("cut_right_tail", True)) \
            and not bool(cfg.get("cut_left_tail", True)):
        mu, sig = pedestal_rows(raw, region, nsig)          # the validated fast path (run_WADERs' usual case)
        return mu, sig, f"median of the {'overscan' if in_ovs else 'active region'} per row"
    keep = region.copy()
    if nsig > 0:
        v = raw[region]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        dev = raw - med
        if cfg.get("cut_right_tail", True):
            keep &= dev <= nsig * mad
        if cfg.get("cut_left_tail", True):
            keep &= dev >= -nsig * mad
    mu = np.full(raw.shape[0], np.nan)
    sig = np.full(raw.shape[0], np.nan)
    for k in range(raw.shape[0]):
        x = raw[k][keep[k]]
        if x.size < 2:
            continue
        if method == "gaus_fit":
            mu[k], sig[k] = gaus_pedestal(x, str(cfg.get("hist_nbins_method", "default")).lower(),
                                          cfg.get("hist_nbins", 0), high)
        else:
            m = float(np.median(x) if method == "median" else np.mean(x))
            mu[k] = m
            sig[k] = float(np.median(np.abs(x - m)) if cfg.get("use_mad", True) else np.std(x))
    where = "overscan" if in_ovs else "active region"
    desc = {"gaus_fit": f"Gaussian fit per row ({cfg.get('hist_nbins_method', 'default')} bins"
                        + (", lowest-peak start" if high else "") + f") in the {where}"}.get(method, f"{method} of the {where} per row")
    return mu, sig, desc


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
    out = {"json": {}, "r": {}}                 # also each CCD's JSON and measurement copy (for --tune)
    for mode, peaks in (("gain", "all"), ("dc", args.peaks)):
        rs = copy.deepcopy(results)
        a = copy.copy(args)
        a.peaks = peaks
        A.choose_peaks(rs, a)
        out[mode], out["json"][mode], out["r"][mode] = {}, {}, {}
        for r in rs:
            r["fit"] = A.fit_settings(r)
            j = json.loads(A.REFERENCE_JSON)
            A.set_dark_current_fit(j, [r])
            out[mode][r["ext"]] = ext_settings(j["process"]["FitDarkCurrentProcess"], r["ext"])
            out["json"][mode][r["ext"]], out["r"][mode][r["ext"]] = j, r
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
    ho = r.get("high_occupancy") or {}
    if ho.get("resolved") and mimic is None:      # > A.LAM_HIGH e-/pix: one fit in its own window around lambda
        s = A.high_occupancy_fit_settings(r)
        # the gain is held at the comb phase: the histogram is a mixture (the dark charge changes over the CCD), and
        # a free gain absorbs that (benchmark: +2 % at the window edge, the comb phase within 0.1 % of the truth)
        fit = fit_dark_current(ps, fit_mask(ps.shape, active, s), s, start, gain_fixed=ho["gain"])
        fit["gain"] = (ho["gain"], float("nan"))
        return fit, (f"high occupancy ({ho['median_e']:.1f} e-/pix median): gain held at the comb phase {ho['gain']:.4f} "
                     f"ADU/e- (coherence {ho['R']:.2f}); noise and dark charge fitted with the peaks lambda +- 4 sqrt(lambda)"), \
            {}, (ho["gain"], None)
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


def fit_limits(fit):
    """parameters of a dark-current fit sitting at (within 0.1 % of the range of) a limit of its settings"""
    s, out = fit["settings"], []
    for p, lo, hi in (("lam", "dc_min", "dc_max"), ("gain", "gain_min", "gain_max"), ("sigma", "sigma_min", "sigma_max"),
                      ("mu", "mu_min", "mu_max")):
        v, a, b = fit[p][0], s[lo], s[hi]
        if b - a <= 1e-4 * max(abs(b), 1e-9):
            continue                                 # held parameter
        if min(abs(v - a), abs(v - b)) < 1e-3 * (b - a):
            out.append(f"{p} = {v:.4g} at its limit [{a:.4g}, {b:.4g}]")
    return out


def _at_limit(fit, s):
    """autoconfig.at_limit on a native fit: mu at a limit or sigma at its upper limit."""
    mu, sg = fit["mu"][0], fit["sigma"][0]
    return abs(mu - s["mu_min"]) < 1e-3 or abs(mu - s["mu_max"]) < 1e-3 or abs(sg - s["sigma_max"]) < 1e-3 * s["sigma_max"]


def tune_ccd(raw, active, ovs, ext, r, fs, args):
    """--tune for one resolved CCD, natively (autoconfig tune_pedestal + refine_peaks, with native pedestals and fits
    in place of panaSKImg runs; same candidates, score and window rules):
      1. pedestal candidates: autoconfig's choice ('chosen'), the per-row 'median', and the other of Gaussian fit /
         lowest-peak finder; each is fitted with the CCD's dark-current settings and ranked by
         fit_quality.quality's total relative error (+1 for mu or sigma at a limit);
      2. with the best pedestal a wide fit (-2.5 to --max-peaks + 0.5 gains) gives the histogram in which the zero
         peak and the Poisson-like peaks are found (autoconfig.peaks_in_histogram); the mu limits and window are
         centred on them (autoconfig.apply_peak_window; --peaks auto keeps the module-wide number of peaks);
      3. the all-peaks (gain) settings follow (gain_dc_split.sync_from_tuned).
    Returns (pedestal settings, {'gain': {ext: settings}, 'dc': {ext: settings}}, report lines)."""
    import contextlib
    import io
    import fit_quality as FQ
    lines = []
    start = dict(gain=r["gain"], sigma=r.get("sigma_raw", r["sigma"]), lam=r.get("lam01") or r.get("lam") or 0.1)
    ped0 = autoconfig_pedestal(r)
    cands = {"chosen": {}, "median": {"method": "median", "n_sigma_to_mask": 1.5}}
    if ped0.get("is_highDC"):
        cands["gausfit"] = {"is_highDC": {k: False for k in ped0["is_highDC"]}, "hist_nbins_method": "manual",
                            "hist_nbins": 100000, "n_sigma_to_mask": 1.5}
    else:
        cands["highDC"] = {"hist_nbins_method": "manual", "hist_nbins": 150, "n_sigma_to_mask": -1,
                           "is_highDC": {ext: True}}
    s = fs["dc"][ext]
    table = []
    for name, upd in cands.items():
        ped = copy.deepcopy(ped0)
        ped.update(upd)
        if name == "median":
            ped.pop("is_highDC", None)
        mu_row, _, _ = pedestal(raw, active, ovs, ped, ext)
        ps = raw - np.nan_to_num(mu_row, nan=0.0)[:, None]
        try:
            fit = fit_dark_current(ps, fit_mask(ps.shape, active, s), s, start)
            q = FQ.quality(fit, r.get("lam01"))
            sc = q["score"] + (1.0 if _at_limit(fit, s) else 0.0)
        except Exception as e:                    # a candidate that cannot be fitted is not chosen
            q, sc = {"error": str(e)}, math.inf
        table.append((sc, name, ped, ps, q))
    table.sort(key=lambda v: v[0])
    lines.append("pedestal tuning (score = total relative fit error, lower is better):")
    for sc, name, _, _, q in table:
        lines.append(f"  {name:8s} " + ("failed" if not np.isfinite(sc) else
                     f"score {sc:7.3f}  chi2 {q['chi2ndf']:.1f} DC {q['dc']:.3f}+-{q['dc_err_s']:.3f} ({q['dc_bias'] or 0:+.3f})"))
    best_sc, best, ped_best, ps, _ = table[0]
    lines.append(f"  -> using '{best}' pedestal settings")
    # ---- window refinement on the histogram of a wide fit
    g = r["gain"]
    scan = dict(s, x_min=round(-2.5 * g, 2), x_max=round((args.max_peaks + 0.5) * g, 2),
                n_peaks=min(args.max_peaks + 3, 20))
    mask = fit_mask(ps.shape, active, scan)
    data = ps[mask]
    nb = int(abs(scan["x_max"] - scan["x_min"]) / scan["binning_size"])
    y, e = np.histogram(data[(data > scan["x_min"]) & (data < scan["x_max"])], nb, (scan["x_min"], scan["x_max"]))
    hp = A.peaks_in_histogram(0.5 * (e[1:] + e[:-1]), y, g, r["lam"], args.max_peaks, args.poisson_tol)
    lines.append(f"peak refinement: zero peak at {hp['zero']:+.2f} ADU, "
                 + " ".join(f"{v:.2f}" for v in hp["ratios"][:8]) + f"  -> {hp['k']} Poisson-like peaks")
    rd = copy.deepcopy(fs["r"]["dc"][ext])
    a = copy.copy(args)
    if args.peaks == "auto" and rd.get("k_fit"):
        a.peaks = str(rd["k_fit"])                 # keep the module-wide number of peaks
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jd = A.apply_peak_window(fs["json"]["dc"][ext], [rd], {ext: hp}, a)
    lines += [l for l in buf.getvalue().splitlines() if l.strip()]
    jd["process"]["PedestalSubtractionProcess"] = ped_best
    jg = copy.deepcopy(fs["json"]["gain"][ext])
    GDS.sync_from_tuned(jg, jd, [ext])
    return ped_best, {"gain": {ext: ext_settings(jg["process"]["FitDarkCurrentProcess"], ext)},
                      "dc": {ext: ext_settings(jd["process"]["FitDarkCurrentProcess"], ext)}}, lines


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


def flooded_background(e_img, region, min_pix=20):
    """Flooding level and charge width of a flooded CCD (no resolved peaks, many e-/pix): per row the median of the
    region's pixels (the level grows along the readout) and their robust spread 1.4826 x MAD. Returns (level per row
    [e-], width [e-] = median of the rows' spreads, median level [e-]) or None if no row has min_pix pixels. The width
    is that of the charge distribution (Poisson + read noise + gain spread), which a pedestal sigma from the overscan
    (read noise only) does not see: 10 x the read noise would be far below the charge fluctuations of a flooded CCD
    and join the whole CCD into one cluster."""
    lev = np.full(e_img.shape[0], np.nan)
    wid = np.full(e_img.shape[0], np.nan)
    for k in range(e_img.shape[0]):
        v = e_img[k][region[k]]
        v = v[np.isfinite(v)]
        if v.size >= min_pix:
            m = np.median(v)
            lev[k], wid[k] = m, 1.4826 * np.median(np.abs(v - m))
    ok = np.isfinite(lev) & np.isfinite(wid) & (wid > 0)
    if not ok.any():
        return None
    lev[~np.isfinite(lev)] = np.median(lev[ok])
    return lev, float(np.median(wid[ok])), float(np.median(lev[ok]))


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


def agmx_valid(shape, active, first_row, binning=(1, 1)):
    """A_GMX's active pixels as agmx_root.valid_mask / comm_clustering: rows from first_row + 3 to the second-last-but-one,
    columns of the active region without a 2-pixel border. The margins are unbinned pixels: with on-chip binning
    (rows, columns) they are ceil((first_row + 3) / b_y) and ceil(2 / b_y) rows and ceil(2 / b_x) columns (a 6-row
    100x1 image keeps rows 1-4)."""
    by, bx = (max(int(b), 1) for b in binning)
    r0, r1, c = math.ceil((first_row + 3) / by), math.ceil(2 / by), math.ceil(2 / bx)
    cols = np.flatnonzero(active.any(axis=0))
    v = np.zeros(shape, bool)
    v[r0:shape[0] - r1, cols[0] + c:cols[-1] + 1 - c] = True
    return v & active


def agmx_clusters(Q, valid, crowded="auto", crowded_shapes="on", fewe="off", fewe_lam=None, fewe_RN=None, bg=None,
                  sigma=None, binning=(1, 1), fewe_threshold=None, fewe_nm=None):
    """A_GMX (adaptive_cluster engine 'gmx') on the image Q [e-]; returns (rows, cols, charge [e-], offsets, extras
    {agmx_*: array, finder, p_event, N_post, sigma_post}, info). Clusters without pixels or a finite energy are
    dropped (as agmx_root). fewe / fewe_lam / fewe_RN: the Bayesian few-electron switch (agmx_switch via
    adaptive_cluster); bg / sigma: background map and read noise already measured (characterise.decide_ccd)."""
    import adaptive as AD
    AD.CROWDED, AD.CROWDED_SHAPES = crowded, crowded_shapes     # also what the crowded-mode record reports
    H0 = Q.shape[0]
    pad = max(0, MIN_AGMX_ROWS - H0)
    if pad:
        # gmx.detection_bg (np.convolve 'same' over 61 rows) fails on images with < 61 rows: pad with invalid rows
        # (no data, no clusters) and give A_GMX a background map of the padded image
        Q = np.vstack([Q, np.zeros((pad, Q.shape[1]))])
        valid = np.vstack([valid, np.zeros((pad, valid.shape[1]), bool)])
        bg = None
    Qv = np.where(valid, Q, 0.0)
    if bg is None:
        try:
            bg = AD.AM.background_map(Qv, valid)
        except Exception:
            bg = None
    cl, info = AD.adaptive_cluster(Qv, valid, bg, sigma=sigma, engine="gmx", return_pixels=True, return_info=True,
                                   crowded=crowded, crowded_shapes=crowded_shapes, fewe=fewe, fewe_lam=fewe_lam,
                                   fewe_RN=fewe_RN, binning=binning, fewe_threshold=fewe_threshold, fewe_nm=fewe_nm)
    if pad:
        info["padded_rows"] = int(pad)
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
    fin = lambda v, d: float(v) if v is not None and np.isfinite(v) else d
    bay = [c.get("finder") == "bayes" for c in cl]
    ex["finder"] = np.array([FINDER["bayes"] if b else FINDER["agmx"] for b in bay], np.int64)
    ex["p_event"] = np.array([fin(c.get("p_ev"), -999.0) if b else BAYES_DEFAULTS["p_event"] for c, b in zip(cl, bay)])
    ex["N_post"] = np.array([fin(c.get("N_mean"), -999.0) if b else BAYES_DEFAULTS["N_post"] for c, b in zip(cl, bay)])
    ex["sigma_post"] = np.array([fin(c.get("sigma_post"), -999.0) if b else BAYES_DEFAULTS["sigma_post"]
                                 for c, b in zip(cl, bay)])
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
    # finder of each cluster and the Bayesian posterior quantities (defaults for threshold / A_GMX clusters)
    B.setdefault("finder", np.full(n, FINDER["threshold"], np.int64))
    for k, v in BAYES_DEFAULTS.items():
        B.setdefault(k, np.full(n, v))
    INT = {"Npix", "ccd", "cluster_id", "has_seed", "nfile", "agmx_cls", "agmx_regime", "finder"}
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
                  keep_arrays=False, gain_override=None, outputs=None, all_ext=True, plots=False, tune=False,
                  fewe="off", fewe_pool=None, mask_cols=None, characterisation=None):
    """Process every CCD of one image (read once) and write OUT/panaSKImg_clustersRec_<stem>_ext<n>.root.

    outputs: [(key, clusterer, folder, curve)] -> one cluster file per CCD for each entry, all from the same pedestal
    and fits (run_WADERs: root/, root_gaincorr/, root_agmx/, ...); [] = fits only, no cluster files. Default (None):
    from curve / clusterer / out_dir below. all_ext: also process CCDs autoconfig flags as unresolved outliers
    (flooded among good ones; run_WADERs skips them unless --all-ext). plots: write panaSKImg's --save-plots PDFs
    (DC fit, baseline and sigma per row; native_plots.py) to out_dir. tune: --tune on every resolved CCD (tune_ccd:
    pedestal method by total fit error, window on the fitted histogram).
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
    panaSKImg's own gain the cluster quantities can be compared to the last bit; with clusterer 'auto' the gain of a
    CCD without a measurable comb, from --gain / --gain-file / siblings, see characterise.py).
    clusterer 'auto': per CCD after calibration and before clustering, characterise.decide_ccd chooses 'unmodelled'
    (A_GMX's hysteresis on the calibrated image), A_GMX crowded mode, A_GMX normal with fewe='auto', or the threshold
    clusterer (decision recorded in the summary, in native_config and in the TNamed 'characterisation').
    fewe: A_GMX's Bayesian few-electron switch for clusterer 'agmx' ('off' | 'auto' | 'on'); fewe_pool: {ext number:
    (lam_map, R_N)} run-pooled dark charge (agmx_switch.prepare_run); mask_cols: {ext number: [columns]} taken out of
    the active region (pedestal, fit, clusters; e.g. the first-column artefact); characterisation: the image's record
    from characterise.py (plans per CCD; written as TNamed 'characterisation').
    Returns a summary dict (per CCD: fit, thresholds, n clusters, timings)."""
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
        if clusterer not in ("threshold", "agmx", "both", "auto"):
            raise ValueError(f"clusterer {clusterer!r}: use 'threshold', 'agmx', 'both' or 'auto'")
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
        rec = dict(ext=ext, resolved=bool(r.get("resolved")), lam01=r.get("lam01") if r.get("resolved") else None)
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
        clus_region = active                        # where clusters are searched (pedestal and fit keep `active`)
        if mask_cols and mask_cols.get(n):
            clus_region = active.copy()
            clus_region[:, list(mask_cols[n])] = False
            rec["masked_columns"] = list(mask_cols[n])
        # ---- 2. pedestal: the settings autoconfig --per-ext chooses for this CCD (or the panaSKImg JSON's);
        #         overscan if they ask for it and there is one (>= 3 columns)
        ped_cfg = dict(mim["ped"] if mim else autoconfig_pedestal(r))
        if r.get("ped_active"):                   # autoconfig's fallback: pedestal from the active region
            ped_cfg["in_overscan"] = False
        fs_ccd = fs
        if tune and r["resolved"] and not mim:
            ped_cfg, fs_ccd, rec["tune"] = tune_ccd(raw, active, ovs, ext, r, fs, args)
        mu_row, sig_row, ped_desc = pedestal(raw, active, ovs, ped_cfg, ext)
        use_ovs = "overscan" in ped_desc
        high_dc = ped_cfg.get("is_highDC")
        high_dc = bool(high_dc.get(ext)) if isinstance(high_dc, dict) else bool(high_dc)
        rec["pedestal"] = dict(method=ped_cfg.get("method", "median"), lowest_peak=high_dc,
                               region="overscan" if use_ovs else "active", description=ped_desc)
        ps = raw - np.nan_to_num(mu_row, nan=0.0)[:, None]
        t_ped = time.time()
        # ---- 3. gain, noise, dark current
        fit, note, fits, held = gain_noise_dc(ps, active, ext, r, fs_ccd, mim["fit"] if mim else None)
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
                   pedestal_region="overscan" if use_ovs else "active", held=held, fit_settings=fit["settings"],
                   fits=dict({k: {p: v[p] for p in FIT_PARAMS + ["chi2", "ndf"]} for k, v in fits.items()},
                             final={p: fit[p] for p in FIT_PARAMS + ["chi2", "ndf"]}))
        ho = r.get("high_occupancy") or {}
        if ho.get("resolved"):                   # high occupancy: per row from the integer charges (truncated Poisson)
            rec["high_occupancy"] = {k: v for k, v in ho.items() if k != "lam_rows"}
            rec["lam_rows"] = [A.truncated_poisson_lam(np.round(e_img[k][active[k]]), ho["kmax"]) if active[k].any()
                               else None for k in range(e_img.shape[0])]
        elif r["resolved"]:                      # dark charge per row from the fraction of 0e- pixels (Poisson)
            with np.errstate(divide="ignore", invalid="ignore"):
                f0 = np.array([np.mean(e_img[k][active[k]] < 0.5) if active[k].any() else np.nan
                               for k in range(e_img.shape[0])])
                rec["lam_rows"] = (-np.log(f0)).tolist()
        lim = fit_limits(fit)
        if lim:
            rec["fit_at_limit"] = lim             # e.g. ultra-low dark charge: the comb fit cannot measure lambda
        info = info_branches(n, mu_row, sig_row, fit, ccd, nfile, h0)
        if plots and write:
            write_plots(os.path.join(out_dir, stem), ext, ps, active, fit, r, mu_row, sig_row, h0, args)
        t_cal = time.time()
        # ---- 5. / 6. one cluster file per output (clusterer x g(n) curve), from the same calibrated image
        thr = thresholds_for(thresholds, r, sigma0_e, mim["thr"] if mim else None)
        # flooded CCD (characterise.plan_ccd: no resolved peaks, > FLOOD_E e-/pix): the threshold clusterer works on the
        # image minus the flooding level and with the width of the CHARGE distribution, not the read noise
        e_thr, flood_bg = e_img, None
        plan_ccd = ((characterisation or {}).get("ccds", {}).get(ext, {}) or {}).get("plan") or {}
        if plan_ccd.get("flooded") and not mim:
            fl = flooded_background(e_img, clus_region)
            if fl is not None:
                flood_bg, width, level = fl
                e_thr = e_img - flood_bg[:, None]
                if not isinstance(thresholds, (tuple, list)):
                    thr = (10 * width, 30 * width, width, 30.0,
                           f"flooded: charge width {width:.3f} e- (per-row MAD of the active pixels; read noise "
                           f"{sigma0_e:.3f} e-) -> 10 / 30 x width above the flooding level ({level:.2f} e-/pix median, "
                           f"subtracted per row)")
                else:
                    thr = thr[:4] + (thr[4] + f", above the flooding level ({level:.2f} e-/pix median, subtracted per "
                                              f"row)",)
                rec["flooded"] = dict(level_e=level, width_e=width, read_noise_e=sigma0_e)
        rec["thresholds"] = dict(pixel_e=thr[0], seed_e=thr[1], sigma_seed_e=thr[2], n_seed=thr[3], note=thr[4])
        sigma_seed_eV = thr[2] * PANASKIMG_EV          # panaSKImg's unit: sigma [e-] x e2eV
        models = {}                                    # curve -> (g(n) model or None, description)
        thr_clusters = None                            # the threshold clusters do not depend on the curve
        for key, cl, odir, cv in outputs:
            if cv not in models:
                model, cdesc = None, None
                if cv is not None:
                    cpath = curve_file(cv, stem)
                    model = GVC.load_model(cpath, n) if cpath else None
                    cdesc = cpath if model else f"{cpath}: no reliable g(n) for {ext} -> fitted gain kept"
                models[cv] = (model, cdesc)
            model, cdesc = models[cv]
            eV = ev_per_e if ev_per_e else (pair_energy_eV() if model is not None else PANASKIMG_EV)
            if key == outputs[0][0]:                   # as before: the first output's curve / eV in the record
                rec.update(curve=cdesc, ev_per_e=eV)
            t_c0 = time.time()
            cl_eff, crowded_eff, fewe_eff, bg_k, sig_k, Q, dec = cl, crowded, fewe, None, None, None, None
            if cl in ("agmx", "auto"):
                adu_img = e_img * g_cal if panaskimg_json else ps      # mimic: as comm_clustering (calibrated x gain)
                Q = (GVC.electrons_with_anchor(adu_img.ravel(), model, g_cal, anchor).reshape(ps.shape)
                     if model is not None else e_img)
                valid = agmx_valid(ps.shape, active, first_row,
                                   (int(h0.get("NPBIN", 1) or 1), int(h0.get("NSBIN", 1) or 1))) & clus_region
                if not valid.any() and active.any():
                    # strips (e.g. 4 x 6294): A_GMX's first-rows (+3) and bottom (2) cuts leave no row -> all rows from
                    # id_row_start (logged)
                    valid = np.zeros(ps.shape, bool)
                    valid[min(first_row, ps.shape[0] - 1):, :] = True
                    valid &= clus_region
                    rec.setdefault("fallbacks", []).append(
                        f"A_GMX valid region empty ({ps.shape[0]} rows): rows {first_row}.. used without the +3 / -2 "
                        f"row cuts")
            if cl == "auto":
                import characterise as CH
                plan = ((characterisation or {}).get("ccds", {}).get(ext, {}) or {}).get("plan") or {}
                dec, bg_k, sig_k = CH.decide_ccd(Q, valid, plan)
                cl_eff = "threshold" if dec["chain"] == "threshold" else "agmx"
                crowded_eff, fewe_eff = dec["crowded"], dec["fewe"]
                if dec["chain"] == "unmodelled":
                    crowded_eff = "off"
                pool = (fewe_pool or {}).get(n)
                dec["fewe_pooled"] = bool(pool is not None and fewe_eff != "off")
                if plan.get("fewe_pooled") and pool is None:
                    dec["fallbacks"].append("run-pooled dark charge planned but not available -> single-image estimate")
                rec.setdefault("decision", dec)       # the first output's decision (root/); each output's chain
                                                      # is in rec[key]["chain"] (a g(n) image can decide otherwise)
            if cl_eff == "threshold":
                if thr_clusters is None:
                    thr_clusters = threshold_clusters(e_thr, clus_region, thr[0], thr[1])
                rows, cols, off = thr_clusters
                e_pix = e_thr[rows, cols]
                E_unc = e_pix * PANASKIMG_EV / 1000.0         # (e- x eV/e-) / 1000, the order of pysimdamicm
                if model is not None:
                    # ADU of each pixel, then A^-1. Mimic mode: back from the pixel energy as GainNonlinearityProcess
                    # (keV -> e- -> ADU with the rounded gain, last bits as panaSKImg); else the pedestal-subtracted ADU
                    adu = (E_unc * 1000.0) / PANASKIMG_EV * g_cal if panaskimg_json else ps[rows, cols]
                    if flood_bg is not None:
                        adu = adu - flood_bg[rows] * g_cal
                    E_fin = GVC.electrons_with_anchor(adu, model, g_cal, anchor) * eV / 1000.0
                else:
                    E_fin = e_pix * eV / 1000.0
                E_props = E_unc if props_from == "uncorrected" else E_fin
                extras, named = None, {}
                E_seed = E_unc
            elif cl_eff == "agmx":
                pool = (fewe_pool or {}).get(n) if fewe_eff != "off" else None
                rows, cols, q, off, extras, ainfo = agmx_clusters(Q, valid, crowded_eff, crowded_shapes, fewe_eff,
                                                                  pool[0] if pool else None,
                                                                  pool[1] if pool else None, bg_k, sig_k,
                                                                  binning=(int(h0.get("NPBIN", 1) or 1), int(h0.get("NSBIN", 1) or 1)),
                                                                  fewe_threshold=pool[2] if pool and len(pool) > 2 else None,
                                                                  fewe_nm=pool[3] if pool and len(pool) > 3 else None)
                if dec is not None:
                    if ainfo.get("padded_rows"):
                        dec["fallbacks"].append(f"image padded with {ainfo['padded_rows']} invalid rows for A_GMX "
                                                f"(gmx.detection_bg needs >= {MIN_AGMX_ROWS} rows)")
                    dec["agmx_info"] = dict(regime=ainfo.get("regime"), core_fill=ainfo.get("core_fill"),
                                            crowded_mode=ainfo.get("crowded_mode"), fewe=_plain(ainfo.get("fewe")),
                                            noise=ainfo.get("noise"), lam=ainfo.get("lam"))
                    if dec["chain"] == "unmodelled" and ainfo.get("regime") != "unmodelled":
                        dec["fallbacks"].append(f"front end said 'unmodelled', A_GMX ran regime {ainfo.get('regime')}")
                    if dec["chain"] in ("agmx", "crowded") and ainfo.get("regime") == "unmodelled":
                        dec["fallbacks"].append("A_GMX switched to 'unmodelled' itself")
                E_fin = q * eV / 1000.0
                E_props = E_seed = E_fin
                named = {"agmx_config": json.dumps({ext: dict(
                    method="A_GMX (superWADERs clustering/adaptive.py engine 'gmx')",
                    gain_corrected=model is not None, gain_anchor=anchor if model is not None else None,
                    parameters=dict(crowded=crowded_eff, crowded_shapes=crowded_shapes, fewe=fewe_eff,
                                    fewe_pooled=pool is not None), cls_codes=AGMX_CLS, finder_codes=FINDER,
                    regime_codes=AGMX_REGIME, not_computed=NOT_COMPUTED, crowded_mode=crowded_record(ainfo, crowded_eff),
                    fewe=_plain(ainfo.get("fewe")),
                    n_clusters=int(off.size - 1), eV_per_e=eV,
                    recommended_selection="agmx_blend < 0.02 and agmx_ped < 10 (and fSTD/sigma cut of the analysis)")})}
            else:
                raise ValueError(f"clusterer {cl!r}: use 'threshold', 'agmx', 'both' or 'auto'")
            t_c1 = time.time()
            ncl = off.size - 1
            table = cluster_table(rows, cols, off, E_props, E_fin, E_seed, ps, raw, sigma_seed_eV, thr[3], ccd, nfile,
                                  times, extras) if ncl else {}
            t_c2 = time.time()
            cfg = dict(version=VERSION, image=path, ext=ext, clusterer=cl if cl != "auto" else f"auto -> {dec['chain']}",
                       thresholds=rec["thresholds"],
                       gain_note=note, curve=cdesc, anchor=anchor if model is not None else None,
                       eV_per_e=eV, props_from=props_from, pedestal=rec["pedestal"],
                       panaskimg_json=panaskimg_json, fit_settings=fit["settings"])
            named = dict(named, native_config=json.dumps(cfg))
            if characterisation is not None or dec is not None:
                named["characterisation"] = characterisation_text(characterisation, ext, dec, rec)
            fname = os.path.join(odir, f"panaSKImg_clustersRec_{stem}_ext{n}.root")
            if write:
                write_ccd(fname, table, ncl, info, config_branches(fit, thr, eV), named)
                summary["files"].append(fname)
            t_c3 = time.time()
            rec[key] = dict(n_clusters=int(ncl), file=fname if write else None, curve=cdesc,
                            chain=dec["chain"] if dec is not None else cl,
                            t_cluster=t_c1 - t_c0, t_props=t_c2 - t_c1, t_write=t_c3 - t_c2)
            if ncl and "finder" in table:
                fv = table["finder"][1]
                rec[key]["n_by_finder"] = {k: int((fv == v).sum()) for k, v in FINDER.items()}
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


def curve_file(curve, stem):
    """the g(n) file for an image: curve is a gain_vs_charge JSON (used for every image) or a gain_vs_charge/ folder
    (run_superWADERs --gain-correction), where 'image_<stem>' (per-image curve) is preferred if curve ends with
    ':per-image', then the curve of the image's gain period ('module_<ACM id>_period<k>', the one whose image list holds
    this image), else 'module_<ACM id>' (pooled curve)"""
    import glob
    if curve is None or os.path.isfile(curve):
        return curve
    path, per_image = (curve[:-len(":per-image")], True) if curve.endswith(":per-image") else (curve, False)
    acm = naming.acm_id(stem)
    if per_image and os.path.isfile(os.path.join(path, f"image_{stem}_gain_vs_charge.json")):
        return os.path.join(path, f"image_{stem}_gain_vs_charge.json")
    for f in sorted(glob.glob(os.path.join(path, f"module_{acm or stem}_period*_gain_vs_charge.json"))):
        with open(f) as fh:
            if any(naming.image_stem(i) == stem for i in json.load(fh).get("images", [])):
                return f
    f = os.path.join(path, f"module_{acm or stem}_gain_vs_charge.json")
    return f if os.path.isfile(f) else None


def _plain(o):
    """JSON-safe copy (numpy scalars / arrays -> Python, large arrays dropped)"""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist()) if o.size <= 64 else f"array{o.shape}"
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def characterisation_text(char, ext, dec, rec):
    """JSON of the TNamed 'characterisation': the file record (characterise.py) with only this CCD, plus the
    post-calibration decision (clusterer 'auto') and the fitted gain / noise / dark charge."""
    out = {}
    if char:
        out = {k: v for k, v in char.items() if k != "ccds"}
        c = (char.get("ccds") or {}).get(ext, {})
        out["ccd"] = {k: v for k, v in c.items() if not k.startswith("_")}
    if dec is not None:
        out["decision"] = {k: v for k, v in dec.items()}
    out["fitted"] = dict(gain=rec.get("gain"), gain_note=rec.get("gain_note"), noise_e=rec.get("noise_e"),
                         dc_e=rec.get("dc_e"), pedestal=rec.get("pedestal"), thresholds=rec.get("thresholds"),
                         masked_columns=rec.get("masked_columns"))
    return json.dumps(_plain(out))


def write_plots(base, ext, ps, active, fit, r, mu_row, sig_row, h0, args):
    """panaSKImg's --save-plots PDFs of one CCD: base_FitDarkCurrentProcess_DCfit_<ext>.pdf (the fit),
    base_PedestalSubtractionProcess_Baseline_<ext>.pdf and _Sigma_<ext>.pdf (per row)."""
    import native_plots as NPL
    s = fit["settings"]
    data = ps[fit_mask(ps.shape, active, s)]
    t0, t1, _ = readout_times(h0)
    info = ([f"t_read={t1 - t0:.1f}s"] if t0 is not None else []) + [
        f"t_exp={float(h0.get('EXPOSURE', 0.0)):.1f}s", f"bin: {h0.get('NPBIN', 1)}x{h0.get('NSBIN', 1)}",
        f"frame: {h0.get('NROW', ps.shape[0])}x{h0.get('NCOL', ps.shape[1])}x{h0.get('NDCM', h0.get('NSKIPS', 1))}",
        f"CCD-{ext}"]
    NPL.dc_fit(f"{base}_FitDarkCurrentProcess_DCfit_{ext}.pdf", data, fit,
               NPL.plot_range(s, r, getattr(args, "plot_peaks", 5)), comb_model, s["n_peaks"], info)
    NPL.per_row(f"{base}_PedestalSubtractionProcess_Baseline_{ext}.pdf", mu_row, "Baseline (ADUs)")
    NPL.per_row(f"{base}_PedestalSubtractionProcess_Sigma_{ext}.pdf", sig_row, "Sigma (ADUs)")


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


def auto_tasks(imgs, out, kw, gain=None, gain_file=None, skip_start=0, log=print, pool=True):
    """the characterisation front end for a list of images (characterise.py): records, gain resolution, groups,
    run-pooled dark charge (agmx_switch.prepare_run) where planned (pool=True; only the 'auto' chain uses it), and one
    process_image task per image. Returns (tasks [(path, out, kw)], records, skipped [(file, reason)])."""
    import characterise as CH
    t0 = time.time()
    recs, groups = CH.characterise_many(imgs, out, gain, gain_file, kw.get("autoconfig_args", ()), kw.get("exts"),
                                        kw.get("curve"), skip_start)
    log(f"characterised {len(recs)} file(s), {len(groups)} run setting(s) ({time.time() - t0:.1f} s)")
    tasks, skipped = [], []
    for rec in recs:
        if not rec.get("process"):
            skipped.append((rec["file"], rec.get("reason")))
            continue
        k = dict(kw)
        gargs, over = CH.gain_args_for(rec)
        aargs = list(kw.get("autoconfig_args", ())) + gargs
        if any(c["channel"]["value"] == "quiet" for c in rec["ccds"].values()):
            aargs.append("--keep-dead")
        k["autoconfig_args"] = aargs
        if over:
            k["gain_override"] = over
        k["mask_cols"] = {int(e[3:]): [c["first_column"]["value"]["column"]] for e, c in rec["ccds"].items()
                          if (c.get("first_column") or {}).get("value") and c["first_column"]["value"]["significant"]}
        k["characterisation"] = CH.public(rec)
        tasks.append([rec["process_path"], out, k])
    # ---- run-pooled dark charge per run setting and CCD (images of equal shape)
    by_group = {}
    for t, rec in zip(tasks, [r for r in recs if r.get("process")]):
        by_group.setdefault(rec["group"]["value"], []).append((t, rec))
    for gkey, members in by_group.items():
        pooled_ext = sorted({int(e[3:]) for _, rec in members for e, c in rec["ccds"].items()
                             if (c.get("plan") or {}).get("fewe_pooled")})
        if not pool or len(members) < 2 or not pooled_ext:
            continue
        t1 = time.time()
        arrays = []
        for t, rec in members:
            k = dict(t[2], outputs=[], write=False, keep_arrays=True, verbose=False, plots=False, save_fits=False)
            k.pop("clusterer", None)
            s = process_image(t[0], out, **k)
            arrays.append(s)
        for n in pooled_ext:
            ext = f"EXT{n}"
            imgs_q, ok = [], True
            for (t, rec), s in zip(members, arrays):
                c = s["ccds"].get(ext, {})
                if "e_img" not in c:
                    ok = False
                    break
                geo = rec["geometry"]["value"]
                act = np.zeros(c["e_img"].shape, bool)
                act[:, geo["first_active_col"]:geo["first_overscan_col"]] = True
                valid = agmx_valid(c["e_img"].shape, act, geo["id_row_start"],
                                   (int(geo.get("npbin", 1) or 1), int(geo.get("nsbin", 1) or 1)))
                for col in (t[2].get("mask_cols") or {}).get(n, []):
                    valid[:, col] = False
                imgs_q.append((np.where(valid, c["e_img"], 0.0), valid))
            if not ok or len({q.shape for q, _ in imgs_q}) != 1:
                for t, rec in members:
                    rec["ccds"].get(ext, {}).setdefault("fallbacks", []).append(
                        "run-pooled dark charge not computed (CCD missing or shapes differ) -> single-image estimate")
                continue
            import agmx_switch as SW
            SW.BINNING = (int(geo.get("npbin", 1) or 1), int(geo.get("nsbin", 1) or 1))
            try:
                # the full run-pooled mode (switch_validation/VALIDATION.md): the run's P(event) cut and noise model
                # too, not each image's own (a per-image budget / noise fit loses single-pixel / single-bin events)
                run = SW.prepare_run(imgs_q, full=True)
                lam_maps, R_N = run["lam_maps"], run["R_N"]
            except Exception as e:                       # noqa: BLE001  logged, single-image estimate instead
                for t, rec in members:
                    rec["ccds"][ext].setdefault("fallbacks", []).append(
                        f"agmx_switch.prepare_run failed ({type(e).__name__}: {e}) -> single-image estimate")
                continue
            for i, (t, rec) in enumerate(members):
                t[2].setdefault("fewe_pool", {})[n] = (lam_maps[i], R_N, run["threshold"], run["nm"])
                c = rec["ccds"][ext]
                c["plan"]["fewe_pool"] = dict(n_images=len(members), lam_median=float(np.median(
                    np.broadcast_to(lam_maps[i], imgs_q[i][1].shape)[imgs_q[i][1]])), R=float(np.sum(R_N)),
                    curve_note=("pooled on the linear-gain image (g(n) curve not applied to the pooling)"
                                if kw.get("curve") else None))
                t[2]["characterisation"] = CH.public(rec)
        log(f"  run pooling {len(members)} images, CCDs {pooled_ext} ({time.time() - t1:.1f} s)")
    return [tuple(t) for t in tasks], recs, skipped


def write_characterisation(out, recs, results):
    """OUT/characterisation/<stem>.json: the front end's record of each image (characterise.py) with, per CCD, the
    result of process_image (chain decision, gain, noise, dark charge, thresholds, cluster counts, fallbacks)."""
    import characterise as CH
    d = os.path.join(out, "characterisation")
    os.makedirs(d, exist_ok=True)
    by_stem = {s["stem"]: s for s in results}
    for r in recs:
        r = CH.public(r)
        s = by_stem.get(r.get("stem"))
        for ext, c in (s or {}).get("ccds", {}).items():
            if ext in r.get("ccds", {}):
                r["ccds"][ext]["result"] = {k: v for k, v in c.items() if k in (
                    "decision", "gain", "gain_note", "noise_e", "dc_e", "skipped", "masked_columns", "thresholds", "flooded",
                    "fit_at_limit", "fallbacks", "auto", "pedestal") or isinstance(v, dict) and "n_clusters" in v}
        with open(os.path.join(d, f"{r['stem']}.json"), "w") as fh:
            fh.write(CH.dumps(r))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="raw skip-averaged images (.fz/.fits/.fits.bz2) or .txt lists")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="images in parallel (processes; keep <= 2)")
    ap.add_argument("--clusterer", choices=("auto", "threshold", "agmx", "both"), default="auto",
                    help="auto (default): the characterisation front end (characterise.py) chooses the chain per CCD")
    ap.add_argument("--gain", default=None, help="gain [ADU/e-] for CCDs whose comb cannot be measured: G or "
                    "EXT1=G1,EXT2=G2 (used through the characterisation; always flagged in its record)")
    ap.add_argument("--gain-file", default=None, help="JSON with gains for CCDs without a comb: {EXTn: G}, "
                    "{acm: {EXTn: G}} or a native_summary.json")
    ap.add_argument("--fewe", default="off", help="A_GMX Bayesian few-electron switch for --clusterer agmx: off|auto|on")
    ap.add_argument("--skip-start", type=int, default=0, help="raw skip files: first skip averaged (0-based)")
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
    ap.add_argument("--plots", action="store_true", help="also write panaSKImg's --save-plots PDFs (DC fit, baseline, sigma)")
    ap.add_argument("--tune", action="store_true", help="pedestal method by total fit error and window on the fitted "
                    "histogram, per resolved CCD (autoconfig --tune, natively)")
    ap.add_argument("--autoconfig-args", default="", help='extra autoconfig options, e.g. "--ccd-cols 6200"')
    a = ap.parse_args(argv)
    thr = a.thresholds
    if "," in thr:
        thr = tuple(float(v) for v in thr.split(","))
    kw = dict(curve=a.curve, clusterer=a.clusterer, thresholds=thr, anchor=a.anchor, ev_per_e=a.ev_per_e,
              exts=a.ext, autoconfig_args=a.autoconfig_args.split(), panaskimg_json=a.panaskimg_json,
              crowded=a.crowded, save_fits=a.save_fits, plots=a.plots, tune=a.tune, fewe=a.fewe)
    imgs = expand_inputs(a.images)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    results, failed = [], []
    recs, skipped = None, []
    if a.clusterer == "auto" or a.gain or a.gain_file or a.skip_start:
        tasks, recs, skipped = auto_tasks(imgs, a.out, kw, a.gain, a.gain_file, a.skip_start)
        for f, why in skipped:
            print(f"NOT PROCESSED {f}: {why}", file=sys.stderr)
    else:
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
    if recs is not None:
        write_characterisation(a.out, recs, results)
    for f in failed:
        print("FAILED", f, file=sys.stderr)
    print(f"native_pipeline: {len(results)} image(s), {len(files)} files in {a.out} ({time.time() - t0:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
