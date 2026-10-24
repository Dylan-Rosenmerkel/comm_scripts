#!/usr/bin/env python3
"""Build a panaSKImg JSON tuned to one skipper-CCD image.

The script reads the raw (skip-averaged) FITS file, measures every extension
(CCD) directly from its pixels, and writes a copy of a reference JSON (--reference) in which
the geometry, PedestalSubtractionProcess, FitDarkCurrentProcess (and optionally
ClusterFinder) settings are chosen so that panaSKImg's fit of the pixel charge
distribution converges on the right noise, gain and dark current.

Where it sits in the pipeline
-----------------------------
    raw image -> autoconfig_panaSKImg.py (this file: one JSON per image or per CCD)
              -> gain_dc_split.py (chooses the gain) -> panaSKImg -> cluster ROOT files
              -> optional gain-vs-charge correction (gain_vs_charge.py) -> MakeComposite
bin/run_WADERs drives the whole chain. panaSKImg (WADERs package, pysimdamicm) fits
the pixel charge histogram with Gaussian peaks (one per electron number, spacing =
gain, width = read noise) weighted by a Poisson distribution (mean = dark current).
That fit only converges if its window, starting values and limits are sensible for
the image at hand; this script measures them from the pixels first.

Entry points
------------
* command line: main() (options in parse_args(); run with -h).
* outputs: the JSON given by -o plus <json>_measured.json (write_measured), or with
  --per-ext a folder with one JSON per CCD, <image>_measured.json and run_<image>.sh.

How to read this file
---------------------
    REFERENCE_JSON              built-in starting config
    find_panaskimg, ...         locating panaSKImg / its optional features
    Small statistics helpers    robust_std, trimmed_mean, clip_tails, compress_ranges
    Electron-peak ("comb")      comb_offset, align_rows, find_gain, refine_gain, ...
    Column and channel checks   find_bad_columns, is_dead_channel
    Per-extension analysis      measure_gain, held_gain_fallback, poisson_peaks,
                                measure_dark_current, split_active_overscan,
                                mask_bad_columns, measure_resolved, measure_unresolved,
                                analyse_extension (uses all of these)
    FitDarkCurrentProcess       fit_settings, choose_peaks, shared_fit_window
    Writing the JSON            set_pedestal, set_dark_current_fit, set_cluster_threshold,
                                skip_clusters, set_plot_range, write_measured,
                                write_per_ext, tune_per_ext, write_run_script
    --tune                      run_candidate, score, tune_pedestal, histogram_peaks,
                                locate_histogram_peaks, apply_peak_window, refine_peaks
    Main                        parse_args, geometry, gain_priors, print_report,
                                analyse_all_extensions, write_single_json, main

Units: charges are in ADU (after pedestal subtraction) unless the name says e-;
gain is ADU/e-; dark current lambda is e-/pixel (mean electrons per pixel over the
exposure + readout). Extensions are named "EXT1", "EXT2", ... as in panaSKImg.

Typical use
-----------
    python autoconfig_panaSKImg.py avg_img_....fits -o auto.json
    python autoconfig_panaSKImg.py avg_img_....fits -o auto.json --tune
    python autoconfig_panaSKImg.py avg_img_....fits -o auto.json --cluster-threshold-e 2 5
    python autoconfig_panaSKImg.py avg_img_....fits -o auto.json --reference my_reference.json
    panaSKImg --json auto.json avg_img_....fits -o ./analysis --acm --save-plots

What is measured
----------------
* Geometry: the active/overscan column boundary from the header
  (ccd_cols = PHYSCOL - NCOLPRE + PRESCAN, in unbinned pixels as panaSKImg expects).
* Channels without a CCD: the active region looks exactly like the overscan (same
  noise, no dark current, no particle tracks). They are removed from 'extensions'.
  (The overscan = columns read after the last real pixel: empty charge packets, so
  they hold only the electronic baseline ("pedestal") and read noise.)
* Bad columns (electronic transient in the first columns, hot columns): columns
  whose mean charge stands out from their neighbours. Masked in the DC fit only.
* Gain and noise, when single-electron peaks are resolved: each row is shifted
  onto a common comb of peaks (removing sub-electron pedestal jitter between rows),
  the peak spacing (= gain) comes from the autocorrelation of the charge histogram
  and is refined as the slope of peak position vs electron number, and the noise
  is the robust width of every pixel's distance to its nearest peak.
* Gain and noise, when peaks are NOT resolved: the read noise comes from the
  overscan and the gain is held at a prior (reference-JSON CalibrationProcess gain or
  --gain), since it cannot be fitted.
* Dark current per row: with resolved peaks the electron count of every pixel is
  known, so lambda(row) = -ln P(0 e-) (for a Poisson of mean lambda the fraction of
  empty pixels is exp(-lambda)). For long readouts it grows with row, so the
  DC fit only uses the first rows (until lambda > --lam-max).
* Pedestal: if most of the rows used in the fit have overscan pixels on the 1 e-
  (or higher) peak, a plain pedestal lands on the wrong peak; the lowest-peak finder
  (is_highDC) is switched on instead. Otherwise the per-row median of the overscan
  is used (closer to the zero peak than a Gaussian fit to ~150 pixels per row).

* Number of peaks to fit (--peaks auto): the fit model is ONE Poisson distribution.
  Extra charge (source/particle clusters, dark current varying across the CCD)
  over-populates the higher peaks; fitting them leaves the model below the data and
  biases lambda upward. lambda is taken from the 0e-/1e- peaks and peaks are added
  one at a time while they still match the Poisson prediction (poisson_peaks).

From these the fit window, number of peaks, binning and all parameter limits of
FitDarkCurrentProcess are set per extension. Optional extras:

* --tune: (1) run panaSKImg's pedestal + DC fit with each candidate pedestal setting
  and keep the one with the smallest total fit error (fit_quality.py: errors scaled
  by sqrt(chi2/ndf), plus the dark-current bias against the 0e-/1e- estimate), with
  a penalty for any parameter stuck at a limit;
  (2) iterative refinement on panaSKImg's own histogram: locate the zero-electron
  peak, re-select the number of Poisson-like peaks, centre the fit window on them,
  and run a final check fit (refine_peaks). Needs PyROOT.
* --cluster-threshold-e PIX SEED: make ClusterFinder use fixed thresholds in
  electrons instead of multiples of the (pedestal-dependent) per-row noise.
* --per-ext: one JSON per CCD (own pedestal method, fit window, binning) in the
  folder given by -o, plus run_<image>.sh that runs panaSKImg per CCD in parallel
  (--acm --ext N), giving one cluster file per CCD (..._extN.root).
* The DC fit plot shows the charge up to --plot-peaks e- (x_max_plot) although only
  the selected peaks are fitted; flooded/unresolved outlier CCDs are skipped by
  ClusterFinder (--cluster-all to keep them).

Requires numpy and astropy (and panaSKImg for --tune).
"""
import argparse
import bz2
import copy
import glob
import json
import math
import os
import re
import io
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
from astropy.io import fits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference JSON: the starting point every generated config is copied from. Only the
# settings this script measures are changed; everything else (input conventions,
# CalibrationProcess, ClusterFinder, BuildClusterMask, ...) comes from here. It is
# built in, so the script does not depend on any file location; use --reference to
# start from your own JSON instead.
REFERENCE_JSON = r"""
{
  "input": {
    "image": {
      "ACM_multi_extension": true,
      "extensions": [
        1,
        2,
        3,
        4
      ],
      "ext_header": 0,
      "skip_image": false,
      "axis_to_compress": 1,
      "correct_leach_bug": false,
      "correct_polarity": false,
      "id_skip_start": 0,
      "id_skip_end": -1,
      "id_row_start": 2,
      "id_row_end": -1,
      "id_col_start": 0,
      "id_col_end": -1,
      "n_rows_prescan": 0,
      "n_rows_overscan": 0,
      "ccd_rows": 250,
      "n_cols_prescan": 0,
      "ccd_cols": 854,
      "n_amp": 1,
      "save_image": true
    },
    "datetime": {
      "exposure_start": "DATEINI",
      "readout_start": "DATEINI",
      "readout_end": "DATEEND"
    },
    "scp": {},
    "convention": {
      "ampdir": "VCKDIRN",
      "itgtime": "TINTEGR",
      "Nskips": "NDCM",
      "Ncols": "NCOL",
      "Nrows": "NROW",
      "Npbin": "NPBIN",
      "Nsbin": "NSBIN",
      "ampl": "AMPL",
      "exposure_time": "EXPOSURE",
      "read_time": "MREAD"
    }
  },
  "process": {
    "sequence": "CompressSkipperProcess;PedestalSubtractionProcess;FitDarkCurrentProcess;CalibrationProcess;ClusterFinder;BuildClusterMask",
    "CompressSkipperProcess": {
      "func_to_compress": [
        "mean",
        "std"
      ],
      "save_image": false
    },
    "PedestalSubtractionProcess": {
      "image": "mean_compressed",
      "method": "gaus_fit",
      "__DEBUG__": false,
      "in_overscan": true,
      "use_mad": false,
      "axis": "row",
      "n_sigma_win_fit": 2,
      "n_sigma_to_mask": 1.5,
      "hist_nbins_method": "manual",
      "hist_nbins": 100000,
      "cut_left_tail": false,
      "cut_right_tail": true
    },
    "TwoDGaussProcess": {
      "__display__": false,
      "col_start": 0,
      "row_start": 0,
      "q_min_ADU": -10,
      "q_max_ADU": 30,
      "bin_size": 0.4,
      "do_fit": true,
      "dc_L": 0.005,
      "dc_U_range": [
        0.0005,
        0.01
      ],
      "dc_U": 0.003,
      "sig_L": 3,
      "sig_L_range": [
        0.5,
        5.0
      ],
      "sig_U": 2.0,
      "theta": 46.0,
      "cal_L": 7.0,
      "cal_L_range": [
        7.0,
        9.5
      ],
      "cal_U": 7.0,
      "cal_U_range": [
        3,
        9.5
      ],
      "fit_opt": "QR LEM"
    },
    "CorrectElectronicColumnTransient": {
      "image": "mean_compressed_pedestal_subtracted",
      "col_start": 1,
      "col_end": 10,
      "subtract_median": true,
      "n_exp": 1,
      "fit_options": "QSMER"
    },
    "CalibrationProcess": {
      "image": "mean_compressed_pedestal_subtracted",
      "gain": {
        "EXT1": 1.26,
        "EXT2": 1.32,
        "EXT3": 1.34,
        "EXT4": 1.31
      },
      "from_dc_fit": true
    },
    "FitCalibrationConstant": {
      "image": "mean_compressed_pedestal_subtracted",
      "n_peaks": 3,
      "calibration": {
        "U": 10.0,
        "L": 10.0
      },
      "n_sigma_win_fit": 3
    },
    "FitDarkCurrentProcess": {
      "image": "mean_compressed_pedestal_subtracted",
      "__DEBUG__": false,
      "__verbose__": false,
      "fit_options": "Q",
      "ratio_plot": false,
      "rows_to_mask": {
        "EXT1": [
          0,
          1,
          2
        ],
        "EXT2": [
          0,
          1,
          2
        ],
        "EXT3": [
          0,
          1,
          2
        ],
        "EXT4": [
          0,
          1,
          2
        ]
      },
      "cols_to_mask": {
        "EXT1": [],
        "EXT2": [],
        "EXT3": [],
        "EXT4": []
      },
      "mask_clusters": false,
      "do_calibration": true,
      "calibration": 1.0,
      "n_peaks": 5,
      "binning_size": 0.01,
      "sigma_max": 0.5,
      "dc_min": 0.0001,
      "dc_max": 10.0,
      "gain_min": 0.8,
      "gain_max": 3.0,
      "x_min": -5,
      "x_max": 10
    },
    "FitDarkCurrentPerRow": {
      "image": "mean_compressed_pedestal_subtracted",
      "row_start": 0,
      "row_end": -1,
      "row_step": 10,
      "calibration": 1.0,
      "rows_to_mask": [],
      "cols_to_mask": [],
      "n_peaks": 2,
      "binning_size": 0.05,
      "gain_min": 0.5,
      "gain_max": 2.5,
      "x_min": -3,
      "x_max": 5
    },
    "FitDarkCurrentPerCol": {
      "image": "mean_compressed_pedestal_subtracted",
      "col_start": 0,
      "col_end": -1,
      "col_step": 10,
      "rows_to_mask": [],
      "cols_to_mask": [],
      "calibration": 1.0,
      "n_peaks": 2,
      "binning_size": 0.05,
      "gain_min": 0.8,
      "gain_max": 1.8,
      "x_min": -3,
      "x_max": 5
    },
    "ChargeLossPlot": {
      "skip_id_list": [
        0,
        1,
        2,
        45
      ],
      "skip_id_baseline": -1,
      "histequ": false,
      "gray_palette": false
    },
    "ChargeLossSkewnessProcess": {
      "use_overscan": false,
      "id_skip_reference": 1,
      "id_skip_start": 2,
      "id_skip_end": -1,
      "skip_step": 20,
      "kcl_threshold": 3.2,
      "kcl_n_sig": 8,
      "display": true
    },
    "FFTNoisePlot": {},
    "RNvsNskipsPlot": {
      "n_skips_per_block": 50,
      "is_blank": false
    },
    "ClusterFinder": {
      "spr_threshold": {
        "EXT1": [
          null,
          10,
          30
        ],
        "EXT2": [
          null,
          10,
          30
        ],
        "EXT3": [
          null,
          10,
          30
        ],
        "EXT4": [
          null,
          10,
          30
        ]
      },
      "spr_image": "mean_compressed_pedestal_subtracted_e",
      "skip_rows": [],
      "skip_cols": [],
      "only_cls_seed": true,
      "method": 0,
      "max_nearest_neighbor": 2
    },
    "BuildClusterMask": {
      "hot_rows": [],
      "hot_cols": [],
      "cluster_size_min": 1,
      "cluster_DX_min": 1,
      "cluster_DY_min": 1,
      "cluster_E_min": 3,
      "cluster_box_E_min": 100000000,
      "N_col_pre": 1,
      "N_col_post": 5,
      "N_row_pre": 1,
      "N_row_post": 1,
      "mask_cluster_column": false
    },
    "ApplySelectionCuts": {
      "cols": [],
      "rows": []
    },
    "GaussianFitProcess": {
      "image": "mean_compressed_pedestal_subtracted",
      "in_overscan": true,
      "hist_nbins_method": "freedman",
      "skip_rows": [],
      "in_trim_mean": true,
      "n_sigma": 5
    }
  }
}
"""


def find_panaskimg():
    """Path of the panaSKImg script, or None.

    Looked up in this order: $PANASKIMG, panaSKImg on the PATH, next to the running
    python (conda environment bin/), and the bin/ folder of a pysimdamicm source checkout.
    """
    import importlib.util
    cands = [os.environ.get("PANASKIMG"), shutil.which("panaSKImg"),
             os.path.join(os.path.dirname(sys.executable), "panaSKImg")]
    try:
        spec = importlib.util.find_spec("pysimdamicm")
        if spec and spec.submodule_search_locations:
            pkg = list(spec.submodule_search_locations)[0]
            cands.append(os.path.join(os.path.dirname(pkg), "bin", "panaSKImg"))
    except (ImportError, ValueError):
        pass
    return next((c for c in cands if c and os.path.isfile(c)), None)


def panaskimg_has_plot_range():
    """True if the installed pysimdamicm's FitDarkCurrentProcess accepts x_max_plot.

    It is a local addition; an unknown key makes panaSKImg stop with an AttributeError,
    so it is only written when supported.
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("pysimdamicm")
        pkg = list(spec.submodule_search_locations)[0]
        with open(os.path.join(pkg, "processes", "skipper_analysis.py")) as fh:
            return "x_max_plot" in fh.read()
    except (ImportError, ValueError, TypeError, OSError):
        return False


DEFAULT_PANASKIMG = find_panaskimg()

# chi2/ndf, scaled errors and dark-current bias of panaSKImg fits (same folder)
sys.path.insert(0, SCRIPT_DIR)
import fit_quality as FQ  # noqa: E402

# Number of rows at the start of the image used to measure gain and noise. These
# rows are read first, so they carry the least dark current.
N_EARLY_ROWS = 30


# =============================================================================
# Small statistics helpers
# =============================================================================
def robust_std(x):
    """Standard deviation estimated from the median absolute deviation (MAD).

    Insensitive to outliers such as particle tracks or hot pixels.
    """
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def trimmed_mean(a, axis=0, lo=10, hi=90):
    """Mean along `axis` using only values between the lo-th and hi-th percentiles."""
    p_lo, p_hi = np.percentile(a, [lo, hi], axis=axis, keepdims=True)
    m = np.where((a >= p_lo) & (a <= p_hi), a, np.nan)
    return np.nanmean(m, axis=axis)


def clip_tails(q, lo=0.5, hi=99.5):
    """Drop the most extreme values (outliers, tracks) of a 1-D array."""
    return q[(q > np.percentile(q, lo)) & (q < np.percentile(q, hi))]


def compress_ranges(idx):
    """Turn a sorted list of indices into panaSKImg mask syntax.

    Runs of 3 or more consecutive indices become [start, end) ranges, e.g.
    [0, 1, 2, 3, 7] -> [[0, 4], 7]. (panaSKImg treats a 2-element list as a range.)
    """
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        if j - i >= 2:
            out.append([idx[i], idx[j] + 1])
        else:
            out.extend(idx[i:j + 1])
        i = j + 1
    return out


# =============================================================================
# Electron-peak ("comb") measurements
#
# With good resolution the pixel charge distribution is a comb of peaks spaced by
# the gain g (ADU/e-). The phase of exp(2*pi*i*q/g) averaged over pixels tells
# where the comb sits; the distance of each pixel to its nearest tooth gives the
# noise.
# =============================================================================
def comb_offset(q, g):
    """Position (ADU, modulo g) of the comb of peaks in the values q.

    Each value is mapped to a phase 2*pi*q/g on the unit circle; values on the comb
    teeth all point the same way, so the angle of the mean vector is the comb
    position. q: 1-D array (ADU), g: peak spacing (ADU/e-). Returns a float in
    (-g/2, g/2].
    """
    return g * np.angle(np.exp(2j * np.pi * q / g).mean()) / (2 * np.pi)


def align_rows(x, g):
    """Shift every row of x so that its pixels sit on a common comb of spacing g.

    Only the sub-electron part of each row's offset is removed; whole-electron
    offsets (pedestal landing on a wrong peak) are kept and handled per row later.
    Why: the baseline of each row is estimated separately and jitters by a fraction
    of an electron, which smears the peaks of the combined histogram.
    x: 2-D array (rows, cols) in ADU. Returns an array of the same shape.
    """
    ph = np.angle(np.exp(2j * np.pi * x / g).mean(axis=1, keepdims=True))
    return x - g * ph / (2 * np.pi)


def find_gain(q, gmin, gmax, prominence=0.03):
    """Spacing of the electron peaks in q (ADU), or None if there is no comb.

    The autocorrelation of the charge histogram has a secondary maximum at a lag
    equal to the peak spacing (and its multiples); for a single Gaussian it just
    falls monotonically. The first prominent secondary maximum between gmin and
    gmax is taken and then refined with refine_gain().
    q: 1-D pixel values (ADU); gmin, gmax: gain search range (ADU/e-);
    prominence: minimum rise of the maximum above the preceding dip (autocorrelation
    normalised to 1). Returns the gain (float, ADU/e-) or None.
    """
    rng = np.random.default_rng(0)
    if q.size > 300000:                       # subsample very large images (speed)
        q = rng.choice(q, 300000, replace=False)

    # fine histogram, lightly smoothed, covering at least two peak spacings
    lo, hi = np.percentile(q, [0.5, 97])
    hi = max(hi, lo + 2.2 * gmax)
    b = 0.005                                 # bin width, ADU
    h, _ = np.histogram(q, bins=int((hi - lo) / b), range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-10, 11) / 3.0) ** 2)
    h = np.convolve(h, k / k.sum(), mode="same")

    # autocorrelation via FFT (zero-padded to avoid wrap-around), normalised to 1 at lag 0
    n = h.size
    f = np.fft.rfft(h - h.mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    lags = np.arange(n) * b

    # first local maximum that rises clearly above the dip before it
    i0, i1 = int(gmin / b), min(int(gmax / b), n - 2)
    for i in range(max(i0, 1), i1):
        if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] - ac[:i].min() > prominence:
            return refine_gain(q, lags[i])[0]
    return None


def refine_gain(q, g, n_iter=3, min_count=200):
    """Refine the gain as the slope of peak position vs electron number.

    Each pixel is assigned to its nearest peak; the median position of every peak
    with at least min_count pixels is fitted with a straight line. Returns
    (gain, number of peaks used). With fewer than two usable peaks the input gain
    is returned unchanged.
    """
    npk = 0
    for _ in range(n_iter):
        off = comb_offset(q, g)
        n = np.round((q - off) / g)               # electron number of each pixel
        d = (q - off) - g * n                     # distance to the nearest peak
        ns, cs, ws = [], [], []
        for k in np.unique(n):
            sel = (n == k) & (np.abs(d) < 0.3 * g)   # core of this peak only
            if sel.sum() >= min_count:
                ns.append(k)
                cs.append(np.median(q[sel]))
                ws.append(sel.sum())
        npk = len(ns)
        if npk < 2:
            break
        g = float(np.polyfit(ns, cs, 1, w=np.sqrt(ws))[0])
    return g, npk


def comb_residual_sigma(q, g):
    """Comb offset and peak width: robust std of each pixel's distance to its nearest peak.

    With resolved peaks this width is the read noise (ADU): every peak has the same
    Gaussian width, so folding all pixels onto one tooth measures it with all the
    statistics of the image. Returns (offset, sigma), both in ADU.
    """
    off = comb_offset(q, g)
    d = (q - off) - g * np.round((q - off) / g)
    return off, robust_std(d)


def zero_peak_sigma(q):
    """(position, width) of the lowest dominant peak, from its left side.

    Used when peaks are not resolved. Only the left side is used so the width is
    not inflated by the dark-current tail on the right.
    q: 1-D pixel values (ADU). Returns (mode, sigma) in ADU; sigma is the MAD of the
    pixels below the mode, scaled to a Gaussian standard deviation.
    """
    # the zero-electron peak sits in the lower part of the distribution; limiting
    # the range keeps the bins fine even with a long high-charge tail
    lo, hi = np.percentile(q, [0.5, 60])
    hi = hi + (hi - lo)
    h, e = np.histogram(q, bins=400, range=(lo, hi))
    hs = np.convolve(h, np.ones(5) / 5, mode="same")
    mode = 0.5 * (e[np.argmax(hs)] + e[np.argmax(hs) + 1])
    left = q[q <= mode]
    return mode, 1.4826 * np.median(mode - left)


# =============================================================================
# Column and channel checks
# =============================================================================
def find_bad_columns(x, min_dev, nsig=6.0, max_frac=0.03):
    """Columns whose mean charge stands out from a running median of their neighbours.

    The threshold is at least nsig robust standard deviations and min_dev (ADU),
    and is raised until at most max_frac of the columns are flagged, so only the
    clearly anomalous ones (transients, hot columns) are masked.
    x: 2-D active-region array (rows, cols), baseline-subtracted, in ADU;
    min_dev: minimum deviation (ADU) for a column to count as bad.
    Returns (list of column indices relative to x, threshold used in ADU).
    """
    xr = x - np.median(x, axis=1, keepdims=True)     # remove per-row level
    cm = trimmed_mean(xr, axis=0).ravel()             # robust mean of every column
    n = cm.size
    # wide window (101 columns) so a transient spanning the first ~10-20 columns stands out
    w = 50
    pad = np.pad(cm, w, mode="reflect")
    smooth = np.array([np.median(pad[i:i + 2 * w + 1]) for i in range(n)])
    dev = cm - smooth
    thr = max(nsig * robust_std(dev), min_dev)
    while np.sum(np.abs(dev) > thr) > max_frac * n:
        thr *= 1.25
    bad = np.where(np.abs(dev) > thr)[0]
    return [int(c) for c in bad], thr


def is_dead_channel(x, xo, gain_prior, args):
    """True if the active region x looks exactly like the overscan xo (no CCD signal).

    A working CCD always shows some charge: dark current and particle tracks.
    An unconnected/broken channel shows only read noise, identical to the overscan.
    Criteria: (1) almost no pixels far above the noise, (2) same noise in active
    region and overscan, (3) no offset between them.
    x, xo: baseline-subtracted active region and overscan (2-D, ADU); gain_prior:
    ADU/e- (or None); args: uses --dead-frac and --keep-dead (never dead with it).
    Returns (dead?, overscan noise in ADU, fraction of pixels with charge).
    """
    # noise from differences of neighbouring pixels: insensitive to slow baseline drifts
    s_ovs = robust_std(np.diff(xo, axis=1).ravel() / np.sqrt(2))
    s_act = robust_std(np.diff(x, axis=1).ravel() / np.sqrt(2))
    hi = max(10 * s_ovs, 5 * (gain_prior or 1.0))   # "clearly charged": >10 sigma and >~5 e-
    frac_hi = float(np.mean(x > hi))
    dead = (not args.keep_dead and s_ovs > 0 and frac_hi < args.dead_frac
            and abs(s_act / s_ovs - 1) < 0.15 and abs(np.median(x)) < 0.3 * s_ovs)
    return dead, s_ovs, frac_hi


# =============================================================================
# Per-extension analysis
# =============================================================================
def measure_gain(x, ext, args, gain_prior, early):
    """Decide whether peaks are resolved and measure the gain.

    "Resolved" means the single-electron peaks are separate: noise/gain below
    --max-sig-over-gain (0.3 by default). Then the gain can be measured from the
    peak spacing and each pixel's electron count is known. Otherwise the histogram
    is one broadened bump and the gain has to come from a prior.

    x: baseline-subtracted active region (2-D, ADU); ext: name for messages;
    gain_prior: ADU/e- (or None); early: slice of the low-dark-current rows used.
    Returns a dict with: g (gain or None), resolved, gain_held (gain could not be
    measured and is held at the prior), credible (a comb was found even if too broad
    to fit), sig_zero (width of the lowest peak, ADU).
    Prints a message when the comb spacing disagrees with the prior or the gain is held.
    """
    # align rows with the prior gain first, so the combined distribution is sharp
    xa = align_rows(x, gain_prior) if gain_prior else x
    q = clip_tails(xa[early].ravel())

    g = find_gain(q, args.gain_min, args.gain_max)
    if g is not None and gain_prior:
        # the autocorrelation can lock onto 2x or 3x the spacing when the peaks are
        # broad; take the harmonic closest to the prior and reject if still far off
        g = min((g / k for k in (1, 2, 3)), key=lambda v: abs(v - gain_prior))
        if abs(g - gain_prior) > args.prior_tol * gain_prior:
            print(f"  {ext}: comb spacing {g:.3f} ADU is not within {args.prior_tol:.0%} "
                  f"of the gain prior {gain_prior:.3f}; treating peaks as unresolved")
            g = None
        else:
            g = refine_gain(q, g)[0]

    # Special case: very sharp peaks but almost no charge (very low dark current),
    # so there is no visible comb. Try to measure the spacing from the few 1e-, 2e-
    # pixels, starting from the prior; if that fails, hold the gain at the prior.
    gain_held = False
    _, sig_zero = zero_peak_sigma(q)
    if g is None and gain_prior and sig_zero / gain_prior < args.max_sig_over_gain:
        g_try, npk = refine_gain(q, gain_prior, min_count=50)
        if npk >= 2 and abs(g_try - gain_prior) <= args.prior_tol * gain_prior:
            g = g_try
        else:
            g, gain_held = gain_prior, True
            print(f"  {ext}: peaks are sharp but almost no pixels above 0e-; "
                  f"holding the gain at the prior {gain_prior:.3f}")

    # resolved = peak width small compared with the spacing
    if g is not None:
        xa = align_rows(x, g)
        q = clip_tails(xa[early].ravel())
        _, sigma = comb_residual_sigma(q, g)
        resolved = sigma / g < args.max_sig_over_gain
    else:
        resolved = False

    # A comb that is too broad to fit the gain still measures it better than the
    # prior, but only if the zero-electron peak is narrower than the spacing
    # (otherwise the "comb" is just noise).
    credible = g is not None and sig_zero < 0.6 * g
    return dict(g=g, resolved=resolved, gain_held=gain_held, credible=credible, sig_zero=sig_zero)


def held_gain_fallback(act, gain_prior, args):
    """Last check for a CCD the comb search called unresolved (used by analyse_extension).

    Very long readouts (e.g. 10000 skips) can hide the comb: the noise is higher, the
    readout leaves charge in later rows, and bright events spill into the overscan so
    their rows get a wrong pedestal. If, with the pedestal taken from the active region
    (row median) and only the clean early rows, the zero-electron peak is still narrower
    than max_sig_over_gain x gain, the 0e-/1e- peaks can be fitted with the gain held at
    the prior (--gain, or the reference CalibrationProcess gain).
    act: raw active-region pixels (2-D, ADU, no baseline subtracted); gain_prior:
    ADU/e-; args: uses --max-sig-over-gain.
    Returns (pedestal-subtracted active pixels (2-D, ADU), zero-peak width (ADU),
    number of clean rows) or None if the CCD does not qualify.
    """
    g = gain_prior
    x = act - np.median(act, axis=1, keepdims=True)
    # clean early rows (within the first ~30, where the long-readout noise and charge are
    # smallest; later rows are noisier and can push a good CCD over the limit): the zero-electron peak still holds
    # >= 20% of the row, and both halves of the row have the same zero level (after a
    # very bright pixel the baseline can step for the rest of the row)
    rows, half = [], x.shape[1] // 2
    for r in range(3, min(N_EARLY_ROWS + 3, x.shape[0])):
        if np.mean(np.abs(x[r]) < 0.5 * g) < 0.2:
            continue
        lo, hi = x[r, :half], x[r, half:]
        lo, hi = lo[np.abs(lo) < 0.5 * g], hi[np.abs(hi) < 0.5 * g]
        if lo.size < 20 or hi.size < 20 or abs(np.median(lo) - np.median(hi)) > 0.25 * g:
            continue
        rows.append(r)
        if len(rows) >= 30:
            break
    if len(rows) < 8:
        return None
    # width of the zero-electron peak from its left side (no dark charge there), within
    # +-1.5 gain: a peak narrower than 0.3 gain lies well inside, a wide one reads >> 0.3
    q = align_rows(x[rows], g).ravel()
    q = q[np.abs(q) < 1.5 * g]
    h, e = np.histogram(q, bins=150, range=(-1.5 * g, 1.5 * g))
    hs = np.convolve(h, np.ones(5) / 5, mode="same")
    mode = 0.5 * (e[np.argmax(hs)] + e[np.argmax(hs) + 1])
    sig = 1.4826 * np.median(mode - q[q < mode]) if np.any(q < mode) else np.inf
    if not sig > 0 or sig / g >= args.max_sig_over_gain:
        return None
    return x, sig, len(rows)


def poisson_peaks(counts, max_peaks, tol=0.25):
    """How many electron peaks follow a single Poisson distribution.

    The fit model is Gaussian peaks weighted by ONE Poisson distribution. Extra
    charge that is not Poisson (particle/source clusters, dark current varying
    between rows) adds pixels to the higher peaks, the fit cannot describe them, and
    forcing it to (a) leaves the fitted peaks below the data and (b) pulls lambda up.

    Iterative selection: lambda is estimated from the 0e- and 1e- peaks only
    (lambda = N1/N0); then peaks n = 2, 3, ... are added one at a time while the
    observed count agrees with the Poisson prediction N0*lambda^n/n! within the
    larger of `tol` (relative) and 3 statistical standard deviations. The first
    peak that disagrees stops the selection.

    Why N1/N0: for a Poisson of mean lambda, P(1)/P(0) = lambda; the 0 e- and 1 e-
    peaks are the tallest and the least affected by extra charge, so this ratio is
    the cleanest estimate of the dark current (lam01).

    counts: number of pixels with 0, 1, 2, ... electrons.
    Returns (K = number of peaks to fit, at least 2; list of data/prediction ratios;
    lambda from the first two peaks).
    """
    n0, n1 = float(counts[0]), float(counts[1])
    if n0 <= 0:
        return 2, [], math.nan
    lam = n1 / n0
    ratios, k = [1.0, 1.0], 2
    for n in range(2, len(counts)):
        expected = n0 * lam ** n / math.factorial(n)
        if expected < 1:
            break                 # nothing measurable left: stop without a verdict
        ratio = counts[n] / expected
        ratios.append(ratio)
        if abs(ratio - 1) > max(tol, 3 / math.sqrt(expected)):
            break                 # first peak with non-Poisson charge
        k = n + 1
    return min(max(k, 2), max_peaks), ratios, lam


def measure_dark_current(xga, xo, off, g, nrows, args):
    """Per-row dark current and the row range to use in the fit (resolved peaks only).

    xga: active pixels (bad columns removed) aligned on the comb, xo: overscan.
    Every pixel gets an electron number; in each row the zero-electron peak is the
    lowest peak holding >= 5% of the pixels, and lambda = -ln(fraction at 0 e-),
    which is robust against tracks and hot pixels.

    Why only part of the rows: rows are read one after the other, so later rows wait
    longer in the CCD and collect more dark current. The fit model has ONE lambda,
    so it is applied to the early rows whose dark current is still low (< --lam-max
    e-/pix) and fairly uniform; the remaining rows are masked in the fit.

    off: comb offset (ADU), g: gain (ADU/e-), nrows: number of rows; args: uses
    --lam-max, --min-rows, --max-peaks, --poisson-tol.
    Returns a dict: row_end (first row NOT used), lam (mean lambda of the used rows,
    e-/pix), lam01, k_poisson, poisson_ratios (see poisson_peaks), ovs_charged_frac
    and ovs_charged_frac_fit (fractions of rows whose overscan sits on >= 1 e-).
    """
    n_act = np.round((xga - off) / g)
    n_ovs = np.round((xo - off) / g)
    lam_row = np.full(nrows, np.nan)
    ovs_row = np.full(nrows, np.nan)
    e_rows = [None] * nrows       # electrons per pixel of every row (for the peak selection)
    for r in range(nrows):
        nr = n_act[r]
        vals, cnt = np.unique(nr, return_counts=True)
        ok = vals[cnt >= max(3, 0.05 * nr.size)]
        if ok.size == 0:          # row so charged that no zero-electron peak remains
            continue
        n0 = ok.min()             # zero-electron peak of this row
        e = nr - n0               # electrons per pixel
        e_rows[r] = e
        p0 = np.sum(e == 0) / max(np.sum((e >= 0) & (e < 50)), 1)
        lam_row[r] = -math.log(max(p0, 1e-6))
        # electrons in this row's overscan, relative to the active zero
        ovs_row[r] = np.median(n_ovs[r] - n0)

    # rows used in the fit: from the start until the (5-row smoothed) lambda exceeds
    # lam_max, but at least min_rows (rows 0-2 are skipped: often a transient)
    lam_s = np.convolve(np.nan_to_num(lam_row, nan=np.inf), np.ones(5) / 5, mode="same")
    above = np.where(lam_s[3:] > args.lam_max)[0]
    row_end = int(above[0] + 3) if above.size else nrows
    row_end = max(row_end, min(args.min_rows, nrows))

    # electron-count histogram of the pixels that go into the fit
    e_fit = np.concatenate([e for e in e_rows[:row_end] if e is not None])
    counts = np.bincount(e_fit[(e_fit >= 0) & (e_fit < 50)].astype(int), minlength=50)
    k_poisson, ratios, lam01 = poisson_peaks(counts, args.max_peaks, args.poisson_tol)
    return dict(
        row_end=row_end,
        lam=float(np.nanmean(lam_row[:row_end])),
        # dark charge over ALL rows (it grows along a long readout): median and 90th percentile,
        # used by --cluster-threshold-e auto
        lam_all_median=float(np.nanmedian(lam_row)),
        lam_all_p90=float(np.nanpercentile(lam_row, 90)),
        lam01=lam01,                  # lambda from the 0e-/1e- peaks only
        k_poisson=k_poisson,          # number of peaks consistent with a single Poisson
        poisson_ratios=ratios,        # data / Poisson prediction for peaks 0, 1, 2, ...
        # fraction of rows whose overscan sits on >= 1 e-: the pedestal would land on
        # the wrong peak there unless the lowest-peak finder is used
        ovs_charged_frac=float(np.nanmean(ovs_row >= 1)),
        # same, only for the rows that go into the DC fit (decides the pedestal method)
        ovs_charged_frac_fit=float(np.nanmean(ovs_row[:row_end] >= 1)),
    )


def split_active_overscan(data, cfg):
    """Cut one extension into active region and overscan and subtract a rough baseline.

    The pedestal (electronic baseline) drifts from row to row, so a per-row baseline
    is subtracted: the median of that row's overscan, similar to what panaSKImg
    starts from. Without an overscan (windowed readout) the row median of the active
    region is used instead.
    data: 2-D extension image (rows, cols) in ADU; cfg: analysis geometry from
    geometry() (id_row_start, n_cols_prescan, ccd_cols = first overscan column).
    Returns (act: raw active pixels, x: baseline-subtracted active pixels,
    xo: baseline-subtracted overscan (= x without overscan), nrows, a0: first
    active column, no_ovs: True if there are fewer than 3 overscan columns).
    """
    r0 = cfg["id_row_start"]
    d = data[r0:, :].astype(float)
    nrows = d.shape[0]
    a0, a1 = cfg["n_cols_prescan"], cfg["ccd_cols"]
    act, ovs = d[:, a0:a1], d[:, a1:]
    if ovs.shape[1] >= 3:
        base = np.median(ovs, axis=1, keepdims=True)
        x, xo = act - base, ovs - base
    else:
        # no overscan (windowed readout): baseline = row median of the active region,
        # which sits on the zero-electron peak while the dark current is < ~0.7 e-/pix;
        # the active pixels also stand in for the overscan in the checks below
        base = np.median(act, axis=1, keepdims=True)
        x = act - base
        xo = x
    no_ovs = ovs.shape[1] < 3
    return act, x, xo, nrows, a0, no_ovs


def mask_bad_columns(x, g, resolved, early, a0):
    """Find the bad columns of x and return them with the good columns only.

    The minimum deviation for a bad column is ~0.3 e- (resolved peaks) or half the
    zero-peak width (unresolved). x: baseline-subtracted active region (2-D, ADU);
    g: gain (ADU/e-, used only if resolved); early: slice of early rows; a0: first
    active column, added so the returned indices are image columns.
    Returns (bad columns as image column indices, x without the bad columns).
    """
    if resolved:
        min_dev = 0.3 * g
    else:
        min_dev = 0.5 * zero_peak_sigma(x[early].ravel())[1]
    bad_cols, _ = find_bad_columns(x, min_dev)
    bad_abs = [c + a0 for c in bad_cols]
    good = np.ones(x.shape[1], bool)
    good[bad_cols] = False
    xg = x[:, good]
    return bad_abs, xg


def measure_resolved(xg, xo, g, nrows, early, args):
    """Noise and dark current of a CCD with resolved peaks.

    Two noise values: sigma, the intrinsic peak width (rows aligned on the comb),
    and sigma_raw, the width as panaSKImg's fit will see it (rows not aligned:
    panaSKImg subtracts one overscan pedestal per row, which keeps the row-to-row
    jitter). Then the per-row dark current and the rows to use (measure_dark_current).
    xg: active region without bad columns, xo: overscan (2-D, ADU); g: gain.
    Returns a dict (gain, sigma, sigma_raw in ADU, plus measure_dark_current's keys).
    """
    xga = align_rows(xg, g)
    off, sigma = comb_residual_sigma(clip_tails(xga[early].ravel()), g)
    _, sigma_raw = comb_residual_sigma(clip_tails(xg[early].ravel()), g)
    out = dict(gain=g, sigma=sigma, sigma_raw=max(sigma_raw, sigma))
    out.update(measure_dark_current(xga, xo, off, g, nrows, args))
    return out


def measure_unresolved(xg, xo, g, nrows, early):
    """Noise and mean charge of a CCD whose peaks are not resolved.

    The width of the zero peak in the early rows is broadened by the unresolved
    charge (sigma_raw); the read noise proper comes from the overscan, which has no
    dark current (sigma = the smaller of the two). The dark current is estimated as
    the median pixel charge divided by the gain (a rough mean charge per pixel).
    xg: active region without bad columns, xo: overscan (2-D, ADU); g: gain used
    (measured comb spacing if credible, else the prior).
    Returns a dict (gain, sigma, sigma_raw in ADU, row_end = nrows, lam in e-/pix,
    ovs_charged_frac(_fit) = 0).
    """
    _, sigma = zero_peak_sigma(xg[early].ravel())
    sig_ovs = robust_std(xo - np.median(xo, axis=1, keepdims=True))
    sig_noise = min(sig_ovs, sigma) if sig_ovs > 0 else sigma
    # mean charge per pixel in electrons (baseline from the overscan)
    lam_e = max(float(np.median(xg)) / g, 0.0)
    return dict(gain=g, sigma=sig_noise, sigma_raw=sigma, row_end=nrows, lam=lam_e,
                ovs_charged_frac=0.0, ovs_charged_frac_fit=0.0)


def analyse_extension(data, ext, cfg, args, gain_prior):
    """Measure one extension (CCD) from its raw pixels.

    Steps: split active/overscan and subtract a per-row baseline; drop the channel if
    it has no CCD (is_dead_channel); measure the gain and decide whether the peaks are
    resolved (measure_gain), with the held-gain fallback for unresolved CCDs; mask
    bad columns; then measure noise and dark current (measure_resolved or
    measure_unresolved).

    data: 2-D image of this extension (ADU); ext: name, e.g. "EXT1"; cfg: analysis
    geometry from geometry(); gain_prior: ADU/e-.
    Returns a dict of results. Always: ext, nrows, dead, no_overscan. Dead channels
    add frac_hi and sigma (overscan noise) and stop there. Otherwise: resolved,
    gain_held, gain_source ("fit", "prior" or "measured"), bad_cols, gain, sigma,
    sigma_raw, row_end, lam, ovs_charged_frac(_fit), and for resolved CCDs lam01,
    k_poisson, poisson_ratios; fallback/ped_active/fallback_note if the fallback was used.
    """
    # ---- split into active region and overscan, and subtract a rough per-row
    #      baseline (median of the overscan, similar to what panaSKImg starts from)
    act, x, xo, nrows, a0, no_ovs = split_active_overscan(data, cfg)
    early = slice(0, min(N_EARLY_ROWS, nrows))
    res = {"ext": ext, "nrows": nrows, "dead": False, "no_overscan": bool(no_ovs)}

    # ---- no CCD behind this channel? then nothing else to measure
    dead, s_ovs, frac_hi = is_dead_channel(x, xo, gain_prior, args)
    if dead:
        res.update(dead=True, frac_hi=frac_hi, sigma=s_ovs)
        return res

    # ---- gain and resolvedness
    gm = measure_gain(x, ext, args, gain_prior, early)
    g, resolved = gm["g"], gm["resolved"]

    # ---- fallback (only for CCDs called unresolved above, so resolved images are not
    #      affected): narrow zero peak in clean early rows -> fit 0e-/1e- with held gain
    fallback = None
    if not resolved and gain_prior and not args.no_fallback:
        fallback = held_gain_fallback(act, gain_prior, args)
        if fallback:
            x, sig_fb, nrows_fb = fallback
            xo = x                              # the active region also replaces the overscan
            g, resolved = gain_prior, True
            gm = dict(gm, g=g, resolved=True, gain_held=True)
            res.update(fallback=True, ped_active=True,
                       fallback_note=f"comb not found; zero peak {sig_fb / g:.2f} x gain wide in "
                                     f"{nrows_fb} clean early rows -> 0e-/1e- fit, gain held at "
                                     f"{g:.3f}, pedestal from the active region")

    res["resolved"] = resolved
    res["gain_held"] = gm["gain_held"] or not resolved
    if resolved:
        res["gain_source"] = "prior" if gm["gain_held"] else "fit"
    else:
        res["gain_source"] = "measured" if gm["credible"] else "prior"

    # ---- bad columns; the minimum deviation is ~0.3 e- (resolved) or half the noise
    res["bad_cols"], xg = mask_bad_columns(x, g, resolved, early, a0)

    if resolved:
        # ---- noise (intrinsic and as the fit sees it), dark current per row and
        #      the rows to use
        res.update(measure_resolved(xg, xo, g, nrows, early, args))
        if no_ovs or fallback:             # pedestal from the active region: "overscan charge" checks do not apply
            res.update(ovs_charged_frac=0.0, ovs_charged_frac_fit=0.0)
    else:
        # ---- unresolved: width of the zero peak in the early rows (broadened by the
        #      unresolved charge) and read noise from the overscan (no dark current there)
        g = g if gm["credible"] else gain_prior
        res.update(measure_unresolved(xg, xo, g, nrows, early))
    return res


# =============================================================================
# FitDarkCurrentProcess settings
# =============================================================================
def fit_settings(r):
    """Per-extension FitDarkCurrentProcess settings from the measurements in r.

    All values are in ADU. The model fitted by panaSKImg is a sum of n_peaks
    Gaussians (width sigma, spacing gain, first at mu) weighted by a Poisson
    distribution of mean lambda. The limits below keep each parameter near its
    measured value so the fit cannot wander to a wrong solution (e.g. twice the gain
    with every other peak empty).

    r: result dict of analyse_extension, with k_fit set by choose_peaks.
    Returns a dict: n_peaks, mu_lo/mu_hi (zero-peak position limits), sigma_min/max,
    gain_min/max, dc_max (lambda upper limit), x_min/x_max (fit window) and binning.
    Three cases: --peaks 1 (zero peak only), resolved peaks, unresolved peaks.
    """
    g, s, s_raw = r["gain"], r["sigma"], r["sigma_raw"]

    if r.get("k_fit") == 1:
        # --peaks 1: only the zero-electron peak (noise measurement, e.g. few skips).
        # The window stops at +1 sigma so the dark-current tail does not widen it; the
        # dark current cannot be measured from one peak and is held ~0 (a free lambda
        # would be degenerate with the normalisation); the gain is held within +-2%.
        w = s_raw if s_raw > 0 else s
        return dict(
            n_peaks=1,
            mu_lo=-max(0.5 * w, 0.2), mu_hi=max(0.5 * w, 0.2),
            sigma_min=w / 3, sigma_max=3 * w,
            gain_min=0.98 * g, gain_max=1.02 * g,
            dc_max=1e-3,
            x_min=-4 * w, x_max=1.0 * w,
            binning=min(max(w / 8, 0.005), 0.5),
        )

    if r["resolved"]:
        lam = max(r["lam"], 0.05)
        held = r.get("gain_held", False)
        k = r["k_fit"]            # number of peaks to fit (0e- ... (k-1)e-), see choose_peaks()
        if k is None:
            # --peaks all: every peak the Poisson distribution populates (~4 sigma)
            n_max = int(math.ceil(lam + 4 * math.sqrt(lam) + 2))
            x_max = g * n_max + max(4 * s_raw, 0.5 * g)
            n_peaks = n_max + 1
        else:
            # window ends halfway between the last fitted peak and the next one; one
            # extra model peak describes the tail of that next peak inside the window
            x_max = g * (k - 0.5)
            n_peaks = k + 1
        # Zero-electron peak position: expected near 0 after pedestal subtraction.
        # Rows whose pedestal landed one electron too high form a small bump at
        # -gain; the window starts above it so it is not taken for the zero peak.
        # If the zero peak is NOT near 0 (pedestal off for most rows), --tune finds it
        # in panaSKImg's own histogram and moves this window (refine_peaks).
        mu_hi = min(0.35 * g, max(4 * s_raw, 0.2))
        return dict(
            n_peaks=min(max(n_peaks, 3), 20),
            mu_lo=-mu_hi, mu_hi=mu_hi,
            # lower limit from the intrinsic width, upper from the width including
            # the row-to-row pedestal jitter the fit will see
            sigma_min=s / 2, sigma_max=max(2.0 * s_raw, s_raw + 0.05),
            # +-12% keeps the fit away from the 2x-gain solution; +-2% if held at the prior
            # (fallback CCDs: the image cannot constrain the gain at all -> +-0.5%)
            gain_min=(0.995 if r.get("fallback") else 0.98 if held else 0.88) * g,
            gain_max=(1.005 if r.get("fallback") else 1.02 if held else 1.12) * g,
            dc_max=g * max(3 * lam, 1.0),
            x_min=-max(3.5 * s_raw, 0.5 * g), x_max=x_max,
            binning=min(max(s / 6, 0.005), 0.05),      # ~6 bins per sigma
        )

    # Unresolved peaks: s = read noise (overscan), s_raw = width of the active-region
    # peak. The gain cannot be fitted, so it is held within +-2%; the distribution is
    # a Gaussian broadened by the Poisson charge, so the window covers both.
    lam = r["lam"] or 0.0
    n_max = int(math.ceil(lam + 4 * math.sqrt(lam) + 3))
    spread = math.sqrt(s ** 2 + lam * g ** 2)
    return dict(
        n_peaks=min(max(n_max + 1, 5), 60),
        mu_lo=-max(1.0, 0.2 * s), mu_hi=max(1.0, 0.2 * s),
        sigma_min=s / 2, sigma_max=max(3 * s, s_raw),
        gain_min=0.98 * g, gain_max=1.02 * g,
        dc_max=g * max(2 * lam + 1, 0.8),
        x_min=-max(6 * s, 3.0), x_max=g * lam + max(6 * spread, 3.0) + 2 * g,
        binning=min(max(s / 8, 0.01), 0.5),
    )


def choose_peaks(results, args):
    """Set r["k_fit"], the number of peaks fitted, for every resolved extension.

    --peaks auto: the number of peaks that follow a single Poisson (poisson_peaks).
        x_min/x_max are shared by all extensions in panaSKImg, so one value is used
        for the whole image: the smallest over the resolved extensions, so that no
        extension gets non-Poisson peaks inside its window.
    --peaks all: previous behaviour, all peaks populated by the Poisson distribution.
    --peaks N: fit exactly N peaks (0e- ... (N-1)e-).
    results: list of analyse_extension dicts (modified in place).
    Returns k (int, or None for 'all' / no resolved extension).
    """
    resolved = [r for r in results if r["resolved"]]
    if args.peaks == "all":
        k = None
    elif args.peaks == "auto":
        k = min((r["k_poisson"] for r in resolved), default=None)
    else:
        k = int(args.peaks)
    for r in results:
        # one peak (zero-electron peak only) also applies to unresolved extensions
        r["k_fit"] = k if (r["resolved"] or k == 1) else None
    return k


def shared_fit_window(results, any_res):
    """Choose the extensions that define panaSKImg's shared x_min/x_max/binning.

    These three settings are global in panaSKImg. They are built from the resolved
    extensions if any, otherwise from those with a typical window. Extensions left
    out (e.g. a very noisy or very charged CCD) are fit in a truncated window and
    flagged as unreliable. Without resolved extensions an extension is an outlier if
    its window span or x_max exceeds twice the median (typically a flooded CCD: so
    much charge, e.g. from light, that the histogram is far wider than the others).
    results: analyse_extension dicts with "fit" (fit_settings) set; any_res: True
    if any extension is resolved.
    Returns (used extensions, outliers, x_lo, x_hi), x_lo/x_hi in ADU.
    """
    F = {r["ext"]: r["fit"] for r in results}
    span = {k: v["x_max"] - v["x_min"] for k, v in F.items()}
    if any_res:
        use = [r["ext"] for r in results if r["resolved"]]
    else:
        med_span = float(np.median(list(span.values())))
        med_hi = float(np.median([v["x_max"] for v in F.values()]))
        use = [k for k in F if span[k] <= 2 * med_span and F[k]["x_max"] <= 2 * med_hi]
    outliers = [k for k in F if k not in use]
    x_lo = min(F[k]["x_min"] for k in use)
    x_hi = max(F[k]["x_max"] for k in use)
    return use, outliers, x_lo, x_hi


# =============================================================================
# Writing the JSON
# =============================================================================
def set_pedestal(j, results, no_overscan=False):
    """PedestalSubtractionProcess settings (shared by all extensions).

    The pedestal is the electronic baseline (ADU of zero charge) that panaSKImg
    subtracts row by row, normally estimated from the overscan. Three choices:
    per-row median (default for resolved peaks, or of the active region when there
    is no overscan), the lowest-peak finder (is_highDC, when the overscan of most
    fit rows already holds >= 1 e-, so a median/Gaussian would sit on the 1 e- peak),
    or the reference method (Gaussian fit by default) with Freedman binning (no
    resolved peaks).
    j: config dict, modified in place; results: analyse_extension dicts;
    no_overscan: force the active-region median. Returns None.
    """
    any_res = any(r["resolved"] for r in results)
    if no_overscan or any(r.get("no_overscan") or r.get("ped_active") for r in results):
        # windowed readout: per-row median of the active region (the zero-electron peak)
        ped = j["process"]["PedestalSubtractionProcess"]
        ped.update({"method": "median", "in_overscan": False, "n_sigma_to_mask": 1.5})
        ped.pop("is_highDC", None)
        return
    # Only the rows used in the DC fit count, and only if most of them have a charged
    # overscan: the lowest-peak finder on ~150 overscan pixels per row lands ~0.3 ADU
    # low, which is worse than a median pedestal that is 1 e- off in a minority of rows
    # (those form a small bump at -gain, outside the fit window).
    charged = any(r["resolved"] and r["ovs_charged_frac_fit"] > 0.5 for r in results)
    ped = j["process"]["PedestalSubtractionProcess"]
    if charged:
        # The overscan holds charge in some rows: use the lowest-peak finder with
        # coarse bins (a few hundred overscan pixels per row cannot fill 100000 bins).
        # Applied to every resolved extension, since the bin settings are shared and
        # the plain Gaussian fit mis-centres with 150 bins.
        ped.update({"hist_nbins_method": "manual", "hist_nbins": 150, "n_sigma_to_mask": -1})
        ped["is_highDC"] = {r["ext"]: bool(r["resolved"]) for r in results}
    elif not any_res:
        # broad peaks: let the histogram binning follow the data
        ped.update({"hist_nbins_method": "freedman"})
        ped.pop("is_highDC", None)
    else:
        # resolved peaks, (mostly) clean overscan: per-row median of the overscan. The
        # Gaussian fit to ~150 overscan pixels per row lands up to ~0.3 ADU off-centre
        # and adds row-to-row jitter; the median sits on the zero peak (tested on
        # AMTEst and skip-noise images: same or lower noise, zero peak closer to 0).
        ped.update({"method": "median"})
        ped.pop("is_highDC", None)


def set_dark_current_fit(j, results):
    """FitDarkCurrentProcess settings. Returns the list of outlier extensions.

    Writes into j (in place): rows to mask (the reference rows plus every row from
    row_end on, where the dark current is too high for the single-lambda model),
    bad columns to mask, the per-extension limits from r["fit"], and the shared
    window x_min/x_max/binning_size (shared_fit_window). Also adds "root" to
    format_figures so fit_quality.py can read the fitted histogram. Raises n_peaks
    where needed so the model covers the whole shared window.
    results: analyse_extension dicts with "fit" set (n_peaks may be raised).
    """
    dc = j["process"]["FitDarkCurrentProcess"]
    any_res = any(r["resolved"] for r in results)

    # rows/columns to leave out of the fit (this only affects the fit, not the image)
    tmpl_rows = dc.get("rows_to_mask", {})
    rows_mask, cols_mask = {}, {}
    for r in results:
        base_rows = tmpl_rows.get(r["ext"], []) if isinstance(tmpl_rows, dict) else list(tmpl_rows)
        # end the range at this extension's last row: with --acm the extensions are
        # stacked in one image, so a longer range would spill into the next one
        cut = [[r["row_end"], r["nrows"]]] if r["resolved"] and r["row_end"] < r["nrows"] else []
        rows_mask[r["ext"]] = list(base_rows) + cut
        cols_mask[r["ext"]] = compress_ranges(r["bad_cols"])

    # shared window; every extension needs enough peaks to cover all of it
    use, outliers, x_lo, x_hi = shared_fit_window(results, any_res)
    for r in results:
        if r.get("k_fit") == 1:            # --peaks 1: the window holds only the zero-electron peak
            continue
        need = int(math.ceil(x_hi / r["gain"])) + 1
        r["fit"]["n_peaks"] = min(max(r["fit"]["n_peaks"], need), 60 if not r["resolved"] else 20)

    F = {r["ext"]: r["fit"] for r in results}
    # also save the fitted histogram + function as ROOT (with --save-plots), so
    # fit_quality.py can report chi2/ndf and the parameter errors
    fmts = list(dc.get("format_figures", ["pdf"]))
    dc["format_figures"] = fmts + ["root"] if "root" not in fmts else fmts
    dc.update({
        "rows_to_mask": rows_mask,
        "cols_to_mask": cols_mask,
        "do_calibration": True,       # fit the gain (limits below)
        # starting value for the fit (panaSKImg uses it for the initial sigma/lambda)
        "calibration": round(float(np.mean([r["gain"] for r in results])), 4),
        "n_peaks": {k: v["n_peaks"] for k, v in F.items()},
        "mu_min": {k: round(v["mu_lo"], 4) for k, v in F.items()},
        "mu_max": {k: round(v["mu_hi"], 4) for k, v in F.items()},
        "sigma_min": {k: round(v["sigma_min"], 4) for k, v in F.items()},
        "sigma_max": {k: round(v["sigma_max"], 4) for k, v in F.items()},
        "gain_min": {k: round(v["gain_min"], 4) for k, v in F.items()},
        "gain_max": {k: round(v["gain_max"], 4) for k, v in F.items()},
        "dc_min": 0.0001,
        "dc_max": {k: round(v["dc_max"], 4) for k, v in F.items()},
        # these three are global in panaSKImg
        "x_min": round(x_lo, 2),
        "x_max": round(x_hi, 2),
        "binning_size": round(min(F[k]["binning"] for k in use), 4),
    })
    return outliers


AUTO_ALPHA = 0.10        # --cluster-threshold-e auto: allowed fraction of dark-current pixels above the threshold
AUTO_SEED_E = 9.5        # --cluster-threshold-e auto: seed threshold [e-] (a cluster needs a pixel with >= 10 e-)


def auto_cluster_thresholds(results, alpha=AUTO_ALPHA, seed_e=AUTO_SEED_E):
    """Pixel thresholds in electrons from the dark charge of each CCD (--cluster-threshold-e auto).

    In a skipper image the pixels around a cluster are not Gaussian noise but dark-current
    electrons: Poisson-distributed with mean lambda (0.2-1.7 e-/pix here, growing along the
    readout). A pixel threshold of k electrons lets a fraction P(Poisson(lambda) >= k) of the
    background pixels join clusters; once that fraction approaches the 8-neighbour percolation
    threshold (~0.41) they link up and clusters merge with the background, and well below it
    they still add charge at the cluster edge. Too high a threshold drops the faint edge of
    diffuse (back-side) clusters instead. The choice: the smallest k >= 2 with
    P(Poisson(lambda) >= k) <= alpha (alpha = 0.10), lambda = the median per-row dark charge
    of that CCD over all rows (the median, because rows crossed by tracks or bright clusters
    inflate the per-row estimate; the 90th percentile was 1.6-1.8x the median on DM10).
    Pixel values are electrons plus a small read noise, so "n >= k" is written as a
    threshold of k - 0.5 e-. Thresholds: k = 2 for lambda <= 0.53, 3 up to 1.10, 4 up to
    1.74, 5 up to 2.43, 6 up to 3.15 e-/pix. alpha = 0.10 was chosen on the 2025-08-27
    DM09/DM10 Image 5s: there pixels >= 2 e- at ~0.25 e-/pix and >= 3 e- at ~1.0 e-/pix gave
    back-side Fe-55 K-alpha within ~1 % (injection-recovery and real clusters), where the
    default (10 x the pedestal width) lost 5-15 % on high-dark-charge images; alpha = 0.05
    would pick 4 e- at 1.0 e-/pix (-1.5 % on injected diffuse clusters).
    CCDs without resolved peaks (no electron counts, so no lambda) keep the reference
    setting. Returns {ext: (pixel_e, seed_e, lambda_used, k) or None}.
    """
    from scipy.stats import poisson
    out = {}
    for r in results:
        lam = r.get("lam_all_median")
        if not r.get("resolved") or lam is None or not np.isfinite(lam):
            out[r["ext"]] = None
            continue
        k = 2
        while poisson.sf(k - 1, lam) > alpha and k < 50:
            k += 1
        out[r["ext"]] = (k - 0.5, max(seed_e, k + 0.5), float(lam), int(k))
    return out


def apply_cluster_thresholds(j, results, setting):
    """Write --cluster-threshold-e into the config: setting is (pixel_e, seed_e) or "auto"."""
    if setting == "auto":
        thr = auto_cluster_thresholds(results)
        cf = j["process"].setdefault("ClusterFinder", {})
        ref = cf.get("spr_threshold")
        new = dict(ref) if isinstance(ref, dict) else {}
        for ext, t in thr.items():
            if t is not None:
                new[ext] = [1.0, t[0], t[1]]
            elif isinstance(ref, dict) and ext in ref:
                new[ext] = ref[ext]
        cf["spr_threshold"] = new
        return thr
    set_cluster_threshold(j, results, *setting)
    return None


def describe_cluster_thresholds(thr):
    """One line per CCD for --cluster-threshold-e auto."""
    lines = []
    for ext, t in sorted(thr.items()):
        if t is None:
            lines.append(f"ClusterFinder {ext}: no resolved peaks - reference thresholds kept")
        else:
            lines.append(f"ClusterFinder {ext}: dark charge {t[2]:.2f} e-/pix (median of rows) -> pixels "
                         f">= {t[3]} e- (> {t[0]:g}), seed > {t[1]:g} e-")
    return lines


def set_cluster_threshold(j, results, pix_e, seed_e):
    """Make ClusterFinder use fixed thresholds in electrons.

    ClusterFinder's spr_threshold per extension is [sigma, n_pixel, n_seed]: a
    pixel joins a cluster if q > n_pixel*sigma and a cluster needs one pixel with
    q > n_seed*sigma. With sigma=None (reference default) sigma is the mean per-row
    pedestal noise, which depends on the pedestal method and is inflated by rows
    where the pedestal fit failed. A numeric sigma is taken in electrons, so
    sigma = 1 e- turns n_pixel/n_seed into thresholds in electrons.
    j: config dict (modified in place); pix_e, seed_e: thresholds in e-.
    """
    cf = j["process"].setdefault("ClusterFinder", {})
    cf["spr_threshold"] = {r["ext"]: [1.0, pix_e, seed_e] for r in results}


def skip_clusters(j, exts):
    """Give ClusterFinder an unreachable threshold (in e-) for these extensions.

    A flooded CCD (hundreds of e-/pixel) is one giant cluster: ClusterFinder spends
    minutes on it and the result is meaningless. With these thresholds no pixel
    passes, so it finds no clusters there; the other extensions are unchanged.
    j: config dict (modified in place); exts: extension names. Does nothing if
    spr_threshold is not a per-extension dict.
    """
    cf = j["process"].setdefault("ClusterFinder", {})
    thr = cf.get("spr_threshold")
    if not isinstance(thr, dict):
        return
    for e in exts:
        thr[e] = [1.0, 1e9, 1e9]


def set_plot_range(j, results, n_show):
    """Show the pixel charge up to n_show e- in the DC fit plot (x_max_plot).

    Needs the x_max_plot option in panaSKImg's FitDarkCurrentProcess: the histogram
    is drawn up to x_max_plot, the fit still uses [x_min, x_max] only, and the
    fitted model is continued as a dashed line beyond x_max.
    j: config dict (modified in place, needs the DC fit settings already set);
    results: analyse_extension dicts; n_show: --plot-peaks (e-). Only cosmetic: the
    fit itself is unchanged. Not written if panaSKImg lacks the option.
    """
    dc = j["process"]["FitDarkCurrentProcess"]
    if any(r.get("k_fit") == 1 for r in results) and panaskimg_has_plot_range():
        # --peaks 1: the fit window stops at +1 sigma, which puts the peak at the right
        # edge (under the statistics box); show more on the right so it sits mid-plot
        dc["x_max_plot"] = round(dc["x_max"] + 0.8 * (dc["x_max"] - dc["x_min"]), 2)
        return
    res = [r for r in results if r["resolved"]]
    if not res or n_show <= 0 or not panaskimg_has_plot_range():
        return
    g = max(r["gain"] for r in res)
    x0 = max(dc["mu_max"][r["ext"]] for r in res)   # zero peak can sit off 0 after --tune
    dc["x_max_plot"] = round(max(dc["x_max"], x0 + (n_show + 0.5) * g), 2)


def write_measured(path, infile, results, outliers=()):
    """<json>_measured.json: what this script measured, per extension.

    fit_quality.py uses lam01 (dark current from the 0e-/1e- pixel counts, independent
    of panaSKImg's fit) as the reference for the dark-current bias.
    Per extension: resolved, gain (ADU/e-), noise_e (e-), lam01 (e-/pix, None if not
    resolved), k_poisson, k_fit, skip (unresolved outlier, e.g. flooded CCD).
    path: file to write; infile: image path (its name without extensions is stored);
    outliers: from set_dark_current_fit.
    """
    ext = {}
    for r in results:
        ext[r["ext"]] = dict(resolved=bool(r["resolved"]), gain=round(float(r["gain"]), 5),
                             noise_e=round(float(r["sigma"] / r["gain"]), 5),
                             lam01=(round(float(r["lam01"]), 5) if r["resolved"] and r.get("lam01") is not None else None),
                             k_poisson=r.get("k_poisson"), k_fit=r.get("k_fit"),
                             # flooded/unresolved CCD among good ones: not worth running
                             skip=bool(r["ext"] in outliers and not r["resolved"]))
    with open(path, "w") as fh:
        json.dump({"image": os.path.basename(infile).split(".")[0], "extensions": ext}, fh, indent=2)


def write_per_ext(base, results, outliers, args):
    """--per-ext: one JSON per CCD and a script that runs panaSKImg once per CCD.

    Each JSON is the reference with extensions=[N] and the settings computed from that
    CCD alone: pedestal method (a CCD with a charged overscan no longer forces the
    lowest-peak finder on the others), fit window and binning. The number of fitted
    peaks is still the module-wide choice (choose_peaks): letting each CCD take more
    peaks put the dark current 5-8% above the 0e-/1e- estimate in tests.
    base: reference config with geometry and extensions only (before any process
    settings); outliers: from set_dark_current_fit on the whole module.
    Writes args.output/<image>_EXTn.json and run_<image>.sh (write_run_script);
    with --tune, tunes each CCD first (tune_per_ext).
    Returns [(ext, json path, resolved, skip)].
    """
    os.makedirs(args.output, exist_ok=True)
    stem = os.path.basename(args.infile).split(".")[0]
    configs = []
    for r in results:
        j = copy.deepcopy(base)
        j["input"]["image"]["extensions"] = [int(r["ext"][3:])]
        set_pedestal(j, [r])
        set_dark_current_fit(j, [r])
        if args.cluster_threshold_e is not None:
            thr = apply_cluster_thresholds(j, [r], args.cluster_threshold_e)
            if thr:
                for line in describe_cluster_thresholds(thr):
                    print("  " + line)
        # a flooded/unresolved CCD among good ones: its fit is not reliable and
        # ClusterFinder would spend minutes on it, so it is left out of the run script
        skip = r["ext"] in outliers and not r["resolved"] and not args.cluster_all
        if skip:
            skip_clusters(j, [r["ext"]])
        configs.append([r, j, skip])

    if args.tune:
        tune_per_ext(configs, args)

    written = []
    for r, j, skip in configs:
        set_plot_range(j, [r], args.plot_peaks)
        path = os.path.join(args.output, f"{stem}_{r['ext']}.json")
        with open(path, "w") as fh:
            json.dump(j, fh, indent=4)
        written.append((r["ext"], path, r["resolved"], skip))
        dc = j["process"]["FitDarkCurrentProcess"]
        ped = j["process"]["PedestalSubtractionProcess"]
        print(f"  {r['ext']}: pedestal {ped['method']}{' (lowest peak)' if 'is_highDC' in ped else ''}, "
              f"window [{dc['x_min']}, {dc['x_max']}] ADU, bin {dc['binning_size']}"
              f"{', NOT in run script (unresolved outlier)' if skip else ''} -> {path}")
    write_run_script(args, stem, written)
    return written


class _ThreadOutput:
    """sys.stdout replacement: a thread can collect its own prints (tune_per_ext).

    The CCDs are tuned in parallel threads; without this their messages would be
    interleaved. A thread that sets sys.stdout.local.buf to a StringIO gets its
    prints there; all other output goes to the original stream.
    """

    def __init__(self, orig):
        """orig: the stream to fall back to (the real sys.stdout)."""
        self.orig, self.local = orig, threading.local()

    def write(self, text):
        """Write text to this thread's buffer if it has one, else to the original stream."""
        buf = getattr(self.local, "buf", None)
        return (buf if buf is not None else self.orig).write(text)

    def flush(self):
        """Flush the original stream (the per-thread buffers are in memory)."""
        self.orig.flush()


def tune_per_ext(configs, args):
    """--tune with --per-ext: tune every resolved CCD on its own, CCDs in parallel.

    Per CCD: tune_pedestal (pedestal candidates ranked by total fit error) and
    refine_peaks (zero peak and window from panaSKImg's own histogram). The number of
    fitted peaks stays the module-wide choice (choose_peaks), as without --tune.
    configs: [[r, json, skip]], updated in place (item[1] becomes the tuned config).
    Each CCD's messages are collected per thread and printed afterwards, CCD by CCD.
    The panaSKImg runs go to <tune_dir or temp dir>/EXTn (temp dir removed afterwards).
    """
    import concurrent.futures as cf
    work_root, src = prepare_workdir(os.path.abspath(args.infile), args.tune_dir)

    def one(item):
        """Tune one CCD (runs in a worker thread); returns the text it printed."""
        r, j, _ = item
        sys.stdout.local.buf = buf = io.StringIO()
        try:
            a = copy.copy(args)
            if args.peaks == "auto" and r.get("k_fit"):
                a.peaks = str(r["k_fit"])          # keep the module-wide number of peaks
            work = os.path.join(work_root, r["ext"])
            os.makedirs(work, exist_ok=True)
            j2, _ = tune_pedestal(j, src, work, [r], args.panaskimg)
            item[1] = refine_peaks(j2, src, work, [r], a)
        finally:
            sys.stdout.local.buf = None
        return buf.getvalue()

    todo = [c for c in configs if c[0]["resolved"] and not c[2]]
    orig = sys.stdout
    sys.stdout = _ThreadOutput(orig)
    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, min(len(todo), 4))) as ex:
            logs = list(ex.map(one, todo))
    finally:
        sys.stdout = orig
    for c, text in zip(todo, logs):
        print(f"--tune {c[0]['ext']}:")
        print("\n".join("  " + l for l in text.rstrip().splitlines()))
    if args.tune_dir is None:
        shutil.rmtree(work_root, ignore_errors=True)


def write_run_script(args, stem, written):
    """run_<image>.sh: panaSKImg once per CCD, all in parallel; output folder = $1.

    stem: image name without extensions; written: [(ext, json path, resolved, skip)]
    from write_per_ext. Skipped CCDs appear as commented-out commands. The script
    lists the logs without "Finished correctly" at the end. Written to args.output
    (made executable).
    """
    infile = os.path.abspath(args.infile)
    note = ""
    if infile.endswith(".bz2"):
        # panaSKImg cannot read .fits.bz2: use the decompressed file next to it
        note = f"# NOTE: panaSKImg cannot read .fits.bz2; decompress first (bunzip2 -k {infile})\n"
        infile = infile[:-4]
    lines = ["#!/bin/bash",
             f"# Made by autoconfig_panaSKImg.py --per-ext for {os.path.basename(args.infile)}.",
             "# Runs panaSKImg once per CCD (in parallel), each with its own JSON.",
             "# Usage: bash " + f"run_{stem}.sh [output_folder]   (default: ./analysis)",
             note.rstrip("\n"),
             'OUT="${1:-analysis}"',
             'mkdir -p "$OUT"',
             f'PY="{sys.executable}"',
             f'PANA="{os.path.abspath(args.panaskimg)}"']
    for ext, path, resolved, skip in written:
        n = ext[3:]
        cmd = (f'"$PY" "$PANA" "{infile}" -j "{os.path.abspath(path)}" -o "$OUT/" --acm --save-plots '
               f'--ext {n} > "$OUT/{stem}_{ext}_panaSKImg.log" 2>&1 &')
        if skip:
            lines.append(f"# {ext} skipped: unresolved outlier CCD (fit unreliable, no clusters). To run it anyway:")
            lines.append("# " + cmd)
        else:
            lines.append(cmd)
    lines += ["wait",
              'echo "done: outputs in $OUT"',
              'grep -L "Finished correctly" "$OUT"/' + stem + '_EXT*_panaSKImg.log 2>/dev/null '
              '| sed "s/^/FAILED (see log): /"']
    path = os.path.join(args.output, f"run_{stem}.sh")
    with open(path, "w") as fh:
        fh.write("\n".join(l for l in lines if l != "") + "\n")
    os.chmod(path, 0o755)
    print(f"  run script: bash {path} <output_folder>")


# =============================================================================
# Optional tuning of the pedestal method with panaSKImg itself (--tune)
#
# The settings above come from this script's own reading of the pixels. --tune
# checks them against panaSKImg's actual result: it runs panaSKImg (pedestal
# subtraction + DC fit only) with a few candidate configs, ranks them by fit
# quality (tune_pedestal), then re-places the fit window on panaSKImg's own
# histogram (refine_peaks).
# =============================================================================
def parse_dcfit_log(text):
    """{ext: dict(mu, sigma, lam, gain)} (ADU) from a panaSKImg log, or None if it failed."""
    if "Finished correctly" not in text:
        return None
    out = {}
    for block in text.split("Process <FitDarkCurrentProcess>")[1:]:
        m = re.search(r"amplifier (EXT\d+)", block)
        if not m:
            continue
        # lines look like "- gain [ADU/e-] = 1.32 +/- 0.001"
        val = lambda key: float(re.search(key + r".*?= ([-\d.eE+]+) \+/-", block).group(1))
        out[m.group(1)] = dict(mu=val("zero-e- peak"), sigma=val("resolution"),
                               lam=val("dark current"), gain=val(r"gain \["))
    return out


def run_candidate(cfg, infile, workdir, name, panaskimg, save_hist=False):
    """Run panaSKImg (pedestal subtraction + DC fit only) with cfg; return parsed results.

    The config and log are written to workdir/<name>.json and workdir/<name>/log.txt.
    With save_hist, the fitted histogram of every extension is also saved as
    workdir/<name>/*_FitDarkCurrentProcess_sideEXTn.root (read by histogram_peaks).
    cfg: config dict (not modified); infile: image panaSKImg reads; panaskimg: path
    of the panaSKImg script (run with the current python).
    Returns parse_dcfit_log's dict, with res[ext]["fit"] (fit_quality.read_side_fit:
    parameters, errors, chi2) when save_hist; None if panaSKImg failed.
    """
    j = copy.deepcopy(cfg)
    j["process"]["sequence"] = "CompressSkipperProcess;PedestalSubtractionProcess;FitDarkCurrentProcess"
    if save_hist:
        j["process"]["FitDarkCurrentProcess"]["format_figures"] = ["root"]
    jpath = os.path.join(workdir, f"{name}.json")
    with open(jpath, "w") as fh:
        json.dump(j, fh, indent=2)
    odir = os.path.join(workdir, name) + "/"
    os.makedirs(odir, exist_ok=True)
    cmd = [sys.executable, panaskimg, infile, "-j", jpath, "-o", odir]
    if j["input"]["image"].get("ACM_multi_extension", False):
        cmd.append("--acm")
    if save_hist:
        cmd.append("--save-plots")      # panaSKImg only writes figure files with this flag
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(odir, "log.txt"), "w") as fh:
        fh.write(p.stdout + p.stderr)
    res = parse_dcfit_log(p.stdout + p.stderr)
    if res and save_hist:
        # parameter errors and chi2 from the stored fit (fit_quality.read_side_fit)
        for e in res:
            files = glob.glob(os.path.join(odir, f"*_side{e}.root"))
            res[e]["fit"] = FQ.read_side_fit(files[0], e) if files else None
    return res


def at_limit(r, dc, e):
    """True if the fitted mu or sigma of extension e sits on its limit.

    A parameter pinned at a limit means the fit wanted to go further: the minimum is
    not inside the allowed range, and the quoted errors are not meaningful.
    r: parsed fit result (parse_dcfit_log), dc: FitDarkCurrentProcess settings.
    """
    return (abs(r["mu"] - dc["mu_min"][e]) < 1e-3 or abs(r["mu"] - dc["mu_max"][e]) < 1e-3
            or abs(r["sigma"] - dc["sigma_max"][e]) < 1e-3 * dc["sigma_max"][e])


def score(res, cfg, exts, refs):
    """Quality of a candidate run (lower is better).

    Sum over the resolved extensions of the total relative fit error
    (fit_quality.quality: gain and noise errors scaled by sqrt(chi2/ndf), dark current
    error combined with its bias against refs[ext], the 0e-/1e- estimate), plus 1 for
    every fit with a parameter stuck at a limit (its errors are not meaningful).
    The quality of each extension is stored in res[ext]["q"].
    """
    if res is None:
        return math.inf
    dc = cfg["process"]["FitDarkCurrentProcess"]
    tot = 0.0
    for e in exts:
        if e not in res or not res[e].get("fit"):
            return math.inf
        q = FQ.quality(res[e]["fit"], refs.get(e))
        res[e]["q"] = q
        tot += q["score"]
        tot += 1.0 if at_limit(res[e], dc, e) else 0.0
    return tot


def prepare_workdir(infile, keep_dir):
    """Working folder for the --tune runs and a panaSKImg-readable copy of the input.

    panaSKImg's file-name parser does not accept *.fits.bz2, so those are decompressed.
    keep_dir: --tune-dir, or None for a new temporary folder (the caller removes it).
    Returns (working folder, path of the image panaSKImg should read).
    """
    work = keep_dir or tempfile.mkdtemp(prefix="autoconfig_")
    os.makedirs(work, exist_ok=True)
    src = infile
    if infile.endswith(".bz2"):
        src = os.path.join(work, os.path.basename(infile)[:-4])
        with bz2.open(infile, "rb") as fi, open(src, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return work, src


def tune_pedestal(j, src, work, results, panaskimg):
    """Try the candidate pedestal settings in panaSKImg and return the best config.

    Candidates: the settings chosen by this script ("chosen"), the median of the
    overscan ("median": lands exactly on a peak when peaks are sharp, so a wrong row
    is off by a whole electron and the comb stays aligned), and the other of
    Gaussian fit / lowest-peak finder. Each is run in panaSKImg (run_candidate) and
    ranked by score(); a ranking table is printed.
    j: config dict (not modified); src: image panaSKImg reads; work: folder for the
    runs; results: analyse_extension dicts; panaskimg: path of the panaSKImg script.
    Returns (config, name of the chosen candidate), or (j, None) if no extension is
    resolved.
    """
    resolved = [r["ext"] for r in results if r["resolved"]]
    if not resolved:
        return j, None        # nothing to compare without electron peaks
    refs = {r["ext"]: r["lam01"] for r in results if r["resolved"]}

    ped0 = j["process"]["PedestalSubtractionProcess"]
    candidates = {"chosen": {},
                  "median": {"method": "median", "n_sigma_to_mask": 1.5}}
    if ped0.get("is_highDC"):
        candidates["gausfit"] = {"is_highDC": {k: False for k in ped0["is_highDC"]},
                                 "hist_nbins_method": "manual", "hist_nbins": 100000, "n_sigma_to_mask": 1.5}
    else:
        candidates["highDC"] = {"hist_nbins_method": "manual", "hist_nbins": 150, "n_sigma_to_mask": -1,
                                "is_highDC": {r["ext"]: bool(r["resolved"]) for r in results}}

    table = []
    for name, upd in candidates.items():
        c = copy.deepcopy(j)
        c["process"]["PedestalSubtractionProcess"].update(upd)
        if name == "median":
            c["process"]["PedestalSubtractionProcess"].pop("is_highDC", None)
        res = run_candidate(c, src, work, name, panaskimg, save_hist=True)
        table.append((score(res, c, resolved, refs), name, c, res))
    table.sort(key=lambda t: t[0])

    print("pedestal tuning (score = total relative fit error, lower is better; per extension: "
          "chi2/ndf, DC +- scaled error and bias; * = parameter at a limit):")
    for sc, name, c, res in table:
        if res is None or sc == math.inf:
            print(f"  {name:8s} failed")
            continue
        dc = c["process"]["FitDarkCurrentProcess"]
        cells = []
        for e in resolved:
            q = res[e]["q"]
            cells.append(f"{e} chi2 {q['chi2ndf']:.1f} DC {q['dc']:.3f}+-{q['dc_err_s']:.3f} "
                         f"({q['dc_bias']:+.3f}){'*' if at_limit(res[e], dc, e) else ''}")
        print(f"  {name:8s} score {sc:7.3f}  " + " | ".join(cells))
    best = table[0]
    print(f"  -> using '{best[1]}' pedestal settings")
    return best[2], best[1]


def histogram_peaks(*a, **k):
    """Thread-safe wrapper (ROOT file access) of _histogram_peaks.

    ROOT is not thread-safe and tune_per_ext runs CCDs in parallel threads, so file
    access is serialised with fit_quality's ROOT_LOCK. Same arguments and return value.
    """
    with FQ.ROOT_LOCK:
        return _histogram_peaks(*a, **k)


def _histogram_peaks(rootfile, ext, g, lam_expected, max_peaks, tol):
    """Electron peaks in the histogram panaSKImg actually fitted.

    Reads the DC-fit histogram saved by panaSKImg (ROOT file) and finds the comb of
    peaks (spacing g). The zero-electron peak is the comb position whose following
    peaks best match a Poisson distribution of mean lam_expected (the dark current
    this script measured row by row): a small bump one electron below the real zero
    (rows whose pedestal landed on the wrong peak) gives the sequence [small, large,
    ...] and is rejected, while a real zero followed by taller peaks (lambda > 1) is
    kept. Then the pixels in every peak above zero are counted and the same Poisson
    test as poisson_peaks() selects the number of peaks.
    rootfile: panaSKImg's *_FitDarkCurrentProcess_sideEXTn.root (histogram "pcd");
    ext: extension name (not used in the lookup); g: gain (ADU/e-);
    lam_expected: e-/pix; max_peaks, tol: as for poisson_peaks.
    Returns dict(zero=position in ADU, k, ratios, lam01).
    """
    import ROOT                       # only needed for --tune; available in the damic env
    ROOT.gROOT.SetBatch(1)
    f = ROOT.TFile(rootfile)
    h = f.Get("pcd")
    x = np.array([h.GetBinCenter(i) for i in range(1, h.GetNbinsX() + 1)])
    y = np.array([h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)])
    f.Close()

    def peak_count(pos):
        """Pixels in the peak at pos (ADU): histogram content within +-0.4 gain."""
        return float(y[np.abs(x - pos) < 0.4 * g].sum())

    # comb position from the count-weighted phase, then all peak positions in range
    off = g * np.angle(np.sum(y * np.exp(2j * np.pi * x / g))) / (2 * np.pi)
    first = math.ceil((x.min() - off) / g)
    positions = [off + n * g for n in range(first, first + int((x.max() - x.min()) / g) + 2)
                 if x.min() <= off + n * g <= x.max()]
    # Shape test: fractions of pixels in the first m peaks above a candidate zero,
    # compared with a Poisson of mean lam_expected (renormalised over those m peaks)
    # through the KL divergence (0 = identical shape).
    m = 4
    lam = max(lam_expected, 0.01)
    prob = np.array([math.exp(-lam) * lam ** n / math.factorial(n) for n in range(m)])
    prob /= prob.sum()

    def zero_score(z):
        """How well z works as the zero peak: minus the KL divergence (higher = better).

        -inf if there are no pixels at z.
        """
        c = np.array([peak_count(z + n * g) for n in range(m)])
        if c[0] <= 0:
            return -math.inf
        q = c / c.sum()
        ok = q > 0
        return -float(np.sum(q[ok] * np.log(q[ok] / prob[ok])))   # higher is better

    # only candidates within 3 electrons of the nominal zero (pedestal-subtracted 0 ADU)
    near = [z for z in positions if abs(z) <= 3 * g] or positions
    zero = max(near, key=zero_score)

    counts = []
    pos = zero
    while pos <= x.max():
        counts.append(peak_count(pos))
        pos += g
    k, ratios, lam01 = poisson_peaks(np.array(counts), max_peaks, tol)
    return dict(zero=zero, k=k, ratios=ratios, lam01=lam01)


def locate_histogram_peaks(resolved, work, args):
    """Step 2 of refine_peaks: zero peak and Poisson-like peaks in the scan histograms.

    resolved: resolved analyse_extension dicts; work: folder holding the
    "peaks_scan" run. Prints one line per extension.
    Returns {ext: histogram_peaks result} for the extensions with a saved histogram.
    """
    print("peak refinement (panaSKImg histogram; data/Poisson for peaks 0,1,2..):")
    found = {}
    for r in resolved:
        files = glob.glob(os.path.join(work, "peaks_scan", f"*_sideEXT{r['ext'][3:]}.root"))
        if not files:
            print(f"  {r['ext']}: no histogram saved, keeping the first-pass settings")
            continue
        hp = histogram_peaks(files[0], r["ext"], r["gain"], r["lam"], args.max_peaks, args.poisson_tol)
        found[r["ext"]] = hp
        print(f"  {r['ext']}: zero peak at {hp['zero']:+.2f} ADU, "
              + " ".join(f"{v:.2f}" for v in hp["ratios"][:8]) + f"  -> {hp['k']} Poisson-like peaks")
    return found


def apply_peak_window(j, resolved, found, args):
    """Step 3 of refine_peaks: move the mu limits and fit window onto the found peaks.

    The mu limits keep their width but are centred on the zero peak found in the
    histogram; x_min is placed below it, and (unless --peaks all) x_max halfway
    between the last fitted peak and the next one. Sets r["k_fit"] of the refined
    extensions. j: config dict (not modified); found: from locate_histogram_peaks.
    Prints the chosen window. Returns the refined copy of j.
    """
    if args.peaks == "auto":
        k = min(hp["k"] for hp in found.values())
    elif args.peaks == "all":
        k = None
    else:
        k = int(args.peaks)
    final = copy.deepcopy(j)
    dc = final["process"]["FitDarkCurrentProcess"]
    lo, hi = [], []
    for r in resolved:
        e = r["ext"]
        if e not in found:
            continue
        x0, g, half = found[e]["zero"], r["gain"], dc["mu_max"][e] - dc["mu_min"][e]
        dc["mu_min"][e] = round(x0 - half / 2, 4)
        dc["mu_max"][e] = round(x0 + half / 2, 4)
        lo.append(x0 - max(3.5 * r["sigma_raw"], 0.5 * g))
        if k is not None:
            dc["n_peaks"][e] = k + 1
            hi.append(x0 + (k - 0.5) * g)
            r["k_fit"] = k
    dc["x_min"] = round(min(lo), 2)
    if hi:
        dc["x_max"] = round(max(hi), 2)
    print(f"  -> fitting {k if k else 'all'} peaks, window [{dc['x_min']}, {dc['x_max']}] ADU")
    return final


def refine_peaks(j, src, work, results, args):
    """Iterative refinement of the DC-fit window with panaSKImg itself (part of --tune).

    Why: this script estimates the zero-peak position and the useful number of peaks
    from its own rough pedestal; panaSKImg's pedestal can differ (e.g. land one
    electron off in many rows), so the window is re-placed on panaSKImg's histogram.
    1. Run panaSKImg with a wide window (up to --max-peaks peaks) and save the histogram.
    2. In that histogram, per extension: locate the zero-electron peak (Poisson shape
       test, see histogram_peaks) and count which peaks follow a single Poisson
       (locate_histogram_peaks).
    3. Centre the mu limits on the zero peak and set the window to the chosen number
       of peaks (--peaks auto: smallest over the resolved extensions, since the
       window is shared) (apply_peak_window).
    4. Run the final config once more to check the fit converged (prints the quality).
    src, work: image and working folder (prepare_workdir); results: analyse_extension
    dicts (k_fit may be updated).
    Returns the refined config (unchanged if ROOT is not available).
    """
    resolved = [r for r in results if r["resolved"]]
    if not resolved:
        return j
    try:
        import ROOT  # noqa: F401
    except ImportError:
        print("peak refinement skipped: PyROOT not available in this python")
        return j

    # ---- 1. wide scan
    gmax = max(r["gain"] for r in resolved)
    scan = copy.deepcopy(j)
    dc = scan["process"]["FitDarkCurrentProcess"]
    dc["x_min"] = round(-2.5 * gmax, 2)
    dc["x_max"] = round((args.max_peaks + 0.5) * gmax, 2)
    dc["n_peaks"] = {k: min(args.max_peaks + 3, 20) for k in dc["n_peaks"]}
    run_candidate(scan, src, work, "peaks_scan", args.panaskimg, save_hist=True)

    # ---- 2. zero peak and Poisson-like peaks per extension
    found = locate_histogram_peaks(resolved, work, args)
    if not found:
        return j

    # ---- 3. final window
    final = apply_peak_window(j, resolved, found, args)
    dc = final["process"]["FitDarkCurrentProcess"]

    # ---- 4. check run
    res = run_candidate(final, src, work, "peaks_final", args.panaskimg, save_hist=True)
    refs = {r["ext"]: r["lam01"] for r in results if r["resolved"]}
    if res:
        for e, v in res.items():
            if e in dc["mu_min"] and e in refs and v.get("fit"):
                flag = "  (parameter at limit!)" if at_limit(v, dc, e) else ""
                print(f"  check {e}: {FQ.fmt(FQ.quality(v['fit'], refs[e]))}, mu {v['mu']:+.2f}{flag}")
    return final


# =============================================================================
# Main
# =============================================================================
def parse_args():
    """Command-line options (see -h); validates --peaks, --cluster-threshold-e, panaSKImg.

    Options are grouped as in the help: gain (prior and search range, when peaks
    count as resolved), dark-current fit (rows and peaks used), geometry and
    channels, clusters, tuning (--tune) and one JSON per extension (--per-ext).
    Exits with a usage error for an invalid --peaks, PIXEL_E/SEED_E not satisfying
    0 < PIXEL_E <= SEED_E, or --tune/--per-ext without a panaSKImg script.
    Returns the argparse namespace.
    """
    ap = argparse.ArgumentParser(description="Generate a panaSKImg JSON tuned to a raw skipper image.")
    ap.add_argument("infile", help="raw (skip-averaged) FITS image (.fits or .fits.bz2)")
    ap.add_argument("-o", "--output", required=True, help="output JSON file")
    ap.add_argument("-r", "--reference", "-t", "--template", dest="reference", default=None,
                    help="reference JSON the generated config starts from; only the measured settings "
                         "are changed (default: the reference built into this script). -t/--template is an alias")

    g = ap.add_argument_group("gain")
    g.add_argument("--gain", default=None,
                   help="gain prior in ADU/e- used when peaks are not resolved, e.g. 1.3 or "
                        "EXT1=1.32,EXT2=1.30 (default: CalibrationProcess gain in the reference JSON)")
    g.add_argument("--gain-min", type=float, default=0.6, help="gain search range low, ADU/e- (default: 0.6)")
    g.add_argument("--gain-max", type=float, default=3.0, help="gain search range high, ADU/e- (default: 3.0)")
    g.add_argument("--prior-tol", type=float, default=0.25,
                   help="max relative difference between measured gain and gain prior (default: 0.25)")
    g.add_argument("--max-sig-over-gain", type=float, default=0.3,
                   help="noise/gain above which peaks count as unresolved (default: 0.3)")

    f = ap.add_argument_group("dark-current fit")
    f.add_argument("--lam-max", type=float, default=1.0,
                   help="max mean dark current (e-/pix) of the rows used in the fit (default: 1.0)")
    f.add_argument("--min-rows", type=int, default=25, help="minimum number of rows in the fit (default: 25)")

    c = ap.add_argument_group("geometry and channels")
    c.add_argument("--ccd-cols", type=int, default=None,
                   help="prescan+active columns in UNBINNED pixels, as panaSKImg's ccd_cols "
                        "(default: from header PHYSCOL-NCOLPRE+PRESCAN, else reference JSON)")
    c.add_argument("--dead-frac", type=float, default=2e-4,
                   help="channels whose active region has less than this fraction of pixels with "
                        "charge (and the same noise as the overscan) are treated as having no CCD "
                        "(default: 2e-4)")
    c.add_argument("--keep-dead", action="store_true",
                   help="keep channels without a CCD in 'extensions' (default: remove them)")

    # more "dark-current fit" options (group f), defined after group c: -h lists them
    # under their group, while the usage line follows the order of definition
    f.add_argument("--no-fallback", action="store_true",
                   help="do not try the held-gain 0e-/1e- fit for CCDs whose peaks the comb search "
                        "could not resolve (see held_gain_fallback)")
    f.add_argument("--peaks", default="auto",
                   help="number of electron peaks to fit: 'auto' (default) = the peaks that follow a "
                        "single Poisson distribution (stops at the first peak with extra, non-Poisson "
                        "charge, e.g. from clusters), 'all' = every peak the Poisson populates "
                        "(previous behaviour), or an integer N (peaks 0..N-1 e-); 1 = only the zero-electron "
                        "peak (noise measurement when the peaks are not resolved, e.g. 1-100 skips)")
    f.add_argument("--poisson-tol", type=float, default=0.25,
                   help="relative tolerance of the Poisson test used by --peaks auto: a peak counts as "
                        "Poisson-like if data/prediction is within 1 +- tol (default: 0.25)")
    f.add_argument("--max-peaks", type=int, default=10,
                   help="upper limit for --peaks auto (default: 10)")

    k = ap.add_argument_group("clusters")
    k.add_argument("--cluster-threshold-e", nargs="+", metavar="PIXEL_E SEED_E | auto",
                   default=None,
                   help="ClusterFinder thresholds in electrons: a pixel joins a cluster if it has more "
                        "than PIXEL_E e-, and a cluster needs at least one pixel above SEED_E e- "
                        "(e.g. 2.5 9.5), or 'auto': per CCD from its dark charge (smallest k with "
                        "P(Poisson(lambda) >= k) <= 10%, pixel > k - 0.5 e-, seed > 9.5 e-). Default: keep "
                        "the reference JSON's spr_threshold, which is in units of the per-row pedestal noise")
    k.add_argument("--cluster-all", action="store_true",
                   help="also search clusters in CCDs flagged as unresolved outliers (e.g. a flooded "
                        "CCD); by default they get an unreachable threshold, since clustering them "
                        "takes minutes and gives one meaningless cluster")
    k.add_argument("--plot-peaks", type=float, default=5,
                   help="draw the DC fit histogram up to about this many e- (x_max_plot; the fit "
                        "range is unchanged, beyond it the model is dashed). 0 = plot only the fit "
                        "range (default: 5). Needs panaSKImg with the x_max_plot option")

    t = ap.add_argument_group("tuning")
    t.add_argument("--tune", action="store_true",
                   help="run panaSKImg (pedestal + DC fit only) with candidate pedestal settings "
                        "and keep the one giving the narrowest, unpinned fit (about 1-2 minutes)")
    t.add_argument("--panaskimg", default=DEFAULT_PANASKIMG,
                   help="panaSKImg script used by --tune and the --per-ext run script (default: found "
                        "automatically: $PANASKIMG, PATH, the python environment)")
    t.add_argument("--tune-dir", default=None,
                   help="keep the --tune runs (configs, logs) in this folder instead of a temp dir")

    p = ap.add_argument_group("one JSON per extension")
    p.add_argument("--per-ext", action="store_true",
                   help="-o is then a FOLDER: write one JSON per CCD (<image>_EXTn.json, each with "
                        "its own pedestal method, fit window and binning) and run_<image>.sh, which "
                        "runs panaSKImg once per CCD in parallel (--acm --ext N) and gives one "
                        "cluster file per CCD (..._extN.root)")

    args = ap.parse_args()
    if args.peaks not in ("auto", "all") and not (args.peaks.isdigit() and int(args.peaks) >= 1):
        ap.error("--peaks must be 'auto', 'all' or an integer >= 1")
    if args.cluster_threshold_e is not None:
        v = args.cluster_threshold_e
        if v == ["auto"]:
            args.cluster_threshold_e = "auto"
        else:
            try:
                pix, seed = (float(x) for x in v)
            except ValueError:
                ap.error("--cluster-threshold-e: give PIXEL_E SEED_E (two numbers) or 'auto'")
            if pix <= 0 or seed < pix:
                ap.error("--cluster-threshold-e: need 0 < PIXEL_E <= SEED_E")
            args.cluster_threshold_e = (pix, seed)
    if (args.tune or args.per_ext) and not args.panaskimg:
        ap.error("panaSKImg not found: activate the environment where pysimdamicm is installed, "
                 "or give --panaskimg /path/to/panaSKImg (or set $PANASKIMG)")
    return args


def geometry(h0, hdul, img, args):
    """Active/overscan column boundary.

    panaSKImg takes ccd_cols in UNBINNED pixels and computes
    n_cols_overscan = int(ncols - ccd_cols/NSBIN). Returns (ccd_cols for the JSON,
    analysis settings for analyse_extension, ncol, nsbin).

    A raw row holds prescan columns, the active (real CCD) columns and then the
    overscan. Priority: --ccd-cols, then the header (PHYSCOL - NCOLPRE + PRESCAN),
    then the reference JSON. A windowed readout (only part of the CCD read) has no
    overscan; then ccd_cols covers the whole row.
    h0: primary FITS header; hdul: open FITS file; img: the reference's
    input.image block; args: uses --ccd-cols. The analysis settings (dict) are in
    BINNED columns: id_row_start, n_cols_prescan, ccd_cols (first overscan
    column), no_overscan.
    """
    ncol = int(h0.get("NCOL", hdul[1].header["NAXIS1"]))
    nsbin = int(h0.get("NSBIN", 1)) or 1
    ccd_cols = args.ccd_cols
    if ccd_cols is None and all(k in h0 for k in ("PHYSCOL", "NCOLPRE", "PRESCAN")):
        guess = int(h0["PHYSCOL"] - h0["NCOLPRE"] + h0["PRESCAN"])
        if 0 < ncol - int(ncol - guess / nsbin) < ncol - 3:   # sanity: leaves an overscan
            ccd_cols = guess
        elif guess / nsbin >= ncol - 3:
            # windowed readout (NCOLPRE columns skipped): the window lies inside the
            # active region, so there is no overscan. panaSKImg then takes the pedestal
            # from the active region (it does so by itself when < 3 overscan columns).
            ccd_cols = ncol * nsbin
    if ccd_cols is None:
        ccd_cols = int(img["ccd_cols"])
        if ccd_cols / nsbin > ncol:                            # reference does not fit this image
            ccd_cols = ncol * nsbin
    first_ovs = ncol - int(ncol - ccd_cols / nsbin)          # first overscan column (binned)
    cfg = {"id_row_start": int(img.get("id_row_start", 0)),
           "n_cols_prescan": int(int(img.get("n_cols_prescan", 0)) / nsbin),
           "ccd_cols": first_ovs,
           "no_overscan": first_ovs >= ncol - 2}
    if cfg["no_overscan"]:
        cfg["n_cols_prescan"] = 0      # windowed readout: no prescan either
    return ccd_cols, cfg, ncol, nsbin


def gain_priors(j, names, args):
    """Gain prior per extension: --gain if given, else the reference-JSON CalibrationProcess gain.

    The prior is used to check the measured gain (--prior-tol) and as the gain when
    the peaks are not resolved. --gain is either one number for all extensions or
    "EXT1=1.32,EXT2=1.30". j: reference config; names: ["EXT1", ...].
    Returns {ext name: gain in ADU/e-} (1.0 for extensions missing in the reference).
    """
    if args.gain:
        if "=" in args.gain:
            return {k: float(v) for k, v in (kv.split("=") for kv in args.gain.split(","))}
        return {n: float(args.gain) for n in names}
    gain_tmpl = j["process"].get("CalibrationProcess", {}).get("gain", {})
    if isinstance(gain_tmpl, dict):
        return {n: float(gain_tmpl.get(n, 1.0)) for n in names}
    return {n: float(gain_tmpl) for n in names}


def print_report(results, outliers, dc, output):
    """Per-extension summary of what was measured and configured.

    Columns: resolved, gain (ADU/e-), sigma (intrinsic noise, ADU), sig_row (noise
    as the fit sees it, ADU), sig[e] (noise in e-), lam[e] (dark current, e-/pix),
    rows (rows used in the fit), ovs>=1e (fraction of fit rows with a charged
    overscan), badcols, npk (model peaks). Then warnings for outlier CCDs.
    dc: the FitDarkCurrentProcess settings, or None (--per-ext: no shared window);
    output: printed as the destination.
    """
    print(f"{'ext':5} {'resolved':8} {'gain':>6} {'sigma':>7} {'sig_row':>7} {'sig[e]':>7} {'lam[e]':>7} "
          f"{'rows':>5} {'ovs>=1e':>7} {'badcols':>7} {'npk':>4}")
    for r in results:
        lam = f"{r['lam']:.2f}" if r["lam"] is not None else "-"
        print(f"{r['ext']:5} {str(r['resolved']):8} {r['gain']:6.3f} {r['sigma']:7.3f} {r['sigma_raw']:7.3f} "
              f"{r['sigma'] / r['gain']:7.3f} {lam:>7} {r['row_end']:5d} {r['ovs_charged_frac_fit']:7.2f} "
              f"{len(r['bad_cols']):7d} {r['fit']['n_peaks']:4d}")
        if r["bad_cols"]:
            print(f"      masked columns: {compress_ranges(r['bad_cols'])}")
        if r.get("fallback"):
            print(f"      fallback: {r['fallback_note']}")
        if not r["resolved"]:
            print(f"      peaks not resolved: gain held at {r['gain']:.3f} ({r['gain_source']}), DC capped")
        elif r.get("poisson_ratios"):
            rat = " ".join(f"{v:.2f}" for v in r["poisson_ratios"][:8])
            print(f"      data/Poisson for peaks 0,1,2..: {rat}  -> {r['k_poisson']} Poisson-like peaks "
                  f"(lambda from 0e-/1e-: {r['lam01']:.3f} e-)")
    for k in outliers:
        fk = next(r["fit"] for r in results if r["ext"] == k)
        print(f"  WARNING {k}: its charge distribution is much wider than the others "
              f"(window {fk['x_min']:.1f}..{fk['x_max']:.1f} ADU); "
              + ("its peaks are not resolved, " if dc is None else "it is fit in the shared window, ")
              + "so its result is not reliable. Check this CCD (light, high DC, bad channel).")
    k = next((r["k_fit"] for r in results if r["resolved"]), None)
    if k is not None:
        print(f"peaks fitted: {k} (0e- to {k - 1}e-)")
    if dc is not None:
        print(f"fit window [{dc['x_min']}, {dc['x_max']}] ADU, bin {dc['binning_size']} ADU -> {output}")


def analyse_all_extensions(hdul, img, j, cfg, args):
    """Run analyse_extension on every extension listed in the reference.

    Channels without a CCD are reported and removed from img["extensions"] (in
    place). Exits if no extension has a working CCD.
    hdul: open FITS file; img: j["input"]["image"]; j: config (for the gain priors);
    cfg: analysis geometry from geometry().
    Returns the result dicts of the working extensions.
    """
    exts = img.get("extensions", [1])
    names = [f"EXT{e}" for e in exts]
    gp = gain_priors(j, names, args)
    results, dead = [], []
    for e, name in zip(exts, names):
        r = analyse_extension(hdul[e].data, name, cfg, args, gp.get(name, 1.0))
        if r["dead"]:
            dead.append((e, r))
            continue
        results.append(r)

    # ---- channels without a CCD are dropped from the processing
    for e, r in dead:
        print(f"  {r['ext']}: no CCD signal (active region identical to overscan, noise "
              f"{r['sigma']:.3f} ADU, {r['frac_hi']:.1e} of pixels with charge) -> removed from 'extensions'")
    if not results:
        sys.exit("No extension shows a working CCD; nothing to configure.")
    if dead:
        img["extensions"] = [e for e in exts if e not in [d[0] for d in dead]]
    return results


def write_single_json(j, results, outliers, args):
    """Finish and write the single JSON (mode without --per-ext) and print the summary.

    Adds the plot range and the ClusterFinder skip for unresolved outlier CCDs
    (unless --cluster-all), writes args.output and <output>_measured.json, then
    prints print_report and notes on ClusterFinder and the plot range.
    """
    set_plot_range(j, results, args.plot_peaks)
    no_cls = [] if args.cluster_all else [e for e in outliers if not any(r["ext"] == e and r["resolved"] for r in results)]
    skip_clusters(j, no_cls)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(j, fh, indent=4)
    write_measured(re.sub(r"\.json$", "", args.output) + "_measured.json", args.infile, results, outliers)

    print_report(results, outliers, j["process"]["FitDarkCurrentProcess"], args.output)
    if args.cluster_threshold_e == "auto":
        for line in describe_cluster_thresholds(auto_cluster_thresholds(results)):
            print(line)
    elif args.cluster_threshold_e is not None:
        pix, seed = args.cluster_threshold_e
        print(f"ClusterFinder: pixels > {pix:g} e-, seed > {seed:g} e- (spr_threshold sigma = 1 e-)")
    if no_cls:
        print(f"ClusterFinder: no clusters searched in {', '.join(no_cls)} (unresolved outlier; "
              f"use --cluster-all to keep it)")
    if "x_max_plot" in j["process"]["FitDarkCurrentProcess"]:
        print(f"DC fit plot drawn up to {j['process']['FitDarkCurrentProcess']['x_max_plot']} ADU "
              f"(about {args.plot_peaks} e-), fit only up to x_max")


def main():
    """Command-line entry point: measure the image and write the config(s).

    Order: load the reference JSON (--reference or the built-in one); geometry from
    the header; analyse every extension; choose the number of peaks and the per-CCD
    fit settings; set pedestal, DC fit and (optionally) ClusterFinder settings;
    optionally --tune with panaSKImg; then write either one JSON (+ _measured.json)
    or, with --per-ext, one JSON per CCD, a run script and _measured.json.
    """
    args = parse_args()

    j = json.load(open(args.reference)) if args.reference else json.loads(REFERENCE_JSON)
    img = j["input"]["image"]
    hdul = fits.open(args.infile)
    h0 = hdul[0].header
    nskips = h0.get("NDCM", h0.get("NSKIPS", "?"))

    # ---- geometry
    ccd_cols, cfg, ncol, nsbin = geometry(h0, hdul, img, args)
    img["ccd_cols"] = ccd_cols
    print(f"{os.path.basename(args.infile)}: {nskips} skips, {ncol} cols (binning {nsbin}), active cols "
          f"{cfg['n_cols_prescan']}-{cfg['ccd_cols'] - 1}, overscan {cfg['ccd_cols']}-{ncol - 1} "
          f"(ccd_cols={ccd_cols})")

    # ---- analyse every extension (channels without a CCD are dropped)
    results = analyse_all_extensions(hdul, img, j, cfg, args)

    # reference + geometry + extensions, before any process settings (for --per-ext)
    base = copy.deepcopy(j)

    # ---- how many peaks to fit, then the per-extension fit settings
    k_fit = choose_peaks(results, args)
    for r in results:
        r["fit"] = fit_settings(r)

    # ---- write the process settings
    set_pedestal(j, results)
    outliers = set_dark_current_fit(j, results)
    if args.cluster_threshold_e is not None:
        apply_cluster_thresholds(j, results, args.cluster_threshold_e)
    if args.tune and not args.per_ext:         # --per-ext tunes every CCD on its own
        work, src = prepare_workdir(os.path.abspath(args.infile), args.tune_dir)
        j, _ = tune_pedestal(j, src, work, results, args.panaskimg)
        j = refine_peaks(j, src, work, results, args)
        if args.tune_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    if args.per_ext:
        print_report(results, outliers, None, args.output)
        print(f"one JSON per CCD in {args.output}:")
        write_per_ext(base, results, outliers, args)
        stem = os.path.basename(args.infile).split(".")[0]
        write_measured(os.path.join(args.output, f"{stem}_measured.json"), args.infile, results, outliers)
        return
    write_single_json(j, results, outliers, args)


if __name__ == "__main__":
    main()
