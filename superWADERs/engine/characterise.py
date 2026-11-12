#!/usr/bin/env python3
"""Characterisation front end of the native pipeline: what kind of image is this, and which chain processes it?

Measured per file and per CCD BEFORE clustering (no cluster is formed here). Every item is a record
    {"value": ..., "method": "how it was measured", "confidence": "high" | "medium" | "low" | "none", ...}
and every fallback is appended to the record's "fallbacks" list (nothing is assumed silently).

Per file
  form        skip-resolved raw (NDCM / NSAMP > 1 and data NCOL x NSKIPS wide, or a 3-D cube) vs skip-averaged; std
              images (var_ / std_) are recognised and not processed. Raw files are compressed as panaSKImg's
              CompressSkipperProcess (mean and std over the skips, axis 1 = the samples of a pixel are consecutive in a
              row) into OUT/compressed/<stem>.fits (+ _std.fits); skip_samples() returns the per-skip values of given
              pixels (for an excursion veto).
  geometry    autoconfig's geometry (header PHYSCOL - NCOLPRE + PRESCAN, else --ccd-cols, else the reference JSON - the
              last one is flagged), prescan, overscan width, binning NPBIN x NSBIN, number of CCD extensions (ACM
              multi-extension), shape class (strip, small window, full width, binned, ...).
  group       readout-settings key (module/ACM, NROW, NCOL, NDCM, binning, NCOLPRE, EXPOSURE, file-name pattern,
              folder): images with the same key form one run setting (run-pooled dark charge, sibling gains).
Per CCD
  artefacts   first active column level (L2 boards: ~2000 / NDCM ADU) -> column masked when significant; saturation
              (pixels at the extension's maximum value, several of them / std 0 for raw data).
  gain        comb measurable (autoconfig: resolved and gain fitted from the comb). If not: --gain, --gain-file, the
              median comb gain of other images of the same run setting (siblings), and only then the reference JSON
              prior - always FLAGGED (confidence low / none) in the record.
  noise, dark charge, counting   read noise [e-], dark charge (Poisson-like peaks, else the rough mean charge),
              electron counting = sharp zero peak (noise < 0.3 gain).
  channel     'no CCD signal' by autoconfig: with a sharp zero peak this cannot be told from an ultra-low dark-charge
              CCD (no charge to see), so such CCDs are kept and flagged ('quiet'); others are skipped (logged).
  structure   rows: dark charge of the first vs last rows (glow / charge along the readout), from autoconfig's per-row
              measurements; columns: bad (hot) columns of autoconfig.
After calibration (decide_ccd, called by native_pipeline.process_image with clusterer='auto', still before
clustering): A_GMX's own background map, read noise, background spread (unmodelled test), core fill (crowded test).

Chain (decision table; ANYIMAGE.md):
  no CCD signal and not sharp                          -> skipped (logged)
  read noise >= 1 e-, or background spread >= 3 x read noise and >= 0.5 e-   -> 'unmodelled' (A_GMX's hysteresis on
                                                          the calibrated image, as adaptive_cluster)
  electron counting, core fill >= 0.10                 -> A_GMX crowded mode
  electron counting                                    -> A_GMX normal, fewe='auto' (Bayesian few-electron switch),
                                                          run-pooled dark charge (agmx_switch.prepare_run) when the
                                                          run setting has >= 2 images and the dark charge is <= POOL_LAM
  otherwise (no counting, noise < 1 e-)                -> threshold clusterer (thresholds 'auto' -> sigma0 rule)

Usage: python characterise.py IMAGES... [-o OUT] [--gain G | EXT1=G1,...] [--gain-file F] [--autoconfig-args "..."]
       -> OUT/characterisation/<stem>.json and a table on stdout.
"""
import argparse
import copy
import functools
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))              # superWADERs/engine
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import autoconfig_panaSKImg as A      # noqa: E402
import naming                         # noqa: E402

VERSION = "characterise-0.1 (2026-11-11)"
SKIP_KEYS = ("NDCM", "NSAMP", "NSKIPS", "ANSAMP")
COUNTING_SIG_OVER_GAIN = 0.3      # autoconfig --max-sig-over-gain: resolved peaks
NOISE_UNMODELLED_E = 1.0          # adaptive_cluster noise_max (physics audit 2026-10-25)
EXCESS_RATIO = 3.0                # adaptive_cluster excess_ratio
CROWD_FILL = 0.10                 # crowded.CROWD_FILL
POOL_LAM = 1.5e-2                 # e-/pix: run-pooled dark charge only below this (agmx_switch FC_AGMX: above it the
                                  # soft-candidate fraction is too high for the Bayesian finder anywhere)
FLOOD_E = 10.0                    # e-/pix: a CCD without resolved peaks and more mean charge than this is 'flooded'
FLOOD_FRAC = 0.30                 # post-calibration flood guard: > 30 % of the pixels above the 'unmodelled' pixel
                                  # threshold (4.5 e-) -> one cluster would cover the CCD (AMtest EXT2: killed, memory)
COL0_NSIG = 5.0                   # first-column artefact: |level| > 5 x noise / sqrt(rows) and > 0.2 e-


def item(value, method, confidence, **kw):
    """one characterisation item"""
    d = dict(value=value, method=method, confidence=confidence)
    d.update(kw)
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dumps(rec):
    return json.dumps(_jsonable(rec), indent=1)


# =============================================================================
# data form: skip-resolved raw vs skip-averaged
# =============================================================================
def _nskips(h0):
    for k in SKIP_KEYS:
        if k in h0:
            try:
                return int(h0[k]), k
            except (TypeError, ValueError):
                pass
    return 1, None


def detect_form(path):
    """form record of one file: kind 'averaged' | 'raw' | 'std' | 'unknown', nskips, layout. Reads headers and the
    shape of the first 2-D/3-D extension only."""
    from astropy.io import fits
    base = os.path.basename(path)
    with fits.open(path) as hd:
        h0 = hd[0].header
        shapes = [(i, h.data.shape) for i, h in enumerate(hd) if h.data is not None and np.ndim(h.data) in (2, 3)]
    nsk, key = _nskips(h0)
    ncol = h0.get("NCOL")
    if not shapes:
        return item("unknown", "no 2-D/3-D image extension", "high", nskips=nsk)
    sh = shapes[0][1]
    name_std = bool(re.match(r"^(var|std)_", base))
    if name_std:
        return item("std", "file name prefix var_/std_ (skip std / variance image of an averaged image)", "medium",
                    nskips=nsk, shape=sh, note="not a charge image: not processed")
    if len(sh) == 3:
        ax = [i for i, s in enumerate(sh) if s == nsk and nsk > 1]
        if ax:
            return item("raw", f"3-D extension {sh}, axis {ax[0]} = {key} = {nsk}", "high", nskips=nsk, shape=sh,
                        layout=f"cube_axis{ax[0]}")
        return item("unknown", f"3-D extension {sh} without an axis equal to the number of skips ({nsk})", "low",
                    nskips=nsk, shape=sh)
    if nsk > 1 and ncol and sh[1] == int(ncol) * nsk:
        return item("raw", f"data {sh[1]} columns = NCOL {ncol} x {key} {nsk} (panaSKImg layout: the skips of a pixel "
                    f"are consecutive along the row)", "high", nskips=nsk, shape=sh, layout="interleaved_axis1")
    if ncol and sh[1] == int(ncol):
        conf = "high" if nsk >= 1 else "medium"
        what = "single skip (NDCM = 1)" if nsk == 1 else f"mean of {nsk} skips"
        return item("averaged", f"data width = NCOL {ncol}; {key or 'no skip keyword'} = {nsk} -> {what}", conf,
                    nskips=nsk, shape=sh)
    if ncol is None:
        return item("averaged", f"no NCOL in the header: taken as skip-averaged (shape {sh})", "low", nskips=nsk,
                    shape=sh)
    return item("unknown", f"data width {sh[1]} is neither NCOL {ncol} nor NCOL x {nsk}", "low", nskips=nsk, shape=sh)


def compress_skips(raw, nskips, skip_start=0, skip_end=None, layout="interleaved_axis1"):
    """(mean, std, saturated) of one raw extension, as CompressSkipperProcess: image (rows, cols, skips), skips
    skip_start .. skip_end-1 (panaSKImg id_skip_start / id_skip_end), np.mean / np.std over them; saturated pixels =
    mean at the image maximum and std 0."""
    cube = _cube(raw, nskips, layout)[:, :, skip_start:skip_end]
    m = cube.mean(axis=2)
    s = cube.std(axis=2)
    sat = (m == m.max()) & (s == 0)
    return m, s, sat


def _cube(raw, nskips, layout):
    raw = np.asarray(raw, np.float64)
    if layout == "interleaved_axis1":
        return raw.reshape(raw.shape[0], raw.shape[1] // nskips, nskips)
    if layout.startswith("cube_axis"):
        return np.moveaxis(raw, int(layout[-1]), 2)
    raise ValueError(f"unknown raw layout {layout!r}")


@functools.lru_cache(maxsize=2)
def _raw_cube(path, ext):
    from astropy.io import fits
    form = detect_form(path)
    with fits.open(path) as hd:
        raw = np.asarray(hd[ext].data, np.float64)
    return _cube(raw, form["nskips"], form.get("layout", "interleaved_axis1"))


def skip_samples(path, ext, rows, cols):
    """per-skip values [ADU] of the given pixels of a skip-resolved raw file: array (n_pixels, n_skips). rows, cols:
    0-based pixel indices of the COMPRESSED image (same as the cluster files' pixels_y - 1, pixels_x - 1); ext: FITS
    extension number. For the excursion veto (a pixel whose charge comes from a few outlying skips)."""
    cube = _raw_cube(os.path.abspath(path), int(ext))
    return cube[np.asarray(rows, int), np.asarray(cols, int), :]


def write_compressed(path, out_dir, skip_start=0, skip_end=None):
    """compress a raw skip file: OUT/compressed/<stem>.fits (mean, header with NDCM = number of skips used and
    NATRAW = source file) and <stem>_std.fits (std). Returns (mean path, std path, record)."""
    from astropy.io import fits
    form = detect_form(path)
    stem = naming.image_stem(path)
    d = os.path.join(out_dir, "compressed")
    os.makedirs(d, exist_ok=True)
    pm, ps = os.path.join(d, f"{stem}.fits"), os.path.join(d, f"{stem}_std.fits")
    t0 = time.time()
    with fits.open(path) as hd:
        h0 = hd[0].header.copy()
        mh, sh, nsat = [fits.PrimaryHDU(header=h0)], [fits.PrimaryHDU(header=h0)], {}
        for i, h in enumerate(hd):
            if i == 0 or h.data is None or np.ndim(h.data) not in (2, 3):
                continue
            m, s, sat = compress_skips(h.data, form["nskips"], skip_start, skip_end, form.get("layout"))
            mh.append(fits.ImageHDU(m.astype(np.float64)))
            sh.append(fits.ImageHDU(s.astype(np.float32)))
            nsat[f"EXT{i}"] = int(sat.sum())
    used = (skip_end if skip_end else form["nskips"]) - skip_start
    for hh in (mh[0].header, sh[0].header):
        hh["NATRAW"] = (os.path.basename(path)[:60], "native: compressed from this skip file")
        hh["NSKIPRAW"] = (form["nskips"], "native: skips in the raw file")
        hh["NSTART"] = (skip_start + 1, "first skip used (1-based, as panaSKImg)")
        hh["NUSED"] = (used, "skips averaged")
        hh["NDCM"] = used
    fits.HDUList(mh).writeto(pm, overwrite=True)
    fits.HDUList(sh).writeto(ps, overwrite=True)
    rec = item(pm, f"CompressSkipperProcess mean/std over skips {skip_start}..{(skip_end or form['nskips']) - 1}",
               "high", std_image=ps, saturated_pixels=nsat, seconds=round(time.time() - t0, 2))
    return pm, ps, rec


# =============================================================================
# geometry and groups
# =============================================================================
def shape_class(nrow, ncol, npbin, nsbin):
    c = []
    if npbin > 1 or nsbin > 1:
        c.append(f"binned {npbin}x{nsbin}")
    if nrow < 20:
        c.append("strip (< 20 rows)")
    if ncol * nsbin >= 6144:
        c.append("full CCD width")
    if nrow * ncol < 100000:
        c.append("small window")
    return c or ["standard"]


def geometry_record(h0, hdul, args):
    ref = json.loads(A.REFERENCE_JSON)
    ccd_cols, geo, ncol, nsbin = A.geometry(h0, hdul, ref["input"]["image"], args)
    nrow = int(h0.get("NROW", hdul[1].header.get("NAXIS2", 0)))
    npbin = int(h0.get("NPBIN", 1) or 1)
    if args.ccd_cols is not None:
        meth, conf = "--ccd-cols", "high"
    elif all(k in h0 for k in ("PHYSCOL", "NCOLPRE", "PRESCAN")):
        meth, conf = (f"header PHYSCOL {h0['PHYSCOL']} - NCOLPRE {h0['NCOLPRE']} + PRESCAN {h0['PRESCAN']} "
                      f"(autoconfig.geometry)"), "high"
    else:
        meth, conf = f"reference JSON ccd_cols {ref['input']['image']['ccd_cols']} (no PHYSCOL/NCOLPRE/PRESCAN)", "low"
    n_ext = sum(1 for i, h in enumerate(hdul) if i > 0 and h.data is not None and np.ndim(h.data) >= 2)
    return item(dict(nrow=nrow, ncol=ncol, npbin=npbin, nsbin=nsbin, ncolpre=h0.get("NCOLPRE"),
                     prescan=h0.get("PRESCAN"), physcol=h0.get("PHYSCOL"), ccd_cols_unbinned=ccd_cols,
                     first_active_col=geo["n_cols_prescan"], first_overscan_col=geo["ccd_cols"],
                     overscan_cols=max(ncol - geo["ccd_cols"], 0), no_overscan=geo["no_overscan"],
                     id_row_start=geo["id_row_start"], n_ccd=n_ext, acm_multi_extension=n_ext > 1,
                     shape_class=shape_class(nrow, ncol, npbin, nsbin)), meth, conf), geo


def group_key(path, h0, shape):
    """readout-settings key: images with equal keys belong to one run setting"""
    stem = naming.image_stem(path)
    pattern = re.sub(r"_\d{8}_\d{6}_\d+$", "", stem)            # drop _<date>_<time>_<n>
    pattern = re.sub(r"\d+$", "", pattern)
    acm = naming.acm_id(stem)
    k = dict(acm=acm, pattern=pattern, folder=os.path.dirname(os.path.abspath(path)), shape=list(shape))
    for h in ("NROW", "NCOL", "NDCM", "NSAMP", "NPBIN", "NSBIN", "NCOLPRE", "EXPOSURE"):
        if h in h0:
            k[h] = h0[h]
    return json.dumps(k, sort_keys=True, default=str)


# =============================================================================
# per CCD
# =============================================================================
def first_column(data, geo, noise_adu, gain, nskips):
    """level of the first active column vs the rest of the row (per-row medians removed) [ADU]"""
    r0, a0, a1 = geo["id_row_start"], geo["n_cols_prescan"], geo["ccd_cols"]
    x = data[r0:, a0:a1].astype(float)
    if x.shape[1] < 4 or x.shape[0] < 2:
        return None
    x = x - np.median(x, axis=1, keepdims=True)
    lvl1 = float(np.median(x[:, 1]))
    lvl = float(np.median(x[:, 0])) - float(np.median(x[:, 1:4]))      # relative to its neighbours
    err = max(noise_adu, 1e-6) * 1.25 / np.sqrt(x.shape[0])
    sig = abs(lvl) > max(COL0_NSIG * err, 0.2 * (gain or 1.0))
    return dict(column=a0, level_adu=lvl, next_column_adu=lvl1, expected_L2_adu=2000.0 / max(nskips, 1),
                significant=bool(sig), n_sigma=abs(lvl) / err)


def saturation(data):
    mx = float(np.max(data))
    n = int(np.sum(data == mx))
    return dict(max_adu=mx, n_at_max=n, flagged=bool(n >= 3))


def ccd_record(data, name, geo, args, gain_prior, nskips, prior_source):
    """pre-calibration characterisation of one CCD (autoconfig measurements + artefacts)"""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        r = A.analyse_extension(data, name, geo, args, gain_prior)
    msgs = [l.strip() for l in buf.getvalue().splitlines() if l.strip()]
    rec = dict(ext=name, autoconfig_messages=msgs, fallbacks=[], warnings=[])
    rec["saturation"] = item(saturation(data), "pixels at the extension's maximum value (>= 3 -> flagged)", "medium")
    if r["dead"]:
        s = float(r.get("sigma", np.nan))
        sharp = gain_prior and s / gain_prior < COUNTING_SIG_OVER_GAIN
        rec["channel"] = item("quiet" if sharp else "no CCD signal",
                              "autoconfig.is_dead_channel: active region = overscan (no charge, same noise, no offset)"
                              + ("; zero peak sharp (noise %.3f x gain prior): an ultra-low dark-charge CCD looks the "
                                 "same -> kept" % (s / gain_prior) if sharp else ""),
                              "low" if sharp else "medium", overscan_noise_adu=s)
        if sharp:
            rec["fallbacks"].append("channel looks dead (no charge) but has a sharp zero peak: processed as an "
                                    "ultra-low dark-charge CCD (autoconfig --keep-dead)")
            rec["gain"] = item(None, "no charge seen: no comb", "none", source=None, comb_measurable=False,
                               prior=gain_prior, prior_source=prior_source)
            rec["noise"] = item(s / gain_prior, "overscan noise (autoconfig.is_dead_channel) / gain", "low", adu=s)
            rec["dark_charge"] = item(0.0, "no charged pixels seen (< dead-frac); measured after calibration", "low")
            rec["counting"] = item(True, "sharp zero peak (noise < 0.3 gain)", "low")
            c0 = first_column(data, geo, s, gain_prior, nskips)
            rec["first_column"] = item(c0, "median of the first active column minus the row medians", "medium")
            if c0 and c0["significant"]:
                rec["warnings"].append(f"first-column artefact: {c0['level_adu']:+.2f} ADU (L2 expectation "
                                       f"~2000/NDCM = {c0['expected_L2_adu']:.2f} ADU) -> column {c0['column']} masked")
        rec["_r"] = r
        return rec
    rec["channel"] = item("signal", "autoconfig.is_dead_channel", "high")
    gain = r.get("gain")
    comb = bool(r.get("resolved")) and r.get("gain_source") == "fit" and not r.get("fallback")
    if comb:
        rec["gain"] = item(gain, "comb spacing (autoconfig find_gain / refine_gain; refined later by the dark-current "
                           "fit)", "high", source="comb")
    else:
        why = ("comb not resolved (noise too large)" if not r.get("resolved") else
               "peaks sharp but too few charged pixels for a comb" if r.get("gain_held") else "comb fallback")
        rec["gain"] = item(None, why, "none", source=None, comb_measurable=False, prior=gain_prior,
                           prior_source=prior_source)
    noise_adu = float(r.get("sigma", np.nan))
    g_for_e = gain if gain else gain_prior
    noise_e = noise_adu / g_for_e if g_for_e else np.nan
    counting = bool(r.get("resolved")) and noise_e < COUNTING_SIG_OVER_GAIN
    rec["noise"] = item(noise_e, "autoconfig: comb residual width (resolved) / zero-peak or overscan width "
                        "(unresolved), in e- with the gain above" + ("" if comb else " (UNCONFIRMED gain)"),
                        "high" if comb else "low", adu=noise_adu)
    lam = r.get("lam01") if r.get("resolved") and r.get("lam01") is not None else r.get("lam")
    rec["dark_charge"] = item(lam, "Poisson-like peaks 0/1 e- (lam01)" if r.get("lam01") is not None else
                              "median pixel charge / gain (unresolved: rough mean charge)",
                              "high" if r.get("lam01") is not None else "low",
                              rows_median=r.get("lam_all_median"), rows_p90=r.get("lam_all_p90"))
    rec["counting"] = item(counting, f"resolved peaks and read noise < {COUNTING_SIG_OVER_GAIN} e- (autoconfig "
                           "--max-sig-over-gain)", "high" if comb else "medium")
    c0 = first_column(data, geo, noise_adu, g_for_e, nskips)
    rec["first_column"] = item(c0, "median of the first active column minus the row medians, vs its error", "medium")
    if c0 and c0["significant"]:
        rec["warnings"].append(f"first-column artefact: {c0['level_adu']:+.2f} ADU (L2 expectation ~2000/NDCM = "
                               f"{c0['expected_L2_adu']:.2f} ADU) -> column {c0['column']} masked")
    bad = list(r.get("bad_cols") or [])
    rec["columns"] = item(dict(n_bad=len(bad), bad_cols=bad[:50]), "autoconfig.find_bad_columns (hot / bad columns: "
                          "masked in the dark-current fit, still clustered)", "medium")
    rows = dict(fit_rows_end=r.get("row_end"), nrows=r.get("nrows"), lam_rows_median=r.get("lam_all_median"),
                lam_rows_p90=r.get("lam_all_p90"), lam_first_rows=lam)
    glow = None
    if lam and r.get("lam_all_median") and lam > 0:
        glow = r["lam_all_median"] / lam
    rows["median_over_first"] = glow
    rec["rows"] = item(rows, "autoconfig per-row dark charge: first (fitted) rows vs all rows; > 1.5 = charge growing "
                       "along the readout (glow / serial-register charge)", "medium",
                       flag=bool(glow and glow > 1.5))
    rec["_r"] = r
    return rec


# =============================================================================
# gains
# =============================================================================
def load_gain_file(path):
    """{acm or '*': {EXTn: gain}} from a JSON file: {"EXT1": 1.3, ...}, {"<acm>": {"EXT1": ...}}, or a native_summary
    .json (list of image summaries; median fitted gain per ACM and CCD of the resolved CCDs)."""
    j = json.load(open(path))
    out = {}
    if isinstance(j, list):                          # native_summary.json
        acc = {}
        for s in j:
            acm = naming.acm_id(s.get("stem", "")) or "*"
            for ext, c in (s.get("ccds") or {}).items():
                if c.get("gain") and c.get("resolved"):
                    acc.setdefault(str(acm), {}).setdefault(ext, []).append(float(c["gain"]))
        return {a: {e: float(np.median(v)) for e, v in d.items()} for a, d in acc.items()}
    if all(str(k).upper().startswith("EXT") for k in j):
        return {"*": {str(k).upper(): float(v) for k, v in j.items()}}
    for a, d in j.items():
        out[str(a)] = {str(k).upper(): float(v) for k, v in d.items()}
    return out


def parse_gain_arg(g):
    if g is None:
        return None
    if isinstance(g, dict):
        return {str(k).upper(): float(v) for k, v in g.items()}
    g = str(g)
    if "=" in g:
        return {k.strip().upper(): float(v) for k, v in (kv.split("=") for kv in g.split(","))}
    return {"*": float(g)}


def resolve_gains(records, gain=None, gain_file=None):
    """fill rec['ccds'][ext]['gain'] for CCDs without a comb, in this order: --gain, --gain-file, siblings (median
    comb gain of the same CCD in the other images of the same run setting), reference JSON prior. Each non-comb
    choice is logged in the CCD's 'fallbacks' and keeps a low / none confidence."""
    garg = parse_gain_arg(gain)
    gfile = load_gain_file(gain_file) if gain_file else {}
    sib = {}
    for rec in records:
        for ext, c in rec.get("ccds", {}).items():
            if c.get("gain", {}).get("source") == "comb":
                sib.setdefault((rec["group"]["value"], ext), []).append(c["gain"]["value"])
    for rec in records:
        acm = str(rec.get("acm") or "*")
        for ext, c in rec.get("ccds", {}).items():
            g = c.get("gain")
            if g is None or g.get("source") == "comb":
                if g is not None and garg and (garg.get(ext) or garg.get("*")):
                    ga = garg.get(ext) or garg.get("*")
                    g["argument_gain"] = ga
                    g["comb_minus_argument"] = g["value"] - ga
                continue
            if garg and (ext in garg or "*" in garg):
                val, src, meth = garg.get(ext, garg.get("*")), "argument", "--gain"
            elif gfile and (ext in gfile.get(acm, {}) or ext in gfile.get("*", {})):
                val = gfile.get(acm, {}).get(ext, gfile.get("*", {}).get(ext))
                src, meth = "file", f"--gain-file {gain_file}"
            elif sib.get((rec["group"]["value"], ext)):
                v = sib[(rec["group"]["value"], ext)]
                val, src = float(np.median(v)), "siblings"
                meth = f"median comb gain of {len(v)} other image(s) of the same run setting"
            else:
                val, src = g["prior"], "reference prior"
                meth = f"NO GAIN MEASURED OR GIVEN: {g['prior_source']} prior (unconfirmed; give --gain or --gain-file)"
            c["gain"] = item(float(val), f"{g['method']} -> {meth}", "low" if src != "reference prior" else "none",
                             source=src, comb_measurable=False, flag="GAIN NOT MEASURED ON THIS CCD")
            c["fallbacks"].append(f"gain from {src} ({float(val):.4f} ADU/e-): {g['method']}")
            # quantities in e- were computed with the prior: redo them with the chosen gain
            n_adu = c["noise"]["adu"]
            c["noise"]["value"] = n_adu / float(val)
            r = c.get("_r", {})
            if "resolved" in r:
                c["counting"]["value"] = bool(r.get("resolved")) and c["noise"]["value"] < COUNTING_SIG_OVER_GAIN
    return records


# =============================================================================
# chain plan (pre-calibration) and post-calibration decision
# =============================================================================
def plan_ccd(c, group_size, curve=None):
    """pre-calibration chain plan of one CCD (the clusterer is confirmed by decide_ccd after calibration)"""
    if c["channel"]["value"] == "no CCD signal":
        return dict(process=False, reason="no CCD signal (autoconfig)")
    r = c["_r"]
    try:
        import native_pipeline as NP
        ped = NP.autoconfig_pedestal(r) if not r.get("dead") else {"method": "median", "in_overscan": True}
    except Exception as e:                                     # noqa: BLE001  reported, not hidden
        ped = {"error": str(e)}
    noise = c.get("noise", {}).get("value")
    counting = bool(c.get("counting", {}).get("value")) if "counting" in c else True     # quiet CCD: sharp peak
    lam = c.get("dark_charge", {}).get("value")
    if c["channel"]["value"] == "quiet":
        lam = 0.0
    flooded = (not counting) and lam is not None and lam > FLOOD_E
    if flooded:
        clus = "threshold"
    elif noise is not None and noise >= NOISE_UNMODELLED_E:
        clus = "unmodelled"
    elif counting:
        clus = "agmx"
    else:
        clus = "threshold"
    pooled = bool(clus == "agmx" and group_size >= 2 and lam is not None and lam <= POOL_LAM)
    gsrc = c.get("gain", {}).get("source") or ("argument/file/prior" if c["channel"]["value"] == "quiet" else None)
    return dict(process=True,
                pedestal=dict(method=ped.get("method"), in_overscan=ped.get("in_overscan"),
                              lowest_peak=bool(ped.get("is_highDC")), settings="autoconfig set_pedestal"),
                calibration=dict(gain_source=gsrc, curve=curve,
                                 note="g(n) curve applied when given and reliable for this CCD, else linear gain"),
                clusterer=clus, fewe="auto" if clus == "agmx" else "off", fewe_pooled=pooled,
                flooded=bool(flooded),
                why=("flooded: no resolved peaks and %.1f e-/pix mean charge > %g -> threshold clusterer with the sigma0 "
                     "rule (thresholds scale with the charge width, as panaSKImg)" % (lam, FLOOD_E) if flooded else
                     "read noise >= %.1f e- (not electron counting)" % NOISE_UNMODELLED_E if clus == "unmodelled" else
                     "electron counting -> A_GMX (crowded mode decided on the core fill after calibration)"
                     if clus == "agmx" else "peaks not resolved, noise < 1 e- -> threshold clusterer (sigma0 rule)"))


def decide_ccd(Q, valid, plan):
    """post-calibration decision (before clustering) on the image Q [e-] that A_GMX would cluster: A_GMX's background
    map, read noise, background spread and core fill. Returns (decision dict, bg, sigma) - bg and sigma are handed to
    adaptive_cluster so that it uses exactly these numbers."""
    import agent_methods as AM
    import crowded as CR
    t0 = time.time()
    fb = []
    Qv = np.where(valid, Q, 0.0)
    if (plan or {}).get("flooded"):
        # no background model / A_GMX on a flooded CCD (memory: one cluster would cover the CCD)
        above = float(np.mean(Qv[valid] > 4.5)) if valid.any() else 0.0
        return dict(chain="threshold", why=plan.get("why", "flooded"), frac_above_4p5e=above, crowded="off",
                    fewe="off", fallbacks=fb, seconds=round(time.time() - t0, 2)), None, None
    sigma = float(AM.read_noise(Qv, valid))
    try:
        bg = AM.background_map(Qv, valid)
        bg_meth = "AM.background_map"
    except Exception as e:                                    # noqa: BLE001
        bg = AM.simple_bg(Qv, valid, sigma)
        bg_meth = "AM.simple_bg"
        fb.append(f"background_map failed ({type(e).__name__}: {e}) -> simple_bg (as adaptive_cluster)")
    noise = float(np.median(bg.s[valid]))
    lam = float(np.median(bg.lam[valid]))
    unmod = (noise >= 0.5 and noise >= EXCESS_RATIO * sigma) or (AM.PHYS_FIX and sigma >= NOISE_UNMODELLED_E)
    fill = float(CR.core_fill(Qv, valid, bg)) if not unmod else float("nan")
    planned = (plan or {}).get("clusterer", "agmx")
    above = float(np.mean(Qv[valid] > 4.5)) if valid.any() else 0.0
    if planned == "threshold" and (plan or {}).get("flooded"):
        chain = "threshold"
        why = (plan or {}).get("why", "flooded")
    elif unmod and above > FLOOD_FRAC:
        chain = "threshold"
        why = (f"flood guard: {above:.0%} of the pixels above 4.5 e- (the 'unmodelled' pixel threshold) -> threshold "
               f"clusterer (sigma0 rule) instead of one CCD-sized cluster")
        fb.append(why)
    elif unmod:
        chain = "unmodelled"
        why = (f"read noise {sigma:.3f} e- >= {NOISE_UNMODELLED_E} e-" if sigma >= NOISE_UNMODELLED_E else
               f"background spread {noise:.3f} e- >= {EXCESS_RATIO} x read noise {sigma:.3f} e- (not dark current + "
               f"read noise)")
    elif planned == "threshold":
        chain = "threshold"
        why = "not electron counting (peaks not resolved) and read noise < 1 e-"
    elif fill >= CROWD_FILL:
        chain = "crowded"
        why = f"core fill {fill:.3f} >= {CROWD_FILL}"
    else:
        chain = "agmx"
        why = f"electron counting, core fill {fill:.3f} < {CROWD_FILL}"
    if chain in ("agmx", "crowded") and abs(fill - CROWD_FILL) < 0.02:
        fb.append(f"core fill {fill:.3f} within 0.02 of the crowded-mode cut {CROWD_FILL}: the mode may flip between "
                  f"images / CCDs of one run")
    if planned == "unmodelled" and chain != "unmodelled":
        fb.append(f"planned 'unmodelled' (autoconfig noise) but A_GMX's own read noise {sigma:.3f} e- / background "
                  f"spread {noise:.3f} e- do not meet the test -> '{chain}'")
    dec = dict(chain=chain, why=why, frac_above_4p5e=above, read_noise_e=item(sigma, "AM.read_noise on the calibrated image", "high"),
               background=item(dict(spread_e=noise, lam_e=lam), bg_meth, "high"),
               core_fill=item(fill, "crowded.core_fill (smoothed charge > max(60 e-, 5 noise)) on this background",
                              "high"),
               crowded="on" if chain == "crowded" else "off",
               fewe=("auto" if chain in ("agmx",) else "off"),
               fallbacks=fb, seconds=round(time.time() - t0, 2))
    if chain == "crowded":
        dec["fewe_note"] = "fewe off: agmx_switch leaves crowded-mode images to A_GMX"
    return dec, bg, sigma


# =============================================================================
# driver
# =============================================================================
def characterise(path, out_dir=None, autoconfig_args=(), exts=None, skip_start=0):
    """pre-calibration record of one file (gains of comb-less CCDs still unresolved: see characterise_many).
    Raw skip files are compressed into out_dir/compressed/ first; rec['process_path'] is the file to process."""
    from astropy.io import fits
    t0 = time.time()
    rec = dict(file=path, stem=naming.image_stem(path), version=VERSION, fallbacks=[], warnings=[])
    form = detect_form(path)
    rec["form"] = form
    proc = path
    if form["value"] == "raw":
        if out_dir is None:
            raise ValueError(f"{path}: skip-resolved raw file needs an output folder for the compressed image")
        proc, std, crec = write_compressed(path, out_dir, skip_start)
        rec["compressed"] = crec
        rec["skip_samples"] = dict(function="characterise.skip_samples(raw_path, ext, rows, cols)", raw_path=path)
    elif form["value"] in ("std", "unknown"):
        rec["process"] = False
        rec["reason"] = f"form '{form['value']}': {form.get('note', form['method'])}"
        rec["seconds"] = round(time.time() - t0, 2)
        return rec
    rec["process_path"] = proc
    args = A.parse_args([proc, "-o", os.devnull] + list(autoconfig_args))
    ref = json.loads(A.REFERENCE_JSON)
    with fits.open(proc) as hd:
        h0 = hd[0].header.copy()
        rec["geometry"], geo = geometry_record(h0, hd, args)
        data = [(i, np.asarray(h.data, np.float64)) for i, h in enumerate(hd)
                if i > 0 and h.data is not None and np.ndim(h.data) == 2]
    if rec["geometry"]["confidence"] == "low":
        rec["fallbacks"].append("geometry: " + rec["geometry"]["method"])
    nsk = int(h0.get("NDCM", form["nskips"]) or 1)          # raw: the number of skips averaged
    rec["acm"] = naming.acm_id(rec["stem"])
    rec["group"] = item(group_key(path, h0, data[0][1].shape if data else ()),
                        "module/ACM + readout settings (NROW, NCOL, NDCM, binning, NCOLPRE, EXPOSURE) + file-name pattern "
                        "+ folder", "medium")
    names = [f"EXT{n}" for n, _ in data]
    gp = A.gain_priors(ref, names, args)
    psrc = "--gain (autoconfig)" if args.gain else "reference JSON CalibrationProcess"
    rec["ccds"] = {}
    for (n, d), nm in zip(data, names):
        if exts and n not in exts:
            continue
        rec["ccds"][nm] = ccd_record(d, nm, geo, args, gp.get(nm, 1.0), nsk, psrc)
    rec["header"] = {k: h0.get(k) for k in ("NROW", "NCOL", "NDCM", "NSAMP", "NPBIN", "NSBIN", "NCOLPRE", "PRESCAN",
                                            "PHYSCOL", "EXPOSURE", "DATEINI") if k in h0}
    rec["process"] = True
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def characterise_many(paths, out_dir=None, gain=None, gain_file=None, autoconfig_args=(), exts=None, curve=None,
                      skip_start=0):
    """records of all files, gains resolved across the run settings, chain plans. Returns (records, groups)."""
    recs = []
    for p in paths:
        try:
            recs.append(characterise(p, out_dir, autoconfig_args, exts, skip_start))
        except Exception as e:                                # noqa: BLE001  reported per file
            import traceback
            recs.append(dict(file=p, stem=naming.image_stem(p), process=False,
                             reason=f"characterisation failed: {type(e).__name__}: {e}",
                             traceback=traceback.format_exc()))
    ok = [r for r in recs if r.get("process")]
    resolve_gains(ok, gain, gain_file)
    groups = {}
    for r in ok:
        groups.setdefault(r["group"]["value"], []).append(r["stem"])
    for r in ok:
        gsize = len(groups[r["group"]["value"]])
        r["group"]["n_images"] = gsize
        r["group"]["images"] = groups[r["group"]["value"]]
        for ext, c in r["ccds"].items():
            c["plan"] = plan_ccd(c, gsize, curve)
            if not c["plan"]["process"]:
                c["fallbacks"].append("CCD skipped: " + c["plan"]["reason"])
    return recs, groups


def public(rec):
    """record without the private autoconfig measurement copies"""
    out = copy.copy(rec)
    if "ccds" in rec:
        out["ccds"] = {e: {k: v for k, v in c.items() if not k.startswith("_")} for e, c in rec["ccds"].items()}
    return out


def gain_args_for(rec):
    """autoconfig --gain string for the CCDs whose gain is not from the comb (others keep the reference prior), and
    {ext number: gain} for process_image(gain_override=...)"""
    ref = json.loads(A.REFERENCE_JSON)
    names = list(rec.get("ccds", {}))
    args = A.parse_args([rec["process_path"], "-o", os.devnull])
    gp = A.gain_priors(ref, names, args)
    over, parts, any_ = {}, [], False
    for e in names:
        g = rec["ccds"][e].get("gain") or {}
        if g.get("source") not in (None, "comb"):
            over[int(e[3:])] = float(g["value"])
            parts.append(f"{e}={g['value']}")
            any_ = True
        else:
            parts.append(f"{e}={gp.get(e, 1.0)}")
    return (["--gain", ",".join(parts)] if any_ else []), over


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--gain", default=None, help="gain [ADU/e-] for CCDs without a comb: G or EXT1=G1,EXT2=G2")
    ap.add_argument("--gain-file", default=None, help="JSON {EXTn: G} / {acm: {EXTn: G}} / a native_summary.json")
    ap.add_argument("--autoconfig-args", default="")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import native_pipeline as NP
    paths = NP.expand_inputs(a.images)
    recs, groups = characterise_many(paths, a.out, a.gain, a.gain_file, a.autoconfig_args.split())
    for r in recs:
        print(f"{r['stem']}: form {r.get('form', {}).get('value')}"
              + ("" if r.get("process") else f"  NOT PROCESSED: {r.get('reason')}"))
        for e, c in r.get("ccds", {}).items():
            p = c.get("plan", {})
            print(f"  {e}: channel {c['channel']['value']}, gain {c.get('gain', {}).get('value')} "
                  f"({c.get('gain', {}).get('source')}), noise {c.get('noise', {}).get('value')}, "
                  f"DC {c.get('dark_charge', {}).get('value')}, counting {c.get('counting', {}).get('value')} -> "
                  f"{p.get('clusterer')} fewe {p.get('fewe')} pooled {p.get('fewe_pooled')}")
        if a.out:
            d = os.path.join(a.out, "characterisation")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"{r['stem']}.json"), "w").write(dumps(public(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
