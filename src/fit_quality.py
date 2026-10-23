#!/usr/bin/env python3
"""Fit quality of panaSKImg's dark-current fits (FitDarkCurrentProcess).

The parameter errors ROOT reports are statistical and assume the fit model is right.
When chi2/ndf > 1 the model does not fully describe the data and those errors are too
small, so they are scaled by k = sqrt(chi2/ndf) (k = 1 if chi2/ndf <= 1).

A small error does not mean a correct result: fitting more peaks shrinks the error of
the dark current but biases it (peaks above ~1 e- hold non-Poisson charge). The bias is
measured against an independent estimate, lambda from the counts of 0e- and 1e- pixels
(lam01, written by autoconfig_panaSKImg.py into <json>_measured.json).

Total error of the dark current = sqrt((k * stat. error)^2 + bias^2).
Score (used by autoconfig --tune, lower is better) = quadrature sum of the relative
scaled errors of gain and noise and the relative total error of the dark current.

As a script, it reports every fit in a panaSKImg output folder (needs the fits saved as
ROOT files: "format_figures": ["pdf", "root"] in FitDarkCurrentProcess, and --save-plots):

    python fit_quality.py <panaSKImg output folder> [--ref <folder with *_measured.json>] [--csv out.csv]

Place in the pipeline: a library for autoconfig_panaSKImg.py (--tune scoring),
gain_dc_split.py (which gain to keep) and bin/run_WADERs (fit_quality.csv, via SIDE_RE and
read_side_fit), and a stand-alone report on any panaSKImg output folder.

How to read this file:
  read_side_fit  fit parameters, chi2 and ndf from panaSKImg's stored fit (ROOT file)
  quality        the numbers above -> gain, noise, dark current with raw/scaled errors,
                 bias against lam01, total error and score
  fmt, load_refs one-line summary; reference dark currents from *_measured.json
  main           command line
Units: gain in ADU/e-, noise in e-, dark current (DC) in e-/pix (per pixel and image).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import threading

# PyROOT is not safe when several threads open files at once: every ROOT file read
# in these scripts goes through this lock (also used by autoconfig_panaSKImg.py)
ROOT_LOCK = threading.RLock()

# Parameters of panaSKImg's fit function, in its order: normalisation, position of the
# 0e- peak [ADU], peak width (readout noise) [ADU], Poisson mean lambda [e-/pix], gain [ADU/e-]
PARAMS = ["norm", "mu", "sigma", "lam", "gain"]
# file name of a stored fit -> (image stem, "EXTn")
SIDE_RE = re.compile(r"(.*)_FitDarkCurrentProcess_side(EXT\d+)\.root$")


def read_side_fit(path, ext):
    """Fit parameters (value, error), chi2 and ndf from a *_FitDarkCurrentProcess_sideEXTn.root.

    ext: "EXTn". Returns {param: (value, error) for PARAMS, "chi2": float, "ndf": int}, or
    None if the file cannot be opened or holds no fit. Thread-safe (ROOT_LOCK).
    """
    with ROOT_LOCK:
        return _read_side_fit(path, ext)


def _read_side_fit(path, ext):
    """read_side_fit() without the lock: histogram "pcd" and its function "fitfunc_<ext>"."""
    import ROOT
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return None
    h = f.Get("pcd")
    fn = h.GetFunction(f"fitfunc_{ext}") if h else None
    if not fn:
        f.Close()
        return None
    out = {n: (fn.GetParameter(i), fn.GetParError(i)) for i, n in enumerate(PARAMS)}
    out["chi2"], out["ndf"] = fn.GetChisquare(), fn.GetNDF()
    f.Close()
    return out


def quality(fit, lam_ref=None):
    """Errors of gain [ADU/e-], noise [e-] and dark current [e-/pix], raw and scaled.

    fit: dict from read_side_fit(); lam_ref: independent dark current lam01 [e-/pix] or None.
    The fit is in ADU, so noise = sigma / gain and dark current = lambda / gain, with the
    relative errors added in quadrature. Each error is also given scaled by
    k = sqrt(max(chi2/ndf, 1)) (keys *_err_s): a chi2/ndf above 1 means the model misses
    part of the data, and scaling by k is the usual way to make the errors account for it.
    dc_bias = dc - lam_ref is the systematic shift of the fitted dark current (e.g. from
    non-Poisson charge in the higher peaks); dc_total = sqrt((k * dc_err)^2 + bias^2), or
    k * dc_err without a reference. score: see the module docstring.
    Returns a flat dict (also the CSV columns of main()).
    """
    g, eg = fit["gain"]
    s, es = fit["sigma"]
    lam, el = fit["lam"]
    chi2ndf = fit["chi2"] / max(fit["ndf"], 1)
    k = math.sqrt(max(chi2ndf, 1.0))
    noise, dc = s / g, lam / g
    e_noise = noise * math.hypot(es / s, eg / g)
    e_dc = dc * math.hypot(el / lam, eg / g) if lam > 0 else el / g
    q = dict(chi2ndf=chi2ndf, ndf=fit["ndf"], k=k, mu=fit["mu"][0],
             gain=g, gain_err=eg, gain_err_s=k * eg,
             noise=noise, noise_err=e_noise, noise_err_s=k * e_noise,
             dc=dc, dc_err=e_dc, dc_err_s=k * e_dc,
             dc_ref=lam_ref, dc_bias=None)
    if lam_ref is not None:
        q["dc_bias"] = dc - lam_ref
        q["dc_total"] = math.hypot(k * e_dc, dc - lam_ref)
    else:
        q["dc_total"] = k * e_dc
    scale_dc = max(dc, lam_ref or 0.0, 0.01)
    q["score"] = math.hypot(k * eg / g, q["dc_total"] / scale_dc, k * e_noise / noise)
    return q


def fmt(q):
    """One-line summary of quality()."""
    bias = f" bias {q['dc_bias']:+.3f} (ref {q['dc_ref']:.3f})" if q["dc_bias"] is not None else ""
    return (f"chi2/ndf {q['chi2ndf']:6.2f} | gain {q['gain']:.4f} +- {q['gain_err_s']:.4f} | "
            f"noise {q['noise']:.4f} +- {q['noise_err_s']:.4f} e- | "
            f"DC {q['dc']:.4f} +- {q['dc_err_s']:.4f} e-/pix{bias} | total DC err {q['dc_total']:.4f}")


def load_refs(folders):
    """{image stem: {ext: lam01}} from the *_measured.json files in folders."""
    refs = {}
    for d in folders:
        for p in glob.glob(os.path.join(d, "*_measured.json")):
            m = json.load(open(p))
            refs.setdefault(m["image"], {}).update(
                {e: v["lam01"] for e, v in m["extensions"].items() if v.get("lam01") is not None})
    return refs


def main():
    """Command line: print fmt() for every stored fit in the folder and write the CSV
    (one row per image and extension, the columns of quality())."""
    ap = argparse.ArgumentParser(description="Report chi2/ndf, scaled errors and dark-current bias "
                                             "of panaSKImg dark-current fits.")
    ap.add_argument("folder", help="panaSKImg output folder (with *_FitDarkCurrentProcess_sideEXTn.root)")
    ap.add_argument("--ref", nargs="*", default=None,
                    help="folder(s) with autoconfig *_measured.json (for the bias); default: the "
                         "output folder and its json/ subfolder")
    ap.add_argument("--csv", default=None, help="CSV file to write (default: <folder>/fit_quality.csv)")
    args = ap.parse_args()

    refdirs = args.ref if args.ref is not None else [args.folder, os.path.join(args.folder, "json")]
    refs = load_refs(refdirs)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.folder, "*_FitDarkCurrentProcess_side*.root"))):
        m = SIDE_RE.match(os.path.basename(path))
        if not m:
            continue
        stem, ext = m.group(1), m.group(2)
        fit = read_side_fit(path, ext)
        if fit is None:
            print(f"{stem} {ext}: no fit stored")
            continue
        q = quality(fit, refs.get(stem, {}).get(ext))
        print(f"{stem} {ext}: {fmt(q)}")
        rows.append(dict(image=stem, ext=ext, **q))
    if not rows:
        print("no *_FitDarkCurrentProcess_sideEXTn.root found (run panaSKImg with --save-plots and "
              '"format_figures": ["pdf", "root"] in FitDarkCurrentProcess)')
        return
    out = args.csv or os.path.join(args.folder, "fit_quality.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")
    if not refs:
        print("(no *_measured.json found: dark-current bias not computed; pass --ref <json folder>)")


if __name__ == "__main__":
    main()
