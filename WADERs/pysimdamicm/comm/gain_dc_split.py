#!/usr/bin/env python3
# Copied from comm_scripts/src/gain_dc_split.py by utils/sync_to_waders.py (imports made package-relative).
# Edit the comm_scripts file and re-run the sync, not this copy.
"""Measure the gain from a many-peak fit and the dark current from a 2-peak fit.

Place in the pipeline (bin/run_WADERs): raw image -> [this script, which calls
autoconfig_panaSKImg.py twice and panaSKImg fit-only] -> one final panaSKImg JSON per CCD
-> full panaSKImg run (clusters) -> optional gain-vs-charge correction (gain_vs_charge.py)
-> MakeComposite. run_WADERs calls split_image() (per_ext=True), image_stem() and
held_quality() directly.

Why: fitting more electron peaks pins down the gain better (more peaks, more lever arm),
but biases the dark current, because peaks above ~1 e- hold non-Poisson charge
(clusters, dark current varying across the CCD). Fitting only the 0e-/1e- peaks gives
an unbiased dark current, but a less precise gain. This script takes each quantity
from the fit that measures it best:

  1. gain JSON : autoconfig with --peaks all (every populated peak). Run with panaSKImg
     (pedestal + DC fit only, ~15 s) -> gain +- error per extension (error scaled by
     sqrt(chi2/ndf), see fit_quality.py).
  2. dc JSON   : autoconfig default (Poisson-like peaks, usually 0e-/1e-). Also run
     (gain free), to compare.
  3. final JSON: the dc JSON with the gain held at the many-peak value, for every
     extension where that gain is more precise (after scaling) and agrees with the
     2-peak gain within 3 sigma. This is the JSON to use for the full analysis:
     CalibrationProcess and ClusterFinder then use that gain.

So two panaSKImg configurations are run first, and the final JSON can only be written
after them. With --run-full OUT the final JSON is also run with the full sequence.

With --per-ext, every JSON is made per CCD (autoconfig --per-ext), in the folders
gain/, dc/ and final/ (<image>_EXTn.json). run_WADERs uses this mode.

Caveat: if the gain changes with charge (non-linearity) the many-peak gain is an
average slope; such extensions fail the 3-sigma agreement test and keep their own gain.

Files written in <folder> (-o): autoconfig_gain.log, autoconfig_dc.log, <image>_measured.json
(autoconfig's 0e-/1e- dark-current reference, for fit_quality.py --ref),
<image>_gain_dc_summary.txt (fit table and gain choice), the JSONs (see above) and the
panaSKImg fit runs in runs/.

Usage:
    python gain_dc_split.py <image> -o <folder> [--per-ext] [--run-full <analysis folder>] [-j N]
Needs PyROOT and panaSKImg; uses autoconfig_panaSKImg.py and fit_quality.py (same folder).

How to read this file:
  helpers      image_stem, autoconfig (run the autoconfig script), fit_run / run_parallel
               (panaSKImg fit-only runs), sync_from_tuned (--tune)
  gain choice  choose_gain (many-peak or 2-peak gain?), hold_gain (fix it in a JSON),
               held_quality / report (summary table)
  driver       split_image() and its steps (_make_jsons, _write_measured, _first_fits,
               _final_configs, _write_finals); main() for the command line
"""
import argparse
import concurrent.futures as cf
import copy
import glob
import json
import math
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from . import autoconfig_panaSKImg as A  # noqa: E402
from . import fit_quality as FQ          # noqa: E402

AUTOCONFIG = os.path.join(SCRIPT_DIR, "autoconfig_panaSKImg.py")


# =============================================================================
# helpers: autoconfig and panaSKImg runs
# =============================================================================
from .naming import image_stem  # noqa: E402,F401  (image name -> output prefix; used here and by run_WADERs)


def autoconfig(infile, out, extra, log):
    """Run autoconfig_panaSKImg.py; its report goes to log. Returns True if it worked."""
    p = subprocess.run([sys.executable, "-m", __package__ + ".autoconfig_panaSKImg", infile, "-o", out] + list(extra),
                       capture_output=True, text=True)
    with open(log, "w") as fh:
        fh.write(p.stdout + p.stderr)
    return p.returncode == 0 and os.path.exists(out)


def fit_run(cfg, src, work, name, panaskimg):
    """panaSKImg pedestal + DC fit only; {ext: fit dict from the stored ROOT fit} ({} if it failed).

    A fit dict holds (value, error) of norm, mu, sigma, lam, gain and chi2, ndf (see
    fit_quality.read_side_fit). The run's files are in work/<name>/.
    """
    res = A.run_candidate(cfg, src, work, name, panaskimg, save_hist=True)
    if not res:
        print(f"  WARNING: panaSKImg failed for '{name}', see {os.path.join(work, name, 'log.txt')}")
        return {}
    return {e: v["fit"] for e, v in res.items() if v.get("fit")}


# =============================================================================
# gain choice
# =============================================================================
def choose_gain(e, fit_g, fit_d, gain_cfg):
    """(gain, scaled error) from the many-peak fit if it should be used, else None; and a note.

    e: extension ("EXTn"); fit_g, fit_d: {ext: fit dict} of the many-peak and 2-peak fits;
    gain_cfg: FitDarkCurrentProcess of the many-peak JSON (for its gain limits). The
    many-peak gain is rejected if it sits at a fit limit, is not more precise than the
    2-peak gain after scaling both errors by sqrt(chi2/ndf), or differs from it by more than
    3 sigma (a sign that the gain changes with charge). Gain in ADU/e-.
    """
    if e not in fit_g:
        return None, "no many-peak fit"
    qg = FQ.quality(fit_g[e])
    g, eg = qg["gain"], max(qg["gain_err_s"], 1e-4)
    lo, hi = gain_cfg["gain_min"][e], gain_cfg["gain_max"][e]
    if min(g - lo, hi - g) < 1e-3 * g:
        return None, f"many-peak gain {g:.4f} sits at its limit"
    if e in fit_d:
        qd = FQ.quality(fit_d[e])
        # more precise after scaling by sqrt(chi2/ndf)? (a many-peak fit that describes
        # the data poorly is not)
        if eg >= qd["gain_err_s"]:
            return None, (f"many-peak gain {g:.4f} +- {eg:.4f} not more precise than "
                          f"2-peak gain {qd['gain']:.4f} +- {qd['gain_err_s']:.4f}")
        # do both agree? (else the gain changes with charge: non-linearity)
        diff, comb = abs(g - qd["gain"]), math.hypot(eg, qd["gain_err_s"])
        if diff > 3 * comb:
            return None, (f"many-peak gain {g:.4f} and 2-peak gain {qd['gain']:.4f} differ by "
                          f"{diff / comb:.1f} sigma (non-linearity?)")
    return (g, eg), f"gain held at {g:.4f} +- {eg:.4f} (many-peak fit)"


def hold_gain(cfg, e, g):
    """Hold extension e's gain at g in cfg (within +-0.001%).

    panaSKImg starts the gain at the middle of its limits, so the fit and then
    CalibrationProcess use this gain. Equal limits do not work: the fit returns gain = 0.
    """
    dcp = cfg["process"]["FitDarkCurrentProcess"]
    dcp["gain_min"][e], dcp["gain_max"][e] = round(g * (1 - 1e-5), 6), round(g * (1 + 1e-5), 6)


def report(stem, refs, fit_g, fit_d, fit_f, held):
    """Text table: each fit's chi2/ndf, gain, noise, dark current, bias, total error.

    refs: {ext: reference dark current lam01 [e-/pix]}; fit_g, fit_d, fit_f: {ext: fit dict}
    of the many-peak, 2-peak and held-gain fits (fit_f may be empty); held: {ext: (gain,
    error)} where the gain was held. Returns the table as one string.
    """
    lines = [stem, "",
             f"{'ext':5} {'fit':26} {'chi2/ndf':>8} {'gain [ADU/e-]':>18} {'noise [e-]':>16} "
             f"{'DC [e-/pix]':>16} {'bias':>7} {'total DC err':>12}"]
    for e in sorted(refs):
        for name, fits in (("all peaks (gain)", fit_g), ("Poisson peaks, gain free", fit_d),
                           ("Poisson peaks, gain held", fit_f)):
            if not fits or e not in fits:
                continue
            q = held_quality(fits[e], refs[e], held.get(e) if name.endswith("held") else None)
            lines.append(f"{e:5} {name:26} {q['chi2ndf']:8.2f} {q['gain']:9.4f} +- {q['gain_err_s']:.4f} "
                         f"{q['noise']:7.4f} +- {q['noise_err_s']:.4f} {q['dc']:7.4f} +- {q['dc_err_s']:.4f} "
                         f"{q['dc_bias']:+7.3f} {q['dc_total']:12.4f}")
        lines.append("")
    lines += ["reference DC (0e-/1e- pixel counts): " + ", ".join(f"{e} {refs[e]:.4f}" for e in sorted(refs)),
              "errors are scaled by sqrt(chi2/ndf); total DC err = sqrt(scaled error^2 + bias^2)"]
    return "\n".join(lines)


def held_quality(fit, ref, held):
    """fit_quality.quality(), with the held gain's error carried into gain and dark current.

    In a fit with the gain held, ROOT reports (almost) no gain error, although the held value
    has the many-peak fit's error. held = (gain, error) or None; with it, gain_err_s becomes
    that error and the dark current (lam / gain) gets its relative error added in quadrature
    (dc_err_s and dc_total updated). ref: reference dark current lam01 [e-/pix] or None.
    """
    q = FQ.quality(fit, ref)
    if held:
        g, eg = held
        q["gain_err_s"] = eg
        q["dc_err_s"] = math.hypot(q["dc_err_s"], q["dc"] * eg / g)
        q["dc_total"] = math.hypot(q["dc_err_s"], q["dc_bias"] or 0.0)
    return q


def run_parallel(tasks, src, work, panaskimg, jobs):
    """Run [(name, cfg)] fit-only in parallel; {name: fits}."""
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {name: ex.submit(fit_run, cfg, src, work, name, panaskimg) for name, cfg in tasks}
        return {name: f.result() for name, f in futs.items()}


def sync_from_tuned(gain_cfg, dc_cfg, exts):
    """After --tune: give the all-peaks JSON the tuned pedestal and zero-peak position.

    The tuning is done on the dc JSON; the gain fit must see the same pedestal-subtracted
    image, so its pedestal settings, mu limits and lower window edge are copied (the
    upper edge is shifted by the same amount as the zero peak).
    """
    gp, dp = gain_cfg["process"], dc_cfg["process"]
    gp["PedestalSubtractionProcess"] = copy.deepcopy(dp["PedestalSubtractionProcess"])
    g, d = gp["FitDarkCurrentProcess"], dp["FitDarkCurrentProcess"]
    shifts = []
    for e in exts:
        if e in g["mu_min"] and e in d["mu_min"]:
            shifts.append(0.5 * (d["mu_min"][e] + d["mu_max"][e]) - 0.5 * (g["mu_min"][e] + g["mu_max"][e]))
            g["mu_min"][e], g["mu_max"][e] = d["mu_min"][e], d["mu_max"][e]
    if shifts:
        g["x_min"] = d["x_min"]
        g["x_max"] = round(g["x_max"] + max(shifts), 2)


# =============================================================================
# driver: the steps of split_image
# =============================================================================
def _make_jsons(infile, outdir, stem, extra, per_ext, tune):
    """Step 1: run autoconfig twice (all peaks -> gain JSON(s), default -> dc JSON(s)).

    per_ext: one JSON per CCD in outdir/gain/ and outdir/dc/ (keyed "EXTn"), else one JSON
    per image (key "all"). With tune, the dc JSON(s) are tuned and the gain JSON(s) follow
    them (sync_from_tuned) and are rewritten. Raises RuntimeError if autoconfig fails.
    Returns ({key: gain cfg}, {key: dc cfg}, path of autoconfig's *_measured.json).
    """
    if per_ext:
        dirs = {k: os.path.join(outdir, k) for k in ("gain", "dc", "final")}
        ok = (autoconfig(infile, dirs["gain"], ["--per-ext", "--peaks", "all"] + extra,
                         os.path.join(outdir, "autoconfig_gain.log"))
              and autoconfig(infile, dirs["dc"], ["--per-ext"] + (["--tune"] if tune else []) + extra,
                             os.path.join(outdir, "autoconfig_dc.log")))
        if not ok:
            raise RuntimeError(f"autoconfig failed for {stem}, see the logs in {outdir}")
        for k in ("gain", "dc"):              # run scripts made by autoconfig are not used here
            for f in glob.glob(os.path.join(dirs[k], "run_*.sh")):
                os.remove(f)
        os.makedirs(dirs["final"], exist_ok=True)
        meas_path = os.path.join(dirs["dc"], f"{stem}_measured.json")
        cfg_g = {os.path.basename(p)[len(stem) + 1:-5]: json.load(open(p))
                 for p in glob.glob(os.path.join(dirs["gain"], f"{stem}_EXT*.json"))}
        cfg_d = {os.path.basename(p)[len(stem) + 1:-5]: json.load(open(p))
                 for p in glob.glob(os.path.join(dirs["dc"], f"{stem}_EXT*.json"))}
    else:
        J = lambda tag: os.path.join(outdir, f"{stem}_{tag}.json")
        ok = (autoconfig(infile, J("gain"), ["--peaks", "all"] + extra, os.path.join(outdir, "autoconfig_gain.log"))
              and autoconfig(infile, J("dc"), (["--tune"] if tune else []) + extra,
                             os.path.join(outdir, "autoconfig_dc.log")))
        if not ok:
            raise RuntimeError(f"autoconfig failed for {stem}, see the logs in {outdir}")
        meas_path = J("dc").replace(".json", "_measured.json")
        cfg_g, cfg_d = {"all": json.load(open(J("gain")))}, {"all": json.load(open(J("dc")))}

    if tune:
        for key in cfg_d:
            if key in cfg_g:
                sync_from_tuned(cfg_g[key], cfg_d[key], [key] if per_ext else list(cfg_d[key]["process"]
                                ["FitDarkCurrentProcess"]["mu_min"]))
        for key, cfg in cfg_g.items():          # keep the files in gain/ = what was fitted
            path = (os.path.join(dirs["gain"], f"{stem}_{key}.json") if per_ext else J("gain"))
            with open(path, "w") as fh:
                json.dump(cfg, fh, indent=4)
    return cfg_g, cfg_d, meas_path


def _write_measured(outdir, stem, meas_path):
    """Step 2: copy autoconfig's measurements to outdir/<stem>_measured.json.

    Returns ({ext: lam01}, set of skipped exts). lam01 is the dark current [e-/pix] from the
    counts of 0e- and 1e- pixels, the reference for the dark-current bias; skipped exts are
    CCDs autoconfig marked as flooded or unresolved.
    """
    measured = json.load(open(meas_path))["extensions"]
    # copy next to the summary, for fit_quality.py --ref
    with open(os.path.join(outdir, f"{stem}_measured.json"), "w") as fh:
        json.dump({"image": stem, "extensions": measured}, fh, indent=2)
    refs = {e: v["lam01"] for e, v in measured.items() if v.get("lam01") is not None and not v.get("skip")}
    skip = {e for e, v in measured.items() if v.get("skip")}
    return refs, skip


def _first_fits(refs, cfg_g, cfg_d, per_ext, src, work, panaskimg, jobs):
    """Step 3: panaSKImg fit-only runs of the gain and dc JSONs; returns (fit_g, fit_d) {ext: fit}."""
    # ---- gain (all peaks) and dark-current (Poisson peaks, gain free) fits
    if per_ext:
        tasks = [(f"{k}_{e}", cfgs[e]) for e in sorted(refs) for k, cfgs in (("gain", cfg_g), ("dc", cfg_d))
                 if e in cfgs]
    else:
        tasks = [("gain_all", cfg_g["all"]), ("dc_all", cfg_d["all"])]
    out = run_parallel(tasks, src, work, panaskimg, jobs)
    fit_g, fit_d = {}, {}
    for name, fits in out.items():
        (fit_g if name.startswith("gain") else fit_d).update(fits)
    return fit_g, fit_d


def _final_configs(refs, cfg_g, cfg_d, fit_g, fit_d, per_ext):
    """Step 4: final cfgs = dc cfgs with the gain held where the many-peak gain is better.

    Also sets FitDarkCurrentProcess "calibration" to the held gain (per CCD; for one JSON
    per image, the mean of the held gains). Returns ({key: final cfg}, {ext: (gain, error)},
    notes = one line per extension on the gain choice).
    """
    # ---- final JSON(s): gain held where the many-peak gain is better
    held, notes = {}, []
    finals_cfg = {k: copy.deepcopy(v) for k, v in cfg_d.items()}
    for e in sorted(refs):
        key = e if per_ext else "all"
        gcfg = cfg_g.get(key, {}).get("process", {}).get("FitDarkCurrentProcess")
        if gcfg is None:
            continue
        h, note = choose_gain(e, fit_g, fit_d, gcfg)
        notes.append(f"  {e}: {note}")
        if h:
            hold_gain(finals_cfg[key], e, h[0])
            held[e] = h
    for key, cfg in finals_cfg.items():
        if held and not per_ext:
            cfg["process"]["FitDarkCurrentProcess"]["calibration"] = \
                round(sum(g for g, _ in held.values()) / len(held), 5)
        elif per_ext and key in held:
            cfg["process"]["FitDarkCurrentProcess"]["calibration"] = round(held[key][0], 5)
    return finals_cfg, held, notes


def _write_finals(finals_cfg, outdir, stem, per_ext, skip):
    """Step 5: write the final JSON(s) (final/<stem>_EXTn.json or <stem>_final.json).

    Returns [(key, path, skip)], skip=True for a CCD autoconfig marked as unresolved.
    """
    finals = []
    for key, cfg in sorted(finals_cfg.items()):
        path = (os.path.join(outdir, "final", f"{stem}_{key}.json") if per_ext
                else os.path.join(outdir, f"{stem}_final.json"))
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=4)
        finals.append((key, path, key in skip))
    return finals


def split_image(infile, outdir, per_ext=False, panaskimg=None, autoconfig_args=(), work=None,
                final_fit=True, jobs=1, tune=False):
    """Make the gain, dc and final JSONs for one image (see module docstring).

    tune: run autoconfig --tune on the dc JSON(s) (pedestal method chosen by total fit
    error, window refined on panaSKImg's histogram); the gain JSON(s) follow it.
    Returns dict(stem, src, finals=[(ext, json path, skip)], held, refs, fits, text).
    finals: the JSONs to run; skip=True for a flooded/unresolved CCD (not worth running).

    Steps: autoconfig JSONs (_make_jsons) -> 0e-/1e- reference (_write_measured) -> gain and
    dc fits (_first_fits) -> gain choice and final JSONs (_final_configs, _write_finals) ->
    optional check fit of the final JSONs (final_fit) -> summary text, also written to
    outdir/<stem>_gain_dc_summary.txt. work: folder for the panaSKImg runs (default
    outdir/runs); autoconfig_args: extra arguments for autoconfig; jobs: parallel runs.
    The returned dict also has notes (gain choice per extension) and skipped (sorted list).
    """
    panaskimg = panaskimg or A.DEFAULT_PANASKIMG
    stem = image_stem(infile)
    os.makedirs(outdir, exist_ok=True)
    work = work or os.path.join(outdir, "runs")
    os.makedirs(work, exist_ok=True)
    work, src = A.prepare_workdir(os.path.abspath(infile), work)
    extra = list(autoconfig_args) + ["--panaskimg", panaskimg]

    cfg_g, cfg_d, meas_path = _make_jsons(infile, outdir, stem, extra, per_ext, tune)
    refs, skip = _write_measured(outdir, stem, meas_path)
    fit_g, fit_d = _first_fits(refs, cfg_g, cfg_d, per_ext, src, work, panaskimg, jobs)
    finals_cfg, held, notes = _final_configs(refs, cfg_g, cfg_d, fit_g, fit_d, per_ext)
    finals = _write_finals(finals_cfg, outdir, stem, per_ext, skip)

    # ---- optional check fit with the held gain
    fit_f = {}
    if final_fit:
        tasks = [(f"final_{k}", c) for k, c in finals_cfg.items() if per_ext is False or k in refs]
        for fits in run_parallel(tasks, src, work, panaskimg, jobs).values():
            fit_f.update(fits)

    text = "\n".join([report(stem, refs, fit_g, fit_d, fit_f, held), "gain choice:"] + notes
                     + ([f"skipped (unresolved outlier CCD): {', '.join(sorted(skip))}"] if skip else []))
    with open(os.path.join(outdir, f"{stem}_gain_dc_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    return dict(stem=stem, src=src, finals=finals, held=held, refs=refs, notes=notes, skipped=sorted(skip),
                fits=dict(gain=fit_g, dc=fit_d, final=fit_f), text=text)


def main():
    """Command line: split one image, print the summary and, with --run-full, run the final
    JSON(s) with the full panaSKImg sequence (log: OUT/<stem>_<key>_panaSKImg.log)."""
    ap = argparse.ArgumentParser(description="Gain from a many-peak fit, dark current from a 2-peak fit.")
    ap.add_argument("infile", help="raw (skip-averaged) image")
    ap.add_argument("-o", "--outdir", required=True, help="folder for the JSONs, the fit runs and the summary")
    ap.add_argument("--per-ext", action="store_true", help="one JSON per CCD (folders gain/, dc/, final/)")
    ap.add_argument("--run-full", default=None, metavar="OUT",
                    help="also run the final JSON(s) with the full panaSKImg sequence (clusters, ...) into OUT")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="panaSKImg runs in parallel (default 4)")
    ap.add_argument("--tune", action="store_true",
                    help="tune the dark-current JSON(s) with panaSKImg (autoconfig --tune): pedestal method "
                         "by total fit error, fit window from panaSKImg's own histogram (slower)")
    ap.add_argument("--autoconfig-args", default="",
                    help='extra arguments for autoconfig_panaSKImg.py, e.g. "--cluster-threshold-e 2 5"')
    ap.add_argument("--panaskimg", default=A.DEFAULT_PANASKIMG, help="panaSKImg script (default: found automatically)")
    args = ap.parse_args()
    if not args.panaskimg:
        ap.error("panaSKImg not found: activate its environment or give --panaskimg")

    r = split_image(args.infile, args.outdir, per_ext=args.per_ext, panaskimg=args.panaskimg,
                    autoconfig_args=args.autoconfig_args.split(), jobs=args.jobs, tune=args.tune)
    print(r["text"])
    print("final JSON(s): " + ", ".join(p for _, p, _ in r["finals"]))

    if args.run_full:
        os.makedirs(args.run_full, exist_ok=True)
        for key, path, skip in r["finals"]:
            if skip:
                continue
            cfg = json.load(open(path))
            cmd = [sys.executable, args.panaskimg, r["src"], "-j", path, "-o", args.run_full.rstrip("/") + "/",
                   "--save-plots"]
            if cfg["input"]["image"].get("ACM_multi_extension", False):
                cmd.append("--acm")
            if args.per_ext:
                cmd += ["--ext", key[3:]]
            print(f"full analysis ({key}) -> {args.run_full}")
            with open(os.path.join(args.run_full, f"{r['stem']}_{key}_panaSKImg.log"), "w") as fh:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
