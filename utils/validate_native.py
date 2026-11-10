#!/usr/bin/env python3
"""Validate the native pipeline (native_pipeline.py) against panaSKImg on images run_WADERs already processed.

Reference: a run_WADERs output folder (--run-dir) made with --gain-correction (g(n) applied inside panaSKImg) and
--clusterer both: root/ (WADERs clusters + g(n)), root_agmx/ (A_GMX + g(n)), json/<image>/final*/ (the exact
panaSKImg settings), gain_vs_charge/ (the g(n) curve). The raw images are listed in --images (a .txt).

Native configurations run (OUT/<config>/):
  mimic        panaSKImg's settings (pedestal masking, fit window/limits/masks, 10/30 sigma0 thresholds, gain rounded
               to 4 decimals for the calibration, cluster shapes from the energies before g(n)) with the NATIVE fit
  mimic_gain   as mimic, but with panaSKImg's fitted gain (from its info tree): isolates the cluster code, so every
               branch can be compared to the last bit
  native       the default: autoconfig settings + gain_dc_split gain choice with native fits, thresholds 'auto'
               (in electrons from the dark charge), g(n)
  native_sigma0  as native but with panaSKImg's 10/30 sigma0 thresholds (separates the effect of the thresholds)
  agmx_mimic   A_GMX with panaSKImg's settings, its gain and its 3.74 eV/e- (root_agmx/ was written that way)
  agmx         the default A_GMX run: native settings, g(n), epsilon(T) = 3.7486 eV/e-
The 'native' and 'agmx' configurations are run through the command line (-j 2) and timed end to end.

Per CCD it reports: pedestal per row, gain / noise / dark current / zero-peak position against the info tree,
cluster counts, cluster-by-cluster match (identical pixel sets), energies of matched clusters, the branches that
are not identical, the Gaussian-fit widths, and the Fe-55 K-alpha of front-side clusters (sigma_xy < 1, as
MakeComposite: chi2 Gaussian fit in 5.3-6.1 keV of a 50-bin 4.5-7 keV histogram, A_GMX files: agmx_cls 1 or 2).

    python validate.py --run-dir RUN --images LIST.txt -o OUT [--skip-runs] [--composite] [--time-panaskimg]
Writes OUT/validation.json and OUT/validation_tables.md (copied into docs/native_pipeline_validation.md by hand with the discussion).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")          # comm_scripts/src: native_pipeline and its modules
sys.path.insert(0, SRC)
PY = sys.executable
CONFIGS = ["mimic", "mimic_gain", "native", "native_sigma0", "agmx_mimic", "agmx"]
REF = {"mimic": "root", "mimic_gain": "root", "native": "root", "native_sigma0": "root",
       "agmx_mimic": "root_agmx", "agmx": "root_agmx"}
FIT_BRANCHES = {"fPosX", "fPosY", "fSTD_X", "fSTD_Y"}


# =============================================================================
# reading ROOT files
# =============================================================================
def read_file(path):
    """{branch: array or list of arrays} of clustersRec, info (as 'info:<name>') of one panaSKImg-format file"""
    import ROOT
    out = {}
    f = ROOT.TFile.Open(path)
    for tname, pre in (("clustersRec", ""), ("info", "info:")):
        t = f.Get(tname)
        if not t:
            continue
        t.GetEntry(0)
        for b in t.GetListOfBranches():
            n = b.GetName()
            v = getattr(t, n)
            cls = b.GetClassName()
            if cls.startswith("vector<vector"):
                out[pre + n] = [np.asarray(list(x), float) for x in v]
            elif cls.startswith("vector"):
                out[pre + n] = np.asarray(list(v), float)
            else:
                try:
                    out[pre + n] = float(v)
                except TypeError:
                    out[pre + n] = str(v)
    f.Close()
    if "Nclusters" not in out:
        out["Nclusters"] = 0.0
    return out


def keys(d):
    """cluster key = sorted (x, y) pixel tuple"""
    if not d.get("Nclusters"):
        return []
    return [tuple(sorted(zip(x.astype(int), y.astype(int)))) for x, y in zip(d["pixels_x"], d["pixels_y"])]


# =============================================================================
# Fe-55 K-alpha (front side), as MakeComposite
# =============================================================================
def sigma_xy(d):
    return np.sqrt(0.5 * (d["fSTD_X"] ** 2 + d["fSTD_Y"] ** 2))


def front_energies(d):
    if not d.get("Nclusters"):
        return np.zeros(0)
    E = d["Energy"]
    sel = (E > 4.5) & (E < 7.0) & (sigma_xy(d) < 1.0)
    if "agmx_cls" in d:
        sel &= np.isin(d["agmx_cls"], (1, 2))
    return E[sel]


def kalpha(E):
    """(mu, mu_err, sigma, n in window) of a chi2 Gaussian fit in [5.3, 6.1] keV of the 50-bin 4.5-7 keV histogram
    (Neyman errors, empty bins left out; start 5.9 / 0.10, sigma in [0.03, 0.40]) - MakeComposite's K-alpha fit"""
    from scipy.optimize import curve_fit
    h, e = np.histogram(E, 50, (4.5, 7.0))
    c = 0.5 * (e[1:] + e[:-1])
    m = (c > 5.3) & (c < 6.1) & (h > 0)
    if m.sum() < 4:
        return (np.nan, np.nan, np.nan, int(h[m].sum()))
    g = lambda x, a, mu, s: a * np.exp(-0.5 * ((x - mu) / s) ** 2)
    try:
        p, cov = curve_fit(g, c[m], h[m], p0=[h.max(), 5.9, 0.1], sigma=np.sqrt(h[m]), absolute_sigma=True,
                           bounds=([0, 5.3, 0.03], [np.inf, 6.1, 0.40]))
        return (float(p[1]), float(np.sqrt(cov[1, 1])), float(p[2]), int(h[m].sum()))
    except RuntimeError:
        return (np.nan, np.nan, np.nan, int(h[m].sum()))


# =============================================================================
# runs
# =============================================================================
def ref_gains(run_dir, stems):
    """{stem: {ext number: panaSKImg's fitted gain}} from the info trees of root/"""
    import ROOT
    out = {}
    for s in stems:
        out[s] = {}
        for p in glob.glob(os.path.join(run_dir, "root", f"panaSKImg_clustersRec_{s}_ext*.root")):
            n = int(p.rsplit("_ext", 1)[1][:-5])
            f = ROOT.TFile.Open(p)
            t = f.Get("info"); t.GetEntry(0)
            out[s][n] = float(getattr(t, f"ext{n}_gainext"))
            f.Close()
    return out


def _job(a):
    import native_pipeline as NP
    path, out, kw = a
    return NP.process_image(path, out, verbose=False, **kw)


def run_configs(images, run_dir, out, jobs=2):
    """the four in-process configurations (mimic, mimic_gain, native_sigma0, agmx_mimic); returns timing summaries"""
    import naming
    from multiprocessing import get_context
    curve = os.path.join(run_dir, "gain_vs_charge")
    jdir = os.path.join(run_dir, "json")
    stems = [naming.image_stem(p) for p in images]
    gains = ref_gains(run_dir, stems)
    tasks = []
    for p, s in zip(images, stems):
        tasks.append((p, os.path.join(out, "mimic"), dict(clusterer="threshold", panaskimg_json=jdir, curve=curve)))
        tasks.append((p, os.path.join(out, "mimic_gain"), dict(clusterer="threshold", panaskimg_json=jdir,
                                                               curve=curve, gain_override=gains[s])))
        tasks.append((p, os.path.join(out, "native_sigma0"), dict(clusterer="threshold", thresholds="sigma0",
                                                                  curve=curve)))
        tasks.append((p, os.path.join(out, "agmx_mimic"), dict(clusterer="agmx", panaskimg_json=jdir, curve=curve,
                                                               ev_per_e=3.74, gain_override=gains[s])))
    res = []
    with get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(_job, tasks):
            res.append(r)
            print(f"  done {r['stem']} {list(r['ccds'])} -> {os.path.dirname(r['files'][0]) if r['files'] else ''} "
                  f"({r['t']['total']:.1f} s)", flush=True)
    return res


def run_cli(images, run_dir, out, cfg, jobs=2):
    """'native' / 'agmx' through the command line, timed end to end (process start, imports, numba cache, ROOT)"""
    lst = os.path.join(out, "images.txt")
    open(lst, "w").write("".join(p + "\n" for p in images))
    cmd = [PY, os.path.join(SRC, "native_pipeline.py"), lst, "-o", os.path.join(out, cfg), "-j", str(jobs),
           "--curve", os.path.join(run_dir, "gain_vs_charge"), "--clusterer", "threshold" if cfg == "native" else "agmx"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    open(os.path.join(out, f"{cfg}.log"), "w").write(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"{cfg} failed, see {out}/{cfg}.log")
    return dict(config=cfg, wall_s=dt, cmd=" ".join(cmd))


def time_panaskimg(image, run_dir, out, ext=1, waders="/Users/dylanrosenmerkel/Documents/Software/WADERS_JHU"):
    """panaSKImg (the WADERS_JHU copy run_WADERs used: g(n) and A_GMX inside ClusterFinder) on one CCD with its final
    and final_agmx JSONs, as run_WADERs step 2 (--save-plots), timed. Returns {json kind: seconds}."""
    import naming
    s = naming.image_stem(image)
    res = {}
    for kind in ("final", "final_agmx"):
        js = os.path.join(run_dir, "json", s, kind, f"{s}_EXT{ext}.json")
        o = os.path.join(out, "panaskimg_timing", kind) + "/"
        os.makedirs(o, exist_ok=True)
        cmd = [PY, os.path.join(waders, "bin", "panaSKImg"), image, "-j", js, "-o", o, "--acm", "--ext", str(ext),
               "--save-plots"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=waders))
        res[kind] = time.time() - t0
        open(os.path.join(o, "log.txt"), "w").write(p.stdout + p.stderr)
        if p.returncode:
            res[kind] = f"failed (rc {p.returncode}, see {o}log.txt)"
    return res


# =============================================================================
# comparison
# =============================================================================
def compare_ccd(nat, ref, n):
    """metrics of one CCD: native file dict vs panaSKImg file dict (ext number n)"""
    E = f"EXT{n}"
    m = dict(n_native=int(nat["Nclusters"]), n_ref=int(ref["Nclusters"]))
    # pedestal (float32 in both files)
    a, b = nat.get(f"info:{E}_pedestal_mu_row_ADU"), ref.get(f"info:{E}_pedestal_mu_row_ADU")
    if a is not None and b is not None:
        a, b = a[0], b[0]
        m["ped_mu_maxdiff_ADU"] = float(np.nanmax(np.abs(a - b)))
        m["ped_mu_identical_rows"] = int(np.sum(a == b))
        m["ped_rows"] = int(a.size)
        sa, sb = nat[f"info:{E}_pedestal_sigma_row_ADU"][0], ref[f"info:{E}_pedestal_sigma_row_ADU"][0]
        m["ped_sigma_maxdiff_ADU"] = float(np.nanmax(np.abs(sa - sb)))
    e = f"ext{n}"
    for q in ("gainext", "sigext", "dcext", "mu0ext"):
        va, vb, eb = nat.get(f"info:{e}_{q}"), ref.get(f"info:{e}_{q}"), ref.get(f"info:{e}_{q}_err")
        if va is not None and vb is not None:
            m[q] = dict(native=va, ref=vb, ref_err=eb, native_err=nat.get(f"info:{e}_{q}_err"),
                        diff_in_ref_err=(va - vb) / eb if eb else None)
    ka, kb = keys(nat), keys(ref)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    common = [k for k in ka if k in ib]
    m["matched"] = len(common)
    m["only_native"] = len(ka) - len(common)
    m["only_ref"] = len(kb) - len(common)
    if common:
        A_ = np.array([ia[k] for k in common]); B_ = np.array([ib[k] for k in common])
        m["same_order"] = bool(np.all(A_ == B_))
        r = nat["Energy"][A_] / ref["Energy"][B_]
        m["energy_ratio_matched"] = dict(median=float(np.median(r)), min=float(np.min(r)), max=float(np.max(r)))
        m["energy_identical"] = int(np.sum(nat["Energy"][A_] == ref["Energy"][B_]))
        # branch-by-branch: clusters whose value is NOT identical (float32 in both files)
        diff = {}
        for k in sorted(set(nat) & set(ref)):
            if k.startswith("info:") or k in ("RUNID", "Nclusters"):
                continue
            va, vb = nat[k], ref[k]
            if isinstance(va, list):
                bad = sum(1 for i, j in zip(A_, B_) if va[i].shape != vb[j].shape or
                          not np.array_equal(va[i], vb[j], equal_nan=True))
            else:
                bad = int(np.sum(~((va[A_] == vb[B_]) | (np.isnan(va[A_]) & np.isnan(vb[B_])))))
            if bad:
                diff[k] = bad
        m["branches_not_identical"] = diff
        m["branches_compared"] = len([k for k in set(nat) & set(ref) if not k.startswith("info:")]) - 2
        fa = np.array([nat[k][A_] for k in ("fSTD_X", "fSTD_Y")]); fb = np.array([ref[k][B_] for k in ("fSTD_X", "fSTD_Y")])
        d = np.abs(fa - fb).ravel()
        m["fSTD_absdiff"] = dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                                 p99=float(np.percentile(d, 99)), max=float(d.max()))
        sa, sb = sigma_xy(nat)[A_], sigma_xy(ref)[B_]
        Ew = (ref["Energy"][B_] > 4.5) & (ref["Energy"][B_] < 7.0)
        m["front_back_flips_FeWindow"] = int(np.sum(Ew & ((sa < 1) != (sb < 1))))
        m["FeWindow_matched"] = int(Ew.sum())
    m["kalpha_native"] = kalpha(front_energies(nat))
    m["kalpha_ref"] = kalpha(front_energies(ref))
    return m


def compare_all(images, run_dir, out):
    import naming
    res = {}
    for cfg in CONFIGS:
        res[cfg] = {}
        pool_nat, pool_ref = {}, {}
        for p in images:
            s = naming.image_stem(p)
            for n in (1, 2, 3, 4):
                fn = os.path.join(out, cfg, f"panaSKImg_clustersRec_{s}_ext{n}.root")
                fr = os.path.join(run_dir, REF[cfg], f"panaSKImg_clustersRec_{s}_ext{n}.root")
                if not (os.path.exists(fn) and os.path.exists(fr)):
                    continue
                nat, ref = read_file(fn), read_file(fr)
                res[cfg][f"{s}:EXT{n}"] = compare_ccd(nat, ref, n)
                pool_nat.setdefault(n, []).append(front_energies(nat))
                pool_ref.setdefault(n, []).append(front_energies(ref))
        res[cfg]["_kalpha_pooled"] = {f"EXT{n}": dict(native=kalpha(np.concatenate(pool_nat[n])),
                                                      ref=kalpha(np.concatenate(pool_ref[n]))) for n in sorted(pool_nat)}
        print(f"compared {cfg}", flush=True)
    return res


# =============================================================================
# tables
# =============================================================================
def tables(res, timing):
    L = []
    w = L.append
    for cfg in CONFIGS:
        R = res.get(cfg, {})
        if not R:
            continue
        w(f"\n### {cfg} vs {REF[cfg]}/\n")
        w("| image:CCD | clusters nat/ref | matched | only nat | only ref | E nat/ref (median, min, max) | "
          "branches differing (clusters) | fSTD |d| med / p99 / max | Fe front<->back flips |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k, m in R.items():
            if k.startswith("_"):
                continue
            er = m.get("energy_ratio_matched", {})
            br = ", ".join(f"{b} {c}" for b, c in m.get("branches_not_identical", {}).items())
            fs = m.get("fSTD_absdiff", {})
            w(f"| {k.split('_Temp_109_')[1]} | {m['n_native']}/{m['n_ref']} | {m['matched']} | {m['only_native']} | "
              f"{m['only_ref']} | {er.get('median', np.nan):.6f}, {er.get('min', np.nan):.6f}, {er.get('max', np.nan):.6f} | "
              f"{br or 'none'} | {fs.get('median', np.nan):.2g} / {fs.get('p99', np.nan):.2g} / {fs.get('max', np.nan):.2g} | "
              f"{m.get('front_back_flips_FeWindow', '-')} of {m.get('FeWindow_matched', '-')} |")
        w("\nK-alpha (front side, 3 images pooled per CCD): native vs reference [keV]\n")
        w("| CCD | native mu +- err (sigma, N) | reference mu +- err (sigma, N) | diff [eV] |")
        w("|---|---|---|---|")
        for e, kk in R["_kalpha_pooled"].items():
            a, b = kk["native"], kk["ref"]
            w(f"| {e} | {a[0]:.4f} +- {a[1]:.4f} ({a[2]:.3f}, {a[3]}) | {b[0]:.4f} +- {b[1]:.4f} ({b[2]:.3f}, {b[3]}) | "
              f"{1000 * (a[0] - b[0]):+.1f} |")
    for cfg in ("mimic", "native"):
        R = res.get(cfg, {})
        w(f"\n### gain / noise / DC / pedestal ({cfg} vs panaSKImg info tree; diff in units of panaSKImg's error)\n")
        w("| image:CCD | gain nat / ref (diff/err) | sigma [ADU] nat / ref (diff/err) | lambda [ADU] nat / ref (diff/err) | "
          "mu0 [ADU] nat / ref | pedestal rows identical | max |d mu_row| [ADU] | max |d sigma_row| [ADU] |")
        w("|---|---|---|---|---|---|---|---|")
        for k, m in R.items():
            if k.startswith("_"):
                continue
            f = lambda q: (f"{m[q]['native']:.5f} / {m[q]['ref']:.5f} ({m[q]['diff_in_ref_err']:+.2f})"
                           if q in m and m[q]['diff_in_ref_err'] is not None else "-")
            w(f"| {k.split('_Temp_109_')[1]} | {f('gainext')} | {f('sigext')} | {f('dcext')} | "
              f"{m['mu0ext']['native']:.5f} / {m['mu0ext']['ref']:.5f} | {m.get('ped_mu_identical_rows')}/{m.get('ped_rows')} | "
              f"{m.get('ped_mu_maxdiff_ADU', np.nan):.2g} | {m.get('ped_sigma_maxdiff_ADU', np.nan):.2g} |")
    if timing:
        w("\n### wall time\n")
        for t in timing:
            w(f"- {t}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--images", required=True, help=".txt with the raw images")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--skip-runs", action="store_true", help="only compare existing outputs")
    ap.add_argument("--time-panaskimg", action="store_true", help="also time one panaSKImg run of one CCD")
    a = ap.parse_args()
    import native_pipeline as NP
    images = NP.expand_inputs([a.images])
    os.makedirs(a.out, exist_ok=True)
    timing = []
    tfile = os.path.join(a.out, "timing.json")
    if not a.skip_runs:
        for cfg in ("native", "agmx"):
            t = run_cli(images, a.run_dir, a.out, cfg, a.jobs)
            timing.append(f"{cfg}: {len(images)} images x 4 CCDs, command line with -j {a.jobs}: {t['wall_s']:.1f} s "
                          f"({t['cmd']})")
            print(timing[-1], flush=True)
        t0 = time.time()
        rs = run_configs(images, a.run_dir, a.out, a.jobs)
        timing.append(f"in-process configurations (mimic, mimic_gain, native_sigma0, agmx_mimic; 2 processes): "
                      f"{time.time() - t0:.1f} s")
        if a.time_panaskimg:
            t = time_panaskimg(images[1], a.run_dir, a.out)
            for kind, dt in t.items():
                timing.append(f"panaSKImg one CCD (EXT1 of {os.path.basename(images[1])}, json {kind}/, g(n), "
                              f"--save-plots, one process): " + (f"{dt:.1f} s" if isinstance(dt, float) else dt))
        json.dump(timing, open(tfile, "w"), indent=1)
    elif os.path.exists(tfile):
        timing = json.load(open(tfile))
    res = compare_all(images, a.run_dir, a.out)
    json.dump(res, open(os.path.join(a.out, "validation.json"), "w"), indent=1, default=str)
    txt = tables(res, timing)
    open(os.path.join(a.out, "validation_tables.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
