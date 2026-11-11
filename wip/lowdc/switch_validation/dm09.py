"""DM09 Image_5 (high dark current) through src/agmx_root.py with AGMX_FEWE=off / auto: the switch must choose A_GMX
(almost) everywhere and leave the clusters unchanged.

    python dm09.py run  --fewe off|auto [--curve]   # agmx_root on every CCD (sequential, one process) -> SCRATCH/dm09/
    python dm09.py compare                          # cluster counts, energies and the Fe-55 K-alpha per CCD
                                                    #   -> results/dm09_compare.csv, results/dm09_decisions.csv

RUN_DIR (read only): the run_WADERs output in the scratchpad (calibrated *_waders.fits.bz2, root/, json/, gain_vs_charge/).
'run' calls agmx_root's own task list and process() in this process, so that adaptive_cluster's decision record
(info['fewe']) can be captured; the output files are agmx_root's.
"""
import os
import sys
import argparse
import glob
import json
import time

import numpy as np

import common as C

RUN_DIR = os.path.join(os.path.dirname(C.SCRATCH), "e2e", "WADERs_analysis_DM09_Image5")
OUTD = os.path.join(C.SCRATCH, "dm09")
SRC = os.path.join(C.LOWDC, "..", "..", "src")


def run(fewe, curve):
    os.environ["AGMX_FEWE"] = fewe
    os.environ["AGMX_FE_CANDIDATES"] = "0"
    sys.path.insert(0, SRC)
    import adaptive as AD
    AD.FEWE = fewe
    import agmx_root as AR
    tag = f"{fewe}_{'gc' if curve else 'plain'}"
    outdir = os.path.join(OUTD, tag)
    os.makedirs(outdir, exist_ok=True)
    rec = []
    orig = AD.adaptive_cluster

    def wrapped(*a, **k):
        res = orig(*a, **k)
        cl, info = res
        f = dict(info.get("fewe", {}))
        f.update(agmx_regime=info.get("regime"), noise=info.get("noise"), lam=info.get("lam"),
                 crowded=bool(info.get("crowded_mode", False)), n_clusters=len(cl),
                 n_bayes=sum(1 for c in cl if c.get("finder") == "bayes"))
        rec.append(f)
        return res
    AD.adaptive_cluster = wrapped
    cpath = os.path.join(RUN_DIR, "gain_vs_charge") if curve else None
    out = []
    for t in AR.tasks_for_run(RUN_DIR):
        t0 = time.perf_counter()
        stem, ext, n, err, note = AR._job(t + (outdir, cpath, "comb"))
        r = dict(rec[-1]) if rec else {}
        r.update(stem=stem, ext=ext, n=n, err=err, t=time.perf_counter() - t0, tag=tag)
        out.append(r)
        print(stem, ext, n, err, r.get("used"), r.get("reason"), f"f_c {r.get('f_c', np.nan):.3g}", f"{r['t']:.1f} s", flush=True)
    json.dump(out, open(os.path.join(outdir, "decisions.json"), "w"), indent=1, default=str)


def read_root(fn):
    """per-cluster Energy [keV], STD_XY [px], PosX, PosY of a panaSKImg cluster file (one entry per image, vectors)"""
    import ROOT
    f = ROOT.TFile.Open(fn)
    t = f.Get("clustersRec")
    out = {k: [] for k in ("Energy", "STD_XY", "PosX", "PosY")}
    if t:
        for ev in t:
            for k in out:
                out[k] += list(getattr(ev, k))
    f.Close()
    return {k: np.array(v, float) for k, v in out.items()}


def kalpha(E, lo=5.4, hi=6.6):
    """Fe-55 K-alpha: Gaussian + flat fit of the energies in [lo, hi] keV (binned ML): (mean, sigma, N)"""
    from scipy import optimize
    x = E[(E > lo) & (E < hi)]
    if x.size < 30:
        return np.nan, np.nan, x.size
    h, b = np.histogram(x, np.linspace(lo, hi, 121)); c = 0.5 * (b[1:] + b[:-1]); w = b[1] - b[0]

    def nll(p):
        n, mu, s, k = p
        m = n * w * np.exp(-0.5 * ((c - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s) + k * w
        m = np.maximum(m, 1e-12)
        return float((m - h * np.log(m)).sum())
    p0 = (x.size, float(np.median(x)), 0.12, 1.0)
    r = optimize.minimize(nll, p0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return float(r.x[1]), float(abs(r.x[2])), float(r.x[0])


def compare():
    import pandas as pd
    rows, dec = [], []
    for kind in ("plain", "gc"):
        d_off, d_on = os.path.join(OUTD, f"off_{kind}"), os.path.join(OUTD, f"auto_{kind}")
        if not os.path.isdir(d_on):
            continue
        for r in json.load(open(os.path.join(d_on, "decisions.json"))):
            dec.append(dict(kind=kind, **{k: r.get(k) for k in ("stem", "ext", "used", "reason", "s", "f_c", "f_c_all", "blocks",
                                                                   "agmx_regime", "noise", "lam", "crowded", "n_clusters", "n_bayes", "t")}))
        t_off = {(r["stem"], r["ext"]): r["t"] for r in json.load(open(os.path.join(d_off, "decisions.json")))}
        for fn in sorted(glob.glob(os.path.join(d_on, "panaSKImg_clustersRec_*.root"))):
            b = os.path.basename(fn)
            A, B_ = read_root(os.path.join(d_off, b)), read_root(fn)
            # Fe-55: front-side clusters (STD_XY < 1 px) only
            k1, k2 = kalpha(A["Energy"][A["STD_XY"] < 1]), kalpha(B_["Energy"][B_["STD_XY"] < 1])
            stem, ext = b[len("panaSKImg_clustersRec_"):-len(".root")].rsplit("_ext", 1)
            same = all(A[k].size == B_[k].size and np.array_equal(A[k], B_[k], equal_nan=True) for k in A)
            rows.append(dict(kind=kind, file=b, n_off=A["Energy"].size, n_auto=B_["Energy"].size, identical=int(same),
                             ka_off=k1[0], ka_auto=k2[0], ka_sig_off=k1[1], ka_sig_auto=k2[1], n_ka_off=k1[2], n_ka_auto=k2[2],
                             t_off=t_off.get((stem, int(ext)), np.nan)))
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "dm09_compare.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(C.OUT, "dm09_decisions.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(dec).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "compare"])
    ap.add_argument("--fewe", default="auto")
    ap.add_argument("--curve", action="store_true")
    a = ap.parse_args()
    if a.what == "run":
        run(a.fewe, a.curve)
    else:
        compare()


if __name__ == "__main__":
    main()
