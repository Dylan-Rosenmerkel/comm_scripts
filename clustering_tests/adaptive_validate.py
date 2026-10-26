"""Validate adaptive.py against MHC (and WADERs) on every test type of the study.
  python adaptive_validate.py crowded   Am-241 runs_29img EXT4 (29 images): Lepy constrained fit, sigma_xy < 0.7
  python adaptive_validate.py tracks    injected objects (mhc_tracks harness) in AMTEst EXT1 / EXT3, 3 images x 3 seeds
  python adaptive_validate.py lowE      few-electron events on simulated blanks (lowE_test harness)
  python adaptive_validate.py fe55      Fe-55 injection + real Ka (compare.py harness, DM09/DM10 EXT1/EXT4 img 0-1)
  python adaptive_validate.py summary"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adaptive as AD
AM, CM, MH = AD.AM, AD.CM, AD.MH
OUT = os.path.join(HERE, os.environ.get("VALIDATE_OUT", "results_adaptive"))
NPROC = int(os.environ.get("NPROC", 3))
METHODS = os.environ.get("METHODS", "MHC,adaptive").split(",")


def factory(Q, valid, bg, sw=None):
    """name -> callable for the methods to validate (env METHODS)"""
    import gmx as GX
    f = {"WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],              # WADERs-like hysteresis (as mhc_tracks)
         "MFD+G": lambda: [dict(c) for c in AM.mf_detect(Q, valid, bg)[0]],
         "MHC": lambda: MH.mhc(Q, valid, bg),
         "adaptive": lambda: AD.adaptive_cluster(Q, valid, bg, sigma_crowd=sw),
         "adaptive+GMX": lambda: AD.adaptive_cluster(Q, valid, bg, sigma_crowd=sw, engine="gmx"),
         "GMX": lambda: GX.general_cluster(Q, valid, bg)}
    return {m: f[m] for m in METHODS}


# ---------------- crowded
def crowded_job(stem):
    import am241_29img_recluster as RC
    import adaptive_tune_am241 as T
    fn = os.path.join(OUT, "crowded", f"{stem}.npz")
    if os.path.exists(fn): return
    Q, valid = RC.load(stem, 4)
    bg = AM.background_map(Q, valid)
    out = {}
    for m, f in factory(Q, valid, bg, T.sigma_wader(stem)).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        key = m.replace("+", "_")
        out[f"{key}_E"] = [c["E"] for c in cl]; out[f"{key}_sxy"] = [c["sxy"] for c in cl]
        out[f"{key}_blend"] = [c.get("blend", 0.0) for c in cl]
    np.savez(fn, **out)
    print(stem[-12:], {m: len(out[f"{m.replace('+', '_')}_E"]) for m in METHODS}, flush=True)


def crowded_fit():
    import am241_paths; am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
    from amfit import Model, summary
    fs = sorted(glob.glob(os.path.join(OUT, "crowded", "*.npz"))); MAIN = ("La", "Lb", "Lg", "g26")
    out = {}
    for m in METHODS:
        key = m.replace("+", "_")
        E, S, B = (np.concatenate([np.load(f)[f"{key}_{k}"] for f in fs]) for k in ("E", "sxy", "blend"))
        cuts = [(f"{m} (sxy<0.7)", S < 0.7)] + ([(f"{m} (sxy<0.7, blend<0.02)", (S < 0.7) & (B < 0.02))] if "GMX" in m else [])
        for lab, sel in cuts:
            r = Model("Ag").fit(E[sel], 0.985); s = summary(r)
            z = np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))
            out[lab] = dict(n=int(sel.sum()), k=s["k"][0], devndf=r["devndf"], Z=z, **{g: s[g][0] / s[g][1] for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0})
            print(f"  {lab:34s} {sel.sum():6d} clusters  k {s['k'][0]:.4f}  dev/ndf {r['devndf']:.2f}  Z_np {z:5.1f}  "
                  + "  ".join(f"{g} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0), flush=True)
    json.dump(out, open(os.path.join(OUT, "crowded_fit.json"), "w"), indent=1, default=float)


# ---------------- tracks
def tracks_job(a):
    import mhc_tracks as MT
    stem, ext, seed = a
    fn = os.path.join(OUT, "tracks", f"{stem}_EXT{ext}_s{seed}.json")
    if os.path.exists(fn): return
    Q, valid, sw = MT.AT.load(stem, ext)
    meth = lambda Q, valid, sw, bg: factory(Q, valid, bg, sw)
    res = MT.inject_eval(Q, valid, sw, np.random.default_rng(seed + 100 * ext), meth=meth)
    json.dump([dict(r, method=m, ext=ext) for m, v in res.items() for r in v], open(fn, "w"))
    print(stem[-12:], ext, seed, "done", flush=True)


# ---------------- low E
def lowE_job(a):
    import lowE_test as LE
    lam, seed = a
    fn = os.path.join(OUT, "lowE", f"sim_{lam}_{seed}.json")
    if os.path.exists(fn): return
    LE.methods = lambda Q, valid, sw, bg: factory(Q, valid, bg)
    rng = np.random.default_rng(seed)
    valid = np.zeros((500, 3500), bool); valid[5:-5, 5:-5] = True
    out = dict(lam=lam, inject=LE.evaluate(np.where(valid, LE.blank(lam, rng), 0), valid, LE.SNOISE, rng),
               fakes=LE.fakes(np.where(valid, LE.blank(lam, np.random.default_rng(seed + 999)), 0), valid, LE.SNOISE))
    json.dump(out, open(fn, "w")); print("lowE", lam, seed, "done", flush=True)


# ---------------- Fe-55
def fe55_job(a):
    import compare as CP
    kind, mod, ext, img = a
    CP.RES = os.path.join(OUT, "fe55")
    def mk(mod, img, lam_row, sw, valid, near=None):
        cache = {}
        def bgf(Q):
            if id(Q) not in cache:
                cache.clear(); cache[id(Q)] = AM.background_map(Q, valid)
            return cache[id(Q)]
        return {m: (lambda m: lambda Q: factory(Q, valid, bgf(Q), sw)[m]())(m) for m in METHODS}
    CP.make_methods = mk
    if os.path.exists(os.path.join(CP.RES, f"{kind}_{mod}_EXT{ext}_img{img}.json")): return
    (CP.run_inject if kind == "inject" else CP.run_real)(mod, ext, img)


def summary():
    # scoring rule as summ_mhc.py (not imported: that script runs on import)
    rs = sum((json.load(open(f)) for f in glob.glob(os.path.join(OUT, "tracks", "*.json"))), [])
    K = ["muon", "worm", "blob", "single", "fe55", "alpha", "hotcol", "lowE", "pair3", "pair4", "pair6"]
    sc = lambda v, k: np.mean([r["resolved"] for r in v]) if k.startswith("pair") else np.mean([r["n"] == 1 and abs(r["frac_largest"] - 1) < 0.05 for r in v])
    for ext in (1, 3):
        print(f"tracks EXT{ext}: " + "".join(f"{k:>8s}" for k in K))
        for m in METHODS:
            v = [r for r in rs if r["ext"] == ext and r["method"] == m]
            print(f"  {m:10s}" + "".join(f"{sc([r for r in v if r['kind'] == k], k):8.2f}" if any(r['kind'] == k for r in v) else "       -" for k in K))
    for f in sorted(glob.glob(os.path.join(OUT, "lowE", "*.json"))):
        d = json.load(open(f)); print("lowE lam", d["lam"], f.split("_")[-1][:-5])
        for m in METHODS:
            a = np.array(d["inject"][m], float)
            eff = [a[a[:, 0] == n, 3].mean() for n in (2, 3, 4, 5, 7, 10)]
            print(f"   {m:9s} eff(2,3,4,5,7,10 e-) " + " ".join(f"{e:.2f}" for e in eff) + f"   fakes {len(d['fakes'][m])}")
    for f in sorted(glob.glob(os.path.join(OUT, "fe55", "inject_*.json"))):
        d = json.load(open(f))
        print(os.path.basename(f), " | ".join(f"s{s}: " + ", ".join(f"{m} {v[m]['med']:.4f}/{v[m]['eff']:.2f}" for m in METHODS) for s, v in d["single"].items()),
              "| pairs 5px resolved", {m: d["pairs"]["5.0"][m]["resolved"] for m in ("MHC", "adaptive")})
    for f in sorted(glob.glob(os.path.join(OUT, "fe55", "real_*.json"))):
        d = json.load(open(f)); print(os.path.basename(f), {m: (round(d["methods"][m]["front"][0] / 5.899, 4) if d["methods"][m]["front"] else None) for m in ("MHC", "adaptive")})


if __name__ == "__main__":
    what = sys.argv[1]
    for d in ("crowded", "tracks", "lowE", "fe55"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    if what == "crowded":
        import am241_29img_recluster as RC
        stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))})
        if not (len(sys.argv) > 2 and sys.argv[2] == "fit"):
            with Pool(NPROC) as p: p.map(crowded_job, stems, chunksize=1)
        crowded_fit()
    elif what == "tracks":
        import am_test as AT
        with Pool(NPROC) as p: p.map(tracks_job, [(s, e, k) for s in AT.IMGS[:3] for e in (1, 3) for k in (1, 2, 3)], chunksize=1)
    elif what == "lowE":
        with Pool(NPROC) as p: p.map(lowE_job, [(l, s) for l in (0.001, 0.01, 0.05) for s in (1, 2)], chunksize=1)
    elif what == "fe55":
        with Pool(NPROC) as p: p.map(fe55_job, [(k, m, e, i) for k in ("real", "inject") for m in ("DM09", "DM10") for e in (1, 4) for i in (0, 1)], chunksize=1)
    else:
        summary()
