"""MHC vs the previous best (MFD+G hybrid v1 / v2, MFD+G, WADERs) on injected objects in real Am-241 images
(EXT1, EXT3), plus classification accuracy of MHC, plus the real Am-241 data."""
import json, os, sys
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
sys.path.insert(0, HERE); import tracks_test as TT
import methods2 as M2
import mhc as MH
AM, CM, AT = TT.AM, TT.CM, TT.AT
NE = 1 / AM.E_PER_E
KINDS = ["muon", "worm", "blob", "pair3", "pair4", "pair6", "single", "fe55", "alpha", "hotcol", "lowE"]
TRUE_CLS = {"muon": "muon", "worm": "worm", "blob": "x-ray diffuse", "single": "x-ray", "fe55": "x-ray", "alpha": "alpha",
            "hotcol": "artifact-column", "lowE": "low-E", "pair3": "x-ray", "pair4": "x-ray", "pair6": "x-ray"}
OUT = os.environ.get("MHC_OUT", "results_tracks")


def sim(rng, kind, y0, x0):
    if kind in TT.KINDS:
        y, x, sub, info = TT.sim(rng, kind, y0, x0)
        if kind == "single":
            info["cls"] = "x-ray diffuse" if info["s"] >= 1.0 else "x-ray"
        return y, x, info
    if kind == "fe55":
        n = rng.poisson(1580); s = rng.uniform(0.3, 1.8)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), dict(E=[n / NE], c=[(y0, x0)], s=s, cls="x-ray diffuse" if s >= 1 else "x-ray")
    if kind == "alpha":
        n = rng.poisson(5.49e3 * NE); s = rng.uniform(2.5, 4.0)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), dict(E=[n / NE], c=[(y0, x0)], s=s)
    if kind == "hotcol":
        L = int(rng.integers(20, 60)); q = rng.poisson(30, L)
        y = np.repeat(y0 + np.arange(L) - L / 2, q); x = np.full(q.sum(), float(round(x0)))
        return y, x, dict(E=[q.sum() / NE], c=[(y0, x0)])
    if kind == "lowE":
        n = int(rng.integers(5, 40)); s = rng.uniform(0.2, 0.6)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), dict(E=[n / NE], c=[(y0, x0)], s=s)


def methods(Q, valid, sw, bg):
    cache = []
    def mf():
        if not cache: cache.append(AM.mf_detect(Q, valid, bg)[0])
        return [dict(c) for c in cache[0]]
    return {
        "WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],
        "MFD+G": lambda: mf(),
        "MFD+G hybrid v1": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), ext_big=14),
        "MFD+G hybrid v2": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), f_big=0.2, ext_big=14),
        "MHC": lambda: MH.mhc(Q, valid, bg),
    }


def inject_eval(Q, valid, sw, rng, per_kind=8, meth=methods, footprint_check=None):
    """as wide_test.inject_eval: footprint_check (default; INJECT_FOOTPRINT=0 for the earlier centre-only check)
    places an object only where its whole footprint (+2 px) avoids real events"""
    import inject_utils as IU
    fc = IU.FOOTPRINT_CHECK if footprint_check is None else footprint_check
    thr = 8.0 if sw < 2 else 60.0
    busy = ndimage.binary_dilation(Q > thr, iterations=12) | ~ndimage.binary_erosion(valid, iterations=40)
    ys, xs = np.nonzero(~busy)
    pos, sims = [], []
    for i in rng.permutation(ys.size):
        y, x = ys[i], xs[i]
        if all((y - a) ** 2 + (x - b) ** 2 > 75 ** 2 for a, b in pos):
            if fc:
                kind = KINDS[len(pos) % len(KINDS)]
                sm = sim(rng, kind, y + rng.uniform(-.5, .5), x + rng.uniform(-.5, .5))
                iy, ix = np.round(sm[0]).astype(int), np.round(sm[1]).astype(int)
                inb = (iy >= 2) & (iy < Q.shape[0] - 2) & (ix >= 2) & (ix < Q.shape[1] - 2)
                if not inb.all() or IU.footprint_busy(busy, iy, ix):
                    continue
                sims.append(sm)
            pos.append((y, x))
        if len(pos) >= per_kind * len(KINDS): break
    Qi = Q.copy(); objs = []
    for j, (y0, x0) in enumerate(pos):
        kind = KINDS[j % len(KINDS)]
        y, x, info = sims[j] if fc else sim(rng, kind, y0 + rng.uniform(-.5, .5), x0 + rng.uniform(-.5, .5))
        iy, ix = np.round(y).astype(int), np.round(x).astype(int)
        ok = (iy >= 0) & (iy < Q.shape[0]) & (ix >= 0) & (ix < Q.shape[1])
        np.add.at(Qi, (iy[ok], ix[ok]), 1.0)
        reg = np.zeros(Q.shape, bool); reg[iy[ok], ix[ok]] = True
        sl = ndimage.find_objects(reg.astype(int))[0]
        sl = (slice(max(sl[0].start - 3, 0), sl[0].stop + 3), slice(max(sl[1].start - 3, 0), sl[1].stop + 3))
        r = ndimage.binary_dilation(reg[sl], structure=M2.EIGHT, iterations=2)
        objs.append(dict(kind=kind, sl=sl, reg=r, **info))
    bg = AM.background_map(Qi, valid)
    out = {}
    for name, f in meth(Qi, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        cy = np.array([c["y"] for c in cl]); cx = np.array([c["x"] for c in cl]); cE = np.array([c["E"] for c in cl])
        res = []
        for o in objs:
            sl = o["sl"]
            iy, ix = np.round(cy).astype(int) - sl[0].start, np.round(cx).astype(int) - sl[1].start
            inside = (iy >= 0) & (iy < o["reg"].shape[0]) & (ix >= 0) & (ix < o["reg"].shape[1])
            inside[inside] = o["reg"][iy[inside], ix[inside]]
            idx = np.nonzero(inside)[0]; idx = idx[np.argsort(-cE[idx])]
            Etot = sum(o["E"])
            r = dict(kind=o["kind"], n=int(idx.size), frac=float(cE[idx].sum() / Etot),
                     frac_largest=float(cE[idx[0]] / Etot) if idx.size else 0.0,
                     cls=cl[idx[0]].get("cls") if idx.size else None,
                     true_cls=o.get("cls", TRUE_CLS[o["kind"]]), s=o.get("s"))
            if o["kind"].startswith("pair"):
                good = 0
                for (ty, tx), te in zip(o["c"], o["E"]):
                    d = np.hypot(cy - ty, cx - tx)
                    j = np.argmin(d) if d.size else None
                    if j is not None and d[j] < 1.5 and abs(cE[j] / te - 1) < 0.10: good += 1
                r["resolved"] = bool(idx.size == 2 and good == 2)
            res.append(r)
        out[name] = res
    return out


def real_eval(Q, valid, sw, bg):
    out = {}
    W, lab = CM.wader(Q, sw, 10, 30)
    big = [c for c in W if c["E"] > 80]
    for name, f in methods(Q, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        Ls = np.array([lab[min(max(int(round(c["y"])), 0), lab.shape[0] - 1), min(max(int(round(c["x"])), 0), lab.shape[1] - 1)] for c in cl])
        split = [int((Ls == lab[int(round(c["y"])), int(round(c["x"]))]).sum()) for c in big if lab[int(round(c["y"])), int(round(c["x"]))] > 0]
        out[name] = dict(E=[float(c["E"]) for c in cl], split=split, cls=[c.get("cls") for c in cl])
    return out


def job(a):
    stem, ext, do_inj = a
    fn = os.path.join(HERE, OUT, f"{stem}_EXT{ext}.json")
    if os.path.exists(fn): return
    Q, valid, sw = AT.load(stem, ext)
    bg = AM.background_map(Q, valid)
    out = dict(stem=stem, ext=ext, real=real_eval(Q, valid, sw, bg))
    if do_inj:
        out["inject"] = []
        for seed in (1, 2, 3):
            out["inject"] += sum(([dict(r, method=m) for r in v] for m, v in
                                  inject_eval(Q, valid, sw, np.random.default_rng(seed + 100 * ext)).items()), [])
    json.dump(out, open(fn, "w"))
    print(stem[-12:], ext, "done", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, OUT), exist_ok=True)
    jobs = [(s, e, i < 3) for i, s in enumerate(AT.IMGS) for e in (1, 3)]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[2]][:int(sys.argv[1])]
    with Pool(int(os.environ.get("NPROC", 10))) as p:
        p.map(job, jobs, chunksize=1)
