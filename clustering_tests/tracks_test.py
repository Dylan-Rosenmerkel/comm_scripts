"""Tracks / big events: WADERs, MFD+G, (1) track-first + MFD+G, (2a) MFD+G + B merge, (2b) MFD+G + saddle merge.
Injection of simulated objects with known truth into real Am-241 images (EXT1 low DC, EXT3 high DC, EXT4 8 e- noise)
and the real Am-241 spectra (cluster counts, lines, splitting of big WADERs clusters)."""
import json, os, sys
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_test as AT
import methods2 as M2
AM, CM = AT.AM, AT.CM
NE = 1 / AM.E_PER_E


def methods(Q, valid, sw, bg):
    cache = []
    def mf():
        if not cache: cache.append(AM.mf_detect(Q, valid, bg)[0])
        return [dict(c) for c in cache[0]]
    return {
        "WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],
        "MFD+G": lambda: mf(),
        "track-first": lambda: M2.track_first(Q, valid, bg),
        "MFD+G+B merge": lambda: M2.mfdg_merge_B(Q, valid, bg, cl=mf()),
        "MFD+G+saddle": lambda: M2.mfdg_merge_saddle(Q, valid, bg, cl=mf()),
        "MFD+G+saddle f0.2": lambda: M2.mfdg_merge_saddle(Q, valid, bg, cl=mf(), f=0.2),
        "MFD+G hybrid": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), ext_big=14),
        "MFD+G hybrid v2": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), f_big=0.2, ext_big=14),
    }


def sim(rng, kind, y0, x0):
    """electron positions (y, x) and sub-event id per electron; info dict with true energies / centres"""
    if kind == "muon":
        L = rng.uniform(15, 60); th = rng.uniform(0, np.pi)
        n = rng.poisson(L * 250); t = rng.uniform(0, 1, n); s = 0.2 + 1.0 * t
        y = y0 + (t - 0.5) * L * np.sin(th) + rng.normal(0, 1, n) * s
        x = x0 + (t - 0.5) * L * np.cos(th) + rng.normal(0, 1, n) * s
        return y, x, np.zeros(n, int), dict(E=[n / NE], c=[(y0, x0)], L=L)
    if kind == "worm":
        ns = rng.integers(8, 31); ang = rng.uniform(0, 2 * np.pi); p = np.array([y0, x0], float); pts = []
        for i in range(ns):
            pts.append(p.copy()); ang += rng.normal(0, 0.5); p += [np.sin(ang), np.cos(ang)]
        pts = np.array(pts); pts -= pts.mean(0) - [y0, x0]
        q = rng.poisson(150 * rng.exponential(1, ns) * np.where(np.arange(ns) >= ns - 3, 3, 1))
        s = rng.uniform(0.3, 1.0)
        y = np.repeat(pts[:, 0], q) + rng.normal(0, s, q.sum()); x = np.repeat(pts[:, 1], q) + rng.normal(0, s, q.sum())
        return y, x, np.zeros(q.sum(), int), dict(E=[q.sum() / NE], c=[(y0, x0)], L=ns)
    if kind == "blob":
        n = rng.poisson(59.54 * NE); s = rng.uniform(1.6, 2.4)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), np.zeros(n, int), dict(E=[n / NE], c=[(y0, x0)], s=s)
    if kind.startswith("pair"):
        d = float(kind[4:]); th = rng.uniform(0, np.pi); s = rng.uniform(0.3, 0.8)
        c = [(y0 + 0.5 * d * np.sin(th), x0 + 0.5 * d * np.cos(th)), (y0 - 0.5 * d * np.sin(th), x0 - 0.5 * d * np.cos(th))]
        n = rng.poisson(1580, 2)
        y = np.concatenate([c[i][0] + rng.normal(0, s, n[i]) for i in range(2)])
        x = np.concatenate([c[i][1] + rng.normal(0, s, n[i]) for i in range(2)])
        return y, x, np.repeat([0, 1], n), dict(E=list(n / NE), c=c, s=s)
    if kind == "single":
        n = rng.poisson(17.75 * NE); s = rng.uniform(0.3, 1.4)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), np.zeros(n, int), dict(E=[n / NE], c=[(y0, x0)], s=s)


KINDS = ["muon", "worm", "blob", "pair3", "pair4", "pair6", "single"]


def inject_eval(Q, valid, sw, rng, per_kind=10):
    busy = ndimage.binary_dilation(Q > 8.0, iterations=12) | ~ndimage.binary_erosion(valid, iterations=40)
    if sw > 2:
        busy = ndimage.binary_dilation(Q > 60.0, iterations=12) | ~ndimage.binary_erosion(valid, iterations=40)
    ys, xs = np.nonzero(~busy)
    pos = []
    for i in rng.permutation(ys.size):
        y, x = ys[i], xs[i]
        if all((y - a) ** 2 + (x - b) ** 2 > 75 ** 2 for a, b in pos):
            pos.append((y, x))
        if len(pos) >= per_kind * len(KINDS): break
    Qi = Q.copy(); objs = []
    for j, (y0, x0) in enumerate(pos):
        kind = KINDS[j % len(KINDS)]
        y, x, sub, info = sim(rng, kind, y0 + rng.uniform(-.5, .5), x0 + rng.uniform(-.5, .5))
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
    for name, f in methods(Qi, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        cy = np.array([c["y"] for c in cl]); cx = np.array([c["x"] for c in cl]); cE = np.array([c["E"] for c in cl])
        res = []
        for o in objs:
            sl = o["sl"]
            iy, ix = np.round(cy).astype(int) - sl[0].start, np.round(cx).astype(int) - sl[1].start
            inside = (iy >= 0) & (iy < o["reg"].shape[0]) & (ix >= 0) & (ix < o["reg"].shape[1])
            inside[inside] = o["reg"][iy[inside], ix[inside]]
            n = int(inside.sum()); Es = sorted(cE[inside].tolist(), reverse=True)
            Etot = sum(o["E"])
            r = dict(kind=o["kind"], n=n, frac=float(sum(Es) / Etot), frac_largest=float(Es[0] / Etot) if Es else 0.0)
            if o["kind"].startswith("pair"):
                # resolved: two clusters, each within 1.5 px of a true centre, each energy within 10 %
                good = 0
                for (ty, tx), te in zip(o["c"], o["E"]):
                    d = np.hypot(cy - ty, cx - tx)
                    j = np.argmin(d) if d.size else None
                    if j is not None and d[j] < 1.5 and abs(cE[j] / te - 1) < 0.10: good += 1
                r["resolved"] = bool(n == 2 and good == 2)
            res.append(r)
        out[name] = res
    return out


def real_eval(Q, valid, sw, bg):
    out = {}
    _, lab = CM.wader(Q, sw, 10, 30)
    W = CM.wader(Q, sw, 10, 30)[0]
    big = [c for c in W if c["E"] > 80]
    for name, f in methods(Q, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        Ls = np.array([lab[min(max(int(round(c["y"])), 0), lab.shape[0] - 1), min(max(int(round(c["x"])), 0), lab.shape[1] - 1)] for c in cl])
        split = []
        for c in big:
            L = lab[int(round(c["y"])), int(round(c["x"]))]
            if L > 0: split.append(int((Ls == L).sum()))
        out[name] = dict(E=[float(c["E"]) for c in cl], split=split)
    return out


def job(a):
    stem, ext, do_inj = a
    fn = os.path.join(HERE, "results4", f"{stem}_EXT{ext}.json")
    if os.path.exists(fn): return
    Q, valid, sw = AT.load(stem, ext)
    bg = AM.background_map(Q, valid)
    out = dict(stem=stem, ext=ext, real=real_eval(Q, valid, sw, bg))
    if do_inj:
        out["inject"] = []
        for seed in (1, 2):
            out["inject"] += sum(([dict(r, method=m) for r in v] for m, v in
                                  inject_eval(Q, valid, sw, np.random.default_rng(seed + 100 * ext)).items()), [])
    json.dump(out, open(fn, "w"))
    print(stem[-12:], ext, "done", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results4"), exist_ok=True)
    jobs = [(s, e, i < 3) for i, s in enumerate(AT.IMGS) for e in (1, 3, 4)]
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    with Pool(10) as p:
        p.map(job, jobs, chunksize=1)
