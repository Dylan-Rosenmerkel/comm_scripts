"""Wide test: every clustering method on every available WADERs-calibrated image on this machine.

Images: all *_waders.fits[.bz2] under ~/Documents (copies / duplicate runs removed, one entry per image stem + CCD).
Input : the WADERs-calibrated image in electrons as stored (linear gain of the image; no g(n) curve, which most
        data sets do not have), active area = image minus a 3-px border.
Methods: WADERs (10 / 30 sigma), B auto (pixel >= k e-, seed 9.5 e-), MFD+G, MFD+G hybrid v1 / v2 (14 px), MHC.
Per image + CCD:
  real     : clusters found, run time, Fe-55 Ka fit (4.5-7 keV) and Am-241 59.5 keV fit where there is a line
  injected : 12 Fe-55 clusters (sigma 0.3-1.6 px), 4 muons (15-35 px), 4 electron worms (8-20 steps),
             4 pairs of 5.9 keV X-rays 4 px apart, added to the image (truth known)
  image    : noise, dark charge, occupancy (fraction of pixels > 1.5 e-)
Output: results_wide/<dataset>__<stem>__EXT<n>.json"""
import glob, json, os, re, sys, time, traceback, zlib
import numpy as np
from astropy.io import fits
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
sys.path.insert(0, HERE)
import agent_methods as AM
import cluster_methods as CM
import methods2 as M2
import mhc as MH
OUT = os.path.join(HERE, os.environ.get("WIDE_OUT", "results_wide"))
NE = 1 / AM.E_PER_E
EXCLUDE = ("Analysis_Image_4_5_6_7", "comm_scripts_data", "/no_gaincorr/", "WADERs_analysis_fz_files", "/Software/")


def inventory():
    files = glob.glob(os.path.expanduser("~/Documents/**/*_waders.fits*"), recursive=True)
    files = sorted(f for f in files if not any(x in f for x in EXCLUDE))
    seen, out = set(), []
    for f in files:
        b = os.path.basename(f)
        m = re.match(r"(.*)_(EXT[0-9-EXT]+)_waders\.fits", b)
        if not m: continue
        stem, exts = m.group(1), m.group(2)
        key = (stem, exts)
        if key in seen: continue
        seen.add(key)
        rel = os.path.relpath(f, os.path.expanduser("~/Documents"))
        ds = "/".join(rel.split("/")[:3]) if rel.startswith("LSM") else "/".join(rel.split("/")[:2])
        out.append((ds, f, stem, exts))
    return out


def kind_of(ds):
    return "Am-241" if ("Am241" in ds or "AMTEst" in ds) else ("skip-noise" if "Skip_Noise" in ds else "Fe-55")


def arrays(f, exts):
    h = fits.open(f)
    names = exts.split("-") if "-" in exts else [exts]
    arrs = [x.data for x in h if x.data is not None and np.ndim(x.data) == 2]
    if len(arrs) == len(names):
        return list(zip(names, arrs))
    if len(arrs) == 1 and len(names) > 1:              # several CCDs side by side in one array
        a = arrs[0]; w = a.shape[1] // len(names)
        return [(n, a[:, i * w:(i + 1) * w]) for i, n in enumerate(names)]
    return [(names[0], arrs[0])] if arrs else []


def methods(Q, valid, sw, bg):
    k = max(int(AM.poisson_tail_k(float(np.median(bg.lam[valid])), 0.10)[0]), 2)
    cache = []
    def mf():
        if not cache: cache.append(AM.mf_detect(Q, valid, bg)[0])
        return [dict(c) for c in cache[0]]
    return {
        "WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],
        "B auto": lambda: CM.topo(Q, 0.0, 9.5, k - 0.5, 0),
        "MFD+G": lambda: mf(),
        "hybrid v1": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), ext_big=14),
        "hybrid v2": lambda: M2.mfdg_hybrid(Q, valid, bg, cl=mf(), f_big=0.2, ext_big=14),
        "MHC": lambda: MH.mhc(Q, valid, bg),
    }


def sim(rng, kind, y0, x0):
    if kind == "fe55":
        n = rng.poisson(1580); s = rng.uniform(0.3, 1.6)
        return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n), [n / NE], [(y0, x0)]
    if kind == "muon":
        L = rng.uniform(15, 35); th = rng.uniform(0, np.pi); n = rng.poisson(L * 250); t = rng.uniform(0, 1, n); s = 0.2 + t
        return (y0 + (t - .5) * L * np.sin(th) + rng.normal(0, 1, n) * s, x0 + (t - .5) * L * np.cos(th) + rng.normal(0, 1, n) * s,
                [n / NE], [(y0, x0)])
    if kind == "worm":
        ns = int(rng.integers(8, 21)); a = rng.uniform(0, 2 * np.pi); p = np.array([y0, x0], float); pts = []
        for i in range(ns):
            pts.append(p.copy()); a += rng.normal(0, .5); p += [np.sin(a), np.cos(a)]
        pts = np.array(pts); pts -= pts.mean(0) - [y0, x0]
        q = rng.poisson(150 * rng.exponential(1, ns) * np.where(np.arange(ns) >= ns - 3, 3, 1)); s = rng.uniform(.3, 1.)
        return np.repeat(pts[:, 0], q) + rng.normal(0, s, q.sum()), np.repeat(pts[:, 1], q) + rng.normal(0, s, q.sum()), [q.sum() / NE], [(y0, x0)]
    if kind == "pair4":
        th = rng.uniform(0, np.pi); s = rng.uniform(.3, .8); c = [(y0 + 2 * np.sin(th), x0 + 2 * np.cos(th)), (y0 - 2 * np.sin(th), x0 - 2 * np.cos(th))]
        n = rng.poisson(1580, 2)
        return (np.concatenate([c[i][0] + rng.normal(0, s, n[i]) for i in range(2)]),
                np.concatenate([c[i][1] + rng.normal(0, s, n[i]) for i in range(2)]), list(n / NE), c)


KINDS = ["fe55"] * 3 + ["muon", "worm", "pair4"]


def inject_eval(Q, valid, sw, rng, n_obj=24, sep=40):
    busy = ndimage.binary_dilation(Q > 8.0, iterations=6) | ~ndimage.binary_erosion(valid, iterations=20)
    ys, xs = np.nonzero(~busy)
    pos = []
    for i in rng.permutation(ys.size)[:200000]:
        y, x = ys[i], xs[i]
        if all((y - a) ** 2 + (x - b) ** 2 > sep ** 2 for a, b in pos):
            pos.append((y, x))
        if len(pos) >= n_obj: break
    Qi = Q.copy(); objs = []
    for j, (y0, x0) in enumerate(pos):
        kind = KINDS[j % len(KINDS)]
        y, x, E, c = sim(rng, kind, y0 + rng.uniform(-.5, .5), x0 + rng.uniform(-.5, .5))
        iy, ix = np.round(y).astype(int), np.round(x).astype(int)
        ok = (iy >= 0) & (iy < Q.shape[0]) & (ix >= 0) & (ix < Q.shape[1])
        np.add.at(Qi, (iy[ok], ix[ok]), 1.0)
        reg = np.zeros(Q.shape, bool); reg[iy[ok], ix[ok]] = True
        sl = ndimage.find_objects(reg.astype(int))[0]
        sl = (slice(max(sl[0].start - 3, 0), sl[0].stop + 3), slice(max(sl[1].start - 3, 0), sl[1].stop + 3))
        objs.append(dict(kind=kind, sl=sl, reg=ndimage.binary_dilation(reg[sl], structure=np.ones((3, 3), bool), iterations=2), E=E, c=c))
    try:
        bg = AM.background_map(Qi, valid)
    except Exception:
        bg = simple_bg(Qi, valid, sw)
    out = {}
    for name, f in methods(Qi, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        cy = np.array([c["y"] for c in cl]); cx = np.array([c["x"] for c in cl]); cE = np.array([c["E"] for c in cl])
        res = []
        for o in objs:
            sl = o["sl"]
            iy, ix = np.round(cy).astype(int) - sl[0].start, np.round(cx).astype(int) - sl[1].start
            ins = (iy >= 0) & (iy < o["reg"].shape[0]) & (ix >= 0) & (ix < o["reg"].shape[1])
            ins[ins] = o["reg"][iy[ins], ix[ins]]
            idx = np.nonzero(ins)[0]; Et = sum(o["E"])
            r = dict(kind=o["kind"], n=int(idx.size), ratio=float(cE[idx].max() / Et) if idx.size else 0.0)
            if o["kind"] == "pair4":
                good = 0
                for (ty, tx), te in zip(o["c"], o["E"]):
                    d = np.hypot(cy - ty, cx - tx) if cy.size else np.array([np.inf])
                    j = int(np.argmin(d))
                    if d[j] < 1.5 and abs(cE[j] / te - 1) < .10: good += 1
                r["resolved"] = bool(idx.size == 2 and good == 2)
            res.append(r)
        out[name] = res
    return out


def job(a):
    ds, f, stem, exts = a
    try:
        for ext, arr in arrays(f, exts):
            fn = os.path.join(OUT, f"{ds.replace('/', '_')}__{stem}__{ext}.json")
            if os.path.exists(fn): continue
            Q = np.nan_to_num(np.asarray(arr, float))
            if Q.shape[0] < 40 or Q.shape[1] < 100: continue
            valid = np.zeros(Q.shape, bool); valid[3:-3, 3:-3] = True
            act = Q[valid]; z = act[np.abs(act) < 0.5]
            sw = float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 100 else float(1.4826 * np.median(np.abs(act - np.median(act))))
            sw = max(sw, 0.05)
            try:
                bg = AM.background_map(Q, valid); bg_kind = "fit"
            except Exception:
                bg = simple_bg(Q, valid, sw); bg_kind = "fallback"
            out = dict(bg=bg_kind, dataset=ds, kind=kind_of(ds), file=f, stem=stem, ext=ext, shape=list(Q.shape), sigma=sw,
                       lam=float(np.median(bg.lam[valid])), noise=float(np.median(bg.s[valid])),
                       occupancy=float((Q[valid] > 1.5).mean()), real={})
            for name, fn_ in methods(Q, valid, sw, bg).items():
                t = time.time()
                try:
                    cl = [c for c in fn_() if np.isfinite(c.get("E", np.nan))]
                except Exception as e:
                    out["real"][name] = dict(error=str(e)); continue
                E = np.array([c["E"] for c in cl])
                r = dict(n=len(cl), time=time.time() - t, n_ka_window=int(((E > 4.5) & (E < 7)).sum()))
                ka = CM_fit_ka(E)
                if ka: r["ka"] = ka
                l59 = fit59(E)
                if l59: r["am59"] = l59
                out["real"][name] = r
            try:
                out["inject"] = inject_eval(Q, valid, sw, np.random.default_rng(zlib.crc32((stem + ext).encode())))
            except Exception as e:
                out["inject_error"] = str(e)
            json.dump(out, open(fn, "w"))
            print(ds, stem[-20:], ext, "done", flush=True)
    except Exception:
        traceback.print_exc()


simple_bg = AM.simple_bg          # fallback background (agent_methods)


def CM_fit_ka(E):
    import run_study as RS
    r = RS.fit_ka(E)
    return list(r) if r else None


def fit59(E):
    import am_test as AT
    if ((E > 55) & (E < 63)).sum() < 60: return None
    r = AT.fit_line(E, 59.54)
    return list(r) if r else None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    inv = inventory()
    print(len(inv), "files", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        import collections
        for k, v in sorted(collections.Counter(d for d, *_ in inv).items()): print(v, k)
        sys.exit()
    with Pool(int(os.environ.get("NPROC", 10)), maxtasksperchild=4) as p:
        p.map(job, inv, chunksize=1)
