"""Compare the agent methods with the WADERs-style baselines on injected and real Fe-55 clusters.

  python compare.py inject  DM09 4 1      -> results/inject_DM09_EXT4_img1.json
  python compare.py real    DM09 4 1      -> results/real_DM09_EXT4_img1.json
  python compare.py all                    (runs everything in parallel worker processes)
"""
import json, os, sys, time
import numpy as np
from scipy import ndimage, stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_methods as AM
import cluster_methods as CM
import run_study as RS

RES = os.path.join(HERE, os.environ.get("RESULTS_DIR", "results"))
os.makedirs(RES, exist_ok=True)
SXYS = (0.3, 0.8, 1.4, 1.8)


def fixed_N(mod, img):
    """Baseline (b): pixel >= 2 e- (Q > 1.5) for DM09 images 1-2, pixel >= 3 e- (Q > 2.5) for DM10 and img 0."""
    return 1.5 if (mod == "DM09" and img in (1, 2)) else 2.5


def auto_N(lam_row, alpha=0.10):
    """Baseline (c): the per-CCD automatic threshold as installed in autoconfig (--cluster-threshold-e auto):
    k from the Poisson tail (alpha 0.10) of the MEDIAN row dark charge, pixel threshold k - 0.5."""
    lm = float(np.median(lam_row))
    return AM.poisson_tail_k(lm, alpha)[0] - 0.5


def make_methods(mod, img, lam_row, sw, valid, near=None):
    """name -> f(Q) returning a list of cluster dicts (E [keV], y, x). Background map computed once per image
    (cached) and shared by the agent methods; `near` restricts the (costly) fits to clusters near given
    positions for the injection study (detection still runs on the whole image)."""
    cache = {}

    def bgf(Q):
        if id(Q) not in cache:
            cache.clear()
            cache[id(Q)] = AM.background_map(Q, valid)
        return cache[id(Q)]
    Nf, Na = fixed_N(mod, img), auto_N(lam_row)

    def naylor_moments(Q):
        bg = bgf(Q)
        det, _ = AM.mf_detect(Q, valid, bg, grow_alpha=0)
        claim = AM.assign_footprints(Q.shape, det)
        out = []
        for i, c in enumerate(det):
            if near is not None and not _is_near(c, near):
                out.append(c); continue
            if not np.isfinite(c["E"]):
                out.append(c); continue
            g = AM.optimal_photometry(Q, bg, valid, c["y"], c["x"], c["sxy"], E0=c["E"] / AM.E_PER_E,
                                      exclude=(claim >= 0) & (claim != i))
            out.append(dict(c, E=g["E"]))
        return out

    def pipe(Q, **kw):
        bg = bgf(Q)
        return pipeline_near(Q, valid, bg, near, **kw)

    return {
        "(a) WADERs 10s/30s": lambda Q: CM.wader(Q, sw, 10, 30)[0],
        f"(b) fixed pix>{Nf:g}, seed>9.5": lambda Q: CM.topo(Q, lam_row, 9.5, Nf, 0),
        f"(c) auto pix>{Na:g}, seed>9.5": lambda Q: CM.topo(Q, lam_row, 9.5, Na, 0),
        "2 PT a=0.02": lambda Q: AM.poisson_tail_clusters(Q, valid, bgf(Q), alpha=0.02)[0],
        "2 PT a=0.05": lambda Q: AM.poisson_tail_clusters(Q, valid, bgf(Q), alpha=0.05)[0],
        "3 MF disk": lambda Q: AM.mf_detect(Q, valid, bgf(Q), grow_alpha=0)[0],
        "3 MF disk+grow": lambda Q: AM.mf_detect(Q, valid, bgf(Q))[0],
        "3+5 MF+Naylor(mom.)": naylor_moments,
        "3+4 MF+Cash": lambda Q: pipe(Q, detector="mf", energy="cash", multi=False),
        "1-3-4-6 pipeline": lambda Q: pipe(Q, detector="mf", energy="cash", multi=True),
        "1-3-4/3-6 hybrid": lambda Q: pipe(Q, detector="mf", energy="hybrid", multi=True),
        "1-2-4-6 PT+Cash+multi": lambda Q: pipe(Q, detector="pt", energy="cash", multi=True),
    }


def _is_near(c, near, r=4.0):
    return any((c["y"] - y) ** 2 + (c["x"] - x) ** 2 < r * r for y, x in near)


def pipeline_near(Q, valid, bg, near, **kw):
    """AM.pipeline, fitting only clusters (or groups) near `near` (all if None)."""
    return AM.pipeline(Q, valid, bg, only_near=near, **kw)[0]


def positions(Q, valid, rng, n, sep=20.0, pair_d=None):
    busy = ndimage.binary_dilation(Q > 8.0, iterations=8) | ~ndimage.binary_erosion(valid, iterations=10)
    ys, xs = np.nonzero(~busy)
    idx = rng.permutation(ys.size)
    pos = []
    for k in idx:
        y, x = ys[k], xs[k]
        if all((y - py) ** 2 + (x - px) ** 2 > sep ** 2 for py, px in pos):
            pos.append((y + rng.uniform(-0.5, 0.5), x + rng.uniform(-0.5, 0.5)))
        if len(pos) >= n:
            break
    if pair_d is None:
        return pos
    out = []
    for (y, x) in pos:
        a = rng.uniform(0, np.pi)
        dy, dx = 0.5 * pair_d * np.sin(a), 0.5 * pair_d * np.cos(a)
        out.append(((y - dy, x - dx), (y + dy, x + dx)))
    return out


def summarize(ratios, n_inj):
    r = np.asarray(ratios, float)
    if r.size == 0:
        return dict(eff=0.0, med=np.nan, rms=np.nan, hi=np.nan, lo=np.nan, n=0)
    med = float(np.median(r))
    return dict(eff=r.size / n_inj, med=med, rms=float(1.4826 * np.median(np.abs(r - med))),
                hi=float(np.mean(r > 1.05)), lo=float(np.mean(r < 0.95)), n=int(r.size))


def run_inject(mod, ext, img, n_round=110, rounds=2, seed=7):
    Q, valid, sigma, lam_row, sw = RS.load(mod, img, ext)
    t0 = time.time()
    out = dict(mod=mod, ext=ext, img=img, lam_row_mean=float(np.mean(lam_row)), single={}, pairs={})
    names = None
    for s in SXYS:
        ratios = {}
        ninj = 0
        for rd in range(rounds):
            rng = np.random.default_rng(seed + 100 * rd + int(10 * s))
            pos = positions(Q, valid, rng, n_round)
            Qi, counts = CM.inject(Q, rng, 1580, s, pos)
            meth = make_methods(mod, img, lam_row, sw, valid, near=pos)
            names = list(meth)
            ninj += len(pos)
            for name, f in meth.items():
                fl = f(Qi)
                for (y, x), ne in zip(pos, counts):
                    m = RS.match(fl, y, x)
                    if m is not None and np.isfinite(m["E"]):
                        ratios.setdefault(name, []).append(m["E"] / (ne * CM.E_PER_E))
        out["single"][str(s)] = {n: summarize(ratios.get(n, []), ninj) for n in names}
        out["single"][str(s)]["_n_injected"] = ninj
        print(mod, ext, img, "sxy", s, f"{time.time() - t0:.0f}s", flush=True)
    # pairs: two 1580-e- clusters at distance d (sxy 1.4): fraction resolved with both energies within 5 %
    for d in (3.0, 5.0):
        rng = np.random.default_rng(seed + 1000 + int(d))
        pp = positions(Q, valid, rng, 100, sep=25.0, pair_d=d)
        flat = [p for pr in pp for p in pr]
        Qi, counts = CM.inject(Q, rng, 1580, 1.4, flat)
        meth = make_methods(mod, img, lam_row, sw, valid, near=flat)
        res = {}
        for name, f in meth.items():
            fl = f(Qi)
            good = merged = 0
            for k, (p1, p2) in enumerate(pp):
                m1, m2 = RS.match(fl, *p1, r=1.5), RS.match(fl, *p2, r=1.5)
                n1, n2 = counts[2 * k], counts[2 * k + 1]
                if m1 is not None and m2 is not None and m1 is not m2 and np.isfinite(m1["E"]) and np.isfinite(m2["E"]) \
                        and abs(m1["E"] / (n1 * CM.E_PER_E) - 1) < 0.05 and abs(m2["E"] / (n2 * CM.E_PER_E) - 1) < 0.05:
                    good += 1
                mm = RS.match(fl, *((np.array(p1) + np.array(p2)) / 2), r=d)
                if mm is not None and np.isfinite(mm["E"]) and abs(mm["E"] / ((n1 + n2) * CM.E_PER_E) - 1) < 0.1:
                    merged += 1
            res[name] = dict(resolved=good / len(pp), merged=merged / len(pp))
        out["pairs"][str(d)] = res
    out["time"] = time.time() - t0
    json.dump(out, open(os.path.join(RES, f"inject_{mod}_EXT{ext}_img{img}.json"), "w"), indent=1, default=float)
    return out


def run_real(mod, ext, img):
    Q, valid, sigma, lam_row, sw = RS.load(mod, img, ext)
    base, _ = CM.wader(Q, sw, 10, 30, 3.0, 8.5)
    side = {"front": [c for c in base if c["sxy"] < 1], "back": [c for c in base if c["sxy"] >= 1]}
    t = time.time(); bg = AM.background_map(Q, valid); t_bg = time.time() - t
    meth = make_methods(mod, img, lam_row, sw, valid)
    out = dict(mod=mod, ext=ext, img=img, n_base=dict(front=len(side["front"]), back=len(side["back"])),
               bg=dict(lam_mean=float(bg.lam[valid].mean()), lam_p90_row=float(np.percentile(bg.rows["lam_smooth"], 90)),
                       o_median=float(np.median(bg.rows["o"][np.isfinite(bg.rows["o"])])), s=float(np.sqrt(np.median(bg.s2))),
                       coord_lam_row_mean=float(np.mean(lam_row)), time=t_bg),
               methods={})
    # make the cached background the one we timed
    for name, f in meth.items():
        t = time.time()
        fl = f(Q)
        dt = time.time() - t
        r = dict(time=dt, n_clusters=len(fl), n_window=int(sum(1 for c in fl if np.isfinite(c["E"]) and 4.5 < c["E"] < 7.0)))
        for sd, cl in side.items():
            E = []
            for c in cl:
                m = RS.match(fl, c["y"], c["x"])
                E.append(m["E"] if m is not None else np.nan)
            r[sd] = RS.fit_ka(E)
            r[sd + "_matched"] = int(np.sum(np.isfinite(E)))
        if any("split" in c for c in fl):
            r["n_split_components"] = int(sum(1 for c in fl if "split" in c))
        out["methods"][name] = r
        print(mod, ext, img, name, f"{dt:.1f}s", r["front"], r["back"], flush=True)
    json.dump(out, open(os.path.join(RES, f"real_{mod}_EXT{ext}_img{img}.json"), "w"), indent=1, default=float)
    return out


def _job(args):
    kind, mod, ext, img = args
    try:
        return (run_inject if kind == "inject" else run_real)(mod, ext, img)
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(error=str(e), args=args)


if __name__ == "__main__":
    if sys.argv[1] in ("inject", "real"):
        _job((sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    elif sys.argv[1] == "all":
        from multiprocessing import Pool
        jobs = [("inject", m, e, i) for m in ("DM09", "DM10") for e in (4, 1) for i in (1, 0)]
        jobs += [("real", m, e, i) for m in ("DM09", "DM10") for e in (1, 4) for i in (0, 1, 2)]
        if len(sys.argv) > 3:          # only the jobs whose result file is missing
            jobs = [j for j in jobs if not os.path.exists(os.path.join(RES, f"{j[0]}_{j[1]}_EXT{j[2]}_img{j[3]}.json"))]
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 10) as p:
            p.map(_job, jobs, chunksize=1)
