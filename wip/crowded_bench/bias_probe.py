"""Physics-bias probe of A_GMX (audit 2026-11-14; $PROBE_REPO: the comm_scripts tree to test, default this one): isolated objects of known charge injected into every image type,
clustered as the pipeline does (adaptive_cluster engine gmx), measured against the truth.

    python bias_probe.py run OUT [-j 4] [--only scen ...]
    python bias_probe.py analyse OUT

Objects (truth in electrons, no Fano: the clustering is judged, not the physics of pair creation):
  X-rays  E in 0.3, 1, 3, 6, 15, 30, 60 keV (n = E / 3.75 eV), depth z uniform 0-675 um, DAMIC diffusion
          sigma^2 = -A ln(1 - b z) (A 285 um^2, b 8.2e-4 /um, 15 um px): sigma 0.0-1.6 px
  few-e   N = 2..10 electrons, the same depth law
Sites: a grid (48 px unbinned, 64 px when binned), shuffled; on real frames a site is used only when the real image
has no charge > 3 e- within 10 px (isolated: intrinsic bias, not merging), except on the crowded L2 frames (no free
sites: random, the local real charge recorded).
Per object: the cluster holding the largest share of its electrons (shares: min(truth, cluster charge) per bin);
complete, foreign, E ratio (cluster electrons / truth), centroid offset (bins, A_GMX convention: bin i centred at i),
width (unbinned: cluster sxy vs injected sigma).
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.environ.get("PROBE_REPO", os.path.expanduser("~/Documents/Software/comm_scripts"))
EPS = 3.75e-3
A_UM2, B_UM, THICK, PIX = 285.0, 8.2e-4, 675.0, 15.0
E_KEV = (0.3, 1.0, 3.0, 6.0, 15.0, 30.0, 60.0)
LINES = (5.9, 13.9, 17.7, 20.8, 26.3, 59.5)      # 'lines' scenarios: Fe-55 K-alpha and the Am-241 / Np L lines

# scenario: base, binning, crowded, fewe, synthetic params
SCEN = {
    "dm09_1x1":      dict(base="dm09", b=(1, 1), crowded="auto", fewe="auto"),
    "dm09_1x1_on":   dict(base="dm09", b=(1, 1), crowded="on", fewe="off"),
    "dm09_2x2":      dict(base="dm09", b=(2, 2), crowded="auto", fewe="auto"),
    "dm09_5x5":      dict(base="dm09", b=(5, 5), crowded="auto", fewe="auto"),
    "dm09_1x10":     dict(base="dm09", b=(1, 10), crowded="auto", fewe="auto"),
    "dm09_10x1":     dict(base="dm09", b=(10, 1), crowded="auto", fewe="auto"),
    "l2_1x1":        dict(base="l2", b=(1, 1), crowded="auto", fewe="off"),
    "l2_1x1_off":    dict(base="l2", b=(1, 1), crowded="off", fewe="off"),
    "lowdc_1x1":     dict(base="syn", b=(1, 1), crowded="auto", fewe="auto", lam=1e-4, noise=0.12, shape=(600, 2000)),
    "lowdc_10x10":   dict(base="syn", b=(10, 10), crowded="auto", fewe="auto", lam=1e-4, noise=0.12, shape=(1200, 2000)),
    "noisy_1x1":     dict(base="syn", b=(1, 1), crowded="auto", fewe="off", lam=1.8, noise=1.5, shape=(600, 1200)),
    "noisy_lam02":   dict(base="syn", b=(1, 1), crowded="auto", fewe="off", lam=0.2, noise=1.5, shape=(600, 1200)),
    "noisy_lam0":    dict(base="syn", b=(1, 1), crowded="auto", fewe="off", lam=0.0, noise=1.5, shape=(600, 1200)),
    # source-like: lines at random positions (no isolation), every object an X-ray
    "l2_lines":      dict(base="l2", b=(1, 1), crowded="auto", fewe="off", lines=True),
    "dm09_lines":    dict(base="dm09", b=(1, 1), crowded="auto", fewe="auto", lines=True),
    "dm09_lines_on": dict(base="dm09", b=(1, 1), crowded="on", fewe="off", lines=True),
}


def frames(base):
    if base == "dm09":
        return [(i, e) for i in range(3) for e in (1, 2, 3, 4)]
    if base == "l2":
        return [(i, e) for i in (9, 10) for e in (1, 2, 4)]
    return [(i, 0) for i in range(4)]


def sigma_px(z):
    return np.sqrt(-A_UM2 * np.log(1 - B_UM * np.minimum(z, THICK))) / PIX


def base_image(sc, img, ext, rng, out):
    """unbinned base [e-], valid, whole-electron flag (re-noised after binning)"""
    sys.path.insert(0, os.path.join(REPO, "wip", "crowded_bench"))
    import bench_crowded as BC
    if sc["base"] == "dm09":                      # whole electrons, cached in out (bench_crowded.load_dm09)
        e, v = BC.load_dm09(img, ext, out)
        return e, v, True, 0.16
    if sc["base"] == "l2":
        Q, v = BC.load(img, ext)
        return Q, v, False, 0.16
    H, W = sc["shape"]
    v = np.zeros((H, W), bool); v[3:-2, 2:-2] = True
    return rng.poisson(sc["lam"], (H, W)).astype(float), v, True, sc["noise"]


def one(task):
    try:
        return _one(task)
    except Exception as e:                       # noqa: BLE001  (recorded, the run goes on)
        name, img, ext, out = task
        f = os.path.join(out, f"{name}_{img}_{ext}.err")
        open(f, "w").write(f"{type(e).__name__}: {e}\n")
        return f


def _one(task):
    name, img, ext, out = task
    f = os.path.join(out, f"{name}_{img}_{ext}.json")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f):
        return f
    for p in ("clustering", "engine"):
        sys.path.insert(0, os.path.join(REPO, "superWADERs", p))
    import adaptive as AD
    import agent_methods as AM
    from scipy import ndimage as ndi
    sc = SCEN[name]
    by, bx = sc["b"]
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{name}_{img}_{ext}".encode()))
    Q, valid, whole, noise = base_image(sc, img, ext, rng, out)
    H, W = Q.shape
    # isolation: real charge > 3 e- (dark charge removed) within 10 px excludes a site
    if sc["base"] == "dm09" and not sc.get("lines"):
        bg0 = AM.background_map(Q + rng.normal(0, 0.16, Q.shape) * valid, valid)
        busy = ndi.binary_dilation((Q - bg0.mean > 4) & valid, iterations=6)
    else:
        busy = np.zeros((H, W), bool)
    sp = 32 if by * bx == 1 else 64
    gy, gx = np.mgrid[20:H - 20:sp, 20:W - 210 if sc["base"] == "l2" else W - 20:sp]
    sites = np.c_[gy.ravel(), gx.ravel()]
    rng.shuffle(sites)
    sites = [s for s in sites if not busy[s[0], s[1]] and valid[s[0], s[1]]][:300 if sc.get("lines") else 180]
    objs, add = [], np.zeros((H, W))
    for k, (sy, sx) in enumerate(sites):
        z = rng.uniform(0, THICK); s = float(sigma_px(z))
        if sc.get("lines"):
            e = LINES[k % len(LINES)]
            kind, n = "xray", int(round(e / EPS)); meta = dict(keV=e)
        elif k % 3 == 2:
            kind, n = "fewe", int(rng.integers(2, 11)); meta = dict(N=n)
        else:
            e = E_KEV[(k // 3 * 2 + k % 3) % len(E_KEV)]
            kind, n = "xray", int(round(e / EPS)); meta = dict(keV=e)
        jit = sp if sc.get("lines") else 1          # lines: anywhere in the cell (random, as a source)
        y0, x0 = sy + rng.uniform(0, jit), sx + rng.uniform(0, jit)
        ys, xs = y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n)
        iy, ix = np.floor(ys).astype(int), np.floor(xs).astype(int)
        ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        np.add.at(add, (iy[ok], ix[ok]), 1.0)
        objs.append((kind, ys[ok], xs[ok], dict(meta, sigma=s, z=float(z), n=int(ok.sum()))))
    Sc = np.where(valid, Q + add, 0.0)
    hb, wb = H // by, W // bx
    bin_ = lambda a: a[:hb * by, :wb * bx].reshape(hb, by, wb, bx).sum(axis=(1, 3))
    if whole:
        Sb = bin_(Sc) + rng.normal(0, noise, (hb, wb))
    else:
        Sb = Sc if by * bx == 1 else bin_(np.round(Sc)) + rng.normal(0, noise, (hb, wb))
    vb = bin_(valid.astype(float)) == by * bx
    Sb = np.where(vb, Sb, 0.0)
    try:
        bg = AM.background_map(Sb, vb) if noise < 1.0 else None
    except Exception:
        bg = None
    cl, info = AD.adaptive_cluster(Sb, vb, bg, engine="gmx", crowded=sc["crowded"], fewe=sc["fewe"], return_pixels=True,
                                   return_info=True, binning=(by, bx))
    cl = [c for c in cl if "_pix" in c and np.isfinite(c.get("E", np.nan)) and len(c["_pix"][0])]
    Ee = np.array([c["E"] / AM.E_PER_E for c in cl])
    PI = np.concatenate([np.asarray(c["_pix"][0], int) * wb + np.asarray(c["_pix"][1], int) for c in cl]) if cl else np.zeros(0, int)
    PK = np.concatenate([np.full(len(c["_pix"][0]), k) for k, c in enumerate(cl, 1)]) if cl else np.zeros(0, int)
    PQ = np.clip(np.concatenate([np.asarray(c["_pix"][2], float) for c in cl]), 0, None) if cl else np.zeros(0)
    res = []
    for kind, ys, xs, meta in objs:
        iy, ix = (np.floor(ys) // by).astype(int), (np.floor(xs) // bx).astype(int)
        T = np.zeros((hb, wb)); okb = (iy < hb) & (ix < wb)
        np.add.at(T, (iy[okb], ix[okb]), 1.0)
        tot = T.sum()
        if tot <= 0:
            continue
        Tf = T.ravel(); sel = Tf[PI] > 0
        sh = np.bincount(PK[sel], np.minimum(Tf[PI[sel]], PQ[sel]), len(cl) + 1); sh[0] = 0
        bi = int(sh.argmax()) if sh.size > 1 and sh.max() > 0 else 0
        ty, tx = float(np.mean(ys)) / by - 0.5, float(np.mean(xs)) / bx - 0.5
        loc = float(np.clip(Q[int(ys.mean()) - 3:int(ys.mean()) + 4, int(xs.mean()) - 3:int(xs.mean()) + 4], 0, None).sum())
        r = dict(kind=kind, tot=float(tot), real_local=loc, **meta)
        if bi:
            c = cl[bi - 1]
            r.update(complete=float(sh[bi] / tot), foreign=float((Ee[bi - 1] - sh[bi]) / tot), e_ratio=float(Ee[bi - 1] / tot),
                     n_rec=float(Ee[bi - 1]), dy=float(c["y"] - ty), dx=float(c["x"] - tx), sxy=float(c.get("sxy", np.nan)), sxy_px=float(c.get("sxy_px", np.nan)),
                     cls=c.get("cls"), finder=c.get("finder", "agmx"), regime=c.get("regime"),
                     n_frag=int((sh[1:] >= 0.05 * tot).sum()))
        else:
            r.update(complete=0.0, foreign=0.0, e_ratio=0.0)
        res.append(r)
    rec = dict(scen=name, img=img, ext=ext, crowded=bool(info.get("crowded_mode")), regime=info.get("regime"),
               fewe=(info.get("fewe") or {}).get("used") if isinstance(info.get("fewe"), dict) else None,
               n_sites=len(sites), n_clusters=len(cl), objects=res)
    json.dump(rec, open(f, "w"))
    return f


def run(out, jobs, only=None):
    os.makedirs(out, exist_ok=True)
    if any(SCEN[n]["base"] == "dm09" for n in SCEN if not only or n in only):
        sys.path.insert(0, os.path.join(REPO, "wip", "crowded_bench"))
        import bench_crowded as BC
        for i in range(3):
            BC.load_dm09(i, 1, out)               # the electron cache, before the workers
    tasks = [(n, i, e, out) for n in SCEN if not only or n in only for i, e in frames(SCEN[n]["base"])]
    tasks.sort(key=lambda t: SCEN[t[0]]["base"] != "l2")          # the slow L2 frames first
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for f in ex.map(one, tasks):
            print("done", os.path.basename(f), flush=True)


def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def analyse(out):
    by = {}
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        if f.endswith("report.json"):
            continue
        r = json.load(open(f))
        d = by.setdefault(r["scen"], dict(objs=[], crowded=0, n=0, regimes=set()))
        d["objs"] += r["objects"]; d["crowded"] += r["crowded"]; d["n"] += 1; d["regimes"].add(str(r["regime"]))
    rep = {}
    L = []
    for name, d in by.items():
        O = d["objs"]
        X = [o for o in O if o["kind"] == "xray"]; F = [o for o in O if o["kind"] == "fewe"]
        good = lambda o: o["complete"] >= 0.9 and o["foreign"] <= 0.1
        rows = []
        for e in E_KEV:
            xs = [o for o in X if o["keV"] == e]
            if not xs: continue
            g = [o for o in xs if good(o)]
            er = np.array([o["e_ratio"] for o in g]); com = np.array([o["complete"] for o in xs])
            # depth tertiles of sigma: energy and efficiency vs depth
            sg = np.array([o["sigma"] for o in xs]); cuts = np.quantile(sg, [1 / 3, 2 / 3])
            dep = []
            for lo, hi in ((-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 9)):
                xx = [o for o in xs if lo < o["sigma"] <= hi]; gg = [o for o in xx if good(o)]
                dep.append(dict(eff=float(np.mean([good(o) for o in xx])) if xx else np.nan,
                                e_ratio=float(np.median([o["e_ratio"] for o in gg])) if gg else np.nan))
            ws = [(o["sxy"], o["sigma"]) for o in g if np.isfinite(o.get("sxy", np.nan)) and o["sigma"] > 0.3]
            rows.append(dict(keV=e, n=len(xs), eff=float(np.mean([good(o) for o in xs])), complete_med=float(np.median(com)),
                             e_ratio_med=float(np.median(er)) if er.size else np.nan, e_ratio_mean=float(np.mean(er)) if er.size else np.nan,
                             e_ratio_iqr=[_q(er, 25), _q(er, 75)], e_ratio_all_med=float(np.median([o["e_ratio"] for o in xs])),
                             dy_med=float(np.median([o["dy"] for o in g])) if g else np.nan,
                             dx_med=float(np.median([o["dx"] for o in g])) if g else np.nan,
                             pos_rms=float(np.sqrt(np.mean([o["dy"] ** 2 + o["dx"] ** 2 for o in g]))) if g else np.nan,
                             width_ratio_med=float(np.median([a / b for a, b in ws])) if ws else np.nan, by_depth=dep))
        frows = []
        for N in range(2, 11):
            ff = [o for o in F if o["N"] == N]
            if not ff: continue
            found = [o for o in ff if o["complete"] >= 0.5]
            frows.append(dict(N=N, n=len(ff), found=float(len(found) / len(ff)),
                              n_rec_med=float(np.median([o["n_rec"] for o in found])) if found else np.nan,
                              n_rec_mean=float(np.mean([o["n_rec"] for o in found])) if found else np.nan,
                              bayes=float(np.mean([o.get("finder") == "bayes" for o in found])) if found else np.nan))
        rep[name] = dict(frames=d["n"], crowded=d["crowded"], regimes=sorted(d["regimes"]), xray=rows, fewe=frows)
        L.append(f"\n### {name}  ({d['n']} frames, crowded {d['crowded']}, regime {', '.join(sorted(d['regimes']))})\n")
        L.append("| keV | n | eff (>=90 %, <=10 % foreign) | E ratio median [IQR] | E ratio mean | E ratio all (median) | dy / dx median [bin] | pos rms | width ratio | eff shallow/mid/deep | E ratio shallow/mid/deep |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['keV']} | {r['n']} | {r['eff']:.3f} | {r['e_ratio_med']:.4f} [{r['e_ratio_iqr'][0]:.3f}, {r['e_ratio_iqr'][1]:.3f}] | "
                     f"{r['e_ratio_mean']:.4f} | {r['e_ratio_all_med']:.3f} | {r['dy_med']:+.3f} / {r['dx_med']:+.3f} | {r['pos_rms']:.3f} | "
                     f"{r['width_ratio_med']:.3f} | {' / '.join(f'{x['eff']:.2f}' for x in r['by_depth'])} | "
                     f"{' / '.join(f'{x['e_ratio']:.3f}' for x in r['by_depth'])} |")
        if frows:
            L.append("\n| N e- | n | found (>= 50 %) | N rec median | N rec mean | by Bayes finder |")
            L.append("|---|---|---|---|---|---|")
            for r in frows:
                L.append(f"| {r['N']} | {r['n']} | {r['found']:.2f} | {r['n_rec_med']:.2f} | {r['n_rec_mean']:.2f} | {r['bayes']:.2f} |")
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyse"))
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.out, a.j, a.only) if a.what == "run" else analyse(a.out)
