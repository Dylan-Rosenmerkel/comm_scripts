"""Crowded-image benchmark with truth for A_GMX (plan 2.7), unbinned and binned: muons and X-rays injected into the
real crowded L2 ACM101 images (2026-09-22, images 9 and 10, core fill 0.27-0.31), the same scene binned in software.

    python bench_crowded.py run OUT [--bins 1x1 2x2 5x5 1x10 10x1] [-j 4]     one JSON per (image, CCD, binning)
    python bench_crowded.py analyse OUT                                      OUT/report.md, report.json

Scene: the calibrated image [e-] of run WADERs_analysis_L2_101_20260922 (linear gain; the injected truth is in e- as
well) + 25 muons (wip/agmx_crowded/inject_mu.sim_muons: L 40-400 px, Moyal knots, width 0.2 -> 0.8-1.6 px with depth)
+ 40 X-rays (1-8 keV, DAMIC diffusion, depth uniform). Binned b_y x b_x: the scene summed per bin (the image's own read
noise is summed too: ~0.15 e- x b, small next to the tracks). A_GMX = superWADERs/clustering adaptive_cluster(engine gmx,
crowded 'auto', binning) with background_map.
--base dm09: the scene is DM09 Image_5 (Fe-55, 3 images x 4 CCDs, whole electrons, core fill ~0.02 unbinned) with
muons and X-rays at the same density per area; binned scenes get one read noise (0.16 e-) per bin.
Scores (charge shares: a cluster holds min(the object's electrons, the cluster's charge) of the object in each bin;
complete_own = the older dominant-cluster label image, where a shared bin goes wholly to its largest cluster):
  muon     complete = largest share of its charge in one cluster; foreign = (that cluster's charge - the muon's) /
           muon charge; n_frag = clusters with >= 5 %; correct = complete >= 0.9 and foreign <= 0.1; overlap = real
           pixels > 50 e- in its footprint (dilated 2 px), binned by tertiles in the report
  X-ray    clean = complete >= 0.9 and foreign <= 0.1; E ratio = its best cluster's charge / its charge
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.expanduser("~/Documents/comm_scripts_data/L2_101_20260922/WADERs_analysis_L2_101_20260922")
STEM = {9: "avg_img_500x3500x500_1x1_L2_101_20260922_134507_9", 10: "avg_img_500x3500x500_1x1_L2_101_20260922_222502_10"}
EXTS = (1, 2, 4)
EPS = 3.75e-3
NOISE_E = 0.16             # e-: read noise of the L2 images (one readout per bin)
A_UM2, B_UM, THICK, PIX = 285.0, 8.2e-4, 675.0, 15.0


def sim_muons(rng, shape, n_mu, margin=10, right=200):
    """the muon model of wip/agmx_crowded/inject_mu.py (copied: that module imports the old clustering_tests)"""
    from scipy.stats import moyal
    dedx = 0.27 / 3.75e-3
    H, W = shape
    img = np.zeros(shape)
    truth = []
    for _ in range(n_mu):
        L = rng.uniform(40, 400)
        th = rng.uniform(0, np.pi)
        dy, dx = np.sin(th) * L, np.cos(th) * L
        for _ in range(50):
            y0 = rng.uniform(margin, H - margin)
            x0 = rng.uniform(margin, W - right - margin)
            if 0 <= y0 + dy < H and 0 <= x0 + dx < W - right:
                break
        else:
            continue
        s_back = rng.uniform(0.8, 1.6)
        front_first = rng.random() < 0.5
        nstep = int(np.ceil(L))
        step_um = np.hypot(15 * L, 675) / nstep
        ys, xs = [], []
        for k in range(nstep):
            t = (k + rng.random()) / nstep
            q = max(dedx * step_um + 0.4 * dedx * step_um * moyal.rvs(random_state=rng), 0)
            n = rng.poisson(q)
            depth = t if front_first else 1 - t
            s = 0.2 + (s_back - 0.2) * depth
            ys.append(y0 + t * dy + rng.normal(0, s, n))
            xs.append(x0 + t * dx + rng.normal(0, s, n))
        truth.append(("muon", np.concatenate(ys), np.concatenate(xs), dict(L=float(L), th=float(np.degrees(th)))))
    return truth


def sim_xrays(rng, shape, n, margin=10, right=200):
    H, W = shape
    out = []
    for _ in range(n):
        e = rng.uniform(1.0, 8.0)
        z = rng.uniform(0, THICK)
        s = np.sqrt(-A_UM2 * np.log(1 - B_UM * z)) / PIX
        y0, x0 = rng.uniform(margin, H - margin), rng.uniform(margin, W - right - margin)
        ne = int(e / EPS)
        out.append(("xray", y0 + rng.normal(0, s, ne), x0 + rng.normal(0, s, ne), dict(keV=float(e), sigma=float(s))))
    return out


def load(img, ext):
    from astropy.io import fits
    with fits.open(f"{RUN}/{STEM[img]}_EXT{ext}_waders.fits.bz2") as h:
        Q = np.asarray(h[0].data, float)
    valid = np.zeros(Q.shape, bool)
    valid[3:Q.shape[0] - 2, 10:Q.shape[1] - 200] = True
    return np.where(valid, Q, 0.0), valid


def load_dm09(img, ext, out):
    """DM09 Image_5 (Fe-55, 480 x 1280) as whole electrons (wip/binned/bench_binned.electrons), cached in out"""
    cache = os.path.join(out, f"dm09_{img}_e.npz")
    if not os.path.exists(cache):
        sys.path.insert(0, os.path.join(REPO, "wip", "binned"))
        import bench_binned as BB
        E = BB.electrons(BB.images()[img])
        np.savez_compressed(cache, **{f"e{k}": v[0] for k, v in E.items()})
    e = np.load(cache)[f"e{ext}"].astype(float)
    valid = np.zeros(e.shape, bool)
    valid[3:e.shape[0] - 2, 2:e.shape[1] - 2] = True
    return np.where(valid, e, 0.0), valid


def one(task):
    base, img, ext, b, out = task
    by, bx = (int(v) for v in b.split("x"))
    f = os.path.join(out, f"{base}_{img}_{ext}_{b}.json")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f):
        return f
    for p in ("clustering", "engine"):
        sys.path.insert(0, os.path.join(REPO, "superWADERs", p))
    import adaptive as AD
    import agent_methods as AM
    if base == "dm09":                    # counts per area as on L2 (25 muons, 40 X-rays per 1.63 Mpix)
        Q, valid = load_dm09(img, ext, out)
        sc, right = valid.sum() / 1.63e6, 2
    else:
        Q, valid = load(img, ext)
        sc, right = 1.0, 200
    rng = np.random.default_rng(1000 + 10 * img + ext)
    objs = sim_muons(rng, Q.shape, int(round(25 * sc)), right=right) + sim_xrays(rng, Q.shape, int(round(40 * sc)), right=right)
    H, W = Q.shape
    add = np.zeros(Q.shape)
    for _, ys, xs, _ in objs:
        iy, ix = np.floor(ys).astype(int), np.floor(xs).astype(int)
        ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        np.add.at(add, (iy[ok], ix[ok]), 1.0)
    real = Q.copy()
    S = np.where(valid, Q + add, 0.0)
    hb, wb = H // by, W // bx
    bin_ = lambda a: a[:hb * by, :wb * bx].reshape(hb, by, wb, bx).sum(axis=(1, 3))
    # binning sums the charge before the one readout: whole electrons (the calibrated image rounded) binned, then one
    # read noise (unbinned: the image as it is)
    if base == "dm09":                    # whole electrons: one read noise per bin at any binning
        Sb = bin_(S) + np.random.default_rng(7 + img + ext).normal(0, NOISE_E, (hb, wb))
    else:
        Sb = S if by * bx == 1 else bin_(np.round(S)) + np.random.default_rng(7 + img + ext).normal(0, NOISE_E, (hb, wb))
    vb = bin_(valid.astype(float)) == by * bx
    bg = AM.background_map(np.where(vb, Sb, 0.0), vb)
    cl, info = AD.adaptive_cluster(np.where(vb, Sb, 0.0), vb, bg, engine="gmx", crowded=os.environ.get("BENCH_CROWDED", "auto"), return_pixels=True,
                                   return_info=True, binning=(by, bx))
    cl = [c for c in cl if "_pix" in c and np.isfinite(c.get("E", np.nan)) and len(c["_pix"][0])]
    own, best = np.zeros((hb, wb), np.int32), np.zeros((hb, wb))
    for k, c in enumerate(cl, 1):
        ys_, xs_, q_ = c["_pix"]
        ys_, xs_, q_ = np.asarray(ys_, int), np.asarray(xs_, int), np.asarray(q_, float)
        m = q_ > best[ys_, xs_]
        own[ys_[m], xs_[m]] = k
        best[ys_[m], xs_[m]] = q_[m]
    Ee = np.array([c["E"] / EPS for c in cl]) if cl else np.zeros(0)
    # every pixel's charge shares: a cluster holds min(the object's electrons, its charge) of the object in each bin
    PI = np.concatenate([np.asarray(c["_pix"][0], int) * wb + np.asarray(c["_pix"][1], int) for c in cl]) if cl else np.zeros(0, int)
    PK = np.concatenate([np.full(len(c["_pix"][0]), k) for k, c in enumerate(cl, 1)]) if cl else np.zeros(0, int)
    PQ = np.clip(np.concatenate([np.asarray(c["_pix"][2], float) for c in cl]), 0, None) if cl else np.zeros(0)
    from scipy import ndimage as ndi
    res = []
    for kind, ys, xs, meta in objs:
        uy, ux = np.floor(ys).astype(int), np.floor(xs).astype(int)       # overlap on the unbinned scene
        uo = (uy >= 0) & (uy < H) & (ux >= 0) & (ux < W)
        U = np.zeros((H, W), bool)
        U[uy[uo], ux[uo]] = True
        overlap = int((real[ndi.binary_dilation(U, iterations=2)] > 50).sum())
        iy, ix = (np.floor(ys) // by).astype(int), (np.floor(xs) // bx).astype(int)
        ok = (iy >= 0) & (iy < hb) & (ix >= 0) & (ix < wb)
        T = np.zeros((hb, wb))
        np.add.at(T, (iy[ok], ix[ok]), 1.0)
        nz = np.nonzero(T)
        q = T[nz]
        tot = q.sum()
        if tot <= 0:
            continue
        k = own[nz]
        sh_own = np.bincount(k, q, len(cl) + 1)
        sh_own[0] = 0
        Tf = T.ravel()
        sel = Tf[PI] > 0
        sh = np.bincount(PK[sel], np.minimum(Tf[PI[sel]], PQ[sel]), len(cl) + 1)
        sh[0] = 0
        bi = int(sh.argmax())
        comp = sh[bi] / tot if bi else 0.0
        foreign = (Ee[bi - 1] - sh[bi]) / tot if bi else 0.0
        merged = 0.0                     # charge of the cluster outside the object's footprint (+1 bin): other events
        if bi:
            F = ndi.binary_dilation(T > 0, iterations=1)
            py, px, pq = (np.asarray(v) for v in cl[bi - 1]["_pix"])
            merged = float(np.asarray(pq, float)[~F[py.astype(int), px.astype(int)]].sum() / tot)
        res.append(dict(kind=kind, n=float(tot), complete=float(comp), complete_own=float(sh_own.max() / tot), foreign=float(foreign), merged=merged,
                        n_frag=int((sh[1:] >= 0.05 * tot).sum()), e_ratio=float(Ee[bi - 1] / tot) if bi else 0.0,
                        overlap=overlap, cls=cl[bi - 1].get("cls") if bi else None, **meta))
    rec = dict(base=base, img=img, ext=ext, binning=b, crowded=bool(info.get("crowded_mode")), core_fill=info.get("core_fill"),
               n_clusters=len(cl), objects=res)
    json.dump(rec, open(f, "w"))
    return f


def run(out, bins, jobs, base="l2"):
    os.makedirs(out, exist_ok=True)
    ids = [(i, e) for i in STEM for e in EXTS] if base == "l2" else [(i, e) for i in range(3) for e in (1, 2, 3, 4)]
    tasks = [(base, i, e, b, out) for b in bins for i, e in ids]
    if base == "dm09":
        for i in range(3):
            load_dm09(i, 1, out)              # the electron cache, before the workers
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for f in ex.map(one, tasks):
            print("done", os.path.basename(f), flush=True)


def analyse(out):
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "*_*_*_*.json"))):
        r = json.load(open(f))
        d = rows.setdefault(r["binning"], dict(mu=[], xr=[], crowded=0, n=0))
        d["n"] += 1
        d["crowded"] += r["crowded"]
        for o in r["objects"]:
            (d["mu"] if o["kind"] == "muon" else d["xr"]).append(o)
    rep, L = {}, ["| binning | CCDs crowded | muons correct | >= 90 % whole | median complete | median foreign | "
                  "fragments (mean) | correct by overlap (low / mid / high) | X-rays clean | X-ray E ratio (median) | "
                  "muons correct (merged) | X-rays clean (merged) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, d in sorted(rows.items()):
        mu, xr = d["mu"], d["xr"]
        ov = np.array([o["overlap"] for o in mu])
        cuts = np.quantile(ov, [1 / 3, 2 / 3]) if ov.size else [0, 0]
        corr = lambda o: o["complete"] >= 0.9 and o["foreign"] <= 0.1
        corm = lambda o: o["complete"] >= 0.9 and o.get("merged", o["foreign"]) <= 0.1   # pixels shared with real charge allowed
        by_ov = [np.mean([corr(o) for o, v in zip(mu, ov) if lo <= v < hi]) if any(lo <= v < hi for v in ov) else np.nan
                 for lo, hi in ((-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9))]
        rep[b] = dict(ccds=d["n"], crowded=d["crowded"], muons=len(mu), correct=float(np.mean([corr(o) for o in mu])),
                      whole=float(np.mean([o["complete"] >= 0.9 for o in mu])),
                      complete_med=float(np.median([o["complete"] for o in mu])),
                      foreign_med=float(np.median([o["foreign"] for o in mu])),
                      frag_mean=float(np.mean([o["n_frag"] for o in mu])), correct_by_overlap=[float(x) for x in by_ov],
                      xrays=len(xr), xray_clean=float(np.mean([corr(o) for o in xr])),
                      xray_eratio=float(np.median([o["e_ratio"] for o in xr])),
                      correct_merged=float(np.mean([corm(o) for o in mu])), xray_clean_merged=float(np.mean([corm(o) for o in xr])))
        v = rep[b]
        L.append(f"| {b} | {v['crowded']}/{v['ccds']} | {v['correct']:.2f} | {v['whole']:.2f} | {v['complete_med']:.2f} | "
                 f"{v['foreign_med']:.3f} | {v['frag_mean']:.2f} | {' / '.join(f'{x:.2f}' for x in v['correct_by_overlap'])} | "
                 f"{v['xray_clean']:.2f} | {v['xray_eratio']:.3f} | {v['correct_merged']:.2f} | {v['xray_clean_merged']:.2f} |")
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "analyse", "all"))
    ap.add_argument("out")
    ap.add_argument("--bins", nargs="*", default=["1x1", "2x2", "5x5", "1x10", "10x1"])
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--base", choices=("l2", "dm09"), default="l2", help="scene: crowded L2 surface images or DM09 Fe-55")
    a = ap.parse_args()
    if a.what in ("run", "all"):
        run(a.out, a.bins, a.j, a.base)
    if a.what in ("analyse", "all"):
        analyse(a.out)


if __name__ == "__main__":
    main()
