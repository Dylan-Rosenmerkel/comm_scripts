"""Scratch test of A_GMX changes on the crowded L2 ACM101 images (nothing in comm_scripts is modified).
Variants (cumulative):
  base    current A_GMX
  bgreg   regional background: dark charge per cell (30 rows x 64 cols) from a truncated-Poisson fit of ALL the
          cell's pixels (no cluster mask, so an extended glow that the mask swallows is still measured), replacing
          the row trend x mesh correction (o and read noise unchanged)
  link    + collinear linking of straight-track fragments (post-pass on the clusters)
python harness.py IMG EXT VARIANT  -> out/IMG_EXT_VARIANT.npz"""
import os, sys, json, glob, time
import numpy as np
CS = os.path.expanduser("~/Documents/Software/comm_scripts")
sys.path[:0] = [f"{CS}/clustering_tests", f"{CS}/src"]
import agent_methods as AM, adaptive as AD, gmx as GX, agmx_root as AR, gain_vs_charge as GV
from scipy import ndimage
RUN = os.path.expanduser("~/Documents/comm_scripts_data/L2_101_20260922/WADERs_analysis_L2_101_20260922")
STEM = {9: "avg_img_500x3500x500_1x1_L2_101_20260922_134507_9", 10: "avg_img_500x3500x500_1x1_L2_101_20260922_222502_10"}

def load(img, ext):
    from astropy.io import fits
    stem = STEM[img]
    wf = f"{RUN}/{stem}_EXT{ext}_waders.fits.bz2"
    with fits.open(wf) as h: cal = h[0].data.astype(float); nsbin = int(h[0].header.get("NSBIN", 1) or 1)
    fj = glob.glob(f"{RUN}/json/{stem}/final/{stem}_EXT{ext}.json")[0]
    ref = AR.read_reference(f"{RUN}/root/panaSKImg_clustersRec_{stem}_ext{ext}.root", ext, fj)
    curve = GV.load_model(f"{RUN}/gain_vs_charge/module_101_gain_vs_charge.json", ext)
    adu = cal * ref["gain"]
    Q = GV.electrons_with_anchor(adu, curve, ref["gain"], "comb").reshape(cal.shape)
    valid = AR.valid_mask(Q.shape, fj, nsbin)
    return np.where(valid, Q, 0.0), valid

def regional_bg(Q, valid, bg, rb=30, cb=64, lam_ok=4.0, use_mask=False, min_free=0.3):
    """regional background: ONE pedestal per row (electronics), the dark charge free per cell.
    1. per cell (rb x cb px, all pixels, no cluster mask): fit_row_fast -> (o_c, lam_c); the fractional pedestal is
       well measured everywhere, the whole-electron part only where lam_c is small (at high charge 'pedestal + N'
       and 'lam + N' look alike)
    2. per row: o = its fractional pedestal + the whole-electron offset I0 = median over the unambiguous cells
       (lam_c < lam_ok) of round(o_c - frac)
    3. per cell: lam = truncated-Poisson MLE of n = round(Q - o) (K from the cell's own level, iterated);
       bilinear interpolation between cell centres; A2 miscount correction as background_map"""
    from scipy import stats
    from scipy.interpolate import RegularGridInterpolator
    H, W = Q.shape
    rr = np.arange(0, H, rb); cc = np.arange(0, W, cb)
    oc = np.full((rr.size, cc.size), np.nan); lc = np.full_like(oc, np.nan)
    for a, r0 in enumerate(rr):
        for b, c0 in enumerate(cc):
            v = Q[r0:r0 + rb, c0:c0 + cb][valid[r0:r0 + rb, c0:c0 + cb]]
            if v.size < 300: continue
            f = AM.fit_row_fast(v)
            if f["resolved"]: oc[a, b], lc[a, b] = f["o"], f["lam"]
    frac_r = bg.o[:, 0] - np.round(bg.o[:, 0])                       # fractional pedestal of each row (comb phase)
    fr_c = np.repeat(frac_r[np.minimum(rr + rb // 2, H - 1)][:, None], cc.size, 1)
    ok = np.isfinite(oc) & (lc < lam_ok)
    I0 = int(np.median(np.round(oc[ok] - fr_c[ok]))) if ok.sum() >= 5 else None
    o_row = frac_r + I0 if I0 is not None else bg.o[:, 0]
    o_map = np.repeat(o_row[:, None], W, 1)
    n = np.round(Q - o_map).astype(int)
    grid = np.full((rr.size, cc.size), np.nan)
    for a, r0 in enumerate(rr):
        for b, c0 in enumerate(cc):
            vm = valid[r0:r0 + rb, c0:c0 + cb]
            fm = vm & ~bg.mask[r0:r0 + rb, c0:c0 + cb]
            # hybrid: the pixels outside the cluster mask when enough of the cell is free (no halo charge), else all
            sel = fm if (use_mask and fm.sum() >= min_free * max(vm.sum(), 1)) else vm
            v = n[r0:r0 + rb, c0:c0 + cb][sel]
            if v.size < 300: continue
            lam = max(float(np.median(v)), 0.05)
            for _ in range(5):
                K = max(3, int(stats.poisson.isf(1e-3, lam)))
                if np.mean((v >= 0) & (v <= K)) < 0.25: lam = np.nan; break
                lam = AM.truncated_poisson_mle(v, K)
            grid[a, b] = lam
    if np.isnan(grid).any():                                         # cells covered by deposits: nearest estimate
        from scipy import ndimage as ND
        idx = ND.distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
        grid = grid[tuple(idx)]
    ip = RegularGridInterpolator((np.minimum(rr + rb / 2, H - 1), np.minimum(cc + cb / 2, W - 1)), grid, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lam_map = np.maximum(ip(np.stack([yy.ravel(), xx.ravel()], 1)).reshape(H, W), AM.LAM_FLOOR)
    p = stats.norm.sf(0.5 / np.sqrt(np.maximum(bg.s2, 1e-6)))
    lam_map = np.maximum((lam_map - p) / np.maximum(1 - 2 * p, 1e-3), AM.LAM_FLOOR)
    return AM.Background(o_map, lam_map, bg.s2, bg.mask, dict(bg.rows, regional=grid, I0=I0)), grid, oc

def pca(ys, xs, w):
    w = np.clip(w, 0, None) + 1e-9; cy, cx = np.average(ys, weights=w), np.average(xs, weights=w)
    C = np.cov(np.vstack([ys - cy, xs - cx]), aweights=w)
    ev, V = np.linalg.eigh(C); u = V[:, 1]
    t = (ys - cy) * u[0] + (xs - cx) * u[1]; perp = -(ys - cy) * u[1] + (xs - cx) * u[0]
    return np.array([cy, cx]), u, t.min(), t.max(), np.sqrt(max(ev[0], 0)), np.sqrt(max(ev[1], 0))

def link_tracks(cl, ang=4.0, gap=15.0, gap_long=25.0, off=2.5, min_len=12.0, min_part=5.0, max_w_over_l=0.12, max_w_long=0.2):
    """merge straight fragments of one track (union-find over all qualifying pairs; E, pixels and centre recomputed).
    A pair links when, with the longer piece L (>= min_len px) and the shorter piece S (>= min_part px):
      - S's endpoints lie within off_L = max(off, 1.5 + 0.01 * len(L)) px of L's axis line (and vice versa for long S)
      - the axes are parallel within `ang` degrees when S is >= min_len long (a short piece's angle is unreliable)
      - the nearest endpoints are < gap px apart (< gap_long px when both are >= 30 px and the offset is < 2 px)
      - both are straight: width / length < max_w_over_l (< max_w_long for pieces >= 50 px, which may carry branches)"""
    T = []
    for k, c in enumerate(cl):
        ys, xs, q = c["_pix"]
        if ys.size < 5: continue
        cen, u, t0, t1, wt, wl = pca(ys.astype(float), xs.astype(float), q)
        Lk = t1 - t0
        if Lk >= min_part and wt / max(Lk, 1e-9) < (max_w_long if Lk >= 50 else max_w_over_l):
            T.append((k, cen + t0 * u, cen + t1 * u, u, Lk))
    par = list(range(len(cl)))
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    from scipy.spatial import cKDTree
    ends = np.array([p for (_, a, b, _, _) in T for p in (a, b)]) if T else np.zeros((0, 2))
    tree = cKDTree(ends) if len(ends) else None
    perp = lambda p, o, u: abs((p - o)[0] * u[1] - (p - o)[1] * u[0])
    nl = 0
    for i, Ti in enumerate(T):
        for e in (Ti[1], Ti[2]):
            for jj in tree.query_ball_point(e, gap_long):
                j = jj // 2
                if j <= i: continue
                Tj = T[j]
                Lg, Sh = (Ti, Tj) if Ti[4] >= Tj[4] else (Tj, Ti)
                if Lg[4] < min_len: continue
                if Sh[4] >= min_len and abs(np.dot(Lg[3], Sh[3])) < np.cos(np.radians(ang)): continue
                o = max(perp(Sh[1], Lg[1], Lg[3]), perp(Sh[2], Lg[1], Lg[3]))
                if o >= max(off, 1.5 + 0.01 * Lg[4]): continue
                g = min(np.linalg.norm(p - q) for p in (Lg[1], Lg[2]) for q in (Sh[1], Sh[2]))
                if g >= (gap_long if (Sh[4] >= 30 and o < 2.0) else gap): continue
                if find(Ti[0]) != find(Tj[0]):
                    par[find(Ti[0])] = find(Tj[0]); nl += 1
    groups = {}
    for k in range(len(cl)): groups.setdefault(find(k), []).append(k)
    out = []
    for g in groups.values():
        if len(g) == 1: out.append(cl[g[0]]); continue
        ys = np.concatenate([cl[k]["_pix"][0] for k in g]); xs = np.concatenate([cl[k]["_pix"][1] for k in g]); q = np.concatenate([cl[k]["_pix"][2] for k in g])
        c = dict(cl[max(g, key=lambda k: cl[k]["E"])]); c["E"] = float(sum(cl[k]["E"] for k in g)); c["_pix"] = (ys, xs, q)
        w = np.clip(q, 0, None) + 1e-9; c["y"], c["x"] = float(np.average(ys, weights=w)), float(np.average(xs, weights=w)); c["n_linked"] = len(g); c["cls"] = "muon"
        out.append(c)
    return out, nl

if __name__ == "__main__":
    img, ext, var = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time(); Q, valid = load(img, ext)
    bg = AM.background_map(Q, valid); extra = {}
    if var in ("bgreg", "link", "bgmask", "linkm"):
        bg, grid, ref = regional_bg(Q, valid, bg, use_mask=var in ("bgmask", "linkm")); extra = dict(grid=grid, ref=ref, o_used=bg.o[:, 0])
    cl, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True)
    cl = [c for c in cl if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
    nl = 0
    if var in ("link", "linkm"):
        cl, nl = link_tracks(cl)
    # label image + table
    lab = np.zeros(Q.shape, np.int32)
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; lab[ys, xs] = k
    keys = ["E", "x", "y", "sxy", "blend", "ped"]
    tab = {k: np.array([c.get(k, np.nan) for c in cl], float) for k in keys}
    tab["npix"] = np.array([c["_pix"][0].size for c in cl]); tab["cls"] = np.array([c.get("cls", "") for c in cl])
    os.makedirs("out", exist_ok=True)
    np.savez_compressed(f"out/{img}_{ext}_{var}.npz", lab=lab, lam=bg.lam.astype(np.float32), n_linked=nl, sec=time.time() - t0, **tab, **extra)
    print(f"{img} EXT{ext} {var}: {len(cl)} clusters, regime {info.get('regime')}, linked {nl}, {time.time() - t0:.0f} s", flush=True)
