"""PWS - persistence-simplified watershed (topological data analysis / component-tree segmentation).

Idea: segment the (lightly smoothed) image into the basins of attraction of its significant local maxima
(watershed from markers), so a dense region never percolates into one cluster, then simplify the basin graph
with 0-dimensional persistence (elder rule): two adjacent basins are merged when the dip between them (peak of
the younger minus the saddle) is not significant given the Poisson + read noise (noise-normalised dynamics), or
when the saddle is high relative to the lower peak (contrast; one deposit with a shallow dip, e.g. along a track).
In bounded islands of significant pixels (sparse images) a lower contrast is enough to merge (tracks / worms with
deep but non-empty dips); in percolating (crowded) islands only the local contrast rule acts.
Energy = sum of R over the basin group (+ 1 ring of unassigned pixels); no model, no fit.

Detection seeds, background and the few-electron counting finder are shared with GMX."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
import gmx as GX
AM, M2, MH = GX.AM, GX.M2, GX.MH
E_PER_E = GX.E_PER_E
EIGHT = GX.EIGHT


def _saddles(lab, S):
    """adjacent label pairs (a < b) and their saddle = max over touching pixel pairs of min(S_p, S_q)"""
    H, W = lab.shape
    A, B, V = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = 0, H - dy
        xa0, xa1 = max(0, -dx), W - max(0, dx)
        la = lab[y0:y1, xa0:xa1]; lb = lab[y0 + dy:y1 + dy, xa0 + dx:xa1 + dx]
        sa = S[y0:y1, xa0:xa1]; sb = S[y0 + dy:y1 + dy, xa0 + dx:xa1 + dx]
        m = (la > 0) & (lb > 0) & (la != lb)
        a, b = la[m], lb[m]
        A.append(np.minimum(a, b)); B.append(np.maximum(a, b)); V.append(np.minimum(sa[m], sb[m]))
    A = np.concatenate(A); B = np.concatenate(B); V = np.concatenate(V)
    if A.size == 0:
        return A, B, V
    key = A.astype(np.int64) * (lab.max() + 1) + B
    order = np.lexsort((-V, key))
    key, A, B, V = key[order], A[order], B[order], V[order]
    first = np.r_[True, key[1:] != key[:-1]]
    return A[first], B[first], V[first]


def general_cluster(Q, valid, bg=None, scales=(0.5, 1.0, 1.5, 2.5), nsig=5.0, smooth=0.5, k_dyn=3.0, c_merge=0.5,
                    c_island=0.1, island_max=4000, link_alpha=0.01, hotcol=True, p_count=1e-5, count_E_max=100,
                    return_info=False):
    bg = GX.get_bg(Q, valid, bg)
    H, W = Q.shape
    Qc = Q
    if hotcol:
        try:
            ex, hot = MH.hot_columns(Q, valid, bg); Qc = Q - ex[None, :]
        except Exception:
            pass
    R = np.where(valid, Qc - bg.mean, 0.0)
    V = np.where(valid, bg.var, 0.0)
    counting = float(np.median(bg.s[valid])) < 0.5
    # seeds (as GMX)
    T = GX.mf_maps(R, V, scales)
    Tm = T.max(0)
    pk = (Tm == ndimage.maximum_filter(Tm, size=5, mode="constant")) & (Tm > nsig) & valid
    py, px = np.nonzero(pk)
    keep = GX.poisson_keep(Qc, valid, bg, py, px, nsig)
    py, px = py[keep], px[keep]
    S = ndimage.gaussian_filter(R, smooth) if smooth else R
    # noise of S per pixel (read noise + dark + Poisson of the signal), smoothing reduces it by 1 / (2 sqrt(pi) s)
    red = 1.0 / (2 * np.sqrt(np.pi) * smooth) if smooth else 1.0
    nS = red * np.sqrt(V + np.clip(S, 0, None))
    link = (Qc > M2.link_threshold(bg, link_alpha)) & valid
    markers = np.zeros((H, W), np.int32)
    markers[py, px] = np.arange(1, py.size + 1)
    support = (link | ndimage.binary_dilation(markers > 0, structure=EIGHT)) & valid
    lab = watershed(-S, markers, mask=support, connectivity=2)
    n = py.size
    out = []
    if n:
        peak = np.r_[0.0, S[py, px]]
        a, b, sv = _saddles(lab, S)
        # islands of the link mask: bounded (sparse) or percolating (crowded)
        il, nil = ndimage.label(support, structure=EIGHT)
        isz = np.bincount(il.ravel())
        bounded = isz[il[py, px]] <= island_max
        bnd = np.r_[False, bounded]
        # elder-rule union-find over saddles in decreasing order
        par = np.arange(n + 1); top = peak.copy()

        def find(u):
            while par[u] != u:
                par[u] = par[par[u]]; u = par[u]
            return u
        for k in np.argsort(-sv):
            u, v, s = find(a[k]), find(b[k]), sv[k]
            if u == v:
                continue
            lo, hi = (u, v) if top[u] < top[v] else (v, u)
            pl = top[lo]
            dyn_sig = (pl - s) < k_dyn * np.hypot(nS[py[lo - 1], px[lo - 1]] if lo > 0 else 0, red * np.sqrt(max(s, 0) + np.median(V[valid])))
            contrast = s > (c_island if (bnd[a[k]] and bnd[b[k]]) else c_merge) * pl
            if dyn_sig or contrast:
                par[lo] = hi
        root = np.array([find(u) for u in range(n + 1)])
        glab = root[lab]                     # label image of the merged groups (root ids; 0 = none)
        # 1 ring of unassigned pixels for the tails
        grown = ndimage.grey_dilation(glab, footprint=EIGHT)
        ring = (glab == 0) & (grown > 0) & valid
        G = np.where(ring, grown, glab)
        ids = np.unique(G[G > 0])
        sl = ndimage.find_objects(G)
        Esum = ndimage.sum(R, G, ids)
        Rp = np.clip(R, 0, None)
        for gid, E in zip(ids, Esum):
            s_ = sl[gid - 1]
            m = G[s_] == gid
            ys, xs = np.nonzero(m); ys = ys + s_[0].start; xs = xs + s_[1].start
            w = Rp[ys, xs]; sw = max(w.sum(), 1e-9)
            my, mx = (w * ys).sum() / sw, (w * xs).sum() / sw
            sxy = float(np.sqrt(max(0.5 * ((w * ((ys - my) ** 2 + (xs - mx) ** 2)).sum() / sw) - 1 / 12, 0)))
            c = dict(E=float(E * E_PER_E), y=float(my), x=float(mx), sxy=sxy, npix=int(m.sum()),
                     n_comp=int((root[1:] == gid).sum()))
            if counting and E < count_E_max:
                c["E"] = float((np.round(Qc[ys, xs] - bg.o[ys, xs]).sum() - bg.lam[ys, xs].sum()) * E_PER_E)
            out.append(c)
        # blend: highest saddle with another group relative to own peak
        bl = np.zeros(n + 1)
        ra, rb = root[a], root[b]
        diff = ra != rb
        for u, v, s in zip(ra[diff], rb[diff], sv[diff]):
            bl[u] = max(bl[u], s / max(peak[u], 1e-9)); bl[v] = max(bl[v], s / max(peak[v], 1e-9))
        for c, gid in zip(out, ids):
            c["blend"] = float(bl[gid])
            c["cls"] = ("low-E" if c["E"] < 0.5 else "alpha" if c["E"] > 1000 else
                        ("x-ray" if c["sxy"] < 1.0 else "x-ray diffuse") if c["n_comp"] == 1 else "extended")
    if counting:
        covered = ndimage.binary_dilation(lab > 0, structure=EIGHT, iterations=2)
        out += GX._counting(Qc, valid, bg, covered, p_count)
    if return_info:
        return out, dict(lab=lab)
    return out
