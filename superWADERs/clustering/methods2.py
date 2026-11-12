"""Two ways to stop MFD+G from breaking tracks / big events into pieces.

(1) track_first:   find big / extended objects first (connected pixels above a Poisson-tail threshold, kept if
                   they are long or elongated), take their energy from the grown connected region, mask them,
                   then run MFD+G on the rest of the image.
(2) mfdg_merge:    run MFD+G everywhere, then decide which of its clusters belong together:
     'B'      - clusters whose peak pixels lie in the same connected region of the B (auto) threshold mask
                (pixel >= k e-, k = Poisson tail of the local dark charge at alpha = 0.10);
     'saddle' - clusters joined through pixels above a Poisson-tail threshold (alpha = 0.01) are merged only if
                the charge along the best path between them never drops below a fraction f of the smaller peak
                (the 'saddle' or 'contrast' test of SExtractor's deblending and of watershed / component trees,
                used here in reverse: a track has no deep valley between its pieces, two X-rays do).
"""
import os, sys
import numpy as np
from scipy import ndimage
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import agent_methods as AM
E_PER_E = AM.E_PER_E
EIGHT = np.ones((3, 3), bool)


def link_threshold(bg, alpha):
    """Poisson-tail threshold map; where the read noise does not resolve single electrons (s >= 0.5 e-) a
    tail fraction alpha of Gaussian noise pixels would percolate, so require mean + 4 sqrt(var) there."""
    t = AM.tail_threshold_map(bg, alpha)
    noisy = bg.s >= 0.5
    if noisy.any():
        t = np.where(noisy, np.maximum(t, bg.mean + 4 * np.sqrt(bg.var)), t)
    return t


def counting_energy(Q, bg, cl):
    """Few-electron mode: E = sum over the cluster's pixels of round(Q - o) minus the expected dark charge"""
    own = owner_map(Q.shape, cl)
    ys, xs = np.nonzero(own >= 0); o = own[ys, xs]
    n = np.round(Q[ys, xs] - bg.o[ys, xs]); l = bg.lam[ys, xs]
    sn = np.bincount(o, n, len(cl)); sl = np.bincount(o, l, len(cl))
    out = []
    for i, c in enumerate(cl):
        out.append(dict(c, E=float((sn[i] - sl[i]) * E_PER_E) if np.isfinite(c["E"]) else np.nan))
    return out


def owner_map(shape, cl):
    """pixel -> index of the MFD+G cluster that owns it (disk or grown pixel), -1 otherwise"""
    own = AM.assign_footprints(shape, cl)
    for i, c in enumerate(cl):
        gp = c.get("grow_pix")
        if gp is not None and len(gp[0]):
            own[gp] = i
    return own


def _merge_groups(Q, bg, cl, own, groups, extra=None):
    """groups: list of lists of cluster indices. extra: optional label image of extra pixels (component id per
    pixel) whose unowned pixels are added to the group with that component. Returns new cluster list."""
    out = []
    for g, members in enumerate(groups):
        cs = [cl[i] for i in members]
        if len(cs) == 1 and extra is None:
            out.append(dict(cs[0], n_merged=1)); continue
        E = sum(c["E"] for c in cs)
        if extra is not None and cs[0].get("_comp", 0) > 0:
            comp = cs[0]["_comp"]
            sl = extra[1][comp - 1]
            m = (extra[0][sl] == comp) & (own[sl] < 0)
            if m.any():
                E += float((Q[sl] - bg.mean[sl])[m].sum() * E_PER_E)
        w = np.array([max(c["E"], 1e-6) if np.isfinite(c["E"]) else 1e-6 for c in cs])
        y = float((w * [c["y"] for c in cs]).sum() / w.sum()); x = float((w * [c["x"] for c in cs]).sum() / w.sum())
        out.append(dict(E=E, y=y, x=x, sxy=max(c["sxy"] for c in cs), n_merged=len(cs), npix=sum(c["npix"] for c in cs)))
    return out


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def mfdg_merge_B(Q, valid, bg, cl=None, alpha=0.10):
    if cl is None:
        cl, _ = AM.mf_detect(Q, valid, bg)
    own = owner_map(Q.shape, cl)
    M = (Q > link_threshold(bg, alpha)) & valid
    lab, n = ndimage.label(M, structure=EIGHT)
    objs = ndimage.find_objects(lab)
    uf = UF(len(cl)); first = {}
    for i, c in enumerate(cl):
        L = lab[c["yp"], c["xp"]]
        c["_comp"] = int(L)
        if L > 0:
            if L in first: uf.union(i, first[L])
            else: first[L] = i
    groups = {}
    for i in range(len(cl)):
        groups.setdefault(uf.find(i), []).append(i)
    # only groups that live in one B component get the component's unowned pixels
    return _merge_groups(Q, bg, cl, own, list(groups.values()), extra=(lab, objs))


def mfdg_merge_saddle(Q, valid, bg, cl=None, f=0.5, alpha=0.01, smooth=0.7):
    """Flooding (descending charge) of the pixels above the alpha tail threshold, smoothed residual S; when two
    flooded regions that each contain a cluster peak meet at level h, their dominant clusters are merged if
    h >= f * min(peak_a, peak_b)."""
    if cl is None:
        cl, _ = AM.mf_detect(Q, valid, bg)
    own = owner_map(Q.shape, cl)
    R = np.where(valid, Q - bg.mean, 0.0)
    S = ndimage.gaussian_filter(R, smooth, mode="constant")
    D = ((Q > link_threshold(bg, alpha)) & valid)
    for c in cl:
        D[c["yp"], c["xp"]] = True
    lab, n = ndimage.label(D, structure=EIGHT)
    uf = UF(len(cl))
    bycomp = {}
    for i, c in enumerate(cl):
        bycomp.setdefault(lab[c["yp"], c["xp"]], []).append(i)
    objs = ndimage.find_objects(lab)
    for L, members in bycomp.items():
        if L == 0 or len(members) < 2:
            continue
        sl = objs[L - 1]
        m = lab[sl] == L
        ys, xs = np.nonzero(m)
        vals = S[sl][m]
        idx = -np.ones(m.shape, int); idx[ys, xs] = np.arange(ys.size)
        peak_of = {}
        mark = -np.ones(ys.size, int)
        for i in members:
            yy, xx = cl[i]["yp"] - sl[0].start, cl[i]["xp"] - sl[1].start
            mark[idx[yy, xx]] = i
            peak_of[i] = float(S[cl[i]["yp"], cl[i]["xp"]])
        order = np.argsort(-vals)
        par = np.arange(ys.size); dom = mark.copy(); done = np.zeros(ys.size, bool)
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for p in order:
            done[p] = True
            h = vals[p]
            y, x = ys[p], xs[p]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                        q = idx[yy, xx]
                        if q < 0 or not done[q]: continue
                        a, b = find(p), find(q)
                        if a == b: continue
                        da, db = dom[a], dom[b]
                        if da >= 0 and db >= 0:
                            if h >= f * min(peak_of[da], peak_of[db]):
                                uf.union(da, db)
                            nd = da if peak_of[da] >= peak_of[db] else db
                        else:
                            nd = da if da >= 0 else db
                        par[a] = b; dom[b] = nd
    groups = {}
    for i in range(len(cl)):
        groups.setdefault(uf.find(i), []).append(i)
    return _merge_groups(Q, bg, cl, own, list(groups.values()))


def track_first(Q, valid, bg, alpha=1e-3, grow_alpha=0.05, ext_min=12, elong=2.0, ext_elong=6, margin=3):
    """Big / extended objects from connected pixels above the alpha tail threshold: 'big' if the bounding extent
    >= ext_min px, or the sqrt eigenvalue ratio of the charge-weighted moments >= elong with extent >= ext_elong.
    Big object footprint = the component grown (reconstruction) through pixels above the grow_alpha threshold,
    plus one ring; E = sum(Q - mean). Everything within `margin` px of a big object is removed from the valid
    mask and MFD+G runs on the rest."""
    thr = link_threshold(bg, alpha)
    M = (Q > thr) & valid
    lab, n = ndimage.label(M, structure=EIGHT)
    R = Q - bg.mean
    big = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == k
        h, w = m.shape
        ext = max(h, w)
        if ext < ext_elong:
            continue
        if ext >= ext_min:
            big[k] = True; continue
        ys, xs = np.nonzero(m)
        q = np.clip(R[sl][m], 0, None); s = q.sum()
        if s <= 0: continue
        my, mx = (q * ys).sum() / s, (q * xs).sum() / s
        C = np.cov(np.vstack([ys - my, xs - mx]), aweights=q + 1e-9)
        ev = np.sort(np.linalg.eigvalsh(C))
        if ev[0] <= 0 or np.sqrt(ev[1] / max(ev[0], 1e-3)) >= elong:
            big[k] = True
    tracks = []
    if big.any():
        seedm = big[lab]
        G = ndimage.binary_propagation(seedm, structure=EIGHT, mask=((Q > link_threshold(bg, grow_alpha)) & valid) | seedm)
        G = ndimage.binary_dilation(G, structure=EIGHT) & valid
        glab, gn = ndimage.label(G, structure=EIGHT)
        for k, sl in enumerate(ndimage.find_objects(glab), start=1):
            m = glab[sl] == k
            ys, xs = np.nonzero(m)
            q = np.clip(R[sl][m], 0, None); s = q.sum()
            E = float(R[sl][m].sum() * E_PER_E)
            y = float((q * (ys + sl[0].start)).sum() / s) if s > 0 else float(ys.mean() + sl[0].start)
            x = float((q * (xs + sl[1].start)).sum() / s) if s > 0 else float(xs.mean() + sl[1].start)
            tracks.append(dict(E=E, y=y, x=x, sxy=np.nan, npix=int(m.sum()), track=True, n_merged=1))
        valid2 = valid & ~ndimage.binary_dilation(G, structure=EIGHT, iterations=margin)
    else:
        valid2 = valid
    cl, _ = AM.mf_detect(np.where(valid2, Q, bg.mean), valid2, bg)
    for c in cl:
        c["track"] = False; c["n_merged"] = 1
    return tracks + cl


def mfdg_hybrid(Q, valid, bg, cl=None, ext_big=10, f=0.5, alpha_B=0.10, f_big=None):
    """(2c) MFD+G, then: all MFD+G clusters whose peaks lie in one B-mask region (pixel >= k e-, alpha_B) that is
    large (bounding extent >= ext_big px, i.e. track-like) are merged into one event; the other clusters are
    merged only by the saddle test with fraction f (keeps close X-ray pairs apart)."""
    if cl is None:
        cl, _ = AM.mf_detect(Q, valid, bg)
    own = owner_map(Q.shape, cl)
    M = (Q > link_threshold(bg, alpha_B)) & valid
    lab, n = ndimage.label(M, structure=EIGHT)
    objs = ndimage.find_objects(lab)
    bigc = np.zeros(n + 1, bool)
    for k, sl in enumerate(objs, start=1):
        bigc[k] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= ext_big
    in_big = {}
    rest = []
    for i, c in enumerate(cl):
        L = lab[c["yp"], c["xp"]]
        if L > 0 and bigc[L]:
            in_big.setdefault(L, []).append(i)
        else:
            rest.append(i)
    out = []
    if f_big is not None:
        # v2: inside large regions use the saddle test with the softer fraction f_big instead of merging all
        big_cl = [cl[i] for m in in_big.values() for i in m]
        out += mfdg_merge_saddle(Q, valid, bg, cl=big_cl, f=f_big) if big_cl else []
        in_big = {}
    for L, members in in_big.items():
        for c in (cl[i] for i in members):
            c["_comp"] = int(L)
        out += _merge_groups(Q, bg, cl, own, [members], extra=(lab, objs))
    sub = [cl[i] for i in rest]
    out += mfdg_merge_saddle(Q, valid, bg, cl=sub, f=f) if sub else []
    return out
