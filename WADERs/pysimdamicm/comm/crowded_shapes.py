# Copied from comm_scripts/clustering_tests/crowded_shapes.py by utils/sync_to_waders.py (imports made package-relative).
# Edit the comm_scripts file and re-run the sync, not this copy.
"""A_GMX crowded mode, shape stage (study: wip/agmx_crowded, scratch shapes.py / topo.py): muons as LINE components of the
mixture and a TOPOLOGY split of the clusters. Used by adaptive.adaptive_cluster when crowded mode runs and
crowded_shapes='on' (the default; $AGMX_CROWDED_SHAPES=off or crowded_shapes='off' falls back to the hint grouping).
It never runs on images below the crowded-mode trigger.

Muon lines. Every straight track found by crowded.find_hints becomes one mixture component with a physical shape
    L(p) = a_j(t) * ends(t) * P(d; sigma(t))
  t, d       position of pixel p along / across the axis (the axis is refined: lateral offset at both ends)
  a_j        charge per px of length in segments of 8 px (free: Landau fluctuations, knots)
  sigma(t)   width, linear along the track (diffusion grows with depth), fitted; capped at 1.2 x the tracker's rms
             within [2, 3] px (crossing charge inflates the fitted width)
  P          pixel-integrated Gaussian across the axis; ends(t): soft ends
The clouds A_GMX put on the axis (not elongated clouds of crossing tracks) are handed to the line; clouds and lines are
then refitted together (EM of the clouds with the lines as a fixed part of the model; each line refitted from its
responsibility share), compact charge they do not explain is re-seeded (residual pass), clouds centred on a line with
the line's own width are its knots (a clearly sharper or wider cloud on the path - e.g. a front-side X-ray - stays
separate), and A_GMX's final stage (gmx.general_cluster with mix0 / fixed / post_lab) assigns everything else. Every
pixel's charge is split between lines and clouds by model responsibility: a touching track keeps its clouds and charge.

Topology. One physical event has a simple skeleton: an X-ray is round, a muon or electron track is a path. On the minimum
spanning tree (MST) of each cluster's cloud centres:
  junction   >= 3 branches of >= 10 px: the two that continue each other most straightly (within 40 deg of opposite)
             stay with the junction, every other branch becomes its own cluster (short spurs stay)
  neck       two compact parts (each <= 15 px across, >= 20 % of the charge) whose model density (smoothed 0.8 px)
             dips below 0.35 between their centres are cut apart
  satellite  a compact round X-ray-like cloud (1-22 keV, sigma <= 1 px, axis ratio < 1.5) attached to a larger group
             through a density dip (< 0.5 to every MST neighbour) is split off

Validation (L2 ACM101, injections into the real images; wip/agmx_crowded/report): muons median foreign charge 0.28 ->
0.085, 'correct' 12.5 -> 43.5 %, >= 90 % in one cluster 62 -> 56 %; electron tracks whole 66 -> 71 %, foreign 0.20 ->
0.056; touching X-ray pairs resolved 72 -> 76 %; Fe-55 clean K-alpha events -3 % (positions and widths unchanged)."""
import numpy as np
from scipy.special import erf
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from . import agent_methods as AM

SIG_MAX = 2.0      # px: width cap of a line (diffusion of a full-thickness back-side track) ...
SIG_CAP = 3.0      # px: ... raised up to this for a diffuse track (1.2 x the tracker's transverse rms)


def _pix_gauss(d, s):
    k = 1.0 / (np.sqrt(2) * s)
    return 0.5 * (erf((d + 0.5) * k) - erf((d - 0.5) * k))


# ------------------------------------------------------------------------------------------------ muon lines
class Line:
    def __init__(self, c, u, t0, t1, w, shape, valid, seg=8, half=6):
        self.c, self.u = np.asarray(c, float), np.asarray(u, float)
        self.t0, self.t1, self.seg = float(t0), float(t1), seg
        self.sig0 = max(0.6 * w, 0.4); self.sig1 = 0.0
        self.sig_max = float(np.clip(1.2 * w, SIG_MAX, SIG_CAP))
        H, W = shape; n = np.array([-self.u[1], self.u[0]])
        pts = self.c + np.outer([self.t0 - 4, self.t1 + 4], self.u)
        y0, y1 = int(max(pts[:, 0].min() - half - 1, 0)), int(min(pts[:, 0].max() + half + 2, H))
        x0, x1 = int(max(pts[:, 1].min() - half - 1, 0)), int(min(pts[:, 1].max() + half + 2, W))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        t = (gy - self.c[0]) * self.u[0] + (gx - self.c[1]) * self.u[1]; d = (gy - self.c[0]) * n[0] + (gx - self.c[1]) * n[1]
        m = (t >= self.t0 - 4) & (t <= self.t1 + 4) & (np.abs(d) <= half) & valid[y0:y1, x0:x1]
        self.py, self.px = gy[m], gx[m]
        self.nseg = max(int(np.ceil((self.t1 - self.t0) / seg)), 1)
        self.a = np.zeros(self.nseg)
        self.off = np.zeros(2)
        self._geom()

    def _geom(self):
        n = np.array([-self.u[1], self.u[0]])
        t = (self.py - self.c[0]) * self.u[0] + (self.px - self.c[1]) * self.u[1]
        d = (self.py - self.c[0]) * n[0] + (self.px - self.c[1]) * n[1]
        f = np.clip((t - self.t0) / max(self.t1 - self.t0, 1.0), 0, 1)
        self.t, self.f = t, f
        self.d = d - (self.off[0] + (self.off[1] - self.off[0]) * f)
        self.j = np.clip(((t - self.t0) / self.seg).astype(int), 0, self.nseg - 1)
        self.ends = 0.5 * (1 + erf((t - self.t0 + 0.5) / 0.8)) * 0.5 * (1 - erf((t - self.t1 - 0.5) / 0.8))

    def sigma(self):
        return np.clip(self.sig0 + self.sig1 * self.f, 0.25, self.sig_max)

    def profile(self):
        return self.ends * _pix_gauss(self.d, self.sigma())

    def model(self):
        return self.a[self.j] * self.profile()

    def update(self, share):
        """M-step from the line's share of each footprint pixel's charge: charge per segment, width and offset"""
        P = self.profile()
        num = np.bincount(self.j, share, self.nseg); den = np.bincount(self.j, P, self.nseg)
        self.a = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        tc, mu, var, wt = [], [], [], []
        for w0 in range(0, self.nseg, 3):
            m = (self.j >= w0) & (self.j < w0 + 3) & (np.abs(self.d) <= 4)
            s = share[m]
            if s.sum() <= 1e-6: continue
            mm = np.average(self.d[m], weights=s)
            tc.append(np.average(self.f[m], weights=s)); mu.append(mm); var.append(np.average((self.d[m] - mm) ** 2, weights=s)); wt.append(s.sum())
        if len(tc) >= 2:
            tc, mu, var, wt = map(np.array, (tc, mu, var, wt))
            sg = np.sqrt(np.clip(var - 1 / 12, 0.04, 9))
            A = np.vstack([np.ones_like(tc), tc]).T; Wm = np.sqrt(wt / wt.sum())[:, None]
            p = np.linalg.lstsq(Wm * A, Wm[:, 0] * sg, rcond=None)[0]
            q = np.linalg.lstsq(Wm * A, Wm[:, 0] * mu, rcond=None)[0]
            self.sig0, self.sig1 = float(p[0]), float(p[1])
            self.off = self.off + np.array([q[0], q[0] + q[1]])
            self._geom()


def _line_image(lines, shape):
    M = np.zeros(shape)
    for L in lines: np.add.at(M, (L.py, L.px), L.model())
    return M


def _refit_lines(lines, Mc, ML, Rp, eps):
    for L in lines:
        tot = Mc[L.py, L.px] + ML[L.py, L.px] + 0.1 * eps[L.py, L.px]
        L.update(Rp[L.py, L.px] * L.model() / np.maximum(tot, 1e-9))


def shape_cluster(Q, valid, bg, hints, return_pixels=False, return_info=False, n_outer=6, em_iter=3, a_min=30.0,
                  nsig_res=5.0, sig_min=4.0, f_pix=0.02, l_min=1.0, topology=True, **kw):
    """A_GMX with the hinted straight tracks as line components and the topology split. Returns the clusters (lines:
    cls 'muon', regime 'line', straight_track True) and, with return_info, info incl. 'lines'."""
    from . import gmx as GX
    from . import crowded as CR
    H, W = Q.shape
    cl0, info = GX.general_cluster(Q, valid, bg, return_info=True, split_crossings=True, **kw)
    bgh, _ = GX.hot_columns_as_dark(Q, valid, bg)
    R = np.where(valid, Q - bgh.mean, 0.0); Rp = np.clip(R, 0, None); V = np.where(valid, bgh.var, 1e6)
    eps = np.sqrt(np.where(valid, bgh.var, 1.0))
    mix = info["mix"]
    if len(mix) and hints:
        _, on_axis = CR.assign_hints(np.zeros(len(mix), int), mix, hints)
        mix = mix.keep(~on_axis)
    lines = [Line(c, u, t0, t1, w, Q.shape, valid) for c, u, t0, t1, w in hints]
    for L in lines:
        L.update(np.where(np.abs(L.d) <= 1.5, Rp[L.py, L.px], 0.0))
    for _ in range(n_outer):
        ML = _line_image(lines, (H, W))
        mix = GX.em(mix, Rp, eps + ML, valid, em_iter)
        mix = mix.keep(mix.A > a_min)
        I, K, G, _, _ = GX.evaluate(mix, (H, W), valid)
        _refit_lines(lines, np.bincount(I, G, H * W).reshape(H, W), ML, Rp, eps)
    # residual pass: compact charge the lines and clouds do not explain
    ML = _line_image(lines, (H, W))
    I, K, G, _, _ = GX.evaluate(mix, (H, W), valid)
    Mc = np.bincount(I, G, H * W).reshape(H, W)
    Rr = np.where(valid, R - Mc - ML, 0.0)
    Tr = GX.mf_maps(Rr, np.where(valid, bgh.var + Mc + ML, 0.0), (0.5, 1.0))
    Trm = Tr.max(0)
    qy, qx = np.nonzero((Trm == GX._local_max(Trm, valid, 3)) & (Trm > nsig_res) & valid)
    if qy.size:
        dmin = cKDTree(mix.mu).query(np.c_[qy, qx])[0] if len(mix) else np.full(qy.size, 9.0)
        qy, qx = qy[dmin > 1.0], qx[dmin > 1.0]
        s1 = np.asarray((0.5, 1.0))[Tr.argmax(0)[qy, qx]]
        mix = mix.concat(GX.Mix(np.maximum(Rr[qy, qx] * 2 * np.pi * (s1 ** 2 + GX.PIX), 1.0), np.c_[qy, qx].astype(float),
                                np.c_[s1 ** 2, s1 ** 2, np.zeros_like(s1)]))
        for _ in range(2):
            mix = GX.em(mix, Rp, eps + ML, valid, em_iter + 2)
            I, K, G, _, _ = GX.evaluate(mix, (H, W), valid)
            _refit_lines(lines, np.bincount(I, G, H * W).reshape(H, W), ML, Rp, eps)
            ML = _line_image(lines, (H, W))
        if len(mix):
            z = GX.significance(mix, Rp, V + ML, valid)
            mix = mix.keep((z > sig_min) & (mix.A > a_min))
    # knots: clouds centred on a line (within 1 px) with the line's own width at that depth
    if len(mix) and lines:
        knot = np.zeros(len(mix), bool)
        sgp = np.sqrt(np.maximum(0.5 * (mix.S[:, 0] + mix.S[:, 1]) - GX.PIX, 0.01))
        for L in lines:
            n = np.array([-L.u[1], L.u[0]]); dd = mix.mu - L.c
            t = dd @ L.u; f = np.clip((t - L.t0) / max(L.t1 - L.t0, 1), 0, 1)
            d = dd @ n - (L.off[0] + (L.off[1] - L.off[0]) * f)
            sl = np.clip(L.sig0 + L.sig1 * f, 0.25, L.sig_max)
            knot |= (np.abs(d) <= 1.0) & (t >= L.t0 - 2) & (t <= L.t1 + 2) & (np.abs(sgp - sl) <= 0.3)
        mix = mix.keep(~knot)
        for _ in range(2):
            I, K, G, _, _ = GX.evaluate(mix, (H, W), valid)
            _refit_lines(lines, np.bincount(I, G, H * W).reshape(H, W), ML, Rp, eps)
            ML = _line_image(lines, (H, W))
            mix = GX.em(mix, Rp, eps + ML, valid, 3)
    # A_GMX's final stage with the lines in every responsibility, and the topology split of the cloud labels
    ML = _line_image(lines, (H, W))
    out, info2 = GX.general_cluster(Q, valid, bg, return_pixels=True, return_info=True, split_crossings=True, mix0=mix, fixed=ML,
                                    post_lab=post_lab if topology else None, **kw)
    I, K, G, _, _ = GX.evaluate(info2["mix"], (H, W), valid)
    T = np.bincount(I, G, H * W) + ML.ravel()
    for L in lines:
        Lm = L.model(); frac = Lm / np.maximum(T[L.py * W + L.px], 1e-12)
        m = (frac >= f_pix) & (Lm >= l_min)
        ys, xs, qq = L.py[m], L.px[m], (frac * R[L.py, L.px])[m]
        if ys.size == 0: continue
        w = np.clip(qq, 0, None) + 1e-9; Mo = T[L.py * W + L.px] - Lm
        out.append(dict(E=float(qq.sum() * AM.E_PER_E), y=float(np.average(ys, weights=w)), x=float(np.average(xs, weights=w)),
                        sxy=float(np.median(L.sigma())), cls="muon", regime="line", straight_track=True, length=float(L.t1 - L.t0),
                        blend=float(np.sum(Lm * Mo) / max(np.sum(Lm * Lm), 1e-12)), sigma0=L.sig0, sigma1=L.sig1, _pix=(ys, xs, qq)))
    GX.fill_flags(out, R, valid)
    if not return_pixels:
        for c in out: c.pop("_pix", None)
    if return_info:
        info2.update(lines=lines, n_lines=len(lines), n_topology_splits=STATS["split"])
        return out, info2
    return out


# ------------------------------------------------------------------------------------------------ topology
def _mst(mu):
    D = squareform(pdist(mu))
    T = minimum_spanning_tree(D).toarray()
    return (T > 0) | (T.T > 0), D


def _branch(adj, j, start):
    seen = {j, start}; st = [start]; out = [start]
    while st:
        a = st.pop()
        for b in np.nonzero(adj[a])[0]:
            if b not in seen: seen.add(b); st.append(b); out.append(b)
    return out


def _density(mu, A, S, pts, smooth):
    """mixture density of the clouds at pts, smoothed with a Gaussian of `smooth` px"""
    out = np.zeros(len(pts))
    for k in range(len(mu)):
        a, c, b = S[k, 0] + 1 / 12 + smooth ** 2, S[k, 1] + 1 / 12 + smooth ** 2, S[k, 2]
        det = max(a * c - b * b, 1e-6); d = pts - mu[k]
        out += A[k] / (2 * np.pi * np.sqrt(det)) * np.exp(-0.5 * (c * d[:, 0] ** 2 + a * d[:, 1] ** 2 - 2 * b * d[:, 0] * d[:, 1]) / det)
    return out


def _neck_ratio(mu, A, S, p, q, smooth):
    """minimum density between the centroids of groups p and q relative to the smaller density at the centroids"""
    cp = np.average(mu[p], 0, weights=A[p]); cq = np.average(mu[q], 0, weights=A[q])
    dens = _density(mu, A, S, cp + np.linspace(0, 1, 41)[:, None] * (cq - cp), smooth)
    return float(dens[8:33].min() / max(min(dens[0], dens[-1]), 1e-9))


def split_cluster(mu, A, S, l_branch=10.0, ang_max=40.0, f_lobe=0.2, neck_max=0.35, lobe_max=15.0, smooth=0.8, neck_max_clouds=150):
    """sub-labels (0 = the main object) for the clouds of one cluster: junction and neck rules"""
    n = len(mu); sub = np.zeros(n, int)
    if n < 2:
        return sub
    adj, D = _mst(mu)
    deg = adj.sum(1); nxt = 1
    for j in np.argsort(-A):
        if deg[j] < 3: continue
        branches = []
        for b in np.nonzero(adj[j])[0]:
            nodes = _branch(adj, j, b); length = D[j, nodes].max()
            if length >= l_branch:
                v = np.average(mu[nodes], 0, weights=A[nodes]) - mu[j]
                branches.append((nodes, v / max(np.linalg.norm(v), 1e-9), length))
        if len(branches) < 3: continue
        best, bp = -2, None
        for a in range(len(branches)):
            for b in range(a + 1, len(branches)):
                c = -float(branches[a][1] @ branches[b][1])
                if c > best: best, bp = c, (a, b)
        if best < np.cos(np.radians(ang_max)):
            bp = (int(np.argmax([br[2] for br in branches])),)
        for k, (nodes, v, length) in enumerate(branches):
            if k in bp: continue
            sub[nodes] = nxt; nxt += 1
            for a in nodes: adj[a, j] = adj[j, a] = False
        deg = adj.sum(1)
    out = sub.copy(); nxt = out.max() + 1
    for g in np.unique(sub):
        idx = np.nonzero(sub == g)[0]
        # the neck test tries every MST edge (cost ~ n^3): only for groups of <= neck_max_clouds clouds; a neck between two
        # compact deposits never involves more (dense Am-241 EXT4 has clusters of thousands of clouds)
        if idx.size < 2 or idx.size > neck_max_clouds: continue
        sa = adj[np.ix_(idx, idx)]; best = None
        for a, b in zip(*np.nonzero(np.triu(sa))):
            cut = sa.copy(); cut[a, b] = cut[b, a] = False
            ncomp, lab = connected_components(cut, directed=False)
            if ncomp != 2: continue
            p, q = idx[lab == 0], idx[lab == 1]
            if min(A[p].sum(), A[q].sum()) < f_lobe * A[idx].sum(): continue
            if max(D[np.ix_(p, p)].max(), D[np.ix_(q, q)].max()) > lobe_max: continue
            r = _neck_ratio(mu, A, S, p, q, smooth)
            if r < neck_max and (best is None or r < best[0]): best = (r, q)
        if best is not None:
            out[best[1]] = nxt; nxt += 1
    return out


def xray_satellites(mu, A, S, sub, a_lo=270.0, a_hi=6000.0, s_max=1.0, ratio_max=1.5, dip_max=0.5, smooth=0.8, r_local=12.0):
    """split off compact round X-ray-like clouds attached to a larger group through a density dip"""
    sub = sub.copy(); nxt = sub.max() + 1
    for g in np.unique(sub):
        idx = np.nonzero(sub == g)[0]
        if idx.size < 2: continue
        adj, _ = _mst(mu[idx])
        tree = cKDTree(mu[idx])
        for i, k in enumerate(idx):
            a, c, b = S[k]; tr = 0.5 * (a + c); dd = np.sqrt(max(0.25 * (a - c) ** 2 + b * b, 0))
            if not (a_lo <= A[k] <= a_hi) or np.sqrt(tr) > s_max or np.sqrt((tr + dd) / max(tr - dd, 1e-6)) >= ratio_max: continue
            if A[k] > 0.5 * A[idx].sum(): continue
            nb = np.nonzero(adj[i])[0]
            if not nb.size: continue
            # the density between the cloud and a neighbour only depends on the clouds near them (within r_local px)
            loc = idx[tree.query_ball_point(mu[k], r_local)]
            lk = int(np.nonzero(loc == k)[0][0])
            ok = True
            for j in nb:
                lj = np.nonzero(loc == idx[j])[0]
                if lj.size == 0:
                    loc2 = np.r_[loc, idx[j]]; lj = np.array([loc2.size - 1])
                else:
                    loc2 = loc
                if _neck_ratio(mu[loc2], A[loc2], S[loc2], np.array([lk]), lj, smooth) >= dip_max:
                    ok = False; break
            if ok:
                sub[k] = nxt; nxt += 1
    return sub


def post_lab(mix, lab, min_clouds=2, max_clouds=1000):
    """topology split of every cluster of min_clouds..max_clouds clouds; new contiguous labels. Larger clusters are left
    alone: the MST works on the full n x n distance matrix (memory ~ 8 n^2 bytes: a 50k-cloud blob on Am-241 EXT4 needs 20 GB),
    and an event-skeleton rule means nothing on a blob of > 1000 clouds"""
    lab = np.asarray(lab).copy(); nxt = int(lab.max()) + 1 if lab.size else 0
    STATS["split"] = 0
    for L in np.unique(lab):
        k = np.nonzero(lab == L)[0]
        if k.size < min_clouds or k.size > max_clouds: continue
        sub = xray_satellites(mix.mu[k], mix.A[k], mix.S[k], split_cluster(mix.mu[k], mix.A[k], mix.S[k]))
        for s in np.unique(sub):
            if s == 0: continue
            lab[k[sub == s]] = nxt; nxt += 1; STATS["split"] += 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab


STATS = {"split": 0}
