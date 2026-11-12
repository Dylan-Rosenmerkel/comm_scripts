"""Bayesian few-electron event finder for skipper CCDs at ultra-low dark current (wip/lowdc/bayes).

Model (all pixel values in electrons, residuals v of noise.pedestals):
  null H0      n_ij = d_ij,          d_ij ~ Poisson(lam_ij)  (dark charge incl. hot pixels / columns)
  event        Neyman-Scott cluster process: centres at rate R per pixel, each with N >= 2 electrons (prior p(N)) that
               land at centre + N(0, sigma^2 I) (pixel-integrated: p_ij(mu, sigma) = product of normal-cdf differences);
               sigma from a depth prior (physics) or the benchmark's flat prior.
  data         P(v_ij | n_ij) = f(v_ij - n_ij) (noise.NoiseModel; never rounded)

Sparsity: with l_ij(n) = f(v_ij - n) / f(v_ij), the null factorises into per-pixel terms, and a pixel whose
max_{n>=1} l(n) < r_min (~1e-2) can be treated as n = 0 in every hypothesis (relative error < r_min). Only 'soft
candidate' pixels are touched; they are grouped by distance (<= R_link px, not connectivity) into neighbourhoods.

Evidence of one event in a neighbourhood C (candidate pixels i, background-convolved likelihoods
g_i(m) = sum_d Pois(d; lam_i) l_i(m + d)): the electron counts m_i ~ Multinomial(N, p_i) restricted to C give

    Z1(mu, sigma, N) / Z0 = N! [t^N] prod_i ( sum_m g_i(m) (p_i t)^m / m! ) / prod_i g_i(0)

(a polynomial product: exact marginalisation of the latent integer counts and of the dark/event attribution of the
electrons). The centre is integrated on a sub-pixel grid (step min(0.25, sigma / 2) px), sigma on a prior grid:

    J(N) = sum_sigma w_sigma  int d^2mu  Z1(mu, sigma, N) / Z0        [per unit event rate R]

so that the posterior odds of 'one event with N electrons' vs background are R_N J(N) (R_N = R p(N)), and
P(no event | data) = 1 / (1 + sum_N R_N J(N) + R^2 J2). J2 (two events) is approximated by spatially separated
bipartitions of C (cuts of the single-linkage tree; events in the same pixels are not separable anyway).
1-e- events are identical to a dark electron (same pixel likelihood, same spatial distribution): they are part of
lam, not of the event model (N >= 2).
"""
import math
import numpy as np
from scipy import special, stats
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from numba import njit

import noise as NZ

PIX_UM = 15.0
THICK_UM = 675.0
# DAMIC at SNOLAB diffusion law sigma_xy^2 = -A ln|1 - b z| (z = depth from the pixel array): A = 285 um^2,
# b = 8.2e-4 / um from surface cosmic-muon tracks, A. Aguilar-Arevalo et al. (DAMIC), PRL 125, 241803 (2020),
# arXiv:2007.15622 -> sigma_max = 15.2 um = 1.01 px at z = 675 um (15 um pixels).
DIFF_A = 285.0
DIFF_B = 8.2e-4
SIG_NODES = np.round(np.arange(0.1, 1.31, 0.1), 3)
KCAP = 20            # largest N of the event model; groups above ~KBIG e- are 'big' deposits (left to A_GMX)
KBIG = 20.0


def sigma_of_z(z_um, A=DIFF_A, b=DIFF_B):
    """lateral diffusion [px] of charge created at depth z [um]"""
    return np.sqrt(-A * np.log(np.abs(1.0 - b * np.asarray(z_um)))) / PIX_UM


def z_of_sigma(sig_px, A=DIFF_A, b=DIFF_B):
    """inverse of sigma_of_z (NaN above sigma_max)"""
    zz = (1.0 - np.exp(-(np.asarray(sig_px) * PIX_UM) ** 2 / A)) / b
    return np.where(zz <= THICK_UM, zz, np.nan)


def sigma_prior(kind="physics", nodes=SIG_NODES):
    """weights of the sigma grid. 'physics': depth uniform in 0..675 um mapped through the diffusion law (each z goes to
    the nearest node); 'bench': sigma uniform 0.2..1.2 px (the benchmark's own distribution)."""
    if kind == "physics":
        z = np.linspace(0, THICK_UM, 20001)
        s = sigma_of_z(z)
        k = np.argmin(np.abs(s[:, None] - nodes[None, :]), axis=1)
        w = np.bincount(k, minlength=nodes.size).astype(float)
    elif kind == "bench":
        s = np.linspace(0.2, 1.2, 20001)
        k = np.argmin(np.abs(s[:, None] - nodes[None, :]), axis=1)
        w = np.bincount(k, minlength=nodes.size).astype(float)
    else:
        raise ValueError(kind)
    return w / w.sum()


def n_prior(kind="flat20"):
    """p(N), N = 0..KCAP (0 for N < 2): 'flat20' uniform 2..20 (default, spectrum-agnostic)"""
    p = np.zeros(KCAP + 1)
    if kind == "flat20":
        p[2:] = 1.0
    elif kind == "flat10":
        p[2:11] = 1.0
    return p / p.sum()


def _lse(a, axis=None):
    """log-sum-exp without scipy's per-call overhead (called ~1e4 times per image); -inf safe"""
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        r = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return r.squeeze(axis) if axis is not None else float(r.reshape(-1)[0])


# ---------------------------------------------------------------------------------------------- per-pixel tables
def bg_convolve(L, lam, K):
    """log g(m) = log sum_d Pois(d; lam) l(m + d), m = 0..K, from the log likelihood ratios L (n = 0..K + D)"""
    D = L.shape[1] - 1 - K
    d = np.arange(D + 1)
    lp = stats.poisson.logpmf(d[None, :], np.asarray(lam)[:, None])        # (k, D+1)
    out = np.empty((L.shape[0], K + 1))
    for m in range(K + 1):
        out[:, m] = special.logsumexp(L[:, m:m + D + 1] + lp, axis=1)
    return out


# ---------------------------------------------------------------------------------------------- numba kernel
@njit(cache=True)
def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(cache=True)
def evidence_kernel(rr, cc, A, Kn, nodes, wsig):
    """sum over centre grid of h^2 N! [t^N] prod_i sum_m A[i, m] (p_i t)^m / m!  for each (N, sigma node).
    rr, cc: candidate pixel rows / cols (relative); A: (k, K+1) linear scaled g_i(m); Kn: max N.
    Returns S (K+1, ns), Sy, Sx (centre-weighted sums)."""
    k = rr.shape[0]
    ns = nodes.shape[0]
    S = np.zeros((Kn + 1, ns)); Sy = np.zeros((Kn + 1, ns)); Sx = np.zeros((Kn + 1, ns))
    rmin = rr.min(); rmax = rr.max() + 1; cmin = cc.min(); cmax = cc.max() + 1
    fact = np.ones(Kn + 1)
    for n in range(1, Kn + 1):
        fact[n] = fact[n - 1] * n
    coef = np.zeros(Kn + 1); tmp = np.zeros(Kn + 1); q = np.zeros(Kn + 1)
    for si in range(ns):
        if wsig[si] <= 0:
            continue
        sg = nodes[si]
        h = min(0.25, sg / 2.0)
        pad = max(2.5 * sg, 0.5)
        ny = int(math.ceil((rmax - rmin + 2 * pad) / h)); nx = int(math.ceil((cmax - cmin + 2 * pad) / h))
        y0 = rmin - pad + 0.5 * h; x0 = cmin - pad + 0.5 * h
        ay = np.empty((ny, k)); bx = np.empty((nx, k))
        for iy in range(ny):
            my = y0 + iy * h
            for i in range(k):
                ay[iy, i] = _phi((rr[i] + 1 - my) / sg) - _phi((rr[i] - my) / sg)
        for ix in range(nx):
            mx = x0 + ix * h
            for i in range(k):
                bx[ix, i] = _phi((cc[i] + 1 - mx) / sg) - _phi((cc[i] - mx) / sg)
        for iy in range(ny):
            my = y0 + iy * h
            for ix in range(nx):
                mx = x0 + ix * h
                for n in range(Kn + 1):
                    coef[n] = 0.0
                coef[0] = 1.0
                deg = 0
                for i in range(k):
                    p = ay[iy, i] * bx[ix, i]
                    # q_m = A[i, m] p^m / m!
                    pm = 1.0
                    mtop = 0
                    for m in range(Kn + 1):
                        if m > 0:
                            pm *= p / m
                        q[m] = A[i, m] * pm
                        if q[m] > 0:
                            mtop = m
                    if mtop == 0:
                        for n in range(deg + 1):
                            coef[n] *= q[0]
                        continue
                    nd = min(Kn, deg + mtop)
                    for n in range(nd + 1):
                        tmp[n] = 0.0
                    for a in range(deg + 1):
                        ca = coef[a]
                        if ca == 0.0:
                            continue
                        top = min(mtop, Kn - a)
                        for m in range(top + 1):
                            tmp[a + m] += ca * q[m]
                    for n in range(nd + 1):
                        coef[n] = tmp[n]
                    deg = nd
                w2 = h * h
                for n in range(deg + 1):
                    val = w2 * fact[n] * coef[n]
                    S[n, si] += val
                    Sy[n, si] += val * my
                    Sx[n, si] += val * mx
    return S, Sy, Sx


# ---------------------------------------------------------------------------------------------- neighbourhoods
def group(rows, cols, r_link):
    """connected components of candidates within r_link px of each other (distance, not pixel connectivity)"""
    n = rows.size
    if n == 0:
        return np.zeros(0, int)
    pts = np.c_[rows, cols].astype(float)
    pairs = cKDTree(pts).query_pairs(r_link, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def nbhd_logJ(rr, cc, LG, Kn, nodes, wsig):
    """log J(N, sigma) of one event in the candidate set (rr, cc) with background-convolved log g (k, >= Kn+1).
    Returns logJ (Kn+1, ns), ybar, xbar arrays (Kn+1, ns)."""
    lg = LG[:, :Kn + 1]
    M = lg.max(1)
    A = np.exp(lg - M[:, None])
    off = float((M - LG[:, 0]).sum())
    S, Sy, Sx = evidence_kernel(rr.astype(np.int64), cc.astype(np.int64), A, Kn, nodes, wsig)
    with np.errstate(divide="ignore", invalid="ignore"):
        logJ = np.log(S) + off
        yb = Sy / S; xb = Sx / S
    return logJ, yb, xb


def _splits(rr, cc, n_max=3, d_min=1.5):
    """up to n_max bipartitions of the candidate set: cut one of the longest single-linkage (MST) edges >= d_min px"""
    k = rr.size
    if k < 2:
        return []
    D = np.hypot(rr[:, None] - rr[None, :], cc[:, None] - cc[None, :])
    T = minimum_spanning_tree(D + 1e-9 * (D == 0) * ~np.eye(k, dtype=bool)).tocoo()
    order = np.argsort(-T.data)
    out = []
    for e in order[:n_max]:
        if T.data[e] < d_min:
            break
        keep = np.ones(T.data.size, bool); keep[e] = False
        g = coo_matrix((T.data[keep], (T.row[keep], T.col[keep])), shape=(k, k))
        lab = connected_components(g, directed=False)[1]
        out.append(lab == lab[0])
    return out


FAST_SINGLE = True
_I1 = {}


def _log_i1(wsig):
    """log int d^2mu p(mu, sigma)^N for one pixel, N = 0..KCAP, every sigma node (cached per prior)"""
    key = tuple(np.round(wsig, 12))
    if key not in _I1:
        S, _, _ = evidence_kernel(np.zeros(1, np.int64), np.zeros(1, np.int64), np.ones((1, KCAP + 1)), KCAP,
                                  SIG_NODES, wsig)
        with np.errstate(divide="ignore"):
            _I1[key] = np.log(S)
    return _I1[key]


class Prior:
    """event-model prior: sigma grid weights, p(N), linking radius, candidate floor"""

    def __init__(self, sig="physics", nk="flat20", r_link=4.0, r_min=1e-2):
        self.sig_kind = sig
        self.wsig = sigma_prior(sig)
        self.pN = n_prior(nk)
        self.r_link = r_link
        self.r_min = r_min


def analyse_image(v, kap, nm, valid, lam_map, prior, keep_logbf=-6.0, big_halo=15.0, cand=None, only_pix=False):
    """all neighbourhoods of one image.
    v: residual image [e-] (noise.pedestals), kap: column noise scales, nm: NoiseModel, lam_map: dark charge per pixel
    (scalar or H x W). Returns (list of neighbourhood dicts, candidate-pixel table for the background fit).
    Each neighbourhood: y, x, pix (charged candidate pixels), cand (all), logJN (log J(N), N = 0..KCAP, p(N) not
    applied), logJ2 (two events, p(N) applied), logJsig (sigma posterior numerator, p(N) applied), q_sum, big, halo."""
    H, W = v.shape
    lam_map = np.broadcast_to(np.asarray(lam_map, float), (H, W))
    pre = valid & (v > min(0.25, 1.5 * nm.core))
    if cand is None:
        cm = np.zeros((H, W), bool)
        if pre.any():
            sub = NZ.candidates(np.where(pre, v, -10.0), kap, nm, pre, prior.r_min, KCAP)
            cm = sub
    else:
        cm = cand
    rows, cols = np.nonzero(cm)
    vals = v[rows, cols]
    D = 4  # dark electrons per pixel considered in g (more for hot pixels, see below)
    lamc = lam_map[rows, cols]
    D = int(max(D, np.ceil(lamc.max() + 6 * np.sqrt(lamc.max()))) if lamc.size else D)
    Kt = KCAP + D
    Lfull = NZ.loglike_table(vals, kap[cols], nm, Kt) if vals.size else np.zeros((0, Kt + 1))
    L = Lfull - Lfull[:, :1]                         # log l(n) = log f(v - n) - log f(v)
    LG = bg_convolve(L, lamc, KCAP) if vals.size else np.zeros((0, KCAP + 1))
    lab = group(rows, cols, prior.r_link)
    nb = []
    # pixel table for the background: under H0 each candidate pixel's posterior expected dark count and P(n >= 1)
    lpd = stats.poisson.logpmf(np.arange(Kt + 1)[None, :], lamc[:, None]) if vals.size else np.zeros((0, Kt + 1))
    post0 = L + lpd
    post0 = np.exp(post0 - special.logsumexp(post0, axis=1, keepdims=True)) if vals.size else post0
    ed0 = (post0 * np.arange(Kt + 1)[None, :]).sum(1) if vals.size else np.zeros(0)
    # 'charged' pixel: some n >= 1 fits better than n = 0 (likelihood, not rounding)
    charged = (L[:, 1:].max(1) > 0) if vals.size else np.zeros(0, bool)
    pixtab = dict(r=rows, c=cols, ed0=ed0, charged=charged, nb=lab, L=L)
    if not vals.size:
        return nb, pixtab
    order = np.argsort(lab, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(lab[order]))[0] + 1, order.size]
    big_pix = []
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        qs = float(np.clip(vals[idx], 0, None).sum())
        groups.append((idx, qs))
        if qs > KBIG:
            big_pix.append(np.c_[rows[idx], cols[idx]])
    tree_big = cKDTree(np.concatenate(big_pix)) if big_pix else None
    for gi, (idx, qs) in enumerate(groups):
        rr, cc = rows[idx], cols[idx]
        d = dict(nb=int(lab[idx[0]]), n_cand=int(idx.size), q_sum=qs, big=qs > KBIG, halo=False,
                 cand=(rr, cc), cidx=idx)
        ch = charged[idx]
        pr, pc = (rr[ch], cc[ch]) if ch.any() else (rr[[np.argmax(vals[idx])]], cc[[np.argmax(vals[idx])]])
        d["pix"] = (pr, pc)
        d["y"] = float(np.mean(pr) + 0.5); d["x"] = float(np.mean(pc) + 0.5)
        if d["big"]:
            nb.append(d); continue
        if tree_big is not None and np.isfinite(tree_big.query(np.c_[rr, cc], distance_upper_bound=big_halo)[0]).any():
            d["halo"] = True; nb.append(d); continue
        if only_pix:
            nb.append(d); continue
        if idx.size > 40:
            # rare large non-big group (s >= 0.15 noise percolation or a dense spot): keep the 40 most charge-like pixels
            idx = idx[np.argsort(-L[idx, 1:].max(1))[:40]]
            rr, cc = rows[idx], cols[idx]
        lgi = LG[idx]
        # largest N any configuration supports: each pixel up to the last m within e^-25 of its best
        nmax = np.array([np.max(np.nonzero(x - x.max() > -25)[0]) for x in lgi])
        Kn = int(min(KCAP, max(2, nmax.sum())))
        r0, c0 = rr.min(), cc.min()
        if idx.size == 1 and FAST_SINGLE:
            # one candidate pixel: J(N, sigma) = l-ratio g(N) / g(0) x int d^2mu p(mu, sigma)^N (location independent)
            logJ = (lgi[0, :Kn + 1] - lgi[0, 0])[:, None] + _log_i1(prior.wsig)[:Kn + 1]
            yb = np.full(logJ.shape, 0.5); xb = np.full(logJ.shape, 0.5)
        else:
            logJ, yb, xb = nbhd_logJ(rr - r0, cc - c0, lgi, Kn, SIG_NODES, prior.wsig)
        logJN = np.full(KCAP + 1, -np.inf)
        with np.errstate(divide="ignore"):
            lw = np.log(prior.wsig)
        jn = _lse(logJ + lw[None, :], axis=1)
        logJN[:Kn + 1] = jn
        pN = prior.pN
        with np.errstate(divide="ignore"):
            lpN = np.log(pN)
        logJ1 = _lse(logJN + lpN)
        d["logJN"] = logJN
        d["logJ1"] = float(logJ1)
        # point deposit (transient single-pixel defect: charge injected into ONE pixel, no diffusion): per unit rate
        # per pixel, J_pt(N) = sum_i g_i(N) / g_i(0) over the neighbourhood's candidates (the others stay n = 0).
        # N >= 2 only (a 1-e- point deposit is dark charge). Used only when posterior() gets R_pt.
        lpt = np.full(KCAP + 1, -np.inf)
        lpt[2:] = _lse(lgi[:, 2:KCAP + 1] - lgi[:, :1], axis=0)
        d["logJpt"] = lpt
        if logJ1 < keep_logbf * np.log(10) - 5:
            d["weak"] = True
        # posterior-mean centre and sigma (p(N) and the sigma prior applied)
        t = logJ + lw[None, :] + lpN[:Kn + 1, None]
        if np.isfinite(t).any():
            wpost = np.exp(t - _lse(t))
            wpost = np.where(np.isfinite(wpost), wpost, 0.0)
            d["y"] = float(np.nansum(wpost * yb) + r0); d["x"] = float(np.nansum(wpost * xb) + c0)
            d["logJsig"] = _lse(t, axis=0)
        # two events: separated bipartitions (only where one event is not negligible)
        d["logJ2"] = -np.inf
        if idx.size >= 2 and logJ1 > np.log(1e-3):
            vals2 = []
            for m in _splits(rr, cc):
                parts = []
                for sel in (m, ~m):
                    a_, b_ = rr[sel], cc[sel]
                    lgs = lgi[sel]
                    nm_ = np.array([np.max(np.nonzero(x - x.max() > -25)[0]) for x in lgs])
                    kn = int(min(KCAP, max(2, nm_.sum())))
                    lj, _, _ = nbhd_logJ(a_ - a_.min(), b_ - b_.min(), lgs, kn, SIG_NODES, prior.wsig)
                    jn_ = _lse(lj + lw[None, :], axis=1)
                    parts.append(_lse(jn_ + lpN[:kn + 1]))
                vals2.append(parts[0] + parts[1])
            if vals2:
                d["logJ2"] = float(_lse(vals2))
        nb.append(d)
    return nb, pixtab


def posterior(nb, R_N, R=None, R_pt=None):
    """P(no event), P(>= 1 event), P(2 events), posterior over N (one-event term) for each neighbourhood, given the
    event-rate spectrum R_N [per pixel, N = 0..KCAP] (R = sum R_N for the two-event term).
    R_pt (optional, N = 0..KCAP): rate spectrum of point deposits (transient single-pixel defects, logJpt) as a
    further non-event alternative; then p0 = 1 - p_ev includes them, p_bg = pure background, p_pt = point deposit."""
    R = float(np.sum(R_N)) if R is None else R
    with np.errstate(divide="ignore"):
        lR = np.log(R_N)
        lRp = np.log(R_pt) if R_pt is not None else None
    for d in nb:
        if "logJN" not in d:
            continue
        t1 = d["logJN"] + lR
        t2 = 2 * np.log(R) + d["logJ2"] if R > 0 else -np.inf
        tp = _lse(d["logJpt"] + lRp) if (lRp is not None and "logJpt" in d) else -np.inf
        lz = _lse(np.r_[0.0, t1, t2, tp])
        lev = _lse(np.r_[t1, t2])
        d["p_ev"] = float(np.exp(lev - lz)) if np.isfinite(lev) else 0.0
        d["p0"] = float(np.exp(_lse(np.r_[0.0, tp]) - lz)) if np.isfinite(tp) else float(np.exp(-lz))
        d["p_bg"] = float(np.exp(-lz))
        d["p_pt"] = float(np.exp(tp - lz)) if np.isfinite(tp) else 0.0
        d["p2"] = float(np.exp(t2 - lz))
        pn = np.exp(t1 - _lse(t1)) if np.isfinite(t1).any() else np.zeros_like(t1)
        d["pN"] = pn
        d["N_mean"] = float((pn * np.arange(pn.size)).sum())
        d["N_map"] = int(np.argmax(pn))
    return nb
