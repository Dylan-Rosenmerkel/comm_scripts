"""GMX - Gaussian-mixture EM deblending with mode/shape merging: one clustering algorithm for every image type.

Idea (crowded-field photometry in astronomy + model-based clustering in statistics):
  the image (dark charge removed) is a sum of elementary charge clouds.  Each cloud is a 2-D Gaussian with its
  own amplitude, centre and FULL covariance (round X-rays, elongated track pieces).  The mixture is fitted to the
  pixel charges with EM (the standard EM for a Gaussian mixture, with the pixel charge as the sample weight);
  every pixel's charge is then shared among the clouds in proportion to their model values ("responsibilities"),
  which conserves charge and needs no threshold, so dense regions do not percolate and a sharp X-ray sitting on
  a diffuse neighbour keeps only its own charge.  Physical clusters are groups of clouds: two clouds are merged
  when the two-cloud model has no significant dip between them (ridge-line / modality criterion of Ray & Lindsay
  2005 and Hennig 2010 for merging mixture components) AND their shapes are compatible with one deposit
  (similar widths, or both elongated along the line joining them = a track).

Steps
  0. background: AM.background_map (dark charge / offset / noise per pixel), hot-column excess removed (MHC)
  1. seeds: multi-scale matched-filter significance (as MFD+G, scales 0.5-2.5 px, Poisson-calibrated), local
     maxima of the max-over-scales map; one Gaussian per seed (width = best scale)
  2. EM fit of all clouds at once (sparse windows, vectorised); residual pass: significant peaks of the residual
     (noise = read noise + dark + Poisson of the model) become new clouds, EM again; insignificant clouds dropped
  3. merge graph (ridge ratio + shape compatibility) -> clusters (connected components)
  4. energies: sum over pixels of R * responsibility of the cluster (+ unexplained significant pixels next to a
     cluster, e.g. CTI trails); width = EM width (one cloud) or second moments of the shared charge
  5. photon-counting CCDs: few-electron groups (>= 1 e- pixels) not covered by a cluster, kept when their Poisson
     probability given the local dark charge is < p_count (as MHC); few-electron clusters get counting energies
  6. class labels (x-ray, x-ray diffuse, muon, worm, alpha, artifact-row/column, low-E) from shape (MHC rules)
"""
import os, sys
import numpy as np
from scipy import ndimage, stats, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
CT = os.path.dirname(os.path.abspath(__file__))
if CT not in sys.path:
    sys.path.insert(0, CT)
import agent_methods as AM
import methods2 as M2
import mhc as MH
E_PER_E = AM.E_PER_E
EIGHT = np.ones((3, 3), bool)
PIX = 1.0 / 12.0


# ------------------------------------------------------------------------------------------------ background
simple_bg = AM.simple_bg


def get_bg(Q, valid, bg=None):
    if bg is not None:
        return bg
    try:
        return AM.background_map(Q, valid)
    except Exception:
        return simple_bg(Q, valid)


# ------------------------------------------------------------------------------------------------ detection
def _sep_conv(img, s, square=False):
    r = int(np.ceil(4 * s + 1)); u = np.arange(-r, r + 1, dtype=float)
    f = AM._f1d(u, 0.0, s)[0]
    if square:
        f = f * f
    return ndimage.correlate1d(ndimage.correlate1d(img, f, axis=0, mode="constant"), f, axis=1, mode="constant")


def mf_maps(R, V, scales):
    out = []
    for s in scales:
        num = _sep_conv(R, s)
        den = np.sqrt(np.maximum(_sep_conv(V, s, square=True), 1e-12))
        out.append(num / den)
    return np.array(out)


def _local_max(T, valid, size):
    """maximum of T over a size x size window, over valid pixels only: the matched-filter significance is not
    defined outside the mask (variance 0 there), so masked pixels must not hide a peak on the mask border"""
    return ndimage.maximum_filter(np.where(valid, T, -np.inf), size=size, mode="constant", cval=-np.inf)


def poisson_keep(Q, valid, bg, py, px, nsig):
    """MFD+G's Poisson calibration of candidates (exact Poisson tail of the 3x3 / 5x5 box sum)."""
    p_thr = stats.norm.sf(nsig)
    keep = np.zeros(py.size, bool)
    R0 = np.where(valid, Q - bg.o, 0.0)
    for size in (3, 5):
        S = ndimage.uniform_filter(R0, size, mode="constant")[py, px] * size * size
        L = ndimage.uniform_filter(np.where(valid, bg.lam, 0.0), size, mode="constant")[py, px] * size * size
        sn = np.sqrt(ndimage.uniform_filter(np.where(valid, bg.s2, 0.0), size, mode="constant")[py, px] * size * size)
        k = np.floor(S - 2 * sn + 0.5)
        keep |= stats.poisson.sf(k - 1, L) < p_thr
    return keep


# ------------------------------------------------------------------------------------------------ mixture
class Mix:
    def __init__(self, A, mu, S):
        self.A = np.asarray(A, float); self.mu = np.asarray(mu, float).reshape(-1, 2); self.S = np.asarray(S, float).reshape(-1, 3)

    # S stored as (syy, sxx, sxy) of the deconvolved (physical) covariance
    def __len__(self):
        return self.A.size

    def keep(self, m):
        return Mix(self.A[m], self.mu[m], self.S[m])

    def concat(self, other):
        return Mix(np.r_[self.A, other.A], np.r_[self.mu, other.mu], np.r_[self.S, other.S])


def _tot_cov(S):
    return S[:, 0] + PIX, S[:, 1] + PIX, S[:, 2]


def evaluate(mix, shape, valid, rmax=16, qmax=16.0, r_min2=4.0):
    """Sparse model: flat pixel index, component index, model value g, offsets dy, dx (pixel - centre)."""
    H, W = shape
    a, c, b = _tot_cov(mix.S)
    tr = 0.5 * (a + c); dd = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    lmax = tr + dd
    rad = np.clip(np.ceil(np.sqrt(qmax * lmax) + 0.5), 2, rmax).astype(int)
    det = np.maximum(a * c - b * b, 1e-6)
    ia, ic, ib = c / det, a / det, -b / det
    norm = mix.A / (2 * np.pi * np.sqrt(det))
    I, K, G, DY, DX = [], [], [], [], []
    for r in np.unique(rad):
        ks = np.nonzero(rad == r)[0]
        o = np.arange(-r, r + 1)
        oy, ox = np.meshgrid(o, o, indexing="ij"); oy = oy.ravel(); ox = ox.ravel()
        cy = np.round(mix.mu[ks, 0]).astype(int); cx = np.round(mix.mu[ks, 1]).astype(int)
        py = cy[:, None] + oy[None, :]; px = cx[:, None] + ox[None, :]
        inside = (py >= 0) & (py < H) & (px >= 0) & (px < W)
        pyc, pxc = np.clip(py, 0, H - 1), np.clip(px, 0, W - 1)
        inside &= valid[pyc, pxc]
        dy = py - mix.mu[ks, 0][:, None]; dx = px - mix.mu[ks, 1][:, None]
        q = ia[ks][:, None] * dy * dy + ic[ks][:, None] * dx * dx + 2 * ib[ks][:, None] * dy * dx
        inside &= (q <= qmax) | (dy * dy + dx * dx <= r_min2)      # footprint: 4 sigma, at least 2 px
        g = norm[ks][:, None] * np.exp(-0.5 * q)
        kk = np.broadcast_to(ks[:, None], py.shape)
        I.append((pyc * W + pxc)[inside]); K.append(kk[inside]); G.append(g[inside]); DY.append(dy[inside]); DX.append(dx[inside])
    if not I:
        z = np.zeros(0)
        return z.astype(int), z.astype(int), z, z, z
    return np.concatenate(I), np.concatenate(K), np.concatenate(G), np.concatenate(DY), np.concatenate(DX)


def em(mix, Rp, eps, valid, n_iter=30, s_min2=0.02, s_max2=25.0, fix_shape=False, iso=False):
    """EM for the pixel-weighted Gaussian mixture. Rp: clipped charge (>= 0) image, eps: per-pixel floor of the
    'unexplained' density (noise) so that far noise pixels are not given to any cloud."""
    H, W = Rp.shape
    rf = Rp.ravel(); ef = eps.ravel()
    for it in range(n_iter):
        if len(mix) == 0:
            break
        I, K, G, DY, DX = evaluate(mix, Rp.shape, valid)
        M = np.bincount(I, G, H * W)
        resp = G / (M[I] + ef[I])
        wr = resp * rf[I]
        n = len(mix)
        Wk = np.bincount(K, wr, n)
        ok = Wk > 1e-3
        my = np.bincount(K, wr * DY, n) / np.maximum(Wk, 1e-12)
        mx = np.bincount(K, wr * DX, n) / np.maximum(Wk, 1e-12)
        A = Wk
        mu = mix.mu + np.c_[my, mx]
        if not fix_shape:
            ey, ex = DY - my[K], DX - mx[K]
            syy = np.bincount(K, wr * ey * ey, n) / np.maximum(Wk, 1e-12) - PIX
            sxx = np.bincount(K, wr * ex * ex, n) / np.maximum(Wk, 1e-12) - PIX
            sxy = np.bincount(K, wr * ey * ex, n) / np.maximum(Wk, 1e-12)
            if iso:
                syy = sxx = 0.5 * (syy + sxx); sxy = np.zeros_like(syy)
            syy = np.clip(syy, s_min2, s_max2); sxx = np.clip(sxx, s_min2, s_max2)
            lim = 0.95 * np.sqrt(syy * sxx)
            sxy = np.clip(sxy, -lim, lim)
            S = np.c_[syy, sxx, sxy]
        else:
            S = mix.S
        # a cloud may move at most 1.5 px per iteration (stability)
        step = mu - mix.mu
        big = np.hypot(step[:, 0], step[:, 1]) > 1.5
        if big.any():
            mu[big] = mix.mu[big] + step[big] * (1.5 / np.hypot(step[big, 0], step[big, 1]))[:, None]
        mix = Mix(A, mu, S).keep(ok)
    return mix


def significance(mix, Rp, V, valid):
    I, K, G, DY, DX = evaluate(mix, Rp.shape, valid)
    H, W = Rp.shape
    M = np.bincount(I, G, H * W)
    # matched-filter S/N of each cloud: A sqrt(sum_p p^2 / var_p), p = normalised profile
    p = G / np.maximum(mix.A[K], 1e-12)
    # null-hypothesis (no cloud) variance: read noise + dark charge + the OTHER clouds' model
    fis = np.bincount(K, p * p / (V.ravel()[I] + np.maximum(M[I] - G, 0)), len(mix))
    return mix.A * np.sqrt(fis)


# ------------------------------------------------------------------------------------------------ merging
def merge_graph(mix, r_merge=0.5, w_ratio=1.6, elong=1.4, cos_min=0.8, nt=17):
    n = len(mix)
    if n < 2:
        return np.arange(n)
    a, c, b = _tot_cov(mix.S)
    sig = np.sqrt(0.5 * (a + c))
    tree = cKDTree(mix.mu)
    P = tree.query_pairs(r=float(min(4 * sig.max() + 2, 16)), output_type="ndarray")
    if P.size == 0:
        return np.arange(n)
    i, j = P[:, 0], P[:, 1]
    d = mix.mu[j] - mix.mu[i]; dist = np.hypot(d[:, 0], d[:, 1])
    near = dist < 3.5 * (sig[i] + sig[j]) + 1.0
    i, j, d, dist = i[near], j[near], d[near], dist[near]
    det = np.maximum(a * c - b * b, 1e-6)
    ia, ic, ib = c / det, a / det, -b / det
    norm = mix.A / (2 * np.pi * np.sqrt(det))

    def dens(k, py, px):
        dy = py - mix.mu[k, 0][:, None]; dx = px - mix.mu[k, 1][:, None]
        q = ia[k][:, None] * dy * dy + ic[k][:, None] * dx * dx + 2 * ib[k][:, None] * dy * dx
        return norm[k][:, None] * np.exp(-0.5 * q)
    t = np.linspace(0, 1, nt)[None, :]
    py = mix.mu[i, 0][:, None] + t * d[:, 0][:, None]; px = mix.mu[i, 1][:, None] + t * d[:, 1][:, None]
    f = dens(i, py, px) + dens(j, py, px)
    ridge = f.min(1) / np.maximum(np.minimum(f[:, 0], f[:, -1]), 1e-12)
    # shape: widths (geometric mean) and elongation / alignment
    gw = (det) ** 0.25
    wr = np.maximum(gw[i], gw[j]) / np.minimum(gw[i], gw[j])
    tr = 0.5 * (a + c); dd = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    l1, l2 = tr + dd, np.maximum(tr - dd, 1e-6)
    el = np.sqrt(l1 / l2)
    ang = 0.5 * np.arctan2(2 * b, a - c)            # major axis angle in the (y, x) frame
    uy, ux = np.cos(ang), np.sin(ang)
    u = d / np.maximum(dist, 1e-9)[:, None]
    cos_i = np.abs(uy[i] * u[:, 0] + ux[i] * u[:, 1]); cos_j = np.abs(uy[j] * u[:, 0] + ux[j] * u[:, 1])
    track = ((el[i] > elong) & (cos_i > cos_min)) | ((el[j] > elong) & (cos_j > cos_min))
    # dissimilar widths: the pair is one deposit only if the narrower cloud is not a mode of the two-cloud model
    # (a shoulder / satellite of the broad one); a narrow cloud that is a local maximum is a separate sharp deposit
    # (e.g. a front-side X-ray on top of a diffuse neighbour)
    narrow_i = gw[i] < gw[j]
    f_n0 = np.where(narrow_i, f[:, 0], f[:, -1]); f_n1 = np.where(narrow_i, f[:, 1], f[:, -2])
    narrow_is_mode = f_n0 > f_n1
    m = (ridge > r_merge) & ((wr < w_ratio) | track | ~narrow_is_mode)
    G = sparse.coo_matrix((np.ones(m.sum()), (i[m], j[m])), shape=(n, n))
    return connected_components(G, directed=False)[1]


def island_arbitration(lab, mix, R, V, Mimg, island, valid, eps, island_max=4000, margin=10.0, f_sys=0.05, n_iter=25):
    """Islands = connected regions of significant pixels (Poisson-tail link threshold). In a bounded island (in
    crowded regions islands percolate and are skipped) that holds K >= 2 clusters, test whether the island is K
    separate X-rays: H_xr = K ROUND Gaussians (4 parameters each, EM with isotropic covariance, started from the
    clusters) against H_free = the free-shape clouds (6 parameters each). BIC(H_xr) - BIC(H_free) > margin ->
    the clusters are not separate point-like deposits -> one deposit (track, worm, streak, irregular blob).
    Pixel weights 1 / (var + b + (f_sys b)^2): f_sys tolerates the non-Gaussian shape of real X-rays."""
    H, W = R.shape
    ilab, nil = ndimage.label(island, structure=EIGHT)
    if nil == 0 or len(mix) == 0:
        return lab, []
    cy = np.clip(np.round(mix.mu[:, 0]).astype(int), 0, H - 1); cx = np.clip(np.round(mix.mu[:, 1]).astype(int), 0, W - 1)
    il = ilab[cy, cx]
    lab = lab.copy()
    objs = ndimage.find_objects(ilab)
    dec = []
    order = np.argsort(il, kind="stable")
    ils = il[order]
    cuts = np.searchsorted(ils, np.arange(nil + 2))
    for L in range(1, nil + 1):
        comps = order[cuts[L]:cuts[L + 1]]
        if comps.size < 2:
            continue
        ul = np.unique(lab[comps])
        if ul.size < 2:
            continue
        sl = objs[L - 1]
        if (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) > 4 * island_max:
            continue
        pad = 3
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        reg = ilab[y0:y1, x0:x1] == L
        if reg.sum() > island_max:
            continue
        use = ndimage.binary_dilation(reg, structure=EIGHT, iterations=2) & valid[y0:y1, x0:x1]
        b = R[y0:y1, x0:x1][use]
        w = 1 / np.sqrt(V[y0:y1, x0:x1][use] + np.clip(b, 0, None) + (f_sys * b) ** 2)
        r = (b - Mimg[y0:y1, x0:x1][use]) * w
        chi_free = float(r @ r)
        # H_xr: one round Gaussian per cluster, started from the cluster's charge-weighted clouds
        A0, mu0, S0 = [], [], []
        for u in ul:
            k = comps[lab[comps] == u]
            a_ = mix.A[k]; A = a_.sum()
            m_ = (a_[:, None] * mix.mu[k]).sum(0) / A
            v = (a_ * (0.5 * (mix.S[k, 0] + mix.S[k, 1]) + 0.5 * ((mix.mu[k] - m_) ** 2).sum(1))).sum() / A
            A0.append(A); mu0.append(m_ - [y0, x0]); S0.append([v, v, 0.0])
        Rp = np.where(use, np.clip(R[y0:y1, x0:x1], 0, None), 0.0)
        mx = em(Mix(A0, mu0, S0), Rp, eps[y0:y1, x0:x1], use, n_iter, iso=True)
        I, K, G, _, _ = evaluate(mx, Rp.shape, use)
        Mx = np.bincount(I, G, Rp.size).reshape(Rp.shape)
        r = (b - Mx[use]) * w
        chi_xr = float(r @ r)
        lnn = np.log(max(b.size, 2))
        d = (chi_xr + 4 * ul.size * lnn) - (chi_free + 6 * comps.size * lnn)
        dec.append((L, comps.size, ul.size, chi_free, chi_xr, d))
        if d > margin:
            lab[np.isin(lab, ul)] = ul[0]
    _, lab = np.unique(lab, return_inverse=True)
    return lab, dec


def scale_space_seeds(T, scales, Qc, valid, bg, nsig, min_sep=2.0):
    """Seeds from the local maxima (5x5) of EACH matched-filter scale, finest scale first: a coarser-scale maximum
    is kept only if no finer seed lies within max(min_sep, 1.5 s) (so two close X-rays give two seeds, not one
    seed at the blend centre of the coarse scale; a lone diffuse X-ray gives one)."""
    acc_y, acc_x, acc_s = [], [], []
    for k, s in enumerate(scales):
        Tk = T[k]
        pk = (Tk == _local_max(Tk, valid, 5)) & (Tk > nsig) & valid
        py, px = np.nonzero(pk)
        if py.size == 0:
            continue
        keep = poisson_keep(Qc, valid, bg, py, px, nsig)
        py, px = py[keep], px[keep]
        o = np.argsort(-Tk[py, px]); py, px = py[o], px[o]
        r = max(min_sep, 1.5 * s)
        if acc_y:
            d, _ = cKDTree(np.c_[np.concatenate(acc_y), np.concatenate(acc_x)]).query(np.c_[py, px])
            ok = d > r
            py, px = py[ok], px[ok]
        # greedy NMS within this scale
        if py.size:
            t = cKDTree(np.c_[py, px]); taken = np.zeros(py.size, bool); sel = []
            for i in range(py.size):
                if taken[i]:
                    continue
                sel.append(i); taken[t.query_ball_point([py[i], px[i]], r)] = True
            sel = np.array(sel)
            acc_y.append(py[sel]); acc_x.append(px[sel]); acc_s.append(np.full(sel.size, s))
    if not acc_y:
        z = np.zeros(0, int)
        return z, z, np.zeros(0)
    return np.concatenate(acc_y), np.concatenate(acc_x), np.concatenate(acc_s)


def _aggregate(I, K, G, M, lab, H, W):
    """cluster responsibility per (pixel, cluster) (duplicates summed) and the dominant cluster of every pixel"""
    nc = int(lab.max()) + 1
    resp = G / np.maximum(M[I], 1e-12)
    key = I.astype(np.int64) * nc + lab[K]
    uk, inv = np.unique(key, return_inverse=True)
    rc = np.bincount(inv, resp)
    pI, pL = uk // nc, uk % nc
    order = np.lexsort((-rc, pI))
    first = np.r_[True, pI[order][1:] != pI[order][:-1]]
    dom = np.full(H * W, -1); dom[pI[order][first]] = pL[order][first]
    return pI, pL, rc, dom.reshape(H, W), nc


# ------------------------------------------------------------------------------------------------ main
def general_cluster(Q, valid, bg=None, scales=(0.5, 1.0, 1.5, 2.5), nsig=5.0, nsig_res=5.0, n_iter=30, sig_min=4.0,
                    r_merge=0.5, w_ratio=1.6, hotcol=True, p_count=1e-5, count_E_max=100, grow=3, return_info=False,
                    residual_pass=True, eps_k=1.0, arbitrate=True, link_alpha=0.01, island_max=4000, bic_margin=30.0, f_sat=0.01, seed_mode="scalespace",
                    sat_e_max=1.0, ped_ring=(5.0, 8.0), ped_q=25, return_pixels=False):
    bg = get_bg(Q, valid, bg)
    H, W = Q.shape
    info = {}
    Qc = Q
    if hotcol:
        try:
            ex, hot = MH.hot_columns(Q, valid, bg)
            Qc = Q - ex[None, :]
            info["hot_columns"] = np.nonzero(hot)[0].tolist()
        except Exception:
            pass
    R = np.where(valid, Qc - bg.mean, 0.0)
    V = np.where(valid, bg.var, 1e6)
    Rp = np.clip(R, 0, None)
    eps = eps_k * np.sqrt(np.where(valid, bg.var, 1.0))
    counting = float(np.median(bg.s[valid])) < 0.5
    # 1. seeds
    T = mf_maps(R, np.where(valid, bg.var, 0.0), scales)
    if seed_mode == "maxscale":
        Tm = T.max(0); bk = T.argmax(0)
        pk = (Tm == _local_max(Tm, valid, 5)) & (Tm > nsig) & valid
        py, px = np.nonzero(pk)
        keep = poisson_keep(Qc, valid, bg, py, px, nsig)
        py, px = py[keep], px[keep]
        s0 = np.asarray(scales)[bk[py, px]]
    else:
        py, px, s0 = scale_space_seeds(T, scales, Qc, valid, bg, nsig)
    A0 = np.maximum(R[py, px] * 2 * np.pi * (s0 ** 2 + PIX), 1.0)
    mix = Mix(A0, np.c_[py, px].astype(float), np.c_[s0 ** 2, s0 ** 2, np.zeros_like(s0)])
    # 2. EM + residual pass
    mix = em(mix, Rp, eps, valid, n_iter)
    if residual_pass and len(mix):
        I, K, G, _, _ = evaluate(mix, R.shape, valid)
        M = np.bincount(I, G, H * W).reshape(H, W)
        Rr = np.where(valid, R - M, 0.0)
        Tr = mf_maps(Rr, np.where(valid, bg.var + np.clip(M, 0, None), 0.0), scales[:2])
        Trm = Tr.max(0)
        pk2 = (Trm == _local_max(Trm, valid, 3)) & (Trm > nsig_res) & valid
        qy, qx = np.nonzero(pk2)
        if qy.size:
            # the residual plus the full background (o + lam): poisson_keep removes o and tests against Poisson(lam)
            k2 = poisson_keep(np.where(valid, Rr + bg.mean, 0.0), valid, bg, qy, qx, nsig) if counting else np.ones(qy.size, bool)
            qy, qx = qy[k2], qx[k2]
            d, _ = cKDTree(mix.mu).query(np.c_[qy, qx]) if len(mix) else (np.full(qy.size, 9.0), None)
            new = d > 1.0
            qy, qx = qy[new], qx[new]
            s1 = np.asarray(scales)[Tr.argmax(0)[qy, qx]]
            add = Mix(np.maximum(Rr[qy, qx] * 2 * np.pi * (s1 ** 2 + PIX), 1.0), np.c_[qy, qx].astype(float),
                      np.c_[s1 ** 2, s1 ** 2, np.zeros_like(s1)])
            info["n_residual_seeds"] = int(len(add))
            mix = em(mix.concat(add), Rp, eps, valid, max(n_iter // 2, 10))
    if len(mix):
        z = significance(mix, Rp, V, valid)
        mix = mix.keep(z > sig_min)
        mix = em(mix, Rp, eps, valid, 5)
    info["n_clouds"] = len(mix)
    # 3. merge
    lab = merge_graph(mix, r_merge=r_merge, w_ratio=w_ratio)
    out = []
    if len(mix):
        I, K, G, DY, DX = evaluate(mix, R.shape, valid)
        M = np.bincount(I, G, H * W)
        if arbitrate:
            island = (Qc > M2.link_threshold(bg, link_alpha)) & valid
            lab, dec = island_arbitration(lab, mix, R, V, M.reshape(H, W), island, valid, eps, island_max=island_max, margin=bic_margin)
            info["decisions"] = dec
        rf = R.ravel()
        pI, pL, rc, dom, nc = _aggregate(I, K, G, M, lab, H, W)
        E = np.bincount(pL, rc * rf[pI], nc)
        if f_sat > 0:
            # satellites: a small cluster whose centre lies in a much bigger cluster's domain is part of it
            w_ = rc * np.clip(rf[pI], 0, None); sw_ = np.maximum(np.bincount(pL, w_, nc), 1e-12)
            cy_ = np.clip(np.round(np.bincount(pL, w_ * (pI // W), nc) / sw_).astype(int), 0, H - 1)
            cx_ = np.clip(np.round(np.bincount(pL, w_ * (pI % W), nc) / sw_).astype(int), 0, W - 1)
            d_ = dom[cy_, cx_]
            # ... or within 6 sigma (Mahalanobis) of a cloud of a much bigger cluster (tail fragments)
            a_, c_, b_ = _tot_cov(mix.S)
            det_ = np.maximum(a_ * c_ - b_ * b_, 1e-6)
            tree = cKDTree(mix.mu)
            for k in np.nonzero(d_ == np.arange(nc))[0]:
                js = tree.query_ball_point([cy_[k], cx_[k]], 20.0)
                best, bE = -1, 0.0
                for j in js:
                    L = lab[j]
                    if L == k or E[L] * f_sat <= E[k] or E[L] <= bE:
                        continue
                    dy, dx = cy_[k] - mix.mu[j, 0], cx_[k] - mix.mu[j, 1]
                    q = (c_[j] * dy * dy + a_[j] * dx * dx - 2 * b_[j] * dy * dx) / det_[j]
                    if q < 36:
                        best, bE = L, E[L]
                if best >= 0:
                    d_[k] = best
            # only small fragments (< sat_e_max keV) are merged: a real X-ray next to a big deposit stays its own cluster
            sat = (d_ >= 0) & (d_ != np.arange(nc)) & (E < f_sat * E[np.maximum(d_, 0)]) & (E * E_PER_E < sat_e_max)
            if sat.any():
                tgt = np.arange(nc); tgt[sat] = d_[sat]
                for _ in range(5):
                    tgt = tgt[tgt]
                _, lab = np.unique(tgt[lab], return_inverse=True)
                pI, pL, rc, dom, nc = _aggregate(I, K, G, M, lab, H, W)
                E = np.bincount(pL, rc * rf[pI], nc)
                info["n_satellites"] = int(sat.sum())
        # 4b. grow: significant pixels next to (<= grow px) and connected to covered pixels, not covered themselves
        if grow:
            covered = dom >= 0
            thr = M2.link_threshold(bg, 0.01)
            cand = (Qc > thr) & valid & ~covered
            if cand.any():
                dist, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
                cand &= dist <= grow
                F = ndimage.binary_propagation(covered & ndimage.binary_dilation(cand, structure=EIGHT), structure=EIGHT, mask=cand | covered)
                extra = F & cand
                ey, ex_ = np.nonzero(extra)
                own = dom[iy[ey, ex_], ix[ey, ex_]]
                E += np.bincount(own, R[ey, ex_], nc)
                gpix = (ey, ex_, own)
            else:
                gpix = None
        else:
            gpix = None
        # moments of the shared charge
        py_ = (pI // W).astype(float); px_ = (pI % W).astype(float)
        w = rc * np.clip(rf[pI], 0, None)
        sw = np.maximum(np.bincount(pL, w, nc), 1e-12)
        cy = np.bincount(pL, w * py_, nc) / sw; cx = np.bincount(pL, w * px_, nc) / sw
        vy = np.bincount(pL, w * (py_ - cy[pL]) ** 2, nc) / sw; vx = np.bincount(pL, w * (px_ - cx[pL]) ** 2, nc) / sw
        sxy_m = np.sqrt(np.clip(0.5 * (vy + vx) - PIX, 0.0, None))
        varE = np.bincount(pL, rc ** 2 * V.ravel()[pI], nc)
        # blend: charge of the other clusters under this cluster's profile, relative to its own (0 = isolated)
        Mc = M[pI] * rc; Mo = M[pI] - Mc
        blend = np.bincount(pL, Mc * Mo, nc) / np.maximum(np.bincount(pL, Mc * Mc, nc), 1e-12)
        ncomp = np.bincount(lab, minlength=nc)
        edge = np.zeros(nc, bool)
        vd = ndimage.binary_erosion(valid, iterations=2)
        edge[np.unique(pL[~vd.ravel()[pI]])] = True
        # quality flags (energies are not changed by them):
        #   peakfrac: share of the cluster's fitted charge carried by its largest cloud (merged neighbours -> < 1)
        #   chi2    : sum (R - M)^2 / (V + M) per pixel over the pixels the cluster dominates (its own model's fit)
        #   ped     : local pedestal = ped_q-th percentile of R in a ring ped_ring px around the centre (charge sheet
        #             under the cluster in crowded fields; its share of that sheet is inside E)
        Asum = np.bincount(lab, mix.A, nc); Amax = np.zeros(nc); np.maximum.at(Amax, lab, mix.A)
        peakfrac = Amax / np.maximum(Asum, 1e-12)
        dmn = rc > 0.5
        Rv, Mv = R.ravel()[pI], M[pI]
        chi = np.bincount(pL[dmn], ((Rv - Mv) ** 2 / (V.ravel()[pI] + np.clip(Mv, 0, None)))[dmn], nc)
        chi2 = chi / np.maximum(np.bincount(pL[dmn], minlength=nc), 1)
        ped = _pedestal(R, valid, cy, cx, ped_ring, ped_q)
        # counting energies for few-electron clusters (photon-counting CCDs)
        if counting:
            maj = rc > 0.5
            n_e = np.round((Qc - bg.o).ravel()[pI])
            En = np.bincount(pL[maj], n_e[maj] - bg.lam.ravel()[pI][maj], nc)
        if return_pixels:
            # per-cluster pixels and the charge [e-] the cluster takes from each (sums exactly to its energy):
            # shared pixels (rc * R), grown pixels (R), or for counting-mode clusters the counted electrons
            Rf = R.ravel()
            po = np.argsort(pL, kind="stable"); pIs, vals = pI[po], (rc * Rf[pI])[po]
            pcut = np.searchsorted(pL[po], np.arange(nc + 1))
            if gpix is not None:
                go = np.argsort(gpix[2], kind="stable"); gys, gxs = gpix[0][go], gpix[1][go]
                gcut = np.searchsorted(gpix[2][go], np.arange(nc + 1))
            if counting:
                mo = np.argsort(pL[maj], kind="stable"); mIs = pI[maj][mo]
                mvals = (n_e[maj] - bg.lam.ravel()[pI][maj])[mo]; mcut = np.searchsorted(pL[maj][mo], np.arange(nc + 1))
        for k in range(nc):
            comps = np.nonzero(lab == k)[0]
            if ncomp[k] == 1:
                syy, sxx, sxy = mix.S[comps[0]]
                sx = float(np.sqrt(max(0.5 * (syy + sxx), 0)))
            else:
                sx = float(sxy_m[k])
            e = float(E[k])
            c = dict(E=e * E_PER_E, y=float(cy[k]), x=float(cx[k]), sxy=sx, sxy_mom=float(sxy_m[k]), n_comp=int(ncomp[k]),
                     sig=float(e / np.sqrt(max(varE[k], 1e-9))), edge=bool(edge[k]), blend=float(blend[k]),
                     ped=float(ped[k]), peakfrac=float(peakfrac[k]), chi2=float(chi2[k]))
            if counting and e < count_E_max:
                c["E"] = float(En[k] * E_PER_E); c["E_fit"] = e * E_PER_E
            if return_pixels:
                if counting and e < count_E_max:
                    ii, vv = mIs[mcut[k]:mcut[k + 1]], mvals[mcut[k]:mcut[k + 1]]
                    py_, px_ = ii // W, ii % W
                else:
                    ii, vv = pIs[pcut[k]:pcut[k + 1]], vals[pcut[k]:pcut[k + 1]]
                    py_, px_ = ii // W, ii % W
                    if gpix is not None and gcut[k + 1] > gcut[k]:
                        gy, gx = gys[gcut[k]:gcut[k + 1]], gxs[gcut[k]:gcut[k + 1]]
                        py_, px_, vv = np.r_[py_, gy], np.r_[px_, gx], np.r_[vv, R[gy, gx]]
                c["_pix"] = (np.asarray(py_, int), np.asarray(px_, int), np.asarray(vv, float))
            out.append(c)
        # 6. classes
        _classify(out, mix, lab, R, Rp, pI, pL, rc, W)
    # 5. counting finder
    if counting:
        covered = np.zeros((H, W), bool)
        if len(mix):
            covered.ravel()[pI[rc > 0.05]] = True
        out += _counting(Qc, valid, bg, covered, p_count)
    fill_flags(out, R, valid, ped_ring, ped_q)
    if return_info:
        info["mix"] = mix; info["lab"] = lab
        return out, info
    return out


def fill_flags(clusters, R, valid, ped_ring=(5.0, 8.0), ped_q=25):
    """quality flags for clusters the mixture did not fit (few-electron finder, hysteresis): edge (a pixel within
    2 px of the mask, as for fitted clusters), ped (local pedestal of R), blend = 0 and peakfrac = 1 (no charge is
    shared), chi2 = NaN (no model). Clusters that already have flags are left alone. Needs '_pix' or y/x."""
    need = [c for c in clusters if "ped" not in c]
    if not need:
        return clusters
    vd = ndimage.binary_erosion(valid, iterations=2)
    cy = np.array([c["y"] for c in need]); cx = np.array([c["x"] for c in need])
    ped = _pedestal(R, valid, cy, cx, ped_ring, ped_q)
    for c, p in zip(need, ped):
        if "_pix" in c:
            ys, xs = np.asarray(c["_pix"][0], int), np.asarray(c["_pix"][1], int)
        else:
            ys, xs = np.array([int(round(c["y"]))]), np.array([int(round(c["x"]))])
        c.setdefault("edge", bool((~vd[ys, xs]).any()))
        c.update(ped=float(p))
        c.setdefault("blend", 0.0); c.setdefault("peakfrac", 1.0); c.setdefault("chi2", np.nan)
    return clusters


def _pedestal(R, valid, cy, cx, ring=(5.0, 8.0), q=25, min_pix=15):
    """local pedestal of each cluster: the q-th percentile of the residual charge R in the ring r_in <= r <= r_out
    (px) around its centre, all active pixels (the 25th percentile is robust against neighbouring deposits in the
    ring). NaN when fewer than min_pix active ring pixels."""
    H, W = R.shape
    r_in, r_out = ring; ro = int(np.ceil(r_out))
    out = np.full(len(cy), np.nan)
    for k in range(len(cy)):
        y, x = cy[k], cx[k]
        yi, xi = int(round(y)), int(round(x))
        y0, y1, x0, x1 = max(yi - ro, 0), min(yi + ro + 1, H), max(xi - ro, 0), min(xi + ro + 1, W)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(yy - y, xx - x)
        m = (r >= r_in) & (r <= r_out) & valid[y0:y1, x0:x1]
        if m.sum() >= min_pix:
            out[k] = float(np.percentile(R[y0:y1, x0:x1][m], q))
    return out


def _classify(out, mix, lab, R, Rp, pI, pL, rc, W):
    order = np.argsort(pL, kind="stable")
    pLs, pIs, rcs = pL[order], pI[order], rc[order]
    cuts = np.searchsorted(pLs, np.arange(len(out) + 1))
    for k, c in enumerate(out):
        E = c["E"]
        if E < 0.5:
            c["cls"] = "low-E"; continue
        a, b = cuts[k], cuts[k + 1]
        sel = rcs[a:b] > 0.5
        ys = pIs[a:b][sel] // W; xs = pIs[a:b][sel] % W
        q = R.ravel()[pIs[a:b][sel]] * rcs[a:b][sel]
        if ys.size == 0:
            c["cls"] = "low-E"; continue
        sh = MH.shape(ys.astype(float), xs.astype(float), q)
        c.update(length=sh["length"], width=sh["width"])
        ext = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        if c["n_comp"] == 1 and sh["ratio"] > 0.5 or ext < 6:
            c["cls"] = "alpha" if E > 1000 else ("x-ray diffuse" if c["sxy"] >= 1.0 else "x-ray")
            continue
        # streak test on the cluster's own pixels
        y0, x0 = ys.min(), xs.min()
        reg = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), bool); reg[ys - y0, xs - x0] = True
        Rr = np.zeros(reg.shape); Rr[ys - y0, xs - x0] = q
        st = MH.streak(reg, reg, Rr)
        if st is not None:
            c["cls"] = st; continue
        comps = np.nonzero(lab == k)[0]
        by, bx = mix.mu[comps, 0], mix.mu[comps, 1]
        sk = MH.shape(by, bx, np.ones(by.size)) if by.size >= 3 else dict(straight=0.0)
        sh["sk_straight"] = sk["straight"]
        c["cls"] = MH.classify_ext(E, sh, ys.size)


def _counting(Qc, valid, bg, covered, p_count):
    out = []
    nn = np.where(valid, np.round(Qc - bg.o), 0).clip(0)
    cl2, k = ndimage.label(nn >= 1, structure=EIGHT)
    for j, sl in enumerate(ndimage.find_objects(cl2), start=1):
        m = cl2[sl] == j
        if m.sum() > 9 or covered[sl][m].any():
            continue
        bsl = (slice(max(sl[0].start - 1, 0), sl[0].stop + 1), slice(max(sl[1].start - 1, 0), sl[1].stop + 1))
        box = ndimage.binary_dilation(cl2[bsl] == j, structure=EIGHT)
        S = float(nn[sl][m].sum())
        mu = float(bg.lam[bsl][box].sum())
        p = float(stats.poisson.sf(S - 1, max(mu, 1e-9)))
        if p < p_count:
            ys, xs = np.nonzero(m); q = nn[sl][m]
            out.append(dict(E=float((S - bg.lam[sl][m].sum()) * E_PER_E), y=float((q * (ys + sl[0].start)).sum() / S),
                            x=float((q * (xs + sl[1].start)).sum() / S), sxy=np.nan, npix=int(m.sum()), cls="low-E",
                            n_comp=0, sig=float(stats.norm.isf(max(p, 1e-300))),
                            _pix=(ys + sl[0].start, xs + sl[1].start, q - bg.lam[sl][m])))
    return out
