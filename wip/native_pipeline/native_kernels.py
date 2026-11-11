"""Numba kernels for the per-cluster quantities of the native pipeline (native_pipeline.py).

Every function works on all clusters of a CCD at once: the pixels of all clusters are concatenated in flat arrays
and off[k]:off[k+1] are the pixels of cluster k. The formulas are those of pysimdamicm's Cluster class
(io/data_formats.py) and ClusterFinder (processes/reconstruction.py), so the branches mean the same as in panaSKImg
files; the two Gaussian profile fits (fSTD_X/Y) replace ROOT's TH1::Fit("gaus", "Q L") by the same Poisson
likelihood minimised with a small damped Newton solver (see fit_gaus_poisson).

Compiled on first use and cached on disk (numba cache=True, __pycache__ next to this file).
"""
import math

import numpy as np
from numba import njit

SQRTPI_ROOT = 2.506628        # ROOT::Fit::InitGaus uses this rounded sqrt(2 pi) for its amplitude guess


# =============================================================================
# moments, widths, brightest pixel, covariance (Cluster.get_properties_on_axes / get_charge_properties)
# =============================================================================
@njit(cache=True)
def _pw_block(a, lo, n):
    """numpy's pairwise_sum for n <= 128: plain loop below 8, else 8 interleaved partial sums"""
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[lo + i]
        return r
    r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
    r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def pw_sum(a, lo, n):
    """sum of a[lo:lo+n] exactly as ndarray.sum() of a contiguous float64 array: numpy reduces in chunks of 8192
    values (its buffer size), each chunk summed pairwise (_pw_tree), the chunk sums added in order"""
    t = 0.0
    i = 0
    while i < n:
        m = min(8192, n - i)
        t += _pw_tree(a, lo + i, m)
        i += m
    return t


@njit(cache=True)
def _pw_tree(a, lo, n):
    """sum of a[lo:lo+n] in numpy's pairwise order (bit-identical to ndarray.sum() of a contiguous float64 array:
    blocks of <= 128 values, larger ranges split in two halves (multiple of 8) and the halves added). Used wherever
    panaSKImg compares or keeps the last bits: e.g. closest_cls_* skips neighbours with exactly the same PosX or PosY,
    and the sign of a near-zero pcaCovXY decides the PCA axis directions. (Written without recursion: numba's
    on-disk cache does not handle recursive functions.)"""
    if n <= 128:
        return _pw_block(a, lo, n)
    sl = np.empty(128, np.int64); sn = np.empty(128, np.int64); ss = np.zeros(128, np.int64)
    vals = np.empty(128)
    top = 0; nv = 0
    sl[0] = lo; sn[0] = n; ss[0] = 0
    while top >= 0:
        l_, n_ = sl[top], sn[top]
        if n_ <= 128:
            vals[nv] = _pw_block(a, l_, n_); nv += 1
            top -= 1
            continue
        n2 = n_ // 2
        n2 -= n2 % 8
        if ss[top] == 0:
            ss[top] = 1
            top += 1; sl[top] = l_; sn[top] = n2; ss[top] = 0
        elif ss[top] == 1:
            ss[top] = 2
            top += 1; sl[top] = l_ + n2; sn[top] = n_ - n2; ss[top] = 0
        else:
            vals[nv - 2] = vals[nv - 2] + vals[nv - 1]; nv -= 1
            top -= 1
    return vals[0]


@njit(cache=True)
def moments(x, y, E, off):
    """Per-cluster scalars; x, y: 1-based pixel coordinates (float64), E: pixel energies [keV].

    Returns a (n, 25) array, columns in the order of MOMENT_NAMES. As pysimdamicm, with numpy's summation order
    (pw_sum) so the values agree to the last bit: RMS = mean of x^2 (not a width), D = max - min + 1, Pos =
    energy-weighted mean (np.average), STD = unweighted std, wSTD = energy-weighted std (nan if the weighted
    variance is negative), STD_XY = unweighted radial rms, wSTD_XY = sqrt((wSTD_X^2 + wSTD_Y^2) / 2), Qmax = first
    brightest pixel, pcaCovXY = weighted covariance."""
    n = off.size - 1
    out = np.empty((n, 25))
    for k in range(n):
        a, b = off[k], off[k + 1]
        m = b - a
        xs = x[a:b]; ys = y[a:b]; es = E[a:b]
        mnx = xs.min(); mxx = xs.max(); mny = ys.min(); mxy = ys.max()
        qi = 0
        for i in range(m):
            if es[i] > es[qi]:
                qi = i
        se = pw_sum(es, 0, m)
        mx = pw_sum(xs, 0, m) / m; my = pw_sum(ys, 0, m) / m
        t = np.empty(m)
        for i in range(m):
            t[i] = xs[i] * xs[i]
        rmsx = pw_sum(t, 0, m) / m
        for i in range(m):
            t[i] = ys[i] * ys[i]
        rmsy = pw_sum(t, 0, m) / m
        for i in range(m):
            t[i] = (xs[i] - mx) ** 2 + (ys[i] - my) ** 2
        stdxy = math.sqrt(pw_sum(t, 0, m) / m)
        for i in range(m):
            t[i] = (xs[i] - mx) * (xs[i] - mx)
        stdx = math.sqrt(pw_sum(t, 0, m) / m)
        for i in range(m):
            t[i] = (ys[i] - my) * (ys[i] - my)
        stdy = math.sqrt(pw_sum(t, 0, m) / m)
        for i in range(m):
            t[i] = xs[i] * es[i]
        posx = pw_sum(t, 0, m) / se
        for i in range(m):
            t[i] = ys[i] * es[i]
        posy = pw_sum(t, 0, m) / se
        for i in range(m):
            t[i] = es[i] * (xs[i] - posx) ** 2
        vx = pw_sum(t, 0, m) / se
        for i in range(m):
            t[i] = es[i] * (ys[i] - posy) ** 2
        vy = pw_sum(t, 0, m) / se
        for i in range(m):
            t[i] = es[i] * (xs[i] - posx) * (ys[i] - posy)
        cxy = pw_sum(t, 0, m) / se
        wsx = math.sqrt(vx) if vx >= 0 else np.nan
        wsy = math.sqrt(vy) if vy >= 0 else np.nan
        o = out[k]
        o[0] = m; o[1] = se; o[2] = stdxy
        o[3] = mnx; o[4] = mxx; o[5] = mx; o[6] = rmsx; o[7] = mxx - mnx + 1.0; o[8] = posx
        o[9] = stdx; o[10] = wsx
        o[11] = mny; o[12] = mxy; o[13] = my; o[14] = rmsy; o[15] = mxy - mny + 1.0; o[16] = posy
        o[17] = stdy; o[18] = wsy
        o[19] = math.sqrt((wsx * wsx + wsy * wsy) / 2.0)
        o[20] = es[qi]; o[21] = xs[qi]; o[22] = ys[qi]
        o[23] = cxy                             # pcaCovXY (pcaCovXX/YY = wSTD^2, set by the caller)
        o[24] = qi                              # index of the brightest pixel inside the cluster
    return out


MOMENT_NAMES = ["Npix", "Energy", "STD_XY", "minX", "maxX", "meanX", "RMSX", "DX", "PosX", "STD_X", "wSTD_X",
                "minY", "maxY", "meanY", "RMSY", "DY", "PosY", "STD_Y", "wSTD_Y", "wSTD_XY",
                "Qmax", "QmaxX", "QmaxY", "pcaCovXY", "_iqmax"]


# =============================================================================
# PCA projections (Cluster.compute_cluster_pca, after the eigen-decomposition done in numpy)
# =============================================================================
@njit(cache=True)
def np_percentile(v, q):
    """np.percentile(v, q) (method 'linear') to the last bit: index (n - 1) q / 100 and numpy's _lerp, which
    interpolates from the upper value when the fraction is >= 0.5 (numba's np.percentile differs in the last bit)"""
    s = np.sort(v)
    n = s.size
    vi = (n - 1) * (q / 100.0)
    lo = int(math.floor(vi))
    hi = min(lo + 1, n - 1)
    t = vi - lo
    a = s[lo]; d = s[hi] - a
    if t >= 0.5:
        return s[hi] - d * (1.0 - t)
    return a + d * t


@njit(cache=True)
def pca_projections(x, y, off, posx, posy, lx, ly, tx, ty):
    """Pixel coordinates on the principal axes (u along, v across) and the robust length / width:
    L = p95(u) - p5(u), W = p84(v) - p16(v) (unweighted percentiles, np_percentile)."""
    n = off.size - 1
    u = np.empty(x.size); v = np.empty(x.size)
    L = np.empty(n); W = np.empty(n)
    for k in range(n):
        a, b = off[k], off[k + 1]
        for i in range(a, b):
            dx = x[i] - posx[k]; dy = y[i] - posy[k]
            u[i] = dx * lx[k] + dy * ly[k]
            v[i] = dx * tx[k] + dy * ty[k]
        if np.isnan(u[a:b]).any():
            L[k] = np.nan
        else:
            L[k] = np_percentile(u[a:b], 95.0) - np_percentile(u[a:b], 5.0)
        if np.isnan(v[a:b]).any():
            W[k] = np.nan
        else:
            W[k] = np_percentile(v[a:b], 84.0) - np_percentile(v[a:b], 16.0)
    return u, v, L, W


# =============================================================================
# Gaussian fits of the x / y profiles (Cluster.get_fitted_STD)
# =============================================================================
@njit(cache=True)
def _nll(c, yv, la, mu, s):
    """Poisson negative log-likelihood of ROOT's "L" fit, sum_j f_j - y_j ln f_j, with ln f = la - z^2 / 2."""
    t = 0.0
    for j in range(c.size):
        z = (c[j] - mu) / s
        lf = la - 0.5 * z * z
        t += math.exp(lf) - yv[j] * lf
    return t


@njit(cache=True)
def _solve3(H, g, free):
    """Newton step -H^-1 g on the free parameters (Gaussian elimination with partial pivoting, 3x3)."""
    M = np.zeros((3, 4))
    for i in range(3):
        for j in range(3):
            M[i, j] = H[i, j] if (free[i] and free[j]) else (1.0 if i == j else 0.0)
        M[i, 3] = -g[i] if free[i] else 0.0
    for col in range(3):
        p = col
        for r in range(col + 1, 3):
            if abs(M[r, col]) > abs(M[p, col]):
                p = r
        if abs(M[p, col]) < 1e-300:
            return np.zeros(3), False
        if p != col:
            for j in range(4):
                tmp = M[col, j]; M[col, j] = M[p, j]; M[p, j] = tmp
        for r in range(col + 1, 3):
            f = M[r, col] / M[col, col]
            for j in range(col, 4):
                M[r, j] -= f * M[col, j]
    st = np.zeros(3)
    for i in range(2, -1, -1):
        acc = M[i, 3]
        for j in range(i + 1, 3):
            acc -= M[i, j] * st[j]
        st[i] = acc / M[i, i]
    return st, True


@njit(cache=True)
def fit_gaus_poisson(c, yv, mu_lo, mu_hi):
    """ROOT's TH1::Fit("gaus", "Q L") of one cluster profile: bin centres c, contents yv.

    As ROOT: starting values from ROOT::Fit::InitGaus (data mean and rms, amplitude 0.5 (max + binw sum /
    (2.506628 rms))), which also limits sigma to [0, 10 rms]; the mean is limited to [mu_lo, mu_hi] (minX, maxX of
    the cluster) and FIXED when mu_lo == mu_hi (ROOT fixes a parameter whose limits are equal). The likelihood
    uses the bin contents as they are (weighted fills, no "WL"), f evaluated at the bin centres, empty bins
    included. Minimised with a damped Newton method in (ln A, mu, sigma), bounds by clipping plus an active set.
    Returns (mu, sigma)."""
    nb = c.size
    sumx = 0.0; sumx2 = 0.0; allcha = 0.0; valmax = 0.0
    binw = c[nb - 1] - c[0] if nb > 1 else 1.0
    if binw <= 0:
        binw = 1.0
    for j in range(nb):
        sumx += yv[j] * c[j]; sumx2 += yv[j] * c[j] * c[j]; allcha += yv[j]
        if yv[j] > valmax:
            valmax = yv[j]
        if j > 0 and c[j] - c[j - 1] < binw:
            binw = c[j] - c[j - 1]
    if allcha <= 0:
        return np.nan, np.nan
    mean = sumx / allcha
    rms2 = sumx2 / allcha - mean * mean
    rms = math.sqrt(rms2) if rms2 > 0 else binw * nb / 4.0
    amp = 0.5 * (valmax + binw * allcha / (SQRTPI_ROOT * rms))
    s_hi = 10.0 * rms
    s_lo = 1e-9 * s_hi
    fix_mu = mu_lo == mu_hi
    mu0 = mean if fix_mu else min(max(mean, mu_lo), mu_hi)
    # ROOT starts once, from InitGaus; Minuit then stops when the estimated distance to the minimum is below its
    # tolerance, which leaves sigma off by up to ~1e-3 (relative) on flat likelihoods. Here the minimum is followed
    # to convergence, from InitGaus and three more widths (a multi-peak profile can trap a single start in a worse
    # local minimum); the lowest likelihood wins.
    best_f = np.inf; bm = np.nan; bs = np.nan
    for fac in (1.0, 0.5, 0.25, 2.0):
        s0 = min(max(rms * fac, s_lo), s_hi)
        m_, s_, f_ = _newton(c, yv, math.log(max(amp, 1e-300)), mu0, s0, mu_lo, mu_hi, s_lo, s_hi, fix_mu)
        if not (best_f < np.inf) or f_ < best_f - 1e-12 * max(1.0, abs(best_f)):
            best_f = f_; bm = m_; bs = s_
    return bm, bs


@njit(cache=True)
def _newton(c, yv, la0, mu0, s0, mu_lo, mu_hi, s_lo, s_hi, fix_mu):
    """damped Newton minimisation of _nll in (ln A, mu, sigma) with box limits; returns (mu, sigma, nll)"""
    nb = c.size
    th = np.array([la0, mu0, s0])
    lo = np.array([-1e300, mu_lo, s_lo]); hi = np.array([1e300, mu_hi, s_hi])
    f0 = _nll(c, yv, th[0], th[1], th[2])
    lam = 1e-3
    for it in range(300):
        la, m_, s = th[0], th[1], th[2]
        g = np.zeros(3); H = np.zeros((3, 3))
        for j in range(nb):
            z = (c[j] - m_) / s
            lf = la - 0.5 * z * z
            f = math.exp(lf)
            r = f - yv[j]
            d0 = 1.0; d1 = z / s; d2 = z * z / s
            g[0] += r * d0; g[1] += r * d1; g[2] += r * d2
            H[0, 0] += f; H[0, 1] += f * d1; H[0, 2] += f * d2
            H[1, 1] += f * d1 * d1 - r / (s * s)
            H[1, 2] += f * d1 * d2 - r * 2.0 * z / (s * s)
            H[2, 2] += f * d2 * d2 - r * 3.0 * z * z / (s * s)
        H[1, 0] = H[0, 1]; H[2, 0] = H[0, 2]; H[2, 1] = H[1, 2]
        free = np.array([True, not fix_mu, True])
        for i in (1, 2):                       # active set: parameter on a bound, gradient pushing outward
            if free[i] and ((th[i] <= lo[i] and g[i] > 0) or (th[i] >= hi[i] and g[i] < 0)):
                free[i] = False
        accepted = False
        best = th.copy(); fbest = f0
        for att in range(40):
            Hd = H.copy()
            for i in range(3):
                Hd[i, i] += lam * (abs(H[i, i]) + 1e-12)
            st, ok = _solve3(Hd, g, free)
            if not ok:
                lam *= 10.0
                continue
            tr = th + st
            for i in range(3):
                tr[i] = min(max(tr[i], lo[i]), hi[i])
            ft = _nll(c, yv, tr[0], tr[1], tr[2])
            if ft < f0:
                best = tr; fbest = ft; accepted = True
                lam = max(lam * 0.1, 1e-12)
                break
            lam *= 10.0
        if not accepted:
            break
        dmax = 0.0
        for i in range(3):
            dmax = max(dmax, abs(best[i] - th[i]))
        df = f0 - fbest
        th = best; f0 = fbest
        if df < 1e-14 * max(1.0, abs(f0)) and dmax < 1e-10:
            break
    return th[1], th[2], f0


@njit(cache=True)
def seg_sums(a, off):
    """per-cluster sums of a in numpy's order (ndarray.sum() of each cluster's values)"""
    n = off.size - 1
    out = np.empty(n)
    for k in range(n):
        out[k] = pw_sum(a, off[k], off[k + 1] - off[k])
    return out


@njit(cache=True)
def profiles(xi, yi, E, off, minx, maxx, miny, maxy):
    """projectionX/Y (TH1D of the cluster with weights E / max(E), 1-pixel bins from min - 1.5 to max + 1.5) and the
    Gaussian fits of both profiles. Returns (projX flat, offX, projY flat, offY, fPosX, fSTD_X, fPosY, fSTD_Y)."""
    n = off.size - 1
    nbx = np.empty(n, np.int64); nby = np.empty(n, np.int64)
    for k in range(n):
        nbx[k] = int(maxx[k] - minx[k]) + 3; nby[k] = int(maxy[k] - miny[k]) + 3
    offx = np.zeros(n + 1, np.int64); offy = np.zeros(n + 1, np.int64)
    for k in range(n):
        offx[k + 1] = offx[k] + nbx[k]; offy[k + 1] = offy[k] + nby[k]
    px = np.zeros(offx[n]); py = np.zeros(offy[n])
    fpx = np.empty(n); fsx = np.empty(n); fpy = np.empty(n); fsy = np.empty(n)
    for k in range(n):
        a, b = off[k], off[k + 1]
        emax = E[a]
        for i in range(a, b):
            if E[i] > emax:
                emax = E[i]
        for i in range(a, b):
            px[offx[k] + xi[i] - int(minx[k]) + 1] += E[i] / emax
            py[offy[k] + yi[i] - int(miny[k]) + 1] += E[i] / emax
        cx = np.empty(nbx[k]); cy = np.empty(nby[k])
        for j in range(nbx[k]):
            cx[j] = minx[k] - 1.0 + j
        for j in range(nby[k]):
            cy[j] = miny[k] - 1.0 + j
        fpx[k], fsx[k] = fit_gaus_poisson(cx, px[offx[k]:offx[k + 1]], minx[k], maxx[k])
        fpy[k], fsy[k] = fit_gaus_poisson(cy, py[offy[k]:offy[k + 1]], miny[k], maxy[k])
    return px, offx, py, offy, fpx, fsx, fpy, fsy


# =============================================================================
# track end points (Cluster.get_tracking_endpoints) and the closest cluster (ClusterFinder.get_closest_cluster)
# =============================================================================
@njit(cache=True)
def _bfs(start, gx, gy, grid, w, h, dist, order):
    """breadth-first search over the 8-connected cluster pixels from pixel index `start` (local grid);
    neighbour order as pysimdamicm: (dx, dy) = (-1,-1) (0,-1) (1,-1) (-1,0) (1,0) (-1,1) (0,1) (1,1).
    Fills dist (-1 = not reached) and order (discovery order); returns the number reached."""
    DX = (-1, 0, 1, -1, 1, -1, 0, 1)
    DY = (-1, -1, -1, 0, 0, 1, 1, 1)
    for i in range(dist.size):
        dist[i] = -1
    dist[start] = 0
    order[0] = start
    head = 0; tail = 1
    while head < tail:
        p = order[head]; head += 1
        for q in range(8):
            nx = gx[p] + DX[q]; ny = gy[p] + DY[q]
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            r = grid[ny, nx]
            if r < 0 or dist[r] >= 0:
                continue
            dist[r] = dist[p] + 1
            order[tail] = r; tail += 1
    return tail


@njit(cache=True)
def _farthest(n_reached, order, dist, gx, gy, ref):
    """first pixel (in discovery order) with the largest (graph distance, squared euclidean distance to ref)"""
    best = order[0]
    bd = dist[best]; be = (gx[best] - gx[ref]) ** 2 + (gy[best] - gy[ref]) ** 2
    for t in range(1, n_reached):
        p = order[t]
        e = (gx[p] - gx[ref]) ** 2 + (gy[p] - gy[ref]) ** 2
        if dist[p] > bd or (dist[p] == bd and e > be):
            best = p; bd = dist[p]; be = e
    return best


@njit(cache=True)
def track_endpoints(xi, yi, off, iqmax):
    """Two end points of each cluster: the pixel farthest (8-connected path length, ties by euclidean distance)
    from the brightest pixel, then the pixel farthest from that one. Duplicate pixels count once (first copy).
    Returns (x1, y1, x2, y2), 1-based like pixels_x / pixels_y."""
    n = off.size - 1
    ex1 = np.empty(n); ey1 = np.empty(n); ex2 = np.empty(n); ey2 = np.empty(n)
    for k in range(n):
        a, b = off[k], off[k + 1]
        x0 = xi[a]; y0 = yi[a]; x1 = xi[a]; y1 = yi[a]
        for i in range(a, b):
            x0 = min(x0, xi[i]); x1 = max(x1, xi[i]); y0 = min(y0, yi[i]); y1 = max(y1, yi[i])
        w = x1 - x0 + 1; h = y1 - y0 + 1
        grid = -np.ones((h, w), np.int64)
        m = b - a
        gx = np.empty(m, np.int64); gy = np.empty(m, np.int64)
        nu = 0
        seed = -1
        for i in range(a, b):
            lx = xi[i] - x0; ly = yi[i] - y0
            if grid[ly, lx] < 0:
                grid[ly, lx] = nu; gx[nu] = lx; gy[nu] = ly
                if i - a == iqmax[k] and seed < 0:
                    seed = nu
                nu += 1
            elif i - a == iqmax[k] and seed < 0:
                seed = grid[ly, lx]
        if nu == 1:
            ex1[k] = xi[a]; ey1[k] = yi[a]; ex2[k] = xi[a]; ey2[k] = yi[a]
            continue
        dist = np.empty(nu, np.int64); order = np.empty(nu, np.int64)
        nr = _bfs(seed, gx, gy, grid, w, h, dist, order)
        e1 = _farthest(nr, order, dist, gx, gy, seed)
        nr = _bfs(e1, gx, gy, grid, w, h, dist, order)
        e2 = _farthest(nr, order, dist, gx, gy, e1)
        ex1[k] = gx[e1] + x0; ey1[k] = gy[e1] + y0; ex2[k] = gx[e2] + x0; ey2[k] = gy[e2] + y0
    return ex1, ey1, ex2, ey2


@njit(cache=True)
def closest(posx, posy, hasseed):
    """closest_cls_*: nearest other cluster with a seed (distance of the energy-weighted centres, |dX|, |dY|,
    1-based index). panaSKImg's rule for "other": PosX != PosX_i AND PosY != PosY_i (a neighbour in exactly the same
    row or column position is skipped too); clusters without a seed get -1. If no candidate exists the result is
    (inf, |dX|, |dY|) of cluster 1, as numpy's argmin over an all-inf array."""
    n = posx.size
    d = np.full(n, -1.0); dx = np.full(n, -1.0); dy = np.full(n, -1.0); cid = np.full(n, -1.0)
    if n <= 1:
        return d, dx, dy, cid
    for i in range(n):
        if not hasseed[i]:
            continue
        best = np.inf; jb = 0
        for j in range(n):
            if posx[j] != posx[i] and posy[j] != posy[i] and hasseed[j]:
                dd = math.sqrt((posx[j] - posx[i]) ** 2 + (posy[j] - posy[i]) ** 2)
                if dd < best:
                    best = dd; jb = j
        d[i] = best; dx[i] = abs(posx[jb] - posx[i]); dy[i] = abs(posy[jb] - posy[i]); cid[i] = jb + 1.0
    return d, dx, dy, cid
