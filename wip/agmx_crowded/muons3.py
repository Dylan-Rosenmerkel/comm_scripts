"""Straight-track (muon) finder v3 = v2 + iterative (EM) charge attribution in the tube (share="em").
v2 docstring follows.
Straight-track (muon) finder v2 for crowded images (scratch prototype).
v1 failed on L2 (core covers 28 % of the CCD): probabilistic Hough at 15 votes fired ~700 spurious segments on the web
of tracks and consumed the muons' pixels; walks stopped at earlier tubes; rms <= 1.5 rejected diffuse muons.
v2:
  seeds   skeleton of the core, standard Hough per tile (tile x tile px, half overlap; votes summed over 3 rho bins so a
          wobbling 1-px skeleton still collects), peaks above seed_votes; lines < min_angle from horizontal dropped
  refit   charge-weighted PCA of core pixels within 2 px of the seed line inside its tile
  walk    along the full image both ways; a step is occupied if a core pixel is within +-1 px (pixels of earlier tubes
          count as occupied: crossings do not stop the walk); stop after max_gap empty steps
  accept  L >= min_len, occupancy >= occ_min, median over 20-px windows of the transverse rms <= rms_max, and at most
          50 % of the line already inside earlier tubes (duplicate)
  tube    transverse sigma per 20-px window, robust linear fit sigma(t) (diffusion grows linearly with depth),
          half-width clip(k_half sigma(t), 1.5, half_max); pixels already in earlier tubes are not re-taken"""
import numpy as np
from scipy import ndimage as nd
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks


from scipy.special import erf


def _pix_gauss(d, s):
    """fraction of a 1-D Gaussian (sigma s, centred on the axis) falling in the pixel at distance d"""
    k = 1 / (np.sqrt(2) * s)
    return 0.5 * (erf((d + 0.5) * k) - erf((d - 0.5) * k))


def _harmonic_fill(F, hole, n_iter=300):
    """Laplace inpainting: values of F inside `hole` replaced by the harmonic interpolation of the values around it"""
    G = F.copy(); G[hole] = 0.0
    if not hole.any(): return G
    ys, xs = np.nonzero(hole)
    y0, y1, x0, x1 = max(ys.min() - 1, 0), min(ys.max() + 2, F.shape[0]), max(xs.min() - 1, 0), min(xs.max() + 2, F.shape[1])
    g = G[y0:y1, x0:x1]; h = hole[y0:y1, x0:x1]
    P = np.pad(g, 1, mode="edge")
    for _ in range(n_iter):
        avg = 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])
        P[1:-1, 1:-1][h] = avg[h]
    G[y0:y1, x0:x1] = P[1:-1, 1:-1]
    return G


def _dir_fill(r, tube, dd, half, u, n_dir=16, min_cross=25.0, exit_pad=0.6):
    """other-track charge in the tube from straight lines across it: for each tube pixel and direction phi (at least
    min_cross deg from the muon axis u), the residual r where the line leaves the tube on both sides (v1, v2);
    o = max over phi of sqrt(v1 v2) (a crossing track is high on both exits along its own direction)"""
    H, W = r.shape
    ys, xs = np.nonzero(tube)
    d0 = dd[ys, xs]; h = half[ys, xs] + exit_pad
    n = np.array([-u[1], u[0]])
    best = np.zeros(ys.size)
    for ph in np.radians(np.linspace(min_cross, 180 - min_cross, n_dir)):
        e = np.cos(ph) * u + np.sin(ph) * n; sn = np.sin(ph)
        v = []
        for sgn in (1, -1):
            s_ = (sgn * h - d0) / (sgn * sn) if sgn > 0 else (d0 + h) / sn     # distance to the exit on that side
            s_ = (h - d0) / sn if sgn > 0 else (h + d0) / sn
            py = np.clip(np.round(ys + sgn * s_ * e[0]).astype(int), 0, H - 1); px = np.clip(np.round(xs + sgn * s_ * e[1]).astype(int), 0, W - 1)
            v.append(r[py, px])
        best = np.maximum(best, np.sqrt(np.clip(v[0], 0, None) * np.clip(v[1], 0, None)))
    out = np.zeros_like(r); out[ys, xs] = best
    return out


_ORACLE = None

def em_share(Rb, tt, dd, inb, tk, sig, half, t0, n_iter=5, med=31, ring=3.0, noise=None, gap=0.0, k_noise=0.0, mean_scale=False, dirfill=False, n_dir=16, min_cross=25.0, u=None, off=(0, 0), dbg=None):
    """muon share of the tube pixels' charge. Model in the band |d| <= half + ring around the axis:
      muon   m = A(t) x pixel-Gaussian(d; sigma(t)), A = running median (med steps) of the muon's step charge
      other  o = (R - m) outside the tube, harmonically interpolated into the tube (crossing / adjacent tracks)
      share  q = R m / (m + o); A refitted from q; n_iter iterations.
    Returns (tube mask, q on the tube pixels)"""
    band = inb & ~tk & (np.abs(dd) <= half + ring)
    tube = band & (np.abs(dd) <= half)
    ti = np.floor(tt - (t0 - 2)).astype(int); ti = np.clip(ti, 0, None)
    nb = int(ti[tube].max()) + 1
    P = np.where(band, _pix_gauss(dd, np.maximum(sig, 0.5)), 0.0)
    Rp = np.where(band, Rb, 0.0)
    fin = np.bincount(ti[tube], P[tube], nb)                                   # expected fraction in the tube per step
    S = np.bincount(ti[tube], np.clip(Rb[tube], 0, None), nb)
    A = nd.median_filter(S / np.maximum(fin, 0.3), size=med, mode="nearest")
    w = np.ones(tube.sum())
    # other-track charge is read from the ring beyond half + gap (the first pixel next to the tube still holds the
    # muon's own tail) and only above k_noise x the pixel noise
    hole = band & (np.abs(dd) < half + gap)
    thr = 0.0 if noise is None else k_noise * noise
    for it in range(n_iter):
        m = A[np.clip(ti, 0, nb - 1)] * P
        r = np.clip(np.where(band, Rp - m - thr, 0.0), 0, None)
        o = np.clip(_harmonic_fill(r, hole), 0, None)
        if dirfill and u is not None:
            o = np.maximum(o, _dir_fill(np.where(hole, 0.0, r), tube, dd, half, u, n_dir, min_cross))
        if _ORACLE is not None:
            sl = (slice(off[0], off[0] + Rb.shape[0]), slice(off[1], off[1] + Rb.shape[1]))
            if "o" in _ORACLE["mode"]: o = np.clip(_ORACLE["O"][sl], 0, None)
            if "m" in _ORACLE["mode"]: m = np.clip(_ORACLE["M"][sl], 0, None)
        mt, ot = m[tube], o[tube]
        w = mt / (mt + ot + 1e-6)
        q = Rb[tube] * w
        Smu = np.bincount(ti[tube], np.clip(q, 0, None), nb)
        A = nd.median_filter(Smu / np.maximum(fin, 0.3), size=med, mode="nearest")
        if mean_scale:
            # step charge is Landau-skewed: scale the running median to the mean over steps without other-track charge
            Ot = np.bincount(ti[tube], ot, nb); clean = (Ot < 0.05 * np.maximum(Smu, 1e-9)) & (fin > 0.5)
            if clean.sum() >= 10:
                A = A * np.clip(np.sum(Smu[clean] / fin[clean]) / np.sum(A[clean]), 1.0, 1.5)
        if dbg is not None: dbg.append(dict(it=it, muon=float(q.sum()), tube=float(Rb[tube].sum()), other=float(ot.sum())))
    return tube, Rb[tube] * w, mt


def _seeds(skel, tile, seed_votes, min_angle, peaks_per_tile):
    H, W = skel.shape
    th = np.deg2rad(np.arange(-90, 90, 0.5))
    th = th[np.abs(np.rad2deg(th)) <= 90 - min_angle]       # normal angle; |theta| -> 90 is a horizontal line
    out = []
    step = tile // 2
    for y0 in range(0, max(H - step, 1), step):
        for x0 in range(0, max(W - step, 1), step):
            sub = skel[y0:y0 + tile, x0:x0 + tile]
            if sub.sum() < seed_votes: continue
            h, a, d = hough_line(sub, theta=th)
            h3 = nd.uniform_filter1d(h.astype(float), 3, axis=0) * 3
            for v, t, r in zip(*hough_line_peaks(h3, a, d, min_distance=4, min_angle=4, threshold=seed_votes, num_peaks=peaks_per_tile)):
                out.append((float(v), x0, y0, float(t), float(r)))
    out.sort(key=lambda s: -s[0])
    return out


def find_muons(Q, valid, bg, min_len=40, core_e=60.0, max_gap=6, occ_min=0.85, rms_max=2.2, min_angle=10.0, smooth=1.0,
               tile=160, seed_votes=40, peaks_per_tile=8, k_half=2.5, half_max=5.0, ridge=2.0, share="em", em_iter=5, em_kw={}, cont_low2=0.15, cont_low=0.08, out_max=0.35, cap_k=1.5, E_PER_E=3.75e-3, dbg=None):
    H, W = Q.shape
    R = np.where(valid, Q - bg.mean, 0.0)
    Rs = nd.gaussian_filter(R, smooth) if smooth else R
    nz = np.sqrt(np.maximum(bg.lam, 0)) / (2 * np.sqrt(np.pi) * smooth) if smooth else np.sqrt(np.maximum(bg.lam, 0))
    core = (Rs > np.maximum(core_e, 5 * nz)) & valid
    NZ = np.sqrt(np.maximum(bg.lam, 0) + np.asarray(bg.s) ** 2) * np.ones_like(R)      # pixel noise [e-]
    skel = skeletonize(core)
    seeds = _seeds(skel, tile, seed_votes, min_angle, peaks_per_tile)
    if dbg is not None: dbg.append(("seeds", seeds, float(core.mean()), float(skel.mean())))
    taken = np.zeros((H, W), bool); mq = np.zeros((H, W)); out = []

    def _at(a, y, x):
        iy, ix = int(round(y)), int(round(x))
        return a[iy, ix] if 0 <= iy < H and 0 <= ix < W else 0

    def occ(c, u, t):
        # occupied: a core pixel within +-1 px of the axis that is a transverse ridge (its smoothed charge >= ridge x
        # the mean at +-3.5 px), or a pixel of an earlier tube (crossing)
        y, x = c + t * u
        if not (0 <= y < H - 0.5 and 0 <= x < W - 0.5): return None
        n = (-u[1], u[0])
        side = min(_at(Rs, y + 3.5 * n[0], x + 3.5 * n[1]), _at(Rs, y - 3.5 * n[0], x - 3.5 * n[1]))
        for k in (-1, 0, 1):
            yy, xx = y + k * n[0], x + k * n[1]
            if _at(taken, yy, xx): return True
            if _at(core, yy, xx) and _at(Rs, yy, xx) >= ridge * max(side, 0): return True
        return False

    for v, x0, y0, th, rho in seeds:
        # seed line in tile coords: x cos th + y sin th = rho; direction (dy, dx) = (cos th, -sin th)
        u = np.array([np.cos(th), -np.sin(th)])
        sub = core[y0:y0 + tile, x0:x0 + tile]
        sy, sx = np.nonzero(sub)
        dd = np.abs(sx * np.cos(th) + sy * np.sin(th) - rho); m = dd <= 2
        if m.sum() < 0.5 * min_len: continue
        sy, sx = sy[m] + y0, sx[m] + x0
        if taken[sy, sx].mean() > 0.5: continue
        w = np.clip(Rs[sy, sx], 0, None) + 1e-6
        c = np.array([np.average(sy, weights=w), np.average(sx, weights=w)])
        C = np.cov(np.vstack([sy - c[0], sx - c[1]]), aweights=w); u2 = np.linalg.eigh(C)[1][:, 1]
        if abs(u2 @ u) > np.cos(np.radians(5)): u = u2 * np.sign(u2 @ u)    # keep the refit only if it agrees with the seed
        if np.degrees(np.arcsin(min(abs(u[0]), 1))) < min_angle: continue
        ends = []
        for sgn in (-1, 1):
            t, last, gap = 0.0, 0.0, 0
            while True:
                t += sgn; o = occ(c, u, t)
                if o is None: break
                if o: last, gap = t, 0
                else:
                    gap += 1
                    if gap > max_gap: break
            ends.append(last)
        t0, t1 = ends; L = t1 - t0
        if L < min_len:
            if dbg is not None: dbg.append(dict(why="short", L=L, c=c, u=u))
            continue
        hits = np.array([bool(occ(c, u, t)) for t in np.arange(t0, t1 + 1)]); occf = hits.mean()
        if occf < occ_min:
            if dbg is not None: dbg.append(dict(why="occ", L=L, occ=occf, c=c, u=u))
            continue
        n = np.array([-u[1], u[0]])
        # continuity along the axis (blob chains in dense X-ray fields dip between the blobs; a muon does not): smoothed charge on the ridge per 1-px step
        _ts = np.arange(t0, t1 + 1); _ax = []
        for _t in _ts:
            _y, _x = c + _t * u
            _ax.append(max(_at(Rs, _y + k * n[0], _x + k * n[1]) for k in (-1, 0, 1)))
        _ax = np.array(_ax); _med = max(np.median(_ax), 1e-9)
        cont = dict(f_low=float(np.mean(_ax < 0.3 * _med)), cv=float(np.std(_ax) / max(np.mean(_ax), 1e-9)), med_ax=float(_med),
                    f_low2=float(np.mean(_ax < 0.5 * _med)))
        if cont["f_low2"] > cont_low2 or cont["f_low"] > cont_low:
            if dbg is not None: dbg.append(dict(why="cont", L=L, c=c, u=u, **cont))
            continue
        pts = c + np.outer([t0 - 2, t1 + 2], u)
        by0, by1 = int(max(pts[:, 0].min() - 8, 0)), int(min(pts[:, 0].max() + 9, H))
        bx0, bx1 = int(max(pts[:, 1].min() - 8, 0)), int(min(pts[:, 1].max() + 9, W))
        gy, gx = np.mgrid[by0:by1, bx0:bx1]
        tt = (gy - c[0]) * u[0] + (gx - c[1]) * u[1]; dd = (gy - c[0]) * n[0] + (gx - c[1]) * n[1]
        inb = (tt >= t0 - 2) & (tt <= t1 + 2) & valid[by0:by1, bx0:bx1]
        tk = taken[by0:by1, bx0:bx1]
        if (inb & (np.abs(dd) <= 1) & tk).sum() > 0.5 * (inb & (np.abs(dd) <= 1)).sum():
            if dbg is not None: dbg.append(dict(why="dup", L=L, c=c, u=u))
            continue
        rr = np.clip(R[by0:by1, bx0:bx1], 0, None)
        # transverse sigma per 20-px window from pixels within 4 px of the axis, not in earlier tubes
        wc, ws = [], []
        for a in np.arange(t0 - 2, t1 + 3, 20):
            win = inb & ~tk & (tt >= a) & (tt < a + 20) & (np.abs(dd) <= 4)
            if win.sum() < 10 or rr[win].sum() <= 0: continue
            wc.append(a + 10); ws.append(np.sqrt(np.average(dd[win] ** 2, weights=rr[win])))
        if not ws: continue
        ws = np.array(ws); wc = np.array(wc)
        rms_med = float(np.median(ws))
        if rms_med > rms_max:
            if dbg is not None: dbg.append(dict(why="rms", L=L, rms=rms_med, c=c, u=u))
            continue
        # robust linear sigma(t): fit, drop windows > 1.5x the fit (crossings), refit
        if ws.size >= 3:
            p = np.polyfit(wc, ws, 1); ok = ws <= 1.5 * np.polyval(p, wc) + 0.2
            p = np.polyfit(wc[ok], ws[ok], 1) if ok.sum() >= 2 else np.array([0, np.median(ws)])
        else:
            p = np.array([0.0, np.median(ws)])
        sig = np.clip(np.polyval(p, tt), 0.3, None)
        half = np.clip(k_half * sig, 1.5, half_max)
        tube = inb & ~tk & (np.abs(dd) <= half)
        ty, tx = np.nonzero(tube)
        q = R[ty + by0, tx + bx0].copy()
        if share == "em":
            tube, q, mtube = em_share(R[by0:by1, bx0:bx1], tt, dd, inb, tk, sig, half, t0, n_iter=em_iter, noise=NZ[by0:by1, bx0:bx1], u=u, off=(by0, bx0), **em_kw)
            ty, tx = np.nonzero(tube)
        elif share:
            # crossings: per 1-px step along the axis, S = tube charge, outer = fraction at |d| > 1.5 sigma; the muon's
            # expected step charge A = running median of S. At steps that are not muon-like (outer > out_max or
            # S > 3 A) the muon gets min(R, cap x A x Gauss(d; sigma)); the rest stays in the image
            ti = np.floor(tt[ty, tx] - (t0 - 2)).astype(int); nb = ti.max() + 1
            qp = np.clip(q, 0, None); sg = sig[ty, tx]; d_ = dd[ty, tx]
            S = np.bincount(ti, qp, nb); O = np.bincount(ti, qp * (np.abs(d_) > 1.5 * sg), nb)
            A = nd.median_filter(S, size=31, mode="nearest")
            bad = (O > out_max * np.maximum(S, 1e-9)) | (S > 3 * A)
            if bad.any():
                bad = nd.binary_dilation(bad, iterations=1)
                pdf = np.exp(-0.5 * (d_ / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
                cap = cap_k * A[ti] * pdf
                m = bad[ti]; q[m] = np.minimum(q[m], cap[m])
        if share != "em": mtube = None
        ty = ty + by0; tx = tx + bx0
        taken[ty, tx] = True
        mq[ty, tx] = q
        if dbg is not None: dbg.append(dict(why="ok", L=L, occ=occf, rms=rms_med, c=c, u=u))
        out.append(dict(E=float(q.sum() * E_PER_E), y=float(np.average(ty, weights=np.clip(q, 1e-9, None))),
                        x=float(np.average(tx, weights=np.clip(q, 1e-9, None))), sxy=rms_med, cls="muon", regime="muon-finder",
                        blend=0.0, ped=0.0, n_comp=1, length=float(L), occ=float(occf), _pix=(ty, tx, q), _m=mtube, _axis=(c, u, t0, t1), cont=cont))
    return taken, out, mq


def install(AD, **kw):
    if getattr(AD, "_muons3", False): return
    _oac = AD.adaptive_cluster

    def adaptive_cluster(Q, valid, bg=None, *a, **k):
        if bg is None: bg = AD.AM.background_map(Q, valid)
        tk, mu, mq = find_muons(Q, valid, bg, **kw)
        # muon charge removed from the image (crossing tracks keep their remainder); pixels fully given to a muon masked
        full = tk & (np.abs(Q - bg.mean - mq) < 1e-9)
        r = _oac(Q - mq, valid & ~full, bg, *a, **k)
        cl, info = r if k.get("return_info") else (r, None)
        cl = list(cl) + mu
        if info is not None: info["n_muons"] = len(mu)
        return (cl, info) if k.get("return_info") else cl
    AD._oac_mu3 = _oac; AD.adaptive_cluster = adaptive_cluster; AD._muons3 = True


def uninstall(AD):
    if getattr(AD, "_muons3", False): AD.adaptive_cluster = AD._oac_mu3; AD._muons3 = False
