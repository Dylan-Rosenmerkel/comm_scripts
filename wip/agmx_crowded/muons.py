"""Straight-track (muon) extraction before A_GMX (scratch prototype).
core = pixels > max(core_e, 5 sqrt(lam)) e- above the background; probabilistic Hough lines on the core (>= min_len px);
each candidate, longest first: refit the axis (charge-weighted PCA of core pixels within 3 px of the line), walk outward
along it while the core is occupied (gaps <= max_gap px), accept if length >= min_len, occupancy >= occ_min and the core
pixels' perpendicular rms <= rms_max; tube half-width per 20-px window = clip(3 x transverse charge sigma, 1.5, 7) px.
Accepted tubes are removed from the core before the next candidate and from the image before A_GMX."""
import numpy as np
from skimage.transform import probabilistic_hough_line

def find_muons(Q, valid, bg, min_len=40, core_e=60.0, max_gap=6, occ_min=0.75, rms_max=1.5, min_angle=10.0, smooth=1.0, E_PER_E=3.75e-3, dbg=None):
    H, W = Q.shape
    R = np.where(valid, Q - bg.mean, 0.0)
    # core on the lightly smoothed image (dotted / knotted tracks become continuous lines); the noise of the smoothed
    # dark charge is ~ sqrt(lam) / (2 sqrt(pi) smooth)
    from scipy import ndimage as _nd
    Rs = _nd.gaussian_filter(R, smooth) if smooth else R
    nz = np.sqrt(np.maximum(bg.lam, 0)) / (2 * np.sqrt(np.pi) * smooth) if smooth else np.sqrt(np.maximum(bg.lam, 0))
    core = (Rs > np.maximum(core_e, 5 * nz)) & valid
    segs = probabilistic_hough_line(core, threshold=15, line_length=min_len, line_gap=4, rng=1)
    # near-horizontal lines are serial-register streaks (found separately), not muons
    segs = [s for s in segs if np.degrees(np.arctan2(abs(s[1][1] - s[0][1]), abs(s[1][0] - s[0][0]))) >= min_angle]
    segs = sorted(segs, key=lambda s: -np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
    taken = np.zeros((H, W), bool); out = []
    yy_all, xx_all = np.nonzero(core)
    if dbg is not None: dbg.append(("segs", segs))
    for (x0, y0), (x1, y1) in segs:
        R_ = lambda r, **e: dbg.append(dict(seg=((x0, y0), (x1, y1)), why=r, **e)) if dbg is not None else None
        if taken[y0, x0] and taken[y1, x1]: R_("taken"); continue
        p0 = np.array([y0, x0], float); u = np.array([y1 - y0, x1 - x0], float); u /= np.linalg.norm(u)
        # refit on core pixels near the segment (not yet taken)
        ylo, yhi = max(min(y0, y1) - 4, 0), min(max(y0, y1) + 5, H); xlo, xhi = max(min(x0, x1) - 4, 0), min(max(x0, x1) + 5, W)
        sy, sx = np.nonzero(core[ylo:yhi, xlo:xhi] & ~taken[ylo:yhi, xlo:xhi]); sy = sy + ylo; sx = sx + xlo
        d = np.abs((sy - p0[0]) * u[1] - (sx - p0[1]) * u[0]); m = d < 3
        if m.sum() < min_len: R_("few"); continue
        w = np.clip(Rs[sy[m], sx[m]], 0, None) + 1e-6; c = np.array([np.average(sy[m], weights=w), np.average(sx[m], weights=w)])
        C = np.cov(np.vstack([sy[m] - c[0], sx[m] - c[1]]), aweights=w); u = np.linalg.eigh(C)[1][:, 1]
        # walk outward along the axis from the centre while the core is occupied
        def occ(t):
            y, x = c + t * u; iy, ix = int(round(y)), int(round(x))
            if not (0 <= iy < H and 0 <= ix < W): return None
            n = np.array([-u[1], u[0]])
            return any(core[int(round(y + k * n[0])) % H, int(round(x + k * n[1])) % W] and not taken[int(round(y + k * n[0])) % H, int(round(x + k * n[1])) % W] for k in (-1, 0, 1))
        ends = []
        for sgn in (-1, 1):
            t, last, gap = 0.0, 0.0, 0
            while True:
                t += sgn; o = occ(t)
                if o is None: break
                if o: last, gap = t, 0
                else:
                    gap += 1
                    if gap > max_gap: break
            ends.append(last)
        t0, t1 = ends; L = t1 - t0
        if L < min_len: R_("short", L=L); continue
        ts = np.arange(t0, t1 + 1); hits = np.array([bool(occ(t)) for t in ts]); occf = hits.mean()
        if occf < occ_min: R_("occ", occ=occf, L=L); continue
        # tube: pixels within the band, charge-based transverse width per 20-px window
        n = np.array([-u[1], u[0]])
        pts = c + np.outer(np.r_[t0 - 2, t1 + 2], u)
        by0, by1 = int(max(pts[:, 0].min() - 9, 0)), int(min(pts[:, 0].max() + 10, H)); bx0, bx1 = int(max(pts[:, 1].min() - 9, 0)), int(min(pts[:, 1].max() + 10, W))
        gy, gx = np.mgrid[by0:by1, bx0:bx1]; tt = (gy - c[0]) * u[0] + (gx - c[1]) * u[1]; dd = (gy - c[0]) * n[0] + (gx - c[1]) * n[1]
        inb = (tt >= t0 - 2) & (tt <= t1 + 2) & (np.abs(dd) <= 8) & valid[by0:by1, bx0:bx1] & ~taken[by0:by1, bx0:bx1]
        rr = np.clip(R[by0:by1, bx0:bx1], 0, None)
        cm = inb & core[by0:by1, bx0:bx1] & (np.abs(dd) <= 3)
        rms = np.sqrt(np.average(dd[cm] ** 2, weights=rr[cm] + 1e-9)) if cm.any() else 99
        if rms > rms_max: R_("rms", rms=rms, L=L); continue
        R_("ok", L=L)
        half = np.zeros_like(tt)
        for a in np.arange(t0 - 2, t1 + 3, 20):
            win = inb & (tt >= a) & (tt < a + 20) & (np.abs(dd) <= 6)
            sg = np.sqrt(np.average(dd[win] ** 2, weights=rr[win] + 1e-9)) if win.any() and rr[win].sum() > 0 else 1.0
            half[(tt >= a) & (tt < a + 20)] = np.clip(3 * sg, 1.5, 7)
        tube = inb & (np.abs(dd) <= half)
        ty, tx = np.nonzero(tube); ty = ty + by0; tx = tx + bx0
        taken[ty, tx] = True
        q = R[ty, tx]
        out.append(dict(E=float(q.sum() * E_PER_E), y=float(np.average(ty, weights=np.clip(q, 1e-9, None))), x=float(np.average(tx, weights=np.clip(q, 1e-9, None))),
                        sxy=float(rms), cls="muon", regime="muon-finder", blend=0.0, ped=0.0, n_comp=1, length=float(L), occ=float(occf), _pix=(ty, tx, q)))
    return taken, out

def install(AD):
    if getattr(AD, "_muons", False): return
    _oac = AD.adaptive_cluster
    def adaptive_cluster(Q, valid, bg=None, *a, **k):
        if bg is None: bg = AD.AM.background_map(Q, valid)
        tk, mu = find_muons(Q, valid, bg)
        r = _oac(Q, valid & ~tk, bg, *a, **k)
        cl, info = r if k.get("return_info") else (r, None)
        cl = list(cl) + mu
        if info is not None: info["n_muons"] = len(mu)
        return (cl, info) if k.get("return_info") else cl
    AD._oac_mu = _oac; AD.adaptive_cluster = adaptive_cluster; AD._muons = True

def uninstall(AD):
    if getattr(AD, "_muons", False): AD.adaptive_cluster = AD._oac_mu; AD._muons = False
