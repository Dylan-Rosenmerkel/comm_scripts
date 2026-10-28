"""A_GMX on crowded CCDs (study: wip/agmx_crowded, report wip/agmx_crowded/report/muon_report.pdf).

Used by adaptive.adaptive_cluster(engine='gmx', crowded='auto'|'on'|'off'). 'auto' (the default, or $AGMX_CROWDED)
switches crowded mode on when the image's core fill (fraction of pixels whose smoothed charge is above
max(60 e-, 5 noise)) is >= CROWD_FILL: L2 surface images 0.27-0.31, Am-241 0.43-0.47, Fe-55 production CCDs
0.015-0.02. On other images nothing here runs and A_GMX is unchanged. In crowded mode:

  regional_background  the per-row background fit fails at high dark charge (integer pedestal ambiguous): rows with
                       too many pixels below the zero-electron level get the good rows' pedestal; per-cell
                       (10 x 64 px) truncated-Poisson dark charge overrides the row model where it differs
                       significantly. Electron-counting images only (read noise < 0.45 e-); otherwise a no-op.
  find_streaks         serial-register streaks (charge in one row, no vertical spread, >= 8 px) are taken out of the
                       image and reported as 'artifact-row' clusters (regime 'serial'), so they do not join tracks.
  find_hints           straight tracks: skeleton of the core -> Hough per 160-px tile -> ridge walk along the line
                       (crossings allowed) -> accepted if long, occupied, narrow and continuous along the axis (a
                       chain of X-ray blobs in a dense field is not). Only the AXIS is used: a hint to GMX.
  assign_hints         after GMX's merge graph and island arbitration: every cloud on a hinted axis (and not an
                       elongated cloud of a crossing track) goes into one cluster. Width, footprint and the charge
                       shared with crossing tracks are A_GMX's own.
  split_crossings      a cluster that holds two straight lines crossing as an X is split into the two tracks.
  link_tracks          collinear straight fragments of one track joined (after clustering).
"""
import numpy as np
from scipy import ndimage as nd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import agent_methods as AM
import methods2 as M2

CROWD_FILL = 0.10          # core fill at which crowded='auto' switches crowded mode on
CORE_E = 60.0              # core threshold [e-] (and 5 x the noise of the smoothed dark charge)


# ------------------------------------------------------------------------------------------------ trigger
def _smoothed(Q, valid, bg, smooth=1.0):
    R = np.where(valid, Q - bg.mean, 0.0)
    Rs = nd.gaussian_filter(R, smooth)
    nz = np.sqrt(np.maximum(bg.lam, 0)) / (2 * np.sqrt(np.pi) * smooth)
    return R, Rs, nz


def core_fill(Q, valid, bg, core_e=CORE_E, smooth=1.0):
    """fraction of the valid pixels in the core (smoothed charge above max(core_e, 5 noise)): the crowding measure"""
    _, Rs, nz = _smoothed(Q, valid, bg, smooth)
    core = (Rs > np.maximum(core_e, 5 * nz)) & valid
    return float(core[valid].mean()) if valid.any() else 0.0


# ------------------------------------------------------------------------------------------------ background
def regional_background(Q, valid, bg, rb=10, cb=64, s_max=0.45, f_neg=0.005, d_min=0.3, n_sig=5.0, k_alpha=0.1):
    """targeted override of AM.background_map where it is demonstrably wrong; returns (bg, report).
      0. electron-counting images only (median read noise < s_max), else bg unchanged
      1. pedestal per row: rows with > f_neg of their pixels below o - 0.5 - 3 s are bad; their whole-electron pedestal
         is replaced by the good rows' consensus (the fractional part is kept); without >= 10 good rows, the lowest
         populated level of the bad rows (10th percentile)
      2. dark charge per rb x cb cell: truncated-Poisson MLE below the (1 - k_alpha) quantile (track halos excluded);
         a cell overrides the row model where |difference| > max(d_min, n_sig x its error); bad rows take the cell
         estimate itself. Overrides are interpolated between cell centres."""
    H, W = Q.shape
    s = float(np.median(bg.s[valid]))
    rep = dict(applied=False, s=s, bad_rows=0, cells_changed=0, ncells=0)
    if s >= s_max:
        return bg, rep
    o = bg.o[:, 0].copy(); frac = o - np.round(o)
    sr = np.sqrt(np.maximum(bg.s2[:, 0], 1e-6))
    fneg = np.array([np.mean(Q[r][valid[r]] - o[r] < -0.5 - 3 * sr[r]) if valid[r].any() else 0.0 for r in range(H)])
    bad = fneg > f_neg
    rep["bad_rows"] = int(bad.sum())
    if bad.any():
        good = ~bad & valid.any(1)
        if good.sum() >= 10:
            I0 = int(np.median(np.round(o[good] - frac[good])))
        else:
            lows = []
            for r in np.nonzero(bad)[0]:
                v = np.round(Q[r][valid[r]] - frac[r]).astype(int)
                lv, cnt = np.unique(v, return_counts=True); cum = np.cumsum(cnt) / cnt.sum()
                lows.append(lv[np.searchsorted(cum, 1e-3)])
            I0 = int(np.percentile(lows, 10))
        o[bad] = frac[bad] + I0; rep["I0"] = I0
    lam_row = np.maximum(bg.mean - o[:, None], AM.LAM_FLOOR)     # the row mean o + lam is kept
    n = np.round(Q - o[:, None]).astype(int)
    rr = np.arange(0, H, rb); cc = np.arange(0, W, cb)
    dgrid = np.zeros((rr.size, cc.size)); lgrid = np.full((rr.size, cc.size), np.nan)
    for a, r0 in enumerate(rr):
        for b, c0 in enumerate(cc):
            vm = valid[r0:r0 + rb, c0:c0 + cb]
            v = n[r0:r0 + rb, c0:c0 + cb][vm]
            if v.size < 150: continue
            rep["ncells"] += 1
            lam = max(float(np.median(v)), 0.05)
            for _ in range(5):
                K = max(3, int(stats.poisson.isf(k_alpha, lam)))
                if np.mean((v >= 0) & (v <= K)) < 0.25 * (1 - k_alpha): lam = np.nan; break
                lam = AM.truncated_poisson_mle(v, K)
            if not np.isfinite(lam): continue
            p = stats.norm.sf(0.5 / max(s, 1e-3)); lam = max((lam - p) / max(1 - 2 * p, 1e-3), AM.LAM_FLOOR)   # miscount
            lgrid[a, b] = lam
            rows_ok = ~bad[r0:r0 + rb]
            if not rows_ok.any(): continue
            ref = float(np.mean(lam_row[r0:r0 + rb][rows_ok][:, c0:c0 + cb][vm[rows_ok]]))
            nfit = max(int(np.sum((v >= 0) & (v <= K))), 1)
            if abs(lam - ref) > max(d_min, n_sig * np.sqrt(max(lam, 0.01) / nfit)):
                dgrid[a, b] = lam - ref; rep["cells_changed"] += 1
    if not (rep["cells_changed"] or rep["bad_rows"]):
        return bg, rep
    centres = (np.minimum(rr + rb / 2, H - 1), np.minimum(cc + cb / 2, W - 1))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], 1)
    lam_map = lam_row + RegularGridInterpolator(centres, dgrid, bounds_error=False, fill_value=None)(pts).reshape(H, W)
    if bad.any():
        if np.isnan(lgrid).any():
            idx = nd.distance_transform_edt(np.isnan(lgrid), return_distances=False, return_indices=True); lgrid = lgrid[tuple(idx)]
        lam_map[bad] = RegularGridInterpolator(centres, lgrid, bounds_error=False, fill_value=None)(pts).reshape(H, W)[bad]
    lam_map = np.maximum(lam_map, AM.LAM_FLOOR)
    rep["applied"] = True
    return AM.Background(np.repeat(o[:, None], W, 1), lam_map, bg.s2, bg.mask, dict(bg.rows, regional=rep)), rep


# ------------------------------------------------------------------------------------------------ serial streaks
def find_streaks(Q, valid, bg, run_min=8, f=0.25, thr_alpha=0.01, gap_join=10):
    """serial-register streaks: significant pixels whose up / down neighbours hold < f of their charge, joined
    horizontally (1-px gaps bridged), >= run_min px; pieces in one row <= gap_join px apart are one streak.
    Returns (mask, clusters with cls 'artifact-row', regime 'serial', '_pix')"""
    R = np.where(valid, Q - bg.mean, 0.0)
    sig = (Q > M2.link_threshold(bg, thr_alpha)) & valid
    up = np.zeros_like(R); dn = np.zeros_like(R); up[:-1] = R[1:]; dn[1:] = R[:-1]
    single = sig & (np.clip(up, 0, None) < f * R) & (np.clip(dn, 0, None) < f * R)
    h = nd.binary_closing(single, structure=np.ones((1, 3), bool)) & sig
    lab, _ = nd.label(h, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool))
    mask = np.zeros_like(single); out = []
    for k, sl in enumerate(nd.find_objects(lab), 1):
        if sl is None or sl[1].stop - sl[1].start < run_min: continue
        ys, xs = np.nonzero(lab[sl] == k); ys = ys + sl[0].start; xs = xs + sl[1].start
        mask[ys, xs] = True
        out.append((ys, xs, R[ys, xs]))
    out.sort(key=lambda p: (int(p[0][0]), int(p[1].min())))
    merged = []
    for ys, xs, q in out:
        if merged and int(merged[-1][0][0]) == int(ys[0]) and xs.min() - merged[-1][1].max() <= gap_join:
            p = merged[-1]; merged[-1] = (np.r_[p[0], ys], np.r_[p[1], xs], np.r_[p[2], q])
        else:
            merged.append((ys, xs, q))
    cl = [dict(E=float(q.sum() * AM.E_PER_E), y=float(ys.mean()), x=float(np.average(xs, weights=np.clip(q, 1e-9, None))),
               sxy=0.0, cls="artifact-row", regime="serial", blend=0.0, ped=0.0, n_comp=1, _pix=(ys, xs, q)) for ys, xs, q in merged]
    return mask, cl


# ------------------------------------------------------------------------------------------------ straight-track hints
def _seeds(skel, tile, seed_votes, min_angle, peaks_per_tile):
    from skimage.transform import hough_line, hough_line_peaks
    H, W = skel.shape
    th = np.deg2rad(np.arange(-90, 90, 0.5))
    th = th[np.abs(np.rad2deg(th)) <= 90 - min_angle]            # normal angle; |theta| -> 90 is a horizontal line
    out = []; step = tile // 2
    for y0 in range(0, max(H - step, 1), step):
        for x0 in range(0, max(W - step, 1), step):
            sub = skel[y0:y0 + tile, x0:x0 + tile]
            if sub.sum() < seed_votes: continue
            h, a, d = hough_line(sub, theta=th)
            h3 = nd.uniform_filter1d(h.astype(float), 3, axis=0) * 3   # 3 rho bins: a wobbling 1-px skeleton still votes
            for v, t, r in zip(*hough_line_peaks(h3, a, d, min_distance=4, min_angle=4, threshold=seed_votes, num_peaks=peaks_per_tile)):
                out.append((float(v), x0, y0, float(t), float(r)))
    out.sort(key=lambda s: -s[0])
    return out


def find_hints(Q, valid, bg, min_len=40, core_e=CORE_E, max_gap=6, occ_min=0.85, rms_max=2.2, min_angle=10.0, smooth=1.0,
               tile=160, seed_votes=40, peaks_per_tile=8, k_half=2.5, half_max=5.0, ridge=2.0, cont_low2=0.15, cont_low=0.08):
    """straight tracks -> [(centre, direction, t0, t1, width)] (axis in (y, x), extent along it, transverse rms).
    Seeds: Hough per tile on the skeleton of the core. Walk: a step is occupied if a core pixel within +-1 px of the
    axis is a transverse ridge (>= ridge x the smaller charge at +-3.5 px) or lies in an earlier track's band
    (crossings). Accepted: length >= min_len, occupancy >= occ_min, median transverse rms <= rms_max, axis
    continuity (fraction of steps below 0.5 / 0.3 x the median axis charge <= cont_low2 / cont_low), not a
    duplicate. Lines < min_angle from horizontal are left to find_streaks. The band (k_half x sigma(t)) of an accepted
    track only marks its pixels for the duplicate test and the crossings of later walks."""
    from skimage.morphology import skeletonize
    H, W = Q.shape
    R, Rs, nz = _smoothed(Q, valid, bg, smooth)
    core = (Rs > np.maximum(core_e, 5 * nz)) & valid
    seeds = _seeds(skeletonize(core), tile, seed_votes, min_angle, peaks_per_tile)
    taken = np.zeros((H, W), bool); out = []

    def at(a, y, x):
        iy, ix = int(round(y)), int(round(x))
        return a[iy, ix] if 0 <= iy < H and 0 <= ix < W else 0

    def occ(c, u, t):
        y, x = c + t * u
        if not (0 <= y < H - 0.5 and 0 <= x < W - 0.5): return None
        n = (-u[1], u[0])
        side = min(at(Rs, y + 3.5 * n[0], x + 3.5 * n[1]), at(Rs, y - 3.5 * n[0], x - 3.5 * n[1]))
        for k in (-1, 0, 1):
            yy, xx = y + k * n[0], x + k * n[1]
            if at(taken, yy, xx): return True
            if at(core, yy, xx) and at(Rs, yy, xx) >= ridge * max(side, 0): return True
        return False

    for _, x0, y0, th, rho in seeds:
        u = np.array([np.cos(th), -np.sin(th)])              # seed line x cos th + y sin th = rho (tile coordinates)
        sy, sx = np.nonzero(core[y0:y0 + tile, x0:x0 + tile])
        m = np.abs(sx * np.cos(th) + sy * np.sin(th) - rho) <= 2
        if m.sum() < 0.5 * min_len: continue
        sy, sx = sy[m] + y0, sx[m] + x0
        if taken[sy, sx].mean() > 0.5: continue
        w = np.clip(Rs[sy, sx], 0, None) + 1e-6
        c = np.array([np.average(sy, weights=w), np.average(sx, weights=w)])
        u2 = np.linalg.eigh(np.cov(np.vstack([sy - c[0], sx - c[1]]), aweights=w))[1][:, 1]
        if abs(u2 @ u) > np.cos(np.radians(5)): u = u2 * np.sign(u2 @ u)
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
        if L < min_len: continue
        if np.mean([bool(occ(c, u, t)) for t in np.arange(t0, t1 + 1)]) < occ_min: continue
        n = np.array([-u[1], u[0]])
        ax = np.array([max(at(Rs, *(c + t * u + k * n)) for k in (-1, 0, 1)) for t in np.arange(t0, t1 + 1)])
        med = max(np.median(ax), 1e-9)
        if np.mean(ax < 0.5 * med) > cont_low2 or np.mean(ax < 0.3 * med) > cont_low: continue
        pts = c + np.outer([t0 - 2, t1 + 2], u)
        by0, by1 = int(max(pts[:, 0].min() - 8, 0)), int(min(pts[:, 0].max() + 9, H))
        bx0, bx1 = int(max(pts[:, 1].min() - 8, 0)), int(min(pts[:, 1].max() + 9, W))
        gy, gx = np.mgrid[by0:by1, bx0:bx1]
        tt = (gy - c[0]) * u[0] + (gx - c[1]) * u[1]; dd = (gy - c[0]) * n[0] + (gx - c[1]) * n[1]
        inb = (tt >= t0 - 2) & (tt <= t1 + 2) & valid[by0:by1, bx0:bx1]
        tk = taken[by0:by1, bx0:bx1]
        if (inb & (np.abs(dd) <= 1) & tk).sum() > 0.5 * (inb & (np.abs(dd) <= 1)).sum(): continue
        rr = np.clip(R[by0:by1, bx0:bx1], 0, None)
        wc, ws = [], []
        for a in np.arange(t0 - 2, t1 + 3, 20):
            win = inb & ~tk & (tt >= a) & (tt < a + 20) & (np.abs(dd) <= 4)
            if win.sum() < 10 or rr[win].sum() <= 0: continue
            wc.append(a + 10); ws.append(np.sqrt(np.average(dd[win] ** 2, weights=rr[win])))
        if not ws: continue
        ws = np.array(ws); wc = np.array(wc); rms = float(np.median(ws))
        if rms > rms_max: continue
        if ws.size >= 3:
            p = np.polyfit(wc, ws, 1); ok = ws <= 1.5 * np.polyval(p, wc) + 0.2
            p = np.polyfit(wc[ok], ws[ok], 1) if ok.sum() >= 2 else np.array([0, np.median(ws)])
        else:
            p = np.array([0.0, np.median(ws)])
        half = np.clip(k_half * np.clip(np.polyval(p, tt), 0.3, None), 1.5, half_max)
        ty, tx = np.nonzero(inb & ~tk & (np.abs(dd) <= half))
        taken[ty + by0, tx + bx0] = True
        out.append((c, u, float(t0), float(t1), rms))
    return out


def assign_hints(lab, mix, hints, d_min=1.5, d_fac=1.0, pad=3.0, elong=1.6, ang_max=30.0):
    """clouds on a hinted axis -> one cluster per hint. On the axis: centre within max(d_min, d_fac x width) of it and
    within its extent (+- pad). A cloud elongated (axis ratio >= elong) at >= ang_max deg to the hint is part of a
    crossing track and keeps its cluster. Returns (lab, hinted: bool per cloud)"""
    lab = np.asarray(lab).copy(); hinted = np.zeros(len(mix), bool)
    if not hints or len(mix) == 0:
        return lab, hinted
    a, c, b = mix.S[:, 0], mix.S[:, 1], mix.S[:, 2]
    tr = 0.5 * (a + c); dd = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    ratio = np.sqrt((tr + dd) / np.maximum(tr - dd, 1e-6))
    th = 0.5 * np.arctan2(2 * b, a - c); vy, vx = np.cos(th), np.sin(th)          # major axis (y, x)
    nxt = int(lab.max()) + 1
    for c0, u, t0, t1, w in sorted(hints, key=lambda h: -(h[3] - h[2])):
        n = np.array([-u[1], u[0]]); d = mix.mu - c0
        t = d @ u; dn = d @ n
        on = (np.abs(dn) <= max(d_min, d_fac * w)) & (t >= t0 - pad) & (t <= t1 + pad)
        crossing = (ratio >= elong) & (np.abs(vy * u[0] + vx * u[1]) < np.cos(np.radians(ang_max)))
        take = on & ~crossing & ~hinted
        if take.sum() < 2: continue
        lab[take] = nxt; nxt += 1; hinted |= take
    _, lab = np.unique(lab, return_inverse=True)
    return lab, hinted


# ------------------------------------------------------------------------------------------------ crossing split
def _lines(mu, A, sig, tol=2.0, min_len=20.0, min_n=3, max_lines=4, seed=0):
    rng = np.random.default_rng(seed)
    L = []; rem = np.arange(len(mu))
    for _ in range(max_lines):
        if rem.size < min_n: break
        best = None
        P = rem if rem.size <= 40 else rng.choice(rem, 40, replace=False)
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                p, q = mu[P[i]], mu[P[j]]; u = q - p; nn = np.linalg.norm(u)
                if nn < 4: continue
                u = u / nn
                d = np.abs((mu[rem, 0] - p[0]) * u[1] - (mu[rem, 1] - p[1]) * u[0])
                inl = rem[d < tol + 0.5 * sig[rem]]
                if inl.size < min_n: continue
                sc = A[inl].sum()
                if best is None or sc > best[0]: best = (sc, inl)
        if best is None: break
        inl = best[1]
        cc = np.average(mu[inl], 0, weights=A[inl]); u = np.linalg.eigh(np.cov((mu[inl] - cc).T, aweights=A[inl]))[1][:, 1]
        t = (mu[inl] - cc) @ u
        if t.max() - t.min() >= min_len: L.append((cc, u, t.min(), t.max()))
        rem = np.setdiff1d(rem, inl)
    return L


def _cross(l1, l2, ang_min=20.0, side=8.0):
    (c1, u1, a1, b1), (c2, u2, a2, b2) = l1, l2
    if abs(u1 @ u2) > np.cos(np.radians(ang_min)): return False
    try: s, t = np.linalg.solve(np.array([u1, -u2]).T, c2 - c1)
    except np.linalg.LinAlgError: return False
    return (s - a1 >= side) and (b1 - s >= side) and (t - a2 >= side) and (b2 - t >= side)


def split_crossings(mix, lab, min_clouds=4):
    """split clusters whose clouds hold two straight lines (RANSAC on cloud centres, >= 20 px, >= 3 clouds) crossing
    as an X (>= 20 deg, both continuing >= 8 px past the crossing); clouds go to the nearest line. (lab, n_split)"""
    lab = np.asarray(lab).copy(); nxt = lab.max() + 1 if lab.size else 0
    sig = np.sqrt(np.maximum(0.5 * (mix.S[:, 0] + mix.S[:, 1]), 1e-6)); nsplit = 0
    for L in np.unique(lab):
        k = np.nonzero(lab == L)[0]
        if k.size < min_clouds: continue
        lines = _lines(mix.mu[k], mix.A[k], sig[k])
        if len(lines) < 2: continue
        keep = sorted({x for i in range(len(lines)) for j in range(i + 1, len(lines)) if _cross(lines[i], lines[j]) for x in (i, j)})
        if not keep: continue
        d = np.stack([np.abs((mix.mu[k, 0] - lines[i][0][0]) * lines[i][1][1] - (mix.mu[k, 1] - lines[i][0][1]) * lines[i][1][0]) for i in keep], 1)
        own = d.argmin(1)
        for m in range(1, len(keep)):
            lab[k[own == m]] = nxt; nxt += 1
        nsplit += len(keep) - 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab, nsplit


# ------------------------------------------------------------------------------------------------ linking
def _pca(ys, xs, w):
    w = np.clip(w, 0, None) + 1e-9; cy, cx = np.average(ys, weights=w), np.average(xs, weights=w)
    ev, V = np.linalg.eigh(np.cov(np.vstack([ys - cy, xs - cx]), aweights=w)); u = V[:, 1]
    t = (ys - cy) * u[0] + (xs - cx) * u[1]
    return np.array([cy, cx]), u, t.min(), t.max(), np.sqrt(max(ev[0], 0))


def link_tracks(cl, ang=4.0, gap=15.0, gap_long=25.0, off=2.5, min_len=12.0, min_part=5.0, max_w_over_l=0.12, max_w_long=0.2):
    """join straight fragments of one track (clusters with '_pix'; union-find over all qualifying pairs). With the
    longer piece L (>= min_len px) and the shorter S (>= min_part px): S's endpoints within max(off, 1.5 + 0.01 len(L))
    px of L's axis; axes parallel within ang deg when S >= min_len; nearest endpoints < gap px apart (< gap_long when
    both >= 30 px and the offset < 2 px); both straight (width / length < max_w_over_l, < max_w_long from 50 px).
    Returns (clusters, number of links)"""
    T = []
    for k, c in enumerate(cl):
        if "_pix" not in c or c.get("regime") == "serial": continue
        ys, xs, q = c["_pix"]
        if ys.size < 5: continue
        cen, u, t0, t1, wt = _pca(ys.astype(float), xs.astype(float), q)
        Lk = t1 - t0
        if Lk >= min_part and wt / max(Lk, 1e-9) < (max_w_long if Lk >= 50 else max_w_over_l):
            T.append((k, cen + t0 * u, cen + t1 * u, u, Lk))
    par = list(range(len(cl)))

    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    if not T: return cl, 0
    tree = cKDTree(np.array([p for (_, a, b, _, _) in T for p in (a, b)]))
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
        ys = np.concatenate([cl[k]["_pix"][0] for k in g]); xs = np.concatenate([cl[k]["_pix"][1] for k in g])
        q = np.concatenate([cl[k]["_pix"][2] for k in g])
        c = dict(cl[max(g, key=lambda k: cl[k]["E"])]); c["E"] = float(sum(cl[k]["E"] for k in g)); c["_pix"] = (ys, xs, q)
        w = np.clip(q, 0, None) + 1e-9; c["y"], c["x"] = float(np.average(ys, weights=w)), float(np.average(xs, weights=w))
        c["n_linked"] = len(g); c["cls"] = "muon"
        out.append(c)
    return out, nl
