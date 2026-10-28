"""Targeted regional background (scratch v3): the current A_GMX background (per-row pedestal + dark charge, mesh
correction) is kept, and replaced only where it is demonstrably wrong.
  0. only for electron-counting images (median read noise < s_max): elsewhere the input map is returned unchanged
  1. pedestal check per row: with the right pedestal almost no pixel lies below the zero-electron level. Rows with
     more than f_neg of their pixels below o - 0.5 - 3 s are 'bad': their whole-electron pedestal is replaced by the
     consensus of the good rows (the fractional part, the comb phase, is kept). Without good rows: the lowest
     populated level per row, its 10th percentile over the bad rows.
  2. dark charge per cell (rb x cb px, all pixels, no cluster mask): truncated-Poisson MLE with the checked pedestal;
     a cell overrides the row model only where they differ by > max(d_min, n_sig x the cell's statistical error);
     the overrides (differences) are interpolated between cell centres, zero elsewhere.
Returns (bg, report)."""
import numpy as np
from scipy import stats, ndimage
from scipy.interpolate import RegularGridInterpolator
import agent_methods as AM

def regional_v3(Q, valid, bg, rb=10, cb=64, s_max=0.45, f_neg=0.005, d_min=0.3, n_sig=5.0, use_mask=False, min_free=0.2, k_alpha=0.1):
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
    # row-model dark charge, re-expressed with the checked pedestal (the row mean o + lam is kept)
    mean_row = bg.mean
    lam_row = np.maximum(mean_row - o[:, None], AM.LAM_FLOOR)
    n = np.round(Q - o[:, None]).astype(int)
    rr = np.arange(0, H, rb); cc = np.arange(0, W, cb)
    dgrid = np.zeros((rr.size, cc.size)); lgrid = np.full((rr.size, cc.size), np.nan)
    for a, r0 in enumerate(rr):
        for b, c0 in enumerate(cc):
            vm = valid[r0:r0 + rb, c0:c0 + cb]
            fm = vm & ~bg.mask[r0:r0 + rb, c0:c0 + cb]
            # pixels outside the cluster mask when enough of the cell is free (track halos raise lam), else all
            v = n[r0:r0 + rb, c0:c0 + cb][fm if (use_mask and fm.sum() >= max(min_free * vm.sum(), 150)) else vm]
            if v.size < 150: continue
            rep["ncells"] += 1
            lam = max(float(np.median(v)), 0.05)
            for _ in range(5):
                K = max(3, int(stats.poisson.isf(k_alpha, lam)))
                if np.mean((v >= 0) & (v <= K)) < 0.25 * (1 - k_alpha): lam = np.nan; break
                lam = AM.truncated_poisson_mle(v, K)
            if not np.isfinite(lam): continue
            p = stats.norm.sf(0.5 / max(s, 1e-3)); lam = max((lam - p) / max(1 - 2 * p, 1e-3), AM.LAM_FLOOR)   # A2 miscount
            lgrid[a, b] = lam
            rows_ok = ~bad[r0:r0 + rb]
            if not rows_ok.any(): continue                       # bad rows take the cell estimate itself (below)
            ref = float(np.mean(lam_row[r0:r0 + rb][rows_ok][:, c0:c0 + cb][vm[rows_ok]]))
            nfit = max(int(np.sum((v >= 0) & (v <= K))), 1)
            err = np.sqrt(max(lam, 0.01) / nfit)
            if abs(lam - ref) > max(d_min, n_sig * err):
                dgrid[a, b] = lam - ref; rep["cells_changed"] += 1
    if rep["cells_changed"] or rep["bad_rows"]:
        ip = RegularGridInterpolator((np.minimum(rr + rb / 2, H - 1), np.minimum(cc + cb / 2, W - 1)), dgrid, bounds_error=False, fill_value=None)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts = np.stack([yy.ravel(), xx.ravel()], 1)
        lam_map = lam_row + ip(pts).reshape(H, W)
        if bad.any():
            # bad rows: the row fit (o + lam) is not trusted at all -> the cell estimates with the checked pedestal
            if np.isnan(lgrid).any():
                idx = ndimage.distance_transform_edt(np.isnan(lgrid), return_distances=False, return_indices=True); lgrid = lgrid[tuple(idx)]
            ipl = RegularGridInterpolator((np.minimum(rr + rb / 2, H - 1), np.minimum(cc + cb / 2, W - 1)), lgrid, bounds_error=False, fill_value=None)
            lam_map[bad] = ipl(pts).reshape(H, W)[bad]
        lam_map = np.maximum(lam_map, AM.LAM_FLOOR)
        rep["applied"] = True
        return AM.Background(np.repeat(o[:, None], W, 1), lam_map, bg.s2, bg.mask, dict(bg.rows, regional_v3=rep, dgrid=dgrid)), rep
    return bg, rep
