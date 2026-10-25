"""Adaptive clustering: choose the method per region of the image from what the region looks like.

Lessons of the 2026-10-23/24 study (docs/CONTEXT_2026-10-24.md):
  - sparse regions (isolated X-rays, tracks, few-electron events): MHC is best (unbiased energies, close X-rays
    resolved, tracks joined, few-electron events without fakes, labels)
  - crowded regions (most pixels hold charge, clusters touch): WADERs-like hysteresis on the image is best; the
    matched filter + disks of MFD+G / MHC mis-measure widths and energies when neighbours overlap

Steps
  1. background map (AM.background_map; simple fallback when no row can be fitted)
  2. global check: if the fitted background spread is >= excess_ratio x the read noise (and >= 0.5 e-), the
     background is not dark current + read noise (e.g. PD06/DM12 Am-241 images: bg spread 1.5-2.9 e- vs 0.37 e-
     read noise; MFD+G-type energies there come out 5-10 % high) -> regime 'unmodelled' for the whole image:
     hysteresis with sigma = read noise (WADERs-like, never below the Poisson-tail threshold of the dark charge)
  3. event occupancy map: fraction of active pixels above max(1.5 e-, 4 x read noise, Poisson tail of the dark
     charge at 1e-3) in a box of `box` px - pixels that hold EVENT charge, not dark current (high-dark-current
     Fe-55 CCDs have 50 % of pixels > 1.5 e- but only ~5 % above the tail; MHC is right for them)
  4. regime: 'crowded' where occupancy >= occ_thr, else 'sparse' (both dilated by `margin` px for the clustering
     so that clusters on the boundary are complete; each cluster is kept by the regime of its centre)
  5. sparse -> MHC on the sparse part; crowded -> hysteresis clustering in electrons on the crowded part:
     pixel > 10 sigma_crowd, seed > 30 sigma_crowd, energy = sum of the core pixels (as WADERs).
     sigma_crowd = the WADERs / panaSKImg pedestal sigma of the CCD when given (on a crowded CCD it is inflated
     by the charge, ~8 e- on the Am-241 EXT4 CCD, which is what makes WADERs work there), else 8 e-
     (pixel > 80 e-, seed > 240 e-). Tuned on the crowded Am-241 CCD (runs_29img EXT4): a threshold scan
     10-160 e- peaks at 80 e- (Np L-line significance); thresholds relative to the local median and
     energies over extra rings were both worse (adaptive_crowded_scan_am241.py)
  6. every cluster gets 'regime', 'occ' and (from MHC) 'cls'; crowded clusters get cls 'x-ray' / 'x-ray diffuse' /
     'big deposit' / 'extended' from their energy and size
"""
import os, sys
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_methods as AM
import cluster_methods as CM
import mhc as MH
EIGHT = np.ones((3, 3), bool)


def simple_bg(Q, valid, sw):
    act = Q[valid]; quiet = act[act < 3 * sw + 0.5]
    p0 = np.mean(np.abs(quiet) < max(0.5, 2 * sw)) if quiet.size else 1.0
    lam = float(-np.log(max(p0, 1e-6))) if sw < 0.5 else 0.0
    mask = ndimage.binary_dilation(Q > 3 * sw + 1, iterations=2) & valid
    f = lambda v: np.full(Q.shape, v, float)
    return AM.Background(f(0.0), f(lam), f(sw * sw), mask, dict(lam_smooth=np.full(Q.shape[0], lam)))


def occupancy(Q, valid, noise, box=31, lam=0.0, bg=None):
    """fraction of active pixels holding EVENT charge in a box: above max(1.5 e-, 4 x read noise, Poisson tail
    (1e-3) of the dark charge). With bg the tail uses the per-pixel dark-charge map (dark charge often grows along
    the rows; the image median would count the dark pixels of the high-dark-charge rows as events)."""
    if bg is not None:
        thr = np.maximum(np.maximum(1.5, 4 * noise), AM.tail_threshold_map(bg, 1e-3) - bg.o)
    else:
        thr = max(1.5, 4 * noise, AM.poisson_tail_k(float(lam), 1e-3)[0] - 0.5)
    num = ndimage.uniform_filter(((Q > thr) & valid).astype(float), box, mode="constant")
    den = ndimage.uniform_filter(valid.astype(float), box, mode="constant")
    return np.where(valid, num / np.maximum(den, 1e-3), 0.0)


def hysteresis(Q, region, sigma, k_pix, n_pix=10.0, n_seed=30.0, seed_e=9.5):
    """WADERs-like clusters inside `region` with thresholds in electrons"""
    tp, ts = max(n_pix * sigma, k_pix - 0.5), max(n_seed * sigma, seed_e)
    lab, n = ndimage.label((Q > tp) & region, structure=EIGHT)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == k; q = Q[sl][m]
        if q.max() <= ts:
            continue
        ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
        my, mx, sxy = CM.moments(Q, ys, xs)
        E = float(q.sum() * AM.E_PER_E)
        ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        cls = "big deposit" if E > 75 else ("extended" if ext >= 10 else ("x-ray diffuse" if sxy >= 1 else "x-ray"))
        out.append(dict(E=E, y=float(my), x=float(mx), sxy=float(sxy), npix=int(m.sum()), cls=cls, qmax=float(q.max()),
                        _pix=(ys, xs, q)))           # pixels and their charge [e-] (sums to E)
    return out


def adaptive_cluster(Q, valid, bg=None, sigma=None, sigma_crowd=None, occ_thr=0.3, box=31, margin=8, excess_ratio=3.0,
                     return_info=False, engine="mhc", return_pixels=False, **mhc_kw):
    """engine='mhc' : unmodelled -> hysteresis; else per region crowded -> hysteresis, sparse -> MHC.
    engine='gmx' : unmodelled -> hysteresis; else GMX on the whole image (GMX shares charge by model responsibility,
                   so it needs no crowded / sparse split; occupancy is still reported per cluster)."""
    if sigma is None:
        act = Q[valid]; z = act[np.abs(act) < 0.5]
        sigma = float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 100 else 1.0
    if bg is None:
        try:
            bg = AM.background_map(Q, valid)
        except Exception:
            bg = simple_bg(Q, valid, sigma)
    noise = float(np.median(bg.s[valid])); lam = float(np.median(bg.lam[valid]))
    k = max(int(AM.poisson_tail_k(lam, 0.10)[0]), 2)
    info = dict(sigma=sigma, noise=noise, lam=lam, sigma_crowd=sigma_crowd)
    if noise >= 0.5 and noise >= excess_ratio * sigma:
        # the background model is not trusted here (its fitted offset can be off by several e-), so cluster the
        # image itself, as WADERs does
        out = [dict(c, regime="unmodelled", occ=np.nan) for c in hysteresis(np.where(valid, Q, 0.0), valid, sigma, k)]
        info.update(regime="unmodelled", crowded_frac=np.nan)
        return (out, info) if return_info else out
    occ = occupancy(Q, valid, sigma, box, lam, bg)
    if engine == "gmx":
        import gmx as GX
        out = []
        for c in GX.general_cluster(Q, valid, bg, return_pixels=return_pixels):
            y, x = int(round(c["y"])), int(round(c["x"]))
            o = float(occ[y, x]) if 0 <= y < Q.shape[0] and 0 <= x < Q.shape[1] else np.nan
            out.append(dict(c, regime="crowded" if o >= occ_thr else "sparse", occ=o))
        info.update(regime="gmx", occ=occ, crowded_frac=float(((occ >= occ_thr) & valid).sum() / max(valid.sum(), 1)))
        return (out, info) if return_info else out
    crowded = (occ >= occ_thr) & valid
    sparse = valid & ~crowded
    out = []
    if sparse.any():
        vs = valid & ~ndimage.binary_erosion(crowded, structure=EIGHT, iterations=margin)
        for c in MH.mhc(np.where(vs, Q, bg.mean), vs, bg, **mhc_kw):
            y, x = int(round(c["y"])), int(round(c["x"]))
            if 0 <= y < Q.shape[0] and 0 <= x < Q.shape[1] and sparse[y, x]:
                out.append(dict(c, regime="sparse", occ=float(occ[y, x])))
    if crowded.any():
        reg = ndimage.binary_dilation(crowded, structure=EIGHT, iterations=margin) & valid
        sc = float(sigma_crowd) if sigma_crowd else 8.0
        for c in hysteresis(np.where(valid, Q - bg.mean, 0.0), reg, sc, k, seed_e=0.0):   # dark charge subtracted
            y, x = int(round(c["y"])), int(round(c["x"]))
            if crowded[y, x]:
                out.append(dict(c, regime="crowded", occ=float(occ[y, x])))
    if return_info:
        info.update(regime="mixed" if crowded.any() and sparse.any() else ("crowded" if crowded.any() else "sparse"),
                    occ=occ, crowded_frac=float(crowded.sum() / max(valid.sum(), 1)))
        return out, info
    return out
