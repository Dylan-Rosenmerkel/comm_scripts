# Copied from comm_scripts/clustering_tests/cluster_methods.py by utils/sync_to_waders.py (imports made package-relative).
# Edit the comm_scripts file and re-run the sync, not this copy.
"""Clustering / energy-estimation methods for skipper-CCD X-ray clusters, compared with WADERs.

All methods work on the same image: WADERs' CALIBRATED image (electrons with panaSKImg's gain), converted
pixel by pixel with the coherent g(n) module curve (electrons = A^-1(e_panaSKImg * g_fit)). The active
region comes from the image geometry.

Methods (each returns a list of clusters with energy [keV], sigma_xy [pix], centroid):
  wader    : WADERs ClusterFinder: pixels > n_pix*sigma, 8-connected, kept if a pixel > n_seed*sigma
             (sigma = pedestal noise; WADERs default n_pix = 10, n_seed = 30)
  topo     : ATLAS-style topological clustering in electrons: seed >= S e-, grow through 8-neighbours
             >= N e-, then add `rings` crowns of neighbours with NO threshold (P = 0); the expected dark
             current of the added pixels (lambda per pixel) is subtracted
  aperture : aperture photometry: around the wader cluster's charge-weighted centroid, sum all pixels within
             R = max(r_min, k * sigma_xy + 1) pixels, minus lambda * (number of pixels in the aperture)
  gfit     : fit of the diffusion model to a box around the cluster: b + N * (pixel-integrated 2D Gaussian),
             weighted least squares (variance = noise^2 + b + model); energy = N
"""
import os
import sys
import numpy as np
from scipy import ndimage, optimize, special
try:
    from .pair_energy import pair_energy_eV
except ImportError:                                 # clustering_tests used without src/ on the path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from .pair_energy import pair_energy_eV

E_PER_E = pair_energy_eV() * 1e-3   # keV per electron: epsilon(T) (src/pair_energy.py), 3.7486 eV at the default 130 K


def lam_at(lam, ys):
    """Dark charge [e-/pix] at rows ys: lam is a number or an array with one value per row."""
    lam = np.asarray(lam, float)
    return lam[np.asarray(ys, int)] if lam.ndim else np.full(np.size(ys), float(lam))
EIGHT = np.ones((3, 3), int)


def moments(Q, ys, xs):
    q = np.clip(Q[ys, xs], 0, None)
    s = q.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan
    my, mx = (q * ys).sum() / s, (q * xs).sum() / s
    vy, vx = (q * (ys - my) ** 2).sum() / s, (q * (xs - mx) ** 2).sum() / s
    return my, mx, float(np.sqrt(0.5 * (vx + vy)))


def wader(Q, sigma, n_pix=10.0, n_seed=30.0, emin=None, emax=None):
    """WADERs-like clusters. Returns list of dicts (E, sxy, y, x, labels-index) and the label image."""
    lab, n = ndimage.label(Q > n_pix * sigma, structure=EIGHT)
    out = []
    objs = ndimage.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        sub = lab[sl] == k
        ys, xs = np.nonzero(sub)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        q = Q[ys, xs]
        if q.max() <= n_seed * sigma:
            continue
        E = q.sum() * E_PER_E
        if emin is not None and not (emin < E < emax):
            continue
        my, mx, sxy = moments(Q, ys, xs)
        out.append(dict(E=E, sxy=sxy, y=my, x=mx, npix=len(q), label=k))
    return out, lab


def topo(Q, lam, S=10.0, N=2.0, rings=1, emin=None, emax=None, exclude=None):
    """Topological clustering in electrons with `rings` threshold-free crowns; DC of crown pixels subtracted."""
    core, n = ndimage.label(Q >= N, structure=EIGHT)
    seeds = ndimage.maximum(Q, core, index=np.arange(1, n + 1)) >= S
    keep = np.zeros(n + 1, bool); keep[1:] = seeds
    grown = keep[core]
    out = []
    lab2, n2 = ndimage.label(grown, structure=EIGHT)
    for k, sl in enumerate(ndimage.find_objects(lab2), start=1):
        pad = rings + 1
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, Q.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, Q.shape[1])
        m = lab2[y0:y1, x0:x1] == k
        mm = ndimage.binary_dilation(m, structure=EIGHT, iterations=rings) if rings > 0 else m
        if exclude is not None:
            mm &= ~exclude[y0:y1, x0:x1]
        ys, xs = np.nonzero(mm)
        ys, xs = ys + y0, xs + x0
        crown = mm & ~m
        cy_, cx_ = np.nonzero(crown)
        E = (Q[ys, xs].sum() - lam_at(lam, cy_ + y0).sum()) * E_PER_E
        if emin is not None and not (emin < E < emax):
            continue
        cy, cx = np.nonzero(m)
        my, mx, sxy = moments(Q, cy + y0, cx + x0)
        out.append(dict(E=E, sxy=sxy, y=my, x=mx, npix=len(ys)))
    return out


def aperture(Q, lam, clusters, k=3.0, r_min=2.5, valid=None):
    """Circular aperture around each cluster centroid; returns list aligned with clusters (E or nan)."""
    out = []
    H, W = Q.shape
    for c in clusters:
        R = max(r_min, k * c["sxy"] + 1.0)
        y0, y1 = int(max(np.floor(c["y"] - R), 0)), int(min(np.ceil(c["y"] + R) + 1, H))
        x0, x1 = int(max(np.floor(c["x"] - R), 0)), int(min(np.ceil(c["x"] + R) + 1, W))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (yy - c["y"]) ** 2 + (xx - c["x"]) ** 2 <= R ** 2
        if valid is not None and not valid[y0:y1, x0:x1][inside].all():
            out.append(np.nan); continue
        q = Q[y0:y1, x0:x1][inside]
        out.append((q.sum() - lam_at(lam, yy[inside]).sum()) * E_PER_E)
    return out


def _gauss_pix(yy, xx, my, mx, s):
    """Pixel-integrated 2D Gaussian (unit total)."""
    r2 = np.sqrt(2) * s
    fx = 0.5 * (special.erf((xx + 0.5 - mx) / r2) - special.erf((xx - 0.5 - mx) / r2))
    fy = 0.5 * (special.erf((yy + 0.5 - my) / r2) - special.erf((yy - 0.5 - my) / r2))
    return fx * fy


def gfit(Q, lam, noise, clusters, half=6, valid=None):
    """Fit b + N * pixel-integrated Gaussian to a (2 half + 1)^2 box around each cluster; returns (E list, sigma list)."""
    E, S = [], []
    H, W = Q.shape
    for c in clusters:
        cy, cx = int(round(c["y"])), int(round(c["x"]))
        y0, y1, x0, x1 = cy - half, cy + half + 1, cx - half, cx + half + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W or (valid is not None and not valid[y0:y1, x0:x1].all()):
            E.append(np.nan); S.append(np.nan); continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        q = Q[y0:y1, x0:x1].ravel()
        yy, xx = yy.ravel().astype(float), xx.ravel().astype(float)
        lb = float(np.mean(lam_at(lam, yy.astype(int))))
        N0 = max(q.sum() - lb * q.size, 10)

        def res(p):
            N, my, mx, s, b = p
            m = b + N * _gauss_pix(yy, xx, my, mx, s)
            return (q - m) / np.sqrt(noise ** 2 + np.clip(m, 0, None) + 0.05)
        try:
            r = optimize.least_squares(res, [N0, c["y"], c["x"], max(c["sxy"], 0.3), lb],
                                       bounds=([0, c["y"] - 2, c["x"] - 2, 0.15, -1], [1e5, c["y"] + 2, c["x"] + 2, 4.0, 5]))
            E.append(r.x[0] * E_PER_E); S.append(r.x[3])
        except Exception:
            E.append(np.nan); S.append(np.nan)
    return E, S


def inject(Q, rng, n_e, sxy, positions):
    """Add simulated X-ray clusters (n_e electrons each, Gaussian spread sxy pixels) at positions (y, x)."""
    Q = Q.copy()
    counts = []
    for (y, x) in positions:
        ne = rng.poisson(n_e)
        counts.append(ne)
        dy, dx = rng.normal(0, sxy, ne), rng.normal(0, sxy, ne)
        iy, ix = np.round(y + dy).astype(int), np.round(x + dx).astype(int)
        ok = (iy >= 0) & (iy < Q.shape[0]) & (ix >= 0) & (ix < Q.shape[1])
        np.add.at(Q, (iy[ok], ix[ok]), 1.0)
    return Q, counts
