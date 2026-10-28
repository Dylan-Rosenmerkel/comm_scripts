"""Cluster finding / energy estimation for skipper-CCD X-ray clusters, built from methods of other fields.

Pixel model used throughout (checked on the DM09/DM10 module-test images, see README):

    Q[r, c] = o[r] + n[r, c] + eps,   n ~ Poisson(lam[r, c] + signal),   eps ~ N(0, s[r]^2)

  o[r]   per-row offset left by the per-row pedestal subtraction (0 or about -1 e- when the 1-e- peak of a
         high-dark-current row was taken as the pedestal; a fraction of an electron on the noisy images)
  lam    dark charge [e-/pix] (grows along the readout)
  s[r]   effective pixel noise: the skipper read noise (0.14-0.4 e-), or ~2-3 e- on the noisy images
         (DM10 img 0), where the pixel distribution is no longer a comb of integers

Methods (each documented with its formula and source):
  1. background_map        robust lam(row, col), o(row), s(row): masked per-row maximum-likelihood fit of the
                           Poisson x Gaussian comb, smoothed along rows (local-linear or Bayesian Blocks),
                           column-bin correction, iterated with the cluster mask
  2. poisson_tail_clusters per-row Poisson-tail pixel threshold k(row) (FDR/percolation reasoning), 8-connected
                           hysteresis with a seed
  3. mf_detect             matched-filter significance map (SoFiA smooth+clip / PowellSnakes / Fermi TS), seeds
                           = local maxima, geometric disk footprint
  4. cash_fit              Poisson maximum likelihood (Cash 1979; Smith et al. 2010) of lam + E * pixel-integrated
                           Gaussian with lam fixed from the map; Fisher error; deviance goodness of fit;
                           'huang' option = shifted-Poisson approximation of Poisson x Gaussian (Huang et al. 2013)
  5. optimal_photometry    Naylor (1998) profile-weighted photometry
  6. multi_fit             N = 1..3 joint Gaussians with shared background, BIC acceptance (DAOSTORM / PyBDSF
                           style), seeded by h-maxima (morphological reconstruction)
  pipeline                 1 -> 3 (or 2) -> 4 (or 5) -> 6 for flagged clusters

Energies are returned in keV with E_PER_E = epsilon(T) x 1e-3 keV/e- (src/pair_energy.py: 3.7486 eV at the default 130 K,
$COMM_TEMPERATURE_K / run_WADERs --temperature for another temperature), cluster dicts with keys E, y, x, sxy (+ extras).
"""
import os
import sys
import numpy as np
from scipy import ndimage, optimize, special, stats
try:
    from pair_energy import pair_energy_eV
except ImportError:                                 # clustering_tests used without src/ on the path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from pair_energy import pair_energy_eV

E_PER_E = pair_energy_eV() * 1e-3   # keV per electron: epsilon(T) (src/pair_energy.py), 3.7486 eV at the default 130 K
EIGHT = np.ones((3, 3), bool)
SQ2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------------------------------------
# 1. background map
# ----------------------------------------------------------------------------------------------------------
def comb_pdf(x, o, lam, s, kmax=None):
    """Density of Q = o + Poisson(lam) + N(0, s^2):  p(x) = sum_k Pois(k; lam) * phi((x - o - k) / s) / s."""
    if kmax is None:
        kmax = int(lam + 8 * np.sqrt(lam + 1) + 8 * s + 5)
    k = np.arange(kmax + 1)
    pk = stats.poisson.pmf(k, lam)
    z = (np.asarray(x)[..., None] - o - k) / s
    return (pk * np.exp(-0.5 * z * z)).sum(-1) / (s * np.sqrt(2 * np.pi))


def fit_comb(vals, o0=0.0, lam0=0.5, s0=0.2, f_out=1e-3, fit_o=True):
    """Maximum-likelihood fit of (o, lam, s) of the comb_pdf to background pixel values, with a small uniform
    outlier component (fraction f, fitted) to absorb residual cluster halos / hot pixels.
    Binned likelihood on a fine histogram (bin 0.05 e-) so the cost does not depend on the number of pixels.
    Several integer starting offsets are tried because o and lam trade off by whole electrons."""
    vals = np.asarray(vals, float)
    lo, hi = np.percentile(vals, 0.05) - 1.0, min(np.percentile(vals, 99.95) + 1.0, o0 + lam0 + 40)
    vals = vals[(vals > lo) & (vals < hi)]
    if vals.size < 50:
        return dict(o=o0, lam=lam0, s=s0, f=f_out, ok=False, n=vals.size)
    bw = 0.05
    edges = np.arange(lo, hi + bw, bw)
    h, _ = np.histogram(vals, edges)
    xc = 0.5 * (edges[1:] + edges[:-1])
    nz = h > 0
    xc_nz, h_nz = xc[nz], h[nz]
    width = hi - lo

    def nll(p):
        o, lam, ls, lf = p
        s = np.exp(ls); f = 1 / (1 + np.exp(-lf))
        if lam < 0:
            return 1e30
        pdf = (1 - f) * comb_pdf(xc_nz, o, lam, s) + f / width
        return -np.sum(h_nz * np.log(np.maximum(pdf * bw, 1e-300)))

    best = None
    starts = [o0 + d for d in (-1, 0, 1)] if fit_o else [o0]
    for os_ in starts:
        mean = vals.mean()
        lam_s = max(mean - os_, 0.02)
        for ss in (s0, max(np.sqrt(max(vals.var() - lam_s, 0.01)), 0.1)):
            p0 = [os_, lam_s, np.log(max(ss, 0.05)), np.log(f_out / (1 - f_out))]
            if fit_o:
                r = optimize.minimize(nll, p0, method="Nelder-Mead",
                                      options=dict(xatol=1e-4, fatol=1e-3, maxiter=800))
            else:
                r = optimize.minimize(lambda q: nll([os_] + list(q)), p0[1:], method="Nelder-Mead",
                                      options=dict(xatol=1e-4, fatol=1e-3, maxiter=800))
                r.x = np.r_[os_, r.x]
            if best is None or r.fun < best.fun:
                best = r
    o, lam, ls, lf = best.x
    return dict(o=float(o), lam=float(max(lam, 0)), s=float(np.exp(ls)), f=float(1 / (1 + np.exp(-lf))),
                ok=True, n=int(vals.size), nll=float(best.fun))


def smooth_local_linear(y, w, half=15, robust=3):
    """Weighted local-linear smoothing (LOWESS, Cleveland 1979: tricube kernel of half-width `half` rows) of
    a series y with weights w (1/variance), with `robust` robustness iterations (bisquare weights on the
    residuals scaled by 6 MAD), so rows with a wrong integer offset (off by one electron) do not pull the
    trend. Keeps a linear ramp exact."""
    n = y.size
    x = np.arange(n, dtype=float)
    ok = np.isfinite(y) & (w > 0)
    y0 = np.where(ok, y, 0.0)
    rw = np.ones(n)
    for it in range(robust + 1):
        out = np.full(n, np.nan)
        for i in range(n):
            d = np.abs(x - i) / (half + 1)
            k = np.where(d < 1, (1 - d ** 3) ** 3, 0) * w * ok * rw
            if k.sum() <= 0:
                continue
            xm = (k * x).sum() / k.sum(); ym = (k * y0).sum() / k.sum()
            sxx = (k * (x - xm) ** 2).sum()
            b = (k * (x - xm) * (y0 - ym)).sum() / sxx if sxx > 0 else 0.0
            out[i] = ym + b * (i - xm)
        if it == robust:
            break
        r = np.where(ok, y - out, 0.0)
        mad = np.median(np.abs(r[ok])) if ok.any() else 0
        u = r / (6 * max(mad, 1e-6))
        rw = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    return out


def smooth_bblocks(y, err, p0=0.01):
    """Bayesian Blocks (Scargle et al. 2013, 'measures' fitness for points with Gaussian errors) piecewise-
    constant representation of y along the rows; returns the block mean at each row."""
    from astropy.stats import bayesian_blocks
    ok = np.isfinite(y) & np.isfinite(err) & (err > 0)
    t = np.arange(y.size, dtype=float)
    edges = bayesian_blocks(t[ok], y[ok], err[ok], fitness="measures", p0=p0)
    out = np.full(y.size, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = ok & (t >= a) & (t <= b)
        if sel.any():
            w = 1 / err[sel] ** 2
            out[(t >= a) & (t <= b)] = (w * y[sel]).sum() / w.sum()
    return out




# physics fixes of the 2026-10-25 audit (docs/aGMX, section 'Physics audit'); AGMX_PHYS_FIXES=0 restores the earlier behaviour
PHYS_FIX = os.environ.get("AGMX_PHYS_FIXES", "1") != "0"
# lowest dark charge the background model can return [e-/pix]: was 1e-3 (map) / 1e-4 (row fit), which is above the
# dark current of the best skipper CCDs (DAMIC-M ~1e-4 e-/pix/day) and made the few-electron tests conservative there
LAM_FLOOR = 1e-6 if PHYS_FIX else 1e-3


def read_noise(Q, valid):
    """read noise [e-] of an image in electrons: MAD of the pixels in the zero-electron peak (|Q| < 0.5); that
    estimate cannot exceed 1.4826 x 0.25 = 0.37 (a uniform spread), so when it comes out above 0.3 (I1, audit) the
    robust spread of all active pixels is used instead (clusters are a small fraction of the pixels)"""
    act = Q[valid]
    z = act[np.abs(act) < 0.5]
    s0 = float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 100 else 1.0
    if PHYS_FIX and s0 > 0.3:
        s0 = max(s0, float(1.4826 * np.median(np.abs(act - np.median(act)))))
    return s0


class Background:
    """Result of background_map: per-pixel maps (arrays of the image shape)
       o    offset [e-]           lam  dark charge [e-/pix]     s2  Gaussian noise variance [e-^2]
       mean = o + lam             var  = lam + s2
    plus the per-row fits (rows) and the final cluster mask."""

    def __init__(self, o, lam, s2, mask, rows):
        self.o, self.lam, self.s2, self.mask, self.rows = o, lam, s2, mask, rows
        self.mean = o + lam
        self.var = lam + s2
        self.s = np.sqrt(s2)


def _cluster_mask(Q, valid, bg, p_box=3e-5, p_pix=1e-5, dilate=4):
    """Pixels that belong to clusters, with Poisson (not Gaussian) tail probabilities so that the mask does
    not explode where the dark charge is small:
      box test : S = sum over the 3x3 box of (Q - o), P(Poisson(9 lam) + noise >= S) < p_box
      pixel    : P(Poisson(lam) + noise >= Q - o) < p_pix
    (noise folded in as a Gaussian approximation of variance 9 s^2 / s^2, adequate for both regimes);
    then dilated by `dilate` pixels (8-connected) to cover the faint halo."""
    R = np.where(valid, Q - bg.o, 0.0)
    S = ndimage.uniform_filter(R, 3, mode="constant") * 9
    lam9 = 9 * bg.lam
    # continuity-corrected Poisson tail with a Gaussian read-noise smearing: use the normal approximation of
    # the Poisson x Gaussian only through the tail of the Poisson at the noise-shifted count
    k_box = np.floor(S - 3 * np.sqrt(9 * bg.s2) + 0.5)
    k_pix = np.floor(R - 3 * np.sqrt(bg.s2) + 0.5)
    pb = stats.poisson.sf(k_box - 1, lam9)
    pp = stats.poisson.sf(k_pix - 1, bg.lam)
    m = ((pb < p_box) | (pp < p_pix)) & valid
    return ndimage.binary_dilation(m, structure=EIGHT, iterations=dilate)


def truncated_poisson_mle(n, K):
    """MLE of lam from integer counts using only n <= K (robust to the cluster-halo / hot-pixel tail):
    maximise sum_{n_i<=K} [n_i ln lam - lam] - N_K ln F(K; lam), F = Poisson CDF."""
    n = n[(n >= 0) & (n <= K)]
    if n.size == 0:
        return np.nan
    S, N = n.sum(), n.size
    # Poisson CDF via special.pdtr (the regularised incomplete gamma): the same value as stats.poisson.logcdf, but
    # without the distribution-object overhead, which made this the slowest part of background_map (2.2x faster)
    Kf = float(np.floor(K))
    f = lambda l: -(S * np.log(l) - N * l - N * np.log(max(special.pdtr(Kf, l), 1e-300)))
    r = optimize.minimize_scalar(f, bounds=(LAM_FLOOR, max(5 * n.mean() + 1, 1.0)), method="bounded",
                                 options=dict(xatol=1e-6))
    return float(r.x)


def fit_row_fast(v, min_frac=0.003):
    """Fast per-row estimate of (o, lam, s) for the comb Q = o + Poisson(lam) + N(0, s^2):
      phase  z = mean exp(2 pi i Q):  o_frac = arg(z) / 2 pi,  s = sqrt(-ln|z| / (2 pi^2))
             (|z| = exp(-2 pi^2 s^2) is the characteristic function of the noise at frequency 1; the Poisson
             part is integer and drops out)
      levels j = round(Q - o_frac); o_int = the integer offset (mode-12 .. mode) maximising the likelihood
             of all pixels under a Poisson + 1e-3 outlier mixture; n = j - o_int
      lam    = truncated-Poisson MLE on n <= K, K = max(3, Poisson isf(1e-3, lam)) (two passes).
    Only valid when the comb is resolved (s < ~0.45); otherwise returns resolved=False."""
    z = np.mean(np.exp(2j * np.pi * v))
    R = np.abs(z)
    # A1 (audit): for unresolved noise |z| fluctuates around sqrt(pi / 4N); the fixed 0.018 threshold sits below that
    # floor for rows of < ~6000 pixels, so pure Gaussian noise was declared 'resolved' with s ~ 0.4 e-
    r_min = max(np.exp(-2 * np.pi ** 2 * 0.45 ** 2), 4.0 / np.sqrt(max(len(v), 1))) if PHYS_FIX else np.exp(-2 * np.pi ** 2 * 0.45 ** 2)
    if R < r_min:
        return dict(resolved=False)
    o_frac = np.angle(z) / (2 * np.pi)
    s = np.sqrt(max(-np.log(R), 1e-8) / (2 * np.pi ** 2))
    j = np.round(v - o_frac).astype(int)
    lv, cnt = np.unique(j, return_counts=True)
    jmode = lv[np.argmax(cnt)]
    best = None
    # candidate integer offsets: from the mode down to the lowest populated level (<= 12 candidates); each
    # scored by the likelihood of all pixels under (1 - f) Poisson(n; lam) + f u, f = 1e-3, u = 1e-3 (a
    # few outlier pixels, e.g. the negative patches next to saturated tracks, cannot set the offset)
    for oi in range(max(lv.min(), jmode - 12), jmode + 1):
        n = j - oi
        sel = n >= 0
        lam0 = max(np.mean(n[sel & (n <= 3 + 3 * max(jmode - oi, 0))]) if (sel & (n <= 3 + 3 * max(jmode - oi, 0))).any() else 0.5, 0.01)
        K = max(3, int(stats.poisson.isf(1e-3, lam0)))
        lam = truncated_poisson_mle(n, K)
        if not np.isfinite(lam):
            continue
        uu, cc = np.unique(n, return_counts=True)
        p = np.where(uu >= 0, (1 - 1e-3) * stats.poisson.pmf(np.maximum(uu, 0), lam), 0) + 1e-6
        ll = float((cc * np.log(p)).sum())
        if best is None or ll > best[0]:
            best = (ll, oi, lam)
    _, o_int, lam = best
    for _ in range(2):
        K = max(3, int(stats.poisson.isf(1e-3, lam)))
        lam = truncated_poisson_mle(j - o_int, K)
    return dict(resolved=True, o=float(o_int + o_frac), lam=float(lam), s=float(s), n=int(v.size))


def simple_bg(Q, valid, sw=None):
    """fallback background when no row can be fitted (high noise / few skips): offset 0, noise sw (read noise; MAD
    of the quiet pixels if None), dark charge from the fraction of empty pixels (-ln P0, as run_study.load),
    clusters masked above 3 sigma (+2 px). (One copy for gmx, adaptive and the wide tests.)"""
    act = Q[valid]
    if sw is None:
        sw = read_noise(Q, valid)
    quiet = act[act < 3 * sw + 0.5]
    p0 = np.mean(np.abs(quiet) < max(0.5, 2 * sw)) if quiet.size else 1.0
    lam = float(-np.log(max(p0, 1e-6))) if sw < 0.5 else 0.0
    mask = ndimage.binary_dilation(Q > 3 * sw + 1, iterations=2) & valid
    f = lambda v: np.full(Q.shape, v, float)
    return Background(f(0.0), f(lam), f(sw * sw), mask, dict(lam_smooth=np.full(Q.shape[0], lam)))


def background_map(Q, valid, n_iter=2, colbin=128, rowbin=60, smooth="local_linear", half=15, init_mask=None,
                   pool=16):
    """Robust dark-charge map.

    Iteration:  (a) mask = clusters (significance mask _cluster_mask, dilated 4 px)
                    [first pass: Q > median + max(4, 5 MAD) dilated 5 px]
                (b) per row r, unmasked pixels: (o_r, lam_r, s_r) with fit_row_fast when the integer comb
                    is resolved (s < 0.45 e-); rows with broad noise: comb ML fit (fit_comb) on blocks of
                    `pool` rows after removing each row's mean, then o_r = row mean - lam_block
                (c) lam_r smoothed along rows (weighted local-linear, tricube half-width 15 rows, or Bayesian
                    Blocks), s_r^2 smoothed likewise; o_r kept per row (per-row pedestal artefact)
                (d) 2D correction: mean residual (Q - o - lam_smooth) of the unmasked pixels in cells of
                    `rowbin` x `colbin` pixels, minus its row-block mean, bilinearly interpolated, added to lam
                    (dark charge or pedestal drift along the row; additive or multiplicative gradients)
    Masked, mesh-wise, iterated background estimation follows SExtractor (Bertin & Arnouts 1996); the
    comb model is the skipper-CCD Poisson x Gaussian pixel model; Bayesian Blocks: Scargle et al. 2013."""
    H, W = Q.shape
    if init_mask is None:
        q = Q[valid]
        med = np.median(q)
        mad = 1.4826 * np.median(np.abs(q - med))
        m = (Q > med + max(4.0, 5 * mad)) & valid
        mask = ndimage.binary_dilation(m, structure=EIGHT, iterations=5)
    else:
        mask = init_mask
    for it in range(n_iter):
        use = valid & ~mask
        o_r = np.full(H, np.nan); lam_r = np.full(H, np.nan); s_r = np.full(H, np.nan); err = np.full(H, np.nan)
        broad = []
        for r in range(H):
            v = Q[r][use[r]]
            if v.size < 100:
                continue
            f = fit_row_fast(v)
            if f["resolved"]:
                o_r[r], lam_r[r], s_r[r] = f["o"], f["lam"], f["s"]
                err[r] = np.sqrt(max(f["lam"], 0.01) / v.size)
            else:
                broad.append(r)
        # broad-noise rows: pooled comb fits
        broad = np.array(broad, int)
        n_res = int(np.isfinite(lam_r).sum())
        if PHYS_FIX and n_res >= max(10, 0.05 * H):
            # A1 (audit): with the stricter resolve test, rows whose comb is washed out (e.g. a charge sheet across
            # the row) are not 'broad noise': the read noise is a property of the amplifier, so when enough rows of
            # the CCD are resolved these rows take o, lam, s from the nearest resolved rows (rows_ok False below)
            # instead of a pooled fit that would read the event charge as noise
            broad = np.zeros(0, int)
        for b0 in range(0, broad.size, pool):
            rows = broad[b0:b0 + pool]
            vals, means = [], {}
            for r in rows:
                v = Q[r][use[r]]
                v = v[np.abs(v - np.median(v)) < 8 * 1.4826 * np.median(np.abs(v - np.median(v))) + 1]
                means[r] = v.mean(); vals.append(v - v.mean())
            vals = np.concatenate(vals)
            f = fit_comb(vals, o0=-0.5, lam0=0.5, s0=max(vals.std(), 0.5), fit_o=True)
            for r in rows:
                lam_r[r] = f["lam"]; s_r[r] = f["s"]; o_r[r] = means[r] - f["lam"]
                err[r] = np.sqrt((f["lam"] + f["s"] ** 2) / max(vals.size, 1))
        rows_ok = np.isfinite(lam_r)
        idx = np.nonzero(rows_ok)[0]
        if idx.size == 0:
            raise RuntimeError("no row could be fitted")
        if smooth == "bblocks":
            lam_s = smooth_bblocks(lam_r, err)
            s2_s = smooth_bblocks(s_r ** 2, np.where(rows_ok, 0.01, np.nan))
        else:
            lam_s = smooth_local_linear(lam_r, np.where(rows_ok, 1 / np.maximum(err, 1e-4) ** 2, 0), half)
            s2_s = smooth_local_linear(s_r ** 2, np.where(rows_ok, 1.0, 0.0), half)
        near = idx[np.clip(np.searchsorted(idx, np.arange(H)), 0, idx.size - 1)]
        # (c') consistency of the integer offsets: the row mean mu_r = o_r + lam_r is well measured, but at
        # high dark charge the split into an integer offset and lam_r can be off by one electron; re-pick
        # o_r = o_frac + round(mu_r - o_frac - lam_smooth) so that lam_r follows the smooth trend, then re-smooth
        fix = rows_ok & np.isin(np.arange(H), broad, invert=True)
        for _pass in range(3):
            lam_tmp = np.where(np.isfinite(lam_s), lam_s, lam_r[near])
            frac = o_r - np.round(o_r)
            changed = 0
            for r in np.nonzero(fix)[0]:
                v = Q[r][use[r]]
                j = np.round(v - frac[r]).astype(int)
                # candidate integer offsets around the current one; pick the one whose truncated-MLE lam is
                # closest to the smooth trend (the shape likelihood alone cannot fix it at lam >~ 4)
                best = None
                lt = max(lam_tmp[r], 0.01)
                uu, cc = np.unique(j, return_counts=True)
                for d in (-1, 0, 1):
                    oi = int(np.round(o_r[r] - frac[r])) + d
                    n = uu - oi
                    # likelihood of all the row's pixels at the trend lam (outlier mixture as in fit_row_fast)
                    p = np.where(n >= 0, (1 - 1e-3) * stats.poisson.pmf(np.maximum(n, 0), lt), 0) + 1e-6
                    sc = -float((cc * np.log(p)).sum())
                    if best is None or sc < best[0]:
                        K = max(3, int(stats.poisson.isf(1e-3, lt)))
                        best = (sc, oi, truncated_poisson_mle(j - oi, K))
                if best is not None:
                    changed += int(best[1] != int(np.round(o_r[r] - frac[r])))
                    o_r[r] = best[1] + frac[r]; lam_r[r] = best[2]
            if smooth == "bblocks":
                lam_s = smooth_bblocks(lam_r, err)
            else:
                lam_s = smooth_local_linear(lam_r, np.where(rows_ok, 1 / np.maximum(err, 1e-4) ** 2, 0), half)
        o_full = np.where(rows_ok, o_r, o_r[near])
        lam_s = np.where(np.isfinite(lam_s), lam_s, lam_r[near])
        s2_s = np.where(np.isfinite(s2_s), s2_s, s_r[near] ** 2)
        if PHYS_FIX:
            # A2 (audit): rows fitted by rounding to whole electrons (fit_row_fast) count read noise > +0.5 e- as an
            # electron and lose noise < -0.5 e- below zero, so their lam (and the row mean o + lam, hence every
            # energy) is high by ~ p (1 - 2 lam) with p = P(noise > 0.5 e-); the offset o (comb phase) is unbiased.
            # Invert the first-order miscount for those rows; pooled comb-likelihood rows (broad) are not affected.
            fast = rows_ok.copy(); fast[broad] = False
            pmis = stats.norm.sf(0.5 / np.sqrt(np.maximum(s2_s, 1e-6)))
            lam_s = np.where(fast, np.maximum((lam_s - pmis) / np.maximum(1.0 - 2.0 * pmis, 1e-3), LAM_FLOOR), lam_s)
        # broad rows: the per-row mean fixes o + lam, keep it when lam is smoothed
        for r in broad:
            o_full[r] = o_r[r] + lam_r[r] - lam_s[r]
        res = np.where(use, Q - o_full[:, None] - lam_s[:, None], np.nan)
        rb = np.arange(0, H, rowbin); cb = np.arange(0, W, colbin)
        grid = np.zeros((rb.size, cb.size))
        for a, r0 in enumerate(rb):
            for bb, c0 in enumerate(cb):
                blk = res[r0:r0 + rowbin, c0:c0 + colbin]
                fin = np.isfinite(blk)
                grid[a, bb] = np.mean(blk[fin]) if fin.sum() > 200 else np.nan
        # empty cells: column-bin median over row blocks, then 0
        colmed = np.nanmedian(np.where(np.isfinite(grid), grid, np.nan), axis=0) if np.isfinite(grid).any() else np.zeros(cb.size)
        grid = np.where(np.isfinite(grid), grid, np.nan_to_num(colmed)[None, :])
        # the row trend already carries the mean of each row block: remove the row-block mean of the cells
        grid = grid - grid.mean(1, keepdims=True)
        from scipy.interpolate import RegularGridInterpolator
        ip = RegularGridInterpolator((np.minimum(rb + rowbin / 2, H - 1), np.minimum(cb + colbin / 2, W - 1)), grid,
                                     bounds_error=False, fill_value=None)
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dmap = ip(np.stack([rr.ravel(), cc.ravel()], 1)).reshape(H, W)
        dcol = dmap.mean(0)
        lam_map = np.maximum(lam_s[:, None] + dmap, LAM_FLOOR)
        o_map = np.repeat(o_full[:, None], W, 1)
        s2_map = np.repeat(np.maximum(s2_s, 1e-4)[:, None], W, 1)
        bg = Background(o_map, lam_map, s2_map, mask, dict(o=o_r, lam=lam_r, s=s_r, err=err, lam_smooth=lam_s,
                                                            s2_smooth=s2_s, dcol=dcol, broad=broad))
        if it < n_iter - 1:
            mask = _cluster_mask(Q, valid, bg)
    return bg


# ----------------------------------------------------------------------------------------------------------
# 2. per-row Poisson-tail threshold
# ----------------------------------------------------------------------------------------------------------
def poisson_tail_k(lam, alpha):
    """k = min{k : P(Poisson(lam) >= k) <= alpha}  (= scipy.stats.poisson.isf(alpha, lam) + 1).
    Rationale: keep the probability that a pure dark-charge pixel passes the pixel threshold, alpha, far
    below the 8-connected site-percolation threshold p_c = 0.407, so dark pixels cannot percolate into
    clusters; equivalently a per-pixel false-positive rate as in FDR source finding (Hopkins et al. 2002)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    out = np.empty(lam.shape, int)
    for i, l in enumerate(lam.ravel()):
        k = 0
        sf = 1.0                          # P(X >= 0)
        pmf = np.exp(-l)
        while sf > alpha:
            sf -= pmf                     # P(X >= k+1)
            k += 1
            pmf *= l / k
        out.ravel()[i] = k
    return out


def comb_tail_threshold(o, lam, s, alpha):
    """Pixel threshold t with P(o + Poisson(lam) + N(0, s^2) > t) = alpha.
    For integer-like data (s < 0.5) it is o + k - 0.5 with k = poisson_tail_k(lam, alpha) (pixel 'n >= k'
    implemented as Q > o + k - 0.5); for broad noise it solves the continuous tail equation."""
    if s < 0.5:
        return o + poisson_tail_k(lam, alpha)[0] - 0.5
    kk = np.arange(int(lam + 10 * np.sqrt(lam + 1) + 10))
    pk = stats.poisson.pmf(kk, lam)
    f = lambda t: (pk * stats.norm.sf((t - o - kk) / s)).sum() - alpha
    return optimize.brentq(f, o - 10 * s, o + lam + 30 * s + 30)


def _cluster_props(Q, bg, ys, xs, subtract=True):
    q = Q[ys, xs]
    E = (q - bg.mean[ys, xs]).sum() if subtract else q.sum()
    w = np.clip(q - bg.mean[ys, xs], 0, None)
    sw = w.sum()
    if sw <= 0:
        return E, np.nan, np.nan, np.nan
    my, mx = (w * ys).sum() / sw, (w * xs).sum() / sw
    sxy = np.sqrt(0.5 * ((w * (ys - my) ** 2).sum() + (w * (xs - mx) ** 2).sum()) / sw)
    return E, my, mx, sxy


def tail_threshold_map(bg, alpha, colstep=64):
    """Per-pixel threshold map t(r, c) = comb_tail_threshold(o_r, lam(r, c), s_r, alpha), evaluated per row
    and per block of `colstep` columns."""
    H, W = bg.lam.shape
    thr = np.empty((H, W))
    cols = np.arange(0, W, colstep)
    cache = {}
    for r in range(H):
        s = float(np.sqrt(bg.s2[r, 0]))
        tc = []
        for c in cols:
            l = float(bg.lam[r, min(c + colstep // 2, W - 1)])
            key = (round(l, 3), round(s, 2), round(float(bg.o[r, 0]), 3)) if s >= 0.5 else (round(l, 3),)
            if key not in cache:
                cache[key] = comb_tail_threshold(0.0 if s < 0.5 else bg.o[r, 0], l, s, alpha) - (0.0 if s < 0.5 else bg.o[r, 0])
            tc.append(bg.o[r, 0] + cache[key])
        thr[r] = np.repeat(tc, colstep)[:W]
    return thr


def poisson_tail_clusters(Q, valid, bg, alpha=0.05, seed=9.5, subtract=True):
    """Hysteresis clustering with a per-row Poisson-tail pixel threshold.
    pixel mask: Q > t(row, col) = comb_tail_threshold(o_r, lam(r, c), s_r, alpha) (computed per row and
    column bin); 8-connected components; a component is kept if one pixel has Q - mean > seed and lies
    above the Poisson tail at 1e-6.
    Energy: sum over the component of (Q - o - lam) [subtract=True: every kept pixel still contains its
    dark charge on top of the signal] or the plain sum (subtract=False, as WADERs)."""
    thr = tail_threshold_map(bg, alpha)
    m = (Q > thr) & valid
    lab, n = ndimage.label(m, structure=EIGHT)
    if n == 0:
        return [], thr
    # seed: Q - mean > seed AND above the Poisson tail at alpha_seed = 1e-6 (at lam ~ 7 e-/pix a fixed
    # 'Q - o > 9.5 e-' seed is passed by 17 % of the dark pixels)
    seed_thr = np.maximum(bg.mean + seed, tail_threshold_map(bg, 1e-6))
    mx = ndimage.maximum(Q - seed_thr, lab, index=np.arange(1, n + 1))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if mx[k - 1] <= 0:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        E, my, mxx, sxy = _cluster_props(Q, bg, ys, xs, subtract)
        out.append(dict(E=E * E_PER_E, y=my, x=mxx, sxy=sxy, npix=len(ys)))
    return out, thr


# ----------------------------------------------------------------------------------------------------------
# pixel-integrated Gaussian and derivatives (used by 3-6)
# ----------------------------------------------------------------------------------------------------------
def _f1d(u, mu, s):
    """1D pixel integral of a unit Gaussian over [u-0.5, u+0.5] and its derivatives d/dmu, d/ds."""
    ap = (u + 0.5 - mu) / (SQ2 * s)
    am = (u - 0.5 - mu) / (SQ2 * s)
    ep, em = np.exp(-ap * ap), np.exp(-am * am)
    f = 0.5 * (special.erf(ap) - special.erf(am))
    dmu = (em - ep) / (np.sqrt(2 * np.pi) * s)
    ds = (am * em - ap * ep) / (np.sqrt(np.pi) * s)
    return f, dmu, ds


def gauss_pix(yv, xv, y0, x0, s, derivs=False):
    """P[i, j] = integral of the unit 2D Gaussian (centre y0, x0, width s) over pixel (yv[i], xv[j]);
    separable: P = fy (outer) fx.  With derivs: (P, dP/dy0, dP/dx0, dP/ds)."""
    fy, dyy, dsy = _f1d(yv, y0, s)
    fx, dxx, dsx = _f1d(xv, x0, s)
    P = np.outer(fy, fx)
    if not derivs:
        return P
    return P, np.outer(dyy, fx), np.outer(fy, dxx), np.outer(dsy, fx) + np.outer(fy, dsx)


def _kernel(s):
    r = int(np.ceil(4 * s + 1))
    u = np.arange(-r, r + 1)
    return gauss_pix(u, u, 0.0, 0.0, s)


# ----------------------------------------------------------------------------------------------------------
# 3. matched-filter detection + geometric footprint
# ----------------------------------------------------------------------------------------------------------
def mf_significance(Q, valid, bg, scales=(0.5, 1.0, 1.5)):
    """Matched-filter significance (the Gaussian limit of the Poisson likelihood-ratio / 'TS' map used by
    PowellSnakes and Fermi, and of SoFiA's smooth+clip):
        T_k = sum_i w_i (Q_i - mean_i) / sqrt(sum_i w_i^2 var_i),   w = pixel-integrated Gaussian of width s_k
    T = max_k T_k; also returns the index of the best scale."""
    R = np.where(valid, Q - bg.mean, 0.0)
    V = np.where(valid, bg.var, 0.0)
    best = np.full(Q.shape, -np.inf); bk = np.zeros(Q.shape, int)
    for i, s in enumerate(scales):
        w = _kernel(s)
        num = ndimage.convolve(R, w, mode="constant")
        den = np.sqrt(np.maximum(ndimage.convolve(V, w * w, mode="constant"), 1e-12))
        T = num / den
        up = T > best
        best[up] = T[up]; bk[up] = i
    best[~valid] = 0
    return best, bk


def _moments_sigma(Q, bg, y, x, s0, valid, n_iter=3, owner=None, me=None):
    """Background-subtracted second moments within r = 3 s + 1.5 around (y, x), iterated; returns (y, x, s)
    with the pixel variance 1/12 removed; s clamped to [0.2, 3]."""
    H, W = Q.shape
    s = s0
    for _ in range(n_iter):
        r = 3 * s + 1.5
        y0, y1 = int(max(np.floor(y - r), 0)), int(min(np.ceil(y + r) + 1, H))
        x0, x1 = int(max(np.floor(x - r), 0)), int(min(np.ceil(x + r) + 1, W))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ins = ((yy - y) ** 2 + (xx - x) ** 2 <= r * r) & valid[y0:y1, x0:x1]
        if owner is not None:
            ins &= owner[y0:y1, x0:x1] == me
        w = (Q[y0:y1, x0:x1] - bg.mean[y0:y1, x0:x1])[ins]
        yy, xx = yy[ins], xx[ins]
        sw = w.sum()
        if sw <= 0:
            break
        ny, nx = (w * yy).sum() / sw, (w * xx).sum() / sw
        if abs(ny - y) > 3 or abs(nx - x) > 3:
            break
        y, x = ny, nx
        v = 0.5 * ((w * ((yy - y) ** 2 + (xx - x) ** 2)).sum() / sw) - 1 / 12
        s = float(np.clip(np.sqrt(max(v, 0.04)), 0.2, 3.0))
    return y, x, s


def mf_detect(Q, valid, bg, scales=(0.5, 1.0, 1.5), nsig=5.0, k_foot=3.5, r_min=2.0, min_sep=2.0, grow_alpha=0.01):
    """Seeds = local maxima (5x5) of the significance map above nsig; per seed: centroid and width from
    background-subtracted moments (_moments_sigma, started at the best matched scale); footprint = disk of
    radius R = max(r_min, k_foot * s + 0.7) (pixel-centre distance; 1 - exp(-R^2 / 2 s^2) > 99.7 % of the
    charge for s >= 0.3). Pixels in several disks go to the seed with the smallest distance / R.
    Energy = sum over the footprint of (Q - mean) [aperture photometry with the background map].
    Clusters whose disk leaves the active region get E = nan."""
    T, bk = mf_significance(Q, valid, bg, scales)
    mx = ndimage.maximum_filter(T, size=5, mode="constant")
    pk = np.nonzero((T == mx) & (T > nsig) & valid)
    # Poisson calibration of the candidates: the Gaussian significance under-states the tail of sparse
    # Poisson counts (at lam = 0.2 one 3-e- pixel is a '5 sigma' peak at scale 0.5). Keep a candidate only if
    # the exact Poisson tail of the 3x3 or 5x5 box sum, P(Poisson(sum lam) >= sum(Q - o)), is below the
    # Gaussian tail of nsig (Li & Ma-type counting test; Gaussian noise folded in as -2 s sqrt(N) on the sum)
    p_thr = stats.norm.sf(nsig)
    keep = np.zeros(pk[0].size, bool)
    R0 = np.where(valid, Q - bg.o, 0.0)
    for size in (3, 5):
        S = ndimage.uniform_filter(R0, size, mode="constant")[pk] * size * size
        L = ndimage.uniform_filter(np.where(valid, bg.lam, 0.0), size, mode="constant")[pk] * size * size
        sn = np.sqrt(ndimage.uniform_filter(np.where(valid, bg.s2, 0.0), size, mode="constant")[pk] * size * size)
        k = np.floor(S - 2 * sn + 0.5)
        keep |= stats.poisson.sf(k - 1, L) < p_thr
    pk = (pk[0][keep], pk[1][keep])
    order = np.argsort(-T[pk])
    seeds = []
    for i in order:
        y, x = pk[0][i], pk[1][i]
        if all((y - a) ** 2 + (x - b) ** 2 >= min_sep ** 2 for a, b, *_ in seeds[-50:]):
            seeds.append((y, x, scales[bk[y, x]], T[y, x]))
    H, W = Q.shape
    cl = []
    # Voronoi cells of the seeds (nearest seed of every pixel): the moments of a seed use only its own cell,
    # so two close clusters do not pull each other's centroid / width
    owner = None
    if seeds:
        mk = np.ones(Q.shape, bool)
        for (y, x, *_r) in seeds:
            mk[y, x] = False
        _, (iy, ix) = ndimage.distance_transform_edt(mk, return_indices=True)
        lut = np.full(Q.shape, -1, int)
        for k, (y, x, *_r) in enumerate(seeds):
            lut[y, x] = k
        owner = lut[iy, ix]
    for k, (y, x, s0, t) in enumerate(seeds):
        my, mxx, s = _moments_sigma(Q, bg, float(y), float(x), s0, valid, owner=owner, me=k)
        if abs(my - y) > 3 or abs(mxx - x) > 3:
            my, mxx = float(y), float(x)
        cl.append(dict(y=my, x=mxx, sxy=s, T=float(t), R=max(r_min, k_foot * s + 0.7), yp=int(y), xp=int(x)))
    claim = assign_footprints(Q.shape, cl)
    grown = None
    if grow_alpha and cl:
        # reconstruction-grown footprint: pixels above the Poisson-tail threshold (alpha = grow_alpha) that are
        # 8-connected to a disk (binary morphological reconstruction, i.e. hysteresis with the disks as markers)
        # catch charge outside the disks - the readout-direction (+y) trails of the real clusters; each grown
        # pixel goes to the cluster owning the nearest disk pixel
        disks = claim >= 0
        M = ((Q > tail_threshold_map(bg, grow_alpha)) & valid) | disks
        F = ndimage.binary_propagation(disks, structure=EIGHT, mask=M)
        extra = F & ~disks
        if extra.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~disks, return_indices=True)
            grown = np.where(extra, claim[iy, ix], -1)
    for i, c in enumerate(cl):
        R = c["R"]
        y0, y1 = int(max(np.floor(c["y"] - R), 0)), int(min(np.ceil(c["y"] + R) + 1, H))
        x0, x1 = int(max(np.floor(c["x"] - R), 0)), int(min(np.ceil(c["x"] + R) + 1, W))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ins = (yy - c["y"]) ** 2 + (xx - c["x"]) ** 2 <= R * R
        mine = ins & (claim[y0:y1, x0:x1] == i)
        if not valid[y0:y1, x0:x1][ins].all():
            c["E"] = np.nan
        else:
            c["E"] = float((Q[y0:y1, x0:x1] - bg.mean[y0:y1, x0:x1])[mine].sum() * E_PER_E)
        c["npix"] = int(mine.sum())
    if grown is not None:
        gy, gx = np.nonzero(grown >= 0)
        own = grown[gy, gx]
        order = np.argsort(own)
        gy, gx, own = gy[order], gx[order], own[order]
        cuts = np.searchsorted(own, np.arange(len(cl) + 1))
        for i, c in enumerate(cl):
            a, b = cuts[i], cuts[i + 1]
            c["grow_pix"] = (gy[a:b], gx[a:b])
            if b > a and np.isfinite(c["E"]):
                c["E"] += float((Q[gy[a:b], gx[a:b]] - bg.mean[gy[a:b], gx[a:b]]).sum() * E_PER_E)
                c["npix"] += int(b - a)
    return cl, T


def trail_correction(Q, bg, c, E_e, y, x, s):
    """Charge [e-] on the cluster's grown footprint pixels outside its disk that the Gaussian model does not
    explain: sum (Q - mean - E P) over those pixels (CTI trails along the readout); 0 if none."""
    gp = c.get("grow_pix")
    if gp is None or len(gp[0]) == 0:
        return 0.0
    gy, gx = gp
    fy = _f1d(gy.astype(float), y, s)[0]
    fx = _f1d(gx.astype(float), x, s)[0]
    return float((Q[gy, gx] - bg.mean[gy, gx]).sum() - E_e * (fy * fx).sum())


def assign_footprints(shape, cl):
    """Label image: pixel -> index of the cluster whose disk contains it with the smallest d / R (-1: none)."""
    H, W = shape
    claim = np.full(shape, -1, int)
    best = np.full(shape, np.inf)
    for i, c in enumerate(cl):
        R = c["R"]
        y0, y1 = int(max(np.floor(c["y"] - R), 0)), int(min(np.ceil(c["y"] + R) + 1, H))
        x0, x1 = int(max(np.floor(c["x"] - R), 0)), int(min(np.ceil(c["x"] + R) + 1, W))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((yy - c["y"]) ** 2 + (xx - c["x"]) ** 2) / R
        sub = best[y0:y1, x0:x1]
        upd = (d <= 1) & (d < sub)
        sub[upd] = d[upd]
        claim[y0:y1, x0:x1][upd] = i
    return claim


# ----------------------------------------------------------------------------------------------------------
# 4. Poisson maximum-likelihood (Cash) fit
# ----------------------------------------------------------------------------------------------------------
_DEV_TAB = None


def _deviance_table():
    """E[D] and Var[D] of the Poisson unit deviance d(n, m) = 2 [m - n + n ln(n / m)] for n ~ Poisson(m),
    tabulated on a log grid of m (used for the goodness-of-fit z-score; Mighell 1999 / McCullagh & Nelder)."""
    global _DEV_TAB
    if _DEV_TAB is None:
        ms = np.logspace(-3, 4, 300)
        Ed, Vd = [], []
        for m in ms:
            n = np.arange(int(m + 12 * np.sqrt(m) + 20))
            p = stats.poisson.pmf(n, m)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = 2 * (m - n + np.where(n > 0, n * np.log(n / m), 0))
            e = (p * d).sum(); Ed.append(e); Vd.append((p * (d - e) ** 2).sum())
        _DEV_TAB = (np.log(ms), np.array(Ed), np.array(Vd))
    return _DEV_TAB


def deviance_z(n, m):
    lm, Ed, Vd = _deviance_table()
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 2 * (m - n + np.where(n > 0, n * np.log(np.maximum(n, 1e-300) / m), 0))
    lmm = np.log(np.clip(m, 1e-3, 1e4))
    e, v = np.interp(lmm, lm, Ed), np.interp(lmm, lm, Vd)
    return float((d.sum() - e.sum()) / np.sqrt(v.sum())), float(d.sum())


def _stamp(Q, bg, valid, y, x, half, exclude=None):
    H, W = Q.shape
    cy, cx = int(round(y)), int(round(x))
    y0, y1, x0, x1 = max(cy - half, 0), min(cy + half + 1, H), max(cx - half, 0), min(cx + half + 1, W)
    sl = (slice(y0, y1), slice(x0, x1))
    use = valid[sl].copy()
    if exclude is not None:
        use &= ~exclude[sl]
    return sl, np.arange(y0, y1, dtype=float), np.arange(x0, x1, dtype=float), use


def _data_model_terms(Q, bg, sl, lik):
    """Data n and background b for the likelihood:
       'poisson': n = round(Q - o) clipped at 0, b = lam   (read noise << 1 e-: the comb is resolved)
       'huang'  : n = Q - o + s^2 (clipped at 0), b = lam + s^2  (Huang et al. 2013 sCMOS approximation of
                  Poisson x Gaussian by a Poisson of mean m + s^2; exact in mean and variance)"""
    if lik == "poisson":
        n = np.clip(np.round(Q[sl] - bg.o[sl]), 0, None)
        b = bg.lam[sl]
    else:
        n = np.clip(Q[sl] - bg.o[sl] + bg.s2[sl], 0, None)
        b = bg.lam[sl] + bg.s2[sl]
    return n, b


def cash_fit(Q, bg, valid, y, x, s0=1.0, E0=None, half=None, exclude=None, lik="auto", fit_s=True):
    """Poisson maximum likelihood of  m_i = b_i + E * P_i(y0, x0, s)  on a (2 half + 1)^2 stamp, b fixed from
    the background map: minimise the Cash (1979) statistic C = 2 sum_i (m_i - n_i ln m_i) (Smith et al. 2010
    MLE of a pixel-integrated Gaussian; Huang et al. 2013 for the read-noise term), analytic gradient,
    L-BFGS-B. Free: E (total charge, e-), y0, x0, s. Fisher matrix I_jk = sum_i dm_i/dth_j dm_i/dth_k / m_i;
    errors = sqrt(diag(I^-1)). Goodness of fit: Poisson deviance z-score (deviance_z).
    lik='auto' -> 'poisson' where the pixel noise s < 0.5 e-, else 'huang'.
    Returns dict(E [keV], E_e [e-], dE [e-], y, x, sxy, ds, C, z, ok)."""
    if lik == "auto":
        lik = "poisson" if np.sqrt(bg.s2[int(round(y)), int(round(x))]) < 0.5 else "huang"
    if half is None:
        half = int(np.clip(np.ceil(3.5 * s0 + 2.5), 4, 9))
    sl, yv, xv, use = _stamp(Q, bg, valid, y, x, half, exclude)
    n, b = _data_model_terms(Q, bg, sl, lik)
    if E0 is None:
        E0 = max(float((n - b)[use].sum()), 20.0)
    sc = np.array([E0, 1.0, 1.0, 1.0])

    def fg(p):
        E, yy, xx, s = p * sc
        P, dy, dx, ds = gauss_pix(yv, xv, yy, xx, s, derivs=True)
        m = b + E * P
        r = np.where(use, 1 - n / m, 0)
        C = 2 * np.sum(np.where(use, m - n * np.log(m), 0))
        g = 2 * np.array([(r * P).sum(), (r * E * dy).sum(), (r * E * dx).sum(), (r * E * ds).sum()]) * sc
        if not fit_s:
            g[3] = 0
        return C, g
    p0 = np.array([1.0, y, x, s0])
    bnds = [(1e-3, 50), (y - 2.5, y + 2.5), (x - 2.5, x + 2.5), (0.08, 4.0) if fit_s else (s0, s0)]
    r = optimize.minimize(fg, p0, jac=True, method="L-BFGS-B", bounds=bnds,
                          options=dict(maxiter=300, ftol=1e-12, gtol=1e-8))
    E, yy, xx, s = r.x * sc
    P, dy, dx, ds = gauss_pix(yv, xv, yy, xx, s, derivs=True)
    m = b + E * P
    J = np.stack([P[use], E * dy[use], E * dx[use], E * ds[use]], 1)
    I = (J / m[use][:, None]).T @ J
    k = 4 if fit_s else 3
    try:
        cov = np.linalg.inv(I[:k, :k])
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(k, np.nan)
    z, D = deviance_z(n[use], m[use])
    # fraction of the model charge that falls on usable stamp pixels (low -> extrapolated energy)
    cover = float(P[use].sum())
    return dict(E=float(E * E_PER_E), E_e=float(E), dE=float(err[0]), y=float(yy), x=float(xx), sxy=float(s),
                ds=float(err[3]) if fit_s else np.nan, C=float(r.fun), z=z, D=D, npix=int(use.sum()),
                cover=cover, ok=bool(r.success), lik=lik)


# ----------------------------------------------------------------------------------------------------------
# 5. optimal (profile-weighted) photometry
# ----------------------------------------------------------------------------------------------------------
def optimal_photometry(Q, bg, valid, y, x, s, half=None, E0=None, n_iter=5, exclude=None):
    """Naylor (1998, MNRAS 296, 339) optimal extraction:  E = sum_i W_i (Q_i - mean_i),
    W_i = (P_i / V_i) / sum_j (P_j^2 / V_j),  V_i = var_i + E P_i  (background + source shot noise),
    which equals  E = sum w_i R_i / sum w_i P_i  with w_i = P_i / V_i; iterated because V depends on E.
    Unbiased for any weights when the profile P is right; variance sigma_E^2 = 1 / sum(P_i^2 / V_i)
    (the Cramer-Rao bound for the amplitude at fixed profile)."""
    if half is None:
        half = int(np.clip(np.ceil(3.5 * s + 2.5), 4, 9))
    sl, yv, xv, use = _stamp(Q, bg, valid, y, x, half, exclude)
    P = gauss_pix(yv, xv, y, x, s)
    R = Q[sl] - bg.mean[sl]
    V0 = bg.var[sl]
    E = max(R[use].sum(), 1.0) if E0 is None else E0
    for _ in range(n_iter):
        w = np.where(use, P / (V0 + max(E, 0) * P), 0)
        E = (w * R).sum() / (w * P).sum()
    w = np.where(use, P / (V0 + max(E, 0) * P), 0)
    dE = 1 / np.sqrt((w * P).sum())
    return dict(E=float(E * E_PER_E), E_e=float(E), dE=float(dE), y=y, x=x, sxy=s)


# ----------------------------------------------------------------------------------------------------------
# 6. multi-emitter fit seeded by h-maxima
# ----------------------------------------------------------------------------------------------------------
def grey_reconstruct(marker, mask):
    """Morphological reconstruction by dilation of marker under mask (Vincent 1993): iterate
    marker <- min(dilate_3x3(marker), mask) until stable."""
    m = np.minimum(marker, mask)
    while True:
        nm = np.minimum(ndimage.grey_dilation(m, size=(3, 3)), mask)
        if np.array_equal(nm, m):
            return m
        m = nm


def h_maxima(f, h):
    """Regional maxima of dynamic >= h (h-maxima transform, Soille 'Morphological Image Analysis'):
    dome = f - R_f(f - h); markers = dome >= h (top plateaus of peaks that rise at least h above any
    saddle to a higher peak). Returns a list of (y, x) at the highest pixel of each marker component."""
    dome = f - grey_reconstruct(f - h, f)
    lab, n = ndimage.label(dome >= h - 1e-9, structure=EIGHT)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        i = np.argmax(f[ys, xs])
        out.append((int(ys[i]), int(xs[i])))
    return out


def _multi_cash(n, b, use, yv, xv, init, sbound=(0.08, 4.0)):
    """Joint Cash fit of N Gaussians (E_j, y_j, x_j, s_j) on a shared fixed background b."""
    N = len(init)
    sc = np.concatenate([[max(e, 20), 1, 1, 1] for e, *_ in init])

    def fg(p):
        q = p * sc
        m = b.copy()
        parts = []
        for j in range(N):
            E, yy, xx, s = q[4 * j:4 * j + 4]
            P, dy, dx, ds = gauss_pix(yv, xv, yy, xx, s, derivs=True)
            m = m + E * P
            parts.append((E, P, dy, dx, ds))
        r = np.where(use, 1 - n / m, 0)
        C = 2 * np.sum(np.where(use, m - n * np.log(m), 0))
        g = []
        for (E, P, dy, dx, ds) in parts:
            g += [(r * P).sum(), (r * E * dy).sum(), (r * E * dx).sum(), (r * E * ds).sum()]
        return C, 2 * np.array(g) * sc
    p0, bnds = [], []
    for (e, y, x, s) in init:
        p0 += [1.0, y, x, s]
        bnds += [(1e-3, 50), (y - 3, y + 3), (x - 3, x + 3), sbound]
    r = optimize.minimize(fg, np.array(p0, float), jac=True, method="L-BFGS-B", bounds=bnds,
                          options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    q = r.x * sc
    return float(r.fun), [tuple(q[4 * j:4 * j + 4]) for j in range(N)]


def multi_fit(Q, bg, valid, c, half=None, exclude=None, nmax=3, h_rel=0.05, h_nsig=4.0, smooth=0.7,
              penalty=None, min_e=100.0, min_sep=1.5):
    """Multi-emitter fit of a flagged cluster (DAOSTORM, Holden et al. 2011; PyBDSF Gaussian decomposition):
    seeds = h-maxima of the stamp smoothed with a sigma=`smooth` Gaussian, h = max(h_rel * peak,
    h_nsig * noise of the smoothed peak incl. signal shot noise); fit N = 1 .. nmax Gaussians jointly (Cash statistic, shared
    background); N+1 is accepted when C_N - C_{N+1} > penalty, default the BIC penalty 4 ln(npix)
    (Schwarz 1978: Delta k = 4 parameters), and only if every component holds >= max(min_e e-, 5 %) of the
    charge and the centres are >= min_sep px apart (no core + halo splits of one cluster). Extra components beyond the h-maxima are started at the pixel of
    largest positive residual; N is limited to the number of h-maxima. Returns a list of component dicts (E [keV], E_e, y, x, sxy) and the chosen N."""
    s0 = c.get("sxy", 1.0) if np.isfinite(c.get("sxy", np.nan)) else 1.0
    if half is None:
        half = int(np.clip(np.ceil(3.5 * s0 + 4.5), 6, 11))
    sl, yv, xv, use = _stamp(Q, bg, valid, c["y"], c["x"], half, exclude)
    lik = "poisson" if np.sqrt(bg.s2[sl].mean()) < 0.5 else "huang"
    n, b = _data_model_terms(Q, bg, sl, lik)
    R = np.where(use, Q[sl] - bg.mean[sl], 0.0)
    Rs = ndimage.gaussian_filter(R, smooth, mode="constant")
    # noise of the smoothed stamp at the peak: background + signal shot noise, var ~ (V + peak) / (4 pi s^2)
    noise_s = np.sqrt((bg.var[sl].mean() + max(Rs.max(), 0)) / (4 * np.pi * smooth ** 2))
    h = max(h_rel * Rs.max(), h_nsig * noise_s)
    seeds = sorted(h_maxima(Rs, h), key=lambda p: -Rs[p])
    if penalty is None:
        penalty = 4 * np.log(max(use.sum(), 2))
    tot = max(R[use].sum(), 20.0)
    fits = []
    init = [(tot, c["y"], c["x"], s0)]
    C1, comp = _multi_cash(n, b, use, yv, xv, init)
    fits.append((C1, comp))
    # at most as many components as there are significant peaks (h-maxima): one peaked but non-Gaussian
    # (elongated, core + halo) cluster stays one cluster
    for N in range(2, min(nmax, max(len(seeds), 1)) + 1):
        prevC, prev = fits[-1]
        m = b.copy()
        for (E, yy, xx, s) in prev:
            m = m + E * gauss_pix(yv, xv, yy, xx, s)
        used = [(p[1], p[2]) for p in prev]
        cand = [(yv[0] + a, xv[0] + bb) for a, bb in seeds]
        cand = [p for p in cand if min((p[0] - u[0]) ** 2 + (p[1] - u[1]) ** 2 for u in used) > 2.0 ** 2]
        if cand:
            ny, nx = cand[0]
        else:
            res = np.where(use, ndimage.gaussian_filter(n - m, smooth), -np.inf)
            a, bb = np.unravel_index(np.argmax(res), res.shape)
            ny, nx = yv[a], xv[bb]
        Eguess = max(float(ndimage.gaussian_filter(n - m, smooth)[int(ny - yv[0]), int(nx - xv[0])])
                     * 4 * np.pi * smooth ** 2, 20.0)
        init = [(E * 0.9, yy, xx, s) for (E, yy, xx, s) in prev] + [(Eguess, ny, nx, s0)]
        CN, compN = _multi_cash(n, b, use, yv, xv, init)
        # guard against splitting one non-Gaussian cluster into core + halo: every component must carry
        # >= max(min_e, 5 %) of the charge and component centres must be >= min_sep pixels apart
        Es = np.array([p[0] for p in compN]); ctr = np.array([(p[1], p[2]) for p in compN])
        dmin = min(np.hypot(*(ctr[i] - ctr[j])) for i in range(N) for j in range(i + 1, N))
        if prevC - CN > penalty and Es.min() >= max(min_e, 0.05 * Es.sum()) and dmin >= min_sep:
            fits.append((CN, compN))
        else:
            break
    C, comp = fits[-1]
    out = [dict(E=E * E_PER_E, E_e=E, y=yy, x=xx, sxy=s) for (E, yy, xx, s) in comp]
    return out, len(comp), len(seeds)


# ----------------------------------------------------------------------------------------------------------
# pipelines
# ----------------------------------------------------------------------------------------------------------
def group_seeds(cl, link=1.0):
    """Groups of detected clusters whose footprints overlap: i ~ j if dist(i, j) < link * (R_i + R_j)
    (DAOPHOT/NSTAR 'groups', fitted jointly; Stetson 1987). Returns a list of index lists."""
    n = len(cl)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ys = np.array([c["y"] for c in cl]); xs = np.array([c["x"] for c in cl]); Rs = np.array([c["R"] for c in cl])
    order = np.argsort(ys)
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if ys[j] - ys[i] > 2 * Rs.max():
                break
            if np.hypot(ys[i] - ys[j], xs[i] - xs[j]) < link * (Rs[i] + Rs[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def group_fit(Q, bg, valid, cl, idx, exclude, margin=3):
    """Joint Cash fit of the Gaussians of one group of overlapping clusters (shared fixed background) on the
    bounding box of their footprints + margin; pixels claimed by clusters outside the group are excluded.
    Returns component dicts (E [keV], E_e, y, x, sxy) aligned with idx, and the deviance z of the joint fit."""
    H, W = Q.shape
    y0 = int(max(min(cl[i]["y"] - cl[i]["R"] for i in idx) - margin, 0))
    y1 = int(min(max(cl[i]["y"] + cl[i]["R"] for i in idx) + margin + 1, H))
    x0 = int(max(min(cl[i]["x"] - cl[i]["R"] for i in idx) - margin, 0))
    x1 = int(min(max(cl[i]["x"] + cl[i]["R"] for i in idx) + margin + 1, W))
    sl = (slice(y0, y1), slice(x0, x1))
    use = valid[sl] & ~exclude[sl]
    lik = "poisson" if np.sqrt(bg.s2[sl].mean()) < 0.5 else "huang"
    n, b = _data_model_terms(Q, bg, sl, lik)
    yv, xv = np.arange(y0, y1, dtype=float), np.arange(x0, x1, dtype=float)
    init = [(max(cl[i]["E"] / E_PER_E, 20.0) if np.isfinite(cl[i]["E"]) else 100.0, cl[i]["y"], cl[i]["x"],
             max(cl[i]["sxy"] if np.isfinite(cl[i]["sxy"]) else 1.0, 0.3)) for i in idx]
    C, comp = _multi_cash(n, b, use, yv, xv, init)
    m = b.copy()
    for (E, yy, xx, s) in comp:
        m = m + E * gauss_pix(yv, xv, yy, xx, s)
    z, _ = deviance_z(n[use], m[use])
    return [dict(E=E * E_PER_E, E_e=E, y=yy, x=xx, sxy=s) for (E, yy, xx, s) in comp], z


def pipeline(Q, valid, bg=None, detector="mf", energy="cash", multi=True, z_flag=4.0, alpha=0.05,
             nsig=5.0, emin_fit=0.3, max_group=4, only_near=None, near_r=4.0, trails=True, grow_alpha=0.01,
             z_good=3.0):
    """1 -> detector (3 'mf' or 2 'pt') -> energy (4 'cash', 5 'naylor', or 'sum' = the detector's own sum)
    -> 6 for flagged clusters.
      * clusters whose footprints overlap are fitted jointly (group_fit, DAOPHOT/NSTAR-style groups of up to
        `max_group` members; larger groups - tracks - keep the detector's footprint sums)
      * isolated clusters: single cash_fit (neighbouring footprints excluded); if its deviance z > z_flag the
        multi-emitter fit (multi_fit, h-maxima seeds, BIC) may split it
      * energy='naylor': profile (centre, width) from the Cash fit, amplitude from optimal_photometry
      * trails=True (mf detector): + trail_correction, the charge on the reconstruction-grown footprint
        pixels outside the disk that the Gaussian does not explain (readout trails of the real clusters)
      * energy='hybrid': Cash energy when the single fit is good (deviance z <= z_good), else the detector's
        model-free footprint sum (disk + grown pixels); overlapping groups always use the joint fit
    Clusters with detector energy below emin_fit keV keep the detector's energy. only_near (list of (y, x)):
    fit only clusters within near_r px of these positions (used to speed up injection tests)."""
    if bg is None:
        bg = background_map(Q, valid)
    if detector == "mf":
        det, _ = mf_detect(Q, valid, bg, nsig=nsig, grow_alpha=grow_alpha if trails else 0)
    else:
        det, _ = poisson_tail_clusters(Q, valid, bg, alpha=alpha)
        for c in det:
            c["R"] = max(2.0, 3.5 * (c["sxy"] if np.isfinite(c["sxy"]) else 1) + 0.7)
    det = [c for c in det if np.isfinite(c.get("y", np.nan)) and np.isfinite(c.get("x", np.nan))]
    if energy == "sum":
        return det, bg
    claim = assign_footprints(Q.shape, det)

    def wanted(c):
        return only_near is None or any((c["y"] - y) ** 2 + (c["x"] - x) ** 2 < near_r ** 2 for y, x in only_near)
    out = []
    for g in group_seeds(det):
        if not any(wanted(det[i]) for i in g):
            continue
        fitme = [i for i in g if np.isfinite(det[i]["E"]) and det[i]["E"] > emin_fit]
        if len(g) > 1 and 1 < len(fitme) <= max_group:
            excl = (claim >= 0) & ~np.isin(claim, g)
            comps, z = group_fit(Q, bg, valid, det, fitme, excl)
            for i, cc in zip(fitme, comps):
                tr = trail_correction(Q, bg, det[i], cc["E_e"], cc["y"], cc["x"], cc["sxy"]) if trails else 0.0
                Eg = cc["E"] + tr * E_PER_E
                if energy == "hybrid" and z > z_good and not (len(fitme) > 1):
                    Eg = det[i]["E"]
                out.append(dict(det[i], E=Eg, y=cc["y"], x=cc["x"], sxy=cc["sxy"], z=z,
                                group=len(fitme), trail=tr))
            out += [det[i] for i in g if i not in fitme]
            continue
        if len(fitme) > max_group:            # extended object (track): footprint sums
            out += [det[i] for i in g]
            continue
        for i in g:
            c = det[i]
            if i not in fitme:
                out.append(c); continue
            excl = (claim >= 0) & (claim != i)
            s0 = c["sxy"] if np.isfinite(c["sxy"]) else 1.0
            f = cash_fit(Q, bg, valid, c["y"], c["x"], s0=max(s0, 0.3), E0=c["E"] / E_PER_E, exclude=excl)
            if energy == "naylor":
                gph = optimal_photometry(Q, bg, valid, f["y"], f["x"], f["sxy"], exclude=excl, E0=f["E_e"])
                e = dict(c, E=gph["E"], E_e=gph["E_e"], dE=gph["dE"], y=f["y"], x=f["x"], sxy=f["sxy"], z=f["z"])
            else:
                e = dict(c, E=f["E"], E_e=f["E_e"], dE=f["dE"], y=f["y"], x=f["x"], sxy=f["sxy"], z=f["z"], cover=f["cover"])
            if trails:
                e["trail"] = trail_correction(Q, bg, c, e["E_e"], e["y"], e["x"], e["sxy"])
                e["E"] = e["E"] + e["trail"] * E_PER_E
            if energy == "hybrid" and f["z"] > z_good:
                # the Gaussian does not describe this cluster (compact front-side core, readout trail):
                # keep the model-free footprint sum of the detector
                e["E"] = c["E"]; e["E_src"] = "footprint"
            if multi and f["z"] > z_flag:
                comps, N, nseed = multi_fit(Q, bg, valid, e, exclude=excl)
                if N > 1:
                    out += [dict(cc, z=f["z"], split=N, npix=c.get("npix", 0)) for cc in comps]
                    continue
            out.append(e)
    return out, bg
