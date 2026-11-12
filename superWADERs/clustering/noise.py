"""Read-noise model for the Bayesian few-electron finder (wip/lowdc/bayes): pedestal nuisances, a symmetric scale-mixture
noise density fitted on the NEGATIVE side of the zero peak, the asymmetry test, and the pixel log-likelihoods.

Pixel model (in electrons, after gain and pedestal):  v_ij = y_ij / g - o_i - u_j = n_ij + kappa_j * eps_ij
  o_i       per-row pedestal residual (low-frequency noise): a nuisance, profiled with a symmetric trimmed mean of the
            zero peak (its statistical error, s / sqrt(n_row) ~ 0.005 e-, is negligible against s and is ignored)
  u_j, kappa_j  per-column offset and noise scale, used only where the column differs significantly (first-column
            artefact, a noisy column); kappa_j = 1 elsewhere
  eps       read noise with density f(eps) = sum_k w_k N(eps; 0, s_k): a zero-mean scale mixture (Gaussian core, a
            wider tail component, a broad 'excursion' component of a few e-). No rounding anywhere.

Why the negative side: no pixel holds negative charge, so v < 0 is (to < 1e-6) pure noise of n = 0 pixels; assuming f is
symmetric, its fold |v| measures f on both sides without any contamination by dark charge or events. The assumption is
tested by comparing the counts at +a and -a for a below 0.5 e- (where n = 1 pixels cannot reach), see asymmetry().
"""
import numpy as np
from scipy import special

LOG2PI = np.log(2.0 * np.pi)


def robust_sigma(x):
    """1.4826 MAD"""
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def pedestals(Y, valid, s_guess=None, n_iter=4, col=True):
    """per-row offset o (and per-column offset u / scale kappa where significant) of the zero peak.
    Symmetric trimmed mean in +-min(2.5 s, 0.4) e- around the current estimate: unbiased for symmetric noise, and a
    1-e- pixel never enters the window. Returns v (residual image, NaN outside valid), o, u, kappa, s_core."""
    Yv = np.where(valid, Y, np.nan)
    o = np.nanmedian(Yv, axis=1)
    o = np.where(np.isfinite(o), o, 0.0)
    v = Yv - o[:, None]
    s = robust_sigma(v[np.abs(v) < 0.5]) if s_guess is None else s_guess
    for _ in range(n_iter):
        hw = min(2.5 * s, 0.4)
        m = np.abs(v) < hw
        cnt = m.sum(1)
        d = np.where(cnt > 5, np.nansum(np.where(m, v, 0.0), 1) / np.maximum(cnt, 1), 0.0)
        o = o + d
        v = Yv - o[:, None]
        s = robust_sigma(v[np.abs(v) < min(5 * s, 0.5)])
    H, W = Y.shape
    u = np.zeros(W); kap = np.ones(W)
    if col and H >= 20:
        # column offsets / scales: only where the deviation is many standard errors (artefact columns), else 0 / 1
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            med = np.nanmedian(v, axis=0)
            vc = np.where(np.abs(v - med[None, :]) < max(1.0, 6 * s), v, np.nan)
            mc = np.nanmedian(vc, axis=0)
            sc = 1.4826 * np.nanmedian(np.abs(vc - mc[None, :]), axis=0)
        sc = np.where(np.isfinite(sc), sc, s)
        nr = valid.sum(0).clip(1)
        big_off = np.abs(med) > 6 * 1.25 * s / np.sqrt(nr)
        big_sc = sc > s * (1 + 6 / np.sqrt(2 * nr)) * 1.1
        u = np.where(big_off | big_sc, med, 0.0)
        kap = np.where(big_sc, sc / s, 1.0)
        u = np.where(np.isfinite(u), u, 0.0); kap = np.where(np.isfinite(kap), kap, 1.0)
        v = Yv - o[:, None] - u[None, :]
    return v, o, u, kap, s


class NoiseModel:
    """zero-mean scale mixture f(eps) = sum_k w_k N(0, s_k)"""

    def __init__(self, w, s):
        self.w = np.asarray(w, float); self.s = np.asarray(s, float)
        self.lw = np.log(np.maximum(self.w, 1e-300)) - np.log(self.s) - 0.5 * LOG2PI

    def logpdf(self, e, kap=1.0):
        """log f(e / kap) - log kap (vectorised; kap broadcast with e)"""
        e = np.asarray(e, float) / kap
        t = self.lw[(None,) * e.ndim] - 0.5 * (e[..., None] / self.s) ** 2
        return special.logsumexp(t, axis=-1) - np.log(kap)

    @property
    def core(self):
        return float(self.s[0])

    def __repr__(self):
        return "NoiseModel(" + ", ".join(f"w={a:.2e} s={b:.3f}" for a, b in zip(self.w, self.s)) + ")"


def fit_negative(v, s0, broad=True, n_iter=200, prior=1.0):
    """EM fit of the scale mixture to the folded NEGATIVE values (half-normal mixture on |v|, v < 0).
    Components: core (s0), tail (3 s0), and if broad an excursion component (init 3 e-, kept >= 1 e-).
    Weights: posterior mean under a Dirichlet(prior, ...) prior, so a component never seen keeps ~prior / n weight
    (e.g. 1 / n_neg for the excursions: a conservative, data-sized allowance instead of an exact zero)."""
    x = -v[np.isfinite(v) & (v < 0)]
    n = x.size
    # weighted EM on a fine histogram of |v| below 1 e- (bin s0 / 40; the fit only, pixel values are never binned) plus
    # every value above 1 e- individually (the rare excursions)
    bw = s0 / 40.0
    lo = x < 1.0
    hc = np.bincount((x[lo] / bw).astype(int))
    xb = np.r_[(np.nonzero(hc)[0] + 0.5) * bw, x[~lo]]
    wb = np.r_[hc[hc > 0], np.ones((~lo).sum())].astype(float)
    s = np.array([s0, 3 * s0] + ([3.0] if broad else []), float)
    w = np.array([0.99, 0.01] + ([1e-4] if broad else []), float); w /= w.sum()
    nk = np.zeros(s.size)
    for _ in range(n_iter):
        lp = np.log(w) - np.log(s) - 0.5 * (xb[:, None] / s) ** 2
        r = np.exp(lp - special.logsumexp(lp, axis=1, keepdims=True)) * wb[:, None]
        nk = r.sum(0)
        w_new = (nk + prior) / (n + prior * len(s))
        s_new = np.sqrt((r * xb[:, None] ** 2).sum(0) / np.maximum(nk, 1e-12))
        # empty components keep their scale; ordering core < tail < broad is kept by bounds
        s_new = np.where(nk > 0.5, s_new, s)
        s_new[0] = max(s_new[0], 1e-3)
        s_new[1] = np.clip(s_new[1], 1.3 * s_new[0], 0.6)
        if broad:
            s_new[2] = max(s_new[2], 1.0)
        done = np.allclose(s_new, s, rtol=1e-7) and np.allclose(w_new, w, rtol=1e-7)
        s, w = s_new, w_new
        if done:
            break
    return NoiseModel(w, s), dict(n_neg=int(n), n_k=nk)


def asymmetry(v, s, levels=(2.5, 3.0, 3.5, 4.0)):
    """symmetry test of the noise: counts of v > a and v < -a for a = level x s (only a < 0.45 e-, where a 1-e- pixel
    needs a noise excursion of > 0.55 e-), and of |v| > 1.5 e- (positive side contaminated by charge: an upper bound)"""
    out = []
    for k in levels:
        a = k * s
        if a >= 0.45:
            continue
        m = (np.abs(v) < 0.45)
        npos = int(((v > a) & m).sum()); nneg = int(((v < -a) & m).sum())
        out.append(dict(level=k, a=a, n_pos=npos, n_neg=nneg, ratio=npos / max(nneg, 1),
                        z=(npos - nneg) / np.sqrt(max(npos + nneg, 1))))
    out.append(dict(level=np.nan, a=1.5, n_pos=int((v > 1.5).sum()), n_neg=int((v < -1.5).sum()),
                    ratio=np.nan, z=np.nan))
    out.append(dict(level=np.nan, a=5.0, n_pos=int((v > 5).sum()), n_neg=int((v < -5).sum()), ratio=np.nan, z=np.nan))
    return out


def loglike_table(vals, kap, nm, K):
    """log P(v | n) = log f((v - n) / kap) - log kap for n = 0..K: shape (len(vals), K + 1)"""
    n = np.arange(K + 1)
    return nm.logpdf(np.asarray(vals)[:, None] - n[None, :], np.asarray(kap)[:, None])


def candidates(v, kap, nm, valid, r_min=1e-2, K=20):
    """soft candidate pixels: max_{n >= 1} f(v - n) / f(v) > r_min. With a permissive prior P(n >= 1) ~ 1e-3 this is
    P(n >= 1 | v) > ~1e-5: any pixel below it changes an event's evidence by < r_min relative (it is treated as n = 0).
    Evaluated at n = 1 and at n = round(v) (the likelihood ratio is unimodal in n for a scale mixture)."""
    vv = np.where(valid, v, -10.0)
    kk = kap[None, :] * np.ones_like(vv)
    l0 = nm.logpdf(vv, kk)
    l1 = nm.logpdf(vv - 1.0, kk)
    nr = np.clip(np.rint(vv), 1, K)
    lr = nm.logpdf(vv - nr, kk)
    lmax = np.maximum(l1, lr) - l0
    return (lmax > np.log(r_min)) & valid


def gain_profile(y_adu, s_adu, nm_e, g_grid, lam_frac=None):
    """profile likelihood of the gain g [ADU/e-] from pixel values y_adu (pedestal-subtracted, ADU) above ~4 s_adu:
    each pixel is n = 0, 1, 2, ... with unknown occupation fractions (EM over pi_n at each g) and noise nm_e scaled
    to ADU (s_adu fixes the noise scale in ADU, independent of g). Returns log L(g)."""
    y = np.asarray(y_adu, float)
    out = []
    for g in g_grid:
        n = np.arange(0, 8)
        # noise in ADU: the fitted shape (in units of the core) times s_adu
        sc = s_adu / nm_e.core
        lp = nm_e.logpdf((y[:, None] - g * n[None, :]) / sc) - np.log(sc)
        pi = np.full(n.size, 1.0 / n.size)
        for _ in range(200):
            t = lp + np.log(pi)
            r = np.exp(t - special.logsumexp(t, axis=1, keepdims=True))
            pi = (r.sum(0) + 0.1) / (y.size + 0.1 * n.size)
        out.append(float(special.logsumexp(lp + np.log(pi), axis=1).sum()))
    return np.array(out)


# ---------------------------------------------------------------------------------------------- non-parametric density
class EmpiricalNoise:
    """non-parametric noise density (fit_empirical): log f tabulated on a grid of t = asinh(|e| / s0) for each side,
    linear interpolation in t; beyond the grid the parametric base model (shifted to be continuous). Drop-in for
    NoiseModel: logpdf(e, kap), core, and w / s / lw of the parametric base (diagnostics only)."""

    def __init__(self, s0, dt, lf_neg, lf_pos, base, info=None):
        self.s0 = float(s0); self.dt = float(dt)
        self.lf_neg = np.asarray(lf_neg, float); self.lf_pos = np.asarray(lf_pos, float)
        self.base = base
        self.w, self.s, self.lw = base.w, base.s, base.lw
        self.info = info or {}
        tn = (self.lf_neg.size - 1) * self.dt
        xe = self.s0 * np.sinh(tn)
        self.t_end = tn
        # continuity at the grid end: base log density + offset per side
        lb = float(base.logpdf(np.array(xe)))
        self.off_neg = self.lf_neg[-1] - lb; self.off_pos = self.lf_pos[-1] - lb

    @property
    def core(self):
        return float(self.base.s[0])

    def logpdf(self, e, kap=1.0):
        e = np.asarray(e, float) / kap
        t = np.arcsinh(np.abs(e) / self.s0) / self.dt
        i = np.minimum(t.astype(np.int64), self.lf_neg.size - 2)
        fr = np.clip(t - i, 0.0, 1.0)
        neg = e < 0
        tab = np.where(neg, self.lf_neg[i] * (1 - fr) + self.lf_neg[i + 1] * fr,
                       self.lf_pos[i] * (1 - fr) + self.lf_pos[i + 1] * fr)
        far = t >= self.lf_neg.size - 1
        if np.any(far):
            tab = np.where(far, self.base.logpdf(e) + np.where(neg, self.off_neg, self.off_pos), tab)
        return tab - np.log(kap)

    def __repr__(self):
        return f"EmpiricalNoise(s0={self.s0:.3f}, {self.info.get('mode', '')}, base={self.base!r})"


def _half_cdf(nm, x):
    """P(|eps| < x) of a zero-mean scale mixture"""
    x = np.asarray(x, float)
    return (nm.w[None, :] * special.erf(x[:, None] / (np.sqrt(2.0) * nm.s[None, :]))).sum(1) / nm.w.sum()


def _smooth_counts(n, dt, h0=0.6, hmin=None):
    """adaptive Gaussian kernel smoothing of histogram counts n on a uniform grid (step dt): bin k is spread with
    bandwidth h_k = clip(h0 / sqrt(n_k + 1), dt / 2, 0.5) (dense bins stay sharp, isolated counts are spread over
    ~+-40 % in amplitude); mass below t = 0 is reflected (folded density). Total count conserved."""
    K = n.size
    tc = (np.arange(K) + 0.5) * dt
    out = np.zeros(K)
    hmin = dt / 2 if hmin is None else hmin
    nz = np.nonzero(n)[0]
    for k in nz:
        h = np.clip(h0 / np.sqrt(n[k] + 1.0), hmin, 0.5)
        if h <= dt / 2 + 1e-12:
            out[k] += n[k]; continue
        kern = np.exp(-0.5 * ((tc - tc[k]) / h) ** 2) + np.exp(-0.5 * ((tc + tc[k]) / h) ** 2)
        out += n[k] * kern / kern.sum()
    return out


def _spline_correction(cnt, mu, tc, t_hi, knot_step=0.25, pen_grid=(1.0, 10.0, 100.0, 1000.0)):
    """semi-parametric density: counts cnt_k ~ Poisson(mu_k exp(delta(t_k))) with delta = B c a cubic B-spline in t
    (knots every knot_step up to t_hi, the end of the data), prior c ~ N(0, (I + pen D2'D2)^-1) (deviations from the
    base model cost, and must be smooth); pen chosen by the Laplace evidence over pen_grid. Returns delta at tc,
    its posterior sd (Laplace) and the chosen pen. Beyond t_hi delta -> 0 (the base model)."""
    from scipy.interpolate import BSpline
    kn = np.arange(0.0, t_hi + knot_step, knot_step)
    t = np.r_[[kn[0]] * 3, kn, [kn[-1]] * 3]
    use = tc <= kn[-1]
    Bm = np.zeros((tc.size, t.size - 4))
    Bm[use] = BSpline.design_matrix(tc[use], t, 3).toarray()
    p = Bm.shape[1]
    D2 = np.diff(np.eye(p), 2, axis=0)
    best = None
    for pen in pen_grid:
        P = np.eye(p) + pen * D2.T @ D2
        c = np.zeros(p)
        for _ in range(50):
            dl = Bm @ c
            m = mu * np.exp(dl)
            g = Bm.T @ (cnt - m) - P @ c
            Hn = Bm.T @ (Bm * m[:, None]) + P
            step = np.linalg.solve(Hn, g)
            c = c + step
            if np.max(np.abs(step)) < 1e-8:
                break
        dl = Bm @ c
        m = mu * np.exp(dl)
        Hn = Bm.T @ (Bm * m[:, None]) + P
        ll = float((cnt * dl - m + mu).sum()) - 0.5 * c @ P @ c
        ev = ll + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * np.linalg.slogdet(Hn)[1]
        if best is None or ev > best[0]:
            Ci = np.linalg.inv(Hn)
            sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Bm, Ci, Bm), 0.0))
            best = (ev, dl, sd, pen)
    return best[1], best[2], best[3]


def fit_empirical(v, base, s0=None, dt=0.05, x_max=2000.0, alpha=10.0, side="neg", e_safe=None, h0=0.6, c_exc=None,
                  method="spline"):
    """non-parametric noise density from the pixel residuals v (no pixel is ever binned for the likelihoods; only
    the density estimate uses a histogram).

    Negative side (pure noise): histogram of |v|, v < 0, on t = asinh(|v| / s0) (resolution ~dt s0 in the core,
    ~dt relative in the tail), adaptively kernel-smoothed (_smooth_counts), and combined with the parametric base
    model as a Dirichlet-process posterior predictive: p_k = (m_k + alpha P_k) / (n + alpha), P_k the base
    probability of bin k. Where data are dense the histogram wins, where they are absent (far tail) the base model
    (with its 1 / n excursion allowance) is the floor. info['rel_err'] = Dirichlet posterior sd / mean per bin.

    side = 'neg'  : symmetric, f(+e) = f(-e) (the base model's assumption)
           'asym' : measured asymmetry. For 0 < e < e_safe (default: where an n = 1 pixel cannot reach, min(0.45,
                    1 - 6 s0)) the positive side gets its own histogram (pure noise there too); beyond e_safe the
                    positive density is c x the negative one, c the positive-excursion scale (fit_excursion_scale;
                    c_exc given or 1). Normalised over both sides.
    Returns EmpiricalNoise."""
    s0 = float(base.core if s0 is None else s0)
    x = -v[np.isfinite(v) & (v < 0)]
    n = x.size
    K = int(np.ceil(np.arcsinh(x_max / s0) / dt))
    te = np.arange(K + 1) * dt
    xe = s0 * np.sinh(te)
    cnt = np.bincount(np.minimum((np.arcsinh(x / s0) / dt).astype(np.int64), K - 1), minlength=K).astype(float)
    P = np.diff(_half_cdf(base, xe)); P = np.maximum(P, 1e-300); P /= P.sum()
    m = _smooth_counts(cnt, dt, h0)
    tc = (np.arange(K) + 0.5) * dt
    if method == "spline":
        # base x exp(smooth correction) (_spline_correction), fitted on the populated range of t
        t_hi = float(tc[np.nonzero(cnt)[0][-1]]) if cnt.any() else 1.0
        dl, sd, pen = _spline_correction(cnt, n * P, tc, t_hi)
        p = P * np.exp(dl); p /= p.sum()
        rel = sd
    else:
        # Dirichlet-process posterior predictive with the base as prior measure (alpha pseudo-pixels)
        p = (m + alpha * P) / (n + alpha)
        rel = np.sqrt(p * (1 - p) / (n + alpha + 1)) / p
        pen = np.nan
    # per-side density at the bin centres: half of the folded probability per unit |e|
    jac = dt * s0 * np.cosh(tc)                    # bin width in |e|
    lf = np.log(0.5 * p / jac)
    info = dict(mode=side, method=method, pen=pen, n_neg=int(n), alpha=alpha, dt=dt, rel_err=rel, counts=cnt,
                e_safe=np.nan, c_exc=1.0)
    lf_pos = lf.copy()
    if side == "asym":
        es = min(0.45, 1.0 - 6.0 * s0) if e_safe is None else e_safe
        es = max(es, 2.0 * s0)
        y = v[np.isfinite(v) & (v > 0) & (v < es)]
        kmax = int(np.arcsinh(es / s0) / dt)                 # bins fully below e_safe
        k1 = int(np.arcsinh(1.0 / s0) / dt)                  # first bin above 1 e- (excursion band)
        cp = np.bincount((np.arcsinh(y / s0) / dt).astype(np.int64), minlength=K)[:K].astype(float)
        mp = _smooth_counts(cp, dt, h0)
        # log(positive / negative) below e_safe from the two adaptively smoothed histograms (pseudo-count 0.5)
        lr0 = np.zeros(K)
        if method == "spline" and kmax > 4:
            # positive / negative ratio as a smooth correction of the negative-side density (same machinery): with
            # symmetric noise the expected positive count of bin k equals the negative one, n p_k
            lr0[:kmax], _, _ = _spline_correction(cp[:kmax], n * p[:kmax], tc[:kmax], float(tc[kmax - 1]))
        else:
            lr0[:kmax] = np.log((mp[:kmax] + 0.5) / (m[:kmax] + 0.5))
        safe_last = float(lr0[max(kmax - 1, 0)])

        def build(c_tail, c_exc):
            lr = lr0.copy()
            lr[kmax:k1] = np.log(c_tail); lr[max(k1, kmax):] = np.log(c_exc)
            lp_ = lf + lr
            z = (np.exp(lf) * jac).sum() + (np.exp(lp_) * jac).sum()      # each side integrates to ~0.5
            return lf - np.log(z), lp_ - np.log(z)
        if c_exc == "fit":
            def mk(ct, ce):
                a_, b_ = build(ct, ce)
                return EmpiricalNoise(s0, dt, np.r_[a_[0], 0.5 * (a_[1:] + a_[:-1])], np.r_[b_[0], 0.5 * (b_[1:] + b_[:-1])], base)
            (ct, ce), ll2 = fit_excursion_scale(v, mk, es)
            info.update(ll_scale=ll2)
        elif c_exc is None:
            ct = ce = 1.0
        else:
            ct, ce = (c_exc, c_exc) if np.isscalar(c_exc) else c_exc
        lf, lf_pos = build(ct, ce)
        info.update(e_safe=es, c_tail=float(ct), c_exc=float(ce), ratio_safe=float(np.exp(safe_last)), k_safe=kmax)
    # tabulate at the grid nodes t_k = k dt (k = 0..K-1) for EmpiricalNoise's interpolation: centre values shifted
    lfn = np.r_[lf[0], 0.5 * (lf[1:] + lf[:-1])]
    lfp = np.r_[lf_pos[0], 0.5 * (lf_pos[1:] + lf_pos[:-1])]
    return EmpiricalNoise(s0, dt, lfn, lfp, base, info)


def fit_excursion_scale(v, mk, e_safe, c_grid=None, K=12, n_iter=25, max_pts=10000, seed=0):
    """positive-side noise scales beyond e_safe: the positive density is c_tail x the negative one for
    e_safe < e < 1 e- and c_exc x for e > 1 e- (the excursion band). Inhomogeneous Poisson-process likelihood of the
    pixel values v > e_safe,
        intensity(v) = Npix [ f_c(v) + sum_{n=1..K} pi_n f_c(v - n) ],   minus its integral over v > e_safe,
    with free occupation fractions pi_n (EM at every grid point: dark charge / events of any n) and f_c = mk(c_tail,
    c_exc); only v <= K - 0.5 is used (above, X-ray / track pixels would count as excursions), the integral runs to
    the same limit. Prior: both log-uniform 1e-2..1e2 (9 x 9 grid, steps of 10^0.5). Charge sits on the integer comb, an excursion need not,
    so the pixels between the comb peaks carry the information. Returns ((c_tail, c_exc) posterior means of log c,
    log-likelihood grid)."""
    c_grid = np.geomspace(1e-2, 1e2, 9) if c_grid is None else c_grid
    vv = v[np.isfinite(v)]
    npix = vv.size
    vmax = K - 0.5
    x = vv[(vv > e_safe) & (vv <= vmax)]
    if x.size > max_pts:
        x = np.random.default_rng(seed).choice(x, max_pts, replace=False)
    wscale = ((vv > e_safe) & (vv <= vmax)).sum() / max(x.size, 1)
    n = np.arange(K + 1)
    g = np.geomspace(e_safe, vmax, 2000)
    ll = np.zeros((c_grid.size, c_grid.size))
    for i, ct in enumerate(c_grid):
        for j, ce in enumerate(c_grid):
            nm = mk(ct, ce)
            lp = nm.logpdf(x[:, None] - n[None, :])
            S0 = float(np.trapz(np.exp(nm.logpdf(g)), g))
            pi = np.full(K, 1e-4)
            for _ in range(n_iter):
                t = lp + np.log(np.r_[1.0, pi])[None, :]
                r = np.exp(t - special.logsumexp(t, axis=1, keepdims=True))
                pi = np.maximum(wscale * r[:, 1:].sum(0) / npix, 1e-12)
            t = lp + np.log(np.r_[1.0, pi])[None, :]
            ll[i, j] = float(wscale * special.logsumexp(t, axis=1).sum() - npix * (S0 + pi[:K - 1].sum()))
    w = np.exp(ll - ll.max())
    lg = np.log(c_grid)
    ct = float(np.exp((w.sum(1) * lg).sum() / w.sum())); ce = float(np.exp((w.sum(0) * lg).sum() / w.sum()))
    return (ct, ce), ll
