"""Skip-resolved transient test for candidate pixels (wip/lowdc/bayes): hook for per-skip data.

A skipper pixel is read M times non-destructively; the image value is the mean of the M samples x_k. Real charge is
the SAME in every sample (it is measured, not created, by the skips); an artefact that leaves a few e- in the mean
need not be: a glitch in one sample (spike), charge that appears / disappears during the readout of the pixel (step,
e.g. the first-read artefact of column 0 = charge only in the first skips), telegraph switching (step). Per pixel:

    H_c     x_k = q + d(k) + e_k                          constant level
    H_spike x_k = q + d(k) + a [k = k0] + e_k             k0 uniform on 0..M-1, a flat of width w_spike
    H_step  x_k = q + d(k) + b [k >= k0] + e_k            k0 uniform on 1..M-1, b flat of width w_step

e_k white Gaussian of known single-sample sigma1 (= read noise x sqrt(M)); d(k) an optional polynomial drift of degree
drift_deg (nuisance for 1/f noise / baseline wander; flat priors, the same in every hypothesis so its Occam factors
cancel). Gaussian marginal likelihoods are exact (generalised least squares): adding a column u to the design changes
log Z by 0.5 (u.x_perp)^2 / (sigma1^2 |u_perp|^2) + 0.5 log(2 pi sigma1^2 / |u_perp|^2) - log w, with x_perp, u_perp
the residuals after the H_c design; for all k0 at once with cumulative sums (O(M p^2)).

    logBF = log[ 0.5 mean_k0 Z_spike(k0) + 0.5 mean_k0 Z_step(k0) ] - log Z_c        (transient vs constant)

Usage: transient_logbf(samples, sigma1) for one pixel; a candidate with logBF above a threshold (calibrated on
constant-charge pixels, skip_study.py) is an excursion, not charge, and is removed from the candidates (or its
pixel likelihood replaced by the noise-only one).
"""
import numpy as np
from scipy import special


def _design(M, deg):
    """H_c design: constant + polynomial drift of degree deg (columns scaled to O(1))"""
    t = (np.arange(M) + 0.5) / M * 2 - 1
    return np.vander(t, deg + 1, increasing=True)


def cov_1f(M, rms, alpha=1.0):
    """covariance of the 1/f^alpha component of one_over_f (circulant, from its power spectrum), variance rms^2"""
    f = np.fft.rfftfreq(M)
    p = np.zeros_like(f); p[1:] = f[1:] ** (-alpha)
    c = np.fft.irfft(p, M)
    c = c / c[0] * rms ** 2
    k = np.arange(M)
    return c[np.abs(k[:, None] - k[None, :])]


class SkipModel:
    """the three hypotheses for pixels with M skips; noise covariance C = sigma1^2 I (+ cov, e.g. cov_1f(M, rms)
    measured from the skip series of empty pixels). Whitening W = L^-1 (C = L L^T) is computed once per run setting;
    then every alternative column (spike at k0, step at k0) is a whitened vector u_w and
        log Z_alt(k0) - log Z_c = 0.5 (u_w . x_perp)^2 / nu + 0.5 log(2 pi / nu) - log w,  nu = |u_w,perp|^2."""

    def __init__(self, M, sigma1, cov=None, drift_deg=0, w_spike=None, w_step=None):
        self.M = M
        self.w_spike = 20.0 * M if w_spike is None else w_spike
        self.w_step = 40.0 if w_step is None else w_step
        A = _design(M, drift_deg)
        if cov is None:
            W = np.eye(M) / sigma1
        else:
            L = np.linalg.cholesky(sigma1 ** 2 * np.eye(M) + cov)
            W = np.linalg.solve(L, np.eye(M))           # whitening (lower triangular)
        self.W = W
        self.Aw = W @ A
        self.Gi = np.linalg.inv(self.Aw.T @ self.Aw)
        self.Us = W                                     # spike at k0: column k0
        self.Ut = np.cumsum(W[:, ::-1], axis=1)[:, ::-1][:, 1:]    # step at k0 = 1..M-1: sum_{k >= k0} W[:, k]
        self.nu_s = self._nu(self.Us); self.nu_t = self._nu(self.Ut)

    def _nu(self, U):
        B = self.Aw.T @ U
        return np.maximum((U * U).sum(0) - (B * (self.Gi @ B)).sum(0), 1e-12)

    def logbf(self, X, return_parts=False):
        """log BF transient vs constant for samples X (M,) or (n, M)"""
        X = np.atleast_2d(np.asarray(X, float))
        xw = X @ self.W.T
        xp = xw - (xw @ self.Aw) @ self.Gi @ self.Aw.T
        ds = xp @ self.Us; dt = xp @ self.Ut
        lz_s = 0.5 * ds ** 2 / self.nu_s + 0.5 * np.log(2 * np.pi / self.nu_s) - np.log(self.w_spike)
        lz_t = 0.5 * dt ** 2 / self.nu_t + 0.5 * np.log(2 * np.pi / self.nu_t) - np.log(self.w_step)
        l_s = special.logsumexp(lz_s, axis=1) - np.log(self.M)
        l_t = special.logsumexp(lz_t, axis=1) - np.log(self.M - 1)
        lbf = np.logaddexp(l_s, l_t) - np.log(2.0)
        if return_parts:
            return lbf, dict(spike=l_s, step=l_t, k_spike=np.argmax(lz_s, 1), k_step=np.argmax(lz_t, 1) + 1)
        return lbf if lbf.size > 1 else float(lbf[0])


def transient_logbf(x, sigma1, drift_deg=0, w_spike=None, w_step=None, cov=None):
    """log Bayes factor transient (spike or step) vs constant for one pixel's per-skip samples x (length M, units of
    sigma1, the single-sample noise; cov: optional correlated-noise covariance, e.g. cov_1f). Prior widths: w_spike =
    20 x M (a glitch moving the mean by up to ~20), w_step = 40 (a level change of up to ~40), in the units of x.
    For many pixels of one run setting build SkipModel once and call its logbf on the stacked samples."""
    x = np.asarray(x, float)
    return SkipModel(x.size, sigma1, cov, drift_deg, w_spike, w_step).logbf(x)


def one_over_f(rng, M, rms, alpha=1.0):
    """1/f^alpha noise series of length M with total rms `rms` (FFT shaping of white noise)"""
    f = np.fft.rfftfreq(M)
    amp = np.zeros_like(f); amp[1:] = f[1:] ** (-alpha / 2)
    z = np.fft.irfft(amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)), M)
    return z / max(z.std(), 1e-30) * rms


def simulate_pixel(rng, M, sigma1, q=0.0, kind="charge", amp=0.0, f1_rms=0.0):
    """per-skip samples [e-] of one pixel: 'charge' = constant q; 'spike' = q plus one sample shifted so that the MEAN
    moves by amp e-; 'step' = q plus amp' from a random skip k0 on, amp' chosen so that the mean moves by amp;
    'step_first' = charge present only in the first skips (k < k0, k0 uniform 1..M/10; mean moves by amp). Plus
    white noise sigma1 and an optional 1/f component of rms f1_rms."""
    x = np.full(M, float(q)) + rng.normal(0.0, sigma1, M)
    if f1_rms > 0:
        x += one_over_f(rng, M, f1_rms)
    if kind == "spike":
        x[rng.integers(0, M)] += amp * M
    elif kind == "step":
        k0 = int(rng.integers(1, M))
        x[k0:] += amp * M / (M - k0)
    elif kind == "step_first":
        k0 = int(rng.integers(1, max(M // 10, 2)))
        x[:k0] += amp * M / k0
    return x
