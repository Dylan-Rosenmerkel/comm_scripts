"""The constrained fit model of the Am-241 spectrum between 10 and 30 keV (the figure of merit of the note).

Why a constrained model
-----------------------
The Np L x-ray "lines" of Am-241 are really groups of many lines (22 in Table 3 of Lepy, Plagnard, Ferreux,
Appl. Radiat. Isot. 66 (2008) 715) closer together than our energy resolution. Fitting each visible bump with a free
Gaussian makes neighbouring lines trade counts and drift. Instead every line is put at its tabulated energy and the
lines within a group keep their tabulated intensity ratios; only a handful of physics parameters are free:

    mu(E) = b0 + b1 (E-20) + b2 (E-20)^2                                   smooth background
          + A_C / 2 * erfc((E - k * 11.25 keV) / (sqrt(2) s_C))             Compton edge of the 59.54 keV gamma ray
          + sum_groups A_g * sum_lines r_i * Gauss(E; k * E_i, sigma_i)     the lines

  k          one energy scale for all lines: observed energy = k * true energy (k = 1 means perfectly calibrated)
  sigma_i    width of line i, sigma^2 = s0^2 + s1 * E (s0 ~ readout noise, s1 ~ charge statistics)
  A_g        total counts in group g (Ll, La, Leta, Lb, Lg, the 26.3 keV gamma, optionally an extra K group)
  r_i        relative intensity of line i inside its group (emission probabilities, normalised to 1)

The Compton edge: a 59.54 keV photon that Compton-scatters in the silicon leaves at most
E_C = E * 2E / (m_e c^2 + 2E) = 11.25 keV with the electron, so the scattered-photon continuum steps down there.

Fitting: the spectrum is histogrammed (0.2 keV bins) and the Poisson likelihood of the bin counts is maximised
(more correct than least squares for bins with few counts). Uncertainties come from the curvature of the likelihood
at its maximum (inverse of the numerical Hessian). Goodness of fit: the deviance per degree of freedom, about 1 for a
model that describes the data within counting fluctuations.

Contents: NP (the line list), EXTRA (optional K-fluorescence groups), Model (build / evaluate / fit), summary(), and
fit_595() for the separate 59.54 keV peak fit.
"""
import numpy as np
from scipy import optimize, special

E_C = 11.25          # Compton edge of the 59.54 keV gamma ray for a free electron [keV]

# Np L lines of Am-241, Lepy et al. 2008 Table 3: group -> [(energy keV, emission probability % per decay)].
# Where Table 3 gives one probability for two lines, the pair sits at the midpoint (Lb + Lb5 at 17.39 keV,
# Lg8 + Lg3 at 21.30 keV). The 26.34 keV gamma ray of Am-241 is one more "group" of one line.
NP = {
    "Ll":   [(11.87, 0.837), (12.24, 0.0260), (13.18, 0.0199)],
    "La":   [(13.76, 1.398), (13.95, 11.60)],
    "Leta": [(15.86, 0.404)],
    "Lb":   [(16.11, 0.248), (16.79, 0.339), (16.84, 2.451), (17.06, 1.736), (17.39, 0.594), (17.75, 11.83),
             (17.99, 1.310), (18.58, 0.054), (18.76, 0.047)],
    "Lg":   [(20.10, 0.087), (20.78, 2.94), (21.10, 0.467), (21.30, 0.520), (21.49, 0.567), (22.12, 0.173),
             (22.40, 0.069)],
    "g26":  [(26.34, 2.40)],
}

# Optional K-fluorescence groups (standard energies, X-ray Data Booklet): K-alpha1 : K-alpha2 = 2 : 1 and K-beta fixed
# to kb_ka times the K-alpha total. In the note the 22.1/24.9 keV lines are fitted as "Ag" (silver, to be confirmed);
# "Sn" is one of the alternatives tested.
EXTRA = {
    "Ag": {"Ka": [(22.16, 2), (21.99, 1)], "Kb": [(24.94, 1)], "kb_ka": 0.20},
    "Sn": {"Ka": [(25.27, 2), (25.04, 1)], "Kb": [(28.49, 1)], "kb_ka": 0.21},
}


class Model:
    """The 10-30 keV model; evaluate it with model(x, p) and fit it with model.fit(energies).

    extra       None (Np lines + 26.3 keV only), "Ag" or "Sn" (add that K group), or "X" (add one Gaussian with free
                energy and size)
    lo, hi, bw  fit range and bin width [keV]
    free_edge   fit the Compton-edge position freely (observed keV, last parameter) instead of fixing it at k * 11.25

    Parameter vector p: k, s0, s1, b0, b1, b2, A_compton, s_compton, then one total count per group (in the order of
    self.groups), then [X_E, X] if extra == "X", then [edge_E] if free_edge. self.names lists them.
    """

    def __init__(self, extra=None, lo=10.0, hi=30.0, bw=0.2, free_edge=False):
        self.extra, self.lo, self.hi, self.bw = extra, lo, hi, bw
        self.free_edge = free_edge
        self.groups = []                                   # (name, [(E_true, relative intensity)])
        for g, lines in NP.items():
            tot = sum(p for _, p in lines)
            self.groups.append((g, [(e, p / tot) for e, p in lines]))
        if extra in EXTRA:
            x = EXTRA[extra]; wa = sum(w for _, w in x["Ka"])
            lines = [(e, w / wa) for e, w in x["Ka"]] + [(e, x["kb_ka"] * w) for e, w in x["Kb"]]
            self.groups.append((f"{extra} K", lines))
        self.free_x = extra == "X"
        self.names = ["k", "s0", "s1", "b0", "b1", "b2", "A_compton", "s_compton"] + [g for g, _ in self.groups]
        if self.free_x:
            self.names += ["X_E", "X"]
        if self.free_edge:
            assert not self.free_x
            self.names += ["edge_E"]

    def __call__(self, x, p):
        """Expected counts per bin at bin centres x [keV] for parameters p (see the class docstring)."""
        k, s0, s1, b0, b1, b2, ac, sc = p[:8]
        edge = p[-1] if self.free_edge else E_C * k
        y = b0 + b1 * (x - 20) + b2 * (x - 20) ** 2 + ac * 0.5 * special.erfc((x - edge) / (np.sqrt(2) * sc))
        amps = p[8:8 + len(self.groups)]
        for (g, lines), a in zip(self.groups, amps):
            for e, rel in lines:
                mu = k * e; sg = np.sqrt(s0 ** 2 + s1 * mu)
                # a * rel counts spread as a normalised Gaussian; times the bin width -> counts per bin
                y = y + a * rel * self.bw / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sg) ** 2)
        if self.free_x:
            xe, xa = p[-2:]; mu = k * xe; sg = np.sqrt(s0 ** 2 + s1 * mu)
            y = y + xa * self.bw / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sg) ** 2)
        return y

    def components(self, x, p):
        """{group name: curve} for each line group alone (background and other groups set to zero), for plotting."""
        out = {}
        for i, (g, _) in enumerate(self.groups):
            q = np.array(p, float); q[3:7] = 0; q[8:8 + len(self.groups)] = 0; q[8 + i] = p[8 + i]
            if self.free_x: q[-1] = 0
            out[g] = self(x, q)
        if self.free_x:
            q = np.array(p, float); q[3:7] = 0; q[8:8 + len(self.groups)] = 0; out["X"] = self(x, q)
        return out

    def fit(self, E, k0=1.0):
        """Fit the cluster energies E [keV] (one entry per cluster) by binned Poisson likelihood.

        k0 is the starting energy scale (about 0.985 after the gain correction, 0.94 without it). The fit is started
        from 9 points (three scales around k0 times three width settings), each refined with L-BFGS-B then Powell,
        and the best is kept - the likelihood has local minima when lines overlap.

        Returns a dict: x, h (bin centres and counts), p, err, cov (best parameters, their uncertainties and
        covariance), nll (minimum of -ln L), chi2ndf (Pearson), devndf (deviance / ndf), names, model.
        """
        h, ed = np.histogram(E, bins=np.arange(self.lo, self.hi + 1e-9, self.bw)); x = 0.5 * (ed[1:] + ed[:-1])
        n_g = len(self.groups)
        tail = max(np.median(h[-10:]), 0.5)
        p0 = [k0, 0.2, 0.004, tail, 0.0, 0.0, max(h[:3].mean() - tail, 1), 0.6] + [max(h.sum() * 0.05, 5)] * n_g
        lb = [0.90, 0.05, 0.0, 0, -np.inf, -np.inf, 0, 0.15] + [0] * n_g
        ub = [1.08, 0.8, 0.05, np.inf, np.inf, np.inf, np.inf, 2.5] + [np.inf] * n_g
        if self.free_x:
            p0 += [25.0, 50]; lb += [23.0, 0]; ub += [26.5, np.inf]
        if self.free_edge:
            p0 += [E_C * k0]; lb += [9.0]; ub += [13.5]
        bounds = list(zip(lb, ub))

        def nll(p):
            """Negative Poisson log-likelihood of the bin counts (constant terms dropped)."""
            mu = np.clip(self(x, p), 1e-9, None)
            return np.sum(mu - h * np.log(mu))

        best = None
        starts = [(kk, s0, s1) for kk in (k0, k0 - 0.012, k0 + 0.012) for s0, s1 in ((0.1, 0.005), (0.3, 0.001), (0.2, 0.003))]
        for kk, ss0, ss1 in starts:
            q0 = list(p0); q0[0], q0[1], q0[2] = kk, ss0, ss1
            r = optimize.minimize(nll, q0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=20000, maxfun=200000))
            r = optimize.minimize(nll, r.x, method="Powell", bounds=bounds, options=dict(maxiter=40000, xtol=1e-6, ftol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        cov = self._cov(nll, p)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        mu = self(x, p)
        pear = np.sum((h - mu) ** 2 / np.maximum(mu, 1e-9)) / max(len(h) - len(p), 1)
        # deviance: 2 * sum[mu - n + n ln(n / mu)], the Poisson analogue of chi^2
        dev = 2 * np.sum(mu - h + np.where(h > 0, h * np.log(np.where(h > 0, h, 1) / mu), 0)) / max(len(h) - len(p), 1)
        return dict(x=x, h=h, p=p, err=err, cov=cov, nll=best.fun, chi2ndf=pear, devndf=dev, names=self.names, model=self)

    @staticmethod
    def _cov(f, p):
        """Covariance matrix: inverse of the numerical Hessian of the negative log-likelihood f at p.

        The Hessian is built from central finite differences with steps of 0.1 % of each parameter (at least 1e-5).
        A parameter sitting on a bound gives a degenerate direction; pinv then returns ~0 uncertainty for it, so treat
        zero errors as "not determined", not as "exact".
        """
        n = len(p); H = np.zeros((n, n)); s = np.maximum(np.abs(p) * 1e-3, 1e-5)
        f0 = f(p)
        for i in range(n):
            for j in range(i, n):
                pp = np.array(p, float)
                if i == j:
                    pp[i] += s[i]; fp = f(pp); pp[i] -= 2 * s[i]; fm = f(pp)
                    H[i, i] = (fp - 2 * f0 + fm) / s[i] ** 2
                else:
                    def g(di, dj):
                        q = np.array(p, float); q[i] += di; q[j] += dj; return f(q)
                    H[i, j] = H[j, i] = (g(s[i], s[j]) - g(s[i], -s[j]) - g(-s[i], s[j]) + g(-s[i], -s[j])) / (4 * s[i] * s[j])
        try:
            return np.linalg.pinv(H)
        except Exception:
            return np.full((n, n), np.nan)


def summary(r):
    """{parameter name: (value, uncertainty)} from a Model.fit result."""
    names, p, e = r["names"], r["p"], r["err"]
    d = {nm: (v, s) for nm, v, s in zip(names, p, e)}
    return d


def fit_595(E):
    """Separate fit of the 59.54 keV gamma-ray peak: Gaussian + linear background.

    Histogram 55-64 keV in 0.15 keV bins, find the maximum, and fit from 1.2 keV below to 1.5 keV above it - the low
    side is cut short because incompletely collected 59.54 keV hits form a tail there. Bin errors sqrt(n); the mean's
    uncertainty is scaled by sqrt(chi2/ndf) when that is above 1.

    Returns (mean, mean error, sigma, amplitude significance) in keV, or None if the fit fails.
    """
    h, ed = np.histogram(E, bins=np.arange(55, 64, 0.15)); x = 0.5 * (ed[1:] + ed[:-1])
    pk = x[np.argmax(h)]
    sel = (x > pk - 1.2) & (x < pk + 1.5)
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - pk)
    try:
        p, cov = optimize.curve_fit(f, x[sel], h[sel], p0=[h.max(), pk, 0.3, 1, 0], sigma=np.sqrt(np.maximum(h[sel], 1)),
                                    bounds=([0, pk - 1, 0.05, -np.inf, -np.inf], [np.inf, pk + 1, 2, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h[sel] - f(x[sel], *p)) / np.sqrt(np.maximum(h[sel], 1))) ** 2) / max(sel.sum() - 5, 1)
    return p[1], np.sqrt(cov[1, 1]) * np.sqrt(max(chi, 1)), abs(p[2]), p[0] / np.sqrt(cov[0, 0])
