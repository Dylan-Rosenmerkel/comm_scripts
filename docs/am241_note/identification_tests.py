#!/usr/bin/env python3
"""Step 5 of the note: what are the two extra lines at ~22.1 and ~24.9 keV? (note Sec. 5.6, Tables 7-10)

The Np L lines and the 26.3 keV gamma ray alone do not describe the 10-30 keV spectrum: two more lines are needed.
This script fits the spectrum (EXT4, sigma_xy < 0.7, gain-corrected per gain period) with the constrained model of
amfit.py plus, in turn, each candidate explanation, and compares the likelihoods:

  none    Np L lines + 26.3 keV only (Lepy Table 3 ratios)
  free2   + two Gaussians with free energy and size      -> where the extra lines are and how big
  lg4     Np Lg4/Lg13 sizes freed (a "downshifted, enhanced" Lg4 explaining the 22.1 keV line)
  lg4x    the same + one free line for the 24.9 keV structure
  l1x     all L1-subshell lines (Lg4, Lg13, Lb4, Lb3) freed + free line: if Lg4 were enhanced, the other lines that
          fill the same L1 vacancies would have to be enhanced too
  am/amx  Am L self-fluorescence lines (labelled in Lepy et al. Figs. 3-5) with free sizes [+ free line]
  ag, sn  silver / tin K fluorescence (amfit.EXTRA)
and a conversion-electron test: if the extra lines were Am-241 conversion electrons (ec(4,2)L + ec(2,1)M + ec(3,1)L,
20.3-25.8 keV, 12.8 per 100 decays), the much stronger ec(2,0)L line at 37.11/37.94/41.93 keV (30.2 per 100 decays)
would be there too; its counts are fitted.

How to compare models: -2 Delta ln L = 2 (nll_model - nll_best); a model worse by many units for the same number of
parameters describes the data clearly less well. Nested models (more free lines) can only fit better, which the
fitting procedure enforces (see fit_with).

Output: <note-dir>/tables/{hypotheses,extra_lines,l1_lines,conversion_electrons}.tex,
        <note-dir>/hypotheses.json, and a printed summary including the Am self-fluorescence line sizes.
Run (after extract_clusters.py):   python identification_tests.py [--run-dir DIR] [--note-dir DIR]
"""
import argparse
import json
import os

import numpy as np
from scipy import optimize

from amfit import Model, NP
from common import CUT, add_path_args, load_clusters, paths

# Am L self-fluorescence lines (americium atoms of the source excited by its own photons), from Lepy et al. Figs. 3-5
AM = [("Am La2", 14.41), ("Am La1", 14.62), ("Am Lb2,15", 17.66), ("Am Lb1", 18.86), ("Am Lg1", 22.07)]
# L1-subshell lines of Np: they all fill vacancies in the L1 subshell, so their intensity ratios are fixed by atomic
# transition rates - an enhanced Lg4 alone is not possible
LG4, LG13, LB4, LB3 = ("Lg4", 22.12), ("Lg13", 22.40), ("Lb4", 17.06), ("Lb3", 17.99)
K_NOTE = 0.984          # energy scale of the main fit, used to place the conversion-electron lines


def fit_with(E, extra_fixed, n_free, base=None):
    """Fit the constrained Np model plus extra Gaussians to the energies E.

    extra_fixed   [(name, true energy keV)]: lines at fixed energies with free sizes (added on top of the model)
    n_free        number of extra lines with free energy and size (0, 1 or 2; started near 22.1 and 24.9 keV)
    base          the amfit.Model to extend (default: Np lines + 26.3 keV only)

    All extra lines share the model's energy scale k and width model sigma^2 = s0^2 + s1 E. The base model is fitted
    first and its best parameters start the extended fit. Each start is refined with L-BFGS-B then Powell, and the
    starting point itself is kept as a candidate - so a fit with more lines can never end worse than the base fit,
    even when a minimiser stalls (it did on the steep Compton edge). Extra fixed-energy lines are started both large
    (20 counts) and at zero.

    Returns dict(p, err, nll, dev, npar, nb): parameters (the base model's nb parameters first, then the extra lines),
    their uncertainties, minimum -ln L, deviance/ndf and the number of parameters.
    """
    base = base or Model(None)
    h, ed = np.histogram(E, bins=np.arange(base.lo, base.hi + 1e-9, base.bw)); x = 0.5 * (ed[1:] + ed[:-1])
    r0 = base.fit(E, 0.985); p0b = list(r0["p"]); nb = len(p0b)

    def model(p):
        y = base(x, p[:nb]); k, s0, s1 = p[0], p[1], p[2]
        j = nb
        for _, e0 in extra_fixed:
            a = p[j]; j += 1; mu = k * e0; sg = np.sqrt(s0 ** 2 + s1 * mu)
            y = y + a * base.bw / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sg) ** 2)
        for _ in range(n_free):
            e0, a = p[j], p[j + 1]; j += 2; mu = k * e0; sg = np.sqrt(s0 ** 2 + s1 * mu)
            y = y + a * base.bw / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sg) ** 2)
        return y

    def nll(p):
        mu = np.clip(model(p), 1e-9, None); return np.sum(mu - h * np.log(mu))

    lbb = [0.90, 0.05, 0.0, 0, -np.inf, -np.inf, 0, 0.15] + [0] * (nb - 8)
    ubb = [1.08, 0.8, 0.05, np.inf, np.inf, np.inf, np.inf, 2.5] + [np.inf] * (nb - 8)
    best = None
    starts_free = [(22.1, 24.9), (21.9, 24.7), (22.3, 25.1)][: max(1, n_free and 3)]
    for st, a0 in [(st, a0) for st in starts_free for a0 in ((20.0, 0.0) if extra_fixed else (20.0,))]:
        p0 = p0b + [a0] * len(extra_fixed); lb = lbb + [0] * len(extra_fixed); ub = ubb + [np.inf] * len(extra_fixed)
        for i in range(n_free):
            e_st = st[i] if n_free == 2 else 24.9
            p0 += [e_st, 100]; lb += [e_st - 1.0, 0]; ub += [e_st + 1.0, np.inf]
        cands = [(nll(np.array(p0, float)), np.array(p0, float))]          # the start itself is a candidate
        r = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=list(zip(lb, ub)), options=dict(maxiter=40000, maxfun=400000))
        cands.append((r.fun, r.x))
        r = optimize.minimize(nll, r.x, method="Powell", bounds=list(zip(lb, ub)), options=dict(maxiter=60000, xtol=1e-7, ftol=1e-10))
        cands.append((r.fun, r.x))
        f, x_ = min(cands, key=lambda c: c[0])
        if best is None or f < best[0]:
            best = (f, x_)
    p = best[1]; C = Model._cov(nll, p); err = np.sqrt(np.clip(np.diag(C), 0, None))
    mu = model(p); npar = len(p)
    dev = 2 * np.sum(mu - h + np.where(h > 0, h * np.log(np.where(h > 0, h, 1) / mu), 0)) / (len(h) - npar)
    return dict(p=p, err=err, nll=best[0], dev=dev, npar=npar, nb=nb)


def fit_fixed_lines(E, lo, hi, lines, bw=0.25):
    """Total counts in Gaussians at k * lines (true keV) on a quadratic background, fitted in [lo, hi].

    Used for the conversion-electron test. Free: background (3 terms), one common width (0.25-1 keV) and one size per
    line. Returns (total counts, uncertainty) with the uncertainty from the covariance of the sizes.
    """
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); x = 0.5 * (ed[1:] + ed[:-1]); xm = 0.5 * (lo + hi)

    def mu(q):
        y = q[0] + q[1] * (x - xm) + q[2] * (x - xm) ** 2
        for a, e0 in zip(q[4:], lines):
            y = y + a * bw / (q[3] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - K_NOTE * e0) / q[3]) ** 2)
        return np.clip(y, 1e-9, None)
    nll = lambda q: np.sum(mu(q) - h * np.log(mu(q)))
    b = [(0, None), (None, None), (None, None), (0.25, 1.0)] + [(0, None)] * len(lines)
    q = optimize.minimize(nll, [np.median(h), 0, 0, 0.45] + [10] * len(lines), method="L-BFGS-B", bounds=b).x
    q = optimize.minimize(nll, q, method="Powell", bounds=b).x
    C = Model._cov(nll, q)
    J = np.zeros(len(q)); J[4:] = 1                         # total of the line amplitudes
    return q[4:].sum(), float(np.sqrt(max(J @ C @ J, 0)))


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p_ = paths(ap.parse_args())
    tab_dir = os.path.join(p_.note, "tables"); os.makedirs(tab_dir, exist_ok=True)
    d = load_clusters(p_.work, 4)
    E = d["Energy"][d["sxy"] < CUT]

    fits = {
        "none":   ("Np L lines + $\\gamma$ 26.3 only (L\\'epy Table~3 ratios)", fit_with(E, [], 0)),
        "free2":  ("two lines, free energy and size", fit_with(E, [], 2)),
        "lg4":    ("Np L$\\gamma_4$/L$\\gamma_{13}$ sizes free (no line at 24.9)", fit_with(E, [LG4, LG13], 0)),
        "lg4x":   ("Np L$\\gamma_4$/L$\\gamma_{13}$ sizes free + free line", fit_with(E, [LG4, LG13], 1)),
        "l1x":    ("all L1-shell lines (L$\\gamma_4$, L$\\gamma_{13}$, L$\\beta_4$, L$\\beta_3$) free + free line",
                   fit_with(E, [LG4, LG13, LB4, LB3], 1)),
        "am":     ("Am L self-fluorescence (5 lines, sizes free)", fit_with(E, AM, 0)),
        "amx":    ("Am L self-fluorescence + free line", fit_with(E, AM, 1)),
    }
    mod = {"ag": ("Ag K lines (K$\\alpha$ 22.16/21.99, K$\\beta$ 24.94, K$\\beta$/K$\\alpha$ fixed 0.20)", Model("Ag").fit(E, 0.985)),
           "sn": ("Sn K lines (K$\\alpha$ 25.27/25.04, K$\\beta$ 28.49)", Model("Sn").fit(E, 0.985))}
    best = min([v[1]["nll"] for v in fits.values()] + [v[1]["nll"] for v in mod.values()])
    out = {}

    # --- model comparison table
    with open(os.path.join(tab_dir, "hypotheses.tex"), "w") as fh:
        for key in ("none", "free2", "lg4", "lg4x", "l1x", "am", "amx"):
            lab, r = fits[key]
            out[key] = dict(two_dnll=2 * (r["nll"] - best), npar=r["npar"], dev=r["dev"])
            fh.write(f"{lab} & {r['npar']} & {2 * (r['nll'] - best):.1f} & {r['dev']:.2f} \\\\\n")
        for key in ("ag", "sn"):
            lab, r = mod[key]
            out[key] = dict(two_dnll=2 * (r["nll"] - best), npar=len(r["p"]), dev=r["devndf"])
            fh.write(f"{lab} & {len(r['p'])} & {2 * (r['nll'] - best):.1f} & {r['devndf']:.2f} \\\\\n")

    # --- the two extra lines with free energy and size
    r = fits["free2"][1]; nb, p, e = r["nb"], r["p"], r["err"]
    fl = [dict(E=p[nb + 2 * i], dE=e[nb + 2 * i], n=p[nb + 2 * i + 1], dn=e[nb + 2 * i + 1]) for i in range(2)]
    ratio = fl[1]["n"] / fl[0]["n"]; dratio = ratio * np.hypot(fl[1]["dn"] / fl[1]["n"], fl[0]["dn"] / fl[0]["n"])
    out["free_lines"] = fl; out["ratio"] = (ratio, dratio); out["k_free"] = p[0]
    with open(os.path.join(tab_dir, "extra_lines.tex"), "w") as fh:
        for i, f in enumerate(fl):
            # observed energy (k * E), true energy, counts
            fh.write(f"line {i + 1} & {p[0] * f['E']:.3f} & {f['E']:.3f} $\\pm$ {f['dE']:.3f} & {f['n']:.0f} $\\pm$ {f['dn']:.0f} \\\\\n")

    # --- L1-subshell test: tabulated sizes vs fitted extra counts
    rl = fits["l1x"][1]; nb, p, e = rl["nb"], rl["p"], rl["err"]
    lg = dict(NP["Lg"]); lb = dict(NP["Lb"]); ig = 8 + list(NP).index("Lg"); ib = 8 + list(NP).index("Lb")
    tab = {"Lg4": p[ig] * lg[22.12] / sum(lg.values()), "Lg13": p[ig] * lg[22.40] / sum(lg.values()),
           "Lb4": p[ib] * lb[17.06] / sum(lb.values()), "Lb3": p[ib] * lb[17.99] / sum(lb.values())}
    names = {"Lg4": ("L$\\gamma_4$", 22.12), "Lg13": ("L$\\gamma_{13}$", 22.40), "Lb4": ("L$\\beta_4$", 17.06),
             "Lb3": ("L$\\beta_3$", 17.99)}
    rh = fits["lg4x"][1]; boost = (rh["p"][rh["nb"]] + tab["Lg4"]) / tab["Lg4"]
    out["lg4_boost"] = boost; out["l1"] = {}
    with open(os.path.join(tab_dir, "l1_lines.tex"), "w") as fh:
        for i, key in enumerate(("Lg4", "Lg13", "Lb4", "Lb3")):
            nm, en = names[key]
            out["l1"][key] = dict(tab=tab[key], extra=p[nb + i], err=e[nb + i])
            fh.write(f"{nm} & {en:.2f} & {tab[key]:.0f} & {tab[key] * boost:.0f} & {p[nb + i]:.0f} $\\pm$ {e[nb + i]:.0f} \\\\\n")

    # --- conversion electrons: the strong ec(2,0)L lines at 37.11/37.94/41.93 keV (30.2 per 100 decays)
    tot, dtot = fit_fixed_lines(E, 33, 46, [37.11, 37.94, 41.93])
    n22 = fl[0]["n"] + fl[1]["n"]; expected = n22 * 30.2 / (8.8 + 3.7 + 0.31)
    out["ce"] = dict(observed=tot, err=dtot, expected=expected)
    with open(os.path.join(tab_dir, "conversion_electrons.tex"), "w") as fh:
        fh.write(f"ec$_{{4,2}}$L + ec$_{{2,1}}$M + ec$_{{3,1}}$L & 20.3--25.8 & 12.8 & -- & {n22:.0f} (two extra lines) \\\\\n")
        fh.write(f"ec$_{{2,0}}$L & 37.11 / 37.94 / 41.93 & 30.2 & $\\geq${expected:.0f} & {tot:.0f} $\\pm$ {dtot:.0f} \\\\\n")

    # --- Am self-fluorescence line sizes (quoted in the text)
    r = fits["am"][1]; nb, p, e = r["nb"], r["p"], r["err"]
    out["am_lines"] = {nm: dict(counts=p[nb + i], err=e[nb + i]) for i, (nm, _) in enumerate(AM)}

    json.dump(out, open(os.path.join(p_.note, "hypotheses.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
