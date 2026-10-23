#!/usr/bin/env python3
"""Step 4 of the note: mean, sigma, FWHM and counts of every peak (note Tables 6, 11 and 12).

For the main fit (EXT4, sigma_xy < 0.7, gain-corrected per gain period) the constrained model of amfit.py does not give
each peak a free mean and width: every line sits at k * E_true and its width follows sigma(E)^2 = s0^2 + s1 * E. So
mean, sigma and FWHM (= 2.355 sigma for one Gaussian) are derived from k, s0 and s1, and their uncertainties are
propagated through the full fit covariance with a numerical Jacobian (propagate()).

Visible peaks are groups of lines (e.g. Np Lb = 9 lines). For a group the script gives the intensity-weighted centroid,
an effective sigma (second moment of the group, so the spread of its lines is included) and the FWHM measured on the
summed curve (blend_fwhm()). Peaks outside 10-30 keV are fitted separately with a free mean and width: the 59.54 keV
gamma ray, the backscatter peak, the Fe-55 Mn Ka/Kb lines, and the position and smearing of the Compton edge.
Everything is done with and without the gain correction.

Output: <work-dir>/peak_table.csv and printed tables.
Run (after extract_clusters.py):   python peak_table.py [--run-dir DIR]
"""
import argparse
import csv
import os
import numpy as np
from scipy import optimize
from amfit import Model, NP, EXTRA
from common import add_path_args, load_clusters, paths

FW = 2 * np.sqrt(2 * np.log(2))          # FWHM / sigma for a Gaussian
NAMES = {11.87: "Ll", 12.24: "Lt", 13.18: "Ls", 13.76: "La2", 13.95: "La1", 15.86: "Leta", 16.11: "Lb6",
         16.79: "Lb15", 16.84: "Lb2", 17.06: "Lb4", 17.39: "Lb + Lb5", 17.75: "Lb1", 17.99: "Lb3", 18.58: "Lb10",
         18.76: "Lb9", 20.10: "Lg5", 20.78: "Lg1", 21.10: "Lg2", 21.30: "Lg8 + Lg3", 21.49: "Lg6", 22.12: "Lg4",
         22.40: "Lg13", 26.34: "gamma 26.3", 22.16: "Ag Ka1", 21.99: "Ag Ka2", 24.94: "Ag Kb"}
GROUP_LABEL = {"Ll": "Np Ll", "La": "Np La", "Leta": "Np Leta", "Lb": "Np Lb", "Lg": "Np Lg", "g26": "gamma 26.3",
               "Ag K": "Ag K"}


def propagate(func, p, cov, eps=1e-6):
    """Values of func(p) (array) and their errors from the covariance, by a numerical Jacobian."""
    f0 = np.asarray(func(p), float)
    J = np.zeros((f0.size, len(p)))
    for j in range(len(p)):
        h = max(abs(p[j]) * eps, 1e-8); q = np.array(p, float); q[j] += h
        J[:, j] = (np.asarray(func(q), float) - f0) / h
    return f0, np.sqrt(np.clip(np.einsum("ij,jk,ik->i", J, cov, J), 0, None))


def blend_fwhm(mus, sgs, w):
    x = np.linspace(min(mus) - 5 * max(sgs), max(mus) + 5 * max(sgs), 6000)
    y = sum(wi * np.exp(-0.5 * ((x - m) / s) ** 2) / s for m, s, wi in zip(mus, sgs, w))
    above = x[y >= y.max() / 2]
    return above[-1] - above[0]


def gfit(E, lo, hi, bw, guess, wguess, wmax):
    """Gaussian + linear: (mean, mean_err, sigma, sigma_err, amplitude significance, counts in the Gaussian, err)."""
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); x = 0.5 * (ed[1:] + ed[:-1])
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - guess)
    p, cov = optimize.curve_fit(f, x, h, p0=[max(h.max() - np.median(h), 1), guess, wguess, np.median(h), 0],
                                sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                bounds=([0, lo, bw / 2, -np.inf, -np.inf], [np.inf, hi, wmax, np.inf, np.inf]))
    chi = np.sum(((h - f(x, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 5, 1)
    cov = cov * max(chi, 1)
    (n, dn), = [propagate(lambda q: [q[0] * abs(q[2]) * np.sqrt(2 * np.pi) / bw], p, cov)]
    e = np.sqrt(np.clip(np.diag(cov), 0, None))
    return p[1], e[1], abs(p[2]), e[2], p[0] / e[0], n[0], dn[0], chi


def main():
    """Fit, derive the peak parameters and write peak_table.csv (in the work folder)."""
    _ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    PATHS = paths(_ap.parse_args())
    os.chdir(PATHS.work)
    d = load_clusters(PATHS.work, 4)
    m = d["sxy"] < 0.7
    rows = []
    for corr, E in (("with", d["Energy"][m]), ("without", d["E_orig"][m])):
        k0 = 0.985 if corr == "with" else 0.94
        mdl = Model("Ag"); r = mdl.fit(E, k0); p, cov = r["p"], r["cov"]
        ig = {g: 8 + i for i, (g, _) in enumerate(mdl.groups)}
        # ---- every individual line
        for g, lines in mdl.groups:
            for e0, rel in lines:
                def q(pp, e0=e0, rel=rel, g=g):
                    mu = pp[0] * e0; sg = np.sqrt(pp[1] ** 2 + pp[2] * mu)
                    return [mu, sg, FW * sg, pp[ig[g]] * rel]
                (mu, sg, fw, n), (dmu, dsg, dfw, dn) = propagate(q, p, cov)
                rows.append(dict(gain_corr=corr, kind="line", group=GROUP_LABEL[g], peak=NAMES.get(e0, f"{e0}"),
                                 E_true_keV=e0, mean_keV=round(mu, 4), mean_err=round(dmu, 4), sigma_keV=round(sg, 4),
                                 sigma_err=round(dsg, 4), fwhm_keV=round(fw, 4), fwhm_err=round(dfw, 4),
                                 counts=round(n, 1), counts_err=round(dn, 1), signif=""))
        # ---- each group as one (blended) peak; Ag K split into Ka (Ka1 + Ka2) and Kb, which are separate peaks
        blends = []
        for g, lines in mdl.groups:
            if g == "Ag K":
                blends += [(g, "Ag Ka", [l for l in lines if l[0] < 23]), (g, "Ag Kb", [l for l in lines if l[0] > 23])]
            else:
                blends.append((g, GROUP_LABEL[g], lines))
        for g, glab, lines in blends:
            es = np.array([e0 for e0, _ in lines]); ws = np.array([w for _, w in lines]); ws = ws / ws.sum()
            def qg(pp, es=es, ws=ws, g=g):
                mus = pp[0] * es; sgs = np.sqrt(pp[1] ** 2 + pp[2] * mus)
                c = np.sum(ws * mus); s_eff = np.sqrt(np.sum(ws * (sgs ** 2 + mus ** 2)) - c ** 2)
                return [c, s_eff, blend_fwhm(mus, sgs, ws), pp[ig[g]] * sum(w for _, w in lines)]
            (c, se, fw, n), (dc, dse, dfw, dn) = propagate(qg, p, cov)
            e_true = float(np.sum(ws * es))
            rows.append(dict(gain_corr=corr, kind="group (blend)", group=GROUP_LABEL[g], peak=glab + " (all lines)",
                             E_true_keV=round(e_true, 3), mean_keV=round(c, 4), mean_err=round(dc, 4), sigma_keV=round(se, 4),
                             sigma_err=round(dse, 4), fwhm_keV=round(fw, 4), fwhm_err=round(dfw, 4), counts=round(n, 1),
                             counts_err=round(dn, 1), signif=round(p[ig[g]] / r["err"][ig[g]], 1) if r["err"][ig[g]] > 0 else ""))
        # ---- fit-wide parameters
        for nm, i in (("energy scale k", 0), ("width s0 [keV]", 1), ("width s1 [keV]", 2)):
            rows.append(dict(gain_corr=corr, kind="fit parameter", group="", peak=nm, E_true_keV="", mean_keV=round(p[i], 6),
                             mean_err=round(r["err"][i], 6), sigma_keV="", sigma_err="", fwhm_keV="", fwhm_err="", counts="",
                             counts_err="", signif=""))
        rows.append(dict(gain_corr=corr, kind="fit parameter", group="", peak="deviance/ndf", E_true_keV="",
                         mean_keV=round(r["devndf"], 3), mean_err="", sigma_keV="", sigma_err="", fwhm_keV="", fwhm_err="",
                         counts="", counts_err="", signif=""))
        # ---- separately fitted peaks
        h, ed = np.histogram(E, bins=np.arange(55, 64, 0.15)); pk = 0.5 * (ed[1:] + ed[:-1])[np.argmax(h)]
        sep = [("gamma 59.54", 59.54, gfit(E, pk - 1.2, pk + 1.5, 0.15, pk, 0.3, 2.0)),
               ("backscatter of 59.54 (180 deg: 48.3)", 48.3, gfit(E, 40, 56, 0.5, 48.3 * pk / 59.54, 2.0, 5.0))]
        for nm, e0, g in sep:
            mu, dmu, sg, dsg, z, n, dn, chi = g
            rows.append(dict(gain_corr=corr, kind="separate fit", group="", peak=nm, E_true_keV=e0, mean_keV=round(mu, 4),
                             mean_err=round(dmu, 4), sigma_keV=round(sg, 4), sigma_err=round(dsg, 4), fwhm_keV=round(FW * sg, 4),
                             fwhm_err=round(FW * dsg, 4), counts=round(n, 1), counts_err=round(dn, 1), signif=round(z, 1)))
        # Fe-55: Mn Ka / Kb with a common scale and width
        h, ed = np.histogram(E, bins=np.arange(4.9, 7.3, 0.05)); x = 0.5 * (ed[1:] + ed[:-1])
        fm = lambda x, k, s, a1, a2, b0, b1: (a1 * np.exp(-0.5 * ((x - 5.899 * k) / s) ** 2) +
                                              a2 * np.exp(-0.5 * ((x - 6.490 * k) / s) ** 2) + b0 + b1 * (x - 6))
        pm, cm = optimize.curve_fit(fm, x, h, p0=[x[np.argmax(h)] / 5.899, 0.1, h.max(), h.max() * 0.15, 5, 0],
                                    sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                    bounds=([0.85, 0.02, 0, 0, -np.inf, -np.inf], [1.1, 0.5, np.inf, np.inf, np.inf, np.inf]))
        chi = np.sum(((h - fm(x, *pm)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 6, 1); cm = cm * max(chi, 1)
        for nm, e0, ia in (("Mn Ka (Fe-55)", 5.899, 2), ("Mn Kb (Fe-55)", 6.490, 3)):
            (mu, sg, n), (dmu, dsg, dn) = propagate(lambda q, e0=e0, ia=ia: [q[0] * e0, abs(q[1]), q[ia] * abs(q[1]) * np.sqrt(2 * np.pi) / 0.05], pm, cm)
            rows.append(dict(gain_corr=corr, kind="separate fit", group="", peak=nm, E_true_keV=e0, mean_keV=round(mu, 4),
                             mean_err=round(dmu, 4), sigma_keV=round(sg, 4), sigma_err=round(dsg, 4), fwhm_keV=round(FW * sg, 4),
                             fwhm_err=round(FW * dsg, 4), counts=round(n, 1), counts_err=round(dn, 1), signif=""))
        # Compton edge: step position and smearing (from the 7.5-30 keV fit with a free edge)
        re_ = Model("Ag", lo=7.5, free_edge=True).fit(E, k0); pe, ee = re_["p"], re_["err"]
        rows.append(dict(gain_corr=corr, kind="separate fit", group="", peak="Compton edge of 59.54 (free e-: 11.25)",
                         E_true_keV=11.25, mean_keV=round(pe[-1], 3), mean_err=round(ee[-1], 3), sigma_keV=round(pe[7], 3),
                         sigma_err=round(ee[7], 3), fwhm_keV=round(FW * pe[7], 3), fwhm_err=round(FW * ee[7], 3), counts="",
                         counts_err="", signif=""))

    with open("peak_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for corr in ("with", "without"):
        print(f"\n===== gain correction: {corr}")
        for kind in ("fit parameter", "group (blend)", "line", "separate fit"):
            print(f"--- {kind}")
            for r in rows:
                if r["gain_corr"] == corr and r["kind"] == kind:
                    if kind == "fit parameter":
                        print(f"  {r['peak']:40s} {r['mean_keV']} ± {r['mean_err']}")
                    else:
                        print(f"  {r['peak']:40s} true {r['E_true_keV']:>7}  mean {r['mean_keV']:>8} ± {r['mean_err']:<7} "
                              f"sigma {r['sigma_keV']:>7} ± {r['sigma_err']:<7} FWHM {r['fwhm_keV']:>7} ± {r['fwhm_err']:<7} "
                              f"counts {r['counts']} ± {r['counts_err']}  {r['signif']}")
    print("-> peak_table.csv")


if __name__ == "__main__":
    main()
