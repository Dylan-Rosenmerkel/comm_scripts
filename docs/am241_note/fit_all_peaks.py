#!/usr/bin/env python3
"""Step 3 of the note: spectra and fits with and without the cluster-width cut, and the band-cut grid.

For each CCD (EXT4, EXT3, EXT1), each selection (all clusters / sigma_xy < 0.7) and with / without gain correction:
  - the constrained 10-30 keV fit (amfit.Model('Ag')),
  - the Compton edge of the 59.54 keV gamma ray with its position free (7.5-30 keV fit),
  - the 59.54 keV gamma-ray peak (amfit.fit_595),
  - the backscatter peak (59.54 keV photons scattered back by the material around the CCD; >= 48.3 keV),
  - the Fe-55 Mn Ka/Kb lines (5.899 / 6.490 keV).
Then the band-cut grid on the CCD with most clusters: lower <= width < upper for sigma_xy and wSTD_XY, scored by the
combined significance of the Np line groups Z_Np = sqrt(sum Z_g^2), on all images and on the even/odd halves (a cut
that is really better must be better on both independent halves).

Output (in the work folder): fits29.csv, band_grid.csv, fits29_spectra.png, fits29_lines.png, fits29_other_peaks.png,
fits29_band_grid.png. The grid is slow (~170 fits, ~40 min); --no-grid reuses band_grid.csv.
Run (after extract_clusters.py):   python fit_all_peaks.py [--run-dir DIR] [--no-grid]
"""
import argparse
import csv
import os
import sys
import numpy as np
from scipy import optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from amfit import Model, summary, fit_595
from common import add_path_args, load_clusters, paths

CUT = 0.7
EXTS = (4, 3, 1)
MAIN = ("La", "Lb", "Lg", "g26")
INK, MUTED, GRID, DATA = "#1a1a19", "#6b6a63", "#d9d8d2", "#9a9890"
ALLC, CUTC = "#eb6834", "#2a78d6"                    # all clusters / after the cut
COL = {"Ll": "#2a78d6", "La": "#eb6834", "Leta": "#1baf7a", "Lb": "#eda100", "Lg": "#e87ba4", "g26": "#008300",
       "Ag K": "#4a3aa7"}
LABEL = {"Ll": "Np Ll", "La": "Np L$\\alpha$", "Leta": "Np L$\\eta$", "Lb": "Np L$\\beta$", "Lg": "Np L$\\gamma$",
         "g26": "$\\gamma$ 26.3", "Ag K": "Ag K"}
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def znp(s):
    return float(np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0)))


def gauss_lin(E, lo, hi, bw, guess, wguess, wmax):
    """Gaussian + linear background in [lo, hi]; (mu, mu_err, sigma, significance, x, h, p, f) or None."""
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); x = 0.5 * (ed[1:] + ed[:-1])
    if h.sum() < 50:
        return None
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - guess)
    try:
        p, cov = optimize.curve_fit(f, x, h, p0=[max(h.max() - np.median(h), 1), guess, wguess, np.median(h), 0],
                                    sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                    bounds=([0, lo, bw / 2, -np.inf, -np.inf], [np.inf, hi, wmax, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h - f(x, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 5, 1)
    e = np.sqrt(np.clip(np.diag(cov), 0, None)) * np.sqrt(max(chi, 1))
    return p[1], e[1], abs(p[2]), p[0] / e[0] if e[0] > 0 else 0, x, h, p, f


def fit_mn(E):
    """Fe-55: Mn Ka 5.899 + Kb 6.490 keV, one scale, free amplitudes, common width, linear background."""
    h, ed = np.histogram(E, bins=np.arange(4.9, 7.3, 0.05)); x = 0.5 * (ed[1:] + ed[:-1])
    if h.max() < 30:
        return None
    def f(x, k, s, a1, a2, b0, b1):
        return (a1 * np.exp(-0.5 * ((x - 5.899 * k) / s) ** 2) + a2 * np.exp(-0.5 * ((x - 6.490 * k) / s) ** 2)
                + b0 + b1 * (x - 6))
    try:
        p, cov = optimize.curve_fit(f, x, h, p0=[x[np.argmax(h)] / 5.899, 0.1, h.max(), h.max() * 0.15, 5, 0],
                                    sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                    bounds=([0.85, 0.02, 0, 0, -np.inf, -np.inf], [1.1, 0.5, np.inf, np.inf, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h - f(x, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 6, 1)
    e = np.sqrt(np.clip(np.diag(cov), 0, None)) * np.sqrt(max(chi, 1))
    return dict(k=p[0], k_err=e[0], sigma=p[1], kb_ka=p[3] / max(p[2], 1e-9), x=x, h=h, p=p, f=f, chi=chi)


def main():
    """All fits, the band grid and the figures (written to the work folder)."""
    _ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    _ap.add_argument("--no-grid", action="store_true", help="reuse band_grid.csv instead of recomputing the grid")
    PATHS = paths(_ap.parse_args())
    os.chdir(PATHS.work)
    data = {e: load_clusters(PATHS.work, e) for e in EXTS}
    SEL = {"all": lambda d: np.ones(len(d["Energy"]), bool), f"sxy<{CUT}": lambda d: d["sxy"] < CUT}
    rows, keep = [], {}
    for e in EXTS:
        d = data[e]
        for sname, sf in SEL.items():
            m = sf(d)
            for corr, E in (("with", d["Energy"]), ("without", d["E_orig"])):
                Em = E[m]; k0 = 0.985 if corr == "with" else 0.94
                r = Model("Ag").fit(Em, k0); s = summary(r)
                re_ = Model("Ag", lo=7.5, free_edge=True).fit(Em, k0); se = summary(re_)
                g = fit_595(Em)
                bs = gauss_lin(Em, 40, 56, 0.5, 48.3 * (g[0] / 59.54 if g else 1), 2.0, 5.0)
                mn = fit_mn(Em)
                bad = any(not (s[q][1] > 0) for q in MAIN) or s["s0"][0] > 0.79 or not (s["k"][1] > 0)
                row = dict(ext=e, selection=sname, gain_corr=corr, n_clusters=int(m.sum()),
                           k_10_30=round(s["k"][0], 5), k_err=round(s["k"][1], 5), devndf=round(r["devndf"], 3),
                           sigma_20keV=round(float(np.sqrt(s["s0"][0] ** 2 + s["s1"][0] * 20)), 3), Z_np=round(znp(s), 1),
                           fit_ok="" if not bad else "NO",
                           **{f"N_{q}": f"{s[q][0]:.0f}+-{s[q][1]:.0f}" for q, _ in r["model"].groups},
                           **{f"Z_{q}": round(s[q][0] / s[q][1], 1) if s[q][1] > 0 else 0 for q, _ in r["model"].groups},
                           compton_edge_keV=round(se["edge_E"][0], 2) if se["edge_E"][1] > 0 else "",
                           compton_edge_err=round(se["edge_E"][1], 2) if se["edge_E"][1] > 0 else "",
                           E595=round(g[0], 3) if g else "", E595_err=round(g[1], 3) if g else "",
                           sigma595=round(g[2], 3) if g else "",
                           backscatter_keV=round(bs[0], 2) if bs else "", backscatter_err=round(bs[1], 2) if bs else "",
                           backscatter_sigma=round(bs[2], 2) if bs else "", backscatter_Z=round(bs[3], 1) if bs else "",
                           MnKa_keV=round(5.899 * mn["k"], 3) if mn else "", MnKa_err=round(5.899 * mn["k_err"], 3) if mn else "",
                           Mn_sigma=round(mn["sigma"], 3) if mn else "", Mn_kb_ka=round(mn["kb_ka"], 3) if mn else "")
                rows.append(row); keep[(e, sname, corr)] = (r, s, re_, se, g, bs, mn, Em)
                print(f"EXT{e} {sname:9s} {corr:7s} n={row['n_clusters']:6d}  10-30: k {row['k_10_30']}±{row['k_err']} "
                      f"dev {row['devndf']} Z_np {row['Z_np']} {row['fit_ok']}  edge {row['compton_edge_keV']}±{row['compton_edge_err']}  "
                      f"59.5 {row['E595']}  backsc {row['backscatter_keV']}±{row['backscatter_err']} ({row['backscatter_Z']}σ)  "
                      f"MnKa {row['MnKa_keV']}", flush=True)
    with open("fits29.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---------------------------------------------------------------- band-cut grid (CCD with most clusters)
    big = max(EXTS, key=lambda e: len(data[e]["Energy"])); d = data[big]
    grid = []
    if "--no-grid" in sys.argv:                  # reuse band_grid.csv from a previous run
        grid = [{k: (v if k == "var" else float(v)) for k, v in r.items()} for r in csv.DictReader(open("band_grid.csv"))]
    for var in (("sxy", "wSTD_XY") if not grid else ()):
        for lo in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
            for hi in (0.6, 0.7, 0.8, 0.9):
                m = (d[var] >= lo) & (d[var] < hi)
                r = Model("Ag").fit(d["Energy"][m], 0.985); s = summary(r)
                z = [znp(summary(Model("Ag").fit(d["Energy"][m & (d["img"] % 2 == h)], 0.985))) for h in (0, 1)]
                grid.append(dict(var=var, lower=lo, upper=hi, n=int(m.sum()), Z_np=round(znp(s), 2), Z_even=round(z[0], 2),
                                 Z_odd=round(z[1], 2), k=round(s["k"][0], 5), k_err=round(s["k"][1], 5), devndf=round(r["devndf"], 3)))
                print(f"band {var:8s} [{lo:.2f},{hi:.2f}) n={m.sum():6d} Z {znp(s):5.1f} (even {z[0]:5.1f}, odd {z[1]:5.1f}) "
                      f"k {s['k'][0]:.4f}±{s['k'][1]:.4f} dev {r['devndf']:.2f}", flush=True)
    if "--no-grid" not in sys.argv:
        with open("band_grid.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

    # ---------------------------------------------------------------- figures
    n_img = len(np.unique(data[EXTS[0]]["img"]))
    # 1. spectra, all vs cut
    fig, axes = plt.subplots(len(EXTS), 1, figsize=(13, 9), sharex=True)
    for ax, e in zip(axes, EXTS):
        d = data[e]; E = d["Energy"]
        ax.hist(E, bins=np.arange(4, 66, 0.25), histtype="step", color=ALLC, lw=1.2, label="all clusters")
        ax.hist(E[d["sxy"] < CUT], bins=np.arange(4, 66, 0.25), histtype="step", color=CUTC, lw=1.2, label=f"sigma_xy < {CUT}")
        ax.set_yscale("log"); ax.set_ylim(bottom=0.8); ax.set_ylabel("clusters / 0.25 keV")
        for e0, nm in ((5.9, "Mn K$\\alpha$"), (11.25, "Compton edge"), (13.93, "L$\\alpha$"), (17.75, "L$\\beta$"),
                       (20.78, "L$\\gamma$"), (22.1, "Ag K$\\alpha$"), (24.94, "Ag K$\\beta$"), (26.34, "26.3"),
                       (48.3, "backscatter"), (59.54, "59.5")):
            ax.axvline(e0, color=MUTED, ls="--", lw=0.6)
            ax.text(e0, ax.get_ylim()[1] * 0.5, " " + nm, rotation=90, va="top", fontsize=7, color=MUTED)
        ax.set_title(f"EXT{e} (gain-corrected)", loc="left", fontsize=10, color=INK)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right"); axes[-1].set_xlabel("cluster energy [keV]")
    fig.suptitle(f"AMtest, {n_img} images: spectra with and without the width cut", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("fits29_spectra.png", dpi=100)

    # 2. 10-30 keV constrained fits, best CCD, all vs cut
    fig = plt.figure(figsize=(14, 10)); gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 0.45, 3, 1], hspace=0.12)
    for b, sname in enumerate(SEL):
        r, s, *_ = keep[(big, sname, "with")]
        x, h, p, mdl = r["x"], r["h"], r["p"], r["model"]
        ax = fig.add_subplot(gs[3 * b]); axr = fig.add_subplot(gs[3 * b + 1], sharex=ax)
        xf = np.linspace(mdl.lo, mdl.hi, 1000)
        ax.errorbar(x, h, yerr=np.sqrt(np.maximum(h, 1)), fmt="o", ms=2.5, color=DATA, elinewidth=0.7, label="data")
        q = np.array(p, float); q[8:8 + len(mdl.groups)] = 0
        ax.plot(xf, mdl(xf, q), color=MUTED, ls="--", lw=1.2, label="background + Compton edge")
        for g, cv in mdl.components(xf, p).items():
            if cv.max() > 0.5:
                ax.plot(xf, cv, color=COL[g], lw=1.3, label=LABEL[g])
        ax.plot(xf, mdl(xf, p), color=INK, lw=1.6, label="total fit")
        sig = "  ".join(f"{LABEL[g]} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0)
        ax.set_title(f"EXT{big}, {sname} ({int(sum(SEL[sname](data[big])))} clusters): k = {s['k'][0]:.4f} ± {s['k'][1]:.4f}, "
                     f"deviance/ndf {r['devndf']:.2f}\n{sig}", loc="left", fontsize=9.5, color=INK)
        plt.setp(ax.get_xticklabels(), visible=False); ax.set_ylabel(f"clusters / {mdl.bw} keV")
        if b == 0:
            ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
        mu = mdl(x, p)
        axr.bar(x, (h - mu) / np.sqrt(np.maximum(mu, 1)), width=mdl.bw * 0.85, color=DATA); axr.axhline(0, color=INK, lw=0.8)
        axr.set_ylim(-4, 4); axr.set_ylabel("pull")
    axr.set_xlabel("cluster energy [keV]")
    fig.suptitle(f"10-30 keV constrained fit (Lepy et al. 2008 line list), {n_img} images, gain-corrected", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.savefig("fits29_lines.png", dpi=100, bbox_inches="tight")

    # 3. the other peaks: Mn Ka/Kb, Compton edge, backscatter, 59.5 - all vs cut, gain-corrected
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for i, sname in enumerate(SEL):
        r, s, re_, se, g, bs, mn, Em = keep[(big, sname, "with")]
        col = ALLC if i == 0 else CUTC
        ax = axes[i, 0]
        if mn:
            ax.step(mn["x"], mn["h"], where="mid", color=DATA); xf = np.linspace(4.9, 7.3, 400)
            ax.plot(xf, mn["f"](xf, *mn["p"]), color=col, lw=1.5)
            ax.set_title(f"{sname}: Mn K$\\alpha$ {5.899 * mn['k']:.3f} keV (true 5.899)\nK$\\beta$/K$\\alpha$ {mn['kb_ka']:.2f}, "
                         f"sigma {mn['sigma'] * 1000:.0f} eV", loc="left", fontsize=9)
        ax = axes[i, 1]
        x, h, p, mdl = re_["x"], re_["h"], re_["p"], re_["model"]
        ax.step(x, h, where="mid", color=DATA); xf = np.linspace(mdl.lo, 16, 400); q = np.array(p, float); q[8:8 + len(mdl.groups)] = 0
        ax.plot(xf, mdl(xf, p), color=col, lw=1.5); ax.plot(xf, mdl(xf, q), color=MUTED, ls="--", lw=1)
        ax.axvline(se["edge_E"][0], color=col, ls=":"); ax.set_xlim(7.5, 16)
        ax.set_title(f"{sname}: Compton edge {se['edge_E'][0]:.2f} ± {se['edge_E'][1]:.2f} keV\n(free electron: 11.25 keV x k = "
                     f"{11.25 * se['k'][0]:.2f})", loc="left", fontsize=9)
        ax = axes[i, 2]
        if bs:
            x, h, p, f = bs[4:8]
            ax.step(x, h, where="mid", color=DATA); xf = np.linspace(40, 56, 400)
            ax.plot(xf, f(xf, *p), color=col, lw=1.5)
            ax.set_title(f"{sname}: backscatter peak {bs[0]:.2f} ± {bs[1]:.2f} keV\n(expected 48.3), sigma {bs[2]:.1f} keV, {bs[3]:.0f}σ",
                         loc="left", fontsize=9)
        ax = axes[i, 3]
        ax.hist(Em, bins=np.arange(55, 63, 0.15), histtype="step", color=col, lw=1.3); ax.axvline(59.54, color=MUTED, ls="--")
        if g:
            ax.set_title(f"{sname}: 59.54 keV at {g[0]:.3f} ± {g[1]:.3f} keV\nsigma {g[2]:.3f} keV", loc="left", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("cluster energy [keV]")
    fig.suptitle(f"EXT{big}, gain-corrected: Fe-55 lines, Compton edge, backscatter peak and 59.5 keV - top: all clusters, "
                 f"bottom: sigma_xy < {CUT}", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("fits29_other_peaks.png", dpi=100)

    # 4. band grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, var in zip(axes, ("sxy", "wSTD_XY")):
        G = [g for g in grid if g["var"] == var]
        los = sorted({g["lower"] for g in G}); his = sorted({g["upper"] for g in G})
        Z = np.array([[next(g["Z_np"] for g in G if g["lower"] == lo and g["upper"] == hi) for hi in his] for lo in los])
        ax.imshow(Z, origin="lower", cmap="Blues", aspect="auto")
        for a in range(len(los)):
            for b in range(len(his)):
                ax.text(b, a, f"{Z[a, b]:.1f}", ha="center", va="center", fontsize=8, color="white" if Z[a, b] > Z.max() * 0.7 else INK)
        ax.set_xticks(range(len(his))); ax.set_xticklabels(his); ax.set_yticks(range(len(los))); ax.set_yticklabels(los)
        ax.set_xlabel(f"upper limit on {var}"); ax.set_ylabel(f"lower limit on {var}"); ax.grid(False)
        ax.set_title(f"EXT{big}: combined Np-line significance of the fit", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig("fits29_band_grid.png", dpi=100)
    print("-> fits29.csv band_grid.csv fits29_spectra.png fits29_lines.png fits29_other_peaks.png fits29_band_grid.png")


if __name__ == "__main__":
    main()
