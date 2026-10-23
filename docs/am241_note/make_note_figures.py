#!/usr/bin/env python3
"""Step 6 of the note: every figure (PDF) and table (.tex) of the note, from the data.

All numbers in the note's tables are written here from the fits, so the note cannot drift from the analysis:
  figures/  fit_10_30.pdf (the figure of merit: constrained 10-30 keV fit with pulls), fit_10_30_allclusters.pdf,
            peak595_versions.pdf and stability595.pdf (gain correction: none / one curve / one curve per period),
            width_motivation.pdf, width_quantisation.pdf, spectra_cut.pdf, band_grid.pdf, other_peaks.pdf,
            gain_curve_img*.png (copied from the run's gain_vs_charge_epochs/)
  tables/   fit_parameters, groups, lines, gain_595, gain_curves, width_slices, band_grid_selected, other_peaks (.tex)
  numbers.json   numbers quoted in the text (pull statistics, fitted gains, slices, ...)
The identification tables (hypotheses, extra_lines, l1_lines, conversion_electrons) come from identification_tests.py.

Pulls (lower panel of the fit figures): (data - fit) / sqrt(fit) per bin; for a good model they scatter like a unit
Gaussian. Uncertainties of derived peak quantities are propagated with the full fit covariance (propagate()).

Run (after extract_clusters.py and fit_all_peaks.py, which makes band_grid.csv):
    python make_note_figures.py [--run-dir DIR] [--note-dir DIR]
"""
import argparse
import csv, glob, json, os, re, shutil
import numpy as np
from scipy import optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ROOT
from amfit import Model, summary, fit_595
from common import add_path_args, load_clusters, paths
ROOT.gROOT.SetBatch(True)

# note folder (figures/, tables/) and the gain-corrected run folder: set in main() from the command line
DOC = FIG = TAB = W = None
CUT = 0.7
FW = 2 * np.sqrt(2 * np.log(2))
INK, MUTED, GRID, DATA = "#1a1a19", "#6b6a63", "#d9d8d2", "#9a9890"
ORIG, POOL, CORR = "#eb6834", "#9a9890", "#2a78d6"      # no correction / pooled curve / per-period curves
ALLC, CUTC = "#eb6834", "#2a78d6"
COL = {"Ll": "#2a78d6", "La": "#eb6834", "Leta": "#1baf7a", "Lb": "#eda100", "Lg": "#e87ba4", "g26": "#008300",
       "Ag K": "#4a3aa7"}
LABEL = {"Ll": "Np L$\\ell$", "La": "Np L$\\alpha$", "Leta": "Np L$\\eta$", "Lb": "Np L$\\beta$", "Lg": "Np L$\\gamma$",
         "g26": "$\\gamma$ 26.3", "Ag K": "Ag K (to be confirmed)"}
MAIN = ("La", "Lb", "Lg", "g26")
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "pdf.fonttype": 42})
num = {}                                                  # numbers quoted in the text -> numbers.json


def znp(s):
    return float(np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0)))


def tex_row(cells):
    return " & ".join(str(c) for c in cells) + r" \\" + "\n"


def pm(v, e, nd):
    return f"${v:.{nd}f} \\pm {e:.{nd}f}$"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def main():
    """Make all figures and tables of the note (see the module docstring)."""
    global DOC, FIG, TAB, W
    _ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    PATHS = paths(_ap.parse_args())
    os.chdir(PATHS.work)
    DOC = PATHS.note; FIG = os.path.join(DOC, "figures"); TAB = os.path.join(DOC, "tables")
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    W = PATHS.corr
    d4, d3 = load_clusters(PATHS.work, 4), load_clusters(PATHS.work, 3)
    n_img = len(np.unique(d4["img"]))
    num["n_images"] = int(n_img)

    # ======================================================================= main fit (figure of merit)
    m4 = d4["sxy"] < CUT
    fits = {}
    for key, E in (("cut_corr", d4["Energy"][m4]), ("cut_orig", d4["E_orig"][m4]), ("all_corr", d4["Energy"])):
        fits[key] = Model("Ag").fit(E, 0.985 if key != "cut_orig" else 0.94)
    r = fits["cut_corr"]; s = summary(r); x, h, p, mdl = r["x"], r["h"], r["p"], r["model"]


    def fit_figure(r, s, ncl, title, name):
        x, h, p, mdl = r["x"], r["h"], r["p"], r["model"]
        fig = plt.figure(figsize=(11, 6.2)); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
        xf = np.linspace(mdl.lo, mdl.hi, 1000)
        ax.errorbar(x, h, yerr=np.sqrt(np.maximum(h, 1)), fmt="o", ms=2.5, color=DATA, elinewidth=0.7, label="data")
        q = np.array(p, float); q[8:8 + len(mdl.groups)] = 0
        ax.plot(xf, mdl(xf, q), color=MUTED, ls="--", lw=1.2, label="background + Compton edge")
        for g, cv in mdl.components(xf, p).items():
            if cv.max() > 0.5:
                ax.plot(xf, cv, color=COL[g], lw=1.3, label=LABEL[g])
        ax.plot(xf, mdl(xf, p), color=INK, lw=1.6, label="total fit")
        ax.axvline(11.25 * p[0], color=MUTED, lw=0.8, ls=":")
        ax.text(11.25 * p[0], ax.get_ylim()[1] * 0.95, " Compton edge of 59.5 keV", fontsize=7.5, color=MUTED, va="top")
        sig = "  ".join(f"{LABEL[g]} {s[g][0] / s[g][1]:.1f}$\\sigma$" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0)
        ax.set_title(f"{title} ({ncl} clusters): scale $k$ = {s['k'][0]:.4f} $\\pm$ {s['k'][1]:.4f}, deviance/ndf "
                     f"{r['devndf']:.2f}\n{sig}", loc="left", fontsize=9.5, color=INK)
        ax.set_ylabel(f"clusters / {mdl.bw} keV"); plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
        mu = mdl(x, p)
        axr.bar(x, (h - mu) / np.sqrt(np.maximum(mu, 1)), width=mdl.bw * 0.85, color=DATA)
        axr.axhline(0, color=INK, lw=0.8); axr.set_ylim(-4, 4); axr.set_ylabel("pull"); axr.set_xlabel("cluster energy [keV]")
        for yv in (-2, 2):
            axr.axhline(yv, color=MUTED, lw=0.6, ls=":")
        save(fig, name)


    fit_figure(r, s, int(m4.sum()), f"EXT4, $\\sigma_{{xy}} < {CUT}$, gain-corrected", "fit_10_30.pdf")
    fit_figure(fits["all_corr"], summary(fits["all_corr"]), len(d4["Energy"]), "EXT4, all clusters, gain-corrected",
               "fit_10_30_allclusters.pdf")
    pulls = (h - mdl(x, p)) / np.sqrt(np.maximum(mdl(x, p), 1))
    num["pull_mean"], num["pull_rms"] = float(np.mean(pulls)), float(np.std(pulls))
    num["pull_frac_within2"] = float(np.mean(np.abs(pulls) < 2))
    num["pull_2425"] = [round(float(v), 2) for v in pulls[(x > 24.0) & (x < 25.2)]]
    num["nbins"], num["npar"] = int(len(h)), int(len(p))
    num["fit"] = {key: dict(k=float(summary(f)["k"][0]), k_err=float(summary(f)["k"][1]), devndf=float(f["devndf"]),
                             Z={g: (float(summary(f)[g][0] / summary(f)[g][1]) if summary(f)[g][1] > 0 else 0.0)
                                for g, _ in f["model"].groups}) for key, f in fits.items()}

    # ---- fit parameter table
    cov = r["cov"]; err = r["err"]
    names_tex = {"k": "energy scale $k$", "s0": "width term $s_0$ [keV]", "s1": "width term $s_1$ [keV]",
                 "b0": "background $b_0$ [counts/bin]", "b1": "background $b_1$ [counts/bin/keV]",
                 "b2": "background $b_2$ [counts/bin/keV$^2$]", "A_compton": "Compton-step height $A_C$ [counts/bin]",
                 "s_compton": "Compton-step width $s_C$ [keV]"}
    with open(os.path.join(TAB, "fit_parameters.tex"), "w") as fh:
        for i, nm in enumerate(r["names"]):
            if nm in names_tex:
                nd = 5 if nm == "k" else 4 if nm in ("s0", "s1", "b2", "s_compton") else 2
                fh.write(tex_row([names_tex[nm], pm(p[i], err[i], nd) + (" (at limit)" if nm == "s0" and p[i] <= 0.0501 else "")]))
            else:
                fh.write(tex_row([f"total counts, {LABEL[nm]}", pm(p[i], err[i], 1)]))
        fh.write(r"\midrule" + "\n")
        fh.write(tex_row(["fit range, bin width", f"{mdl.lo:.1f}--{mdl.hi:.1f} keV, {mdl.bw} keV ({len(h)} bins)"]))
        fh.write(tex_row(["free parameters", f"{len(p)}"]))
        fh.write(tex_row(["deviance / ndf", f"{r['devndf']:.2f}"]))
        fh.write(tex_row(["Pearson $\\chi^2$ / ndf", f"{r['chi2ndf']:.2f}"]))

    # ---- peak tables (groups and individual lines) - same propagation as peak_table.py
    NAMES = {11.87: "L$\\ell$", 12.24: "L$t$", 13.18: "L$s$", 13.76: "L$\\alpha_2$", 13.95: "L$\\alpha_1$", 15.86: "L$\\eta$",
             16.11: "L$\\beta_6$", 16.79: "L$\\beta_{15}$", 16.84: "L$\\beta_2$", 17.06: "L$\\beta_4$", 17.39: "L$\\beta$ + L$\\beta_5$",
             17.75: "L$\\beta_1$", 17.99: "L$\\beta_3$", 18.58: "L$\\beta_{10}$", 18.76: "L$\\beta_9$", 20.10: "L$\\gamma_5$",
             20.78: "L$\\gamma_1$", 21.10: "L$\\gamma_2$", 21.30: "L$\\gamma_8$ + L$\\gamma_3$", 21.49: "L$\\gamma_6$",
             22.12: "L$\\gamma_4$", 22.40: "L$\\gamma_{13}$", 26.34: "$\\gamma$ 26.3", 22.16: "Ag K$\\alpha_1$ (t.b.c.)", 21.99: "Ag K$\\alpha_2$ (t.b.c.)",
             24.94: "Ag K$\\beta_1$ (t.b.c.)"}
    from amfit import NP
    PROB = {e0: pr for g, lines in NP.items() for e0, pr in lines}
    ig = {g: 8 + i for i, (g, _) in enumerate(mdl.groups)}


    def propagate(func, p, cov, eps=1e-6):
        f0 = np.asarray(func(p), float); J = np.zeros((f0.size, len(p)))
        for j in range(len(p)):
            hh = max(abs(p[j]) * eps, 1e-8); q = np.array(p, float); q[j] += hh
            J[:, j] = (np.asarray(func(q), float) - f0) / hh
        return f0, np.sqrt(np.clip(np.einsum("ij,jk,ik->i", J, cov, J), 0, None))


    def blend_fwhm(mus, sgs, w):
        xx = np.linspace(min(mus) - 5 * max(sgs), max(mus) + 5 * max(sgs), 6000)
        yy = sum(wi * np.exp(-0.5 * ((xx - m_) / s_) ** 2) / s_ for m_, s_, wi in zip(mus, sgs, w))
        ab = xx[yy >= yy.max() / 2]; return ab[-1] - ab[0]


    with open(os.path.join(TAB, "lines.tex"), "w") as fh:
        for g, lines in mdl.groups:
            for e0, rel in lines:
                (mu, sg, fw, n), (dmu, dsg, dfw, dn) = propagate(
                    lambda q, e0=e0, rel=rel, g=g: [q[0] * e0, np.sqrt(q[1] ** 2 + q[2] * q[0] * e0),
                                                    FW * np.sqrt(q[1] ** 2 + q[2] * q[0] * e0), q[ig[g]] * rel], p, cov)
                src = f"{PROB[e0]:g}" if e0 in PROB and g != "Ag K" else ("--" if g == "Ag K" else "")
                fh.write(tex_row([NAMES[e0], f"{e0:.2f}", src, f"{mu:.3f} $\\pm$ {dmu:.3f}", f"{sg:.3f} $\\pm$ {dsg:.3f}",
                                  f"{fw:.3f} $\\pm$ {dfw:.3f}", f"{n:.1f} $\\pm$ {dn:.1f}"]))
    blends = []
    for g, lines in mdl.groups:
        if g == "Ag K":
            blends += [(g, "Ag K$\\alpha$ (to be confirmed)", [l for l in lines if l[0] < 23]),
                       (g, "Ag K$\\beta$ (to be confirmed)", [l for l in lines if l[0] > 23])]
        else:
            blends.append((g, LABEL[g], lines))
    num["groups"] = {}
    with open(os.path.join(TAB, "groups.tex"), "w") as fh:
        for g, lab, lines in blends:
            es = np.array([e for e, _ in lines]); ws = np.array([w for _, w in lines]); ws = ws / ws.sum()

            def qg(q, es=es, ws=ws, g=g, lines=lines):
                mus = q[0] * es; sgs = np.sqrt(q[1] ** 2 + q[2] * mus); c = np.sum(ws * mus)
                return [c, np.sqrt(np.sum(ws * (sgs ** 2 + mus ** 2)) - c ** 2), blend_fwhm(mus, sgs, ws),
                        q[ig[g]] * sum(w for _, w in lines)]
            (c, se, fw, n), (dc, dse, dfw, dn) = propagate(qg, p, cov)
            z = p[ig[g]] / err[ig[g]] if err[ig[g]] > 0 else 0
            et = float(np.sum(ws * es))
            num["groups"][lab] = dict(true=et, mean=c, dmean=dc, sigma=se, fwhm=fw, counts=n, dcounts=dn, Z=z)
            zt = f"{z:.1f}" if not lab.startswith("Ag K$\\beta") else "(with K$\\alpha$)"
            fh.write(tex_row([lab, f"{et:.3f}", f"{c:.3f} $\\pm$ {dc:.3f}", f"{c / et:.4f}", f"{se:.3f} $\\pm$ {dse:.3f}",
                              f"{fw:.3f} $\\pm$ {dfw:.3f}", f"{n:.0f} $\\pm$ {dn:.0f}", zt]))

    # ---- model comparison (what the 24.9 keV bump and the 22 keV excess are)
    mc = []
    for lab, extra in (("Np L lines + $\\gamma$ 26.3 only", None), ("+ Sn K (25.27/25.04, 28.49 keV)", "Sn"),
                       ("+ Ag K (22.16/21.99, 24.94 keV)", "Ag")):
        rr = fits["cut_corr"] if extra == "Ag" else Model(extra).fit(d4["Energy"][m4], 0.985)
        mc.append((lab, rr["nll"], rr["devndf"], len(rr["p"])))
    best = min(v[1] for v in mc)
    with open(os.path.join(TAB, "model_comparison.tex"), "w") as fh:
        for lab, nll, dev, npar in mc:
            fh.write(tex_row([lab, npar, f"{2 * (nll - best):.1f}", f"{dev:.2f}"]))
    num["model_comparison"] = {lab: dict(two_dnll=2 * (nll - best), devndf=dev) for lab, nll, dev, _ in mc}

    # ======================================================================= gain correction
    def read_energies(sub, ext):
        out = {}
        for f in sorted(glob.glob(os.path.join(W, sub, f"panaSKImg_clustersRec_*_ext{ext}.root"))):
            nimg = int(re.search(r"_(\d+)_ext", f).group(1))
            tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec"); E = []
            for i in range(t.GetEntries()):
                t.GetEntry(i); E.append(np.array(list(t.Energy), float))
            tf.Close(); out[nimg] = np.concatenate(E)
        return out


    versions = [("no correction", "root", ORIG), ("one curve, all images", "root_gaincorr", POOL),
                ("one curve per gain period", "root_gaincorr_epochs", CORR)]
    EV = {(v, e): read_energies(sub, e) for v, sub, _ in versions for e in (3, 4)}
    rows = []
    num["gain"] = {}
    for e in (4, 3):
        for v, sub, _ in versions:
            out = [v if e == 4 else "", f"EXT{e}" if v == "no correction" else ""]
            for lab, sel in (("21-40", lambda n: n <= 40), ("41-50", lambda n: n >= 41), ("all", lambda n: True)):
                g = fit_595(np.concatenate([E for n, E in EV[(v, e)].items() if sel(n)]))
                out.append(f"{g[0] / 59.54:.4f}"); num["gain"][f"EXT{e} {v} {lab}"] = dict(ratio=g[0] / 59.54, sigma=g[2])
                if lab == "all":
                    out.append(f"{g[2]:.3f}")
            rows.append(out)
    with open(os.path.join(TAB, "gain_595.tex"), "w") as fh:
        for e in (4, 3):
            for v, sub, _ in versions:
                gg = [num["gain"][f"EXT{e} {v} {lab}"] for lab in ("21-40", "41-50", "all")]
                fh.write(tex_row([f"EXT{e}" if v == "no correction" else "", v] + [f"{q['ratio']:.4f}" for q in gg]
                                 + [f"{gg[2]['sigma']:.3f}"]))
            if e == 4:
                fh.write(r"\midrule" + "\n")
    # per-period fitted gains (what changed)
    fq = list(csv.DictReader(open(os.path.join(PATHS.orig, "fit_quality.csv"))))
    num["fitted_gain"] = {}
    for ext in ("EXT1", "EXT3", "EXT4"):
        a = [float(q["gain"]) for q in fq if q["ext"] == ext and int(re.search(r"_(\d+)$", q["image"]).group(1)) <= 40]
        b = [float(q["gain"]) for q in fq if q["ext"] == ext and int(re.search(r"_(\d+)$", q["image"]).group(1)) >= 41]
        num["fitted_gain"][ext] = dict(p1=np.mean(a), p1_rms=np.std(a), p2=np.mean(b), p2_rms=np.std(b), ratio=np.mean(b) / np.mean(a))
    curves = {}
    for pth in sorted(glob.glob(os.path.join(W, "gain_vs_charge_epochs", "*_gain_vs_charge.json"))):
        js = json.load(open(pth)); lab = re.search(r"img(\d+-\d+)", pth).group(1); curves[lab] = js["extensions"]
        shutil.copy(pth.replace(".json", ".png"), os.path.join(FIG, f"gain_curve_img{lab}.png"))
    with open(os.path.join(TAB, "gain_curves.tex"), "w") as fh:
        for lab, ex in curves.items():
            for e in ("EXT3", "EXT4", "EXT1"):
                c = ex.get(e, {})
                fg = num["fitted_gain"][e]["p1" if lab.startswith("21") else "p2"]
                if c.get("ok"):
                    fh.write(tex_row([f"{lab}", e, f"{c['model']['a']:.4f}", f"{fg:.4f}", f"{c['drop_at_1000'] * 100:.1f}",
                                      f"{c['n_max_measured']:.0f}"]))
                else:
                    fh.write(tex_row([f"{lab}", e, "--", f"{fg:.4f}", "--", "no reliable curve"]))
    num["curves"] = {lab: {e: dict(a=c["model"]["a"], drop=c["drop_at_1000"], nmax=c["n_max_measured"])
                           for e, c in ex.items() if c.get("ok")} for lab, ex in curves.items()}

    # figure: 59.5 keV peak per version (EXT4, EXT3), and per image
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, e in zip(axes, (4, 3)):
        for v, sub, col in versions:
            E = np.concatenate(list(EV[(v, e)].values()))
            ax.hist(E, bins=np.arange(53, 62, 0.15), histtype="step", color=col, lw=1.5, label=v)
        ax.axvline(59.54, color=INK, ls="--", lw=0.8); ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters / 0.15 keV")
        ax.set_title(f"EXT{e}, all clusters, {n_img} images", loc="left", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "peak595_versions.pdf")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    imgs = sorted(EV[("no correction", 4)])
    for ax, e in zip(axes, (4, 3)):
        for k_, (v, sub, col) in enumerate(versions):
            xs, ys, es_ = [], [], []
            for j, n in enumerate(imgs):
                g = fit_595(EV[(v, e)][n])
                if g and g[1] < 0.3:
                    xs.append(j + (k_ - 1) * 0.2); ys.append(g[0] / 59.54); es_.append(g[1] / 59.54)
            ax.errorbar(xs, ys, yerr=es_, fmt="o", ms=4, color=col, capsize=2, label=v)
        x0 = imgs.index(41) - 0.5
        ax.axvline(x0, color=MUTED, ls=":", lw=1); ax.axhline(1, color=INK, ls="--", lw=0.8)
        ax.text(x0, ax.get_ylim()[1], " gain period 2 (images 41-50)", va="top", fontsize=8, color=MUTED)
        ax.set_ylabel("59.5 keV peak / true"); ax.set_title(f"EXT{e}", loc="left", fontsize=10)
    axes[1].set_xticks(range(len(imgs))); axes[1].set_xticklabels([f"#{n}" for n in imgs], rotation=60, fontsize=7.5)
    axes[1].set_xlabel("image"); axes[0].legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
    save(fig, "stability595.pdf")

    # ======================================================================= width cut: motivation
    E, S = d4["Energy"], d4["sxy"]
    sl = [(0, 0.33), (0.33, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]
    kk = []
    for lo, hi in sl:
        mm = (S >= lo) & (S < hi)
        rr = Model("Ag").fit(E[mm], 0.985); ss = summary(rr); g = fit_595(E[mm])
        kk.append((lo, hi, int(mm.sum()), ss["k"][0], ss["k"][1], ss["Lb"][0] / ss["Lb"][1] if ss["Lb"][1] > 0 else 0,
                   g[0] / 59.54 if g else np.nan, g[1] / 59.54 if g else np.nan))
    with open(os.path.join(TAB, "width_slices.tex"), "w") as fh:
        for lo, hi, n, k, dk, zb, r5, dr5 in kk:
            fh.write(tex_row([f"{lo:.2f}--{hi:.2f}", n, f"{k:.4f} $\\pm$ {dk:.4f}", f"{zb:.1f}", f"{r5:.4f}"]))
    num["width_slices"] = [dict(lo=a, hi=b, n=n, k=k, dk=dk, Zb=zb, r595=r5) for a, b, n, k, dk, zb, r5, _ in kk]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]; mm = (E > 4) & (E < 64)
    ax.hist2d(E[mm], S[mm], bins=[np.arange(4, 64, 0.4), np.linspace(0, 2, 70)], norm=LogNorm(), cmap="Blues")
    ax.axhline(CUT, color=CUTC, lw=1.5); ax.axhspan(0, CUT, color=CUTC, alpha=0.08)
    ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("$\\sigma_{xy}$ [pixels]"); ax.grid(False)
    ax.set_title(f"EXT4: width vs energy (kept: $\\sigma_{{xy}} < {CUT}$)", loc="left", fontsize=10)
    ax = axes[1]
    win = np.zeros_like(E, bool); side = np.zeros_like(E, bool)
    for (a, b), (c_, d_) in (((17.0, 17.9), (15.4, 16.3)), ((20.2, 21.2), (22.7, 23.6)), ((25.5, 26.3), (27.5, 28.4))):
        win |= (E > a) & (E < b); side |= (E > c_) & (E < d_)
    for sel, col, lab, ls in ((win, CUTC, "L$\\beta$/L$\\gamma$/26.3 keV windows", "-"), (side, ALLC, "neighbouring continuum", "-"),
                              ((E > 58.5) & (E < 60.3), INK, "59.5 keV peak", "--")):
        ax.hist(S[sel], bins=np.linspace(0, 2, 41), density=True, histtype="step", color=col, lw=1.6, ls=ls, label=lab)
    ax.axvline(CUT, color=CUTC, ls=":"); ax.set_xlabel("$\\sigma_{xy}$ [pixels]"); ax.set_ylabel("normalised")
    ax.set_title("EXT4: width of line vs continuum clusters", loc="left", fontsize=10); ax.legend(fontsize=7.5, frameon=False)
    ax = axes[2]
    xc = [0.5 * (a + b) for a, b, *_ in kk]
    ax.errorbar(xc, [q[3] for q in kk], yerr=[q[4] for q in kk], fmt="o-", color=CUTC, ms=6, capsize=3, label="10-30 keV lines (fit scale $k$)")
    ax.plot(xc, [q[6] for q in kk], "s--", color=INK, ms=6, label="59.5 keV peak")
    ax.axhline(1, color=MUTED, lw=0.8); ax.axvline(CUT, color=CUTC, ls=":")
    ax.set_xlabel("$\\sigma_{xy}$ slice [pixels]"); ax.set_ylabel("measured / true energy")
    ax.set_title("EXT4: energy scale vs cluster width", loc="left", fontsize=10); ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); save(fig, "width_motivation.pdf")

    # width quantisation (why sigma_xy has a gap at 0.30-0.33)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    mm = (E > 8) & (E < 30)
    axes[0].hist(S[mm], bins=np.arange(0.1, 0.8, 0.01), histtype="stepfilled", color=CUTC, alpha=0.7)
    axes[0].set_xlabel("$\\sigma_{xy}$ (Gaussian fit of the projections) [pixels]"); axes[0].set_ylabel("clusters, 8-30 keV")
    axes[0].set_title("fitted width: spikes and a gap at 0.30-0.33", loc="left", fontsize=10)
    axes[1].hist(d4["wSTD_XY"][mm], bins=np.arange(0.1, 0.8, 0.01), histtype="stepfilled", color=ALLC, alpha=0.7)
    axes[1].set_xlabel("wSTD$_{XY}$ (charge-weighted RMS) [pixels]"); axes[1].set_title("charge-weighted RMS: continuous", loc="left", fontsize=10)
    fig.tight_layout(); save(fig, "width_quantisation.pdf")

    # ======================================================================= spectra with/without cut, band grid
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.hist(E, bins=np.arange(4, 66, 0.25), histtype="step", color=ALLC, lw=1.2, label="all clusters")
    ax.hist(E[m4], bins=np.arange(4, 66, 0.25), histtype="step", color=CUTC, lw=1.2, label=f"$\\sigma_{{xy}} < {CUT}$")
    ax.set_yscale("log"); ax.set_ylim(bottom=0.8)
    for e0, nm in ((5.9, "Mn K$\\alpha$"), (11.25, "Compton edge"), (13.93, "L$\\alpha$"), (17.54, "L$\\beta$"), (21.0, "L$\\gamma$"),
                   (22.1, "Ag K$\\alpha$ (tbc)"), (24.94, "Ag K$\\beta$ (tbc)"), (26.34, "26.3"), (48.3, "backscatter"), (59.54, "59.5")):
        ax.axvline(e0, color=MUTED, ls="--", lw=0.6); ax.text(e0, 8e3, " " + nm, rotation=90, va="top", fontsize=7, color=MUTED)
    ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters / 0.25 keV"); ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"EXT4, {n_img} images, gain-corrected per period", loc="left", fontsize=10)
    save(fig, "spectra_cut.pdf")

    grid = list(csv.DictReader(open("band_grid.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var, lab in zip(axes, ("sxy", "wSTD_XY"), ("$\\sigma_{xy}$", "wSTD$_{XY}$")):
        G = [g for g in grid if g["var"] == var]
        los = sorted({float(g["lower"]) for g in G}); his = sorted({float(g["upper"]) for g in G})
        Z = np.array([[float(next(g["Z_np"] for g in G if float(g["lower"]) == lo and float(g["upper"]) == hi)) for hi in his] for lo in los])
        ax.imshow(Z, origin="lower", cmap="Blues", aspect="auto")
        for a_ in range(len(los)):
            for b_ in range(len(his)):
                ax.text(b_, a_, f"{Z[a_, b_]:.1f}", ha="center", va="center", fontsize=8, color="white" if Z[a_, b_] > Z.max() * 0.7 else INK)
        ax.set_xticks(range(len(his))); ax.set_xticklabels(his); ax.set_yticks(range(len(los))); ax.set_yticklabels(los)
        ax.set_xlabel(f"upper limit on {lab}"); ax.set_ylabel(f"lower limit on {lab}"); ax.grid(False)
        ax.set_title(f"combined significance $Z_{{\\rm Np}}$, all {n_img} images", loc="left", fontsize=10)
    fig.tight_layout(); save(fig, "band_grid.pdf")
    with open(os.path.join(TAB, "band_grid_selected.tex"), "w") as fh:
        for var, lo, hi in (("sxy", 0.0, 0.6), ("sxy", 0.0, 0.7), ("sxy", 0.2, 0.7), ("sxy", 0.3, 0.7), ("sxy", 0.0, 0.8),
                            ("sxy", 0.0, 0.9), ("sxy", 0.5, 0.9), ("wSTD_XY", 0.0, 0.7), ("wSTD_XY", 0.0, 0.9)):
            g = next(q for q in grid if q["var"] == var and float(q["lower"]) == lo and float(q["upper"]) == hi)
            vl = "$\\sigma_{xy}$" if var == "sxy" else "wSTD$_{XY}$"
            fh.write(tex_row([f"{lo:.1f} $\\le$ {vl} $<$ {hi:.1f}", g["n"], f"{float(g['Z_np']):.1f}", f"{float(g['Z_even']):.1f}",
                              f"{float(g['Z_odd']):.1f}", f"{float(g['k']):.4f}", f"{float(g['devndf']):.2f}"]))

    # ======================================================================= other peaks
    def gfit(E, lo, hi, bw, guess, wguess, wmax):
        hh, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); xx = 0.5 * (ed[1:] + ed[:-1])
        f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - guess)
        pp, cc = optimize.curve_fit(f, xx, hh, p0=[max(hh.max() - np.median(hh), 1), guess, wguess, np.median(hh), 0],
                                    sigma=np.sqrt(np.maximum(hh, 1)), maxfev=20000,
                                    bounds=([0, lo, bw / 2, -np.inf, -np.inf], [np.inf, hi, wmax, np.inf, np.inf]))
        chi = np.sum(((hh - f(xx, *pp)) / np.sqrt(np.maximum(hh, 1))) ** 2) / max(len(hh) - 5, 1); cc = cc * max(chi, 1)
        ee = np.sqrt(np.clip(np.diag(cc), 0, None))
        return dict(mu=pp[1], dmu=ee[1], s=abs(pp[2]), ds=ee[2], z=pp[0] / ee[0], x=xx, h=hh, p=pp, f=f)


    others = {}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, (sname, sel) in enumerate((("all clusters", np.ones(len(E), bool)), (f"$\\sigma_{{xy}} < {CUT}$", m4))):
        Em = E[sel]; col = ALLC if i == 0 else CUTC
        hh, ed = np.histogram(Em, bins=np.arange(55, 64, 0.15)); pk = 0.5 * (ed[1:] + ed[:-1])[np.argmax(hh)]
        g595 = gfit(Em, pk - 1.2, pk + 1.5, 0.15, pk, 0.3, 2.0)
        bs = gfit(Em, 40, 56, 0.5, 48.3 * pk / 59.54, 2.0, 5.0)
        hh, ed = np.histogram(Em, bins=np.arange(4.9, 7.3, 0.05)); xx = 0.5 * (ed[1:] + ed[:-1])
        fm = lambda x, k, s, a1, a2, b0, b1: (a1 * np.exp(-0.5 * ((x - 5.899 * k) / s) ** 2) + a2 * np.exp(-0.5 * ((x - 6.490 * k) / s) ** 2) + b0 + b1 * (x - 6))
        pmn, cmn = optimize.curve_fit(fm, xx, hh, p0=[xx[np.argmax(hh)] / 5.899, 0.1, hh.max(), hh.max() * 0.15, 5, 0],
                                      sigma=np.sqrt(np.maximum(hh, 1)), maxfev=20000,
                                      bounds=([0.85, 0.02, 0, 0, -np.inf, -np.inf], [1.1, 0.5, np.inf, np.inf, np.inf, np.inf]))
        chi = np.sum(((hh - fm(xx, *pmn)) / np.sqrt(np.maximum(hh, 1))) ** 2) / max(len(hh) - 6, 1); emn = np.sqrt(np.diag(cmn) * max(chi, 1))
        re_ = Model("Ag", lo=7.5, free_edge=True).fit(Em, 0.985); pe, ee = re_["p"], re_["err"]
        others[sname] = dict(g595=g595, bs=bs, mn=(pmn, emn), edge=(pe[-1], ee[-1], pe[7], ee[7], 11.25 * pe[0]))
        ax = axes[i, 0]; ax.step(xx, hh, where="mid", color=DATA); xf = np.linspace(4.9, 7.3, 400); ax.plot(xf, fm(xf, *pmn), color=col, lw=1.5)
        ax.set_title(f"{sname}: Mn K$\\alpha$ / K$\\beta$ (Fe-55)", loc="left", fontsize=9)
        ax = axes[i, 1]; mdl_e = re_["model"]
        ax.step(re_["x"], re_["h"], where="mid", color=DATA); xf = np.linspace(7.5, 16, 400)
        qe = np.array(pe, float); qe[8:8 + len(mdl_e.groups)] = 0
        ax.plot(xf, mdl_e(xf, pe), color=col, lw=1.5); ax.plot(xf, mdl_e(xf, qe), color=MUTED, ls="--", lw=1)
        ax.axvline(pe[-1], color=col, ls=":"); ax.set_xlim(7.5, 16); ax.set_title(f"{sname}: Compton edge of 59.54 keV", loc="left", fontsize=9)
        for ax, gg, ttl, lo_, hi_ in ((axes[i, 2], bs, "backscatter peak", 40, 56), (axes[i, 3], g595, "59.54 keV", 55, 64)):
            ax.step(gg["x"], gg["h"], where="mid", color=DATA); xf = np.linspace(gg["x"][0], gg["x"][-1], 400)
            ax.plot(xf, gg["f"](xf, *gg["p"]), color=col, lw=1.5); ax.set_title(f"{sname}: {ttl}", loc="left", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("cluster energy [keV]")
    fig.tight_layout(); save(fig, "other_peaks.pdf")
    with open(os.path.join(TAB, "other_peaks.tex"), "w") as fh:
        for sname in others:
            o = others[sname]; g, b = o["g595"], o["bs"]; pmn, emn = o["mn"]; ed = o["edge"]
            sn = "all" if sname == "all clusters" else "cut"
            fh.write(r"\multicolumn{7}{l}{\emph{" + sname + r"}} \\" + "\n")
            fh.write(tex_row(["$\\gamma$ 59.54", "59.54", f"{g['mu']:.3f} $\\pm$ {g['dmu']:.3f}", f"{g['mu'] / 59.54:.4f}",
                              f"{g['s']:.3f} $\\pm$ {g['ds']:.3f}", f"{FW * g['s']:.3f} $\\pm$ {FW * g['ds']:.3f}", f"{g['z']:.0f}"]))
            fh.write(tex_row(["backscatter peak", "$\\geq$ 48.3", f"{b['mu']:.2f} $\\pm$ {b['dmu']:.2f}", "--",
                              f"{b['s']:.2f} $\\pm$ {b['ds']:.2f}", f"{FW * b['s']:.2f} $\\pm$ {FW * b['ds']:.2f}", f"{b['z']:.0f}"]))
            for nm, e0 in (("Mn K$\\alpha_1$ (Fe-55)", 5.899), ("Mn K$\\beta$ (Fe-55)", 6.490)):
                fh.write(tex_row([nm, f"{e0:.3f}", f"{pmn[0] * e0:.3f} $\\pm$ {emn[0] * e0:.3f}", f"{pmn[0]:.4f}",
                                  f"{pmn[1]:.3f} $\\pm$ {emn[1]:.3f}", f"{FW * pmn[1]:.3f} $\\pm$ {FW * emn[1]:.3f}", "--"]))
            fh.write(tex_row(["Compton edge (step)", "11.25", f"{ed[0]:.2f} $\\pm$ {ed[1]:.2f}", f"{ed[0] / 11.25:.3f}",
                              f"{ed[2]:.2f} $\\pm$ {ed[3]:.2f}", f"{FW * ed[2]:.2f} $\\pm$ {FW * ed[3]:.2f}", "--"]))
    num["others"] = {sn: dict(E595=o["g595"]["mu"], s595=o["g595"]["s"], bs=o["bs"]["mu"], dbs=o["bs"]["dmu"], bs_z=o["bs"]["z"],
                              mn_k=o["mn"][0][0], mn_s=o["mn"][0][1], edge=o["edge"][0], dedge=o["edge"][1], edge_free=o["edge"][4])
                     for sn, o in others.items()}
    json.dump(num, open(os.path.join(DOC, "numbers.json"), "w"), indent=1, default=float)
    print("figures:", sorted(os.listdir(FIG))); print("tables:", sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()
