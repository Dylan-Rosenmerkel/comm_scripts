"""Figures for docs/gain_correction_method/gain_correction_method.tex (written to its figures/ folder)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import ROOT
ROOT.gROOT.SetBatch(True)
sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/src"))
import gain_vs_charge as GVC
import autoconfig_panaSKImg as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
DATA = "/Users/dylanrosenmerkel/Documents/LSM_Module_Testing_Analysis/Production_Modules_Data/2025-08-27-DM07_DM09_DM10_Low_Temp"
STUDY = os.path.join(DATA, "gain_curve_study_frontside")
ROOTD = os.path.join(STUDY, "WADERs_analysis_Image5_DM09_DM10", "root")
STAMP = {0: "20250827_182132_0", 1: "20250827_212852_1", 2: "20250828_003611_2"}
KA, KB = 5.899, 6.490
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
COL = {"none": "#8c8c8c", "interpolated": "#1f5fbf", "broken": "#2a9d8f", "linear": "#d1495b"}


def model(mod, tag, form="interpolated"):
    return json.load(open(os.path.join(STUDY, "models", f"{mod}_img{tag}_{form}.json")))


# ----------------------------------------------------------------------------- 1. the comb
img = os.path.join(DATA, f"avg_Image_5_Low_Temp_109_{STAMP[1]}.fz")
ref = json.loads(A.REFERENCE_JSON)
h = fits.open(img)
import argparse
_, cfg, _, _ = A.geometry(h[0].header, h, ref["input"]["image"], argparse.Namespace(ccd_cols=None))
g0 = model("DM09", "1")["EXT4"]["g_1e"]
x = GVC.ext_pixels(h[4].data, cfg, g0)
mask, _ = GVC.clean_rows(x, g0)
v = x[mask].ravel()
fig, axs = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))
ax = axs[0]
sel = v[(v > -1.5 * g0) & (v < 6.5 * g0)]
ax.hist(sel / g0, bins=np.arange(-1.5, 6.5, 0.02), color="#1f5fbf", histtype="stepfilled", alpha=0.85)
ax.set_yscale("log"); ax.set_xlabel("pixel value / $g_0$"); ax.set_ylabel("pixels per bin")
ax.set_title(f"0-6 e$^-$: the comb, spacing $g_0$ = {g0:.4f} ADU", fontsize=10)
from gain_vs_charge import best_period
for ax, (lo, hi) in zip(axs[1:], ((20, 35), (270, 360), (640, 850))):
    sel = v[(v > lo * g0) & (v < hi * g0)]
    gl, R = best_period(sel, g0)
    R0 = np.abs(np.mean(np.exp(2j * np.pi * sel / g0)))
    for gg, col, lab in ((g0, "0.55", f"folded with $g_0$: R = {R0:.2f}"),
                         (gl, "#d1495b", f"folded with $g$ = {gl:.4f}: R = {R:.2f}")):
        phi0 = np.angle(np.mean(np.exp(2j * np.pi * sel / gg))) / (2 * np.pi)   # centre on the mean phase
        ph = np.mod(sel / gg - phi0 + 0.5, 1.0) - 0.5
        ax.hist(ph, bins=np.linspace(-0.5, 0.5, 51), histtype="step", lw=1.6, color=col, label=lab, density=True)
    ax.set_xlabel("position within a comb tooth, frac$(x/g)$ (centred)")
    ax.set_title(f"{lo}-{hi} e$^-$ ({sel.size} pixels)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
axs[1].set_ylabel("density")
fig.suptitle("The electron comb and how its local spacing is measured (DM09 EXT4, one Image 5, clean rows). Left: the comb "
             "at low charge. Right: all pixels of a charge bin folded\nonto one tooth. With the low-charge gain $g_0$ the "
             "teeth no longer line up at high charge (flat); with the best local spacing $g(n)$ they do (peak). R = comb visibility",
             fontsize=9.5)
plt.tight_layout(rect=(0, 0, 1, 0.86))
plt.savefig(os.path.join(OUT, "comb.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- 2. g(n) and the three forms
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
nn = np.logspace(1, np.log10(6000), 400)
for k, (mod, ext) in enumerate((("DM09", "EXT4"), ("DM09", "EXT2"))):
    r = model(mod, "012")[ext]
    pts = [p for p in r["points"] if p["significant"]]
    n = np.array([p["n"] for p in pts]); g = np.array([p["g"] for p in pts]); e = np.array([p["err"] for p in pts])
    ax, ar = axs[0, k], axs[1, k]
    gi = GVC.g_of(nn, r["model"])
    ax.errorbar(n, g, e, fmt="ko", ms=4, label="measured local gain (3 images)", zorder=5)
    for form in ("interpolated", "broken", "linear"):
        m = model(mod, "012", form)[ext]["model"]
        ax.plot(nn, GVC.g_of(nn, m), color=COL[form], lw=1.4, label=form + (" (default)" if form == "interpolated" else ""))
        ar.plot(nn, (GVC.g_of(nn, m) / gi - 1) * 100, color=COL[form], lw=1.4)
    ar.errorbar(n, (g / GVC.g_of(n, r["model"]) - 1) * 100, e / g * 100, fmt="ko", ms=4)
    for a_ in (ax, ar):
        a_.axvspan(r["n_max_measured"], r["model"]["n_cap"], color="0.9", zorder=0)
        a_.axvspan(r["model"]["n_cap"], 6000, color="0.8", zorder=0)
        a_.set_xscale("log")
    ax.set_title(f"{mod} {ext}", fontsize=11); ax.set_ylabel("$g(n)$ [ADU/e$^-$]")
    ar.set_ylabel("relative to interpolated [%]"); ar.set_xlabel("electrons in the pixel $n$"); ar.set_ylim(-2, 2)
    ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Local gain $g(n)$ and the three model forms. Light grey: beyond the highest measured charge $n_{max}$, "
             "continued linearly; dark grey: above $n_{cap}=3\\,n_{max}$, held constant", fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(os.path.join(OUT, "forms.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- 3. nonlinearity curve
def wader_gain(acm, stamp, ext):
    f = ROOT.TFile.Open(os.path.join(ROOTD, f"panaSKImg_clustersRec_avg_Image_5_Low_Temp_{acm}_{stamp}_ext{ext}.root"))
    info = f.Get("info"); info.GetEntry(0); gv = float(getattr(info, f"ext{ext}_gainext")); f.Close()
    return gv


fig, axs = plt.subplots(2, 4, figsize=(17, 7.6))
for row, mod in enumerate(("DM09", "DM10")):
    res = model(mod, "012")
    for k, ext in enumerate(("EXT1", "EXT2", "EXT3", "EXT4")):
        ax, r = axs[row, k], res.get(ext, {})
        if not r.get("ok"):
            ax.text(0.5, 0.5, "no reliable curve\n(gain rises with charge)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{mod} {ext}", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); continue
        m = r["model"]
        nl = GVC.nonlinearity([p for p in r["points"] if p["significant"]], m)
        x, d, ed = np.array(nl["n_meas"]), np.array(nl["delta"]), np.array(nl["delta_err"])
        ax.errorbar(x, d, ed, fmt="o", ms=4, color="#1f5fbf", label="measured local gains, integrated")
        nt, At, _ = GVC._table(m)
        grid = np.linspace(0, r["n_max_measured"], 300); Ag = np.interp(grid, nt, At)
        ax.plot(Ag / nl["g_ref"], Ag / nl["g_ref"] - grid, color="0.3", lw=1.2, label="interpolated model")
        xx = np.linspace(0, x.max() * 1.05, 200)
        ax.plot(xx, nl["c2"] * xx ** 2 + nl["c1"] * xx, color="#d1495b", lw=1.2,
                label=f"${nl['c2']:.2e}\\,x^2 {nl['c1']:+.4f}\\,x$".replace("e-0", "e-"))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{mod} {ext}: $g(0)$ = {nl['g_ref']:.4f}", fontsize=10)
        ax.set_xlabel("measured pixel charge $A(n)/g(0)$ [e$^-$]", fontsize=9)
        ax.set_ylabel("measured $-$ peak number [e$^-$]", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Nonlinearity curves (reference gain $g(0)$) of DM09 and DM10 from the 3-image $g(n)$ measurement, "
             "as written by run_WADERs --gain-correction (no peak fits)", fontsize=10.5)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(os.path.join(OUT, "nonlinearity.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- 4. Fe-55 front side
def front(acm, i, ext):
    f = os.path.join(ROOTD, f"panaSKImg_clustersRec_avg_Image_5_Low_Temp_{acm}_{STAMP[i]}_ext{ext}.root")
    tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec"); info = tf.Get("info")
    info.GetEntry(0); gfit = float(getattr(info, f"ext{ext}_gainext"))
    out = []
    for k in range(t.GetEntries()):
        t.GetEntry(k)
        sx, sy = np.array(list(t.fSTD_X)), np.array(list(t.fSTD_Y)); E = np.array(list(t.Energy))
        for j in np.where((np.sqrt(0.5 * (sx ** 2 + sy ** 2)) < 1) & (E > 4.5) & (E < 7.0))[0]:
            out.append((np.array(list(t.pixels_E[int(j)]), float), gfit))
    tf.Close()
    return out


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (mod, acm, ext) in zip(axs, (("DM09", "109", 1), ("DM09", "109", 2), ("DM10", "106", 2))):
    cl = [c for i in STAMP for c in front(acm, i, ext)]
    mm = model(mod, "012")[f"EXT{ext}"]["model"]
    E0 = np.array([c.sum() for c, _ in cl])
    E1 = np.array([GVC.electrons_of_adu(c * 1e3 / GVC.E_PER_E * gv, mm).sum() * GVC.E_PER_E * 1e-3 for c, gv in cl])
    ed = np.arange(4.8, 6.9, 0.025)
    ax.hist(E0, ed, histtype="step", color=COL["none"], lw=1.6, label="without correction")
    ax.hist(E1, ed, histtype="step", color=COL["interpolated"], lw=1.6, label="with correction")
    ax.axvline(KA, color="k", ls="--", lw=0.8); ax.axvline(KB, color="k", ls=":", lw=0.8)
    ax.set_title(f"{mod} EXT{ext}, front side, 3 images ({len(cl)} clusters)", fontsize=10)
    ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters / 25 eV")
axs[0].legend(fontsize=8)
fig.suptitle("Front-side Fe-55 ($\\sigma_{xy}<1$) of the three Image 5s of a module, with the 3-image curve "
             "(dashed: K$\\alpha$ 5.899 keV, dotted: K$\\beta$ 6.490 keV)", fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(OUT, "fe55.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- 5. Am-241 (needs am241_forms.py)
sp = os.path.join(HERE, "data", "am241_spectra.json")
if os.path.exists(sp):
    S = json.load(open(sp)); edges = np.arange(0, 70.001, 0.1); c = 0.5 * (edges[1:] + edges[:-1])
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    for row, e in enumerate(("3", "4")):
        for col_, (lo, hi, reb) in enumerate(((12, 30, 4), (54, 63, 2))):
            ax = axs[row, col_]
            for k in ("none", "linear", "interpolated"):
                y = np.array(S[e][k], float)
                y = y[: len(y) // reb * reb].reshape(-1, reb).sum(axis=1); x = c[: len(c) // reb * reb].reshape(-1, reb).mean(axis=1)
                w = (x > lo) & (x < hi)
                ax.step(x[w], y[w], where="mid", color=COL[k], lw=1.3,
                        label={"none": "without correction", "linear": "linear", "interpolated": "interpolated (default)"}[k])
            for E0 in ((13.93, 17.75, 20.80, 26.34) if col_ == 0 else (59.54,)):
                ax.axvline(E0, color="k", ls="--", lw=0.7)
            ax.set_title(f"EXT{e}", fontsize=10); ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel(f"clusters / {0.1 * reb:.1f} keV")
            ax.legend(fontsize=8)
    fig.suptitle("Am-241 (AMTEst, 10 Image 8s, board 106, all clusters): Np L lines and 26.3 keV (left), 59.54 keV (right); "
                 "dashed: true energies", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(os.path.join(OUT, "am241.png"), dpi=150); plt.close(fig)
print("figures in", OUT)
