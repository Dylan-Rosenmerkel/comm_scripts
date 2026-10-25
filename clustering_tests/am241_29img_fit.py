"""Constrained Am-241 10-30 keV fit (amfit.Model('Ag'), as runs_29img/analysis/final_fit.py) on the EXT4 clusters of
the MFD+G hybrid clustering, same style as final_fit.png; plus a width-cut scan and the WADERs numbers."""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import am241_paths  # noqa: E402
AN = am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
from amfit import Model, summary
METHOD = sys.argv[1] if len(sys.argv) > 1 else "hybrid_v1"
CUT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
INK, MUTED, GRID, DATA = "#1a1a19", "#6b6a63", "#d9d8d2", "#9a9890"
COL = {"Ll": "#2a78d6", "La": "#eb6834", "Leta": "#1baf7a", "Lb": "#eda100", "Lg": "#e87ba4", "g26": "#008300", "Ag K": "#4a3aa7"}
LABEL = {"Ll": "Np Ll", "La": "Np L$\\alpha$", "Leta": "Np L$\\eta$", "Lb": "Np L$\\beta$", "Lg": "Np L$\\gamma$",
         "g26": "$\\gamma$ 26.3", "Ag K": "Ag K (fluorescence)"}
MAIN = ("La", "Lb", "Lg", "g26")
znp = lambda s: np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))

CDIR = os.environ.get("CDIR", "clusters_occ0.3")
files = sorted(glob.glob(os.path.join(HERE, CDIR, os.environ.get("CGLOB", "*_ext4.npz"))))   # CDIR may be absolute
E = np.concatenate([np.load(f)[f"{METHOD}_E"] for f in files]); S = np.concatenate([np.load(f)[f"{METHOD}_sxy"] for f in files])
if os.environ.get("BLEND"):          # purity cut on the blend flag (GMX / A_GMX), e.g. BLEND=0.02
    B = np.concatenate([np.load(f)[f"{METHOD}_blend"] for f in files]); keep = B < float(os.environ["BLEND"])
    E, S = E[keep], S[keep]
print(f"{len(files)} images, {METHOD}: {E.size} clusters")
print("width-cut scan (MFD+G moment width sigma_xy):")
for wc in (0.5, 0.6, 0.7, 0.8, 1.0, 9.0):
    m = S < wc
    r = Model("Ag").fit(E[m], 0.985); s = summary(r)
    print(f"  sigma_xy < {wc}: {m.sum():6d} clusters  k {s['k'][0]:.4f}±{s['k'][1]:.4f}  dev/ndf {r['devndf']:.2f}  Z_np {znp(s):.1f}  "
          + "  ".join(f"{g} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0))
ref = [r for r in csv.DictReader(open(os.path.join(AN, "final_fit.csv"))) if r["ext"] == "4" and r["gain_corr"] == "with"]
if ref:
    print("WADERs reference (final_fit.csv, EXT4, sigma_xy < 0.7, gain-corrected): k", ref[0]["k"], "dev/ndf", ref[0]["devndf"], "clusters", ref[0]["n_clusters"],
          "  " + "  ".join(f"{r['group']} {r['signif']}σ" for r in ref))

m = S < CUT
r = Model("Ag").fit(E[m], 0.985); s = summary(r); x, h, p, mdl = r["x"], r["h"], r["p"], r["model"]
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig = plt.figure(figsize=(14, 6.2)); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
xf = np.linspace(mdl.lo, mdl.hi, 1000)
ax.errorbar(x, h, yerr=np.sqrt(np.maximum(h, 1)), fmt="o", ms=2.5, color=DATA, elinewidth=0.7, label="data", zorder=1)
q = np.array(p, float); q[8:8 + len(mdl.groups)] = 0
ax.plot(xf, mdl(xf, q), color=MUTED, ls="--", lw=1.2, label="background + Compton edge")
for g, cv in mdl.components(xf, p).items():
    if cv.max() > 0.5:
        ax.plot(xf, cv, color=COL[g], lw=1.3, label=LABEL[g])
ax.plot(xf, mdl(xf, p), color=INK, lw=1.6, label="total fit")
ax.axvline(11.25 * p[0], color=MUTED, lw=0.8, ls=":")
ax.text(11.25 * p[0], ax.get_ylim()[1] * 0.95, " Compton edge of 59.5 keV", fontsize=7.5, color=MUTED, va="top")
ax.set_ylabel(f"clusters / {mdl.bw} keV"); plt.setp(ax.get_xticklabels(), visible=False)
sig = "  ".join(f"{LABEL[g]} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0)
name = {"hybrid_v1": "MFD+G hybrid (v1, 14 px)", "hybrid_v2": "MFD+G hybrid (v2, 14 px)", "MFDG": "MFD+G", "MHC": "MHC", "GMX": "GMX (blend < 0.02)", "adaptive_GMX": "adaptive+GMX (blend < 0.02)"}[METHOD] + (f", occupancy mask < {CDIR.split('occ')[-1]}" if "occ" in CDIR else "")
ax.set_title(f"EXT4, {name}, sigma_xy < {CUT:g} (MFD+G moment width), gain-corrected ({int(m.sum())} clusters):  scale k = {s['k'][0]:.4f} ± {s['k'][1]:.4f},  "
             f"deviance/ndf {r['devndf']:.2f}\n{sig}", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
mu = mdl(x, p)
axr.bar(x, (h - mu) / np.sqrt(np.maximum(mu, 1)), width=mdl.bw * 0.85, color=DATA)
axr.axhline(0, color=INK, lw=0.8); axr.set_ylim(-4, 4); axr.set_ylabel("pull"); axr.set_xlabel("cluster energy [keV]")
fig.suptitle(f"Am-241 10-30 keV: constrained fit (Np L lines and intensity ratios from Lepy et al. 2008, Table 3) - 29 images, gain correction per gain period, {name} clustering",
             fontsize=11, weight="bold", color=INK, x=0.01, ha="left")
fn = os.path.join(HERE, f"final_fit_{METHOD}_cut{CUT:g}_{os.path.basename(os.path.normpath(CDIR))}.png"); fig.savefig(fn, dpi=150, bbox_inches="tight"); print(fn)
