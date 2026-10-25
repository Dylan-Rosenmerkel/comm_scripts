"""Figures for the 'contamination' section of docs/aGMX (the high-side tail of the 59.5 keV line on the crowded CCD,
its cause and the quality flags). Reads the investigation results saved in <data>/excess/."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/excess")
OUT = os.path.join(HERE, "..", "docs", "aGMX", "figures")
AN = os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis")
sys.path.insert(0, AN)
plt.rcParams.update({"font.size": 9})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig); print(name)


from scan import load as wload
cwd = os.getcwd(); os.chdir(AN); dW = wload(4); os.chdir(cwd)
fs = sorted(glob.glob(os.path.join(X, "res_ped_real2", "*.npz")))
P = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("E", "sxy", "blend", "b_q25", "b_med")}
base = (P["sxy"] < 0.7) & (P["blend"] < 0.02)

# fig30: the tail
fig, ax = plt.subplots(figsize=(8, 4.2)); b = np.arange(54, 66, 0.1)
for E, lab, c in ((dW["Energy"][dW["sxy"] < 0.7], "WADERs, sigma_xy < 0.7", "0.4"), (P["E"][base], "A_GMX, sigma_xy < 0.7, blend < 0.02", "tab:green"),
                  (P["E"][base & ~(P["b_q25"] >= 10)], "A_GMX, + pedestal < 10 e-/pix", "tab:blue")):
    h, _ = np.histogram(E, b); ax.step(b[:-1], h / max(h.max(), 1), where="post", color=c, lw=1.3, label=lab)
ax.axvline(59.54, color="r", ls=":", lw=0.8); ax.set_yscale("log"); ax.set_ylim(5e-3, 1.5); ax.set_xlabel("cluster energy [keV]")
ax.set_ylabel("clusters (peak = 1)"); ax.legend(fontsize=8)
ax.set_title("crowded Am-241 CCD (29 images): the 59.5 keV line", fontsize=9.5, loc="left")
save(fig, "fig30_tail.png")

# fig31: injected clusters, isolated vs next to real events
rs = [r for r in sum((json.load(open(f)) for f in glob.glob(os.path.join(X, "res_ped_inj2", "*.json"))), []) if r["E0"] > 10]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); bins = np.linspace(0.9, 1.4, 101)
for a, (ext, t) in zip(ax, ((1, "clean CCD (EXT1)"), (4, "crowded CCD (EXT4)"))):
    for near, c, lab in ((False, "tab:blue", "injected away from real events"), (True, "tab:red", "injected next to real events")):
        x = np.array([r["share"] for r in rs if r["ext"] == ext and r["near"] == near])
        if x.size: a.hist(np.clip(x, 0.9, 1.4), bins, histtype="step", color=c, lw=1.3, density=True, label=f"{lab} (n={x.size}, median {np.median(x):.3f})")
    a.axvline(1, color="k", lw=0.6); a.set_yscale("log"); a.set_xlabel("E_rec / E_true (A_GMX), X-rays 17.75-59.5 keV"); a.legend(fontsize=7)
    a.set_title(t, fontsize=9.5, loc="left")
save(fig, "fig31_injected_contamination.png")

# fig32: the charge around tail clusters
r = np.load(os.path.join(X, "ring_rows.npy"))
fig, ax = plt.subplots(figsize=(7.5, 4))
edges = np.array([57, 59, 60, 60.5, 61.5, 63, 66]); mid = 0.5 * (edges[1:] + edges[:-1])
med = [np.median(r[(r[:, 0] >= a) & (r[:, 0] < b_), 1]) for a, b_ in zip(edges[:-1], edges[1:])]
ax.scatter(r[:, 0], np.clip(r[:, 1], 0.5, None), s=3, alpha=0.25, color="0.4", label="clusters (sigma_xy<0.7, blend<0.02)")
ax.plot(mid, med, "o-", color="tab:red", label="median per energy bin"); ax.set_yscale("log"); ax.axvline(59.54, color="k", ls=":", lw=0.6)
ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("median charge in the ring 5-8 px [e-/pix]"); ax.legend(fontsize=8)
ax.set_title("clusters above the line sit on a charge sheet (crowded CCD, 3 images)", fontsize=9.5, loc="left")
save(fig, "fig32_ring_level.png")

# fig33: flags of injected clusters, clean vs contaminated
fr = [r for r in sum((json.load(open(f)) for f in glob.glob(os.path.join(X, "res_feat", "*.json"))), []) if r["method"] == "A_GMX" and r["E0"] > 10]
ratio = np.array([r["ratio"] for r in fr]); cont = ratio > 1.02; clean = np.abs(ratio - 1) < 0.01
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for a, (k, bins_, lab) in zip(ax, (("peakfrac", np.linspace(0.3, 1.0, 36), "peak fraction (largest cloud / all)"), ("chi2", np.logspace(-1, 3, 41), "chi2 per pixel of the cluster's own model"),
                                   ("n_comp", np.arange(0.5, 10.5, 1), "number of clouds"))):
    v = np.array([r[k] for r in fr], float)
    a.hist(v[clean], bins_, histtype="step", color="tab:blue", lw=1.3, density=True, label=f"clean (within 1 %), n={clean.sum()}")
    a.hist(v[cont], bins_, histtype="step", color="tab:red", lw=1.3, density=True, label=f"contaminated (> +2 %), n={cont.sum()}")
    if k == "chi2": a.set_xscale("log")
    a.set_xlabel(lab); a.legend(fontsize=7)
ax[0].set_title("injected X-rays: what distinguishes contaminated clusters", fontsize=9.5, loc="left")
save(fig, "fig33_flags.png")

# fig34: trade-off of the pedestal cut
from amfit import Model, summary
cuts = [None, 20, 10, 5, 2]; T, Z, N = [], [], []
for x in cuts:
    m = base if x is None else base & ~(P["b_q25"] >= x); E = P["E"][m]
    T.append(np.sum((E > 60.5) & (E < 64)) / max(np.sum((E > 59) & (E < 60)), 1)); N.append(m.sum())
    rr = Model("Ag").fit(E, 0.985); s = summary(rr); Z.append(np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in ("La", "Lb", "Lg", "g26") if s[g][1] > 0)))
EW = dW["Energy"][dW["sxy"] < 0.7]; rW = Model("Ag").fit(EW, 0.985); sW = summary(rW)
ZW = np.sqrt(sum((sW[g][0] / sW[g][1]) ** 2 for g in ("La", "Lb", "Lg", "g26") if sW[g][1] > 0)); TW = np.sum((EW > 60.5) & (EW < 64)) / np.sum((EW > 59) & (EW < 60))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(T, Z, "o-", color="tab:green", label="A_GMX, pedestal cut (labels: e-/pix)")
for t, z, x, n in zip(T, Z, cuts, N): ax.annotate(f"{'none' if x is None else '< ' + str(x)}\n{n} cl.", (t, z), textcoords="offset points", xytext=(6, -12), fontsize=7)
ax.plot([TW], [ZW], "s", color="0.4", label="WADERs (sigma_xy < 0.7)")
ax.set_xlabel("59.5 keV high-side tail / peak (N 60.5-64 / N 59-60)"); ax.set_ylabel("combined Np L significance Z"); ax.legend(fontsize=8)
ax.set_title("purity vs statistics on the crowded CCD", fontsize=9.5, loc="left")
save(fig, "fig34_pedestal_tradeoff.png")
