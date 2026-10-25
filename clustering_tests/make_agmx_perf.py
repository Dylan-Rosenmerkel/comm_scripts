"""Performance figures + tables for docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip) (A_GMX vs WADERs and the other methods).
Reads the saved results in ~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/."""
import glob, json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25")
OUT = os.path.join(HERE, "..", "docs", "aGMX", "figures")
COL = {"WADERs": "0.45", "B auto": "tab:olive", "MFD+G": "tab:red", "hybrid v1": "tab:orange", "hybrid v2": "gold",
       "MHC": "tab:blue", "adaptive": "0.1", "GMX": "tab:purple", "adaptive+GMX": "tab:green"}
LAB = {"adaptive+GMX": "A_GMX", "adaptive": "adaptive (MHC engine)"}
plt.rcParams.update({"font.size": 9})
TABLES = []


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig); print(name)


def _tex(x):
    return str(x).replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&").replace(">=", "$\\ge$").replace("adaptive+GMX", "A\\_GMX")


def md(rows, header):
    """LaTeX tabular (name kept from the Markdown version)"""
    cols = "l" + "c" * (len(header) - 1)
    lines = ["\\begin{table}[H]\\centering\\footnotesize", "\\begin{tabular}{" + cols + "}", "\\toprule",
             " & ".join(_tex(h) for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(_tex(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ inputs
def latest(name, sub="", n_min=1):
    """the re-run of a result folder with the audited code (<name>_v3, 2026-10-25) when it is complete, else <name>"""
    v3 = name + "_v3"
    return v3 if len(glob.glob(os.path.join(D, v3, sub, "*"))) >= n_min else name


# ------------------------------------------------------------------------------------------------ wide test
_n_all = len(glob.glob(os.path.join(D, "results_wide", "*.json")))
_v2 = os.path.join(D, "results_wide_adaptive_v2")
EXTRA = {"adaptive": "results_wide_adaptive_v2" if os.path.isdir(_v2) and len(glob.glob(os.path.join(_v2, "*.json"))) >= _n_all else "results_wide_adaptive_merged",
         "GMX": latest("results_wide_gmx", n_min=_n_all), "adaptive+GMX": latest("results_wide_adaptive_gmx", n_min=_n_all)}
VAL = latest("results_adaptive_gmx", "crowded")     # GMX / A_GMX validation (fe55, tracks, lowE, crowded)
print("inputs:", EXTRA, VAL)
recs = []
for f in sorted(glob.glob(os.path.join(D, "results_wide", "*.json"))):
    rec = json.load(open(f)); ok = True
    for n, d in EXTRA.items():
        fa = os.path.join(D, d, os.path.basename(f))
        if not os.path.exists(fa): ok = False; break
        a = json.load(open(fa)); rec["real"][n] = a["real"].get(n, {})
        if "inject" in rec and "inject" in a: rec["inject"][n] = a["inject"][n]
        else: rec.pop("inject", None)
    if ok and "inject" in rec: recs.append(rec)
M = ["WADERs", "B auto", "MFD+G", "hybrid v1", "hybrid v2", "MHC", "adaptive", "GMX", "adaptive+GMX"]
ok_ = lambda r, k: r["resolved"] if k == "pair4" else (r["n"] == 1 and abs(r["ratio"] - 1) < 0.05)


def wide_row(sel, m):
    v = collections.defaultdict(list); rat = []
    for rec in sel:
        for r in rec["inject"][m]:
            v[r["kind"]].append(ok_(r, r["kind"]))
            if r["kind"] == "fe55" and r["n"] >= 1: rat.append(r["ratio"])
    rat = np.array(rat); t = np.median([rec["real"][m]["time"] for rec in sel if "time" in rec["real"].get(m, {})])
    return [LAB.get(m, m), f"{np.mean(v['fe55']):.3f}", f"{100 * 1.4826 * np.median(np.abs(rat - np.median(rat))):.2f} %",
            f"{np.mean(v['muon']):.2f}", f"{np.mean(v['worm']):.2f}", f"{np.mean(v['pair4']):.2f}", f"{t:.2f}"]


groups = [("all", lambda r: True), ("Fe-55 data sets", lambda r: r["kind"] == "Fe-55"), ("Am-241 data sets", lambda r: r["kind"] == "Am-241"),
          ("skip-noise study", lambda r: r["kind"] == "skip-noise"), ("noisy (bg spread >= 1 e-)", lambda r: r["noise"] >= 1),
          ("dark >= 1 e-/pix", lambda r: r["lam"] >= 1), ("occupancy >= 30 %", lambda r: r["occupancy"] >= 0.3)]
for gname, f in groups:
    sel = [r for r in recs if f(r)]
    TABLES.append((f"Wide test, {gname} ({len(sel)} image-CCDs)", md([wide_row(sel, m) for m in M],
                   ["method", "Fe-55 ok", "Fe-55 E rms", "muon", "worm", "pair 4 px", "s / image-CCD (median)"])))
fig, axs = plt.subplots(2, 2, figsize=(14, 8)); MS = ["WADERs", "MHC", "GMX", "adaptive+GMX"]; Wd = 0.8 / len(MS)
for a, (k, t) in zip(axs.ravel(), [("fe55", "Fe-55 clusters, sigma 0.3-1.6 px (one cluster, E within 5 %)"), ("muon", "muons"),
                                   ("worm", "electron worms"), ("pair4", "two X-rays 4 px apart (both resolved)")]):
    for i, m in enumerate(MS):
        v = [np.mean([ok_(r, k) for rec in recs if f(rec) for r in rec["inject"][m] if r["kind"] == k]) for _, f in groups]
        a.bar(np.arange(len(groups)) + (i - len(MS) / 2 + .5) * Wd, v, Wd, color=COL[m], label=LAB.get(m, m))
    a.set_xticks(range(len(groups))); a.set_xticklabels([g for g, _ in groups], rotation=20, ha="right", fontsize=7.5)
    a.set_ylim(0, 1.05); a.set_title(t, fontsize=9.5, loc="left"); a.grid(axis="y", alpha=.3)
axs[0, 0].legend(fontsize=8)
fig.suptitle(f"Wide test: {len(recs)} image-CCDs, {len({r['dataset'] for r in recs})} data sets, injected objects with known truth", fontsize=11)
fig.tight_layout(); save(fig, "fig20_wide_vs_waders.png")

# adaptive switch
fig, ax = plt.subplots(figsize=(7.5, 5))
for kind, c in (("Fe-55", "tab:blue"), ("Am-241", "tab:red"), ("skip-noise", "tab:green")):
    s = [r for r in recs if r["kind"] == kind]
    ax.scatter([r["sigma"] for r in s], [r["noise"] for r in s], s=8, c=c, alpha=.6, label=kind)
x = np.logspace(-1.5, 1, 50); ax.plot(x, np.maximum(3 * x, 0.5), "k--", lw=1, label="switch: s >= 3 sigma_read and >= 0.5 e-")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("read noise sigma_read (spread of pixels near 0 e-) [e-]")
ax.set_ylabel("fitted background spread s [e-]"); ax.legend(fontsize=8)
ax.set_title("the adaptive switch on all image-CCDs: above the line -> 'unmodelled' (hysteresis)", fontsize=9, loc="left")
save(fig, "fig21_adaptive_switch.png")

# ------------------------------------------------------------------------------------------------ Fe-55 injection
S = collections.defaultdict(list); P = collections.defaultdict(list); K = collections.defaultdict(list)
for dd, skip_mhc in ((os.path.join(D, "results_fe55"), False), (os.path.join(D, VAL, "fe55"), True)):
    for f in glob.glob(os.path.join(dd, "inject_*.json")):
        d = json.load(open(f))
        for s, v in d["single"].items():
            for m, x in v.items():
                if not m.startswith("_") and not (skip_mhc and m == "MHC"): S[(float(s), m)].append(x)
        for dist, v in d["pairs"].items():
            for m, x in v.items():
                if not (skip_mhc and m == "MHC"): P[(float(dist), m)].append(x)
    for f in glob.glob(os.path.join(dd, "real_*.json")):
        d = json.load(open(f))
        for m, x in d["methods"].items():
            if x["front"] and not (skip_mhc and m == "MHC"): K[m].append(x["front"][0] / 5.899)
MF = ["WADERs", "MFD+G", "MHC", "GMX", "adaptive+GMX"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
sig = (0.3, 0.8, 1.4, 1.8)
for m in MF:
    if not S[(0.3, m)]: continue
    ax[0].plot(sig, [np.median([x["med"] for x in S[(s, m)]]) for s in sig], "o-", color=COL[m], label=LAB.get(m, m))
    ax[0].plot(sig, [min((x["med"] for x in S[(s, m)]), key=lambda q: -abs(q - 1)) for s in sig], "x:", color=COL[m])
    ax[1].plot([3, 5], [np.mean([x["resolved"] for x in P[(d, m)]]) for d in (3.0, 5.0)], "o-", color=COL[m], label=LAB.get(m, m))
ax[0].axhline(1, color="k", lw=0.6); ax[0].set_xlabel("cluster width sigma_xy [px]"); ax[0].set_ylabel("E_rec / E_true")
ax[0].set_title("Fe-55 injection (8 image-CCDs, 220 per width): median (o) and worst CCD (x)", fontsize=9, loc="left"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("distance between two diffuse (sigma 1.4 px) Fe-55 clusters [px]"); ax[1].set_ylabel("fraction resolved (both within 5 %)")
ax[1].set_title("close pairs", fontsize=9, loc="left"); ax[1].set_ylim(-0.02, 1.02); ax[1].legend(fontsize=8)
for i, m in enumerate(MF):
    if K[m]: ax[2].errorbar(i, np.mean(K[m]), yerr=[[np.mean(K[m]) - min(K[m])], [max(K[m]) - np.mean(K[m])]], fmt="o", color=COL[m], capsize=4)
ax[2].set_xticks(range(len(MF))); ax[2].set_xticklabels([LAB.get(m, m) for m in MF], rotation=15)
ax[2].set_ylabel("real front-side Ka peak / 5.899 keV"); ax[2].set_title("real Fe-55 Ka (mean, range over the 8 CCDs)", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig22_fe55.png")
TABLES.append(("Fe-55 injection (DM09 / DM10, EXT1 / EXT4, images 0-1): median E_rec/E_true (worst CCD) per width, pairs, real Ka",
               md([[LAB.get(m, m)] + [f"{np.median([x['med'] for x in S[(s, m)]]):.4f} ({min((x['med'] for x in S[(s, m)]), key=lambda q: -abs(q - 1)):.3f})" for s in sig]
                   + [f"{np.mean([x['resolved'] for x in P[(5.0, m)]]):.2f}", f"{np.mean(K[m]):.4f}", f"{100 * (max(K[m]) - min(K[m])):.2f} %"] for m in MF if S[(0.3, m)]],
                  ["method", "sigma 0.3", "0.8", "1.4", "1.8", "pair 5 px resolved", "real Ka / 5.899", "Ka spread"])))

# ------------------------------------------------------------------------------------------------ tracks
rs = []
for f in glob.glob(os.path.join(D, "results_tracks", "*.json")):
    d = json.load(open(f)); rs += [dict(r, ext=d["ext"]) for r in d.get("inject", [])]
for f in glob.glob(os.path.join(D, VAL, "tracks", "*.json")):
    rs += [r for r in json.load(open(f)) if r["method"] != "MHC"]
KT = ["muon", "worm", "blob", "single", "fe55", "alpha", "hotcol", "lowE", "pair3", "pair4", "pair6"]
sc = lambda v, k: np.mean([r["resolved"] for r in v]) if k.startswith("pair") else np.mean([r["n"] == 1 and abs(r["frac_largest"] - 1) < 0.05 for r in v])
fig, axs = plt.subplots(2, 1, figsize=(14, 7.5))
for a, ext in zip(axs, (1, 3)):
    rows = []
    for i, m in enumerate(MS):
        v = [sc([r for r in rs if r["ext"] == ext and r["method"] == m and r["kind"] == k], k) for k in KT]
        a.bar(np.arange(len(KT)) + (i - len(MS) / 2 + .5) * Wd, v, Wd, color=COL[m], label=LAB.get(m, m))
    for m in ["WADERs", "MFD+G", "hybrid v1", "MHC", "GMX", "adaptive+GMX"]:
        v = [r for r in rs if r["ext"] == ext and r["method"] == m]
        if v: rows.append([LAB.get(m, m)] + [f"{sc([r for r in v if r['kind'] == k], k):.2f}" for k in KT])
    TABLES.append((f"Injected objects in real Am-241 images, EXT{ext} (3 images x 3 seeds, success fraction)", md(rows, ["method"] + KT)))
    a.set_xticks(range(len(KT))); a.set_xticklabels(KT); a.set_ylim(0, 1.05); a.grid(axis="y", alpha=.3)
    a.set_title(f"EXT{ext} ({'low background, 0.12 e- noise' if ext == 1 else 'denser CCD'}): one cluster with E within 5 % (pairs: both resolved)", fontsize=9.5, loc="left")
axs[0].legend(fontsize=8, ncol=4)
fig.tight_layout(); save(fig, "fig23_tracks.png")

# ------------------------------------------------------------------------------------------------ few-electron
R_ = collections.defaultdict(list); F_ = collections.defaultdict(list)
for dd, skip in ((os.path.join(D, "results_lowE"), ()), (os.path.join(D, VAL, "lowE"), ("MHC",))):
    for f in glob.glob(os.path.join(dd, "sim_*.json")):
        d = json.load(open(f))
        for m, rows in d["inject"].items():
            if m not in skip: R_[(d["lam"], m)] += rows
        for m, E in d.get("fakes", {}).items():
            if m not in skip: F_[(d["lam"], m)].append(E)
ML = ["WADERs", "MHC", "GMX", "adaptive+GMX", "DM-style >=1e-"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ne = [1, 2, 3, 4, 5, 7, 10, 15, 20]; rows = []
for a, lam in zip(ax, (0.001, 0.01, 0.05)):
    for m in ML:
        x = np.array(R_[(lam, m)], float)
        if not x.size: continue
        eff = [x[x[:, 0] == n, 3].mean() for n in ne]
        E = np.concatenate([np.round(e) for e in F_[(lam, m)]]) if F_[(lam, m)] else np.array([])
        fk = np.sum((E >= 2) & (E <= 5)) / (490 * 3490 * max(len(F_[(lam, m)]), 1) / 1e6)
        a.plot(ne, eff, "o-", color=COL.get(m, "tab:cyan"), label=f"{LAB.get(m, m)} (fakes 2-5 e-: {fk:.1f}/Mpix)")
        rows.append([lam, LAB.get(m, m)] + [f"{e:.2f}" for e in eff[1:6]] + [f"{fk:.1f}"])
    a.set_xscale("log"); a.set_ylim(-0.02, 1.02); a.set_xlabel("event charge [e-]"); a.set_title(f"dark charge {lam} e-/pix, 0.15 e- noise", fontsize=9, loc="left")
    a.legend(fontsize=6.5)
ax[0].set_ylabel("efficiency")
fig.tight_layout(); save(fig, "fig24_few_electron.png")
TABLES.append(("Few-electron events on simulated blanks (efficiency by event charge; fakes = clusters of 2-5 e- per Mpix with nothing injected)",
               md(rows, ["dark e-/pix", "method", "2 e-", "3 e-", "4 e-", "5 e-", "7 e-", "fakes/Mpix"])))

# ------------------------------------------------------------------------------------------------ crowded EXT4
import am241_paths; am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
from amfit import Model, summary
from scan import load as wload
_cwd = os.getcwd(); os.chdir(os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis"))
dW = wload(4); os.chdir(_cwd); EW = dW["Energy"][dW["sxy"] < 0.7]
fs = sorted(glob.glob(os.path.join(D, VAL, "crowded", "*.npz")))
EA, SA, BA = (np.concatenate([np.load(f)[f"adaptive_GMX_{k}"] for f in fs]) for k in ("E", "sxy", "blend"))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
b = np.arange(10, 30.01, 0.2)
ax[0].hist(EW, b, histtype="step", color=COL["WADERs"], lw=1.4, label=f"WADERs, sigma_xy < 0.7 ({EW.size} clusters)")
m = (SA < 0.7) & (BA < 0.02)
ax[0].hist(EA[m], b, histtype="step", color=COL["adaptive+GMX"], lw=1.4, label=f"A_GMX, sigma_xy < 0.7, blend < 0.02 ({m.sum()} clusters)")
for e, n in ((13.95, "Np La"), (17.75, "Np Lb"), (20.78, "Np Lg"), (22.16, "Ag Ka"), (26.34, "gamma 26.3")):
    ax[0].axvline(e, color="k", lw=0.5, ls=":"); ax[0].text(e, ax[0].get_ylim()[1] * 0.97, n, rotation=90, fontsize=7, va="top")
ax[0].set_xlabel("cluster energy [keV]"); ax[0].set_ylabel("clusters / 0.2 keV"); ax[0].legend(fontsize=8)
ax[0].set_title("crowded Am-241 CCD (29 images, EXT4): the same events, two clusterings", fontsize=9, loc="left")
cuts = (0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 10); Z = []; N = []
MAIN = ("La", "Lb", "Lg", "g26")
for c in cuts:
    mm = (SA < 0.7) & (BA < c); r = Model("Ag").fit(EA[mm], 0.985); s = summary(r)
    Z.append(np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))); N.append(mm.sum())
rW = Model("Ag").fit(EW, 0.985); sW = summary(rW); ZW = np.sqrt(sum((sW[g][0] / sW[g][1]) ** 2 for g in MAIN if sW[g][1] > 0))
ax[1].plot(range(len(cuts)), Z, "o-", color=COL["adaptive+GMX"], label="A_GMX, sigma_xy < 0.7 and blend < cut")
ax[1].axhline(ZW, color=COL["WADERs"], ls="--", label=f"WADERs (sigma_xy < 0.7): {ZW:.1f}")
for i, n in enumerate(N): ax[1].text(i, Z[i] + 0.6, f"{n}", fontsize=7, ha="center")
ax[1].set_xticks(range(len(cuts))); ax[1].set_xticklabels([str(c) if c < 10 else "none" for c in cuts]); ax[1].set_xlabel("blend cut (numbers: clusters kept)")
ax[1].set_ylabel("combined Np L-line significance Z"); ax[1].legend(fontsize=8)
ax[1].set_title("the purity cut: keep clusters whose charge is not shared with neighbours", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig25_crowded.png")

# ------------------------------------------------------------------------------------------------ run time
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for m in ["WADERs", "MFD+G", "MHC", "GMX", "adaptive+GMX"]:
    px = np.array([r["shape"][0] * r["shape"][1] / 1e6 for r in recs if "time" in r["real"].get(m, {})])
    t = np.array([r["real"][m]["time"] for r in recs if "time" in r["real"].get(m, {})])
    ax.scatter(px, np.maximum(t, 1e-3), s=6, alpha=.5, color=COL[m], label=f"{LAB.get(m, m)} (median {np.median(t):.2f} s)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("image-CCD size [Mpix]"); ax.set_ylabel("run time [s]"); ax.legend(fontsize=8)
ax.set_title("run time per image-CCD (one core, Apple M-series)", fontsize=9, loc="left")
save(fig, "fig26_runtime.png")

open(os.path.join(OUT, "..", "perf_tables.tex"), "w").write("% generated by clustering_tests/make_agmx_perf.py\n" +
    "\n\n".join(f"\\paragraph{{{_tex(t)}}}\n{x}" for t, x in TABLES) + "\n")
print("tables:", len(TABLES))
