"""lower-background regression: package (newx) vs package + finder (newmu = v2 cap rule, newmu3 = v3 EM + continuity cut)"""
import glob, re, json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from reg_analyse import core, sel
tags = sorted({re.sub(r"_newmu3\.npz$", "", f[4:]) for f in glob.glob("reg/*_newmu3.npz")})
rows = []
for t in tags:
    import os
    z = {v: np.load(f"reg/{t}_{v}.npz", allow_pickle=True) for v in ("newx", "newmu", "newmu3") if os.path.exists(f"reg/{t}_{v}.npz")}
    if "newx" not in z or "newmu3" not in z: continue
    r = dict(tag=t)
    for v, d in z.items():
        r[v] = dict(n=len(d["E"]), crumbs=int((d["npix"] < 5).sum()), largest=int(d["npix"].max()), nmu=int(d["n_mu"]), tube=float(d["tube_frac"]),
                    fe=core(d["E"][sel(d)], 5.0, 6.6, 5.88, 0.08), nE=int(sel(d).sum()))
    rows.append(r)
json.dump(rows, open("report/lowbg_rows.json", "w"), default=float)
fe = [r for r in rows if not r["tag"].startswith("Am241")]
am = [r for r in rows if r["tag"].startswith("Am241")]
print(f"{len(fe)} Fe-55/production CCDs, {len(am)} Am-241")
lab = [re.sub(r"^_2025-", "", r["tag"])[:11] + (" e1" if r["tag"].endswith("ext1") else " e4") for r in fe]
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
x = np.arange(len(fe))
for v, c, o in (("newmu", "#d07a2a", -0.15), ("newmu3", "#2a64c8", 0.15)):
    fe_v = [r for r in fe if v in r]; d = np.array([1e3 * (r[v]["fe"][0] - r["newx"]["fe"][0]) if v in r else np.nan for r in fe]); e = np.array([1e3 * np.hypot(r[v]["fe"][1], r["newx"]["fe"][1]) if v in r else np.nan for r in fe])
    axs[0, 0].errorbar(x + o, d, e, fmt="o", color=c, label={"newmu": "v2 cap rule", "newmu3": "v3 EM + continuity cut"}[v])
    w = np.array([1e3 * (r[v]["fe"][2] - r["newx"]["fe"][2]) if v in r else np.nan for r in fe]); axs[0, 1].plot(x + o, w, "o", color=c)
    axs[1, 0].plot(x + o, [r[v]["n"] - r["newx"]["n"] if v in r else np.nan for r in fe], "o", color=c)
    axs[1, 1].plot(x + o, [r[v]["nmu"] if v in r else np.nan for r in fe], "o", color=c)
    print(v, f"Fe Ka shift median {np.nanmedian(d):+.2f} eV (max |{np.nanmax(np.abs(d)):.1f}|), width {np.nanmedian(w):+.2f} eV, clusters {sum(r[v]['n'] - r['newx']['n'] for r in fe_v):+d}, crumbs {sum(r[v]['crumbs'] for r in fe_v)} vs {sum(r['newx']['crumbs'] for r in fe_v)}, Fe events {sum(r[v]['fe'][3] for r in fe_v)} vs {sum(r['newx']['fe'][3] for r in fe_v)}")
for a, t in zip(axs.ravel(), ("Fe Kα (front side) shift vs package [eV]", "Fe Kα width change [eV]", "change in number of clusters", "finder tracks per CCD")):
    a.axhline(0, color="k", lw=0.6); a.set_title(t, fontsize=12, loc="left", weight="bold"); a.set_xticks(x); a.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
axs[0, 0].legend()
fig.tight_layout(); fig.savefig("report/lowbg_fe.png", dpi=75)
# Am-241 pooled
lines = [(13.9, "Np Lα"), (17.75, "Np Lβ"), (20.8, "Np Lγ"), (26.34, "γ 26.3"), (59.54, "γ 59.5")]
out = {}
for v in ("newx", "newmu", "newmu3"):
    if not all(v in r for r in am): continue
    E = np.concatenate([np.load(f"reg/{r['tag']}_{v}.npz", allow_pickle=True)["E"][sel(np.load(f"reg/{r['tag']}_{v}.npz", allow_pickle=True))] for r in am]) if am else np.array([])
    out[v] = {n: core(E, l - 1.2, l + 1.2, l, 0.25 if l < 30 else 0.4) for l, n in lines}
    out[v]["n"] = int(E.size)
    print("Am241", v, " | ".join(f"{n} {out[v][n][0]:.3f} (n {out[v][n][3]})" for _, n in lines), "| clusters", sum(r[v]["n"] for r in am), "tracks", [r[v]["nmu"] for r in am])
json.dump(out, open("report/lowbg_am.json", "w"), default=float)
if am:
    fig, ax = plt.subplots(figsize=(14, 5))
    B = np.linspace(8, 64, 561)
    for v, c, n in (("newx", "#8c8c8c", "package"), ("newmu", "#d07a2a", "v2 cap rule"), ("newmu3", "#2a64c8", "v3 EM + continuity cut")):
        if not all(v in r for r in am): continue
        E = np.concatenate([np.load(f"reg/{r['tag']}_{v}.npz", allow_pickle=True)["E"][sel(np.load(f"reg/{r['tag']}_{v}.npz", allow_pickle=True))] for r in am])
        ax.hist(E, B, histtype="step", color=c, lw=1.2, label=f"{n} ({E.size} clusters)")
    ax.set_yscale("log"); ax.set_xlabel("energy [keV]"); ax.legend(); ax.set_title(f"Am-241, {len(am)} images (EXT4), front-side clean selection", loc="left", weight="bold")
    fig.tight_layout(); fig.savefig("report/lowbg_am.png", dpi=80)
