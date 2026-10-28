"""injection summary: per-variant metrics and by length"""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = [("pkg", "package"), ("mu_c60_g6", "finder v1 (Hough)"), ("mu2", "v2, cap rule"), ("mu2_noshare", "v2, no sharing"), ("mu3", "v3, EM sharing")]
res = {v: sum((json.load(open(f))["res"] for f in glob.glob(f"inj/*_{v}.json")), []) for v, _ in V}
met = [("complete", "median share of the muon\nin its best cluster", lambda r: np.median([x["complete"] for x in r])),
       ("one", "muons with >= 90 %\nin one cluster", lambda r: np.mean([x["complete"] >= 0.9 for x in r])),
       ("frags", "mean fragments (>= 5 %)", lambda r: np.mean([x["n_frag"] for x in r])),
       ("foreign", "median foreign charge\nin the muon's cluster", lambda r: np.nanmedian([x["foreign"] for x in r])),
       ("correct", "correct (>= 90 % complete\nand <= 10 % foreign)", lambda r: np.mean([x["correct"] for x in r]))]
fig, axs = plt.subplots(1, 5, figsize=(26, 5.5))
cols = ["#8c8c8c", "#c9a227", "#d07a2a", "#b04a4a", "#2a64c8"]
table = {}
for a, (k, t, f) in zip(axs, met):
    vals = [f(res[v]) if res[v] else np.nan for v, _ in V]; table[k] = vals
    a.bar(range(len(V)), vals, color=cols); a.set_xticks(range(len(V))); a.set_xticklabels([n for _, n in V], rotation=30, ha="right", fontsize=10)
    a.set_title(t, fontsize=12)
    for i, x in enumerate(vals): a.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=10)
fig.suptitle(f"200 muons injected into L2 images 9/10 EXT1/EXT4 (2 seeds each), full clustering", fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig("report/inj_bars.png", dpi=80)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
bins = [40, 100, 160, 220, 280, 340, 400]
for (v, n), c in zip(V, cols):
    r = res[v]
    if not r: continue
    L = np.array([x["L"] for x in r]); cp = np.array([x["complete"] for x in r]); fr = np.array([x["n_frag"] for x in r])
    idx = np.digitize(L, bins) - 1; xc = 0.5 * (np.array(bins[1:]) + bins[:-1])
    axs[0].plot(xc, [np.median(cp[idx == i]) for i in range(len(xc))], "o-", color=c, label=n)
    axs[1].plot(xc, [np.mean(fr[idx == i]) for i in range(len(xc))], "o-", color=c, label=n)
axs[0].set_ylabel("median completeness"); axs[1].set_ylabel("mean fragments"); [a.set_xlabel("projected muon length [px]") for a in axs]
axs[0].legend(); fig.tight_layout(); fig.savefig("report/inj_vs_length.png", dpi=80)
json.dump(dict(variants=[v for v, _ in V], **table, n={v: len(res[v]) for v, _ in V}), open("report/inj_table.json", "w"))
print(json.dumps(table, default=float))
