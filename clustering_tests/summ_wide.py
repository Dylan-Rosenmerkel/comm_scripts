"""Summary of the wide test (results_wide/*.json): tables (markdown) + figures."""
import json, glob, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results_wide")
# extra methods run separately on the same images / injected objects: name=dir arguments
EXTRA = dict(a.split("=", 1) for a in sys.argv[2:]) or {"adaptive": "results_wide_adaptive", "GMX": "results_wide_gmx"}
M = ["WADERs", "B auto", "MFD+G", "hybrid v1", "hybrid v2", "MHC"] + list(EXTRA)
COL = {"adaptive": "k", "GMX": "tab:purple", "WADERs": "0.45", "B auto": "tab:olive", "MFD+G": "tab:red", "hybrid v1": "tab:orange", "hybrid v2": "tab:green", "MHC": "tab:blue"}
KINDS = ["fe55", "muon", "worm", "pair4"]
recs = []
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    rec = json.load(open(f)); ok_all = True
    for name, d in EXTRA.items():
        fa = os.path.join(HERE, d, os.path.basename(f))
        if not os.path.exists(fa): ok_all = False; break      # keep only image-CCDs that have all methods
        a = json.load(open(fa))
        rec["real"][name] = a["real"].get(name, {})
        if "inject" in rec and "inject" in a: rec["inject"][name] = a["inject"][name]
        else: rec.pop("inject", None)
    if ok_all: recs.append(rec)
print(f"{len(recs)} image-CCDs from {len({r['dataset'] for r in recs})} data sets")


def short(ds):
    return ds.split("/")[-1].replace("_Low_Temp", "").replace("Production_Modules_Data", "")


def ok(r, k):
    if k == "pair4": return r["resolved"]
    return r["n"] == 1 and abs(r["ratio"] - 1) < 0.05


def inj_table(sel):
    rows = {}
    for m in M:
        v = collections.defaultdict(list); ratios = []
        for rec in sel:
            for r in rec.get("inject", {}).get(m, []):
                v[r["kind"]].append(ok(r, r["kind"]))
                if r["kind"] == "fe55" and r["n"] >= 1: ratios.append(r["ratio"])
        rows[m] = dict({k: (np.mean(v[k]) if v[k] else np.nan) for k in KINDS},
                       fe55_med=np.median(ratios) if ratios else np.nan,
                       fe55_rms=1.4826 * np.median(np.abs(np.array(ratios) - np.median(ratios))) if ratios else np.nan,
                       n=sum(len(x) for x in v.values()))
    return rows


def ka_table(sel):
    rows = {}
    for m in M:
        pk = [rec["real"][m]["ka"][0] for rec in sel if rec["real"].get(m, {}).get("ka")]
        w = [rec["real"][m]["ka"][2] for rec in sel if rec["real"].get(m, {}).get("ka")]
        rows[m] = dict(n=len(pk), mean=np.mean(pk) if pk else np.nan, sd=np.std(pk) if pk else np.nan, width=np.median(w) * 1000 if w else np.nan)
    return rows


lines = ["# Wide test: all clustering methods on all available images\n",
         f"{len(recs)} image-CCDs, {len({r['dataset'] for r in recs})} data sets. Injection: per image-CCD 12 Fe-55 clusters "
         "(sigma 0.3-1.6 px), 4 muons, 4 worms, 4 pairs 4 px apart. Success = one cluster with E within 5 % (pairs: both "
         "found, each within 10 %). Ka = fit of each method's own clusters in 4.5-7 keV (peak in keV, spread = standard "
         "deviation over image-CCDs, width = median fitted sigma).\n"]
groups = [("ALL", recs)] + [(k, [r for r in recs if r["kind"] == k]) for k in ("Fe-55", "Am-241", "skip-noise")]
for name, sel in groups:
    if not sel: continue
    t = inj_table(sel)
    lines.append(f"\n## {name} ({len(sel)} image-CCDs)\n")
    lines.append("| method | Fe-55 success | Fe-55 E/E_true (median, rms) | muon | worm | pair 4 px | real Ka peak [keV] (n, spread) | Ka width [eV] | clusters / image | time / image [s] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    ka = ka_table(sel)
    for m in M:
        nc = np.median([rec["real"][m]["n"] for rec in sel if "n" in rec["real"].get(m, {})])
        tt = np.median([rec["real"][m]["time"] for rec in sel if "time" in rec["real"].get(m, {})])
        a = t[m]; k = ka[m]
        lines.append(f"| {m} | {a['fe55']:.3f} | {a['fe55_med']:.4f}, {a['fe55_rms']:.4f} | {a['muon']:.2f} | {a['worm']:.2f} | {a['pair4']:.2f} | "
                     + (f"{k['mean']:.4f} ({k['n']}, {k['sd']:.4f})" if k["n"] else "-") + f" | {k['width']:.0f} | {nc:.0f} | {tt:.1f} |")
# by data set (injection Fe-55 success, worm success, pairs)
lines.append("\n## Per data set: Fe-55 success / muon / worm / pair (injection)\n")
lines.append("| data set | n | " + " | ".join(M) + " |"); lines.append("|---|---|" + "---|" * len(M))
for ds in sorted({r["dataset"] for r in recs}):
    sel = [r for r in recs if r["dataset"] == ds]; t = inj_table(sel)
    lines.append(f"| {short(ds)} | {len(sel)} | " + " | ".join(f"{t[m]['fe55']:.2f}/{t[m]['muon']:.2f}/{t[m]['worm']:.2f}/{t[m]['pair4']:.2f}" for m in M) + " |")
# by image condition
lines.append("\n## By image condition (all data sets): Fe-55 success / worm / pair\n")
conds = [("noise < 0.3 e-", lambda r: r["noise"] < 0.3), ("noise 0.3-1 e-", lambda r: 0.3 <= r["noise"] < 1), ("noise >= 1 e-", lambda r: r["noise"] >= 1),
         ("dark < 0.1 e-/pix", lambda r: r["lam"] < 0.1), ("dark 0.1-1", lambda r: 0.1 <= r["lam"] < 1), ("dark >= 1", lambda r: r["lam"] >= 1),
         ("occupancy < 5 %", lambda r: r["occupancy"] < 0.05), ("occupancy 5-30 %", lambda r: 0.05 <= r["occupancy"] < 0.3), ("occupancy >= 30 %", lambda r: r["occupancy"] >= 0.3)]
lines.append("| condition | n | " + " | ".join(M) + " |"); lines.append("|---|---|" + "---|" * len(M))
for cn, f in conds:
    sel = [r for r in recs if f(r)]
    if not sel: continue
    t = inj_table(sel)
    lines.append(f"| {cn} | {len(sel)} | " + " | ".join(f"{t[m]['fe55']:.2f}/{t[m]['worm']:.2f}/{t[m]['pair4']:.2f}" for m in M) + " |")
errs = collections.Counter(m for r in recs for m, v in r["real"].items() if "error" in v)
lines.append(f"\nMethod errors (image-CCDs where the method raised): {dict(errs) or 'none'}; background fallback used on "
             f"{sum(r.get('bg') == 'fallback' for r in recs)} image-CCDs.\n")
open(os.path.join(HERE, "wide_summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))

# figure: success by kind and by condition
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
t = inj_table(recs); W = 0.8 / len(M)
for i, m in enumerate(M):
    ax[0].bar(np.arange(4) + (i - len(M) / 2 + .5) * W, [t[m][k] for k in KINDS], W, color=COL[m], label=m)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Fe-55 (sigma 0.3-1.6)", "muon", "electron worm", "X-ray pair 4 px"])
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("success fraction"); ax[0].set_title(f"injection, all {len(recs)} image-CCDs", fontsize=10); ax[0].legend(fontsize=8, ncol=3)
cn = [c for c, f in conds if any(f(r) for r in recs)]
for i, m in enumerate(M):
    v = []
    for c, f in conds:
        sel = [r for r in recs if f(r)]
        if sel: v.append(np.nanmean([inj_table(sel)[m][k] for k in KINDS]))
    ax[1].bar(np.arange(len(v)) + (i - len(M) / 2 + .5) * W, v, W, color=COL[m])
ax[1].set_xticks(range(len(cn))); ax[1].set_xticklabels(cn, rotation=30, ha="right", fontsize=8); ax[1].set_ylim(0, 1.05)
ax[1].set_title("mean success over the 4 kinds, by image condition", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_wide.png"), dpi=130)
