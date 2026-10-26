"""Build docs/clustering_methods_report.pdf from the saved results of the clustering study (2026-10-23 .. 25).
All numbers are recomputed here from the result files listed in DATA (copies in
~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/)."""
import glob, json, os, sys, collections, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_text_methods as RT
import report_figs as RF

D = os.path.expanduser(os.environ.get("CLUSTER_DATA", "~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25"))
DATA = dict(
    wide=os.path.join(D, "results_wide"),
    wide_extra={"adaptive": os.path.join(D, "results_wide_adaptive_v2"), "GMX": os.path.join(D, "results_wide_gmx"),
                "adaptive+GMX": os.path.join(D, "results_wide_adaptive_gmx")},
    tracks=os.path.join(D, "results_tracks"), tracks_extra=[os.path.join(D, "results_adaptive_v2", "tracks"), os.path.join(D, "results_adaptive_gmx", "tracks")],
    lowE=os.path.join(D, "results_lowE"), lowE_extra=[os.path.join(D, "results_adaptive_v2", "lowE"), os.path.join(D, "results_adaptive_gmx", "lowE")],
    fe55=os.path.join(D, "results_fe55"), fe55_extra=[os.path.join(D, "results_adaptive_v2", "fe55"), os.path.join(D, "results_adaptive_gmx", "fe55")],
    crowded=os.path.join(D, "crowded_fits.json"),
    am_plots=[os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis/final_fit.png"), os.path.join(D, "final_fit_GMX_cut0.7_clusters_gmx.png")],
)
M_ALL = ["WADERs", "B auto", "MFD+G", "hybrid v1", "hybrid v2", "MHC", "adaptive", "GMX", "adaptive+GMX"]
COL = {"WADERs": "0.45", "B auto": "tab:olive", "MFD+G": "tab:red", "hybrid v1": "tab:orange", "hybrid v2": "tab:green",
       "MHC": "tab:blue", "adaptive": "k", "GMX": "tab:purple", "adaptive+GMX": "tab:cyan"}


# ------------------------------------------------------------------ pages
def text_page(pdf, title, blocks, mono=False, fs=8.4, width=100):
    fig = plt.figure(figsize=(8.5, 11)); fig.text(0.06, 0.955, title, fontsize=14, weight="bold"); y = 0.92
    for h, body in blocks:
        if h:
            if y < 0.08:
                pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=(8.5, 11)); y = 0.95
            fig.text(0.06, y, h, fontsize=10.5, weight="bold"); y -= 0.022
        for p in body:
            lines = sum((textwrap.wrap(L, 115) or [""] for L in p.split("\n")), []) if mono else textwrap.wrap(p, width)
            for L in lines:
                if y < 0.04:
                    pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=(8.5, 11)); y = 0.95
                fig.text(0.07, y, L, fontsize=fs - (1.2 if mono else 0), family="monospace" if mono else None); y -= 0.0155 if not mono else 0.0138
            y -= 0.007
        y -= 0.008
    pdf.savefig(fig); plt.close(fig)


def table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    f = lambda r: "  ".join(str(c).rjust(w[i]) if i else str(c).ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([f(header), "-" * (sum(w) + 2 * len(w))] + [f(r) for r in rows])


# ------------------------------------------------------------------ wide test
def load_wide():
    recs = []
    for f in sorted(glob.glob(os.path.join(DATA["wide"], "*.json"))):
        rec = json.load(open(f)); ok = True
        for name, d in DATA["wide_extra"].items():
            fa = os.path.join(d, os.path.basename(f))
            if not os.path.exists(fa): ok = False; break
            a = json.load(open(fa)); rec["real"][name] = a["real"].get(name, {})
            if "inject" in rec and "inject" in a: rec["inject"][name] = a["inject"][name]
            else: rec.pop("inject", None)
        if ok: recs.append(rec)
    return recs


def wide_ok(r, k):
    return r["resolved"] if k == "pair4" else (r["n"] == 1 and abs(r["ratio"] - 1) < 0.05)


def wide_table(sel, M):
    rows = []
    for m in M:
        v = collections.defaultdict(list); rat = []
        for rec in sel:
            for r in rec.get("inject", {}).get(m, []):
                v[r["kind"]].append(wide_ok(r, r["kind"]))
                if r["kind"] == "fe55" and r["n"] >= 1: rat.append(r["ratio"])
        rat = np.array(rat); t = np.median([rec["real"][m]["time"] for rec in sel if "time" in rec["real"].get(m, {})])
        rows.append([m, f"{np.mean(v['fe55']):.3f}", f"{np.median(rat):.4f}", f"{100 * 1.4826 * np.median(np.abs(rat - np.median(rat))):.2f}",
                     f"{np.mean(v['muon']):.2f}", f"{np.mean(v['worm']):.2f}", f"{np.mean(v['pair4']):.2f}", f"{t:.1f}"])
    return table(rows, ["method", "Fe-55 ok", "E/E_true", "rms [%]", "muon", "worm", "pair 4px", "s/img"])


def fig_wide(recs, M):
    conds = [("all", lambda r: True), ("Fe-55 sets", lambda r: r["kind"] == "Fe-55"), ("Am-241 sets", lambda r: r["kind"] == "Am-241"),
             ("skip-noise", lambda r: r["kind"] == "skip-noise"), ("noise >= 1 e-", lambda r: r["noise"] >= 1),
             ("dark >= 1 e-/pix", lambda r: r["lam"] >= 1), ("occupancy >= 30 %", lambda r: r["occupancy"] >= 0.3)]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5)); W = 0.8 / len(M)
    for a, (k, lab) in zip(axs.ravel(), [("fe55", "Fe-55 clusters (sigma 0.3-1.6 px): one cluster, E within 5 %"), ("muon", "muons: one cluster, E within 5 %"),
                                         ("worm", "electron worms: one cluster, E within 5 %"), ("pair4", "two 5.9 keV X-rays 4 px apart: both resolved")]):
        for i, m in enumerate(M):
            v = []
            for cn, f in conds:
                s = [wide_ok(r, k) for rec in recs if f(rec) for r in rec.get("inject", {}).get(m, []) if r["kind"] == k]
                v.append(np.mean(s) if s else np.nan)
            a.bar(np.arange(len(conds)) + (i - len(M) / 2 + .5) * W, v, W, color=COL[m], label=m)
        a.set_xticks(range(len(conds))); a.set_xticklabels([c for c, _ in conds], fontsize=7.5, rotation=20, ha="right")
        a.set_ylim(0, 1.05); a.set_title(lab, fontsize=9.5); a.grid(axis="y", alpha=.3)
    axs[0, 0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"Wide test: {len(recs)} image-CCDs from {len({r['dataset'] for r in recs})} data sets (injected objects, truth known)", fontsize=11)
    fig.tight_layout(); return fig


# ------------------------------------------------------------------ tracks / lowE / Fe-55
def load_tracks():
    """WADERs, MFD+G, hybrids, MHC from the main track run; adaptive / GMX / adaptive+GMX from the validation runs
    (same images, same seeds -> same injected objects)"""
    rs = []
    for f in glob.glob(os.path.join(DATA["tracks"], "*.json")):
        d = json.load(open(f))
        for r in d.get("inject", []): rs.append(dict(r, ext=d["ext"]))
    for dd in DATA["tracks_extra"]:
        for f in glob.glob(os.path.join(dd, "*.json")):
            rs += [r for r in json.load(open(f)) if r["method"] != "MHC"]
    return rs


def tracks_table(rs, ext):
    K = ["muon", "worm", "blob", "single", "fe55", "alpha", "hotcol", "lowE", "pair3", "pair4", "pair6"]
    sc = lambda v, k: np.mean([r["resolved"] for r in v]) if k.startswith("pair") else np.mean([r["n"] == 1 and abs(r["frac_largest"] - 1) < 0.05 for r in v])
    rows = []
    for m in M_ALL:
        v = [r for r in rs if r["ext"] == ext and r["method"] == m]
        if v: rows.append([m] + [f"{sc([r for r in v if r['kind'] == k], k):.2f}" if any(r["kind"] == k for r in v) else "-" for k in K])
    return table(rows, ["method"] + K)


def lowE_table():
    R = collections.defaultdict(list); F = collections.defaultdict(list)
    for dd in [DATA["lowE"]] + DATA["lowE_extra"]:
        for f in glob.glob(os.path.join(dd, "*.json")):
            if "real" in os.path.basename(f): continue
            d = json.load(open(f))
            for m, rows in d["inject"].items():
                if dd != DATA["lowE"] and m == "MHC": continue
                R[(d["lam"], m)] += rows
            for m, E in d.get("fakes", {}).items():
                if dd != DATA["lowE"] and m == "MHC": continue
                F[(d["lam"], m)].append(E)
    out = []
    for lam in (0.001, 0.01, 0.05):
        rows = []
        for m in ["WADERs", "MFD+G 5sig", "MFD+G 3sig count", "MHC", "adaptive", "GMX", "adaptive+GMX", "DM-style >=1e-"]:
            a = np.array(R[(lam, m)], float)
            if not a.size: continue
            eff = [a[a[:, 0] == n, 3].mean() for n in (2, 3, 4, 5, 7, 10)]
            ex = a[(a[:, 0] == 3) & (a[:, 3] == 1)]
            E = np.concatenate([np.round(e) for e in F[(lam, m)]]) if F[(lam, m)] else np.array([])
            npx = 490 * 3490 * max(len(F[(lam, m)]), 1) / 1e6
            rows.append([m] + [f"{e:.2f}" for e in eff] + [f"{np.mean(np.round(ex[:, 4]) == 3):.2f}" if len(ex) else "-",
                                                           f"{np.sum((E >= 2) & (E <= 5)) / npx:.1f}", f"{np.sum(E == 1) / npx:.0f}"])
        out.append((f"dark charge {lam} e-/pix, read noise 0.15 e- (simulated blank, 2 x 1.7 Mpix)",
                    table(rows, ["method", "2 e-", "3 e-", "4 e-", "5 e-", "7 e-", "10 e-", "exact@3", "fakes 2-5e/Mpix", "1e/Mpix"])))
    return out


def fe55_table():
    S = collections.defaultdict(list); P = collections.defaultdict(list); K = collections.defaultdict(list)
    for dd in [DATA["fe55"]] + DATA["fe55_extra"]:
        for f in glob.glob(os.path.join(dd, "inject_*.json")):
            d = json.load(open(f))
            for s, v in d["single"].items():
                for m, x in v.items():
                    if m.startswith("_") or (dd != DATA["fe55"] and m == "MHC"): continue
                    S[(s, m)].append(x)
            for dist, v in d["pairs"].items():
                for m, x in v.items():
                    if dd != DATA["fe55"] and m == "MHC": continue
                    P[(dist, m)].append(x["resolved"])
        for f in glob.glob(os.path.join(dd, "real_*.json")):
            d = json.load(open(f))
            for m, x in d["methods"].items():
                if dd != DATA["fe55"] and m == "MHC": continue
                if x["front"]: K[m].append(x["front"][0] / 5.899)
    rows = []
    for m in M_ALL:
        if not S[("0.3", m)]: continue
        r = [m]
        for s in ("0.3", "0.8", "1.4", "1.8"):
            v = S[(s, m)]; r.append(f"{np.median([x['med'] for x in v]):.4f}/{min((x['med'] for x in v), key=lambda q: -abs(q - 1)):.3f}")
        r += [f"{np.mean(P[('5.0', m)]):.2f}", f"{np.mean(K[m]):.4f}", f"{100 * (max(K[m]) - min(K[m])):.2f}"]
        rows.append(r)
    return table(rows, ["method", "s0.3 med/worst", "s0.8", "s1.4", "s1.8", "pair5px", "Ka/5.899", "spread%"])


# ------------------------------------------------------------------ main
def main():
    out = os.path.join(HERE, "..", "docs", "clustering_methods_report.pdf")
    recs = load_wide(); Mw = [m for m in M_ALL if all(m in r["real"] for r in recs[:1])]
    crowded = json.load(open(DATA["crowded"]))
    figm = RF.fig_methods(os.path.join(HERE, "fig_methods.png"))
    with PdfPages(out) as pdf:
        text_page(pdf, "Clustering methods for skipper-CCD images: report", RT.SUMMARY)
        pdf.savefig(figm); plt.close(figm)
        text_page(pdf, "The methods: what they do, where the ideas come from, why they work", RT.METHODS)
        text_page(pdf, "Wide test: every method on every image", [
            ("", [RT.WIDE_INTRO]),
            ("All image-CCDs", [wide_table(recs, Mw)]),
            ("Fe-55 data sets", [wide_table([r for r in recs if r["kind"] == "Fe-55"], Mw)]),
            ("Am-241 data sets", [wide_table([r for r in recs if r["kind"] == "Am-241"], Mw)]),
            ("Noisy images (fitted background spread >= 1 e-: PD06/DM12 Am-241)", [wide_table([r for r in recs if r["noise"] >= 1], Mw)]),
            ("High dark charge (>= 1 e-/pix)", [wide_table([r for r in recs if r["lam"] >= 1], Mw)]),
        ], mono=True)
        f = fig_wide(recs, Mw); pdf.savefig(f); plt.close(f)
        rows = [[k, v["n"], f"{v['k']:.4f}", f"{v['devndf']:.2f}", f"{v['Z']:.1f}"] + [f"{v.get(g, 0):.1f}" for g in ("La", "Lb", "Lg", "g26", "Ag K")] for k, v in crowded.items()]
        text_page(pdf, "Crowded CCD: Am-241 10-30 keV constrained fit (29 images, EXT4)", [
            ("", [RT.CROWDED_INTRO]),
            ("", [table(rows, ["clustering (cut)", "clusters", "k", "dev/ndf", "Z_np", "La", "Lb", "Lg", "g26", "Ag K"])]),
            ("", [RT.CROWDED_NOTES])], mono=False)
        for p in DATA["am_plots"]:
            if os.path.exists(p):
                im = plt.imread(p); fig = plt.figure(figsize=(11, 11 * im.shape[0] / im.shape[1] + 0.3)); plt.imshow(im); plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        rs = load_tracks()
        text_page(pdf, "Injected objects in real Am-241 images (truth known)", [
            ("", [RT.TRACKS_INTRO]),
            ("EXT1 (0.12 e- read noise, low background)", [tracks_table(rs, 1)]),
            ("EXT3 (denser CCD)", [tracks_table(rs, 3)])], mono=True)
        text_page(pdf, "Few-electron events (1-20 e- in 1-5 pixels)", [("", [RT.LOWE_INTRO])] + [(h, [t]) for h, t in lowE_table()], mono=True)
        text_page(pdf, "Fe-55: injection and the real Ka peak (DM09 / DM10, EXT1 / EXT4, images 0-1)", [
            ("", [RT.FE55_INTRO]), ("", [fe55_table()])], mono=True)
        text_page(pdf, "Recommendations, corrections, files", RT.CLOSING)
    print(os.path.abspath(out))


if __name__ == "__main__":
    main()
