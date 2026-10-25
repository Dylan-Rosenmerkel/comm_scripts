"""Effect of the 2026-10-25 A_GMX changes on the Fe-55 Ka peak of every data set and on the injection test:
old A_GMX (results_wide_adaptive_gmx) vs new (results_wide_agmx_flags) without cuts, and the new front-side Ka
(sigma_xy < 1) with the flag cuts. Writes docs/aGMX/fe55_flags_tables.tex and prints the tables."""
import glob, json, os, re, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25")
NEW = os.environ.get("FLAGS_NEW", os.path.join(HERE, "results_wide_agmx_flags"))   # FLAGS_NEW: another run
OLD = os.path.join(D, "results_wide_adaptive_gmx")
recs = []
for f in sorted(glob.glob(os.path.join(NEW, "*.json"))):
    n = json.load(open(f)); fo = os.path.join(OLD, os.path.basename(f))
    if not os.path.exists(fo): continue
    o = json.load(open(fo)); recs.append((n, o))
print(f"{len(recs)} image-CCDs with old and new results")
ok_ = lambda r, k: r["resolved"] if k == "pair4" else (r["n"] == 1 and abs(r["ratio"] - 1) < 0.05)


def ka(rows):
    """median K-alpha over image-CCDs, robust spread (1.4826 MAD), median width [eV], number fitted. Median, not mean: a
    single low-statistics fit that lands on another peak moves a mean by tens of eV"""
    v = [r for r in rows if r]
    if not v:
        return (np.nan, np.nan, np.nan, 0)
    k = np.array([r[0] for r in v])
    return (float(np.median(k)), float(1.4826 * np.median(np.abs(k - np.median(k)))), np.median([1000 * r[2] for r in v]), len(v))


def tex(rows, header, caption=""):
    """LaTeX table (shrunk to the text width only when wider), the caption below it"""
    esc = lambda x: str(x).replace("%", "\\%").replace("_", "\\_").replace(">=", "$\\ge$")
    L = ["\\begin{table}[H]\\centering\\footnotesize", "\\begin{adjustbox}{max width=\\linewidth}",
         "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule", " & ".join(esc(h) for h in header) + " \\\\", "\\midrule"]
    L += [" & ".join(esc(c) for c in r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if caption:
        L.append("\\caption*{" + caption + "}")
    return "\n".join(L + ["\\end{table}"])


out = []
# 1. Fe-55 Ka per data set
by = collections.defaultdict(list)
for n, o in recs:
    if n["kind"] == "Fe-55":     # "2025-09-07-DM18_DM19_DM20_Low_Temp" -> "2025-09-07-DM18 DM19 DM20"
        by[re.sub(r"_Low.*$", "", n["dataset"].split("/")[-1]).replace("_", " ")].append((n, o))
rows = []
CUTS = ["front", "front, ped<10", "front, peakfrac>0.95", "front, ped<10, peakfrac>0.95"]
print("\nFe-55 Ka (median peak keV over image-CCDs, width eV, clusters per image-CCD in 4.5-7 keV):")
for ds, v in sorted(by.items()):
    old = ka([o["real"]["adaptive+GMX"].get("ka") for n, o in v]); new = ka([n["real"]["all"]["ka"] for n, o in v])
    cols = [ka([n["real"][c]["ka"] for n, o in v]) for c in CUTS]
    nk = [np.mean([n["real"][c]["n"] for n, o in v]) for c in ["all"] + CUTS]
    rows.append([ds, len(v), f"{old[0]:.4f}", f"{new[0]:.4f}"] + [f"{c[0]:.4f} ({c[2]:.0f})" for c in cols] + [f"{100 * (1 - nk[2] / max(nk[1], 1)):.1f} %", f"{100 * (1 - nk[4] / max(nk[1], 1)):.1f} %"])
    print("  " + " | ".join(str(x) for x in rows[-1]))
nofit = [r[0] for r in rows if r[2] == "nan"]
rows = [r for r in rows if r[2] != "nan"]          # data sets where no image-CCD has a K-alpha fit are listed below the table
H = ["data set", "n", "Ka old (all)", "Ka new (all)", "front", "front+ped", "front+pf", "front+both", "front lost to ped", "front lost to both"]
out.append(tex(rows, H, "Fe-55 K-alpha per data set: median fitted peak over image-CCDs [keV] (median width eV); old vs new A\\_GMX "
                  "without cuts, new front side with the flag cuts. No K-alpha fit in any image-CCD of: " + ", ".join(nofit) + "."))
# 2. all Fe-55 image-CCDs together
allv = [(n, o) for n, o in recs if n["kind"] == "Fe-55"]
summ = []
for lab, getter in (("old A_GMX, all clusters", lambda n, o: o["real"]["adaptive+GMX"].get("ka")), ("new A_GMX, all clusters", lambda n, o: n["real"]["all"]["ka"])) + tuple(
        (f"new, {c}", (lambda c: lambda n, o: n["real"][c]["ka"])(c)) for c in CUTS):
    k = ka([getter(n, o) for n, o in allv])
    summ.append([lab, k[3], f"{k[0]:.4f}", f"{1000 * k[1]:.0f}", f"{k[2]:.0f}"])
print("\nall Fe-55 image-CCDs:"); [print("  ", r) for r in summ]
out.append(tex(summ, ["selection", "image-CCDs", "median K-alpha [keV]", "spread [eV]", "width [eV]"],
                  "Fe-55 K-alpha, all Fe-55 image-CCDs: number fitted, median peak, robust spread over image-CCDs, median width. "
                  "old = A\\_GMX before the 25 October changes, new = with them (satellite limit, flags, audit fixes)."))
# 3. injection: old vs new
inj = []
for kind in ("Fe-55", "Am-241", "skip-noise"):
    v = [(n, o) for n, o in recs if n["kind"] == kind and "inject" in n and "inject" in o]
    row = [kind, len(v)]
    for k in ("fe55", "muon", "worm", "pair4"):
        so = np.mean([ok_(r, k) for n, o in v for r in o["inject"]["adaptive+GMX"] if r["kind"] == k])
        sn = np.mean([ok_(r, k) for n, o in v for r in n["inject"]["A_GMX flags"] if r["kind"] == k])
        row.append(f"{so:.3f} / {sn:.3f}")
    inj.append(row)
print("\ninjection success old / new:"); [print("  ", r) for r in inj]
out.append(tex(inj, ["data sets", "image-CCDs", "Fe-55", "muon", "worm", "pair 4 px"],
                  "Injection test (same objects): success old / new A\\_GMX."))
open(os.path.join(HERE, "..", "docs", "aGMX", "fe55_flags_tables.tex"), "w").write(
    "% generated by clustering_tests/summ_wide_flags.py\n" + "\n\n".join(out) + "\n")
