import glob, json, numpy as np
from collections import defaultdict
R = defaultdict(list); N = defaultdict(list)
for f in glob.glob("inj/*.json"):
    d = json.load(open(f)); R[d["var"]] += d["res"]; N[d["var"]].append(d["n_cl"])
print(f"{'variant':18s} {'muons':>5s} {'correct':>8s} {'one-cluster>=90%':>16s} {'complete med':>12s} {'frags mean':>10s} {'foreign med':>11s} {'foreign>10%':>11s} | correct by length 40-100 / 100-200 / 200-400")
for v in sorted(R, key=lambda v: (v != "pkg", v)):
    r = R[v]; c = np.array([x["correct"] for x in r]); comp = np.array([x["complete"] for x in r]); fr = np.array([x["n_frag"] for x in r]); fo = np.array([x["foreign"] for x in r]); L = np.array([x["L"] for x in r])
    byL = [c[(L >= a) & (L < b)].mean() if ((L >= a) & (L < b)).any() else np.nan for a, b in ((40, 100), (100, 200), (200, 400))]
    print(f"{v:18s} {len(r):5d} {c.mean():8.2f} {np.mean(comp >= 0.9):16.2f} {np.median(comp):12.2f} {fr.mean():10.1f} {np.nanmedian(fo):11.2f} {np.mean(fo > 0.1):11.2f} | " + " / ".join(f"{x:.2f}" for x in byL))
