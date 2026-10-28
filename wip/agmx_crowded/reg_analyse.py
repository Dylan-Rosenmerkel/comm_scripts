"""current vs new A_GMX on the regression set: per CCD Fe-55 K-alpha (front: sigma_xy < 1, blend < 0.02, ped < 10) core fit;
Am-241 (29 images pooled): 59.5 keV and Fe K-alpha; cluster counts; whether the v3 background triggered."""
import glob, json, re, numpy as np
from scipy.optimize import curve_fit
def core(E, lo, hi, guess, w0):
    E = E[(E > lo) & (E < hi)]
    if E.size < 60: return np.nan, np.nan, np.nan, E.size
    mu, sg = guess, w0
    for _ in range(6):
        s = E[(E > mu - 1.5 * sg) & (E < mu + 1.5 * sg)]
        if s.size < 30: return np.nan, np.nan, np.nan, E.size
        h, b = np.histogram(s, 30); c = 0.5 * (b[1:] + b[:-1])
        try: p, C = curve_fit(lambda x, a, m, w: a * np.exp(-0.5 * ((x - m) / w) ** 2), c, h, p0=[h.max(), mu, sg], sigma=np.sqrt(np.maximum(h, 1)))
        except Exception: return np.nan, np.nan, np.nan, E.size
        mu, sg = p[1], abs(p[2])
    return mu, np.sqrt(C[1, 1]), sg, s.size
def sel(z, front=True):
    m = (z["blend"] < 0.02) & (z["ped"] < 10)
    return m & ((z["sxy"] < 1) if front else (z["sxy"] > 1))
rows = []
tags = sorted({re.sub(r"_(base|new)\.npz$", "", f[4:]) for f in glob.glob("reg/*_new.npz")})
for t in tags:
    try: b = np.load(f"reg/{t}_base.npz", allow_pickle=True); n = np.load(f"reg/{t}_new.npz", allow_pickle=True)
    except FileNotFoundError: continue
    rep = json.loads(str(n["rep"]))
    r = dict(tag=t, applied=rep["applied"], bad=rep["bad_rows"], cells=rep["cells_changed"], ncells=rep["ncells"], s=rep["s"],
             nb=len(b["E"]), nn=len(n["E"]), streaks=int(n["n_streaks"]), linked=int(n["n_linked"]))
    for v, z in (("b", b), ("n", n)):
        r[f"fe_{v}"] = core(z["E"][sel(z)], 5.0, 6.6, 5.88, 0.08)
    rows.append(r)
json.dump(rows, open("reg_rows.json", "w"), default=float)
def grp(r):
    return "Am241" if r["tag"].startswith("Am241") else ("L2" if r["tag"].startswith("L2") else ("HighTemp" if "High_Temp" in r["tag"] else "Fe55 production"))
print(f"{len(rows)} CCD images compared")
for g in ("Fe55 production", "HighTemp", "L2", "Am241"):
    R = [r for r in rows if grp(r) == g]
    if not R: continue
    app = [r for r in R if r["applied"] and (r["cells"] > 0 or r["bad"] > 10)]
    d = np.array([1e3 * (r["fe_n"][0] - r["fe_b"][0]) for r in R]); w = np.array([1e3 * (r["fe_n"][2] - r["fe_b"][2]) for r in R])
    nb = np.array([r["fe_b"][3] for r in R]); nn = np.array([r["fe_n"][3] for r in R])
    ok = np.isfinite(d)
    print(f"\n== {g}: {len(R)} CCDs, v3 substantially active on {len(app)} (cells changed or >10 bad rows)")
    print(f"   Fe Ka shift new-current [eV]: median {np.nanmedian(d):+.1f}, 5-95% {np.nanpercentile(d,5):+.1f} .. {np.nanpercentile(d,95):+.1f}  (n={ok.sum()})")
    print(f"   Fe Ka width change [eV]: median {np.nanmedian(w):+.1f}, 5-95% {np.nanpercentile(w,5):+.1f} .. {np.nanpercentile(w,95):+.1f}")
    print(f"   Fe events in core: {np.nansum(nb):.0f} -> {np.nansum(nn):.0f};  clusters {sum(r['nb'] for r in R)} -> {sum(r['nn'] for r in R)}; streaks {sum(r['streaks'] for r in R)}, linked {sum(r['linked'] for r in R)}")
    big = sorted([r for r in R if np.isfinite(r["fe_n"][0]) and np.isfinite(r["fe_b"][0])], key=lambda r: -abs(r["fe_n"][0] - r["fe_b"][0]))[:4]
    for r in big:
        print(f"     largest shift: {r['tag'][:60]:60s} Ka {r['fe_b'][0]:.3f} -> {r['fe_n'][0]:.3f}, sigma {1e3*r['fe_b'][2]:.0f} -> {1e3*r['fe_n'][2]:.0f} eV, n {r['fe_b'][3]} -> {r['fe_n'][3]}, bad rows {r['bad']}, cells {r['cells']}")
# Am-241 pooled
for v in ("base", "new"):
    E = np.concatenate([np.load(f, allow_pickle=True)["E"][sel(np.load(f, allow_pickle=True))] for f in sorted(glob.glob(f"reg/Am241*_{v}.npz"))]) if glob.glob(f"reg/Am241*_{v}.npz") else np.array([])
    if E.size:
        g = core(E, 57, 62, 59.6, 0.4); f = core(E, 5.0, 6.6, 5.9, 0.1)
        print(f"Am241 pooled {v}: 59.54 keV line {g[0]:.3f} ({100*(g[0]/59.541-1):+.2f} %) sigma {1e3*g[2]:.0f} eV n {g[3]};  Fe Ka {f[0]:.3f} sigma {1e3*f[2]:.0f} eV n {f[3]}")
