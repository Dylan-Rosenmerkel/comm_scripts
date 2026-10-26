"""Tune the occupancy switch of adaptive.py on the crowded Am-241 CCD (runs_29img, EXT4, 29 images).
Per image: MHC clusters and hysteresis clusters (sigma = WADERs pedestal sigma from the log, and sigma = read
noise), each with the occupancy at its centre -> clusters_adapt/<stem>.npz.
  python adaptive_tune_am241.py          (cluster)
  python adaptive_tune_am241.py fit      (scan occ_thr: Lepy constrained fit, sigma_xy < 0.7)"""
import glob, os, re, sys
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am241_29img_recluster as RC
import adaptive as AD
AM, MH = AD.AM, AD.MH
OUT = os.path.join(HERE, "clusters_adapt")


def sigma_wader(stem, ext=4):
    log = open(os.path.join(RC.G, "logs", f"{stem}_EXT{ext}_panaSKImg.log")).read()
    m = re.search(r"Sigma for SPR: ([0-9.eE+-]+) ADUs \(([0-9.eE+-]+) ADU/e-", log)
    return float(m.group(1)) / float(m.group(2))


def job(stem):
    fn = os.path.join(OUT, f"{stem}.npz")
    if os.path.exists(fn): return
    Q, valid = RC.load(stem, 4)
    bg = AM.background_map(Q, valid)
    noise = float(np.median(bg.s[valid])); sw = sigma_wader(stem)
    occ = AD.occupancy(Q, valid, noise)
    k = max(int(AM.poisson_tail_k(float(np.median(bg.lam[valid])), 0.10)[0]), 2)
    res = {"MHC": MH.mhc(Q, valid, bg),
           "hyst_sw": AD.hysteresis(np.where(valid, Q - bg.o, 0), valid, sw, k),
           "hyst_noise": AD.hysteresis(np.where(valid, Q - bg.o, 0), valid, noise, k)}
    out = dict(sw=sw, noise=noise)
    for name, cl in res.items():
        cl = [c for c in cl if np.isfinite(c.get("E", np.nan))]
        o = np.array([occ[min(max(int(round(c["y"])), 0), Q.shape[0] - 1), min(max(int(round(c["x"])), 0), Q.shape[1] - 1)] for c in cl])
        out[f"{name}_E"] = np.array([c["E"] for c in cl]); out[f"{name}_sxy"] = np.array([c["sxy"] for c in cl]); out[f"{name}_occ"] = o
    np.savez(fn, **out)
    print(stem[-12:], f"sigma_w {sw:.2f} noise {noise:.2f}", {k: len(v) for k, v in res.items()}, flush=True)


def fit():
    import am241_paths; am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
    from amfit import Model, summary
    fs = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    D = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files if k not in ("sw", "noise")}
    MAIN = ("La", "Lb", "Lg", "g26")
    def show(lab, E, S):
        m = S < 0.7
        r = Model("Ag").fit(E[m], 0.985); s = summary(r)
        z = np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))
        print(f"  {lab:34s} {m.sum():6d} clusters  k {s['k'][0]:.4f}±{s['k'][1]:.4f}  dev/ndf {r['devndf']:.2f}  Z_np {z:5.1f}  "
              + "  ".join(f"{g} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0))
    print(f"{len(fs)} images; WADERs reference (panaSKImg, fitted sigma < 0.7): Lb 15.5σ Lg 16.0σ g26 14.8σ AgK 12.9σ, k 0.984")
    for h in ("hyst_sw", "hyst_noise"):
        show(f"hysteresis only ({h})", D[f"{h}_E"], D[f"{h}_sxy"])
        show("MHC only", D["MHC_E"], D["MHC_sxy"])
        for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
            a, b = D["MHC_occ"] < t, D[f"{h}_occ"] >= t
            show(f"adaptive occ_thr {t} ({h})", np.r_[D["MHC_E"][a], D[f"{h}_E"][b]], np.r_[D["MHC_sxy"][a], D[f"{h}_sxy"][b]])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))})
    with Pool(int(os.environ.get("NPROC", 4))) as p:
        p.map(job, stems, chunksize=1)
