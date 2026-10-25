"""Crowded-regime threshold scan (Am-241 runs_29img EXT4): hysteresis clustering with pixel threshold T and seed
3 T for T in 10-160 e-, on the whole active area; Lepy constrained fit (sigma_xy < 0.7) for each T.
  python adaptive_thr_scan_am241.py         (cluster, -> clusters_thr/<stem>.npz)
  python adaptive_thr_scan_am241.py fit"""
import glob, os, sys
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am241_29img_recluster as RC
import adaptive as AD
OUT = os.path.join(HERE, "clusters_thr")
TS = (10, 20, 40, 60, 80, 120, 160)


def job(stem):
    fn = os.path.join(OUT, f"{stem}.npz")
    if os.path.exists(fn): return
    Q, valid = RC.load(stem, 4)
    bg = AD.AM.background_map(Q, valid)
    R = np.where(valid, Q - bg.o, 0)
    out = {}
    for T in TS:
        cl = AD.hysteresis(R, valid, T / 10.0, 2, n_pix=10, n_seed=30, seed_e=0)
        out[f"T{T}_E"] = np.array([c["E"] for c in cl]); out[f"T{T}_sxy"] = np.array([c["sxy"] for c in cl])
    np.savez(fn, **out); print(stem[-12:], {T: len(out[f"T{T}_E"]) for T in TS}, flush=True)


def fit():
    sys.path.insert(0, os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis"))
    from amfit import Model, summary
    fs = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    MAIN = ("La", "Lb", "Lg", "g26")
    for T in TS:
        E = np.concatenate([np.load(f)[f"T{T}_E"] for f in fs]); S = np.concatenate([np.load(f)[f"T{T}_sxy"] for f in fs])
        for cut in (0.7, 1.0):
            m = S < cut
            r = Model("Ag").fit(E[m], 0.985); s = summary(r)
            z = np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))
            print(f"  pixel > {T:3d} e-, seed > {3 * T:3d} e-, sigma_xy < {cut}: {m.sum():6d} clusters  k {s['k'][0]:.4f}  dev/ndf {r['devndf']:.2f}  Z_np {z:5.1f}  "
                  + "  ".join(f"{g} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))})
    with Pool(int(os.environ.get("NPROC", 3))) as p:
        p.map(job, stems, chunksize=1)
