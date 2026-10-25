"""Crowded regime variants on the Am-241 runs_29img EXT4 images (Lepy constrained fit, sigma_xy < 0.7):
  fixed T      : pixel > T, seed > 3 T (T = 80 e- ~ WADERs), energy = sum of the core pixels (as WADERs)
  local T      : pixel > m + d, seed > m + 3 d, m = local median (31 x 31 px, the level of the charge sheet)
  energy 'ring r' : sum over the core dilated by r rings of (Q - m)   (tails recovered, sheet subtracted)
  python adaptive_crowded_scan_am241.py ; python adaptive_crowded_scan_am241.py fit"""
import glob, os, sys
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am241_29img_recluster as RC
import adaptive as AD
OUT = os.path.join(HERE, "clusters_crowd")
EIGHT = np.ones((3, 3), bool)
VARIANTS = [("fixed", 80), ("fixed", 60), ("local", 20), ("local", 40), ("local", 60), ("local", 80)]


def clusters(Q, valid, m, kind, d):
    tp = (d if kind == "fixed" else m + d); ts = (3 * d if kind == "fixed" else m + 3 * d)
    lab, n = ndimage.label((Q > tp) & valid, structure=EIGHT)
    seeds = ndimage.maximum(Q - ts, lab, index=np.arange(1, n + 1)) > 0
    out = {f"ring{r}": [] for r in (0, 1, 2)}; sx = []
    owner = np.where(lab > 0, lab, 0)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if not seeds[k - 1]: continue
        pad = 3
        s2 = (slice(max(sl[0].start - pad, 0), sl[0].stop + pad), slice(max(sl[1].start - pad, 0), sl[1].stop + pad))
        core = lab[s2] == k
        ys, xs = np.nonzero(core)
        my, mx, sxy = AD.CM.moments(Q, ys + s2[0].start, xs + s2[1].start)
        sx.append(sxy)
        for r in (0, 1, 2):
            fp = ndimage.binary_dilation(core, structure=EIGHT, iterations=r) if r else core
            fp &= valid[s2] & ((owner[s2] == 0) | core)            # do not take other cores' pixels
            sub = 0.0 if r == 0 and kind == "fixed" else m[s2]
            out[f"ring{r}"].append(float(((Q[s2] - sub) * fp).sum() * AD.AM.E_PER_E))
    return out, np.array(sx)


def job(stem):
    fn = os.path.join(OUT, f"{stem}.npz")
    if os.path.exists(fn): return
    Q, valid = RC.load(stem, 4)
    bg = AD.AM.background_map(Q, valid)
    R = np.where(valid, Q - bg.o, 0.0)
    m = ndimage.median_filter(np.where(valid, R, 0.0), size=31, mode="nearest")
    res = {}
    for kind, d in VARIANTS:
        e, s = clusters(R, valid, m, kind, d)
        for r, v in e.items(): res[f"{kind}{d}_{r}_E"] = np.array(v)
        res[f"{kind}{d}_sxy"] = s
    np.savez(fn, **res); print(stem[-12:], "done", flush=True)


def fit():
    sys.path.insert(0, os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis"))
    from amfit import Model, summary
    fs = sorted(glob.glob(os.path.join(OUT, "*.npz"))); MAIN = ("La", "Lb", "Lg", "g26")
    for kind, d in VARIANTS:
        S = np.concatenate([np.load(f)[f"{kind}{d}_sxy"] for f in fs])
        for r in (0, 1, 2):
            E = np.concatenate([np.load(f)[f"{kind}{d}_ring{r}_E"] for f in fs]); mm = S < 0.7
            res = Model("Ag").fit(E[mm], 0.985); s = summary(res)
            z = np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))
            print(f"  {kind:5s} d={d:3d} energy ring{r}: {mm.sum():6d} clusters  k {s['k'][0]:.4f}  dev/ndf {res['devndf']:.2f}  Z_np {z:5.1f}  "
                  + "  ".join(f"{g} {s[g][0] / s[g][1]:.1f}σ" for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))})
    with Pool(int(os.environ.get("NPROC", 3))) as p:
        p.map(job, stems, chunksize=1)
