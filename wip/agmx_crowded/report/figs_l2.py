"""report figures on L2: full maps, zooms of all methods, crossing charge sharing, Fe-55"""
import os, sys, numpy as np, ROOT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT.gROOT.SetBatch(True)
sys.path.insert(0, ".")
from harness import load, STEM
D = os.path.expanduser("~/Documents/comm_scripts_data/L2_101_20260922")
O = "report"
def root_lab(f, shape):
    lab = np.zeros(shape, np.int32); size = []; tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec"); k = 0
    for ev in t:
        for i in range(len(ev.Energy)):
            k += 1; px = np.array(list(ev.pixels_x[i]), int); py = np.array(list(ev.pixels_y[i]), int)
            ok = (py >= 0) & (py < shape[0]) & (px >= 0) & (px < shape[1]); lab[py[ok], px[ok]] = k; size.append(len(px))
    tf.Close(); return lab, np.array(size), None
def npz_lab(img, ext, v):
    z = np.load(f"out/{img}_{ext}_{v}.npz", allow_pickle=True); return z["lab"], z["npix"], z["streak"] if "streak" in z else None
METHODS = [("WADERs default", lambda i, e, sh: root_lab(f"{D}/WADERs_analysis_L2_101_20260922/root/panaSKImg_clustersRec_{STEM[i]}_ext{e}.root", sh)),
           ("WADERs auto thr.", lambda i, e, sh: root_lab(f"{D}/thr_auto/WADERs_analysis_L2_101_20260922/root/panaSKImg_clustersRec_{STEM[i]}_ext{e}.root", sh)),
           ("A_GMX current", lambda i, e, sh: npz_lab(i, e, "base")),
           ("package (bg + xsplit + streaks + link)", lambda i, e, sh: npz_lab(i, e, "pkg")),
           ("package + muon finder, cap rule (mu2)", lambda i, e, sh: npz_lab(i, e, "pkgmu2")),
           ("package + muon finder, EM sharing (mu3)", lambda i, e, sh: npz_lab(i, e, "pkgmu3"))]
def colour(lab, size, st, seed=5):
    rng = np.random.default_rng(seed); cols = rng.uniform(0.25, 0.85, (lab.max() + 1, 3)); cols[0] = 0.07
    if st is not None and len(st): cols[1:][st] = (1.0, 0.95, 0.0)
    cols[np.argmax(size) + 1] = (0.9, 0.1, 0.1)
    return cols[lab]
cache = {}
def get(i, e, m, sh):
    if (i, e, m) not in cache: cache[(i, e, m)] = dict(METHODS)[m](i, e, sh)
    return cache[(i, e, m)]

# 1. full maps
for img, ext in ((9, 1), (9, 4), (10, 1), (10, 4)):
    Q, valid = load(img, ext); sh = Q.shape
    fig, axs = plt.subplots(7, 1, figsize=(24, 24))
    vv = np.percentile(Q[valid], [5, 99.5])
    axs[0].imshow(Q, cmap="gray", vmin=vv[0], vmax=vv[1], origin="lower", aspect="auto", interpolation="nearest")
    axs[0].set_title(f"image {img} EXT{ext} (L2 ACM101)", loc="left", fontsize=14, weight="bold")
    for a, (m, _) in zip(axs[1:], METHODS):
        lab, size, st = get(img, ext, m, sh)
        a.imshow(colour(lab, size, st), origin="lower", aspect="auto", interpolation="nearest")
        a.set_title(f"{m}: {len(size):,} clusters; red = largest ({size.max():,} px)", loc="left", fontsize=14, weight="bold")
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{O}/full_img{img}_ext{ext}.png", dpi=60); plt.close(fig)
    print("full", img, ext, flush=True)

# 2. zooms: rows = regions, cols = image + methods
REG = [(9, 1, 150, 500, 360, 700, "EXT1 'blob' (vertical muon crossed by ~10 tracks)"),
       (9, 1, 0, 500, 700, 1150, "long diagonal muon"),
       (9, 4, 0, 500, 1600, 2050, "crowded region with long straight tracks"),
       (10, 4, 0, 500, 2600, 3050, "crowded region"),
       (10, 1, 0, 500, 1000, 1450, "crowded region")]
# largest cluster of pkg on 10_4 as an extra region
lab, size, _ = npz_lab(10, 4, "pkg"); yy, xx = np.nonzero(lab == np.argmax(size) + 1)
REG.append((10, 4, 0, 500, int(max(xx.mean() - 225, 0)), int(max(xx.mean() - 225, 0)) + 450, "largest cluster of the package (10 EXT4)"))
for r0 in range(0, len(REG), 3):
    regs = REG[r0:r0 + 3]
    fig, axs = plt.subplots(len(regs), 7, figsize=(35, 6.2 * len(regs)))
    axs = np.atleast_2d(axs)
    for row, (img, ext, y0, y1, x0, x1, title) in zip(axs, regs):
        Q, valid = load(img, ext); sh = Q.shape
        vv = np.percentile(Q[valid], [5, 99.7])
        row[0].imshow(Q[y0:y1, x0:x1], cmap="gray", vmin=vv[0], vmax=vv[1], origin="lower", interpolation="nearest", extent=(x0, x1, y0, y1))
        row[0].set_title(f"img {img} EXT{ext}: {title}", loc="left", fontsize=12, weight="bold")
        for a, (m, _) in zip(row[1:], METHODS):
            lab, size, st = get(img, ext, m, sh)
            sub = lab[y0:y1, x0:x1]; ids = np.unique(sub[sub > 0])
            a.imshow(colour(lab, size, st)[y0:y1, x0:x1], origin="lower", interpolation="nearest", extent=(x0, x1, y0, y1))
            a.set_title(f"{m}\n{ids.size} clusters here; largest {size[ids - 1].max() if ids.size else 0:,} px", fontsize=11)
        for a in row: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{O}/zoom_{r0 // 3 + 1}.png", dpi=55); plt.close(fig)
    print("zoom", r0, flush=True)
