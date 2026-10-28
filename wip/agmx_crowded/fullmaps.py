"""Full-CCD cluster maps: WADERs default, WADERs auto threshold, A_GMX current, A_GMX new (regional bg + linking)."""
import os, sys, numpy as np, ROOT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT.gROOT.SetBatch(True)
sys.path.insert(0, "."); from harness import load, STEM
D = os.path.expanduser("~/Documents/comm_scripts_data/L2_101_20260922")
img, ext = int(sys.argv[1]), int(sys.argv[2])
def root_lab(f, shape):
    lab = np.zeros(shape, np.int32); size = []; tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec"); k = 0
    for ev in t:
        for i in range(len(ev.Energy)):
            k += 1; px = np.array(list(ev.pixels_x[i]), int); py = np.array(list(ev.pixels_y[i]), int)
            ok = (py >= 0) & (py < shape[0]) & (px >= 0) & (px < shape[1]); lab[py[ok], px[ok]] = k; size.append(len(px))
    tf.Close(); return lab, np.array(size)
Q, valid = load(img, ext); sh = Q.shape; st = STEM[img]
cases = [("WADERs (default thresholds, 10σ / 30σ)", *root_lab(f"{D}/WADERs_analysis_L2_101_20260922/root/panaSKImg_clustersRec_{st}_ext{ext}.root", sh)),
         ("WADERs, auto thresholds in electrons", *root_lab(f"{D}/thr_auto/WADERs_analysis_L2_101_20260922/root/panaSKImg_clustersRec_{st}_ext{ext}.root", sh))]
for v, nm in (("base", "A_GMX current"), ("linkm", "A_GMX new (regional background + track linking)"),
              ("pkg", "A_GMX new: regional background + crossing split + serial streaks (yellow) + track linking")):
    z = np.load(f"out/{img}_{ext}_{v}.npz", allow_pickle=True); cases.append((nm, z["lab"], z["npix"]) + ((z["streak"],) if "streak" in z else ()))
fig, axs = plt.subplots(6, 1, figsize=(24, 22))
vv = np.percentile(Q[valid], [5, 99.5])
axs[0].imshow(Q, cmap="gray", vmin=vv[0], vmax=vv[1], origin="lower", aspect="auto", interpolation="nearest")
axs[0].set_title(f"image {img} EXT{ext} (electrons, g(n)-corrected)", loc="left", fontsize=13, weight="bold")
rng = np.random.default_rng(5)
for a, (nm, lab, size, *st) in zip(axs[1:], cases):
    cols = rng.uniform(0.25, 0.85, (lab.max() + 1, 3)); cols[0] = 0.07
    if st: cols[1:][st[0]] = (1.0, 0.95, 0.0)
    big = np.argsort(size)[::-1][:1] + 1; cols[big] = (0.85, 0.12, 0.12)
    a.imshow(cols[lab], origin="lower", aspect="auto", interpolation="nearest")
    a.set_title(f"{nm}: {len(size):,} clusters; red = largest, {size.max():,} px ({size.max() / Q.size:.0%} of the CCD)", loc="left", fontsize=13, weight="bold")
for a in axs: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"fullmaps_img{img}_ext{ext}.png", dpi=75); print("ok")
