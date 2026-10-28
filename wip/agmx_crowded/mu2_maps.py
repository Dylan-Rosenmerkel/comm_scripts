"""full-CCD maps: image, A_GMX package (pkg), package + straight-track finder v2 (pkgmu2); red = largest cluster"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); from harness import load
img, ext = int(sys.argv[1]), int(sys.argv[2])
Q, valid = load(img, ext)
fig, axs = plt.subplots(3, 1, figsize=(24, 12))
vv = np.percentile(Q[valid], [5, 99.5])
axs[0].imshow(Q, cmap="gray", vmin=vv[0], vmax=vv[1], origin="lower", aspect="auto", interpolation="nearest")
axs[0].set_title(f"image {img} EXT{ext}", loc="left", fontsize=13, weight="bold")
for a, (v, nm) in zip(axs[1:], (("pkg", "A_GMX package (current scratch version)"), ("pkgmu2", "package + straight-track finder v2 (crossing charge shared)"))):
    z = np.load(f"out/{img}_{ext}_{v}.npz", allow_pickle=True); lab, size = z["lab"], z["npix"]
    rng = np.random.default_rng(5); cols = rng.uniform(0.25, 0.85, (lab.max() + 1, 3)); cols[0] = 0.07
    cols[1:][z["streak"]] = (1.0, 0.95, 0.0); cols[np.argmax(size) + 1] = (0.85, 0.12, 0.12)
    a.imshow(cols[lab], origin="lower", aspect="auto", interpolation="nearest")
    a.set_title(f"{nm}: {size.size:,} clusters; red = largest, {size.max():,} px", loc="left", fontsize=13, weight="bold")
for a in axs: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"mu2_maps_img{img}_ext{ext}.png", dpi=75)
