import sys, numpy as np
sys.path.insert(0, "."); import harness as HX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Q, valid = HX.load(9, 1); z = np.load("out/9_1_pkg.npz", allow_pickle=True); lab = z["lab"]
ids, cnt = np.unique(lab[lab > 0], return_counts=True); top = ids[np.argsort(cnt)[::-1][:3]]
fig, axs = plt.subplots(2, 3, figsize=(21, 13)); rng = np.random.default_rng(3)
for j, k in enumerate(top):
    ys, xs = np.nonzero(lab == k); y0, y1, x0, x1 = 0, 500, max(xs.min() - 60, 0), min(xs.max() + 60, Q.shape[1])
    w = Q[y0:y1, x0:x1]
    axs[0, j].imshow(w, cmap="gray", vmin=0, vmax=np.percentile(w, 99), origin="lower", extent=(x0, x1, y0, y1), aspect="auto")
    cols = rng.uniform(0.25, 0.7, (lab.max() + 1, 3)); cols[0] = 0.07; cols[k] = (1, 0.15, 0.15)
    axs[1, j].imshow(cols[lab[y0:y1, x0:x1]], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", aspect="auto")
    E = z["E"][k - 1] if "E" in z else np.nan
    axs[0, j].set_title(f"image, columns {x0}-{x1}", loc="left"); axs[1, j].set_title(f"cluster #{j+1} by size: {cnt[ids == k][0]:,} px, {E/1e3:.1f} MeV, class {z['cls'][k-1]}", loc="left")
    # column profile of the floor around it
    prof = np.percentile(Q[300:490, x0:x1], 20, axis=0)
    print(f"cluster {j+1}: x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}, {cnt[ids==k][0]} px; floor (20th pct, rows 300-490) min {prof.min():.1f} max {prof.max():.1f} at col {x0 + prof.argmax()}")
fig.tight_layout(); fig.savefig("blobview.png", dpi=70)
