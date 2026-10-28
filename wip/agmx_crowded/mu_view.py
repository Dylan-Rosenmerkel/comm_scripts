import sys, numpy as np
sys.path.insert(0, "."); import harness as HX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Q, valid = HX.load(9, 1); lab = np.load("mu_lab_9_1.npy")
wins = [(339, 900), (1500, 2200), (2400, 3100)]
fig, axs = plt.subplots(len(wins), 2, figsize=(22, 6 * len(wins))); rng = np.random.default_rng(9)
cols = rng.uniform(0.3, 1, (lab.max() + 1, 3)); cols[0] = 0
for r, (x0, x1) in enumerate(wins):
    w = Q[:, x0:x1]; axs[r, 0].imshow(w, cmap="gray", vmin=0, vmax=np.percentile(w, 99), origin="lower", extent=(x0, x1, 0, 500), aspect="auto")
    g = w / np.percentile(w, 99); g = np.clip(g, 0, 1)[..., None] * 0.6
    ov = np.where(lab[:, x0:x1, None] > 0, cols[lab[:, x0:x1]], g)
    axs[r, 1].imshow(ov, origin="lower", extent=(x0, x1, 0, 500), aspect="auto", interpolation="nearest")
    axs[r, 1].set_title(f"found straight tracks (colour) over the image (grey): {len(np.unique(lab[:, x0:x1][lab[:, x0:x1] > 0]))} here", loc="left")
fig.tight_layout(); fig.savefig("mu_view.png", dpi=65)
