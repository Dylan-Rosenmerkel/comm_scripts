import sys, numpy as np
sys.path.insert(0, "."); import harness as HX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Q, valid = HX.load(9, 1)
A = np.load("out/9_1_pkg_linkv1.npz")["lab"]; B = np.load("out/9_1_pkg.npz")["lab"]
wins = [(1620, 1860, 210, 390), (2400, 3100, 0, 500), (0, 700, 0, 500)]
fig, axs = plt.subplots(3, 3, figsize=(24, 17), gridspec_kw=dict(width_ratios=[1, 1, 1])); rng = np.random.default_rng(7)
for r, (x0, x1, y0, y1) in enumerate(wins):
    w = Q[y0:y1, x0:x1]; axs[r, 0].imshow(w, cmap="gray", vmin=np.percentile(w, 5), vmax=np.percentile(w, 99.5), origin="lower", extent=(x0, x1, y0, y1), aspect="auto")
    for j, (lab, nm) in enumerate(((A, "strict linking"), (B, "loosened linking")), 1):
        L = lab[y0:y1, x0:x1]; cols = rng.uniform(0.25, 1, (lab.max() + 1, 3)); cols[0] = 0.07
        axs[r, j].imshow(cols[L], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", aspect="auto")
        axs[r, j].set_title(f"{nm}: {len(np.unique(L[L > 0]))} clusters in window", fontsize=11)
fig.tight_layout(); fig.savefig("linkview.png", dpi=70)
