import sys, numpy as np
sys.path.insert(0, "."); import harness as HX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, figsize=(20, 9))
for r, img in enumerate((9, 10)):
    Q, valid = HX.load(img, 1)
    lab = np.load(f"out/{img}_1_pkg.npz")["lab"]
    x0, x1 = 420, 510
    w = Q[:, x0:x1]
    axs[r, 0].imshow(w, cmap="gray", vmin=0, vmax=np.percentile(w, 99), origin="lower", extent=(x0, x1, 0, 500), aspect="auto"); axs[r, 0].set_title(f"image {img} EXT1, columns {x0}-{x1}")
    rng = np.random.default_rng(1); cols = rng.uniform(0.25, 1, (lab.max() + 1, 3)); cols[0] = 0.07
    axs[r, 1].imshow(cols[lab[:, x0:x1]], origin="lower", extent=(x0, x1, 0, 500), aspect="auto", interpolation="nearest"); axs[r, 1].set_title("new A_GMX clusters")
    for rows, c in ((slice(50, 150), "C0"), (slice(200, 300), "C1"), (slice(400, 490), "C3")):
        prof = np.percentile(Q[rows, x0:x1], 20, axis=0)
        axs[r, 2].plot(np.arange(x0, x1), prof, color=c, label=f"rows {rows.start}-{rows.stop}: 20th percentile per column")
    axs[r, 2].set_xlabel("column"); axs[r, 2].set_ylabel("e- per pixel"); axs[r, 2].legend(fontsize=8); axs[r, 2].set_yscale("log")
    p = np.percentile(Q[400:490, x0:x1], 20, axis=0); k = int(np.argmax(p))
    hw = np.sum(p > 0.5 * p.max())
    print(f"image {img}: peak column {x0 + k}, 20th pct level {p.max():.0f} e-/pix (rows 400-490), FWHM {hw} columns; level vs row:",
          [int(np.percentile(Q[a:a + 50, x0 + k], 20)) for a in range(0, 500, 50)])
fig.tight_layout(); fig.savefig("blob_ext1.png", dpi=80)
