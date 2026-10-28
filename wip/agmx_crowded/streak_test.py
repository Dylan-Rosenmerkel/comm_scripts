import sys, numpy as np
sys.path.insert(0, "."); import harness as HX, bgv3, streaks, xsplit
AM, AD, GX = HX.AM, HX.AD, HX.GX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Q, valid = HX.load(9, 1)
bg, rep = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid)); print("v3:", rep)
st, sc = streaks.find_streaks(Q, valid, bg)
L = np.array([c["_pix"][0].size for c in sc]); E = np.array([c["E"] for c in sc])
print(f"streaks: {len(sc)}, length median {np.median(L):.0f} max {L.max()}, energy median {np.median(E):.2f} keV; rows with streaks: {len(np.unique([int(c['y']) for c in sc]))}")
def lab_of(cl):
    lab = np.zeros(Q.shape, int)
    for k, c in enumerate(sorted(cl, key=lambda c: -c["_pix"][0].size), 1):
        ys, xs, q = c["_pix"]; m = q > 0.5; lab[ys[m], xs[m]] = k
    return lab
xsplit.install(GX)
cl0 = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True)
streaks.install(AD); cl1 = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True); streaks.uninstall(AD)
cl0 = [c for c in cl0 if "_pix" in c]; cl1 = [c for c in cl1 if "_pix" in c]
print("clusters without / with streak removal:", len(cl0), len(cl1))
np.save("streak_mask_9_1.npy", st)
l0, l1 = lab_of(cl0), lab_of(cl1)
# crops centred on the longest streaks
order = np.argsort(L)[::-1][:4]
fig, axs = plt.subplots(4, 3, figsize=(16, 15)); rng = np.random.default_rng(4)
for r, i in enumerate(order):
    y, x = int(sc[i]["y"]), int(sc[i]["x"]); y0, y1, x0, x1 = max(y - 40, 0), min(y + 40, Q.shape[0]), max(x - 70, 0), min(x + 70, Q.shape[1])
    w = (Q - bg.mean)[y0:y1, x0:x1]; axs[r, 0].imshow(w, cmap="gray", vmin=0, vmax=np.percentile(w, 99.5), origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
    axs[r, 0].contour(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5, st[y0:y1, x0:x1], [0.5], colors="r", linewidths=0.7)
    axs[r, 0].set_title(f"image, red = detected streak pixels ({L[i]} px, {E[i]:.1f} keV)", fontsize=9)
    for j, (lab, nm) in enumerate(((l0, "new A_GMX (v3 bg + crossing split)"), (l1, "+ serial streaks separated")), 1):
        cols = rng.uniform(0.3, 1, (lab.max() + 1, 3)); cols[0] = 0.07
        axs[r, j].imshow(cols[lab[y0:y1, x0:x1]], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest"); axs[r, j].set_title(nm, fontsize=9)
fig.tight_layout(); fig.savefig("streaks.png", dpi=80)
