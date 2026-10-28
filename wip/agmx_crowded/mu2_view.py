import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import harness as HX, bgv3, muons2
AM = HX.AM
img, ext = int(sys.argv[1]), int(sys.argv[2])
Q, valid = HX.load(img, ext)
bg, _ = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid))
tk, mu, mq = muons2.find_muons(Q, valid, bg)
R = np.where(valid, Q - bg.mean, 0)
print(len(mu), "tracks; length quantiles", np.percentile([m["length"] for m in mu], [10, 50, 90]).round())
lab = np.zeros(Q.shape); rng = np.random.default_rng(0)
for m in mu: lab[m["_pix"][0], m["_pix"][1]] = rng.uniform(0.05, 1)
lab = np.ma.masked_equal(lab, 0)
fig, ax = plt.subplots(2, 2, figsize=(20, 17))
for j, (x0, x1) in enumerate([(1600, 2200), (2700, 3300)]):
    for k in range(2):
        a = ax[k, j]
        a.imshow(np.log10(np.clip(R[:, x0:x1], 1, None)), origin="lower", cmap="gray_r", vmin=0, vmax=3.5, extent=(x0, x1, 0, R.shape[0]), interpolation="nearest")
        if k: a.imshow(lab[:, x0:x1], origin="lower", cmap="hsv", alpha=.55, extent=(x0, x1, 0, R.shape[0]), interpolation="nearest", vmin=0, vmax=1)
        a.set_title(("muon-finder v2 tubes (colour)" if k else "image") + f"  cols {x0}-{x1}")
plt.tight_layout(); plt.savefig(f"mu2_view_{img}_{ext}.png", dpi=60)
