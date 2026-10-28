import numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); from harness import load
V = ["base", "bgreg", "bgmask", "linkm"]; COL = {"base": "#9a958f", "bgreg": "#d07a2a", "bgmask": "#2a8a4a", "link": "#2a64c8", "linkm": "#2a64c8"}
def fe(z):
    s = (z["E"] > 5.2) & (z["E"] < 6.6) & (z["sxy"] < 1) & (z["blend"] < 0.02) & (z["ped"] < 10)
    return z["x"][s], z["E"][s]
print(f"{'CCD':10s} {'variant':6s} {'clusters':>8s} {'largest px':>10s} {'>5000 px':>8s} {'Fe n':>5s} {'Fe med':>7s} {'Fe 16-84% width':>15s} {'col slope %/1000':>16s}")
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
for a, (img, ext) in zip(axs.ravel(), [(9, 1), (10, 1), (9, 4), (10, 4)]):
    for v in V:
        z = np.load(f"out/{img}_{ext}_{v}.npz", allow_pickle=True)
        x, E = fe(z); p = np.polyfit(x, E, 1) if x.size > 50 else [np.nan, np.nan]
        lo, md, hi = np.percentile(E, [16, 50, 84])
        print(f"img{img} EXT{ext} {v:6s} {len(z['E']):8d} {int(z['npix'].max()):10,d} {int((z['npix'] > 5000).sum()):8d} {x.size:5d} {md:7.3f} {1e3*(hi-lo)/2:12.0f} eV {100*p[0]*1000/5.9:16.2f}")
        b = np.arange(0, 2600, 150); idx = np.digitize(x, b)
        med = [np.median(E[idx == i]) if (idx == i).sum() > 10 else np.nan for i in range(1, len(b))]
        a.plot(0.5 * (b[1:] + b[:-1]), med, "o-", color=COL[v], label=v, ms=4)
    a.axhline(5.899, color="r", lw=0.8); a.set_title(f"image {img} EXT{ext}: Fe Kα median vs column", loc="left", fontsize=10, weight="bold"); a.set_ylim(5.4, 6.1)
axs[0, 0].legend(); fig.tight_layout(); fig.savefig("fe_vs_col_variants.png", dpi=100)
# maps
Q, valid = load(9, 1)
rng = np.random.default_rng(3)
wins = [("glow region", 456, 1210, 13, 500), ("top-left", 0, 450, 280, 500), ("long tracks", 2400, 3100, 0, 500)]
fig, axs = plt.subplots(len(wins), 5, figsize=(27, 4.2 * len(wins)))
for r, (nm, x0, x1, y0, y1) in enumerate(wins):
    w = Q[y0:y1, x0:x1]; vv = np.percentile(w, [5, 99.5])
    axs[r, 0].imshow(w, cmap="gray", vmin=vv[0], vmax=vv[1], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest"); axs[r, 0].set_title(f"image 9 EXT1, {nm}", loc="left", weight="bold")
    for j, v in enumerate(V):
        z = np.load(f"out/9_1_{v}.npz", allow_pickle=True); lab = z["lab"]; cols = rng.uniform(0.25, 1, (lab.max() + 1, 3)); cols[0] = 0.08
        big = np.argsort(z["npix"])[::-1][:1] + 1; cols[big] = (0.85, 0.15, 0.15)
        L = lab[y0:y1, x0:x1]; axs[r, j + 1].imshow(cols[L], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
        axs[r, j + 1].set_title(f"{v}: {len(np.unique(L[L > 0]))} clusters here (red: CCD's largest, {int(z['npix'].max()):,} px)", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig("maps_variants.png", dpi=80)
# background maps
zb = np.load("out/9_1_base.npz"); zr = np.load("out/9_1_bgreg.npz")
fig, axs = plt.subplots(1, 3, figsize=(18, 3.6))
for a, (L, t) in zip(axs, [(zb["lam"], "current background map λ [e⁻/pix]"), (zr["lam"], "regional background map λ [e⁻/pix]")]):
    im = a.imshow(L, origin="lower", aspect="auto", vmin=0, vmax=np.percentile(zr["lam"], 99.5)); plt.colorbar(im, ax=a); a.set_title(t, loc="left", fontsize=10)
rows = slice(300, 500)
axs[2].plot(zb["lam"][rows].mean(0), color=COL["base"], label="current"); axs[2].plot(zr["lam"][rows].mean(0), color=COL["bgreg"], label="regional")
nq = np.where(valid, Q, np.nan)[rows]; prof = np.array([np.nanpercentile(nq[:, c:c + 50], 10) for c in range(0, 3500, 50)])
axs[2].plot(np.arange(0, 3500, 50) + 25, prof, "k:", label="image 10th percentile"); axs[2].set_title("rows 300-500, mean over rows", loc="left", fontsize=10); axs[2].legend(fontsize=8); axs[2].set_xlabel("column")
fig.tight_layout(); fig.savefig("bg_maps.png", dpi=90); print("ok")
