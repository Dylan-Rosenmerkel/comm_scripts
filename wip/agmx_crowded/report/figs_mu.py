"""report: crossing charge sharing close-up; injected-muon examples per method"""
import sys, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, ".")
import harness as HX, bgv3, muons2, muons3, xsplit, streaks, inject_mu as IM
AM, AD, GX = HX.AM, HX.AD, HX.GX
O = "report"
# --- crossing close-up
for img, ext, y0, y1, x0, x1, nm in ((9, 1, 240, 470, 400, 540, "a"), (9, 4, 150, 330, 1880, 2060, "b")):
    Q, valid = HX.load(img, ext); bg, _ = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid))
    R = np.where(valid, Q - bg.mean, 0)
    _, _, m2 = muons2.find_muons(Q, valid, bg); _, _, m3 = muons3.find_muons(Q, valid, bg)
    S = lambda a: np.clip(a[y0:y1, x0:x1], 3, None)
    pan = [("image (charge above background)", S(R)), ("muon share, cap rule (mu2)", S(m2)), ("left for A_GMX, cap rule", S(R - m2)),
           ("muon share, EM (mu3)", S(m3)), ("left for A_GMX, EM", S(R - m3))]
    fig, axs = plt.subplots(1, 5, figsize=(25, 7.5))
    for a, (t, im) in zip(axs, pan):
        a.imshow(im, norm=LogNorm(3, 3000), cmap="magma", origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest"); a.set_title(t, fontsize=13)
    fig.suptitle(f"img {img} EXT{ext}: charge sharing where straight tracks cross others (log colour, 3-3000 e-)", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(f"{O}/crossing_{nm}.png", dpi=60); plt.close(fig); print("crossing", nm, flush=True)

# --- injected-muon examples: img 9 EXT4 seed 1, pkg / mu2 / mu3 cluster labels
img, ext, seed = 9, 4, 1
Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 * seed + 10 * img + ext)
add, truth = IM.sim_muons(rng, Q.shape, 25); Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
labs = {}
for v in ("pkg", "mu2", "mu3"):
    xsplit.install(GX); streaks.install(AD)
    if v == "mu2": muons2.install(AD)
    if v == "mu3": muons3.install(AD)
    cl = [c for c in AD.adaptive_cluster(Qi, valid, bg, engine="gmx", return_pixels=True) if "_pix" in c and np.isfinite(c.get("E", np.nan))]
    cl, _ = HX.link_tracks(cl)
    muons2.uninstall(AD); muons3.uninstall(AD); streaks.uninstall(AD); xsplit.uninstall(GX)
    own = np.zeros(Q.shape, np.int32); best = np.zeros(Q.shape)
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; m = q > best[ys, xs]; own[ys[m], xs[m]] = k; best[ys[m], xs[m]] = q[m]
    labs[v] = own; print("inj", v, flush=True)
np.savez_compressed(f"{O}/inj_labels.npz", **labs)
sel = sorted(range(len(truth)), key=lambda i: -truth[i]["L"])[:12:2]
fig, axs = plt.subplots(len(sel), 4, figsize=(20, 5 * len(sel)))
for row, i in zip(axs, sel):
    t = truth[i]; yy0, yy1 = max(t["py"].min() - 15, 0), min(t["py"].max() + 16, Q.shape[0]); xx0, xx1 = max(t["px"].min() - 15, 0), min(t["px"].max() + 16, Q.shape[1])
    R = np.where(valid, Qi - bg.mean, 0)
    row[0].imshow(np.clip(R[yy0:yy1, xx0:xx1], 3, None), norm=LogNorm(3, 3000), cmap="gray_r", origin="lower", extent=(xx0, xx1, yy0, yy1), interpolation="nearest", aspect="auto")
    row[0].plot(t["px"] + .5, t["py"] + .5, ",", color="cyan", alpha=0.25)
    row[0].set_title(f"injected muon, L {t['L']:.0f} px (cyan = its pixels)", fontsize=12)
    for a, v in zip(row[1:], ("pkg", "mu2", "mu3")):
        own = labs[v]; sub = own[yy0:yy1, xx0:xx1]
        k = own[t["py"], t["px"]]; sh = np.bincount(k, t["q"]); sh[0] = 0; tot = t["q"].sum()
        nfr = int((sh >= 0.05 * tot).sum()); comp = sh.max() / tot
        rng2 = np.random.default_rng(3); cols = rng2.uniform(0.3, 0.9, (own.max() + 1, 3)); cols[0] = 0.07
        img_ = cols[sub].copy(); mine = np.isin(sub, np.nonzero(sh >= 0.05 * tot)[0]) & (sub > 0)
        a.imshow(img_, origin="lower", extent=(xx0, xx1, yy0, yy1), interpolation="nearest", aspect="auto")
        a.contour(np.arange(xx0, xx1) + .5, np.arange(yy0, yy1) + .5, mine, levels=[0.5], colors="white", linewidths=0.8)
        a.set_title(f"{v}: {nfr} fragment(s) >= 5 %, largest holds {comp:.0%}", fontsize=12)
    for a in row: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{O}/inj_examples.png", dpi=55); plt.close(fig); print("examples done")
