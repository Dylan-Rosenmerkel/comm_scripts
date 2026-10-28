"""finder tubes (mu3) overlaid on lower-background images: Fe-55 production CCD and an Am-241 CCD"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, ".")
import harness as HX, regress as RG, bgv3, muons3
AM = HX.AM
J = {j[4]: j for j in RG.jobs()}
pick = [t for t in J if "DM07_DM09" in t and t.endswith("_ext1")][:1] + [t for t in J if "DM18_DM19" in t and "20250907" in t and t.endswith("_ext4")][:1] + \
       [t for t in J if t.startswith("Am241") and "_28_" in t][:1]
sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/clustering_tests"))
for t in pick:
    kind, run, stem, ext, tag = J[t]
    if kind == "am":
        import am241_29img_recluster as RC; Q, valid = RC.load(stem, ext)
    else:
        Q, valid, _ = RG.load_run(run, stem, ext)
    bg, _ = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid))
    tk, mu, mq = muons3.find_muons(Q, valid, bg)
    R = np.where(valid, Q - bg.mean, 0)
    lab = np.zeros(Q.shape); rng = np.random.default_rng(0)
    for m in mu: lab[m["_pix"][0], m["_pix"][1]] = rng.uniform(0.05, 1)
    lab = np.ma.masked_equal(lab, 0)
    W = Q.shape[1]; segs = [(0, W // 3), (W // 3, 2 * W // 3), (2 * W // 3, W)]
    fig, axs = plt.subplots(3, 2, figsize=(30, 14))
    for row, (x0, x1) in zip(axs, segs):
        for k, a in enumerate(row):
            a.imshow(np.clip(R[:, x0:x1], 3, None), norm=LogNorm(3, 3000), cmap="gray_r", origin="lower", aspect="auto", interpolation="nearest", extent=(x0, x1, 0, Q.shape[0]))
            if k: a.imshow(lab[:, x0:x1], cmap="hsv", alpha=.6, origin="lower", aspect="auto", interpolation="nearest", extent=(x0, x1, 0, Q.shape[0]), vmin=0, vmax=1)
            a.set_yticks([])
    axs[0, 0].set_title(f"{tag}: image", loc="left", fontsize=14, weight="bold")
    axs[0, 1].set_title(f"straight-track finder (mu3): {len(mu)} tracks, {tk[valid].mean():.2%} of pixels", loc="left", fontsize=14, weight="bold")
    fig.tight_layout(); nm = "am241" if kind == "am" else tag[1:20]
    fig.savefig(f"report/tubes_{nm}.png", dpi=50); plt.close(fig)
    print(tag, len(mu), tk[valid].mean(), "lengths", np.percentile([m["length"] for m in mu], [10, 50, 90]).round() if mu else None, flush=True)
