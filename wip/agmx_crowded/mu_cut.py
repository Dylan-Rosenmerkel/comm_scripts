import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, muons
from scipy import ndimage as nd
AM = HX.AM
Q, valid = HX.load(9, 4); rng = np.random.default_rng(1094)
add, truth = IM.sim_muons(rng, Q.shape, 25)
Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
dbg = []; tk, mu = muons.find_muons(Qi, valid, bg, core_e=60, max_gap=6, dbg=dbg); segs = dbg[0][1]
R = np.where(valid, Qi - bg.mean, 0); Rs = nd.gaussian_filter(R, 1)
nz = np.sqrt(np.maximum(bg.lam, 0)) / (2*np.sqrt(np.pi)); core = (Rs > np.maximum(60, 5*nz)) & valid
print("core fill", core.mean(), "lam median", np.median(bg.lam))
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
for j, i in enumerate([0, 1, 2]):
    t = truth[i]; y0, y1, x0, x1 = t["py"].min()-30, t["py"].max()+30, t["px"].min()-30, t["px"].max()+30
    y0, x0 = max(y0, 0), max(x0, 0)
    a = ax[0, j]; a.imshow(np.log10(np.clip(R[y0:y1, x0:x1], 1, None)), origin="lower", cmap="magma", extent=(x0, x1, y0, y1))
    a.plot(t["px"], t["py"], ",", color="cyan", alpha=.3); a.set_title(f"inj L {t['L']:.0f} th {t['th']:.0f}  (cyan = injected px)")
    a = ax[1, j]; a.imshow(core[y0:y1, x0:x1] + 2*tk[y0:y1, x0:x1], origin="lower", cmap="viridis", extent=(x0, x1, y0, y1))
    for (p, q) in segs:
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1: a.plot([p[0], q[0]], [p[1], q[1]], "r-", lw=1)
    a.set_title("core (green), taken tubes (yellow), Hough segs (red)")
plt.tight_layout(); plt.savefig("mu_cut.png", dpi=80)
for t in truth[:5]:
    y, x = t["py"], t["px"]
    print(f"L {t['L']:.0f}: core at truth px {core[y,x].mean():.2f}  R median {np.median(R[y,x]):.0f}  Rs median {np.median(Rs[y,x]):.0f}  Qi median {np.median(Qi[y,x]):.0f}  tk {tk[y,x].mean():.2f}")
