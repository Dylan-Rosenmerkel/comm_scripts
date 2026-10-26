"""Figures for docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip): every step of adaptive+GMX run explicitly on a real low-background
Am-241 image (AMTEst EXT1) with known objects injected in a quiet region, plus the performance comparisons.
  python make_agmx_figures.py            -> docs/aGMX/figures/*.png
  AGMX_FIG_OUT=DIR (relative to docs/aGMX) writes elsewhere; MUON_PHYS=1 injects a minimum-ionising muon (inject_utils)"""
import json, os, sys
import numpy as np
from scipy import ndimage, stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_test as AT
import gmx as GX
AM, CM = GX.AM, AT.CM
NE = 1 / AM.E_PER_E
OUT = os.path.join(HERE, "..", "docs", "aGMX", os.environ.get("AGMX_FIG_OUT", "figures"))
os.makedirs(OUT, exist_ok=True)
D = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig); print(name)


# ------------------------------------------------------------------------------------------------ the scene
def quiet_window(Q, valid, h=130, w=560):
    busy = ndimage.uniform_filter((Q > 3).astype(float), (h, w), mode="constant")
    ok = np.zeros_like(valid); ok[h // 2 + 10:-h // 2 - 10, w // 2 + 10:-w // 2 - 10] = True
    busy[~(ok & valid)] = 9
    y, x = np.unravel_index(np.argmin(busy), busy.shape)
    return y - h // 2, x - w // 2


def gauss_cloud(rng, n, y0, x0, s):
    return y0 + rng.normal(0, s, n), x0 + rng.normal(0, s, n)


def make_scene():
    Q, valid, sw = AT.load(AT.IMGS[2], 1)
    y0, x0 = quiet_window(Q, valid)
    rng = np.random.default_rng(5)
    objs = []
    def add(name, ys, xs, E, centres):
        iy, ix = np.round(ys).astype(int), np.round(xs).astype(int)
        np.add.at(Q, (iy, ix), 1.0); objs.append(dict(name=name, E=E, c=centres))
    Y = y0 + 40
    n = rng.poisson(1580); add("sharp Fe-55", *gauss_cloud(rng, n, Y, x0 + 25, 0.35), n / NE, [(Y, x0 + 25)])
    n = rng.poisson(1580); add("diffuse Fe-55", *gauss_cloud(rng, n, Y, x0 + 85, 1.6), n / NE, [(Y, x0 + 85)])
    for dx in (0, 4):
        n = rng.poisson(1580); add(f"pair {'a' if dx == 0 else 'b'}", *gauss_cloud(rng, n, Y, x0 + 145 + dx, 0.6), n / NE, [(Y, x0 + 145 + dx)])
    n = rng.poisson(4746); add("diffuse 17.75 keV", *gauss_cloud(rng, n, Y, x0 + 225, 2.0), n / NE, [(Y, x0 + 225)])
    n = rng.poisson(1580); add("sharp Fe-55 on it", *gauss_cloud(rng, n, Y, x0 + 229, 0.3), n / NE, [(Y, x0 + 229)])
    n = rng.poisson(15920); add("59.5 keV", *gauss_cloud(rng, n, Y + 45, x0 + 300, 1.0), n / NE, [(Y + 45, x0 + 300)])
    n = rng.poisson(400); add("faint 1.5 keV", *gauss_cloud(rng, n, Y + 45, x0 + 306, 0.5), n / NE, [(Y + 45, x0 + 306)])  # 6 px from the 59.5 keV
    import inject_utils as IU
    L = 40; n = IU.muon_electrons(rng, L, AM.E_PER_E) if IU.MUON_PHYS else 40 * 250
    t = rng.uniform(0, 1, n); s = 0.2 + t
    add("muon", Y + 45 + (t - .5) * L * np.sin(.35) + rng.normal(0, 1, n) * s, x0 + 380 + (t - .5) * L * np.cos(.35) + rng.normal(0, 1, n) * s, n / NE, [(Y + 45, x0 + 380)])
    ns = 18; a = 0.3; p = np.array([Y + 40.0, x0 + 470.0]); pts = []
    for i in range(ns):
        pts.append(p.copy()); a += rng.normal(0, .55); p += [np.sin(a), np.cos(a)]
    pts = np.array(pts); q = rng.poisson(150 * rng.exponential(1, ns) * np.where(np.arange(ns) >= ns - 3, 3, 1))
    add("worm", np.repeat(pts[:, 0], q) + rng.normal(0, .5, q.sum()), np.repeat(pts[:, 1], q) + rng.normal(0, .5, q.sum()), q.sum() / NE, [tuple(pts.mean(0))])
    add("5 e- event", *gauss_cloud(rng, 5, Y, x0 + 300, 0.4), 5 / NE, [(Y, x0 + 300)])
    return Q, valid, sw, (y0, x0), objs


Q, valid, sw, (Y0, X0), OBJ = make_scene()
H, W = Q.shape
CROP = (slice(Y0, Y0 + 130), slice(X0, X0 + 560))
EXT = (X0 - 0.5, X0 + 560 - 0.5, Y0 + 130 - 0.5, Y0 - 0.5)
bg = AM.background_map(Q, valid)
R = np.where(valid, Q - bg.mean, 0.0); V = np.where(valid, bg.var, 1e6); Rp = np.clip(R, 0, None)
eps = np.sqrt(np.where(valid, bg.var, 1.0))
SC = (0.5, 1.0, 1.5, 2.5)


def show(ax, img, title, cmap="Greys", stretch=True, vmax=None):
    v = np.sign(img) * np.sqrt(np.abs(img)) if stretch else img
    ax.imshow(v[CROP], cmap=cmap, extent=EXT, vmin=None if vmax is None else 0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, loc="left"); ax.set_xticks([]); ax.set_yticks([])


def ellipses(ax, mix, col="tab:red", nsig=2, lw=1.0):
    for k in range(len(mix)):
        y, x = mix.mu[k]
        if not (Y0 <= y < Y0 + 130 and X0 <= x < X0 + 560): continue
        a, c, b = mix.S[k, 0] + GX.PIX, mix.S[k, 1] + GX.PIX, mix.S[k, 2]
        ev, evec = np.linalg.eigh(np.array([[a, b], [b, c]]))
        ang = np.degrees(np.arctan2(evec[0, 1], evec[1, 1]))
        ax.add_patch(Ellipse((x, y), 2 * nsig * np.sqrt(ev[1]), 2 * nsig * np.sqrt(ev[0]), angle=ang, fill=False, ec=col, lw=lw))


# ------------------------------------------------------------------------------------------------ fig 1: overview
cl, info = GX.general_cluster(Q, valid, bg, return_info=True)
WA = CM.wader(Q, sw, 10, 30)[0]
fig, ax = plt.subplots(3, 1, figsize=(13, 9.5))
show(ax[0], Q, "the test scene: a real low-background CCD image (Am-241 EXT1, 0.12 e- noise) with 11 known deposits injected (sqrt stretch)")
for o in OBJ:
    y, x = o["c"][0]; ax[0].annotate(o["name"], (x, y), (x - 10, y - 18 if not any(t in o["name"] for t in ("pair b", "on it", "faint")) else y + 26), fontsize=7, color="tab:blue",
                                    arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.6))
def labels(a, items, col):
    """cluster markers + labels; a cluster close to an already labelled one is labelled below its marker"""
    placed = []
    for x, y, t in sorted(items):
        # a cluster within 12 px of an already labelled one gets its label below its marker
        dy = 14 if any(abs(x - px) < 12 and abs(y - py) < 12 for px, py in placed) else -5
        placed.append((x, y))
        a.plot(x, y, "o", mfc="none", mec=col, ms=12); a.text(x + 4, y + dy, t, fontsize=6.5, color=col)


show(ax[1], Q, f"WADERs (10 / 30 sigma): {sum(1 for c in WA if Y0 <= c['y'] < Y0 + 130 and X0 <= c['x'] < X0 + 560)} clusters in the window")
labels(ax[1], [(c["x"], c["y"], f"{c['E']:.2f}") for c in WA if Y0 <= c["y"] < Y0 + 130 and X0 <= c["x"] < X0 + 560], "tab:orange")
show(ax[2], Q, f"A_GMX: {sum(1 for c in cl if Y0 <= c['y'] < Y0 + 130 and X0 <= c['x'] < X0 + 560)} clusters in the window (energy keV, class)")
labels(ax[2], [(c["x"], c["y"], f"{c['E']:.2f} {c.get('cls', '')}") for c in cl if Y0 <= c["y"] < Y0 + 130 and X0 <= c["x"] < X0 + 560], "tab:red")
fig.tight_layout(); save(fig, "fig01_overview.png")
truth = [(o["name"], o["E"], o["c"][0]) for o in OBJ]
json.dump(dict(truth=[(n, float(e), [float(v) for v in c]) for n, e, c in truth],
               agmx=[dict(E=c["E"], y=c["y"], x=c["x"], cls=c.get("cls"), blend=c.get("blend")) for c in cl if Y0 <= c["y"] < Y0 + 130 and X0 <= c["x"] < X0 + 560],
               waders=[dict(E=c["E"], y=c["y"], x=c["x"]) for c in WA if Y0 <= c["y"] < Y0 + 130 and X0 <= c["x"] < X0 + 560]),
          open(os.path.join(OUT, "scene_results.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------------------------------------ fig 2: background
fig = plt.figure(figsize=(13, 7.5)); gs = fig.add_gridspec(2, 3)
for i, (m, t) in enumerate(((bg.o, "offset o [e-]"), (bg.lam, "dark charge lambda [e-/pix]"), (bg.s, "read noise s [e-]"))):
    a = fig.add_subplot(gs[0, i]); im = a.imshow(np.where(valid, m, np.nan), aspect="auto", cmap="viridis"); plt.colorbar(im, ax=a); a.set_title(t, loc="left")
    a.set_xlabel("column"); a.set_ylabel("row")
a = fig.add_subplot(gs[1, :2])
quiet = valid & ~bg.mask
v = Q[quiet]; bins = np.arange(-1, 4, 0.04)
a.hist(v, bins, histtype="step", color="k", label="pixels away from clusters (data)")
lam, s, o = float(np.median(bg.lam[quiet])), float(np.median(bg.s[quiet])), float(np.median(bg.o[quiet]))
xs = np.linspace(-1, 4, 1000)
mod = sum(stats.poisson.pmf(k, lam) * stats.norm.pdf(xs, o + k, s) for k in range(6)) * v.size * 0.04
a.plot(xs, mod, "r", lw=1.2, label=f"model: sum_k Poisson(k; lambda={lam:.3f}) x Normal(o + k, s={s:.2f})")
a.set_yscale("log"); a.set_ylim(0.5, None); a.set_xlabel("pixel value Q [e-]"); a.set_ylabel("pixels / 0.04 e-"); a.legend(fontsize=8)
a.set_title("the empty-pixel model: integer electrons (Poisson dark charge) smeared by Gaussian read noise", loc="left")
a = fig.add_subplot(gs[1, 2]); a.axis("off")
a.text(0, 0.95, "per pixel:\n  mean = o + lambda\n  var  = lambda + s^2\n\nresidual charge:\n  R = Q - mean\n\nempty pixel: R ~ 0 +- sqrt(var)\ndeposit:     R = its charge", va="top", family="monospace", fontsize=10)
fig.tight_layout(); save(fig, "fig02_background.png")

# ------------------------------------------------------------------------------------------------ fig 3: matched filter + seeds
T = GX.mf_maps(R, np.where(valid, bg.var, 0.0), SC)
py, px, s0 = GX.scale_space_seeds(T, SC, Q, valid, bg, 5.0)
fig, ax = plt.subplots(5, 1, figsize=(13, 13))
for k, s in enumerate(SC):
    a = ax[k]; a.imshow(np.clip(T[k][CROP], 0, 60), cmap="magma", extent=EXT, vmin=0, vmax=60)
    pk = (T[k] == ndimage.maximum_filter(T[k], size=5, mode="constant")) & (T[k] > 5) & valid
    yy, xx = np.nonzero(pk[CROP]); a.plot(xx + X0, yy + Y0, "c+", ms=7, mew=1.2, label="local maxima T > 5 at this scale")
    sel = s0 == s; m = (py >= Y0) & (py < Y0 + 130) & (px >= X0) & (px < X0 + 560) & sel
    a.plot(px[m], py[m], "o", mfc="none", mec="lime", ms=11, mew=1.5, label="kept as seed (finest scale first)")
    a.set_title(f"matched-filter significance T at kernel width w = {s} px (clipped at 60)", loc="left"); a.set_xticks([]); a.set_yticks([])
    if k == 0: a.legend(fontsize=7, loc="upper right")
show(ax[4], R, "all seeds on the residual image R (colour = scale where the seed was taken)")
for s, col in zip(SC, ("lime", "cyan", "yellow", "magenta")):
    m = (s0 == s) & (py >= Y0) & (py < Y0 + 130) & (px >= X0) & (px < X0 + 560)
    ax[4].plot(px[m], py[m], "o", mfc="none", mec=col, ms=10, mew=1.4, label=f"w = {s}")
ax[4].legend(fontsize=7, ncol=4, loc="upper right")
fig.tight_layout(); save(fig, "fig03_matched_filter_seeds.png")

# ------------------------------------------------------------------------------------------------ fig 4: Poisson check
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for lam_, c in ((0.001, "tab:blue"), (0.01, "tab:green"), (0.1, "tab:red")):
    k = np.arange(0, 12); mu9 = 9 * lam_
    ax[0].semilogy(k, stats.poisson.sf(k - 1, mu9), "o-", color=c, label=f"P(box sum >= k), lambda = {lam_}")
    z = (k - mu9) / np.sqrt(mu9 + 9 * 0.12 ** 2)
    ax[0].semilogy(k, np.clip(stats.norm.sf(z), 1e-30, 1), "--", color=c, alpha=.6)
ax[0].axhline(stats.norm.sf(5), color="k", lw=0.8); ax[0].text(0.2, stats.norm.sf(5) * 2, "required: below the Gaussian 5 sigma tail (2.9e-7)", fontsize=7.5)
ax[0].set_ylim(1e-25, 2); ax[0].set_xlabel("charge k in the 3x3 box [e-]"); ax[0].set_ylabel("tail probability")
ax[0].legend(fontsize=7); ax[0].set_title("exact Poisson tail (solid) vs Gaussian approximation (dashed)", loc="left")
ax[1].axis("off")
ax[1].text(0, 0.95, "Why: at lambda = 0.01 a 3x3 box of empty pixels has variance\n9 (lambda + s^2) = 0.22 e-^2, so a lone 3-electron box is\n(3 - 0.09) / 0.47 = 6.2 'Gaussian sigma' - but its true\nprobability is P(Poisson(0.09) >= 3) = 1.1e-4, 400x more\nlikely than 5 sigma. Integer counts have heavier tails than\nthe Gaussian; the seed must pass the exact Poisson test\n(Li & Ma 1983 idea) at the same false-alarm rate.", va="top", family="monospace", fontsize=9)
fig.tight_layout(); save(fig, "fig04_poisson_check.png")

# ------------------------------------------------------------------------------------------------ fig 5: EM evolution
A0 = np.maximum(R[py, px] * 2 * np.pi * (s0 ** 2 + GX.PIX), 1.0)
mix0 = GX.Mix(A0, np.c_[py, px].astype(float), np.c_[s0 ** 2, s0 ** 2, np.zeros_like(s0)])
snaps = {0: mix0}
for n in (1, 3, 30):
    snaps[n] = GX.em(mix0, Rp, eps, valid, n)
zoom = (slice(Y0 + 25, Y0 + 60), slice(X0 + 130, X0 + 245)); zext = (X0 + 129.5, X0 + 244.5, Y0 + 59.5, Y0 + 24.5)
fig, ax = plt.subplots(4, 1, figsize=(12, 10))
for a, (n, mx) in zip(ax, snaps.items()):
    a.imshow(np.sqrt(np.clip(R[zoom], 0, None)), cmap="Greys", extent=zext, interpolation="nearest")
    vis = [k for k in range(len(mx)) if zoom[0].start <= mx.mu[k, 0] < zoom[0].stop and zoom[1].start <= mx.mu[k, 1] < zoom[1].stop]
    vis = sorted(vis, key=lambda k: mx.mu[k, 1])            # left to right: labels alternate above / below
    for j, k in enumerate(vis):
        y, x = mx.mu[k]
        aa, cc, bb = mx.S[k, 0] + GX.PIX, mx.S[k, 1] + GX.PIX, mx.S[k, 2]
        ev, evec = np.linalg.eigh(np.array([[aa, bb], [bb, cc]]))
        a.add_patch(Ellipse((x, y), 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=np.degrees(np.arctan2(evec[0, 1], evec[1, 1])), fill=False, ec="tab:red", lw=1.2))
        a.text(x - 4, y - 4.5 if (j % 2 == 0) else y + 6.5, f"A={mx.A[k]:.0f} e-", color="tab:red", fontsize=7)
    a.set_title(f"after {n} EM iteration(s): red = 2-sigma ellipse of each cloud, A = its charge (truth: pair 1580 + 1580 e-, diffuse 4746 + sharp 1580 e-)", loc="left", fontsize=8.5)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig05_em_evolution.png")

# ------------------------------------------------------------------------------------------------ fig 6: responsibilities + charge sharing
mix = snaps[30]
I, K, G, DY, DX = GX.evaluate(mix, R.shape, valid)
M = np.bincount(I, G, H * W)
yc = int(round(OBJ[4]["c"][0][0])); xs_ = np.arange(X0 + 212, X0 + 242)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
row = R[yc, xs_]
ks = [k for k in range(len(mix)) if abs(mix.mu[k, 0] - yc) < 3 and X0 + 212 <= mix.mu[k, 1] < X0 + 242]
comp = {}
for k in ks:
    sel = K == k; g = np.zeros(H * W); np.add.at(g, I[sel], G[sel]); comp[k] = g.reshape(H, W)[yc, xs_]
ax[0].bar(xs_, row, width=0.9, color="0.8", label="data R along the row through both deposits")
for k, c in zip(ks, ("tab:blue", "tab:red", "tab:green", "tab:purple")):
    ax[0].plot(xs_, comp[k], "-", color=c, lw=1.5, label=f"cloud {k}: sigma {np.sqrt(0.5 * (mix.S[k, 0] + mix.S[k, 1])):.2f} px, A {mix.A[k]:.0f} e-")
ax[0].plot(xs_, M.reshape(H, W)[yc, xs_], "k--", lw=1, label="model M = sum of clouds")
ax[0].set_xlabel("column"); ax[0].set_ylabel("e- per pixel"); ax[0].legend(fontsize=7); ax[0].set_title("sharp X-ray on a diffuse one: two clouds explain the row", loc="left")
tot = sum(comp.values()) + eps.reshape(H, W)[yc, xs_]
for k, c in zip(ks, ("tab:blue", "tab:red", "tab:green", "tab:purple")):
    ax[1].plot(xs_, comp[k] / tot, "-", color=c, lw=1.5, label=f"responsibility of cloud {k}")
ax[1].plot(xs_, eps.reshape(H, W)[yc, xs_] / tot, ":", color="0.4", label="noise floor eps / (M + eps)")
ax[1].set_xlabel("column"); ax[1].set_ylim(-0.02, 1.02); ax[1].legend(fontsize=7)
ax[1].set_title("r(p, k) = g_k(p) / (M(p) + eps(p)): the share of each pixel", loc="left")
fig.tight_layout(); save(fig, "fig06_responsibilities.png")

# ------------------------------------------------------------------------------------------------ fig 7: residual pass + pruning
Mimg = M.reshape(H, W); Rr = np.where(valid, R - Mimg, 0.0)
Tr = GX.mf_maps(Rr, np.where(valid, bg.var + np.clip(Mimg, 0, None), 0.0), SC[:2]).max(0)
zoom2 = (slice(Y0 + 70, Y0 + 100), slice(X0 + 285, X0 + 320)); z2 = (X0 + 284.5, X0 + 319.5, Y0 + 99.5, Y0 + 69.5)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for a, (img, t) in zip(ax[:3], ((R, "data R: 59.5 keV + faint 1.5 keV 6 px away"), (Mimg, "model after EM"), (Rr, "residual R - M"))):
    a.imshow(np.sign(img[zoom2]) * np.sqrt(np.abs(img[zoom2])), cmap="RdBu_r" if img is Rr else "Greys", extent=z2, interpolation="nearest",
             vmin=-6 if img is Rr else None, vmax=6 if img is Rr else None)
    a.set_title(t, fontsize=8.5, loc="left"); a.set_xticks([]); a.set_yticks([])
pk2 = (Tr == ndimage.maximum_filter(Tr, size=3, mode="constant")) & (Tr > 5) & valid
yy, xx = np.nonzero(pk2[zoom2]); ax[2].plot(xx + z2[0] + 0.5, yy + z2[3] + 0.5, "o", mfc="none", mec="lime", ms=12, mew=1.5)
z = GX.significance(mix, Rp, V, valid)
ax[3].hist(np.log10(np.clip(z, 1e-2, None)), 60, color="0.5"); ax[3].axvline(np.log10(4), color="r"); ax[3].set_xlabel("log10 cloud S/N")
ax[3].set_title(f"pruning: keep S/N > 4 ({(z > 4).sum()} of {len(z)} clouds)", fontsize=8.5, loc="left")
fig.tight_layout(); save(fig, "fig07_residual_pruning.png")

# ------------------------------------------------------------------------------------------------ fig 8: ridge test
mixf = info["mix"]; labf = info["lab"]
a_, c_, b_ = GX._tot_cov(mixf.S); det = a_ * c_ - b_ * b_
def dens(k, yy, xx):
    dy, dx = yy - mixf.mu[k, 0], xx - mixf.mu[k, 1]
    return mixf.A[k] / (2 * np.pi * np.sqrt(det[k])) * np.exp(-0.5 * (c_[k] * dy * dy + a_[k] * dx * dx - 2 * b_[k] * dy * dx) / det[k])
def nearest(y, x):
    return int(np.argmin(np.hypot(mixf.mu[:, 0] - y, mixf.mu[:, 1] - x)))
cases = [("two X-rays 4 px apart", OBJ[2]["c"][0], OBJ[3]["c"][0]), ("sharp X-ray on a diffuse one", OBJ[4]["c"][0], OBJ[5]["c"][0])]
mu_clouds = [k for k in range(len(mixf)) if np.hypot(mixf.mu[k, 0] - OBJ[8]["c"][0][0], mixf.mu[k, 1] - OBJ[8]["c"][0][1]) < 22]
if len(mu_clouds) >= 2:
    mu_clouds = sorted(mu_clouds, key=lambda k: mixf.mu[k, 1])
    cases.append(("two clouds of the muon", tuple(mixf.mu[mu_clouds[0]]), tuple(mixf.mu[mu_clouds[1]])))
fig, ax = plt.subplots(1, len(cases), figsize=(14, 3.8))
for a, (t, p1, p2) in zip(np.atleast_1d(ax), cases):
    i, j = nearest(*p1), nearest(*p2)
    tt = np.linspace(-0.4, 1.4, 200); yy = mixf.mu[i, 0] + tt * (mixf.mu[j, 0] - mixf.mu[i, 0]); xx = mixf.mu[i, 1] + tt * (mixf.mu[j, 1] - mixf.mu[i, 1])
    fi, fj = dens(i, yy, xx), dens(j, yy, xx); f = fi + fj
    inner = (tt >= 0) & (tt <= 1)
    ridge = f[inner].min() / min(f[np.argmin(np.abs(tt))], f[np.argmin(np.abs(tt - 1))])
    a.plot(tt, fi, "tab:blue", lw=1); a.plot(tt, fj, "tab:red", lw=1); a.plot(tt, f, "k", lw=1.6)
    a.axhline(f[inner].min(), color="0.5", ls=":")
    same = labf[i] == labf[j]
    a.set_title(f"{t}\nridge = {ridge:.2f} -> {'merged' if same else 'kept apart'}", fontsize=9, loc="left")
    a.set_xlabel("position along the line joining the two centres (0 = cloud 1, 1 = cloud 2)"); a.set_yticks([])
fig.tight_layout(); save(fig, "fig08_ridge_test.png")

# ------------------------------------------------------------------------------------------------ fig 9: island BIC test
dec = info.get("decisions", [])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
if dec:
    d = np.array([(x[3], x[4], x[5], x[1], x[2]) for x in dec])
    ax[0].scatter(d[:, 0], d[:, 1], c=np.where(d[:, 2] > 30, "tab:red", "tab:blue"), s=14)
    lim = [max(d[:, :2].min(), 1), d[:, :2].max() * 1.2]; ax[0].plot(lim, lim, "k:", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("chi^2 of H_free (free-shape clouds)"); ax[0].set_ylabel("chi^2 of H_xr (K round X-rays)")
    ax[0].set_title("islands of this image: red = one deposit (dBIC > 30), blue = separate", fontsize=9, loc="left")
    ax[1].hist(np.clip(d[:, 2], -200, 2000), 60, color="0.5"); ax[1].axvline(30, color="r"); ax[1].set_xlabel("dBIC = BIC(H_xr) - BIC(H_free) (clipped)")
    ax[1].set_title("decision: dBIC > 30 -> merge the island into one deposit", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig09_island_bic.png")

# ------------------------------------------------------------------------------------------------ fig 10: final ownership + blend
If, Kf, Gf, _, _ = GX.evaluate(mixf, R.shape, valid)
pI, pL, rc, dom, nc = GX._aggregate(If, Kf, Gf, np.bincount(If, Gf, H * W), labf, H, W)
fig, ax = plt.subplots(2, 1, figsize=(13, 6.5))
cm = plt.get_cmap("tab20"); rgb = np.ones((130, 560, 3))
dd = dom[CROP]; m = dd >= 0; rgb[m] = cm(dd[m] % 20)[:, :3]
rgb *= (0.35 + 0.65 * np.clip(np.sqrt(np.clip(R[CROP], 0, None)) / 6, 0, 1))[..., None] ** 0.3
ax[0].imshow(rgb, extent=EXT, interpolation="nearest"); ax[0].set_title("which cluster owns most of each pixel's charge (colour = cluster; pixels can be shared)", loc="left")
ax[0].set_xticks([]); ax[0].set_yticks([])
show(ax[1], Q, "blend of each cluster = charge of the OTHER clusters under its profile / its own (0 = isolated)")
for c in cl:
    if Y0 <= c["y"] < Y0 + 130 and X0 <= c["x"] < X0 + 560 and c["E"] > 0.5:
        n_lab = sum(1 for q in cl if abs(q["y"] - c["y"]) < 3 and q["x"] < c["x"] and c["x"] - q["x"] < 12)
        ax[1].text(c["x"] - 6, c["y"] - 6 if n_lab % 2 == 0 else c["y"] + 10, f"{c.get('blend', 0):.3f}", fontsize=7, color="tab:red")
fig.tight_layout(); save(fig, "fig10_ownership_blend.png")

# ------------------------------------------------------------------------------------------------ fig 11: counting mode
o5 = OBJ[-1]["c"][0]; zy, zx = int(round(o5[0])), int(round(o5[1]))
sub = Q[zy - 3:zy + 4, zx - 3:zx + 4] - bg.o[zy - 3:zy + 4, zx - 3:zx + 4]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
im = ax[0].imshow(sub, cmap="Greys"); plt.colorbar(im, ax=ax[0])
for (i, j), v in np.ndenumerate(sub): ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="tab:red")
ax[0].set_title("a 5-electron event: pixel values Q - o [e-]", loc="left", fontsize=9)
ax[1].imshow(np.round(sub).clip(0), cmap="Greys")
for (i, j), v in np.ndenumerate(np.round(sub).clip(0)): ax[1].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="tab:red")
_nn = np.round(sub).clip(0); _lab, _ = ndimage.label(_nn >= 1, structure=np.ones((3, 3), bool))
_g = _lab[np.unravel_index(np.argmax(_nn), _nn.shape)]
ax[1].set_title(f"rounded to whole electrons:\nthe event (8-connected group) holds {_nn[_lab == _g].sum():.0f} e-"
                + (f",\n{_nn[(_lab > 0) & (_lab != _g)].sum():.0f} e- apart = dark charge / noise" if ((_lab > 0) & (_lab != _g)).any() else ""), loc="left", fontsize=9)
rng = np.random.default_rng(3); n_pix = 13
analog = [rng.normal(0, 0.15, n_pix).sum() for _ in range(20000)]
ax[2].hist(np.array(analog) + 5, 80, histtype="step", color="tab:blue", label="sum of 13 analog pixels (noise 0.15 e- each)")
ax[2].axvline(5, color="tab:red", lw=2, label="counting: exactly 5")
ax[2].set_xlabel("measured charge [e-]"); ax[2].legend(fontsize=7)
ax[2].set_title("why count: analog sums add sqrt(N) x noise = 0.54 e-", loc="left", fontsize=9)
fig.tight_layout(); save(fig, "fig11_counting.png")
print("scene figures done")


# ------------------------------------------------------------------------------------------------ fig 0: flowchart
import report_figs as RF
fig, a = plt.subplots(figsize=(13, 5.2)); a.axis("off")
B = RF._box; AR = RF._arr
B(a, 0.08, 0.5, "calibrated image\nQ [e-]", "#eee", fs=8.5)
B(a, 0.25, 0.5, "0  background model\no, lambda, s per pixel\nR = Q - (o + lambda)", "#eee", fs=8)
B(a, 0.43, 0.5, "A  adaptive switch:\ns >= 3 sigma_read and\ns >= 0.5 e-, or\nsigma_read >= 1 e- ?", "#ffe", fs=8)
B(a, 0.43, 0.13, "'unmodelled':\nhysteresis on Q\n(10/30 sigma of the\nzero peak, >= 4.5 e-)", "#ddd", fs=7.5)
B(a, 0.62, 0.86, "1  seeds: matched filter\nw = 0.5/1/1.5/2.5 px,\nfinest first, Poisson check", "#fde", fs=7.5)
B(a, 0.62, 0.5, "2  EM: all Gaussian clouds\nat once (exact pixel\nintegrals), charge SHARED", "#ffe", fs=7.5)
B(a, 0.62, 0.14, "3  residual pass\n+ prune S/N < 4", "#fde", fs=7.5)
B(a, 0.82, 0.86, "4  merge clouds:\nridge > 0.5 and\ncompatible shapes;\nfragments attach once", "#eed", fs=7.5)
B(a, 0.82, 0.5, "5  island BIC test:\nK round X-rays vs\none free-shape deposit", "#dfd", fs=7.5)
B(a, 0.82, 0.14, "6-8  energy by sharing,\nwidth, flags, counting,\nfew-e finder, class\n(9: few-e catalogue)", "#eef", fs=7.5)
AR(a, 0.13, 0.5, 0.17, 0.5); AR(a, 0.33, 0.5, 0.36, 0.5); AR(a, 0.43, 0.4, 0.43, 0.24, "yes"); AR(a, 0.5, 0.56, 0.55, 0.8, "no")
AR(a, 0.62, 0.78, 0.62, 0.6); AR(a, 0.62, 0.4, 0.62, 0.22); AR(a, 0.69, 0.18, 0.75, 0.8); AR(a, 0.82, 0.78, 0.82, 0.6); AR(a, 0.82, 0.4, 0.82, 0.24)
a.text(0.5, 0.98, "A_GMX = adaptive switch + GMX (Gaussian-mixture EM deblending)", ha="center", fontsize=12, weight="bold", transform=a.transAxes)
save(fig, "fig00_flowchart.png")

# ------------------------------------------------------------------------------------------------ fig 12: threshold vs sharing (1-D)
x = np.linspace(-8, 12, 801); dx = x[1] - x[0]
g = lambda A, m, s: A * np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
fig, ax = plt.subplots(2, 2, figsize=(13, 6.5))
d1 = g(100, 0, 2.0); thr = 4
for a, (title, comps) in zip(ax[:, 0], (("a diffuse cluster (sigma 2 px)", [d1]), ("two X-rays 4 px apart", [g(100, 0, 0.8), g(100, 4, 0.8)]))):
    tot = sum(comps); a.fill_between(x, tot, color="0.85"); a.plot(x, tot, "k")
    a.axhline(thr, color="tab:orange", ls="--"); above = tot > thr
    a.fill_between(x, 0, tot, where=above, color="tab:orange", alpha=.35)
    a.set_title(f"threshold + connect (WADERs): {title}\nkept charge {tot[above].sum() * dx:.0f} of {tot.sum() * dx:.0f}; "
                + ("one cluster" if np.sum(np.diff(above.astype(int)) == 1) <= 1 else "several clusters"), fontsize=9, loc="left")
    a.set_yticks([])
for a, (title, comps) in zip(ax[:, 1], (("a diffuse cluster", [d1]), ("two X-rays 4 px apart", [g(100, 0, 0.8), g(100, 4, 0.8)]))):
    tot = sum(comps); a.plot(x, tot, "k", lw=1.5)
    for c, col in zip(comps, ("tab:blue", "tab:red")):
        a.fill_between(x, 0, c, color=col, alpha=.3)
    a.set_title(f"fit + share (GMX): {title}\n" + ", ".join(f"cloud {i + 1}: {c.sum() * dx:.0f}" for i, c in enumerate(comps)) + " (all charge, split by model)", fontsize=9, loc="left")
    a.set_yticks([])
fig.tight_layout(); save(fig, "fig12_threshold_vs_sharing.png")

# ------------------------------------------------------------------------------------------------ fig 13: the island test as images
lab0 = GX.merge_graph(mixf)
island = (Q > GX.M2.link_threshold(bg, 0.01)) & valid
ilab, _ = ndimage.label(island, structure=GX.EIGHT)
fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
for row, (name, ctr) in enumerate((("worm", OBJ[9]["c"][0]), ("pair 4 px", OBJ[2]["c"][0]))):
    cy_, cx_ = int(round(ctr[0])), int(round(ctr[1]))
    win = ilab[cy_ - 8:cy_ + 9, cx_ - 8:cx_ + 9]; wq = R[cy_ - 8:cy_ + 9, cx_ - 8:cx_ + 9]
    L = int(win[np.unravel_index(np.argmax(np.where(win > 0, wq, -np.inf)), win.shape)])   # island of the brightest pixel nearby
    sl = ndimage.find_objects((ilab == L).astype(int))[0]
    y0, y1, x0, x1 = sl[0].start - 4, sl[0].stop + 4, sl[1].start - 4, sl[1].stop + 4
    comps = [k for k in range(len(mixf)) if y0 <= mixf.mu[k, 0] < y1 and x0 <= mixf.mu[k, 1] < x1 and ilab[int(round(mixf.mu[k, 0])), int(round(mixf.mu[k, 1]))] == L]
    sub = R[y0:y1, x0:x1]; use = np.ones(sub.shape, bool)
    mf_ = mixf.keep(np.isin(np.arange(len(mixf)), comps)); mf_ = GX.Mix(mf_.A, mf_.mu - [y0, x0], mf_.S)
    I_, K_, G_, _, _ = GX.evaluate(mf_, sub.shape, use); Mfree = np.bincount(I_, G_, sub.size).reshape(sub.shape)
    ul = np.unique(lab0[comps]); A0, mu0, S0 = [], [], []
    for u in ul:
        k = [c for c in comps if lab0[c] == u]; a_ = mixf.A[k]; Aa = a_.sum(); m_ = (a_[:, None] * mixf.mu[k]).sum(0) / Aa
        v = (a_ * (0.5 * (mixf.S[k, 0] + mixf.S[k, 1]) + 0.5 * ((mixf.mu[k] - m_) ** 2).sum(1))).sum() / Aa
        A0.append(Aa); mu0.append(m_ - [y0, x0]); S0.append([v, v, 0.0])
    mx = GX.em(GX.Mix(A0, mu0, S0), np.clip(sub, 0, None), eps[y0:y1, x0:x1], use, 25, iso=True)
    I_, K_, G_, _, _ = GX.evaluate(mx, sub.shape, use); Mxr = np.bincount(I_, G_, sub.size).reshape(sub.shape)
    w = 1 / (V[y0:y1, x0:x1] + np.clip(sub, 0, None) + (0.05 * sub) ** 2)
    cf, cx_ = float(((sub - Mfree) ** 2 * w).sum()), float(((sub - Mxr) ** 2 * w).sum())
    lnn = np.log(sub.size); dB = (cx_ + 4 * len(ul) * lnn) - (cf + 6 * len(comps) * lnn)
    vmax = np.sqrt(max(sub.max(), 1))
    for a, (img, t) in zip(ax[row], ((sub, f"{name}: data R"), (Mxr, f"H_xr: {len(ul)} round Gaussian(s)\nchi2 = {cx_:.0f}"),
                                     (Mfree, f"H_free: {len(comps)} free-shape clouds\nchi2 = {cf:.0f}"), (sub - Mxr, f"data - H_xr\ndBIC = {dB:.0f} -> {'one deposit' if dB > 30 else 'separate X-rays'}"))):
        a.imshow(np.sign(img) * np.sqrt(np.abs(img)), cmap="RdBu_r" if a is ax[row][3] else "Greys", vmin=-vmax if a is ax[row][3] else 0, vmax=vmax, interpolation="nearest")
        a.set_title(t, fontsize=8.5, loc="left"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig13_island_example.png")
print("extra figures done")
