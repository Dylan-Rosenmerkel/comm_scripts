"""Injection test of straight-track handling: realistic muons injected into the real L2 ACM101 images.
Muon model: projected length L ~ U(40, 400) px, direction uniform; 3-D path sqrt((15 L)^2 + 675^2) um; the charge along the
projection in 1-px steps, each step Moyal-fluctuated (knots); diffusion sigma = s_front + (s_back - s_front) t along the
track (t = depth fraction), s_front 0.2, s_back ~ U(0.8, 1.6) px.
Scoring per injected muon (dominant-cluster label image):
  best   = cluster holding the largest share of the muon's injected charge; complete = that share
  n_frag = clusters holding >= 5 % of it;  foreign = (best cluster's charge - muon's charge in it) / muon charge
  correct = complete >= 0.9 and foreign <= 0.1
python inject_mu.py IMG EXT SEED VARIANT  -> inj/IMG_EXT_SEED_VARIANT.json"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as HX, bgv3, xsplit, streaks, muons, muons2, muons3
from scipy.stats import moyal
AM, AD, GX = HX.AM, HX.AD, HX.GX
DEDX_E = 0.27 / 3.75e-3          # e- per um, most probable

def sim_muons(rng, shape, n_mu, margin=10):
    H, W = shape; img = np.zeros(shape); truth = []
    for m in range(n_mu):
        L = rng.uniform(40, 400); th = rng.uniform(0, np.pi)
        dy, dx = np.sin(th) * L, np.cos(th) * L
        for _ in range(50):
            y0 = rng.uniform(margin, H - margin); x0 = rng.uniform(margin, W - 200 - margin)
            if 0 <= y0 + dy < H and 0 <= x0 + dx < W - 200: break
        else: continue
        s_back = rng.uniform(0.8, 1.6); front_first = rng.random() < 0.5
        nstep = int(np.ceil(L)); step_um = np.hypot(15 * L, 675) / nstep
        pts_y, pts_x = [], []
        for k in range(nstep):
            t = (k + rng.random()) / nstep
            q = max(DEDX_E * step_um + 0.4 * DEDX_E * step_um * moyal.rvs(random_state=rng), 0)   # knots
            n = rng.poisson(q); depth = t if front_first else 1 - t
            s = 0.2 + (s_back - 0.2) * depth
            pts_y.append(y0 + t * dy + rng.normal(0, s, n)); pts_x.append(x0 + t * dx + rng.normal(0, s, n))
        ys = np.concatenate(pts_y); xs = np.concatenate(pts_x)
        iy = np.floor(ys).astype(int); ix = np.floor(xs).astype(int); ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        T = np.zeros(shape); np.add.at(T, (iy[ok], ix[ok]), 1.0)
        img += T; nz = np.nonzero(T); truth.append(dict(L=L, th=float(np.degrees(th)), s_back=s_back, n=float(T.sum()), py=nz[0], px=nz[1], q=T[nz]))
    return img, truth

def score(cl, truth, shape):
    H, W = shape
    own = np.zeros(shape, np.int32); best = np.zeros(shape)
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; m = q > best[ys, xs]; own[ys[m], xs[m]] = k; best[ys[m], xs[m]] = q[m]
    Ee = np.array([c["E"] / 3.75e-3 for c in cl])           # cluster charge [e-]
    res = []
    for t in truth:
        k = own[t["py"], t["px"]]; tot = t["q"].sum()
        sh = np.bincount(k, t["q"], len(cl) + 1); sh[0] = 0
        b = int(sh.argmax()); comp = sh[b] / tot if b else 0.0
        nfrag = int((sh[1:] >= 0.05 * tot).sum())
        foreign = (Ee[b - 1] - sh[b]) / tot if b else np.nan
        res.append(dict(L=t["L"], th=t["th"], s_back=t["s_back"], complete=float(comp), n_frag=nfrag, foreign=float(foreign),
                        correct=bool(comp >= 0.9 and foreign <= 0.1), best_cls=cl[b - 1].get("cls", "") if b else ""))
    return res

VARIANTS = {
    "pkg": {},
    "mu_c30_g6":  dict(core_e=30, max_gap=6), "mu_c30_g15": dict(core_e=30, max_gap=15),
    "mu_c60_g6":  dict(core_e=60, max_gap=6), "mu_c60_g15": dict(core_e=60, max_gap=15),
    "mu_c30_g15_r25": dict(core_e=30, max_gap=15, rms_max=2.5),
    "mu2": {}, "mu2_tight": dict(out_max=0.25, cap_k=1.2), "mu2_noshare": dict(share=False), "mu3": {},
}

if __name__ == "__main__":
    img, ext, seed, var = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs("inj", exist_ok=True); fn = f"inj/{img}_{ext}_{seed}_{var}.json"
    if os.path.exists(fn): sys.exit()
    Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 * seed + 10 * img + ext)
    add, truth = sim_muons(rng, Q.shape, 25)
    Qi = np.where(valid, Q + add, 0.0)
    t0 = time.time()
    bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
    xsplit.install(GX); streaks.install(AD)
    if var == "mu3":
        muons3.install(AD)
    elif var.startswith("mu2"):
        muons2.install(AD, **VARIANTS[var])
    elif var != "pkg":
        kw = VARIANTS[var]
        _fm = muons.find_muons; muons.find_muons = lambda *a, **k: _fm(*a, **{**k, **kw})
        muons.install(AD)
    cl = [c for c in AD.adaptive_cluster(Qi, valid, bg, engine="gmx", return_pixels=True) if "_pix" in c and np.isfinite(c.get("E", np.nan))]
    cl, _ = HX.link_tracks(cl)
    res = score(cl, truth, Q.shape)
    json.dump(dict(img=img, ext=ext, seed=seed, var=var, sec=time.time() - t0, n_cl=len(cl), res=res), open(fn, "w"))
    c = np.array([r["correct"] for r in res]); print(f"{img} EXT{ext} seed {seed} {var:16s}: correct {c.mean():.2f} ({c.sum()}/{c.size}), median complete {np.median([r['complete'] for r in res]):.2f}, frags {np.mean([r['n_frag'] for r in res]):.1f}, {time.time()-t0:.0f} s", flush=True)
