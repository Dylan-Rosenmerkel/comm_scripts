"""sanity: inject the same muons into a quiet synthetic image (lambda 0.5, noise 0.17); where does the muon charge go?"""
import sys, numpy as np
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, xsplit, streaks, muons
AM, AD, GX = HX.AM, HX.AD, HX.GX
var = sys.argv[1] if len(sys.argv) > 1 else "pkg"
H, W = 500, 3700
rng = np.random.default_rng(7)
Q0 = rng.poisson(0.5, (H, W)) + rng.normal(0, 0.17, (H, W)); valid = np.ones((H, W), bool)
add, truth = IM.sim_muons(rng, (H, W), 25)
Qi = Q0 + add
bg = AM.background_map(Qi, valid)
pass
xsplit.install(GX); streaks.install(AD)
if var != "pkg":
    kw = IM.VARIANTS[var]; _fm = muons.find_muons; muons.find_muons = lambda *a, **k: _fm(*a, **{**k, **kw}); muons.install(AD)
cl = [c for c in AD.adaptive_cluster(Qi, valid, bg, engine="gmx", return_pixels=True) if "_pix" in c and np.isfinite(c.get("E", np.nan))]
cl, _ = HX.link_tracks(cl)
own = np.zeros((H, W), np.int32); best = np.zeros((H, W))
for k, c in enumerate(cl, 1):
    ys, xs, q = c["_pix"]; m = q > best[ys, xs]; own[ys[m], xs[m]] = k; best[ys[m], xs[m]] = q[m]
Ee = np.array([c["E"] / 3.75e-3 for c in cl])
for t in sorted(truth, key=lambda t: t["L"]):
    k = own[t["py"], t["px"]]; tot = t["q"].sum(); sh = np.bincount(k, t["q"], len(cl) + 1)
    b = int(sh[1:].argmax()) + 1
    big = np.sort(sh[1:])[::-1][:4] / tot
    print(f"L {t['L']:5.0f} th {t['th']:5.1f} sb {t['s_back']:.2f} Q {tot:8.0f} | unclustered {sh[0]/tot:.2f} top shares {np.round(big,2)} | best E/muon {Ee[b-1]/tot:.2f} cls {cl[b-1].get('cls','')} npix {cl[b-1]['_pix'][0].size} truth npix {t['py'].size}")
