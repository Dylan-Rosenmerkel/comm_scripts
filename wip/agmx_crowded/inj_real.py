"""where does injected muon charge go on a real L2 image (pkg)"""
import sys, numpy as np
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, xsplit, streaks
AM, AD, GX = HX.AM, HX.AD, HX.GX
img, ext = int(sys.argv[1]), int(sys.argv[2])
Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 + 10 * img + ext)
add, truth = IM.sim_muons(rng, Q.shape, 25)
Qi = np.where(valid, Q + add, 0.0)
bg, rep = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
xsplit.install(GX); streaks.install(AD)
cl = [c for c in AD.adaptive_cluster(Qi, valid, bg, engine="gmx", return_pixels=True) if "_pix" in c and np.isfinite(c.get("E", np.nan))]
cl0 = cl
cl, _ = HX.link_tracks(cl)
H, W = Q.shape
def own_of(cl):
    own = np.zeros((H, W), np.int32); best = np.zeros((H, W))
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; m = q > best[ys, xs]; own[ys[m], xs[m]] = k; best[ys[m], xs[m]] = q[m]
    return own
own = own_of(cl); Ee = np.array([c["E"] / 3.75e-3 for c in cl])
# real-image charge in the muon footprint (dilated 2 px) -> crowding measure
from scipy import ndimage as ndi
bgarr = np.asarray(getattr(bg, "map", bg)) if not isinstance(bg, np.ndarray) else bg
for t in sorted(truth, key=lambda t: t["L"]):
    fp = np.zeros((H, W), bool); fp[t["py"], t["px"]] = True; fp = ndi.binary_dilation(fp, iterations=2)
    k = own[t["py"], t["px"]]; tot = t["q"].sum(); sh = np.bincount(k, t["q"], len(cl) + 1)
    b = int(sh[1:].argmax()) + 1; big = np.sort(sh[1:])[::-1][:4] / tot
    nreal = int((Q[fp] > 50).sum())
    print(f"L {t['L']:4.0f} th {t['th']:5.1f} | uncl {sh[0]/tot:.2f} shares {np.round(big,2)} nfrag5% {(sh[1:]>=.05*tot).sum()} | best E/mu {Ee[b-1]/tot:5.2f} npix {cl[b-1]['_pix'][0].size:6d} cls {cl[b-1].get('cls','')[:8]:8s} | real px>50e in footprint {nreal}")
