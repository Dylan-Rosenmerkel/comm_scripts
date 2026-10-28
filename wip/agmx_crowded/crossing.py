"""Track-crossing experiments on crops of image 9 EXT1 (regional background)."""
import sys, numpy as np
sys.path.insert(0, "."); import harness as HX
AM, AD, GX = HX.AM, HX.AD, HX.GX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Q, valid = HX.load(9, 1)
bg = HX.regional_bg(Q, valid, AM.background_map(Q, valid), use_mask=True)[0]
class Sub:  # background restricted to a crop
    def __init__(s, b, sl): s.o, s.lam, s.s2, s.mask, s.rows = b.o[sl], b.lam[sl], b.s2[sl], b.mask[sl], b.rows; s.mean, s.var, s.s = s.o + s.lam, s.lam + s.s2, np.sqrt(s.s2)
def labels(Qc, vc, bc, **kw):
    out = GX.general_cluster(Qc, vc, bc, return_pixels=True, **kw)
    lab = np.zeros(Qc.shape, int)
    for k, c in enumerate(sorted(out, key=lambda c: -c["_pix"][0].size), 1):
        ys, xs, q = c["_pix"]; m = q > 0.5; lab[ys[m], xs[m]] = k     # pixels with > 0.5 e- of the cluster's charge
    return lab, out
crops = [(int(a), int(b), int(c), int(d)) for a, b, c, d in (s.split(",") for s in sys.argv[1:])] or [(2690, 2890, 200, 400)]
variants = [("current", {}), ("crossing split", "X")]
fig, axs = plt.subplots(len(crops), len(variants) + 1, figsize=(5.2 * (len(variants) + 1), 5 * len(crops)), squeeze=False)
rng = np.random.default_rng(2)
for r, (x0, x1, y0, y1) in enumerate(crops):
    sl = (slice(y0, y1), slice(x0, x1)); Qc, vc = Q[sl].copy(), valid[sl].copy(); bc = Sub(bg, sl)
    w = Qc - bc.mean; axs[r, 0].imshow(w, cmap="gray", vmin=0, vmax=np.percentile(w, 99.5), origin="lower", extent=(x0, x1, y0, y1)); axs[r, 0].set_title(f"image ({x0}-{x1}, {y0}-{y1})", fontsize=9)
    for j, (nm, kw) in enumerate(variants, 1):
        import xsplit, importlib
        if kw == 'X':
            xsplit.install(GX); lab, out = labels(Qc, vc, bc); xsplit.uninstall(GX)
        else:
            assert not getattr(GX, '_xsplit', False); lab, out = labels(Qc, vc, bc, **kw)
        cols = rng.uniform(0.3, 1, (lab.max() + 1, 3)); cols[0] = 0.07
        axs[r, j].imshow(cols[lab], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
        big = sorted(out, key=lambda c: -c["E"])[:3]
        axs[r, j].set_title(f"{nm}: {len(out)} clusters; top E " + ", ".join(f"{c['E']:.0f}" for c in big) + " keV", fontsize=8)
        print(nm, (x0, y0), len(out), [round(c["E"]) for c in big], flush=True)
fig.tight_layout(); fig.savefig("crossing.png", dpi=85)
