import sys, numpy as np
from scipy import ndimage
sys.path.insert(0, "."); import harness as HX, bgv3
AM = HX.AM
if __name__ == "__main__":
    for ext in (1, 4):
        Q, valid = HX.load(9, ext); bg = AM.background_map(Q, valid)
        lab = np.load(f"out/9_{ext}_pkg.npz")["lab"]; far = valid & ~ndimage.binary_dilation(lab > 0, iterations=4)
        for kw in (dict(), dict(use_mask=True), dict(k_alpha=0.1), dict(use_mask=True, k_alpha=0.1), dict(use_mask=True, k_alpha=0.3)):
            b3, rep = bgv3.regional_v3(Q, valid, bg, **kw)
            r = (Q - b3.mean)[far]
            # by region: glow vs rest (EXT1 cols 600-1200 rows 250-500)
            g = np.zeros_like(far); g[250:500, 600:1200] = True
            print(f"EXT{ext} {str(kw):40s} far mean(Q-bg) {r.mean():+.3f}  median {np.median(r):+.3f}  | glow box {(Q - b3.mean)[far & g].mean():+.3f}  rest {(Q - b3.mean)[far & ~g].mean():+.3f}", flush=True)
