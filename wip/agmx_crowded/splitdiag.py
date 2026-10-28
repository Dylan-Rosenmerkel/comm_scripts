"""remaining split straight tracks in the new package (image 9 EXT1): loose candidate search, and which of the linking
criteria (min_len 12, w/L < 0.12, angle < 4 deg, offset < 2.5 px, gap < 15 px) each candidate fails"""
import sys, numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, "."); import harness as HX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
img, ext = 9, 1
Q, valid = HX.load(img, ext); lab = np.load(f"out/{img}_{ext}_pkg.npz")["lab"]
R = Q - np.median(Q)
segs = []
for k, sl in enumerate(ndimage.find_objects(lab), 1):
    if sl is None: continue
    m = lab[sl] == k; ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
    if ys.size < 6: continue
    w = np.clip(Q[ys, xs], 0, None) + 1e-6
    c = np.array([np.average(ys, weights=w), np.average(xs, weights=w)]); C = np.cov(np.vstack([ys - c[0], xs - c[1]]), aweights=w)
    ev, V = np.linalg.eigh(C); u = V[:, 1]; t = (ys - c[0]) * u[0] + (xs - c[1]) * u[1]
    L = t.max() - t.min(); wt = np.sqrt(max(ev[0], 0))
    if L >= 8 and wt / L < 0.25: segs.append((k, c + t.min() * u, c + t.max() * u, u, L, wt / L))
ends = np.array([p for s in segs for p in (s[1], s[2])]); tree = cKDTree(ends)
cands = []
for i, (ki, a1, a2, ua, La, ra) in enumerate(segs):
    for e in (a1, a2):
        for jj in tree.query_ball_point(e, 40):
            j = jj // 2
            if j <= i: continue
            kj, b1, b2, ub, Lb, rb_ = segs[j]
            ang = np.degrees(np.arccos(min(abs(ua @ ub), 1)))
            if ang > 8: continue
            off = max(abs((p - a1)[0] * ua[1] - (p - a1)[1] * ua[0]) for p in (b1, b2))
            if off > 4: continue
            gap = min(np.linalg.norm(p - q) for p in (a1, a2) for q in (b1, b2))
            fails = [n for n, bad in (("short<12", min(La, Lb) < 12), ("wide w/L>=0.12", max(ra, rb_) >= 0.12), ("angle>=4", ang >= 4),
                                      ("offset>=2.5", off >= 2.5), ("gap>=15", gap >= 15)) if bad]
            cands.append((ki, kj, ang, off, gap, La, Lb, fails))
from collections import Counter
print(f"{len(segs)} elongated clusters; {len(cands)} loose collinear pairs still split")
cnt = Counter(f for c in cands for f in c[7]); print("failed criteria:", dict(cnt), " | pairs failing only one:", Counter(c[7][0] for c in cands if len(c[7]) == 1))
# show the 6 longest combined
cands.sort(key=lambda c: -(c[5] + c[6]))
fig, axs = plt.subplots(2, 6, figsize=(26, 9)); rng = np.random.default_rng(1)
for col, (ki, kj, ang, off, gap, La, Lb, fails) in enumerate(cands[:6]):
    m = (lab == ki) | (lab == kj); ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 15, 0), min(ys.max() + 15, Q.shape[0]), max(xs.min() - 15, 0), min(xs.max() + 15, Q.shape[1])
    w = Q[y0:y1, x0:x1]; axs[0, col].imshow(w, cmap="gray", vmin=np.percentile(w, 5), vmax=np.percentile(w, 99.5), origin="lower", extent=(x0, x1, y0, y1))
    L = lab[y0:y1, x0:x1]; cols = rng.uniform(0.25, 0.6, (lab.max() + 1, 3)); cols[0] = 0.07; cols[ki] = (1, 0.2, 0.2); cols[kj] = (0.2, 0.6, 1)
    axs[1, col].imshow(cols[L], origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
    axs[0, col].set_title(f"lengths {La:.0f} + {Lb:.0f} px", fontsize=9)
    axs[1, col].set_title(f"angle {ang:.1f}°, offset {off:.1f} px, gap {gap:.0f} px\nfails: {', '.join(fails) or 'none?'}", fontsize=9)
fig.tight_layout(); fig.savefig("splits_remaining.png", dpi=75)
