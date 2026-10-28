import sys, numpy as np
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, muons
AM = HX.AM
img, ext = int(sys.argv[1]), int(sys.argv[2])
Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 + 10 * img + ext)
add, truth = IM.sim_muons(rng, Q.shape, 25)
Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
dbg = []; tk, mu = muons.find_muons(Qi, valid, bg, core_e=60, max_gap=6, dbg=dbg)
segs = dbg[0][1]; ev = dbg[1:]
print("hough segments", len(segs))
for t in sorted(truth, key=lambda t: t["L"]):
    # truth line endpoints from charge-weighted PCA
    y, x, q = t["py"], t["px"], t["q"]; c = np.array([np.average(y, weights=q), np.average(x, weights=q)])
    u = np.array([np.sin(np.radians(t["th"])), np.cos(np.radians(t["th"]))])
    def match(s):
        (x0, y0), (x1, y1) = s; d = [abs((py - c[0]) * u[1] - (px - c[1]) * u[0]) for px, py in ((x0, y0), (x1, y1))]
        a = np.degrees(np.arccos(min(1, abs(((y1 - y0) * u[0] + (x1 - x0) * u[1]) / np.hypot(y1 - y0, x1 - x0)))))
        tt = [((py - c[0]) * u[0] + (px - c[1]) * u[1]) for px, py in ((x0, y0), (x1, y1))]
        return max(d) < 6 and a < 8 and max(abs(v) for v in tt) < t["L"] / 2 + 10
    ms = [s for s in segs if match(s)]; whys = [e["why"] + (f"(L{e.get('L',0):.0f},occ{e.get('occ',0):.2f},rms{e.get('rms',0):.1f})" if e["why"] != "taken" else "") for e in ev if match(e["seg"])]
    tl = sum(np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]) for s in ms)
    print(f"L {t['L']:4.0f} th {t['th']:5.1f}: {len(ms):2d} hough segs (sum len {tl:4.0f}) -> {', '.join(whys[:6])}")
print("--- segments in bbox of first truth muon", truth[0]["L"])
t = truth[0]; y0,y1,x0,x1 = t["py"].min(), t["py"].max(), t["px"].min(), t["px"].max()
for s in segs:
    (a,b),(c,d) = s
    if x0-5 <= a <= x1+5 and y0-5 <= b <= y1+5: print(s, np.degrees(np.arctan2(d-b, c-a)))
for e in ev:
    (a,b),(c,d) = e["seg"]
    if x0-5 <= a <= x1+5 and y0-5 <= b <= y1+5: print(e)
