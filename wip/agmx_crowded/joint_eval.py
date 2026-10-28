"""muon charge attribution with A_GMX as the model of the other tracks (scratch):
  1. muons3 finder: tubes, muon model m (A(t) x pixel-Gaussian), EM share (harmonic other) as the start
  2. GMX (gmx.general_cluster) on Q - m: its clouds model everything that is not the muon model
  3. clouds whose model lies >= f_in inside a tube are the muon's own fluctuations (knots) -> muon; the rest = other
  4. muon share q = R (m + knots) / (m + knots + other + eps)  [n_outer times, A(t) refitted from q]"""
import sys, time, numpy as np
from scipy import ndimage as nd
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, muons3
AM, GX = HX.AM, HX.GX

def joint_share(Q, valid, bg, mu, f_in=0.8, n_outer=1, dbg=None):
    H, W = Q.shape
    R = np.where(valid, Q - bg.mean, 0.0)
    Mm = np.zeros((H, W)); tid = np.zeros((H, W), np.int32)
    for k, m in enumerate(mu, 1):
        ty, tx, q = m["_pix"]; Mm[ty, tx] += np.clip(m["_m"], 0, None); tid[ty, tx] = k
    for it in range(n_outer):
        t_ = time.time()
        cl, info = GX.general_cluster(Q - Mm, valid, bg, return_info=True)
        mix = info["mix"]
        I, K, G, DY, DX = GX.evaluate(mix, (H, W), valid)
        n = len(mix)
        intube = tid.ravel()[I] > 0
        lab = np.asarray(info["lab"]); nl = int(lab.max()) + 1
        # knot or other per A_GMX cluster (clouds grouped by merge_graph / arbitration): a crossing track's cluster
        # reaches outside the tube, a knot's does not
        fin_c = np.bincount(lab[K], G * intube, nl) / np.maximum(np.bincount(lab[K], G, nl), 1e-12)
        knot = fin_c[lab] >= f_in
        Mk = np.bincount(I, G * knot[K], H * W).reshape(H, W)
        Mo = np.bincount(I, G * ~knot[K], H * W).reshape(H, W)
        eps = 1e-3
        out = []
        for k, m in enumerate(mu, 1):
            ty, tx, _ = m["_pix"]
            num = np.clip(m["_m"], 0, None) + Mk[ty, tx]
            q = R[ty, tx] * num / (num + Mo[ty, tx] + eps)
            out.append(q)
        if dbg is not None: dbg.append(dict(it=it, clouds=n, knots=int(knot.sum()), sec=time.time() - t_))
    return out

if __name__ == "__main__":
    img, ext, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1
    Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 * seed + 10 * img + ext)
    add, truth = IM.sim_muons(rng, Q.shape, 25)
    Qi = np.where(valid, Q + add, 0.0)
    bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
    H, W = Q.shape
    tk, mu, mq = muons3.find_muons(Qi, valid, bg, em_kw=dict(dirfill=False))
    dbg = []
    qs = {"EM harm": [m["_pix"][2] for m in mu]}
    for f in (0.6, 0.8, 0.95):
        qs[f"GMX f_in {f}"] = joint_share(Qi, valid, bg, mu, f_in=f, dbg=dbg)
    print(dbg)
    own = np.zeros((H, W), np.int32)
    for k, m in enumerate(mu, 1): own[m["_pix"][0], m["_pix"][1]] = k
    for name, Ql in qs.items():
        rows = []
        for t in truth:
            k = own[t["py"], t["px"]]; tot = t["q"].sum(); sh = np.bincount(k, t["q"], len(mu) + 1)
            b = int(sh[1:].argmax()) + 1 if len(mu) else 0
            if not b or sh[b] < 0.5 * tot: continue
            ty, tx = mu[b - 1]["_pix"][:2]; q = Ql[b - 1]; a = add[ty, tx]
            rows.append(((q.sum() - tot) / tot, np.abs(q - a).sum() / tot, np.clip(q - a, 0, None).sum() / tot, np.clip(a - q, 0, None).sum() / tot))
        r = np.array(rows)
        print(f"{name:13s} {len(r)} muons: E error median {np.median(r[:,0]):+.3f} (|err| < 5 %: {np.mean(np.abs(r[:,0]) < .05):.2f}) | foreign {np.median(r[:,2]):.3f} | given away {np.median(r[:,3]):.3f} | misassigned {np.median(r[:,1]):.3f}", flush=True)
