"""per injected muon (finder level): best tube; energy error (sum q - muon charge)/muon, misassigned charge sum|q - inj|/muon over
the tube pixels, foreign = sum max(q - inj, 0)/muon, lost-to-others = sum max(inj - q, 0) in tube / muon"""
import sys, numpy as np
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, muons2, muons3
AM = HX.AM
img, ext, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1
Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 * seed + 10 * img + ext)
add, truth = IM.sim_muons(rng, Q.shape, 25)
Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
H, W = Q.shape
R0 = np.where(valid, Qi - bg.mean, 0.0)
ORC = {"oracle o": dict(M=add, O=R0 - add, mode="o"), "oracle m": dict(M=add, O=R0 - add, mode="m"), "oracle mo": dict(M=add, O=R0 - add, mode="mo")}
for name, fm, kw in (("oracle o", muons3.find_muons, dict(em_kw=dict(dirfill=False))), ("oracle m", muons3.find_muons, dict(em_kw=dict(dirfill=False))), ("oracle mo", muons3.find_muons, dict(em_kw=dict(dirfill=False))), ("cap (v2)", muons2.find_muons, {}), ("none", muons2.find_muons, dict(share=False)), ("EM harm", muons3.find_muons, dict(em_kw=dict(dirfill=False))), ("EM dir", muons3.find_muons, {}), ("EM dir+ms", muons3.find_muons, dict(em_kw=dict(mean_scale=True)))):
    muons3._ORACLE = ORC.get(name)
    tk, mu, mq = fm(Qi, valid, bg, **kw)
    own = np.zeros((H, W), np.int32)
    for k, m in enumerate(mu, 1): own[m["_pix"][0], m["_pix"][1]] = k
    rows = []
    for t in truth:
        k = own[t["py"], t["px"]]; tot = t["q"].sum(); sh = np.bincount(k, t["q"], len(mu) + 1)
        b = int(sh[1:].argmax()) + 1 if len(mu) else 0
        if not b or sh[b] < 0.5 * tot: continue
        ty, tx, q = mu[b - 1]["_pix"]; a = add[ty, tx]
        rows.append(((q.sum() - tot) / tot, np.abs(q - a).sum() / tot, np.clip(q - a, 0, None).sum() / tot, np.clip(a - q, 0, None).sum() / tot))
    r = np.array(rows)
    print(f"{name:9s} {len(r)} muons found: E error median {np.median(r[:,0]):+.3f} (|err| < 5 %: {np.mean(np.abs(r[:,0]) < .05):.2f}) | foreign {np.median(r[:,2]):.3f} | muon charge given away {np.median(r[:,3]):.3f} | misassigned {np.median(r[:,1]):.3f}")
