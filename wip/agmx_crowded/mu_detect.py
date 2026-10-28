"""finder-level diagnosis: for each injected muon, best-matching finder track (share of its charge inside one tube)"""
import sys, numpy as np
sys.path.insert(0, '.')
import inject_mu as IM, harness as HX, bgv3, muons
AM = HX.AM
img, ext = int(sys.argv[1]), int(sys.argv[2]); kw = eval("dict(" + (sys.argv[3] if len(sys.argv) > 3 else "") + ")")
Q, valid = HX.load(img, ext); rng = np.random.default_rng(1000 + 10 * img + ext)
add, truth = IM.sim_muons(rng, Q.shape, 25)
Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid))
tk, mu = muons.find_muons(Qi, valid, bg, **kw)
H, W = Q.shape; own = np.zeros((H, W), np.int32)
for k, m in enumerate(mu, 1): own[m["_pix"][0], m["_pix"][1]] = k
print(f"{len(mu)} finder tracks; lengths", sorted(int(m['length']) for m in mu)[::-1][:40])
sh_all = []
for t in sorted(truth, key=lambda t: t["L"]):
    k = own[t["py"], t["px"]]; tot = t["q"].sum(); sh = np.bincount(k, t["q"], len(mu) + 1)
    b = int(sh[1:].argmax()) + 1 if len(mu) else 0; s = sh[b] / tot if b else 0
    sh_all.append(s)
    extra = f"tube len {mu[b-1]['length']:.0f} occ {mu[b-1]['occ']:.2f} E/mu {mu[b-1]['E']/3.75e-3/tot:.2f}" if b and s > 0.05 else ""
    print(f"L {t['L']:4.0f} th {t['th']:5.1f} sb {t['s_back']:.2f} | best tube share {s:.2f} in-any-tube {1-sh[0]/tot:.2f} {extra}")
print("median best share", np.median(sh_all), " >=0.9:", np.mean(np.array(sh_all) >= .9))
