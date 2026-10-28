import os, sys, numpy as np
sys.path.insert(0, ".")
import harness as HX, regress as RG, bgv3, muons3, inject_mu as IM
AM = HX.AM
def stats(name, mus):
    a = np.array([[m["cont"]["f_low"], m["cont"]["f_low2"], m["cont"]["cv"], m["length"]] for m in mus])
    if not len(a): print(name, "none"); return
    print(f"{name:28s} n {len(a):4d} | f_low(<0.3 med) pct 10/50/90 {np.percentile(a[:,0],[10,50,90]).round(3)} | f_low2(<0.5) {np.percentile(a[:,1],[10,50,90]).round(3)} | cv {np.percentile(a[:,2],[10,50,90]).round(2)}")
# injected truth on L2
Q, valid = HX.load(9, 4); rng = np.random.default_rng(1094); add, truth = IM.sim_muons(rng, Q.shape, 25); Qi = np.where(valid, Q + add, 0.0)
bg, _ = bgv3.regional_v3(Qi, valid, AM.background_map(Qi, valid)); tk, mu, mq = muons3.find_muons(Qi, valid, bg)
H, W = Q.shape; own = np.zeros((H, W), np.int32)
for k, m in enumerate(mu, 1): own[m["_pix"][0], m["_pix"][1]] = k
inj = set()
for t in truth:
    k = own[t["py"], t["px"]]; sh = np.bincount(k, t["q"], len(mu) + 1); sh[0] = 0
    if sh.max() > 0.5 * t["q"].sum(): inj.add(int(sh.argmax()) - 1)
stats("L2 injected muons", [mu[i] for i in inj]); stats("L2 other (real) tracks", [m for i, m in enumerate(mu) if i not in inj])
sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/clustering_tests")); import am241_29img_recluster as RC
J = {j[4]: j for j in RG.jobs()}
for t in [t for t in J if t.startswith("Am241") and "_28_" in t][:1] + [t for t in J if "DM07_DM09" in t and t.endswith("_ext1")][:1]:
    kind, run, stem, ext, tag = J[t]
    Q, valid = RC.load(stem, ext) if kind == "am" else RG.load_run(run, stem, ext)[:2]
    bg, _ = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid)); tk, mu, mq = muons3.find_muons(Q, valid, bg)
    stats(tag[:28], mu)
