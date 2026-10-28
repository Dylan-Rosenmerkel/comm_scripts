import os, sys, numpy as np
sys.path.insert(0, ".")
import harness as HX, regress as RG, bgv3, muons3
from scipy import ndimage as nd
AM = HX.AM
sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/clustering_tests")); import am241_29img_recluster as RC
J = {j[4]: j for j in RG.jobs()}
picks = [t for t in J if t.startswith("L2_")] + [t for t in J if t.startswith("Am241")][::7] + \
        [t for t in J if not t.startswith(("L2_", "Am241")) and "High_Temp" not in t and t.endswith("ext1")][::6]
for t in picks:
    kind, run, stem, ext, tag = J[t]
    Q, valid = RC.load(stem, ext) if kind == "am" else RG.load_run(run, stem, ext)[:2]
    bg, _ = bgv3.regional_v3(Q, valid, AM.background_map(Q, valid))
    R = np.where(valid, Q - bg.mean, 0); Rs = nd.gaussian_filter(R, 1)
    nz = np.sqrt(np.maximum(bg.lam, 0)) / (2 * np.sqrt(np.pi)); core = (Rs > np.maximum(60, 5 * nz)) & valid
    dbg = []; tk, mu, mq = muons3.find_muons(Q, valid, bg, dbg=dbg)
    print(f"{tag[:58]:58s} core fill {core[valid].mean():.4f}  tracks {len(mu):4d}  tube frac {tk[valid].mean():.4f}", flush=True)
