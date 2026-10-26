"""Constrained Am-241 10-30 keV fit (amfit.Model('Ag'), Lepy line list) of every clustering on the crowded CCD
(runs_29img EXT4, 29 images) from the saved cluster files -> <data>/crowded_fits.json (used by the reports)."""
import glob, json, os, sys
import numpy as np
D = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import am241_paths  # noqa: E402
AN = am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
from amfit import Model, summary
from scan import load as wload
MAIN = ("La", "Lb", "Lg", "g26")


def fit(E):
    r = Model("Ag").fit(E, 0.985); s = summary(r)
    return dict(n=int(E.size), k=float(s["k"][0]), devndf=float(r["devndf"]),
                Z=float(np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in MAIN if s[g][1] > 0))),
                **{g: float(s[g][0] / s[g][1]) for g in ("La", "Lb", "Lg", "g26", "Ag K") if s[g][1] > 0})


def cat(pattern, *keys):
    fs = sorted(glob.glob(pattern))
    return [np.concatenate([np.asarray(np.load(f)[k], float) for f in fs]) for k in keys]


out = {}
cwd = os.getcwd(); os.chdir(AN); d = wload(4); os.chdir(cwd)
out["WADERs (sigma_xy < 0.7)"] = fit(d["Energy"][d["sxy"] < 0.7])
for name, key in (("MFD+G", "MFDG"), ("hybrid v1", "hybrid_v1"), ("hybrid v2", "hybrid_v2"), ("MHC", "MHC")):
    E, S = cat(os.path.join(D, "am241_hybrid", "clusters_occ2", "*.npz"), f"{key}_E", f"{key}_sxy")
    out[f"{name} (sigma_xy < 0.7)"] = fit(E[S < 0.7])
E, S = cat(os.path.join(D, "results_adaptive_v2", "crowded", "*.npz"), "adaptive_E", "adaptive_sxy")
out["adaptive, MHC engine (sigma_xy < 0.7)"] = fit(E[S < 0.7])
E, S, B = cat(os.path.join(D, "verify_gmx", "res", "*.npz"), "E", "sxy", "blend")
out["GMX (sigma_xy < 0.7)"] = fit(E[S < 0.7]); out["GMX (sigma_xy < 0.7, blend < 0.02)"] = fit(E[(S < 0.7) & (B < 0.02)])
E, S, B = cat(os.path.join(D, "results_adaptive_gmx", "crowded", "*.npz"), "adaptive_GMX_E", "adaptive_GMX_sxy", "adaptive_GMX_blend")
out["A_GMX (sigma_xy < 0.7)"] = fit(E[S < 0.7]); out["A_GMX (sigma_xy < 0.7, blend < 0.02)"] = fit(E[(S < 0.7) & (B < 0.02)])
json.dump(out, open(os.path.join(D, "crowded_fits.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:42s} {v['n']:7d}  k {v['k']:.4f}  dev/ndf {v['devndf']:.2f}  Z {v['Z']:5.1f}  " + "  ".join(f"{g} {v.get(g, 0):.1f}" for g in ("La", "Lb", "Lg", "g26", "Ag K")))
