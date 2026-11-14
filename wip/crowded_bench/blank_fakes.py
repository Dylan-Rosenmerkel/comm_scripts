"""fake rate of A_GMX (fewe auto, single-image mode) on blank binned images of 24 unbinned Mpix (12000 x 2000 px):
    python blank_fakes.py LAM_PER_BIN N_IMAGES BIN     budget 0.08 per unbinned Mpix (audit 2026-11-14)"""
import os, sys
import numpy as np
R = os.environ.get("PROBE_REPO", os.path.expanduser("~/Documents/Software/comm_scripts"))
for p in ("clustering", "engine"):
    sys.path.insert(0, os.path.join(R, "superWADERs", p))
import adaptive as AD, agent_methods as AM
lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
b = int(sys.argv[3]) if len(sys.argv) > 3 else 100
shape = (12000 // b, 2000 // b)
n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(5)
nf, sizes = 0, []
for i in range(n_img):
    Q = rng.poisson(lam, shape) + rng.normal(0, 0.12, shape)
    v = np.ones(Q.shape, bool)
    bg = AM.background_map(Q, v)
    cl = AD.adaptive_cluster(Q, v, bg, engine="gmx", crowded="auto", fewe="auto", binning=(b, b))
    small = [c for c in cl if c["E"] / AM.E_PER_E < 20]
    nf += len(small); sizes += [round(c["E"] / AM.E_PER_E, 1) for c in small]
print(f"{b}x{b} lam {lam}/bin: {nf} clusters < 20 e- in {n_img} blank images = {nf / (24.0 * n_img):.3f} per unbinned Mpix "
      f"(budget 0.08); charges {sorted(sizes)[:30]}")
