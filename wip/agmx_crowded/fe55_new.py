import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = np.linspace(5.0, 7.2, 89); C = 0.5 * (B[1:] + B[:-1]); W = B[1] - B[0]
def model(x, na, mu, sg, rb, b0, b1):
    g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s) * W
    return na * (g(mu, sg) + rb * g(mu * 6.4904 / 5.8988, sg * np.sqrt(6.4904 / 5.8988))) + b0 + b1 * (x - 6)
def cat(ext, v):
    out = {k: [] for k in ("E", "x", "sxy", "blend", "ped", "cls", "dist")}
    for img in (9, 10):
        z = np.load(f"out/{img}_{ext}_{v}.npz", allow_pickle=True)
        d, _ = cKDTree(np.c_[z["y"], z["x"]]).query(np.c_[z["y"], z["x"]], k=2)
        for k in out: out[k].append(d[:, 1] if k == "dist" else z[k])
    return {k: np.concatenate(v) for k, v in out.items()}
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
print(f"{'':16s} {'cut':6s} {'N Ka':>6s} {'Ka keV':>8s} {'sigma eV':>8s} {'Kb/Ka':>12s}")
for j, ext in enumerate((1, 2, 4)):
    for r, (v, nm) in enumerate((("base", "A_GMX current"), ("pkg", "A_GMX new (fixed background)"))):
        z = cat(ext, v); a = axs[r, j]
        rec = (z["blend"] < 0.02) & (z["ped"] < 10) & (z["sxy"] < 1)
        clean = rec & (z["x"] < 2300) & (z["cls"] == "x-ray") & (z["dist"] > 6) & (z["sxy"] > 0.15) & (z["sxy"] < 0.55)
        for s, col, lab, kind in ((np.ones_like(rec), "#9a958f", "all clusters", "step"), (rec, "#d07a2a", "blend<0.02, ped<10, σxy<1", "step"),
                                  (clean, "#2a64c8", "+ x-ray, isolated >6 px, 0.15<σxy<0.55, col<2300", "stepfilled")):
            h = np.histogram(z["E"][s], B)[0]
            a.hist(z["E"][s], B, histtype=kind, color=col, lw=1.4, alpha=0.45 if kind == "stepfilled" else 1, label=f"{lab} ({h.sum()})")
            if s is clean or s is rec:
                try:
                    p, Cv = curve_fit(model, C, h, p0=[h.sum() * .7, 5.9, .08, .13, 1, 0], sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000); er = np.sqrt(np.diag(Cv))
                    tag = "clean" if s is clean else "rec"
                    print(f"{nm:16s} EXT{ext} {tag:6s} {p[0]:6.0f} {p[1]:8.3f} {abs(p[2])*1e3:8.0f} {p[3]:6.3f}±{er[3]:.3f}")
                    if s is clean: a.text(0.98, 0.62, f"clean fit: Kα {p[1]:.3f} keV\nσ {abs(p[2])*1e3:.0f} eV, Kβ/Kα {p[3]:.2f}", transform=a.transAxes, ha="right", fontsize=8.5, bbox=dict(fc="white", ec="0.8"))
                except Exception: pass
        for x, t in ((5.899, "Kα"), (6.490, "Kβ")): a.axvline(x, color="r", lw=0.8)
        a.set_title(f"{nm}, EXT{ext} (both images)", loc="left", weight="bold", fontsize=10)
        if j == 0: a.set_ylabel("clusters / 25 eV")
        if r == 1: a.set_xlabel("cluster energy [keV]")
    axs[0, j].set_ylim(top=max(axs[0, j].get_ylim()[1], axs[1, j].get_ylim()[1])); axs[1, j].set_ylim(axs[0, j].get_ylim())
axs[0, 2].legend(fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("fe55_new.png", dpi=105)
