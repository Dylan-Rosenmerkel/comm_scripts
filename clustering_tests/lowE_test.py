"""(3) Low-energy, few-pixel events (1-20 e- in 1-5 pixels, the dark-matter search regime) on low-background images.
 - efficiency and charge accuracy: events injected into real AMT EXT1 images (0.125 e- noise, 0.001-0.07 e-/pix)
   and into simulated blank images (Poisson dark charge lam + Gaussian 0.15 e- noise) at several lam
 - fake rate: clusters found on simulated blanks with nothing injected (per Mpix, by reconstructed charge)
Methods: WADERs default (10 / 30 sigma), B auto (pixel >= k e-, seed 9.5 e-), MFD+G (nsig 5, 4, 3), and the
low-threshold 'DM-style' clustering used by skipper-CCD dark-matter searches (8-connected pixels with >= 1 e-,
charge = sum of the rounded pixel values)."""
import json, os, sys
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_test as AT
import methods2 as M2
AM, CM = AT.AM, AT.CM
NE = 1 / AM.E_PER_E
EIGHT = np.ones((3, 3), bool)
NES = [1, 2, 3, 4, 5, 7, 10, 15, 20]
SIGS = [0.2, 0.4, 0.7]
LAMS = [0.001, 0.01, 0.05, 0.2]
SNOISE = 0.15


def dm_style(Q, valid):
    n = np.where(valid, np.round(Q), 0).clip(0)
    lab, k = ndimage.label(n >= 1, structure=EIGHT)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i; q = n[sl][m]; ys, xs = np.nonzero(m)
        s = q.sum()
        out.append(dict(E=float(s / NE), y=float((q * (ys + sl[0].start)).sum() / s), x=float((q * (xs + sl[1].start)).sum() / s)))
    return out


def methods(Q, valid, sw, bg):
    k = max(int(AM.poisson_tail_k(float(np.median(bg.lam[valid])), 0.10)[0]), 2)
    return {
        "WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],
        "B auto": lambda: CM.topo(Q, 0.0, 9.5, k - 0.5, 0),
        "MFD+G 5sig": lambda: AM.mf_detect(Q, valid, bg)[0],
        "MFD+G 4sig": lambda: AM.mf_detect(Q, valid, bg, nsig=4.0)[0],
        "MFD+G 3sig": lambda: AM.mf_detect(Q, valid, bg, nsig=3.0)[0],
        "MFD+G 5sig count": lambda: M2.counting_energy(Q, bg, AM.mf_detect(Q, valid, bg)[0]),
        "MFD+G 3sig count": lambda: M2.counting_energy(Q, bg, AM.mf_detect(Q, valid, bg, nsig=3.0)[0]),
        "DM-style >=1e-": lambda: dm_style(Q, valid),
    }


def inject(Q, valid, rng, n_ev=800, sep=20):
    busy = ndimage.binary_dilation(Q > 1.5, iterations=6) | ~ndimage.binary_erosion(valid, iterations=8)
    ys, xs = np.nonzero(~busy)
    pos = []
    occ = np.zeros(Q.shape, bool)
    for i in rng.permutation(ys.size):
        y, x = ys[i], xs[i]
        if occ[y, x]: continue
        pos.append((y + rng.uniform(-.5, .5), x + rng.uniform(-.5, .5)))
        occ[max(y - sep, 0):y + sep, max(x - sep, 0):x + sep] = True
        if len(pos) >= n_ev: break
    Qi = Q.copy(); ev = []
    for j, (y, x) in enumerate(pos):
        ne = NES[j % len(NES)]; s = SIGS[(j // len(NES)) % len(SIGS)]
        iy = np.round(y + rng.normal(0, s, ne)).astype(int); ix = np.round(x + rng.normal(0, s, ne)).astype(int)
        np.add.at(Qi, (iy, ix), 1.0)
        ev.append((y, x, ne, s, len(set(zip(iy, ix)))))
    return Qi, ev


def evaluate(Q, valid, sw, rng):
    Qi, ev = inject(Q, valid, rng)
    bg = AM.background_map(Qi, valid)
    out = {}
    for name, f in methods(Qi, valid, sw, bg).items():
        cl = [c for c in f() if np.isfinite(c.get("E", np.nan))]
        cy = np.array([c["y"] for c in cl]); cx = np.array([c["x"] for c in cl]); cE = np.array([c["E"] for c in cl]) * NE
        rows = []
        for (y, x, ne, s, npx) in ev:
            d = np.hypot(cy - y, cx - x) if cy.size else np.array([np.inf])
            j = int(np.argmin(d))
            rows.append([ne, s, npx, bool(d[j] < 1.5 + s), float(cE[j]) if d[j] < 1.5 + s else np.nan])
        out[name] = rows
    return out


def fakes(Q, valid, sw):
    bg = AM.background_map(Q, valid)
    return {name: [float(c["E"] * NE) for c in f() if np.isfinite(c.get("E", np.nan))] for name, f in methods(Q, valid, sw, bg).items()}


def blank(lam, rng, shape=(500, 3500)):
    return rng.poisson(lam, shape) + rng.normal(0, SNOISE, shape)


def job(a):
    kind, key, seed = a
    fn = os.path.join(HERE, "results_lowE2", f"{kind}_{key}_{seed}.json")
    if os.path.exists(fn): return
    rng = np.random.default_rng(seed)
    if kind == "real":
        Q, valid, sw = AT.load(key, 1)
        out = dict(lam=float(np.median(AM.background_map(Q, valid).lam[valid])), inject=evaluate(Q, valid, sw, rng))
    else:
        lam = float(key)
        valid = np.zeros((500, 3500), bool); valid[5:-5, 5:-5] = True
        out = dict(lam=lam, inject=evaluate(np.where(valid, blank(lam, rng), 0), valid, SNOISE, rng),
                   fakes=fakes(np.where(valid, blank(lam, np.random.default_rng(seed + 999)), 0), valid, SNOISE))
    json.dump(out, open(fn, "w"))
    print(kind, key, seed, "done", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results_lowE2"), exist_ok=True)
    jobs = [("real", s, i) for i, s in enumerate(AT.IMGS[:4])] + [("sim", str(l), s) for l in LAMS for s in (1, 2)]
    with Pool(4) as p:
        p.map(job, jobs, chunksize=1)
