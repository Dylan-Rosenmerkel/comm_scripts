"""Oracle ceiling of few-electron detection on the low-background binned benchmark (bench_lowbg.py): the best
efficiency any clusterer can reach at a given false-alarm rate, per binning and dark current. It tells how much a
bin-aware A_GMX / Bayesian finder can still gain over the methods measured on the benchmark.

    python ceiling_lowbg.py [--F 0.08] [--n-mc 20000] [--compare DIR ...]   -> ceiling.md (next to this file's results)

Events as in bench_lowbg: N electrons (2-10), depth uniform in 0-675 um, DAMIC diffusion law, centre uniform; binned
b_y x b_x. Dark charge per bin lam_b = dc x b_y x b_x (Poisson), read noise neglected (0.12 e-: counts are exact).
Two oracle tests, each with everything true; the ceiling is the better of the two for every event:
  pattern    Neyman-Pearson on the charge pattern (occupied bins and counts): select patterns by P(pattern | event) /
             P(pattern | dark charge) until the chance rate of dark-charge patterns reaches F per Mpix of unbinned
             area (generalises wip/lowdc/bayes/ceiling.py 'pairs' to b_y != b_x, the diffusion law and exact Poisson).
             Best where lam_b << 1.
  footprint  the event's footprint (its m bins) known: total charge in them against Poisson(m lam_b); threshold k so
             that the bins per Mpix x P(Poisson(m lam_b) >= k) <= F. Best where lam_b is ~1 (every bin has charge).
F = 0.08 per Mpix is the Bayesian finder's operating point; the measured runs had 0-0.5 fakes per Mpix.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.stats import poisson

A_UM2, B_UM, THICK, PIX = 285.0, 8.2e-4, 675.0, 15.0
BINS = ["1x1", "1x10", "10x1", "10x10", "1x100", "100x1", "100x100"]
DCS = [1e-6, 1e-5, 1e-4]
CATS = {"2 e-": [2], "3 e-": [3], "4-5 e-": [4, 5], "6-10 e-": list(range(6, 11))}


def patterns(N, by, bx, n_mc, rng):
    """list of event patterns (tuple of (dy, dx, count), first occupied bin first) and their frequencies"""
    z = rng.uniform(0, THICK, n_mc)
    sig = np.sqrt(-A_UM2 * np.log(1 - B_UM * z)) / PIX
    cy, cx = rng.uniform(0, by, n_mc), rng.uniform(0, bx, n_mc)
    py = np.floor((cy[:, None] + rng.normal(size=(n_mc, N)) * sig[:, None]) / by).astype(np.int64)
    px = np.floor((cx[:, None] + rng.normal(size=(n_mc, N)) * sig[:, None]) / bx).astype(np.int64)
    out = {}
    for a, b in zip(py, px):
        k, c = np.unique(a * 100003 + b, return_counts=True)
        y, x = np.divmod(k, 100003)
        key = tuple(zip((y - y[0]).tolist(), (x - x[0]).tolist(), c.tolist()))
        out[key] = out.get(key, 0) + 1
    return {k: v / n_mc for k, v in out.items()}


def eff_pattern(probs, lam_b, bins_per_mpix, F):
    """Neyman-Pearson efficiency with exact counts (exact Poisson for the occupied bins of the pattern)"""
    rows = []
    for p, pe in probs.items():
        nr = math.prod(poisson.pmf(n, lam_b) for _, _, n in p) * bins_per_mpix
        rows.append((pe, nr))
    rows.sort(key=lambda t: -t[0] / max(t[1], 1e-300))
    eff = fk = 0.0
    for pe, nr in rows:
        if fk + nr <= F:
            eff += pe
            fk += nr
        else:
            eff += pe * (F - fk) / nr
            break
    return eff


def eff_footprint(probs, N, lam_b, bins_per_mpix, F):
    """efficiency of the known-footprint charge sum test, averaged over the event patterns"""
    eff = 0.0
    shapes = {}                                                # distinct footprints of m bins: each is a trial
    for p in probs:
        shapes.setdefault(len(p), set()).add(tuple((dy, dx) for dy, dx, _ in p))
    for p, pe in probs.items():
        m = len(p)
        mu = m * lam_b
        trials = bins_per_mpix * len(shapes[m])
        k = 1
        while trials * poisson.sf(k - 1, mu) > F:             # P(Poisson(mu) >= k) = sf(k - 1)
            k += 1
        eff += pe * poisson.sf(k - N - 1, mu)                  # P(N + Poisson(mu) >= k)
    return eff


def ceiling(F, n_mc, seed=7):
    rng = np.random.default_rng(seed)
    res = {}
    for b in BINS:
        by, bx = (int(v) for v in b.split("x"))
        P = {N: patterns(N, by, bx, n_mc, rng) for N in range(2, 11)}
        for dc in DCS:
            lam_b, bpm = dc * by * bx, 1e6 / (by * bx)
            e = {N: max(eff_pattern(P[N], lam_b, bpm, F), eff_footprint(P[N], N, lam_b, bpm, F)) for N in P}
            res[f"{b}_dc{dc:g}"] = {c: float(np.mean([e[N] for N in Ns])) for c, Ns in CATS.items()}
        print("ceiling", b, flush=True)
    return res


def measured(d):
    """{setting: {category: efficiency}} of a bench_lowbg report.json"""
    r = json.load(open(os.path.join(d, "report.json")))
    key = {"2 e-": "fewe_2", "3 e-": "fewe_3", "4-5 e-": "fewe_4-5", "6-10 e-": "fewe_6-10"}
    return {s: {c: (v["eff"].get(k) or [None])[0] for c, k in key.items()} for s, v in r.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--F", type=float, default=0.08)
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--compare", nargs="*", default=[], help="bench_lowbg output folders (label=path)")
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    C = ceiling(a.F, a.n_mc)
    M = {lab: measured(p) for lab, p in (x.split("=", 1) for x in a.compare)}
    os.makedirs(a.o, exist_ok=True)
    json.dump(dict(F=a.F, ceiling=C, measured=M), open(os.path.join(a.o, "ceiling.json"), "w"), indent=1)
    labs = ["ceiling"] + list(M)
    L = [f"# Few-electron detection: oracle ceiling vs measured (F = {a.F} fakes / Mpix)", "",
         "Each cell: " + " / ".join(labs), "", "| setting | " + " | ".join(CATS) + " |", "|---" * (len(CATS) + 1) + "|"]
    f = lambda v: "-" if v is None else f"{v:.2f}"
    for s in C:
        cells = [" / ".join(f((C if lab == "ceiling" else M[lab]).get(s, {}).get(c)) for lab in labs) for c in CATS]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    open(os.path.join(a.o, "ceiling.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
