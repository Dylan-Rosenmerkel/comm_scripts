#!/usr/bin/env python3
"""Smoke-detector foil check (note Sec. 5.6, "Source construction"): are there gold L x rays?

The Am-241 source comes from an ionisation smoke detector. Such foils are commonly americium oxide in a gold matrix
on a silver backing with a thin gold cover; if this one is, gold L x rays (Au La 9.71 keV, Lb1/Lb2 11.44/11.58 keV)
could be in the spectrum next to the suspected silver K lines.

Local fits (EXT4, sigma_xy < 0.7, gain-corrected): a cubic background plus Gaussians at k * E_true with the width
model of the main fit (sigma^2 = s1 * E), sizes free. The region sits on the steep Compton edge, where a global fit
with gold lines did not converge reliably, hence the local fits. A null result is not conclusive: low energies are
strongly suppressed in these data (Np La comes out at about a quarter of its tabulated size relative to Lb).

Run (after extract_clusters.py):   python gold_check.py [--run-dir DIR]
"""
import argparse

import numpy as np
from scipy import optimize

from amfit import Model
from common import CUT, add_path_args, load_clusters, paths

K, S1 = 0.984, 0.0063     # energy scale and width term of the main fit (note Table 5)


def local(E, lo, hi, lines, bw=0.1):
    """Sizes [counts] and uncertainties of Gaussians at K * lines on a cubic background, fitted in [lo, hi] keV."""
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); x = 0.5 * (ed[1:] + ed[:-1]); xm = 0.5 * (lo + hi)

    def mu(q):
        y = q[0] + q[1] * (x - xm) + q[2] * (x - xm) ** 2 + q[3] * (x - xm) ** 3
        for a, e0 in zip(q[4:], lines):
            sg = np.sqrt(S1 * K * e0); y = y + a * bw / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - K * e0) / sg) ** 2)
        return np.clip(y, 1e-9, None)
    nll = lambda q: np.sum(mu(q) - h * np.log(mu(q)))
    b = [(None, None)] * 4 + [(0, None)] * len(lines)
    q = optimize.minimize(nll, [np.median(h), 0, 0, 0] + [1.0] * len(lines), method="Powell", bounds=b).x
    e = np.sqrt(np.clip(np.diag(Model._cov(nll, q)), 0, None))
    return q[4:], e[4:]


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p = paths(ap.parse_args())
    d = load_clusters(p.work, 4)
    E = d["Energy"][d["sxy"] < CUT]
    for nm, lo, hi, lines in (("Au La 9.71", 8.3, 11.0, [9.71]), ("Au Lb 11.44/11.58", 10.4, 12.6, [11.44, 11.58])):
        a, e = local(E, lo, hi, lines)
        print(f"{nm:18s}: " + "  ".join(f"{v:.0f} ± {u:.0f}" for v, u in zip(a, e)) + " counts")


if __name__ == "__main__":
    main()
