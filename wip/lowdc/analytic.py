"""Analytic few-electron detection limits of the skipper pixel null (wip/lowdc): for dark charge lam and read noise s,
the per-pixel rates of a counted pixel (P(n = 1): dark charge + rounding miscounts P(e > 0.5)) and of a 2-e- pixel,
and the smallest window charge N_min(w) whose null probability P(sum of w x w null pixels >= N) is below the
threshold for F false windows per Mpix (one trial per pixel and window size, 4 sizes) - what any finder that
groups electrons by a w x w window can reach at that fake rate. Also the A_GMX statistic T of the two simplest
2-e- events (one pixel with 2 e-; two touching 1-e- pixels), with the true lam and with the lam A_GMX uses.

    python analytic.py [--F 0.08]  -> results/analytic.md
"""
import argparse, os
import numpy as np
import pandas as pd
import metrics as MT
from scipy import stats
from proto_box import pixel_pmf, window_sf, WINDOWS

HERE = os.path.dirname(os.path.abspath(__file__))
POLY = np.array([1, 1, 4, 20, 110, 638, 3832, 23592, 147941, 940982], float)   # gmx.POLYPLETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=float, default=0.08)
    a = ap.parse_args()
    p_thr = a.F / 1e6 / len(WINDOWS)
    rows = []
    for s in (0.05, 0.06, 0.10, 0.15, 0.20, 0.25):
        for lam in (3e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            p = pixel_pmf(lam, s)
            r = dict(s=s, lam=lam, miscount=f"{stats.norm.sf(0.5 / s):.1e}", P1=f"{p[1]:.1e}", P2=f"{p[2]:.1e}",
                     T_2e_1px=f"{-np.log(p[2]):.1f}", T_1e1e=f"{-2 * np.log(p[1]) - np.log(POLY[2]):.1f}",
                     T_3e_3px=f"{-3 * np.log(p[1]) - np.log(POLY[3]):.1f}",
                     ln_Npix_per_F=f"{np.log(1e6 / a.F):.1f}")
            for w in WINDOWS:
                sf = window_sf(lam, s, w)
                r[f"Nmin w{w}"] = int(np.argmax(sf < p_thr))
            rows.append(r)
    md = (f"# Analytic limits (F = {a.F} false windows per Mpix; p_thr = {p_thr:.1e} per pixel and window)\n\n"
          "T_* = -ln P(observed counts | null) - ln N_shapes: A_GMX's group statistic (gmx._group_stats) with the TRUE lam; "
          "a group is kept when T > thr_T (~14-16 for 0.6 Mpix images, see tables.md).\n\n" + MT.md_table(pd.DataFrame(rows)) + "\n")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", "analytic.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
