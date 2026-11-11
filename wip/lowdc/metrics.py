"""Matching of found clusters / candidates to the simulated truth, and the benchmark figures of merit (wip/lowdc).

Matching (per image): a found object (cluster or candidate with pixels) belongs to the truth deposit with which it
shares the most TRUE electrons (pixel overlap weighted by the deposit's electron count in the shared pixels); objects
sharing no pixel fall back to the nearest deposit centre within r_match px (a 1-e- event can be found one pixel off when
read noise moves the count). An event is found when at least one object belongs to it; split = more than one object;
the measured charge is the sum over its objects. Unmatched objects are fakes; those within r_big of a large deposit
(Fe-55, muon) are counted apart ('near_big': CTI / halo / diffusion-tail pieces, not dark-charge fakes).
"""
import numpy as np


def _pixdict(truth, W):
    """pixel key -> (event index, electrons of that event in the pixel); overlapping deposits: the larger count wins"""
    d = {}
    for i, t in enumerate(truth):
        ys, xs, cnt = t["pix"]
        for k, n in zip(ys * W + xs, cnt):
            if k not in d or d[k][1] < n:
                d[int(k)] = (i, int(n))
    return d


def match(objs, truth, W, r_match=2.0, r_big=40.0):
    """objs: list of dicts with y, x and pix = (rows, cols); returns (owner per object: event index or -1 (fake) or
    -2 (near a large deposit)), and per event the list of object indices"""
    pd = _pixdict(truth, W)
    cy = np.array([t["y"] for t in truth]); cx = np.array([t["x"] for t in truth])
    big = np.array([t["kind"] != "fe" for t in truth], bool)
    owner = np.full(len(objs), -1, int)
    per_ev = [[] for _ in truth]
    for j, o in enumerate(objs):
        score = {}
        ys, xs = o["pix"][0], o["pix"][1]
        for k in (np.asarray(ys, int) * W + np.asarray(xs, int)):
            if int(k) in pd:
                i, n = pd[int(k)]
                score[i] = score.get(i, 0) + n
        if score:
            owner[j] = max(score, key=score.get)
        elif len(truth):
            d = np.hypot(cy - o["y"], cx - o["x"])
            i = int(np.argmin(d))
            if d[i] <= r_match:
                owner[j] = i
            elif big.any() and np.min(np.where(big, d, np.inf)) <= r_big:
                owner[j] = -2
        if owner[j] >= 0:
            per_ev[owner[j]].append(j)
    return owner, per_ev


def binom_err(k, n):
    """efficiency and its error (Wilson-like: never 0 at k = 0 or n)"""
    n = np.maximum(n, 1)
    p = k / n
    return p, np.sqrt(np.maximum(p * (1 - p), 1.0 / n) / n)


def rate_err(k, mpix):
    """rate per Mpix and its Poisson error (1.0 / mpix when k = 0: the one-sided ~63 % bound would be 1.0)"""
    return k / mpix, np.sqrt(max(k, 1)) / mpix


def md_table(df):
    """a pandas DataFrame as a GitHub markdown table (no tabulate dependency)"""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)
