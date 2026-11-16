"""The production threshold clusterer (native_pipeline.threshold_clusters with --thresholds auto) on an image in
electrons, as cluster dicts the benchmarks score like A_GMX's: E [keV], y, x, cls 'threshold', '_pix' = (rows, cols,
electrons). Thresholds (autoconfig_panaSKImg.auto_cluster_thresholds): lam = median over rows of the row dark charge
(background_map), k = smallest k >= 2 with P(Poisson(lam) >= k) <= 0.10, pixel >= k e- (> k - 0.5), seed >= max(9.5, k +
0.5) e-; images without resolved electron peaks (read noise >= 0.5 e-) take the sigma0 rule (10 / 30 x the read
noise), as the pipeline does; energy = sum of the pixels (WADERs: the dark charge under the cluster is not subtracted).

    BENCH_CLUSTERER=threshold python bench_crowded.py ...    /    python bias_probe.py ... (same switch)
"""
import numpy as np
from scipy.stats import poisson

AUTO_ALPHA, AUTO_SEED_E = 0.10, 9.5


def thresholds(lam):
    k = 2
    while poisson.sf(k - 1, lam) > AUTO_ALPHA and k < 100000:
        k += 1 if lam < 50 else max(1, int(np.sqrt(lam) / 4))
    return k - 0.5, max(AUTO_SEED_E, k + 0.5), k


def threshold_cluster(Q, valid, bg, e_per_e):
    import native_pipeline as NP
    import agent_methods as AM
    s0 = float(AM.read_noise(np.where(valid, Q, 0.0), valid))
    if s0 >= 0.5:
        t_pix, t_seed, k, lam = 10 * s0, 30 * s0, None, float("nan")
        return _clusters(NP, Q, valid, t_pix, t_seed, e_per_e), dict(regime="threshold", t_pix=t_pix, t_seed=t_seed,
                                                                    lam=lam, crowded_mode=False, rule="sigma0")
    lam_rows = np.array([np.median(bg.lam[r][valid[r]]) for r in range(Q.shape[0]) if valid[r].any()])
    lam = float(np.median(lam_rows)) if lam_rows.size else 0.0
    t_pix, t_seed, k = thresholds(lam)
    return _clusters(NP, Q, valid, t_pix, t_seed, e_per_e), dict(regime="threshold", t_pix=t_pix, t_seed=t_seed, lam=lam,
                                                                crowded_mode=False, rule="auto")


def _clusters(NP, Q, valid, t_pix, t_seed, e_per_e):
    img = np.where(valid, Q, 0.0)
    rows, cols, off = NP.threshold_clusters(img, valid, t_pix, t_seed)
    out = []
    for a, b in zip(off[:-1], off[1:]):
        ys, xs = rows[a:b], cols[a:b]
        q = img[ys, xs]
        w = np.clip(q, 1e-9, None)
        out.append(dict(E=float(q.sum() * e_per_e), y=float(np.average(ys, weights=w)), x=float(np.average(xs, weights=w)),
                        sxy=np.nan, cls="threshold", regime="threshold", _pix=(ys, xs, q)))
    return out
