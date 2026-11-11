"""Power of the skip-resolved transient test (skips.py) on simulated per-skip series (wip/lowdc/bayes).

Null: real charge (constant q = 1..10 e- in every skip) + white noise sigma1 per sample (+ optional 1/f component).
Transients: an excursion that moves the pixel MEAN by a = 1..10 e- (the art-config amplitudes): one bad sample
(spike), a level change from a random skip on (step), charge present only in the first skips (step_first, the
column-0 first-read artefact). Threshold: logBF at a false-veto rate of 1e-3 per real-charge pixel (the 99.9 %
quantile of the null), reported with the detection fraction of each transient kind.

    python skip_study.py [--n 20000]  -> results/v2/skip_study.md
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import skips as SK  # noqa: E402
import metrics as MT  # noqa: E402

SIGMA1 = 3.3        # e- per single sample (L2 ACM 103: s x sqrt(NDCM) ~ 3.3 e- at NDCM 1000-5000, gain 1.3 assumed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="null pixels per setting")
    ap.add_argument("--nt", type=int, default=2000, help="transient pixels per kind and amplitude")
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    rows = []
    for M in (100, 1000, 3000):
        for f1 in (0.0, 0.5, 1.0):
            models = [("white", SK.SkipModel(M, SIGMA1))]
            if f1 > 0:
                models += [("white+drift2", SK.SkipModel(M, SIGMA1, drift_deg=2)),
                           ("1/f cov", SK.SkipModel(M, SIGMA1, cov=SK.cov_1f(M, f1 * SIGMA1)))]
            t0 = time.time()
            X0 = np.array([SK.simulate_pixel(rng, M, SIGMA1, rng.integers(1, 11), "charge", f1_rms=f1 * SIGMA1)
                           for _ in range(a.n)])
            XT = {(k, amp): np.array([SK.simulate_pixel(rng, M, SIGMA1, rng.integers(0, 3), k, amp, f1 * SIGMA1)
                                      for _ in range(a.nt)])
                  for k in ("spike", "step", "step_first") for amp in (1.0, 2.0, 5.0)}
            for mname, mod in models:
                null = mod.logbf(X0)
                thr = float(np.quantile(null, 1 - 1e-3))
                r = dict(M=M, s_mean=round(SIGMA1 / np.sqrt(M), 3), f1_rms=f"{f1:g} sigma1", model=mname,
                         null_median=round(float(np.median(null)), 2), thr_1e3=round(thr, 2),
                         false_veto_at_logBF0=f"{(null > 0).mean():.1e}")
                for (k, amp), X in XT.items():
                    r[f"{k} a{amp:g}"] = round(float((mod.logbf(X) > thr).mean()), 3)
                rows.append(r)
                print(r, f"{time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    md = ("# Skip-resolved transient test: power (skip_study.py)\n\n"
          f"sigma1 = {SIGMA1} e- per sample; null = constant charge 1..10 e- (+ 1/f of rms f1 x sigma1); threshold = "
          "99.9 % quantile of the null logBF (false veto 1e-3 per charge pixel); columns: fraction of transients "
          "(mean moved by a e-) above it.\n\n" + MT.md_table(df) + "\n")
    os.makedirs(os.path.join(HERE, "results", "v2"), exist_ok=True)
    open(os.path.join(HERE, "results", "v2", "skip_study.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
