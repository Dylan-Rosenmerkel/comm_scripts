"""proto_box (../proto_box.py, window counts with an exact null) at other nominal fake rates, on the same images, for the
ROC comparison (wip/lowdc/bayes). ../results/proto_box_*_F0.08.csv hold the F = 0.08 point; this writes the others
into bayes/results/ (nothing is written outside this folder).

    python bench_proto_F.py --F 0.01,1,10
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import proto_box as P, bench_agmx as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", default="0.01,1,10")
    a = ap.parse_args()
    for F in [float(x) for x in a.F.split(",")]:
        ri, re_, rf = [], [], []
        t0 = time.time()
        for name, (st, n, par) in B.configs().items():
            if name.startswith(("art", "busy")) or any(t in name for t in ("_s0.20", "_s0.25", "_l0.01")):
                continue
            for i in range(n):
                im, ev, fk = P.run(name, i, par, F)
                ri.append(im); re_ += ev; rf += fk
        tag = f"proto_F{F:g}"
        pd.DataFrame(ri).to_csv(os.path.join(HERE, "results", f"{tag}_images.csv"), index=False)
        pd.DataFrame(re_).to_csv(os.path.join(HERE, "results", f"{tag}_events.csv"), index=False)
        pd.DataFrame(rf).to_csv(os.path.join(HERE, "results", f"{tag}_fakes.csv"), index=False)
        print(f"F={F}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
