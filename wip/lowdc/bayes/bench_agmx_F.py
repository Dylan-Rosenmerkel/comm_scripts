"""A_GMX at other nominal fake rates (count_fakes) on the benchmark images, for the ROC comparison (wip/lowdc/bayes).

bench_agmx.py ran A_GMX only at its default count_fakes = 0.08 per Mpix; its results/*.csv therefore give one operating
point per configuration. Here the same pipeline (background_map -> adaptive_cluster(engine='gmx', return_pixels=True))
is rerun on the SAME images (same seeds, ../bench_agmx.configs) with count_fakes = F, recording only what the ROC needs:
per injected deposit found / measured charge, and the unmatched clusters (fakes).

    python bench_agmx_F.py --F 0.01 --only base_s0.10_l1e-05,... [--procs 2]
       -> results/agmx_F<F>_events.csv, agmx_F<F>_fakes.csv, agmx_F<F>_images.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, sys, time, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOWDC = os.path.dirname(HERE)
CT = os.path.join(LOWDC, "..", "..", "clustering_tests")
for _p in (LOWDC, CT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bench_agmx as B  # noqa: E402

KEY = ["base_s0.06_l1e-05", "base_s0.10_l1e-05", "base_s0.15_l1e-05", "base_s0.15_l0.0001",
       "blank_s0.06_l1e-05", "blank_s0.10_l1e-05", "blank_s0.15_l1e-05", "blank_s0.15_l0.0001",
       "l2_s0.06_l1e-05", "l2_s0.10_l1e-05"]


def run_image(task):
    """one image at count_fakes = F: (image row, event rows, fake rows)"""
    import sim, metrics as MT
    import adaptive as AD
    name, i, par, F = task
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(B._seed(name, i))
    Q, valid, truth, p, rate = sim.simulate(rng, **par)
    Q = np.where(valid, Q, 0.0)
    W = Q.shape[1]
    t0 = time.perf_counter()
    try:
        bg = AD.AM.background_map(Q, valid)
    except Exception:
        bg = None
    cl, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, count_fakes=F)
    cl = [c for c in cl if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
    dt = time.perf_counter() - t0
    own, per = MT.match([dict(y=c["y"], x=c["x"], pix=c["_pix"]) for c in cl], truth, W)
    E1 = AD.AM.E_PER_E
    ev = [dict(config=name, img=i, ev=k, kind=t["kind"], n_e=t["n_e"], sigma=t["sigma"], found=int(bool(per[k])),
               q=sum(cl[j]["E"] for j in per[k]) / E1 if per[k] else 0.0) for k, t in enumerate(truth)]
    fk = [dict(config=name, img=i, near_big=int(own[j] == -2), y=c["y"], x=c["x"], q=c["E"] / E1, npix=len(c["_pix"][0]))
          for j, c in enumerate(cl) if own[j] < 0]
    return dict(config=name, img=i, mpix=valid.sum() / 1e6, t=dt), ev, fk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--F", type=float, required=True)
    ap.add_argument("--only", default=",".join(KEY))
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cf = B.configs()
    tasks = [(n, i, cf[n][2], a.F) for n in a.only.split(",") for i in range(cf[n][1])]
    print(f"{len(tasks)} images at count_fakes = {a.F}", flush=True)
    ri, re_, rf = [], [], []
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool:
        for k, (im, ev, fk) in enumerate(pool.imap_unordered(run_image, tasks, chunksize=1)):
            ri.append(im); re_ += ev; rf += fk
            if k % 50 == 0:
                print(f"{k + 1}/{len(tasks)} {im['config']}  {time.time() - t0:.0f} s", flush=True)
    tag = f"agmx_F{a.F:g}"
    out = os.path.join(HERE, "results")
    pd.DataFrame(ri).to_csv(os.path.join(out, f"{tag}_images.csv"), index=False)
    pd.DataFrame(re_).to_csv(os.path.join(out, f"{tag}_events.csv"), index=False)
    pd.DataFrame(rf).to_csv(os.path.join(out, f"{tag}_fakes.csv"), index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
