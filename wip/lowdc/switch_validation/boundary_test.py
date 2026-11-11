"""Block-boundary test: small events (2-10 e-) straddling 100 x 100 block edges / corners where the neighbouring
blocks have different regimes (read noise checkerboard, synth.boundary_image).

    python boundary_test.py [--procs 2] [--n 30]  -> results/boundary_events.csv, boundary_fakes.csv, boundary_images.csv

Runs per image: A_GMX alone (cached), the v0 switch and the current one (single-image mode). Double counting = an
event matched by more than one object from different finders (agmx + bayes); loss = found by A_GMX alone but by
neither finder after the switch.
"""
import os
import argparse
import copy
import time
import warnings

import numpy as np
import pandas as pd

import common as C
import synth as S

VER = os.environ.get("SWV_VERSION", "v2")      # frozen switch version evaluated as 'cur'

CFG = {"bnd_checker_s0.10_0.20": dict(s_lo=0.10, s_hi=0.20, pattern="checker"),
       "bnd_checker_s0.06_0.25": dict(s_lo=0.06, s_hi=0.25, pattern="checker"),
       "bnd_uniform_s0.10": dict(s_lo=0.10, pattern="uniform")}


def run(task):
    name, i = task
    warnings.filterwarnings("ignore")
    import adaptive as AD
    E1 = AD.AM.E_PER_E
    rng = np.random.default_rng(S.seed(name, i))
    Q, valid, truth, s_map = S.boundary_image(rng, **CFG[name])
    cl, info, t_ag = C.agmx(name, i, Q, valid)
    W = Q.shape[1]
    ev, fk, im = [], [], []
    for tag in ("agmx", "v0", "cur"):
        if tag == "agmx":
            out, rec = cl, {}
        else:
            SW = C.load_switch(VER if tag == "cur" else tag)
            out, inf = SW.apply(Q, valid, copy.deepcopy(cl), copy.deepcopy(info), E1, "auto", None, None, True)
            rec = inf["fewe"]
        e, f = C.score(out, truth, W, E1, name, i, tag)
        for r, t in zip(e, truth):
            yy, xx = int(np.clip(np.floor(t["y"]), 0, Q.shape[0] - 1)), int(np.clip(np.floor(t["x"]), 0, W - 1))
            r.update(edge=t["edge"], s_here=float(s_map[yy, xx]))
        ev += e; fk += f
        b = rec.get("blocks", {})
        im.append(dict(tag=tag, config=name, img=i, used=rec.get("used", np.nan), nb_bayes=b.get("bayes", np.nan),
                       nb_both=b.get("both", np.nan), nb_agmx=b.get("agmx", np.nan), n_bayes=rec.get("n_bayes", np.nan),
                       n_replaced=rec.get("n_agmx_replaced", np.nan)))
    return ev, fk, im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    tasks = [(n, i) for n in CFG for i in range(a.n)]
    from multiprocessing import get_context
    E, F, I = [], [], []
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=20) as pool:
        for k, (e, f, im) in enumerate(pool.imap_unordered(run, tasks, chunksize=1)):
            E += e; F += f; I += im
            if k % 10 == 0:
                print(f"{k + 1}/{len(tasks)} [{time.time() - t0:.0f} s]", flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    pd.DataFrame(E).to_csv(os.path.join(C.OUT, "boundary_events.csv"), index=False)
    pd.DataFrame(F).to_csv(os.path.join(C.OUT, "boundary_fakes.csv"), index=False)
    pd.DataFrame(I).to_csv(os.path.join(C.OUT, "boundary_images.csv"), index=False)


if __name__ == "__main__":
    main()
