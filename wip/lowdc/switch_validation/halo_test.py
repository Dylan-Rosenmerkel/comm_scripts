"""Halo hand-over test: small events (2-10 e-) 5-40 px from Fe-55 deposits and muon tracks (synth.halo_image).

    python halo_test.py [--procs 2] [--n 40]  -> results/halo_events.csv, halo_fakes.csv, halo_big.csv

Per image: A_GMX alone (cached), the v0 switch (centroid halo 15 px) and the current switch with HALO in
HALOS (distance to the nearest big-deposit pixel), single-image mode. Recorded: per small event found / q / finder vs
its distance d_pix to the nearest big-deposit pixel; unmatched objects < 20 e- with their distance; whether every
A_GMX cluster > 20 e- comes out unchanged.
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

CFG = {"halo_s0.10_l1e-05": dict(s=0.10, lam=1e-5), "halo_s0.15_l0.0001": dict(s=0.15, lam=1e-4)}
HALOS = (0.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)


def _big_same(out, cl, e1):
    a = [c for c in cl if c["E"] / e1 > 20]
    b = [{k: v for k, v in c.items() if k != "finder"} for c in out if c.get("finder", "agmx") == "agmx" and c["E"] / e1 > 20]
    return int(C.same_clusters(a, b))


def run(task):
    name, i = task
    warnings.filterwarnings("ignore")
    import adaptive as AD
    E1 = AD.AM.E_PER_E
    rng = np.random.default_rng(S.seed(name, i))
    Q, valid, truth = S.halo_image(rng, k=i, **CFG[name])
    cl, info, t_ag = C.agmx(name, i, Q, valid)
    W = Q.shape[1]
    bp = np.concatenate([np.c_[t["pix"][0], t["pix"][1]] for t in truth if t["kind"] != "fe"])
    runs = [("agmx", None, None)] + [("v0", "v0", 15.0)] + [(f"cur_h{h:g}", VER, h) for h in HALOS]
    ev, fk, bg = [], [], []
    for tag, ver, h in runs:
        if ver is None:
            out, rec = cl, {}
        else:
            SW = C.load_switch(ver)
            SW.HALO = h
            out, inf = SW.apply(Q, valid, copy.deepcopy(cl), copy.deepcopy(info), E1, "auto", None, None, True)
            rec = inf["fewe"]
        e, f = C.score(out, truth, W, E1, name, i, tag)
        for r, t in zip(e, truth):
            r.update(d=t.get("d", np.nan), d_pix=t.get("d_pix", np.nan), near=t.get("near", ""))
        for r in f:
            r["d_big"] = float(np.min(np.hypot(bp[:, 0] - r["y"], bp[:, 1] - r["x"])))
        ev += e; fk += f
        bg.append(dict(tag=tag, config=name, img=i, big_same=_big_same(out, cl, E1), used=rec.get("used", np.nan),
                       nb_agmx=rec.get("blocks", {}).get("agmx", np.nan), f_c=rec.get("f_c", np.nan),
                       f_c_all=rec.get("f_c_all", np.nan), n_bayes=rec.get("n_bayes", np.nan)))
    return ev, fk, bg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    tasks = [(n, i) for n in CFG for i in range(a.n)]
    from multiprocessing import get_context
    E, F, Bg = [], [], []
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=20) as pool:
        for k, (e, f, b) in enumerate(pool.imap_unordered(run, tasks, chunksize=1)):
            E += e; F += f; Bg += b
            if k % 10 == 0:
                print(f"{k + 1}/{len(tasks)} [{time.time() - t0:.0f} s]", flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    pd.DataFrame(E).to_csv(os.path.join(C.OUT, "halo_events.csv"), index=False)
    pd.DataFrame(F).to_csv(os.path.join(C.OUT, "halo_fakes.csv"), index=False)
    pd.DataFrame(Bg).to_csv(os.path.join(C.OUT, "halo_big.csv"), index=False)


if __name__ == "__main__":
    main()
