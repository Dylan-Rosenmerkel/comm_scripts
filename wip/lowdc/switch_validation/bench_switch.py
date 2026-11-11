"""Benchmark of the A_GMX / Bayes switch (agmx_switch.apply) on the bench_agmx images (same configs and seeds) plus
the 'both'-regime configs of common.EXTRA.

    python bench_switch.py --phase agmx  [--procs 2]                    # fill the A_GMX cache (fewe='off')
    python bench_switch.py --phase switch --version v0|v1|... [--procs 2]  # switch on every group, both modes

Per group (base_x + blank_x pooled, as bayes/bench_bayes.py; every other config alone) and image:
  agmx    A_GMX alone (the cached fewe='off' output)
  single  apply(..., 'auto') with the single-image dark charge and event rates
  pooled  apply(..., 'auto', lam_map=lam_maps[k], R_N=R_N) with prepare_run over the images of the group
-> results/<version>/<group>__{events,fakes,images}.csv (column tag = agmx | single | pooled; agmx rows only for version cur)
"""
import os
import argparse
import copy
import time
import warnings

import numpy as np
import pandas as pd

import common as C


ON_TEST = {"base_s0.17_l1e-05": 15, "base_s0.20_l1e-05": 3}     # group -> images (forced Bayes at s 0.20: ~240 s / image)


def run_agmx(task):
    name, i, par = task
    warnings.filterwarnings("ignore")
    Q, valid, truth, p, rate = C.simulate(name, i, par)
    _, info, t = C.agmx(name, i, Q, valid)
    return name, i, t


def _img_row(tag, name, i, valid, info, rec, t_sw, t_prep, ident):
    r = dict(tag=tag, config=name, img=i, mpix=valid.sum() / 1e6, agmx_regime=info.get("regime"),
             crowded=int(bool(info.get("crowded_mode", False))), agmx_lam=info.get("lam", np.nan),
             agmx_noise=info.get("noise", np.nan), t_switch=t_sw, t_prep=t_prep, identical=ident)
    if rec is not None:
        b = rec.get("blocks", {})
        r.update(used=int(bool(rec.get("used"))), reason=rec.get("reason", ""), s=rec.get("s", np.nan), f_c=rec.get("f_c", np.nan),
                 f_c_bg=rec.get("f_c_bg", np.nan), nb_bayes=b.get("bayes", 0), nb_both=b.get("both", 0), nb_agmx=b.get("agmx", 0),
                 lam=rec.get("lam", np.nan), R=rec.get("R", np.nan), threshold=rec.get("threshold", np.nan),
                 n_replaced=rec.get("n_agmx_replaced", 0), n_bayes=rec.get("n_bayes", 0), n_big=rec.get("n_big", 0),
                 only3=int(bool(rec.get("only_N_ge_3", False))), pooled=int(bool(rec.get("pooled_lam", False))),
                 area_bayes=rec.get("area_bayes", np.nan))
    return r


def run_group(task):
    gname, cnames, version, modes = task
    warnings.filterwarnings("ignore")
    import adaptive as AD
    SW = C.load_switch(version)
    E1 = AD.AM.E_PER_E
    cf = C.configs()
    imgs = []
    for cn in cnames:
        st, n, par = cf[cn]
        for i in range(n):
            Q, valid, truth, p, rate = C.simulate(cn, i, par)
            cl, info, t_ag = C.agmx(cn, i, Q, valid)
            imgs.append((cn, i, Q, valid, truth, cl, info, t_ag))
    ev, fk, im = [], [], []
    W = imgs[0][2].shape[1]
    # A_GMX alone
    if "agmx" in modes:
        for cn, i, Q, valid, truth, cl, info, t_ag in imgs:
            e, f = C.score(cl, truth, W, E1, cn, i, "agmx")
            ev += e; fk += f
            im.append(dict(_img_row("agmx", cn, i, valid, info, None, t_ag, 0.0, 1)))
    # single-image mode
    for cn, i, Q, valid, truth, cl, info, t_ag in (imgs if "single" in modes else []):
        t0 = time.perf_counter()
        out, inf = SW.apply(Q, valid, copy.deepcopy(cl), copy.deepcopy(info), E1, "auto", None, None, True)
        t = time.perf_counter() - t0
        rec = inf["fewe"]
        e, f = C.score(out, truth, W, E1, cn, i, "single")
        ev += e; fk += f
        im.append(_img_row("single", cn, i, valid, info, rec, t, 0.0, int(C.same_clusters(out, cl))))
    # forced Bayes ('on') where the rule says A_GMX, to test the upper f_c cut
    if "on" in modes and gname in ON_TEST:
        for cn, i, Q, valid, truth, cl, info, t_ag in imgs[:ON_TEST[gname]]:
            t0 = time.perf_counter()
            out, inf = SW.apply(Q, valid, copy.deepcopy(cl), copy.deepcopy(info), E1, "on", None, None, True)
            t = time.perf_counter() - t0
            e, f = C.score(out, truth, W, E1, cn, i, "on")
            ev += e; fk += f
            im.append(_img_row("on", cn, i, valid, info, inf["fewe"], t, 0.0, int(C.same_clusters(out, cl))))
    # run-pooled mode
    if "pooled" not in modes:
        return gname, ev, fk, im
    t0 = time.perf_counter()
    t_run, nm = None, None
    try:
        if "full" in SW.prepare_run.__code__.co_varnames:
            run = SW.prepare_run([(Q, valid) for _, _, Q, valid, *_ in imgs], full=True)
            lam_maps, R_N, t_run, nm = run["lam_maps"], run["R_N"], run["threshold"], run["nm"]
        elif hasattr(SW, "run_threshold"):
            lam_maps, R_N, t_run = SW.prepare_run([(Q, valid) for _, _, Q, valid, *_ in imgs], with_threshold=True)
        else:
            lam_maps, R_N = SW.prepare_run([(Q, valid) for _, _, Q, valid, *_ in imgs])
        err = ""
    except Exception as ex:             # noqa: BLE001 - recorded, the group falls back to single-image estimates
        lam_maps, R_N, err = [None] * len(imgs), None, repr(ex)
    t_prep = (time.perf_counter() - t0) / len(imgs)
    # 'pooled': run-pooled lam, R_N and (when the switch has it) the run's P(event) cut; 'pooled_it': the same with
    # each image's own fake budget
    runs = [("pooled", t_run)] + ([("pooled_it", None)] if t_run is not None and "pooled_it" in modes else [])
    for tag, thr in runs:
        for k, (cn, i, Q, valid, truth, cl, info, t_ag) in enumerate(imgs):
            t0 = time.perf_counter()
            kw = dict(threshold=thr) if thr is not None else {}
            if nm is not None:
                kw["nm"] = nm
            out, inf = SW.apply(Q, valid, copy.deepcopy(cl), copy.deepcopy(info), E1, "auto", lam_maps[k], R_N, True, **kw)
            t = time.perf_counter() - t0
            rec = inf["fewe"]
            e, f = C.score(out, truth, W, E1, cn, i, tag)
            ev += e; fk += f
            r = _img_row(tag, cn, i, valid, info, rec, t, t_prep, int(C.same_clusters(out, cl)))
            r.update(prep_error=err, t_run=t_run if t_run is not None else np.nan)
            im.append(r)
    return gname, ev, fk, im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="switch", choices=["agmx", "switch"])
    ap.add_argument("--version", default="cur")
    ap.add_argument("--groups", default="")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--modes", default="agmx,single,on,pooled", help="runs per group: agmx (A_GMX alone), single, on "
                    "(forced Bayes, noisy groups only), pooled, pooled_it (run statistics, image budget)")
    a = ap.parse_args()
    from multiprocessing import get_context
    cf = C.configs()
    if a.phase == "agmx":
        tasks = [(n, i, par) for n, (st, k, par) in cf.items() for i in range(k)
                 if not os.path.exists(os.path.join(C.CACHE, f"{n}__{i}.pkl"))]
        tasks.sort(key=lambda t: -t[2].get("H", 480))
        print(f"{len(tasks)} A_GMX images to run", flush=True)
        t0 = time.time()
        with get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool:
            for k, (n, i, t) in enumerate(pool.imap_unordered(run_agmx, tasks, chunksize=1)):
                if k % 25 == 0:
                    print(f"{k + 1}/{len(tasks)} {n} #{i} {t:.1f} s  [{time.time() - t0:.0f} s]", flush=True)
        return
    out = os.path.join(C.OUT, a.version)
    os.makedirs(out, exist_ok=True)
    G = C.groups()
    names = [g for g in a.groups.split(",") if g] or list(G)
    names = [n for n in names if not os.path.exists(os.path.join(out, f"{n}__images.csv"))]
    names.sort(key=lambda g: -sum(cf[c][1] * cf[c][2].get("H", 480) for c in G[g]))
    tasks = [(g, G[g], a.version, a.modes) for g in names]
    print(f"{len(tasks)} groups to run", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for gname, ev, fk, im in pool.imap_unordered(run_group, tasks, chunksize=1):
            # one file per group and table (rows of the three tags differ in columns); images last = group done
            for key, rows in (("events", ev), ("fakes", fk), ("images", im)):
                if rows or key == "images":
                    pd.DataFrame(rows).to_csv(os.path.join(out, f"{gname}__{key}.csv"), index=False)
            print(f"{gname}: {len(im)} rows  [{time.time() - t0:.0f} s]", flush=True)


def load(version):
    """concatenated tables of one version: dict(events, fakes, images)"""
    import glob
    d = os.path.join(C.OUT, version)
    res = {}
    for key in ("events", "fakes", "images"):
        fs = sorted(glob.glob(os.path.join(d, f"*__{key}.csv")))
        res[key] = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return res


if __name__ == "__main__":
    main()
