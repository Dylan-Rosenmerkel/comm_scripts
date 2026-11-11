"""Shared helpers of the switch validation (wip/lowdc/switch_validation): paths, configs, A_GMX cache, matching.

A_GMX's own output (adaptive_cluster(engine='gmx', fewe='off', return_pixels=True)) is cached per image in the
scratchpad, so that the switch (agmx_switch.apply, a pure post-processing step of that output) can be re-evaluated
after a fix without rerunning A_GMX. check_hook.py verifies that adaptive_cluster(fewe='auto') gives exactly
apply(cached A_GMX output).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOWDC = os.path.dirname(HERE)
BAYES = os.path.join(LOWDC, "bayes")
CT = os.path.join(LOWDC, "..", "..", "clustering_tests")
for _p in (HERE, LOWDC, BAYES, CT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SCRATCH = os.environ.get("SWV_SCRATCH", "/private/tmp/claude-501/-Users-dylanrosenmerkel/"
                         "511be665-1540-4ae6-86c3-c4f8e8738785/scratchpad/switch_validation")
CACHE = os.path.join(SCRATCH, "agmx_cache")
# the finder modules (noise, finder, pipeline, background) are being changed by another agent: the benchmark runs on a
# snapshot of them (SCRATCH/finder_snapshot, md5.txt there) unless SWV_LIVE=1; agmx_switch.py itself is the live file
SNAP = os.path.join(SCRATCH, "finder_snapshot")
if os.environ.get("SWV_LIVE", "0") != "1" and os.path.isdir(SNAP):
    sys.path.insert(0, SNAP)
OUT = os.path.join(HERE, "results")

import bench_agmx as B   # noqa: E402

# configs added for the validation (not in bench_agmx): the 'both' regime between the two f_c cuts
EXTRA = {"base_s0.16_l0.0001": ("grid", 15, dict(lam=1e-4, s=0.16)),
         "base_s0.17_l1e-05": ("grid", 15, dict(lam=1e-5, s=0.17)),
         # realistic event density (2 events per 0.6-Mpix image instead of 30): per-image vs run fake budget
         "sparse2_s0.15_l1e-05": ("grid", 40, dict(lam=1e-5, s=0.15, n_events=2)),
         "sparse2_s0.15_l0.0001": ("grid", 40, dict(lam=1e-4, s=0.15, n_events=2))}


def configs():
    c = dict(B.configs())
    c.update(EXTRA)
    return c


def groups():
    """run groups: base_x pooled with blank_x (as bayes/bench_bayes.py), every other config alone; incl. noisy ones"""
    cf = configs()
    g = {}
    for name in cf:
        if name.startswith("blank_"):
            continue
        g[name] = [name]
        b = "blank_" + name[len("base_"):] if name.startswith("base_") else None
        if b and b in cf:
            g[name].append(b)
    return g


def load_switch(version):
    """'v0' = the switch as implemented on 2026-11-10, 'v1', ... = frozen copies of the fixed switch in this folder;
    'cur' = the live bayes/agmx_switch.py"""
    if version == "cur":
        import agmx_switch as SW
        return SW
    fn = os.path.join(HERE, f"agmx_switch_{version}.py")
    spec = importlib.util.spec_from_file_location(f"agmx_switch_{version}", fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def simulate(name, i, par):
    import sim
    rng = np.random.default_rng(B._seed(name, i))
    Q, valid, truth, p, rate = sim.simulate(rng, **par)
    return np.where(valid, Q, 0.0), valid, truth, p, rate


def agmx(name, i, Q, valid, cache=True):
    """A_GMX alone (fewe='off'), cached: (clusters with _pix, info without the occupancy map, runtime [s])"""
    import adaptive as AD
    fn = os.path.join(CACHE, f"{name}__{i}.pkl")
    if cache and os.path.exists(fn):
        with open(fn, "rb") as fh:
            return pickle.load(fh)
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    try:
        bg = AD.AM.background_map(Q, valid)
    except Exception:
        bg = None
    cl, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="off")
    t = time.perf_counter() - t0
    info = {k: v for k, v in info.items() if k != "occ"}
    res = (cl, info, t)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        tmp = fn + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, fn)
    return res


def pix_of(c):
    p = c.get("_pix")
    return (np.asarray(p[0], int), np.asarray(p[1], int)) if p is not None else (np.zeros(0, int), np.zeros(0, int))


def score(cl, truth, W, e_per_e, config, img, tag):
    """match clusters to the truth: event rows (found, n_obj, q [e-], finder of the best object) and fake rows"""
    import metrics as MT
    cl = [c for c in cl if np.isfinite(c.get("E", np.nan))]
    objs = [dict(y=c["y"], x=c["x"], pix=pix_of(c)) for c in cl]
    own, per = MT.match(objs, truth, W)
    ev = []
    for k, t in enumerate(truth):
        js = per[k]
        q = sum(cl[j]["E"] for j in js) / e_per_e if js else 0.0
        fnd = sorted({cl[j].get("finder", "agmx") for j in js})
        ev.append(dict(tag=tag, config=config, img=img, ev=k, kind=t["kind"], n_e=t["n_e"], sigma=round(float(t["sigma"]), 4),
                       y=round(float(t["y"]), 2), x=round(float(t["x"]), 2), found=int(bool(js)), n_obj=len(js),
                       q=round(q, 3), finder="+".join(fnd)))
    fk = []
    for j, c in enumerate(cl):
        if own[j] < 0:
            fk.append(dict(tag=tag, config=config, img=img, near_big=int(own[j] == -2), y=round(c["y"], 2),
                           x=round(c["x"], 2), q=round(c["E"] / e_per_e, 3), npix=len(pix_of(c)[0]), cls=c.get("cls", ""),
                           finder=c.get("finder", "agmx"), p_ev=c.get("p_ev", np.nan)))
    return ev, fk


def same_clusters(a, b):
    """exact equality of two cluster lists (keys, scalars, pixel arrays)"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y):
            return False
        for k in x:
            u, v = x[k], y[k]
            if isinstance(u, tuple):
                if len(u) != len(v) or not all(np.array_equal(np.asarray(p), np.asarray(q), equal_nan=True)
                                               if np.asarray(p).dtype.kind == "f" else np.array_equal(p, q) for p, q in zip(u, v)):
                    return False
            elif isinstance(u, float) or isinstance(v, float):
                if not ((u == v) or (np.isnan(u) and np.isnan(v))):
                    return False
            elif isinstance(u, np.ndarray):
                if not np.array_equal(u, v):
                    return False
            elif u != v:
                return False
    return True
