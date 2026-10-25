"""Few-electron test (as mfdg_tracks/lowE_test.py) with MHC added."""
import os, sys, json
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
sys.path.insert(0, HERE); import lowE_test as LE
import mhc as MH
AM, CM, AT, M2 = LE.AM, LE.CM, LE.AT, LE.M2
OUT = os.path.join(HERE, "results_lowE")


def methods(Q, valid, sw, bg):
    cache = []
    def mf():
        if not cache: cache.append(AM.mf_detect(Q, valid, bg, nsig=3.0)[0])
        return [dict(c) for c in cache[0]]
    return {
        "WADERs": lambda: CM.wader(Q, sw, 10, 30)[0],
        "MFD+G 5sig": lambda: AM.mf_detect(Q, valid, bg)[0],
        "MFD+G 3sig count": lambda: M2.counting_energy(Q, bg, mf()),
        "MFD+G hybrid v1": lambda: M2.mfdg_hybrid(Q, valid, bg, ext_big=14),
        "MHC": lambda: MH.mhc(Q, valid, bg),
        "DM-style >=1e-": lambda: LE.dm_style(Q, valid),
    }


LE.methods = methods


def job(a):
    kind, key, seed = a
    fn = os.path.join(OUT, f"{kind}_{key}_{seed}.json")
    if os.path.exists(fn): return
    rng = np.random.default_rng(seed)
    if kind == "real":
        Q, valid, sw = AT.load(key, 1)
        out = dict(lam=float(np.median(AM.background_map(Q, valid).lam[valid])), inject=LE.evaluate(Q, valid, sw, rng))
    else:
        lam = float(key)
        valid = np.zeros((500, 3500), bool); valid[5:-5, 5:-5] = True
        out = dict(lam=lam, inject=LE.evaluate(np.where(valid, LE.blank(lam, rng), 0), valid, LE.SNOISE, rng),
                   fakes=LE.fakes(np.where(valid, LE.blank(lam, np.random.default_rng(seed + 999)), 0), valid, LE.SNOISE))
    json.dump(out, open(fn, "w"))
    print(kind, key, seed, "done", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [("real", s, i) for i, s in enumerate(AT.IMGS[:4])] + [("sim", str(l), s) for l in LE.LAMS for s in (1, 2)]
    with Pool(int(os.environ.get("NPROC", 4))) as p:
        p.map(job, jobs, chunksize=1)
