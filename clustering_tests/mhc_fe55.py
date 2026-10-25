"""Fe-55 (DM09 / DM10, EXT1 / EXT4, images 0-1): WADERs, MFD+G, MFD+G hybrid v1 / v2, MHC.
Injection of 1580-e- clusters (sigma 0.3-1.8 px) and pairs, and the real Ka peak (front side), with the framework
of explore_agent/compare.py."""
import os, sys, json
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
sys.path.insert(0, HERE); import compare as CP
import methods2 as M2
import mhc as MH
AM, CM = CP.AM, CP.CM
CP.RES = os.path.join(HERE, "results_fe55")


def make_methods(mod, img, lam_row, sw, valid, near=None):
    cache = {}
    def bgf(Q):
        if id(Q) not in cache:
            cache.clear(); cache[id(Q)] = AM.background_map(Q, valid)
        return cache[id(Q)]
    return {
        "WADERs": lambda Q: CM.wader(Q, sw, 10, 30)[0],
        "MFD+G": lambda Q: AM.mf_detect(Q, valid, bgf(Q))[0],
        "MFD+G hybrid v1": lambda Q: M2.mfdg_hybrid(Q, valid, bgf(Q), ext_big=14),
        "MFD+G hybrid v2": lambda Q: M2.mfdg_hybrid(Q, valid, bgf(Q), f_big=0.2, ext_big=14),
        "MHC": lambda Q: MH.mhc(Q, valid, bgf(Q)),
    }


CP.make_methods = make_methods


def job(a):
    kind, mod, ext, img = a
    fn = os.path.join(CP.RES, f"{kind}_{mod}_EXT{ext}_img{img}.json")
    if os.path.exists(fn): return
    try:
        (CP.run_inject if kind == "inject" else CP.run_real)(mod, ext, img)
    except Exception:
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    os.makedirs(CP.RES, exist_ok=True)
    jobs = [(k, m, e, i) for k in ("real", "inject") for m in ("DM09", "DM10") for e in (1, 4) for i in (0, 1)]
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        p.map(job, jobs, chunksize=1)
