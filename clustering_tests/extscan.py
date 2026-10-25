import tracks_test as TT, methods2 as M2, numpy as np, json, sys
from multiprocessing import Pool
AM=TT.AM
def meth(Q, valid, sw, bg):
    cl = AM.mf_detect(Q, valid, bg)[0]
    d = {}
    for e in (10, 12, 14, 18):
        d[f"v2 ext{e}"] = (lambda e: lambda: M2.mfdg_hybrid(Q, valid, bg, cl=[dict(c) for c in cl], f_big=0.2, ext_big=e))(e)
        d[f"v1 ext{e}"] = (lambda e: lambda: M2.mfdg_hybrid(Q, valid, bg, cl=[dict(c) for c in cl], ext_big=e))(e)
    return d
TT.methods = meth
def job(a):
    stem, ext = a
    Q, valid, sw = TT.AT.load(stem, ext); out = []
    for seed in (1, 2):
        for m, v in TT.inject_eval(Q, valid, sw, np.random.default_rng(seed + 100 * ext)).items():
            out += [dict(r, method=m, ext=ext) for r in v]
    return out
if __name__ == "__main__":
    with Pool(6) as p: res = sum(p.map(job, [(s, e) for s in TT.AT.IMGS[:3] for e in (1, 3)]), [])
    json.dump(res, open("extscan.json", "w"))
