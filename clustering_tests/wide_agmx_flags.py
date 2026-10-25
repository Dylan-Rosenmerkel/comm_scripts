"""Wide test of A_GMX with the quality flags (2026-10-25): every image of wide_test.py; per image-CCD the Fe-55 Ka fit
(front side, sigma_xy < 1) and the 59.5 keV fit without / with the flag cuts, plus the injection test (same objects as
wide_test.py) -> results_wide_agmx_flags/."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("WIDE_OUT", "results_wide_agmx_flags")
import wide_test as W
import adaptive as AD

CUTS = {"all": lambda c: True, "front": lambda c: c.get("sxy", 9) < 1,
        "front, ped<10": lambda c: c.get("sxy", 9) < 1 and not (c.get("ped", 0) >= 10),
        "front, peakfrac>0.95": lambda c: c.get("sxy", 9) < 1 and c.get("peakfrac", 1) > 0.95,
        "front, ped<10, peakfrac>0.95": lambda c: c.get("sxy", 9) < 1 and not (c.get("ped", 0) >= 10) and c.get("peakfrac", 1) > 0.95}


def methods(Q, valid, sw, bg):
    return {"A_GMX flags": lambda: AD.adaptive_cluster(Q, valid, bg, sigma=sw, engine="gmx")}


W.methods = methods


def job(a):
    ds, f, stem, exts = a
    try:
        for ext, arr in W.arrays(f, exts):
            fn = os.path.join(W.OUT, f"{ds.replace('/', '_')}__{stem}__{ext}.json")
            if os.path.exists(fn): continue
            Q = np.nan_to_num(np.asarray(arr, float))
            if Q.shape[0] < 40 or Q.shape[1] < 100: continue
            valid = np.zeros(Q.shape, bool); valid[3:-3, 3:-3] = True
            act = Q[valid]; z = act[np.abs(act) < 0.5]
            sw = max(float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 100 else float(1.4826 * np.median(np.abs(act - np.median(act)))), 0.05)
            try:
                bg = W.AM.background_map(Q, valid)
            except Exception:
                bg = W.simple_bg(Q, valid, sw)
            t = time.time()
            cl = [c for c in AD.adaptive_cluster(Q, valid, bg, sigma=sw, engine="gmx") if np.isfinite(c.get("E", np.nan))]
            out = dict(dataset=ds, kind=W.kind_of(ds), stem=stem, ext=ext, time=time.time() - t, n=len(cl), real={})
            for name, f_ in CUTS.items():
                E = np.array([c["E"] for c in cl if f_(c)])
                out["real"][name] = dict(n=int(E.size), ka=W.CM_fit_ka(E), am59=W.fit59(E))
            out["inject"] = W.inject_eval(Q, valid, sw, np.random.default_rng(W.zlib.crc32((stem + ext).encode())))
            json.dump(out, open(fn, "w"))
            print(ds, stem[-20:], ext, "done", flush=True)
    except Exception:
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(W.OUT, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", 8)), maxtasksperchild=4) as p:
        p.map(job, W.inventory(), chunksize=1)
