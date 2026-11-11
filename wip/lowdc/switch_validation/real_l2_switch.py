"""The switch on the real L2 ACM 103 noise-test images through adaptive_cluster(engine='gmx', fewe='auto').

Data (read only): ~/Documents/comm_scripts_data/ndcm_images_2_L2_103/*.fits. Per extension (CCD): ADU -> e- with
a per-row median pedestal (valid pixels) and gain 1.3 ADU/e- (EXT1-3; EXT4 unknown: 1.3 and 0.82), column 0 masked.
The 100 x 500 images are grouped by NDCM (two per setting) and run in run-pooled mode (prepare_run over the pair);
the five single images of other shapes (NDCM 3000) run alone in single-image mode.

    python real_l2_switch.py  -> results/real_l2_images.csv (decision per image / CCD), results/real_l2_objects.csv
                                 (every Bayes event and every A_GMX cluster < 20 e- of fewe='off' and 'auto')
"""
import os
import glob
import re
import time
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits

import common as C

DATA = os.path.expanduser("~/Documents/comm_scripts_data/ndcm_images_2_L2_103")
GAINS = {1: (1.3,), 2: (1.3,), 3: (1.3,), 4: (1.3, 0.82)}


def to_e(Y, valid, gain):
    """per-row median pedestal of the valid pixels, ADU -> e-"""
    ped = np.array([np.median(r[v]) for r, v in zip(Y, valid)])
    return np.where(valid, (Y - ped[:, None]) / gain, 0.0)


def main():
    warnings.filterwarnings("ignore")
    import adaptive as AD
    import agmx_switch as SW
    E1 = AD.AM.E_PER_E
    files = sorted(glob.glob(os.path.join(DATA, "avg_img_*.fits")))
    im_rows, ob_rows = [], []
    for ext, gains in GAINS.items():
        for g in gains:
            groups = {}
            for f in files:
                m = re.search(r"img_(\d+)x(\d+)x(\d+)_", os.path.basename(f))
                H, W, nd = map(int, m.groups())
                key = ("pair", nd) if (H, W) == (100, 500) else ("single", os.path.basename(f))
                groups.setdefault(key, []).append((f, nd))
            for key, fl in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
                ims = []
                for f, nd in fl:
                    with fits.open(f) as h:
                        Y = h[ext].data.astype(float)
                    valid = np.ones(Y.shape, bool); valid[:, 0] = False
                    ims.append((to_e(Y, valid, g), valid, os.path.basename(f), nd))
                t0 = time.perf_counter()
                if key[0] == "pair":
                    run = SW.prepare_run([(Q, v) for Q, v, _, _ in ims], full=True)
                else:
                    run = dict(lam_maps=[None] * len(ims), R_N=None, threshold=None, nm=None)
                t_prep = time.perf_counter() - t0
                for k, (Q, valid, fn, nd) in enumerate(ims):
                    ta = time.perf_counter()
                    bg = AD.AM.background_map(Q, valid)
                    off, inf0 = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="off")
                    t1 = time.perf_counter()
                    on, inf = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="auto",
                                                  fewe_lam=run["lam_maps"][k], fewe_RN=run["R_N"],
                                                  fewe_threshold=run["threshold"], fewe_nm=run["nm"])
                    t2 = time.perf_counter()
                    rec = inf.get("fewe", {})
                    b = rec.get("blocks", {})
                    small_off = [c for c in off if np.isfinite(c["E"]) and c["E"] / E1 <= 20]
                    big_off = [c for c in off if np.isfinite(c["E"]) and c["E"] / E1 > 20]
                    nby = [c for c in on if c.get("finder") == "bayes"]
                    kept_small = [c for c in on if c.get("finder") == "agmx" and np.isfinite(c["E"]) and c["E"] / E1 <= 20]
                    im_rows.append(dict(ext=ext, gain=g, mode=key[0], file=fn, ndcm=nd, shape=f"{Q.shape[0]}x{Q.shape[1]}",
                                        agmx_regime=inf0.get("regime"), agmx_noise=inf0.get("noise"), agmx_lam=inf0.get("lam"),
                                        crowded=int(bool(inf0.get("crowded_mode", False))),
                                        used=int(bool(rec.get("used"))), reason=rec.get("reason", ""), s=rec.get("s", np.nan),
                                        f_c=rec.get("f_c", np.nan), f_c_all=rec.get("f_c_all", np.nan),
                                        nb_bayes=b.get("bayes", 0), nb_both=b.get("both", 0), nb_agmx=b.get("agmx", 0),
                                        lam=rec.get("lam", np.nan), R=rec.get("R", np.nan), threshold=rec.get("threshold", np.nan),
                                        only3=int(bool(rec.get("only_N_ge_3", False))), n_off=len(off), n_off_small=len(small_off),
                                        n_off_big=len(big_off), n_replaced=rec.get("n_agmx_replaced", 0),
                                        n_kept_small=len(kept_small), n_bayes=len(nby), t_agmx=t1 - ta,
                                        t_auto=t2 - t1, t_prep=t_prep / len(ims)))
                    for tag, cs in (("agmx_off_small", small_off), ("bayes", nby)):
                        for c in cs:
                            ob_rows.append(dict(ext=ext, gain=g, file=fn, ndcm=nd, tag=tag, y=round(c["y"], 2), x=round(c["x"], 2),
                                                q=round(c["E"] / E1, 3), npix=len(C.pix_of(c)[0]), cls=c.get("cls"),
                                                p_ev=c.get("p_ev", np.nan), N_map=c.get("N_map", np.nan),
                                                logJ1=c.get("logJ1", np.nan)))
                print(f"EXT{ext} g{g} {key}: {[r['used'] for r in im_rows[-len(ims):]]}", flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    pd.DataFrame(im_rows).to_csv(os.path.join(C.OUT, "real_l2_images.csv"), index=False)
    pd.DataFrame(ob_rows).to_csv(os.path.join(C.OUT, "real_l2_objects.csv"), index=False)


if __name__ == "__main__":
    main()
