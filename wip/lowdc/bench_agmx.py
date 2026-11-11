"""Benchmark of the current A_GMX few-electron finding in the ultra-low dark-current regime (wip/lowdc).

Per simulated image (sim.py): background_map -> adaptive_cluster(engine='gmx') -> fe_candidates.candidates (as
src/agmx_root.process does when the noise is < 0.35 e-), matched to the truth (metrics.py). Recorded per image
(results/images.csv: background estimates vs truth, count threshold, runtimes, fakes), per injected deposit
(results/events.csv) and per unmatched object (results/fakes.csv).

    python bench_agmx.py --set grid|blanks|l2|all [--procs 2] [--scale 1.0]

Sets: grid = event images over (lam, s, tail / hot column / artifacts / big deposits); blanks = event-free images
for the fake rates; l2 = the L2 ACM 103 geometry (100 x 500, no overscan). Every image has a fixed seed
(config name, image index), finished images are skipped on a rerun (resume). At most --procs worker processes,
each single-threaded (machine-load rule: <= 2 heavy python processes).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, sys, time, warnings, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "..", "..", "clustering_tests")
for _p in (HERE, CT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = os.path.join(HERE, "results")
TAIL = dict(tail="mix", f_tail=0.005, k_tail=3.0)        # 0.5 % of pixels with 3 x the read noise (both signs)
HOTCOL = dict(hot_cols=(640,), hot_col_rate=0.02)        # one faint hot column, 0.02 e-/pix excess
# L2 artefacts (coordinator note 2026-11-10): first column ~2000 / NDCM ADU at ~1.3 ADU/e- (NDCM 1000 -> ~1.5 e-),
# rare large excursions of both signs, a little serial-register charge, a few hot pixels, row pedestal residual
ART = dict(first_col=1.5, first_col_jit=0.3, f_out=1e-5, out_lo=1.0, out_hi=10.0, lam_ser=2e-5, n_hot_pix=3, ped_row=0.01)


def configs():
    """name -> (set, n_images, sim parameters)"""
    c = {}
    for lam in (1e-5, 1e-4, 1e-3, 1e-2):
        c[f"base_s0.15_l{lam:g}"] = ("grid", 30, dict(lam=lam, s=0.15))
        c[f"tail_s0.15_l{lam:g}"] = ("grid", 15, dict(lam=lam, s=0.15, **TAIL))
        c[f"hotcol_s0.15_l{lam:g}"] = ("grid", 15, dict(lam=lam, s=0.15, **HOTCOL))
    for s in (0.05, 0.06, 0.10, 0.20, 0.25):
        for lam in (1e-5, 1e-4):
            c[f"base_s{s:.2f}_l{lam:g}"] = ("grid", 15, dict(lam=lam, s=s))
    for s in (0.06, 0.15):
        c[f"base_s{s:.2f}_l3e-06"] = ("grid", 15, dict(lam=3e-6, s=s))
    for s in (0.10, 0.20):
        c[f"tail_s{s:.2f}_l0.0001"] = ("grid", 15, dict(lam=1e-4, s=s, **TAIL))
        c[f"hotcol_s{s:.2f}_l0.0001"] = ("grid", 15, dict(lam=1e-4, s=s, **HOTCOL))
    c["art_s0.10_l1e-05"] = ("grid", 15, dict(lam=1e-5, s=0.10, **ART))
    c["busy_s0.15_l0.0001"] = ("grid", 15, dict(lam=1e-4, s=0.15, n_fe55=3, n_muon=1))
    c["grad_s0.15_l0.0001"] = ("grid", 15, dict(lam=1e-4, s=0.15, lam_grad=2.0))
    for s, lam in ((0.15, 1e-5), (0.15, 1e-4), (0.15, 1e-3), (0.15, 1e-2), (0.06, 1e-5), (0.10, 1e-5)):
        c[f"blank_s{s:.2f}_l{lam:g}"] = ("blanks", 60, dict(lam=lam, s=s, n_events=0))
    for s, lam in ((0.06, 1e-5), (0.10, 1e-5), (0.15, 1e-4)):
        c[f"l2_s{s:.2f}_l{lam:g}"] = ("l2", 200, dict(lam=lam, s=s, H=100, W=500, n_os=0, n_events=4, sep=25))
    return c


def _seed(name, i):
    return np.random.SeedSequence([zlib.crc32(name.encode()), i])


def _parse_pix(sp):
    a = np.array([[int(v) for v in t.split(":")] for t in sp.split(";")]) if sp else np.zeros((0, 3), int)
    return a[:, 0], a[:, 1]


def run_image(task):
    """one image: simulate, cluster, match; returns (image row, event rows, fake rows)"""
    import sim, metrics as MT
    import adaptive as AD, gmx as GX, fe_candidates as FE
    name, i, par = task
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(_seed(name, i))
    Q, valid, truth, p, rate = sim.simulate(rng, **par)
    Q = np.where(valid, Q, 0.0)                    # as agmx_root.process: inactive pixels set to 0
    H, W = Q.shape
    t0 = time.perf_counter()
    try:
        bg = AD.AM.background_map(Q, valid)
    except Exception:
        bg = None
    t1 = time.perf_counter()
    cl, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True)
    cl = [c for c in cl if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
    t2 = time.perf_counter()
    cands, summ = [], {}
    if bg is not None and info.get("regime") != "unmodelled" and info.get("noise", 9) < 0.35:
        cands, summ = FE.candidates(Q, valid, bg, cl)
    t3 = time.perf_counter()
    # background diagnostics: what the counting finder uses (detection_bg after the hot-column step, as
    # general_cluster) and the count threshold it derives (not part of the timed pipeline)
    ib = dict()
    if bg is not None:
        bg2, hot = GX.hot_columns_as_dark(Q, valid, bg)
        detc = GX.detection_bg(Q, valid, bg2)
        deto = GX.detection_bg(Q, valid, bg, mode="occupancy")
        lam_c = np.where(valid, detc.lam, 0.0); s_c = np.where(valid, detc.s, 1.0)
        t_thr, t_an, t_emp = GX.count_threshold(valid, lam_c, s_c, 0.08)
        lv = bg.lam[valid]
        ib = dict(bg_lam_med=float(np.median(lv)), bg_lam_mean=float(lv.mean()), bg_lam_p90=float(np.percentile(lv, 90)),
                  det_lam_cons=float(np.median(detc.lam[valid])), det_lam_occ=float(np.median(deto.lam[valid])),
                  n_hotcol_flag=int(hot.sum()), hotcol_found=int(any(h in hot.nonzero()[0] for h in p["hot_cols"])),
                  thr_T=t_thr, thr_T_an=t_an, thr_T_emp=t_emp)
    # matching
    big = [t for t in truth if t["kind"] != "fe"]
    objs = [dict(y=c["y"], x=c["x"], pix=c["_pix"]) for c in cl]
    own, per = MT.match(objs, truth, W)
    cobj = [dict(y=c["y"], x=c["x"], pix=_parse_pix(c["pixels"])) for c in cands]
    cown, cper = MT.match(cobj, truth, W)
    E1 = AD.AM.E_PER_E
    ev_rows = []
    for k, t in enumerate(truth):
        js = per[k]
        q = sum(cl[j]["E"] for j in js) / E1 if js else 0.0
        cj = cper[k]
        clean = [j for j in cj if cands[j]["clean"]]
        ev_rows.append(dict(config=name, img=i, ev=k, kind=t["kind"], n_e=t["n_e"], sigma=round(float(t["sigma"]), 4),
                            n_pix=int(len(t["pix"][0])), max_pix=int(t["pix"][2].max()) if len(t["pix"][2]) else 0,
                            found=int(bool(js)), n_obj=len(js), q=round(q, 3),
                            cls=cl[js[0]]["cls"] if js else "", lowE=int(any(cl[j]["cls"] == "low-E" for j in js)),
                            cand_any=int(bool(cj)), cand_clean=int(bool(clean)),
                            cand_nexp=min((cands[j]["n_exp"] for j in clean), default=np.nan),
                            cand_q=sum(cands[j]["charge_e"] for j in cj)))
    fk_rows = []
    for j, c in enumerate(cl):
        if own[j] < 0:
            fk_rows.append(dict(config=name, img=i, src="agmx", near_big=int(own[j] == -2), y=round(c["y"], 2), x=round(c["x"], 2),
                                q=round(c["E"] / E1, 3), npix=int(c.get("npix", len(c["_pix"][0]))), cls=c["cls"],
                                sig=round(float(c.get("sig", np.nan)), 3), n_exp=np.nan, clean=np.nan))
    for j, c in enumerate(cands):
        if cown[j] < 0:
            fk_rows.append(dict(config=name, img=i, src="cand", near_big=int(cown[j] == -2), y=round(c["y"], 2), x=round(c["x"], 2),
                                q=c["charge_e"], npix=c["npix"], cls="", sig=round(c["T"], 3), n_exp=c["n_exp"], clean=int(c["clean"])))
    lam_true = float(np.median(rate[valid]))
    exp_sum = float(sum(v["mean"] for v in summ.get("expected_clean", {}).values()))
    obs_sum = float(sum(summ.get("observed_clean", {}).values()))
    img = dict(config=name, img=i, H=H, W=int(valid.sum(1).max()), mpix=valid.sum() / 1e6, n_truth=len(truth), n_big=len(big),
               lam_true=lam_true, s_true=p["s"], regime=info.get("regime"), agmx_lam=info.get("lam", np.nan),
               agmx_noise=info.get("noise", np.nan), crowded=int(bool(info.get("crowded_mode", False))),
               n_cl=len(cl), n_cand=len(cands), n_cand_clean=summ.get("n_clean", 0), cand_exp_clean=exp_sum,
               cand_obs_clean_all=obs_sum, t_bg=t1 - t0, t_agmx=t2 - t1, t_fe=t3 - t2, **ib)
    return img, ev_rows, fk_rows


IMG_COLS = ["config", "img", "H", "W", "mpix", "n_truth", "n_big", "lam_true", "s_true", "regime", "agmx_lam", "agmx_noise",
            "crowded", "n_cl", "n_cand", "n_cand_clean", "cand_exp_clean", "cand_obs_clean_all", "t_bg", "t_agmx", "t_fe",
            "bg_lam_med", "bg_lam_mean", "bg_lam_p90", "det_lam_cons", "det_lam_occ", "n_hotcol_flag", "hotcol_found",
            "thr_T", "thr_T_an", "thr_T_emp"]
EV_COLS = ["config", "img", "ev", "kind", "n_e", "sigma", "n_pix", "max_pix", "found", "n_obj", "q", "cls", "lowE",
           "cand_any", "cand_clean", "cand_nexp", "cand_q"]
FK_COLS = ["config", "img", "src", "near_big", "y", "x", "q", "npix", "cls", "sig", "n_exp", "clean"]


def _writer(fn, cols):
    new = not os.path.exists(fn)
    fh = open(fn, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if new:
        w.writeheader()
    return fh, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="grid", choices=["grid", "blanks", "l2", "all"])
    ap.add_argument("--only", default="", help="comma-separated config names (default: every config of the set)")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the number of images per config")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    done = set()
    fn_img = os.path.join(OUT, "images.csv")
    if os.path.exists(fn_img):
        with open(fn_img) as fh:
            done = {(r["config"], int(r["img"])) for r in csv.DictReader(fh)}
    only = set(filter(None, a.only.split(",")))
    tasks = []
    for name, (st, n, par) in configs().items():
        if (a.set == "all" or st == a.set) and (not only or name in only):
            tasks += [(name, i, par) for i in range(max(int(round(n * a.scale)), 1)) if (name, i) not in done]
    print(f"{len(tasks)} images to run ({len(done)} done)", flush=True)
    files = [_writer(fn_img, IMG_COLS), _writer(os.path.join(OUT, "events.csv"), EV_COLS), _writer(os.path.join(OUT, "fakes.csv"), FK_COLS)]
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool:
        for k, (img, evs, fks) in enumerate(pool.imap_unordered(run_image, tasks, chunksize=1)):
            files[0][1].writerow(img); files[1][1].writerows(evs); files[2][1].writerows(fks)
            if k % 10 == 0:
                for fh, _ in files: fh.flush()
                print(f"{k + 1}/{len(tasks)} {img['config']} #{img['img']}  {time.time() - t0:.0f} s", flush=True)
    for fh, _ in files:
        fh.close()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
