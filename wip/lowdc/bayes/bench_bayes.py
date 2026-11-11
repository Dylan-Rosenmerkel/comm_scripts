"""Benchmark of the Bayesian few-electron finder on the wip/lowdc simulation (identical images: ../bench_agmx configs
and seeds). One process per group of configurations that share (s, lam) (base + blank images pooled, like a run with
mostly empty images); the background and the event-rate spectrum are fitted per group.

    python bench_bayes.py --prior physics|bench [--groups g1,g2] [--procs 2] [--shuffle 2]
      -> results/<prior>/objects.csv   every neighbourhood with log J1 > -2 (matched owner, scores, posteriors)
         results/<prior>/events.csv    every injected deposit with its best-matching neighbourhood
         results/<prior>/images.csv    per image: noise model, candidates, time
         results/<prior>/groups.json   per group: background fit vs truth, R_N, noise asymmetry
         results/<prior>/null.csv      off-image coincidence null (shuffled) neighbourhoods with log J1 > -2
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOWDC = os.path.dirname(HERE)
for _p in (HERE, LOWDC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bench_agmx as B  # noqa: E402

LOGJ_KEEP = -2.0
# local persistence test configuration (not in ../bench_agmx.py): base s0.10 l1e-5 plus PERSISTENT defects at fixed
# positions (seeded per group): Poisson hot pixels, blinking pixels (a 2-8 e- burst in 20 % of the images), one
# hot row (serial-register streak) - every detection on them is a fake
PERSIST = dict(n_hot=20, hot_rate=(0.02, 1.0), n_blink=5, blink_p=0.2, blink_amp=(2, 8), hot_row_rate=2e-3)
EXTRA = {"persist_s0.10_l1e-05": ("grid", 15, dict(lam=1e-5, s=0.10), PERSIST)}


def add_defects(Q, valid, rng_img, gname, d):
    """persistent defects (positions from the group seed, charge per image from rng_img): returns Q, rate add-on"""
    H, W = Q.shape
    W_act = int(valid.sum(1).max())
    rg = np.random.default_rng(B._seed(gname, 9999))
    hr, hc = rg.integers(0, H, d["n_hot"]), rg.integers(1, W_act, d["n_hot"])
    hrate = rg.uniform(*d["hot_rate"], d["n_hot"])
    br, bc = rg.integers(0, H, d["n_blink"]), rg.integers(1, W_act, d["n_blink"])
    row = int(rg.integers(0, H))
    Q = Q.copy()
    Q[hr, hc] += rng_img.poisson(hrate)
    fire = rng_img.random(d["n_blink"]) < d["blink_p"]
    Q[br[fire], bc[fire]] += rng_img.integers(d["blink_amp"][0], d["blink_amp"][1] + 1, fire.sum())
    Q[row, :W_act] += rng_img.poisson(d["hot_row_rate"], W_act)
    return Q, dict(hot=list(zip(hr.tolist(), hc.tolist())), blink=list(zip(br.tolist(), bc.tolist())), row=row)


def groups():
    """group name -> list of config names (same simulation parameters apart from n_events)"""
    cf = B.configs()
    g = {}
    for name in cf:
        if name.startswith("blank_"):
            continue
        key = name
        g[key] = [name]
        b = "blank_" + name[len("base_"):] if name.startswith("base_") else None
        if b and b in cf:
            g[key].append(b)
    # out of scope (read noise >= 0.20 or lam = 1e-2: the candidates are no longer sparse); kept out of the default run
    skip = [k for k in g if any(t in k for t in ("_s0.20", "_s0.25", "_l0.01"))]
    g = {k: v for k, v in g.items() if k not in skip}
    for k in EXTRA:
        g[k] = [k]
    return g, skip


def _cfg(cn):
    cf = B.configs()
    return cf[cn] if cn in cf else EXTRA[cn][:3]


def run_group(task):
    """simulate the group's images, run the finder, match; returns tables"""
    import sim, metrics as MT
    import finder as FD, pipeline as PL
    gname, cnames, prior_kind, n_shuffle, opts = task
    warnings.filterwarnings("ignore")
    prior = FD.Prior(prior_kind)
    images, truths, meta = [], [], []
    defects = None
    for cn in cnames:
        st, n, par = _cfg(cn)
        for i in range(n):
            rng = np.random.default_rng(B._seed(cn, i))
            Q, valid, truth, p, rate = sim.simulate(rng, **par)
            if cn in EXTRA:
                Q, defects = add_defects(Q, valid, rng, gname, EXTRA[cn][3])
            images.append((np.where(valid, Q, 0.0), valid))
            truths.append(truth)
            n1 = sum(1 for t in truth if t["kind"] == "fe" and t["n_e"] == 1)
            meta.append(dict(config=cn, img=i, lam_true=float(np.median(rate[valid])), lam_true_mean=float(rate[valid].mean()),
                             n1=n1, mpix=valid.sum() / 1e6, hot_cols=list(p["hot_cols"]), hot_pix=p.get("hot_pix_list", []),
                             rate=rate if (p["hot_cols"] or p["n_hot_pix"]) else None))
    t0 = time.perf_counter()
    res = PL.process_group(images, prior, nm_pooled=True, n_shuffle=n_shuffle, seed=7, **opts)
    t_all = time.perf_counter() - t0
    W = images[0][0].shape[1]
    ev_rows, ob_rows, im_rows, nu_rows = [], [], [], []
    nodes = FD.SIG_NODES
    zn = FD.z_of_sigma(nodes)
    for k, (nb, truth, m) in enumerate(zip(res["nbs"], truths, meta)):
        objs = [d for d in nb if "logJN" in d]
        own, per = MT.match(objs, truth, W)
        for j, d in enumerate(objs):
            ps = None
            if "logJsig" in d and np.isfinite(d["logJsig"]).any():
                ps = np.exp(d["logJsig"] - np.logaddexp.reduce(d["logJsig"]))
            d["sig_mean"] = float((ps * nodes).sum()) if ps is not None else np.nan
            if ps is not None and np.isfinite(zn).any():
                okz = np.isfinite(zn)
                d["z_mean"] = float((ps[okz] * zn[okz]).sum() / max(ps[okz].sum(), 1e-300))
            else:
                d["z_mean"] = np.nan
            if d["logJ1"] > LOGJ_KEEP:
                ob_rows.append(dict(group=gname, config=m["config"], img=m["img"], owner=int(own[j]), y=round(d["y"], 2),
                                    x=round(d["x"], 2), n_cand=d["n_cand"], npix=len(d["pix"][0]), q_sum=round(d["q_sum"], 2),
                                    logJ1=round(d["logJ1"], 3), logJ2=round(d["logJ2"], 3) if np.isfinite(d["logJ2"]) else -999,
                                    p0=d["p0"], p_ev=d["p_ev"], p2=d["p2"], N_mean=round(d["N_mean"], 3), N_map=d["N_map"],
                                    sig_mean=round(d["sig_mean"], 3), z_mean=round(d["z_mean"], 1),
                                    p_pt=d.get("p_pt", 0.0), lJ2=round(float(d["logJN"][2]), 3),
                                    lJ3=round(float(d["logJN"][3]), 3),
                                    defect=_defect_kind(d, defects)))
        for e, t in enumerate(truth):
            js = per[e]
            best = max(js, key=lambda j: objs[j]["logJ1"]) if js else None
            dd = objs[best] if best is not None else {}
            ev_rows.append(dict(group=gname, config=m["config"], img=m["img"], ev=e, kind=t["kind"], n_e=t["n_e"],
                                sigma=round(float(t["sigma"]), 4), n_pix=int(len(t["pix"][0])),
                                max_pix=int(t["pix"][2].max()) if len(t["pix"][2]) else 0,
                                n_obj=len(js), logJ1=round(dd.get("logJ1", -np.inf), 3), p_ev=dd.get("p_ev", 0.0),
                                p0=dd.get("p0", 1.0), p_pt=dd.get("p_pt", 0.0),
                                lJ2=round(float(dd["logJN"][2]), 3) if dd else -np.inf,
                                lJ3=round(float(dd["logJN"][3]), 3) if dd else -np.inf,
                                N_mean=round(dd.get("N_mean", np.nan), 3), N_map=dd.get("N_map", -1),
                                N_sum=round(sum(objs[j]["N_mean"] for j in js if objs[j]["p_ev"] > 0.5), 3),
                                n_obj50=sum(1 for j in js if objs[j]["p_ev"] > 0.5),
                                sig_mean=round(dd.get("sig_mean", np.nan), 3), z_mean=round(dd.get("z_mean", np.nan), 1)))
        nsd = res["noise"][k]
        nm = nsd["nm"]
        im_rows.append(dict(group=gname, config=m["config"], img=m["img"], mpix=m["mpix"], lam_true=m["lam_true"],
                            n1_events=m["n1"], n_cand=int(res["pixtabs"][k]["r"].size), n_nb=len(nb),
                            n_big=sum(1 for d in nb if d.get("big")), s_core=nm.s[0], w_tail=nm.w[1], s_tail=nm.s[1],
                            w_broad=nm.w[2], s_broad=nm.s[2], lam_init=nsd["lam1"], t_noise=res["t_noise"][k],
                            t_find=res["t_find"][k], noise_kind=type(nm).__name__))
    for rep, nbn in enumerate(res["null"]):
        for k, nb in enumerate(nbn):
            for d in nb:
                if "logJN" in d and d["logJ1"] > LOGJ_KEEP:
                    nu_rows.append(dict(group=gname, rep=rep, img=k, y=d["y"], x=d["x"], n_cand=d["n_cand"],
                                        logJ1=d["logJ1"], p0=d["p0"], p_ev=d["p_ev"], N_map=d["N_map"]))
    # background vs truth
    bg = res["bg"]
    truth_hot = {}
    if meta[0]["rate"] is not None:
        hc = meta[0]["hot_cols"]
        truth_hot["hot_cols"] = hc
        truth_hot["hot_pix"] = sorted({tuple(h) for m in meta for h in m["hot_pix"]})
    asym = pd.DataFrame([dict(img=k, **a) for k, n in enumerate(res["noise"]) for a in n["asym"]])
    asym_sum = asym.groupby("a")[["n_pos", "n_neg"]].sum().reset_index().to_dict("records")
    summ = dict(group=gname, configs=cnames, n_img=len(images), mpix=res["area"] / 1e6,
                lam_true_median=float(np.median([m["lam_true"] for m in meta])),
                lam_true_mean=float(np.mean([m["lam_true_mean"] for m in meta])),
                lam_1e_events=float(sum(m["n1"] for m in meta) / res["area"]),
                lam0=bg["lam0"], beta=bg["beta"], lam_mean_fit=bg["lam_mean"],
                lam_init_median=float(np.median([n["lam1"] for n in res["noise"]])),
                hot_cols_fit=bg["hot_cols"], hot_pix_fit=bg["hot_pix"][:50], n_hot_pix_fit=len([h for h in bg["hot_pix"] if h["P"] > 0.5]),
                truth_hot=truth_hot, R_N=[float(x) for x in res["R_N"]],
                R_true=float(sum(1 for tr in truths for t in tr if t["kind"] == "fe" and t["n_e"] >= 2) / res["area"]),
                asym=asym_sum, t_all=t_all, t_bg=res["t_bg"], opts=opts,
                R_pt=[float(x) for x in res["R_pt"]] if res.get("R_pt") is not None else None,
                pt_info=res.get("pt_info", {}), hot_rows_fit=bg.get("hot_rows", []), blink_fit=bg.get("blink_pix", []), n_veto=len(bg.get("veto", [])),
                defects=defects, noise_info={k: v for k, v in getattr(res["noise"][0]["nm"], "info", {}).items()
                                             if isinstance(v, (str, int, float))})
    return gname, ev_rows, ob_rows, im_rows, nu_rows, summ


def _defect_kind(d, defects):
    """'hot' / 'blink' / 'row' when a neighbourhood touches a persistent defect of the local test config, else ''"""
    if not defects:
        return ""
    rr, cc = d["cand"]
    pts = set(zip(np.asarray(rr).tolist(), np.asarray(cc).tolist()))
    if pts & set(map(tuple, defects["blink"])):
        return "blink"
    if pts & set(map(tuple, defects["hot"])):
        return "hot"
    if defects["row"] in set(np.asarray(rr).tolist()):
        return "row"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="physics", choices=["physics", "bench"])
    ap.add_argument("--groups", default="")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--shuffle", type=int, default=2)
    ap.add_argument("--tag", default="", help="output folder results/<tag> (default results/<prior>)")
    ap.add_argument("--noise", default="mix", help="pipeline noise option: mix | empirical | empirical_sym | empirical_asym")
    ap.add_argument("--point", type=int, default=0, help="1: point-deposit (transient single-pixel defect) component")
    ap.add_argument("--mask-col0", type=int, default=0)
    ap.add_argument("--hot-rows", type=int, default=0)
    ap.add_argument("--blink", type=int, default=0)
    a = ap.parse_args()
    opts = dict(noise=a.noise, point_defects=bool(a.point), mask_cols=(0,) if a.mask_col0 else (),
                hot_rows=bool(a.hot_rows), blink=bool(a.blink))
    G, skip = groups()
    names = [g for g in a.groups.split(",") if g] or [g for g in G if g not in EXTRA]
    out = os.path.join(HERE, "results", a.tag or a.prior)
    os.makedirs(out, exist_ok=True)
    fs = {k: os.path.join(out, f"{k}.csv") for k in ("events", "objects", "images", "null")}
    fj = os.path.join(out, "groups.json")
    done = json.load(open(fj)) if os.path.exists(fj) else {}
    names = [n for n in names if n not in done]
    # biggest groups first (load balance)
    names.sort(key=lambda g: -sum(_cfg(c)[1] * _cfg(c)[2].get("H", 480) for c in G[g]))
    tasks = [(g, G[g], a.prior, a.shuffle, opts) for g in names]
    print(f"{len(tasks)} groups ({len(done)} done); skipped (out of scope): {skip}", flush=True)
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for gname, ev, ob, im, nu, summ in pool.imap_unordered(run_group, tasks, chunksize=1):
            for key, rows in (("events", ev), ("objects", ob), ("images", im), ("null", nu)):
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(fs[key], mode="a", index=False, header=not os.path.exists(fs[key]))
            done[gname] = summ
            json.dump(done, open(fj, "w"), indent=1, default=float)
            print(f"{gname}: {summ['n_img']} img, {summ['t_all']:.0f} s (lam fit {summ['lam_mean_fit']:.2e} "
                  f"true {summ['lam_true_mean']:.2e}+1e {summ['lam_1e_events']:.1e})  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
