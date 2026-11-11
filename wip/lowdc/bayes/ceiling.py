"""Ceilings for the hard cases (s = 0.15, lam 1e-4 / 1e-3) and population-level rate sensitivity (wip/lowdc/bayes).

    python ceiling.py pairs                       analytic accidental-pair limit (noiseless counts) -> results/v2/pairs.md
    python ceiling.py oracle --group G            finder vs oracle on the SAME images of a benchmark group
    python ceiling.py pop --group G               population R_N fit: unbiasedness on blanks, sensitivity vs counting
    python ceiling.py grid                        oracle 2-/3-e- efficiency at 0.08 fakes / Mpix on an (s, lam) grid

Oracle = Neyman-Pearson test with everything true: the true lam map (dark + serial + hot; plus the 1-e- event density,
which is indistinguishable), the true noise (Gaussian s, no pedestal estimate), the true sigma distribution (uniform
0.2-1.2 px, 'bench' prior) and the true N spectrum (uniform 2..10); score log J1 (rate-free). Variants on the same
images: 'bkg' = true background + noise but the finder's prior (physics sigma, N flat 2..20); the finder itself is
taken from a benchmark run (results/<tag>). Gap split: finder -> bkg = background / noise estimation, bkg -> oracle =
prior, oracle -> 1 = fundamental (chance coincidences of dark electrons and noise, analytic pair limit for s -> 0).
Single process each (machine-load rule); outputs in results/v2/.
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
import sim, metrics as MT  # noqa: E402
import finder as FD, noise as NZ, pipeline as PL  # noqa: E402
import bench_bayes as BB  # noqa: E402

OUT = os.path.join(HERE, "results", "v2")
F_OP = 0.08


# ------------------------------------------------------------------------------------------------ analytic pairs
def pattern_probs(N, sig_lo=0.2, sig_hi=1.2, n_mc=400000, binning=1, seed=1):
    """P(pattern) of an N-electron event (centre uniform in a pixel, sigma uniform sig_lo..sig_hi px, pixel size
    `binning` original px): pattern = sorted tuple of (dy, dx, count) relative to the first occupied pixel"""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(sig_lo, sig_hi, n_mc)
    c = rng.uniform(0, binning, (n_mc, 2))
    pos = np.floor((c[:, None, :] + rng.normal(size=(n_mc, N, 2)) * sig[:, None, None]) / binning).astype(int)
    out = {}
    for p in pos:
        k = np.unique(p, axis=0, return_counts=True)
        a = np.c_[k[0] - k[0][0], k[1]]
        key = tuple(map(tuple, a))
        out[key] = out.get(key, 0) + 1
    return {k: v / n_mc for k, v in out.items()}


def null_rate(pattern, lam):
    """chance rate per pixel (anchor) of exactly this dark-charge pattern: prod_i Pois(n_i; lam) ~ prod lam^n / n!"""
    from math import factorial
    r = 1.0
    for (_, _, n) in pattern:
        r *= lam ** n / factorial(n)
    return r


def np_limit(probs, lam, F=F_OP, binning=1):
    """Neyman-Pearson efficiency with noiseless counts: select patterns by P_ev / P_null until the chance rate
    reaches F per Mpix (of ORIGINAL pixels; a binned pixel covers binning^2 of them), fractional last pattern"""
    lam_b = lam * binning ** 2
    rows = [(p, pe, null_rate(p, lam_b) * 1e6 / binning ** 2) for p, pe in probs.items()]
    rows.sort(key=lambda t: -t[1] / t[2])
    eff, fk = 0.0, 0.0
    for p, pe, nr in rows:
        if fk + nr <= F:
            eff += pe; fk += nr
        else:
            eff += pe * (F - fk) / nr
            break
    return eff


def cmd_pairs(a):
    rows = []
    for N in (2, 3):
        for b in (1, 2):
            probs = pattern_probs(N, binning=b)
            for lam in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 3e-3):
                rows.append(dict(N=N, binning=b, lam=lam, eff_F0p08=round(np_limit(probs, lam, binning=b), 3),
                                 eff_F1=round(np_limit(probs, lam, 1.0, binning=b), 3)))
        p1 = pattern_probs(N)
        top = sorted(p1.items(), key=lambda t: -t[1])[:6]
        print(N, [(k, round(v, 3)) for k, v in top])
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["N", "binning"], columns="lam", values="eff_F0p08").round(3)
    md = ("# Analytic accidental-coincidence limit (noiseless counts, ceiling.py pairs)\n\n"
          "Neyman-Pearson selection of charge patterns by P(pattern | event) / P(pattern | dark charge), events with "
          "sigma uniform 0.2-1.2 px (the benchmark), until the chance rate of dark-charge patterns reaches F per Mpix. "
          "binning 2 = 2x2 on-chip binning (lam per binned pixel x4, sigma / 2; F per Mpix of original pixels).\n\n"
          "Efficiency at F = 0.08 / Mpix:\n\n" + MT.md_table(piv.reset_index()) + "\n\nFull table:\n\n" + MT.md_table(df) + "\n")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "pairs.md"), "w").write(md)
    print(md)


# ------------------------------------------------------------------------------------------------ group images
def load_group(gname):
    G, _ = BB.groups()
    images, truths, rates, is_blank = [], [], [], []
    for cn in G[gname]:
        st, n, par = BB._cfg(cn)
        for i in range(n):
            rng = np.random.default_rng(B._seed(cn, i))
            Q, valid, truth, p, rate = sim.simulate(rng, **par)
            images.append((np.where(valid, Q, 0.0), valid)); truths.append(truth); rates.append(rate)
            is_blank.append(cn.startswith("blank_"))
    return images, truths, rates, np.array(is_blank), par


def true_rates(truths, area):
    """true event-rate spectrum R_N per pixel (N = 0..KCAP) of the group and the 1-e- event density"""
    R = np.zeros(FD.KCAP + 1)
    for tr in truths:
        for t in tr:
            if t["kind"] == "fe" and t["n_e"] <= FD.KCAP:
                R[t["n_e"]] += 1
    return R / area


def run_variant(images, truths, rates, s, prior, lam_add, R_true):
    """analyse every image with the true noise / true lam map; returns objects table (owner, logJ1, p0, lJ2, lJ3)"""
    nm = NZ.NoiseModel([1.0], [s])
    rows, nbs = [], []
    W = images[0][0].shape[1]
    for k, ((Q, valid), truth, rate) in enumerate(zip(images, truths, rates)):
        v = np.where(valid, Q, -10.0)
        nb, _ = FD.analyse_image(v, np.ones(W), nm, valid, rate + lam_add, prior)
        FD.posterior(nb, R_true)
        objs = [d for d in nb if "logJN" in d]
        own, per = MT.match(objs, truth, W)
        for j, d in enumerate(objs):
            if d["logJ1"] > -2:
                rows.append(dict(img=k, owner=int(own[j]), q_sum=d["q_sum"], logJ1=d["logJ1"], p0=d["p0"], p_ev=d["p_ev"],
                                 lJ2=float(d["logJN"][2]), lJ3=float(d["logJN"][3])))
        ev = []
        for e, t in enumerate(truth):
            js = per[e]
            best = max(js, key=lambda j: objs[j]["logJ1"]) if js else None
            ev.append(dict(img=k, kind=t["kind"], n_e=t["n_e"], sigma=t["sigma"],
                           logJ1=objs[best]["logJ1"] if best is not None else -np.inf,
                           p_ev=objs[best]["p_ev"] if best is not None else 0.0))
        nbs.append(ev)
    return pd.DataFrame(rows), pd.DataFrame([e for ev in nbs for e in ev])


def roc_point(ob, ev, mpix, F=F_OP, ne=(2, 3, 4)):
    """best efficiency with measured fakes <= F x Mpix on the score logJ1 (threshold just above the (k+1)-th fake),
    and the analytic operating point (sum of p0 over the selection = F x Mpix)"""
    fk = np.sort(ob[(ob.owner == -1) & (ob.q_sum < 20)].logJ1.values)[::-1]
    nmax = int(np.floor(F * mpix))
    thr = fk[nmax] + 1e-9 if fk.size > nmax else -np.inf
    e = ev[ev.kind == "fe"]
    out = {f"eff{n}_meas": float((e[e.n_e == n].logJ1 >= thr).mean()) for n in ne}
    out["thr"] = float(thr); out["n_fake_allowed"] = nmax
    s = ob.sort_values("p_ev", ascending=False)
    cum = np.cumsum(s.p0.values)
    k = int(np.searchsorted(cum, F * mpix, side="right"))
    t = s.p_ev.values[k - 1] if k > 0 else 1.0
    out.update({f"eff{n}_op": float((e[e.n_e == n].p_ev >= t).mean()) for n in ne})
    out["fakes_op"] = int(((s.iloc[:k].owner == -1) & (s.iloc[:k].q_sum < 20)).sum())
    return out


def cmd_oracle(a):
    warnings.filterwarnings("ignore")
    images, truths, rates, is_blank, par = load_group(a.group)
    area = float(sum(v.sum() for _, v in images)); mpix = area / 1e6
    R_true = true_rates(truths, area)
    lam1 = R_true[1]                                   # 1-e- events: part of the background for every method
    R_true[1] = 0.0
    res = {}
    for name, prior in (("bkg", FD.Prior("physics")), ("oracle", FD.Prior("bench", nk="flat10"))):
        t0 = time.time()
        R_use = R_true if name == "oracle" else R_true.sum() * prior.pN
        ob, ev = run_variant(images, truths, rates, par["s"], prior, lam1, R_use)
        ob.to_csv(os.path.join(OUT, f"oracle_{a.group}_{name}_objects.csv"), index=False)
        ev.to_csv(os.path.join(OUT, f"oracle_{a.group}_{name}_events.csv"), index=False)
        res[name] = roc_point(ob, ev, mpix)
        print(name, res[name], f"{time.time() - t0:.0f} s", flush=True)
    # the finder (benchmark run)
    for tag in a.tags.split(","):
        d = os.path.join(HERE, "results", tag)
        if os.path.exists(os.path.join(d, "objects.csv")):
            ob = pd.read_csv(os.path.join(d, "objects.csv")); ev = pd.read_csv(os.path.join(d, "events.csv"))
            ob = ob[ob.group == a.group]; ev = ev[ev.group == a.group]
            if len(ev):
                res[f"finder {tag}"] = roc_point(ob, ev, mpix)
    df = pd.DataFrame([dict(variant=k, **v) for k, v in res.items()])
    df.to_csv(os.path.join(OUT, f"oracle_{a.group}.csv"), index=False)
    print(MT.md_table(df.round(3)))


# ------------------------------------------------------------------------------------------------ population
def em_fit(LJ, LJ2, area, pN0, n_iter=200):
    """population_em on arrays (rows = neighbourhoods)"""
    nbs = [[dict(logJN=l, logJ2=l2) for l, l2 in zip(LJ, LJ2)]]
    return PL.population_em(nbs, area, pN0, n_iter=n_iter)


def cmd_pop(a):
    """R_N from the population EM: (1) unbiasedness on the blank images alone; (2) sensitivity to a 2-e- (3-e-)
    event rate rho per Mpix at exposure E Mpix-images: bootstrap of blank-image neighbourhoods (null) plus a Poisson
    number of matched event neighbourhoods (signal, from the base images, incl. events with no neighbourhood) ->
    EM -> R_2; vs per-event counting at 0.08 fakes / Mpix with the measured fake rate subtracted"""
    warnings.filterwarnings("ignore")
    images, truths, rates, is_blank, par = load_group(a.group)
    prior = FD.Prior("physics")
    t0 = time.time()
    res = PL.process_group(images, prior, nm_pooled=True)
    print(f"process_group {time.time() - t0:.0f} s", flush=True)
    W = images[0][0].shape[1]
    null, sig = [], {2: [], 3: []}
    n_ev = {2: 0, 3: 0}
    for k, (nb, truth) in enumerate(zip(res["nbs"], truths)):
        objs = [d for d in nb if "logJN" in d]
        own, per = MT.match(objs, truth, W)
        if is_blank[k]:
            # neighbourhoods with log J1 < -5 change R by < ~1e-3 relative (sum R J << area R): dropped for speed
            null.append([(d["logJN"], d["logJ2"], d["logJ1"]) for d in objs if d["logJ1"] > -5])
        else:
            for e, t in enumerate(truth):
                if t["kind"] == "fe" and t["n_e"] in sig:
                    n_ev[t["n_e"]] += 1
                    # the event's neighbourhoods (usually one)
                    sig[t["n_e"]].append([(objs[j]["logJN"], objs[j]["logJ2"], objs[j]["logJ1"]) for j in per[e]])
    mp_img = float(images[0][1].sum()) / 1e6
    nb_blank = int(is_blank.sum())
    LJb = np.array([x[0] for im in null for x in im]); LJ2b = np.array([x[1] for im in null for x in im])
    area_b = nb_blank * mp_img * 1e6
    R_b = em_fit(LJb, LJ2b, area_b, prior.pN)
    out = dict(group=a.group, blank_mpix=nb_blank * mp_img, R_blank=R_b.sum(), R2_blank=R_b[2], R3_blank=R_b[3],
               pseudo=1e-3 * (prior.pN > 0).sum() / area_b)
    print("blank-only EM:", out, flush=True)
    # counting threshold: logJ1 above which the blanks give F_OP fakes / Mpix
    l1b = np.sort(np.array([x[2] for im in null for x in im]))[::-1]
    nmax = int(F_OP * nb_blank * mp_img)
    thr = l1b[nmax] + 1e-9
    b_rate = (l1b >= thr).sum() / (nb_blank * mp_img)
    eff = {n: np.mean([any(x[2] >= thr for x in s_) for s_ in sig[n]]) for n in sig}
    out.update(count_thr=thr, count_fake_rate=b_rate, count_eff2=eff[2], count_eff3=eff[3])
    rng = np.random.default_rng(5)
    rows = []
    for E in (10.0, 100.0):
        n_img = int(round(E / mp_img))
        for rho in (0.1, 0.3, 1.0):
            for n in (2, 3):
                r_hat, c_hat = [], []
                for rep in range(a.reps):
                    ims = rng.integers(0, len(null), n_img)
                    rows_n = [x for i in ims for x in null[i]]
                    k_ev = rng.poisson(rho * E)
                    evs = rng.integers(0, len(sig[n]), k_ev)
                    rows_s = [x for i in evs for x in sig[n][i]]
                    allr = rows_n + rows_s
                    LJ = np.array([x[0] for x in allr]); LJ2 = np.array([x[1] for x in allr])
                    R = em_fit(LJ, LJ2, E * 1e6, prior.pN)
                    r_hat.append(R[n] * 1e6)                     # per Mpix
                    cnt = sum(1 for x in allr if x[2] >= thr)
                    c_hat.append((cnt - b_rate * E) / (eff[n] * E) if eff[n] > 0 else np.nan)
                r_hat = np.array(r_hat); c_hat = np.array(c_hat)
                rows.append(dict(E_Mpix=E, rho_per_Mpix=rho, N=n, em_mean=r_hat.mean(), em_sd=r_hat.std(),
                                 count_mean=np.nanmean(c_hat), count_sd=np.nanstd(c_hat),
                                 em_bias=r_hat.mean() - rho, sd_ratio_count_over_em=np.nanstd(c_hat) / max(r_hat.std(), 1e-30)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"pop_{a.group}.csv"), index=False)
    json.dump(out, open(os.path.join(OUT, f"pop_{a.group}.json"), "w"), indent=1, default=float)
    print(MT.md_table(df.round(4)))


# ------------------------------------------------------------------------------------------------ grid
def _grid_task(t):
    s, lam, n_ev_img, n_bl_img, seed = t
    warnings.filterwarnings("ignore")
    rows = []
    nm = NZ.NoiseModel([1.0], [s])
    prior = FD.Prior("bench", nk="flat10")
    H, W = 480, 1280
    area = H * W
    ob_rows, ev_rows = [], []
    for kind, n_img in (("ev", n_ev_img), ("blank", n_bl_img)):
        for i in range(n_img):
            rng = np.random.default_rng([seed, int(s * 1000), int(lam * 1e7), i, kind == "ev"])
            ne = 80 if kind == "ev" else 0
            Q, valid, truth, p, rate = sim.simulate(rng, H=H, W=W, n_os=0, lam=lam, s=s, n_events=ne, ne_range=(2, 3))
            R = np.zeros(FD.KCAP + 1)
            if kind == "ev":
                R[2] = R[3] = 40.0 / area
            else:
                R[2] = R[3] = 1e-12
            nb, _ = FD.analyse_image(np.where(valid, Q, -10.0), np.ones(W), nm, valid, rate, prior)
            FD.posterior(nb, R)
            objs = [d for d in nb if "logJN" in d]
            own, per = MT.match(objs, truth, W)
            for j, d in enumerate(objs):
                if d["logJ1"] > -2:
                    ob_rows.append(dict(kind=kind, owner=int(own[j]), q_sum=d["q_sum"], logJ1=d["logJ1"], p0=d["p0"]))
            for e, t in enumerate(truth):
                js = per[e]
                ev_rows.append(dict(n_e=t["n_e"], logJ1=max((objs[j]["logJ1"] for j in js), default=-np.inf)))
    ob = pd.DataFrame(ob_rows); ev = pd.DataFrame(ev_rows)
    mp_ev, mp_bl = n_ev_img * area / 1e6, n_bl_img * area / 1e6
    # null tail: blanks (measured) + event images (unmatched objects); fake rate per Mpix at threshold
    fk = np.sort(ob[(ob.owner == -1) & (ob.q_sum < 20)].logJ1.values)[::-1]
    mp = mp_ev + mp_bl
    out = dict(s=s, lam=lam, mpix_null=round(mp, 1))
    for F in (F_OP, 1.0):
        nmax = int(np.floor(F * mp))
        thr = fk[nmax] + 1e-9 if fk.size > nmax else -np.inf
        for n in (2, 3):
            e = ev[ev.n_e == n]
            out[f"eff{n}_F{F:g}"] = round(float((e.logJ1 >= thr).mean()), 3)
        out[f"thr_F{F:g}"] = round(float(thr), 2)
    return out


def cmd_grid(a):
    from multiprocessing import get_context
    tasks = [(s, lam, a.n_ev, a.n_blank, 3) for s in (0.06, 0.10, 0.12, 0.15) for lam in (1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 1e-3)]
    rows = []
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_grid_task, tasks):
            rows.append(r)
            print(r, f"{time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).sort_values(["s", "lam"])
    df.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    md = ("# Oracle 2-/3-e- efficiency on an (s, lam) grid (ceiling.py grid)\n\n"
          "Oracle finder (true noise, true lam, true sigma U(0.2, 1.2), N in {2, 3}); events-only images (40 2-e- + 40 "
          "3-e- per 480 x 1280 image) plus blank images; threshold on log J1 at the measured fake rate F per Mpix "
          "(unmatched objects of all images).\n\n" + MT.md_table(df) + "\n")
    open(os.path.join(OUT, "grid.md"), "w").write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pairs", "oracle", "pop", "grid"])
    ap.add_argument("--group", default="base_s0.15_l0.0001")
    ap.add_argument("--tags", default="v2/before,v2/after")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-ev", type=int, default=10)
    ap.add_argument("--n-blank", type=int, default=200)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(pairs=cmd_pairs, oracle=cmd_oracle, pop=cmd_pop, grid=cmd_grid)[a.cmd](a)


if __name__ == "__main__":
    main()
