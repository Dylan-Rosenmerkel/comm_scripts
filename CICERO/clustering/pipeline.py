"""Group-level driver of the Bayesian few-electron finder (wip/lowdc/bayes).

A 'group' is a set of images sharing detector, noise and dark charge (one run setting; in the benchmark the images of
one configuration, e.g. base + blank images of the same (s, lam)). Steps:

  1. noise      per image (or pooled over the group): pedestals (row / column nuisances) and the negative-side
                scale-mixture noise model (noise.py)
  2. pass 1     candidates + neighbourhood evidences with a single-image dark-charge estimate (weak prior)
  3. rates      population EM for the event-rate spectrum R_N (N >= 2) over all neighbourhoods of the group
  4. background pooled hierarchical dark-charge model (background.py) using P(H0 | neighbourhood)
  5. pass 2     evidences with the pooled lam map, R_N EM again, posteriors per neighbourhood
  6. null       (optional) off-image coincidence null: the candidate pixels of the group are permuted among the
                images independently per pixel location (stationary defects stay, spatial event correlations are
                destroyed) and the finder is rerun on the shuffled images
"""
import time
import numpy as np
from scipy import special

import noise as NZ
import finder as FD
import background as BG


ASYM_Z = 4.0        # |z| of the symmetry test above which noise='empirical' switches to the asymmetric density
ASYM_LBF = 3.0      # log Bayes factor of the positive-side scales vs symmetric above which it does the same


def fit_noise(vv, s, noise="mix"):
    """noise density from residuals vv (valid pixels): 'mix' = negative-side scale mixture (NoiseModel);
    'empirical' = non-parametric (noise.fit_empirical) on top of it, symmetric unless the symmetry test is
    significant (max |z| > ASYM_Z), then asymmetric with fitted positive-side scales; 'empirical_sym' /
    'empirical_asym' force the side."""
    nm, info = NZ.fit_negative(vv, s)
    if noise == "mix":
        return nm, info
    side = {"empirical_sym": "neg", "empirical_asym": "asym"}.get(noise)
    if side is not None:
        return NZ.fit_empirical(vv, nm, side=side, c_exc="fit" if side == "asym" else None), info
    # auto: asymmetric only if the data demand it - the core symmetry test (|z| > ASYM_Z below 0.45 e-) or the
    # positive-side scales (c_tail, c_exc) != (1, 1) by a log Bayes factor > ASYM_LBF
    em_a = NZ.fit_empirical(vv, nm, side="asym", c_exc="fit")
    ll = em_a.info["ll_scale"]
    i1 = ll.shape[0] // 2                         # c = 1 is the middle of the symmetric log grid
    lbf = float(special.logsumexp(ll) - np.log(ll.size) - ll[i1, i1])
    z = [abs(a["z"]) for a in NZ.asymmetry(vv, nm.core) if np.isfinite(a["z"])]
    zmax = max(z) if z else 0.0
    em = em_a if (zmax > ASYM_Z or lbf > ASYM_LBF) else NZ.fit_empirical(vv, nm, side="neg")
    em.info.update(auto_zmax=zmax, auto_lbf_scale=lbf)
    return em, info


def noise_stage(Y, valid, gain=1.0, s_guess=None, col=True, noise="mix", mask_cols=()):
    """residual v [e-], column scales, noise model, asymmetry table. noise: see fit_noise; mask_cols: columns
    always excluded (e.g. (0,) for the first-read artefact column), returned as valid_eff"""
    valid = np.array(valid, bool)
    for c in mask_cols:
        valid[:, c] = False
    v, o, u, kap, s = NZ.pedestals(np.asarray(Y, float) / gain, valid, s_guess, col=col)
    nm, info = fit_noise(v[valid], s, noise)
    return dict(v=np.where(valid, v, -10.0), kap=kap, nm=nm, s=s, o=o, u=u, n_neg=info["n_neg"],
                asym=NZ.asymmetry(v[valid], nm.core), valid_eff=valid)


def population_em(nbs, area, pN0, n_iter=500, a=1e-3):
    """event-rate spectrum R_N [events / pixel], N = 0..KCAP, by EM over the neighbourhood evidences of all images:
    responsibilities r_iN = R_N J_i(N) / (1 + sum R J_i + R^2 J2_i), R_N = (sum_i r_iN + 2 r2_i R_N / R + a) / area.
    Sub-threshold neighbourhoods contribute their fractional responsibilities (no threshold anywhere)."""
    rows = [d for nb in nbs for d in nb if "logJN" in d]
    if not rows:
        return np.zeros(FD.KCAP + 1)
    LJ = np.array([d["logJN"] for d in rows])
    LJ2 = np.array([d["logJ2"] for d in rows])
    mask = pN0 > 0
    R_N = np.where(mask, 1e-6 * pN0 / pN0.max(), 0.0)
    for _ in range(n_iter):
        R = R_N.sum()
        with np.errstate(divide="ignore"):
            t1 = LJ + np.log(R_N)[None, :]
            t2 = 2 * np.log(max(R, 1e-300)) + LJ2
        lz = special.logsumexp(np.c_[np.zeros(len(rows)), t1, t2], axis=1)
        r1 = np.exp(t1 - lz[:, None]).sum(0)
        r2 = np.exp(t2 - lz).sum()
        new = np.where(mask, (r1 + 2 * r2 * R_N / max(R, 1e-300) + a) / area, 0.0)
        if np.allclose(new, R_N, rtol=1e-6, atol=1e-14):
            R_N = new
            break
        R_N = new
    return R_N


PT_DLNL = 5.0       # population log-likelihood gain required to keep the point-deposit spectrum (else R_pt = 0)


def _pop_lnl(LJ, LJ2, LP, R_N, R_pt, area):
    """extended (Poisson-process) log-likelihood of the population rates, relative to 'no events'"""
    R = R_N.sum()
    with np.errstate(divide="ignore"):
        t = [np.zeros(LJ.shape[0]), special.logsumexp(LJ + np.log(R_N)[None, :], axis=1),
             2 * np.log(max(R, 1e-300)) + LJ2]
        if R_pt is not None and R_pt.sum() > 0:
            t.append(special.logsumexp(LP + np.log(R_pt)[None, :], axis=1))
    return float(np.logaddexp.reduce(np.c_[tuple(t)], axis=1).sum() - area * (R + (R_pt.sum() if R_pt is not None else 0.0)))


def population_em_pt(nbs, area, pN0, n_iter=500, a=1e-3, a_pt=1e-3, info=None):
    """as population_em, plus a rate spectrum R_pt(N) of point deposits (transient single-pixel defects, logJpt):
    responsibilities r_iN = R_N J_i(N) / Z_i, q_iN = R_pt(N) Jpt_i(N) / Z_i, Z_i = 1 + sum R J + R^2 J2 + sum R_pt Jpt.
    The two are told apart by the event model's spatial spread: an N-electron event puts all its charge into one
    pixel only with probability int p^N d^2mu (the sigma prior), so single-pixel deposits beyond that fraction are
    learned as point deposits. Model selection: R_pt is kept only if it raises the population log-likelihood by more
    than PT_DLNL over the event-only fit (population_em), else (R_N of that fit, zeros) - a compact-event excess
    caused by a sigma-prior mismatch must not turn real events into defects. Returns (R_N, R_pt); info (dict, optional)
    receives dlnL."""
    rows = [d for nb in nbs for d in nb if "logJN" in d]
    if not rows:
        return np.zeros(FD.KCAP + 1), np.zeros(FD.KCAP + 1)
    LJ = np.array([d["logJN"] for d in rows])
    LJ2 = np.array([d["logJ2"] for d in rows])
    LP = np.array([d.get("logJpt", np.full(FD.KCAP + 1, -np.inf)) for d in rows])
    mask = pN0 > 0
    mpt = np.arange(FD.KCAP + 1) >= 2
    R_N = np.where(mask, 1e-6 * pN0 / pN0.max(), 0.0)
    R_pt = np.where(mpt, 1e-8, 0.0)
    for _ in range(n_iter):
        R = R_N.sum()
        with np.errstate(divide="ignore"):
            t1 = LJ + np.log(R_N)[None, :]
            t2 = 2 * np.log(max(R, 1e-300)) + LJ2
            tp = LP + np.log(R_pt)[None, :]
        lz = special.logsumexp(np.c_[np.zeros(len(rows)), t1, t2, tp], axis=1)
        r1 = np.exp(t1 - lz[:, None]).sum(0)
        r2 = np.exp(t2 - lz).sum()
        rp = np.exp(tp - lz[:, None]).sum(0)
        new = np.where(mask, (r1 + 2 * r2 * R_N / max(R, 1e-300) + a) / area, 0.0)
        newp = np.where(mpt, (rp + a_pt) / area, 0.0)
        done = np.allclose(new, R_N, rtol=1e-6, atol=1e-14) and np.allclose(newp, R_pt, rtol=1e-6, atol=1e-14)
        R_N, R_pt = new, newp
        if done:
            break
    R0 = population_em(nbs, area, pN0, n_iter, a)
    dl = _pop_lnl(LJ, LJ2, LP, R_N, R_pt, area) - _pop_lnl(LJ, LJ2, LP, R0, None, area)
    if info is not None:
        info.update(dlnL_pt=dl, R_pt_fit=float(R_pt.sum()))
    if dl < PT_DLNL:
        return R0, np.zeros_like(R_pt)
    return R_N, R_pt


def _init_lam(nb, pixtab, valid, a0=0.5, lam_prior=1e-5):
    """single-image dark charge: EM fixed point of the soft dark counts of the neighbourhoods with at most ~one
    electron (sum of positive residuals < 1.5 e-: no room for an N >= 2 event) plus a weak Gamma prior (0.5
    pseudo-electrons at lam_prior)"""
    single = {d["nb"] for d in nb if d["q_sum"] < 1.5 and not d.get("big")}
    m = np.array([int(k) in single for k in pixtab["nb"]], bool) if pixtab["r"].size else np.zeros(0, bool)
    L = pixtab["L"][m]
    n = np.arange(L.shape[1])
    lam = lam_prior
    for _ in range(200):
        t = L + n[None, :] * np.log(lam) - special.gammaln(n + 1)[None, :]
        s = float((np.exp(t - special.logsumexp(t, axis=1, keepdims=True)) * n[None, :]).sum()) if L.size else 0.0
        new = (s + a0) / (valid.sum() + a0 / lam_prior)
        if abs(new - lam) < 1e-4 * lam:
            return new
        lam = new
    return lam


def process_group(images, prior, gain=1.0, nm_pooled=False, R_fixed=None, n_shuffle=0, seed=0, fit_hotpix=True,
                  noise_info=None, col=True, noise="mix", point_defects=False, mask_cols=(), hot_rows=False,
                  blink=False, nm_fixed=None, lam_fixed=None):
    """images: list of (Y, valid) of equal shape. Returns dict(nbs, pixtabs, noise, bg, R_N, lam_maps, times, null).
    Options (defaults = the original behaviour):
      noise          'mix' (negative-side scale mixture) | 'empirical' | 'empirical_sym' | 'empirical_asym' (fit_noise)
      point_defects  also fit a rate spectrum of point deposits (transient single-pixel defects, population_em_pt);
                     posteriors use it (p_ev excludes them); out['R_pt']
      mask_cols      columns always masked (e.g. (0,): first-read artefact column)
      hot_rows, blink  background.fit components (hot rows; blinking pixels -> masked in pass 2, out['veto'])
      nm_fixed       a given noise model (e.g. run-level or the true one) instead of fitting
      lam_fixed      given dark-charge maps (list per image, or a scalar / map for all) instead of the pooled fit"""
    t_noise = []
    NS = []
    images = [(Y, np.array(v, bool)) for Y, v in images]
    for (Y, valid) in images:
        t0 = time.perf_counter()
        NS.append(noise_stage(Y, valid, gain, col=col, noise="mix" if (nm_pooled and len(images) > 1) else noise,
                              mask_cols=mask_cols))
        t_noise.append(time.perf_counter() - t0)
    if mask_cols:
        images = [(Y, n["valid_eff"]) for n, (Y, _) in zip(NS, images)]
    if nm_fixed is not None:
        for n in NS:
            n["nm"] = nm_fixed
    elif nm_pooled and len(images) > 1:
        t0 = time.perf_counter()
        vv = np.concatenate([n["v"][valid] for n, (_, valid) in zip(NS, images)])
        nm, _ = fit_noise(vv, float(np.median([n["s"] for n in NS])), noise)
        for n in NS:
            n["nm"] = nm
        t_noise[0] += time.perf_counter() - t0
    area = float(sum(v.sum() for _, v in images))

    pt_info = {}

    def rates(nbl):
        if R_fixed is not None:
            return R_fixed, None
        if point_defects:
            return population_em_pt(nbl, area, prior.pN, info=pt_info)
        return population_em(nbl, area, prior.pN), None
    # pass 1
    nbs, pts, t_find = [], [], []
    for n, (Y, valid) in zip(NS, images):
        t0 = time.perf_counter()
        nb, pt = FD.analyse_image(n["v"], n["kap"], n["nm"], valid, 1e-5, prior, only_pix=True)
        lam1 = _init_lam(nb, pt, valid)
        cand = np.zeros(valid.shape, bool); cand[pt["r"], pt["c"]] = True
        nb, pt = FD.analyse_image(n["v"], n["kap"], n["nm"], valid, lam1, prior, cand=cand)
        nbs.append(nb); pts.append(pt); t_find.append(time.perf_counter() - t0)
        n["cand"] = cand; n["lam1"] = lam1
    R_N, R_pt = rates(nbs)
    for nb in nbs:
        FD.posterior(nb, R_N, R_pt=R_pt)
    # pooled background
    t0 = time.perf_counter()
    lam_maps, bgres = BG.fit(pts, nbs, [v for _, v in images], fit_hotpix=fit_hotpix, hot_rows=hot_rows, blink=blink)
    if lam_fixed is not None:
        lam_maps = list(lam_fixed) if isinstance(lam_fixed, (list, tuple)) else [lam_fixed] * len(images)
    t_bg = time.perf_counter() - t0
    veto = np.zeros(images[0][1].shape, bool)
    for r_, c_ in bgres.get("veto", []):
        veto[r_, c_] = True
    # pass 2
    nbs2, pts2 = [], []
    for k, (n, (Y, valid)) in enumerate(zip(NS, images)):
        t0 = time.perf_counter()
        if veto.any():
            n["cand"] = n["cand"] & ~veto
        nb, pt = FD.analyse_image(n["v"], n["kap"], n["nm"], valid, lam_maps[k], prior, cand=n["cand"])
        nbs2.append(nb); pts2.append(pt); t_find[k] += time.perf_counter() - t0
    R_N, R_pt = rates(nbs2)
    for nb in nbs2:
        FD.posterior(nb, R_N, R_pt=R_pt)
    out = dict(nbs=nbs2, pixtabs=pts2, noise=NS, bg=bgres, R_N=R_N, lam_maps=lam_maps, area=area,
               t_noise=t_noise, t_find=t_find, t_bg=t_bg, null=[], R_pt=R_pt, veto=veto, pt_info=pt_info)
    # off-image coincidence null
    rng = np.random.default_rng(seed)
    T = len(images)
    for rep in range(n_shuffle if T > 1 else 0):
        H, W = images[0][1].shape
        # candidates of neighbourhoods already accepted as events (P(event) > 0.5) are removed first: what is
        # shuffled is the background (dark charge, noise, defects) plus any sub-threshold content
        keep = []
        for pt, nb in zip(pts2, nbs2):
            ev = {d["nb"] for d in nb if d.get("p_ev", 0.0) > 0.5 or d.get("big") or d.get("halo")}
            keep.append(np.array([int(k) not in ev for k in pt["nb"]], bool) if pt["r"].size else np.zeros(0, bool))
        allr = np.concatenate([p["r"][m] for p, m in zip(pts2, keep)]); allc = np.concatenate([p["c"][m] for p, m in zip(pts2, keep)])
        allt = np.concatenate([np.full(m.sum(), k) for k, m in enumerate(keep)])
        key = allr * W + allc
        # one random permutation of the image indices per pixel location: new image = perm_key[t]
        uk, inv = np.unique(key, return_inverse=True)
        perms = np.argsort(rng.random((uk.size, T)), axis=1)
        newt = perms[inv, allt]
        nbn = []
        for k in range(T):
            sel = newt == k
            v = np.full((H, W), -10.0)
            v[allr[sel], allc[sel]] = np.array([NS[t]["v"][r, c] for t, r, c in zip(allt[sel], allr[sel], allc[sel])])
            cand = np.zeros((H, W), bool); cand[allr[sel], allc[sel]] = True
            valid = images[k][1]
            nb, _ = FD.analyse_image(v, NS[k]["kap"], NS[k]["nm"], valid, lam_maps[k], prior, cand=cand & valid)
            FD.posterior(nb, R_N, R_pt=R_pt)
            nbn.append(nb)
        out["null"].append(nbn)
    return out
