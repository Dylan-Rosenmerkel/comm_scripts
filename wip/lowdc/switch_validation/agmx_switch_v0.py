"""A_GMX + Bayesian few-electron finder: the adaptive switch of RESULTS.md section 12 (not yet validated).

Called by clustering_tests/adaptive.adaptive_cluster(..., fewe='auto' | 'on') after A_GMX has clustered the image:

  1. noise model and soft candidates on the image (noise.py): read noise s and the soft-candidate fraction f_c, per
     image and per BLOCK x BLOCK block. f_c is the share of pixels with max_{n>=1} f(v - n) / f(v) > 1e-2; it combines
     read noise, its tail and the dark charge in one measure of sparsity (~50 f_c other candidates per 4-px link disk)
  2. regime per block: 'bayes' (f_c <= FC_BAYES), 'both' (<= FC_AGMX), 'agmx' (above, or A_GMX regime 'unmodelled' /
     crowded mode for the whole image). fewe='on' forces 'bayes' wherever A_GMX is not unmodelled
  3. big deposits stay with A_GMX: A_GMX clusters above BIG_E electrons, with a HALO-px halo the Bayesian finder skips
  4. in 'bayes' / 'both' blocks the A_GMX clusters of <= BIG_E electrons outside the halo are replaced by the Bayesian
     catalogue: neighbourhoods with P(event) >= t, t the analytic operating point (sum of P(no event) over the selected
     = F_FAKE x Mpix of the image). With lam >= LAM_2E only events with posterior N >= 3 are kept (2-e- events cannot
     be told from chance pairs of dark electrons there; they are left to a population-rate fit)
  5. dark charge: lam_map (and R_N) of the run when given (pipeline.process_group pools them over the images of a run
     setting); otherwise the finder's single-image two-pass estimate and the event rates fitted on this image alone

Bayesian clusters get: E [keV] = N_mean x E_PER_E (posterior-mean electrons), y, x, sxy = posterior-mean sigma,
cls 'few-e', regime 'bayes', p_ev, logJ1, N_mean, N_map, sigma_post, and '_pix' = (rows, cols, electrons) on the charged
candidate pixels (expected counts under the event hypothesis, scaled to sum to N_mean). Kept A_GMX clusters get
finder = 'agmx'. info['fewe'] records the decision (s, f_c, lam, blocks per regime, threshold, counts).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import noise as NZ        # noqa: E402
import finder as FD       # noqa: E402
import pipeline as PL     # noqa: E402

# switch parameters (RESULTS.md section 12)
FC_BAYES = 6e-3     # f_c at or below: Bayesian finder for groups <= BIG_E
FC_AGMX = 1.5e-2    # f_c above: A_GMX only (neighbourhoods start to percolate)
LAM_2E = 3e-4       # dark charge [e-/pix] at or above: no individual 2-e- events
BIG_E = 20.0        # electrons: larger groups are A_GMX's (X-rays, tracks)
HALO = 15.0         # px around big deposits that the Bayesian finder skips
BLOCK = 100         # px, block size of the per-region decision
F_FAKE = 0.08       # expected fakes per Mpix at the analytic operating point
MIN_BLOCK_PIX = 2000  # blocks with fewer valid pixels take the image's regime


def regime_of(fc):
    return "bayes" if fc <= FC_BAYES else ("both" if fc <= FC_AGMX else "agmx")


def measure(Q, valid, prior=None):
    """noise stage + soft candidates: dict(ns (pipeline.noise_stage), cand, s, f_c, block regimes)"""
    prior = prior or FD.Prior()
    ns = PL.noise_stage(Q, valid, gain=1.0)
    cand = NZ.candidates(ns["v"], ns["kap"], ns["nm"], valid, prior.r_min)
    nv = max(int(valid.sum()), 1)
    fc = float(cand.sum() / nv)
    H, W = valid.shape
    nby, nbx = -(-H // BLOCK), -(-W // BLOCK)
    blk_fc = np.full((nby, nbx), np.nan)
    blk_reg = np.empty((nby, nbx), object)
    for i in range(nby):
        for j in range(nbx):
            sl = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
            nvb = int(valid[sl].sum())
            if nvb >= MIN_BLOCK_PIX:
                blk_fc[i, j] = cand[sl].sum() / nvb
                blk_reg[i, j] = regime_of(blk_fc[i, j])
            else:
                blk_reg[i, j] = regime_of(fc)
    return dict(ns=ns, cand=cand, s=float(ns["nm"].core), f_c=fc, blk_fc=blk_fc, blk_reg=blk_reg, prior=prior)


def _block_regime(m, y, x):
    i = min(max(int(y) // BLOCK, 0), m["blk_reg"].shape[0] - 1)
    j = min(max(int(x) // BLOCK, 0), m["blk_reg"].shape[1] - 1)
    return m["blk_reg"][i, j]


def bayes_catalogue(m, valid, lam_map=None, R_N=None):
    """neighbourhoods of the image with posteriors; single-image dark charge and event rates when not given"""
    ns, prior = m["ns"], m["prior"]
    if lam_map is None:
        nb, pt = FD.analyse_image(ns["v"], ns["kap"], ns["nm"], valid, 1e-5, prior, only_pix=True)
        lam_map = PL._init_lam(nb, pt, valid)
    cand = m["cand"]
    nb, pt = FD.analyse_image(ns["v"], ns["kap"], ns["nm"], valid, lam_map, prior, cand=cand)
    if R_N is None:
        R_N = PL.population_em([nb], float(valid.sum()), prior.pN)
    FD.posterior(nb, R_N)
    return nb, lam_map, R_N


def operating_threshold(nb, area, F=F_FAKE):
    """P(event) threshold t: the largest selection whose summed P(no event) stays <= F x area [Mpix]"""
    p = np.sort(np.array([d["p_ev"] for d in nb if "p_ev" in d and not d.get("big") and not d.get("halo")]))[::-1]
    if not p.size:
        return 1.0
    k = int(np.searchsorted(np.cumsum(1.0 - p), F * area / 1e6, side="right"))
    return float(p[k - 1]) if k > 0 else 1.0


def _sigma_post(d):
    if "logJsig" not in d:
        return np.nan
    w = np.exp(d["logJsig"] - FD._lse(d["logJsig"]))
    return float((w * FD.SIG_NODES).sum())


def _bayes_cluster(d, e_per_e, v):
    pr, pc = d["pix"]
    q = np.clip(v[pr, pc], 0, None)
    q = q / q.sum() * d["N_mean"] if q.sum() > 0 else np.full(pr.size, d["N_mean"] / max(pr.size, 1))
    sp = _sigma_post(d)
    return dict(E=d["N_mean"] * e_per_e, y=d["y"], x=d["x"], sxy=sp, cls="few-e", regime="bayes", finder="bayes",
                occ=np.nan, p_ev=d["p_ev"], logJ1=d.get("logJ1", np.nan), N_mean=d["N_mean"], N_map=d["N_map"],
                sigma_post=sp, edge=False, blend=0.0, ped=0.0, peakfrac=1.0, chi2=np.nan, sig=d.get("logJ1", np.nan),
                _pix=(pr.astype(int), pc.astype(int), q.astype(float)))


def apply(Q, valid, clusters, info, e_per_e, mode="auto", lam_map=None, R_N=None, return_pixels=True):
    """the switch on one image after A_GMX: returns (clusters, info) with info['fewe']"""
    rec = dict(mode=mode)
    if info.get("regime") == "unmodelled":
        rec.update(used=False, reason="A_GMX regime unmodelled (not electron-counting)")
        info["fewe"] = rec
        return clusters, info
    m = measure(Q, valid)
    rec.update(s=m["s"], f_c=m["f_c"], blocks={r: int((m["blk_reg"] == r).sum()) for r in ("bayes", "both", "agmx")})
    if mode == "on":
        m["blk_reg"][:] = "bayes"
    elif info.get("crowded_mode"):
        m["blk_reg"][:] = "agmx"
        rec["reason"] = "A_GMX crowded mode"
    if not (m["blk_reg"] != "agmx").any():
        rec.update(used=False, reason=rec.get("reason", "f_c above the cut everywhere"))
        info["fewe"] = rec
        return clusters, info
    nb, lam_map, R_N = bayes_catalogue(m, valid, lam_map, R_N)
    lam = float(np.median(np.broadcast_to(lam_map, valid.shape)[valid]))
    area = float(valid.sum())
    t = operating_threshold(nb, area)
    # big deposits (A_GMX) and their halo
    big = [c for c in clusters if c["E"] / e_per_e > BIG_E]
    by = np.array([c["y"] for c in big]); bx = np.array([c["x"] for c in big])

    def in_halo(y, x):
        return by.size > 0 and bool((np.hypot(by - y, bx - x) < HALO).any())

    out = []
    n_drop = 0
    for c in clusters:
        small = c["E"] / e_per_e <= BIG_E
        if small and _block_regime(m, c["y"], c["x"]) != "agmx" and not in_halo(c["y"], c["x"]):
            n_drop += 1                     # replaced by the Bayesian catalogue
            continue
        out.append(dict(c, finder="agmx"))
    n_add = 0
    for d in nb:
        if d.get("big") or d.get("halo") or "p_ev" not in d or d["p_ev"] < t:
            continue
        if _block_regime(m, d["y"], d["x"]) == "agmx" or in_halo(d["y"], d["x"]):
            continue
        if lam >= LAM_2E and d["N_mean"] < 3.0:
            continue
        bc = _bayes_cluster(d, e_per_e, m["ns"]["v"])
        if not return_pixels:
            bc.pop("_pix")
        out.append(bc)
        n_add += 1
    rec.update(used=True, lam=lam, R=float(np.sum(R_N)), threshold=t, n_agmx_replaced=n_drop, n_bayes=n_add,
               n_big=len(big), only_N_ge_3=bool(lam >= LAM_2E), pooled_lam=lam_map is not None and np.ndim(lam_map) > 0)
    info["fewe"] = rec
    return out, info


def prepare_run(images, prior=None, **kw):
    """pooled dark charge and event rates for the images of one run setting: [(Q, valid)] -> (lam_maps, R_N);
    pass lam_map=lam_maps[k], R_N=R_N to adaptive_cluster(..., fewe='auto') for image k"""
    res = PL.process_group(images, prior or FD.Prior(), gain=1.0, **kw)
    return res["lam_maps"], res["R_N"]
