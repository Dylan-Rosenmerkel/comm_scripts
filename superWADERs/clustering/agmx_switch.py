"""A_GMX + Bayesian few-electron finder: the adaptive switch of RESULTS.md section 12 (validated in
wip/lowdc/switch_validation/VALIDATION.md).

Called by clustering_tests/adaptive.adaptive_cluster(..., fewe='auto' | 'on') after A_GMX has clustered the image:

  1. big deposits: A_GMX clusters above BIG_E electrons (footprint = their '_pix', else a disc around the centroid)
     stay A_GMX's; together with the finder's own > BIG_E candidate groups they define a HALO-px halo (distance to
     the nearest deposit PIXEL, not to the centroid: a muon track is long)
  2. noise model and soft candidates on the image (noise.py): read noise s and the soft-candidate fraction f_c, per
     image and per BLOCK x BLOCK block, counted OUTSIDE the big-deposit footprints (padded by BIG_PAD px) - a deposit
     must not turn its block to A_GMX. f_c is the share of pixels with max_{n>=1} f(v - n) / f(v) > 1e-2; it
     combines read noise, its tail and the dark charge in one measure of sparsity (~50 f_c other candidates per 4-px
     link disk)
  3. regime per block: 'bayes' (f_c <= FC_BAYES), 'both' (<= fc_agmx(); treated as 'bayes': Bayes catalogue), 'agmx'
     (above, or A_GMX regime 'unmodelled' / crowded mode for the whole image). fewe='on' forces 'bayes' wherever
     A_GMX is not unmodelled. The Bayes region B = valid pixels of non-'agmx' blocks outside the halo
  4. ownership, decided ONCE per event so that a block edge or the halo edge cannot double count or lose it: a
     Bayesian neighbourhood is in B when its centre is in B and the finder did not flag it big / halo; an A_GMX
     cluster <= BIG_E e- that touches (1 px) a neighbourhood follows that neighbourhood, one that touches none
     follows its own centre. A_GMX clusters in B are replaced by the Bayesian catalogue of B
  5. the catalogue: neighbourhoods in B with P(event) >= t, t the analytic operating point (sum of P(no event) over
     the selected = F_FAKE x Mpix of B). With lam >= LAM_2E only events with P(N >= 3 | event) >= 0.5 are reported
     and budgeted (2-e- events cannot be told from chance pairs of dark electrons there; they are left to a
     population-rate fit)
  6. run-pooled mode (recommended): prepare_run(..., full=True) pools the noise model, the dark charge and the event
     rates over the images of a run setting and sets the P(event) cut on the whole run; single-image mode: this
     image's own noise fit, two-pass dark charge, event rates and fake budget (conservative, see VALIDATION.md)

Bayesian clusters get: E [keV] = N_mean x E_PER_E (posterior-mean electrons), y, x (A_GMX convention: pixel i centred
at i), sxy = posterior-mean sigma, cls 'few-e', regime 'bayes', finder 'bayes', p_ev, logJ1, N_mean, N_map,
sigma_post, and '_pix' = (rows, cols, electrons) on the charged candidate pixels (expected counts under the event
hypothesis, scaled to sum to N_mean). Kept A_GMX clusters get finder = 'agmx'. info['fewe'] records the decision
(s, f_c, f_c_all, lam, blocks per regime, Bayes area, threshold, counts).
"""
import os
import sys

import math
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import noise as NZ        # noqa: E402
import finder as FD       # noqa: E402
import pipeline as PL     # noqa: E402

# switch parameters (RESULTS.md section 12; switch_validation/VALIDATION.md)
FC_BAYES = 6e-3     # f_c at or below: Bayesian finder for groups <= BIG_E
FC_AGMX = 1.5e-2    # f_c above: A_GMX only (neighbourhoods start to percolate)
LAM_2E = 3e-4       # dark charge [e-/pix] at or above: no individual 2-e- events
BIG_E = 20.0        # electrons: larger groups are A_GMX's (X-rays, tracks)
HALO = 5.0          # px from the nearest pixel of a big deposit that the Bayesian finder skips (halo test 2026-11-13:
                    # 3-10 px find 91-96 % of 2-10 e- events at 10-20 px from Fe-55 / muons, 15 px 86-89 %; no fakes)
BIG_PAD = 2         # px: big-deposit footprints padded by this are left out of f_c
BLOCK = 100         # px, block size of the per-region decision
F_FAKE = 0.08       # expected fakes per Mpix at the analytic operating point
MIN_BLOCK_PIX = 2000  # blocks with fewer valid (non-deposit) pixels take the image's regime
BINNING = (1, 1)    # on-chip binning (rows, columns) of the image being clustered: set by adaptive_cluster(binning=)
LINK = 1            # px: an A_GMX cluster within this of a neighbourhood's candidate pixel follows that neighbourhood


def regime_of(fc):
    return "bayes" if fc <= FC_BAYES else ("both" if fc <= fc_agmx() else "agmx")


def big_footprint(clusters, e_per_e, shape):
    """pixels of the A_GMX clusters above BIG_E e- ('_pix'; without pixel lists a disc of radius max(3, 3 sxy) px
    around the centroid) and the list of those clusters"""
    H, W = shape
    m = np.zeros(shape, bool)
    big = []
    for c in clusters:
        if not c["E"] / e_per_e > BIG_E:
            continue
        big.append(c)
        p = c.get("_pix")
        if p is not None and len(p[0]):
            r, cc = np.asarray(p[0], int), np.asarray(p[1], int)
            ok = (r >= 0) & (r < H) & (cc >= 0) & (cc < W)
            m[r[ok], cc[ok]] = True
        else:
            s = c.get("sxy", np.nan)
            rad = max(3.0, 3.0 * s) if np.isfinite(s) else 3.0
            y0, x0 = int(round(c["y"])), int(round(c["x"]))
            k = int(np.ceil(rad))
            ys, xs = np.ogrid[max(y0 - k, 0):min(y0 + k + 1, H), max(x0 - k, 0):min(x0 + k + 1, W)]
            m[ys, xs] |= (ys - y0) ** 2 + (xs - x0) ** 2 <= rad * rad
    return m, big


def measure(Q, valid, prior=None, exclude=None, nm=None):
    """noise stage + soft candidates: dict(ns (pipeline.noise_stage), cand, s, f_c, f_c_all, block f_c and regimes).
    exclude: pixels left out of f_c (big-deposit footprints); f_c_all counts every valid pixel. nm: the run's pooled
    noise model (prepare_run) instead of this image's own fit"""
    prior = prior or FD.Prior(binning=BINNING)
    ns = PL.noise_stage(Q, valid, gain=1.0)
    if nm is not None:
        ns["nm"] = nm
    cand = NZ.candidates(ns["v"], ns["kap"], ns["nm"], valid, prior.r_min)
    use = valid & ~exclude if exclude is not None else valid
    fc_all = float(cand.sum() / max(int(valid.sum()), 1))
    fc = float((cand & use).sum() / max(int(use.sum()), 1))
    H, W = valid.shape
    nby, nbx = -(-H // BLOCK), -(-W // BLOCK)
    blk_fc = np.full((nby, nbx), np.nan)
    blk_reg = np.empty((nby, nbx), object)
    for i in range(nby):
        for j in range(nbx):
            sl = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
            nvb = int(use[sl].sum())
            if nvb >= MIN_BLOCK_PIX:
                blk_fc[i, j] = (cand[sl] & use[sl]).sum() / nvb
                blk_reg[i, j] = regime_of(blk_fc[i, j])
            else:
                blk_reg[i, j] = regime_of(fc)
    return dict(ns=ns, cand=cand, s=float(ns["nm"].core), f_c=fc, f_c_all=fc_all, blk_fc=blk_fc, blk_reg=blk_reg,
                prior=prior)


def _block_regime(m, y, x):
    """regime of the block holding pixel (round(y), round(x)) (A_GMX convention: pixel i centred at i)"""
    i = min(max(int(round(y)) // BLOCK, 0), m["blk_reg"].shape[0] - 1)
    j = min(max(int(round(x)) // BLOCK, 0), m["blk_reg"].shape[1] - 1)
    return m["blk_reg"][i, j]


def fc_agmx():
    """FC_AGMX per bin: in binned images the soft candidates are mostly single dark electrons and the fake budget per
    bin is b_y x b_x times larger: FC_AGMX x sqrt(b_y b_x), at most 0.1 (neighbourhoods start to percolate); FC_AGMX
    itself for unbinned images"""
    return FC_AGMX if BINNING == (1, 1) else min(FC_AGMX * math.sqrt(BINNING[0] * BINNING[1]), 0.1)


def lam_2e():
    """LAM_2E per bin: the same chance rate of dark-charge 2-e- patterns per Mpix of UNBINNED area (~ lam_bin^2 /
    (b_y b_x)) as LAM_2E per pixel, i.e. LAM_2E x sqrt(b_y b_x); LAM_2E itself for unbinned images"""
    return LAM_2E * math.sqrt(BINNING[0] * BINNING[1])


def _halo_bins():
    """the finder's halo radius in bins: HALO unbinned pixels, at least 1.5 bins"""
    return HALO if BINNING == (1, 1) else max(HALO / max(BINNING), 1.5)


def bayes_catalogue(m, valid, lam_map=None, R_N=None):
    """neighbourhoods of the image with posteriors; single-image dark charge and event rates when not given"""
    ns, prior = m["ns"], m["prior"]
    if lam_map is None:
        nb, pt = FD.analyse_image(ns["v"], ns["kap"], ns["nm"], valid, 1e-5, prior, only_pix=True, big_halo=_halo_bins())
        lam_map = PL._init_lam(nb, pt, valid)
    cand = m["cand"]
    nb, pt = FD.analyse_image(ns["v"], ns["kap"], ns["nm"], valid, lam_map, prior, cand=cand, big_halo=_halo_bins())
    if R_N is None:
        R_N = PL.population_em([nb], float(valid.sum()), prior.pN)
    FD.posterior(nb, R_N)
    return nb, lam_map, R_N


def p_ge3(d):
    """P(N >= 3 | one event) of a neighbourhood"""
    pn = d.get("pN")
    return float(np.sum(pn[3:])) if pn is not None else 0.0


def operating_threshold(nb, area, F=F_FAKE):
    """P(event) threshold t: the largest selection whose summed P(no event) stays <= F x area [Mpix]
    (nb: the neighbourhoods eligible for the catalogue)"""
    p = np.sort(np.array([d["p_ev"] for d in nb if "p_ev" in d and not d.get("big") and not d.get("halo")]))[::-1]
    if not p.size:
        return 1.0
    # the budget is per Mpix of UNBINNED area: a bin covers BINNING[0] x BINNING[1] pixels of the CCD
    k = int(np.searchsorted(np.cumsum(1.0 - p), F * area * BINNING[0] * BINNING[1] / 1e6, side="right"))
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
    # the finder's centre is continuous with pixel i = [i, i + 1); A_GMX (and panaSKImg) centre pixel i at i
    return dict(E=d["N_mean"] * e_per_e, y=d["y"] - 0.5, x=d["x"] - 0.5, sxy=sp, cls="few-e", regime="bayes",
                finder="bayes", occ=np.nan, p_ev=d["p_ev"], logJ1=d.get("logJ1", np.nan), N_mean=d["N_mean"],
                N_map=d["N_map"], sigma_post=sp, edge=False, blend=0.0, ped=0.0, peakfrac=1.0, chi2=np.nan,
                sig=d.get("logJ1", np.nan), _pix=(pr.astype(int), pc.astype(int), q.astype(float)))


def _inside(mask, y, x):
    H, W = mask.shape
    r, c = int(round(y)), int(round(x))
    return 0 <= r < H and 0 <= c < W and bool(mask[r, c])


def apply(Q, valid, clusters, info, e_per_e, mode="auto", lam_map=None, R_N=None, return_pixels=True, threshold=None,
          nm=None):
    """the switch on one image after A_GMX: returns (clusters, info) with info['fewe'].
    lam_map, R_N, threshold, nm: the run's dark-charge map of this image, event rates, P(event) cut and pooled noise
    model (prepare_run(..., full=True)); each None: this image's own estimate (single-image mode: its own noise fit
    keeps a ~1 / n_pixels excursion weight and its own fake budget F_FAKE x its Bayes area - both strict on small
    images and on images with many events, see switch_validation/VALIDATION.md)"""
    rec = dict(mode=mode)
    if info.get("regime") == "unmodelled":
        rec.update(used=False, reason="A_GMX regime unmodelled (not electron-counting)")
        info["fewe"] = rec
        return clusters, info
    if mode != "on" and info.get("crowded_mode"):
        rec.update(used=False, reason="A_GMX crowded mode")
        info["fewe"] = rec
        return clusters, info
    foot, big = big_footprint(clusters, e_per_e, valid.shape)
    pad = ndimage.binary_dilation(foot, iterations=BIG_PAD) if BIG_PAD and foot.any() else foot
    m = measure(Q, valid, exclude=pad, nm=nm)
    rec.update(s=m["s"], f_c=m["f_c"], f_c_all=m["f_c_all"], n_big=len(big),
               blocks={r: int((m["blk_reg"] == r).sum()) for r in ("bayes", "both", "agmx")})
    if mode == "on":
        m["blk_reg"][:] = "bayes"
    if not (m["blk_reg"] != "agmx").any():
        rec.update(used=False, reason="f_c above the cut everywhere")
        info["fewe"] = rec
        return clusters, info
    nb, lam_map, R_N = bayes_catalogue(m, valid, lam_map, R_N)
    lam = float(np.median(np.broadcast_to(lam_map, valid.shape)[valid]))
    only3 = lam >= lam_2e()
    # Bayes region: valid pixels of non-'agmx' blocks farther than HALO from every big-deposit pixel
    for d in nb:
        if d.get("big"):
            foot[d["cand"]] = True
    H, W = valid.shape
    blk = np.repeat(np.repeat(m["blk_reg"] != "agmx", BLOCK, 0), BLOCK, 1)[:H, :W]
    # HALO in unbinned pixels (distance with the bin size per axis), the bins next to a deposit always included
    halo = ((ndimage.distance_transform_edt(~foot, sampling=BINNING) < HALO) | ndimage.binary_dilation(foot, np.ones((3, 3), bool))
            if foot.any() and BINNING != (1, 1) else
            ndimage.distance_transform_edt(~foot) < HALO if foot.any() else np.zeros((H, W), bool))
    region = valid & blk & ~halo
    area = float(region.sum())
    # neighbourhoods in B and their catalogue
    inB = [("p_ev" in d and not d.get("big") and not d.get("halo") and _inside(region, d["y"] - 0.5, d["x"] - 0.5))
           for d in nb]
    elig = [d for d, b in zip(nb, inB) if b and (not only3 or p_ge3(d) >= 0.5)]
    t = operating_threshold(elig, area) if threshold is None else float(threshold)
    # ownership of the A_GMX clusters: the neighbourhood they touch, else their own centre
    lab = np.zeros((H, W), np.int32)
    for k, d in enumerate(nb):
        lab[d["cand"]] = k + 1
    if LINK > 0 and len(nb):
        lab = ndimage.grey_dilation(lab, size=(2 * LINK + 1, 2 * LINK + 1))
    out = []
    n_drop = 0
    for c in clusters:
        if c["E"] / e_per_e > BIG_E or not np.isfinite(c["E"]):
            out.append(dict(c, finder="agmx"))
            continue
        p = c.get("_pix")
        if p is not None and len(p[0]):
            r, cc = np.asarray(p[0], int), np.asarray(p[1], int)
        else:
            r, cc = np.array([int(round(c["y"]))]), np.array([int(round(c["x"]))])
        ok = (r >= 0) & (r < H) & (cc >= 0) & (cc < W)
        ks = np.unique(lab[r[ok], cc[ok]])
        ks = ks[ks > 0] - 1
        mine = any(inB[k] for k in ks) if ks.size else _inside(region, c["y"], c["x"])
        if mine:
            n_drop += 1                     # replaced by the Bayesian catalogue
            continue
        out.append(dict(c, finder="agmx"))
    n_add = 0
    for d in elig:
        if d["p_ev"] < t:
            continue
        bc = _bayes_cluster(d, e_per_e, m["ns"]["v"])
        if not return_pixels:
            bc.pop("_pix")
        out.append(bc)
        n_add += 1
    rec.update(used=True, lam=lam, R=float(np.sum(R_N)), threshold=t, threshold_from="image" if threshold is None else "run",
               pooled_nm=nm is not None, area_bayes=area, n_agmx_replaced=n_drop,
               n_bayes=n_add, only_N_ge_3=bool(only3), pooled_lam=np.ndim(lam_map) > 0)
    info["fewe"] = rec
    return out, info


def run_threshold(nbs, area, lam, F=F_FAKE):
    """the analytic operating point of a whole run: P(event) cut whose selection over all images (neighbourhoods not
    big / halo; with lam >= LAM_2E only P(N >= 3) >= 0.5) has summed P(no event) <= F x area [Mpix]"""
    only3 = lam >= lam_2e()
    elig = [d for nb in nbs for d in nb if "p_ev" in d and not d.get("big") and not d.get("halo")
            and (not only3 or p_ge3(d) >= 0.5)]
    return operating_threshold(elig, area, F)


def prepare_run(images, prior=None, full=False, **kw):
    """pooled noise model, dark charge and event rates for the images of one run setting (pipeline.process_group,
    nm_pooled=True unless given): [(Q, valid)] -> (lam_maps, R_N); with full=True a dict(lam_maps, R_N, threshold,
    nm, lam) - threshold = the run's P(event) cut (run_threshold over all images). For image k pass
    adaptive_cluster(..., fewe='auto', fewe_lam=lam_maps[k], fewe_RN=R_N, fewe_threshold=threshold, fewe_nm=nm)
    (the recommended run-pooled mode; switch_validation/VALIDATION.md)"""
    kw.setdefault("nm_pooled", True)
    res = PL.process_group(images, prior or FD.Prior(binning=BINNING), gain=1.0, **kw)
    if not full:
        return res["lam_maps"], res["R_N"]
    lam = float(np.median(np.concatenate([np.asarray(m)[v] for m, (_, v) in zip(res["lam_maps"], images)])))
    return dict(lam_maps=res["lam_maps"], R_N=res["R_N"], threshold=run_threshold(res["nbs"], res["area"], lam),
                nm=res["noise"][0]["nm"], lam=lam)
