""":mod:`pysimdamicm.processes.comm_clustering`
    Additions to ClusterFinder from comm_scripts:

    - ``auto_spr_threshold``: pixel and seed thresholds in electrons chosen from the dark charge of the CCD
      (ClusterFinder ``spr_threshold_auto``), the rule of comm_scripts' autoconfig_panaSKImg
      --cluster-threshold-e auto;
    - ``find_clusters_agmx``: A_GMX clustering (ClusterFinder ``algorithm: "agmx"``): a model of the image
      (background, Gaussian charge clouds, in crowded mode line components for straight tracks) whose clusters
      share every pixel's charge (pysimdamicm/comm/adaptive.py engine 'gmx'). Its clusters become the same
      Cluster objects as the WADERs clusters, with the extra agmx_* branches of comm_scripts' agmx_root.py.

.. moduleauthor:: comm_scripts (D. Rosenmerkel)
"""
import copy
import json
import math
import os

import numpy as np

from pysimdamicm.io.data_formats import Cluster
from pysimdamicm.utils.units import Units
u = Units()

AUTO_ALPHA = 0.10       # allowed fraction of dark-current pixels above the pixel threshold
AUTO_SEED_E = 9.5       # seed threshold [e-] (a cluster needs a pixel with >= 10 e-)
MAX_SIG_OVER_GAIN = 0.3 # electron peaks resolved if the pixel noise is below this [e-] (autoconfig's limit)

AGMX_DEFAULTS = dict(crowded="auto", crowded_shapes="on", fe_candidates=True, temperature=None, first_row=None)


def _active_box(hits, amp):
    """rows, cols (slices) of the active region of one amplifier"""
    act = np.logical_not(hits.amplifier[amp]['mask_image_active_region'])
    rows, cols = np.where(act.any(axis=1))[0], np.where(act.any(axis=0))[0]
    return act, slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _calibrated_image(hits, image_attr):
    img = getattr(hits, hits.get_image_attribute_name(image_attr))
    return np.asarray(np.ma.getdata(img), float)


def row_dark_charge(e_img):
    """Dark charge per row [e-/pix] from the zero-electron share, as autoconfig_panaSKImg: pixels rounded to
    electrons, the lowest well-populated value is the zero peak, lambda = -ln P(0) of the pixels with 0-49 e-."""
    lam = np.full(e_img.shape[0], np.nan)
    for r, row in enumerate(e_img):
        n = np.round(row)
        vals, cnt = np.unique(n, return_counts=True)
        ok = vals[cnt >= max(3, 0.05 * n.size)]
        if ok.size == 0:
            continue
        e = n - ok.min()
        p0 = np.sum(e == 0) / max(np.sum((e >= 0) & (e < 50)), 1)
        lam[r] = -math.log(max(p0, 1e-6))
    return lam


def pixel_noise(e_img):
    """Pixel noise [e-] from the visibility of the electron comb (0-5 e- pixels): R = exp(-2 pi^2 s^2)."""
    v = e_img[(e_img > -0.5) & (e_img < 5.5)]
    if v.size < 100:
        return np.inf
    R = abs(np.mean(np.exp(2j * np.pi * v)))
    return math.sqrt(-math.log(R) / (2 * math.pi ** 2)) if R > 1e-6 else np.inf


def auto_spr_threshold(hits, amp, image_attr, alpha=AUTO_ALPHA, seed_e=AUTO_SEED_E):
    """[1.0, k - 0.5, max(seed_e, k + 0.5)] (sigma = 1 e-, so thresholds in electrons) with k the smallest number
    of electrons >= 2 with P(Poisson(lambda) >= k) <= alpha, lambda = median per-row dark charge of the active
    region; None if the electron peaks are not resolved (then the configured spr_threshold is kept).
    Returns (threshold, lambda, k, noise)."""
    from scipy.stats import poisson
    act, rs, cs = _active_box(hits, amp)
    e = _calibrated_image(hits, image_attr)[rs, cs]
    noise = pixel_noise(e)
    if not np.isfinite(noise) or noise >= MAX_SIG_OVER_GAIN:
        return None, float("nan"), None, noise
    lam = float(np.nanmedian(row_dark_charge(e)))
    if not np.isfinite(lam):
        return None, lam, None, noise
    k = 2
    while poisson.sf(k - 1, lam) > alpha and k < 50:
        k += 1
    return [1.0, k - 0.5, max(seed_e, k + 0.5)], lam, k, noise


def find_clusters_agmx(finder, hits, opts):
    """A_GMX on the calibrated image of the current amplifier; appends Cluster objects to hits.evt_clusters (and
    the other per-cluster lists ClusterFinder keeps). opts: ClusterFinder.agmx merged with AGMX_DEFAULTS.
    With GainNonlinearityProcess the image is first converted to electrons with g(n) (as run_WADERs
    --gain-correction --clusterer agmx). Returns the crowded-mode record of the amplifier."""
    if opts.get("temperature") not in (None, "", -1):
        os.environ["COMM_TEMPERATURE_K"] = str(opts["temperature"])    # read by the comm modules (epsilon(T))
    from pysimdamicm.comm import adaptive as AD
    from pysimdamicm.comm import agmx_root as AR
    amp = hits.execute_process_in_amp
    cal = _calibrated_image(hits, finder.spr_image)
    gain = u._tohdr.get(f"GAIN_{amp}", (None,))[0]
    if gain is None:
        gain = hits.ADC2e[amp] if hasattr(hits, 'ADC2e') else u.ADC2e
    act, rs, cs = _active_box(hits, amp)
    # active pixels as in comm_scripts' agmx_root.valid_mask: rows from first_row + 3 (first_row: the JSON's
    # input.image.id_row_start; default the first active row) and a 2-px border left out
    r0 = rs.start if opts.get("first_row") in (None, "") else int(opts["first_row"])
    valid = np.zeros(cal.shape, bool)
    valid[r0 + 3:rs.stop - 2, cs.start + 2:cs.stop - 2] = True
    valid &= act
    curve = getattr(hits, 'gain_curve', {}).get(amp)
    if curve is not None:
        from pysimdamicm.processes.gain_nonlinearity import electrons
        Q = electrons(cal * gain, curve)
    else:
        Q = cal
    Q = np.where(valid, Q, 0.0)
    kw = dict(crowded=str(opts["crowded"]), crowded_shapes=str(opts["crowded_shapes"]))
    AD.CROWDED, AD.CROWDED_SHAPES = kw["crowded"], kw["crowded_shapes"]     # also what the crowded-mode record reports
    try:
        bg = AD.AM.background_map(Q, valid)
    except Exception:
        bg = None
    print(f"Process <ClusterFinder> A_GMX on {amp} (crowded {kw['crowded']}, shapes {kw['crowded_shapes']}"
          + (", gain-corrected image" if curve is not None else "") + ")")
    clusters, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, **kw)
    clusters = [c for c in clusters if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
    if opts.get("fe_candidates") and bg is not None and info.get("regime") != "unmodelled" and info.get("noise", 9) < 0.35:
        from pysimdamicm.comm import fe_candidates as FE
        cands, summ = FE.candidates(Q, valid, bg, clusters)
        out = getattr(hits, "output", "") or ""
        summ.update(source_image=os.path.basename(getattr(hits, "_file_name", "")), ext=str(amp),
                    gain_corrected=curve is not None)
        FE.write(cands, summ, os.path.join(os.path.dirname(out) or ".", "fe_candidates",
                                           os.path.basename(out) + str(amp)))
    _append_clusters(finder, hits, amp, clusters)
    rec = dict(method="A_GMX (pysimdamicm/comm/adaptive.py engine 'gmx')", version=AR.AGMX_VERSION,
               gain_corrected=curve is not None, gain_anchor=curve["anchor"] if curve is not None else None,
               parameters=kw, cls_codes=AR.CLS_CODES, regime_codes=AR.REGIME_CODES, not_computed=AR.NOT_COMPUTED,
               crowded_mode=AR.crowded_summary(info), n_clusters=len(clusters),
               recommended_selection="agmx_blend < 0.02 and agmx_ped < 10 (and fSTD/sigma cut of the analysis)")
    if not hasattr(hits, "agmx_config"):
        hits.agmx_config = {}
    hits.agmx_config[str(amp)] = rec
    return rec


def _append_clusters(finder, hits, amp, clusters):
    """A_GMX clusters (with '_pix' = rows, cols, charge [e-]) -> Cluster objects, as agmx_root.build_clusters but
    with this image's own attributes; the per-cluster bookkeeping of ClusterFinder.get_cluster_object_list."""
    from pysimdamicm.comm import agmx_root as AR
    if not hasattr(hits, 'evt_clusters'):
        hits.evt_clusters, hits.Nclusters, hits.evt_clusters_in_amp = [], 0, []
        hits._PosX, hits._PosY, hits._hasseed = [], [], []
    if not hasattr(hits, 'me_reconstructed_spectrum'):
        hits.me_reconstructed_spectrum = []
    for i, c in enumerate(clusters, start=1):
        ys, xs, q = c["_pix"]
        if len(ys) == 0:
            continue
        h = copy.copy(hits)
        h.__attr_to_clusterize__ = "_pixels"
        h.x_pixels, h.y_pixels = np.asarray(xs, int), np.asarray(ys, int)
        h.z_pixels = -np.ones(len(ys))
        h.Edep_pixels = np.asarray(q, float) * u.e2eV
        cl = Cluster(i, h, np.ones(len(ys), bool))
        try:
            cl.get_cluster_properties(__DEBUG__=finder.__DEBUG__, get_fitted_STD=finder.get_fitted_STD)
        except Exception:
            continue
        cl.agmx_cls = int(AR.CLS_CODES.get(c.get("cls"), 0))
        for k in ("blend", "ped", "peakfrac", "chi2", "sig"):
            v = c.get(k, np.nan)
            setattr(cl, f"agmx_{k}", float(v) if v is not None and np.isfinite(v) else AR.NOT_COMPUTED)
        cl.agmx_regime = int(AR.REGIME_CODES.get(c.get("regime"), -1))
        hits.evt_clusters.append(cl)
        hits.Nclusters += 1
        cl.cluster_id = hits.Nclusters
        hits.evt_clusters_in_amp.append(amp)
        hits._PosX.append(cl.PosX)
        hits._PosY.append(cl.PosY)
        if finder.evaluate_event_mask:
            cl.mask_flags = finder.evaluate_cluster_mask(cl, hits.n_cols_prescan, hits.n_cols_overscan)
        try:
            hits._hasseed.append(cl.has_seed)
            hits.me_reconstructed_spectrum.append(cl.Energy)
        except AttributeError:
            pass


def agmx_config_json(hits):
    """the A_GMX record of all amplifiers (for the agmx_config TNamed of the cluster file)"""
    return json.dumps(getattr(hits, "agmx_config", {}))
