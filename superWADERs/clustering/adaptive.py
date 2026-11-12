"""Adaptive clustering: choose the method per region of the image from what the region looks like.

Lessons of the 2026-10-23/24 study (docs/notes/CONTEXT_2026-10-24.md):
  - sparse regions (isolated X-rays, tracks, few-electron events): MHC is best (unbiased energies, close X-rays
    resolved, tracks joined, few-electron events without fakes, labels)
  - crowded regions (most pixels hold charge, clusters touch): WADERs-like hysteresis on the image is best; the
    matched filter + disks of MFD+G / MHC mis-measure widths and energies when neighbours overlap

Steps
  1. background map (AM.background_map; simple fallback when no row can be fitted)
  2. global check: if the fitted background spread is >= excess_ratio x the read noise (and >= 0.5 e-), the
     background is not dark current + read noise (e.g. PD06/DM12 Am-241 images: bg spread 1.5-2.9 e- vs 0.37 e-
     read noise; MFD+G-type energies there come out 5-10 % high) -> regime 'unmodelled' for the whole image:
     hysteresis with sigma = read noise (WADERs-like, never below the Poisson-tail threshold of the dark charge)
  3. event occupancy map: fraction of active pixels above max(1.5 e-, 4 x read noise, Poisson tail of the dark
     charge at 1e-3) in a box of `box` px - pixels that hold EVENT charge, not dark current (high-dark-current
     Fe-55 CCDs have 50 % of pixels > 1.5 e- but only ~5 % above the tail; MHC is right for them)
  4. regime: 'crowded' where occupancy >= occ_thr, else 'sparse' (both dilated by `margin` px for the clustering
     so that clusters on the boundary are complete; each cluster is kept by the regime of its centre)
  5. sparse -> MHC on the sparse part; crowded -> hysteresis clustering in electrons on the crowded part:
     pixel > 10 sigma_crowd, seed > 30 sigma_crowd, energy = sum of the core pixels (as WADERs).
     sigma_crowd = the WADERs / panaSKImg pedestal sigma of the CCD when given (on a crowded CCD it is inflated
     by the charge, ~8 e- on the Am-241 EXT4 CCD, which is what makes WADERs work there), else 8 e-
     (pixel > 80 e-, seed > 240 e-). Tuned on the crowded Am-241 CCD (runs_29img EXT4): a threshold scan
     10-160 e- peaks at 80 e- (Np L-line significance); thresholds relative to the local median and
     energies over extra rings were both worse (adaptive_crowded_scan_am241.py)
  6. every cluster gets 'regime', 'occ' and (from MHC) 'cls'; crowded clusters get cls 'x-ray' / 'x-ray diffuse' /
     'big deposit' / 'extended' from their energy and size
  7. engine 'gmx', crowded CCDs (crowded.py): with crowded='auto' (default; $AGMX_CROWDED overrides) crowded mode
     is on when the core fill (fraction of pixels whose smoothed charge is above max(60 e-, 5 noise)) is >= 0.10:
     regional background override, hot-column bands, serial-register streaks removed (clusters 'artifact-row', regime 'serial'),
     straight tracks found and passed to GMX as hints (their clouds -> one cluster, cls 'muon'), clusters that
     cross as an X split, collinear track fragments linked. Other images are clustered exactly as before.
     With crowded_shapes='on' the straight tracks are line components of the mixture instead of hints, and the clusters
     get a topology split (crowded_shapes.py).
"""
import os, sys
import numpy as np
from scipy import ndimage
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import agent_methods as AM
import cluster_methods as CM
import mhc as MH
EIGHT = np.ones((3, 3), bool)


simple_bg = AM.simple_bg


def occupancy(Q, valid, noise, box=31, lam=0.0, bg=None):
    """fraction of active pixels holding EVENT charge in a box: above max(1.5 e-, 4 x read noise, Poisson tail
    (1e-3) of the dark charge). With bg the tail uses the per-pixel dark-charge map (dark charge often grows along
    the rows; the image median would count the dark pixels of the high-dark-charge rows as events)."""
    if bg is not None:
        thr = np.maximum(np.maximum(1.5, 4 * noise), AM.tail_threshold_map(bg, 1e-3) - bg.o)
    else:
        thr = max(1.5, 4 * noise, AM.poisson_tail_k(float(lam), 1e-3)[0] - 0.5)
    num = ndimage.uniform_filter(((Q > thr) & valid).astype(float), box, mode="constant")
    den = ndimage.uniform_filter(valid.astype(float), box, mode="constant")
    return np.where(valid, num / np.maximum(den, 1e-3), 0.0)


def hysteresis(Q, region, sigma, k_pix, n_pix=10.0, n_seed=30.0, seed_e=9.5):
    """WADERs-like clusters inside `region` with thresholds in electrons"""
    tp, ts = max(n_pix * sigma, k_pix - 0.5), max(n_seed * sigma, seed_e)
    lab, n = ndimage.label((Q > tp) & region, structure=EIGHT)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == k; q = Q[sl][m]
        if q.max() <= ts:
            continue
        ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
        my, mx, sxy = CM.moments(Q, ys, xs)
        E = float(q.sum() * AM.E_PER_E)
        ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        cls = "big deposit" if E > 75 else ("extended" if ext >= 10 else ("x-ray diffuse" if sxy >= 1 else "x-ray"))
        out.append(dict(E=E, y=float(my), x=float(mx), sxy=float(sxy), npix=int(m.sum()), cls=cls, qmax=float(q.max()),
                        _pix=(ys, xs, q)))           # pixels and their charge [e-] (sums to E)
    return out


CROWDED = os.environ.get("AGMX_CROWDED", "auto")      # default of adaptive_cluster(crowded=...): auto | on | off
CROWDED_SHAPES = os.environ.get("AGMX_CROWDED_SHAPES", "on")    # in crowded mode: muon lines + topology split: on | off
FEWE = os.environ.get("AGMX_FEWE", "off")    # Bayesian few-electron finder switch (wip/lowdc/bayes): off | auto | on


def adaptive_cluster(Q, valid, bg=None, sigma=None, sigma_crowd=None, occ_thr=0.3, box=31, margin=8, excess_ratio=3.0,
                     return_info=False, engine="mhc", return_pixels=False, noise_max=1.0, crowded=None, crowd_fill=None, crowded_shapes=None,
                     fewe=None, fewe_lam=None, fewe_RN=None, fewe_threshold=None, fewe_nm=None, **kw):
    """engine='mhc' : unmodelled -> hysteresis; else per region crowded -> hysteresis, sparse -> MHC.
    engine='gmx' : unmodelled -> hysteresis; else GMX on the whole image (GMX shares charge by model responsibility,
                   so it needs no crowded / sparse split; occupancy is still reported per cluster). A_GMX.

    Q: image [e-]; valid: active pixels; bg: AM.background_map (computed if None)
    sigma: READ noise [e-] (estimated from the quiet pixels if None); used by the 'unmodelled' switch
    sigma_crowd, margin: engine 'mhc' only (pedestal sigma of the crowded-region hysteresis, region overlap)
    **kw: passed to MH.mhc (engine 'mhc') or GX.general_cluster (engine 'gmx')
    return_pixels: engine 'gmx' (and 'unmodelled'): every cluster gets '_pix' = (rows, cols, charge [e-]) summing to E
    crowded: engine 'gmx': 'auto' | 'on' | 'off' (default $AGMX_CROWDED, else 'auto') - crowded mode (step 7) when the core fill
             >= crowd_fill (default crowded.CROWD_FILL = 0.10) / always / never. info: core_fill, crowded_mode,
             bg_override, bg_columns (reports), n_streaks, n_hints, n_linked, n_glow (cls 'glow': large faint clusters)
    crowded_shapes: 'on' | 'off' (default $AGMX_CROWDED_SHAPES, else 'on'): in crowded mode the straight tracks become line
             components of the mixture and the clusters get the topology split (crowded_shapes.py) instead of the
             hint grouping. info: crowded_shapes, n_lines, n_topology_splits
    fewe: engine 'gmx' only: 'off' | 'auto' | 'on' (default $AGMX_FEWE, else 'off'): the Bayesian few-electron finder
             (wip/lowdc/bayes/agmx_switch.py, validation: wip/lowdc/switch_validation/VALIDATION.md): after GMX, in
             sparse regions (soft-candidate fraction f_c <= 1.5e-2 per 100 x 100 block, counted outside the big deposits;
             'on': everywhere unless unmodelled) the clusters of <= 20 e- farther than 15 px from every pixel of a bigger
             one are replaced by the Bayesian catalogue (cls 'few-e', regime 'bayes', finder 'bayes', p_ev, N_mean,
             sigma_post); kept clusters get finder 'agmx'. Crowded mode and 'unmodelled' images stay A_GMX. fewe_lam /
             fewe_RN / fewe_threshold / fewe_nm: the run-pooled dark-charge map of this image, event rates, P(event) cut
             and noise model (agmx_switch.prepare_run(..., full=True): lam_maps[k], R_N, threshold, nm); without them
             single-image estimates and this image's own fake budget are used (conservative). info: fewe (decision record)
    Each cluster: E [keV], y, x, sxy, cls, regime ('sparse' / 'crowded' / 'unmodelled'), occ; with engine 'gmx'
    also the quality flags edge, blend, ped [e-/pix], peakfrac, chi2, sig (gmx.fill_flags for clusters that were
    not fitted by the mixture: blend 0, peakfrac 1, chi2 NaN)."""
    if engine not in ("mhc", "gmx"):
        raise ValueError(f"adaptive_cluster: unknown engine {engine!r} (use 'mhc' or 'gmx')")
    if return_pixels and engine != "gmx":
        raise ValueError("adaptive_cluster: return_pixels needs engine='gmx' (MHC clusters have no pixel lists)")
    if sigma is None:
        sigma = AM.read_noise(Q, valid)
    if bg is None:
        try:
            bg = AM.background_map(Q, valid)
        except Exception:
            bg = simple_bg(Q, valid, sigma)
    info = dict(sigma=sigma, sigma_crowd=sigma_crowd)
    streak_cl, hints = [], None
    cmode = (crowded or CROWDED).lower()
    if cmode not in ("auto", "on", "off"):
        raise ValueError(f"adaptive_cluster: crowded={cmode!r} (use 'auto', 'on' or 'off')")
    if engine == "gmx" and cmode != "off":
        import crowded as CR
        fill = CR.core_fill(Q, valid, bg)
        on = cmode == "on" or fill >= (CR.CROWD_FILL if crowd_fill is None else crowd_fill)
        info.update(core_fill=fill, crowded_mode=bool(on))
        if on:
            bg, info["bg_override"] = CR.regional_background(Q, valid, bg)
            bg, info["bg_columns"] = CR.column_background(Q, valid, bg)
            hints = CR.find_hints(Q, valid, bg)
            st, streak_cl = CR.find_streaks(Q, valid, bg)
            valid = valid & ~st
            info.update(n_hints=len(hints), n_streaks=len(streak_cl))
    noise = float(np.median(bg.s[valid])); lam = float(np.median(bg.lam[valid]))
    k = max(int(AM.poisson_tail_k(lam, 0.10)[0]), 2)
    info.update(noise=noise, lam=lam)
    # 'unmodelled': the background is not dark current + read noise (bg spread >= excess_ratio x read noise), or the
    # image is not electron-counting skipper data (read noise >= noise_max e-, physics audit 2026-10-25): with a few
    # e- of noise per pixel the GMX energy (sum over the whole footprint) collects sqrt(Npix) x noise and the lines
    # broaden (59.5 keV width +40 % on PD06/DM12), so such images are clustered on Q with WADERs-like thresholds
    if (noise >= 0.5 and noise >= excess_ratio * sigma) or (AM.PHYS_FIX and sigma >= noise_max):
        # the background model is not trusted here (its fitted offset can be off by several e-), so cluster the
        # image itself, as WADERs does
        # thresholds as before the audit: 10 / 30 x the width of the zero-electron peak (|Q| < 0.5), which saturates at
        # 0.37 e- (pixel > ~3.7 e-, seed > ~11 e-, i.e. ~1 / 3 x the true noise on PD06/DM12); 10 / 30 x the true noise
        # would cut the cluster tails (choosing thresholds for noisy images is an open item)
        act = Q[valid]; z = act[np.abs(act) < 0.5]
        s_thr = float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 100 else sigma
        # the dark-charge term of the pixel threshold uses lam from the background model, which is not trusted here:
        # fixed at k = 5 (pixel > 4.5 e-), what the earlier code used on the PD06/DM12 images (lam ~ 1.8 there);
        # 3.7 e- (k = 2) and 5.5 e- (k = 6) were both worse (Fe-55 injections 0.81 / 0.88 vs 0.93). Open item.
        k_unm = 5 if AM.PHYS_FIX else k
        out = [dict(c, regime="unmodelled", occ=np.nan) for c in hysteresis(np.where(valid, Q, 0.0), valid, s_thr, k_unm)]
        if engine == "gmx":
            import gmx as GX
            GX.fill_flags(out, np.where(valid, Q, 0.0), valid)   # no trusted background here: ped from the image
        info.update(regime="unmodelled", crowded_frac=np.nan)
        return (out, info) if return_info else out
    occ = occupancy(Q, valid, sigma, box, lam, bg)
    if engine == "gmx":
        import gmx as GX
        fmode = (fewe or FEWE).lower()
        if fmode not in ("off", "auto", "on"):
            raise ValueError(f"adaptive_cluster: fewe={fmode!r} (use 'off', 'auto' or 'on')")
        # the switch needs the pixel lists (footprints of the big deposits, ownership); dropped again below
        rpix = return_pixels or fmode != "off"
        out = []
        cm = info.get("crowded_mode", False)
        shapes = cm and (crowded_shapes or CROWDED_SHAPES).lower() == "on"
        info["crowded_shapes"] = bool(shapes)
        if shapes:
            import crowded_shapes as CS
            res, inf2 = CS.shape_cluster(Q, valid, bg, hints, return_pixels=True, return_info=True, **kw)
            info.update(n_lines=inf2["n_lines"], n_topology_splits=inf2["n_topology_splits"])
        elif cm:
            res = GX.general_cluster(Q, valid, bg, return_pixels=True, split_crossings=True, hints=hints, **kw)
        else:
            res = GX.general_cluster(Q, valid, bg, return_pixels=rpix, **kw)
        for c in res:
            y, x = int(round(c["y"])), int(round(c["x"]))
            o = float(occ[y, x]) if 0 <= y < Q.shape[0] and 0 <= x < Q.shape[1] else np.nan
            reg = c["regime"] if c.get("regime") == "line" else ("crowded" if o >= occ_thr else "sparse")
            out.append(dict(c, regime=reg, occ=o))
        if cm:
            out = out + [dict(c, occ=np.nan) for c in streak_cl]
            out, info["n_linked"] = CR.link_tracks(out)
            info["n_glow"] = CR.glow_flag(out, bg)
            if not rpix:
                for c in out: c.pop("_pix", None)
        info.update(regime="gmx", occ=occ, crowded_frac=float(((occ >= occ_thr) & valid).sum() / max(valid.sum(), 1)))
        if fmode != "off":
            import agmx_switch as SW
            out, info = SW.apply(Q, valid, out, info, AM.E_PER_E, fmode, fewe_lam, fewe_RN, return_pixels, fewe_threshold,
                                 fewe_nm)
            if not return_pixels:
                for c in out: c.pop("_pix", None)
        return (out, info) if return_info else out
    crowded = (occ >= occ_thr) & valid
    sparse = valid & ~crowded
    out = []
    if sparse.any():
        vs = valid & ~ndimage.binary_erosion(crowded, structure=EIGHT, iterations=margin)
        for c in MH.mhc(np.where(vs, Q, bg.mean), vs, bg, **kw):
            y, x = int(round(c["y"])), int(round(c["x"]))
            if 0 <= y < Q.shape[0] and 0 <= x < Q.shape[1] and sparse[y, x]:
                out.append(dict(c, regime="sparse", occ=float(occ[y, x])))
    if crowded.any():
        reg = ndimage.binary_dilation(crowded, structure=EIGHT, iterations=margin) & valid
        sc = float(sigma_crowd) if sigma_crowd else 8.0
        for c in hysteresis(np.where(valid, Q - bg.mean, 0.0), reg, sc, k, seed_e=0.0):   # dark charge subtracted
            y, x = int(round(c["y"])), int(round(c["x"]))
            if crowded[y, x]:
                out.append(dict(c, regime="crowded", occ=float(occ[y, x])))
    if return_info:
        info.update(regime="mixed" if crowded.any() and sparse.any() else ("crowded" if crowded.any() else "sparse"),
                    occ=occ, crowded_frac=float(crowded.sum() / max(valid.sum(), 1)))
        return out, info
    return out
