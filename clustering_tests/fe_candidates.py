"""Few-electron candidate catalogue: the low-background (DAMIC-M / SENSEI style) stage that runs after A_GMX.

A_GMX decides per event at a fixed fake rate (good for a cluster catalogue). A search for rare few-electron events
works differently: it keeps EVERY candidate, masks the regions where spurious charge is expected, and compares the
observed counts per class with the background expected from dark charge alone. This module produces the inputs for
that analysis, without deciding anything:

  candidates   every group of touching pixels with 2..s_max counted electrons (rounded Q - offset), <= m_max
               pixels: position, charge [e-], pixels, local dark charge and read noise, the null statistic
               T = -sum ln P(n_i | lam_i, s_i) - ln N_shapes(m) (as gmx._group_stats), n_exp = the expected number of
               dark-charge groups per image at least as unlikely (unmasked area; simulated blanks, analytic tail
               N_pix exp(-T) beyond them) and p_image = 1 - exp(-n_exp); mask flags; clean = no flag
  masks        (flags, nothing is removed)
               halo     within halo_px(E) of an A_GMX cluster of energy E (spurious charge around large deposits)
               trail    in the row or column of a cluster with E > trail_E_min, within trail_px (charge-transfer
                        inefficiency trails along the serial / vertical transfer)
               hotcol   column flagged hot (extra dark charge) or with a single-electron rate > hot_ratio x median
               edge     within edge_px of the mask edge
               cluster  overlaps the footprint of an A_GMX cluster (A_GMX already reports it)
  summary      isolated single-electron rate (dark charge + miscounts), unmasked area, observed clean candidates per
               class (charge, pixels) and the expected number from dark charge alone in the same area (simulated
               blanks of the image's own detection background; mean and Monte-Carlo error)

    import fe_candidates as FE
    cands, summary = FE.candidates(Q, valid, bg, agmx_clusters)        # Q [e-], bg = agent_methods.background_map
    FE.write(cands, summary, "out/stem_ext1")                           # stem_ext1_fe_candidates.csv / .json
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import agent_methods as AM  # noqa: E402
import gmx as GX  # noqa: E402

E_PER_E = AM.E_PER_E
EIGHT = np.ones((3, 3), bool)
VERSION = "2026-10-26"


def halo_px(E_keV):
    """halo radius [px] around a cluster of energy E: 5 px for small deposits, growing as sqrt(E), at most 60 px"""
    return float(np.clip(5.0 * np.sqrt(max(E_keV, 0.0)), 5.0, 60.0))


def masks(Q, valid, bg, clusters, edge_px=5, trail_E_min=1.0, trail_px=50, hot_ratio=5.0, hot_columns=()):
    """boolean maps (halo, trail, hotcol, edge, cluster) of the image shape; clusters: A_GMX clusters (dicts with y, x,
    E [keV] and optionally '_pix')"""
    H, W = Q.shape
    yy, xx = np.ogrid[:H, :W]
    halo = np.zeros((H, W), bool); trail = np.zeros((H, W), bool); clus = np.zeros((H, W), bool)
    for c in clusters:
        E = float(c.get("E", 0.0))
        if not np.isfinite(E):
            continue
        cy, cx = float(c["y"]), float(c["x"])
        if "_pix" in c and len(c["_pix"][0]):
            ys, xs = np.asarray(c["_pix"][0], int), np.asarray(c["_pix"][1], int)
            clus[ys, xs] = True
        else:
            clus[int(round(cy)), int(round(cx))] = True
        if E < 2.0 * 20 * E_PER_E:          # A_GMX clusters below ~20 e- are few-electron events themselves
            continue
        r = halo_px(E)
        y0, y1, x0, x1 = max(int(cy - r), 0), min(int(cy + r) + 1, H), max(int(cx - r), 0), min(int(cx + r) + 1, W)
        halo[y0:y1, x0:x1] |= (yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2 <= r * r
        if E >= trail_E_min:
            iy, ix = int(round(cy)), int(round(cx))
            trail[iy, max(ix - trail_px, 0):min(ix + trail_px + 1, W)] = True
            trail[max(iy - trail_px, 0):min(iy + trail_px + 1, H), ix] = True
    edge = ~ndimage.binary_erosion(valid, iterations=edge_px)
    # hot columns: flagged by A_GMX (extra dark charge) or with an isolated single-electron rate >> the median
    hot = np.zeros(W, bool)
    hot[list(hot_columns)] = True
    nn = np.where(valid, np.round(Q - bg.o), 0)
    one = (nn == 1) & valid & ~clus
    rate = one.sum(0) / np.maximum(valid.sum(0), 1)
    med = np.median(rate[valid.any(0)]) if valid.any() else 0.0
    nmin = 3.0 / max(valid.sum(0).max(), 1)
    hot |= rate > max(hot_ratio * med, nmin * hot_ratio)
    hotcol = np.zeros((H, W), bool); hotcol[:, hot] = True
    return dict(halo=halo & valid, trail=trail & valid, hotcol=hotcol & valid, edge=edge & valid, cluster=clus & valid)


def _groups(nn, valid, lam, s, s_min=2, s_max=20, m_max=9):
    """candidate groups: labels, ids, statistic T, charge S, pixel count m (groups with s_min <= S <= s_max, m <= m_max)"""
    lab, ids, T = GX._group_stats(nn, valid, lam, s, m_max=m_max, link=1)
    if ids.size == 0:
        return lab, ids, T, np.zeros(0), np.zeros(0, int)
    k = lab.max()
    S = np.bincount(lab.ravel(), np.where(nn > 0, nn, 0).ravel(), k + 1)[ids]
    m = np.bincount(lab.ravel(), minlength=k + 1)[ids]
    sel = (S >= s_min) & (S <= s_max)
    return lab, ids[sel], T[sel], S[sel], m[sel]


def candidates(Q, valid, bg, clusters=(), s_max=20, m_max=9, n_blanks=20, seed=2026, hot_columns=(), **mask_kw):
    """the few-electron candidate list and the per-image summary (see the module docstring). Q: image [e-], valid:
    active pixels, bg: background (agent_methods.background_map), clusters: the A_GMX clusters of the image."""
    det = GX.detection_bg(Q, valid, bg, mode="occupancy")       # accurate occupancy map (calibrated expectation)
    lam = np.where(valid, det.lam, 0.0); s = np.where(valid, det.s, 1.0)
    nn = np.where(valid, np.round(Q - bg.o), 0).clip(0)
    M = masks(Q, valid, bg, clusters, hot_columns=hot_columns, **mask_kw)
    masked = M["halo"] | M["trail"] | M["hotcol"] | M["edge"] | M["cluster"]
    clean_area = valid & ~masked
    npix_clean = int(clean_area.sum())
    lab, ids, T, S, m = _groups(nn, valid, lam, s, s_max=s_max, m_max=m_max)
    # background expectation: blanks simulated from the detection background, the same grouping, clean area only
    rng = np.random.default_rng(seed)
    blankT, cls_counts = [], {}
    for _ in range(n_blanks):
        q = np.where(valid, np.round(rng.poisson(lam) + rng.normal(0.0, 1.0, lam.shape) * s), 0).clip(0)
        lb, idb, Tb, Sb, mb = _groups(q, valid, lam, s, s_max=s_max, m_max=m_max)
        if idb.size:
            # a blank group counts when its pixels are all in the clean area (as for the real candidates)
            cnt_bad = np.bincount(lb.ravel(), (~clean_area).ravel().astype(float), lb.max() + 1)[idb]
            inside = cnt_bad == 0
            blankT.append(Tb[inside])
            for a, b in zip(Sb[inside], mb[inside]):
                cls_counts[(int(a), int(b))] = cls_counts.get((int(a), int(b)), 0) + 1
    blankT = np.sort(np.concatenate(blankT)) if blankT else np.zeros(0)
    # candidates
    objs = ndimage.find_objects(lab)
    cands = []
    for g, Tg, Sg, mg in zip(ids, T, S, m):
        sl = objs[g - 1]
        sub = lab[sl] == g
        ys, xs = np.nonzero(sub); ys = ys + sl[0].start; xs = xs + sl[1].start
        q = nn[ys, xs]
        flags = {k: bool(v[ys, xs].any()) for k, v in M.items()}
        # expected dark-charge groups per image at least as unlikely, in the clean area: simulated blanks, and the
        # analytic tail N_clean exp(-T) where the blanks have no group that unlikely
        n_mc = float(np.sum(blankT >= Tg)) / max(n_blanks, 1)
        n_exp = n_mc if n_mc > 0 else float(npix_clean * np.exp(-Tg))
        cands.append(dict(y=float((q * ys).sum() / q.sum()), x=float((q * xs).sum() / q.sum()), charge_e=int(Sg), npix=int(mg),
                          E_keV=float(Sg * E_PER_E), lam=float(lam[ys, xs].mean()), noise=float(s[ys, xs].mean()), T=float(Tg),
                          n_exp=n_exp, p_image=float(1.0 - np.exp(-n_exp)), clean=not any(flags.values()),
                          pixels=";".join(f"{a}:{b}:{int(c)}" for a, b, c in zip(ys, xs, q)), **{f"mask_{k}": v for k, v in flags.items()}))
    # summary
    iso = (nn == 1) & valid & ~masked
    nb = ndimage.uniform_filter((nn >= 1).astype(float), 5, mode="constant") * 25 - (nn >= 1)
    iso &= nb < 0.5
    obs = {}
    for c in cands:
        if c["clean"]:
            key = f"{c['charge_e']}e_{c['npix']}px"; obs[key] = obs.get(key, 0) + 1
    exp = {f"{a}e_{b}px": dict(mean=n / max(n_blanks, 1), mc_err=np.sqrt(n) / max(n_blanks, 1)) for (a, b), n in sorted(cls_counts.items())}
    summary = dict(version=VERSION, npix_active=int(valid.sum()), npix_clean=npix_clean,
                   masked_fraction={k: float((v & valid).sum() / max(valid.sum(), 1)) for k, v in M.items()},
                   single_e_rate=float(iso.sum() / max(clean_area.sum(), 1)),
                   lam_det_median=float(np.median(lam[valid])) if valid.any() else float("nan"),
                   noise_median=float(np.median(s[valid])) if valid.any() else float("nan"),
                   n_blanks=n_blanks, observed_clean=obs, expected_clean=exp,
                   n_candidates=len(cands), n_clean=int(sum(c["clean"] for c in cands)))
    return cands, summary


COLUMNS = ["y", "x", "charge_e", "npix", "E_keV", "lam", "noise", "T", "n_exp", "p_image", "clean", "mask_halo", "mask_trail",
           "mask_hotcol", "mask_edge", "mask_cluster", "pixels"]


def write(cands, summary, prefix):
    """<prefix>_fe_candidates.csv (one row per candidate) and <prefix>_fe_candidates.json (the summary)"""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + "_fe_candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for c in cands:
            w.writerow({k: c[k] for k in COLUMNS})
    with open(prefix + "_fe_candidates.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return prefix + "_fe_candidates.csv", prefix + "_fe_candidates.json"
