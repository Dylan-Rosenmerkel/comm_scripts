"""Multi-hypothesis clustering (MHC): specialised finders for each event type + significance arbitration.

Step 0  background removal: dark-charge / offset map (AM.background_map) and hot-column correction (per-column
        excess of the quiet pixels, subtracted where it is > 5 sigma).
Step 1  candidate finders, each specialised to one kind of deposit
        - point   : matched-filter peaks (MFD+G) -> X-rays (front / diffuse back), small deposits
        - extended: connected regions of pixels above a Poisson-tail threshold (alpha = 0.01) -> tracks / blobs
        - counting: groups of pixels with >= 1 e- (photon-counting CCDs only), kept if the Poisson probability of
                    their charge given the local dark charge is < p_count -> few-electron deposits
Step 2  arbitration where finders overlap: inside every extended region that holds >= 2 point candidates or is
        >= ext_min px long, compare two explanations of the same pixels
            H_pt : the point candidates, each a pixel-integrated Gaussian (its MFD+G centre and width), free
                   non-negative amplitudes
            H_ext: one extended deposit, a chain of Gaussian beads along the region's skeleton (1.5 px apart,
                   common width from a grid), free non-negative amplitudes
        with the Poisson-weighted chi^2 of each fit. H_ext wins only if it is better by > dchi_min AND by a
        fraction > frac_min of chi^2(H_pt) (a significant, not a cosmetic, improvement). Counting candidates that
        overlap a point or extended event are absorbed by it.
Step 3  classification of every final event (cls): 'x-ray', 'x-ray diffuse', 'low-E', 'muon', 'worm', 'alpha',
        'artifact-column' (hot column / vertical CTI streak), 'artifact-row' (serial-register / horizontal streak)
        and a significance sig = sum(R) / sqrt(sum var) over its pixels.
Energies: point = MFD+G footprint energy (counting-mode for < count_E_max e- on photon-counting CCDs),
          extended = sum of R over the region + 1 ring.
"""
import os, sys
import numpy as np
from scipy import ndimage, optimize, stats
from skimage.morphology import skeletonize
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import agent_methods as AM
import methods2 as M2
E_PER_E = AM.E_PER_E
EIGHT = np.ones((3, 3), bool)


def hot_columns(Q, valid, bg, nsig=5.0, min_excess=0.05):
    """per-column excess of the quiet (non-cluster) pixels over the background map; returns the excess per
    column (0 where not significant) and the flagged columns"""
    quiet = valid & ~bg.mask
    R = np.where(quiet, Q - bg.mean, 0.0)
    n = quiet.sum(0)
    ex = R.sum(0) / np.maximum(n, 1)
    err = np.sqrt(np.where(quiet, bg.var, 0).sum(0)) / np.maximum(n, 1)
    hot = (n > 20) & (ex > nsig * err) & (ex > min_excess)
    return np.where(hot, ex, 0.0), hot


def _gauss_cols(ys, xs, centres, sig):
    """sparse design matrix: pixel-integrated unit Gaussians (centre, width) at pixels (ys, xs), each column
    evaluated only within 4 sigma + 1 px of its centre"""
    from scipy import sparse
    rows, cols, vals = [], [], []
    for j, ((cy, cx), s) in enumerate(zip(centres, sig)):
        r = 4 * s + 1
        k = np.nonzero((np.abs(ys - cy) <= r) & (np.abs(xs - cx) <= r))[0]
        if k.size == 0:
            continue
        v = AM._f1d(ys[k].astype(float), cy, s)[0] * AM._f1d(xs[k].astype(float), cx, s)[0]
        rows.append(k); cols.append(np.full(k.size, j)); vals.append(v)
    if not rows:
        return sparse.csr_matrix((ys.size, max(len(centres), 1)))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ys.size, len(centres)))


def _nnls_chi2(A, b, w):
    """non-negative weighted least squares (bounded, sparse LSMR-based trust region); returns chi^2, amplitudes"""
    from scipy import sparse
    Aw = sparse.diags(w) @ A
    res = optimize.lsq_linear(Aw, b * w, bounds=(0, np.inf), method="trf", lsmr_tol="auto", max_iter=200)
    r = (b - A @ res.x) * w
    return float(r @ r), res.x


def _beads(mask, sep=1.5):
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    if ys.size == 0:
        ys, xs = np.nonzero(mask)
    chosen = []
    for y, x in zip(ys, xs):
        if all((y - a) ** 2 + (x - b) ** 2 >= sep * sep for a, b in chosen[-40:]):
            chosen.append((y, x))
    return chosen


def shape(ys, xs, w):
    """charge-weighted principal axes: length (uniform-line equivalent), width, orientation, straightness"""
    w = np.clip(w, 0, None); s = w.sum()
    if s <= 0 or ys.size < 2:
        return dict(length=1.0, width=0.3, angle=0.0, straight=0.0, ratio=1.0)
    my, mx = (w * ys).sum() / s, (w * xs).sum() / s
    C = np.cov(np.vstack([ys - my, xs - mx]), aweights=w + 1e-12)
    ev, evec = np.linalg.eigh(C)
    l1, l2 = max(ev[1], 1e-6), max(ev[0], 1e-6)
    d = evec[:, 1]
    perp = (ys - my) * (-d[1]) + (xs - mx) * d[0]
    return dict(length=float(np.sqrt(12 * l1)), width=float(np.sqrt(l2)), angle=float(np.degrees(np.arctan2(d[0], d[1])) % 180),
                straight=float(np.sqrt((w * perp ** 2).sum() / s)), ratio=float(np.sqrt(l2 / l1)))


def streak(reg, above, R, frac=0.85, run_min=8):
    """row / column streak test on one region (2-D sub-arrays): >= frac of the positive charge in <= 2 adjacent
    rows (columns) and a run of >= run_min consecutive above-threshold pixels along them.
    Returns 'artifact-row', 'artifact-column' or None"""
    q = np.where(reg, np.clip(R, 0, None), 0.0); tot = q.sum()
    if tot <= 0:
        return None
    for axis, name in ((1, "artifact-row"), (0, "artifact-column")):
        prof = q.sum(axis)                       # charge per row (axis=1) or per column (axis=0)
        two = prof[:-1] + prof[1:] if prof.size > 1 else prof
        k = int(np.argmax(two)); sel = slice(k, k + 2)
        if two[k] < frac * tot:
            continue
        line = (above & reg)[sel, :].any(0) if axis == 1 else (above & reg)[:, sel].any(1)
        best = cur = 0
        for v in line:
            cur = cur + 1 if v else 0; best = max(best, cur)
        if best >= run_min:
            return name
    return None


# physics fixes of the 2026-10-25 audit (docs/aGMX, section 'Physics audit'); AGMX_PHYS_FIXES=0 restores the earlier behaviour
PHYS_FIX = __import__("os").environ.get("AGMX_PHYS_FIXES", "1") != "0"
MIP_MIN = 2.5      # keV per projected pixel: a minimum-ionising particle loses ~0.27 keV/um, >= ~4 keV per 15-um pixel


def classify_ext(E, sh, npix):
    a = sh["angle"]
    if sh["width"] < 0.6 and sh["length"] >= 5:
        if min(abs(a - 90), abs(a - 270)) < 5: return "artifact-column"
        if min(a, abs(a - 180)) < 5: return "artifact-row"
    if E > 1000 and sh["ratio"] > 0.4: return "alpha"
    if sh["length"] >= 10 and sh.get("sk_straight", 9) < 1.0 and sh.get("sk_straight", 9) < 0.06 * sh["length"]:
        # G1 (audit): a straight track is a muon only if it deposits at least the minimum-ionising rate
        if PHYS_FIX and E / max(sh["length"], 1.0) < MIP_MIN:
            return "extended"
        return "muon"
    if sh["ratio"] > 0.6 and sh["length"] < 12: return "x-ray diffuse"
    return "worm"


def mhc(Q, valid, bg=None, ext_min=8, dchi_min=50.0, frac_min=0.3, link_alpha=0.01, p_count=1e-5,
        count_E_max=100, hotcol=True, return_info=False, max_region=30000, rule="bic", bic_margin=10.0, x_max=75.0):
    if bg is None:
        bg = AM.background_map(Q, valid)
    info = {}
    # step 0: hot columns
    if hotcol:
        ex, hot = hot_columns(Q, valid, bg)
        Qc = Q - ex[None, :]
        info["hot_columns"] = np.nonzero(hot)[0].tolist()
    else:
        Qc = Q
    R = np.where(valid, Qc - bg.mean, 0.0)
    counting = float(np.median(bg.s[valid])) < 0.5
    # step 1a: points
    pts = AM.mf_detect(Qc, valid, bg)[0]
    pts = [c for c in pts if np.isfinite(c["E"])]
    if counting and pts:
        ce = M2.counting_energy(Qc, bg, pts)
        for c, d in zip(pts, ce):
            if c["E"] / E_PER_E < count_E_max:
                c["E"] = d["E"]
    own = M2.owner_map(Q.shape, pts)
    # step 1b: extended regions
    M = (Qc > M2.link_threshold(bg, link_alpha)) & valid
    lab, n = ndimage.label(M, structure=EIGHT)
    objs = ndimage.find_objects(lab)
    in_reg = {}
    for i, c in enumerate(pts):
        L = lab[c["yp"], c["xp"]]
        in_reg.setdefault(int(L), []).append(i)
    out = []
    used = np.zeros(len(pts), bool)
    dec = []
    for L in range(1, n + 1):
        sl = objs[L - 1]
        mem = in_reg.get(L, [])
        ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if len(mem) < 2 and ext < ext_min:
            continue
        if len(mem) == 0 and ext < ext_min:
            continue
        # pixels of the comparison: region + 2 rings
        pad = 3
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, Q.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, Q.shape[1])
        reg = lab[y0:y1, x0:x1] == L
        if reg.sum() > max_region:
            continue
        use = ndimage.binary_dilation(reg, structure=EIGHT, iterations=2) & valid[y0:y1, x0:x1]
        ys, xs = np.nonzero(use)
        b = R[y0:y1, x0:x1][use]
        w = 1 / np.sqrt(bg.var[y0:y1, x0:x1][use] + np.clip(b, 0, None))
        if mem:
            cen = [(pts[i]["y"] - y0, pts[i]["x"] - x0) for i in mem]
            sig = [float(np.clip(pts[i]["sxy"], 0.3, 4.0)) for i in mem]
            chi_pt, _ = _nnls_chi2(_gauss_cols(ys, xs, cen, sig), b, w)
        else:
            chi_pt = float((b * w) @ (b * w))
        st = streak(reg, M[y0:y1, x0:x1], R[y0:y1, x0:x1])
        if st is not None:
            rr = ndimage.binary_dilation(reg, structure=EIGHT) & valid[y0:y1, x0:x1]
            ry, rx = np.nonzero(rr); q = R[y0:y1, x0:x1][rr]
            wq = np.clip(q, 0, None); sw = max(wq.sum(), 1e-9)
            out.append(dict(E=float(q.sum() * E_PER_E), y=float((wq * (ry + y0)).sum() / sw), x=float((wq * (rx + x0)).sum() / sw),
                            sxy=np.nan, npix=int(rr.sum()), cls=st, n_pieces=len(mem), dchi2=np.nan,
                            sig=float(q.sum() / np.sqrt(bg.var[y0:y1, x0:x1][rr].sum()))))
            used[mem] = True; own[y0:y1, x0:x1][rr] = -2
            dec.append(dict(L=L, n_pt=len(mem), chi_pt=np.nan, chi_ext=np.nan, ext=True, streak=st))
            continue
        bd = _beads(reg)
        chi_ext = np.inf
        for s in (0.5, 0.8, 1.2, 1.8):
            c2, _ = _nnls_chi2(_gauss_cols(ys, xs, bd, [s] * len(bd)), b, w)
            chi_ext = min(chi_ext, c2)
        d = chi_pt - chi_ext
        if rule == "bic":
            # Bayesian information criterion: chi^2 + k ln(n), k = 4 per point (A, y, x, sigma), beads + 1 for the chain
            lnn = np.log(max(b.size, 2))
            d = (chi_pt + 4 * len(mem) * lnn) - (chi_ext + (len(bd) + 1) * lnn)
            win_ext = d > bic_margin or (len(mem) == 0)
        else:
            win_ext = (d > dchi_min and d > frac_min * chi_pt) or (len(mem) == 0)
        dec.append(dict(L=L, n_pt=len(mem), chi_pt=chi_pt, chi_ext=chi_ext, ext=win_ext))
        if not win_ext:
            continue
        rr = ndimage.binary_dilation(reg, structure=EIGHT) & valid[y0:y1, x0:x1]
        ry, rx = np.nonzero(rr)
        q = R[y0:y1, x0:x1][rr]
        E = float(q.sum() * E_PER_E)
        sh = shape(ry + y0, rx + x0, q)
        by = np.array([p[0] for p in bd], float); bx = np.array([p[1] for p in bd], float)
        sk = shape(by, bx, np.ones(by.size)) if by.size >= 3 else dict(straight=0.0, length=1.0)
        sh["sk_straight"], sh["sk_length"] = sk["straight"], sk["length"]
        wq = np.clip(q, 0, None); sw = max(wq.sum(), 1e-9)
        ev = dict(E=E, y=float((wq * (ry + y0)).sum() / sw), x=float((wq * (rx + x0)).sum() / sw), sxy=sh["width"],
                  npix=int(rr.sum()), cls=classify_ext(E, sh, rr.sum()), n_pieces=len(mem), dchi2=float(d),
                  sig=float(q.sum() / np.sqrt(bg.var[y0:y1, x0:x1][rr].sum())), **{k: sh[k] for k in ("length", "width", "straight", "sk_straight")})
        out.append(ev)
        used[mem] = True
        # pixels of this event are owned by it (for the counting step)
        own[y0:y1, x0:x1][rr] = -2
    for i, c in enumerate(pts):
        if used[i]:
            continue
        E = c["E"]
        cls = ("alpha" if E > 1000 else "big deposit" if E > x_max else "low-E" if E < 0.5
               else "x-ray diffuse" if c["sxy"] >= 1.0 else "x-ray")
        out.append(dict(c, cls=cls, n_pieces=1, sig=float(c["T"])))
    # step 1c: counting candidates not covered by anything else
    if counting:
        nn = np.where(valid, np.round(Qc - bg.o), 0).clip(0)
        cl2, k = ndimage.label(nn >= 1, structure=EIGHT)
        for j, sl in enumerate(ndimage.find_objects(cl2), start=1):
            m = cl2[sl] == j
            if m.sum() > 9 or (own[sl][m] != -1).any():
                continue
            bsl = (slice(max(sl[0].start - 1, 0), sl[0].stop + 1), slice(max(sl[1].start - 1, 0), sl[1].stop + 1))
            box = ndimage.binary_dilation(cl2[bsl] == j, structure=EIGHT)
            S = float(nn[sl][m].sum())
            mu = float(bg.lam[bsl][box].sum())
            p = float(stats.poisson.sf(S - 1, max(mu, 1e-9)))
            if p < p_count:
                ys, xs = np.nonzero(m); q = nn[sl][m]
                out.append(dict(E=float((S - bg.lam[sl][m].sum()) * E_PER_E), y=float((q * (ys + sl[0].start)).sum() / S),
                                x=float((q * (xs + sl[1].start)).sum() / S), sxy=np.nan, npix=int(m.sum()), cls="low-E",
                                n_pieces=1, sig=float(stats.norm.isf(max(p, 1e-300)))))
    if return_info:
        info["decisions"] = dec
        return out, info
    return out
