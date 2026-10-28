"""Serial-register streaks (scratch): charge clocked into the serial register lands in ONE row with no vertical diffusion.
A pixel is 'single-row' when it is significant (above the Poisson-tail link threshold) and the pixels directly above and
below hold < f of its excess charge. Horizontal runs of single-row pixels (gaps <= 1 px) of >= run_min px are streaks.
They are removed from the image before clustering (valid = False) and reported as separate 'artifact-row' clusters."""
import numpy as np
from scipy import ndimage
import methods2 as M2

def find_streaks(Q, valid, bg, run_min=8, f=0.25, thr_alpha=0.01, gap_join=10):
    R = np.where(valid, Q - bg.mean, 0.0)
    sig = (Q > M2.link_threshold(bg, thr_alpha)) & valid
    up = np.zeros_like(R); dn = np.zeros_like(R); up[:-1] = R[1:]; dn[1:] = R[:-1]
    single = sig & (np.clip(up, 0, None) < f * R) & (np.clip(dn, 0, None) < f * R)
    h = ndimage.binary_closing(single, structure=np.ones((1, 3), bool)) & sig      # bridge 1-px gaps (only significant pixels)
    lab, n = ndimage.label(h, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool))                    # horizontal connectivity only
    streak = np.zeros_like(single); out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl[1].stop - sl[1].start < run_min: continue
        m = lab[sl] == k; ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
        streak[ys, xs] = True
        q = R[ys, xs]
        out.append(dict(E=float(q.sum() * 3.75e-3), y=float(ys.mean()), x=float(np.average(xs, weights=np.clip(q, 1e-9, None))), sxy=0.0,
                        cls="artifact-row", blend=0.0, ped=0.0, regime="serial", _pix=(ys, xs, q), n_comp=1))
    # pieces of one streak cut by a track or X-ray crossing it: same row, gap <= gap_join px -> one cluster
    out.sort(key=lambda c: (int(c["_pix"][0][0]), int(c["_pix"][1].min())))
    merged = []
    for c in out:
        p = merged[-1] if merged else None
        if p is not None and int(p["_pix"][0][0]) == int(c["_pix"][0][0]) and c["_pix"][1].min() - p["_pix"][1].max() <= gap_join:
            ys = np.r_[p["_pix"][0], c["_pix"][0]]; xs = np.r_[p["_pix"][1], c["_pix"][1]]; q = np.r_[p["_pix"][2], c["_pix"][2]]
            p.update(E=float(q.sum() * 3.75e-3), x=float(np.average(xs, weights=np.clip(q, 1e-9, None))), _pix=(ys, xs, q))
        else:
            merged.append(c)
    return streak, merged

def install(AD):
    """wrap adaptive_cluster: streak pixels out of the image, streak clusters appended"""
    if getattr(AD, "_streaks", False): return
    _oac = AD.adaptive_cluster
    def adaptive_cluster(Q, valid, bg=None, *a, **k):
        if bg is None: bg = AD.AM.background_map(Q, valid)
        st, sc = find_streaks(Q, valid, bg)
        r = _oac(Q, valid & ~st, bg, *a, **k)
        cl, info = r if k.get("return_info") else (r, None)
        cl = list(cl) + sc
        if info is not None: info["n_streaks"] = len(sc)
        return (cl, info) if k.get("return_info") else cl
    AD._oac_orig = _oac; AD.adaptive_cluster = adaptive_cluster; AD._streaks = True

def uninstall(AD):
    if getattr(AD, "_streaks", False): AD.adaptive_cluster = AD._oac_orig; AD._streaks = False
