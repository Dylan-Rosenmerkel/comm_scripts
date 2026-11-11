"""QUICK PROTOTYPE (not a method yet): a counting-box finder with an analytic false-alarm probability (wip/lowdc).

Purpose: support the ranking of improvements in RESULTS.md by measuring, on the SAME simulated images as
bench_agmx.py (same seeds), what a finder gains when it (a) groups electrons by a window instead of pixel
connectivity (diffuse events whose electrons do not touch stay one candidate), (b) takes its threshold from an
analytic null probability instead of simulated blanks, and (c) uses an occupancy-based background (isolated
single-electron pixels, which includes the rounding miscounts of the read noise).

  counts       nn = round(Q - o_r), o_r = row median (lam << 1), counts clipped at 0
  background   q = fraction of isolated counted pixels (no other counted pixel in 5 x 5), per image; corrected for the
               dark pixels that do have a counted neighbour (exp(-24 q)); the dark charge lam is the value whose pixel
               model P(round(Poisson(lam) + N(0, s)) = 1) gives q (s = MAD of the zero peak), so the rounding miscounts
               are in the null but cannot make 2 e- in one pixel
  statistic    for windows w = 1, 2, 3, 5 px (all positions): N_w = window count, p_w = P(sum of w^2 null pixels >= N_w)
               (exact: the per-pixel null pmf convolved w^2 times)
  threshold    p_w < F / 1e6 / 4 (Bonferroni over 4 window sizes, F = fakes per Mpix, one trial per pixel and window)
  detections   8-connected groups of significant window corners, footprint = counted pixels of their windows

    python proto_box.py [--F 0.08] [--set grid,blanks,l2]  -> results/proto_box_events.csv, proto_box_images.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, sys, time
import numpy as np
import pandas as pd
from scipy import ndimage, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, metrics as MT, bench_agmx as B  # noqa: E402

WINDOWS = (1, 2, 3, 5)


def occupancy(nn, valid):
    """isolated counted-pixel rate (dark charge + rounding miscounts), corrected for dark pixels with a counted neighbour"""
    occ = (nn >= 1) & valid
    nb = ndimage.uniform_filter(occ.astype(float), 5, mode="constant") * 25 - occ
    iso = (nn == 1) & valid & (nb < 0.5)
    q = iso.sum() / max(valid.sum(), 1)
    return float(max(q / np.exp(-24.0 * q), 1e-7))


def pixel_pmf(lam, s, K=24):
    """P(round(k + e) = n), n = 0..K, k ~ Poisson(lam), e ~ N(0, s) (the skipper pixel null, as gmx.pix_null_logp)"""
    n = np.arange(K + 1)
    k = np.arange(K + 4)
    pk = stats.poisson.pmf(k, lam)
    pn = (stats.norm.cdf((n[:, None] + 0.5 - k[None, :]) / s) - stats.norm.cdf((n[:, None] - 0.5 - k[None, :]) / s)) @ pk
    return pn / pn.sum()


def lam_from_q(q, s):
    """dark charge whose pixel model gives the measured counted-pixel rate q (P(n = 1) rises with lam for lam < 1)"""
    g = np.geomspace(1e-8, 0.5, 300)
    p1 = np.array([pixel_pmf(l, s, 3)[1] for l in g])
    return float(np.interp(q, p1, g, left=1e-8))


def window_sf(lam, s, w, K=24):
    """P(window sum >= N) for N = 0..K, w x w null pixels"""
    p = pixel_pmf(lam, s, K)
    t = np.array([1.0])
    for _ in range(w * w):
        t = np.convolve(t, p)[:K + 1]
    t = np.r_[t, np.zeros(K + 1 - t.size)]
    return np.maximum(1.0 - np.r_[0.0, np.cumsum(t)[:-1]], 1e-300)


def box_finder(Q, valid, F=0.08, q=None):
    """detections [dict(y, x, pix, N, w, p)] and the background q used"""
    H, W = Q.shape
    o = np.median(np.where(valid, Q, np.nan), axis=1)
    o = np.where(np.isfinite(o), o, 0.0)
    nn = np.where(valid, np.round(Q - o[:, None]), 0).clip(0)
    q = occupancy(nn, valid) if q is None else q
    v = np.where(valid, Q - o[:, None], np.nan)
    z = v[np.abs(v) < 0.5]
    sn = float(max(1.4826 * np.median(np.abs(z - np.median(z))), 0.02))
    lam = lam_from_q(q, sn)
    p_thr = F / 1e6 / len(WINDOWS)
    sig = np.zeros((H, W), bool)
    best = np.ones((H, W)); wbest = np.zeros((H, W), int)
    for w in WINDOWS:
        # window anchored so that the window covering pixels [r - (w-1)//2 ... ] is centred (even w: shifted by half a pixel)
        N = np.rint(ndimage.uniform_filter(nn, w, mode="constant") * w * w)
        sf = window_sf(lam, sn, w)
        p = sf[np.clip(N, 0, sf.size - 1).astype(int)]
        s = (p < p_thr) & valid
        upd = s & (p < best)
        best[upd] = p[upd]; wbest[upd] = w
        # footprint of a significant window: dilate the centre by the window half-size
        if s.any():
            # (even w: one pixel larger, covering both alignments of the window)
            wf = w + (w % 2 == 0)
            sig |= ndimage.binary_dilation(s, structure=np.ones((wf, wf), bool)) if w > 1 else s
    # join counted pixels whose significant windows overlap: label the significant footprint, keep counted pixels
    lab2, n2 = ndimage.label(sig, structure=np.ones((3, 3), bool))
    out = []
    for k2, sl in enumerate(ndimage.find_objects(lab2), start=1):
        m = (lab2[sl] == k2) & (nn[sl] >= 1)
        if not m.any():
            continue
        ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
        c = nn[ys, xs]
        pb = best[sl][lab2[sl] == k2].min()
        out.append(dict(y=float((c * ys).sum() / c.sum()), x=float((c * xs).sum() / c.sum()), pix=(ys, xs), N=float(c.sum()), p=float(pb)))
    return out, (q, lam, sn)


def run(name, i, par, F):
    rng = np.random.default_rng(B._seed(name, i))
    Q, valid, truth, p, rate = sim.simulate(rng, **par)
    Q = np.where(valid, Q, 0.0)
    det, (q, lam, sn) = box_finder(Q, valid, F)
    own, per = MT.match(det, truth, Q.shape[1])
    ev = [dict(config=name, img=i, ev=k, kind=t["kind"], n_e=t["n_e"], sigma=t["sigma"], n_pix=len(t["pix"][0]),
               found=int(bool(per[k])), n_obj=len(per[k]), q=sum(det[j]["N"] for j in per[k])) for k, t in enumerate(truth)]
    fk = [dict(config=name, img=i, y=d["y"], x=d["x"], N=d["N"], npix=len(d["pix"][0]), p=d["p"], near_big=int(own[j] == -2))
          for j, d in enumerate(det) if own[j] < 0]
    return dict(config=name, img=i, mpix=valid.sum() / 1e6, q_occ=q, lam_fit=lam, s_fit=sn, lam_true=float(np.median(rate[valid])), n_det=len(det)), ev, fk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=float, default=0.08)
    ap.add_argument("--set", default="grid,blanks,l2")
    a = ap.parse_args()
    sets = a.set.split(",")
    rows_i, rows_e, rows_f = [], [], []
    t0 = time.time()
    for name, (st, n, par) in B.configs().items():
        if st not in sets or name.startswith(("art", "busy")):
            continue
        for i in range(n):
            im, ev, fk = run(name, i, par, a.F)
            rows_i.append(im); rows_e += ev; rows_f += fk
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    tag = f"_F{a.F:g}"
    pd.DataFrame(rows_i).to_csv(os.path.join(B.OUT, f"proto_box_images{tag}.csv"), index=False)
    pd.DataFrame(rows_e).to_csv(os.path.join(B.OUT, f"proto_box_events{tag}.csv"), index=False)
    pd.DataFrame(rows_f).to_csv(os.path.join(B.OUT, f"proto_box_fakes{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
